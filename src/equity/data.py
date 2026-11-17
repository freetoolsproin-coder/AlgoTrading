"""
Market data for the equity service (Upstox, real data only - nothing here is simulated).

Endpoints (checked against the live API on 2026-11-16)
    v3 /historical-candle/{key}/{unit}/{interval}/{to}/{from}   past sessions, NOT including today
           minutes 1-15: at most ~1 month per request      hours: one quarter      days / weeks: years
    v3 /historical-candle/intraday/{key}/minutes/1               today's 1-minute candles
    v2 /market-quote/quotes                                      LTP, OHLC, volume, VWAP, 5-level depth, circuit limits
    v3 /feed/market-data-feed/authorize                          WebSocket stream (optional, [SERVICE] websocket)

The shared UpstoxClient gives the per-second spacing, 429 back-off and token reload; RateBudget adds the per-minute
and per-30-minute ceilings so that a backtest download cannot starve the other services. History is cached on disk
(data/equity/history/<SYMBOL>_<tf>.csv.gz) and only the missing days are downloaded again.

A missing value is NaN / None all the way to the dashboard; it is never replaced by zero.
"""
from __future__ import annotations

import collections
import json
import logging
import threading
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote as urlquote

import numpy as np
import pandas as pd

from src.equity.indicators import OHLCV, resample_bars
from src.stocks.universe import NIFTY_INDEX_KEY, Stock
from src.utils.numbers import fnum

logger = logging.getLogger("scanner.equity.data")
TZ = "Asia/Kolkata"
V3 = "https://api.upstox.com/v3"
OPEN, CLOSE = time(9, 15), time(15, 30)
# timeframe id -> (unit, interval, days per request, intraday?)
TIMEFRAMES = {"1m": ("minutes", 1, 28, True), "5m": ("minutes", 5, 28, True), "15m": ("minutes", 15, 28, True),
              "30m": ("minutes", 30, 85, True), "1h": ("hours", 1, 85, True), "1d": ("days", 1, 3000, False)}
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def is_trading_day(day: date) -> bool:
    try:
        from src.utils.market_calendar import is_trading_day as f
        return f(day)
    except Exception:      # noqa: BLE001 - calendar file missing: weekends only
        return day.weekday() < 5


def market_open(now: pd.Timestamp) -> bool:
    return is_trading_day(now.date()) and OPEN <= now.time() < CLOSE


def last_session(now: pd.Timestamp) -> date:
    """The latest session that has started (today once the market has opened on a trading day)."""
    d = now.date()
    if not (is_trading_day(d) and now.time() >= OPEN):
        d -= timedelta(days=1)
        while not is_trading_day(d):
            d -= timedelta(days=1)
    return d


def to_bars(frame: pd.DataFrame, session: bool = True) -> pd.DataFrame:
    """candles_frame (datetime column) -> OHLCV indexed by bar start. `session` keeps 09:15-15:30 only."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=TZ))
    out = frame.set_index("datetime")[OHLCV].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if session:
        clock = out.index.time
        out = out[(clock >= OPEN) & (clock < CLOSE)]
    return out


class RateBudget:
    """Sliding-window ceilings on top of the client's per-second spacing."""

    def __init__(self, per_minute: int = 300, per_30min: int = 1500):
        self.limits = ((60.0, per_minute), (1800.0, per_30min))
        self.calls = collections.deque()
        self.lock = threading.Lock()
        self.waited = 0.0

    def acquire(self):
        while True:
            with self.lock:
                now = _time.monotonic()
                while self.calls and now - self.calls[0] > 1800.0:
                    self.calls.popleft()
                wait = 0.0
                for window, limit in self.limits:
                    recent = [t for t in self.calls if now - t <= window]
                    if len(recent) >= limit:
                        wait = max(wait, recent[0] + window - now + 0.05)
                if wait <= 0:
                    self.calls.append(now)
                    return
            self.waited += wait
            _time.sleep(min(wait, 5.0))

    def status(self) -> dict:
        now = _time.monotonic()
        with self.lock:
            return {"last_minute": sum(1 for t in self.calls if now - t <= 60), "last_30min": len(self.calls),
                    "limit_minute": self.limits[0][1], "limit_30min": self.limits[1][1], "waited_seconds": round(self.waited, 1)}


def normalize_quote(raw: dict) -> dict:
    """Upstox quote -> flat dict. Absent numbers stay NaN (bid / ask are NaN when that side of the book is empty)."""
    ohlc, depth = raw.get("ohlc") or {}, raw.get("depth") or {}
    ltp, change = fnum(raw.get("last_price")), fnum(raw.get("net_change"))

    def levels(side):
        return [{"price": fnum(x.get("price")), "qty": fnum(x.get("quantity")), "orders": fnum(x.get("orders"))}
                for x in (depth.get(side) or []) if fnum(x.get("price"), 0.0) > 0]
    buy, sell = levels("buy"), levels("sell")
    tbq, tsq = fnum(raw.get("total_buy_quantity")), fnum(raw.get("total_sell_quantity"))
    return {"ltp": ltp, "open": fnum(ohlc.get("open")), "high": fnum(ohlc.get("high")), "low": fnum(ohlc.get("low")),
            "prev_close": ltp - change if np.isfinite(ltp) and np.isfinite(change) else float("nan"),
            "change": change, "volume": fnum(raw.get("volume")), "vwap": fnum(raw.get("average_price")),
            "bid": buy[0]["price"] if buy else float("nan"), "ask": sell[0]["price"] if sell else float("nan"),
            "bid_qty": buy[0]["qty"] if buy else float("nan"), "ask_qty": sell[0]["qty"] if sell else float("nan"),
            "depth": {"buy": buy, "sell": sell}, "total_buy_qty": tbq, "total_sell_qty": tsq,
            "lower_circuit": fnum(raw.get("lower_circuit_limit")), "upper_circuit": fnum(raw.get("upper_circuit_limit")),
            "trade_time": raw.get("last_trade_time"), "symbol": raw.get("symbol")}


class EquityData:
    def __init__(self, client=None, cache_dir: Optional[Path] = None, requests_per_second: float = 4.0,
                 per_minute: int = 300, per_30min: int = 1500, universe: Optional[List[Stock]] = None):
        if client is None:
            from src.broker.client import UpstoxClient
            client = UpstoxClient(requests_per_second=requests_per_second)
        self.client = client
        if cache_dir is None:
            from src.equity.config import DATA_DIR
            cache_dir = DATA_DIR / "history"
        self.cache_dir = Path(cache_dir)
        self.budget = RateBudget(per_minute, per_30min)
        self._universe = universe
        self._mem: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._retry: Dict[Tuple[str, str], float] = {}     # history that Upstox has not published yet: ask again later
        self._locks: Dict[str, threading.Lock] = collections.defaultdict(threading.Lock)
        self.feed = None                       # optional WebSocket LTP stream (start_feed)
        self.calls = 0

    # ------------------------------------------------------------------ universe
    def universe(self) -> List[Stock]:
        if self._universe is None:
            from src.stocks.universe import load_universe
            self._universe = load_universe()
        return self._universe

    def stock(self, symbol: str) -> Optional[Stock]:
        symbol = symbol.strip().upper()
        return next((s for s in self.universe() if s.symbol == symbol), None)

    # ------------------------------------------------------------------ raw API
    def _candles(self, path: str) -> pd.DataFrame:
        from src.broker.client import candles_frame
        self.budget.acquire()
        self.calls += 1
        return candles_frame(self.client.get(path, base=V3).get("data", {}).get("candles", []))

    def fetch(self, key: str, tf: str, start: date, end: date) -> pd.DataFrame:
        """Past sessions [start, end] of one timeframe, in as many requests as the API's range limit needs."""
        unit, interval, span, intraday = TIMEFRAMES[tf]
        frames, cur = [], start
        while cur <= end:
            stop = min(end, cur + timedelta(days=span - 1))
            frames.append(self._candles(f"/historical-candle/{urlquote(key, safe='')}/{unit}/{interval}/"
                                        f"{stop.isoformat()}/{cur.isoformat()}"))
            cur = stop + timedelta(days=1)
        frames = [f for f in frames if not f.empty]
        return to_bars(pd.concat(frames) if frames else None, session=intraday)

    def today_1m(self, key: str) -> pd.DataFrame:
        return to_bars(self._candles(f"/historical-candle/intraday/{urlquote(key, safe='')}/minutes/1"))

    def quotes(self, keys: Iterable[str]) -> Dict[str, dict]:
        keys = [k for k in dict.fromkeys(keys) if k]
        out: Dict[str, dict] = {}
        for i in range(0, len(keys), 400):
            self.budget.acquire()
            self.calls += 1
            data = self.client.get("/market-quote/quotes", {"instrument_key": ",".join(keys[i:i + 400])}).get("data") or {}
            for raw in data.values():
                token = raw.get("instrument_token")
                if token:
                    out[token] = normalize_quote(raw)
        if self.feed is not None:              # a fresher streamed LTP wins over the polled one
            for k, q in out.items():
                tick = self.feed.latest(k)
                if tick is not None:
                    q["ltp"], q["streamed"] = tick[0], True
        return out

    # ------------------------------------------------------------------ disk-cached history
    def _file(self, symbol: str, tf: str) -> Path:
        return self.cache_dir / f"{symbol.replace('&', '_')}_{tf}.csv.gz"

    def _read(self, path: Path) -> Tuple[pd.DataFrame, Optional[date]]:
        meta = path.with_suffix(".json")
        try:
            frame = pd.read_csv(path, compression="gzip")
            frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert(TZ)
            since = date.fromisoformat(json.loads(meta.read_text(encoding="utf-8"))["since"])
            return frame.set_index("datetime")[OHLCV].astype(float), since
        except (OSError, ValueError, KeyError):
            return to_bars(None), None

    def _write(self, path: Path, bars: pd.DataFrame, since: date):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        out = bars.copy()
        out.index = out.index.tz_convert("UTC")
        out.to_csv(tmp, compression="gzip", index_label="datetime")
        tmp.replace(path)
        path.with_suffix(".json").write_text(json.dumps({"since": since.isoformat()}), encoding="utf-8")

    def history(self, symbol: str, key: str, tf: str, start: date, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Completed past sessions from `start` up to yesterday (today is never in the historical endpoint)."""
        now = now or now_ist()
        end = now.date() - timedelta(days=1)
        with self._locks[f"{symbol}|{tf}"]:
            cached = self._mem.get((symbol, tf))
            if cached is not None and cached.attrs.get("since") <= start and \
                    (cached.attrs.get("end") == end or _time.monotonic() < self._retry.get((symbol, tf), 0.0)):
                return cached[cached.index.date >= start]
            bars, since = self._read(self._file(symbol, tf))
            parts, changed = [bars], False
            if since is None or bars.empty:
                parts, since, changed = [self.fetch(key, tf, start, end)], start, True
            else:
                if start < since:
                    parts.insert(0, self.fetch(key, tf, start, since - timedelta(days=1)))
                    since, changed = start, True
                last = bars.index[-1].date()
                if last < end and any(is_trading_day(last + timedelta(days=i)) for i in range(1, (end - last).days + 1)):
                    parts.append(self.fetch(key, tf, last + timedelta(days=1), end))
                    changed = True
            parts = [p for p in parts if not p.empty]
            bars = pd.concat(parts) if parts else to_bars(None)
            bars = bars[~bars.index.duplicated(keep="last")].sort_index()
            if changed and not bars.empty:
                self._write(self._file(symbol, tf), bars, since)
            expected = end                                   # the last session the history should already contain
            while not is_trading_day(expected):
                expected -= timedelta(days=1)
            complete = len(bars) > 0 and bars.index[-1].date() >= expected
            if not complete:                                 # Upstox publishes a session's history some hours later
                self._retry[(symbol, tf)] = _time.monotonic() + 600.0
            bars.attrs.update(end=end if complete else None, since=since)
            self._mem[(symbol, tf)] = bars
            return bars[bars.index.date >= start]

    def forget(self, tf: Optional[str] = None):
        """Drop the in-memory copies (the disk cache stays). Long histories of a backtest are not kept in RAM."""
        for k in [k for k in self._mem if tf is None or k[1] == tf]:
            self._mem.pop(k, None)

    # ------------------------------------------------------------------ frames the engine needs
    def intraday_bars(self, stock: Stock, now: pd.Timestamp, history_days: int = 20) -> Dict[str, pd.DataFrame]:
        """{"1m": today's COMPLETED 1-minute bars, "5m": completed 5-minute bars of the last sessions incl. today}."""
        hist = self.history(stock.symbol, stock.key, "5m", now.date() - timedelta(days=history_days), now)
        m1 = self.today_1m(stock.key) if is_trading_day(now.date()) and now.time() >= OPEN else to_bars(None)
        m1 = m1[m1.index + pd.Timedelta(minutes=1) <= now]
        frames = [hist]
        if not m1.empty:
            m5 = resample_bars(m1, 5)
            frames.append(m5[m5.index + pd.Timedelta(minutes=5) <= now])
        frames = [f for f in frames if not f.empty]
        m5 = pd.concat(frames) if frames else to_bars(None)
        return {"1m": m1, "5m": m5[~m5.index.duplicated(keep="last")].sort_index()}

    def daily_bars(self, stock: Stock, now: pd.Timestamp, days: int = 1100, today_1m: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Completed daily bars. Today's bar is appended only after the close (built from the 1-minute candles)."""
        d = self.history(stock.symbol, stock.key, "1d", now.date() - timedelta(days=days), now)
        if today_1m is not None and not today_1m.empty and now.time() >= CLOSE and today_1m.index[-1].date() == now.date():
            day = pd.Timestamp(now.date()).tz_localize(TZ)
            if d.empty or d.index[-1] < day:
                row = pd.DataFrame({"open": today_1m["open"].iloc[0], "high": today_1m["high"].max(),
                                    "low": today_1m["low"].min(), "close": today_1m["close"].iloc[-1],
                                    "volume": today_1m["volume"].sum()}, index=pd.DatetimeIndex([day]))
                d = pd.concat([d, row])
        return d

    def hourly_bars(self, stock: Stock, now: pd.Timestamp, days: int = 100, today_1m: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        h = self.history(stock.symbol, stock.key, "1h", now.date() - timedelta(days=days), now)
        if today_1m is not None and not today_1m.empty:
            t = resample_bars(today_1m, 60)
            close = pd.Timestamp(datetime.combine(now.date(), CLOSE)).tz_localize(TZ)     # the 15:15 bar ends at 15:30
            done = [min(x + pd.Timedelta(minutes=60), close) <= now for x in t.index]
            h = pd.concat([h, t[done]])
        return h[~h.index.duplicated(keep="last")].sort_index()

    # ------------------------------------------------------------------ optional LTP stream
    def start_feed(self, keys: List[str]):
        """Upstox market-data WebSocket (LTP) in its own thread, with the existing reconnect / re-authorise loop."""
        import asyncio

        from src.broker.websocket_feed import WebSocketFeed
        if self.feed is not None:
            return
        self._feed_loop = asyncio.new_event_loop()

        def run():
            asyncio.set_event_loop(self._feed_loop)
            self.feed = WebSocketFeed(self.client, lambda: keys, max_age_seconds=20.0)
            self._feed_loop.run_until_complete(self.feed.run())
        threading.Thread(target=run, daemon=True, name="equity-feed").start()

    def stop_feed(self):
        import asyncio
        if self.feed is not None:
            asyncio.run_coroutine_threadsafe(self.feed.stop(), self._feed_loop)

    def status(self) -> dict:
        out = {"calls": self.calls, "budget": self.budget.status()}
        if self.feed is not None:
            out["websocket"] = self.feed.status()
        return out


INDEX = Stock(symbol="NIFTY", name="NIFTY 50", industry="Index", key=NIFTY_INDEX_KEY, nifty50=False)
