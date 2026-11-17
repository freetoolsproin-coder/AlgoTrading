"""
Market data for the stock-options service.

LiveSource  Upstox REST (shared rate limit, token from upstox_token.json / .env):
    v3 /historical-candle/{key}/days/1/{to}/{from}            daily candles (1 year, cached per day)
    v3 /historical-candle/{key}/minutes/15/{to}/{from}        15-minute candles of the last sessions (cached per day)
    v3 /historical-candle/intraday/{key}/minutes/15           today's 15-minute candles (every cycle)
    v2 /market-quote/quotes                                   LTP / OHLC of all 50 stocks in one call
    v2 /option/contract?instrument_key=NSE_EQ|...             expiries + lot size (cached per day)
    v2 /option/chain                                          chain with greeks for one expiry

MockSource  deterministic synthetic prices and Black-Scholes chains - no token needed, any time of day.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote as urlquote

import numpy as np
import pandas as pd

from src.stocks.universe import NIFTY_INDEX_KEY, Stock

logger = logging.getLogger("scanner.stocks.data")
TZ = "Asia/Kolkata"
V3 = "https://api.upstox.com/v3"
OPEN, CLOSE = time(9, 15), time(15, 30)


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def is_market_open(now: pd.Timestamp) -> bool:
    try:
        from src.utils.market_calendar import is_trading_day
        trading = is_trading_day(now.date())
    except Exception:
        trading = now.weekday() < 5
    return trading and OPEN <= now.time() < CLOSE


def to_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """candles_frame (datetime column) -> DataFrame indexed by bar start time."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = frame.set_index("datetime")[["open", "high", "low", "close", "volume"]].copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    clock = out.index.time
    return out[(clock >= OPEN) & (clock < CLOSE)]


class _DayCache:
    def __init__(self):
        self.day, self.data, self.lock = None, {}, threading.Lock()

    def get(self, key, loader):
        today = date.today()
        with self.lock:
            if self.day != today:
                self.day, self.data = today, {}
            if key in self.data:
                return self.data[key]
        value = loader()
        with self.lock:
            self.data[key] = value
        return value


# =============================================================================
# Live (Upstox)
# =============================================================================
class LiveSource:
    mock = False

    def __init__(self, client=None, requests_per_second: float = 5.0):
        if client is None:
            from src.broker.client import UpstoxClient
            client = UpstoxClient(requests_per_second=requests_per_second)
        self.client = client
        self.cache = _DayCache()
        self._stocks: Optional[List[Stock]] = None
        self._equity_keys: Optional[Dict[str, str]] = None

    def now(self) -> pd.Timestamp:
        return now_ist()

    def stocks(self) -> List[Stock]:
        if self._stocks is None:
            from src.stocks.universe import load_universe
            self._stocks = load_universe()
        return self._stocks

    def lookup(self, symbol: str) -> Optional[Stock]:
        """A NIFTY 50 stock, or any NSE equity in the Upstox master (research for non-index stocks)."""
        symbol = symbol.strip().upper()
        for s in self.stocks():
            if s.symbol == symbol:
                return s
        if self._equity_keys is None:
            from src.stocks.universe import master_equity_keys
            self._equity_keys = master_equity_keys()
        key = self._equity_keys.get(symbol)
        return Stock(symbol=symbol, name=symbol, industry="", key=key, nifty50=False) if key else None

    def all_symbols(self) -> List[str]:
        if self._equity_keys is None:
            from src.stocks.universe import master_equity_keys
            self._equity_keys = master_equity_keys()
        return sorted(self._equity_keys)

    # ---- candles
    def daily(self, key: str) -> pd.DataFrame:
        def load():
            end = date.today()
            return self.client.daily_candles(key, end - timedelta(days=400), end)
        return self.cache.get(("daily", key), load)

    def index_daily(self) -> pd.DataFrame:
        return self.daily(NIFTY_INDEX_KEY)

    def _v3(self, path):
        from src.broker.client import candles_frame
        return candles_frame(self.client.get(path, base=V3).get("data", {}).get("candles", []))

    def bars15(self, key: str, now: pd.Timestamp) -> pd.DataFrame:
        def load_history():
            end = date.today() - timedelta(days=1)
            return to_bars(self._v3(f"/historical-candle/{urlquote(key, safe='')}/minutes/15/"
                                    f"{end.isoformat()}/{(end - timedelta(days=14)).isoformat()}"))
        history = self.cache.get(("15m", key), load_history)
        today = pd.DataFrame()
        if now.weekday() < 5 and now.time() >= OPEN:
            try:
                today = to_bars(self._v3(f"/historical-candle/intraday/{urlquote(key, safe='')}/minutes/15"))
            except Exception as error:
                if "Auth" in type(error).__name__:
                    raise
                logger.warning("intraday candles failed for %s: %s", key, error)
        frames = [f for f in (history, today) if f is not None and not f.empty]
        if not frames:
            return to_bars(None)
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep="last")].sort_index()

    # ---- quotes / options
    def quotes(self, keys: List[str]) -> Dict[str, dict]:
        return self.client.quotes(keys)

    def contracts(self, key: str) -> Tuple[List[date], Dict[date, int]]:
        def load():
            data = self.client.get("/option/contract", {"instrument_key": key}).get("data") or []
            expiries, lots = set(), {}
            for row in data:
                try:
                    e = date.fromisoformat(str(row.get("expiry"))[:10])
                except ValueError:
                    continue
                expiries.add(e)
                if row.get("lot_size"):
                    lots[e] = int(float(row["lot_size"]))
            return sorted(expiries), lots
        return self.cache.get(("contracts", key), load)

    def chain(self, key: str, expiry: date) -> pd.DataFrame:
        from src.options.chain import chain_frame
        return chain_frame(self.client.option_chain(key, expiry))


# =============================================================================
# Mock
# =============================================================================
MOCK_PRICES = {"RELIANCE": 1450, "TCS": 3150, "HDFCBANK": 990, "ICICIBANK": 1380, "INFY": 1520, "SBIN": 960,
               "ITC": 410, "LT": 3900, "BHARTIARTL": 2050, "KOTAKBANK": 2100, "AXISBANK": 1230, "MARUTI": 16000}


class MockSource:
    """Synthetic but internally consistent market: 15-minute random walks with trend regimes per stock,
    daily candles built from them, Black-Scholes chains around the spot. Same seed -> same market."""
    mock = True

    def __init__(self, clock: Optional[pd.Timestamp] = None, days: int = 300):
        self.fixed_now = clock
        self.days = days
        self._cache: Dict[str, pd.DataFrame] = {}
        self._options: Dict[str, tuple] = {}       # option key -> (underlying key, strike, kind, expiry, iv)
        from src.stocks.universe import FALLBACK_SYMBOLS
        self._stocks = [Stock(symbol=s, name=f"{s.title()} (mock)", industry="Mock", key=f"NSE_EQ|MOCK{i:07d}X")
                        for i, s in enumerate(FALLBACK_SYMBOLS)]

    def now(self) -> pd.Timestamp:
        if self.fixed_now is not None:
            return self.fixed_now
        n = now_ist()
        if n.weekday() >= 5 or not (time(9, 45) <= n.time() < time(15, 30)):     # replay a session at 11:30
            d = n.date() if n.weekday() < 5 and n.time() >= CLOSE else n.date() - timedelta(days=1)
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            n = pd.Timestamp(datetime.combine(d, time(11, 30))).tz_localize(TZ)
        return n.floor("min")

    def stocks(self) -> List[Stock]:
        return self._stocks

    def lookup(self, symbol: str) -> Optional[Stock]:
        return next((s for s in self._stocks if s.symbol == symbol.strip().upper()), None)

    def all_symbols(self) -> List[str]:
        return [s.symbol for s in self._stocks]

    def _seed(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)

    def _series(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]
        rng = np.random.default_rng(self._seed(key))
        sym = next((s.symbol for s in self._stocks if s.key == key), "NIFTY")
        base = MOCK_PRICES.get(sym, 22000 if key == NIFTY_INDEX_KEY else float(rng.uniform(300, 4000)))
        end_day = self.now().date()
        days = pd.bdate_range(end=end_day, periods=self.days)
        stamps = [pd.Timestamp(datetime.combine(d.date(), OPEN)).tz_localize(TZ) + pd.Timedelta(minutes=15 * i)
                  for d in days for i in range(25)]
        n = len(stamps)
        drift = np.repeat(rng.normal(0, 0.00035, n // 75 + 1), 75)[:n]          # regimes of ~3 sessions
        rets = drift + rng.normal(0, 0.0028, n)
        close = base * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[base], close[:-1]]) * (1 + rng.normal(0, 0.0008, n))
        wick = np.abs(rng.normal(0, 0.0025, n)) * close
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0025, n)) * close
        vol = rng.lognormal(11, 0.5, n) * (1 + 2 * np.abs(rets) / 0.0028)
        frame = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol.round()},
                             index=pd.DatetimeIndex(stamps))
        self._cache[key] = frame
        return frame

    def _visible(self, key: str) -> pd.DataFrame:
        s = self._series(key)
        return s[s.index + pd.Timedelta(minutes=15) <= self.now()]

    def daily(self, key: str) -> pd.DataFrame:
        s = self._visible(key)
        g = s.groupby(s.index.date)
        d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                          "close": g["close"].last(), "volume": g["volume"].sum()})
        d["datetime"] = [pd.Timestamp(x).tz_localize(TZ) for x in d.index]
        return d.reset_index(drop=True)[["datetime", "open", "high", "low", "close", "volume"]]

    def index_daily(self) -> pd.DataFrame:
        return self.daily(NIFTY_INDEX_KEY)

    def bars15(self, key: str, now: pd.Timestamp) -> pd.DataFrame:
        s = self._visible(key)
        return s[s.index.normalize() >= (now.normalize() - pd.Timedelta(days=16))]

    def _option_quote(self, key: str) -> Optional[dict]:
        from src.stocks.analytics import bs_price
        underlying, strike, kind, expiry, iv = self._options[key]
        spot = float(self._visible(underlying)["close"].iloc[-1])
        years = max((pd.Timestamp(datetime.combine(expiry, CLOSE)).tz_localize(TZ) - self.now()).total_seconds()
                    / (365 * 86400), 1 / 365)
        return {"ltp": round(bs_price(spot, strike, years, iv, kind), 2)}

    def quotes(self, keys: List[str]) -> Dict[str, dict]:
        out = {}
        for k in keys:
            if k in self._options:
                out[k] = self._option_quote(k)
                continue
            if k.startswith("NSE_FO|"):            # contract of an earlier run: unknown to this mock
                continue
            s = self._visible(k)
            d = self.daily(k)
            if s.empty or len(d) < 2:
                continue
            today = s[s.index.date == self.now().date()]
            out[k] = {"ltp": float(s["close"].iloc[-1]), "prev_close": float(d["close"].iloc[-2]),
                      "open": float(today["open"].iloc[0]) if not today.empty else float("nan"),
                      "high": float(today["high"].max()) if not today.empty else float("nan"),
                      "low": float(today["low"].min()) if not today.empty else float("nan"),
                      "volume": float(today["volume"].sum()), "symbol": None}
        return out

    def contracts(self, key: str):
        today = self.now().date()
        exps = []
        for m in range(3):
            y, mo = today.year + (today.month - 1 + m) // 12, (today.month - 1 + m) % 12 + 1
            last = (date(y + mo // 12, mo % 12 + 1, 1) - timedelta(days=1))
            while last.weekday() != 1:              # last Tuesday (NSE monthly stock expiry)
                last -= timedelta(days=1)
            if last >= today:
                exps.append(last)
        spot = float(self._visible(key)["close"].iloc[-1])
        lot = int(max(1, round(250000 / spot / 25) * 25)) if spot < 10000 else 25
        return exps, {e: lot for e in exps}

    def chain(self, key: str, expiry: date) -> pd.DataFrame:
        from src.stocks.analytics import bs_greeks, bs_price
        rng = np.random.default_rng(self._seed(key + expiry.isoformat()))
        spot = float(self._visible(key)["close"].iloc[-1])
        step = 5 if spot < 500 else 10 if spot < 1000 else 20 if spot < 2500 else 50 if spot < 5000 else 100
        atm = round(spot / step) * step
        years = max((pd.Timestamp(datetime.combine(expiry, CLOSE)).tz_localize(TZ) - self.now()).total_seconds()
                    / (365 * 86400), 1 / 365)
        rows = []
        for k in range(-12, 13):
            strike = atm + k * step
            row = {"strike": float(strike)}
            for kind in ("ce", "pe"):
                iv = 0.24 + 0.02 * abs(k) / 6 + rng.normal(0, 0.005)
                price = bs_price(spot, strike, years, iv, kind.upper())
                g = bs_greeks(spot, strike, years, iv, kind.upper())
                spread = max(0.05, price * 0.006)
                oi = float(rng.integers(50, 600) * 1000 * math.exp(-abs(k) / 6))
                okey = f"NSE_FO|MOCK{self._seed(f'{key}{expiry}{strike}{kind}') % 10**7}"
                self._options[okey] = (key, float(strike), kind.upper(), expiry, iv)
                row.update({f"{kind}_ltp": round(price, 2), f"{kind}_close": round(price * 0.97, 2), f"{kind}_oi": oi,
                            f"{kind}_prev_oi": oi * float(rng.uniform(0.85, 1.1)), f"{kind}_volume": oi * 3,
                            f"{kind}_bid": round(price - spread / 2, 2), f"{kind}_ask": round(price + spread / 2, 2),
                            f"{kind}_iv": round(iv * 100, 2), f"{kind}_delta": g["delta"], f"{kind}_gamma": g["gamma"],
                            f"{kind}_theta": g["theta"], f"{kind}_vega": g["vega"],
                            f"{kind}_key": okey})
            rows.append(row)
        frame = pd.DataFrame(rows)
        for side in ("ce", "pe"):
            frame[f"{side}_oi_chg"] = frame[f"{side}_oi"] - frame[f"{side}_prev_oi"]
            frame[f"{side}_px_chg"] = frame[f"{side}_ltp"] - frame[f"{side}_close"]
        return frame
