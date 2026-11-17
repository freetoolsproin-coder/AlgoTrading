"""
Market data for the F&O engine.

LiveData   Upstox REST through the shared rate-limited client (token from upstox_token.json / .env):
             instrument master (daily)                      universe, expiries, lot sizes, strikes
             v2 /market-quote/quotes                        every underlying in batches of 450 keys
             v3 /historical-candle/.../minutes/1            1-minute history (once a day per underlying)
             v3 /historical-candle/intraday/.../minutes/1   today's 1-minute candles
             v3 /historical-candle/.../days/1               daily candles (daily trend)
             v2 /option/chain                               live chain with bid / ask / OI / IV / greeks
           Calls are counted; failures, auth errors and the time of the last good answer are reported by
           status() for the dashboard's connection monitor. Nothing is filled in when a call fails.

MockData   a synthetic market for --mock and the tests. Every price it produces is artificial; the engine
           labels the whole dashboard MOCK and stores results in a separate database.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import time as clock
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.broker.market_data import empty_minutes, to_minutes
from src.fno.analysis import attach_futures
from src.fno.config import FnoConfig
from src.fno.universe import Underlying, Universe, load_universe
from src.options.chain import chain_frame
from src.stocks.analytics import bs_greeks, bs_price

logger = logging.getLogger("scanner.fno.data")
TZ = "Asia/Kolkata"
OPEN, CLOSE = time(9, 15), time(15, 30)


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def completed_minutes(m1: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the minute that is still forming."""
    if m1 is None or m1.empty:
        return empty_minutes()
    return m1[m1.index + pd.Timedelta(minutes=1) <= now]


# =============================================================================
# Live (Upstox)
# =============================================================================
class LiveData:
    mock = False

    def __init__(self, cfg: FnoConfig, client=None, universe_loader=load_universe):
        if client is None:
            from src.broker.client import UpstoxClient
            client = UpstoxClient(requests_per_second=cfg.requests_per_second)
        self.cfg, self.client, self.universe_loader = cfg, client, universe_loader
        self._universe: Optional[Universe] = None
        self._day: Optional[date] = None
        self._history: Dict[str, pd.DataFrame] = {}
        self._daily: Dict[str, Optional[pd.DataFrame]] = {}
        self._lock = threading.Lock()
        self.calls = self.failures = self.streak = 0
        self.last_ok: Optional[pd.Timestamp] = None
        self.last_error = ""
        self.auth_ok: Optional[bool] = None

    def now(self) -> pd.Timestamp:
        return now_ist()

    # ---- monitored calls
    def _call(self, fn, *args, **kw):
        from src.broker.client import UpstoxAuthError
        self.calls += 1
        try:
            out = fn(*args, **kw)
        except UpstoxAuthError as error:
            self.failures += 1
            self.streak += 1
            self.auth_ok, self.last_error = False, str(error)
            raise
        except Exception as error:
            self.failures += 1
            self.streak += 1
            self.last_error = f"{type(error).__name__}: {error}"[:200]
            raise
        self.streak, self.auth_ok, self.last_ok = 0, True, now_ist()
        return out

    def status(self) -> dict:
        age = (now_ist() - self.last_ok).total_seconds() if self.last_ok is not None else None
        state = "no token" if self.auth_ok is False else "down" if self.streak >= 5 else \
            "degraded" if self.streak else "ok" if self.last_ok is not None else "starting"
        return {"source": "Upstox REST", "state": state, "calls": self.calls, "failures": self.failures,
                "consecutive_failures": self.streak, "seconds_since_ok": None if age is None else round(age),
                "last_error": self.last_error}

    # ---- reference data
    def universe(self) -> Universe:
        today = self.now().date()
        with self._lock:
            if self._universe is None or self._day != today:
                self._universe = self.universe_loader(self.cfg.indices, today)
                self._day, self._history, self._daily = today, {}, {}
                logger.info("F&O universe: %d indices, %d stocks with options.", len(self._universe.indices),
                            len(self._universe.stocks))
            return self._universe

    def quotes(self, keys: List[str]) -> Dict[str, dict]:
        return self._call(self.client.quotes, keys)

    def _minutes(self, key: str, now: pd.Timestamp) -> pd.DataFrame:
        if key not in self._history:
            end = now.date() - timedelta(days=1)
            start = end - timedelta(days=self.cfg.history_days + 6)
            self._history[key] = to_minutes(self._call(self.client.minute_candles, key, start, end))
        today = to_minutes(self._call(self.client.minute_candles_today, key))
        today = today[today.index.date == now.date()]
        prior = self._history[key]
        out = pd.concat([prior, today]) if len(prior) else today
        return out[~out.index.duplicated(keep="last")].sort_index()

    def minutes(self, und: Underlying, now: pd.Timestamp) -> pd.DataFrame:
        """1-minute candles; an index gets the volume / OI of its near-month futures attached."""
        spot = self._minutes(und.key, now)
        if und.kind == "INDEX":
            fut = und.near_future(now.date())
            if fut:
                try:
                    spot = attach_futures(spot, self._minutes(fut, now))
                except Exception as error:
                    if "Auth" in type(error).__name__:
                        raise
                    logger.warning("[%s] futures candles unavailable (no volume / VWAP this cycle): %s", und.symbol, error)
        return spot

    def daily(self, und: Underlying) -> Optional[pd.DataFrame]:
        if und.symbol not in self._daily:
            end = self.now().date() - timedelta(days=1)
            try:
                self._daily[und.symbol] = self._call(self.client.daily_candles, und.key, end - timedelta(days=200), end)
            except Exception as error:
                if "Auth" in type(error).__name__:
                    raise
                logger.warning("[%s] daily candles unavailable: %s", und.symbol, error)
                return None
        return self._daily[und.symbol]

    def chain(self, und: Underlying, expiry: date) -> pd.DataFrame:
        return chain_frame(self._call(self.client.option_chain, und.key, expiry))

    def option_minutes(self, key: str) -> pd.DataFrame:
        """Today's 1-minute candles of one option contract (real traded premiums)."""
        return to_minutes(self._call(self.client.minute_candles_today, key))


# =============================================================================
# Mock (synthetic, labelled)
# =============================================================================
MOCK_BASE = {"NIFTY": (25000.0, 50, 65, "NSE_INDEX|Nifty 50"), "BANKNIFTY": (57000.0, 100, 30, "NSE_INDEX|Nifty Bank"),
             "FINNIFTY": (27000.0, 50, 60, "NSE_INDEX|Nifty Fin Service"), "SENSEX": (82000.0, 100, 20, "BSE_INDEX|SENSEX"),
             "RELIANCE": (1450.0, 10, 500, "NSE_EQ|MOCKRELIANCE"), "HDFCBANK": (990.0, 5, 1100, "NSE_EQ|MOCKHDFCBANK"),
             "ICICIBANK": (1380.0, 10, 700, "NSE_EQ|MOCKICICIBANK"), "INFY": (1520.0, 20, 400, "NSE_EQ|MOCKINFY"),
             "TCS": (3150.0, 20, 175, "NSE_EQ|MOCKTCS"), "SBIN": (960.0, 5, 750, "NSE_EQ|MOCKSBIN"),
             "LT": (3900.0, 20, 175, "NSE_EQ|MOCKLT"), "ITC": (410.0, 5, 1600, "NSE_EQ|MOCKITC")}


class MockData:
    """Deterministic synthetic market: 1-minute random walks with trend regimes, Black-Scholes chains.
    The values of MOCK_BASE (prices, strike steps, lot sizes) are invented for the simulation only."""
    mock = True

    def __init__(self, cfg: FnoConfig, clock_start: Optional[pd.Timestamp] = None, speed: float = 20.0, days: int = 14,
                 seed: int = 7):
        self.cfg, self.speed, self.days, self.seed = cfg, speed, days, seed
        n = now_ist()
        if clock_start is None:
            d = n.date()
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            clock_start = pd.Timestamp(datetime.combine(d, time(9, 45))).tz_localize(TZ)
        self.start, self._t0 = clock_start, clock.monotonic()
        self.fixed: Optional[pd.Timestamp] = None
        self._series: Dict[str, pd.DataFrame] = {}
        self._contracts: Dict[str, tuple] = {}
        self._universe: Optional[Universe] = None
        self.calls = 0

    def now(self) -> pd.Timestamp:
        if self.fixed is not None:
            return self.fixed
        t = self.start + pd.Timedelta(seconds=(clock.monotonic() - self._t0) * self.speed)
        end = pd.Timestamp(datetime.combine(self.start.date(), time(15, 29))).tz_localize(TZ)
        return min(t, end).floor("s")

    def status(self) -> dict:
        return {"source": "MOCK (synthetic prices)", "state": "ok", "calls": self.calls, "failures": 0,
                "consecutive_failures": 0, "seconds_since_ok": 0, "last_error": ""}

    def _expiries(self, symbol: str, today: date) -> List[date]:
        weekly = {"NIFTY": 1, "SENSEX": 3}.get(symbol)
        out = []
        if weekly is not None:
            d = today
            while len(out) < 2:
                if d.weekday() == weekly:
                    out.append(d)
                d += timedelta(days=1)
        for m in range(2):
            y, mo = today.year + (today.month - 1 + m) // 12, (today.month - 1 + m) % 12 + 1
            last = date(y + mo // 12, mo % 12 + 1, 1) - timedelta(days=1)
            while last.weekday() != 1:
                last -= timedelta(days=1)
            if last >= today and last not in out:
                out.append(last)
        return sorted(out)

    def universe(self) -> Universe:
        if self._universe is not None:
            return self._universe
        today, rows = self.start.date(), []
        for symbol, (base, step, lot, ukey) in MOCK_BASE.items():
            index = ukey.split("|")[0].endswith("INDEX")
            segment = "BSE_FO" if symbol == "SENSEX" else "NSE_FO"
            base = float(self._visible(symbol, self.start)["close"].iloc[-1])   # strikes are listed around the current price
            atm = round(base / step) * step
            for e in self._expiries(symbol, today):
                ms = int(pd.Timestamp(datetime.combine(e, time(23, 59, 59))).tz_localize(TZ).timestamp() * 1000)
                for k in range(-40, 41):
                    for kind in ("CE", "PE"):
                        key = f"{segment}|MOCK{symbol}{e:%y%m%d}{atm + k * step:g}{kind}"
                        self._contracts[key] = (symbol, e, float(atm + k * step), kind)
                        rows.append({"segment": segment, "instrument_type": kind, "instrument_key": key,
                                     "underlying_symbol": symbol, "underlying_key": ukey,
                                     "underlying_type": "INDEX" if index else "EQUITY", "expiry": ms,
                                     "strike_price": float(atm + k * step), "lot_size": lot, "tick_size": 5.0,
                                     "name": f"{symbol} (mock)"})
        self._universe = Universe(rows, today, self.cfg.indices)
        return self._universe

    def _full(self, symbol: str) -> pd.DataFrame:
        if symbol in self._series:
            return self._series[symbol]
        rng = np.random.default_rng(int(hashlib.md5(f"{symbol}{self.seed}".encode()).hexdigest()[:8], 16))
        base = MOCK_BASE[symbol][0]
        days = pd.bdate_range(end=self.start.date(), periods=self.days)
        stamps = pd.DatetimeIndex([pd.Timestamp(datetime.combine(d.date(), OPEN)).tz_localize(TZ) + pd.Timedelta(minutes=i)
                                   for d in days for i in range(375)])
        n = len(stamps)
        sigma = 0.00045 if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX") else 0.0009
        drift = np.repeat(rng.normal(0, sigma * 0.22, n // 90 + 1), 90)[:n]          # ~90-minute trend regimes
        rets = drift + rng.normal(0, sigma, n)
        gaps = rng.normal(0, sigma * 6, len(days))                                   # overnight gaps
        rets[::375] += gaps
        close = base * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[base], close[:-1]])
        open_[::375] *= np.exp(gaps)                                                 # the session OPENS at the gap
        wick = np.abs(rng.normal(0, sigma * 0.6, (2, n))) * close
        vol = rng.lognormal(9.5, 0.6, n) * (1 + 3 * np.abs(rets) / sigma)
        frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick[0],
                              "low": np.minimum(open_, close) - wick[1], "close": close, "volume": vol.round(),
                              "oi": np.nan}, index=stamps)
        self._series[symbol] = frame
        return frame

    def _visible(self, symbol: str, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        return completed_minutes(self._full(symbol), now or self.now())

    def _symbol_of(self, key: str) -> Optional[str]:
        return next((s for s, v in MOCK_BASE.items() if v[3] == key), None)

    def minutes(self, und: Underlying, now: pd.Timestamp) -> pd.DataFrame:
        self.calls += 1
        return self._visible(und.symbol, now)

    def daily(self, und: Underlying) -> Optional[pd.DataFrame]:
        return None

    def _iv(self, symbol: str, strike: float, spot: float) -> float:
        base = 0.13 if symbol in ("NIFTY", "SENSEX", "FINNIFTY") else 0.16 if symbol == "BANKNIFTY" else 0.28
        return base + 0.6 * abs(strike / spot - 1)

    def _years(self, expiry: date, now: pd.Timestamp) -> float:
        end = pd.Timestamp(datetime.combine(expiry, CLOSE)).tz_localize(TZ)
        return max((end - now).total_seconds() / (365 * 86400.0), 1 / (365 * 24.0))

    def _premium(self, key: str, spot: float, now: pd.Timestamp) -> float:
        symbol, expiry, strike, kind = self._contracts[key]
        return max(bs_price(spot, strike, self._years(expiry, now), self._iv(symbol, strike, spot), kind), 0.05)

    def quotes(self, keys: List[str]) -> Dict[str, dict]:
        self.calls += 1
        now, out = self.now(), {}
        for key in keys:
            if key in self._contracts:
                symbol = self._contracts[key][0]
                bars = self._visible(symbol, now)
                if bars.empty:
                    continue
                price = self._premium(key, float(bars["close"].iloc[-1]), now)
                half = max(0.05, price * 0.004)
                out[key] = {"ltp": round(price, 2), "bid": round(price - half, 2), "ask": round(price + half, 2),
                            "trade_time": now}
                continue
            symbol = self._symbol_of(key)
            if symbol is None:
                continue
            bars = self._visible(symbol, now)
            today = bars[bars.index.date == now.date()]
            prior = bars[bars.index.date < now.date()]
            if today.empty or prior.empty:
                continue
            out[key] = {"ltp": float(today["close"].iloc[-1]), "open": float(today["open"].iloc[0]),
                        "high": float(today["high"].max()), "low": float(today["low"].min()),
                        "prev_close": float(prior["close"].iloc[-1]), "volume": float(today["volume"].sum()),
                        "trade_time": now}
        return out

    def chain(self, und: Underlying, expiry: date) -> pd.DataFrame:
        self.calls += 1
        now = self.now()
        bars = self._visible(und.symbol, now)
        spot = float(bars["close"].iloc[-1])
        prev = float(bars["close"].iloc[-11]) if len(bars) > 11 else spot
        years, lot = self._years(expiry, now), und.lot_size(expiry)
        rng = np.random.default_rng(int(hashlib.md5(f"{und.symbol}{expiry}".encode()).hexdigest()[:8], 16))
        rows = []
        for strike in und.strikes.get(expiry, []):
            row = {"strike": strike}
            dist = abs(strike / spot - 1)
            for kind in ("CE", "PE"):
                s = kind.lower()
                iv = self._iv(und.symbol, strike, spot)
                price = max(bs_price(spot, strike, years, iv, kind), 0.05)
                g = bs_greeks(spot, strike, years, iv, kind)
                half = max(0.05, price * 0.004)
                oi = float(rng.integers(2000, 9000)) * lot * math.exp(-dist * 60)
                row.update({f"{s}_ltp": round(price, 2), f"{s}_close": round(max(bs_price(prev, strike, years, iv, kind), 0.05), 2),
                            f"{s}_oi": oi, f"{s}_prev_oi": oi * 0.96, f"{s}_volume": oi * 2.5,
                            f"{s}_bid": round(price - half, 2), f"{s}_ask": round(price + half, 2), f"{s}_iv": round(iv * 100, 2),
                            f"{s}_delta": g["delta"], f"{s}_gamma": g["gamma"], f"{s}_theta": g["theta"], f"{s}_vega": g["vega"],
                            f"{s}_key": und.contracts.get((expiry, strike, kind))})
            rows.append(row)
        frame = pd.DataFrame(rows)
        for s in ("ce", "pe"):
            frame[f"{s}_oi_chg"] = frame[f"{s}_oi"] - frame[f"{s}_prev_oi"]
            frame[f"{s}_px_chg"] = frame[f"{s}_ltp"] - frame[f"{s}_close"]
        return frame

    def option_minutes(self, key: str) -> pd.DataFrame:
        self.calls += 1
        now = self.now()
        symbol = self._contracts[key][0]
        bars = self._visible(symbol, now)
        bars = bars[bars.index.date == now.date()]
        out = empty_minutes().reindex(bars.index)
        for col in ("open", "high", "low", "close"):
            out[col] = [self._premium(key, float(v), t) for t, v in zip(bars.index, bars[col])]
        hi, lo = out[["open", "high", "low", "close"]].max(axis=1), out[["open", "high", "low", "close"]].min(axis=1)
        out["high"], out["low"], out["volume"] = hi, lo, bars["volume"]
        return out
