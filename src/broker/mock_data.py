"""
Synthetic NIFTY / SENSEX markets for trying the scanner without credentials (`python -m src.main --mock`).

Consolidations that end in breakouts, near-month futures with volume and OI, a Black-Scholes option chain
with bid/ask, OI, volume and greeks, India VIX, breadth, and a fast clock (`speed` simulated seconds per
real second, starting at 09:15). Everything it produces is labelled MOCK, and mock paper trades never mix
with live ones in the database. The contract specifications below are NOT real ones.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from src.broker.market_data import EMPTY, TZ, IndexSnapshot, MarketSnapshot
from src.config.instruments import INSTRUMENTS, get_instrument
from src.config.settings import PBConfig
from src.options.chain import choose_expiry
from src.options.selection import bs_greeks, years_to_expiry
from src.scanner import indicators as ind
from src.utils import market_calendar

MOCK_SPECS = {  # synthetic only - NOT real contract specifications
    "NIFTY": {"base": 24800.0, "step": 50, "lot": 65, "vol": 0.11, "fut_vol": 9000, "weekday": 1},
    "SENSEX": {"base": 81500.0, "step": 100, "lot": 20, "vol": 0.11, "fut_vol": 1500, "weekday": 3},
}


class MockProvider:
    """Deterministic synthetic markets. Clock: `speed` simulated seconds per real second, starting at 09:15."""
    mock = True

    def __init__(self, cfg: PBConfig, indices=None, speed: float = 60.0, seed: int = 7,
                 session: Optional[date] = None, start_clock: time = time(9, 15)):
        self.cfg = cfg
        self.speed = speed
        self.names = [get_instrument(n).name for n in (indices or MOCK_SPECS)]
        today = session or pd.Timestamp.now(tz=TZ).date()
        if not market_calendar.is_trading_day(today):
            today = market_calendar.previous_trading_day(today)
        self.session = today
        self.start_real = pd.Timestamp.now(tz=TZ)
        self.start_sim = pd.Timestamp(datetime.combine(today, start_clock)).tz_localize(TZ)
        self.rng_seed = seed
        self.days = [today]
        d = today
        for _ in range(12):
            d = market_calendar.previous_trading_day(d)
            self.days.insert(0, d)
        self.data = {n: self._generate(n, i) for i, n in enumerate(self.names)}
        self.vix_path = self._vix_path()

    def now(self) -> pd.Timestamp:
        elapsed = (pd.Timestamp.now(tz=TZ) - self.start_real).total_seconds() * self.speed
        return self.start_sim + pd.Timedelta(seconds=elapsed)

    def _generate(self, name: str, k: int):
        spec = MOCK_SPECS[name]
        rng = np.random.default_rng(self.rng_seed * 100 + k)
        rows_spot, rows_fut = [], []
        price = spec["base"]
        oi = 1.2e7 / spec["lot"]
        sigma_min = spec["vol"] / math.sqrt(252 * 375)
        for day in self.days:
            gap = rng.normal(0, 0.003)
            price *= 1 + gap
            t0 = pd.Timestamp(datetime.combine(day, time(9, 15))).tz_localize(TZ)
            # regimes: consolidation blocks that end with a breakout burst, then drift
            regime, left, anchor, burst_dir = "drift", 0, price, 1
            for minute in range(375):
                if left <= 0:
                    regime = rng.choice(["consolidate", "drift", "trend"], p=[0.45, 0.35, 0.20])
                    left = int(rng.integers(20, 70))
                    anchor, burst_dir = price, (1 if rng.random() < 0.5 else -1)
                left -= 1
                if regime == "consolidate":
                    vol = sigma_min * 0.55
                    drift = -0.08 * (price - anchor) / price
                    if left < 6:  # the breakout burst at the end of a consolidation
                        vol, drift = sigma_min * 1.8, burst_dir * sigma_min * 1.6
                elif regime == "trend":
                    vol, drift = sigma_min * 1.1, burst_dir * sigma_min * 0.45
                else:
                    vol, drift = sigma_min, 0.0
                ret = drift + rng.normal(0, vol)
                o = price
                c = price * (1 + ret)
                hi = max(o, c) * (1 + abs(rng.normal(0, vol * 0.6)))
                lo = min(o, c) * (1 - abs(rng.normal(0, vol * 0.6)))
                t = t0 + pd.Timedelta(minutes=minute)
                rows_spot.append((t, o, hi, lo, c, 0.0, np.nan))
                burst = regime == "consolidate" and left < 6 or regime == "trend"
                base_vol = spec["fut_vol"] * (1.6 if minute < 30 or minute > 345 else 1.0)
                v = max(1.0, rng.gamma(4.0, base_vol / 4.0) * (2.2 if burst else (0.6 if regime == "consolidate" else 1.0)))
                oi += (v * 0.08 if burst else -v * 0.01) * (1 if rng.random() < 0.6 else -1)
                basis = 1 + 0.0009
                rows_fut.append((t, o * basis, hi * basis, lo * basis, c * basis, v, oi))
                price = c
        cols = ["datetime"] + EMPTY
        spot = pd.DataFrame(rows_spot, columns=cols).set_index("datetime")
        fut = pd.DataFrame(rows_fut, columns=cols).set_index("datetime")
        return {"spot": spot, "fut": fut}

    def _vix_path(self):
        rng = np.random.default_rng(self.rng_seed + 999)
        return 13.5 + np.cumsum(rng.normal(0, 0.02, 375 * len(self.days)))

    def _expiries(self, name: str, today: date) -> List[date]:
        wd = MOCK_SPECS[name]["weekday"]
        first = today + timedelta(days=(wd - today.weekday()) % 7)
        return [first + timedelta(days=7 * i) for i in range(4)]

    def snapshot(self, now: Optional[pd.Timestamp] = None, want_chain: Optional[set] = None) -> MarketSnapshot:
        now = now or self.now()
        out = {}
        day = self.session
        for name in self.names:
            spec = MOCK_SPECS[name]
            spot = self.data[name]["spot"]
            fut = self.data[name]["fut"]
            visible = spot[spot.index + pd.Timedelta(minutes=1) <= now]
            fvisible = fut[fut.index + pd.Timedelta(minutes=1) <= now]
            daily = ind.daily_from_minutes(spot[spot.index.date < day])
            today = visible[visible.index.date == day]
            ltp = float(visible["close"].iloc[-1]) if len(visible) else spec["base"]
            snap = IndexSnapshot(name=name, spot_minutes=visible, fut_minutes=fvisible, daily=daily, ltp=ltp,
                                 ltp_time=(visible.index[-1] + pd.Timedelta(minutes=1)) if len(visible) else now,
                                 day_open=float(today["open"].iloc[0]) if len(today) else np.nan,
                                 prev_close=float(daily["close"].iloc[-1]) if len(daily) else np.nan,
                                 exchange=INSTRUMENTS[name].fo_segment, mock=True,
                                 sources={"spot": "MOCK", "futures": "MOCK"})
            expiries = self._expiries(name, day)
            snap.expiry, snap.nearest_expiry = choose_expiry(expiries, now, self.cfg)
            snap.lot_size = spec["lot"]
            if want_chain is None or name in want_chain:
                snap.chain = self._chain(name, ltp, now, snap.expiry)
                snap.chain_time, snap.chain_source = now, "MOCK"
                snap.step = float(spec["step"])
            out[name] = snap
        i = min(len(self.vix_path) - 1, max(0, int((now - self.start_sim).total_seconds() // 60) + 375 * (len(self.days) - 1)))
        breadth = {}
        for name, snap in out.items():  # synthetic: follows the index's last 15-minute move
            m = snap.spot_minutes
            if len(m) > 16:
                move = (m["close"].iloc[-1] / m["close"].iloc[-16] - 1) * 100
                pct = float(min(95, max(5, 50 + move * 60)))
                breadth[name] = {"pct_above_vwap": pct, "pct_advancing": pct, "count": 50, "proxy": False}
        return MarketSnapshot(time=now, indices=out, vix=float(self.vix_path[i]),
                              vix_prev_close=float(self.vix_path[375 * (len(self.days) - 1) - 1]), mock=True,
                              breadth=breadth)

    def _chain(self, name, spot, now, expiry) -> pd.DataFrame:
        spec = MOCK_SPECS[name]
        rng = np.random.default_rng(int(now.timestamp()) // 60 + sum(map(ord, name)))
        step = spec["step"]
        atm = round(spot / step) * step
        years = years_to_expiry(now, expiry)
        rows = []
        for k in range(-8, 9):
            strike = atm + k * step
            row = {"strike": float(strike)}
            for side, is_call in (("ce", True), ("pe", False)):
                moneyness = (strike - spot) / spot
                iv = 13.0 + 40 * moneyness ** 2 * 100 + (1.0 if not is_call else 0.0)
                price, delta, gamma, theta, vega = bs_greeks(spot, strike, years, iv / 100, is_call)
                price = max(price, 0.05)
                spread = max(0.05, round(price * (0.004 + 0.002 * abs(k)), 2))
                liquidity = math.exp(-abs(k) / 3.0)
                row.update({f"{side}_ltp": round(price, 2), f"{side}_close": price, f"{side}_bid": round(max(0.05, price - spread / 2), 2),
                            f"{side}_ask": round(price + spread / 2, 2), f"{side}_iv": iv, f"{side}_delta": delta,
                            f"{side}_gamma": gamma, f"{side}_theta": theta, f"{side}_vega": vega,
                            f"{side}_volume": float(int(rng.gamma(3, 3e5 * liquidity))),
                            f"{side}_oi": float(int(rng.gamma(4, 2e6 * liquidity))), f"{side}_prev_oi": np.nan,
                            f"{side}_key": f"MOCK|{name}|{int(strike)}|{'CE' if is_call else 'PE'}"})
            rows.append(row)
        return pd.DataFrame(rows)
