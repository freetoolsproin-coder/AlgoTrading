"""
Strategy library (src/movement/strategies.py): individual rules on built scenarios, signal de-duplication,
config loading, and a full mock session run through every strategy without errors.

    python -m unittest tests.test_strategies
"""
import tempfile
import unittest
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.broker.market_data import TZ, IndexSnapshot, MarketSnapshot
from src.broker.mock_data import MockProvider
from src.config.settings import PBConfig
from src.movement import strategies as S

DAY = pd.Timestamp("2026-11-03 09:15", tz=TZ)


def bars_1m(closes, start, volume=1000.0, oi=None, spread=1.0):
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread, "close": c,
                         "volume": volume if np.isscalar(volume) else np.asarray(volume, float),
                         "oi": oi if oi is not None else np.full(len(c), 1e6)}, index=idx)


def session(path_today, prev_days=2, base=24000.0, seed=1):
    """Previous sessions of gentle noise + today's 1-min closes."""
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(prev_days, 0, -1):
        closes = base + np.cumsum(rng.normal(0, 2, 375))
        frames.append(bars_1m(closes, DAY - pd.Timedelta(days=d)))
    frames.append(bars_1m(path_today, DAY))
    return pd.concat(frames)


def snap_of(m1, now, fut=None, chain=None, name="NIFTY"):
    daily = S.ind.daily_from_minutes(m1[m1.index.date < now.date()])
    return IndexSnapshot(name=name, spot_minutes=m1, fut_minutes=fut if fut is not None else m1, daily=daily,
                         ltp=float(m1["close"].iloc[-1]), ltp_time=now, chain=chain, step=50.0)


def ctx_for(m1, now, fut=None, memory=None, tf=5):
    eng = S.StrategyEngine(list(S.REGISTRY), {}, tf)
    c = eng.context(snap_of(m1, now, fut), now)
    if memory is not None:
        c.memory = memory
    return c


def p(key, **kw):
    return dict(S.REGISTRY[key].params, **kw)


class StrategyRuleTests(unittest.TestCase):
    def test_supertrend_flips(self):
        path = list(24000 - np.arange(120) * 3.0) + list(23640 + np.arange(1, 61) * 6.0)
        m1 = session(path)
        found = set()
        for minutes in range(120, 180, 5):
            now = DAY + pd.Timedelta(minutes=minutes)
            r = S.supertrend_flip(ctx_for(m1[m1.index < now], now), p("supertrend"))
            found.add(r.direction)
        self.assertIn("UP", found)

    def test_orb_first_break_only(self):
        path = [24000.0] * 15 + [24000 + i * 4 for i in range(1, 30)]
        m1 = session(path)
        now = DAY + pd.Timedelta(minutes=20)                 # the 09:30 bar is the first close above the range
        first = S.orb(ctx_for(m1[m1.index < now], now), p("orb"))
        self.assertEqual(first.direction, "UP")
        later = DAY + pd.Timedelta(minutes=40)
        self.assertIsNone(S.orb(ctx_for(m1[m1.index < later], later), p("orb")).direction)
        early = DAY + pd.Timedelta(minutes=10)
        self.assertIn("not complete", S.orb(ctx_for(m1[m1.index < early], early), p("orb")).reason)

    def test_atr_breakout(self):
        path = [24000.0] * 10 + [24000 + i * 10 for i in range(1, 40)]
        m1 = session(path, prev_days=20)
        fired = set()
        for minutes in range(15, 50, 5):
            now = DAY + pd.Timedelta(minutes=minutes)
            fired.add(S.atr_breakout(ctx_for(m1[m1.index < now], now), p("atr_breakout", k=0.3)).direction)
        self.assertIn("UP", fired)

    def test_first_hour(self):
        path = [24000 + i * 2 for i in range(60)] + [24120 + i * 3 for i in range(30)]
        m1 = session(path)
        fired = set()
        for minutes in range(65, 90, 5):
            now = DAY + pd.Timedelta(minutes=minutes)
            fired.add(S.first_hour(ctx_for(m1[m1.index < now], now), p("first_hour")).direction)
        self.assertIn("UP", fired)

    def test_oi_buildup(self):
        path = [24000.0] * 30 + [24000 + i * 3 for i in range(1, 16)]
        oi = np.r_[np.full(30, 1e6), np.linspace(1.002e6, 1.03e6, 15)]
        m1 = session(path)
        fut = m1.copy()
        fut.loc[fut.index >= DAY, "oi"] = oi
        now = DAY + pd.Timedelta(minutes=45)
        r = S.oi_buildup(ctx_for(m1, now, fut), p("oi_buildup"))
        self.assertEqual(r.direction, "UP")
        self.assertIn("long build-up", r.reason)

    def test_no_futures_volume_is_no_data(self):
        m1 = session([24000.0] * 60)
        fut = m1.assign(volume=0.0)
        now = DAY + pd.Timedelta(minutes=60)
        c = ctx_for(m1, now, fut)
        self.assertIn("no data", S.vwap_reversal(c, p("vwap_reversal")).reason)
        self.assertIn("no data", S.volume_profile(c, p("volume_profile")).reason)

    def test_volume_profile_levels(self):
        closes = np.r_[np.full(100, 100.0), np.linspace(100, 110, 50)]
        vp = S.volume_profile_levels(bars_1m(closes, DAY, spread=0.2), bins=40)
        self.assertLess(abs(vp["poc"] - 100), 1.0)
        self.assertLessEqual(vp["val"], vp["poc"])
        self.assertGreaterEqual(vp["vah"], vp["poc"])

    def test_pcr_and_premium_need_history(self):
        m1 = session([24000.0] * 60)
        now = DAY + pd.Timedelta(minutes=60)
        mem = S.Memory()
        c = ctx_for(m1, now, memory=mem)
        self.assertIn("no data", S.pcr(c, p("pcr")).reason)
        mem.pcr.extend([(now - pd.Timedelta(minutes=35), 0.95), (now, 1.10)])
        c.b5.loc[c.b5.index[-1], "close"] = c.b5["ema20"].iloc[-1] + 5
        self.assertEqual(S.pcr(c, p("pcr")).direction, "UP")
        mem.premiums.extend([(now - pd.Timedelta(minutes=11), {"atm": 24000.0, "legs": {24000.0: (100.0, 100.0)}}),
                             (now, {"atm": 24000.0, "legs": {24000.0: (112.0, 92.0)}})])
        self.assertEqual(S.premium_momentum(c, p("premium_momentum")).direction, "UP")

    def test_bollinger_needs_a_range(self):
        rng = np.random.default_rng(3)
        path = list(24000 + rng.normal(0, 3, 150))
        m1 = session(path)
        now = DAY + pd.Timedelta(minutes=150)
        r = S.bollinger_reversion(ctx_for(m1, now), p("bollinger_reversion"))
        self.assertTrue(r.reason)


class EngineTests(unittest.TestCase):
    def test_signal_fires_once_per_bar(self):
        eng = S.StrategyEngine(["orb"], {})
        path = [24000.0] * 15 + [24000 + i * 4 for i in range(1, 30)]
        m1 = session(path)
        now = DAY + pd.Timedelta(minutes=20)
        snap = MarketSnapshot(time=now, indices={"NIFTY": snap_of(m1[m1.index < now], now)})
        self.assertEqual(len(eng.evaluate(snap, now)), 1)
        self.assertEqual(eng.evaluate(snap, now + pd.Timedelta(seconds=10)), [])
        self.assertEqual(eng.evaluate(snap, now, stale={"NIFTY"}), [])

    def test_a_failing_strategy_does_not_stop_the_others(self):
        eng = S.StrategyEngine(["orb", "supertrend"], {})
        eng.params["orb"] = {}          # broken parameters -> KeyError inside the rule
        m1 = session([24000.0] * 60)
        now = DAY + pd.Timedelta(minutes=60)
        eng.evaluate(MarketSnapshot(time=now, indices={"NIFTY": snap_of(m1, now)}), now)
        self.assertTrue(eng.last["orb"]["NIFTY"]["reason"].startswith("error"))
        self.assertFalse(eng.last["supertrend"]["NIFTY"]["reason"].startswith("error"))

    def test_config(self):
        eng, profiles = S.load_config()
        self.assertEqual(len(eng.enabled), len(S.REGISTRY))
        self.assertEqual(profiles["orb"].entry_windows, ["SIGNAL"])
        with self.assertRaises(ValueError):
            S.coerce_params("orb", {"nonsense": "1"})
        trig = S.to_trigger({"index": "NIFTY", "direction": "UP", "title": "ORB", "reason": "x"}, 24000.0)
        self.assertEqual(trig["window"], "SIGNAL")

    def test_full_mock_session_without_errors(self):
        cfg = PBConfig()
        provider = MockProvider(cfg, ["NIFTY", "SENSEX"], speed=1.0, start_clock=time(9, 15), seed=11)
        eng = S.StrategyEngine(list(S.REGISTRY), {})
        start = provider.start_sim
        signals = []
        for minute in range(5, 370, 5):
            now = start + pd.Timedelta(minutes=minute, seconds=6)
            snap = provider.snapshot(now)
            signals += eng.evaluate(snap, now)
            errors = [(k, i, r["reason"]) for k, d in eng.last.items() for i, r in d.items()
                      if r["reason"].startswith("error")]
            self.assertEqual(errors, [], f"at {now}")
        self.assertGreater(len(signals), 5)
        self.assertGreater(len({s["strategy"] for s in signals}), 5)


if __name__ == "__main__":
    unittest.main()
