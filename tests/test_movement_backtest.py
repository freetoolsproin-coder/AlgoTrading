"""
Movement paper-trading backtest (src/movement/backtest.py): replay through the live monitor / paper trader,
modelled option fills, statistics and the out-of-sample split.

    python -m unittest tests.test_movement_backtest
"""
import math
import unittest
from datetime import date

import numpy as np
import pandas as pd

from src.broker.market_data import TZ
from src.movement import backtest as bt
from src.movement.rules import from_ini

DAY1, DAY2 = date(2026, 10, 29), date(2026, 10, 30)
RAMP_START = pd.Timestamp("2026-10-30 10:00", tz=TZ)


def synthetic(ramp_points=8.0, ramp_minutes=15, day2_end="11:00", seed=7):
    """Quiet NIFTY around 24000 (yesterday + today), with a steady rally from 10:00 today, then flat."""
    rng = np.random.default_rng(seed)
    day1 = pd.date_range(pd.Timestamp(f"{DAY1} 12:00", tz=TZ), pd.Timestamp(f"{DAY1} 15:29", tz=TZ), freq="1min")
    day2 = pd.date_range(pd.Timestamp(f"{DAY2} 09:15", tz=TZ), pd.Timestamp(f"{DAY2} {day2_end}", tz=TZ),
                         freq="1min", inclusive="left")
    index = day1.append(day2)
    steps = rng.normal(0, 1.5, len(index))
    ramp = (index >= RAMP_START) & (index < RAMP_START + pd.Timedelta(minutes=ramp_minutes))
    steps[ramp] = ramp_points
    steps[index > RAMP_START + pd.Timedelta(minutes=ramp_minutes)] = 0.0
    closes = 24000 + np.cumsum(steps)
    opens = np.r_[closes[0], closes[:-1]]
    spot = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 1, "low": np.minimum(opens, closes) - 1,
                         "close": closes, "volume": 0.0}, index=index)
    vix = pd.DataFrame({"open": 14.0, "high": 14.0, "low": 14.0, "close": 14.0, "volume": 0.0}, index=index)
    return spot, vix


def run(spot, vix, keep_rvol=False, overrides=None):
    rules, profiles, cfg, costs = bt.backtest_settings(overrides=overrides, keep_rvol=keep_rvol)
    return bt.simulate("NIFTY", spot, vix, rules, profiles, cfg, costs)


class SettingsTests(unittest.TestCase):
    def test_rvol_is_switched_off_for_both_books_unless_kept(self):
        rules, profiles, _, _ = bt.backtest_settings()
        self.assertEqual(rules.min_rvol, 0)
        self.assertEqual(float(profiles["scalp"].triggers["min_rvol"]), 0)
        rules, profiles, _, _ = bt.backtest_settings(keep_rvol=True)
        self.assertEqual(rules.min_rvol, from_ini().min_rvol)
        self.assertGreater(float(profiles["scalp"].triggers["min_rvol"]), 0)

    def test_overrides_apply_to_this_run_only(self):
        _, profiles, _, _ = bt.backtest_settings(overrides=bt.parse_overrides(["PAPER_SCALP.stop_pct=11"]))
        self.assertEqual(profiles["scalp"].stop_pct, 11.0)
        _, profiles, _, _ = bt.backtest_settings()
        self.assertNotEqual(profiles["scalp"].stop_pct, 11.0)

    def test_bad_override(self):
        with self.assertRaises(ValueError):
            bt.parse_overrides(["stop_pct=10"])
        with self.assertRaises(ValueError):
            bt.backtest_settings(overrides=bt.parse_overrides(["PAPER_SCALP.no_such_setting=1"]))


class StatsTests(unittest.TestCase):
    def trades(self, nets):
        start = pd.Timestamp("2026-10-30 10:00", tz=TZ)
        return [{"net": n, "charges": 50.0, "entry_time": (start + pd.Timedelta(minutes=10 * i)).isoformat(),
                 "exit_time": (start + pd.Timedelta(minutes=10 * i + 5)).isoformat()} for i, n in enumerate(nets)]

    def test_stats(self):
        s = bt.stats(self.trades([100, -50, 200, -150, -100, 300]))
        self.assertEqual(s["trades"], 6)
        self.assertAlmostEqual(s["win_rate"], 50.0)
        self.assertAlmostEqual(s["avg_win"], 200.0)
        self.assertAlmostEqual(s["avg_loss"], -100.0)
        self.assertAlmostEqual(s["profit_factor"], 2.0)
        self.assertAlmostEqual(s["net"], 300.0)
        self.assertAlmostEqual(s["max_drawdown"], 250.0)   # 250 peak -> 0 after -150, -100
        self.assertAlmostEqual(s["avg_hold"], 5.0)

    def test_no_losses_and_no_trades(self):
        self.assertTrue(math.isinf(bt.stats(self.trades([10, 20]))["profit_factor"]))
        self.assertEqual(bt.stats([]), {"trades": 0})

    def test_split_is_chronological_70_30(self):
        days = [date(2026, 10, d) for d in range(10, 0, -1)]
        first, last = bt.split_days(days)
        self.assertEqual(len(first), 7)
        self.assertEqual(len(last), 3)
        self.assertLess(max(first), min(last))


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spot, cls.vix = synthetic()
        cls.result = run(cls.spot, cls.vix)

    def test_rally_buys_calls_after_the_move(self):
        trades = self.result["trades"]
        self.assertTrue(any(t["book"] == "momentum" for t in trades), trades)
        self.assertEqual({t["side"] for t in trades}, {"CE"})
        for t in trades:
            entry = pd.Timestamp(t["entry_time"])
            self.assertGreater(entry, RAMP_START)   # never before the move has happened
            self.assertEqual(entry.date(), DAY2)
            self.assertIsNotNone(t["exit_time"])     # all closed the same day
            self.assertEqual(pd.Timestamp(t["exit_time"]).date(), DAY2)
            self.assertAlmostEqual(t["net"], t["gross"] - t["charges"], places=1)
            self.assertGreater(t["charges"], 0)
            self.assertTrue(t["instrument_key"].startswith("NIFTY|"))
        self.assertEqual(self.result["days"], [DAY2])   # the first session only supplies history

    def test_entry_at_the_modelled_ask(self):
        t = self.result["trades"][0]
        und = t["underlying_entry"]
        self.assertAlmostEqual(und, float(self.spot["close"][self.spot.index < pd.Timestamp(t["entry_time"])].iloc[-1]))
        rules, profiles, cfg, costs = bt.backtest_settings()
        provider = bt.ReplayProvider("NIFTY", self.spot, self.vix, cfg)
        ask = provider.model.ask(und, t["strike"], pd.Timestamp(t["entry_time"]), date.fromisoformat(t["expiry"]), True)
        self.assertAlmostEqual(t["entry_price"], round(ask * (1 + cfg.slippage_pct / 100), 2), places=2)
        self.assertGreaterEqual(t["entry_price"], cfg.min_premium)

    def test_no_look_ahead(self):
        """Changing everything after the first entry leaves that entry unchanged."""
        first = self.result["trades"][0]
        entry = pd.Timestamp(first["entry_time"])
        spot = self.spot.copy()
        later = spot.index >= entry
        spot.loc[later, ["open", "high", "low", "close"]] -= 300.0
        again = run(spot, self.vix)["trades"][0]
        for key in ("book", "entry_time", "entry_price", "strike", "side"):
            self.assertEqual(again[key], first[key])

    def test_no_trades_without_a_move_or_with_rvol_kept(self):
        flat, vix = synthetic(ramp_points=0.0)
        self.assertEqual(run(flat, vix)["trades"], [])
        kept = run(self.spot, self.vix, keep_rvol=True)
        self.assertEqual(kept["trades"], [])        # no futures volume: every trigger is UNCONFIRMED
        self.assertTrue(kept["events"])
        self.assertEqual({e["quality"] for e in kept["events"] if e["index"] == "NIFTY"}, {"UNCONFIRMED"})

    def test_report_mentions_the_approximations(self):
        text = bt.report("NIFTY", self.result, DAY2, DAY2, ["momentum", "scalp"], False, {})
        self.assertIn("MODELLED", text)
        self.assertIn("rvol check OFF", text)
        self.assertIn("== MOMENTUM book ==", text)
        self.assertIn("first 70%", text)


if __name__ == "__main__":
    unittest.main()
