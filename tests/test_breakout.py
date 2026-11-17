"""NIFTY 50 stock pre-breakout scanner: stages, no look-ahead, plans, equity sizing, paper book, backtest, API."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _breakout_synth import synth  # noqa: E402

from src.execution.costs import EquityCostConfig, equity_intraday_charges, make_cost_fn, make_equity_cost_fn  # noqa: E402
from src.stocks.breakout import (BreakoutBook, BreakoutConfig, Tracker, equity_leg, evaluate,  # noqa: E402
                                 features)

CFG = BreakoutConfig()


def run_day(sign, far=False, cfg=CFG):
    bars, daily, level = synth(sign, far=far)
    f = features(bars, daily, cfg)
    side = "CE" if sign > 0 else "PE"
    tr = Tracker()
    tr.roll(f.index[-1].date())
    out = []
    for i in range(len(f) - 6, len(f)):
        r = evaluate(f, i, cfg)[side]
        r["close"] = float(f["close"].iloc[i])
        tr.update("TEST", r, f.index[i], float(f["atr"].iloc[i]), cfg)
        out.append(r)
    return f, out, level


class StageTests(unittest.TestCase):
    def check_flow(self, sign):
        f, out, level = run_day(sign)
        stages = [r["stage"] for r in out]
        self.assertIn("READY", stages[:-1], stages)                  # suggested BEFORE the move
        last = out[-1]
        self.assertEqual(last["stage"], "CONFIRMED", last["reason"])
        self.assertTrue(last["prior_ready"])
        p = last["plan"]
        self.assertEqual(p["sign"], sign)
        self.assertGreater(sign * (p["trigger"] - p["stop"]), 0)
        self.assertGreater(sign * (p["t1"] - p["trigger"]), 0)
        self.assertGreaterEqual(sign * (p["t2"] - p["t1"]), 0)
        self.assertGreaterEqual(p["rr1"], CFG.min_rr)
        a = p["atr15"]
        self.assertLessEqual(p["risk"], CFG.max_stop_atr * a + 0.01)
        self.assertGreaterEqual(p["risk"], CFG.min_stop_atr * a - 0.01)
        self.assertTrue(all(g["ok"] for g in last["gates"]))
        ready = next(r for r in out if r["stage"] == "READY")
        self.assertGreaterEqual(ready["score"], CFG.ready_score)
        self.assertLessEqual(ready["dist_atr"], CFG.ready_atr)
        self.assertGreaterEqual(ready["families"]["Compression"], CFG.min_compression)

    def test_ce_ready_then_confirmed(self):
        self.check_flow(1)

    def test_pe_ready_then_confirmed(self):
        self.check_flow(-1)

    def test_far_breakout_is_missed_not_chased(self):
        _f, out, _ = run_day(1, far=True)
        self.assertEqual(out[-1]["stage"], "MISSED")

    def test_breakout_without_ready_is_not_marked_prior_ready(self):
        cfg = replace(CFG, ready_score=99.0)              # READY impossible
        _f, out, _ = run_day(1, cfg=cfg)
        self.assertNotIn("READY", [r["stage"] for r in out])
        self.assertEqual(out[-1]["stage"], "CONFIRMED")
        self.assertFalse(out[-1]["prior_ready"])

    def test_ready_outside_entry_window_is_only_watch(self):
        cfg = replace(CFG, entry_end="10:00")
        _f, out, _ = run_day(1, cfg=cfg)
        self.assertNotIn("READY", [r["stage"] for r in out])

    def test_score_is_bounded_and_families_capped(self):
        f, out, _ = run_day(1)
        from src.stocks.breakout import FAMILY_MAX
        for r in out:
            self.assertTrue(0 <= r["score"] <= 100)
            for k, v in r["families"].items():
                self.assertLessEqual(v, FAMILY_MAX[k] + 1e-9)

    def test_features_have_no_look_ahead(self):
        bars, daily, _ = synth(1)
        full = features(bars, daily, CFG)
        cols = [c for c in full.columns if c != "day"]
        for k in (len(bars) - 10, len(bars) - 3, len(bars) - 1):
            part = features(bars.iloc[:k], daily, CFG)
            a = pd.to_numeric(full[cols].iloc[k - 1], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(part[cols].iloc[k - 1], errors="coerce").to_numpy(dtype=float)
            same = np.isclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9)
            self.assertTrue(same.all(), [c for c, ok in zip(cols, same) if not ok])

    def test_evaluate_rows_fast_path_matches(self):
        bars, daily, _ = synth(-1)
        f = features(bars, daily, CFG)
        rows = f.to_dict("records")
        for i in (len(f) - 3, len(f) - 1):
            a, b = evaluate(f, i, CFG), evaluate(f, i, CFG, rows=rows)
            for side in ("CE", "PE"):
                self.assertEqual(a[side]["stage"], b[side]["stage"])
                self.assertEqual(a[side]["score"], b[side]["score"])


class PlanTests(unittest.TestCase):
    def test_equity_leg_respects_risk_budget_and_margin(self):
        plan = {"trigger": 1000.0, "stop": 995.0, "t1": 1007.5, "t2": 1012.5, "sign": 1}
        eq = equity_leg(plan, CFG, make_equity_cost_fn())
        self.assertEqual(eq["qty"], int(CFG.capital * CFG.risk_pct / 100 // 5))
        self.assertLessEqual(eq["qty"] * 5, CFG.capital * CFG.risk_pct / 100)
        self.assertGreater(eq["risk_rs"], eq["qty"] * 5)           # charges included
        tight = dict(plan, stop=999.9)
        eq2 = equity_leg(tight, CFG, make_equity_cost_fn())
        self.assertLessEqual(eq2["qty"] * 1000.0, CFG.capital * CFG.leverage)
        short = {"trigger": 1000.0, "stop": 1005.0, "t1": 992.5, "t2": 987.5, "sign": -1}
        self.assertEqual(equity_leg(short, CFG, make_equity_cost_fn())["action"], "SELL SHORT")

    def test_equity_charges(self):
        c = EquityCostConfig()
        buy = equity_intraday_charges("BUY", 1000, 100, c)          # turnover 1 lakh
        sell = equity_intraday_charges("SELL", 1000, 100, c)
        self.assertAlmostEqual(sell - buy, 100000 * (0.025 - 0.003) / 100, places=2)   # STT vs stamp
        # Rs 20 brokerage (cap) + 2.97 exchange + 0.10 SEBI + 3.00 stamp + 18% GST on the first three
        self.assertAlmostEqual(buy, 20 + 2.97 + 0.10 + 3.00 + 0.18 * (20 + 2.97 + 0.10), places=2)

    def test_option_premium_is_estimated_at_the_trigger(self):
        from src.stocks.data import MockSource
        from src.stocks.engine import choose_expiry
        from src.stocks.signals import option_leg
        src = MockSource(clock=pd.Timestamp("2026-11-12 11:30", tz="Asia/Kolkata"))
        s = src.stocks()[0]
        exps, lots = src.contracts(s.key)
        e = choose_expiry(exps, src.now().date(), 3)
        chain = src.chain(s.key, e)
        spot = float(src.bars15(s.key, src.now())["close"].iloc[-1])
        now_leg, _, _ = option_leg(s.symbol, "CE", spot, spot, spot * 0.99, spot * 1.015, spot * 1.025,
                                   src.daily(s.key), src.now(), chain, e, lots[e])
        trig_leg, _, _ = option_leg(s.symbol, "CE", spot, spot * 1.005, spot * 0.995, spot * 1.015, spot * 1.025,
                                    src.daily(s.key), src.now(), chain, e, lots[e])
        self.assertGreater(trig_leg["entry"], now_leg["entry"])
        self.assertLess(trig_leg["stop"], trig_leg["entry"])
        self.assertLess(trig_leg["entry"], trig_leg["t1"])


class BookTests(unittest.TestCase):
    def item(self, sign=1):
        p = {"trigger": 1000.0, "stop": 995.0 if sign > 0 else 1005.0, "t1": 1000 + sign * 7.5,
             "t2": 1000 + sign * 12.5, "sign": sign, "risk": 5.0}
        return {"symbol": "TEST", "side": "CE" if sign > 0 else "PE", "level": 999.5, "score": 80, "stage": "CONFIRMED",
                "plan": p, "equity": equity_leg(p, CFG, make_equity_cost_fn()),
                "option": {"contract": "TEST NOV 1000 CE", "key": "K", "lot_size": 100, "entry": 20.0,
                           "price_now": 20.0, "delta": 0.5}}

    def test_t1_moves_stop_to_cost_then_exit_at_cost(self):
        b = BreakoutBook(Path(tempfile.mkdtemp()), make_cost_fn(), make_equity_cost_fn(), mock=True)
        now = pd.Timestamp("2026-11-12 11:00", tz="Asia/Kolkata")
        b.roll(now.date())
        t = b.open(self.item(), now, 1000.5)
        self.assertIsNotNone(t)
        self.assertIsNone(b.open(self.item(), now, 1000.5))                 # one trade per stock per day
        b.mark({"TEST": 1008.0}, {"K": 24.0}, now + pd.Timedelta(minutes=15), "15:15")
        self.assertEqual(t["status"], "T1_HIT")
        self.assertEqual(t["stop"], 1000.5)
        self.assertGreater(t["eq_pnl"], 0)
        self.assertGreater(t["opt_pnl"], 0)
        b.mark({"TEST": 1000.0}, {}, now + pd.Timedelta(minutes=30), "15:15")
        self.assertEqual(t["status"], "EXIT_AT_COST")
        self.assertLess(t["eq_pnl"], 0)                                     # charges + 0.5 slippage
        b.save({})
        b2 = BreakoutBook(b.folder, make_cost_fn(), make_equity_cost_fn(), mock=True)
        b2.roll(now.date())
        self.assertEqual(b2.trades["TEST"]["status"], "EXIT_AT_COST")

    def test_short_stop_and_square_off(self):
        b = BreakoutBook(Path(tempfile.mkdtemp()), make_cost_fn(), make_equity_cost_fn(), mock=True)
        now = pd.Timestamp("2026-11-12 11:00", tz="Asia/Kolkata")
        b.roll(now.date())
        t = b.open(self.item(-1), now, 999.5)
        b.mark({"TEST": 1006.0}, {}, now + pd.Timedelta(minutes=15), "15:15")
        self.assertEqual(t["status"], "STOP_HIT")
        self.assertLess(t["eq_pnl"], 0)
        t2 = b.open(dict(self.item(-1), symbol="OTHER"), now, 999.5)
        b.mark({"OTHER": 999.0}, {}, pd.Timestamp("2026-11-12 15:16", tz="Asia/Kolkata"), "15:15")
        self.assertEqual(t2["status"], "TIME_EXIT")


class ConfigAndBacktestTests(unittest.TestCase):
    def test_config_from_ini(self):
        f = Path(tempfile.mkdtemp()) / "x.ini"
        f.write_text("[STOCK_BREAKOUT]\nready_score = 75\nentry_mode = touch\nnotify = true\nmax_trades = 3\n")
        cfg = BreakoutConfig.from_ini(f)
        self.assertEqual((cfg.ready_score, cfg.entry_mode, cfg.notify, cfg.max_trades), (75.0, "touch", True, 3))
        f.write_text("[STOCK_BREAKOUT]\nready_score = 40\nwatch_score = 60\n")
        with self.assertRaises(ValueError):
            BreakoutConfig.from_ini(f)

    def test_shipped_ini_is_valid(self):
        from src.config.settings import CONFIG_FILE
        cfg = BreakoutConfig.from_ini(CONFIG_FILE)
        self.assertTrue(cfg.enabled)

    def test_exit_simulation_is_conservative(self):
        from src.stocks.breakout_backtest import simulate_exit
        idx = pd.date_range("2026-11-12 10:00", periods=4, freq="15min", tz="Asia/Kolkata")
        f = pd.DataFrame({"open": [100, 100, 100, 100], "high": [100, 110, 100, 100], "low": [100, 90, 100, 100],
                          "close": [100, 100, 100, 100]}, index=idx)
        f["day"] = f.index.date
        px, status, _ = simulate_exit(f, 0, 100.0, 1, 95.0, 105.0, 108.0)
        self.assertEqual((px, status), (95.0, "STOP_HIT"))

    def test_backtest_runs_on_mock(self):
        from src.stocks import breakout_backtest as bt
        data, idx = bt.load_mock(None, 6)
        data = dict(list(data.items())[:6])
        trades, readies = bt.run(data, idx, CFG, make_equity_cost_fn())
        for t in trades:
            self.assertIn(t["status"], ("STOP_HIT", "EXIT_AT_COST", "T2_HIT", "TIME_EXIT"))
            self.assertGreater(t["qty"], 0)
        text = bt.report(trades, readies, trades, CFG, "test")
        self.assertIn("System (READY first)", text)


class EngineTests(unittest.TestCase):
    def test_engine_and_api_expose_breakout(self):
        from src.stocks.data import MockSource
        from src.stocks.engine import StockEngine
        from src.stocks.web import create_app
        e = StockEngine(MockSource(clock=pd.Timestamp("2026-11-12 11:30", tz="Asia/Kolkata")),
                        cost_fn=make_cost_fn(), book_dir=Path(tempfile.mkdtemp()))
        e.refresh()
        b = e.state()["breakout"]
        self.assertIsNotNone(b)
        self.assertTrue(b["items"])
        for x in b["items"]:
            if x["stage"] in ("READY", "CONFIRMED") and x.get("plan"):
                p = x["plan"]
                self.assertGreater(p["sign"] * (p["trigger"] - p["stop"]), 0)
                self.assertIn("checks", x)
        client = create_app(e, api_token="t").test_client()
        r = client.get("/api/breakout")
        self.assertEqual(r.status_code, 200)
        self.assertIn("items", r.get_json())
        self.assertIn(b"Before the breakout", client.get("/").data)

    def test_switched_off(self):
        from src.stocks.data import MockSource
        from src.stocks.engine import StockEngine
        e = StockEngine(MockSource(clock=pd.Timestamp("2026-11-12 11:30", tz="Asia/Kolkata")),
                        book_dir=Path(tempfile.mkdtemp()), breakout_cfg=replace(CFG, enabled=False))
        self.assertIsNone(e.breakout)


if __name__ == "__main__":
    unittest.main()
