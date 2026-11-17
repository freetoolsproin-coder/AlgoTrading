"""NIFTY 50 stock-options service: plan validity, paper book, research report and web API (mock data)."""
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.execution.costs import make_cost_fn
from src.stocks.analytics import chain_stats, classic_pivots, supertrend
from src.stocks.data import MockSource
from src.stocks.engine import PaperBook, StockEngine, choose_expiry
from src.stocks.precision import PrecisionConfig, evaluate, nifty_context, select
from src.stocks.signals import StockConfig, build_plan, completed
from src.stocks.universe import parse_nifty50_csv

CLOCK = pd.Timestamp("2026-11-12 11:30", tz="Asia/Kolkata")


def engine():
    return StockEngine(MockSource(clock=CLOCK), cost_fn=make_cost_fn(), book_dir=Path(tempfile.mkdtemp()))


class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = engine()
        cls.e.refresh()
        cls.state = cls.e.state()

    def test_all_stocks_scanned_without_errors(self):
        self.assertEqual(self.state["errors"], {})
        self.assertEqual(len(self.state["plans"]), 50)

    def test_valid_plans_have_consistent_levels(self):
        valid = [p for p in self.state["plans"] if p["status"] == "VALID"]
        self.assertTrue(valid, "mock market should produce at least one valid plan")
        cfg = StockConfig()
        for p in valid:
            s, o = p["spot"], p["option"]
            sign = 1 if p["direction"] == "CE" else -1
            self.assertGreater(sign * (s["entry"] - s["stop"]), 0, p["symbol"])
            self.assertGreater(sign * (s["t1"] - s["entry"]), 0, p["symbol"])
            self.assertGreaterEqual(sign * (s["t2"] - s["t1"]), 0, p["symbol"])
            self.assertGreaterEqual(s["rr1"], cfg.min_rr - 0.01)
            self.assertLess(o["stop"], o["entry"])                       # long option: SL below entry
            self.assertLess(o["entry"], o["t1"])
            self.assertLessEqual(o["t1"], o["t2"])
            self.assertGreaterEqual(o["stop"], round(o["entry"] * (1 - cfg.max_premium_loss_pct / 100), 1) - 0.1)
            self.assertGreaterEqual(o["rr"], cfg.min_rr)
            self.assertEqual(o["kind"], p["direction"])
            self.assertTrue(all(c["ok"] for c in p["checks"]))
            self.assertTrue(o["contract"].startswith(p["symbol"]) and o["contract"].endswith(p["direction"]))

    def test_watch_plans_name_a_failed_check(self):
        for p in self.state["plans"]:
            if p["status"] == "WATCH" and p["checks"]:
                self.assertTrue(any(not c["ok"] for c in p["checks"]) or abs(p["score"]) < StockConfig().min_score)

    def test_valid_plans_open_paper_trades(self):
        # precision mode: only top picks (VALID + every precision check) are paper-traded, within the caps
        picks = {p["symbol"] for p in self.state["plans"] if (p.get("precision") or {}).get("ok")}
        traded = {t["symbol"] for t in self.state["book"]["trades"]}
        self.assertLessEqual(traded, picks)
        self.assertLessEqual(len(traded), PrecisionConfig().max_open)
        for p in self.state["plans"]:
            self.assertEqual(p.get("precision") is not None, p["status"] == "VALID", p["symbol"])

    def test_precision_off_trades_every_valid_plan(self):
        e = StockEngine(MockSource(clock=CLOCK), cost_fn=make_cost_fn(), book_dir=Path(tempfile.mkdtemp()),
                        precision_cfg=PrecisionConfig(enabled=False))
        e.refresh()
        st = e.state()
        valid = {p["symbol"] for p in st["plans"] if p["status"] == "VALID"}
        self.assertEqual({t["symbol"] for t in st["book"]["trades"]}, valid)

    def test_outside_entry_window_is_never_valid(self):
        src = MockSource(clock=pd.Timestamp("2026-11-12 15:00", tz="Asia/Kolkata"))
        e = StockEngine(src, book_dir=Path(tempfile.mkdtemp()))
        e.refresh()
        self.assertEqual(e.state()["counts"]["VALID"], 0)

    def test_forming_bar_is_ignored(self):
        src = MockSource(clock=CLOCK)
        bars = src.bars15(src.stocks()[0].key, CLOCK)
        self.assertTrue((completed(bars, CLOCK - pd.Timedelta(minutes=5)).index[-1] + pd.Timedelta(minutes=15))
                        <= CLOCK - pd.Timedelta(minutes=5))

    def test_short_history_is_no_trade(self):
        src = MockSource(clock=CLOCK)
        k = src.stocks()[0].key
        plan = build_plan("X", src.bars15(k, CLOCK).tail(10), src.daily(k), float("nan"), CLOCK)
        self.assertEqual(plan["status"], "NO_TRADE")


class PieceTests(unittest.TestCase):
    def test_expiry_skips_too_close(self):
        d = date(2026, 11, 23)
        self.assertEqual(choose_expiry([date(2026, 11, 24), date(2026, 12, 29)], d, 3), date(2026, 12, 29))
        self.assertEqual(choose_expiry([date(2026, 11, 24), date(2026, 12, 29)], date(2026, 11, 2), 3), date(2026, 11, 24))

    def test_pivots(self):
        p = classic_pivots(110, 90, 100)
        self.assertAlmostEqual(p["P"], 100)
        self.assertAlmostEqual(p["R1"], 110)
        self.assertAlmostEqual(p["S1"], 90)

    def test_supertrend_follows_trend(self):
        idx = pd.date_range("2026-01-01", periods=60, freq="15min", tz="Asia/Kolkata")
        up = pd.DataFrame({"open": range(100, 160), "close": [x + 0.8 for x in range(100, 160)]}, index=idx)
        up["high"], up["low"] = up["close"] + 0.5, up["open"] - 0.5
        self.assertEqual(int(supertrend(up)[1].iloc[-1]), 1)

    def test_chain_stats(self):
        chain = pd.DataFrame({"strike": [90.0, 100.0, 110.0], "ce_oi": [10.0, 50.0, 200.0], "pe_oi": [300.0, 60.0, 5.0],
                              "ce_iv": [20.0, 21.0, 22.0], "pe_iv": [20.0, 21.0, 22.0], "ce_ltp": [11, 4, 1],
                              "pe_ltp": [1, 4, 11]})
        s = chain_stats(chain, 100.0)
        self.assertEqual(s["ce_wall"], 110.0)
        self.assertEqual(s["pe_wall"], 90.0)
        self.assertEqual(s["max_pain"], 100.0)
        self.assertAlmostEqual(s["pcr"], round(365 / 260, 2))

    def test_nse_csv(self):
        text = "Company Name,Industry,Symbol,Series,ISIN Code\nReliance Industries Ltd.,Oil Gas,RELIANCE,EQ,INE002A01018\n"
        s = parse_nifty50_csv(text)
        self.assertEqual(s[0].key, "NSE_EQ|INE002A01018")
        self.assertEqual(parse_nifty50_csv("garbage"), [])


def _trend_bars(step: float, days: int = 3, start: float = 1000.0) -> pd.DataFrame:
    """15m bars, 25 per day, drifting by `step` per bar, with volume."""
    idx = []
    for d in pd.bdate_range("2026-11-10", periods=days):
        idx += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=25, freq="15min"))
    idx = pd.DatetimeIndex(idx).tz_localize("Asia/Kolkata")
    close = [start + step * i + (0.4 if i % 2 else -0.4) for i in range(len(idx))]
    df = pd.DataFrame({"close": close}, index=idx)
    df["open"] = df["close"] - step * 0.8
    df["high"] = df[["open", "close"]].max(axis=1) + 0.3
    df["low"] = df[["open", "close"]].min(axis=1) - 0.3
    df["volume"] = 1000.0
    return df


class PrecisionTests(unittest.TestCase):
    NOW = pd.Timestamp("2026-11-12 12:00", tz="Asia/Kolkata")

    def _plan(self, kind="CE", entry=20.0, bid=19.9, ask=20.1, rr=2.0):
        return {"symbol": "X", "status": "VALID", "direction": kind,
                "option": {"entry": entry, "bid": bid, "ask": ask, "rr": rr}}

    def _bars(self, step=0.25):
        return _trend_bars(step)

    def test_nifty_context_votes(self):
        up = nifty_context(_trend_bars(2.0, start=24000), self.NOW)
        dn = nifty_context(_trend_bars(-2.0, start=24000), self.NOW)
        self.assertEqual(up["bias"], 3)
        self.assertEqual(dn["bias"], -3)
        self.assertFalse(nifty_context(None, self.NOW)["available"])

    def _names(self, res):
        return set(res["failed"])

    def test_ce_against_falling_nifty_is_rejected(self):
        nifty = nifty_context(_trend_bars(-2.0, start=24000), self.NOW)
        res = evaluate(self._plan("CE"), self._bars(0.05), nifty, self.NOW)
        self.assertIn("NIFTY alignment", self._names(res))
        self.assertFalse(res["ok"])

    def test_cheap_or_wide_premium_and_low_rr_rejected(self):
        nifty = nifty_context(_trend_bars(2.0, start=24000), self.NOW)
        bars = self._bars(0.05)
        self.assertIn("Premium and spread", self._names(evaluate(self._plan(entry=5.0, bid=4.9, ask=5.1), bars, nifty, self.NOW)))
        self.assertIn("Premium and spread", self._names(evaluate(self._plan(bid=19.0, ask=21.0), bars, nifty, self.NOW)))
        self.assertIn("Reward:risk", self._names(evaluate(self._plan(rr=1.2), bars, nifty, self.NOW)))

    def _done(self, bars):
        return bars[bars.index + pd.Timedelta(minutes=15) <= self.NOW]

    def test_clean_aligned_ce_passes(self):
        nifty = nifty_context(_trend_bars(0.5, start=24000), self.NOW)
        res = evaluate(self._plan(), self._done(_trend_bars(0.2)), nifty, self.NOW)
        self.assertTrue(res["ok"], res["failed"])
        self.assertGreater(res["rank"], 0)

    def test_stretched_ce_rejected(self):
        nifty = nifty_context(_trend_bars(0.5, start=24000), self.NOW)
        res = evaluate(self._plan(), self._done(_trend_bars(0.5)), nifty, self.NOW)
        self.assertIn("Not stretched from VWAP", self._names(res))
        self.assertIn("RSI not extreme", self._names(res))

    def test_red_last_bar_fails_confirmation_for_ce(self):
        nifty = nifty_context(_trend_bars(0.5, start=24000), self.NOW)
        bars = self._done(_trend_bars(0.2)).copy()
        bars.iloc[-1, bars.columns.get_loc("open")] = bars["close"].iloc[-1] + 1.0
        bars.iloc[-1, bars.columns.get_loc("high")] = bars["open"].iloc[-1] + 0.3
        self.assertIn("Confirmation bar", self._names(evaluate(self._plan(), bars, nifty, self.NOW)))

    def test_cutoff_and_not_valid(self):
        late = pd.Timestamp("2026-11-12 14:45", tz="Asia/Kolkata")
        nifty = nifty_context(_trend_bars(2.0, start=24000), late)
        self.assertIn("Pick cut-off", self._names(evaluate(self._plan(), self._bars(0.05), nifty, late)))
        self.assertIsNone(evaluate(dict(self._plan(), status="WATCH"), self._bars(0.05), nifty, self.NOW))

    def test_select_respects_caps_and_rank(self):
        cfg = PrecisionConfig(max_picks_per_day=3, max_open=2)
        mk = lambda s, r, ok=True: {"symbol": s, "precision": {"ok": ok, "rank": r}}
        c = [mk("A", 50), mk("B", 80), mk("C", 70), mk("D", 99, ok=False)]
        self.assertEqual([p["symbol"] for p in select(c, 0, 0, set(), cfg)], ["B", "C"])
        self.assertEqual([p["symbol"] for p in select(c, 2, 0, {"B", "X"}, cfg)], ["C"])
        self.assertEqual(select(c, 3, 0, set(), cfg), [])
        self.assertEqual(select(c, 1, 2, set(), cfg), [])

    def test_config_from_ini(self):
        f = Path(tempfile.mkdtemp()) / "x.ini"
        f.write_text("[STOCK_PRECISION]\nenabled = false\nmax_open = 1\nlast_entry = 12:30\nmin_rr = 2\n",
                     encoding="utf-8")
        cfg = PrecisionConfig.from_ini(f)
        self.assertEqual((cfg.enabled, cfg.max_open, cfg.last_entry, cfg.min_rr), (False, 1, "12:30", 2.0))


class PaperBookTests(unittest.TestCase):
    def _book(self):
        b = PaperBook(Path(tempfile.mkdtemp()), lambda *a, **k: 0.0, mock=True)
        b.roll(CLOCK.date())
        plan = {"symbol": "ABC", "score": 5.0,
                "spot": {"entry": 100.0, "stop": 98.0, "t1": 103.0, "t2": 105.0},
                "option": {"kind": "CE", "contract": "ABC NOV 100 CE", "key": "K", "lot_size": 10, "entry": 10.0,
                           "stop": 8.0, "t1": 12.0, "t2": 14.0}}
        b.open(plan, CLOCK)
        return b

    def test_t1_moves_stop_to_cost_then_exit_at_cost(self):
        b = self._book()
        b.mark({"K": 12.5}, {"ABC": 103.2}, CLOCK, "15:15")
        t = b.trades["ABC"]
        self.assertEqual((t["status"], t["stop"]), ("T1_HIT", 10.0))
        b.mark({"K": 9.9}, {"ABC": 99.9}, CLOCK, "15:15")
        self.assertEqual(t["status"], "EXIT_AT_COST")

    def test_stop_and_target_and_square_off(self):
        b = self._book()
        b.mark({"K": 7.5}, {"ABC": 99.0}, CLOCK, "15:15")
        self.assertEqual((b.trades["ABC"]["status"], b.trades["ABC"]["pnl"]), ("STOP_HIT", -25.0))
        b = self._book()
        b.mark({"K": 15.0}, {"ABC": 106.0}, CLOCK, "15:15")
        self.assertEqual((b.trades["ABC"]["status"], b.trades["ABC"]["exit"]), ("T2_HIT", 14.0))
        b = self._book()
        b.mark({"K": 11.0}, {"ABC": 101.0}, CLOCK + pd.Timedelta(hours=4), "15:15")
        self.assertEqual(b.trades["ABC"]["status"], "TIME_EXIT")

    def test_spot_stop_closes_even_if_premium_holds(self):
        b = self._book()
        b.mark({"K": 8.5}, {"ABC": 97.9}, CLOCK, "15:15")
        self.assertEqual(b.trades["ABC"]["status"], "STOP_HIT")

    def test_book_survives_restart(self):
        b = self._book()
        b.save()
        b2 = PaperBook(b.folder, lambda *a, **k: 0.0, mock=True)
        b2.roll(CLOCK.date())
        self.assertIn("ABC", b2.trades)


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.stocks.web import create_app
        cls.e = engine()
        cls.e.refresh()
        cls.client = create_app(cls.e, api_token="").test_client()

    def test_pages(self):
        for path in ("/", "/report/RELIANCE", "/static/stocks.css", "/static/search.js", "/health"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_state_and_search(self):
        self.assertEqual(len(self.client.get("/api/state").get_json()["plans"]), 50)
        hits = self.client.get("/api/search?q=rel").get_json()
        self.assertEqual(hits[0]["symbol"], "RELIANCE")

    def test_report(self):
        r = self.client.get("/api/report/RELIANCE").get_json()
        for section in ("snapshot", "performance", "risk", "technicals", "levels", "options", "plan", "verdict"):
            self.assertIn(section, r)
        self.assertEqual(self.client.get("/api/report/NOSUCH").status_code, 404)
        self.assertEqual(self.client.get("/api/report/bad%20sym!").status_code, 404)


if __name__ == "__main__":
    unittest.main()
