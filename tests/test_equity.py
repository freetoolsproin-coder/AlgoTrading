"""NIFTY 50 cash-equity service (src/equity): strategies, no look-ahead, risk, execution modes, backtest, API.
No network: prices are deterministic synthetic bars and the broker is a scripted fake."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from src.equity import backtest as bt
from src.equity import strategies as strat
from src.equity.alerts import Alerter
from src.equity.config import AlertConfig, EquityConfig, enabled_ids
from src.equity.data import EquityData, RateBudget, normalize_quote
from src.equity.engine import EquityEngine
from src.equity.execution import Trader, sessions_held
from src.equity.indicators import add_cross_section, enrich_daily, enrich_index, enrich_intraday, resample_bars, weekly_bars
from src.equity.risk import Costs, RiskManager, Sizing, delivery_charges, DeliveryCostConfig
from src.equity.signals import entry_zone, in_zone, levels, opportunities, raw_signals, stop_distance, tick
from src.equity.store import Store
from src.execution.broker_api import FILLED, OPEN, Broker, BrokerError, Funds, OrderStatus
from src.stocks.universe import Stock

TZ = "Asia/Kolkata"
NOW = pd.Timestamp("2026-11-12 11:32", tz=TZ)          # a Thursday, market open


def sessions(end: date, n: int):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def bars_5m(seed: int, days: int = 30, end: date = date(2026, 11, 12), base: float = 1000.0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = [pd.Timestamp(f"{d} 09:15", tz=TZ) + pd.Timedelta(minutes=5 * i) for d in sessions(end, days) for i in range(75)]
    n = len(stamps)
    regime = np.repeat(rng.normal(drift, 0.0006, n // 25 + 1), 25)[:n]
    close = base * np.exp(np.cumsum(regime + rng.normal(0, 0.0014, n)))
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    vol = (rng.lognormal(10, 0.6, n) * (1 + 3 * np.abs(close / open_ - 1) / 0.0014)).round()
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=pd.DatetimeIndex(stamps))


def daily_from(b: pd.DataFrame) -> pd.DataFrame:
    g = b.groupby(b.index.normalize())
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum()})


def daily_bars(seed: int, days: int = 420, end: date = date(2026, 11, 12), drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([pd.Timestamp(d.isoformat(), tz=TZ) for d in sessions(end, days)])
    regime = np.repeat(rng.normal(drift, 0.004, days // 15 + 1), 15)[:days]
    close = 800.0 * np.exp(np.cumsum(regime + rng.normal(0, 0.012, days)))
    open_ = np.concatenate([[800.0], close[:-1]]) * (1 + rng.normal(0, 0.003, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, days)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.lognormal(14, 0.5, days).round()}, index=idx)


def sig(module="intraday", symbol="AAA", side="BUY", price=1000.0, stop=990.0, sid="s1", strategy="vwap_breakout", **kw):
    s = 1 if side == "BUY" else -1
    d = {"id": sid, "module": module, "symbol": symbol, "name": symbol, "side": side, "strategy": strategy,
         "strategy_name": strategy, "kind": "breakout", "time": NOW.isoformat(), "day": NOW.date().isoformat(), "price": price,
         "entry": price, "stop": stop, "t1": price + s * 15, "t2": price + s * 25, "atr": 8.0, "score": 70.0,
         "zone": [min(price - s * 5, price + s * 3.5), max(price - s * 5, price + s * 3.5)], "reason": "test", "key": f"NSE_EQ|{symbol}"}
    d.update(kw)
    return d


def make_trader(mode="paper", broker=None, armed=True, **risk):
    cfg = EquityConfig()
    cfg.mode = mode
    for k, v in risk.items():
        setattr(cfg.risk, k, v)
    store = Store(":memory:")
    rm = RiskManager(cfg.risk, Costs(), kill_file=Path(tempfile.mkdtemp()) / "KILL")
    alerts = Alerter(AlertConfig(), store, push=False, clock=lambda: NOW)
    return Trader(cfg, store, rm, alerts, broker=broker, gate=lambda today: (armed, [] if armed else ["not armed today"]))


# =============================================================================
class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b = bars_5m(1)
        nb = bars_5m(99, base=22000.0)
        nb["volume"] = 0.0
        cls.f5 = enrich_intraday(b, None, enrich_index(nb, None), 5, 30)
        nd = daily_bars(50)
        cls.frames = {f"S{i}": enrich_daily(daily_bars(i), nd) for i in range(6)}
        add_cross_section(cls.frames, {f"S{i}": "Bank" if i < 3 else "IT" for i in range(6)})

    def test_registry_has_25_plus_25_unique_strategies(self):
        self.assertEqual(len(strat.INTRADAY), 25)
        self.assertEqual(len(strat.SWING), 25)
        self.assertEqual(len({s.name for s in strat.INTRADAY.values()}), 25)
        self.assertEqual(len({s.name for s in strat.SWING.values()}), 25)
        for row in strat.describe():
            self.assertTrue(row["about"])

    def test_every_strategy_returns_aligned_boolean_masks(self):
        for table, f in ((strat.INTRADAY, self.f5), (strat.SWING, self.frames["S0"])):
            for st in table.values():
                out = st.run(f)
                for mask in (out.long, out.short):
                    self.assertEqual(len(mask), len(f), st.id)
                    self.assertEqual(mask.dtype, bool, st.id)

    def test_intraday_strategies_fire_on_synthetic_data(self):
        fired = [st.id for st in strat.INTRADAY.values() if st.run(self.f5).long.any() or st.run(self.f5).short.any()]
        self.assertGreaterEqual(len(fired), 20, fired)

    def test_enabled_ids_spec(self):
        ids = list(strat.SWING)
        self.assertEqual(enabled_ids("all", ids), ids)
        self.assertNotIn("cup_handle", enabled_ids("all,-cup_handle", ids))
        self.assertEqual(enabled_ids("donchian, high_20d", ids), ["donchian", "high_20d"])


class NoLookAheadTests(unittest.TestCase):
    """A value on a bar must not change when later bars are appended (no look-ahead, no repainting)."""

    def test_intraday_frame_and_signals_are_causal(self):
        b = bars_5m(3)
        nb = bars_5m(98, base=22000.0)
        full = enrich_intraday(b, None, enrich_index(nb, None), 5, 30)
        cut = len(b) - 137                                  # mid-session cut
        part = enrich_intraday(b.iloc[:cut], None, enrich_index(nb.iloc[:cut], None), 5, 30)
        cols = [c for c in full.columns if full[c].dtype != object]
        a, c = full.iloc[:cut][cols].tail(150), part[cols].tail(150)
        pd.testing.assert_frame_equal(a, c, check_exact=False, rtol=1e-9, atol=1e-9)
        for st in strat.INTRADAY.values():
            x, y = st.run(full), st.run(part)
            np.testing.assert_array_equal(x.long.to_numpy()[:cut][-150:], y.long.to_numpy()[-150:], err_msg=st.id)
            np.testing.assert_array_equal(x.short.to_numpy()[:cut][-150:], y.short.to_numpy()[-150:], err_msg=st.id)

    def test_daily_frame_and_signals_are_causal(self):
        d, nd = daily_bars(7), daily_bars(51)
        full = enrich_daily(d, nd)
        cut = len(d) - 23
        part = enrich_daily(d.iloc[:cut], nd.iloc[:cut])
        cols = [c for c in full.columns if c not in ("rs_rank", "sector_rs", "sector_rank")]
        pd.testing.assert_frame_equal(full.iloc[:cut][cols].tail(60), part[cols].tail(60), check_exact=False, rtol=1e-9, atol=1e-9)
        for st in strat.SWING.values():
            x, y = st.run(full), st.run(part)
            np.testing.assert_array_equal(x.long.to_numpy()[:cut][-60:], y.long.to_numpy()[-60:], err_msg=st.id)

    def test_higher_timeframe_is_known_only_after_its_bar_closed(self):
        b = bars_5m(5, days=12)
        f = enrich_intraday(b, None, None)
        day = f[f.index.normalize() == f.index.normalize()[-1]]
        # inside the first hour of a session the 1h state cannot have changed: no 60-minute bar has completed yet
        self.assertEqual(day["h60"].iloc[:11].nunique(), 1)
        self.assertEqual(len(resample_bars(b, 60)), 12 * 7)
        self.assertEqual(len(weekly_bars(daily_from(b))), len({(d.isocalendar()[0], d.isocalendar()[1]) for d in b.index.normalize().unique()}))

    def test_previous_day_levels_come_from_the_previous_session(self):
        b = bars_5m(6, days=8)
        f = enrich_intraday(b, None, None)
        d = daily_from(b)
        last, prev = d.index[-1], d.index[-2]
        row = f[f.index.normalize() == last].iloc[0]
        self.assertAlmostEqual(row["pdh"], d.loc[prev, "high"])
        self.assertAlmostEqual(row["pdl"], d.loc[prev, "low"])
        self.assertTrue(np.isnan(f[f.index.normalize() == last]["orb_hi"].iloc[:6]).all())      # range not complete yet


class SignalTests(unittest.TestCase):
    def test_levels_targets_and_reward_risk(self):
        cfg = EquityConfig().intraday
        lv = levels("BUY", 1000.0, 990.0, cfg)
        self.assertEqual((lv["stop"], lv["t1"], lv["t2"], lv["rr"]), (990.0, 1015.0, 1025.0, 2.5))
        lv = levels("SELL", 1000.0, 1010.0, cfg)
        self.assertEqual((lv["t1"], lv["t2"]), (985.0, 975.0))
        self.assertEqual(tick(100.123), 100.1)

    def test_stop_distance_is_clamped_by_atr_and_minimum_percent(self):
        cfg = EquityConfig().intraday
        close, atr = np.array([1000.0, 1000.0, 1000.0, 1000.0]), np.array([4.0, 4.0, 4.0, np.nan])
        d = stop_distance("BUY", close, np.array([999.9, 950.0, np.nan, 990.0]), atr, cfg, True)
        self.assertAlmostEqual(d[0], max(cfg.stop_atr_min * 4, 1000 * cfg.min_stop_pct / 100))
        self.assertAlmostEqual(d[1], cfg.stop_atr_max * 4)
        self.assertGreater(d[2], 0)
        self.assertTrue(np.isnan(d[3]))                      # no ATR yet -> no signal, never a zero stop

    def test_entry_zone_does_not_chase(self):
        z = entry_zone("BUY", 1000.0, 10.0, 8.0)
        self.assertTrue(in_zone(1002.0, z))
        self.assertFalse(in_zone(1006.0, z))
        z = entry_zone("SELL", 1000.0, 10.0, 8.0)
        self.assertTrue(in_zone(998.0, z))
        self.assertFalse(in_zone(994.0, z))

    def test_raw_signals_carry_every_required_field(self):
        f = enrich_intraday(bars_5m(1), None, enrich_index(bars_5m(99, base=22000.0), None))
        out = raw_signals(f, strat.INTRADAY.values(), "intraday", EquityConfig().intraday, "AAA", "Aaa Ltd", start=len(f) - 75)
        self.assertTrue(out)
        for s in out:
            for k in ("symbol", "name", "price", "entry", "stop", "t1", "t2", "rr", "strategy_name", "score", "components",
                      "volume_ok", "indicators", "time", "reason", "zone"):
                self.assertIn(k, s)
            self.assertEqual(len({s["id"] for s in out}), len(out))
            d = 1 if s["side"] == "BUY" else -1
            self.assertGreater((s["entry"] - s["stop"]) * d, 0)
            self.assertGreater((s["t2"] - s["t1"]) * d, 0)
            self.assertTrue(0 <= s["score"] <= 100)
            self.assertAlmostEqual(sum(s["components"].values()), s["score"], delta=0.5)

    def test_swing_signals_have_zone_holding_period_and_exits(self):
        nd = daily_bars(50)
        frames = {f"S{i}": enrich_daily(daily_bars(i), nd) for i in range(5)}
        add_cross_section(frames, {k: "X" for k in frames})
        cfg = EquityConfig()
        got = []
        for k, f in frames.items():
            got += raw_signals(f, strat.SWING.values(), "swing", cfg.swing, k, k, start=len(f) - 120, max_gap_atr=0.75)
        self.assertTrue(got)
        for s in got:
            self.assertTrue(cfg.swing.min_hold_days <= s["hold_days"] <= cfg.swing.max_hold_days)
            self.assertIn(s["trend"], ("UP", "DOWN", "SIDEWAYS"))
            self.assertEqual(len(s["exits"]), 5)
            self.assertLessEqual(s["zone"][0], s["zone"][1])

    def test_opportunities_merge_duplicates_and_count_confluence(self):
        a = sig(sid="1", strategy="orb", time="2026-11-12T10:00:00+05:30")
        b = sig(sid="2", strategy="vwap_breakout", time="2026-11-12T10:10:00+05:30")
        c = sig(sid="3", strategy="macd_cross", time="2026-11-12T12:30:00+05:30")
        d = sig(sid="4", symbol="BBB", side="SELL", strategy="orb")
        opp = opportunities([a, b, c, d], validated={"orb"}, window_minutes=30)
        self.assertEqual(len(opp), 2)                        # one row per stock and side
        aaa = next(o for o in opp if o["symbol"] == "AAA")
        self.assertEqual(aaa["confluence"], 1)               # only the 12:30 signal is within 30 minutes of the latest
        self.assertEqual(opportunities([a, b], window_minutes=30)[0]["confluence"], 2)


class RiskTests(unittest.TestCase):
    def rm(self, **kw):
        cfg = EquityConfig().risk
        for k, v in kw.items():
            setattr(cfg, k, v)
        return RiskManager(cfg, Costs(), kill_file=Path(tempfile.mkdtemp()) / "KILL")

    def test_quantity_is_risk_over_stop_distance_in_whole_shares(self):
        rm = self.rm(capital=1_000_000, risk_per_trade_pct=0.5, max_position_pct=100, slippage_bps=0)
        s = rm.size("intraday", "BUY", 500.0, 495.0, 512.5)
        self.assertEqual(s.qty, 1000)                        # 5,000 / 5
        self.assertIsInstance(s.qty, int)
        self.assertAlmostEqual(s.rr_gross, 2.5)
        self.assertLess(s.rr_net, s.rr_gross)                # charges reduce it

    def test_no_leverage_position_value_cap_and_free_cash(self):
        rm = self.rm(capital=500_000, max_position_pct=20)
        s = rm.size("intraday", "BUY", 1000.0, 999.0, 1002.5)
        self.assertLessEqual(s.qty * 1000.0, 100_000)
        self.assertEqual(rm.size("intraday", "BUY", 1000.0, 990.0, 1025.0, free_cash=500.0).qty, 0)

    def test_swing_sizing_adds_gap_allowance(self):
        rm = self.rm(max_position_pct=100)
        a = rm.size("swing", "BUY", 1000.0, 980.0, 1050.0, atr=0.0)
        b = rm.size("swing", "BUY", 1000.0, 980.0, 1050.0, atr=20.0)
        self.assertLess(b.qty, a.qty)

    def test_entry_gate(self):
        rm = self.rm()
        good = Sizing(10, 5, 50, 10000, 30, 2.5, 2.2)
        book = {"open": [], "realized_today": 0.0, "trades_today": 0, "loss_streak": 0}
        self.assertEqual(rm.check_entry("intraday", "AAA", good, book), (True, []))
        cases = {"daily loss": {**book, "realized_today": -rm.daily_loss_limit()}, "trade limit": {**book, "trades_today": rm.cfg.daily_trade_limit},
                 "in a row": {**book, "loss_streak": rm.cfg.max_consecutive_losses},
                 "maximum open": {**book, "open": [{"module": "intraday", "symbol": f"X{i}"} for i in range(rm.cfg.max_open_intraday)]},
                 "no averaging": {**book, "open": [{"module": "swing", "symbol": "AAA"}]}}
        for text, b in cases.items():
            ok, why = rm.check_entry("intraday", "AAA", good, b)
            self.assertFalse(ok, text)
            self.assertIn(text, " ".join(why))
        self.assertFalse(rm.check_entry("intraday", "AAA", Sizing(10, 5, 50, 10000, 30, 1.5, 1.4), book)[0])     # R:R below 1:2
        self.assertFalse(rm.check_entry("intraday", "AAA", Sizing(10, 5, 50, 10000, 30, 2.5, 1.2), book)[0])     # costs eat it
        self.assertFalse(rm.check_entry("intraday", "AAA", Sizing(0, 5, 0, 0, 0, 0, 0, "x"), book)[0])
        rm.kill_flag = True
        self.assertIn("kill switch", " ".join(rm.check_entry("intraday", "AAA", good, book)[1]))

    def test_kill_file_blocks(self):
        rm = self.rm()
        self.assertFalse(rm.killed())
        rm.kill_file.write_text("x")
        self.assertTrue(rm.killed())

    def test_delivery_charges_include_stt_both_sides_and_dp_on_sell(self):
        c = DeliveryCostConfig()
        buy, sell = delivery_charges("BUY", 1000.0, 100, c), delivery_charges("SELL", 1000.0, 100, c)
        self.assertGreater(buy, 100.0)                       # 0.1% STT of 1,00,000 alone is 100
        self.assertGreater(sell, buy - 15.0 + 15.0 - 16.0)
        self.assertEqual(delivery_charges("BUY", 0, 10, c), 0.0)
        self.assertGreater(Costs().charge("swing", "SELL", 1000.0, 100), Costs().charge("intraday", "SELL", 1000.0, 100))


class PaperTradingTests(unittest.TestCase):
    def test_intraday_lifecycle_target1_then_stop_at_cost(self):
        t = make_trader()
        st, note = t.take(sig(), 1000.0, NOW)
        self.assertEqual(st, "TAKEN", note)
        pos = t.store.positions(("OPEN",))[0]
        qty = pos["qty"]
        self.assertGreater(qty, 1)
        self.assertGreater(pos["entry"], 1000.0)             # slippage against us
        t.on_quotes({"AAA": pos["t1"] + 0.5}, NOW, True)
        pos = t.store.position(pos["id"])
        self.assertEqual(pos["qty_open"], qty - qty // 2)
        self.assertEqual(pos["stop"], pos["entry"])          # stop moved to cost
        self.assertTrue(pos["data"]["t1_done"])
        t.on_quotes({"AAA": pos["entry"] - 0.5}, NOW, True)
        pos = t.store.position(pos["id"])
        self.assertEqual((pos["status"], pos["exit_reason"]), ("CLOSED", "STOP_AT_COST"))
        self.assertGreater(pos["charges"], 0)
        self.assertEqual(len(t.store.fills(position_id=pos["id"])), 3)

    def test_stop_target2_and_square_off(self):
        t = make_trader()
        t.take(sig(sid="a", symbol="AAA"), 1000.0, NOW)
        t.take(sig(sid="b", symbol="BBB"), 1000.0, NOW)
        t.take(sig(sid="c", symbol="CCC", side="SELL", stop=1010.0), 1000.0, NOW)
        t.on_quotes({"AAA": 989.0, "BBB": 1040.0, "CCC": 1001.0}, NOW, True)
        by = {p["symbol"]: p for p in t.store.positions()}
        self.assertEqual(by["AAA"]["exit_reason"], "STOP")
        self.assertLess(by["AAA"]["gross"], 0)
        self.assertEqual(by["BBB"]["exit_reason"], "TARGET2")
        self.assertGreater(by["BBB"]["gross"] - by["BBB"]["charges"], 0)
        self.assertEqual(by["CCC"]["status"], "OPEN")
        t.on_quotes({"CCC": 999.0}, pd.Timestamp("2026-11-12 15:11", tz=TZ), True)
        c = t.store.position(by["CCC"]["id"])
        self.assertEqual((c["status"], c["exit_reason"]), ("CLOSED", "SQUARE_OFF"))
        self.assertGreater(c["gross"], 0)                    # short: sold 1000, bought back 999
        book = t.book(pd.Timestamp("2026-11-12 15:12", tz=TZ))
        self.assertEqual(book["trades_today"], 3)
        self.assertAlmostEqual(book["realized_today"], sum(p["gross"] - p["charges"] for p in t.store.positions()), places=2)

    def test_duplicates_and_averaging_are_refused(self):
        t = make_trader()
        self.assertEqual(t.take(sig(sid="a"), 1000.0, NOW)[0], "TAKEN")
        self.assertEqual(t.take(sig(sid="a"), 1000.0, NOW)[0], "TAKEN")                 # same signal: nothing new
        st, why = t.take(sig(sid="b", strategy="orb"), 1000.0, NOW)                     # same stock again
        self.assertEqual(st, "BLOCKED")
        self.assertIn("no averaging", why)
        self.assertEqual(len(t.store.positions()), 1)

    def test_price_outside_zone_and_analysis_mode(self):
        t = make_trader()
        self.assertEqual(t.take(sig(), 1009.0, NOW)[0], "BLOCKED")
        self.assertEqual(t.take(sig(), None, NOW)[0], "BLOCKED")
        a = make_trader("analysis")
        self.assertEqual(a.take(sig(), 1000.0, NOW)[0], "ANALYSIS")
        self.assertEqual(a.store.positions(), [])

    def test_daily_loss_limit_stops_new_entries(self):
        t = make_trader(max_daily_loss_pct=0.05, max_position_pct=100)        # limit = 250 rupees
        t.take(sig(sid="a"), 1000.0, NOW)
        t.on_quotes({"AAA": 980.0}, NOW, True)
        st, why = t.take(sig(sid="b", symbol="BBB"), 1000.0, NOW)
        self.assertEqual(st, "BLOCKED")
        self.assertIn("daily loss limit", why)
        self.assertTrue(any(a["kind"] == "risk_limit" for a in t.alerts.recent))

    def test_swing_entry_waits_for_zone_then_time_exit(self):
        t = make_trader()
        s = sig("swing", sid="w", price=1000.0, stop=975.0, zone=[987.5, 1006.0], atr=20.0, strategy="donchian")
        st, note = t.take(s, None, pd.Timestamp("2026-11-12 15:40", tz=TZ))
        self.assertEqual(st, "PENDING", note)
        nxt = pd.Timestamp("2026-11-13 09:20", tz=TZ)
        t.on_quotes({"AAA": 1012.0}, nxt, True)                                  # gap above the zone: wait
        self.assertEqual(t.store.positions()[0]["status"], "PENDING")
        t.on_quotes({"AAA": 1003.0}, nxt, True, {"AAA": {"h1": -1}})             # 1h trend against: wait
        self.assertEqual(t.store.positions()[0]["status"], "PENDING")
        t.on_quotes({"AAA": 1003.0}, nxt, True, {"AAA": {"h1": 1}})
        pos = t.store.positions()[0]
        self.assertEqual(pos["status"], "OPEN")
        self.assertEqual(pos["stop"], 975.0)                                     # the stop never moves away
        self.assertEqual(sessions_held(pos["opened"], date(2026, 11, 19)), 5)
        t.on_quotes({"AAA": 1010.0}, pd.Timestamp("2026-11-19 15:16", tz=TZ), True)
        self.assertEqual(t.store.positions()[0]["exit_reason"], "TIME_EXIT")

    def test_swing_pending_expires_and_gap_through_stop_cancels(self):
        t = make_trader()
        t.take(sig("swing", sid="w1", stop=975.0, zone=[987.5, 1006.0]), None, pd.Timestamp("2026-11-12 15:40", tz=TZ))
        t.take(sig("swing", sid="w2", symbol="BBB", stop=975.0, zone=[987.5, 1006.0]), None, pd.Timestamp("2026-11-12 15:40", tz=TZ))
        t.on_quotes({"BBB": 970.0}, pd.Timestamp("2026-11-13 09:20", tz=TZ), True)
        t.on_quotes({}, pd.Timestamp("2026-11-13 15:31", tz=TZ), False)
        by = {p["symbol"]: p for p in t.store.positions()}
        self.assertEqual(by["BBB"]["status"], "CANCELLED")
        self.assertIn("gap", by["BBB"]["exit_reason"])
        self.assertEqual(by["AAA"]["status"], "CANCELLED")

    def test_swing_sell_is_an_alert_and_flags_an_exit(self):
        t = make_trader()
        self.assertEqual(t.take(sig("swing", side="SELL", stop=1020.0), None, NOW)[0], "ALERT")
        t.take(sig("swing", sid="w", stop=975.0, zone=[987.5, 1006.0]), None, pd.Timestamp("2026-11-12 15:40", tz=TZ))
        t.on_quotes({"AAA": 1000.0}, pd.Timestamp("2026-11-13 09:20", tz=TZ), True)
        t.flag_exit("AAA", "TREND_EXIT", "close below the 20 EMA", pd.Timestamp("2026-11-13 15:40", tz=TZ))    # day 1: too early
        self.assertFalse(t.store.positions()[0]["data"].get("exit_flag"))
        t.flag_exit("AAA", "TREND_EXIT", "close below the 20 EMA", pd.Timestamp("2026-11-16 15:40", tz=TZ))
        t.on_quotes({"AAA": 995.0}, pd.Timestamp("2026-11-17 09:16", tz=TZ), True)
        self.assertEqual(t.store.positions()[0]["exit_reason"], "TREND_EXIT")

    def test_kill_switch_closes_everything_and_blocks(self):
        t = make_trader()
        t.take(sig(sid="a"), 1000.0, NOW)
        t.take(sig("swing", sid="w", symbol="BBB", stop=975.0, zone=[987.5, 1006.0]), None, NOW)
        t.on_quotes({"AAA": 1001.0}, NOW, True)
        t.kill(True, NOW)
        by = {p["symbol"]: p for p in t.store.positions()}
        self.assertEqual((by["AAA"]["status"], by["AAA"]["exit_reason"]), ("CLOSED", "KILL"))
        self.assertEqual(by["BBB"]["status"], "CANCELLED")
        self.assertEqual(t.take(sig(sid="c", symbol="CCC"), 1000.0, NOW)[0], "BLOCKED")
        t.kill(False, NOW)
        self.assertEqual(t.take(sig(sid="c", symbol="CCC"), 1000.0, NOW)[0], "TAKEN")

    def test_unvalidated_strategies_paper_only_when_allowed(self):
        t = make_trader()
        self.assertEqual(t.take(sig(), 1000.0, NOW, validated=False)[0], "TAKEN")
        t2 = make_trader()
        t2.cfg.paper_unvalidated = False
        self.assertEqual(t2.take(sig(), 1000.0, NOW, validated=False)[0], "BLOCKED")


class FakeBroker(Broker):
    name, live = "fake", True

    def __init__(self):
        self.placed, self.cancelled, self.status, self.fail = [], [], {}, None
        self.net = {}

    def place_order(self, req):
        if self.fail:
            err, self.fail = self.fail, None
            raise err
        oid = f"B{len(self.placed) + 1}"
        self.placed.append((oid, req))
        self.status[oid] = OrderStatus(oid, OPEN, quantity=req.quantity, tag=req.tag)
        return [oid]

    def fill(self, oid, price):
        req = dict(self.placed)[oid]
        self.status[oid] = OrderStatus(oid, FILLED, quantity=req.quantity, filled_quantity=req.quantity, average_price=price, tag=req.tag)

    def modify_order(self, order_id, **kw):
        pass

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)

    def order_status(self, order_id="", tag=""):
        if order_id:
            return self.status.get(order_id)
        return next((s for s in self.status.values() if s.tag == tag), None)

    def positions(self):
        return []

    def funds(self):
        return Funds(1e6)


class LiveTradingTests(unittest.TestCase):
    def test_live_needs_arming_and_a_confirmation_per_order(self):
        br = FakeBroker()
        t = make_trader("live", br, armed=False)
        st, note = t.take(sig(), 1000.0, NOW)
        self.assertEqual(st, "PROPOSED", note)
        self.assertEqual(br.placed, [])                                     # nothing sent without a click
        oid = t.store.orders(("PROPOSED",))[0]["id"]
        t.ltp["AAA"] = 1000.0
        ok, msg = t.confirm(oid, NOW)
        self.assertFalse(ok)
        self.assertIn("not armed", msg)
        self.assertEqual(br.placed, [])
        self.assertEqual(t.set_mode("live", NOW.date())[0], False)

    def test_confirm_sends_once_then_fill_places_the_protective_stop(self):
        br = FakeBroker()
        t = make_trader("live", br)
        t.take(sig(), 1000.0, NOW)
        oid = t.store.orders(("PROPOSED",))[0]["id"]
        t.ltp["AAA"] = 1000.0
        self.assertTrue(t.confirm(oid, NOW)[0])
        self.assertFalse(t.confirm(oid, NOW)[0])                            # duplicate click: not sent again
        self.assertEqual(len(br.placed), 1)
        b1, req = br.placed[0]
        self.assertEqual((req.side, req.product, req.order_type), ("BUY", "I", "LIMIT"))
        self.assertTrue(req.instrument_key.startswith("NSE_EQ|"))
        self.assertEqual(t.store.positions()[0]["status"], "ENTERING")
        br.fill(b1, 1000.4)
        t.reconcile(NOW)
        pos = t.store.positions()[0]
        self.assertEqual((pos["status"], pos["entry"]), ("OPEN", 1000.4))
        self.assertEqual(len(br.placed), 2)
        b2, stop = br.placed[1]
        self.assertEqual((stop.side, stop.order_type, stop.trigger_price, stop.quantity), ("SELL", "SL-M", 990.0, pos["qty"]))
        t.reconcile(NOW)
        self.assertEqual(len(br.placed), 2)                                 # no second stop while one is working
        br.fill(b2, 989.8)
        t.reconcile(NOW)
        pos = t.store.positions()[0]
        self.assertEqual((pos["status"], pos["exit_reason"]), ("CLOSED", "STOP"))
        self.assertEqual(len({o["tag"] for o in t.store.orders()}), len(t.store.orders()))

    def test_timeout_is_looked_up_by_tag_never_resent(self):
        br = FakeBroker()
        t = make_trader("live", br)
        t.take(sig(), 1000.0, NOW)
        o = t.store.orders(("PROPOSED",))[0]
        t.ltp["AAA"] = 1000.0
        br.fail = BrokerError("timeout", retryable=True, maybe_sent=True)
        self.assertFalse(t.confirm(o["id"], NOW)[0])
        self.assertEqual(t.store.order(o["id"])["status"], "UNKNOWN")
        self.assertFalse(t.confirm(o["id"], NOW)[0])
        self.assertEqual(br.placed, [])
        br.status["B9"] = OrderStatus("B9", FILLED, quantity=o["qty"], filled_quantity=o["qty"], average_price=1000.2, tag=o["tag"])
        t.reconcile(NOW)                                                    # it did arrive: found by its tag
        self.assertEqual(t.store.positions()[0]["status"], "OPEN")

    def test_rejected_entry_and_user_reject_cancel_the_position(self):
        br = FakeBroker()
        t = make_trader("live", br)
        t.take(sig(sid="a"), 1000.0, NOW)
        t.take(sig(sid="b", symbol="BBB"), 1000.0, NOW)
        a, b = t.store.orders(("PROPOSED",))
        t.ltp.update(AAA=1000.0, BBB=1000.0)
        br.fail = BrokerError("insufficient funds")
        t.confirm(a["id"], NOW)
        self.assertTrue(t.reject(b["id"], NOW)[0])
        self.assertEqual({p["status"] for p in t.store.positions()}, {"CANCELLED"})

    def test_target_exit_is_proposed_but_square_off_and_kill_are_automatic(self):
        br = FakeBroker()
        t = make_trader("live", br)
        t.take(sig(), 1000.0, NOW)
        t.ltp["AAA"] = 1000.0
        t.confirm(t.store.orders(("PROPOSED",))[0]["id"], NOW)
        br.fill("B1", 1000.0)
        t.reconcile(NOW)
        t.on_quotes({"AAA": 1030.0}, NOW, True)                              # beyond target 2
        self.assertEqual(len(br.placed), 2)                                  # entry + stop only: the exit waits
        self.assertEqual([o["purpose"] for o in t.store.orders(("PROPOSED",))], ["EXIT"])
        t.on_quotes({"AAA": 1030.0}, pd.Timestamp("2026-11-12 15:11", tz=TZ), True)
        self.assertEqual(len(br.placed), 3)                                  # square-off sent without a click
        self.assertEqual(br.placed[2][1].side, "SELL")
        self.assertIn("B2", br.cancelled)                                    # the stop order was cancelled first
        br.fill("B3", 1029.5)
        t.reconcile(pd.Timestamp("2026-11-12 15:12", tz=TZ))
        self.assertEqual(t.store.positions()[0]["exit_reason"], "SQUARE_OFF")

    def test_equity_broker_refuses_non_equity_and_naked_delivery_sell(self):
        from src.equity.broker import UpstoxEquityBroker
        from src.execution.broker_api import OrderRequest
        b = UpstoxEquityBroker(session=object())
        with self.assertRaises(BrokerError):
            b.place_order(OrderRequest("NSE_FO|12345", "BUY", 1, product="I"))
        with self.assertRaises(BrokerError):
            b.place_order(OrderRequest("NSE_EQ|INE002A01018", "SELL", 1, product="D"))
        with self.assertRaises(BrokerError):
            b.place_order(OrderRequest("NSE_EQ|INE002A01018", "BUY", 0, product="I"))


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.charge = lambda side, price, qty: 0.0

    def run_trade(self, lows, highs, opens=None, closes=None, side="BUY", qty=10):
        n = len(lows)
        o = np.array(opens or [100.0] * n)
        c = np.array(closes or [100.0] * n)
        return bt.simulate_trade(side, 100.0, 98.0, 103.0, 105.0, qty, o, np.array(highs), np.array(lows), c, 0, n - 1, c[-1], "TIME_EXIT", self.charge, 0.0)

    def test_stop_first_when_a_bar_touches_stop_and_target(self):
        r = self.run_trade([97.0], [106.0])
        self.assertEqual(r["reason"], "STOP")
        self.assertAlmostEqual(r["gross"], -20.0)

    def test_target1_books_half_then_stop_at_cost(self):
        r = self.run_trade([99.5, 99.0, 99.8], [103.5, 101.0, 100.5])
        self.assertEqual([f[3] for f in r["fills"]], ["TARGET1", "STOP_AT_COST"])
        self.assertAlmostEqual(r["gross"], 5 * 3.0)

    def test_target2_and_time_exit(self):
        r = self.run_trade([99.5, 101.0], [103.5, 105.5])
        self.assertEqual([f[3] for f in r["fills"]], ["TARGET1", "TARGET2"])
        self.assertAlmostEqual(r["gross"], 5 * 3.0 + 5 * 5.0)
        r = self.run_trade([99.5, 99.5], [101.0, 101.0], closes=[100.5, 101.0])
        self.assertEqual(r["reason"], "TIME_EXIT")
        self.assertAlmostEqual(r["gross"], 10.0)

    def test_gap_through_the_stop_fills_at_the_open(self):
        r = self.run_trade([99.5, 94.0], [101.0, 96.0], opens=[100.0, 95.0])
        self.assertEqual(r["reason"], "STOP")
        self.assertAlmostEqual(r["gross"], -50.0)

    def test_short_trade_is_mirrored(self):
        r = bt.simulate_trade("SELL", 100.0, 102.0, 97.0, 95.0, 10, np.array([100.0, 99.0]), np.array([100.5, 98.0]),
                              np.array([96.5, 94.0]), np.array([97.0, 95.0]), 0, 1, 95.0, "SQUARE_OFF", self.charge, 0.0)
        self.assertEqual([f[3] for f in r["fills"]], ["TARGET1", "TARGET2"])
        self.assertAlmostEqual(r["gross"], 5 * 3.0 + 5 * 5.0)

    def test_metrics(self):
        t0 = pd.Timestamp("2026-01-05", tz=TZ)
        trades = [{"net": n, "r": n / 100, "charges": 5.0, "exit_time": t0 + pd.Timedelta(days=i)} for i, n in enumerate([100, -50, 200, -100, -100, 150])]
        m = bt.metrics(trades, 100000.0, pd.date_range(t0, periods=30, freq="D", tz=TZ))
        self.assertEqual((m["trades"], m["win_rate"], m["net"]), (6, 50.0, 200.0))
        self.assertAlmostEqual(m["profit_factor"], 1.8)
        self.assertEqual(m["max_dd"], 200.0)
        self.assertIsNotNone(m["sharpe"])
        self.assertEqual(bt.metrics([], 1.0)["trades"], 0)

    def test_book_rules_limit_open_positions_and_daily_trades(self):
        cfg = EquityConfig()
        cfg.risk.max_open_intraday, cfg.risk.daily_trade_limit = 2, 3
        t0 = pd.Timestamp("2026-01-05 10:00", tz=TZ)
        mk = lambda i, hold: {"strategy": "orb", "symbol": f"S{i}", "entry_time": t0 + pd.Timedelta(minutes=5 * i),
                              "exit_time": t0 + pd.Timedelta(minutes=5 * i + hold), "net": 10.0, "score": 60.0}
        kept = bt.apply_book_rules([mk(0, 60), mk(1, 60), mk(2, 60), mk(20, 5), mk(22, 5), mk(24, 5)], cfg, "intraday")
        self.assertEqual([t["symbol"] for t in kept], ["S0", "S1", "S20"])          # third blocked by open limit, then daily limit

    def test_end_to_end_on_synthetic_history_and_validation_needs_both_samples(self):
        class Hist:
            def __init__(self):
                self.stocks = [Stock(f"S{i}", f"S{i} Ltd", "Bank" if i < 3 else "IT", f"NSE_EQ|S{i}") for i in range(5)]
                self.d = {s.symbol: daily_bars(i, 520) for i, s in enumerate(self.stocks)}
                self.d["NIFTY"] = daily_bars(77, 520)

            def universe(self):
                return self.stocks

            def history(self, symbol, key, tf, start, now=None):
                return self.d[symbol]

            def forget(self, tf=None):
                pass
        cfg = EquityConfig()
        store = Store(":memory:")
        res = bt.run("swing", cfg, Hist(), store, years=1.5, now=pd.Timestamp("2026-11-12 18:00", tz=TZ))
        self.assertEqual(len(res["strategies"]), 25)
        self.assertGreater(res["summary"]["all_strategies_combined"]["trades"], 0)
        self.assertEqual(len(res["walk_forward"]), cfg.backtest.walk_forward_folds)
        self.assertTrue(any("survivorship" in n for n in res["notes"]))
        for r in res["strategies"]:
            self.assertEqual(r["all"]["trades"], r["in_sample"]["trades"] + r["out_of_sample"]["trades"])
            if r["validated"]:
                self.assertGreater(r["in_sample"]["net"], 0)
                self.assertGreater(r["out_of_sample"]["net"], 0)
                self.assertGreaterEqual(r["out_of_sample"]["trades"], cfg.backtest.min_oos_trades)
        self.assertEqual(store.get("validated")["swing"], res["validated"])
        self.assertEqual(set(store.get("perf")["swing"]), set(strat.SWING))
        for t in res["recent_trades"]:
            self.assertGreater(t["entry_time"], t["signal_time"])           # never filled on the signal bar

    def test_intraday_trades_enter_next_bar_and_close_same_day(self):
        cfg = EquityConfig()
        f = enrich_intraday(bars_5m(4, days=40), None, enrich_index(bars_5m(97, days=40, base=22000.0), None))
        rm = RiskManager(cfg.risk, Costs(), kill_file=Path(tempfile.mkdtemp()) / "K")
        trades = bt.trades_for_stock(f, "intraday", "AAA", list(strat.INTRADAY.values()), cfg, rm)
        self.assertGreater(len(trades), 20)
        for t in trades:
            self.assertEqual(t["entry_time"], t["signal_time"] + pd.Timedelta(minutes=5))
            self.assertEqual(t["entry_time"].date(), t["exit_time"].date())
            self.assertLessEqual(t["exit_time"].time(), pd.Timestamp("15:10").time())
            self.assertGreater(t["charges"], 0)
            self.assertGreaterEqual(t["score"], cfg.intraday.min_strength)

    def test_membership_file_restricts_the_period(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.csv"
            p.write_text("symbol,from,to\nAAA,2026-10-01,\n", encoding="utf-8")
            m = bt.load_membership(p)
        self.assertEqual(m["AAA"][0][0], date(2026, 10, 1))
        cfg = EquityConfig()
        f = enrich_intraday(bars_5m(4, days=40), None, None)
        rm = RiskManager(cfg.risk, Costs(), kill_file=Path(tempfile.mkdtemp()) / "K")
        trades = bt.trades_for_stock(f, "intraday", "AAA", list(strat.INTRADAY.values()), cfg, rm, [(date(2026, 11, 1), date(2999, 1, 1))])
        self.assertTrue(all(t["signal_time"].date() >= date(2026, 11, 1) for t in trades))


class FakeClient:
    def __init__(self):
        self.paths = []

    def get(self, path, params=None, timeout=20, base=""):
        self.paths.append(path)
        if "market-quote" in path:
            return {"data": {"NSE_EQ:AAA": {"instrument_token": "NSE_EQ|AAA", "last_price": 101.5, "net_change": 1.5, "volume": 1000,
                                            "average_price": 100.9, "ohlc": {"open": 100, "high": 102, "low": 99.5, "close": 101.5},
                                            "depth": {"buy": [{"price": 0.0, "quantity": 0, "orders": 0}],
                                                      "sell": [{"price": 101.55, "quantity": 40, "orders": 2}]},
                                            "total_buy_quantity": 0.0, "total_sell_quantity": 40.0}}}
        to, frm = path.rstrip("/").split("/")[-2:]
        days = pd.bdate_range(frm, to)
        return {"data": {"candles": [[f"{d.date()}T00:00:00+05:30", 10, 11, 9, 10.5, 100, 0] for d in days]}}


class DataTests(unittest.TestCase):
    def test_quote_keeps_missing_values_missing(self):
        d = EquityData(client=FakeClient(), cache_dir=tempfile.mkdtemp(), universe=[])
        q = d.quotes(["NSE_EQ|AAA"])["NSE_EQ|AAA"]
        self.assertEqual(q["ltp"], 101.5)
        self.assertAlmostEqual(q["prev_close"], 100.0)
        self.assertTrue(np.isnan(q["bid"]))                  # empty bid side is NaN, not 0
        self.assertEqual(q["ask"], 101.55)
        self.assertEqual(q["depth"]["buy"], [])
        self.assertTrue(np.isnan(normalize_quote({})["ltp"]))

    def test_history_is_cached_on_disk_and_only_missing_days_are_fetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = FakeClient()
            d = EquityData(client=c, cache_dir=tmp, universe=[])
            now = pd.Timestamp("2026-11-12 10:00", tz=TZ)
            a = d.history("AAA", "NSE_EQ|AAA", "1d", date(2026, 10, 1), now)
            self.assertEqual(a.index[-1].date(), date(2026, 11, 11))            # never today
            self.assertEqual(len(c.paths), 1)
            d.history("AAA", "NSE_EQ|AAA", "1d", date(2026, 10, 1), now)
            self.assertEqual(len(c.paths), 1)                                   # memory
            d2 = EquityData(client=c, cache_dir=tmp, universe=[])
            d2.history("AAA", "NSE_EQ|AAA", "1d", date(2026, 10, 1), now)
            self.assertEqual(len(c.paths), 1)                                   # disk
            later = d2.history("AAA", "NSE_EQ|AAA", "1d", date(2026, 10, 1), pd.Timestamp("2026-11-16 10:00", tz=TZ))
            self.assertEqual(len(c.paths), 2)                                   # only the new days
            self.assertIn("2026-11-12", c.paths[-1])
            self.assertEqual(later.index[-1].date(), date(2026, 11, 13))
            self.assertFalse(later.index.has_duplicates)

    def test_minute_history_is_split_into_monthly_requests(self):
        c = FakeClient()
        d = EquityData(client=c, cache_dir=tempfile.mkdtemp(), universe=[])
        d.fetch("NSE_EQ|AAA", "5m", date(2026, 1, 1), date(2026, 3, 31))
        self.assertEqual(len(c.paths), 4)

    def test_rate_budget_counts_calls(self):
        b = RateBudget(per_minute=5, per_30min=10)
        for _ in range(5):
            b.acquire()
        self.assertEqual(b.status()["last_minute"], 5)


class FakeData:
    """The engine's data source, served from synthetic bars."""

    def __init__(self, n=6):
        self._universe = [Stock(f"S{i}", f"S{i} Ltd", "Bank" if i < 3 else "IT", f"NSE_EQ|S{i}") for i in range(n)]
        self.m5 = {s.symbol: bars_5m(i + 10, 22) for i, s in enumerate(self._universe)}
        self.m5["NIFTY"] = bars_5m(90, 22, base=22000.0)
        self.dly = {s.symbol: daily_bars(i + 10, 420, end=date(2026, 11, 11)) for i, s in enumerate(self._universe)}
        self.dly["NIFTY"] = daily_bars(91, 420, end=date(2026, 11, 11))
        self.feed = None

    def universe(self):
        return self._universe

    def intraday_bars(self, stock, now, history_days=20):
        b = self.m5[stock.symbol]
        b = b[b.index + pd.Timedelta(minutes=5) <= now]
        return {"1m": b[b.index.normalize() == now.normalize()], "5m": b}

    def daily_bars(self, stock, now, days=1100, today_1m=None):
        return self.dly[stock.symbol]

    def hourly_bars(self, stock, now, days=80, today_1m=None):
        return resample_bars(self.intraday_bars(stock, now)["5m"], 60)

    def quotes(self, keys):
        out = {}
        for s in self._universe:
            b = self.m5[s.symbol]
            b = b[b.index + pd.Timedelta(minutes=5) <= NOW]
            out[s.key] = normalize_quote({"last_price": float(b["close"].iloc[-1]), "net_change": 1.0, "volume": 1e5})
        return out

    def status(self):
        return {"calls": 0}

    def stop_feed(self):
        pass


def make_engine(mode="paper"):
    cfg = EquityConfig()
    cfg.mode, cfg.data_dir = mode, Path(tempfile.mkdtemp())
    cfg.intraday.signal_valid_minutes = 400                     # every signal of the synthetic morning counts as fresh
    cfg.alerts.desktop_popup = False
    return EquityEngine(cfg, data=FakeData(), store=Store(":memory:"), clock=lambda: NOW, push=False, gate=lambda d: (False, ["not armed today"]))


class EngineAndApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = make_engine()
        cls.e.risk.kill_file = Path(tempfile.mkdtemp()) / "KILL"
        cls.e.quote_cycle(NOW)
        cls.e.scan_cycle(NOW)
        cls.e.quote_cycle(NOW)
        cls.e.status = "ok"
        cls.state = cls.e.state()
        from src.equity.web import create_app
        cls.client = create_app(cls.e, api_token="").test_client()

    def test_scan_builds_both_modules_from_completed_bars(self):
        s = self.state
        self.assertEqual(len(s["intraday"]["scanner"]), 6)
        self.assertGreater(s["intraday"]["signals_today"], 0)
        self.assertEqual(s["mode"], "paper")
        self.assertTrue(s["market_open"])
        for row in s["intraday"]["scanner"]:
            self.assertIn(row["signal"], ("BUY", "SELL", "NO TRADE"))
            self.assertTrue(row["reason"])
            self.assertLessEqual(pd.Timestamp(row["last_signal"]["time"]) if row["last_signal"] else NOW, NOW)
        self.assertEqual(len(s["swing"]["watchlist"]), 6)
        self.assertEqual(len(s["swing"]["sectors"]), 2)
        for k in ("intraday_buy", "intraday_sell", "swing_buy", "swing_sell", "breakout", "high_volume", "momentum", "strong_trend"):
            self.assertIn(k, s["scanners"])
        json.dumps(s, default=str)

    def test_no_signal_is_newer_than_the_last_completed_bar(self):
        last_bar_close = pd.Timestamp("2026-11-12 11:30", tz=TZ)
        for sg in self.e.store.signals("intraday"):
            self.assertLessEqual(pd.Timestamp(sg["time"]), last_bar_close)
            self.assertGreaterEqual(pd.Timestamp(sg["time"]).time(), pd.Timestamp("09:30").time())      # entry window

    def test_one_signal_per_stock_setup_and_side_a_day(self):
        seen = [(s["symbol"], s["strategy"], s["side"]) for s in self.e.store.signals("intraday")]
        self.assertEqual(len(seen), len(set(seen)))
        before = len(seen)
        self.e.scan_cycle(NOW)                                   # scanning again adds nothing
        self.assertEqual(len(self.e.store.signals("intraday")), before)

    def test_paper_book_respects_the_limits(self):
        r = self.e.cfg.risk
        open_ = self.e.store.positions(("OPEN",), module="intraday")
        self.assertLessEqual(len(open_), r.max_open_intraday)
        self.assertEqual(len({p["symbol"] for p in open_}), len(open_))
        self.assertLessEqual(self.e.trader.book(NOW)["trades_today"], r.daily_trade_limit)
        for p in open_:
            self.assertLessEqual(p["qty"] * p["entry"], r.capital * r.max_position_pct / 100 + 1)

    def test_api_state_chart_and_strategies(self):
        r = self.client.get("/api/state")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("NaN", r.get_data(as_text=True))
        c = self.client.get("/api/chart/S0?tf=15m").get_json()
        self.assertGreater(len(c["candles"]), 20)
        self.assertIn("EMA 9", c["lines"])
        self.assertEqual(len(c["lines"]["EMA 9"]), len(c["candles"]))
        self.assertEqual(self.client.get("/api/chart/S0?tf=1d").get_json()["tf"], "1d")
        self.assertEqual(self.client.get("/api/chart/S0?tf=7m").status_code, 404)
        st = self.client.get("/api/strategies").get_json()
        self.assertEqual((len(st["intraday"]), len(st["swing"])), (25, 25))
        self.assertEqual(self.client.get("/api/backtest/swing").status_code, 404)
        self.assertEqual(self.client.get("/health").get_json()["status"], "ok")

    def test_api_mode_live_refused_unless_armed_and_confirm_needs_explicit_flag(self):
        r = self.client.post("/api/mode", json={"mode": "live"})
        self.assertEqual(r.status_code, 409)
        self.assertIn("not armed", r.get_json()["error"])
        self.assertEqual(self.e.trader.mode, "paper")
        self.assertEqual(self.client.post("/api/mode", json={"mode": "bogus"}).status_code, 409)
        self.assertEqual(self.client.post("/api/orders/1/confirm", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/orders/1/confirm", json={"confirm": True}).status_code, 409)

    def test_api_risk_settings_are_validated_and_persisted(self):
        self.assertEqual(self.client.post("/api/risk", json={"risk_per_trade_pct": 50}).status_code, 400)
        r = self.client.post("/api/risk", json={"risk_per_trade_pct": 0.25, "unknown": 1})
        self.assertEqual(r.get_json()["risk"]["risk_per_trade_pct"], 0.25)
        self.assertEqual(self.e.store.get("risk")["risk_per_trade_pct"], 0.25)
        self.client.post("/api/risk", json={"risk_per_trade_pct": 0.5})

    def test_api_posts_are_local_only(self):
        r = self.client.post("/api/kill", json={"on": True}, environ_base={"REMOTE_ADDR": "10.0.0.9"})
        self.assertEqual(r.status_code, 403)
        self.assertFalse(self.e.risk.killed())

    def test_z_kill_switch_via_api(self):
        r = self.client.post("/api/kill", json={"on": True})
        self.assertTrue(r.get_json()["kill_switch"])
        self.assertEqual(self.e.store.positions(("OPEN", "PENDING", "PROPOSED")), [])
        self.assertTrue(self.e.state()["risk"]["kill_switch"])
        self.client.post("/api/kill", json={"on": False})


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _reply(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.dumps({"ok": True, "method": self.command, "path": self.path, "body": self.rfile.read(n).decode() if n else ""}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply


class DeskProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.desk.__main__ import make_handler
        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        dead = "http://127.0.0.1:9"
        cls.desk = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dead, dead, dead, f"http://127.0.0.1:{cls.up.server_port}"))
        cls.off = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dead, dead, dead, dead))
        for s in (cls.up, cls.desk, cls.off):
            threading.Thread(target=s.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for s in (cls.up, cls.desk, cls.off):
            s.shutdown()
            s.server_close()

    def call(self, server, path, data=None):
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", data=data, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_equity_api_is_passed_through(self):
        status, body = self.call(self.desk, "/api/eq/chart/M%26M?tf=5m")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["path"], "/api/chart/M%26M?tf=5m")
        status, body = self.call(self.desk, "/api/eq/kill", b'{"on": true}')
        self.assertEqual((status, json.loads(body)["method"], json.loads(body)["body"]), (200, "POST", '{"on": true}'))
        self.assertEqual(self.call(self.desk, "/api/other", b"{}")[0], 404)

    def test_offline_service_is_reported_not_faked(self):
        status, body = self.call(self.off, "/api/eq/state")
        self.assertEqual(status, 503)
        self.assertTrue(json.loads(body)["offline"])

    def test_page_and_assets_are_served(self):
        status, body = self.call(self.desk, "/")
        self.assertEqual(status, 200)
        for needle in (b'href="#indices"', b'id="fno"', b'id="breakouts"'):
            self.assertIn(needle, body)
        for gone in (b'id="intraday"', b'id="swing"', b"/static/equity.js", b"Intraday equity", b"eq_pnl"):   # cash equity removed
            self.assertNotIn(gone, body)
        self.assertEqual(self.call(self.desk, "/static/equity.js")[0], 200)
        self.assertEqual(self.call(self.desk, "/static/equity.css")[0], 200)
        self.assertEqual(self.call(self.desk, "/static/../__main__.py")[0], 404)


class StoreAndConfigTests(unittest.TestCase):
    def test_signal_ids_are_unique_and_kv_roundtrips(self):
        s = Store(":memory:")
        self.assertTrue(s.add_signal(sig(), "ACTIVE"))
        self.assertFalse(s.add_signal(sig(), "ACTIVE"))
        s.signal_status("s1", "TAKEN", "ok")
        self.assertEqual(s.signals("intraday")[0]["status"], "TAKEN")
        s.set("mode", "paper")
        self.assertEqual(s.get("mode"), "paper")
        self.assertEqual(s.get("missing", 7), 7)

    def test_config_file_defaults(self):
        cfg = EquityConfig.from_ini()
        self.assertEqual(cfg.mode, "paper")                                  # the shipped default is PAPER
        self.assertEqual((cfg.risk.risk_per_trade_pct, cfg.risk.max_daily_loss_pct, cfg.risk.min_rr), (0.5, 2.0, 2.0))
        self.assertTrue(cfg.live_confirm_each_order)
        self.assertTrue(cfg.require_validation)
        self.assertGreaterEqual(cfg.intraday.target2_r, cfg.risk.min_rr)
        self.assertGreaterEqual(cfg.swing.target2_r, cfg.risk.min_rr)
        self.assertEqual((cfg.swing.min_hold_days, cfg.swing.max_hold_days), (2, 5))
        self.assertNotIn("token", json.dumps(cfg.public()).lower())

    def test_alerts_are_once_only_with_a_key_and_respect_kind_switches(self):
        a = Alerter(AlertConfig(notify_trend_reversal=False), Store(":memory:"), push=False, clock=lambda: NOW)
        self.assertIsNotNone(a.emit("risk_limit", "x", key="k"))
        self.assertIsNone(a.emit("risk_limit", "x", key="k"))
        self.assertFalse(a.wants("trend_reversal"))
        self.assertTrue(a.wants("stop"))
        self.assertEqual(len(a.store.alerts()), 1)


if __name__ == "__main__":
    unittest.main()
