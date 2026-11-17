"""
Strategy, signal, option, risk, paper-trading and storage tests (no network, no broker).
Ported from the original test_prebreakout.py suite; NIFTY / SENSEX only.

    python -m unittest discover -s tests
"""
import os
import tempfile
import unittest
from datetime import date, time

import numpy as np
import pandas as pd

from src.broker.mock_data import MockProvider
from src.config import instruments as inst
from src.config import settings
from src.execution import paper_trading as paper_mod
from src.notifications import service as notify_mod
from src.options import selection as opt_sel
from src.options.chain import choose_expiry
from src.risk import manager as riskm
from src.scanner import indicators as ind
from src.scanner import signals as sig
from src.scanner import strategies as strat
from src.scanner.engine import ScannerEngine
from src.scanner.store import Store

CFG_PATH = settings.CONFIG_FILE


def make_engine(d, cfg, names=("NIFTY", "SENSEX"), session=date(2026, 10, 16), **app_changes):
    """A ScannerEngine on mock data writing only inside the temporary directory `d`."""
    app = settings.load_app_settings(CFG_PATH)
    app.indices = tuple(names)
    app.kill_switch_file = os.path.join(d, "KILL_SWITCH")
    for k, v in app_changes.items():
        setattr(app, k, v)
    prov = MockProvider(cfg, list(names), session=session)
    n = notify_mod.Notifier({}, csv_path=os.path.join(d, "a.csv"), enabled=False)
    return ScannerEngine(cfg, app, prov, n, Store(os.path.join(d, "pb.db")), push=False), prov


class _OutputsInTemp:
    """Redirect the state / HTML outputs of ScannerEngine._publish into a temp dir."""

    def __init__(self, d):
        self.d = d

    def __enter__(self):
        self.old = settings.STATE_FILE, settings.HTML_FILE
        settings.STATE_FILE = settings.Path(self.d) / "s.json"
        settings.HTML_FILE = settings.Path(self.d) / "s.html"
        return self

    def __exit__(self, *exc):
        settings.STATE_FILE, settings.HTML_FILE = self.old


def synthetic_minutes(days=6, seed=3, start="2026-10-05"):
    rng = np.random.default_rng(seed)
    rows = []
    price = 24000.0
    for day in pd.bdate_range(start, periods=days):
        t0 = pd.Timestamp(day.date()).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)
        for m in range(375):
            o = price
            c = price * (1 + rng.normal(0, 0.0004))
            rows.append((t0 + pd.Timedelta(minutes=m), o, max(o, c) + 2, min(o, c) - 2, c, 0.0, np.nan))
            price = c
    f = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume", "oi"]).set_index("datetime")
    return f


def fut_from(spot, seed=4):
    rng = np.random.default_rng(seed)
    f = spot.copy() * 1.001
    f["volume"] = rng.gamma(4, 2000, len(f))
    f["oi"] = 1e6 + np.cumsum(rng.normal(0, 100, len(f)))
    return f


class IndicatorTests(unittest.TestCase):
    def test_rsi_bounds_and_ema(self):
        close = pd.Series(np.linspace(100, 120, 60))
        r = ind.rsi(close, 14)
        self.assertTrue((r.dropna() <= 100).all() and (r.dropna() >= 0).all())
        self.assertAlmostEqual(r.iloc[-1], 100.0)  # only gains
        self.assertTrue(np.isnan(ind.ema(close, 20).iloc[5]))

    def test_resample_labels_and_session(self):
        m = synthetic_minutes(1)
        bars = ind.resample(m, 5)
        self.assertEqual(len(bars), 75)
        self.assertEqual(bars.index[0].time(), time(9, 15))
        self.assertEqual(bars["minutes_in_bar"].iloc[0], 5)
        self.assertEqual(bars["high"].iloc[0], m["high"].iloc[:5].max())

    def test_vwap_needs_volume(self):
        m = synthetic_minutes(1)
        self.assertTrue(ind.session_vwap(m).isna().all())
        f = fut_from(m)
        v = ind.session_vwap(f)
        self.assertTrue(v.notna().all())

    def test_daily_levels_use_only_earlier_days(self):
        m = synthetic_minutes(8)
        daily = ind.daily_from_minutes(m)
        day = daily.index[5].date()
        lv = ind.daily_levels(daily, day)
        self.assertEqual(lv.pdh, daily["high"].iloc[4])
        self.assertEqual(lv.pdc, daily["close"].iloc[4])
        # changing the target day's own data must not change its levels
        m2 = m.copy()
        m2.loc[m2.index.date == day, "high"] += 500
        self.assertEqual(ind.daily_levels(ind.daily_from_minutes(m2), day).pdh, lv.pdh)

    def test_swing_pivot_known_only_after_right_bars(self):
        bars = pd.DataFrame({"high": [1, 2, 5, 2, 1, 1, 1], "low": [0, 1, 4, 1, 0, 0, 0]},
                            index=pd.date_range("2026-10-05 09:15", periods=7, freq="5min", tz="Asia/Kolkata"))
        p = ind.swing_pivots(bars, 2, 2)
        hi = p[p["kind"] == "H"].iloc[0]
        self.assertEqual(hi["pos"], 2)
        self.assertEqual(hi["known_at"], 4)
        self.assertTrue(ind.pivots_known(p, 3).query("kind == 'H'").empty)

    def test_zones_merge(self):
        zones = ind.build_zones([(100.0, "PDH"), (100.4, "swingH"), (110.0, "ORH")], atr_value=2.0, merge_atr=0.3)
        self.assertEqual(len(zones), 2)
        self.assertEqual(zones[0].price, 100.0)  # anchored on the named level
        self.assertEqual(zones[0].high, 100.4)


class NoLookAheadTests(unittest.TestCase):
    def test_features_identical_on_truncated_data(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        m = synthetic_minutes(6)
        f = fut_from(m)
        full = strat.build_features(m, cfg, f)
        cut_time = m.index[375 * 4 + 200]
        part = strat.build_features(m[m.index < cut_time], cfg, f[f.index < cut_time])
        pos = len(part.bars) - 2  # a complete bar in the truncated set
        a = strat.evaluate_bar(full, pos, cfg, "NIFTY")
        b = strat.evaluate_bar(part, pos, cfg, "NIFTY")
        self.assertEqual(a.time, b.time)
        self.assertAlmostEqual(a.ce.score, b.ce.score, places=6)
        self.assertAlmostEqual(a.pe.score, b.pe.score, places=6)
        self.assertEqual(a.zones, b.zones)
        for k in ("ema20", "rsi", "adx", "atr", "f_rvol", "f_vwap"):
            self.assertEqual(a.indicators[k], b.indicators[k], k)

    def test_opening_range_unknown_before_window_closes(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        m = synthetic_minutes(3)
        ff = strat.build_features(m, cfg)
        day3 = sorted(set(ff.dates))[-1]
        first = int(np.nonzero(ff.dates == day3)[0][0])
        ev = strat.evaluate_bar(ff, first + 1, cfg)  # 09:20 bar closes 09:25 < 09:30
        self.assertIsNone(ev.levels["orh"])
        ev = strat.evaluate_bar(ff, first + 2, cfg)  # 09:25 bar closes 09:30
        self.assertIsNotNone(ev.levels["orh"])


def _ev(close, level, score, breakout=False, extension=-0.5, quality=None, pos=100, t="2026-10-06 11:00", atr=10.0,
        direction=inst.CE):
    ts = pd.Timestamp(t, tz="Asia/Kolkata")
    def de(d, s, b):
        dist = (level - close) / atr if d == inst.CE else (close - level) / atr
        return strat.DirectionEval(direction=d, score=s, families=[], supporting=["x"], contradicting=[],
                                 level={"sources": "PDH", "price": level}, level_price=level, distance_atr=dist,
                                 breakout=b, breakout_quality=quality or [], extension_atr=-dist, next_level=None)
    ce = de(inst.CE, score if direction == inst.CE else 0, breakout if direction == inst.CE else False)
    pe = de(inst.PE, score if direction == inst.PE else 0, breakout if direction == inst.PE else False)
    pe.level = None
    pe.level_price = np.nan
    pe.distance_atr = np.nan
    return strat.BarEval(index="NIFTY", time=ts, close_time=ts + pd.Timedelta(minutes=5), close=close, atr=atr,
                       position=pos, ce=ce, pe=pe, indicators={}, levels={}, zones=[], structure={})


class StageTests(unittest.TestCase):
    def setUp(self):
        self.cfg = settings.PBConfig.from_ini(CFG_PATH)
        self.costs = lambda s, p, q: 0.0

    def resolver(self, direction, plan, ev):
        c = opt_sel.OptionCandidate(symbol="NIFTY 24000 CE", instrument_key="K", strike=24000, option_type=direction,
                                expiry="2026-10-13", ltp=100, bid=99.5, ask=100.5, spread_pct=1.0, volume=1e6, oi=1e7,
                                iv=13, delta=0.55, gamma=0.002, theta=-5, vega=8, greeks_source="feed", lot_size=65)
        opt_sel.option_plan(c, plan, self.cfg, self.costs)
        return c, [c], ""

    def step(self, tr, ev, **kw):
        args = dict(watch_block=None, entry_block=None, data_ok=True, data_reason="", risk=riskm.RiskManager(self.cfg),
                    resolver=self.resolver)
        args.update(kw)
        return tr.step(ev, **args)

    def test_watch_confirm_entry_and_no_duplicates(self):
        tr = sig.StageTracker("NIFTY", self.cfg)
        a = self.step(tr, _ev(23995, 24000, 70, pos=100))
        self.assertEqual([x.kind for x in a], ["high_watch"])
        a = self.step(tr, _ev(23996, 24000, 71, pos=101, t="2026-10-06 11:05"))
        self.assertEqual(a, [])  # same tier, same setup: no duplicate
        a = self.step(tr, _ev(24004, 24000, 72, breakout=True, pos=102, t="2026-10-06 11:10"))
        kinds = [x.kind for x in a]
        self.assertEqual(kinds, ["confirmed", "entry"])
        entry = a[1]
        self.assertGreater(entry.option["lots"], 0)
        self.assertIsNotNone(entry.invalidation)
        self.assertGreaterEqual(entry.rr, self.cfg.min_rr)

    def test_missed_move_do_not_chase(self):
        tr = sig.StageTracker("NIFTY", self.cfg)
        self.step(tr, _ev(23995, 24000, 70, pos=100))
        a = self.step(tr, _ev(24030, 24000, 70, breakout=True, pos=101))  # 3 ATR beyond
        self.assertEqual([x.kind for x in a], ["missed"])
        self.assertIn("do not chase", a[0].message.lower())

    def test_stale_data_blocks_entry(self):
        tr = sig.StageTracker("NIFTY", self.cfg)
        self.step(tr, _ev(23995, 24000, 70, pos=100))
        a = self.step(tr, _ev(24004, 24000, 72, breakout=True, pos=101), data_ok=False, data_reason="stale")
        self.assertEqual(a, [])

    def test_entry_needs_prior_watch(self):
        tr = sig.StageTracker("NIFTY", self.cfg)
        a = self.step(tr, _ev(24004, 24000, 72, breakout=True, pos=101))
        self.assertEqual([x.kind for x in a], ["confirmed"])
        self.assertIn("Stage A", tr.active[inst.CE].avoid_reason)

    def test_failed_breakout_invalidated(self):
        tr = sig.StageTracker("NIFTY", self.cfg)
        self.step(tr, _ev(23995, 24000, 70, pos=100))
        self.step(tr, _ev(24004, 24000, 72, breakout=True, pos=101))
        a = self.step(tr, _ev(23990, 24000, 50, pos=102))
        self.assertEqual([x.kind for x in a], ["invalidated"])

    def test_risk_limits(self):
        r = riskm.RiskManager(self.cfg)
        r.roll(date(2026, 10, 6))
        for _ in range(self.cfg.max_trades_per_index):
            r.record_open("NIFTY")
            r.state.open_positions -= 1
        ok, why = r.can_open("NIFTY")
        self.assertFalse(ok)
        r.update(-self.cfg.max_daily_loss_rupees - 1, 0, 0)
        self.assertFalse(r.can_open("SENSEX")[0])
        self.assertTrue(r.state.halted)

    def test_sizing_skips_when_one_lot_too_risky(self):
        cfg = self.cfg.replace(capital=100000, risk_per_trade_pct=0.5)
        r = riskm.RiskManager(cfg)
        c = opt_sel.OptionCandidate(symbol="X", instrument_key=None, strike=1, option_type="CE", expiry=None, ltp=300,
                                bid=299, ask=301, spread_pct=0.6, volume=1, oi=1, iv=1, delta=.5, gamma=0, theta=0,
                                vega=0, greeks_source="feed", lot_size=65)
        c.entry_premium, c.sl_premium, c.est_costs = 300, 200, 50
        lots, risk, why = r.size(c)
        self.assertEqual(lots, 0)
        self.assertIn("budget", why)


class OptionTests(unittest.TestCase):
    def test_rank_prefers_delta_and_tight_spread_not_cheapness(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        chain = MockProvider(cfg, ["NIFTY"], session=date(2026, 10, 16))._chain(
            "NIFTY", 24000, pd.Timestamp("2026-10-16 11:00", tz="Asia/Kolkata"), date(2026, 10, 20))
        plan = strat.UnderlyingPlan(inst.CE, 24000, (24000, 24010), 23980, 24030, 24060, 20, None)
        best, ranked, note = opt_sel.select_option(chain, 24000, inst.CE, plan, 50, 65, date(2026, 10, 20),
                                               pd.Timestamp("2026-10-16 11:00", tz="Asia/Kolkata"), cfg,
                                               lambda s, p, q: 40.0)
        ok = [c for c in ranked if not c.rejected]
        for c in ok:
            self.assertTrue(cfg.delta_min <= abs(c.delta) <= cfg.delta_max)
        if best is not None:
            self.assertLessEqual(abs(abs(best.delta) - cfg.delta_target), 0.2)

    def test_missing_greeks_are_estimated_and_flagged(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        chain = pd.DataFrame([{"strike": 24000.0, "ce_ltp": 120, "ce_bid": 119.5, "ce_ask": 120.5, "ce_volume": 1e6,
                               "ce_oi": 1e7, "ce_iv": np.nan, "ce_delta": np.nan, "ce_gamma": np.nan,
                               "ce_theta": np.nan, "ce_vega": np.nan, "ce_key": "K"}])
        cands = opt_sel.chain_candidates(chain, 24000, inst.CE, 50, 65, date(2026, 10, 20),
                                     pd.Timestamp("2026-10-16 11:00", tz="Asia/Kolkata"), cfg, iv_fallback=13)
        self.assertEqual(cands[0].greeks_source, "estimated")
        self.assertTrue(0.4 < cands[0].delta < 0.7)

    def test_expiry_choice_skips_0dte(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        now = pd.Timestamp("2026-10-20 10:00", tz="Asia/Kolkata")
        trade, nearest = choose_expiry([date(2026, 10, 20), date(2026, 10, 27)], now, cfg)
        self.assertEqual(nearest, date(2026, 10, 20))
        self.assertEqual(trade, date(2026, 10, 27))


class PaperTests(unittest.TestCase):
    def test_paper_exits_and_costs(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        book = paper_mod.PaperBook(cfg, lambda s, p, q, e="NSE_FO": 30.0)
        a = sig.Alert(time="2026-10-06T11:10:00+05:30", index="NIFTY", direction=inst.CE, stage=sig.ENTRY, kind="entry",
                      price=24004, level=24000, distance_atr=-0.4, score=72, invalidation=23995, setup_key="k",
                      option={"symbol": "S", "strike": 24000, "lots": 2, "lot_size": 65, "entry_premium": 100,
                              "sl_premium": 85, "t1_premium": 115, "t2_premium": 130})
        p = book.open(a, "2026-10-06T11:10:30+05:30", "NSE_FO")
        self.assertIsNotNone(p)
        self.assertIsNone(book.open(a, "2026-10-06T11:11:00+05:30", "NSE_FO"))  # no averaging / second position
        now = pd.Timestamp("2026-10-06 11:20", tz="Asia/Kolkata")
        book.mark(p, bid=116, underlying=24020, now=now, exchange="NSE_FO", bar_close_underlying=24020)
        self.assertTrue(p.t1_done)
        self.assertEqual(p.qty_open, 65)
        self.assertGreaterEqual(p.sl, 100)  # stop moved to breakeven
        book.mark(p, bid=99, underlying=24010, now=now, exchange="NSE_FO", bar_close_underlying=24010)
        self.assertTrue(p.closed)
        self.assertAlmostEqual(p.charges, 90.0)

    def test_underlying_invalidation(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        book = paper_mod.PaperBook(cfg, lambda *a: 0.0)
        a = sig.Alert(time="t", index="NIFTY", direction=inst.PE, stage=sig.ENTRY, kind="entry", price=24000, level=24010,
                      distance_atr=0, score=70, invalidation=24020, setup_key="k",
                      option={"symbol": "S", "strike": 24000, "lots": 1, "lot_size": 65, "entry_premium": 100,
                              "sl_premium": 80, "t1_premium": 120, "t2_premium": 140})
        p = book.open(a, "2026-10-06T11:10:00+05:30", "NSE_FO")
        book.mark(p, bid=95, underlying=24025, now=pd.Timestamp("2026-10-06 11:15", tz="Asia/Kolkata"),
                  exchange="NSE_FO", bar_close_underlying=24025)
        self.assertEqual(p.exit_reason, "underlying invalidation")


class SessionTests(unittest.TestCase):
    def test_blocks(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        t = lambda hm: pd.Timestamp(f"2026-10-06 {hm}", tz="Asia/Kolkata")
        self.assertIsNotNone(riskm.session_block(t("09:20"), cfg)[0])
        self.assertEqual(riskm.session_block(t("11:00"), cfg), (None, None))
        self.assertIsNotNone(riskm.session_block(t("14:50"), cfg)[1])
        self.assertIsNotNone(riskm.session_block(t("13:45"), cfg, is_expiry_day=True)[1])
        self.assertIsNotNone(riskm.session_block(t("09:40"), cfg, gap_atr_mult=1.2)[1])
        self.assertIsNotNone(riskm.session_block(t("11:00"), cfg, vix=35)[1])
        self.assertIsNotNone(riskm.session_block(t("11:00"), cfg, event="RBI")[1])


class AlertFormatTests(unittest.TestCase):
    def test_all_fields_present(self):
        a = {"time": "2026-10-06T11:10:00+05:30", "index": "NIFTY", "direction": "CE", "stage": "ENTRY SETUP",
             "kind": "entry", "price": 24004, "level": 24000, "distance_atr": -0.4, "score": 72, "supporting": ["a"],
             "contradicting": [], "option": {"symbol": "NIFTY 24000 CE", "ltp": 100, "bid": 99, "ask": 101,
                                              "spread_pct": 2, "lots": 1, "lot_size": 65},
             "entry_conditions": "zone", "invalidation": 23990, "option_risk": "SL", "targets": [1, 2, 3, 4],
             "rr": 1.5, "expiry_info": "exp", "freshness": "spot 5s", "message": "m", "mock": True}
        title, body = notify_mod.format_alert(a)
        for label in ("Index:", "Timestamp:", "Asia/Kolkata", "Direction:", "Stage:", "Current underlying price:",
                      "level:", "Distance to level:", "Setup score:", "Supporting signals:", "Contradicting signals:",
                      "Selected option contract:", "bid-ask spread", "Entry conditions:", "Underlying invalidation level:",
                      "Option risk-control method:", "Targets:", "Estimated risk-reward:",
                      "Expiry and liquidity information:", "Data freshness:"):
            self.assertIn(label, body, label)
        self.assertIn("MOCK", title)
        self.assertIn("not a probability", body)

    def test_notifier_respects_toggles_and_never_raises(self):
        sent = []
        with tempfile.TemporaryDirectory() as d:
            n = notify_mod.Notifier({"notify_early_watch": "false", "telegram": "true", "desktop_popup": "false",
                                     "min_seconds_between_messages": "0"},
                                    csv_path=os.path.join(d, "a.csv"), telegram_sender=lambda t, c, m: sent.append(m),
                                    popup=lambda *a: None)
            n.token, n.chat, n.telegram_on = "t", "c", True
            n.emit({"kind": "early_watch", "index": "NIFTY", "time": "x"})
            n.emit({"kind": "entry", "index": "NIFTY", "time": "x"})
            n.flush()
            import time as _t
            _t.sleep(0.2)
            self.assertEqual(len(sent), 1)
            self.assertTrue(os.path.exists(os.path.join(d, "a.csv")))
            n.close()
            self.assertFalse(n._worker.is_alive())   # clean shutdown of the worker thread


class ServiceMockTests(unittest.TestCase):
    def test_mock_session_runs_and_writes_state(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        with tempfile.TemporaryDirectory() as d, _OutputsInTemp(d):
            s, prov = make_engine(d, cfg)
            t = prov.start_sim
            for _ in range(130):  # 09:15 -> 11:25
                state = s.cycle(t + pd.Timedelta(seconds=20))
                t += pd.Timedelta(minutes=1)
            self.assertEqual(set(state["indices"]), {"NIFTY", "SENSEX"})
            self.assertTrue(state["mock"])
            self.assertTrue(os.path.exists(settings.STATE_FILE))
            self.assertIn("MOCK DATA", open(settings.HTML_FILE, encoding="utf-8").read())
            # restart mid-session: nothing already sent may be sent again
            before = len(s.store.recent_alerts(1000))
            s2 = ScannerEngine(cfg, s.app, prov, s.notifier, s.store, push=False)
            s2.cycle(t + pd.Timedelta(seconds=20))
            after = s.store.recent_alerts(1000)
            keys = [(a.get("setup_key"), a.get("kind")) for a in after if a.get("kind") != "data"]
            self.assertEqual(len(keys), len(set(keys)), "duplicate alert after restart")
            self.assertGreaterEqual(len(after), before)
            s.notifier.close()


class ConfigTests(unittest.TestCase):
    def test_config_validates(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        self.assertEqual(cfg.indices, ("NIFTY", "SENSEX"))
        with self.assertRaises(ValueError):
            cfg.replace(early_watch=90, high_watch=60).validate()

    def test_weights_sum_to_100(self):
        self.assertAlmostEqual(sum(settings.PBConfig.from_ini(CFG_PATH).weights.values()), 100)


class KillSwitchTests(unittest.TestCase):
    """The KILL_SWITCH file blocks new entries and squares off paper positions (live mode only)."""

    def test_kill_switch_blocks_entries_and_clears(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        with tempfile.TemporaryDirectory() as d:
            s, _ = make_engine(d, cfg, names=("NIFTY",))
            s.mock = False  # the kill switch only applies to live data
            now = pd.Timestamp("2026-10-16 11:00", tz="Asia/Kolkata")
            self.assertIsNone(s._kill_switch_reason())
            open(os.path.join(d, "KILL_SWITCH"), "w").close()
            self.assertIn("kill switch", s._kill_switch_reason())
            s._kill_switch(now)
            self.assertFalse(s.risk.can_open("NIFTY")[0])
            self.assertTrue(s.risk.snapshot()["halted"])
            os.remove(os.path.join(d, "KILL_SWITCH"))
            s._kill_switch(now)
            self.assertTrue(s.risk.can_open("NIFTY")[0])   # cleared again
            s.notifier.close()

    def test_paper_close_on_halt(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        book = paper_mod.PaperBook(cfg, lambda *a: 10.0)
        a = sig.Alert(time="t", index="NIFTY", direction=inst.CE, stage=sig.ENTRY, kind="entry", price=24000, level=23990,
                      distance_atr=0, score=70, invalidation=23980, setup_key="k",
                      option={"symbol": "S", "strike": 24000, "lots": 1, "lot_size": 65, "entry_premium": 100,
                              "sl_premium": 80, "t1_premium": 120, "t2_premium": 140})
        p = book.open(a, "2026-10-16T11:00:00+05:30", "NSE_FO")
        ev = book.close(p, bid=90, now=pd.Timestamp("2026-10-16 11:30", tz="Asia/Kolkata"), exchange="NSE_FO",
                        reason="risk: kill switch is on")
        self.assertTrue(p.closed)
        self.assertEqual(ev[0]["reason"], "risk: kill switch is on")


class GateTests(unittest.TestCase):
    """Optional compression / dominance gates (off by default)."""

    def _ev_with(self, ce, pe, comp_points, dist=0.2, pos=100, breakout=False, close=23998):
        ev = _ev(close if not breakout else 24004, 24000, ce, breakout=breakout, pos=pos, atr=10.0)
        ev.ce.families = [strat.Family("Compression", comp_points, 15)]
        ev.pe.score = pe
        return ev

    def test_watch_needs_compression_and_dominance_when_enabled(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH).replace(min_compression_pct=65, watch_min_dominance=15)
        tr = sig.StageTracker("NIFTY", cfg)
        args = dict(watch_block=None, entry_block=None, data_ok=True, data_reason="", risk=None, resolver=None)
        self.assertEqual(tr.step(self._ev_with(70, 60, 15), **args), [])          # dominance 10 < 15
        self.assertEqual(tr.step(self._ev_with(70, 40, 6, pos=101), **args), [])  # compression 40% < 65%
        a = tr.step(self._ev_with(70, 40, 12, pos=102), **args)                   # 80% compression, lead 30
        self.assertEqual([x.kind for x in a], ["high_watch"])

    def test_entry_dominance(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH).replace(entry_min_dominance=20)
        tr = sig.StageTracker("NIFTY", cfg)
        st = StageTests()
        st.cfg, st.costs = cfg, (lambda s, p, q: 0.0)
        args = dict(watch_block=None, entry_block=None, data_ok=True, data_reason="", risk=riskm.RiskManager(cfg),
                    resolver=st.resolver)
        tr.step(self._ev_with(70, 40, 12), **args)
        a = tr.step(self._ev_with(72, 60, 12, pos=101, breakout=True), **args)
        self.assertEqual([x.kind for x in a], ["confirmed"])
        self.assertIn("leads the other side", tr.active[inst.CE].avoid_reason)

    def test_insufficient_history_is_not_evaluated(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        ff = strat.build_features(synthetic_minutes(1), cfg)
        self.assertIsNone(strat.evaluate_bar(ff, cfg.min_history_bars - 1, cfg))


class ScoreV2Tests(unittest.TestCase):
    def test_v2_components_and_causality(self):
        from src.scanner import score_v2 as v2
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        m = synthetic_minutes(5)
        f = fut_from(m)
        ff = strat.build_features(m, cfg, f)
        mf = v2.build_minutes(m, f)
        pos = len(ff.bars) - 10
        ev = strat.evaluate_bar(ff, pos, cfg, "NIFTY")
        ctx = {"breadth": {"pct_above_vwap": 70.0, "count": 50}, "agree_ce": 1.0, "agree_pe": 0.0}
        out = v2.rescore(ev, ff, mf, ctx)
        names = [x.name for x in out.ce.families]
        self.assertEqual(names, list(v2.WEIGHTS))
        self.assertEqual(sum(x.max_points for x in out.ce.families), 100)
        self.assertTrue(0 <= out.ce.score <= 100 and 0 <= out.pe.score <= 100)
        self.assertEqual(out.ce.level_price, ev.ce.level_price)       # levels shared with v1
        b = {x.name: x for x in out.ce.families}["Breadth"]
        self.assertAlmostEqual(b.points, 10 * (70 - 50) / 15 if 70 < 65 else 10.0)
        # truncating the data after the bar must not change its v2 score
        cut = ev.close_time
        mf2 = v2.build_minutes(m[m.index < cut], f[f.index < cut])
        ff2 = strat.build_features(m[m.index < cut], cfg, f[f.index < cut])
        ev2 = strat.evaluate_bar(ff2, pos, cfg, "NIFTY")
        self.assertAlmostEqual(v2.rescore(ev2, ff2, mf2, ctx).ce.score, out.ce.score, places=6)

    def test_missing_feeds_are_marked_and_rescaled(self):
        from src.scanner import score_v2 as v2
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        m = synthetic_minutes(4)
        ff = strat.build_features(m, cfg)
        ev = strat.evaluate_bar(ff, len(ff.bars) - 5, cfg, "NIFTY")
        out = v2.rescore(ev, ff, v2.build_minutes(m, None), {})
        missing = {x.name for x in out.ce.families if not x.available}
        self.assertEqual(missing, {"VWAP", "Futures P-V-OI", "Breadth", "Cross-index"})


class MLTests(unittest.TestCase):
    def test_logistic_learns_and_auc(self):
        from src.scanner import ml_filter as ml
        rng = np.random.default_rng(1)
        X = rng.normal(size=(800, 3))
        y = (rng.random(800) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - X[:, 1])))).astype(float)
        X[rng.random(800) < 0.1, 2] = np.nan
        model = ml.Logistic(1.0).fit(X, y, ["a", "b", "c"])
        p = model.predict(X)
        self.assertGreater(ml.auc(y, p), 0.75)
        self.assertGreater(model.w[0], 0)
        self.assertLess(model.w[1], 0)
        again = ml.Logistic.from_dict(model.to_dict())
        self.assertTrue(np.allclose(again.predict(X), p))

    def test_walk_forward_rejects_noise(self):
        from src.scanner import ml_filter as ml
        rng = np.random.default_rng(2)
        t0 = pd.Timestamp("2024-01-01 10:00", tz="Asia/Kolkata")
        samples = [{"time": (t0 + pd.Timedelta(hours=6 * i)).isoformat(), "features": {"a": float(rng.normal()),
                    "b": float(rng.normal())}, "y": int(rng.random() < 0.4), "r": 0.0} for i in range(900)]
        for s in samples:
            s["r"] = 1.5 if s["y"] else -1.0
        res = ml.walk_forward(samples)
        self.assertTrue(res["ok"])
        self.assertFalse(res["approved"])  # pure noise must never be approved to block trades

    def test_label_outcome(self):
        from src.scanner import ml_filter as ml
        idx = pd.date_range("2026-10-06 10:00", periods=30, freq="1min", tz="Asia/Kolkata")
        up = pd.DataFrame({"open": 100.0, "high": np.linspace(100, 120, 30), "low": 99.5, "close": np.linspace(100, 119, 30)}, index=idx)
        self.assertEqual(ml.label_outcome(up, idx[0], inst.CE, 100, 96, 120, time(15, 15)), (1, ml.LABEL_R_TARGET))
        down = up.copy()
        down["low"] = np.linspace(100, 90, 30)
        down["high"] = 100.5
        self.assertEqual(ml.label_outcome(down, idx[0], inst.CE, 100, 96, 120, time(15, 15))[0], 0)
        self.assertIsNone(ml.label_outcome(up.iloc[:3].assign(high=100.5), idx[0], inst.CE, 100, 96, 120, time(15, 15)))

    def test_gate_never_shows_or_blocks_unapproved(self):
        import json as _json
        from src.scanner import ml_filter as ml
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            model = ml.Logistic(1.0).fit(np.array([[0.0], [1.0], [0.5], [0.2]]), np.array([0.0, 1.0, 1.0, 0.0]), ["v1_score"])
            for approved in (False, True):
                with open(path, "w") as h:
                    _json.dump({"approved": approved, "threshold": 1.5, "model": model.to_dict()}, h)
                os.utime(path, (time_mod() + (1 if approved else 0),) * 2)
                gate = ml.MLGate(path)
                ev = _ev(24004, 24000, 72, breakout=True)
                p, block = gate.gate(inst.CE, ev, ev.ce, None)
                if approved:
                    self.assertIsNotNone(p)
                    self.assertIn("ML filter", block)
                else:
                    self.assertEqual((p, block), (None, None))


def time_mod():
    import time as _t
    return _t.time()


def time_mod():
    import time as _t
    return _t.time()


def _entry_alert(direction=inst.CE, index="NIFTY", key="k", lots=2):
    return sig.Alert(time="2026-10-06T11:10:00+05:30", index=index, direction=direction, stage=sig.ENTRY, kind="entry",
                     price=24004, level=24000, distance_atr=-0.4, score=72,
                     invalidation=23900 if direction == inst.CE else 24100, setup_key=key,
                     option={"symbol": "S", "strike": 24000, "lots": lots, "lot_size": 65, "entry_premium": 100,
                             "sl_premium": 85, "t1_premium": 115, "t2_premium": 160})


class SpecRiskExitTests(unittest.TestCase):
    """20 Oct 2026: consecutive-loss limit, same-side exposure limit, trailing stop, failed-breakout exit."""

    def test_losing_streak(self):
        self.assertEqual(riskm.losing_streak([]), 0)
        self.assertEqual(riskm.losing_streak([-5, 10, -1, -2]), 2)
        self.assertEqual(riskm.losing_streak([-5, -1, 3]), 0)

    def test_consecutive_loss_limit_halts_for_the_day(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH).replace(max_consecutive_losses=3)
        r = riskm.RiskManager(cfg)
        r.roll(date(2026, 10, 20))
        r.update(-3000, 0, 0, open_directions=[], closed_nets=[500, -1000, -1000])
        self.assertTrue(r.can_open("NIFTY", inst.CE)[0])
        r.update(-4000, 0, 0, open_directions=[], closed_nets=[500, -1000, -1000, -1000])
        ok, why = r.can_open("NIFTY", inst.CE)
        self.assertFalse(ok)
        self.assertIn("in a row", why)
        r.update(-1000, 0, 0, open_directions=[], closed_nets=[500, -1000, -1000, -1000, 4000])
        self.assertFalse(r.can_open("NIFTY", inst.CE)[0])   # a halt lasts the rest of the day
        r.roll(date(2026, 10, 21))
        self.assertTrue(r.can_open("NIFTY", inst.CE)[0])

    def test_consecutive_loss_limit_off_by_default(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        self.assertEqual(cfg.max_consecutive_losses, 0)
        r = riskm.RiskManager(cfg)
        r.roll(date(2026, 10, 20))
        r.update(-5000, 0, 0, open_directions=[], closed_nets=[-1000] * 5)
        self.assertTrue(r.can_open("NIFTY", inst.CE)[0])

    def test_same_direction_limit(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH).replace(max_same_direction=1, max_open_positions=5)
        r = riskm.RiskManager(cfg)
        r.roll(date(2026, 10, 20))
        self.assertTrue(r.can_open("SENSEX", inst.CE)[0])
        r.record_open("NIFTY", inst.CE)
        ok, why = r.can_open("SENSEX", inst.CE)
        self.assertFalse(ok)
        self.assertIn("CE positions", why)
        self.assertTrue(r.can_open("SENSEX", inst.PE)[0])   # the other side is still allowed
        snap = r.snapshot()
        self.assertEqual((snap["open_ce"], snap["open_pe"]), (1, 0))

    def test_book_risk_inputs(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        book = paper_mod.PaperBook(cfg, lambda *a: 0.0)
        p1 = book.open(_entry_alert(index="NIFTY", key="a"), "2026-10-06T11:10:30+05:30", "NSE_FO")
        book.open(_entry_alert(direction=inst.PE, index="SENSEX", key="b"), "2026-10-06T11:12:00+05:30", "BSE_FO")
        book.close(p1, bid=90, now=pd.Timestamp("2026-10-06 11:30", tz="Asia/Kolkata"), exchange="NSE_FO", reason="x")
        inputs = book.risk_inputs(date(2026, 10, 6))
        self.assertEqual(inputs["open_directions"], [inst.PE])
        self.assertEqual(len(inputs["closed_nets"]), 1)
        self.assertLess(inputs["closed_nets"][0], 0)
        self.assertIsNotNone(book.find_open("b"))
        self.assertIsNone(book.find_open("a"))

    def test_trailing_stop_after_t1(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH).replace(trail_after_t1_pct=20.0)
        book = paper_mod.PaperBook(cfg, lambda *a: 0.0)
        p = book.open(_entry_alert(), "2026-10-06T11:10:30+05:30", "NSE_FO")
        now = pd.Timestamp("2026-10-06 11:20", tz="Asia/Kolkata")
        book.mark(p, bid=116, underlying=24020, now=now, exchange="NSE_FO", bar_close_underlying=24020)
        self.assertTrue(p.t1_done)
        self.assertAlmostEqual(p.sl, 100)                     # breakeven first
        book.mark(p, bid=150, underlying=24060, now=now, exchange="NSE_FO", bar_close_underlying=24060)
        self.assertAlmostEqual(p.sl, 120)                     # 20% under the best bid 150
        book.mark(p, bid=140, underlying=24050, now=now, exchange="NSE_FO", bar_close_underlying=24050)
        self.assertAlmostEqual(p.sl, 120)                     # never loosened
        book.mark(p, bid=119, underlying=24030, now=now, exchange="NSE_FO", bar_close_underlying=24030)
        self.assertTrue(p.closed)
        self.assertEqual(p.exit_reason, "trailing stop")

    def test_no_trailing_by_default_keeps_breakeven(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        self.assertEqual(cfg.trail_after_t1_pct, 0)
        book = paper_mod.PaperBook(cfg, lambda *a: 0.0)
        p = book.open(_entry_alert(), "2026-10-06T11:10:30+05:30", "NSE_FO")
        now = pd.Timestamp("2026-10-06 11:20", tz="Asia/Kolkata")
        book.mark(p, bid=116, underlying=24020, now=now, exchange="NSE_FO", bar_close_underlying=24020)
        book.mark(p, bid=150, underlying=24060, now=now, exchange="NSE_FO", bar_close_underlying=24060)
        self.assertAlmostEqual(p.sl, 100)

    def test_config_rejects_bad_new_values(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        with self.assertRaises(ValueError):
            cfg.replace(trail_after_t1_pct=150).validate()
        with self.assertRaises(ValueError):
            cfg.replace(max_same_direction=-1).validate()

    def _service(self, d, cfg):
        return make_engine(d, cfg, names=("NIFTY",))[0]

    def test_service_exits_on_failed_breakout(self):
        from types import SimpleNamespace
        cfg = settings.PBConfig.from_ini(CFG_PATH).replace(exit_on_failed_breakout=True)
        with tempfile.TemporaryDirectory() as d:
            s = self._service(d, cfg)
            pos = s.book.open(_entry_alert(key="setup-1"), "2026-10-16T11:00:00+05:30", "NSE_FO")
            chain = pd.DataFrame([{"strike": 24000.0, "ce_bid": 93.0, "ce_ltp": 94.0, "pe_bid": 50.0, "pe_ltp": 51.0}])
            s.snapshots = {"NIFTY": SimpleNamespace(chain=chain, exchange="NSE_FO", ltp=23995.0)}
            now = pd.Timestamp("2026-10-16 11:15", tz="Asia/Kolkata")
            inval = sig.Alert(time=now.isoformat(), index="NIFTY", direction=inst.CE, stage=sig.INVALIDATED,
                              kind="invalidated", price=23995, level=24000, distance_atr=None, score=None,
                              setup_key="setup-1")
            s.feed["NIFTY"] = {"ok": True, "chain_ok": False}
            s._exit_failed_breakout(s.book, inval, now, push=False)
            self.assertFalse(pos.closed)                      # never exits on a stale option price
            s.feed["NIFTY"] = {"ok": True, "chain_ok": True}
            s._exit_failed_breakout(s.book, inval, now, push=False)
            self.assertTrue(pos.closed)
            self.assertEqual(pos.exit_reason, "failed breakout")
            self.assertAlmostEqual(pos.exits[-1]["price"], round(93.0 * (1 - cfg.slippage_pct / 100), 2))


class BacktestStatsTests(unittest.TestCase):
    def test_bootstrap_ci_brackets_the_mean(self):
        from src.backtest import stats as bs
        rng = np.random.default_rng(1)
        nets = list(rng.normal(100, 1000, 400))
        ci = bs.bootstrap_ci(nets)
        lo, hi = ci["expectancy"]
        self.assertLess(lo, np.mean(nets))
        self.assertGreater(hi, np.mean(nets))
        self.assertGreater(ci["profit_factor"][1], ci["profit_factor"][0])
        self.assertEqual(bs.bootstrap_ci(nets), ci)               # reproducible (fixed seed)
        self.assertIn("note", bs.bootstrap_ci([1, 2, 3]))

    def test_monte_carlo_drawdown(self):
        from src.backtest import stats as bs
        nets = [1000] * 20 + [-800] * 10
        mc = bs.monte_carlo_drawdown(nets, 500000, loss_limit=60000)
        self.assertEqual(mc["actual_max_dd"], -8000)               # all losses at the end
        self.assertEqual(mc["actual_losing_streak"], 10)
        self.assertGreaterEqual(mc["dd_median"], mc["dd_p95"])    # drawdowns are negative
        self.assertGreaterEqual(mc["dd_p95"], mc["dd_worst"])
        self.assertEqual(mc["p_dd_beyond_limit"], 0.0)

    def test_segments_and_buckets(self):
        from src.backtest import stats as bs
        from types import SimpleNamespace as T
        rows = [T(net=100, d="CE"), T(net=-50, d="CE"), T(net=-20, d="PE")]
        seg = {r["group"]: r for r in bs.segment_table(rows, lambda r: r.d)}
        self.assertEqual(seg["CE"]["trades"], 2)
        self.assertAlmostEqual(seg["CE"]["profit_factor"], 2.0)
        self.assertEqual(seg["PE"]["profit_factor"], 0.0)
        self.assertEqual(bs.bucket_time("2026-10-06T09:40:00+05:30"), "09:15-10:30")
        self.assertEqual(bs.bucket_time("2026-10-06T14:00:00+05:30"), "13:30-close")
        self.assertEqual(bs.bucket_dte(0), "0-1 day")
        self.assertEqual(bs.bucket_dte(5), "4+ days")
        self.assertEqual(bs.bucket_vix(18.2), "VIX 17+")
        first, last = bs.split_halves([T(entry_time=str(i)) for i in range(10)])
        self.assertEqual((len(first), len(last)), (7, 3))


if __name__ == "__main__":
    unittest.main()


class AlertOnlyModeTests(unittest.TestCase):
    def test_alert_only_mode_opens_no_paper_positions(self):
        cfg = settings.PBConfig.from_ini(CFG_PATH)
        alert = _entry_alert(key="setup-x")
        for paper, expected in ((True, 1), (False, 0)):
            with tempfile.TemporaryDirectory() as d, _OutputsInTemp(d):
                s, prov = make_engine(d, cfg, names=("NIFTY",), paper_trading=paper)
                t = prov.start_sim + pd.Timedelta(hours=2)
                s.cycle(t)                                  # day set-up and silent replay first
                tracker = s.trackers["NIFTY"]
                tracker.step = lambda *a, **k: [alert]      # the next completed bar produces an ENTRY SETUP
                s.cycle(t + pd.Timedelta(minutes=5))
                self.assertEqual(len(s.book.positions), expected, f"paper_trading={paper}")
                state = s._publish(t, market_open=True)
                self.assertEqual(state["trading_mode"], "paper" if paper else "alerts only")
                s.notifier.close()
