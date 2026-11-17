"""
Tests for the v3 layer and the execution engine (no network, no broker):

    extended features (causal), regime, strategy modules, trap engine, derivatives, v3 score, stop models,
    early entries, exit ladder / trailing, portfolio risk limits, order manager (idempotency, validation,
    partial fills, repricing, protective stop, reconciliation, emergency close, journal), Upstox order requests,
    the live-trading gate, settings validation, risk-adjusted statistics and the engine wiring.

    python -m unittest tests.test_v3 -v
"""
import os
import tempfile
import unittest
from datetime import date, time
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
import pandas as pd

from src.backtest import stats as bs
from src.broker.mock_data import MockProvider
from src.config import instruments as inst
from src.config import settings
from src.execution import live_gate
from src.execution import paper_trading as paper_mod
from src.execution.broker_api import CANCELLED, FILLED, OPEN, REJECTED, BrokerError, OrderRequest
from src.execution.order_manager import ExecSettings, OrderManager, _tick, make_tag
from src.execution.paper_broker import PaperBroker
from src.execution.upstox_broker import UpstoxBroker
from src.options import derivatives as deriv_mod
from src.options import selection as opt_sel
from src.risk import manager as riskm
from src.scanner import features_ext as fx
from src.scanner import modules as mods
from src.scanner import regime as reg
from src.scanner import score_v3 as v3
from src.scanner import signals as sig
from src.scanner import strategies as strat
from src.scanner import traps as trap_mod
from tests.test_strategy import _ev, _OutputsInTemp, fut_from, make_engine, synthetic_minutes

CFG_PATH = settings.CONFIG_FILE
NOW = pd.Timestamp("2026-10-26 10:00", tz="Asia/Kolkata")


def cfg():
    return settings.PBConfig.from_ini(CFG_PATH)


def trending_minutes(days=8, drift=0.00012, seed=5):
    rng = np.random.default_rng(seed)
    rows, price = [], 24000.0
    for day in pd.bdate_range("2026-10-05", periods=days):
        t0 = pd.Timestamp(day.date()).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)
        for m in range(375):
            o = price
            c = price * (1 + drift + rng.normal(0, 0.00025))
            rows.append((t0 + pd.Timedelta(minutes=m), o, max(o, c) + 1, min(o, c) - 1, c, 0.0, np.nan))
            price = c
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume", "oi"]).set_index("datetime")


# =============================================================================
# Features, regime, modules, traps, derivatives, v3 score
# =============================================================================
class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg()
        self.m = synthetic_minutes(8)
        self.f = fut_from(self.m)
        self.ff = strat.build_features(self.m, self.cfg, self.f)
        self.ext = fx.build(self.ff, self.cfg)

    def test_columns_are_causal(self):
        pos = len(self.ff.bars) - 10
        cut = self.ff.bars.index[pos] + pd.Timedelta(minutes=self.cfg.signal_minutes)
        ff2 = strat.build_features(self.m[self.m.index < cut], self.cfg, self.f[self.f.index < cut])
        e2 = fx.build(ff2, self.cfg)
        for k in self.ext.cols:
            a, b = self.ext.v(k, pos), e2.v(k, pos)
            if np.isnan(a) and np.isnan(b):
                continue
            self.assertAlmostEqual(a, b, places=9, msg=k)
        self.assertEqual(self.ext.opening_range(pos, 30), e2.opening_range(pos, 30))
        self.assertEqual(fx.liquidity_pools(self.ext, pos, 10.0), fx.liquidity_pools(e2, pos, 10.0))

    def test_opening_ranges_known_only_after_their_window(self):
        day = sorted(set(self.ff.dates))[-1]
        first = int(np.nonzero(self.ff.dates == day)[0][0])
        self.assertTrue(np.isnan(self.ext.opening_range(first, 15)[0]))      # 09:15 bar closes 09:20
        self.assertFalse(np.isnan(self.ext.opening_range(first, 5)[0]))      # 5-min range known at 09:20
        self.assertTrue(np.isnan(self.ext.initial_balance(first + 6)[0]))    # 10:00 < 10:15
        self.assertFalse(np.isnan(self.ext.initial_balance(first + 11)[0]))

    def test_vwap_source_and_proxy(self):
        self.assertEqual(self.ext.vwap_source, "futures")
        ff = strat.build_features(self.m, self.cfg)
        self.assertEqual(fx.build(ff, self.cfg).vwap_source, "twap-proxy")

    def test_sweep_detection(self):
        ext = self.ext
        pos = len(self.ff.bars) - 3
        a = self.ff.arrays
        lvl = float(a["high"][pos]) - 0.01                       # wick just above, close below (if bar closed below)
        if a["close"][pos] < lvl and a["close"][pos - 1] < lvl:
            out = fx.sweep(ext, pos, [(lvl - 1.0, "test")], 10.0, 1)
            self.assertTrue(out["bear"] or a["high"][pos] <= lvl - 1.0 + 0.5)
        self.assertEqual(fx.sweep(ext, pos, [], 10.0), {"bull": [], "bear": []})

    def test_regime_classes(self):
        seen = set()
        for pos in range(60, len(self.ff.bars), 7):
            r = reg.classify(self.ext, pos)
            self.assertIn(r.name, reg.ALL)
            seen.add(r.name)
        self.assertGreater(len(seen), 2)
        ff = strat.build_features(trending_minutes(), self.cfg)
        ext = fx.build(ff, self.cfg)
        names = [reg.classify(ext, p).name for p in range(200, len(ff.bars))]
        self.assertIn(reg.STRONG_BULL, names)
        self.assertNotIn(reg.STRONG_BEAR, names[-50:])

    def test_adapted_weights_sum_to_100(self):
        w = reg.adapted_weights(self.cfg.weights_v3, reg.Regime(reg.STRONG_BULL, 1.0))
        self.assertAlmostEqual(sum(w.values()), 100.0, places=6)
        self.assertGreater(w["Multi-timeframe"], self.cfg.w3_mtf)

    def test_modules_and_v3_score(self):
        pos = len(self.ff.bars) - 20
        ev = strat.evaluate_bar(self.ff, pos, self.cfg, "NIFTY")
        e3 = v3.rescore(ev, self.ff, self.cfg, {}, None)
        self.assertEqual([f.name for f in e3.ce.families], list(self.cfg.weights_v3))
        self.assertTrue(0 <= e3.ce.score <= 100 and 0 <= e3.pe.score <= 100)
        self.assertFalse(next(f for f in e3.ce.families if f.name == "Derivatives").available)   # no chain
        self.assertEqual(set(e3.extras["modules"]), {"Volatility compression", "Market structure", "Momentum",
                                                      "VWAP & acceptance", "Opening range", "Liquidity sweep",
                                                      "Derivatives"})
        for m in e3.extras["modules"].values():
            for side in ("ce", "pe"):
                self.assertTrue(0 <= m[side]["confidence"] <= 1)
        # levels and breakout tests are v1's, unchanged
        self.assertEqual(e3.ce.level_price, ev.ce.level_price)
        self.assertEqual(e3.ce.breakout, ev.ce.breakout)
        # causal: truncating after the bar does not change its v3 score
        cut = ev.close_time
        ff2 = strat.build_features(self.m[self.m.index < cut], self.cfg, self.f[self.f.index < cut])
        ev2 = strat.evaluate_bar(ff2, pos, self.cfg, "NIFTY")
        self.assertAlmostEqual(v3.rescore(ev2, ff2, self.cfg, {}, None).ce.score, e3.ce.score, places=6)

    def test_uncertain_regime_vetoes_entries(self):
        pos = len(self.ff.bars) - 20
        ev = strat.evaluate_bar(self.ff, pos, self.cfg, "NIFTY")
        with mock.patch.object(reg, "classify", return_value=reg.Regime(reg.UNCERTAIN, 0.3)):
            e3 = v3.rescore(ev, self.ff, self.cfg, {}, None)
        self.assertIn("uncertain", e3.ce.entry_veto)
        c2 = self.cfg.replace(block_uncertain_regime=False, trap_action="report")
        with mock.patch.object(reg, "classify", return_value=reg.Regime(reg.UNCERTAIN, 0.3)):
            e3 = v3.rescore(ev, self.ff, c2, {}, None)
        self.assertIsNone(e3.ce.entry_veto)

    def test_tracker_respects_entry_veto(self):
        c = self.cfg
        tr = sig.StageTracker("NIFTY", c)
        res = lambda d, plan, ev: (None, [], "unused")
        tr.step(_ev(23995, 24000, 70, pos=100), watch_block=None, entry_block=None, data_ok=True, data_reason="",
                risk=riskm.RiskManager(c), resolver=res)
        ev = _ev(24004, 24000, 72, breakout=True, pos=101, t="2026-10-06 11:05")
        ev.ce.entry_veto = "market regime uncertain: no new v3 entries"
        out = tr.step(ev, watch_block=None, entry_block=None, data_ok=True, data_reason="", risk=riskm.RiskManager(c),
                      resolver=res)
        self.assertEqual([a.kind for a in out], ["confirmed"])
        self.assertIn("uncertain", tr.active[inst.CE].avoid_reason)


class TrapTests(unittest.TestCase):
    def test_verdict_modes(self):
        t = [trap_mod.Trap("extended", 10, "x"), trap_mod.Trap("IV spike", 8, "y")]
        c = cfg()
        self.assertEqual(trap_mod.verdict(t, c.replace(trap_action="report")), (0.0, None))
        pts, veto = trap_mod.verdict(t, c.replace(trap_action="penalize", trap_block_points=25))
        self.assertEqual((pts, veto), (18.0, None))
        _, veto = trap_mod.verdict(t, c.replace(trap_action="penalize", trap_block_points=15))
        self.assertIn("trap points", veto)
        _, veto = trap_mod.verdict(t[:1], c.replace(trap_action="block"))
        self.assertIn("extended", veto)
        self.assertEqual(trap_mod.verdict([], c), (0.0, None))

    def test_immediate_rejection_detected(self):
        c = cfg()
        m = synthetic_minutes(6)
        ff = strat.build_features(m, c)
        ext = fx.build(ff, c)
        pos = len(ff.bars) - 5
        a = ff.arrays
        level = (a["close"][pos - 1] + a["close"][pos]) / 2
        up = a["close"][pos - 1] > a["close"][pos]       # previous close above the level, this one below
        direction = inst.CE if up else inst.PE
        ev = strat.evaluate_bar(ff, pos, c, "NIFTY")
        d = ev.side(direction)
        d.level_price = level
        traps = trap_mod.detect(direction, ev, d, ext, {}, None, c)
        self.assertIn("immediate rejection", [t.name for t in traps])


class DerivativesTests(unittest.TestCase):
    def test_chain_analysis_and_evidence(self):
        c = cfg()
        prov = MockProvider(c, ["NIFTY"], session=date(2026, 10, 16))
        snap = prov.snapshot(prov.start_sim + pd.Timedelta(hours=2))
        isnap = snap.indices["NIFTY"]
        d = deriv_mod.analyse_chain(isnap.chain, isnap.ltp, 15.0, isnap.step)
        self.assertTrue(d.available)
        self.assertGreater(d.pcr_oi, 0)
        self.assertGreaterEqual(d.call_wall, isnap.ltp - 1e-6)
        self.assertLessEqual(d.put_wall, isnap.ltp + 1e-6)
        self.assertTrue(np.isnan(d.iv_percentile))            # no history: never invented
        conf, sup, con, notes = deriv_mod.evidence(d, True)
        self.assertTrue(0 <= conf <= 1)
        self.assertTrue(any("IV percentile unavailable" in n for n in notes))
        none = deriv_mod.analyse_chain(None, 24000, 10)
        self.assertFalse(none.available)
        self.assertIsNone(deriv_mod.evidence(none, True)[0])


# =============================================================================
# Stops, early entries, exits
# =============================================================================
class TradePlanTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg()

    def test_stop_models(self):
        d = _ev(24004, 24000, 70, breakout=True).ce
        lvl = strat.underlying_plan(inst.CE, d, 24004, 10, self.cfg)
        self.assertAlmostEqual(lvl.stop, 24000 - self.cfg.sl_atr * 10)
        atr = strat.underlying_plan(inst.CE, d, 24004, 10, self.cfg.replace(stop_model="atr", atr_stop_mult=1.0))
        self.assertAlmostEqual(atr.stop, 23994)
        sw = strat.underlying_plan(inst.CE, d, 24004, 10, self.cfg.replace(stop_model="swing"), swing=23995)
        self.assertAlmostEqual(sw.stop, 23994)
        capped = strat.underlying_plan(inst.CE, d, 24004, 10, self.cfg.replace(stop_model="swing"), swing=23900)
        self.assertAlmostEqual(capped.stop, 24004 - self.cfg.max_sl_atr * 10)
        early = strat.underlying_plan(inst.CE, _ev(23997, 24000, 80).ce, 23997, 10, self.cfg, early=True)
        self.assertLess(early.stop, 23997 - 1.9)              # never on the wrong side of the entry

    def test_early_entry_reduced_risk(self):
        c = self.cfg.replace(entry_mode="early", early_min_score=75, early_max_distance_atr=0.35)

        def resolver(direction, plan, ev):
            cand = opt_sel.OptionCandidate(symbol="NIFTY 24000 CE", instrument_key="NSE_FO|1", strike=24000,
                                           option_type=direction, expiry="2026-10-13", ltp=100, bid=99.5, ask=100.5,
                                           spread_pct=1.0, volume=1e6, oi=1e7, iv=13, delta=0.55, gamma=0.002,
                                           theta=-5, vega=8, greeks_source="feed", lot_size=65)
            opt_sel.option_plan(cand, plan, c, lambda s, p, q: 0.0)
            return cand, [cand], ""

        def run(cc):
            tr = sig.StageTracker("NIFTY", cc)
            ev = _ev(23998, 24000, 82, pos=100)
            ev.extras = {"prev_high": 23990, "prev_low": 23980}
            out = tr.step(ev, watch_block=None, entry_block=None, data_ok=True, data_reason="",
                          risk=riskm.RiskManager(cc), resolver=resolver)
            return tr, out

        tr, out = run(c)
        entries = [a for a in out if a.kind == "entry"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].entry_type, "early")
        self.assertEqual(entries[0].signal_status, "TRIGGERED")
        self.assertTrue(tr.active[inst.CE].early)
        _, out_full = run(c.replace(early_risk_mult=1.0))
        full = [a for a in out_full if a.kind == "entry"][0]
        self.assertLessEqual(entries[0].option["lots"], full.option["lots"])
        # the early setup is invalidated by its stop, not by being "below the level"
        stop = tr.active[inst.CE].plan["stop"]
        out = tr.step(_ev(23999, 24000, 60, pos=101, t="2026-10-06 11:05"), watch_block=None, entry_block=None,
                      data_ok=True, data_reason="", risk=None, resolver=None)
        self.assertEqual(out, [])
        out = tr.step(_ev(stop - 1, 24000, 40, pos=102, t="2026-10-06 11:10"), watch_block=None, entry_block=None,
                      data_ok=True, data_reason="", risk=None, resolver=None)
        self.assertEqual([a.kind for a in out], ["invalidated"])

    def _book_and_pos(self, c, lots=8):
        book = paper_mod.PaperBook(c, lambda s, p, q, e="NSE_FO": 0.0)
        alert = sig.Alert(time=NOW.isoformat(), index="NIFTY", direction=inst.CE, stage=sig.ENTRY, kind="entry",
                          price=24010, level=24000, distance_atr=-1, score=70, invalidation=23990,
                          setup_key="v3|k", message="Entry setup: BUY X",
                          option={"lots": lots, "lot_size": 75, "entry_premium": 100.0, "sl_premium": 80.0,
                                  "t1_premium": 120.0, "t2_premium": 140.0, "symbol": "X", "strike": 24000,
                                  "expiry": "2026-10-28", "instrument_key": "NSE_FO|1",
                                  "ladder": [110.0, 115.0, 120.0, 130.0]})
        return book, book.open(alert, NOW.isoformat(), "NSE_FO")

    def test_ladder_partial_booking_breakeven_and_atr_trail(self):
        c = self.cfg.replace(exit_profile="ladder", ladder_r="1,1.5,2,3", ladder_book_pct="25,25,0,0",
                             trail_mode="atr", trail_atr_mult=1.0, slippage_pct=0.0)
        book, pos = self._book_and_pos(c)
        self.assertEqual(pos.profile, "ladder")
        self.assertEqual([s["qty"] for s in pos.ladder], [150, 150, 0, 0])
        t = NOW + pd.Timedelta(minutes=5)
        ev = book.mark(pos, bid=111, underlying=24020, now=t, exchange="NSE_FO", bar_close_underlying=24020,
                       ctx={"atr": 10})
        self.assertEqual([e["qty"] for e in ev], [150])
        self.assertTrue(pos.t1_done)
        self.assertEqual(pos.sl, 100.0)                         # breakeven after the first step
        ev = book.mark(pos, bid=116, underlying=24040, now=t + pd.Timedelta(minutes=5), exchange="NSE_FO",
                       bar_close_underlying=24040, ctx={"atr": 10})
        self.assertEqual([e["qty"] for e in ev], [150])
        self.assertEqual(pos.qty_open, 300)
        self.assertAlmostEqual(pos.und_trail, 24030)            # best close 24040 - 1 ATR
        ev = book.mark(pos, bid=112, underlying=24028, now=t + pd.Timedelta(minutes=10), exchange="NSE_FO",
                       bar_close_underlying=24028, ctx={"atr": 10})
        self.assertTrue(pos.closed)
        self.assertEqual(ev[-1]["reason"], "trailing stop (atr)")

    def test_classic_profile_unchanged_by_ctx(self):
        c = self.cfg.replace(slippage_pct=0.0)
        book, pos = self._book_and_pos(c, lots=2)
        self.assertEqual(pos.profile, "classic")
        ev = book.mark(pos, bid=121, underlying=24030, now=NOW + pd.Timedelta(minutes=5), exchange="NSE_FO",
                       ctx={"atr": 10, "momentum_exhausted_ce": True})
        self.assertEqual([e["reason"] for e in ev], ["target 1 (half)"])
        self.assertFalse(pos.closed)


# =============================================================================
# Portfolio risk
# =============================================================================
class RiskTests(unittest.TestCase):
    def cand(self, premium=100.0, sl=80.0, lot=75, expiry="2026-10-28", side=inst.CE):
        c = opt_sel.OptionCandidate(symbol="X", instrument_key="K", strike=24000, option_type=side, expiry=expiry,
                                    ltp=premium, bid=premium, ask=premium, spread_pct=0.5, volume=1e6, oi=1e7, iv=13,
                                    delta=0.5, gamma=0.001, theta=-3, vega=5, greeks_source="feed", lot_size=lot)
        c.entry_premium, c.sl_premium, c.est_costs = premium, sl, 0.0
        return c

    def test_daily_pct_and_weekly_limits(self):
        c = cfg().replace(max_daily_loss_rupees=60000, max_daily_loss_pct=2.0, capital=500000)
        r = riskm.RiskManager(c)
        self.assertEqual(r.daily_limit(), 10000)
        r.update(-10001, 0, 0)
        self.assertTrue(r.state.halted)
        r = riskm.RiskManager(cfg().replace(max_weekly_loss_pct=3.0, capital=500000, max_daily_loss_rupees=0))
        r.update(-6000, 0, 0, week_realised=-9500)
        self.assertTrue(r.state.halted)
        self.assertIn("weekly", r.state.halt_reason)

    def test_portfolio_caps(self):
        c = cfg().replace(capital=500000, risk_per_trade_pct=1.0, max_open_premium_pct=10, max_lots_per_expiry=3,
                          correlated_risk_mult=1.5)
        r = riskm.RiskManager(c)
        lots, _, why = r.size(self.cand())
        self.assertEqual(lots, 3)                               # 5000 / 1500 = 3 lots
        r.update(0, 0, 1, open_details=[{"index": "NIFTY", "direction": inst.CE, "expiry": "2026-10-28", "lots": 2,
                                         "premium": 15000, "risk": 3000}])
        lots, _, why = r.size(self.cand())
        self.assertEqual(lots, 1)                               # expiry cap 3 - 2 open
        r.update(0, 0, 1, open_details=[{"index": "NIFTY", "direction": inst.CE, "expiry": "2026-11-04", "lots": 4,
                                         "premium": 30000, "risk": 6000}])
        lots, _, why = r.size(self.cand())
        self.assertEqual(lots, 1)                               # correlated CE risk: 7500 - 6000 = 1 lot
        lots, _, _ = r.size(self.cand(side=inst.PE))
        self.assertEqual(lots, 2)                               # premium cap: 50000 - 30000 = 20000 / 7500
        r.update(0, 0, 1, open_details=[{"index": "NIFTY", "direction": inst.CE, "expiry": "2026-11-04", "lots": 6,
                                         "premium": 50000, "risk": 9000}])
        lots, _, why = r.size(self.cand())
        self.assertEqual(lots, 0)
        self.assertIn("premium", why)


# =============================================================================
# Execution
# =============================================================================
def paper_pos(**kw):
    d = dict(id="k1", index="NIFTY", instrument_key="NSE_FO|12345", symbol="NIFTY 24000 CE", lot_size=75, qty_open=150,
             entry_price=100.0, sl=70.0, expiry="2026-10-28", closed=False)
    d.update(kw)
    return NS(**d)


class OrderManagerTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.b = PaperBroker()
        self.om = OrderManager(self.b, ExecSettings(max_live_lots=2), journal=os.path.join(self.d, "e.db"))

    def entry(self, pos=None, quote=(99.5, 100.0, NOW), **kw):
        args = dict(data_ok=True, market_open=True, last_entry=time(14, 45))
        args.update(kw)
        return self.om.on_entry(pos or paper_pos(), quote, NOW, **args)

    def test_entry_fill_stop_and_idempotency(self):
        ok, _ = self.entry()
        self.assertTrue(ok)
        o = self.om.orders["k1|entry"]
        self.assertEqual((o.status, o.filled), (FILLED, 150))
        self.assertLessEqual(o.req["price"], 102.0)
        self.assertTrue(o.tag.startswith("pb") and len(o.tag) <= 20)
        stops = [x for x in self.om.orders.values() if x.purpose == "sl"]
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0].req["trigger_price"], 70.0)
        ok, why = self.entry()                                   # same position again
        self.assertFalse(ok)
        self.assertEqual(sum(1 for x in self.om.orders.values() if x.purpose == "entry"), 1)

    def test_validation_refusals(self):
        cases = [(dict(pos=paper_pos(instrument_key="NSE_INDEX|Nifty 50")), "not a valid F&O"),
                 (dict(pos=paper_pos(expiry="2026-10-20")), "expired"),
                 (dict(quote=(99, 100, NOW - pd.Timedelta(minutes=5))), "old"),
                 (dict(data_ok=False), "not fresh"),
                 (dict(pos=paper_pos(sl=120.0)), "invalid premium stop"),
                 (dict(pos=paper_pos(qty_open=50)), "whole number of lots")]
        for i, (kw, text) in enumerate(cases):
            kw.setdefault("pos", paper_pos())
            kw["pos"].id = f"v{i}"
            ok, why = self.entry(**kw)
            self.assertFalse(ok, text)
            self.assertIn(text, why)
        self.b.capital = 1000.0
        ok, why = self.entry(pos=paper_pos(id="m1"))
        self.assertFalse(ok)
        self.assertIn("margin", why)

    def test_partial_fill_then_timeout_cancels_remainder(self):
        self.b.partial_fill_ratio = 0.5
        self.b.set_quote("NSE_FO|12345", 99.5, 100.0)
        ok, _ = self.entry()
        lp = self.om.positions["k1"]
        self.assertEqual(lp.qty, 75)
        self.b.set_quote("NSE_FO|12345", 103, 104)               # moves away: the rest is not filled
        self.om.poll(NOW + pd.Timedelta(seconds=30))
        o = self.om.orders["k1|entry"]
        self.assertEqual(o.status, CANCELLED)
        self.assertEqual(lp.qty, 75)
        stop = self.om.orders[lp.sl_key]
        self.assertEqual(stop.req["quantity"], 75)

    def test_rejection_and_exit_reprice_to_market(self):
        self.b.reject_next = "insufficient funds"
        ok, why = self.entry()
        self.assertFalse(ok)
        self.assertEqual(self.om.orders["k1|entry"].status, REJECTED)
        self.entry(pos=paper_pos(id="k2", instrument_key="NSE_FO|222"))
        self.b.quotes["NSE_FO|222"] = (118, 119)                  # the market already fell below the exit limit
        with mock.patch.object(self.b, "set_quote"):
            self.om.on_exit(paper_pos(id="k2", instrument_key="NSE_FO|222", closed=True), 150, 150, (130, 131), NOW,
                            "target")
        ex = next(o for o in self.om.orders.values() if o.purpose == "exit")
        self.assertEqual((ex.status, ex.req["order_type"]), (OPEN, "LIMIT"))
        for k in range(1, 5):                                     # no fresh quote: re-price attempts, then MARKET
            self.om.poll(NOW + pd.Timedelta(seconds=25 * k), {})
        self.assertEqual((ex.status, ex.req["order_type"]), (FILLED, "MARKET"))
        self.assertTrue(self.om.positions["k2"].closed)
        self.assertEqual(self.b.positions(), [])

    def test_reconciliation_mismatch_blocks_entries(self):
        self.entry()
        self.b.net["NSE_FO|12345"] = 75                           # someone sold half outside this program
        res = self.om.reconcile(NOW)
        self.assertFalse(res["ok"])
        self.assertIn("mismatch", self.om.blocked)
        ok, why = self.entry(pos=paper_pos(id="k9", instrument_key="NSE_FO|999"))
        self.assertFalse(ok)
        self.assertIn("blocked", why)
        self.b.net["NSE_FO|12345"] = 150
        self.om.reconcile(NOW)
        self.assertIsNone(self.om.blocked)

    def test_emergency_close_and_journal_reload(self):
        self.entry()
        self.om.close_all(NOW, "kill switch", {"NSE_FO|12345": (90, 91)})
        self.assertTrue(self.om.positions["k1"].closed)
        self.assertEqual(self.b.positions(), [])
        om2 = OrderManager(self.b, ExecSettings(), journal=os.path.join(self.d, "e.db"))
        om2.roll(NOW.date())
        self.assertIn("k1|entry", om2.orders)
        self.assertTrue(om2.positions["k1"].closed)
        ok, _ = om2.on_entry(paper_pos(), (99.5, 100, NOW), NOW, data_ok=True, market_open=True)
        self.assertFalse(ok)                                      # the key survives a restart: no duplicate

    def test_tick_and_tag(self):
        self.assertEqual(_tick(100.01, up=True), 100.05)
        self.assertEqual(_tick(100.04, up=False), 100.0)
        self.assertEqual(make_tag("pb", "a"), make_tag("pb", "a"))
        self.assertNotEqual(make_tag("pb", "a"), make_tag("pb", "b"))


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append((method, url, params, json, headers))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class UpstoxBrokerTests(unittest.TestCase):
    def broker(self, responses):
        tok = NS(token=lambda: "TOKEN")
        return UpstoxBroker(session=_Session(responses), token_source=tok, requests_per_second=1000)

    def test_place_order_request(self):
        b = self.broker([_Resp({"status": "success", "data": {"order_ids": ["111", "112"]}})])
        ids = b.place_order(OrderRequest("NSE_FO|43919", "BUY", 150, "LIMIT", price=100.123, tag="pbabc"))
        self.assertEqual(ids, ["111", "112"])
        method, url, _, body, headers = b.session.calls[0]
        self.assertEqual((method, url), ("POST", "https://api-hft.upstox.com/v3/order/place"))
        self.assertEqual(body["instrument_token"], "NSE_FO|43919")
        self.assertEqual((body["quantity"], body["order_type"], body["transaction_type"]), (150, "LIMIT", "BUY"))
        self.assertEqual(body["price"], 100.12)
        self.assertNotIn("market_protection", body)
        self.assertEqual(headers["Authorization"], "Bearer TOKEN")

    def test_market_order_has_protection_and_index_refused(self):
        b = self.broker([_Resp({"status": "success", "data": {"order_ids": ["1"]}})])
        b.place_order(OrderRequest("BSE_FO|1", "SELL", 20, "MARKET"))
        self.assertEqual(b.session.calls[0][3]["market_protection"], -1)
        with self.assertRaises(BrokerError):
            b.place_order(OrderRequest("NSE_INDEX|Nifty 50", "BUY", 75, "LIMIT", price=1))

    def test_timeout_on_place_is_maybe_sent_and_not_retried(self):
        b = self.broker([TimeoutError("read timed out")])
        with self.assertRaises(BrokerError) as ctx:
            b.place_order(OrderRequest("NSE_FO|1", "BUY", 75, "LIMIT", price=10))
        self.assertTrue(ctx.exception.maybe_sent)
        self.assertEqual(len(b.session.calls), 1)

    def test_status_mapping_and_errors(self):
        b = self.broker([_Resp({"status": "success", "data": {"order_id": "9", "status": "complete", "quantity": 75,
                                                               "filled_quantity": 75, "average_price": 101.5}}),
                         _Resp({"status": "error", "errors": [{"message": "Insufficient funds"}]}, 400)])
        st = b.order_status("9")
        self.assertEqual((st.status, st.filled_quantity, st.average_price), (FILLED, 75, 101.5))
        with self.assertRaises(BrokerError) as ctx:
            b.cancel_order("9")
        self.assertIn("Insufficient funds", str(ctx.exception))


class LiveGateTests(unittest.TestCase):
    def test_gate_requires_every_step(self):
        d = tempfile.mkdtemp()
        f = os.path.join(d, "LIVE_ARMED")
        today = date(2026, 10, 26)
        with mock.patch.dict(os.environ, {"LIVE_TRADING_CONFIRM": ""}):
            armed, missing = live_gate.status({"mode": "live"}, today, path=f)
            self.assertFalse(armed)
            self.assertEqual(len(missing), 2)
        with mock.patch.dict(os.environ, {"LIVE_TRADING_CONFIRM": live_gate.CONFIRM_PHRASE}):
            live_gate.write_arm(date(2026, 10, 23), {}, path=f)
            self.assertFalse(live_gate.status({"mode": "live"}, today, path=f)[0])   # yesterday's arm expired
            live_gate.write_arm(today, {}, path=f)
            self.assertTrue(live_gate.status({"mode": "live"}, today, path=f)[0])
            self.assertFalse(live_gate.status({"mode": "paper"}, today, path=f)[0])
            self.assertFalse(live_gate.status({"mode": "live"}, today, mock=True, path=f)[0])
            self.assertTrue(live_gate.disarm(path=f))
            self.assertFalse(live_gate.status({"mode": "live"}, today, path=f)[0])

    def test_execution_falls_back_to_paper_when_not_armed(self):
        from src.execution.order_manager import Execution
        with mock.patch.dict(os.environ, {"LIVE_TRADING_CONFIRM": ""}):
            ex = Execution({"mode": "live"}, 500000, mock=False)
            mgr = ex.ensure(date(2026, 10, 26))
        self.assertFalse(mgr.broker.live)
        self.assertIn("not armed", ex.gate_note)


# =============================================================================
# Settings, statistics, engine wiring
# =============================================================================
class SettingsTests(unittest.TestCase):
    def test_overrides_and_validation(self):
        c = cfg()
        c3 = c.with_overrides({"entry_min_score": "41", "exit_profile": "ladder", "regime_adapt": "false"})
        self.assertEqual((c3.entry_min_score, c3.exit_profile, c3.regime_adapt), (41.0, "ladder", False))
        self.assertEqual(c.entry_min_score, cfg().entry_min_score)          # the original is untouched
        for bad in ({"exit_profile": "x"}, {"ladder_book_pct": "60,60"}, {"w3_mtf": "30"}, {"entry_mode": "x"},
                    {"trap_action": "x"}, {"early_risk_mult": "2"}):
            with self.assertRaises(ValueError):
                c.with_overrides(bad)
        with self.assertRaises(ValueError):
            ExecSettings.from_mapping({"mode": "yolo"})
        with self.assertRaises(ValueError):
            ExecSettings.from_mapping({"unknown": "1"})
        app = settings.load_app_settings(CFG_PATH)
        self.assertEqual(app.execution.get("mode"), "paper")      # default: simulated orders only

    def test_risk_adjusted(self):
        trades = [NS(day=date(2026, 1, d), entry_time=f"2026-01-{d:02d}T10:00", net=n)
                  for d, n in zip(range(1, 11), [500, -200, 300, -100, 400, -300, 200, 100, -50, 250])]
        r = bs.risk_adjusted(trades, 100000, 20)
        self.assertGreater(r["sharpe"], 0)
        self.assertGreater(r["sortino"], r["sharpe"])
        self.assertEqual(r["max_consecutive_losses"], 1)
        self.assertAlmostEqual(r["gross_profit"], 1750)


class EngineV3Tests(unittest.TestCase):
    def test_mock_session_with_v3_and_paper_execution(self):
        c = cfg()
        with tempfile.TemporaryDirectory() as d, _OutputsInTemp(d):
            s, prov = make_engine(d, c, v3_overrides={"exit_profile": "ladder", "trail_mode": "atr",
                                                      "entry_mode": "early"},
                                  execution={"mode": "paper", "lane": "v1"})
            t = prov.start_sim
            for i in range(40):
                state = s.cycle(t + pd.Timedelta(minutes=3 * i, seconds=20))
            self.assertIsNotNone(s.v3)
            self.assertEqual(s.cfg_v3.exit_profile, "ladder")
            for name, st in state["indices"].items():
                self.assertIn("v3", st)
                self.assertIn(st["regime"]["name"], reg.ALL)
                self.assertEqual(len(st["v3"]["modules"]), 7)
            self.assertEqual(state["execution"]["mode"], "paper")
            self.assertFalse(state["execution"]["live"])
            with open(settings.HTML_FILE, encoding="utf-8") as handle:
                html_text = handle.read()
            self.assertIn("Order execution", html_text)
            self.assertIn("Regime:", html_text)
            s.notifier.close()

    def test_engine_mirrors_paper_trades_to_the_order_path(self):
        c = cfg()
        with tempfile.TemporaryDirectory() as d, _OutputsInTemp(d):
            s, prov = make_engine(d, c, names=("NIFTY",), execution={"mode": "paper", "lane": "v1"})
            t = prov.start_sim + pd.Timedelta(minutes=40)
            s.cycle(t)
            isnap = s.snapshots["NIFTY"]
            row = isnap.chain.iloc[len(isnap.chain) // 2]
            ask = float(row["ce_ask"])
            alert = sig.Alert(time=t.isoformat(), index="NIFTY", direction=inst.CE, stage=sig.ENTRY, kind="entry",
                              price=isnap.ltp, level=isnap.ltp - 5, distance_atr=-0.5, score=70,
                              invalidation=isnap.ltp - 60, setup_key="2026-10-16|NIFTY|CE|test",
                              message="Entry setup: BUY test",
                              option={"lots": 2, "lot_size": isnap.lot_size, "entry_premium": ask, "sl_premium": ask * 0.7,
                                      "t1_premium": ask * 1.3, "t2_premium": ask * 1.6, "symbol": "NIFTY TEST CE",
                                      "strike": float(row["strike"]), "expiry": isnap.expiry.isoformat(),
                                      "instrument_key": row["ce_key"]})
            pos = s.book.open(alert, t.isoformat(), isnap.exchange, mock=True)
            s._exec_entry("v1", pos, isnap, t, True)
            mgr = s.execution.manager
            lp = mgr.positions[pos.id]
            self.assertEqual(lp.qty, isnap.lot_size)                    # capped at max_live_lots = 1
            self.assertTrue(any(o.purpose == "sl" for o in mgr.orders.values()))
            s._exec_entry("v1", pos, isnap, t, True)                     # a second call never doubles the order
            self.assertEqual(sum(1 for o in mgr.orders.values() if o.purpose == "entry"), 1)
            s._exec_entry("v3", pos, isnap, t, True)                     # another lane is not mirrored
            self.assertEqual(sum(1 for o in mgr.orders.values() if o.purpose == "entry"), 1)
            with open(os.path.join(d, "KILL_SWITCH"), "w"):
                pass
            s.mock = False                                               # the kill switch is ignored in mock mode
            s._kill_switch(t + pd.Timedelta(minutes=1))
            self.assertTrue(pos.closed)
            self.assertTrue(lp.closed)
            self.assertEqual(lp.qty, 0)
            s.notifier.close()


if __name__ == "__main__":
    unittest.main()
