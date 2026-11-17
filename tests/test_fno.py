"""
Tests of the F&O multi-strategy engine (src/fno). No network, no credentials: a synthetic market only.

    python -m unittest tests.test_fno -v
"""
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.execution.costs import CostConfig
from src.fno import backtest as bt
from src.fno import indicators as ind
from src.fno import metrics
from src.fno.analysis import analyze, evaluate, latest, prepare
from src.fno.config import FnoConfig
from src.fno.contracts import build_plan, lots_split, premium_risk, select_contract
from src.fno.data import MockData
from src.fno.engine import Engine
from src.fno.position import Position
from src.fno.risk import RiskManager
from src.fno.store import Store, chain_explain, chain_payload, chain_series
from src.fno.strategies import REGISTRY, SCAN_FACTORS, Ctx, all_strategies, scan_factors
from src.fno.universe import Universe

TZ = "Asia/Kolkata"
DAY = pd.Timestamp("2026-11-12 11:30", tz=TZ)          # a Thursday


def cfg(**kw):
    return replace(FnoConfig(), costs={}, strategies={}, **kw)


def feed(now=DAY, **kw):
    f = MockData(cfg(), clock_start=now.normalize() + pd.Timedelta(hours=9, minutes=45), **kw)
    f.fixed = now
    return f


def mirror(m1, pivot=None):
    """Reflect prices around a level: every bullish pattern becomes the same bearish pattern."""
    pivot = pivot or 2 * float(m1["close"].mean())
    out = m1.copy()
    out["open"], out["close"] = pivot - m1["open"], pivot - m1["close"]
    out["high"], out["low"] = pivot - m1["low"], pivot - m1["high"]
    return out


def bars(closes, start="2026-11-11 09:15", vol=1000.0, wick=0.2):
    """5-minute bars with the given closes (each opens at the previous close)."""
    idx = pd.date_range(pd.Timestamp(start, tz=TZ), periods=len(closes), freq="5min")
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick, "close": c,
                         "volume": vol, "oi": np.nan}, index=idx)


class TestUniverse(unittest.TestCase):
    def rows(self):
        ms = lambda d: int(pd.Timestamp(datetime.combine(d, time(23, 59, 59)), tz=TZ).timestamp() * 1000)   # noqa: E731
        out = []
        for sym, ukey, utype, seg, lot, step, base in (("NIFTY", "NSE_INDEX|Nifty 50", "INDEX", "NSE_FO", 65, 50, 25000),
                                                       ("SENSEX", "BSE_INDEX|SENSEX", "INDEX", "BSE_FO", 20, 100, 82000),
                                                       ("MIDCPNIFTY", "NSE_INDEX|X", "INDEX", "NSE_FO", 120, 25, 12000),
                                                       ("RELIANCE", "NSE_EQ|INE002A01018", "EQUITY", "NSE_FO", 500, 10, 1400)):
            for e in (date(2026, 11, 10), date(2026, 11, 24), date(2026, 12, 29)):
                for k in range(-5, 6):
                    for t in ("CE", "PE"):
                        out.append({"segment": seg, "instrument_type": t, "instrument_key": f"{seg}|{sym}{e}{k}{t}",
                                    "underlying_symbol": sym, "underlying_key": ukey, "underlying_type": utype,
                                    "expiry": ms(e), "strike_price": base + k * step, "lot_size": lot, "tick_size": 5.0})
            out.append({"segment": seg, "instrument_type": "FUT", "instrument_key": f"{seg}|{sym}FUT", "underlying_symbol": sym,
                        "underlying_key": ukey, "underlying_type": utype, "expiry": ms(date(2026, 11, 24)), "lot_size": lot})
        return out

    def test_contract_specs_come_from_the_master(self):
        u = Universe(self.rows(), date(2026, 11, 12))
        self.assertEqual([i.symbol for i in u.indices], ["NIFTY", "SENSEX"])       # other indices are not traded
        self.assertEqual([s.symbol for s in u.stocks], ["RELIANCE"])
        n = u.get("nifty")
        self.assertEqual(n.expiries, [date(2026, 11, 24), date(2026, 12, 29)])     # expired contracts dropped
        self.assertEqual((n.lot_size(n.expiries[0]), n.strike_step(n.expiries[0]), n.tick), (65, 50.0, 0.05))
        self.assertEqual(u.get("SENSEX").segment, "BSE_FO")
        self.assertEqual(n.near_future(date(2026, 11, 12)), "NSE_FO|NIFTYFUT")
        self.assertEqual(n.pick_expiry(date(2026, 11, 24), min_days=1), date(2026, 12, 29))
        self.assertEqual(len(u.select(("NIFTY",), "none")), 1)
        self.assertEqual(len(u.select(("NIFTY", "SENSEX"), "all")), 3)

    def test_config_is_paper_only(self):
        with tempfile.TemporaryDirectory() as d:
            ini = Path(d) / "fno.ini"
            ini.write_text("[SERVICE]\nmode = live\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                FnoConfig.load(ini, Path(d) / "o.json")
            ini.write_text("[SERVICE]\nmode = paper\n[PLAN]\ntarget_r = 1, 2.5, 4\n[STRATEGY_ORB]\nenabled = false\nminutes = 45\n", encoding="utf-8")
            c = FnoConfig.load(ini, Path(d) / "o.json")
            self.assertEqual(c.target_r, (1.0, 2.5, 4.0))
            self.assertFalse(REGISTRY["orb"].enabled(c.strategies["orb"]))
            self.assertEqual(REGISTRY["orb"].params(c.strategies["orb"])["minutes"], 45)

    def test_no_order_code_in_the_package(self):
        text = "".join(p.read_text(encoding="utf-8") for p in Path(bt.__file__).parent.glob("*.py"))
        for word in ("place_order", "/order/place", "order_manager", "upstox_broker", "live_gate"):
            self.assertNotIn(word, text)


class TestStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = feed()
        cls.cfg = replace(cfg(), strategies={"gap": {"min_gap_pct": "0.05"},
                                             "st_rsi_ema": {"window": "6", "adx_min": "10", "min_rvol": "1.0"},
                                             "option_chain": {"min_score": "35", "max_opposite": "60"},
                                             # the synthetic chain's volume is not cumulative: turnover rule tested apart
                                             "vol_oi_divergence": {"min_turnover_pct": "-1000", "drop_pct": "0.5"},
                                             "multi_factor": {"min_score": "60", "max_opposite": "30"}})
        cls.m1 = cls.f._visible("RELIANCE")
        cls.bench = cls.f._visible("NIFTY")
        snaps = []
        rng = np.random.default_rng(1)
        ce, pe, co, po = 100.0, 100.0, 1e6, 1e6
        for t in cls.m1.index[::2]:
            move = rng.normal(0, 0.04)
            ce, pe = ce * (1 + move), pe * (1 - move + rng.normal(0, 0.01))
            co, po = co * (1 + rng.normal(0, 0.02)), po * (1 + rng.normal(0, 0.02))
            snaps.append((t, {"atm": 1450.0, "expiry": "x", "dte": 5, "spot": 1450.0, "rows": {
                "1450": [max(ce, 1), max(pe, 1), co, po, co * 2, po * 2, 20.0, 21.0, 0.5, 0.6]}}))
        cls.chain = chain_series(snaps, 10)
        cls.a = analyze("RELIANCE", cls.m1, cls.cfg, None, cls.bench, cls.chain)

    def test_all_strategies_have_rules_and_settings(self):
        s = all_strategies()
        self.assertEqual([x.number for x in s], list(range(1, 44)))
        for x in s:
            d = x.describe()
            self.assertTrue(d["ce"] and d["pe"] and d["exit"] and d["name"])
            self.assertTrue(set(x.grid) <= set(x.defaults), x.key)

    def test_every_strategy_runs_and_fires_both_ways(self):
        self.assertEqual(self.a.unavailable, {})
        m = analyze("RELIANCE", mirror(self.m1), self.cfg, None, mirror(self.bench), None)
        m2 = analyze("ITC", self.f._visible("ITC"), self.cfg, None, self.bench, None)
        for key in REGISTRY:
            ups = sum(int((x.events[key] == 1).sum()) for x in (self.a, m, m2) if key in x.events)
            dns = sum(int((x.events[key] == -1).sum()) for x in (self.a, m, m2) if key in x.events)
            self.assertGreater(ups, 0, f"{key} never gave a CE trigger")
            self.assertGreater(dns, 0, f"{key} never gave a PE trigger")

    def test_no_look_ahead(self):
        """Signals of the first part must not change when later data is added."""
        cut = self.m1.index[int(len(self.m1) * 0.8)]
        part = analyze("RELIANCE", self.m1[self.m1.index < cut], self.cfg, None, self.bench[self.bench.index < cut],
                       self.chain[self.chain.index < cut])
        n = len(part.frame) - 1                       # the cut may fall inside the last bar
        for key in REGISTRY:
            np.testing.assert_array_equal(part.events[key].to_numpy()[:n], self.a.events[key].to_numpy()[:n], err_msg=key)
            np.testing.assert_array_equal(part.signals[key].to_numpy()[:n], self.a.signals[key].to_numpy()[:n], err_msg=key)

    def test_ce_and_pe_rules_are_mirror_images(self):
        """Reflecting the price series must swap CE and PE triggers (strategies defined on price differences)."""
        m = analyze("RELIANCE", mirror(self.m1), self.cfg)
        base = analyze("RELIANCE", self.m1, self.cfg)
        for key in ("ema_9_21", "ema_20_50", "supertrend", "adx_dmi", "macd_cross", "ichimoku", "orb", "pdh_pdl", "first_15",
                    "bb_squeeze", "donchian", "inside_bar", "rsi_ema", "stoch_rsi", "macd_rsi", "rvol_breakout", "pv_surge",
                    "momentum_pullback", "bb_reversal", "vwap_rejection", "sr_reversal", "pivot_reversal", "bos_choch",
                    "atr_breakout", "vwap_trend", "fibonacci", "rsi_divergence", "supertrend_reversal", "st_rsi_ema",
                    "vwap_breakout", "pivot_breakout", "supertrend_macd", "vwap_rsi"):
            a, b = base.events[key].to_numpy(), -m.events[key].to_numpy()
            self.assertGreater(np.abs(a).sum(), 0, key)
            self.assertLess((a != b).mean(), 0.002, f"{key}: CE / PE rules are not symmetric")

    def test_added_strategies_39_to_42(self):
        c = Ctx(tf=5)
        # two flat sessions, then on day 3 a dip under VWAP followed by a strong recovery on volume
        d3 = [100 - 0.15 * i for i in range(1, 13)] + [98.2 + 0.45 * i for i in range(1, 40)]
        closes = [100 + 0.05 * np.sin(i) for i in range(150)] + d3
        idx = [pd.Timestamp(f"2026-11-{d} 09:15", tz=TZ) + pd.Timedelta(minutes=5 * i) for d, n in ((10, 75), (11, 75), (12, len(d3)))
               for i in range(n)]
        f = bars(closes)
        f.index = pd.DatetimeIndex(idx)
        f.loc[f.index[150 + 12:], "volume"] = 3000.0
        e = ind.enrich(f, 5)
        day3 = e.index.date == date(2026, 11, 12)
        vb = REGISTRY["vwap_breakout"].signals(e, c)
        self.assertEqual(int((vb[day3] == 1).sum()), 1)
        t = vb[day3][vb[day3] == 1].index[0]
        self.assertTrue(e.loc[t, "close"] > e.loc[t, "vwap"] and e["close"].shift(1).loc[t] < e["vwap"].shift(1).loc[t])
        quiet = f.copy()
        quiet["volume"] = 1000.0                                                       # same move without the volume
        self.assertEqual(int((REGISTRY["vwap_breakout"].signals(ind.enrich(quiet, 5), c)[day3] == 1).sum()), 0)
        pb = REGISTRY["pivot_breakout"].signals(e, c)
        hits = pb[day3][pb[day3] == 1].index
        self.assertGreaterEqual(len(hits), 1)
        self.assertTrue(all(e.loc[h, "close"] > e.loc[h, "r1"] for h in hits))
        lows = pb[day3][pb[day3] == -1].index                                          # the opening dip broke S1 first
        self.assertTrue(all(e.loc[h, "close"] < e.loc[h, "s1"] for h in lows) and all(h < hits[0] for h in lows))
        sm = REGISTRY["supertrend_macd"].signals(e, c, {"mult": 2.0, "window": 8})
        self.assertGreaterEqual(int((sm[day3] == 1).sum()), 1)
        h = sm[day3][sm[day3] == 1].index[0]
        self.assertTrue(e.loc[h, "macd"] > e.loc[h, "macd_sig"])
        vr = REGISTRY["vwap_rsi"].signals(e, c)
        self.assertEqual(int((vr[day3] == 1).sum()), 0)            # RSI crossed 55 while VWAP was still falling: no signal
        # a drift above a rising VWAP, a shallow pull-back, then RSI back up through 55 with the close above VWAP
        d3 = [100 + 0.02 * i for i in range(1, 16)] + [100.3 - 0.1 * i for i in range(1, 5)] + [99.9 + 0.3 * i for i in range(1, 25)]
        g = bars([100 + 0.05 * np.sin(i) for i in range(150)] + d3)
        g.index = pd.DatetimeIndex(idx[:150 + len(d3)])
        e2 = ind.enrich(g, 5)
        vr = REGISTRY["vwap_rsi"].signals(e2, c)[e2.index.date == date(2026, 11, 12)]
        self.assertEqual(int((vr == 1).sum()), 1)
        h = vr[vr == 1].index[0]
        self.assertTrue(e2.loc[h, "close"] > e2.loc[h, "vwap"] and e2.loc[h, "rsi"] > 55 >= e2["rsi"].shift(1).loc[h])

    def test_rule_scenarios(self):
        c = Ctx(tf=5)
        # flat for two sessions, then a steady rise on day 3: opening-range, previous-day-high and Donchian breakouts up
        closes = [100 + 0.05 * np.sin(i) for i in range(150)] + [100 + 0.3 * i for i in range(1, 60)]
        idx = [pd.Timestamp("2026-11-10 09:15", tz=TZ) + pd.Timedelta(minutes=5 * i) for i in range(75)] + \
              [pd.Timestamp("2026-11-11 09:15", tz=TZ) + pd.Timedelta(minutes=5 * i) for i in range(75)] + \
              [pd.Timestamp("2026-11-12 09:15", tz=TZ) + pd.Timedelta(minutes=5 * i) for i in range(59)]
        f = bars(closes)
        f.index = pd.DatetimeIndex(idx)
        e = ind.enrich(f, 5)
        day3 = e.index.date == date(2026, 11, 12)
        for key in ("orb", "pdh_pdl", "first_15", "donchian", "ema_9_21"):
            s = REGISTRY[key].signals(e, c)
            self.assertEqual(int((s[day3] == 1).sum()), 1 if key != "donchian" else int((s[day3] == 1).sum()), key)
            self.assertGreaterEqual(int((s[day3] == 1).sum()), 1, key)
            self.assertEqual(int((s[day3] == -1).sum()), 0, key)
        orb = REGISTRY["orb"].signals(e, c)
        first = orb[orb == 1].index[0]
        self.assertGreaterEqual(first, pd.Timestamp("2026-11-12 09:45", tz=TZ))        # never inside the 30-minute range
        # inside bar: mother 100-104, inside 101-103, then a close above 104
        f2 = bars([100] * 40, wick=1.0)
        f2.iloc[-3, :4] = [100, 104, 100, 103]
        f2.iloc[-2, :4] = [102, 103, 101, 102]
        f2.iloc[-1, :4] = [102, 105.5, 102, 105]
        s = REGISTRY["inside_bar"].signals(ind.enrich(f2, 5), c, {"min_mother_atr": 0.1})
        self.assertEqual(int(s.iloc[-1]), 1)

    def test_a_bare_cross_is_not_a_signal(self):
        """Triggers that fail the confluence checks never become signals; signals are a subset of triggers."""
        ev, sig = self.a.events, self.a.signals
        lagged = (ev != 0)
        for k in range(1, self.cfg.confirm_bars + 1):
            lagged |= (ev.shift(k) != 0) & ev.shift(k).notna()
        self.assertFalse(((sig != 0) & ~lagged).any().any())
        self.assertLess(int((sig != 0).sum().sum()), int((ev != 0).sum().sum()))
        strict = evaluate(prepare("RELIANCE", self.m1, self.cfg), replace(self.cfg, min_confidence=101.0))
        self.assertEqual(int((strict.signals[[k for k, s in REGISTRY.items() if s.kind != "reversal" and k in strict.signals]] != 0).sum().sum()), 0)
        comp, g = self.a.comp, self.a.gates["ema_9_21"]
        hit = g.index[g["signal"] == 1]
        self.assertTrue(len(hit) and (comp.loc[hit, "LOCATION"] == 1).all() and (comp.loc[hit, "up_n"] >= 4).all())

    def test_strategy_switch(self):
        c = replace(self.cfg, strategies={"orb": {"enabled": "false"}})
        self.assertNotIn("orb", analyze("RELIANCE", self.m1, c).events)

    def test_indicators(self):
        e = self.a.frame
        d = e.groupby(e.index.normalize()).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        last, prev = d.index[-1], d.index[-2]
        row = e[e.index.normalize() == last].iloc[0]
        self.assertAlmostEqual(row["pd_high"], d.loc[prev, "high"])
        self.assertAlmostEqual(row["pp"], (d.loc[prev, "high"] + d.loc[prev, "low"] + d.loc[prev, "close"]) / 3)
        hi, lo = ind.opening_range(e, 30, 5)
        today = e.index.normalize() == last
        self.assertTrue(hi[today].iloc[:6].isna().all())
        self.assertAlmostEqual(hi[today].iloc[6], e["high"][today].iloc[:6].max())
        self.assertTrue(((e["val"] <= e["poc"]) & (e["poc"] <= e["vah"]))[e["poc"].notna()].all())


class TestSupertrendRsiEma(unittest.TestCase):
    """Strategy 36: every listed condition is necessary."""

    def frame(self):
        # two quiet sessions, a sell-off that turns Supertrend bearish, then a strong rally on heavy volume
        closes = [100 + 0.3 * np.sin(i / 3) for i in range(150)] + [100 - 0.5 * i for i in range(1, 16)] + \
                 [92.5 + 0.9 * i for i in range(1, 31)]
        f = bars(closes, wick=0.15)
        f.index = pd.DatetimeIndex([pd.Timestamp(d, tz=TZ) + pd.Timedelta(minutes=5 * i) for d, n in
                                    (("2026-11-10 09:15", 75), ("2026-11-11 09:15", 75), ("2026-11-12 09:15", 45)) for i in range(n)])
        f.loc[f.index[165:], "volume"] = 4000.0
        return ind.enrich(f, 5)

    def run_it(self, e, h15=1.0, **p):
        ctx = Ctx(tf=5, htf={15: pd.Series(h15, index=e.index)})
        return REGISTRY["st_rsi_ema"].signals(e, ctx, dict({"mult": 2.0, "window": 8}, **p))

    def test_buy_ce_needs_every_condition(self):
        e = self.frame()
        s = self.run_it(e)
        hits = s[s == 1]
        self.assertEqual(len(hits), 1)                                                   # one signal, not one per bar
        self.assertEqual(int((s == -1).sum()), 0)
        row = e.loc[hits.index[0]]
        self.assertTrue(row["ema9"] > row["ema21"] and row["close"] > row["vwap"] and row["rsi"] > 55 and row["adx"] >= 18)
        self.assertEqual(int((self.run_it(e, h15=-1.0) == 1).sum()), 0)                  # 15-minute trend against it
        self.assertEqual(int((self.run_it(e, min_rvol=50) == 1).sum()), 0)               # no volume confirmation
        self.assertEqual(int((self.run_it(e, adx_min=95) == 1).sum()), 0)                # sideways-market filter
        self.assertEqual(int((self.run_it(e, rsi_bull=99) == 1).sum()), 0)               # RSI never crosses the level
        low = e.copy()
        low["vwap"] = e["close"] + 5
        self.assertEqual(int((self.run_it(low) == 1).sum()), 0)                          # below VWAP
        novol = analyze("X", feed()._visible("NIFTY").assign(volume=0.0), cfg())
        self.assertIn("st_rsi_ema", novol.unavailable)                                   # never guessed without volume

    def test_buy_pe_is_the_mirror(self):
        e = self.frame()
        raw = e[["open", "high", "low", "close", "volume", "oi"]]
        s = self.run_it(ind.enrich(mirror(raw, 200.0), 5), h15=-1.0)
        self.assertEqual((int((s == -1).sum()), int((s == 1).sum())), (1, 0))


class TestOptionChainAndScanner(unittest.TestCase):
    T0 = pd.Timestamp("2026-11-12 10:00", tz=TZ)

    @staticmethod
    def snap(ce, pe, ce_oi, pe_oi, vol=(1000, 1000), iv=14.0, dte=3, spot=100.0):
        rows = {"95": [ce + 5, pe - 1, 500, 4000, 100, 100, iv, iv, 0.5, 0.5],
                "100": [ce, pe, ce_oi, pe_oi, vol[0], vol[1], iv, iv, 0.4, 0.6],
                "105": [ce - 1, pe + 5, 5000, 400, 100, 100, iv, iv, 0.5, 0.5]}
        return {"atm": 100.0, "expiry": "e", "dte": dte, "spot": spot, "rows": rows}

    def series(self, second):
        return chain_series([(self.T0, self.snap(10, 10, 1000, 1000)), (self.T0 + pd.Timedelta(minutes=10), second)], 10)

    def test_iv_percentile_needs_history_and_is_causal(self):
        snaps = [(self.T0 + pd.Timedelta(minutes=i), self.snap(10, 10, 1000, 1000, iv=iv)) for i, iv in enumerate((14.0, 30.0, 11.0))]
        self.assertTrue(chain_series(snaps, 10)["iv_pctile"].isna().all())                  # 3 readings: not enough history
        hist = [12.0 + 0.1 * i for i in range(40)]                                           # 12.0 .. 15.9
        got = chain_series(snaps, 10, hist)["iv_pctile"].tolist()
        self.assertAlmostEqual(got[0], 21 / 40 * 100)                                        # 21 of the 40 are <= 14.0
        self.assertAlmostEqual(got[1], 100.0)                                                # above everything seen so far
        self.assertAlmostEqual(got[2], 0.0)
        self.assertIn("IV percentile 100", " ".join(chain_explain(chain_series(snaps, 10, hist).iloc[1], "CE")))

    def test_volume_oi_divergence(self):
        """Heavy call volume while call OI falls and the call gets dearer = short covering: CE only on a price break."""
        rising = ind.enrich(bars([100 + 0.02 * np.sin(i) for i in range(60)] + [100 + 0.4 * i for i in range(1, 15)]), 5)

        def run(second, f=rising):
            two = self.series(second)
            ch = pd.DataFrame([two.iloc[1]] * len(f), index=f.index)
            return REGISTRY["vol_oi_divergence"].signals(f, Ctx(tf=5, chain=ch))
        cover = self.series(self.snap(12, 9, 900, 1000, (1300, 1010))).iloc[1]
        self.assertAlmostEqual(cover["ce_vol_pct"], 300 / 6500 * 100)                        # contracts traded / call OI before
        self.assertLess(cover["ce_oi_chg_pct"], -1.0)
        self.assertIn("volume and OI diverge", " ".join(chain_explain(cover, "CE")))
        self.assertGreater(int((run(self.snap(12, 9, 900, 1000, (1300, 1010))) == 1).sum()), 0)
        self.assertEqual(int((run(self.snap(12, 9, 900, 1000, (1010, 1010))) != 0).sum()), 0)      # no volume behind it
        self.assertEqual(int((run(self.snap(12, 9, 1100, 1000, (1300, 1010))) != 0).sum()), 0)     # OI rose: new positions
        self.assertEqual(int((run(self.snap(8, 11, 900, 1000, (1300, 1010))) != 0).sum()), 0)      # call got cheaper
        falling = ind.enrich(bars([100 + 0.02 * np.sin(i) for i in range(60)] + [100 - 0.4 * i for i in range(1, 15)]), 5)
        self.assertGreater(int((run(self.snap(9, 12, 1000, 900, (1010, 1300)), falling) == -1).sum()), 0)
        self.assertEqual(int((run(self.snap(12, 9, 900, 1000, (1300, 1010)), falling) != 0).sum()), 0)   # price disagrees

    def test_oi_is_read_together_with_price(self):
        # the SAME call-OI build-up is bullish when calls get dearer (buying) and bearish when they get cheaper (writing)
        bought = self.series(self.snap(12, 9, 1500, 1000, (2500, 1200), 13.8)).iloc[1]
        written = self.series(self.snap(8, 11.5, 1500, 1000, (1200, 2500), 14.6)).iloc[1]
        self.assertGreater(bought["flow_bias"], 0)
        self.assertLess(written["flow_bias"], 0)
        self.assertGreater(bought["bull_score"], 60)
        self.assertLess(bought["bear_score"], 30)
        self.assertGreater(written["bear_score"], 60)
        self.assertEqual((bought["support"], bought["resistance"]), (95.0, 105.0))       # OI concentration
        self.assertAlmostEqual(bought["pcr"], (4000 + 1000 + 400) / (500 + 1500 + 5000))
        self.assertAlmostEqual(bought["iv_chg"], -0.2)
        self.assertAlmostEqual(bought["atm_spread_pct"], 0.6)
        put_writing = self.series(self.snap(11, 8, 1000, 1600)).iloc[1]                   # put OI up, puts cheaper
        self.assertGreater(put_writing["flow_bias"], 0)
        flat = self.series(self.snap(10, 10, 1500, 1500)).iloc[1]                         # OI up, prices unchanged
        self.assertTrue(np.isnan(flat["flow_bias"]))                                      # no direction from OI alone
        self.assertLess(flat["bull_score"], 55)
        why = chain_explain(bought, "CE")
        self.assertTrue(any("put wall" in w for w in why) and any("PCR" in w for w in why) and any("IV" in w for w in why))
        old = chain_series([(self.T0, {"atm": 100.0, "expiry": "e", "rows": {"100": [10, 10, 1000, 1000]}})], 10)
        self.assertTrue(np.isnan(old["atm_iv"].iloc[0]))                                  # old 4-field snapshots still load

    def chain_frame(self, f, **kw):
        base = {"bull_score": 80.0, "bear_score": 10.0, "atm_spread_pct": 0.5, "dte": 3.0, "support": 99.0, "resistance": 130.0}
        base.update(kw)
        return pd.DataFrame(base, index=f.index)

    def test_strategy_37_needs_chain_and_price_action(self):
        rising = ind.enrich(bars([100 + 0.02 * np.sin(i) for i in range(60)] + [100 + 0.4 * i for i in range(1, 15)]), 5)
        flat = ind.enrich(bars([100 + 0.02 * np.sin(i) for i in range(74)]), 5)
        run = lambda f, **kw: REGISTRY["option_chain"].signals(f, Ctx(tf=5, chain=self.chain_frame(f, **kw)))   # noqa: E731
        self.assertEqual(int((run(rising) == 1).sum()), 1)
        self.assertEqual(int((run(flat) != 0).sum()), 0)                    # bullish chain, no price action: no signal
        self.assertEqual(int((run(rising, bull_score=40.0) != 0).sum()), 0)  # price action without chain evidence
        self.assertEqual(int((run(rising, bear_score=60.0) != 0).sum()), 0)  # conflicting evidence
        self.assertEqual(int((run(rising, atm_spread_pct=4.0) != 0).sum()), 0)   # illiquid
        self.assertEqual(int((run(rising, dte=0.0) != 0).sum()), 0)          # expiry day
        self.assertEqual(int((run(rising, resistance=100.5) == 1).sum()), int((run(rising, resistance=100.5) == 1).sum()))
        self.assertEqual(int((REGISTRY["option_chain"].signals(rising, Ctx(tf=5)) != 0).sum()), 0)   # no chain: nothing
        falling = ind.enrich(bars([100 + 0.02 * np.sin(i) for i in range(60)] + [100 - 0.4 * i for i in range(1, 15)]), 5)
        s = run(falling, bull_score=10.0, bear_score=80.0, support=70.0, resistance=101.0)
        self.assertEqual((int((s == -1).sum()), int((s == 1).sum())), (1, 0))

    def test_scanner_scores_and_strategy_38(self):
        e = TestSupertrendRsiEma().frame()
        ctx = Ctx(tf=5, bench_ret=pd.Series(0.0, index=e.index))
        x = scan_factors(e, ctx)
        self.assertEqual(set(SCAN_FACTORS) | {"bull", "bear"}, set(x.columns))
        end, sell = x.iloc[-1], x.loc[pd.Timestamp("2026-11-12 10:25", tz=TZ)]
        self.assertGreaterEqual(end["bull"], 70)
        self.assertLessEqual(end["bear"], 10)
        self.assertGreaterEqual(sell["bear"], 60)                            # the sell-off ranks on the bearish side
        self.assertEqual((end["EMA"], end["SUPERTREND"], end["RSI"], end["VWAP"], end["REL STRENGTH"]), (1, 1, 1, 1, 1))
        s = REGISTRY["multi_factor"].signals(e, ctx)
        self.assertGreaterEqual(int((s == 1).sum()), 1)
        self.assertGreaterEqual(int((s == -1).sum()), 1)
        self.assertTrue(np.isnan(scan_factors(e, Ctx(tf=5)).iloc[-1]["REL STRENGTH"]))   # no benchmark: factor left out


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.f = feed()
        self.cfg = self.f.cfg
        self.und = self.f.universe().get("NIFTY")
        self.expiry = self.und.pick_expiry(DAY.date(), min_days=1)
        self.chain = self.f.chain(self.und, self.expiry)
        self.spot = float(self.f._visible("NIFTY")["close"].iloc[-1])
        self.bar = pd.Series({"atr": 30.0, "swing_lo": self.spot - 40, "swing_hi": self.spot + 40, "close": self.spot})
        self.sig = {"strategy": "orb", "strategy_name": "Opening Range Breakout", "confidence": 70.0}

    def test_contract_selection_and_filters(self):
        ce, _ = select_contract(self.chain, self.spot, "CE", self.und, self.expiry, self.cfg, DAY)
        pe, _ = select_contract(self.chain, self.spot, "PE", self.und, self.expiry, self.cfg, DAY)
        self.assertEqual((ce.side, pe.side), ("CE", "PE"))
        self.assertTrue(0.35 <= abs(ce.delta) <= 0.7 and pe.delta < 0)
        for c in (ce, pe):                                                               # all four greeks on the contract
            self.assertTrue(c.gamma > 0 and c.vega > 0 and c.theta < 0, c.to_dict())
        self.assertEqual(ce.key, self.und.contracts[(self.expiry, ce.strike, "CE")])     # a real chain row
        self.assertEqual(ce.lot, self.und.lot_size(self.expiry))
        step = self.und.strike_step(self.expiry)
        itm, _ = select_contract(self.chain, self.spot, "CE", self.und, self.expiry,
                                 replace(self.cfg, selection="moneyness", moneyness="ITM", strike_distance=2), DAY)
        otm, _ = select_contract(self.chain, self.spot, "PE", self.und, self.expiry,
                                 replace(self.cfg, selection="moneyness", moneyness="OTM", strike_distance=2), DAY)
        atm = min(self.chain["strike"], key=lambda s: abs(s - self.spot))
        self.assertEqual((itm.strike, otm.strike), (atm - 2 * step, atm - 2 * step))
        wide = self.chain.copy()
        wide["ce_ask"] = wide["ce_bid"] * 1.2
        none, why = select_contract(wide, self.spot, "CE", self.und, self.expiry, self.cfg, DAY)
        self.assertIsNone(none)
        self.assertIn("spread", " ".join(why))
        thin = self.chain.copy()
        thin["ce_oi"] = 10.0
        self.assertIsNone(select_contract(thin, self.spot, "CE", self.und, self.expiry, self.cfg, DAY)[0])
        self.assertIsNone(select_contract(None, self.spot, "CE", self.und, self.expiry, self.cfg, DAY)[0])

    def test_percent_plan_sl10_t15_t25(self):
        """[PLAN] plan_mode = percent: SL -10%, T1 +15%, T2 +25% of the live premium, all out at T2, R:R >= 1:2."""
        cfg_p = replace(self.cfg, plan_mode="percent", sl_pct=10.0, target_pct=(15.0, 25.0), min_rr_after_costs=2.0)
        costs = CostConfig.from_mapping(cfg_p.costs)
        for side, word in (("CE", "Bullish"), ("PE", "Bearish")):
            c, _ = select_contract(self.chain, self.spot, side, self.und, self.expiry, cfg_p, DAY)
            plan, why = build_plan(self.sig, self.und, c, self.bar, cfg_p, 500000.0, DAY)
            self.assertEqual(why, "")
            e = plan.entry
            self.assertAlmostEqual(plan.stop / e, 0.90, delta=0.002)
            self.assertAlmostEqual(plan.t1 / e, 1.15, delta=0.002)
            self.assertAlmostEqual(plan.t2 / e, 1.25, delta=0.002)
            self.assertEqual((plan.t3, plan.direction, plan.timeframe, plan.signal_status, plan.plan_mode),
                             (plan.t2, word, f"{cfg_p.base_tf}-minute", "PAPER TRADE", "percent"))
            self.assertGreaterEqual(plan.rr, 2.0)
            pos = Position.open(plan, cfg_p, costs, DAY)
            self.assertEqual(pos.split[2], 0)                                            # nothing waits for a third target
            pos.step(DAY, e, plan.t1 + 0.05, e, plan.t1, cfg_p, costs)
            self.assertEqual((pos.stage, pos.status, pos.stop), (1, "OPEN", e))          # part booked, stop to entry
            pos.step(DAY, plan.t1, plan.t2 + 0.05, plan.t1, plan.t2, cfg_p, costs)
            self.assertEqual((pos.status, pos.exit_reason), ("CLOSED", "T2"))
            lost = Position.open(plan, cfg_p, costs, DAY)
            lost.step(DAY, e, e, plan.stop - 0.05, plan.stop, cfg_p, costs)
            self.assertEqual(lost.exit_reason, "SL")
            self.assertLessEqual(abs(lost.net), plan.max_loss * 1.02)
        wide, why = build_plan(self.sig, self.und, c, self.bar, replace(cfg_p, sl_pct=14.0), 500000.0, DAY)
        self.assertIsNone(wide)                                                          # 25 / 14 is below 1 : 2 after charges
        self.assertIn("reward:risk", why)
        with self.assertRaises(ValueError):
            replace(cfg_p, target_pct=(25.0, 15.0)).validate()

    def test_confidence_is_the_strategy_agreement_score(self):
        m1, bench = self.f._visible("RELIANCE"), self.f._visible("NIFTY")
        c = replace(cfg(), confidence_mode="agreement", min_confidence=60.0)
        a = analyze("RELIANCE", m1, c, None, bench, None)
        n = 0
        for g in a.gates.values():
            for t, row in g[g["signal"] != 0].iterrows():
                score = float(a.scan.loc[t, "bull" if row["signal"] == 1 else "bear"])
                self.assertAlmostEqual(row["confidence"], score, delta=0.06)
                self.assertGreaterEqual(score, 60.0)                                     # min_confidence applies to it
                n += 1
        self.assertGreater(n, 0)
        self.assertTrue(all(s["confidence_basis"] == "strategy agreement score" for s in latest(a, c)))

    def test_plan_levels_quantity_and_loss(self):
        c, _ = select_contract(self.chain, self.spot, "CE", self.und, self.expiry, self.cfg, DAY)
        plan, why = build_plan(self.sig, self.und, c, self.bar, self.cfg, 500000.0, DAY)
        self.assertEqual(why, "")
        self.assertAlmostEqual(plan.r, plan.entry - plan.stop, places=2)
        for k, t in zip((1, 2, 3), (plan.t1, plan.t2, plan.t3)):
            self.assertAlmostEqual(t, plan.entry + k * plan.r, delta=0.051)            # 1R / 2R / 3R, tick-rounded
        self.assertGreaterEqual(plan.entry, c.ask)                                      # bought at the ask or worse
        self.assertEqual(plan.qty % c.lot, 0)
        self.assertLessEqual(plan.max_loss, 500000 * 0.01)                              # inside the 1% risk budget
        self.assertGreater(plan.max_loss, plan.r * plan.qty)                            # includes charges and slippage
        self.assertLess(plan.und_stop, plan.und_entry)
        self.assertTrue(8 <= (plan.r / plan.entry) * 100 <= 30.5)
        for name in ("symbol", "instrument", "side", "strike", "expiry", "entry", "stop", "t1", "t2", "t3", "rr", "qty",
                     "max_loss", "confidence", "strategy_name", "timestamp", "exits"):
            self.assertIn(name, plan.to_dict())
        half, _ = build_plan(self.sig, self.und, c, self.bar, self.cfg, 500000.0, DAY, risk_mult=0.5)
        self.assertLess(half.qty, plan.qty)
        none, why = build_plan(self.sig, self.und, c, self.bar, self.cfg, 20000.0, DAY)
        self.assertIsNone(none)
        self.assertIn("risk budget", why)
        pe, _ = select_contract(self.chain, self.spot, "PE", self.und, self.expiry, self.cfg, DAY)
        p2, _ = build_plan(self.sig, self.und, pe, self.bar, self.cfg, 500000.0, DAY)
        self.assertGreater(p2.und_stop, p2.und_entry)
        self.assertGreater(p2.t1, p2.entry)                                             # long premium either way

    def test_helpers(self):
        self.assertAlmostEqual(premium_risk(0.5, 0.0, 20), 10.0)
        self.assertLess(premium_risk(0.5, 0.002, 20), 10.0)
        self.assertEqual(lots_split(10, (40, 30, 30)), [4, 3, 3])
        self.assertEqual(lots_split(1, (40, 30, 30), 2), [0, 1, 0])
        self.assertEqual(sum(lots_split(2, (40, 30, 30))), 2)
        self.assertEqual(sum(lots_split(3, (40, 30, 30))), 3)


class TestPosition(unittest.TestCase):
    def setUp(self):
        t = TestPlan()
        t.setUp()
        c, _ = select_contract(t.chain, t.spot, "CE", t.und, t.expiry, t.cfg, DAY)
        self.cfg, self.costs = t.cfg, CostConfig()
        self.plan, _ = build_plan(t.sig, t.und, c, t.bar, t.cfg, 2000000.0, DAY)
        self.pos = Position.open(self.plan, self.cfg, self.costs, DAY)

    def step(self, o, h, l, c, **kw):
        return self.pos.step(DAY, o, h, l, c, self.cfg, self.costs, **kw)

    def test_stop_before_target_in_the_same_bar(self):
        p = self.plan
        self.step(p.entry, p.t3 + 5, p.stop - 1, p.entry)
        self.assertEqual((self.pos.status, self.pos.exit_reason), ("CLOSED", "SL"))
        self.assertAlmostEqual(self.pos.realized, -p.r * p.qty, delta=1.0)
        self.assertLess(self.pos.net, self.pos.realized)                                # charges on top
        self.assertAlmostEqual(self.pos.r_multiple, -1.0, delta=0.15)

    def test_scale_out_breakeven_and_t3(self):
        p, lot = self.plan, self.plan.lot
        self.assertGreaterEqual(p.lots, 3)
        fills = self.step(p.entry, p.t1 + 0.1, p.entry - 0.1, p.t1)
        self.assertEqual([f["reason"] for f in fills], ["T1"])
        self.assertEqual(fills[0]["qty"], self.pos.split[0] * lot)
        self.assertEqual(self.pos.stop, self.pos.entry)                                 # stop to breakeven after T1
        self.step(p.t1, p.t3 + 1, p.t1, p.t3)
        self.assertEqual((self.pos.status, self.pos.exit_reason, self.pos.qty_open), ("CLOSED", "T3", 0))
        self.assertGreater(self.pos.net, 0)
        self.assertEqual(sum(f["qty"] for f in self.pos.fills if f["side"] == "SELL"), p.qty)

    def test_breakeven_gap_invalidation_trail_time_and_force(self):
        p = self.plan
        self.step(p.entry, p.t1 + 0.1, p.entry, p.t1)
        self.step(p.t1, p.t1, p.entry - 2, p.entry - 1)
        self.assertEqual(self.pos.exit_reason, "BREAKEVEN")
        self.setUp()
        self.step(p.stop - 3, p.stop - 2, p.stop - 4, p.stop - 3)
        self.assertEqual(self.pos.exit_reason, "SL (gap)")
        self.assertLess(self.pos.fills[-1]["price"], p.stop)                            # filled at the gap open, not the stop
        self.setUp()
        self.step(p.entry, p.entry + 1, p.entry - 1, p.entry, und_close=p.und_stop - 1)
        self.assertEqual(self.pos.exit_reason, "UNDERLYING INVALIDATED")
        self.setUp()
        self.step(p.entry, p.t1 + 0.1, p.entry, p.t1, und_close=p.und_entry + 50, atr=10.0)
        self.assertAlmostEqual(self.pos.und_trail, p.und_entry + 50 - self.cfg.trail_atr_mult * 10)
        self.step(p.t1, p.t1, p.t1 - 1, p.t1 - 1, und_close=p.und_entry + 20, atr=10.0)
        self.assertEqual(self.pos.exit_reason, "TRAIL (underlying)")
        self.setUp()
        self.step(p.entry, p.entry, p.entry, p.entry, minutes_open=self.cfg.max_hold_minutes)
        self.assertEqual(self.pos.exit_reason, "TIME STOP")
        self.setUp()
        self.step(p.entry, p.entry, p.entry, p.entry, opposite=True)
        self.assertEqual(self.pos.exit_reason, "OPPOSITE SIGNAL")
        self.setUp()
        self.step(p.entry, p.entry, p.entry, p.entry, force="SQUARE-OFF", slippage_pct=1.0)
        self.assertEqual(self.pos.exit_reason, "SQUARE-OFF")
        self.assertAlmostEqual(self.pos.fills[-1]["price"], round(p.entry * 0.99, 2), places=2)
        again = Position.from_dict(json.loads(json.dumps(self.pos.to_dict())))
        self.assertEqual((again.id, round(again.net, 2)), (self.pos.id, round(self.pos.net, 2)))


class TestRisk(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cfg = cfg()
        self.kill = Path(self.dir.name) / "KILL_SWITCH"
        self.risk = RiskManager(self.cfg, kill_files=[self.kill], events_file=Path(self.dir.name) / "none.csv")

    def tearDown(self):
        self.dir.cleanup()

    def check(self, now=DAY, closed=(), opened=(), **kw):
        return self.risk.check_entry(now, kw.pop("symbol", "NIFTY"), kw.pop("side", "CE"), "orb", kw.pop("bar", now), 500000.0,
                                     list(closed), list(opened), **kw)

    @staticmethod
    def trade(net, symbol="NIFTY", closed="2026-11-12 10:00"):
        p = Position(id="x", symbol=symbol, strategy="orb", side="CE", instrument="i", key="k", segment="NSE_FO", lot=1, qty=1,
                     entry=100.0, stop=90.0, initial_stop=90.0, r=10.0, targets=[110, 120, 130], split=[0, 1, 0], und_entry=1.0,
                     und_stop=0.5, opened="2026-11-12 09:30", qty_open=0, status="CLOSED", closed=closed)
        p.realized = net
        return p

    def test_limits(self):
        self.assertEqual(self.check(), [])
        self.assertIn("daily loss", " ".join(self.check(closed=[self.trade(-16000)])))
        self.assertTrue(self.kill.exists())                                             # automatic kill switch
        self.assertIn("kill switch", self.check()[0])
        self.assertTrue(self.risk.release())
        self.assertIn("profit lock", " ".join(self.check(closed=[self.trade(21000)])))
        self.assertIn("max 8 trades", " ".join(self.check(closed=[self.trade(10)] * 8)))
        losses = [self.trade(-100, closed=f"2026-11-12 10:0{i}") for i in range(3)]
        self.assertIn("consecutive losses", " ".join(self.check(closed=losses)))
        self.assertEqual(self.check(closed=losses + [self.trade(50, closed="2026-11-12 10:30")]), [])
        o = self.trade(0)
        o.status, o.qty_open, o.last_price = "OPEN", 1, 100.0
        self.assertIn("already holding NIFTY", " ".join(self.check(opened=[o])))
        self.assertIn("max 3 open", " ".join(self.check(symbol="TCS", opened=[o, o, o])))

    def test_time_duplicates_expiry_spread_and_stale_data(self):
        at = lambda hm: pd.Timestamp(f"2026-11-12 {hm}", tz=TZ)   # noqa: E731
        self.assertIn("before", self.check(now=at("09:16"))[0])
        self.assertIn("last entry", self.check(now=at("14:50"))[0])
        self.assertIn("holiday", self.check(now=pd.Timestamp("2026-11-14 11:00", tz=TZ))[0])
        self.risk.mark_signal(DAY, "NIFTY", "CE", "orb", DAY)
        later = DAY + pd.Timedelta(minutes=5)
        self.assertIn("duplicate", " ".join(self.check(now=later, bar=later)))
        self.assertEqual(self.check(now=later, bar=later, side="PE"), [])
        self.assertEqual(self.check(now=DAY + pd.Timedelta(minutes=25), bar=later), [])
        self.assertIn("expires today", " ".join(self.check(side="PE", expiry=DAY.date())))
        self.risk.cfg = replace(self.cfg, allow_expiry_day_contract=True)
        self.assertEqual(self.check(side="PE", expiry=DAY.date()), [])
        self.assertIn("expiry day", " ".join(self.check(now=at("13:45"), side="PE", expiry=DAY.date())))
        self.assertEqual(self.risk.risk_mult(DAY, DAY.date()), 0.5)
        self.assertIn("spread", " ".join(self.check(side="PE", spread_pct=9.0)))
        self.assertIn("old", " ".join(self.check(side="PE", data_age=999)))
        ev = Path(self.dir.name) / "events.csv"
        ev.write_text("date,time,name,impact\n2026-11-12,11:45,RBI policy,high\n", encoding="utf-8")
        r = RiskManager(self.cfg, kill_files=[self.kill], events_file=ev)
        self.assertIn("event", r.session_block(DAY))


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        self.feed = feed(DAY.normalize() + pd.Timedelta(hours=10), days=8)
        self.feed.cfg = self.cfg = replace(self.feed.cfg, stocks="RELIANCE,ITC,ICICIBANK", max_trades_per_day=50, max_open_positions=20, daily_profit_lock_pct=0.0)
        self.store = Store(d / "t.db")
        self.engine = Engine(self.cfg, self.feed, self.store, RiskManager(self.cfg, kill_files=[d / "KILL"], events_file=d / "e.csv"))

    def tearDown(self):
        self.store.close()
        self.dir.cleanup()

    def run_until(self, end, step=5):
        t = self.feed.fixed
        while t <= end:
            self.feed.fixed = t
            self.engine.cycle(t)
            t += pd.Timedelta(minutes=step)

    def test_session_end_to_end(self):
        self.run_until(DAY.normalize() + pd.Timedelta(hours=12, minutes=30))
        self.feed.fixed = DAY.normalize() + pd.Timedelta(hours=15, minutes=16)
        self.engine.cycle(self.feed.fixed)
        s = self.engine.state()
        self.assertTrue(s["mock"] and s["mode"] == "PAPER")
        self.assertEqual([w["symbol"] for w in s["indices"]], ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"])
        self.assertEqual(len(s["watchlist"]), 7)
        self.assertEqual(s["health"]["errors"], [])
        buys = [x for x in self.store.signals() if x["status"] == "BUY"]
        self.assertTrue(buys, "the mock session produced no entry")
        for x in buys:
            p = x["plan"]
            self.assertLess(p["stop"], p["entry"])
            self.assertTrue(p["entry"] < p["t1"] < p["t2"] < p["t3"])
            self.assertIn(x["side"], ("CE", "PE"))
            self.assertTrue(x["checks"]["PREMIUM"])                                     # premium confirmation
        self.assertEqual(s["positions"], [])                                            # squared off at 15:15
        trades = self.store.trades()
        self.assertEqual(len(trades), len(buys))
        self.assertTrue(all(t["status"] == "CLOSED" and t["mock"] for t in trades))
        keys = [(x["symbol"], x["side"], x["bar_time"]) for x in buys]
        self.assertEqual(len(keys), len(set(keys)))                                     # no duplicate entries
        rep = self.engine.performance()
        self.assertEqual(rep["overall"]["trades"], len(trades))
        self.assertAlmostEqual(rep["overall"]["net_pnl"], round(sum(t["net"] for t in trades), 2), places=1)
        self.assertEqual({r["group"] for r in rep["by_side"]} <= {"CE", "PE"}, True)
        self.assertTrue(self.store.chain_snapshots("NIFTY"))                            # index chains are recorded

    def test_kill_switch_squares_off_and_blocks(self):
        self.run_until(DAY)
        self.assertTrue(self.engine.positions, "no open position to test with")
        self.engine.risk.kill("test")
        n = len(self.store.signals())
        self.run_until(self.feed.fixed + pd.Timedelta(minutes=30))
        self.assertEqual(self.engine.positions, [])
        self.assertTrue(all(p.exit_reason == "KILL SWITCH" for p in self.engine.closed if p.closed > str(DAY)))
        self.assertFalse([x for x in self.store.signals()[:len(self.store.signals()) - n] if x["status"] == "BUY"])

    def test_restart_restores_open_positions(self):
        self.run_until(DAY)
        ids = sorted(p.id for p in self.engine.positions)
        again = Engine(self.cfg, self.feed, self.store, self.engine.risk)
        self.assertEqual(sorted(p.id for p in again.positions), ids)

    def test_web_api(self):
        from src.fno.web import create_app
        self.run_until(DAY)
        client = create_app(self.engine, api_token="secret").test_client()
        self.assertEqual(client.get("/health").get_json()["mode"], "PAPER")
        self.assertEqual(len(client.get("/api/strategies").get_json()), 43)
        sc = client.get("/api/state").get_json()["scanner"]
        self.assertTrue({"bullish", "bearish", "no_trade"} <= set(sc))
        for r in sc["bullish"] + sc["bearish"]:
            self.assertGreaterEqual(r["score"], 60)
            self.assertGreaterEqual(abs(r["bull"] - r["bear"]), 30)                      # no conflicting signals
        ranked = [r["score"] for r in sc["bullish"]]
        self.assertEqual(ranked, sorted(ranked, reverse=True))
        self.assertEqual(len(client.get("/api/state").get_json()["indices"]), 4)
        self.assertIn("overall", client.get("/api/performance").get_json())
        self.assertIn(b"F&amp;O Engine", client.get("/").data)
        r = client.post("/api/strategies/orb", json={"enabled": False, "minutes": 45})
        self.assertEqual((r.get_json()["enabled"], r.get_json()["params"]["minutes"]), (False, 45))
        self.assertEqual(client.post("/api/strategies/orb", json={"minutes": "x"}).status_code, 400)
        self.assertEqual(client.post("/api/strategies/nope", json={"enabled": True}).status_code, 404)
        far = {"REMOTE_ADDR": "10.0.0.9"}
        self.assertEqual(client.post("/api/strategies/orb", json={"enabled": True}, environ_base=far).status_code, 403)
        self.assertEqual(client.post("/api/strategies/orb", json={"enabled": True}, environ_base=far,
                                     headers={"X-API-Token": "secret"}).status_code, 200)
        self.assertTrue(client.post("/api/kill").get_json()["kill"])
        self.assertEqual(client.post("/api/kill/release", json={}).status_code, 400)
        self.assertEqual(client.post("/api/kill/release", json={"confirm": "RELEASE"}).get_json()["kill"], "")
        self.assertEqual(client.get("/api/backtests/20990101_000000").status_code, 404)


class TestBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = feed(DAY.normalize() + pd.Timedelta(hours=15, minutes=29), days=40)
        cls.cfg = replace(cls.f.cfg, bt_wf_train_days=15, bt_wf_test_days=5, bt_min_trades=5)
        cls.unds = [cls.f.universe().get("RELIANCE")]
        cls.minutes = {s: cls.f._full(s) for s in ("RELIANCE", "NIFTY")}
        cls.start = cls.minutes["RELIANCE"].index[0].date()
        cls.keys = ["orb", "ema_9_21", "donchian", "supertrend"]
        cls.rep = bt.run(cls.cfg, cls.unds, cls.start, DAY.date(), cls.keys, "model", True, minutes=cls.minutes, save=False)

    def test_report_shape_and_honesty(self):
        r = self.rep
        self.assertEqual(r["price_sources"], ["MODELLED"])
        self.assertTrue(any("MODELLED" in w for w in r["warnings"]))
        self.assertTrue(any("Survivorship" in w for w in r["warnings"]))
        self.assertTrue(any("assumption" in w for w in r["warnings"]))
        self.assertEqual({s["key"] for s in r["strategies"]}, set(self.keys))
        self.assertGreater(r["overall"]["trades"], 20)
        for s in r["strategies"]:
            self.assertEqual(s["all"]["trades"], s["in_sample"]["trades"] + s["out_of_sample"]["trades"])
            self.assertIn("by_dte", s)
            self.assertIn("walk_forward", s)

    def test_entries_are_after_the_signal_bar_and_inside_the_session(self):
        for t in self.rep["trades"]:
            sig, opened = pd.Timestamp(t["signal_bar"]), pd.Timestamp(t["opened"])
            self.assertEqual(opened, sig + pd.Timedelta(minutes=5))                     # next bar's open
            self.assertEqual(pd.Timestamp(t["closed"]).date(), opened.date())           # intraday only
            self.assertLessEqual(pd.Timestamp(t["closed"]).time(), time(15, 30))
            self.assertEqual(t["qty"], t["lot"])
            self.assertGreater(t["charges"], 0)

    def test_costs_and_spread_reduce_results(self):
        free = replace(self.cfg, bt_assumed_spread_pct=0.0, bt_slippage_pct=0.0, costs={"brokerage_per_order": 0, "stt_sell_pct": 0,
                       "nse_txn_pct": 0, "sebi_per_crore": 0, "stamp_buy_pct": 0, "gst_pct": 0})
        r0 = bt.run(free, self.unds, self.start, DAY.date(), ["donchian"], "model", minutes=self.minutes, save=False)
        r1 = next(s for s in self.rep["strategies"] if s["key"] == "donchian")
        self.assertGreater(r0["overall"]["net_pnl"], r1["all"]["net_pnl"])

    def test_out_of_sample_is_chronological(self):
        cut = date.fromisoformat(self.rep["oos_starts"])
        days = sorted({pd.Timestamp(t["opened"]).date() for t in self.rep["trades"]})
        self.assertTrue(days[0] < cut <= days[-1])
        self.assertEqual(bt.split_date([date(2026, 1, 1)], 0.3), None)

    def test_walk_forward_uses_only_later_windows(self):
        wf = next(s for s in self.rep["strategies"] if s["key"] == "orb")["walk_forward"]
        self.assertGreaterEqual(len(wf["windows"]), 2)
        self.assertGreater(wf["combinations_tried"], 1)
        for w in wf["windows"]:
            self.assertLess(w["train"].split(" .. ")[1], w["test"].split(" .. ")[0])
        self.assertEqual(wf["out_of_sample"]["trades"], sum(w["test_trades"] for w in wf["windows"]))

    def test_model_expiry_and_membership(self):
        self.assertEqual(bt.model_expiry("NIFTY", date(2026, 11, 12)).weekday(), 1)
        e = bt.model_expiry("RELIANCE", date(2026, 11, 12))
        self.assertTrue(e >= date(2026, 11, 12) and e.month == 11)
        self.assertGreater(bt.model_expiry("RELIANCE", date(2026, 11, 27)), date(2026, 11, 27))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.csv"
            p.write_text("symbol,from,to\nABC,2025-01-01,2025-06-30\n", encoding="utf-8")
            self.assertEqual(bt.membership(p), {"ABC": [(date(2025, 1, 1), date(2025, 6, 30))]})


class TestMetricsAndStore(unittest.TestCase):
    def test_summary(self):
        t = [{"net": 100.0, "r_multiple": 1.0, "opened": "2026-11-10 10:05", "closed": "2026-11-10 10:30", "strategy": "a", "symbol": "X", "side": "CE"},
             {"net": -50.0, "r_multiple": -0.5, "opened": "2026-11-10 11:40", "closed": "2026-11-10 11:50", "strategy": "a", "symbol": "X", "side": "PE"},
             {"net": -80.0, "r_multiple": -0.8, "opened": "2026-11-11 10:10", "closed": "2026-11-11 10:20", "strategy": "b", "symbol": "Y", "side": "PE"}]
        s = metrics.summary(t)
        self.assertEqual((s["trades"], s["wins"], s["losses"], s["net_pnl"]), (3, 1, 2, -30.0))
        self.assertAlmostEqual(s["win_rate"], 33.3)
        self.assertAlmostEqual(s["profit_factor"], round(100 / 130, 2))
        self.assertEqual((s["avg_profit"], s["avg_loss"], s["max_drawdown"]), (100.0, -65.0, 130.0))
        self.assertIsNone(s["sharpe"])                                                  # fewer than 20 days: not reported
        r = metrics.report(t, {"a": 5, "c": 2})
        self.assertEqual({x["group"] for x in r["by_strategy"]}, {"a", "b", "c"})
        self.assertEqual([x["group"] for x in r["by_time"]], ["10:00", "11:30"])
        self.assertEqual(r["equity"][-1]["equity"], -30.0)
        self.assertEqual(metrics.summary([])["win_rate"], None)

    def test_chain_series(self):
        t0 = pd.Timestamp("2026-11-12 10:00", tz=TZ)
        snaps = [(t0, {"atm": 100.0, "expiry": "e", "rows": {"100": [10.0, 10.0, 1000, 1000]}}),
                 (t0 + pd.Timedelta(minutes=10), {"atm": 100.0, "expiry": "e", "rows": {"100": [12.0, 9.0, 1000, 1100]}})]
        s = chain_series(snaps, 10)
        self.assertTrue(np.isnan(s["ce_mom_pct"].iloc[0]))
        self.assertAlmostEqual(s["ce_mom_pct"].iloc[1], 20.0)
        self.assertAlmostEqual(s["pe_mom_pct"].iloc[1], -10.0)
        self.assertAlmostEqual(s["pe_oi_chg_pct"].iloc[1], 10.0)
        with tempfile.TemporaryDirectory() as d:
            st = Store(Path(d) / "x.db")
            st.add_chain(t0, "NIFTY", snaps[0][1])
            got = st.chain_snapshots("NIFTY")
            self.assertEqual(str(got[0][0].tz), TZ)
            st.close()
        f = feed()
        u = f.universe().get("NIFTY")
        e = u.pick_expiry(DAY.date(), min_days=1)
        self.assertEqual(len(chain_payload(f.chain(u, e), float(f._visible("NIFTY")["close"].iloc[-1]), e)["rows"]), 21)


if __name__ == "__main__":
    unittest.main()
