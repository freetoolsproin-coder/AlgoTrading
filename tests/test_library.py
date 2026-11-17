"""
Full strategy library (src/movement/library.py + engine changes), the ML signal classifier
(src/movement/ml_signal.py), the Flask strategy / ML endpoints and the per-strategy backtest.

    python -m unittest tests.test_library
"""
import tempfile
import time as clock
import unittest
from datetime import date, time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.broker.market_data import TZ, IndexSnapshot, MarketSnapshot
from src.movement import backtest as bt
from src.movement import library as L
from src.movement import ml_signal as ML
from src.movement import strategies as S

DAY = pd.Timestamp("2026-11-03 09:15", tz=TZ)


def bars_1m(closes, start, volume=1000.0, spread=1.0):
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread, "close": c,
                         "volume": volume, "oi": 1e6}, index=idx)


def history(path_today, prev_days=3, base=24000.0, seed=1, noise=2.0, prev_paths=None):
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(prev_days, 0, -1):
        closes = prev_paths[prev_days - d] if prev_paths else base + np.cumsum(rng.normal(0, noise, 375))
        frames.append(bars_1m(closes, DAY - pd.Timedelta(days=d)))
    frames.append(bars_1m(path_today, DAY))
    return pd.concat(frames)


def snap_of(m1, now, name="NIFTY", fut=None):
    daily = S.ind.daily_from_minutes(m1[m1.index.date < now.date()])
    return IndexSnapshot(name=name, spot_minutes=m1, fut_minutes=fut if fut is not None else m1, daily=daily,
                         ltp=float(m1["close"][m1.index < now].iloc[-1]), ltp_time=now, step=50.0)


def run_rule(key, m1, now, tf=5, peers=None, **params):
    """Evaluate one strategy on data cut at `now`; returns the Result."""
    eng = S.StrategyEngine([key], {key: dict(S.REGISTRY[key].params, **params, timeframe=tf)}, tf)
    snaps = {"NIFTY": snap_of(m1[m1.index < now], now)}
    for name, frame in (peers or {}).items():
        snaps[name] = snap_of(frame[frame.index < now], now, name)
    eng.evaluate(MarketSnapshot(time=now, indices=snaps, vix=14.0), now)
    r = eng.last[key]["NIFTY"]
    return S.Result(r["direction"], r["reason"], r["values"], r["bar"])


def scan(key, m1, minutes, **kw):
    """Directions a strategy gives at each `minutes` offset from today's open."""
    return [run_rule(key, m1, DAY + pd.Timedelta(minutes=m, seconds=5), **kw).direction for m in minutes]


class LibraryShapeTests(unittest.TestCase):
    def test_fifty_plus_strategies_with_complete_metadata(self):
        self.assertGreaterEqual(len(S.REGISTRY), 50)
        for key, spec in S.REGISTRY.items():
            with self.subTest(key=key):
                self.assertTrue(spec.title and spec.category and spec.ce and spec.pe and spec.data)
                self.assertNotIn("timeframe", spec.params)

    def test_every_requested_strategy_is_present(self):
        wanted = ["supertrend", "ma_crossover", "ema_9_21", "ema_20_50_200", "adx_trend", "donchian", "ichimoku",
                  "psar", "rsi_reversal", "macd_crossover", "stochastic", "roc", "cci", "williams_r",
                  "breakout_momentum", "sr_breakout", "pdh_pdl_breakout", "orb", "inside_bar", "engulfing",
                  "pin_bar", "vwap_breakout", "vwap_reversal", "bollinger_breakout", "bollinger_squeeze",
                  "atr_breakout", "keltner", "volatility_expansion", "india_vix", "rsi_mean_reversion",
                  "bollinger_reversion", "vwap_mean_reversion", "zscore_reversion", "ma_reversion",
                  "supertrend_rsi", "macd_adx", "vwap_ema", "mtf_confirmation", "fibonacci", "pivot_points",
                  "cpr_breakout", "volume_profile", "relative_strength", "pair_trading", "stat_arb", "pcr",
                  "oi_buildup", "gamma_iv", "ensemble", "ml_signal"]
        self.assertEqual(len(wanted), 50)
        self.assertEqual([k for k in wanted if k not in S.REGISTRY], [])

    def test_catalogue(self):
        rows = S.catalogue()
        self.assertEqual(len(rows), len(S.REGISTRY))
        orb = next(r for r in rows if r["key"] == "orb")
        self.assertTrue(orb["enabled"])
        self.assertAlmostEqual(orb["reward_risk"], orb["target_pct"] / orb["stop_pct"], places=2)


class NoLookAheadTests(unittest.TestCase):
    """Every strategy gives the same answer whether or not the data after `now` exists."""

    def test_future_bars_change_nothing(self):
        rng = np.random.default_rng(5)
        today = list(24000 + np.cumsum(np.r_[rng.normal(0.6, 3, 150), rng.normal(-0.8, 4, 225)]))
        prev = [24000 + np.cumsum(rng.normal(0, 3, 375)) for _ in range(5)]
        m1 = history(today, prev_days=5, prev_paths=prev)
        other = m1.copy()
        other[["open", "high", "low", "close"]] *= 3.3
        other[["open", "high", "low", "close"]] += np.cumsum(rng.normal(0, 4, len(other)))[:, None]
        future_shock = m1.copy()
        later = future_shock.index >= DAY + pd.Timedelta(minutes=200)
        future_shock.loc[later, ["open", "high", "low", "close"]] += 500.0
        keys = list(S.REGISTRY)
        for minute in (40, 95, 160, 199):
            now = DAY + pd.Timedelta(minutes=minute, seconds=5)
            results = []
            for frame in (m1[m1.index < now], future_shock):
                eng = S.StrategyEngine(keys, {}, 5)
                snaps = {"NIFTY": snap_of(frame, now), "SENSEX": snap_of(other, now, "SENSEX")}
                for v in (14.0, 14.2):
                    eng.remember_vix("NIFTY", v, now)
                eng.evaluate(MarketSnapshot(time=now, indices=snaps, vix=14.0), now)
                results.append({k: (d["NIFTY"]["direction"], d["NIFTY"]["reason"]) for k, d in eng.last.items()})
            for k in keys:
                with self.subTest(minute=minute, strategy=k):
                    self.assertEqual(results[0][k], results[1][k])
                    self.assertFalse(results[0][k][1].startswith("error"), results[0][k][1])


class RuleTests(unittest.TestCase):
    def test_crossovers_fire_once(self):
        path = list(24000 - np.arange(150) * 1.5) + list(23775 + np.arange(1, 150) * 3.0)
        m1 = history(path)
        for key in ("ema_9_21", "ma_crossover", "macd_crossover", "psar"):
            with self.subTest(key=key):
                dirs = scan(key, m1, range(150, 299, 5))
                self.assertIn("UP", dirs)
                self.assertLessEqual(dirs.count("UP"), 3)

    def test_donchian_and_keltner_breakouts(self):
        path = [24000.0 + (i % 6) for i in range(120)] + [24006 + 6 * i for i in range(1, 40)]
        m1 = history(path, noise=0.5)
        for key in ("donchian", "keltner", "bollinger_breakout"):
            with self.subTest(key=key):
                dirs = scan(key, m1, range(120, 160, 5))
                self.assertEqual(dirs.count("UP"), 1, dirs)

    def test_pdh_pdl_breakout_first_close_only(self):
        prev = [24000 + 20 * np.sin(i / 30) for i in range(375)]          # yesterday's high ~24020
        today = [24000.0] * 30 + [24000 + 2 * i for i in range(1, 60)]
        m1 = history(today, prev_days=1, prev_paths=[prev])
        dirs = scan("pdh_pdl_breakout", m1, range(10, 90, 5))
        self.assertEqual(dirs.count("UP"), 1, dirs)
        pdh = run_rule("pdh_pdl_breakout", m1, DAY + pd.Timedelta(minutes=60)).values["pdh"]
        self.assertAlmostEqual(pdh, max(prev) + 1, places=0)

    def test_inside_bar_breakout(self):
        bar = lambda o, h, l, c: [o, l, h, c, c]                             # 5 one-minute closes, one 5-min bar
        path = [24000.0 + (i % 4) for i in range(100)]
        path += bar(24000, 24060, 23960, 24010) + bar(24010, 24030, 23990, 24005) + bar(24005, 24080, 24000, 24075)
        m1 = history(path, noise=0.5)
        r = run_rule("inside_bar", m1, DAY + pd.Timedelta(minutes=115, seconds=5), max_mother_atr=100.0)
        self.assertEqual(r.direction, "UP", r.reason)

    def test_pivots_and_cpr_use_only_yesterday(self):
        prev = list(24000 + np.linspace(0, 30, 375))
        today = [24040.0] * 10 + [24040 + 3 * i for i in range(1, 50)]
        m1 = history(today, prev_days=1, prev_paths=[prev])
        piv = run_rule("pivot_points", m1, DAY + pd.Timedelta(minutes=60))
        y = S.ind.daily_from_minutes(m1[m1.index.date < DAY.date()]).iloc[-1]
        self.assertAlmostEqual(piv.values["P"], (y["high"] + y["low"] + y["close"]) / 3, places=1)
        self.assertIn("UP", scan("pivot_points", m1, range(10, 60, 5)))
        cpr = scan("cpr_breakout", m1, range(1, 60, 5), max_width_pct=1.0)
        self.assertEqual(cpr.count("UP"), 1, cpr)

    def test_zscore_and_mean_reversion(self):
        path = [24000.0 + (i % 3) for i in range(150)] + [24000 - 8 * i for i in range(1, 8)] + \
               [23944 + 4 * i for i in range(1, 30)]
        m1 = history(path, noise=0.5)
        self.assertIn("UP", scan("zscore_reversion", m1, range(150, 190, 5)))

    def test_pair_trading_needs_both_and_fires_on_the_rich_leg(self):
        rng = np.random.default_rng(2)
        common = np.cumsum(rng.normal(0, 2, 375 * 3 + 300))
        nifty_c = 24000 + common
        sensex_c = 80000 + common * 3.3
        nifty_c[-60:-30] += np.linspace(0, 120, 30)                  # NIFTY gets rich, then gives it back
        nifty_c[-30:] += np.linspace(120, 0, 30)
        prev = [nifty_c[i * 375:(i + 1) * 375] for i in range(3)]
        m1 = history(nifty_c[3 * 375:], prev_days=3, prev_paths=prev)
        sx = history(sensex_c[3 * 375:], prev_days=3, prev_paths=[sensex_c[i * 375:(i + 1) * 375] for i in range(3)])
        alone = run_rule("pair_trading", m1, DAY + pd.Timedelta(minutes=290))
        self.assertIn("no data", alone.reason)
        dirs = scan("pair_trading", m1, range(250, 300, 5), peers={"SENSEX": sx}, lookback=40)
        self.assertIn("DOWN", dirs)
        self.assertNotIn("UP", dirs)

    def test_india_vix_needs_samples_then_fires_on_a_break(self):
        m1 = history([24000.0] * 60 + [24000 + 5 * i for i in range(1, 20)], noise=0.5)
        now = DAY + pd.Timedelta(minutes=65, seconds=5)               # the first bar that breaks out
        c = S.StrategyEngine(["india_vix"], {}).context(snap_of(m1[m1.index < now], now), now)
        self.assertIn("no data", L.india_vix(c, S.REGISTRY["india_vix"].params).reason)
        c.memory.vix.extend([(now - pd.Timedelta(minutes=31), 15.0), (now, 14.2)])
        r = L.india_vix(c, S.REGISTRY["india_vix"].params)
        self.assertEqual(r.direction, "UP", r.reason)

    def test_futures_volume_strategies_say_no_data_without_it(self):
        m1 = history([24000.0] * 120)
        now = DAY + pd.Timedelta(minutes=120)
        snap = snap_of(m1, now, fut=m1.assign(volume=0.0))
        eng = S.StrategyEngine(["vwap_breakout", "vwap_mean_reversion", "vwap_ema"], {})
        eng.evaluate(MarketSnapshot(time=now, indices={"NIFTY": snap}), now)
        for key in eng.enabled:
            self.assertTrue(eng.last[key]["NIFTY"]["reason"].startswith("no data"), key)

    def test_ensemble_counts_votes(self):
        path = list(24000 - np.arange(150) * 1.0) + list(23850 + np.arange(1, 120) * 3.0)
        dirs = scan("ensemble", history(path), range(150, 269, 5))
        self.assertIn("UP", dirs)


class EngineTests(unittest.TestCase):
    def test_per_strategy_timeframe(self):
        with tempfile.TemporaryDirectory() as d:
            ini = Path(d) / "s.ini"
            ini.write_text("[STRATEGIES]\nenabled = orb, supertrend\ntimeframe = 5\n"
                           "[STRATEGY_SUPERTREND]\ntimeframe = 15\nmult = 2.5\n", encoding="utf-8")
            eng, profiles = S.load_config(ini)
            self.assertEqual(eng.tfs, {"orb": 5, "supertrend": 15})
            self.assertEqual(eng.params["supertrend"]["mult"], 2.5)
            ini.write_text("[STRATEGIES]\nenabled = orb\n[STRATEGY_ORB]\ntimeframe = 7\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                S.load_config(ini)

    def test_bar_strategies_evaluated_once_per_bar(self):
        calls = []
        spec = S.REGISTRY["donchian"]
        counting = S.replace(spec, fn=lambda c, p: calls.append(c.now) or spec.fn(c, p))
        m1 = history([24000.0 + (i % 5) for i in range(60)])
        with mock.patch.dict(S.REGISTRY, {"donchian": counting}):
            eng = S.StrategyEngine(["donchian"], {})
            for sec in range(0, 300, 10):                       # 09:45:00 .. 09:49:50, one 5-min bar
                now = DAY + pd.Timedelta(minutes=30, seconds=sec)
                eng.evaluate(MarketSnapshot(time=now, indices={"NIFTY": snap_of(m1[m1.index < now], now)}), now)
        self.assertLessEqual(len(calls), 2)                     # the bar set changes at most once (the 09:45 bar)

    def test_signal_carries_its_timeframe(self):
        path = [24000.0] * 15 + [24000 + i * 4 for i in range(1, 30)]
        m1 = history(path)
        now = DAY + pd.Timedelta(minutes=20)
        eng = S.StrategyEngine(["orb"], {})
        sigs = eng.evaluate(MarketSnapshot(time=now, indices={"NIFTY": snap_of(m1[m1.index < now], now)}), now)
        self.assertEqual(sigs[0]["timeframe"], 5)


def planted_minutes(days=60, seed=3):
    """1-min data where a strong 5-min up bar tends to be followed by more up (a learnable pattern)."""
    rng = np.random.default_rng(seed)
    frames, start = [], pd.Timestamp("2026-01-05 09:15", tz=TZ)
    day = start
    for _ in range(days):
        n = 375
        steps = rng.normal(0, 3, n)
        for k in range(0, n - 60, 30):                 # every 30 min: an impulse then follow-through
            if rng.random() < 0.5:
                sign = 1 if rng.random() < 0.5 else -1
                steps[k:k + 5] += sign * 6
                steps[k + 5:k + 30] += sign * 2.5
        closes = 24000 + np.cumsum(steps)
        frames.append(bars_1m(closes, day))
        day = day + pd.Timedelta(days=1)
        while day.weekday() >= 5:
            day = day + pd.Timedelta(days=1)
    return pd.concat(frames)


class MLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.minutes = planted_minutes()
        cls.path = Path(cls.dir.name) / "ml_signal_NIFTY_5m.json"
        cls.model = ML.train(cls.minutes, "NIFTY", tf=5, horizon=4, min_move_atr=1.0, holdout=0.3, path=cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_split_is_chronological_and_learns_the_pattern(self):
        m = self.model
        self.assertLess(date.fromisoformat(m["train_end"]), date.fromisoformat(m["test_start"]))
        self.assertEqual(m["train_days"] + m["test_days"], 60)
        self.assertGreater(m["report"]["test"]["up"]["auc"], 0.6)
        self.assertTrue(self.path.exists())
        self.assertEqual(ML.load_model(self.path)["train_end"], m["train_end"])

    def test_labels_never_cross_sessions(self):
        b = S.add_indicators(S.ind.resample(self.minutes, 5))
        lab = ML.labels(b, 6, 1.0)
        last_bars = lab.groupby(lab.index.date).tail(6)
        self.assertTrue(last_bars["fwd_atr"].isna().all())

    def test_logistic_and_auc(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2000, 2))
        y = (x[:, 0] + 0.3 * rng.normal(size=2000) > 0).astype(float)
        m = ML.fit_logistic(x, y, l2=0.1)
        self.assertGreater(m.w[0], 1.0)
        self.assertGreater(ML.auc(y, m.proba(x)), 0.9)
        self.assertAlmostEqual(ML.auc([0, 1], [0.1, 0.9]), 1.0)

    def test_strategy_refuses_training_days_and_trades_after(self):
        params = {"threshold": 0.0, "model": str(self.path)}
        in_train = pd.Timestamp(f"{self.model['train_end']} 13:00", tz=TZ)
        m1 = self.minutes[self.minutes.index < in_train]
        c = S.StrategyEngine(["ml_signal"], {}).context(snap_of(m1, in_train), in_train)
        self.assertIn("training period", ML.evaluate(c, params).reason)
        later = pd.Timestamp(f"{self.model['test_end']} 13:00", tz=TZ)
        m1 = self.minutes[self.minutes.index < later]
        c = S.StrategyEngine(["ml_signal"], {}).context(snap_of(m1, later), later)
        r = ML.evaluate(c, params)
        self.assertIn("p_up", r.values)
        missing = ML.evaluate(c, {"threshold": 0.0, "model": str(Path(self.dir.name) / "none.json")})
        self.assertTrue(missing.reason.startswith("no data"))

    def test_too_little_data(self):
        with self.assertRaises(ValueError):
            ML.train(self.minutes.iloc[:375 * 5], "NIFTY", save=False)


class FlaskTests(unittest.TestCase):
    def setUp(self):
        from src.movement.monitor import MovementMonitor
        from src.movement.rules import RulesStore
        from src.movement.store import EventStore
        from src.movement.web import create_app
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)

        class Provider:
            mock = True

            def now(self):
                return DAY

            def snapshot(self, now=None, want_chain=None):
                return MarketSnapshot(time=now, indices={}, vix=15.0, mock=True)

        self.engine = S.StrategyEngine(["orb", "ml_signal"], {})
        self.monitor = MovementMonitor(Provider(), RulesStore(override_file=d / "o.json"), EventStore(d / "m.db"),
                                       mock=True, strategies=self.engine)
        self.client = create_app(self.monitor, api_token="").test_client()

    def tearDown(self):
        self.dir.cleanup()

    def test_pages_and_catalogue(self):
        for page in ("/strategies", "/ml"):
            r = self.client.get(page)
            self.assertEqual(r.status_code, 200)
            r.close()
        data = self.client.get("/api/strategies").get_json()
        self.assertEqual(data["count"], len(S.REGISTRY))
        enabled = {r["key"] for r in data["strategies"] if r["enabled"]}
        self.assertEqual(enabled, {"orb", "ml_signal"})

    def test_predict_reports_engine_state(self):
        self.engine.last = {"ml_signal": {"NIFTY": {"direction": None, "reason": "P(up) 0.3", "values": {"p_up": 0.3}}}}
        data = self.client.get("/api/ml/predict").get_json()
        self.assertTrue(data["enabled"])
        self.assertEqual(data["indices"]["NIFTY"]["values"]["p_up"], 0.3)

    def test_train_is_local_only_and_validated(self):
        r = self.client.post("/api/ml/train", json={}, environ_overrides={"REMOTE_ADDR": "192.168.1.9"})
        self.assertEqual(r.status_code, 403)
        r = self.client.post("/api/ml/train", json={"index": "BANKNIFTY"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/ml/train", json={"horizon": 500})
        self.assertEqual(r.status_code, 400)

    def test_train_job_runs_in_background(self):
        minutes = planted_minutes(days=30)
        with mock.patch.object(ML, "load_history", return_value=minutes), \
                mock.patch.object(ML, "model_path", side_effect=lambda i, tf: Path(self.dir.name) / f"m_{i}_{tf}.json"):
            r = self.client.post("/api/ml/train", json={"index": "NIFTY", "horizon": 4})
            self.assertEqual(r.status_code, 202)
            job = r.get_json()
            for _ in range(100):
                status = self.client.get(f"/api/ml/jobs/{job['id']}").get_json()
                if status["status"] != "running":
                    break
                clock.sleep(0.1)
        self.assertEqual(status["status"], "done", status)
        self.assertIn("test", status["report"])
        self.assertTrue((Path(self.dir.name) / "m_NIFTY_5.json").exists())
        self.assertEqual(self.client.get("/api/ml/jobs/nope").status_code, 404)


class BacktestTests(unittest.TestCase):
    def test_strategy_settings(self):
        eng, profiles = bt.strategy_settings(keys=["donchian", "orb"])
        self.assertEqual(eng.enabled, ["donchian", "orb"])
        self.assertEqual(set(profiles), {"donchian", "orb"})
        eng, _ = bt.strategy_settings(keys=["all"])
        self.assertEqual(len(eng.enabled), len(S.REGISTRY))
        with self.assertRaises(ValueError):
            bt.strategy_settings(keys=["no_such_strategy"])

    def test_replay_with_strategy_books_and_report(self):
        rng = np.random.default_rng(4)
        frames, day = [], pd.Timestamp("2026-10-20 09:15", tz=TZ)
        days = []
        while len(days) < 8:
            if day.weekday() < 5:
                steps = rng.normal(0, 4, 375)
                steps[100:160] += 3.0 if len(days) % 2 else -3.0     # a trend leg every day
                frames.append(bars_1m(24000 + np.cumsum(steps), day, volume=0.0))
                days.append(day.date())
            day += pd.Timedelta(days=1)
        spot = pd.concat(frames)
        vix = spot.assign(open=14.0, high=14.0, low=14.0, close=14.0)
        rules, profiles, cfg, costs = bt.backtest_settings(overrides={"PAPER_MOMENTUM": {"enabled": "false"},
                                                                      "PAPER_SCALP": {"enabled": "false"}})
        eng, sprof = bt.strategy_settings(keys=["donchian", "supertrend", "pair_trading"])
        result = bt.simulate("NIFTY", spot, vix, rules, {**profiles, **sprof}, cfg, costs, days[1:], engine=eng)
        books = {t["book"] for t in result["trades"]}
        self.assertTrue(books)
        self.assertTrue(books <= {"donchian", "supertrend"})
        for t in result["trades"]:
            self.assertEqual(pd.Timestamp(t["entry_time"]).date(), pd.Timestamp(t["exit_time"]).date())
        self.assertIn("no data: needs the other index", " ".join(result["strategy_status"]["pair_trading"]))
        text = bt.strategy_report("NIFTY", result, days[1], days[-1], eng.enabled, {})
        self.assertIn("SUMMARY", text)
        self.assertIn("NO TRADES - why", text)
        self.assertIn("consecutive losses", text)
        self.assertIn("MODELLED", text)

    def test_stats_extras(self):
        start = pd.Timestamp("2026-10-30 10:00", tz=TZ)
        trades = [{"net": n, "charges": 10.0, "entry_time": (start + pd.Timedelta(days=i)).isoformat(),
                   "exit_time": (start + pd.Timedelta(days=i, minutes=5)).isoformat()}
                  for i, n in enumerate([100, 100, -50, -50, -50, 200])]
        s = bt.stats(trades)
        self.assertEqual(s["max_consecutive_wins"], 2)
        self.assertEqual(s["max_consecutive_losses"], 3)
        self.assertAlmostEqual(s["gross_profit"], 400.0)
        self.assertAlmostEqual(s["reward_risk"], (400 / 3) / 50)
        days = [(start + pd.Timedelta(days=i)).date() for i in range(25)]
        r = bt.daily_ratios(trades, days)
        self.assertIsNotNone(r["sharpe"])
        self.assertIsNone(bt.daily_ratios(trades, days[:5])["sharpe"])


if __name__ == "__main__":
    unittest.main()
