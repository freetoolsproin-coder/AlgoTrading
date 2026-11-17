"""
Movement trigger service (src/movement): window metrics, triggers, rules, cooldown, Flask API.

    python -m unittest tests.test_movement
"""
import json
import tempfile
import unittest
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.broker.market_data import TZ, IndexSnapshot, MarketSnapshot
from src.movement import detector
from src.movement.monitor import MovementMonitor, Webhook
from src.movement.rules import MovementRules, RulesStore, parse_window_map
from src.movement.store import EventStore
from src.movement.web import create_app
from src.notifications.service import format_alert

DAY = pd.Timestamp("2026-11-03 09:15", tz=TZ)


def minutes(closes, start=DAY, volume=1000.0, oi=None, up_bodies=True):
    """1-min candles from a list of closes (open = previous close)."""
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    closes = np.asarray(closes, dtype=float)
    opens = np.r_[closes[0], closes[:-1]]
    vol = np.full(len(closes), volume) if np.isscalar(volume) else np.asarray(volume, dtype=float)
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 1, "low": np.minimum(opens, closes) - 1,
                         "close": closes, "volume": vol, "oi": oi if oi is not None else np.full(len(closes), 1e6)},
                        index=idx)


def snapshot(spot, fut=None, now=None, name="NIFTY"):
    now = now or spot.index[-1] + pd.Timedelta(minutes=1)
    return IndexSnapshot(name=name, spot_minutes=spot, fut_minutes=fut if fut is not None else spot.assign(volume=0.0),
                         daily=pd.DataFrame(), ltp=float(spot["close"].iloc[-1]), ltp_time=now,
                         prev_close=float(spot["close"].iloc[0]))


class Provider:
    """Returns prepared snapshots; `now` is set by the test."""
    mock = True

    def __init__(self, snaps):
        self.snaps, self.t = snaps, None

    def now(self):
        return self.t

    def snapshot(self, now=None, want_chain=None):
        return MarketSnapshot(time=now, indices=self.snaps(now), vix=15.0, mock=True)


class RulesTests(unittest.TestCase):
    def test_window_map_parsing(self):
        self.assertEqual(parse_window_map("5:0.15, 10:0.25"), {5: 0.15, 10: 0.25})
        self.assertEqual(parse_window_map({"15": "0.4"}), {15: 0.4})
        with self.assertRaises(ValueError):
            parse_window_map("5")

    def test_validation_and_updates(self):
        r = MovementRules().updated({"windows": "5, 30", "move_pct": "5:0.2, 30:0.6", "oi_confirm": "true"})
        self.assertEqual(r.windows, [5, 30])
        self.assertTrue(r.oi_confirm)
        with self.assertRaises(ValueError):
            MovementRules().updated({"windows": "5, 20"})          # 20 has no threshold
        with self.assertRaises(ValueError):
            MovementRules().updated({"nonsense": 1})
        with self.assertRaises(ValueError):
            MovementRules().updated({"push_min_quality": "MAYBE"})

    def test_points_threshold_alone_is_enough(self):
        r = MovementRules().updated({"move_pct": "5:0", "windows": "5", "move_points": {"NIFTY": "5:30"}})
        self.assertEqual(r.threshold_points("NIFTY", 5), 30)

    def test_store_overrides_persist_and_reset(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "o.json"
            store = RulesStore(override_file=f)
            store.update({"min_rvol": 2.0})
            self.assertEqual(RulesStore(override_file=f).rules.min_rvol, 2.0)
            with self.assertRaises(ValueError):
                store.update({"strong_mult": 0.5})
            self.assertEqual(store.rules.min_rvol, 2.0)              # a bad change keeps the previous rules
            store.reset()
            self.assertFalse(f.exists())

    def test_ini_section_loads(self):
        r = RulesStore(override_file=Path(tempfile.gettempdir()) / "movement-none.json").defaults
        self.assertEqual(r.windows, [5, 10, 15])
        self.assertIn("NIFTY", r.move_points)


class DetectorTests(unittest.TestCase):
    rules = MovementRules().updated({"min_strength": 0, "min_rvol": 0, "min_consistency": 0})

    def test_window_change_and_reference(self):
        closes = [100.0] * 30 + [100 + i * 0.1 for i in range(1, 16)]   # +1.5 over the last 15 min
        m = detector.index_metrics(snapshot(minutes(closes)), DAY + pd.Timedelta(minutes=45), self.rules)
        w5, w15 = m["windows"][5], m["windows"][15]
        self.assertTrue(w5["ready"])
        self.assertAlmostEqual(w5["change"], 0.5, places=6)
        self.assertAlmostEqual(w15["change"], 1.5, places=6)
        self.assertEqual(w15["direction"], "UP")
        self.assertEqual(w15["consistency"], 1.0)

    def test_no_window_across_the_overnight_gap(self):
        prev = minutes([100.0] * 30, start=DAY - pd.Timedelta(days=1))
        today = minutes([105.0, 105.0, 105.0])
        m = detector.index_metrics(snapshot(pd.concat([prev, today])), DAY + pd.Timedelta(minutes=3), self.rules)
        self.assertFalse(m["windows"][5]["ready"])
        self.assertEqual(detector.evaluate(m, self.rules), [])

    def test_trigger_threshold_and_quality(self):
        closes = [24000.0] * 40 + [24000 + 12 * i for i in range(1, 6)]   # +60 pts in 5 min = 0.25 %
        m = detector.index_metrics(snapshot(minutes(closes)), DAY + pd.Timedelta(minutes=45), self.rules)
        found = {t["window"]: t for t in detector.evaluate(m, self.rules)}
        self.assertIn("5", found)
        self.assertEqual(found["5"]["direction"], "UP")
        self.assertEqual(found["5"]["quality"], "STRONG")               # no checks on, 0.25 >= 1.5 x 0.15
        self.assertNotIn("15", found)                                   # 0.25 % < 0.35 %

    def test_failed_check_is_unconfirmed(self):
        rules = self.rules.updated({"min_rvol": 1.5})
        closes = [24000.0] * 40 + [24000 + 12 * i for i in range(1, 6)]
        fut = minutes(closes, volume=1000.0)                            # flat volume: rvol 1.0
        m = detector.index_metrics(snapshot(minutes(closes), fut), DAY + pd.Timedelta(minutes=45), rules)
        t = [t for t in detector.evaluate(m, rules) if t["window"] == "5"][0]
        self.assertEqual(t["quality"], "UNCONFIRMED")
        self.assertEqual(t["checks"]["rvol"], "fail")

    def test_volume_and_oi(self):
        closes = [24000.0] * 40 + [24000 + 12 * i for i in range(1, 6)]
        vol = [1000.0] * 40 + [3000.0] * 5
        oi = np.r_[np.full(40, 1e6), np.linspace(1.01e6, 1.05e6, 5)]
        fut = minutes(closes, volume=vol, oi=oi)
        m = detector.index_metrics(snapshot(minutes(closes), fut), DAY + pd.Timedelta(minutes=45), self.rules)
        w5 = m["windows"][5]
        self.assertAlmostEqual(w5["rvol"], 3.0, places=2)
        self.assertGreater(w5["oi_change_pct"], 0)
        self.assertEqual(w5["oi_reading"], "Long build-up")

    def test_alignment_trigger(self):
        closes = [24000.0] * 30 + [24000 + 7 * i for i in range(1, 16)]   # steady climb, 105 pts in 15 min
        m = detector.index_metrics(snapshot(minutes(closes)), DAY + pd.Timedelta(minutes=45), self.rules)
        windows = {t["window"] for t in detector.evaluate(m, self.rules)}
        self.assertIn("ALL", windows)

    def test_stale_data_never_triggers(self):
        closes = [24000.0] * 40 + [24000 + 30 * i for i in range(1, 6)]
        snap = snapshot(minutes(closes))
        snap.ltp_time = DAY                                             # 45 min old
        m = detector.index_metrics(snap, DAY + pd.Timedelta(minutes=45), self.rules)
        self.assertTrue(m["stale"])
        self.assertEqual(detector.evaluate(m, self.rules), [])

    def test_vix(self):
        samples = [(DAY + pd.Timedelta(minutes=i), 14.0) for i in range(10)] + [(DAY + pd.Timedelta(minutes=10), 15.0)]
        _, found = detector.vix_triggers(samples, DAY + pd.Timedelta(minutes=10), MovementRules())
        self.assertTrue(any(t["window"] == "5" and t["direction"] == "UP" for t in found))


class MonitorAndWebTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        closes = [24000.0] * 40 + [24000 + 12 * i for i in range(1, 6)]
        self.provider = Provider(lambda now: {"NIFTY": snapshot(minutes(closes), now=now)})
        self.rules = RulesStore(override_file=d / "o.json")
        self.rules.update({"min_strength": 0, "min_rvol": 0, "min_consistency": 0, "alignment": False,
                           "vix_move_pct": "5:0"})
        self.posted = []
        self.monitor = MovementMonitor(self.provider, self.rules, EventStore(d / "m.db"),
                                       webhook=Webhook("http://example.invalid", sender=lambda u, e: self.posted.append(e)),
                                       mock=True)
        self.client = create_app(self.monitor, api_token="").test_client()

    def tearDown(self):
        self.dir.cleanup()

    def test_cooldown(self):
        t = DAY + pd.Timedelta(minutes=45)
        first = self.monitor.run_once(t)
        self.assertTrue(first)
        self.assertEqual(self.monitor.run_once(t + pd.Timedelta(minutes=1)), [])     # within 5 min
        self.assertEqual(len(self.monitor.store.recent()), len(first))

    def test_restart_keeps_cooldown(self):
        t = DAY + pd.Timedelta(minutes=45)
        self.monitor.run_once(t)
        again = MovementMonitor(self.provider, self.rules, self.monitor.store, mock=True)
        self.assertEqual(again.run_once(t + pd.Timedelta(minutes=1)), [])

    def test_api(self):
        self.monitor.run_once(DAY + pd.Timedelta(minutes=45))
        self.assertEqual(self.client.get("/health").json["status"], "ok")
        state = self.client.get("/api/state").json
        self.assertIn("NIFTY", state["indices"])
        self.assertTrue(self.client.get("/api/events?index=NIFTY").json)
        self.assertEqual(self.client.get("/").status_code, 200)
        r = self.client.put("/api/config", json={"min_rvol": "1.3"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["rules"]["min_rvol"], 1.3)
        self.assertEqual(self.client.put("/api/config", json={"windows": "7"}).status_code, 400)
        self.assertEqual(self.client.post("/api/config/reset").status_code, 200)

    def test_changes_only_from_this_computer_or_with_token(self):
        remote = {"REMOTE_ADDR": "192.168.1.20"}
        self.assertEqual(self.client.put("/api/config", json={"min_rvol": 1}, environ_base=remote).status_code, 403)
        guarded = create_app(self.monitor, api_token="s3cret").test_client()
        self.assertEqual(guarded.post("/api/evaluate").status_code, 401)
        self.assertEqual(guarded.post("/api/evaluate", headers={"X-API-Token": "s3cret"}).status_code, 200)

    def test_alert_text(self):
        e = self.monitor.run_once(DAY + pd.Timedelta(minutes=45))[0]
        title, body = format_alert(dict(e, kind="movement", stage="5 MIN MOVE"))
        self.assertIn("NIFTY", title)
        self.assertIn("Window: 5 minutes", body)
        self.assertIn("not an order", body)


def chain_for(spot, step=50.0, prem=100.0):
    """A small option chain around ATM with liquid, tight quotes and feed deltas."""
    rows = []
    atm = round(spot / step) * step
    for k in range(-3, 4):
        strike = atm + k * step
        row = {"strike": strike}
        for side, sign in (("ce", -1), ("pe", 1)):
            p = max(5.0, prem + sign * k * 20)
            delta = 0.5 - k * 0.08 if side == "ce" else -(0.5 + k * 0.08)
            row.update({f"{side}_ltp": p, f"{side}_bid": p - 0.5, f"{side}_ask": p + 0.5, f"{side}_iv": 14.0,
                        f"{side}_delta": delta, f"{side}_gamma": 0.001,
                        f"{side}_theta": -5.0, f"{side}_vega": 10.0, f"{side}_volume": 1e6, f"{side}_oi": 1e6,
                        f"{side}_key": f"KEY|{int(strike)}|{side}"})
        rows.append(row)
    return pd.DataFrame(rows)


class PaperTests(unittest.TestCase):
    def setUp(self):
        from src.config.settings import PBConfig
        from src.execution.costs import make_cost_fn
        from src.movement import paper
        self.paper = paper
        self.dir = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.dir.name) / "m.db")
        profiles = {"momentum": paper.Profile(name="momentum").validate(),
                    "scalp": paper.Profile(name="scalp", entry_windows=["1", "3"], stop_pct=8, target_pct=12,
                                           breakeven_pct=5, trail_pct=0, fade_exit=0.5, max_hold_minutes=8).validate()}
        self.kill = Path(self.dir.name) / "KILL"
        self.trader = paper.PaperTrader(profiles, PBConfig(), make_cost_fn(), store=self.store, mock=True,
                                        kill_switch_file=self.kill)
        self.now = DAY + pd.Timedelta(hours=1)
        self.isnap = snapshot(minutes([24000.0] * 60), now=self.now)
        self.isnap.chain, self.isnap.lot_size, self.isnap.step = chain_for(24000.0), 65, 50.0
        self.isnap.expiry = (DAY + pd.Timedelta(days=2)).date()
        self.isnap.exchange = "NSE_FO"

    def tearDown(self):
        self.dir.cleanup()

    def trig(self, direction="UP", window="5", quality="CONFIRMED", ref=23950.0):
        return {"index": "NIFTY", "direction": direction, "window": window, "quality": quality, "ref_price": ref,
                "price": 24000.0, "change": 24000.0 - ref, "change_pct": 0.2, "message": "test",
                "threshold_ratio": 1.2}

    def open(self, book="momentum", **kw):
        return self.trader.try_open(book, self.trig(**kw), self.isnap, self.now)

    def test_up_buys_call_down_buys_put(self):
        t = self.open()
        self.assertEqual(t.side, "CE")
        self.assertEqual(t.qty, 65)
        row = self.isnap.chain[self.isnap.chain["strike"] == t.strike].iloc[0]
        self.assertAlmostEqual(t.entry_price, row["ce_ask"] * 1.0025, places=2)    # ask + slippage
        self.assertTrue(0.35 <= row["ce_delta"] <= 0.70)
        self.assertGreater(t.charges, 0)
        self.trader.close_manual(t.id, self.now + pd.Timedelta(minutes=10))
        p = self.trader.try_open("momentum", self.trig("DOWN"), self.isnap, self.now + pd.Timedelta(minutes=30))
        self.assertEqual(p.side, "PE")

    def test_wants_filters(self):
        self.assertTrue(self.trader.wants("momentum", self.trig()))
        self.assertFalse(self.trader.wants("momentum", self.trig(quality="UNCONFIRMED")))
        self.assertFalse(self.trader.wants("scalp", self.trig(window="5")))
        self.assertFalse(self.trader.wants("momentum", dict(self.trig(), index="INDIAVIX")))

    def test_one_position_per_index_and_reentry_wait(self):
        t = self.open()
        self.assertIsNone(self.open())
        self.trader.close_manual(t.id, self.now)
        self.assertIsNone(self.trader.try_open("momentum", self.trig(), self.isnap, self.now + pd.Timedelta(minutes=1)))
        self.assertIn("waiting", self.trader.skipped[-1]["reason"])

    def test_target_stop_and_breakeven(self):
        t = self.open()
        k = t.instrument_key
        self.assertFalse(self.trader.mark({k: {"bid": t.entry_price * 1.2}}, {}, self.now + pd.Timedelta(minutes=1)))
        self.assertGreaterEqual(t.stop, t.entry_price)                        # breakeven / trail moved the stop
        closed = self.trader.mark({k: {"bid": t.entry_price * 1.5}}, {}, self.now + pd.Timedelta(minutes=2))
        self.assertEqual(closed[0].exit_reason, "target")
        self.assertGreater(closed[0].net(), 0)

    def test_stop_loss(self):
        t = self.open()
        closed = self.trader.mark({t.instrument_key: {"bid": t.entry_price * 0.7}}, {},
                                  self.now + pd.Timedelta(minutes=1))
        self.assertEqual(closed[0].exit_reason, "stop")
        self.assertLess(closed[0].net(), 0)

    def test_no_exit_on_missing_quote(self):
        t = self.open()
        self.assertFalse(self.trader.mark({}, {}, self.now + pd.Timedelta(minutes=1)))
        self.assertTrue(t.open)

    def test_scalp_fade_and_time_stop(self):
        t = self.open("scalp", window="1")
        k = t.instrument_key
        closed = self.trader.mark({k: {"bid": t.entry_price}}, {"NIFTY": 23970.0}, self.now + pd.Timedelta(minutes=1))
        self.assertEqual(closed[0].exit_reason, "momentum faded")            # gave back 30 of a 50-pt move
        later = self.now + pd.Timedelta(minutes=30)                         # after the re-entry + loss wait
        t2 = self.trader.try_open("scalp", self.trig(window="1"), self.isnap, later)
        k = t2.instrument_key
        closed = self.trader.mark({k: {"bid": t2.entry_price}}, {"NIFTY": 24000.0}, later + pd.Timedelta(minutes=9))
        self.assertEqual(closed[0].exit_reason, "time stop")

    def test_reverse_and_last_entry(self):
        t = self.open()
        closed = self.trader.mark({t.instrument_key: {"bid": t.entry_price}}, {}, self.now + pd.Timedelta(minutes=1),
                                  reverse={("momentum", "NIFTY"): "DOWN"})
        self.assertEqual(closed[0].exit_reason, "opposite trigger")
        late = pd.Timestamp("2026-11-03 15:16", tz=TZ)
        self.assertIsNone(self.trader.try_open("momentum", self.trig(), self.isnap, late))

    def test_daily_loss_limit_halts_book(self):
        self.trader.profiles["momentum"].max_daily_loss = 500
        t = self.open()
        closed = self.trader.mark({t.instrument_key: {"bid": t.entry_price * 0.85}}, {},
                                  self.now + pd.Timedelta(minutes=1))
        self.assertEqual(closed[0].exit_reason, "daily loss limit")
        self.assertTrue(self.trader.halted.get("momentum"))
        self.assertIsNone(self.trader.try_open("momentum", self.trig(), self.isnap, self.now + pd.Timedelta(hours=1)))

    def test_kill_switch(self):
        t = self.open()
        self.kill.write_text("")
        closed = self.trader.mark({t.instrument_key: {"bid": t.entry_price}}, {}, self.now + pd.Timedelta(minutes=1))
        self.assertEqual(closed[0].exit_reason, "kill switch")
        self.assertIsNone(self.trader.try_open("scalp", self.trig(window="1"), self.isnap,
                                               self.now + pd.Timedelta(minutes=2)))

    def test_open_trades_survive_restart(self):
        from src.config.settings import PBConfig
        from src.execution.costs import make_cost_fn
        today = pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(hours=10)
        t = self.trader.try_open("momentum", self.trig(), self.isnap, today)
        again = self.paper.PaperTrader(self.trader.profiles, PBConfig(), make_cost_fn(), store=self.store, mock=True)
        self.assertEqual([x.id for x in again.trades if x.open], [t.id])

    def test_ini_profiles_and_scalp_rules(self):
        profiles = self.paper.load_profiles()
        self.assertEqual(profiles["scalp"].entry_windows[:2], ["1", "3"])
        r = self.paper.scalp_rules(MovementRules(), profiles["scalp"])
        self.assertEqual(r.windows, [1, 3])
        with self.assertRaises(ValueError):
            self.paper.Profile(sides=["XX"]).validate()

    def test_alert_text(self):
        from src.movement.monitor import trade_alert
        title, body = format_alert(trade_alert(self.open(), "open"))
        self.assertIn("BUY", title)
        self.assertIn("PAPER TRADE", body)


if __name__ == "__main__":
    unittest.main()
