"""
Upstox integration with fakes (no network, no credentials): REST client, token handling, market data
service, WebSocket feed.

    python -m unittest tests.test_broker
"""
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

from src.broker import client as client_mod
from src.broker.client import TokenSource, UpstoxAuthError, UpstoxClient
from src.broker.market_data import LiveProvider
from src.broker.websocket_feed import WebSocketFeed, parse_feed_message
from src.config import settings
from src.scanner import indicators as ind
from tests.test_strategy import fut_from, synthetic_minutes


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {"status": "success", "data": {}}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}", response=self)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        return self.responses.pop(0)


class StaticToken(TokenSource):
    def token(self):
        return "test-token-0123456789"


class TokenTests(unittest.TestCase):
    def test_env_token_wins_and_file_is_reloaded(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "upstox_token.json"
            src = TokenSource(path)
            with mock.patch.dict(os.environ, {"UPSTOX_ACCESS_TOKEN": ""}), \
                    mock.patch.object(settings, "ENV_FILE", Path(d) / ".env"):
                with self.assertRaises(UpstoxAuthError):
                    src.token()
                path.write_text(json.dumps({"access_token": "file-token"}), encoding="utf-8")
                self.assertEqual(src.token(), "file-token")
            with mock.patch.dict(os.environ, {"UPSTOX_ACCESS_TOKEN": "env-token"}):
                self.assertEqual(src.token(), "env-token")

    def test_missing_token_message_explains_the_fix(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"UPSTOX_ACCESS_TOKEN": ""}), \
                mock.patch.object(client_mod, "get_secret", lambda *a, **k: None):
            with self.assertRaises(UpstoxAuthError) as ctx:
                TokenSource(Path(d) / "none.json").token()
        self.assertIn("src.broker.auth", str(ctx.exception))

    def test_redact(self):
        self.assertEqual(settings.redact(None), "<not set>")
        self.assertNotIn("secret", settings.redact("secret-value"))


class ClientTests(unittest.TestCase):
    def test_401_raises_auth_error_and_429_retries(self):
        s = FakeSession([FakeResponse(429), FakeResponse(200, {"status": "success", "data": {"x": 1}})])
        c = UpstoxClient(requests_per_second=100, session=s, token_source=StaticToken())
        with mock.patch.object(client_mod._time, "sleep"):
            self.assertEqual(c.get("/a")["data"], {"x": 1})
        self.assertEqual(len(s.calls), 2)
        self.assertTrue(s.calls[0][2]["Authorization"].startswith("Bearer "))
        c = UpstoxClient(requests_per_second=100, session=FakeSession([FakeResponse(401)]), token_source=StaticToken())
        with self.assertRaises(UpstoxAuthError):
            c.get("/a")

    def test_rate_limit_spacing(self):
        s = FakeSession([FakeResponse() for _ in range(3)])
        c = UpstoxClient(requests_per_second=2, session=s, token_source=StaticToken())
        slept = []
        with mock.patch.object(client_mod._time, "sleep", side_effect=lambda x: slept.append(x)):
            for _ in range(3):
                c.get("/a")
        self.assertGreaterEqual(len(slept), 2)
        self.assertTrue(all(0 < x <= 0.5 for x in slept))

    def test_v3_endpoints_are_used_for_candles(self):
        s = FakeSession([FakeResponse(200, {"status": "success", "data": {"candles": []}}) for _ in range(3)])
        c = UpstoxClient(requests_per_second=100, session=s, token_source=StaticToken())
        c.minute_candles_today("NSE_INDEX|Nifty 50")
        c.daily_candles("BSE_INDEX|SENSEX", date(2026, 1, 1), date(2026, 2, 1))
        c.minute_candles("NSE_INDEX|Nifty 50", date(2026, 10, 1), date(2026, 10, 10))
        urls = [u for u, _, _ in s.calls]
        self.assertTrue(urls[0].startswith(client_mod.V3_BASE + "/historical-candle/intraday/NSE_INDEX%7CNifty%2050/minutes/1"))
        self.assertIn("/days/1/2026-02-01/2026-01-01", urls[1])
        self.assertIn("/minutes/1/2026-10-10/2026-10-01", urls[2])

    def test_quotes_isolate_a_bad_key(self):
        import requests
        c = UpstoxClient(requests_per_second=100, session=FakeSession([]), token_source=StaticToken())

        def fake_get(path, params=None, **kw):
            keys = params["instrument_key"].split(",")
            if "NSE_FO|BAD" in keys:
                raise requests.HTTPError("400", response=types.SimpleNamespace(status_code=400))
            return {"data": {k.replace("|", ":"): {"last_price": 1.0, "instrument_token": k} for k in keys}}
        c.get = fake_get
        out = c.quotes(["NSE_INDEX|Nifty 50", "NSE_FO|BAD", "BSE_INDEX|SENSEX"])
        self.assertEqual(set(k for k in out if "|" in k), {"NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"})
        self.assertIn("NSE_FO|BAD", c.bad_keys)

    def test_instrument_key_validation(self):
        self.assertTrue(client_mod.valid_instrument_key("NSE_FO|54321"))
        self.assertTrue(client_mod.valid_instrument_key("BSE_INDEX|SENSEX"))
        self.assertFalse(client_mod.valid_instrument_key("NSE_FO"))
        self.assertFalse(client_mod.valid_instrument_key(None))


class LiveProviderTests(unittest.TestCase):
    """NIFTY (NSE_FO) and SENSEX (BSE_FO) through fake Upstox calls: expiries / lots from the master,
    strike step from the chain, chain contracts keep their own keys, WebSocket price wins when newer."""

    def _provider(self, now, feed=None):
        m = synthetic_minutes(6, start="2026-10-09")
        m = m[m.index + pd.Timedelta(minutes=1) <= now]
        f = fut_from(m)
        master = [
            {"instrument_type": "INDEX", "exchange": "NSE_INDEX", "instrument_key": "NSE_INDEX|Nifty 50", "name": "Nifty 50"},
            {"instrument_type": "INDEX", "exchange": "BSE_INDEX", "instrument_key": "BSE_INDEX|SENSEX", "name": "SENSEX"},
            {"instrument_type": "INDEX", "exchange": "NSE_INDEX", "instrument_key": "NSE_INDEX|India VIX", "name": "India VIX"},
            {"instrument_type": "FUTIDX", "exchange": "NSE_FO", "instrument_key": "NSE_FO|900", "name": "NIFTY", "expiry": "2026-10-27"},
            {"instrument_type": "FUTIDX", "exchange": "BSE_FO", "instrument_key": "BSE_FO|901", "name": "SENSEX", "expiry": "2026-10-29"},
        ]
        for e in ("2026-10-20", "2026-10-27"):
            master.append({"instrument_type": "OPTIDX", "exchange": "NSE_FO", "instrument_key": f"NSE_FO|1{e[-2:]}",
                           "name": "NIFTY", "expiry": e, "lot_size": 65})
        for e in ("2026-10-22", "2026-10-29"):
            master.append({"instrument_type": "OPTIDX", "exchange": "BSE_FO", "instrument_key": f"BSE_FO|2{e[-2:]}",
                           "name": "SENSEX", "expiry": e, "lot_size": 20})
        calls = []

        def candles(frame):
            return pd.DataFrame([{"datetime": t, "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                                  "volume": r.volume, "oi": r.oi} for t, r in frame.iterrows()])

        class FakeClient:
            def quotes(self, keys):
                calls.append(("quotes", tuple(keys)))
                return {k: {"ltp": 24000.0, "open": 23990, "prev_close": 23950,
                            "trade_time": str(int((now - pd.Timedelta(seconds=5)).timestamp() * 1000))} for k in keys}

            def daily_candles(self, key, a, b):
                return ind.daily_from_minutes(m).reset_index().rename(columns={"index": "datetime"})

            def minute_candles(self, key, a, b):
                calls.append(("history", key))
                src = f if "_FO|" in key else m
                return candles(src[src.index.date < now.date()])

            def minute_candles_today(self, key):
                calls.append(("today", key))
                src = f if "_FO|" in key else m
                return candles(src[src.index.date == now.date()])

            def option_chain(self, key, expiry):
                calls.append(("chain", key, expiry))
                seg = "BSE_FO" if key.startswith("BSE") else "NSE_FO"
                step = 100 if seg == "BSE_FO" else 50
                return [{"strike_price": 24000 + k * step,
                         "call_options": {"instrument_key": f"{seg}|C{k}", "market_data": {"ltp": 100, "bid_price": 99.5,
                                          "ask_price": 100.5, "volume": 1e6, "oi": 1e7}, "option_greeks": {"delta": 0.5, "iv": 13}},
                         "put_options": {"instrument_key": f"{seg}|P{k}", "market_data": {"ltp": 90, "bid_price": 89.5,
                                         "ask_price": 90.5, "volume": 1e6, "oi": 1e7}, "option_greeks": {"delta": -0.5, "iv": 13}}}
                        for k in range(-3, 4)]

        cfg = settings.PBConfig.from_ini(settings.CONFIG_FILE)
        p = LiveProvider(["NIFTY", "SENSEX"], cfg, client=FakeClient(), live_feed=feed,
                         master_loader=lambda today: master, constituents_loader=lambda m_: {})
        return p, calls

    def test_snapshot_nifty_and_sensex(self):
        now = pd.Timestamp("2026-10-16 11:02:30", tz="Asia/Kolkata")
        p, calls = self._provider(now)
        snap = p.snapshot(now)
        self.assertEqual(set(snap.indices), {"NIFTY", "SENSEX"})
        n, s = snap.indices["NIFTY"], snap.indices["SENSEX"]
        self.assertEqual((n.expiry, n.lot_size, n.step, n.exchange), (date(2026, 10, 20), 65, 50.0, "NSE_FO"))
        self.assertEqual((s.expiry, s.lot_size, s.step, s.exchange), (date(2026, 10, 22), 20, 100.0, "BSE_FO"))
        self.assertTrue(n.chain["ce_key"].str.startswith("NSE_FO|").all())
        self.assertTrue(s.chain["pe_key"].str.startswith("BSE_FO|").all())
        self.assertFalse(n.spot_minutes.empty)
        self.assertFalse(n.fut_minutes.empty)            # futures supply volume / OI
        self.assertLess((now - n.ltp_time).total_seconds(), 10)
        subscribed = set(p.subscription_keys())
        self.assertEqual(subscribed, {"NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX", "NSE_FO|900", "BSE_FO|901",
                                      "NSE_INDEX|India VIX"})
        # second cycle within the cache window: no new history / chain calls
        before = len([c for c in calls if c[0] in ("history", "chain")])
        p.snapshot(now + pd.Timedelta(seconds=10))
        self.assertEqual(len([c for c in calls if c[0] in ("history", "chain")]), before)

    def test_expiry_rolls_and_skips_0dte(self):
        now = pd.Timestamp("2026-10-20 10:00", tz="Asia/Kolkata")   # NIFTY expiry day
        p, _ = self._provider(now)
        n = p.snapshot(now).indices["NIFTY"]
        self.assertEqual(n.nearest_expiry, date(2026, 10, 20))
        self.assertEqual(n.expiry, date(2026, 10, 27))
        self.assertTrue(n.is_expiry_day)

    def test_websocket_price_used_when_newer(self):
        now = pd.Timestamp("2026-10-16 11:02:30", tz="Asia/Kolkata")

        class Feed:
            def latest(self, key):
                return (24123.5, now - pd.Timedelta(seconds=1)) if key == "NSE_INDEX|Nifty 50" else None
        p, _ = self._provider(now, feed=Feed())
        n = p.snapshot(now).indices["NIFTY"]
        self.assertEqual(n.ltp, 24123.5)
        self.assertEqual(n.sources["spot"], "WebSocket")

    def test_auth_error_propagates(self):
        now = pd.Timestamp("2026-10-16 11:02:30", tz="Asia/Kolkata")
        p, _ = self._provider(now)

        def boom(keys):
            raise UpstoxAuthError("expired")
        p.client.quotes = boom
        with self.assertRaises(UpstoxAuthError):
            p.snapshot(now)


def _feed_bytes(prices):
    from src.broker import MarketDataFeedV3_pb2 as pb
    r = pb.FeedResponse()
    for key, (ltp, ltt, full) in prices.items():
        if full:
            r.feeds[key].fullFeed.indexFF.ltpc.ltp = ltp
            r.feeds[key].fullFeed.indexFF.ltpc.ltt = ltt
        else:
            r.feeds[key].ltpc.ltp = ltp
            r.feeds[key].ltpc.ltt = ltt
    return r.SerializeToString()


class FakeWS:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.closed = False

    async def send(self, data):
        self.sent.append(json.loads(data.decode("utf-8")))

    async def recv(self):
        if self.messages:
            return self.messages.pop(0)
        await asyncio.sleep(0.05)
        raise ConnectionError("server closed")

    async def close(self):
        self.closed = True


class WebSocketTests(unittest.TestCase):
    def test_parse_ltpc_and_full_feed(self):
        ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        out = parse_feed_message(_feed_bytes({"NSE_INDEX|Nifty 50": (24001.5, ms, False),
                                              "BSE_INDEX|SENSEX": (81000.25, ms, True)}))
        self.assertEqual(out["NSE_INDEX|Nifty 50"][0], 24001.5)
        self.assertEqual(out["BSE_INDEX|SENSEX"][0], 81000.25)
        self.assertEqual(str(out["BSE_INDEX|SENSEX"][1].tz), "Asia/Kolkata")

    def test_connect_subscribe_receive_reconnect_and_stop(self):
        ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        sockets = [FakeWS([_feed_bytes({"NSE_INDEX|Nifty 50": (24001.5, ms, False)})]), FakeWS([])]
        connects = []

        class Ctx:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, *a):
                return False

        def connect(url, **kw):
            connects.append(url)
            return Ctx(sockets[min(len(connects) - 1, len(sockets) - 1)])

        fake = types.ModuleType("websockets")
        fake.connect = connect

        class C:
            def websocket_url(self):
                return "wss://example.invalid/feed"

        async def scenario():
            feed = WebSocketFeed(C(), lambda: ["NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"])
            with mock.patch.dict(sys.modules, {"websockets": fake}), \
                    mock.patch.object(WebSocketFeed, "_sleep", new=lambda self, s: _short(self)):
                task = asyncio.create_task(feed.run())
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    if len(connects) >= 2:
                        break
                tick = feed.latest("NSE_INDEX|Nifty 50")
                await feed.stop()
                await asyncio.wait_for(task, timeout=5)
            return feed, tick

        async def _short(self):
            await asyncio.sleep(0.01)
            return self._stop.is_set()

        feed, tick = asyncio.run(scenario())
        self.assertEqual(tick[0], 24001.5)
        self.assertGreaterEqual(len(connects), 2)            # reconnected after the server closed
        sub = sockets[0].sent[0]
        self.assertEqual(sub["method"], "sub")
        self.assertEqual(sub["data"]["mode"], "ltpc")
        self.assertEqual(set(sub["data"]["instrumentKeys"]), {"NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX"})
        self.assertFalse(feed.connected)
        self.assertGreaterEqual(feed.reconnects, 1)

    def test_stale_ticks_are_ignored(self):
        feed = WebSocketFeed(None, lambda: [], max_age_seconds=60)
        feed._store({"X|Y": (1.0, pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(minutes=5))})
        self.assertIsNone(feed.latest("X|Y"))

    def test_auth_error_waits_instead_of_crashing(self):
        class C:
            def websocket_url(self):
                raise UpstoxAuthError("no token")

        async def scenario():
            feed = WebSocketFeed(C(), lambda: ["NSE_INDEX|Nifty 50"])
            with mock.patch.dict(sys.modules, {"websockets": types.ModuleType("websockets")}):
                task = asyncio.create_task(feed.run())
                await asyncio.sleep(0.1)
            self.assertFalse(task.done())                    # waiting, not crashed
            await feed.stop()
            await asyncio.wait_for(task, timeout=2)
            return feed
        feed = asyncio.run(scenario())
        self.assertIn("no token", feed.last_error)


if __name__ == "__main__":
    unittest.main()
