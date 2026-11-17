"""
Upstox Market Data Feed V3 (WebSocket) for the NIFTY / SENSEX scanner.

Streams the last traded price of the subscribed instruments: the NIFTY and SENSEX indices, their
near-month index futures and India VIX (the keys come from the market data service once the
instrument master is loaded). Mode "ltpc". Messages are protobuf (MarketDataFeedV3_pb2.py).

Lifecycle
  * ONE connection per process: the application controller starts `run()` once as an asyncio task;
    calling start twice is refused by the controller.
  * Authorize (REST, v3) -> connect -> subscribe -> receive. On any disconnect it re-authorizes and
    reconnects with exponential backoff (5 s doubling to 60 s). A missing / expired token waits 60 s and
    retries, so a new token from `python -m src.broker.auth` is picked up without a restart.
  * The subscription follows `keys_fn()`: added keys are subscribed and removed keys unsubscribed every
    30 s, never by opening another connection.
  * `stop()` closes the socket cleanly; `run()` then returns.

The scanner keeps working without the feed (REST quotes every cycle); the feed only makes prices fresher.
Prices older than `max_age_seconds` are ignored by `latest()`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.broker.client import UpstoxAuthError, UpstoxClient

logger = logging.getLogger("scanner.websocket")
TZ = "Asia/Kolkata"


def parse_feed_message(payload: bytes) -> Dict[str, Tuple[float, pd.Timestamp]]:
    """Protobuf FeedResponse -> {instrument_key: (ltp, last trade time)}. Unknown / empty feeds are skipped."""
    from src.broker import MarketDataFeedV3_pb2 as pb

    response = pb.FeedResponse()
    response.ParseFromString(payload)
    out = {}
    for key, feed in response.feeds.items():
        kind = feed.WhichOneof("FeedUnion")
        ltpc = None
        if kind == "ltpc":
            ltpc = feed.ltpc
        elif kind == "fullFeed":
            sub = feed.fullFeed.WhichOneof("FullFeedUnion")
            ltpc = getattr(feed.fullFeed, sub).ltpc if sub else None
        elif kind == "firstLevelWithGreeks":
            ltpc = feed.firstLevelWithGreeks.ltpc
        if ltpc is None or ltpc.ltp <= 0 or ltpc.ltt <= 0:
            continue
        out[key] = (float(ltpc.ltp), pd.Timestamp(int(ltpc.ltt), unit="ms", tz="UTC").tz_convert(TZ))
    return out


class WebSocketFeed:
    def __init__(self, client: UpstoxClient, keys_fn: Callable[[], List[str]], max_age_seconds: float = 120.0,
                 mode: str = "ltpc"):
        self.client = client
        self.keys_fn = keys_fn
        self.max_age_seconds = max_age_seconds
        self.mode = mode
        self._prices: Dict[str, Tuple[float, pd.Timestamp]] = {}
        self._lock = threading.Lock()          # latest() is read from the scanner's worker thread
        self._stop = asyncio.Event()
        self._ws = None
        self.connected = False
        self.subscribed: set = set()
        self.messages = 0
        self.reconnects = 0
        self.last_error = ""

    # ---------- read side (any thread) ----------
    def latest(self, key: str) -> Optional[Tuple[float, pd.Timestamp]]:
        with self._lock:
            tick = self._prices.get(key)
        if tick is None:
            return None
        if (pd.Timestamp.now(tz=TZ) - tick[1]).total_seconds() > self.max_age_seconds:
            return None
        return tick

    def status(self) -> dict:
        return {"connected": self.connected, "subscribed": len(self.subscribed), "messages": self.messages,
                "reconnects": self.reconnects, "last_error": self.last_error}

    def _store(self, ticks: Dict[str, Tuple[float, pd.Timestamp]]):
        with self._lock:
            self._prices.update(ticks)

    # ---------- lifecycle (event loop) ----------
    async def stop(self):
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception as error:  # closing a dead socket must not block shutdown
                logger.debug("WebSocket close: %s", error)

    async def _sleep(self, seconds: float) -> bool:
        """Sleep unless stopped. Returns True when stop was requested."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
            return True
        except asyncio.TimeoutError:
            return False

    async def run(self):
        import websockets

        backoff = 5.0
        while not self._stop.is_set():
            keys = self.keys_fn()
            if not keys:  # instrument master not loaded yet
                if await self._sleep(5):
                    break
                continue
            try:
                url = await asyncio.to_thread(self.client.websocket_url)
            except UpstoxAuthError as error:
                self.last_error = str(error)
                logger.error("WebSocket: %s Retrying in 60 s.", error)
                if await self._sleep(60):
                    break
                continue
            except Exception as error:
                self.last_error = f"authorize failed: {type(error).__name__}"
                logger.warning("WebSocket authorize failed (%s); retrying in %.0f s.", error, backoff)
                if await self._sleep(backoff):
                    break
                backoff = min(backoff * 2, 60.0)
                continue
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5,
                                              max_size=2 ** 22) as ws:
                    self._ws = ws
                    self.connected = True
                    self.subscribed = set()
                    backoff = 5.0
                    logger.info("WebSocket connected.")
                    await self._sync_subscriptions(ws)
                    resync = asyncio.get_running_loop().time() + 30
                    while not self._stop.is_set():
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            message = None
                        if isinstance(message, (bytes, bytearray)):
                            try:
                                self._store(parse_feed_message(bytes(message)))
                                self.messages += 1
                            except Exception as error:
                                logger.debug("Undecodable feed message skipped: %s", error)
                        if asyncio.get_running_loop().time() >= resync:
                            await self._sync_subscriptions(ws)
                            resync = asyncio.get_running_loop().time() + 30
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"[:200]
                if not self._stop.is_set():
                    logger.warning("WebSocket disconnected (%s); reconnecting in %.0f s.", self.last_error, backoff)
            finally:
                self.connected = False
                self._ws = None
            if self._stop.is_set():
                break
            self.reconnects += 1
            if await self._sleep(backoff):
                break
            backoff = min(backoff * 2, 60.0)
        logger.info("WebSocket feed stopped.")

    async def _sync_subscriptions(self, ws):
        wanted = set(self.keys_fn())
        add, remove = sorted(wanted - self.subscribed), sorted(self.subscribed - wanted)
        if remove:
            await ws.send(json.dumps({"guid": uuid.uuid4().hex, "method": "unsub",
                                      "data": {"instrumentKeys": remove}}).encode("utf-8"))
        if add:
            await ws.send(json.dumps({"guid": uuid.uuid4().hex, "method": "sub",
                                      "data": {"mode": self.mode, "instrumentKeys": add}}).encode("utf-8"))
            logger.info("WebSocket subscribed: %s", ", ".join(add))
        self.subscribed = wanted
