"""
Background monitor: polls the market data provider, computes the 5 / 10 / 15-minute movement metrics,
fires triggers (with a cooldown per index / window / direction) and hands them to

    the event store      data/movement.db (always)
    the live stream      every open web page (Server-Sent Events, always)
    the notifier         Telegram / Windows popup, when the quality reaches push_min_quality
    a webhook            MOVEMENT_WEBHOOK_URL in .env (optional): the event as JSON, POSTed

One worker thread; the Flask web server only reads `state` and the store, so a slow browser never slows it.
"""
from __future__ import annotations

import collections
import json
import logging
import queue
import threading
from datetime import time
from typing import Dict, List, Optional

import pandas as pd

from src.broker.client import UpstoxAuthError
from src.movement import detector
from src.movement import paper as paper_mod
from src.movement.rules import RulesStore
from src.movement.store import EventStore
from src.scanner.store import json_default
from src.utils import market_calendar

logger = logging.getLogger("scanner.movement.monitor")
SESSION_START, SESSION_END = time(9, 15), time(15, 30)


class Broadcaster:
    """Fan-out to Server-Sent-Event clients: one bounded queue per connected page."""

    def __init__(self, max_clients: int = 50):
        self._clients: List[queue.Queue] = []
        self._lock = threading.Lock()
        self.max_clients = max_clients

    def subscribe(self) -> Optional[queue.Queue]:
        with self._lock:
            if len(self._clients) >= self.max_clients:
                return None
            q: queue.Queue = queue.Queue(maxsize=100)
            self._clients.append(q)
            return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, kind: str, data: dict):
        message = f"event: {kind}\ndata: {json.dumps(data, default=json_default)}\n\n"
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(message)
            except queue.Full:   # a stalled page: drop its oldest message rather than block the monitor
                try:
                    q.get_nowait()
                    q.put_nowait(message)
                except (queue.Empty, queue.Full):
                    pass

    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._clients)


class Webhook:
    """Best-effort POST of each pushed event to one URL, from its own thread."""

    def __init__(self, url: Optional[str], sender=None):
        self.url = (url or "").strip()
        self._send = sender or self._post
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._warned = False
        if self.url:
            threading.Thread(target=self._run, name="movement-webhook", daemon=True).start()

    def send(self, event: dict):
        if self.url:
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                logger.warning("Webhook queue full: an event was not posted (it is still stored).")

    def _run(self):
        while True:
            event = self._queue.get()
            try:
                self._send(self.url, event)
            except Exception as error:
                if not self._warned:
                    self._warned = True
                    logger.warning("Webhook POST failed (%s); events are still stored and shown.", type(error).__name__)

    @staticmethod
    def _post(url, event):
        import requests
        requests.post(url, data=json.dumps(event, default=json_default),
                      headers={"Content-Type": "application/json"}, timeout=8).raise_for_status()


def to_alert(e: dict) -> dict:
    """A trigger as a Notifier alert (src/notifications/service.py formats kind 'movement')."""
    return dict(e, kind="movement", stage=f"{e['window']} MIN MOVE" if e["window"] != "ALL" else "ALL WINDOWS",
                direction=e["direction"], score=e.get("threshold_ratio"))


def trade_alert(t, action: str) -> dict:
    """A paper trade entry / exit as a Notifier alert (kind 'paper_trade')."""
    d = t.to_dict()
    verb = "BUY" if action == "open" else "SELL"
    price = d["entry_price"] if action == "open" else d["exit_price"]
    msg = f"PAPER {d['book'].upper()} {verb} {d['qty']} {d['symbol']} @ {price:.2f}"
    if action == "close":
        msg += f" - {d['exit_reason']} - net Rs {d['net']:,.0f}"
    return dict(d, kind="paper_trade", action=action, stage=f"PAPER {d['book'].upper()} {verb}",
                direction=d["side"], time=d["exit_time"] if action == "close" else d["entry_time"],
                price=d["underlying_entry"], message=msg)


class MovementMonitor:
    def __init__(self, provider, rules: RulesStore, store: EventStore, notifier=None, broadcaster=None,
                 webhook: Optional[Webhook] = None, mock: bool = False, push_enabled: bool = True,
                 trader: Optional[paper_mod.PaperTrader] = None, trade_notifier=None, strategies=None):
        self.provider = provider
        self.strategies = strategies          # src.movement.strategies.StrategyEngine (optional)
        self.trader = trader
        self.trade_notifier = trade_notifier or notifier
        self._scalp_rules = (None, None)
        self.push_enabled = push_enabled   # False: mock run without --notify (events are stored and shown only)
        self.rules_store = rules
        self.store = store
        self.notifier = notifier
        self.broadcaster = broadcaster or Broadcaster()
        self.webhook = webhook or Webhook(None)
        self.mock = mock
        self.state: dict = {"status": "starting", "indices": {}, "vix": {}, "generated_at": None}
        self.vix_samples: collections.deque = collections.deque(maxlen=2000)
        self.last_fired: Dict[tuple, pd.Timestamp] = {}
        self.cycles = 0
        self.last_error = ""
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._seed_cooldowns()

    @property
    def rules(self):
        return self.rules_store.rules

    def _seed_cooldowns(self):
        """A restart must not re-send today's triggers: start the cooldowns from the stored events."""
        try:
            for e in reversed(self.store.recent(500, mock=self.mock)):
                ts = pd.Timestamp(e["time"])
                self.last_fired[(e["index"], e["window"], e["direction"])] = ts
        except Exception:
            logger.exception("Could not read earlier movement events")

    # ------------------------------------------------------------------ thread
    def start(self):
        if self._thread is not None:
            raise RuntimeError("monitor already started")
        self._thread = threading.Thread(target=self._run, name="movement-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15.0):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def poke(self):
        """Evaluate now instead of waiting for the next poll (POST /api/evaluate, rule changes)."""
        self._wake.set()

    def _run(self):
        while not self._stop.is_set():
            wait = min(self.rules.poll_seconds, 2) if self.mock else self.rules.poll_seconds   # mock clock is fast
            try:
                if not self.mock and not self._in_session(self.provider.now()) and self.cycles > 0:
                    self._set_status("Market closed - no triggers outside 09:15-15:30 on trading days")
                    wait = 60
                else:
                    self.run_once()
            except UpstoxAuthError as error:
                self.last_error = str(error)
                self._set_status("Upstox token missing or expired - run: python -m src.broker.auth")
                logger.error("%s Waiting 60 s for a new token.", error)
                wait = 60
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"[:300]
                self._set_status("Cycle failed - see logs/scanner.log; retrying")
                logger.exception("Movement cycle failed; the monitor continues.")
                wait = max(wait, 15)
            self._wake.wait(wait)
            self._wake.clear()

    @staticmethod
    def _in_session(now: pd.Timestamp) -> bool:
        return market_calendar.is_trading_day(now.date()) and SESSION_START <= now.time() <= SESSION_END

    def _set_status(self, status: str):
        with self._lock:
            self.state = dict(self.state, status=status, last_error=self.last_error)
        self.broadcaster.publish("state", self.state)

    # ------------------------------------------------------------------ one cycle
    def run_once(self, now: Optional[pd.Timestamp] = None) -> List[dict]:
        rules = self.rules
        now = now or self.provider.now()
        chain_needed = self.strategies is not None and self.strategies.needs_chain
        snap = self.provider.snapshot(now, want_chain=None if chain_needed else set())
        in_session = self.mock or self._in_session(now)
        if pd.notna(snap.vix):
            self.vix_samples.append((now, float(snap.vix)))
        indices, fired = {}, []
        for name, index_snap in snap.indices.items():
            metrics = detector.index_metrics(index_snap, now, rules)
            indices[name] = metrics
            if in_session:
                fired += self._fire(detector.evaluate(metrics, rules), now, rules)
        vix_windows, vix_found = detector.vix_triggers(list(self.vix_samples), now, rules, mock=self.mock)
        if in_session:
            fired += self._fire(vix_found, now, rules)
        scalp_view, strategy_triggers = {}, []
        if self.strategies is not None:
            strategy_triggers = self._strategies(snap, indices, now, in_session)
        if self.trader is not None:
            scalp_view = self._paper(snap, indices, fired, now, rules, in_session, strategy_triggers)
        self.cycles += 1
        self.last_error = ""
        stale = [n for n, m in indices.items() if m["stale"]]
        status = ("Running" if in_session else "Market closed - showing the last data, no triggers")
        if stale and in_session:
            status += f" - data stale for {', '.join(stale)} (no triggers for it)"
        state = {"status": status, "mode": "MOCK DATA" if self.mock else "LIVE DATA", "generated_at": now.isoformat(),
                 "indices": indices, "vix": {"value": detector._f(snap.vix), "windows": vix_windows},
                 "errors": list(snap.errors or []), "rules": rules.to_dict(), "cycles": self.cycles,
                 "last_error": self.last_error, "fired_this_cycle": len(fired), "scalp": scalp_view,
                 "strategies": self.strategies.table() if self.strategies is not None else None,
                 "paper": self.trader.summary(now) if self.trader is not None else None}
        with self._lock:
            self.state = state
        self.broadcaster.publish("state", state)
        return fired

    def _fire(self, found: List[dict], now: pd.Timestamp, rules) -> List[dict]:
        out = []
        for e in found:
            key = (e["index"], e["window"], e["direction"])
            window_minutes = max(rules.windows) if e["window"] == "ALL" else int(e["window"])
            last = self.last_fired.get(key)
            if last is not None and now - last < pd.Timedelta(minutes=rules.cooldown(window_minutes)):
                continue
            self.last_fired[key] = now
            pushed = self.push_enabled and rules.should_push(e["quality"])
            e = dict(e, pushed=pushed, mock=self.mock)
            e["id"] = self.store.add(e)
            logger.warning("MOVEMENT %s", e["message"])
            self.broadcaster.publish("movement", e)
            if self.notifier is not None:
                self.notifier.emit(to_alert(e), push=pushed)
            if pushed:
                self.webhook.send(e)
            out.append(e)
        return out

    # ------------------------------------------------------------------ paper trading
    def _scalp(self, rules):
        """Scalp trigger rules, rebuilt only when the base rules change."""
        if self._scalp_rules[0] is not rules:
            self._scalp_rules = (rules, paper_mod.scalp_rules(rules, self.trader.profiles["scalp"]))
        return self._scalp_rules[1]

    def _strategies(self, snap, indices, now, in_session) -> List[tuple]:
        """Evaluate the strategy library; returns [(book, trigger)] for the paper trader."""
        from src.movement.strategies import to_trigger
        stale = {n for n, m in indices.items() if m["stale"]}
        signals = self.strategies.evaluate(snap, now, stale)
        out = []
        if not in_session:
            return out
        for sig in signals:
            sig = dict(sig, time=now.isoformat(), mock=self.mock)
            logger.warning("SIGNAL %s %s %s - %s", sig["title"], sig["index"], sig["direction"], sig["reason"])
            self.broadcaster.publish("signal", sig)
            out.append((sig["strategy"], to_trigger(sig, indices[sig["index"]].get("price"), self.mock)))
        return out

    def _paper(self, snap, indices, fired, now, rules, in_session, extra=()) -> dict:
        trader = self.trader
        trader.new_day(now.date())
        wanted: Dict[tuple, dict] = {}            # (book, index) -> strongest trigger this cycle
        reverse: Dict[tuple, str] = {}

        def consider(book, trig):
            if not trader.wants(book, trig):
                return
            key = (book, trig["index"])
            reverse[key] = trig["direction"]
            best = wanted.get(key)
            if best is None or (trig.get("threshold_ratio") or 0) > (best.get("threshold_ratio") or 0):
                wanted[key] = trig

        for e in fired:
            consider("momentum", e)
        for book, trig in extra:
            consider(book, trig)
        scalp_view = {}
        sp = trader.profiles.get("scalp")
        if sp is not None and sp.enabled:
            srules = self._scalp(rules)
            for name, isnap in snap.indices.items():
                m = detector.index_metrics(isnap, now, srules)
                scalp_view[name] = m["windows"]
                if in_session and not indices[name]["stale"]:
                    for trig in detector.evaluate(m, srules):
                        consider("scalp", dict(trig, mock=self.mock))
        if trader.open_trades():
            quotes = self._option_quotes(trader.open_trades(), now)
            underlying = {n: m.get("price") for n, m in indices.items()}
            for t in trader.mark(quotes, underlying, now, reverse):
                self._trade_event(t, "close")
        if in_session and wanted:
            needed = {idx for _, idx in wanted}
            chain_snap = self.provider.snapshot(now, want_chain=needed)
            fresh = self._fresh_quote if not self.mock and hasattr(self.provider, "client") else None
            for (book, idx), trig in wanted.items():
                isnap = chain_snap.indices.get(idx)
                if isnap is None or indices[idx]["stale"]:
                    continue
                t = trader.try_open(book, trig, isnap, now, fresh)
                if t is not None:
                    self._trade_event(t, "open")
        return scalp_view

    def _fresh_quote(self, key: str) -> Optional[dict]:
        try:
            return self.provider.client.quotes([key]).get(key)
        except UpstoxAuthError:
            raise
        except Exception as error:
            logger.warning("Option quote failed for %s: %s", key, error)
            return None

    def _option_quotes(self, trades, now) -> Dict[str, dict]:
        """{instrument_key: {bid, ask, ltp}} for open trades: a live quote call, or the mock option chain."""
        keys = [t.instrument_key for t in trades if t.instrument_key]
        if not self.mock and hasattr(self.provider, "client"):
            try:
                return self.provider.client.quotes(keys)
            except UpstoxAuthError:
                raise
            except Exception as error:
                logger.warning("Option quotes failed (open trades are not marked this cycle): %s", error)
                return {}
        out = {}
        snap = self.provider.snapshot(now, want_chain={t.index for t in trades})
        for isnap in snap.indices.values():
            chain = isnap.chain
            if chain is None or chain.empty:
                continue
            for side in ("ce", "pe"):
                for _, row in chain.iterrows():
                    out[row.get(f"{side}_key")] = {"bid": row.get(f"{side}_bid"), "ask": row.get(f"{side}_ask"),
                                                   "ltp": row.get(f"{side}_ltp")}
        return out

    def _trade_event(self, t, action: str):
        alert = trade_alert(t, action)
        logger.warning("%s", alert["message"])
        self.broadcaster.publish("trade", alert)
        if self.trade_notifier is not None:
            self.trade_notifier.emit(alert, push=self.push_enabled)
        if self.push_enabled:
            self.webhook.send(alert)

    def close_trade(self, trade_id: str) -> Optional[dict]:
        if self.trader is None:
            return None
        t = self.trader.close_manual(trade_id, self.provider.now())
        if t is None:
            return None
        self._trade_event(t, "close")
        return t.to_dict()

    def snapshot_state(self) -> dict:
        with self._lock:
            return dict(self.state, stream_clients=self.broadcaster.clients)
