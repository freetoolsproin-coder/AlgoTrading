"""
Order execution engine. It MIRRORS the decisions of one lane's paper book ([EXECUTION] lane, default v1) to a
broker: when the paper book opens, scales out of or closes a position, the order manager sends the matching
orders. The strategy never talks to a broker directly.

    [EXECUTION] mode = off     no order path (paper book only, as before)
                       paper   (default) every decision goes through this order path to the simulated
                               PaperBroker: validation, journal, partial fills, reconciliation - no real orders
                       live    UpstoxBroker, ONLY while src/execution/live_gate.py reports "armed" for today;
                               otherwise it behaves exactly like paper and says why on the dashboard

Pre-trade validation (every entry; any failure = no order, reason logged and shown)
    data freshness, quote age, contract validity (F&O key, expiry not past), market session / last entry time,
    available margin, maximum lots (max_live_lots) and risk, stop-loss validity, position limits, duplicate
    positions / orders (idempotency key + broker positions), broker connectivity, daily order count

Orders
    entry   LIMIT at ask x (1 + entry_buffer_pct%), never above the planned premium x (1 + max_entry_slip_pct%).
            Not filled within order_timeout_seconds -> the remainder is cancelled (entries never chase);
            a partial fill becomes the live quantity.
    stop    protective_sl = true: a SELL SL order at the paper stop rests at the broker after every fill, so a
            stop exists even if this program stops. It is moved with the paper stop (breakeven / trailing) and
            cancelled before any other exit.
    exit    LIMIT at bid x (1 - exit_buffer_pct%), re-priced to the new bid up to max_reprice times, then MARKET
            with exchange market protection (exit_market_after_retries). Exits always complete.
    idempotency: each order has a key (position, purpose, sequence) and a short broker tag derived from it; a key
            is never sent twice, and after a timeout the order is looked up by its tag instead of being re-sent.
    reconciliation: every reconcile_seconds the expected live quantity per contract is compared with the broker's
            positions; a mismatch blocks new live entries and raises an alert until it is resolved.
    emergency close: cancel every working order and close every live position (kill switch, risk halt).

Live quantity is capped at max_live_lots per position. Scale-outs are mirrored proportionally in whole lots,
rounded DOWN (a 1-lot live position therefore exits in full on the final paper exit).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.broker.client import valid_instrument_key
from src.config import settings
from src.execution import live_gate
from src.execution.broker_api import (CANCELLED, FILLED, OPEN, PENDING, REJECTED, TERMINAL, UNKNOWN, Broker,
                                      BrokerError, OrderRequest)
from src.execution.paper_broker import PaperBroker
from src.utils.numbers import finite, fnum

logger = logging.getLogger("scanner.execution")


@dataclass
class ExecSettings:
    mode: str = "paper"                  # off | paper | live
    lane: str = "v1"
    entry_buffer_pct: float = 0.5
    max_entry_slip_pct: float = 2.0
    exit_buffer_pct: float = 1.0
    order_timeout_seconds: float = 20
    max_reprice: int = 2
    exit_market_after_retries: bool = True
    protective_sl: bool = True
    sl_limit_gap_pct: float = 5.0        # SL order limit price this % below its trigger
    reconcile_seconds: float = 60
    max_live_lots: int = 1
    max_orders_per_day: int = 20
    max_quote_age_seconds: float = 30
    min_margin_buffer_pct: float = 10
    product: str = "I"
    tag_prefix: str = "pb"
    algo_name: str = ""

    @classmethod
    def from_mapping(cls, raw: Optional[dict]) -> "ExecSettings":
        s = cls()
        for k, v in (raw or {}).items():
            if not hasattr(s, k):
                raise ValueError(f"config/scanner.ini: [EXECUTION] unknown setting {k!r}")
            setattr(s, k, settings._coerce(getattr(cls(), k), v))
        s.mode = s.mode.lower()
        if s.mode not in ("off", "paper", "live"):
            raise ValueError("config/scanner.ini: [EXECUTION] mode must be off, paper or live")
        if s.lane not in ("v1", "v2", "v3"):
            raise ValueError("config/scanner.ini: [EXECUTION] lane must be v1, v2 or v3")
        if s.max_live_lots < 1:
            raise ValueError("config/scanner.ini: [EXECUTION] max_live_lots must be >= 1")
        return s


@dataclass
class ManagedOrder:
    key: str
    tag: str
    position_id: str
    purpose: str                 # entry | exit | sl
    req: dict
    status: str = PENDING
    order_ids: List[str] = field(default_factory=list)
    filled: int = 0
    avg_price: float = 0.0
    created: str = ""
    updated: str = ""
    reprices: int = 0
    message: str = ""
    accounted: int = 0           # filled quantity already applied to the live position
    acted: str = ""              # time of the last action (placement / re-price) - timeouts count from here

    def to_dict(self):
        return asdict(self)


@dataclass
class LivePosition:
    position_id: str
    index: str
    instrument_key: str
    symbol: str
    lot_size: int
    qty: int = 0                 # filled, still open
    entry_avg: float = 0.0
    planned_qty: int = 0
    sl_trigger: float = float("nan")
    sl_key: str = ""
    exit_seq: int = 0
    closed: bool = False
    realised: float = 0.0
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        d["sl_trigger"] = None if not finite(self.sl_trigger) else round(self.sl_trigger, 2)
        return d


def make_tag(prefix: str, key: str) -> str:
    return (prefix + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12])[:20]


class OrderManager:
    def __init__(self, broker: Broker, s: ExecSettings, journal: Optional[Path] = None,
                 alert_fn: Optional[Callable[[str, str], None]] = None, mock: bool = False):
        self.broker, self.s, self.mock = broker, s, mock
        self.journal = Path(journal) if journal else settings.DATA_DIR / "execution.db"
        self.alert_fn = alert_fn or (lambda level, msg: logger.warning("[execution] %s", msg))
        self.orders: Dict[str, ManagedOrder] = {}
        self.positions: Dict[str, LivePosition] = {}
        self.day: Optional[date] = None
        self.orders_today = 0
        self.blocked: Optional[str] = None           # reconciliation mismatch / broker problem
        self.last_reconcile: Optional[pd.Timestamp] = None
        self.last_problem = ""
        self.rejections: List[str] = []
        self._lock = threading.RLock()
        self._init_journal()

    # ---------------------------------------------------------------- journal
    @contextlib.contextmanager
    def _conn(self):
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.journal), timeout=10)
        try:
            with conn:          # commit / rollback
                yield conn
        finally:
            conn.close()

    def _init_journal(self):
        try:
            with self._conn() as c:
                c.execute("CREATE TABLE IF NOT EXISTS orders (key TEXT PRIMARY KEY, day TEXT, broker TEXT, payload TEXT)")
                c.execute("CREATE TABLE IF NOT EXISTS live_positions (id TEXT PRIMARY KEY, day TEXT, broker TEXT, payload TEXT)")
        except sqlite3.Error:
            logger.exception("execution journal unavailable")

    def _save(self, o: Optional[ManagedOrder] = None, p: Optional[LivePosition] = None):
        try:
            with self._conn() as c:
                day = (self.day or date.today()).isoformat()
                if o is not None:
                    c.execute("INSERT OR REPLACE INTO orders VALUES (?,?,?,?)", (o.key, day, self.broker.name,
                                                                                 json.dumps(o.to_dict())))
                if p is not None:
                    c.execute("INSERT OR REPLACE INTO live_positions VALUES (?,?,?,?)",
                              (p.position_id, day, self.broker.name, json.dumps(p.to_dict())))
        except sqlite3.Error:
            logger.exception("could not journal an order")

    def load_day(self, day: date):
        """After a restart: today's orders and live positions come back from the journal (same broker only)."""
        self.day = day
        try:
            with self._conn() as c:
                for (payload,) in c.execute("SELECT payload FROM orders WHERE day = ? AND broker = ?",
                                            (day.isoformat(), self.broker.name)):
                    o = ManagedOrder(**json.loads(payload))
                    self.orders[o.key] = o
                for (payload,) in c.execute("SELECT payload FROM live_positions WHERE day = ? AND broker = ?",
                                            (day.isoformat(), self.broker.name)):
                    d = json.loads(payload)
                    d["sl_trigger"] = float("nan") if d.get("sl_trigger") is None else d["sl_trigger"]
                    p = LivePosition(**d)
                    self.positions[p.position_id] = p
        except (sqlite3.Error, TypeError, ValueError):
            logger.exception("could not reload the execution journal")
        self.orders_today = sum(1 for o in self.orders.values() if o.purpose == "entry")

    def roll(self, day: date):
        if self.day != day:
            self.orders, self.positions = {}, {}
            self.orders_today, self.blocked, self.rejections = 0, None, []
            self.load_day(day)

    # ---------------------------------------------------------------- sending
    def _send(self, key: str, position_id: str, purpose: str, req: OrderRequest, now: pd.Timestamp) -> Optional[ManagedOrder]:
        with self._lock:
            if key in self.orders:
                logger.warning("[execution] duplicate order key %s refused", key)
                return None
            req.tag = make_tag(self.s.tag_prefix, key)
            req.product = self.s.product
            o = ManagedOrder(key=key, tag=req.tag, position_id=position_id, purpose=purpose, req=req.to_dict(),
                             created=now.isoformat(), updated=now.isoformat(), acted=now.isoformat())
            self.orders[key] = o
            self._save(o)
            try:
                o.order_ids = self.broker.place_order(req)
                o.status = OPEN
            except BrokerError as error:
                if error.maybe_sent:
                    o.status = UNKNOWN
                    o.message = f"outcome unknown ({error}); will look it up by tag {req.tag}"
                    self.alert_fn("warning", f"{purpose} order {req.tag}: outcome unknown, checking by tag")
                else:
                    o.status = REJECTED
                    o.message = str(error)
                    self.rejections.append(f"{now:%H:%M} {purpose}: {error}")
                    self.alert_fn("warning", f"{purpose} order refused: {error}")
            self._refresh(o, now)
            self._save(o)
            return o

    def _refresh(self, o: ManagedOrder, now: pd.Timestamp):
        """Pull the latest status of every broker order behind a managed order."""
        if o.status in TERMINAL and o.filled == o.accounted:
            return
        try:
            if o.status == UNKNOWN and not o.order_ids:
                st = self.broker.order_status(tag=o.tag)
                if st is None:
                    age = (now - pd.Timestamp(o.created)).total_seconds()
                    if age > max(30.0, self.s.order_timeout_seconds):
                        o.status, o.message = REJECTED, "not found at the broker after a timeout: treated as not sent"
                    return
                o.order_ids = [st.order_id]
            sts = [self.broker.order_status(order_id=i) for i in o.order_ids]
        except BrokerError as error:
            self.last_problem = str(error)
            return
        sts = [x for x in sts if x is not None]
        if not sts:
            return
        filled = sum(x.filled_quantity for x in sts)
        o.avg_price = (sum(x.average_price * x.filled_quantity for x in sts) / filled) if filled else 0.0
        o.filled = filled
        states = {x.status for x in sts}
        if states <= {FILLED}:
            o.status = FILLED
        elif states & {OPEN, PENDING, UNKNOWN}:
            o.status = OPEN
        elif REJECTED in states and not filled:
            o.status = REJECTED
            o.message = "; ".join(x.message for x in sts if x.message)[:200]
            if o.message:
                self.rejections.append(f"{now:%H:%M} {o.purpose}: {o.message}")
                self.alert_fn("warning", f"{o.purpose} order rejected: {o.message}")
        else:
            o.status = CANCELLED if filled < int(o.req["quantity"]) else FILLED
        o.updated = now.isoformat()

    # ---------------------------------------------------------------- validation
    def validate_entry(self, pos, quote: Tuple[float, float, Optional[pd.Timestamp]], now: pd.Timestamp, *,
                       data_ok: bool, market_open: bool, last_entry, live_open_limit: int, qty: int) -> List[str]:
        reasons = []
        bid, ask, qtime = quote
        if self.blocked:
            reasons.append(f"execution blocked: {self.blocked}")
        if not data_ok:
            reasons.append("market data not fresh")
        if qtime is not None and (now - qtime).total_seconds() > self.s.max_quote_age_seconds:
            reasons.append(f"option quote {(now - qtime).total_seconds():.0f}s old")
        if not (finite(ask) and ask > 0 and finite(bid) and bid > 0):
            reasons.append("no live bid / ask for the contract")
        key = pos.instrument_key
        mock_key = (not self.broker.live) and str(key).startswith("MOCK|")   # synthetic contracts in --mock
        if not mock_key and not (valid_instrument_key(key) and str(key).split("|")[0].endswith("_FO")):
            reasons.append(f"contract key {key!r} is not a valid F&O instrument")
        try:
            if pos.expiry and date.fromisoformat(str(pos.expiry)[:10]) < now.date():
                reasons.append(f"contract expired {pos.expiry}")
        except ValueError:
            reasons.append(f"bad expiry {pos.expiry!r}")
        if not market_open:
            reasons.append("market not open")
        if last_entry is not None and now.time() >= last_entry:
            reasons.append(f"after the last entry time {last_entry:%H:%M}")
        if qty < pos.lot_size or qty % max(pos.lot_size, 1):
            reasons.append(f"quantity {qty} is not a whole number of lots ({pos.lot_size})")
        if qty > self.s.max_live_lots * pos.lot_size:
            reasons.append(f"quantity above max_live_lots {self.s.max_live_lots}")
        if not (finite(pos.sl) and 0 < pos.sl < pos.entry_price):
            reasons.append(f"invalid premium stop {pos.sl} for entry {pos.entry_price}")
        open_live = [p for p in self.positions.values() if not p.closed and p.qty > 0]
        if len(open_live) >= live_open_limit:
            reasons.append(f"{len(open_live)} live positions open (max {live_open_limit})")
        if any(p.instrument_key == key and not p.closed for p in self.positions.values()):
            reasons.append("a live position in this contract already exists")
        if any(o.purpose == "entry" and o.status in (OPEN, PENDING, UNKNOWN) and o.req.get("instrument_key") == key
               for o in self.orders.values()):
            reasons.append("an entry order for this contract is already working")
        if self.orders_today >= self.s.max_orders_per_day:
            reasons.append(f"{self.orders_today} entry orders today (max {self.s.max_orders_per_day})")
        if reasons:
            return reasons
        ok, why = self.broker.healthy()
        if not ok:
            return [f"broker not reachable: {why}"]
        try:
            held = {p.instrument_key: p.quantity for p in self.broker.positions() if p.quantity}
            if held.get(key):
                reasons.append(f"broker already holds {held[key]} of {key}")
            funds = self.broker.funds()
            need = ask * qty * (1 + self.s.min_margin_buffer_pct / 100.0)
            if funds.available_margin < need:
                reasons.append(f"available margin Rs {funds.available_margin:,.0f} < Rs {need:,.0f} needed")
        except BrokerError as error:
            reasons.append(f"broker check failed: {error}")
        return reasons

    # ---------------------------------------------------------------- mirror the paper book
    def on_entry(self, pos, quote, now: pd.Timestamp, *, data_ok: bool, market_open: bool, last_entry=None,
                 live_open_limit: int = 3) -> Tuple[bool, str]:
        with self._lock:
            self.roll(now.date())
            if hasattr(self.broker, "set_quote") and finite(quote[0]) and finite(quote[1]):
                self.broker.set_quote(pos.instrument_key, quote[0], quote[1])
            qty = min(pos.qty_open, self.s.max_live_lots * pos.lot_size)
            qty -= qty % max(pos.lot_size, 1)
            reasons = self.validate_entry(pos, quote, now, data_ok=data_ok, market_open=market_open,
                                          last_entry=last_entry, live_open_limit=live_open_limit, qty=qty)
            if reasons:
                why = "; ".join(reasons)
                self.rejections.append(f"{now:%H:%M} entry {pos.symbol}: {why}")
                logger.warning("[execution] entry for %s not sent: %s", pos.symbol, why)
                return False, why
            ask = quote[1]
            limit = min(ask * (1 + self.s.entry_buffer_pct / 100.0), pos.entry_price * (1 + self.s.max_entry_slip_pct / 100.0))
            limit = _tick(limit, up=True)
            lp = LivePosition(position_id=pos.id, index=pos.index, instrument_key=pos.instrument_key, symbol=pos.symbol,
                              lot_size=pos.lot_size, planned_qty=qty, sl_trigger=pos.sl)
            self.positions[pos.id] = lp
            o = self._send(f"{pos.id}|entry", pos.id, "entry",
                           OrderRequest(pos.instrument_key, "BUY", qty, "LIMIT", price=limit), now)
            self.orders_today += 1
            self._save(p=lp)
            self._apply_fills(now)
            return (o is not None and o.status != REJECTED), (o.message if o else "duplicate")

    def on_exit(self, pos, qty_paper_sold: int, qty_paper_before: int, quote, now: pd.Timestamp, reason: str):
        """The paper book sold `qty_paper_sold` of `qty_paper_before`: sell the proportional live quantity."""
        with self._lock:
            lp = self.positions.get(pos.id)
            if lp is None or lp.closed:
                return
            if hasattr(self.broker, "set_quote") and finite(quote[0]) and finite(quote[1]):
                self.broker.set_quote(lp.instrument_key, quote[0], quote[1])
            self._cancel_entry_remainder(lp, now)
            full = pos.closed or qty_paper_sold >= qty_paper_before
            if not full:
                lots = int(lp.qty // lp.lot_size * qty_paper_sold / max(qty_paper_before, 1))
                if lots < 1:
                    return          # a scale-out smaller than one live lot: the live lot waits for the final exit
            self._cancel_stop(lp, now)   # also folds in a stop that filled in the meantime
            if full:
                qty = lp.qty
            else:
                qty = min(lp.qty, int(lp.qty // lp.lot_size * qty_paper_sold / max(qty_paper_before, 1)) * lp.lot_size)
            if qty <= 0:
                return
            self._exit(lp, qty, quote[0], now, reason)

    def update_stop(self, pos, now: pd.Timestamp):
        """Move the resting stop with the paper stop (breakeven / trailing). Only ever tightened."""
        with self._lock:
            lp = self.positions.get(pos.id)
            if lp is None or lp.closed or not self.s.protective_sl or not finite(pos.sl):
                return
            if finite(lp.sl_trigger) and pos.sl <= lp.sl_trigger + 0.05:
                return
            lp.sl_trigger = pos.sl
            o = self.orders.get(lp.sl_key) if lp.sl_key else None
            if o is not None and o.status == OPEN and o.order_ids:
                try:
                    trig = _tick(pos.sl, up=False)
                    self.broker.modify_order(o.order_ids[0], quantity=lp.qty, trigger_price=trig,
                                             price=_tick(trig * (1 - self.s.sl_limit_gap_pct / 100.0), up=False),
                                             order_type="SL")
                    o.req.update(trigger_price=trig)
                    self._save(o, lp)
                except BrokerError as error:
                    self.alert_fn("warning", f"could not move the stop for {lp.symbol}: {error}")

    def close_all(self, now: pd.Timestamp, reason: str, quotes: Optional[Dict[str, Tuple[float, float]]] = None):
        """Emergency close: cancel every working order, then sell every live quantity."""
        with self._lock:
            for o in list(self.orders.values()):
                if o.status in (OPEN, PENDING) and o.order_ids:
                    for oid in o.order_ids:
                        try:
                            self.broker.cancel_order(oid)
                        except BrokerError:
                            pass
                    self._refresh(o, now)
            self._apply_fills(now)
            for lp in self.positions.values():
                if lp.closed or lp.qty <= 0:
                    continue
                bid = (quotes or {}).get(lp.instrument_key, (float("nan"), float("nan")))[0]
                self._exit(lp, lp.qty, bid, now, f"emergency close: {reason}", market=not finite(bid))

    # ---------------------------------------------------------------- internals
    def _exit(self, lp: LivePosition, qty: int, bid: float, now: pd.Timestamp, reason: str, market: bool = False):
        lp.exit_seq += 1
        if market or not finite(bid) or bid <= 0:
            req = OrderRequest(lp.instrument_key, "SELL", qty, "MARKET")
        else:
            req = OrderRequest(lp.instrument_key, "SELL", qty, "LIMIT",
                               price=_tick(bid * (1 - self.s.exit_buffer_pct / 100.0), up=False))
        o = self._send(f"{lp.position_id}|exit|{lp.exit_seq}", lp.position_id, "exit", req, now)
        if o is not None:
            o.message = reason
        self._apply_fills(now)

    def _cancel_entry_remainder(self, lp, now):
        o = self.orders.get(f"{lp.position_id}|entry")
        if o is not None and o.status in (OPEN, PENDING) and o.order_ids:
            for oid in o.order_ids:
                try:
                    self.broker.cancel_order(oid)
                except BrokerError:
                    pass
            self._refresh(o, now)

    def _cancel_stop(self, lp: LivePosition, now):
        o = self.orders.get(lp.sl_key) if lp.sl_key else None
        if o is None or o.status not in (OPEN, PENDING):
            return
        for oid in o.order_ids:
            try:
                self.broker.cancel_order(oid)
            except BrokerError:
                pass
        self._refresh(o, now)
        self._apply_fills(now)      # the stop may have filled in the meantime

    def _place_stop(self, lp: LivePosition, now):
        if not self.s.protective_sl or lp.qty <= 0 or not finite(lp.sl_trigger) or lp.sl_trigger <= 0:
            return
        current = self.orders.get(lp.sl_key) if lp.sl_key else None
        if current is not None and current.status in (OPEN, PENDING, UNKNOWN):
            if int(current.req["quantity"]) - current.filled == lp.qty:
                return
            self._cancel_stop(lp, now)
        trig = _tick(lp.sl_trigger, up=False)
        n = sum(1 for k in self.orders if k.startswith(f"{lp.position_id}|sl|")) + 1
        o = self._send(f"{lp.position_id}|sl|{n}", lp.position_id, "sl",
                       OrderRequest(lp.instrument_key, "SELL", lp.qty, "SL", trigger_price=trig,
                                    price=_tick(trig * (1 - self.s.sl_limit_gap_pct / 100.0), up=False)), now)
        if o is not None:
            lp.sl_key = o.key

    def _apply_fills(self, now):
        """Fold new fills into the live positions; place / resize the protective stop after entry fills."""
        for o in list(self.orders.values()):
            self._refresh(o, now)
            new = o.filled - o.accounted
            if new <= 0:
                continue
            lp = self.positions.get(o.position_id)
            o.accounted = o.filled
            if lp is None:
                continue
            if o.purpose == "entry":
                total = lp.qty + new
                lp.entry_avg = (lp.entry_avg * lp.qty + o.avg_price * new) / total if total else 0.0
                lp.qty = total
            else:
                lp.realised += (o.avg_price - lp.entry_avg) * new
                lp.qty = max(0, lp.qty - new)
                if o.purpose == "sl":
                    self.alert_fn("warning", f"protective stop filled at the broker: {lp.symbol} {new} @ {o.avg_price:.2f}")
                if lp.qty == 0 and not self._entry_working(lp):
                    lp.closed = True
            self._save(o, lp)
            exit_working = any(x.purpose == "exit" and x.position_id == lp.position_id and x.status in (OPEN, PENDING, UNKNOWN)
                               for x in self.orders.values())
            if lp.qty > 0 and not exit_working and o.purpose in ("entry", "exit"):
                self._place_stop(lp, now)   # (re)size the resting stop to the live quantity

    def _entry_working(self, lp) -> bool:
        o = self.orders.get(f"{lp.position_id}|entry")
        return o is not None and o.status in (OPEN, PENDING, UNKNOWN)

    def poll(self, now: pd.Timestamp, quotes: Optional[Dict[str, Tuple[float, float]]] = None):
        """Every cycle: refresh working orders, handle timeouts (cancel entries, re-price exits), reconcile."""
        with self._lock:
            self.roll(now.date())
            for key, (bid, ask) in (quotes or {}).items():
                if hasattr(self.broker, "set_quote") and finite(bid) and finite(ask):
                    self.broker.set_quote(key, bid, ask)
            self._apply_fills(now)
            for o in list(self.orders.values()):
                if o.status not in (OPEN, PENDING):
                    continue
                age = (now - pd.Timestamp(o.acted or o.created)).total_seconds()
                if age < self.s.order_timeout_seconds:
                    continue
                if o.purpose == "entry":
                    for oid in o.order_ids:
                        try:
                            self.broker.cancel_order(oid)
                        except BrokerError:
                            pass
                    self._refresh(o, now)
                    o.message = f"entry not filled in {self.s.order_timeout_seconds:.0f}s: remainder cancelled (no chasing)"
                    self._save(o)
                elif o.purpose == "exit":
                    self._reprice_exit(o, (quotes or {}).get(o.req["instrument_key"]), now)
            self._apply_fills(now)
            for lp in self.positions.values():
                if not lp.closed and lp.qty == 0 and not self._entry_working(lp):
                    lp.closed = True
                    lp.note = lp.note or "entry not filled"
                    self._save(p=lp)
            if self.last_reconcile is None or (now - self.last_reconcile).total_seconds() >= self.s.reconcile_seconds:
                self.reconcile(now)

    def _reprice_exit(self, o: ManagedOrder, quote, now):
        remaining = int(o.req["quantity"]) - o.filled
        if remaining <= 0 or not o.order_ids:
            return
        if o.reprices >= self.s.max_reprice and self.s.exit_market_after_retries:
            try:
                self.broker.modify_order(o.order_ids[0], quantity=int(o.req["quantity"]), order_type="MARKET")
                o.req["order_type"] = "MARKET"
            except BrokerError as error:
                self.alert_fn("critical", f"exit for {o.req['instrument_key']} could not be made MARKET: {error}")
        elif quote is not None and finite(quote[0]) and quote[0] > 0:
            price = _tick(quote[0] * (1 - self.s.exit_buffer_pct / 100.0), up=False)
            try:
                self.broker.modify_order(o.order_ids[0], quantity=int(o.req["quantity"]), price=price, order_type="LIMIT")
                o.req["price"] = price
            except BrokerError as error:
                self.alert_fn("warning", f"exit re-price failed: {error}")
        o.reprices += 1
        o.updated = o.acted = now.isoformat()
        self._save(o)

    def reconcile(self, now: pd.Timestamp) -> dict:
        """Expected live quantity per contract vs the broker's positions."""
        self.last_reconcile = now
        expected: Dict[str, int] = {}
        for lp in self.positions.values():
            if lp.qty:
                expected[lp.instrument_key] = expected.get(lp.instrument_key, 0) + lp.qty
        try:
            held = {p.instrument_key: p.quantity for p in self.broker.positions() if p.quantity}
        except BrokerError as error:
            self.last_problem = f"reconciliation skipped: {error}"
            return {"ok": None, "note": self.last_problem}
        ours = set(expected)
        mismatch = {k: (expected.get(k, 0), held.get(k, 0)) for k in ours | (set(held) & ours)
                    if expected.get(k, 0) != held.get(k, 0)}
        if mismatch:
            text = ", ".join(f"{k}: expected {e}, broker {h}" for k, (e, h) in mismatch.items())
            if self.blocked != f"position mismatch: {text}":
                self.alert_fn("critical", f"position mismatch - new live entries blocked: {text}")
            self.blocked = f"position mismatch: {text}"
        elif self.blocked and self.blocked.startswith("position mismatch"):
            self.alert_fn("warning", "positions reconciled again: live entries allowed")
            self.blocked = None
        return {"ok": not mismatch, "mismatch": mismatch, "untracked": sorted(set(held) - ours)}

    def snapshot(self) -> dict:
        with self._lock:
            return {"broker": self.broker.name, "live": bool(self.broker.live), "lane": self.s.lane,
                    "blocked": self.blocked, "orders_today": self.orders_today,
                    "working": [o.to_dict() for o in self.orders.values() if o.status in (OPEN, PENDING, UNKNOWN)],
                    "recent": [o.to_dict() for o in list(self.orders.values())[-15:]],
                    "positions": [p.to_dict() for p in self.positions.values()],
                    "rejections": self.rejections[-10:], "last_problem": self.last_problem,
                    "last_reconcile": self.last_reconcile.isoformat() if self.last_reconcile is not None else None}


def _tick(price: float, up: bool, tick: float = 0.05) -> float:
    """Round to the option tick size (0.05), up for buys / down for sells."""
    if not finite(price):
        return price
    n = price / tick
    n = math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)
    return round(max(tick, n * tick), 2)


# =============================================================================
# Factory used by the scanner engine
# =============================================================================
class Execution:
    """What the engine holds: the order manager plus the live-gate state, rebuilt when the gate changes."""

    def __init__(self, raw: dict, capital: float, mock: bool, alert_fn=None, broker_factory=None):
        self.s = ExecSettings.from_mapping(raw)
        self.raw, self.capital, self.mock = raw, capital, mock
        self.alert_fn = alert_fn
        self.broker_factory = broker_factory
        self.manager: Optional[OrderManager] = None
        self.gate_note = ""
        self.armed = False
        self._built_for: Optional[Tuple[str, date]] = None

    @property
    def enabled(self) -> bool:
        return self.s.mode != "off"

    def ensure(self, today: date) -> Optional[OrderManager]:
        if not self.enabled:
            return None
        armed, missing = live_gate.status(self.raw, today, mock=self.mock) if self.s.mode == "live" else (False, [])
        want = "live" if armed else "paper"
        if self._built_for != (want, today):
            if want == "live":
                if self.broker_factory is not None:
                    broker = self.broker_factory()
                else:
                    from src.execution.upstox_broker import UpstoxBroker
                    broker = UpstoxBroker(algo_name=self.s.algo_name)
                logger.critical("LIVE ORDER EXECUTION ARMED for %s (lane %s, max %d lot(s) per position).",
                                today, self.s.lane, self.s.max_live_lots)
            else:
                broker = PaperBroker(capital=self.capital)
            self.manager = OrderManager(broker, self.s, alert_fn=self.alert_fn, mock=self.mock)
            self.manager.roll(today)
            self._built_for = (want, today)
        self.armed = armed
        self.gate_note = "" if armed or self.s.mode != "live" else "live requested but not armed: " + "; ".join(missing)
        return self.manager

    def snapshot(self) -> dict:
        base = {"mode": self.s.mode, "armed": self.armed, "gate": self.gate_note, "lane": self.s.lane}
        if self.manager is not None:
            base.update(self.manager.snapshot())
        return base
