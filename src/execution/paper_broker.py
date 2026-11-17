"""
Simulated broker for PAPER execution: the same order path as live (order manager, validation, journal,
reconciliation) without sending anything anywhere.

Fills use the latest option quote supplied by the engine (`set_quote(instrument_key, bid, ask)`):
    BUY LIMIT   fills when ask <= limit price, at the ask
    SELL LIMIT  fills when bid >= limit price, at the bid
    MARKET      fills at the ask (BUY) / bid (SELL)
    SL (sell)   triggers when bid <= trigger, then fills at the bid (never better than the limit)
Optional realism for tests: `partial_fill_ratio` fills only that share of each order on the first check,
`reject_next` makes the next placement fail with a message.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional

from src.execution.broker_api import (CANCELLED, FILLED, OPEN, REJECTED, Broker, BrokerError, BrokerPosition, Funds,
                                      OrderRequest, OrderStatus)


class PaperBroker(Broker):
    name = "paper"
    live = False

    def __init__(self, capital: float = 500000.0, partial_fill_ratio: float = 1.0):
        self.capital = float(capital)
        self.partial_fill_ratio = partial_fill_ratio
        self.reject_next: Optional[str] = None
        self.quotes: Dict[str, tuple] = {}
        self.quote_seq: Dict[str, int] = {}
        self.orders: Dict[str, dict] = {}
        self.net: Dict[str, int] = {}
        self.avg: Dict[str, float] = {}
        self.cash_used = 0.0
        self._ids = itertools.count(1)

    def set_quote(self, instrument_key: str, bid: float, ask: float):
        self.quotes[instrument_key] = (float(bid), float(ask))
        self.quote_seq[instrument_key] = self.quote_seq.get(instrument_key, 0) + 1

    # ---------------------------------------------------------------- Broker
    def place_order(self, req: OrderRequest) -> List[str]:
        if self.reject_next:
            reason, self.reject_next = self.reject_next, None
            oid = f"P{next(self._ids)}"
            self.orders[oid] = {"req": req, "status": REJECTED, "filled": 0, "avg": 0.0, "msg": reason, "triggered": False,
                                "seq": -1}
            return [oid]
        if req.quantity <= 0:
            raise BrokerError("quantity must be positive")
        oid = f"P{next(self._ids)}"
        self.orders[oid] = {"req": req, "status": OPEN, "filled": 0, "avg": 0.0, "msg": "", "triggered": False, "seq": -1}
        self._try_fill(oid, first=True)
        return [oid]

    def modify_order(self, order_id, *, quantity=None, price=None, trigger_price=None, order_type=None):
        o = self._get(order_id)
        if o["status"] != OPEN:
            raise BrokerError(f"order {order_id} is {o['status']}, cannot modify")
        r = o["req"]
        if quantity is not None:
            if quantity < o["filled"]:
                raise BrokerError("new quantity below the filled quantity")
            r.quantity = int(quantity)
        if price is not None:
            r.price = float(price)
        if trigger_price is not None:
            r.trigger_price = float(trigger_price)
        if order_type is not None:
            r.order_type = order_type
        self._try_fill(order_id)

    def cancel_order(self, order_id):
        o = self._get(order_id)
        if o["status"] == OPEN:
            o["status"] = CANCELLED

    def order_status(self, order_id="", tag="") -> Optional[OrderStatus]:
        if not order_id and tag:
            order_id = next((k for k, v in self.orders.items() if v["req"].tag == tag), "")
        if not order_id or order_id not in self.orders:
            return None
        self._try_fill(order_id)
        o = self.orders[order_id]
        r = o["req"]
        return OrderStatus(order_id=order_id, status=o["status"], quantity=r.quantity, filled_quantity=o["filled"],
                           average_price=o["avg"], pending_quantity=max(0, r.quantity - o["filled"]) if o["status"] == OPEN else 0,
                           message=o["msg"], tag=r.tag, raw_status=o["status"].lower())

    def positions(self) -> List[BrokerPosition]:
        return [BrokerPosition(k, q, self.avg.get(k, 0.0)) for k, q in self.net.items() if q != 0]

    def funds(self) -> Funds:
        return Funds(available_margin=self.capital - self.cash_used, used_margin=self.cash_used)

    # ---------------------------------------------------------------- simulation
    def _get(self, order_id) -> dict:
        if order_id not in self.orders:
            raise BrokerError(f"unknown order {order_id}")
        return self.orders[order_id]

    def _try_fill(self, oid: str, first: bool = False):
        o = self.orders[oid]
        if o["status"] != OPEN:
            return
        r = o["req"]
        q = self.quotes.get(r.instrument_key)
        if q is None:
            return
        bid, ask = q
        price = None
        if r.order_type == "MARKET":
            price = ask if r.side == "BUY" else bid
        elif r.order_type in ("SL", "SL-M"):
            if not o["triggered"]:
                hit = (bid <= r.trigger_price) if r.side == "SELL" else (ask >= r.trigger_price)
                if not hit:
                    return
                o["triggered"] = True
            if r.order_type == "SL-M":
                price = ask if r.side == "BUY" else bid
            elif r.side == "SELL" and bid >= r.price:
                price = bid
            elif r.side == "BUY" and ask <= r.price:
                price = ask
        elif r.side == "BUY" and ask <= r.price:
            price = ask
        elif r.side == "SELL" and bid >= r.price:
            price = bid
        if price is None or price <= 0:
            return
        remaining = r.quantity - o["filled"]
        qty = remaining
        if self.partial_fill_ratio < 1.0:
            # partial-fill simulation: at most one fill per quote update, the first one only a share of the order
            seq = self.quote_seq.get(r.instrument_key, 0)
            if o["seq"] == seq:
                return
            o["seq"] = seq
            if first:
                qty = max(1, int(remaining * self.partial_fill_ratio))
        o["avg"] = (o["avg"] * o["filled"] + price * qty) / (o["filled"] + qty)
        o["filled"] += qty
        signed = qty if r.side == "BUY" else -qty
        prev = self.net.get(r.instrument_key, 0)
        if signed > 0:
            self.avg[r.instrument_key] = (self.avg.get(r.instrument_key, 0.0) * max(prev, 0) + price * qty) / max(prev + qty, 1)
        self.net[r.instrument_key] = prev + signed
        self.cash_used += price * signed
        if o["filled"] >= r.quantity:
            o["status"] = FILLED
