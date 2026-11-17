"""
Upstox order routing for NSE cash equity (LIVE). It extends the existing UpstoxBroker (same HTTP layer, rate limit,
token source, status mapping, send-once rule) and changes only what differs for equity:

    * place_order accepts NSE_EQ instrument keys only (the parent accepts F&O keys only)
    * product "I" = intraday, "D" = delivery (swing)
    * holdings() reads the demat holdings for the swing reconciliation

Endpoints: POST api-hft.upstox.com/v3/order/place · PUT /v3/order/modify · DELETE /v3/order/cancel ·
GET api.upstox.com/v2/order/details · /v2/order/retrieve-all · /v2/portfolio/short-term-positions ·
/v2/portfolio/long-term-holdings · /v2/user/get-funds-and-margin. Order endpoints cannot be exercised without
sending an order, so check them against https://upstox.com/developer/api-documentation/orders before arming.

Nothing imports this module unless live trading is armed for today (src/equity/execution.py).
"""
from __future__ import annotations

from typing import List

from src.broker.client import valid_instrument_key
from src.execution.broker_api import BrokerError, BrokerPosition, OrderRequest
from src.execution.upstox_broker import API_BASE, HFT_BASE, UpstoxBroker
from src.utils.numbers import fnum


class UpstoxEquityBroker(UpstoxBroker):
    name = "upstox-equity"

    def place_order(self, req: OrderRequest) -> List[str]:
        if not valid_instrument_key(req.instrument_key) or not req.instrument_key.startswith("NSE_EQ|"):
            raise BrokerError(f"refusing to trade {req.instrument_key!r}: not an NSE cash-equity instrument key")
        if req.product not in ("I", "D") or req.side not in ("BUY", "SELL") or int(req.quantity) < 1:
            raise BrokerError("invalid equity order (product must be I or D, side BUY or SELL, quantity >= 1)")
        if req.product == "D" and req.side == "SELL" and not getattr(req, "closing", False):
            raise BrokerError("refusing a delivery SELL that does not close a holding (no short selling in delivery)")
        body = {"quantity": int(req.quantity), "product": req.product, "validity": req.validity,
                "price": round(float(req.price), 2) if req.order_type in ("LIMIT", "SL") else 0,
                "tag": req.tag[:20], "instrument_token": req.instrument_key, "order_type": req.order_type,
                "transaction_type": req.side, "disclosed_quantity": 0,
                "trigger_price": round(float(req.trigger_price), 2) if req.order_type in ("SL", "SL-M") else 0,
                "is_amo": False, "slice": False}
        if req.order_type in ("MARKET", "SL-M"):
            body["market_protection"] = -1
        payload = self._call("POST", f"{HFT_BASE}/v3/order/place", body=body, retry=False)
        data = payload.get("data") or {}
        ids = data.get("order_ids") or ([data["order_id"]] if data.get("order_id") else [])
        if not ids:
            raise BrokerError("Upstox accepted the request but returned no order id", maybe_sent=True)
        return [str(i) for i in ids]

    def holdings(self) -> List[BrokerPosition]:
        payload = self._call("GET", f"{API_BASE}/v2/portfolio/long-term-holdings")
        out = []
        for p in payload.get("data") or []:
            key = p.get("instrument_token") or p.get("instrument_key")
            if key:
                out.append(BrokerPosition(instrument_key=key, quantity=int(fnum(p.get("quantity"), 0)),
                                          average_price=fnum(p.get("average_price"), 0.0),
                                          last_price=fnum(p.get("last_price"), 0.0), pnl=fnum(p.get("pnl"), 0.0)))
        return out

    def order_book(self) -> List[dict]:
        payload = self._call("GET", f"{API_BASE}/v2/order/retrieve-all")
        return list(payload.get("data") or [])
