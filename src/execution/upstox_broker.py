"""
Upstox implementation of the Broker interface (LIVE orders). Used only when live trading is armed
(src/execution/live_gate.py); nothing else in the application imports it.

Endpoints (Upstox developer API; check https://upstox.com/developer/api-documentation/orders before arming)
    POST   https://api-hft.upstox.com/v3/order/place              place (returns data.order_ids)
    PUT    https://api-hft.upstox.com/v3/order/modify             modify an open order
    DELETE https://api-hft.upstox.com/v3/order/cancel?order_id=   cancel an open order
    GET    https://api.upstox.com/v2/order/details?order_id=|tag= one order's latest status
    GET    https://api.upstox.com/v2/portfolio/short-term-positions
    GET    https://api.upstox.com/v2/user/get-funds-and-margin?segment=SEC

Safety rules in this class
    * A placement is sent ONCE. On a timeout / connection error it is reported as maybe_sent=True; the order
      manager then looks the order up by its tag instead of sending it again (no duplicate orders).
    * Every call shares the REST client's rate limit; HTTP 429 backs off and retries (reads, modify, cancel).
    * The access token comes from the same TokenSource as the market data (.env / upstox_token.json) and is
      never logged.
"""
from __future__ import annotations

import logging
import time as _time
from typing import List, Optional

from src.broker.client import TokenSource, UpstoxAuthError, valid_instrument_key
from src.execution.broker_api import (CANCELLED, FILLED, OPEN, PENDING, REJECTED, UNKNOWN, Broker, BrokerError,
                                      BrokerPosition, Funds, OrderRequest, OrderStatus)
from src.utils.numbers import fnum

logger = logging.getLogger("scanner.execution")
HFT_BASE = "https://api-hft.upstox.com"
API_BASE = "https://api.upstox.com"

STATUS_MAP = {"complete": FILLED, "rejected": REJECTED, "cancelled": CANCELLED, "open": OPEN,
              "trigger pending": OPEN, "modify pending": OPEN, "cancel pending": OPEN, "open pending": PENDING,
              "validation pending": PENDING, "put order req received": PENDING, "modify validation pending": OPEN,
              "after market order req received": PENDING, "not cancelled": OPEN, "not modified": OPEN}


class UpstoxBroker(Broker):
    name = "upstox"
    live = True

    def __init__(self, session=None, token_source: Optional[TokenSource] = None, requests_per_second: float = 5.0,
                 timeout: float = 8.0, algo_name: str = ""):
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.tokens = token_source or TokenSource()
        self.min_interval = 1.0 / max(requests_per_second, 0.5)
        self.timeout = timeout
        self.algo_name = algo_name
        self._last = 0.0

    # ---------------------------------------------------------------- http
    def _headers(self) -> dict:
        h = {"Accept": "application/json", "Content-Type": "application/json",
             "Authorization": f"Bearer {self.tokens.token()}"}
        if self.algo_name:
            h["X-Algo-Name"] = self.algo_name
        return h

    def _call(self, method: str, url: str, *, params=None, body=None, retry: bool = True) -> dict:
        attempts = 3 if retry else 1
        for attempt in range(attempts):
            wait = self._last + self.min_interval - _time.monotonic()
            if wait > 0:
                _time.sleep(wait)
            self._last = _time.monotonic()
            try:
                response = self.session.request(method, url, params=params, json=body, headers=self._headers(),
                                                timeout=self.timeout)
            except Exception as error:   # timeout / connection reset: outcome unknown for a placement
                if retry and attempt + 1 < attempts:
                    _time.sleep(0.5 * (attempt + 1))
                    continue
                raise BrokerError(f"{method} {url.split('?')[0]}: {type(error).__name__}", retryable=True,
                                  maybe_sent=(method == "POST")) from error
            status = getattr(response, "status_code", 200)
            if status == 401:
                raise UpstoxAuthError("Upstox token rejected (401).")
            if status == 429 and retry and attempt + 1 < attempts:
                _time.sleep(2 ** attempt)
                continue
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if status >= 400 or (isinstance(payload, dict) and payload.get("status") == "error"):
                errors = payload.get("errors") if isinstance(payload, dict) else None
                msg = "; ".join(str(e.get("message", e)) for e in errors) if isinstance(errors, list) else f"HTTP {status}"
                raise BrokerError(f"Upstox refused {method} {url.split('?')[0]}: {msg}", retryable=status >= 500)
            return payload if isinstance(payload, dict) else {}
        raise BrokerError(f"{method} {url}: rate limited", retryable=True)

    # ---------------------------------------------------------------- Broker
    def place_order(self, req: OrderRequest) -> List[str]:
        if not valid_instrument_key(req.instrument_key) or not req.instrument_key.split("|")[0].endswith("_FO"):
            raise BrokerError(f"refusing to trade {req.instrument_key!r}: not an F&O instrument key")
        body = {"quantity": int(req.quantity), "product": req.product, "validity": req.validity,
                "price": round(float(req.price), 2) if req.order_type in ("LIMIT", "SL") else 0,
                "tag": req.tag[:20], "instrument_token": req.instrument_key, "order_type": req.order_type,
                "transaction_type": req.side, "disclosed_quantity": 0,
                "trigger_price": round(float(req.trigger_price), 2) if req.order_type in ("SL", "SL-M") else 0,
                "is_amo": False, "slice": True}
        if req.order_type in ("MARKET", "SL-M"):
            body["market_protection"] = -1   # exchange-guideline protection (0 would be rejected)
        payload = self._call("POST", f"{HFT_BASE}/v3/order/place", body=body, retry=False)
        ids = (payload.get("data") or {}).get("order_ids") or []
        if not ids and (payload.get("data") or {}).get("order_id"):
            ids = [payload["data"]["order_id"]]
        if not ids:
            raise BrokerError("Upstox accepted the request but returned no order id", maybe_sent=True)
        return [str(i) for i in ids]

    def modify_order(self, order_id, *, quantity=None, price=None, trigger_price=None, order_type=None):
        current = self.order_status(order_id)
        if current is None:
            raise BrokerError(f"order {order_id} not found")
        body = {"order_id": order_id, "validity": "DAY", "disclosed_quantity": 0,
                "quantity": int(quantity if quantity is not None else current.quantity),
                "order_type": order_type or "LIMIT",
                "price": round(float(price), 2) if price is not None else 0,
                "trigger_price": round(float(trigger_price), 2) if trigger_price is not None else 0}
        self._call("PUT", f"{HFT_BASE}/v3/order/modify", body=body)

    def cancel_order(self, order_id):
        self._call("DELETE", f"{HFT_BASE}/v3/order/cancel", params={"order_id": order_id})

    def order_status(self, order_id="", tag="") -> Optional[OrderStatus]:
        params = {"order_id": order_id} if order_id else {"tag": tag}
        try:
            payload = self._call("GET", f"{API_BASE}/v2/order/details", params=params)
        except BrokerError as error:
            if "not found" in str(error).lower() or "UDAPI100059" in str(error):
                return None
            raise
        d = payload.get("data") or {}
        if isinstance(d, list):
            d = d[-1] if d else {}
        if not d:
            return None
        raw = str(d.get("status", "")).lower()
        return OrderStatus(order_id=str(d.get("order_id", order_id)), status=STATUS_MAP.get(raw, UNKNOWN),
                           quantity=int(fnum(d.get("quantity"), 0)), filled_quantity=int(fnum(d.get("filled_quantity"), 0)),
                           average_price=fnum(d.get("average_price"), 0.0),
                           pending_quantity=int(fnum(d.get("pending_quantity"), 0)),
                           message=str(d.get("status_message") or ""), tag=str(d.get("tag") or ""), raw_status=raw)

    def positions(self) -> List[BrokerPosition]:
        payload = self._call("GET", f"{API_BASE}/v2/portfolio/short-term-positions")
        out = []
        for p in payload.get("data") or []:
            key = p.get("instrument_token") or p.get("instrument_key")
            if not key:
                continue
            out.append(BrokerPosition(instrument_key=key, quantity=int(fnum(p.get("quantity"), 0)),
                                      average_price=fnum(p.get("average_price"), 0.0),
                                      last_price=fnum(p.get("last_price"), 0.0), pnl=fnum(p.get("pnl"), 0.0)))
        return out

    def funds(self) -> Funds:
        payload = self._call("GET", f"{API_BASE}/v2/user/get-funds-and-margin", params={"segment": "SEC"})
        data = payload.get("data") or {}
        eq = data.get("equity") or data
        return Funds(available_margin=fnum(eq.get("available_margin"), 0.0), used_margin=fnum(eq.get("used_margin"), 0.0),
                     raw={k: eq.get(k) for k in ("available_margin", "used_margin")})
