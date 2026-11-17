"""
Read-only broker monitor (Upstox). It sends GET requests only - a guard refuses any other method - so this module
can never place, modify or cancel an order. Answers are cached for [CONSOLE] broker_cache_seconds to respect the
broker's rate limits.

    profile    GET /v2/user/profile
    funds      GET /v2/user/get-funds-and-margin?segment=SEC
    positions  GET /v2/portfolio/short-term-positions
    orders     GET /v2/order/retrieve-all            (today's order book)
    trades     GET /v2/order/trades/get-trades-for-day
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.broker.client import TokenSource, UpstoxAuthError
from src.config.settings import get_secret

logger = logging.getLogger("scanner.console")
API = "https://api.upstox.com"
IST = timezone(timedelta(hours=5, minutes=30))
ENDPOINTS = {
    "profile": ("/v2/user/profile", None),
    "funds": ("/v2/user/get-funds-and-margin", {"segment": "SEC"}),
    "positions": ("/v2/portfolio/short-term-positions", None),
    "orders": ("/v2/order/retrieve-all", None),
    "trades": ("/v2/order/trades/get-trades-for-day", None),
}


class ReadOnlyViolation(RuntimeError):
    pass


class _GetOnlySession:
    """Wraps a requests-like session and refuses anything but GET."""

    def __init__(self, session):
        self._s = session

    def get(self, url, **kw):
        return self._s.get(url, **kw)

    def request(self, method, url, **kw):
        if str(method).upper() != "GET":
            raise ReadOnlyViolation(f"broker monitor is read-only; refused {method} {url}")
        return self._s.request("GET", url, **kw)

    def __getattr__(self, name):
        if name in ("post", "put", "delete", "patch"):
            raise ReadOnlyViolation(f"broker monitor is read-only; refused {name.upper()}")
        raise AttributeError(name)


def token_info(token_file: Path) -> dict:
    """Where the token comes from and roughly when it expires (Upstox tokens end at 03:30 IST the next day)."""
    if get_secret("UPSTOX_ACCESS_TOKEN"):
        return {"present": True, "source": "UPSTOX_ACCESS_TOKEN (.env / environment)", "issued": None,
                "expires": None, "expired": None}
    p = Path(token_file)
    if not p.exists():
        return {"present": False, "source": p.name, "issued": None, "expires": None, "expired": None}
    issued = datetime.fromtimestamp(p.stat().st_mtime, IST)
    expiry = (issued + timedelta(days=1)).replace(hour=3, minute=30, second=0, microsecond=0)
    if issued.hour < 3 or (issued.hour == 3 and issued.minute < 30):
        expiry = issued.replace(hour=3, minute=30, second=0, microsecond=0)
    return {"present": True, "source": p.name, "issued": issued.isoformat(timespec="minutes"),
            "expires": expiry.isoformat(timespec="minutes"), "expired": datetime.now(IST) >= expiry}


def _rows(res: dict) -> list:
    data = res.get("data")
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


class BrokerMonitor:
    def __init__(self, token_file: Path, cache_seconds: int = 15, session=None, timeout: float = 6.0):
        if session is None:
            import requests
            session = requests.Session()
        self.session = _GetOnlySession(session)
        self.tokens = TokenSource(token_file)
        self.token_file = Path(token_file)
        self.cache_seconds = cache_seconds
        self.timeout = timeout
        self._cache: dict = {}
        self._lock = threading.Lock()

    def _get(self, name: str) -> dict:
        path, params = ENDPOINTS[name]
        started = time.monotonic()
        try:
            token = self.tokens.token()
        except UpstoxAuthError as error:
            return {"ok": False, "error": "no_token", "message": str(error), "latency_ms": None}
        try:
            r = self.session.request("GET", f"{API}{path}", params=params, timeout=self.timeout,
                                     headers={"Accept": "application/json", "Authorization": f"Bearer {token}"})
        except ReadOnlyViolation:
            raise
        except Exception as error:   # noqa: BLE001 - network problems are reported, never raised
            return {"ok": False, "error": "network", "message": f"{type(error).__name__}", "latency_ms": None}
        latency = round((time.monotonic() - started) * 1000)
        status = getattr(r, "status_code", 200)
        try:
            payload = r.json()
        except ValueError:
            payload = {}
        if status == 401:
            return {"ok": False, "error": "unauthorized", "latency_ms": latency,
                    "message": "Upstox rejected the token (401): run the daily login (python -m src.broker.auth)."}
        if status >= 400 or (isinstance(payload, dict) and payload.get("status") == "error"):
            errs = payload.get("errors") if isinstance(payload, dict) else None
            msg = "; ".join(str(e.get("message", e)) for e in errs)[:200] if isinstance(errs, list) else f"HTTP {status}"
            return {"ok": False, "error": "http", "message": msg, "latency_ms": latency}
        return {"ok": True, "data": (payload or {}).get("data"), "latency_ms": latency}

    def fetch(self, name: str, force: bool = False) -> dict:
        with self._lock:
            hit = self._cache.get(name)
            if hit and not force and time.monotonic() - hit[0] < self.cache_seconds:
                return hit[1]
        res = self._get(name)
        res["fetched_at"] = datetime.now(IST).isoformat(timespec="seconds")
        with self._lock:
            self._cache[name] = (time.monotonic(), res)
        return res

    def snapshot(self, force: bool = False) -> dict:
        tok = token_info(self.token_file)
        if not tok["present"]:
            return {"connected": False, "status": "no_token", "token": tok,
                    "message": "No Upstox token. Run the daily login: python -m src.broker.auth"}
        profile = self.fetch("profile", force)
        if not profile["ok"]:
            return {"connected": False, "status": profile["error"], "message": profile["message"], "token": tok}
        out = {"connected": True, "status": "ok", "token": tok, "latency_ms": profile.get("latency_ms"),
               "fetched_at": profile.get("fetched_at")}
        p = profile.get("data") or {}
        out["profile"] = {"user_id": p.get("user_id"), "user_name": p.get("user_name"), "broker": p.get("broker"),
                          "exchanges": p.get("exchanges"), "products": p.get("products"), "is_active": p.get("is_active")}
        funds = self.fetch("funds", force)
        eq = ((funds.get("data") or {}).get("equity") or funds.get("data") or {}) if funds["ok"] else {}
        out["funds"] = {"ok": funds["ok"], "message": funds.get("message", ""),
                        "available_margin": eq.get("available_margin"), "used_margin": eq.get("used_margin"),
                        "payin_amount": eq.get("payin_amount")}
        pos = self.fetch("positions", force)
        out["positions"] = {"ok": pos["ok"], "message": pos.get("message", ""), "rows": [
            {"symbol": x.get("trading_symbol") or x.get("tradingsymbol"), "instrument_key": x.get("instrument_token"),
             "product": x.get("product"), "quantity": x.get("quantity"), "average_price": x.get("average_price"),
             "last_price": x.get("last_price"), "pnl": x.get("pnl"), "realised": x.get("realised"),
             "unrealised": x.get("unrealised")} for x in _rows(pos)]}
        orders = self.fetch("orders", force)
        out["orders"] = {"ok": orders["ok"], "message": orders.get("message", ""), "rows": [
            {"time": x.get("order_timestamp"), "symbol": x.get("trading_symbol") or x.get("tradingsymbol"),
             "side": x.get("transaction_type"), "type": x.get("order_type"), "qty": x.get("quantity"),
             "filled": x.get("filled_quantity"), "price": x.get("price"), "avg": x.get("average_price"),
             "status": x.get("status"), "tag": x.get("tag"), "message": x.get("status_message"),
             "order_id": x.get("order_id")} for x in _rows(orders)]}
        trades = self.fetch("trades", force)
        out["trades"] = {"ok": trades["ok"], "message": trades.get("message", ""), "rows": [
            {"time": x.get("exchange_timestamp") or x.get("order_timestamp"),
             "symbol": x.get("trading_symbol") or x.get("tradingsymbol"), "side": x.get("transaction_type"),
             "qty": x.get("quantity"), "price": x.get("average_price"), "order_id": x.get("order_id"),
             "trade_id": x.get("trade_id")} for x in _rows(trades)]}
        return out
