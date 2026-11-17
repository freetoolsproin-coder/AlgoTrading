"""
Flask app of the NIFTY 50 stock-options service.

    GET  /                        signals page: CE / PE plan per stock, paper book, search bar
    GET  /report/<SYMBOL>         research report page for one stock
    GET  /api/state               every plan + paper book + status
    GET  /api/plan/<SYMBOL>       the latest plan of one stock
    GET  /api/search?q=rel        search suggestions (NIFTY 50 first, then any NSE equity in live mode)
    GET  /api/report/<SYMBOL>     research report as JSON
    POST /api/refresh             run a cycle now (this computer only, or X-API-Token = MOVEMENT_API_TOKEN)
    GET  /health
"""
from __future__ import annotations

import hmac
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from src.config.settings import get_secret
from src.stocks.research import build_report

STATIC = Path(__file__).resolve().parent / "static"
LOCAL = ("127.0.0.1", "::1", "localhost")
SYMBOL_RE = re.compile(r"^[A-Z0-9&\-]{1,20}$")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _json(payload, status: int = 200) -> Response:
    return Response(json.dumps(_clean(payload), default=_default), status=status, mimetype="application/json")


def create_app(engine, api_token=None) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    source = engine.source
    token = api_token if api_token is not None else (get_secret("MOVEMENT_API_TOKEN") or "")

    def guard():
        if token:
            if not hmac.compare_digest(request.headers.get("X-API-Token", ""), token):
                abort(401, "X-API-Token missing or wrong")
        elif request.remote_addr not in LOCAL:
            abort(403, "accepted from this computer only")

    def symbol_or_404(symbol: str) -> str:
        symbol = symbol.strip().upper()
        if not SYMBOL_RE.match(symbol):
            abort(404, "not a valid NSE symbol")
        return symbol

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    def error(e):
        return jsonify(error=getattr(e, "description", str(e))), e.code

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/report/<symbol>")
    def report_page(symbol):
        symbol_or_404(symbol)
        return send_from_directory(STATIC, "report.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", engine=engine.status, generated_at=engine.generated_at, cycles=engine.cycles)

    @app.get("/api/state")
    def state():
        return _json(engine.state())

    @app.get("/api/plan/<symbol>")
    def plan(symbol):
        p = engine.plan_for(symbol_or_404(symbol))
        if p is None:
            abort(404, "no plan for that symbol yet")
        return _json(p)

    @app.get("/api/search")
    def search():
        q = request.args.get("q", "").strip().upper()
        stocks = source.stocks()
        if not q:
            return _json([s.to_dict() for s in stocks])
        hits = [s for s in stocks if s.symbol.startswith(q)] + \
               [s for s in stocks if not s.symbol.startswith(q) and (q in s.symbol or q in s.name.upper())]
        out = [s.to_dict() for s in hits]
        if len(out) < 12 and not source.mock and len(q) >= 2:
            known = {s.symbol for s in stocks}
            try:
                extra = [x for x in source.all_symbols() if x.startswith(q) and x not in known]
            except Exception:
                extra = []
            out += [{"symbol": x, "name": x, "industry": "", "nifty50": False} for x in extra[:12 - len(out)]]
        return _json(out[:12])

    @app.get("/api/report/<symbol>")
    def report(symbol):
        symbol = symbol_or_404(symbol)
        try:
            r = build_report(source, engine, symbol, use_cache=request.args.get("fresh") != "1")
        except Exception as e:
            if "Auth" in type(e).__name__:
                abort(401, "Upstox token missing or expired: run python -m src.broker.auth")
            raise
        if r is None:
            abort(404, f"{symbol} not found - check the NSE symbol")
        return _json(r)

    @app.post("/api/refresh")
    def refresh():
        guard()
        engine.poke()
        return jsonify(ok=True)

    return app
