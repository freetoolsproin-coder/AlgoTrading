"""
Flask app of the equity service (the trading desk on :8500 proxies /api/eq/* to these routes).

    GET  /api/state                     everything both dashboard tabs show (signals, scanners, positions, risk, alerts)
    GET  /api/chart/<SYMBOL>?tf=5m      candles + indicator lines + entry / stop / target levels (1m 5m 15m 30m 1h 4h 1d 1w)
    GET  /api/strategies                the 50 strategies with their rules
    GET  /api/signals?module=&day=      stored signals of one day
    GET  /api/backtest/<module>         the latest full backtest result
    POST /api/backtest/run              {"module": "swing", "years": 3}   (runs in the background)
    POST /api/mode                      {"mode": "analysis" | "paper" | "live"}   (live only when armed for today)
    POST /api/kill                      {"on": true | false}              emergency kill switch
    POST /api/risk                      {"risk_per_trade_pct": 0.5, ...}  risk settings
    POST /api/orders/<id>/confirm       send one proposed LIVE order      POST /api/orders/<id>/reject
    POST /api/positions/<id>/close      close (or cancel) one position
    POST /api/refresh                   scan now
    GET  /health

Every POST is accepted from this computer only, or with X-API-Token = MOVEMENT_API_TOKEN when that is set.
"""
from __future__ import annotations

import hmac
import re

from flask import Flask, abort, jsonify, request

from src.config.settings import get_secret
from src.equity import strategies as strat
from src.stocks.web import _json

LOCAL = ("127.0.0.1", "::1", "localhost")
SYMBOL_RE = re.compile(r"^[A-Z0-9&\-]{1,20}$")
TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w")


def create_app(engine, api_token=None) -> Flask:
    app = Flask(__name__)
    token = api_token if api_token is not None else (get_secret("MOVEMENT_API_TOKEN") or "")

    def guard():
        if token:
            if not hmac.compare_digest(request.headers.get("X-API-Token", ""), token):
                abort(401, "X-API-Token missing or wrong")
        elif request.remote_addr not in LOCAL:
            abort(403, "accepted from this computer only")

    def body() -> dict:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    def error(e):
        return jsonify(ok=False, error=getattr(e, "description", str(e))), e.code

    @app.get("/health")
    def health():
        return jsonify(status="ok", engine=engine.status, mode=engine.trader.mode, scans=engine.scans, cycles=engine.cycles)

    @app.get("/api/state")
    def state():
        return _json(engine.state())

    @app.get("/api/chart/<symbol>")
    def chart(symbol):
        symbol = symbol.strip().upper()
        tf = request.args.get("tf", "5m")
        if not SYMBOL_RE.match(symbol) or tf not in TIMEFRAMES:
            abort(404, "unknown symbol or timeframe")
        out = engine.chart(symbol, tf)
        if out is None:
            abort(404, "no chart for that symbol")
        return _json(out)

    @app.get("/api/strategies")
    def strategies():
        return _json({"intraday": engine._strategy_table("intraday"), "swing": engine._strategy_table("swing")})

    @app.get("/api/signals")
    def signals():
        module = request.args.get("module", "intraday")
        if module not in ("intraday", "swing"):
            abort(400, "module must be intraday or swing")
        day = request.args.get("day") or engine.clock().date().isoformat()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", day):
            abort(400, "day must be YYYY-MM-DD")
        return _json(engine.store.signals(module, day=day))

    @app.get("/api/backtest/<module>")
    def backtest(module):
        if module not in strat.REGISTRY:
            abort(404, "module must be intraday or swing")
        row = engine.store.latest_backtest(module)
        if not row:
            abort(404, "no backtest has been run for this module yet")
        return _json({"created": row["created"], "params": row["params"], **row["result"]})

    @app.post("/api/backtest/run")
    def backtest_run():
        guard()
        b = body()
        module = b.get("module")
        if module not in strat.REGISTRY:
            abort(400, "module must be intraday or swing")
        years = b.get("years")
        if years is not None:
            try:
                years = min(5.0, max(0.25, float(years)))
            except (TypeError, ValueError):
                abort(400, "years must be a number")
        if not engine.run_backtest(module, years):
            abort(409, "a backtest is already running")
        return jsonify(ok=True)

    @app.post("/api/mode")
    def mode():
        guard()
        ok, msg = engine.trader.set_mode(str(body().get("mode", "")), engine.clock().date())
        if not ok:
            abort(409, msg)
        return jsonify(ok=True, mode=msg)

    @app.post("/api/kill")
    def kill():
        guard()
        engine.trader.kill(bool(body().get("on", True)), engine.clock())
        return jsonify(ok=True, kill_switch=engine.risk.killed())

    @app.post("/api/risk")
    def risk():
        guard()
        try:
            return _json({"ok": True, "risk": engine.update_risk(body())})
        except (TypeError, ValueError) as e:
            abort(400, str(e))

    @app.post("/api/orders/<int:order_id>/<action>")
    def order(order_id, action):
        guard()
        if action not in ("confirm", "reject"):
            abort(404, "unknown action")
        if action == "confirm" and body().get("confirm") is not True:
            abort(400, 'send {"confirm": true} to place this live order')
        fn = engine.trader.confirm if action == "confirm" else engine.trader.reject
        ok, msg = fn(order_id, engine.clock())
        if not ok:
            abort(409, msg)
        return jsonify(ok=True, message=msg)

    @app.post("/api/positions/<int:position_id>/close")
    def close(position_id):
        guard()
        ok, msg = engine.trader.close(position_id, engine.clock())
        if not ok:
            abort(409, msg)
        return jsonify(ok=True, message=msg)

    @app.post("/api/refresh")
    def refresh():
        guard()
        engine.poke()
        return jsonify(ok=True)

    return app
