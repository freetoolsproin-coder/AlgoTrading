"""
Flask app of the F&O engine (dashboard + JSON API). It reads the engine's state; it never places orders.

    GET  /                          dashboard
    GET  /api/state                 watchlist, index cards, signals, open positions, risk usage, health
    GET  /api/stream                Server-Sent Events: a state summary every few seconds, each signal as it happens
    GET  /api/trades                paper trade history
    GET  /api/performance           overall + by strategy / symbol / CE-PE / time of day / exit, equity curve
    GET  /api/strategies            the 43 strategies with rules, settings and signal counts
    POST /api/strategies/<key>      {"enabled": bool, "<param>": value}   (this computer, or X-API-Token)
    GET  /api/backtests             saved backtest reports;  /api/backtests/<id> one report
    POST /api/backtests/run         {"symbols": "...", "from": "...", "to": "...", "strategies": "...", "walk_forward": bool}
    GET  /api/backtests/job         status of the running backtest
    POST /api/kill | /api/kill/release | /api/positions/<id>/close
    GET  /health
"""
from __future__ import annotations

import hmac
import json
import queue
import threading
import time as clock
from datetime import date, timedelta
from pathlib import Path

from flask import Flask, Response, abort, request, send_from_directory

from src.config.settings import get_secret
from src.fno import backtest as bt
from src.fno.store import dumps
from src.fno.strategies import REGISTRY

STATIC = Path(__file__).resolve().parent / "static"
LOCAL = ("127.0.0.1", "::1", "localhost")


def _json(payload, status: int = 200) -> Response:
    return Response(dumps(payload), status=status, mimetype="application/json")


def create_app(engine, api_token=None) -> Flask:
    app = Flask(__name__, static_folder=None)
    token = api_token if api_token is not None else (get_secret("MOVEMENT_API_TOKEN") or "")
    subscribers: list = []
    job = {"state": "idle", "message": "", "id": None}
    engine.listeners.append(lambda sig: [q.put(("signal", sig)) for q in list(subscribers)])

    def allowed() -> bool:
        if request.remote_addr in LOCAL:
            return True
        sent = request.headers.get("X-API-Token", "")
        return bool(token) and hmac.compare_digest(sent, token)

    def guard():
        if not allowed():
            abort(403)

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'"
        return response

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    @app.get("/health")
    def health():
        s = engine.state()
        return _json({"status": s.get("status"), "time": s.get("time"), "mode": "PAPER", "mock": engine.feed.mock})

    @app.get("/api/state")
    def state():
        return _json(engine.state())

    @app.get("/api/stream")
    def stream():
        q: queue.Queue = queue.Queue(maxsize=200)
        subscribers.append(q)

        def gen():
            try:
                last = None
                while True:
                    try:
                        kind, payload = q.get(timeout=3)
                        yield f"event: {kind}\ndata: {dumps(payload)}\n\n"
                    except queue.Empty:
                        s = engine.state()
                        if s.get("cycle") != last:
                            last = s.get("cycle")
                            yield f"event: cycle\ndata: {json.dumps({'cycle': last, 'time': s.get('time')})}\n\n"
                        else:
                            yield ": keep-alive\n\n"
            finally:
                if q in subscribers:
                    subscribers.remove(q)
        return Response(gen(), mimetype="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.get("/api/trades")
    def trades():
        return _json(engine.store.trades(limit=int(request.args.get("limit", 500))))

    @app.get("/api/signals")
    def signals():
        return _json(engine.store.signals(request.args.get("day"), int(request.args.get("limit", 300))))

    @app.get("/api/performance")
    def performance():
        return _json(engine.performance())

    @app.get("/api/strategies")
    def strategies():
        return _json(engine.strategies())

    @app.post("/api/strategies/<key>")
    def set_strategy(key):
        guard()
        s = REGISTRY.get(key)
        if s is None:
            abort(404)
        body = request.get_json(silent=True) or {}
        values = {}
        if "enabled" in body:
            values["enabled"] = "true" if body["enabled"] in (True, "true", 1, "1") else "false"
        for name, default in s.defaults.items():
            if name in body:
                try:
                    values[name] = type(default)(float(body[name]))
                except (TypeError, ValueError):
                    return _json({"error": f"{name} must be a number"}, 400)
        if not values:
            return _json({"error": "nothing to change"}, 400)
        with engine._lock:
            engine.cfg.save_strategy_override(key, values)
        return _json(s.describe(engine.cfg.strategies.get(key)))

    @app.get("/api/backtests")
    def backtests():
        return _json({"reports": bt.list_reports(), "job": job})

    @app.get("/api/backtests/job")
    def backtest_job():
        return _json(job)

    @app.get("/api/backtests/<run_id>")
    def backtest(run_id):
        report = bt.load_report(run_id)
        if report is None:
            abort(404)
        return _json(report)

    @app.post("/api/backtests/run")
    def backtest_run():
        guard()
        if job["state"] == "running":
            return _json({"error": "a backtest is already running"}, 409)
        body = request.get_json(silent=True) or {}
        try:
            end = date.fromisoformat(body["to"]) if body.get("to") else date.today() - timedelta(days=1)
            start = date.fromisoformat(body["from"]) if body.get("from") else end - timedelta(days=60)
        except ValueError:
            return _json({"error": "dates must be YYYY-MM-DD"}, 400)
        symbols = [s.strip().upper() for s in str(body.get("symbols") or "NIFTY").split(",") if s.strip()][:25]
        keys = None if str(body.get("strategies") or "all") == "all" else \
            [k for k in str(body["strategies"]).lower().split(",") if k.strip() in REGISTRY]

        def work():
            job.update(state="running", message="starting", id=None)
            try:
                uni = engine.feed.universe()
                unds = [uni.get(s) for s in symbols if uni.get(s)]
                if not unds:
                    raise ValueError("none of the symbols is in the F&O list")
                minutes = client = None
                if engine.feed.mock:
                    minutes = {u.symbol: engine.feed._full(u.symbol) for u in unds}
                    minutes.setdefault(engine.cfg.benchmark, engine.feed._full(engine.cfg.benchmark))
                    s_, e_ = min(m.index[0].date() for m in minutes.values()), engine.feed.now().date()
                else:
                    s_, e_ = start, end
                    try:
                        engine.feed.client.tokens.token()
                        client = engine.feed.client
                    except Exception:
                        client = None
                report = bt.run(engine.cfg, unds, s_, e_, keys, "model" if client is None else "auto",
                                bool(body.get("walk_forward")), client, client is not None, minutes,
                                engine.store, lambda m: job.update(message=m))
                job.update(state="done", message=f"{report['overall']['trades']} trades", id=report["id"])
            except Exception as error:
                job.update(state="failed", message=f"{type(error).__name__}: {error}"[:300])
        threading.Thread(target=work, name="fno-backtest", daemon=True).start()
        return _json({"started": True}, 202)

    @app.post("/api/kill")
    def kill():
        engine.risk.kill("engaged from the dashboard")      # anyone who can see the page may stop trading
        return _json({"kill": engine.risk.killed()})

    @app.post("/api/kill/release")
    def release():
        guard()
        body = request.get_json(silent=True) or {}
        if body.get("confirm") != "RELEASE":
            return _json({"error": 'send {"confirm": "RELEASE"}'}, 400)
        engine.risk.release()
        return _json({"kill": engine.risk.killed()})

    @app.post("/api/positions/<pos_id>/close")
    def close(pos_id):
        guard()
        return _json({"closed": engine.close_position(pos_id)})

    return app
