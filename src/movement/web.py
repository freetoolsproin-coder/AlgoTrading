"""
Flask web app of the movement service.

    GET  /                      live page (metrics per window, trigger feed, rule editor)
    GET  /api/state             latest metrics for every index + India VIX
    GET  /api/events            stored triggers   ?limit=200&day=YYYY-MM-DD&index=NIFTY
    GET  /api/stream            Server-Sent Events: "state" every cycle, "movement" for each trigger
    GET  /api/config            current rules (ini defaults + saved overrides)
    PUT  /api/config            change rules, JSON body e.g. {"move_pct": "5:0.2, 10:0.3, 15:0.4"}
    POST /api/config/reset      back to the config/scanner.ini values
    POST /api/evaluate          run a cycle now
    POST /api/test-alert        send a test message through Telegram / popup / webhook
    GET  /api/paper             paper books: open and closed trades today, P&L, limits, skipped entries
    GET  /api/paper/history     stored paper trades   ?day=YYYY-MM-DD&book=scalp
    POST /api/paper/close/<id>  close one open paper trade now
    GET  /strategies            strategy library page (rules, live results per index)
    GET  /api/strategies        every strategy: category, CE / PE rule, data, timeframe, exits, latest result
    GET  /ml                    machine-learning page (train, test-period report, live probabilities)
    GET  /api/ml/models         trained ML signal models with their train / test reports
    POST /api/ml/train          train a model in the background, JSON body e.g.
                                {"index": "NIFTY", "timeframe": 5, "horizon": 6, "min_move_atr": 1.0, "holdout": 0.3}
    GET  /api/ml/jobs/<id>      progress / result of a training job
    GET  /api/ml/predict        the model's latest P(up) / P(down) per index (from the running strategy engine)
    GET  /health

Changing endpoints (PUT / POST) are accepted from this computer only, unless MOVEMENT_API_TOKEN is set in
.env: then every caller must send it in the X-API-Token header.
"""
from __future__ import annotations

import hmac
import json
import queue
import threading
import time as clock
import uuid
from datetime import date
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from src.config.settings import get_secret
from src.movement.monitor import MovementMonitor, to_alert
from src.scanner.store import json_default

STATIC = Path(__file__).resolve().parent / "static"
LOCAL = ("127.0.0.1", "::1", "localhost")
HEARTBEAT_SECONDS = 15


def _json(payload, status: int = 200) -> Response:
    return Response(json.dumps(payload, default=json_default), status=status, mimetype="application/json")


def create_app(monitor: MovementMonitor, api_token: str | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    app.config["JSON_SORT_KEYS"] = False
    token = api_token if api_token is not None else (get_secret("MOVEMENT_API_TOKEN") or "")

    def guard():
        if token:
            if not hmac.compare_digest(request.headers.get("X-API-Token", ""), token):
                abort(401, "X-API-Token missing or wrong")
        elif request.remote_addr not in LOCAL:
            abort(403, "changes are accepted from this computer only (set MOVEMENT_API_TOKEN to allow others)")

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    def error(e):
        return jsonify(error=getattr(e, "description", str(e))), e.code

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/health")
    def health():
        s = monitor.snapshot_state()
        return jsonify(status="ok", monitor=s.get("status"), generated_at=s.get("generated_at"),
                       cycles=monitor.cycles, stream_clients=monitor.broadcaster.clients)

    @app.get("/api/state")
    def state():
        return _json(monitor.snapshot_state())

    @app.get("/api/events")
    def events():
        limit = min(max(request.args.get("limit", 200, type=int), 1), 2000)
        return _json(monitor.store.recent(limit, mock=monitor.mock, day=request.args.get("day"),
                                          index=request.args.get("index")))

    @app.get("/api/config")
    def get_config():
        rs = monitor.rules_store
        return _json({"rules": rs.rules.to_dict(), "overrides": sorted(rs.overrides)})

    @app.put("/api/config")
    def put_config():
        guard()
        changes = request.get_json(silent=True)
        if not isinstance(changes, dict) or not changes:
            abort(400, "send a JSON object of settings to change")
        try:
            rules = monitor.rules_store.update(changes)
        except (ValueError, TypeError) as e:
            abort(400, str(e))
        monitor.poke()
        return _json({"rules": rules.to_dict(), "overrides": sorted(monitor.rules_store.overrides)})

    @app.post("/api/config/reset")
    def reset_config():
        guard()
        rules = monitor.rules_store.reset()
        monitor.poke()
        return _json({"rules": rules.to_dict(), "overrides": []})

    @app.post("/api/evaluate")
    def evaluate():
        guard()
        monitor.poke()
        return jsonify(ok=True)

    @app.post("/api/test-alert")
    def test_alert():
        guard()
        event = {"index": "TEST", "window": "5", "direction": "UP", "quality": "CONFIRMED",
                 "time": monitor.snapshot_state().get("generated_at") or "", "price": None, "change": 0.0,
                 "change_pct": 0.0, "checks": {}, "mock": monitor.mock,
                 "message": "Test message from the movement service - not a market event"}
        if monitor.notifier is not None:
            monitor.notifier.emit(to_alert(event), push=True)
        monitor.webhook.send(event)
        channels = monitor.notifier.channels() if monitor.notifier is not None else []
        return jsonify(ok=True, channels=channels + (["webhook"] if monitor.webhook.url else []))

    @app.get("/api/paper")
    def paper():
        if monitor.trader is None:
            return _json({})
        return _json(monitor.trader.summary(monitor.provider.now()))

    @app.get("/api/paper/history")
    def paper_history():
        return _json(monitor.store.trades(day=request.args.get("day"), book=request.args.get("book"),
                                          mock=monitor.mock, limit=min(request.args.get("limit", 500, type=int), 5000)))

    @app.post("/api/paper/close/<path:trade_id>")
    def paper_close(trade_id):
        guard()
        closed = monitor.close_trade(trade_id)
        if closed is None:
            abort(404, "no open paper trade with that id")
        return _json(closed)

    # ------------------------------------------------------------------ strategy library + machine learning
    jobs: dict = {}
    jobs_lock = threading.Lock()

    @app.get("/strategies")
    def strategies_page():
        return send_from_directory(STATIC, "strategies.html")

    @app.get("/ml")
    def ml_page():
        return send_from_directory(STATIC, "ml.html")

    @app.get("/api/strategies")
    def strategies_api():
        from src.movement import strategies as strategies_mod
        engine = monitor.strategies
        rows = strategies_mod.catalogue(getattr(monitor, "config_path", None))
        last = engine.last if engine is not None else {}
        for r in rows:
            r["latest"] = last.get(r["key"], {})
            if engine is not None and r["key"] in engine.tfs:
                r["timeframe"], r["enabled"] = engine.tfs[r["key"]], True
            elif engine is not None:
                r["enabled"] = False
        return _json({"running": engine is not None, "count": len(rows), "strategies": rows})

    @app.get("/api/ml/models")
    def ml_models():
        from src.movement import ml_signal
        return _json({"models": ml_signal.list_models()})

    @app.post("/api/ml/train")
    def ml_train():
        guard()
        from src.config.instruments import SUPPORTED_INDICES
        from src.movement import ml_signal
        body = request.get_json(silent=True) or {}
        try:
            index = str(body.get("index", "NIFTY")).upper()
            if index not in SUPPORTED_INDICES:
                raise ValueError(f"index must be one of {', '.join(SUPPORTED_INDICES)}")
            args = dict(tf=int(body.get("timeframe", 5)), horizon=int(body.get("horizon", 6)),
                        min_move_atr=float(body.get("min_move_atr", 1.0)), holdout=float(body.get("holdout", 0.3)),
                        signal_rate=float(body.get("signal_rate", 0.05)))
            start = date.fromisoformat(str(body.get("start", "2023-01-01")))
            end = date.fromisoformat(str(body.get("end", date.today().isoformat())))
            if args["tf"] not in (1, 3, 5, 10, 15, 30, 60) or not 1 <= args["horizon"] <= 60:
                raise ValueError("timeframe must be 1/3/5/10/15/30/60 and horizon 1-60 bars")
            if not 0 < args["min_move_atr"] <= 10 or not 0 <= args["holdout"] < 0.9 \
                    or not 0 < args["signal_rate"] < 0.5:
                raise ValueError("min_move_atr 0-10, holdout 0-0.9, signal_rate 0-0.5")
        except (TypeError, ValueError) as e:
            abort(400, str(e))
        with jobs_lock:
            if any(j["status"] == "running" for j in jobs.values()):
                abort(409, "a training job is already running")
            job_id = uuid.uuid4().hex[:12]
            jobs[job_id] = {"id": job_id, "status": "running", "index": index, "params": args,
                            "started": clock.time()}

        def work():
            try:
                minutes = ml_signal.load_history(index, start, end)
                if minutes.empty:
                    raise ValueError(f"no {index} 1-minute history for {start}..{end} in data/history or orb_data")
                model = ml_signal.train(minutes, index, **args)
                result = {"status": "done", "report": ml_signal.format_report(model),
                          "model": {k: model[k] for k in ("index", "timeframe", "train_end", "test_start",
                                                         "test_end", "report", "thresholds", "top_features")}}
            except Exception as error:              # reported to the page, never kills the service
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            with jobs_lock:
                jobs[job_id].update(result, finished=clock.time())

        threading.Thread(target=work, name=f"ml-train-{job_id}", daemon=True).start()
        return _json(jobs[job_id], status=202)

    @app.get("/api/ml/jobs/<job_id>")
    def ml_job(job_id):
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                abort(404, "no such job")
            return _json(dict(job))

    @app.get("/api/ml/predict")
    def ml_predict():
        engine = monitor.strategies
        if engine is None or "ml_signal" not in engine.enabled:
            return _json({"enabled": False, "indices": {},
                          "note": "ml_signal is not enabled in [STRATEGIES] (or no strategies run)"})
        return _json({"enabled": True, "timeframe": engine.tfs["ml_signal"],
                      "indices": engine.last.get("ml_signal", {})})

    @app.get("/api/stream")
    def stream():
        q = monitor.broadcaster.subscribe()
        if q is None:
            abort(503)

        def generate():
            try:
                yield "retry: 3000\n\n"
                yield (f"event: state\ndata: {json.dumps(monitor.snapshot_state(), default=json_default)}\n\n")
                last = clock.monotonic()
                while True:
                    try:
                        yield q.get(timeout=1.0)
                        last = clock.monotonic()
                    except queue.Empty:
                        if clock.monotonic() - last >= HEARTBEAT_SECONDS:
                            last = clock.monotonic()
                            yield ": keep-alive\n\n"
            finally:
                monitor.broadcaster.unsubscribe(q)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    return app
