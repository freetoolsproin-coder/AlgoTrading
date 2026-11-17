"""
Trading console web application (Flask). Created by `create_app()`; served by `python -m src.console`.

Security model
  * Every page except /login, /healthz and static files needs a signed-in user. Roles: admin (everything) and
    viewer (read-only, plus the two SAFE actions: engage the kill switch, disarm live trading).
  * Passwords: PBKDF2-SHA256 (600k iterations); optional TOTP second factor; per-user lockout after repeated
    failures and a per-IP throttle; the session is rotated at sign-in and expires after [CONSOLE] session_minutes
    of inactivity.
  * Every POST carries a CSRF token. Cookies are HttpOnly, SameSite=Strict (Secure with secure_cookies = true).
  * Strict Content-Security-Policy (no inline scripts), frame-ancestors none, nosniff, no-referrer.
  * Every sign-in, setting change, safety action, credential change and download is written to the hash-chained
    audit log.

Trading safety
  * No route places, modifies or cancels an order. The broker is read with GET requests only.
  * Live trading can be DISARMED here; ARMING stays a terminal command with a typed confirmation.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import secrets
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template, request, send_file, session,
                   url_for)

from src.config import settings
from src.console import analytics, data as data_mod, reports, security
from src.console.broker_monitor import BrokerMonitor, token_info
from src.console.config import ConsoleConfig, load_console_config
from src.console.health import fmt_age, system_health, age_seconds
from src.console.jobs import BACKTEST_KINDS, KINDS, JobError, JobRunner
from src.console import ml as ml_mod
from src.console.assistant import Assistant, AssistantError
from src.console.store import ConsoleStore, now_iso
from src.console.vault import MANAGED_KEYS, Vault, VaultError

logger = logging.getLogger("scanner.console")
IST = timezone(timedelta(hours=5, minutes=30))
HERE = Path(__file__).resolve().parent

DOC_PAGES = {
    "getting-started": "Getting started",
    "data-sources": "Live, paper and backtest data",
    "fees-and-slippage": "Fees and slippage",
    "risk-controls": "Risk controls",
    "live-trading": "Live trading safeguards",
    "security": "Security and credentials",
    "operations": "Daily operations",
    "ai-and-ml": "ML filter and AI assistant",
    "api": "JSON API",
}


# ============================================================================= formatting
def inr(value, decimals: int = 0, sign: bool = False) -> str:
    """Indian digit grouping: 12,34,567."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    neg = v < 0
    v = abs(v)
    whole, frac = f"{v:.{decimals}f}".split(".") if decimals else (f"{v:.0f}", "")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    s = whole + (f".{frac}" if frac else "")
    prefix = "−" if neg else ("+" if sign and v > 0 else "")
    return f"{prefix}₹{s}"


def num(value, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def pnl_class(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return "pos" if v > 0 else "neg" if v < 0 else ""


def short_time(iso) -> str:
    if not iso:
        return "—"
    s = str(iso)
    return s[:16].replace("T", " ")


# ============================================================================= app factory
def create_app(cfg: Optional[ConsoleConfig] = None, *, broker_session=None, ai_session=None,
               testing: bool = False) -> Flask:
    cfg = cfg or load_console_config()
    cfg.console_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"),
                static_url_path="/static")
    app.config.update(
        SECRET_KEY=_session_key(cfg.console_dir / "session.key"),
        SESSION_COOKIE_NAME="tc_session", SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=cfg.secure_cookies, PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        MAX_CONTENT_LENGTH=64 * 1024, TESTING=testing, JSON_SORT_KEYS=False)
    app.jinja_env.filters.update(inr=inr, num=num, pnl=pnl_class, t=short_time, age=lambda s: fmt_age(age_seconds(s)))

    store = ConsoleStore(cfg.console_db)
    vault = Vault(cfg.vault_file)
    monitor = BrokerMonitor(cfg.token_file, cfg.broker_cache_seconds, session=broker_session)
    jobs = JobRunner(on_finish=lambda j: store.audit(j.get("actor") or "system", "backtest.finished",
                                                     target=j["kind"], outcome=j["status"],
                                                     detail={"command": j["command"], "exit_code": j["exit_code"]}))

    def scanner_cfg():
        try:
            return settings.PBConfig.from_ini(cfg.config_file)
        except Exception:   # noqa: BLE001
            return settings.PBConfig()

    def app_settings():
        try:
            return settings.load_app_settings(cfg.config_file)
        except Exception:   # noqa: BLE001
            return settings.AppSettings()

    def make_sources(include_mock=False) -> data_mod.DataSources:
        from src.execution.costs import make_cost_fn
        a, pb = app_settings(), scanner_cfg()
        return data_mod.DataSources(cfg.scanner_db, cfg.execution_db, cfg.movement_db, cfg.output_dir, cfg.state_file,
                                    slippage_pct=pb.slippage_pct, cost_fn=make_cost_fn(a.costs), include_mock=include_mock)

    sources = make_sources()
    app.extensions["console"] = {"cfg": cfg, "store": store, "vault": vault, "monitor": monitor, "jobs": jobs,
                                 "sources": sources}

    def gate_status():
        from src.execution import live_gate
        a = app_settings()
        armed, missing = live_gate.status(a.execution, date.today(), mock=(a.mode == "mock"), path=cfg.arm_file)
        return {"armed": armed, "missing": missing, "mode": a.execution.get("mode", "paper"),
                "lane": a.execution.get("lane", "v1"), "max_live_lots": a.execution.get("max_live_lots", "1")}

    def client_ip() -> str:
        return request.remote_addr or ""

    def audit(action, target="", outcome="ok", detail=None, actor=None):
        store.audit(actor or (g.user["username"] if getattr(g, "user", None) else "anonymous"), action,
                    target=target, outcome=outcome, detail=detail, ip=client_ip())

    # ------------------------------------------------------------------ request hooks
    @app.before_request
    def load_user():
        g.user = None
        g.nonce = secrets.token_urlsafe(12)
        username = session.get("user")
        if username:
            last = session.get("seen", 0)
            if time.time() - last > cfg.session_minutes * 60:
                session.clear()
                flash("Signed out after inactivity.", "info")
                return
            u = store.user(username)
            if not u or u["disabled"]:
                session.clear()
                return
            g.user = u
            session["seen"] = time.time()
        if "csrf" not in session:
            session["csrf"] = security.new_csrf_token()
        if request.method == "POST":
            token = request.form.get("csrf") or request.headers.get("X-CSRF-Token")
            if not security.csrf_ok(session.get("csrf"), token):
                audit("security.csrf_rejected", target=request.path, outcome="denied")
                abort(400, description="The form expired or did not come from this console. Reload the page and try again.")

    @app.after_request
    def headers(resp: Response):
        if "Content-Security-Policy" not in resp.headers:
            resp.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; font-src 'self'; frame-src 'self'; frame-ancestors 'none'; form-action 'self'; "
                "base-uri 'none'; object-src 'none'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "SAMEORIGIN" if resp.headers.get("X-Report") else "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if cfg.secure_cookies:
            resp.headers["Strict-Transport-Security"] = "max-age=31536000"
        if request.path.startswith(("/api/", "/reports/")) or g.get("user"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def inject():
        return {"me": g.get("user"), "csrf": session.get("csrf", ""), "strip": status_strip() if g.get("user") else None,
                "doc_pages": DOC_PAGES, "SOURCE_LABEL": data_mod.SOURCE_LABEL, "now": datetime.now(IST)}

    _strip_cache = {"t": 0, "v": None}

    def status_strip():
        if time.time() - _strip_cache["t"] < 3 and _strip_cache["v"]:
            return _strip_cache["v"]
        st = sources.state() or {}
        gate = gate_status()
        cached = monitor._cache.get("profile")
        broker = None if not cached else ("ok" if cached[1].get("ok") else cached[1].get("error"))
        a = age_seconds(st.get("generated_at"))
        v = {"engine": "running" if a is not None and a < 120 else ("stale" if st else "not running"),
             "engine_age": fmt_age(a), "session": st.get("status") or "unknown", "mock": bool(st.get("mock")),
             "trading_mode": (st.get("execution") or {}).get("mode") or gate["mode"], "armed": gate["armed"],
             "kill": Path(cfg.kill_switch_file).exists(), "broker": broker, "market_open": st.get("market_open")}
        _strip_cache.update(t=time.time(), v=v)
        return v

    # ------------------------------------------------------------------ auth decorators
    def login_required(view):
        @functools.wraps(view)
        def wrapped(*a, **kw):
            if not g.user:
                if request.path.startswith("/api/"):
                    return jsonify({"error": "authentication required"}), 401
                return redirect(url_for("login", next=request.path))
            if not g.user["risk_ack"] and request.endpoint not in ("accept_disclosure", "disclosures", "logout"):
                return redirect(url_for("accept_disclosure"))
            if (cfg.require_2fa_for_admin and g.user["role"] == "admin" and not g.user["totp_secret"]
                    and request.endpoint not in ("account", "account_2fa", "logout", "accept_disclosure", "disclosures")):
                flash("Two-factor sign-in is required for admin accounts. Set it up below.", "warn")
                return redirect(url_for("account"))
            return view(*a, **kw)
        return wrapped

    def admin_required(view):
        @functools.wraps(view)
        @login_required
        def wrapped(*a, **kw):
            if g.user["role"] != "admin":
                audit("security.forbidden", target=request.path, outcome="denied")
                abort(403)
            return view(*a, **kw)
        return wrapped

    # ------------------------------------------------------------------ errors
    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    @app.errorhandler(500)
    def error_page(err):
        code = getattr(err, "code", 500) or 500
        if code == 500:
            logger.exception("console: unhandled error on %s", request.path)
        messages = {400: "That request could not be processed.", 403: "Your role does not allow this action.",
                    404: "There is nothing at this address.", 413: "The submitted form is too large.",
                    500: "Something failed on the server. The error is in logs/scanner.log."}
        desc = getattr(err, "description", "") if code == 400 else ""
        if request.path.startswith("/api/"):
            return jsonify({"error": messages.get(code, "error"), "detail": desc}), code
        return render_template("error.html", code=code, message=messages.get(code, "Error"), detail=desc), code

    # ------------------------------------------------------------------ public
    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "time": now_iso()})

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if g.user:
            return redirect(url_for("overview"))
        no_users = store.count_users() == 0
        pending = session.get("pending_2fa")
        if request.method == "POST" and not no_users:
            ip = client_ip()
            if store.recent_ip_failures(ip, cfg.lockout_minutes) >= cfg.max_login_failures * 4:
                audit("auth.throttled", target=ip, outcome="denied", actor="anonymous")
                flash(f"Too many failed sign-ins from this address. Try again in {cfg.lockout_minutes} minutes.", "error")
                return render_template("login.html", no_users=no_users, step="password"), 429
            if pending and request.form.get("step") == "2fa":
                u = store.user(pending.get("u", ""))
                if not u or time.time() - pending.get("t", 0) > 300:
                    session.pop("pending_2fa", None)
                    flash("The sign-in took too long. Start again.", "error")
                    return redirect(url_for("login"))
                if security.verify_totp(u["totp_secret"], request.form.get("code", "")):
                    return _finish_login(u)
                store.record_attempt(u["username"], ip, False)
                audit("auth.2fa_failed", target=u["username"], outcome="denied", actor=u["username"])
                flash("That code is not valid. Use the current 6-digit code from your authenticator app.", "error")
                return render_template("login.html", step="2fa", no_users=False), 401
            ok_name, username = security.split_username(request.form.get("username", ""))
            password = request.form.get("password", "")
            u = store.user(username) if ok_name else None
            if u and store.recent_failures(username, cfg.lockout_minutes) >= cfg.max_login_failures:
                audit("auth.locked", target=username, outcome="denied", actor="anonymous")
                flash(f"This account is locked for {cfg.lockout_minutes} minutes after repeated failed sign-ins.", "error")
                return render_template("login.html", no_users=no_users, step="password"), 429
            valid = bool(u) and not u["disabled"] and security.verify_password(password, u["password_hash"])
            if not u:   # equalise timing for unknown users
                security.verify_password(password, security.hash_password("x", iterations=1000))
            if not valid:
                store.record_attempt(username or "?", ip, False)
                audit("auth.login_failed", target=username or "?", outcome="denied", actor="anonymous")
                flash("Username or password is incorrect.", "error")
                return render_template("login.html", no_users=no_users, step="password"), 401
            if u["totp_secret"]:
                session.clear()
                session["csrf"] = security.new_csrf_token()
                session["pending_2fa"] = {"u": u["username"], "t": time.time(), "next": request.args.get("next", "")}
                return render_template("login.html", step="2fa", no_users=False)
            return _finish_login(u)
        return render_template("login.html", no_users=no_users, step="2fa" if pending else "password")

    def _finish_login(u):
        nxt = (session.get("pending_2fa") or {}).get("next") or request.args.get("next") or ""
        session.clear()
        session["user"] = u["username"]
        session["seen"] = time.time()
        session["csrf"] = security.new_csrf_token()
        session.permanent = True
        store.record_attempt(u["username"], client_ip(), True)
        store.update_user(u["username"], last_login=now_iso())
        audit("auth.login", target=u["username"], actor=u["username"], detail={"2fa": bool(u["totp_secret"])})
        if not nxt.startswith("/") or nxt.startswith("//"):
            nxt = url_for("overview")
        return redirect(nxt)

    @app.post("/logout")
    def logout():
        if g.user:
            audit("auth.logout", target=g.user["username"])
        session.clear()
        flash("Signed out.", "info")
        return redirect(url_for("login"))

    # ------------------------------------------------------------------ disclosures
    @app.get("/disclosures")
    def disclosures():
        return render_template("disclosures.html", accept=False)

    @app.route("/disclosures/accept", methods=["GET", "POST"])
    def accept_disclosure():
        if not g.user:
            return redirect(url_for("login"))
        if request.method == "POST":
            if request.form.get("ack") != "yes":
                flash("Tick the box to confirm you have read the risk disclosure.", "error")
            else:
                store.update_user(g.user["username"], risk_ack=now_iso())
                audit("disclosure.accepted", target="risk-disclosure-v1")
                return redirect(url_for("overview"))
        return render_template("disclosures.html", accept=True)

    @app.get("/faq")
    def faq():
        return render_template("faq.html")

    @app.get("/docs")
    @app.get("/docs/<page>")
    def docs(page="getting-started"):
        if page not in DOC_PAGES:
            abort(404)
        return render_template(f"docs/{page}.html", page=page, title=DOC_PAGES[page])

    # ------------------------------------------------------------------ overview
    def source_summary(source):
        """Live: one account, one total. Paper / backtest: books are alternative models or variants over the same
        signals or history, so they are summarised per book and never summed (that would double-count)."""
        pb = scanner_cfg()
        ts = sources.trades(source)
        m = analytics.compute(ts, pb.capital)
        rows = analytics.breakdown(ts, lambda t: t.book, pb.capital) if source != data_mod.LIVE else []
        return {"source": source, "m": m, "books": len({t.book_key for t in ts}), "rows": rows}

    def default_book(source, book):
        """Paper and backtest default to one book; 'all' must be asked for explicitly."""
        if source == data_mod.LIVE or book:
            return "" if book == "all" else book
        bs = sources.books(source)
        if not bs:
            return ""
        keys = [b["key"] for b in bs]
        if "scanner-v1" in keys:
            return "scanner-v1"
        counts = {}
        for t in sources.trades(source):
            counts[t.book_key] = counts.get(t.book_key, 0) + 1
        return max(keys, key=lambda k: counts.get(k, 0))

    @app.get("/")
    @login_required
    def overview():
        st = sources.state() or {}
        lanes = []
        for key, label in (("risk", "v1"), ("risk_v2", "v2"), ("risk_v3", "v3")):
            r = st.get(key)
            if r:
                lanes.append({"lane": label, **r})
        today = []
        for key, label in (("paper", "v1"), ("paper_v2", "v2"), ("paper_v3", "v3")):
            for t in ((st.get(key) or {}).get("today") or []):
                today.append({"lane": label, **t})
        open_mv = [t for t in sources.trades(data_mod.PAPER) if t.open]
        return render_template("overview.html", summaries=[source_summary(s) for s in data_mod.SOURCES],
                               state=st, lanes=lanes, today=today, open_mv=open_mv, gate=gate_status(),
                               alerts=(st.get("recent_alerts") or [])[:8], events=(st.get("events") or {}),
                               orders=sources.orders(8))

    # ------------------------------------------------------------------ performance
    def filtered(source, book="", start="", end=""):
        ts = sources.trades(source)
        if book and book != "all":
            ts = [t for t in ts if t.book_key == book]
        if start:
            ts = [t for t in ts if t.day >= start]
        if end:
            ts = [t for t in ts if t.day <= end]
        return ts

    def _date_arg(name):
        v = request.args.get(name, "").strip()
        if not v:
            return ""
        try:
            return date.fromisoformat(v).isoformat()
        except ValueError:
            flash(f"Ignored {name}: use a date like 2026-01-31.", "warn")
            return ""

    @app.get("/performance")
    @login_required
    def performance():
        source = request.args.get("source", data_mod.PAPER)
        if source not in data_mod.SOURCES:
            abort(404)
        raw_book, start, end = request.args.get("book", ""), _date_arg("from"), _date_arg("to")
        book = default_book(source, raw_book)
        pb = scanner_cfg()
        ts = filtered(source, book, start, end)
        m = analytics.compute(ts, pb.capital)
        return render_template("performance.html", source=source, book=book, combined=(raw_book == "all"), start=start, end=end, m=m,
                               books=sources.books(source),
                               by_book=analytics.breakdown(ts, lambda t: t.book, pb.capital),
                               by_index=analytics.breakdown(ts, lambda t: f"{t.index} {t.side}", pb.capital),
                               by_exit=analytics.breakdown(ts, lambda t: t.exit_reason or "—", pb.capital),
                               months=analytics.monthly(ts), chart={"equity": m["equity"], "daily": m["daily"]})

    @app.get("/trades")
    @login_required
    def trades():
        source = request.args.get("source", data_mod.PAPER)
        if source not in data_mod.SOURCES:
            abort(404)
        book, start, end = request.args.get("book", ""), _date_arg("from"), _date_arg("to")
        ts = sorted(filtered(source, book, start, end), key=lambda t: t.entry_time or "", reverse=True)
        page = max(1, int(request.args.get("page", "1") or 1)) if request.args.get("page", "1").isdigit() else 1
        per = 100
        return render_template("trades.html", source=source, book=book, start=start, end=end,
                               books=sources.books(source), rows=ts[(page - 1) * per: page * per], total=len(ts),
                               page=page, pages=max(1, (len(ts) + per - 1) // per))

    # ------------------------------------------------------------------ paper
    @app.get("/paper")
    @login_required
    def paper():
        pb = scanner_cfg()
        ts = sources.trades(data_mod.PAPER)
        books = []
        for b in sources.books(data_mod.PAPER):
            bt = [t for t in ts if t.book_key == b["key"]]
            books.append({**b, "m": analytics.compute(bt, pb.capital), "open": [t for t in bt if t.open]})
        st = sources.state() or {}
        return render_template("paper.html", books=books, state=st, pb=pb, exec=(st.get("execution") or {}))

    # ------------------------------------------------------------------ backtests
    @app.get("/backtests")
    @login_required
    def backtests():
        pb = scanner_cfg()
        ts = sources.trades(data_mod.BACKTEST)
        sets = analytics.breakdown(ts, lambda t: t.book_key, pb.capital)
        labels = {b["key"]: b["label"] for b in sources.books(data_mod.BACKTEST)}
        for s in sets:
            s["label"] = labels.get(s["key"], s["key"])
        return render_template("backtests.html", reports=data_mod.backtest_reports(cfg.output_dir), sets=sets,
                               jobs=[j for j in jobs.list() if j["kind"] in BACKTEST_KINDS],
                               kinds={k: KINDS[k] for k in BACKTEST_KINDS}, indices=settings.SUPPORTED_INDICES)

    @app.get("/backtests/reports/<name>")
    @login_required
    def backtest_report(name):
        allowed = {r["name"] for r in data_mod.backtest_reports(cfg.output_dir)}
        if name not in allowed:
            abort(404)
        path = Path(cfg.output_dir) / name
        resp = send_file(path, mimetype="text/plain; charset=utf-8" if name.endswith(".txt") else "text/html; charset=utf-8")
        # the generated reports carry their own inline styles / scripts: sandbox them in a unique origin
        resp.headers["Content-Security-Policy"] = ("sandbox allow-scripts; default-src 'none'; img-src data:; "
                                                   "style-src 'unsafe-inline'; script-src 'unsafe-inline' https://cdn.plot.ly; "
                                                   "frame-ancestors 'self'")
        resp.headers["X-Report"] = "1"
        return resp

    @app.post("/backtests/run")
    @admin_required
    def backtest_run():
        if request.form.get("kind") not in BACKTEST_KINDS:
            abort(400)
        try:
            job = jobs.start(request.form.get("kind", ""), request.form.get("instrument", ""),
                             request.form.get("start", ""), request.form.get("end", ""), actor=g.user["username"])
        except JobError as error:
            audit("backtest.start", target=request.form.get("kind", ""), outcome="rejected", detail=str(error))
            flash(str(error), "error")
            return redirect(url_for("backtests"))
        audit("backtest.start", target=job["kind"], detail={"command": job["command"]})
        flash(f"Started: {job['command']}. Results appear below and in the reports list when it finishes.", "ok")
        return redirect(url_for("backtests"))

    # ------------------------------------------------------------------ risk
    def risk_context():
        pb, a = scanner_cfg(), app_settings()
        st = sources.state() or {}
        limit_daily = pb.max_daily_loss_rupees
        if pb.max_daily_loss_pct:
            limit_daily = min(limit_daily, pb.capital * pb.max_daily_loss_pct / 100)
        lanes = []
        for key, label in (("risk", "v1"), ("risk_v2", "v2"), ("risk_v3", "v3")):
            r = st.get(key)
            if not r:
                continue
            loss = max(0.0, -(r.get("day_pnl") or 0))
            lanes.append({"lane": label, "r": r, "bars": [
                {"label": "Daily loss used", "value": loss, "limit": r.get("limit") or limit_daily, "fmt": "inr"},
                {"label": "Trades this session", "value": r.get("trades_total") or 0, "limit": pb.max_trades_per_session, "fmt": "n"},
                {"label": "Open positions", "value": r.get("open_positions") or 0, "limit": pb.max_open_positions, "fmt": "n"},
                {"label": "Open premium", "value": r.get("open_premium") or 0,
                 "limit": pb.capital * pb.max_premium_pct_of_capital / 100, "fmt": "inr"},
            ]})
        ts = sources.trades(data_mod.PAPER)
        hist = []
        for b in sources.books(data_mod.PAPER):
            m = analytics.compute([t for t in ts if t.book_key == b["key"]], pb.capital)
            hist.append({"label": b["label"], "m": m})
        limits = [
            ("Capital (sizing base)", inr(pb.capital), "[RISK] capital"),
            ("Risk per trade", f"{pb.risk_per_trade_pct:g}% = {inr(pb.capital * pb.risk_per_trade_pct / 100)}", "[RISK] risk_per_trade_pct"),
            ("Max daily loss", inr(limit_daily), "[RISK] max_daily_loss_rupees / _pct"),
            ("Max weekly loss", f"{pb.max_weekly_loss_pct:g}% of capital" if pb.max_weekly_loss_pct else "off", "[RISK] max_weekly_loss_pct"),
            ("Max open positions", pb.max_open_positions, "[RISK] max_open_positions"),
            ("Trades per index / session", f"{pb.max_trades_per_index} / {pb.max_trades_per_session}", "[RISK]"),
            ("Max premium of capital", f"{pb.max_premium_pct_of_capital:g}%", "[RISK] max_premium_pct_of_capital"),
            ("Losing-streak stop", pb.max_consecutive_losses or "off", "[RISK] max_consecutive_losses"),
            ("Max same-direction positions", pb.max_same_direction or "off", "[RISK] max_same_direction"),
            ("Correlated risk multiple", pb.correlated_risk_mult or "off", "[RISK] correlated_risk_mult"),
            ("Max lots per expiry", pb.max_lots_per_expiry or "off", "[RISK] max_lots_per_expiry"),
            ("Premium stop cap", f"{pb.max_premium_loss_pct:g}% of premium", "[TRADE_PLAN] max_premium_loss_pct"),
            ("Minimum reward:risk after costs", f"{pb.min_rr:g}", "[TRADE_PLAN] min_rr"),
            ("Last entry / square-off", f"{pb.last_entry.strftime('%H:%M')} / {pb.square_off.strftime('%H:%M')}", "[TIMEFRAME]"),
            ("Expiry-day cut-off", pb.expiry_day_cutoff.strftime("%H:%M"), "[RISK] expiry_day_cutoff"),
            ("VIX ceiling", f"{pb.max_vix:g}", "[RISK] max_vix"),
            ("Live lots cap", a.execution.get("max_live_lots", "1"), "[EXECUTION] max_live_lots"),
            ("Live orders per day", a.execution.get("max_orders_per_day", "20"), "[EXECUTION] max_orders_per_day"),
        ]
        from src.execution.costs import CostConfig
        costs = CostConfig.from_mapping(a.costs)
        return {"lanes": lanes, "limits": limits, "hist": hist, "state": st, "gate": gate_status(),
                "kill": Path(cfg.kill_switch_file).exists(), "kill_file": str(cfg.kill_switch_file), "costs": costs,
                "pb": pb, "events": st.get("events") or {}}

    @app.get("/risk")
    @login_required
    def risk():
        return render_template("risk.html", **risk_context())

    @app.post("/risk/kill-switch")
    @login_required
    def kill_switch():
        action = request.form.get("action")
        f = Path(cfg.kill_switch_file)
        if action == "engage":
            reason = (request.form.get("reason") or "").strip()[:200]
            f.write_text(f"engaged {now_iso()} by {g.user['username']} from the console. {reason}\n", encoding="utf-8")
            audit("risk.kill_switch", target="engage", detail={"reason": reason, "file": str(f)})
            flash("Kill switch engaged. The scanner blocks new entries and squares off open positions on its next cycle.", "warn")
        elif action == "release":
            if g.user["role"] != "admin":
                audit("risk.kill_switch", target="release", outcome="denied")
                abort(403)
            if request.form.get("confirm", "").strip() != "RELEASE":
                flash('Type RELEASE to confirm releasing the kill switch.', "error")
                return redirect(url_for("risk"))
            try:
                f.unlink()
            except FileNotFoundError:
                pass
            audit("risk.kill_switch", target="release")
            flash("Kill switch released. New entries are allowed again within the risk limits.", "ok")
        else:
            abort(400)
        _strip_cache["t"] = 0
        return redirect(request.form.get("back") if (request.form.get("back") or "").startswith("/") else url_for("risk"))

    @app.post("/risk/disarm")
    @login_required
    def disarm():
        from src.execution import live_gate
        was = live_gate.disarm(path=cfg.arm_file)
        audit("live.disarm", target=str(date.today()), detail={"was_armed_file": was})
        flash("Live trading disarmed for today. Orders go to the paper broker only." if was else
              "Live trading was not armed. Nothing changed.", "ok" if was else "info")
        _strip_cache["t"] = 0
        return redirect(url_for("risk"))

    # ------------------------------------------------------------------ broker
    @app.get("/broker")
    @login_required
    def broker():
        force = request.args.get("refresh") == "1"
        snap = monitor.snapshot(force=force)
        if force:
            audit("broker.refresh", outcome="ok" if snap.get("connected") else snap.get("status", "error"))
        st = sources.state() or {}
        return render_template("broker.html", snap=snap, orders=sources.orders(200), exec=(st.get("execution") or {}),
                               gate=gate_status())

    # ------------------------------------------------------------------ health
    @app.get("/health")
    @login_required
    def health():
        snap = monitor.snapshot()
        h = system_health(cfg, sources.state(), snap, gate_status(), app_settings())
        return render_template("health.html", h=h)

    # ------------------------------------------------------------------ audit
    @app.get("/audit")
    @login_required
    def audit_page():
        action = request.args.get("action", "")
        actor = request.args.get("actor", "")
        rows = store.audit_rows(500, 0, action, actor)
        return render_template("audit.html", rows=rows, action=action, actor=actor,
                               verify=store.verify_chain() if request.args.get("verify") else None)

    # ------------------------------------------------------------------ reports
    def _download(text, filename, mimetype):
        resp = Response(text, mimetype=mimetype)
        resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp

    @app.get("/reports")
    @login_required
    def reports_page():
        return render_template("reports.html", books={s: sources.books(s) for s in data_mod.SOURCES})

    @app.get("/reports/trades.<fmt>")
    @login_required
    def report_trades(fmt):
        source = request.args.get("source", data_mod.PAPER)
        if source not in data_mod.SOURCES or fmt not in ("csv", "json"):
            abort(404)
        book, start, end = request.args.get("book", ""), _date_arg("from"), _date_arg("to")
        ts = filtered(source, book, start, end)
        meta = {"source": source, "source_label": data_mod.SOURCE_LABEL[source], "book": book or "all", "from": start,
                "to": end, "generated": now_iso(), "generated_by": g.user["username"], "trades": len(ts),
                "note": "net = gross - charges; slippage is inside fill prices (estimate shown separately)"}
        audit("report.download", target=f"trades.{fmt}", detail=meta)
        stamp = datetime.now(IST).strftime("%Y%m%d-%H%M")
        if fmt == "json":
            return _download(reports.trades_json(ts, meta), f"trades-{source}-{stamp}.json", "application/json")
        pre = [f"{k}: {v}" for k, v in meta.items()]
        return _download(reports.trades_csv(ts, pre), f"trades-{source}-{stamp}.csv", "text/csv")

    @app.get("/reports/orders.csv")
    @login_required
    def report_orders():
        audit("report.download", target="orders.csv")
        return _download(reports.to_csv(sources.orders(100000), reports.ORDER_COLUMNS,
                                        [f"order journal (data/execution.db), generated {now_iso()}"]),
                         "orders.csv", "text/csv")

    @app.get("/reports/audit.csv")
    @admin_required
    def report_audit():
        audit("report.download", target="audit.csv")
        return _download(reports.to_csv(store.audit_rows(1000000), reports.AUDIT_COLUMNS,
                                        [f"audit log, generated {now_iso()}", json.dumps(store.verify_chain())]),
                         "audit.csv", "text/csv")

    @app.get("/reports/performance")
    @login_required
    def report_performance():
        source = request.args.get("source", data_mod.PAPER)
        if source not in data_mod.SOURCES:
            abort(404)
        raw_book, start, end = request.args.get("book", ""), _date_arg("from"), _date_arg("to")
        book = default_book(source, raw_book)
        pb = scanner_cfg()
        ts = filtered(source, book, start, end)
        m = analytics.compute(ts, pb.capital)
        audit("report.view", target="performance", detail={"source": source, "book": book})
        label = next((b["label"] for b in sources.books(source) if b["key"] == book),
                     "All books combined (overlapping models; double-counts shared signals)" if source != data_mod.LIVE else "All")
        return render_template("report_print.html", source=source, book_label=label, start=start, end=end, m=m,
                               by_book=analytics.breakdown(ts, lambda t: t.book, pb.capital),
                               months=analytics.monthly(ts), chart={"equity": m["equity"], "daily": m["daily"]},
                               generated=now_iso())

    # ------------------------------------------------------------------ account
    @app.route("/account", methods=["GET", "POST"])
    @login_required
    def account():
        if request.method == "POST":
            current, new, again = (request.form.get(k, "") for k in ("current", "new", "again"))
            if not security.verify_password(current, g.user["password_hash"]):
                audit("account.password_change", outcome="denied", detail="wrong current password")
                flash("Your current password is incorrect.", "error")
            elif new != again:
                flash("The two new passwords do not match.", "error")
            elif security.password_problems(new, g.user["username"]):
                for p in security.password_problems(new, g.user["username"]):
                    flash(p, "error")
            else:
                store.update_user(g.user["username"], password_hash=security.hash_password(new), password_changed=now_iso())
                audit("account.password_change")
                flash("Password changed.", "ok")
            return redirect(url_for("account"))
        setup = session.get("totp_setup")
        from src.console.security import totp_uri
        return render_template("account.html", setup=setup, uri=totp_uri(setup, g.user["username"]) if setup else "")

    @app.post("/account/2fa")
    @login_required
    def account_2fa():
        action = request.form.get("action")
        if action == "begin":
            session["totp_setup"] = security.new_totp_secret()
        elif action == "confirm":
            secret = session.get("totp_setup")
            if secret and security.verify_totp(secret, request.form.get("code", "")):
                store.update_user(g.user["username"], totp_secret=secret)
                session.pop("totp_setup", None)
                audit("account.2fa_enabled")
                flash("Two-factor sign-in is on. You will need a code from your authenticator app at every sign-in.", "ok")
            else:
                flash("That code did not match. Check the time on your phone and try the current code.", "error")
        elif action == "disable":
            if g.user["totp_secret"] and security.verify_totp(g.user["totp_secret"], request.form.get("code", "")):
                store.update_user(g.user["username"], totp_secret=None)
                audit("account.2fa_disabled")
                flash("Two-factor sign-in is off.", "warn")
            else:
                audit("account.2fa_disabled", outcome="denied")
                flash("Enter a current code to turn two-factor sign-in off.", "error")
        elif action == "cancel":
            session.pop("totp_setup", None)
        return redirect(url_for("account"))

    # ------------------------------------------------------------------ admin: users
    @app.route("/admin/users", methods=["GET", "POST"])
    @admin_required
    def users():
        if request.method == "POST":
            action = request.form.get("action")
            ok, name = security.split_username(request.form.get("username", ""))
            if not ok:
                flash("Usernames are 3-32 characters: letters, digits, dot, dash, underscore.", "error")
                return redirect(url_for("users"))
            if action == "create":
                pw, role = request.form.get("password", ""), request.form.get("role", "viewer")
                if role not in ("admin", "viewer"):
                    abort(400)
                if store.user(name):
                    flash("That username exists.", "error")
                elif security.password_problems(pw, name):
                    for p in security.password_problems(pw, name):
                        flash(p, "error")
                else:
                    store.create_user(name, security.hash_password(pw), role)
                    audit("user.create", target=name, detail={"role": role})
                    flash(f"Created {name} ({role}). Share the password privately; they can change it under Account.", "ok")
            elif action in ("disable", "enable", "reset_2fa", "make_admin", "make_viewer"):
                if name == g.user["username"] and action in ("disable", "make_viewer"):
                    flash("You cannot disable or demote your own account.", "error")
                elif not store.user(name):
                    abort(404)
                else:
                    changes = {"disable": {"disabled": 1}, "enable": {"disabled": 0}, "reset_2fa": {"totp_secret": None},
                               "make_admin": {"role": "admin"}, "make_viewer": {"role": "viewer"}}[action]
                    store.update_user(name, **changes)
                    audit(f"user.{action}", target=name)
                    flash(f"{name}: {action.replace('_', ' ')} done.", "ok")
            else:
                abort(400)
            return redirect(url_for("users"))
        return render_template("users.html", users=store.users())

    # ------------------------------------------------------------------ admin: credentials
    @app.route("/admin/credentials", methods=["GET", "POST"])
    @admin_required
    def credentials():
        if request.method == "POST":
            action, name = request.form.get("action"), request.form.get("name", "")
            if name not in MANAGED_KEYS:
                abort(400)
            try:
                if action == "set":
                    vault.set(name, request.form.get("value", ""), actor=g.user["username"])
                    audit("vault.set", target=name)
                    flash(f"{name} saved encrypted.", "ok")
                elif action == "delete":
                    vault.delete(name)
                    audit("vault.delete", target=name)
                    flash(f"{name} removed from the vault.", "ok")
                elif action == "import":
                    value = settings.read_env_file().get(name, "").strip()
                    if not value:
                        flash(f"{name} is not set in .env.", "error")
                    else:
                        vault.set(name, value, actor=g.user["username"])
                        audit("vault.import_env", target=name)
                        flash(f"{name} copied from .env into the vault. You can now delete that line from .env.", "ok")
                else:
                    abort(400)
            except VaultError as error:
                audit(f"vault.{action}", target=name, outcome="error", detail=str(error))
                flash(str(error), "error")
            return redirect(url_for("credentials"))
        env = settings.read_env_file()
        try:
            rows = vault.describe()
            vault_error = ""
        except VaultError as error:
            rows, vault_error = [], str(error)
        for r in rows:
            r["in_env"] = bool(env.get(r["name"], "").strip())
            r["in_environ"] = bool(os.environ.get(r["name"]))
        key_source = "CONSOLE_MASTER_KEY environment variable" if os.environ.get("CONSOLE_MASTER_KEY") else (
            "vault.key protected by Windows DPAPI" if os.name == "nt" else "vault.key (owner-only file)")
        return render_template("credentials.html", rows=rows, vault_error=vault_error, key_source=key_source,
                               token=token_info(cfg.token_file),
                               live_confirm_set=bool(settings.get_secret("LIVE_TRADING_CONFIRM")))

    # ------------------------------------------------------------------ JSON API
    @app.get("/api/status")
    @login_required
    def api_status():
        _strip_cache["t"] = 0
        return jsonify(status_strip())

    @app.get("/api/summary")
    @login_required
    def api_summary():
        out = {}
        for s in data_mod.SOURCES:
            summ = source_summary(s)
            m = summ["m"]
            if s == data_mod.LIVE:
                out[s] = {k: m[k] for k in ("trades", "open_positions", "net", "gross", "charges", "win_rate",
                                            "profit_factor", "avg_trade", "sample_ok", "sample_note")}
                out[s]["max_drawdown"] = m["drawdown"]["amount"]
            else:
                out[s] = {"trades": m["trades"], "books": summ["rows"],
                          "note": "books are alternative models / variants over the same signals: not summed"}
        return jsonify({"generated": now_iso(), "sources": out,
                        "note": "live, paper and backtest figures are separate and never added together"})

    @app.get("/api/performance")
    @login_required
    def api_performance():
        source = request.args.get("source", data_mod.PAPER)
        if source not in data_mod.SOURCES:
            return jsonify({"error": "source must be live, paper or backtest"}), 400
        ts = filtered(source, request.args.get("book", ""), _date_arg("from"), _date_arg("to"))
        return jsonify({"source": source, "metrics": analytics.compute(ts, scanner_cfg().capital)})

    @app.get("/api/health")
    @login_required
    def api_health():
        return jsonify(system_health(cfg, sources.state(), monitor.snapshot(), gate_status(), app_settings()))

    @app.get("/api/broker")
    @login_required
    def api_broker():
        return jsonify(monitor.snapshot())

    @app.get("/api/risk")
    @login_required
    def api_risk():
        st = sources.state() or {}
        return jsonify({"lanes": {k: st.get(k) for k in ("risk", "risk_v2", "risk_v3")},
                        "kill_switch": Path(cfg.kill_switch_file).exists(), "live_gate": gate_status()})


    # ------------------------------------------------------------------ ML lab
    def ml_context():
        st = sources.state() or {}
        return {"model": ml_mod.model_summary(cfg.ml_model_file), "samples": ml_mod.sample_stats(cfg.scanner_db),
                "live": st.get("ml") or {}}

    @app.get("/ml")
    @login_required
    def ml_lab():
        ctx = ml_context()
        return render_template("ml.html", **ctx, jobs=[j for j in jobs.list() if j["kind"] == "ml"],
                               indices=settings.SUPPORTED_INDICES, ml_cfg={"enabled": app_settings().ml_enabled,
                                                                          "allow_block": app_settings().ml_allow_block},
                               chart={"calibration": ctx["model"].get("calibration", []) if ctx["model"].get("exists") else []})

    @app.post("/ml/train")
    @admin_required
    def ml_train():
        names = [i for i in settings.SUPPORTED_INDICES if request.form.get(f"inst_{i}")]
        try:
            job = jobs.start("ml", ",".join(names), request.form.get("start", ""), "", actor=g.user["username"])
        except JobError as error:
            audit("ml.train", outcome="rejected", detail=str(error))
            flash(str(error), "error")
            return redirect(url_for("ml_lab"))
        audit("ml.train", target=",".join(names), detail={"command": job["command"]})
        flash("Training started. The scanner picks up the new model by itself; it can only block entries if it "
              "passes every out-of-sample check.", "ok")
        return redirect(url_for("ml_lab"))

    @app.get("/api/ml")
    @login_required
    def api_ml():
        return jsonify(ml_context())

    # ------------------------------------------------------------------ AI assistant tools (read-only)
    def _tool_book(source, book):
        if source not in data_mod.SOURCES:
            raise ValueError("source must be live, paper or backtest")
        return book or ("" if source == data_mod.LIVE else default_book(source, ""))

    def _iso(v):
        if not v:
            return ""
        return date.fromisoformat(str(v)).isoformat()

    def t_overview():
        st = sources.state() or {}
        out = {"note": "sources are separate; paper/backtest books overlap and are listed per book, never summed",
               "scanner_status": st.get("status"), "scanner_updated": st.get("generated_at"),
               "kill_switch_engaged": Path(cfg.kill_switch_file).exists(), "live_gate": gate_status(), "sources": {}}
        for s_ in data_mod.SOURCES:
            summ = source_summary(s_)
            m = summ["m"]
            out["sources"][s_] = ({"closed_trades": m["trades"], "net": m["net"], "charges": m["charges"],
                                   "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
                                   "max_drawdown": m["drawdown"]["amount"], "sample_note": m["sample_note"]}
                                  if s_ == data_mod.LIVE else {"books": summ["rows"]})
        out["risk_today"] = {k: st.get(k) for k in ("risk", "risk_v2", "risk_v3") if st.get(k)}
        return out

    def t_list_books(source):
        if source not in data_mod.SOURCES:
            raise ValueError("bad source")
        counts = {}
        for t in sources.trades(source):
            counts[t.book_key] = counts.get(t.book_key, 0) + 1
        return [{**b, "trades": counts.get(b["key"], 0)} for b in sources.books(source)]

    def t_performance(source, book="", date_from="", date_to=""):
        book = _tool_book(source, book)
        m = analytics.compute(filtered(source, book, _iso(date_from), _iso(date_to)), scanner_cfg().capital)
        m = {k: v for k, v in m.items() if k != "equity"}
        m["daily"] = m["daily"][-60:]
        return {"source": source, "book": book or "all", "metrics": m}

    BREAKDOWN_KEYS = {
        "book": lambda t: t.book, "month": lambda t: (t.day or "")[:7], "exit_reason": lambda t: t.exit_reason or "-",
        "index_side": lambda t: f"{t.index} {t.side}", "hour": lambda t: (t.entry_time or "")[11:13] + ":00",
        "weekday": lambda t: date.fromisoformat(t.day).strftime("%A") if t.day else "-"}

    def t_breakdown(source, by, book="", date_from="", date_to=""):
        if by not in BREAKDOWN_KEYS:
            raise ValueError("bad dimension")
        book = "" if by == "book" else _tool_book(source, book)
        ts = filtered(source, book, _iso(date_from), _iso(date_to))
        return {"source": source, "book": book or "all", "by": by,
                "rows": analytics.breakdown(ts, BREAKDOWN_KEYS[by], scanner_cfg().capital)}

    def t_trades(source, book="", date_from="", date_to="", only="all", limit=20):
        ts = filtered(source, book if source == data_mod.LIVE else (book or ""), _iso(date_from), _iso(date_to))
        if only == "winners":
            ts = [t for t in ts if not t.open and (t.net or 0) > 0]
        elif only == "losers":
            ts = [t for t in ts if not t.open and (t.net or 0) < 0]
        elif only == "open":
            ts = [t for t in ts if t.open]
        ts = sorted(ts, key=lambda t: t.entry_time or "", reverse=True)[:max(1, min(50, int(limit or 20)))]
        keep = ("book", "index", "side", "symbol", "entry_time", "exit_time", "qty", "entry_price", "exit_price",
                "gross", "charges", "slippage", "net", "exit_reason", "open", "open_pnl")
        return {"source": source, "count": len(ts), "trades": [{k: t.to_dict()[k] for k in keep} for t in ts]}

    def t_risk():
        r = risk_context()
        from dataclasses import asdict
        return {"limits": [{"limit": a, "value": str(b), "setting": c} for a, b, c in r["limits"]],
                "today": [{"lane": l["lane"], **{k: l["r"].get(k) for k in ("day_pnl", "limit", "trades_total",
                          "open_positions", "halted", "halt_reason", "entries_allowed", "week_realised",
                          "consecutive_losses")}} for l in r["lanes"]],
                "costs": asdict(r["costs"]), "paper_slippage_pct": r["pb"].slippage_pct,
                "kill_switch_engaged": r["kill"], "live_gate": r["gate"], "events": r["events"]}

    def t_health():
        h = system_health(cfg, sources.state(), monitor.snapshot(), gate_status(), app_settings())
        return {"overall": h["overall"], "checks": h["checks"], "recent_log_problems": h["logs"].get("recent", [])[:10]}

    def t_ml():
        c = ml_context()
        c["samples"].pop("recent", None)
        return c

    assistant = Assistant(
        {"get_overview": t_overview, "list_books": t_list_books, "get_performance": t_performance,
         "get_breakdown": t_breakdown, "get_trades": t_trades, "get_risk": t_risk, "get_health": t_health,
         "get_ml_model": t_ml},
        api_key_fn=lambda: settings.get_secret("ANTHROPIC_API_KEY"), model=cfg.ai_model,
        max_tokens=cfg.ai_max_tokens, per_hour=cfg.ai_requests_per_hour, session=ai_session)
    app.extensions["console"]["assistant"] = assistant

    @app.get("/assistant")
    @login_required
    def assistant_page():
        return render_template("assistant.html", enabled=cfg.ai_enabled, configured=assistant.configured(),
                               history=assistant.history(g.user["username"]), model=cfg.ai_model,
                               per_hour=cfg.ai_requests_per_hour)

    @app.post("/api/assistant")
    @login_required
    def api_assistant():
        if not cfg.ai_enabled:
            return jsonify({"error": "The assistant is turned off ([CONSOLE] ai_enabled = false)."}), 403
        body = request.get_json(silent=True) or {}
        question = str(body.get("question") or request.form.get("question") or "")
        try:
            res = assistant.ask(g.user["username"], question)
        except AssistantError as error:
            audit("ai.ask", outcome=error.kind, detail={"question": question[:300], "error": str(error)[:200]})
            code = {"input": 400, "rate": 429, "config": 503, "auth": 502, "network": 502, "api": 502}.get(error.kind, 500)
            return jsonify({"error": str(error)}), code
        audit("ai.ask", detail={"question": question[:300], "tools": [t["name"] for t in res["tools"]],
                                "tokens": res["usage"], "model": res["model"]})
        return jsonify(res)

    @app.post("/api/assistant/clear")
    @login_required
    def api_assistant_clear():
        assistant.clear(g.user["username"])
        return jsonify({"ok": True})

    return app


def _session_key(path: Path) -> bytes:
    env = os.environ.get("CONSOLE_SECRET_KEY")
    if env:
        return env.encode()
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    key = secrets.token_bytes(48)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key
