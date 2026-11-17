"""System health: scanner heartbeat, data feed, WebSocket, broker, token, services, databases, disk, logs."""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from src.console.data import connect_ro

IST = timezone(timedelta(hours=5, minutes=30))
OK, WARN, FAIL, UNKNOWN = "ok", "warn", "fail", "unknown"


def _check(name: str, status: str, detail: str, group: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "group": group}


def probe(url: str, timeout: float = 1.5) -> Optional[int]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 - fixed local URLs
            return r.status
    except Exception:   # noqa: BLE001
        return None


def age_seconds(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=IST)
    return (datetime.now(IST) - t).total_seconds()


def fmt_age(sec: Optional[float]) -> str:
    if sec is None:
        return "never"
    sec = int(sec)
    if sec < 90:
        return f"{sec}s ago"
    if sec < 5400:
        return f"{sec // 60} min ago"
    if sec < 172800:
        return f"{sec // 3600} h ago"
    return f"{sec // 86400} days ago"


LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ (\w+)\s+(\S+): (.*)$")


def log_tail(path: Path, lines: int = 400) -> dict:
    path = Path(path)
    if not path.exists():
        return {"exists": False, "errors": 0, "warnings": 0, "recent": []}
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return {"exists": True, "errors": 0, "warnings": 0, "recent": []}
    rows = text.splitlines()[-lines:]
    recent, errors, warnings = [], 0, 0
    for raw in rows:
        m = LOG_LINE.match(raw)
        if not m:
            continue
        level = m.group(2)
        if level in ("ERROR", "CRITICAL"):
            errors += 1
        elif level == "WARNING":
            warnings += 1
        if level in ("WARNING", "ERROR", "CRITICAL"):
            recent.append({"time": m.group(1), "level": level, "logger": m.group(3), "message": m.group(4)[:300]})
    return {"exists": True, "errors": errors, "warnings": warnings, "recent": recent[-25:][::-1]}


def db_check(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"name": path.name, "exists": False}
    c = connect_ro(path)
    res = {"name": path.name, "exists": True, "size": path.stat().st_size}
    if c is None:
        res["ok"] = False
        return res
    try:
        res["ok"] = c.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    except sqlite3.Error as error:
        res["ok"], res["error"] = False, str(error)
    finally:
        c.close()
    return res


def system_health(cfg, state: Optional[dict], broker: Optional[dict], gate: dict, scanner_settings=None) -> dict:
    checks: List[dict] = []
    loop = getattr(scanner_settings, "loop_seconds", 30) if scanner_settings else 30
    # scanner heartbeat
    gen = (state or {}).get("generated_at")
    a = age_seconds(gen)
    if state is None:
        checks.append(_check("Scanner heartbeat", FAIL, "no state file: the scanner has not run (python -m src.main)", "Engine"))
    elif a is not None and a <= max(90, 3 * loop):
        checks.append(_check("Scanner heartbeat", OK, f"last cycle {fmt_age(a)}", "Engine"))
    else:
        checks.append(_check("Scanner heartbeat", WARN, f"last cycle {fmt_age(a)} - scanner stopped or stalled", "Engine"))
    if state:
        err = (state.get("cycle") or {}).get("last_error") or ""
        checks.append(_check("Cycle errors", WARN if err else OK, err[:200] or "none in the last cycle", "Engine"))
        mode = state.get("mode")
        checks.append(_check("Data mode", WARN if state.get("mock") else OK,
                             f"{mode}" + (" - synthetic data, not for decisions" if state.get("mock") else ""), "Engine"))
        market_open = state.get("market_open")
        for name, idx in (state.get("indices") or {}).items():
            feed = idx.get("feed") or {}
            if feed.get("ok"):
                checks.append(_check(f"{name} market data", OK, "fresh", "Market data"))
            else:
                checks.append(_check(f"{name} market data", WARN if market_open else UNKNOWN,
                                     (feed.get("reason") or "no data") + ("" if market_open else " (market closed)"),
                                     "Market data"))
        ws = state.get("websocket") or {}
        if ws.get("connected"):
            checks.append(_check("WebSocket feed", OK, f"{ws.get('subscribed', 0)} instruments, "
                                 f"{ws.get('reconnects', 0)} reconnects", "Market data"))
        else:
            checks.append(_check("WebSocket feed", WARN if market_open else UNKNOWN,
                                 (ws.get("last_error") or "not connected") + ("" if market_open else " (market closed)"),
                                 "Market data"))
        ex = state.get("execution") or {}
        prob = ex.get("last_problem") or ex.get("blocked") or ""
        checks.append(_check("Order engine", WARN if prob else OK,
                             f"mode {ex.get('mode', '?')}, broker {ex.get('broker', '?')}" + (f": {prob}" if prob else ""),
                             "Execution"))
        ev = state.get("events") or {}
        checks.append(_check("Event calendar", OK if ev.get("active") else WARN, ev.get("note") or
                             f"{len(ev.get('upcoming') or [])} upcoming events", "Risk"))
    # broker / token
    if broker is not None:
        tok = broker.get("token") or {}
        if not tok.get("present"):
            checks.append(_check("Upstox token", FAIL, "missing: run python -m src.broker.auth", "Broker"))
        elif tok.get("expired"):
            checks.append(_check("Upstox token", FAIL, f"expired {tok.get('expires')}: run the daily login", "Broker"))
        else:
            checks.append(_check("Upstox token", OK, f"from {tok.get('source')}" +
                                 (f", expires about {tok.get('expires')[11:16]} on {tok.get('expires')[:10]}"
                                  if tok.get("expires") else ""), "Broker"))
        if broker.get("connected"):
            checks.append(_check("Broker API", OK, f"reachable, {broker.get('latency_ms')} ms", "Broker"))
        else:
            checks.append(_check("Broker API", FAIL if tok.get("present") else UNKNOWN,
                                 broker.get("message") or broker.get("status", ""), "Broker"))
    # safety switches
    ks = Path(cfg.kill_switch_file).exists()
    checks.append(_check("Kill switch", WARN if ks else OK, "ENGAGED: new entries blocked, positions squared off"
                         if ks else "released", "Risk"))
    checks.append(_check("Live trading gate", WARN if gate.get("armed") else OK,
                         "ARMED for today: real orders possible" if gate.get("armed") else "not armed (no real orders)",
                         "Execution"))
    # local services
    try:
        from src.config import settings
        p = settings.read_ini(cfg.config_file)
        dport = p.getint("DASHBOARD", "port", fallback=8502)
        mport = p.getint("MOVEMENT", "port", fallback=8503)
    except Exception:   # noqa: BLE001
        dport, mport = 8502, 8503
    for label, port in (("Scanner dashboard", dport), ("Movement service", mport)):
        code = probe(f"http://127.0.0.1:{port}/health")
        checks.append(_check(label, OK if code == 200 else UNKNOWN,
                             f"port {port} answering" if code == 200 else f"port {port} not answering (not running?)",
                             "Services"))
    # storage
    dbs = [db_check(p) for p in (cfg.scanner_db, cfg.execution_db, cfg.movement_db, cfg.console_db)]
    for d in dbs:
        if not d.get("exists"):
            checks.append(_check(f"Database {d['name']}", UNKNOWN, "not created yet", "Storage"))
        else:
            checks.append(_check(f"Database {d['name']}", OK if d.get("ok") else FAIL,
                                 f"{d['size'] / 1e6:.1f} MB, integrity {'ok' if d.get('ok') else 'FAILED'}", "Storage"))
    try:
        du = shutil.disk_usage(cfg.data_dir if Path(cfg.data_dir).exists() else ".")
        free_pct = du.free / du.total * 100
        checks.append(_check("Disk space", OK if free_pct > 10 else (WARN if free_pct > 3 else FAIL),
                             f"{du.free / 1e9:.1f} GB free ({free_pct:.0f}%)", "Storage"))
    except OSError:
        pass
    logs = log_tail(Path(cfg.log_dir) / "scanner.log")
    if logs["exists"]:
        checks.append(_check("Scanner log", WARN if logs["errors"] else OK,
                             f"{logs['errors']} errors, {logs['warnings']} warnings in the latest lines", "Engine"))
    proc = {}
    try:
        import psutil
        me = psutil.Process()
        proc = {"cpu_pct": psutil.cpu_percent(interval=None), "mem_pct": psutil.virtual_memory().percent,
                "console_rss_mb": round(me.memory_info().rss / 1e6, 1)}
    except Exception:   # noqa: BLE001 - psutil is optional
        pass
    worst = FAIL if any(c["status"] == FAIL for c in checks) else WARN if any(c["status"] == WARN for c in checks) else OK
    return {"overall": worst, "checks": checks, "logs": logs, "process": proc,
            "checked_at": datetime.now(IST).isoformat(timespec="seconds")}
