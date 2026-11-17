"""
Trading console.

    python -m src.console --create-admin       first run: create an admin account (asks for a password)
    python -m src.console                      start the console at http://127.0.0.1:8510
    python -m src.console --port 8600          another port
    python -m src.console --reset-password U   set a new password for user U (and unlock it)
    python -m src.console --verify-audit       check the audit log's hash chain
    python -m src.console --hash-password      print a hash for CONSOLE_ADMIN_PASSWORD_HASH (hosted deploys, see app.py)

It runs beside the scanner (python -m src.main) and the movement service; it does not need them to be running and
never places orders. Keep it on 127.0.0.1 unless it sits behind an HTTPS reverse proxy ([CONSOLE] in
config/scanner.ini).
"""
from __future__ import annotations

import argparse
import getpass
import logging
import sys

from src.config import settings
from src.console import security
from src.console.config import load_console_config
from src.console.store import ConsoleStore, now_iso


def _ask_password(username: str) -> str:
    while True:
        pw = getpass.getpass("Password (12+ characters, mixed): ")
        problems = security.password_problems(pw, username)
        if problems:
            print("  " + "\n  ".join(problems))
            continue
        if getpass.getpass("Repeat: ") != pw:
            print("  The two passwords differ.")
            continue
        return pw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.console", description="Trading console (web UI).")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--create-admin", action="store_true", help="create an admin account")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--reset-password", metavar="USER")
    ap.add_argument("--verify-audit", action="store_true")
    ap.add_argument("--hash-password", action="store_true",
                    help="print a password hash for CONSOLE_ADMIN_PASSWORD_HASH (hosted deploys)")
    a = ap.parse_args(argv)
    if a.hash_password:
        ok, name = security.split_username(a.username)
        if not sys.stdin.isatty():
            print("Run this in an interactive terminal so you can type the password.")
            return 2
        print(security.hash_password(_ask_password(name)))
        return 0
    settings.ensure_dirs()
    from src.utils.logging_config import setup_logging
    log = setup_logging()
    cfg = load_console_config()
    store = ConsoleStore(cfg.console_db)

    if a.create_admin:
        ok, name = security.split_username(a.username)
        if not ok:
            print("Usernames are 3-32 characters: letters, digits, dot, dash, underscore.")
            return 2
        if store.user(name):
            print(f"User {name} already exists. Use --reset-password {name} to change its password.")
            return 2
        if not sys.stdin.isatty():
            print("Run this in an interactive terminal so you can type the password.")
            return 2
        store.create_user(name, security.hash_password(_ask_password(name)), "admin")
        store.audit("cli", "user.create", target=name, detail={"role": "admin"})
        print(f"Admin {name} created. Start the console with: python -m src.console")
        return 0
    if a.reset_password:
        ok, name = security.split_username(a.reset_password)
        if not ok or not store.user(name):
            print("No such user.")
            return 2
        store.update_user(name, password_hash=security.hash_password(_ask_password(name)), disabled=0,
                          password_changed=now_iso())
        store.record_attempt(name, "cli", True)   # clears the lockout counter
        store.audit("cli", "user.reset_password", target=name)
        print(f"Password for {name} reset.")
        return 0
    if a.verify_audit:
        r = store.verify_chain()
        print(f"Audit log: {r['rows']} rows, " + ("chain intact." if r["ok"] else f"BROKEN at id {r['broken_at']}."))
        return 0 if r["ok"] else 1

    host, port = a.host or cfg.host, a.port or cfg.port
    if host not in ("127.0.0.1", "localhost", "::1") and not cfg.secure_cookies:
        log.warning("Console bound to %s without secure_cookies: put it behind HTTPS before exposing it on a network.",
                    host)
    if store.count_users() == 0:
        log.warning("No console users yet: run python -m src.console --create-admin")
    from src.console.web import create_app
    app = create_app(cfg)
    url = f"http://{host}:{port}/"
    log.info("Trading console: %s (read-only on trading data; never places orders)", url)
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8, ident="console")
    except ImportError:
        logging.getLogger("scanner.console").warning("waitress not installed; using the development server")
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    except OSError as error:
        log.error("Console not started: %s:%s is in use or not allowed (%s). Use --port.", host, port, error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
