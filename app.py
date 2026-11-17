"""
WSGI entry point for hosted deploys (Vercel and similar), which look for a top-level Flask instance named `app`.

It serves the trading console (src/console). Locally keep using `python -m src.console`.

On a serverless host the project folder is read-only, so runtime data (console users, audit log, session key)
goes to /tmp and is lost whenever the host recycles the instance. Set CONSOLE_SECRET_KEY in the host's
environment variables so sign-ins survive restarts.

First admin on a host (no terminal there): set these environment variables on the host
    CONSOLE_ADMIN_USER            e.g. admin
    CONSOLE_ADMIN_PASSWORD_HASH   output of `python -m src.console --hash-password` run on your computer
The account is created at start-up whenever the console has no users (so again after each /tmp reset).
"""
import logging
import os
import tempfile

if os.environ.get("VERCEL") and not os.environ.get("SCANNER_RUNTIME_DIR"):
    os.environ["SCANNER_RUNTIME_DIR"] = os.path.join(tempfile.gettempdir(), "algotrading")

from src.config import settings  # noqa: E402  (must come after SCANNER_RUNTIME_DIR is set)
from src.console.config import load_console_config  # noqa: E402
from src.console.web import create_app  # noqa: E402

settings.ensure_dirs()
_cfg = load_console_config()
if os.environ.get("VERCEL"):
    _cfg.secure_cookies = True   # served over HTTPS
app = create_app(_cfg)


def _bootstrap_admin() -> None:
    user = os.environ.get("CONSOLE_ADMIN_USER", "").strip()
    pw_hash = os.environ.get("CONSOLE_ADMIN_PASSWORD_HASH", "").strip()
    if not (user and pw_hash):
        return
    from src.console import security
    log = logging.getLogger("scanner.console")
    ok, name = security.split_username(user)
    if not ok or not pw_hash.startswith("pbkdf2_sha256$") or pw_hash.count("$") != 3:
        log.error("CONSOLE_ADMIN_USER / CONSOLE_ADMIN_PASSWORD_HASH invalid; no admin created.")
        return
    store = app.extensions["console"]["store"]
    if store.count_users() == 0:
        store.create_user(name, pw_hash, "admin")
        store.audit("bootstrap", "user.create", target=name, detail={"role": "admin", "source": "environment"})
        log.info("Admin %s created from environment variables.", name)


_bootstrap_admin()
