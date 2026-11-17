"""
WSGI entry point for hosted deploys (Vercel and similar), which look for a top-level Flask instance named `app`.

It serves the trading console (src/console). Locally keep using `python -m src.console`.

On a serverless host the project folder is read-only, so runtime data (console users, audit log, session key)
goes to /tmp and is lost whenever the host recycles the instance. Set CONSOLE_SECRET_KEY in the host's
environment variables so sign-ins survive restarts.
"""
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
