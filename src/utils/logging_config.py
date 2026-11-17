"""
One logging setup for the whole application: console + logs/scanner.log (rotated at midnight, 14 days).
Every module logs through a child of the "scanner" logger. Secrets are never passed to the logger;
`RedactingFilter` is a second line of defence that masks anything that looks like a bearer token or a
Telegram bot token.
"""
from __future__ import annotations

import logging
import re
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from src.config.settings import LOG_FILE

FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_PATTERNS = [
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}"), r"\1***"),
    (re.compile(r"\b\d{6,}:[A-Za-z0-9_\-]{30,}\b"), "***telegram-token***"),
    (re.compile(r"(access_token[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-]{12,}"), r"\1***"),
]


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = message
        for pattern, repl in _PATTERNS:
            redacted = pattern.sub(repl, redacted)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


def setup_logging(level: int = logging.INFO, log_file: Path = LOG_FILE) -> logging.Logger:
    """Idempotent: calling it twice does not add duplicate handlers."""
    for stream in (sys.stdout, sys.stderr):  # Windows consoles: alerts contain ₹ and emoji
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    root = logging.getLogger("scanner")
    root.setLevel(level)
    if getattr(root, "_scanner_configured", False):
        return root
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(FORMAT)
    redact = RedactingFilter()
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=14, encoding="utf-8")
    console = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, console):
        handler.setFormatter(formatter)
        handler.addFilter(redact)
        root.addHandler(handler)
    root.propagate = False
    root._scanner_configured = True
    return root


def shutdown_logging():
    """Flush and close every handler (called last on exit)."""
    root = logging.getLogger("scanner")
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)
    root._scanner_configured = False
