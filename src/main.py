"""
Single entry point of the NIFTY & SENSEX pre-breakout CE/PE scanner.

    python -m src.main                     live (Upstox), runs all day; dashboard http://127.0.0.1:8502
    python -m src.main --mock              synthetic data, no credentials, 60x clock (a session takes ~6 min)
    python -m src.main --mock --speed 300  faster mock
    python -m src.main --once              one cycle, print a summary, exit
    python -m src.main --no-dashboard      no built-in web server (the HTML file is still written)
    python -m src.main --no-websocket      REST quotes only

Run it from the project root (VS Code / PyCharm: the "Scanner" run configuration does that).
Alerts and paper trading only: this program cannot place orders.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.app import Application, RunOptions
from src.config import settings
from src.utils.logging_config import setup_logging, shutdown_logging

MIN_PYTHON = (3, 10)


def parse_args(argv=None) -> RunOptions:
    p = argparse.ArgumentParser(prog="python -m src.main", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true", help="synthetic data, no credentials")
    p.add_argument("--speed", type=float, default=60.0, help="mock clock: simulated seconds per real second")
    p.add_argument("--start", default="09:15", help="mock clock start time HH:MM")
    p.add_argument("--notify", action="store_true", help="in mock mode, also send Telegram / popups (off by default)")
    p.add_argument("--once", action="store_true", help="run one cycle, print a summary, exit")
    p.add_argument("--no-dashboard", action="store_true", help="do not start the built-in dashboard server")
    p.add_argument("--no-websocket", action="store_true", help="do not open the Upstox WebSocket (REST only)")
    p.add_argument("--config", type=Path, default=settings.CONFIG_FILE, help="path to scanner.ini")
    p.add_argument("--debug", action="store_true", help="verbose logging")
    a = p.parse_args(argv)
    return RunOptions(mock=a.mock, speed=a.speed, start=a.start, notify=a.notify, once=a.once,
                      dashboard=not a.no_dashboard, websocket=not a.no_websocket, config=a.config), a.debug


def main(argv=None) -> int:
    if sys.version_info < MIN_PYTHON:
        print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required (found {sys.version.split()[0]}).")
        return 1
    options, debug = parse_args(argv)
    settings.ensure_dirs()
    setup_logging(logging.DEBUG if debug else logging.INFO)
    log = logging.getLogger("scanner.main")
    try:
        code = asyncio.run(Application(options).run())
    except KeyboardInterrupt:  # Ctrl+C pressed again during shutdown
        log.warning("Interrupted during shutdown.")
        code = 130
    except FileNotFoundError as error:
        log.error("%s", error)
        code = 1
    finally:
        logging.getLogger("scanner.main").info("Exit.")
        shutdown_logging()
    return code


if __name__ == "__main__":
    sys.exit(main())
