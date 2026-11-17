"""
Movement trigger service: watches NIFTY / SENSEX (and India VIX) for 5 / 10 / 15-minute price moves and
fires alerts. Flask web page + JSON API + live stream on http://127.0.0.1:8503.

    python -m src.movement                     live Upstox data (needs today's token: python -m src.broker.auth)
    python -m src.movement --mock              synthetic data, no credentials (fast clock)
    python -m src.movement --mock --notify     ... and also send Telegram / popups
    python -m src.movement --port 8600         another port
    python -m src.movement --server flask      Flask's development server instead of Waitress

Rules: [MOVEMENT] in config/scanner.ini, or edit them on the web page (saved in data/movement_rules.json).
It runs beside the scanner (python -m src.main) and does not need it. Alerts only - it never places orders.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.broker.client import TokenSource, UpstoxAuthError, UpstoxClient
from src.broker.market_data import LiveProvider
from src.broker.mock_data import MockProvider
from src.config import settings
from src.config.settings import PBConfig
from src.execution.costs import make_cost_fn
from src.movement import paper as paper_mod
from src.movement import rules as rules_mod
from src.movement import strategies as strategies_mod
from src.movement.monitor import Broadcaster, MovementMonitor, Webhook
from src.movement.store import EventStore
from src.movement.web import create_app
from src.notifications.service import Notifier
from src.utils import single_instance
from src.utils.logging_config import setup_logging, shutdown_logging
from src.utils.numbers import hhmm

LOCK_FILE = settings.DATA_DIR / "movement.lock"
LOG_FILE = settings.LOG_DIR / "movement.log"
logger = logging.getLogger("scanner.movement.main")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.movement", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true", help="synthetic data, no credentials")
    p.add_argument("--speed", type=float, default=20.0, help="mock clock: simulated seconds per real second")
    p.add_argument("--start", default="09:15", help="mock clock start time HH:MM")
    p.add_argument("--notify", action="store_true", help="in mock mode, also send Telegram / popups / webhook")
    p.add_argument("--host", help="bind address (default [MOVEMENT] host, 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default [MOVEMENT] port, 8503)")
    p.add_argument("--server", choices=("waitress", "flask"), default="waitress")
    p.add_argument("--config", default=settings.CONFIG_FILE, help="path to scanner.ini")
    p.add_argument("--debug", action="store_true", help="verbose logging")
    return p.parse_args(argv)


def build(args):
    parser = settings.read_ini(args.config)
    app_settings = settings.load_app_settings(args.config)
    cfg = PBConfig.from_ini(args.config)
    sec = parser["MOVEMENT"] if parser.has_section("MOVEMENT") else {}
    rules_store = rules_mod.RulesStore(args.config)
    mock = args.mock or app_settings.mode == "mock"
    names = list(app_settings.indices)
    if mock:
        provider = MockProvider(cfg, names, speed=args.speed, start_clock=hhmm(args.start))
        push = args.notify
    else:
        provider = LiveProvider(names, cfg, client=UpstoxClient(requests_per_second=app_settings.requests_per_second),
                                constituents_loader=lambda _master: {})
        push = True
        try:
            TokenSource().token()
        except UpstoxAuthError as error:
            logger.error("%s The service will start and wait for a token.", error)
    notify = dict(app_settings.notify, telegram=sec.get("telegram", "true"),
                  desktop_popup=sec.get("desktop_popup", "false"), notify_movement="true")
    notifier = Notifier(notify, csv_path=settings.OUTPUT_DIR / "movement_alerts.csv", enabled=push)
    trade_notifier = Notifier(dict(notify, desktop_popup=sec.get("trade_popup", "true"), notify_paper_trade="true"),
                              csv_path=settings.OUTPUT_DIR / "movement_paper_alerts.csv", enabled=push)
    webhook = Webhook(settings.get_secret("MOVEMENT_WEBHOOK_URL") if push else None)
    store = EventStore()
    profiles = paper_mod.load_profiles(args.config)
    engine, strategy_profiles = strategies_mod.load_config(args.config)
    profiles = {**profiles, **strategy_profiles}
    trader = None
    if any(p.enabled for p in profiles.values()):
        trader = paper_mod.PaperTrader(profiles, cfg, make_cost_fn(app_settings.costs), store=store, mock=mock,
                                       kill_switch_file=app_settings.kill_switch_file)
        paper_mod.scalp_rules(rules_store.rules, profiles["scalp"])   # fail at start-up on bad scalp settings
    monitor = MovementMonitor(provider, rules_store, store, notifier=notifier, broadcaster=Broadcaster(),
                              webhook=webhook, mock=mock, push_enabled=push, trader=trader,
                              trade_notifier=trade_notifier, strategies=engine)
    monitor.config_path = args.config
    host = args.host or str(sec.get("host", "127.0.0.1")).strip()
    port = args.port or int(sec.get("port", 8503))
    logger.info("Movement service | %s | %s | %s | channels: %s%s", "MOCK DATA" if mock else "LIVE DATA",
                ", ".join(names), rules_mod.describe(rules_store.rules), ", ".join(notifier.channels()),
                ", webhook" if webhook.url else "")
    if trader is not None:
        logger.info("Paper trading: %s (simulated, no orders)", ", ".join(
            f"{n} {'on' if p.enabled else 'off'}" for n, p in profiles.items() if n in paper_mod.PROFILES))
    if engine is not None:
        logger.info("Strategies (%d, each its own paper book, default %d-min bars): %s", len(engine.enabled), engine.tf,
                    ", ".join(f"{k}@{engine.tfs[k]}m" for k in engine.enabled))
    return monitor, [notifier, trade_notifier], host, port


def main(argv=None) -> int:
    args = parse_args(argv)
    settings.ensure_dirs()
    setup_logging(logging.DEBUG if args.debug else logging.INFO, log_file=LOG_FILE)
    lock = single_instance.acquire(LOCK_FILE)
    if lock is None:
        logger.error("The movement service is already running (lock: %s).", LOCK_FILE)
        shutdown_logging()
        return 2
    monitor, notifiers = None, []
    code = 0
    try:
        monitor, notifiers, host, port = build(args)
        monitor.start()
        app = create_app(monitor)
        logger.info("Web page: http://%s:%s/  (Ctrl+C to stop)", host, port)
        if args.server == "flask":
            app.run(host=host, port=port, threaded=True, use_reloader=False)
        else:
            from waitress import create_server
            server = create_server(app, host=host, port=port, threads=16, channel_timeout=60)
            try:
                server.run()
            finally:
                server.close()
    except KeyboardInterrupt:
        logger.info("Stop requested.")
    except (FileNotFoundError, ValueError) as error:
        logger.error("Cannot start: %s", error)
        code = 1
    except OSError as error:
        logger.error("Cannot start the web server: %s (port in use? use --port)", error)
        code = 1
    finally:
        if monitor is not None:
            monitor.stop()
        for notifier in notifiers:
            notifier.close(5.0)
        single_instance.release(lock)
        logger.info("Stopped.")
        shutdown_logging()
    return code


if __name__ == "__main__":
    sys.exit(main())
