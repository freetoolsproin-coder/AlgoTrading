"""
F&O multi-strategy engine (paper trading only).

    python -m src.fno                     live Upstox data (after python -m src.broker.auth), dashboard http://127.0.0.1:8520
    python -m src.fno --mock              synthetic market, no token; everything is labelled MOCK
    python -m src.fno --once              one cycle printed in the terminal, no web server
    python -m src.fno strategies          the 43 strategies with their CE / PE / exit rules and settings
    python -m src.fno backtest --help     per-strategy backtest, out-of-sample split, walk-forward

Settings: config/fno.ini. This engine never places orders.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import settings

logger = logging.getLogger("scanner.fno.main")


def print_strategies(cfg):
    from src.fno.strategies import all_strategies
    for s in all_strategies():
        d = s.describe(cfg.strategies.get(s.key))
        print(f"\n{s.number:>2}. {s.name}  [{s.key}]  {s.category} / {s.kind}  {'ON' if d['enabled'] else 'OFF'}"
              f"{'  needs: ' + ', '.join(s.needs) if s.needs else ''}")
        print(f"    BUY CE : {s.ce}\n    BUY PE : {s.pe}\n    EXIT   : {s.exit}\n    params : {d['params']}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "backtest":
        from src.fno.backtest import main as bt_main
        return bt_main(argv[1:])
    p = argparse.ArgumentParser(prog="python -m src.fno", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", choices=("serve", "strategies"), default="serve")
    p.add_argument("--mock", action="store_true", help="synthetic data, no credentials")
    p.add_argument("--once", action="store_true", help="run one cycle, print it and exit")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    from src.fno.config import FnoConfig
    from src.utils.logging_config import setup_logging
    settings.ensure_dirs()
    setup_logging(logging.DEBUG if args.debug else logging.INFO, log_file=settings.LOG_DIR / "fno.log")
    try:
        cfg = FnoConfig.load(args.config)
    except ValueError as error:
        print(f"config/fno.ini: {error}")
        return 2
    if args.command == "strategies":
        print_strategies(cfg)
        return 0
    from src.fno.data import LiveData, MockData
    from src.fno.engine import Engine
    feed = MockData(cfg) if args.mock else LiveData(cfg)
    engine = Engine(cfg, feed)
    if args.once:
        try:
            engine.cycle()
        except Exception as error:
            logger.error("%s: %s", type(error).__name__, error)
            return 1
        s = engine.state()
        print(f"\n{s['status']}  {s['time']}  universe {len(s['watchlist'])}  scanned {s['health']['scanned']}  ({s['duration']}s)"
              f"{'  MOCK DATA' if s['mock'] else ''}")
        for w in s["watchlist"]:
            if w.get("bar_time"):
                print(f"  {w['symbol']:12} {w.get('ltp') or 0:>10.2f}  {w['status']:9} checks up {w['checks_up']}/{w['checks_avail']} "
                      f"down {w['checks_dn']}/{w['checks_avail']}  trend {w['trend']}  {', '.join(w['strategies'])}")
        for sig in s["signals"]:
            plan = sig.get("plan") or {}
            print(f"  SIGNAL {sig['status']:8} {sig['symbol']} {sig['side']} {sig['strategy_name']} conf {sig['confidence']}"
                  f"  {plan.get('instrument', '')} entry {plan.get('entry', '-')} SL {plan.get('stop', '-')} "
                  f"T1 {plan.get('t1', '-')} T2 {plan.get('t2', '-')} T3 {plan.get('t3', '-')}  {sig.get('reason', '')}")
        return 0
    from src.fno.web import create_app
    app = create_app(engine)
    host, port = args.host or cfg.host, args.port or cfg.port
    engine.start()
    logger.info("F&O engine (PAPER%s) on http://%s:%s", ", MOCK DATA" if args.mock else "", host, port)
    try:
        try:
            from waitress import serve
            serve(app, host=host, port=port, threads=8)
        except ImportError:
            app.run(host=host, port=port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
