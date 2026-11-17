"""
NIFTY 50 stock options service: intraday CE / PE plans with validated stop-loss and targets, paper tracking,
and a research report for any stock (search bar). Flask page + JSON API on http://127.0.0.1:8504.

    python -m src.stocks                  live Upstox data (needs today's token: python -m src.broker.auth)
    python -m src.stocks --mock           synthetic data, no token, works any time
    python -m src.stocks --once           one scan printed to the terminal, no web server
    python -m src.stocks --port 8600      another port
    python -m src.stocks --server flask   Flask's development server instead of Waitress

Settings: [STOCK_OPTIONS] in config/scanner.ini (charges from [COSTS]). Paper only - it never places orders.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import settings

logger = logging.getLogger("scanner.stocks.main")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.stocks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true", help="synthetic data, no credentials")
    p.add_argument("--once", action="store_true", help="scan once, print the plans and exit")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8504)
    p.add_argument("--server", choices=("waitress", "flask"), default="waitress")
    p.add_argument("--config", default=settings.CONFIG_FILE)
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def build(args):
    from src.execution.costs import make_cost_fn
    from src.stocks.data import LiveSource, MockSource
    from src.stocks.engine import StockEngine
    from src.stocks.signals import StockConfig
    cfg = StockConfig.from_ini(args.config)
    parser = settings.read_ini(args.config)
    costs = dict(parser["COSTS"]) if parser.has_section("COSTS") else None
    source = MockSource() if args.mock else LiveSource(requests_per_second=cfg.requests_per_second)
    return StockEngine(source, cfg, cost_fn=make_cost_fn(costs))


def print_plans(state: dict):
    print(f"\n{state['status']}  {state['generated_at']}  ({state['duration']}s)  {state['counts']}")
    for p in state["plans"]:
        if p["status"] == "NO_TRADE":
            continue
        o, s = p.get("option") or {}, p.get("spot") or {}
        print(f"{p['status']:6} {p['symbol']:11} {p['direction'] or '-':2} score {p['score']:+5.1f}  "
              f"{o.get('contract', '-'):28} entry {o.get('entry')} SL {o.get('stop')} T1 {o.get('t1')} T2 {o.get('t2')}"
              f"  | spot SL {s.get('stop')} T1 {s.get('t1')} T2 {s.get('t2')}  {p['reason']}")
    for sym, err in state["errors"].items():
        print(f"ERROR  {sym}: {err}")


def main(argv=None) -> int:
    args = parse_args(argv)
    from src.utils.logging_config import setup_logging
    settings.ensure_dirs()
    setup_logging(logging.DEBUG if args.debug else logging.INFO, log_file=settings.LOG_DIR / "stocks.log")
    engine = build(args)
    if args.once:
        try:
            engine.refresh()
        except Exception as error:
            logger.error("%s: %s", type(error).__name__, error)
            return 1
        print_plans(engine.state())
        return 0
    from src.stocks.web import create_app
    app = create_app(engine)
    engine.start()
    logger.info("Stock options service on http://%s:%s (%s)", args.host, args.port, "MOCK" if args.mock else "live")
    try:
        if args.server == "waitress":
            try:
                from waitress import serve
            except ImportError:
                logger.warning("waitress not installed; using Flask's development server")
                app.run(host=args.host, port=args.port, threaded=True)
            else:
                serve(app, host=args.host, port=args.port, threads=8)
        else:
            app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
