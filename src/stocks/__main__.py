"""
NIFTY 50 stock options service: intraday CE / PE plans with validated stop-loss and targets, paper tracking,
and a research report for any stock (search bar). Flask page + JSON API on http://127.0.0.1:8504.

    python -m src.stocks                  live Upstox data (needs today's token: python -m src.broker.auth)
    python -m src.stocks --mock           synthetic data, no token, works any time
    python -m src.stocks --once           one scan printed to the terminal, no web server
    python -m src.stocks --port 8600      another port
    python -m src.stocks --server flask   Flask's development server instead of Waitress

The same service runs the pre-breakout scanner ("Before the breakout" on the page, /api/breakout):
NIFTY 50 stocks coiled next to a level, with the trigger, an intraday equity plan and a CE / PE plan.
Backtest it with: python -m src.stocks.breakout_backtest

Settings: [STOCK_OPTIONS] and [STOCK_BREAKOUT] in config/scanner.ini (charges from [COSTS]). Paper only - it never places orders.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

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
    from src.execution.costs import make_cost_fn, make_equity_cost_fn
    from src.stocks.breakout import BreakoutConfig, telegram_pusher
    from src.stocks.data import LiveSource, MockSource
    from src.stocks.engine import StockEngine
    from src.stocks.precision import PrecisionConfig
    from src.stocks.signals import StockConfig
    cfg = StockConfig.from_ini(args.config)
    bcfg = BreakoutConfig.from_ini(args.config)
    parser = settings.read_ini(args.config)
    costs = dict(parser["COSTS"]) if parser.has_section("COSTS") else None
    eq_costs = dict(parser["COSTS_EQUITY"]) if parser.has_section("COSTS_EQUITY") else None
    source = MockSource() if args.mock else LiveSource(requests_per_second=cfg.requests_per_second)
    pusher = telegram_pusher() if bcfg.notify and not args.mock else None
    if bcfg.notify and pusher is None and not args.mock:
        logger.warning("[STOCK_BREAKOUT] notify = true but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set in .env")
    return StockEngine(source, cfg, cost_fn=make_cost_fn(costs), breakout_cfg=bcfg,
                       eq_cost_fn=make_equity_cost_fn(eq_costs), pusher=pusher,
                       precision_cfg=PrecisionConfig.from_ini(args.config))


def print_breakout(b: Optional[dict]):
    if not b:
        return
    print(f"\nBEFORE THE BREAKOUT  {b['counts']}")
    for x in b["items"]:
        if x["stage"] not in ("READY", "TRIGGERED", "CONFIRMED", "WATCH"):
            continue
        p, e, o = x.get("plan") or {}, x.get("equity") or {}, x.get("option") or {}
        print(f"{x['stage']:9} {x['symbol']:11} {x['side']} score {x['score']:5.1f}  {(x.get('zone') or '')[:34]:34} "
              f"trigger {p.get('trigger')} SL {p.get('stop')} T1 {p.get('t1')} T2 {p.get('t2')} | "
              f"{e.get('action', '-')} {e.get('qty', '-')} sh | {o.get('contract', '-')} @{o.get('entry', '-')} "
              f"SL {o.get('stop', '-')} T1 {o.get('t1', '-')}{'  OK' if x.get('tradable') else ''}")


def print_plans(state: dict):
    print(f"\n{state['status']}  {state['generated_at']}  ({state['duration']}s)  {state['counts']}")
    n = state.get("nifty") or {}
    if n.get("available"):
        print(f"NIFTY bias {n['bias']:+d}  " + ", ".join(f"{v['name']} {v['value']}" for v in n["votes"]))
    picks = [p for p in state["plans"] if (p.get("precision") or {}).get("ok")]
    print(f"\nTOP PICKS (precision mode): {len(picks)}")
    for p in picks:
        o, s = p["option"], p["spot"]
        print(f"  rank {p['precision']['rank']:5.1f}  {p['symbol']:11} {o['contract']:28} buy {o['entry']} SL {o['stop']} "
              f"T1 {o['t1']} T2 {o['t2']}  R:R {o['rr']}  risk/lot {o['risk_rs']}  | spot SL {s['stop']} T1 {s['t1']}")
    print()
    for p in state["plans"]:
        if p["status"] == "NO_TRADE":
            continue
        o, s = p.get("option") or {}, p.get("spot") or {}
        pr = p.get("precision")
        why = p["reason"]
        if pr:
            why = "TOP PICK" if pr["ok"] else "valid, not a pick: " + ", ".join(pr["failed"]).lower()
        print(f"{p['status']:6} {p['symbol']:11} {p['direction'] or '-':2} score {p['score']:+5.1f}  "
              f"{o.get('contract', '-'):28} entry {o.get('entry')} SL {o.get('stop')} T1 {o.get('t1')} T2 {o.get('t2')}"
              f"  | spot SL {s.get('stop')} T1 {s.get('t1')} T2 {s.get('t2')}  {why}")
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
        state = engine.state()
        print_plans(state)
        print_breakout(state.get("breakout"))
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
