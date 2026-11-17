"""
NIFTY 50 cash-equity service: intraday module + swing module. Flask JSON API on http://127.0.0.1:8530; the trading
desk (python -m src.desk, http://127.0.0.1:8500) hosts it and shows it as the "Intraday cash" and "Swing" tabs.

    python -m src.equity                  live Upstox data (needs today's token: python -m src.broker.auth)
    python -m src.equity --once           one scan printed to the terminal, no web server
    python -m src.equity --port 8600      another port
    python -m src.equity --server flask   Flask's development server instead of Waitress
    python -m src.equity --arm            arm LIVE orders for today (asks you to type a confirmation)
    python -m src.equity --disarm         remove today's arming
    python -m src.equity.backtest --module swing        backtest + validation (see that module)
    python -m src.equity.strategies                     list the 50 strategies

Settings: config/equity.ini. Default mode is PAPER. There is no mock mode: every signal comes from real market data,
and without a token the service says so instead of inventing prices.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from src.config import settings

logger = logging.getLogger("scanner.equity.main")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.equity", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", action="store_true", help="scan once, print the signals and exit")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--server", choices=("waitress", "flask"), default="waitress")
    p.add_argument("--arm", action="store_true", help="arm live orders for today")
    p.add_argument("--disarm", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def arm(cfg) -> int:
    from src.equity.execution import ARM_FILE
    from src.execution import live_gate
    today = date.today()
    if (settings.get_secret("LIVE_TRADING_CONFIRM") or "").strip() != live_gate.CONFIRM_PHRASE:
        print(f"Cannot arm: LIVE_TRADING_CONFIRM={live_gate.CONFIRM_PHRASE} is not set in .env")
        return 2
    if not sys.stdin.isatty():
        print("Arming needs an interactive terminal (type the confirmation yourself).")
        return 2
    r = cfg.risk
    limits = {"capital": r.capital, "risk_per_trade_pct": r.risk_per_trade_pct, "max_daily_loss_pct": r.max_daily_loss_pct,
              "max_open_intraday": r.max_open_intraday, "max_open_swing": r.max_open_swing, "daily_trade_limit": r.daily_trade_limit,
              "confirm_each_order": cfg.live_confirm_each_order, "auto_exits": cfg.live_auto_exits}
    print("Live limits for today (NIFTY 50 cash equity):")
    for k, v in limits.items():
        print(f"  {k}: {v}")
    try:
        from src.equity.broker import UpstoxEquityBroker
        funds = UpstoxEquityBroker().funds()
        print(f"  broker check: available margin Rs {funds.available_margin:,.0f}")
    except Exception as error:      # noqa: BLE001
        print(f"Cannot arm: broker check failed ({type(error).__name__}: {str(error)[:150]})")
        return 2
    print("\nReal orders can be placed on your Upstox account after you switch the dashboard to LIVE and confirm each "
          "order. Confirming an entry also authorises its protective stop order, the intraday square-off and kill-switch "
          "exits. Nothing in this software guarantees a profit; losses are real.")
    if input(f'Type "ARM {today.isoformat()}" to arm live equity trading for today: ').strip() != f"ARM {today.isoformat()}":
        print("Not armed.")
        return 1
    live_gate.write_arm(today, limits, path=ARM_FILE)
    print("ARMED for today. Switch the mode to LIVE on the dashboard (it stays PAPER until you do).")
    return 0


def print_state(state: dict):
    print(f"\n{state['status']} {state['status_text']}  {state['time']}  mode {state['mode']}  scan {state['scan_seconds']}s")
    m = state["market"]
    print(f"NIFTY {m.get('ltp')} ({m.get('day_pct')}%)  trend {m.get('trend')}  adv/dec {m.get('advancing')}/{m.get('declining')}")
    for title, rows in (("INTRADAY BUY", state["intraday"]["buy"]), ("INTRADAY SELL", state["intraday"]["sell"]),
                        ("SWING BUY", state["swing"]["buy"]), ("SWING SELL / EXIT", state["swing"]["sell"])):
        print(f"\n{title} ({len(rows)})")
        for o in rows[:12]:
            print(f"  {o['symbol']:11} {o['time'][:16]} entry {o['price']:>9.2f} SL {o['stop']:>9.2f} T1 {o['t1']:>9.2f} T2 {o['t2']:>9.2f} "
                  f"score {o['score']:>5.1f} x{o['confluence']} {o['status']:8} {o['reason'][:90]}")
    for sym, err in list(state["errors"].items())[:10]:
        print(f"ERROR {sym}: {err}")


def main(argv=None) -> int:
    args = parse_args(argv)
    from src.equity.config import EquityConfig
    from src.utils.logging_config import setup_logging
    settings.ensure_dirs()
    setup_logging(logging.DEBUG if args.debug else logging.INFO, log_file=settings.LOG_DIR / "equity.log")
    cfg = EquityConfig.from_ini()
    if args.disarm:
        from src.equity.execution import ARM_FILE
        from src.execution import live_gate
        print("Disarmed." if live_gate.disarm(ARM_FILE) else "Was not armed.")
        return 0
    if args.arm:
        return arm(cfg)
    from src.equity.engine import EquityEngine
    engine = EquityEngine(cfg, push=not args.once)
    if args.once:
        now = engine.clock()
        try:
            engine.quote_cycle(now)
            engine.scan_cycle(now)
            engine.quote_cycle(engine.clock())
        except Exception as error:      # noqa: BLE001
            logger.error("%s: %s", type(error).__name__, error)
            return 1
        engine.status, engine.status_text = "ok", ""
        print_state(engine.state())
        return 0
    from src.equity.web import create_app
    app = create_app(engine)
    engine.start()
    host, port = args.host or cfg.host, args.port or cfg.port
    logger.info("Equity service on http://%s:%s (mode %s)", host, port, engine.trader.mode)
    try:
        if args.server == "waitress":
            try:
                from waitress import serve
            except ImportError:
                app.run(host=host, port=port, threaded=True)
            else:
                serve(app, host=host, port=port, threads=8)
        else:
            app.run(host=host, port=port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
