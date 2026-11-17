"""
Live-trading activation gate. Real orders are possible ONLY when all of these hold:

    1. config/scanner.ini   [EXECUTION] mode = live
    2. .env                 LIVE_TRADING_CONFIRM=I ACCEPT REAL-MONEY RISK
    3. today's arm file     data/LIVE_ARMED, written by the interactive command below for TODAY only
    4. not mock mode

    python -m src.execution.live_gate --status     what is missing (changes nothing)
    python -m src.execution.live_gate --arm        checks the token, funds and open positions, shows the limits,
                                                   and asks you to type "ARM <today's date>"
    python -m src.execution.live_gate --disarm     removes the arm file (same effect as the next day)

The arm file expires at the end of the day, so live trading never continues silently into a new session.
The KILL_SWITCH file still blocks entries and squares off everything, live included.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import settings

CONFIRM_PHRASE = "I ACCEPT REAL-MONEY RISK"
ARM_FILE = settings.DATA_DIR / "LIVE_ARMED"


def arm_file(path: Optional[Path] = None) -> Path:
    return Path(path) if path else ARM_FILE


def status(exec_cfg: dict, today: date, mock: bool = False, path: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """(armed, what is missing)."""
    missing = []
    if str(exec_cfg.get("mode", "paper")).strip().lower() != "live":
        missing.append("[EXECUTION] mode is not 'live'")
    if (settings.get_secret("LIVE_TRADING_CONFIRM") or "").strip() != CONFIRM_PHRASE:
        missing.append(f"LIVE_TRADING_CONFIRM={CONFIRM_PHRASE} is not set in .env")
    f = arm_file(path)
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("day") != today.isoformat():
            missing.append(f"armed for {data.get('day')}, not today: run python -m src.execution.live_gate --arm")
    except (OSError, ValueError):
        missing.append("not armed today: run python -m src.execution.live_gate --arm")
    if mock:
        missing.append("mock mode never trades live")
    return not missing, missing


def write_arm(today: date, limits: dict, path: Optional[Path] = None):
    f = arm_file(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"day": today.isoformat(), "armed_at": datetime.now().isoformat(timespec="seconds"),
                             "limits": limits}, indent=1), encoding="utf-8")


def disarm(path: Optional[Path] = None) -> bool:
    f = arm_file(path)
    try:
        f.unlink()
        return True
    except OSError:
        return False


def _preflight(exec_cfg: dict) -> List[str]:
    """Token, broker reachability, funds and open positions. Returns problems (empty = ok)."""
    problems = []
    try:
        from src.execution.upstox_broker import UpstoxBroker
        broker = UpstoxBroker()
        funds = broker.funds()
        print(f"  funds: available margin Rs {funds.available_margin:,.0f}")
        if funds.available_margin <= 0:
            problems.append("no available margin")
        open_pos = [p for p in broker.positions() if p.quantity != 0]
        if open_pos:
            print(f"  WARNING: {len(open_pos)} open position(s) at the broker that this app did not open:")
            for p in open_pos[:10]:
                print(f"    {p.instrument_key} qty {p.quantity}")
    except Exception as error:  # noqa: BLE001
        problems.append(f"broker check failed: {type(error).__name__}: {str(error)[:150]}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Arm / disarm live order execution for today.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true")
    g.add_argument("--arm", action="store_true")
    g.add_argument("--disarm", action="store_true")
    args = ap.parse_args(argv)
    app = settings.load_app_settings()
    cfg = settings.PBConfig.from_ini()
    ex = app.execution
    today = date.today()
    if args.disarm:
        print("Disarmed." if disarm() else "Was not armed.")
        return 0
    armed, missing = status(ex, today, mock=(app.mode == "mock"))
    if args.status:
        print("LIVE TRADING ARMED for today." if armed else "Live trading is OFF:\n  - " + "\n  - ".join(missing))
        return 0
    blockers = [m for m in missing if not m.startswith("not armed") and not m.startswith("armed for")]
    if blockers:
        print("Cannot arm:\n  - " + "\n  - ".join(blockers))
        return 2
    if not sys.stdin.isatty():
        print("Arming needs an interactive terminal (type the confirmation yourself).")
        return 2
    print("Pre-flight checks ...")
    problems = _preflight(ex)
    if problems:
        print("Cannot arm:\n  - " + "\n  - ".join(problems))
        return 2
    limits = {"lane": ex.get("lane", "v1"), "max_live_lots": ex.get("max_live_lots", "1"),
              "max_orders_per_day": ex.get("max_orders_per_day", "20"), "risk_per_trade_pct": cfg.risk_per_trade_pct,
              "daily_loss_limit": cfg.max_daily_loss_rupees, "max_open_positions": cfg.max_open_positions}
    print("\nLive limits for today:")
    for k, v in limits.items():
        print(f"  {k}: {v}")
    print("\nReal orders will be placed on your Upstox account for the signals of lane "
          f"{limits['lane']}. Nothing in this software guarantees a profit; losses are real.")
    typed = input(f'Type "ARM {today.isoformat()}" to arm live trading for today: ').strip()
    if typed != f"ARM {today.isoformat()}":
        print("Not armed.")
        return 1
    write_arm(today, limits)
    print("ARMED for today. Start (or keep running) python -m src.main; it picks this up within one cycle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
