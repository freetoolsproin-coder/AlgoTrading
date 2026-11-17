"""
Risk manager of the F&O paper book. `check_entry` returns every reason a new entry is blocked (empty = allowed).

Controls: maximum daily loss (realised + open), daily profit lock, trades per day, open positions, positions per
underlying, consecutive losses, duplicate signals (same underlying + side inside the cooldown, or the same
strategy bar twice), no-trade windows, first / last entry time, scheduled events (config/events.csv), expiry-day
cutoff and reduced size, spread / slippage limits, stale data, and the kill switch.

Kill switch: the file KILL_SWITCH in the project folder (shared with the other services) or data/fno/KILL_SWITCH
(this engine only; also written automatically when the daily loss limit is hit). While it exists no entry is
taken and open paper positions are squared off. Only a person removes it.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import settings
from src.fno.config import FnoConfig, data_dir
from src.utils import market_calendar

logger = logging.getLogger("scanner.fno.risk")


class RiskManager:
    def __init__(self, cfg: FnoConfig, kill_files: Optional[List[Path]] = None, events_file: Optional[Path] = None):
        self.cfg = cfg
        self.local_kill = (data_dir() / "KILL_SWITCH") if kill_files is None else Path(kill_files[0])
        self.kill_files = [self.local_kill, settings.PROJECT_ROOT / "KILL_SWITCH"] if kill_files is None \
            else [Path(p) for p in kill_files]
        self.events = market_calendar.load_events(events_file or settings.DEFAULT_EVENTS_FILE)
        self.last_signal: Dict[tuple, pd.Timestamp] = {}
        self.seen_bars: set = set()

    # ------------------------------------------------------------------ kill switch
    def killed(self) -> str:
        for path in self.kill_files:
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8").strip()
                except OSError:
                    text = ""
                return text or f"kill switch file {path.name} present"
        return ""

    def kill(self, reason: str):
        if not self.local_kill.exists():
            self.local_kill.parent.mkdir(parents=True, exist_ok=True)
            self.local_kill.write_text(reason, encoding="utf-8")
            logger.critical("KILL SWITCH engaged: %s", reason)

    def release(self) -> bool:
        """Removes this engine's own kill file only (never the shared one)."""
        if self.local_kill.exists():
            self.local_kill.unlink()
            logger.warning("F&O kill switch released by the operator.")
            return True
        return False

    # ------------------------------------------------------------------ day statistics
    @staticmethod
    def day_stats(closed_today: List, open_positions: List) -> dict:
        realized = sum(p.net for p in closed_today)
        open_pnl = sum(p.unrealized + p.net for p in open_positions)
        streak = 0
        for p in sorted(closed_today, key=lambda x: x.closed, reverse=True):
            if p.net < 0:
                streak += 1
            else:
                break
        return {"realized": realized, "open_pnl": open_pnl, "total": realized + open_pnl,
                "trades": len(closed_today) + len(open_positions), "consecutive_losses": streak}

    def limits(self, capital: float) -> dict:
        return {"max_daily_loss": capital * self.cfg.max_daily_loss_pct / 100.0,
                "profit_lock": capital * self.cfg.daily_profit_lock_pct / 100.0,
                "risk_per_trade": capital * self.cfg.risk_per_trade_pct / 100.0}

    # ------------------------------------------------------------------ gates
    def session_block(self, now: pd.Timestamp) -> str:
        t = now.time()
        if not market_calendar.is_trading_day(now.date()):
            return "market holiday"
        if t < self.cfg.first_entry:
            return f"before {self.cfg.first_entry:%H:%M}"
        if t >= self.cfg.last_entry:
            return f"after the last entry time {self.cfg.last_entry:%H:%M}"
        for a, b in self.cfg.windows():
            if a <= t < b:
                return f"no-trade window {a:%H:%M}-{b:%H:%M}"
        event = market_calendar.event_now(self.events, now, self.cfg.event_block_minutes, self.cfg.event_block_minutes)
        if event:
            return f"event: {event}"
        return ""

    def risk_mult(self, now: pd.Timestamp, expiry: Optional[date]) -> float:
        return self.cfg.expiry_day_risk_mult if expiry is not None and expiry == now.date() else 1.0

    def check_entry(self, now: pd.Timestamp, symbol: str, side: str, strategy: str, bar_time, capital: float,
                    closed_today: List, open_positions: List, expiry: Optional[date] = None,
                    spread_pct: Optional[float] = None, data_age: Optional[float] = None) -> List[str]:
        cfg, why = self.cfg, []
        killed = self.killed()
        if killed:
            return [f"kill switch: {killed}"]
        block = self.session_block(now)
        if block:
            why.append(block)
        stats, lim = self.day_stats(closed_today, open_positions), self.limits(capital)
        if stats["total"] <= -lim["max_daily_loss"]:
            why.append(f"daily loss limit reached (Rs {stats['total']:,.0f})")
            if cfg.auto_kill_on_daily_loss:
                self.kill(f"daily loss limit reached on {now:%Y-%m-%d}: Rs {stats['total']:,.0f}")
        if cfg.daily_profit_lock_pct > 0 and stats["realized"] >= lim["profit_lock"]:
            why.append(f"daily profit lock reached (Rs {stats['realized']:,.0f})")
        if stats["trades"] >= cfg.max_trades_per_day:
            why.append(f"max {cfg.max_trades_per_day} trades per day")
        if len(open_positions) >= cfg.max_open_positions:
            why.append(f"max {cfg.max_open_positions} open positions")
        if sum(1 for p in open_positions if p.symbol == symbol) >= cfg.max_positions_per_underlying:
            why.append(f"already holding {symbol}")
        if stats["consecutive_losses"] >= cfg.max_consecutive_losses:
            why.append(f"{stats['consecutive_losses']} consecutive losses: trading paused for today")
        last = self.last_signal.get((symbol, side))
        if last is not None and (now - last).total_seconds() < cfg.signal_cooldown_minutes * 60:
            why.append(f"duplicate: {symbol} {side} signalled {int((now - last).total_seconds() // 60)} min ago")
        if (symbol, strategy, side, str(bar_time)) in self.seen_bars:
            why.append("duplicate: this bar's signal was already handled")
        if expiry is not None and expiry == now.date():
            if not cfg.allow_expiry_day_contract:
                why.append("contract expires today (allow_expiry_day_contract = false)")
            elif now.time() >= cfg.expiry_day_last_entry:
                why.append(f"expiry day: no entries after {cfg.expiry_day_last_entry:%H:%M}")
        if spread_pct is not None and spread_pct > cfg.max_spread_pct:
            why.append(f"spread {spread_pct:.1f}% > {cfg.max_spread_pct:g}%")
        if cfg.slippage_pct > cfg.max_slippage_pct:
            why.append("configured slippage exceeds max_slippage_pct")
        if data_age is not None and data_age > cfg.max_data_age_seconds:
            why.append(f"price data is {int(data_age)} s old")
        return why

    def mark_signal(self, now: pd.Timestamp, symbol: str, side: str, strategy: str, bar_time):
        self.last_signal[(symbol, side)] = now
        self.seen_bars.add((symbol, strategy, side, str(bar_time)))
        if len(self.seen_bars) > 20000:
            self.seen_bars = set(list(self.seen_bars)[-5000:])
