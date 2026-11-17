"""
Risk manager for alerts / paper trading: daily loss limit, trade counts, open-position limits,
same-direction exposure, losing streak, external blocks (kill switch), and session gates
(time of day, expiry day, gap, India VIX, event calendar).

All limits come from the [RISK] / [TIMEFRAME] sections of config/scanner.ini.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.instruments import CE, PE
from src.config.settings import PBConfig
from src.options.selection import OptionCandidate
from src.risk.position_sizing import size_in_lots
from src.utils.numbers import finite


# =============================================================================
# Risk manager (alerts / paper only)
# =============================================================================
@dataclass
class RiskState:
    day: Optional[date] = None
    realised: float = 0.0
    open_pnl: float = 0.0
    trades_total: int = 0
    trades_by_index: Dict[str, int] = field(default_factory=dict)
    open_positions: int = 0
    open_directions: List[str] = field(default_factory=list)
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    week_realised: float = 0.0                 # realised net P&L earlier this week (before today)
    open_details: List[dict] = field(default_factory=list)   # index, direction, expiry, lots, premium, risk


def losing_streak(closed_nets: List[float]) -> int:
    """Losing trades in a row at the end of the list (chronological net P&L of closed trades)."""
    n = 0
    for net in reversed(closed_nets):
        if net < 0:
            n += 1
        else:
            break
    return n


class RiskManager:
    def __init__(self, cfg: PBConfig):
        self.cfg = cfg
        self.state = RiskState()
        self.external_block: Optional[str] = None   # e.g. the KILL_SWITCH file

    def roll(self, day: date):
        if self.state.day != day:
            self.state = RiskState(day=day)

    def daily_limit(self) -> float:
        """Rupee daily loss limit: the tighter of max_daily_loss_rupees and max_daily_loss_pct of capital (0 = off)."""
        limits = [abs(self.cfg.max_daily_loss_rupees)] if self.cfg.max_daily_loss_rupees > 0 else []
        if self.cfg.max_daily_loss_pct > 0:
            limits.append(self.cfg.capital * self.cfg.max_daily_loss_pct / 100.0)
        return min(limits) if limits else 0.0

    def update(self, realised: float, open_pnl: float, open_positions: int,
               open_directions: Optional[List[str]] = None, closed_nets: Optional[List[float]] = None,
               open_details: Optional[List[dict]] = None, week_realised: Optional[float] = None):
        s = self.state
        s.realised, s.open_pnl, s.open_positions = realised, open_pnl, open_positions
        if open_directions is not None:
            s.open_directions = list(open_directions)
        if closed_nets is not None:
            s.consecutive_losses = losing_streak(closed_nets)
        if open_details is not None:
            s.open_details = list(open_details)
        if week_realised is not None:
            s.week_realised = float(week_realised)
        limit = self.daily_limit()
        if limit > 0 and realised + open_pnl <= -limit:
            if not s.halted:
                s.halted, s.halt_reason = True, f"daily loss limit Rs {limit:,.0f} reached"
        if self.cfg.max_weekly_loss_pct > 0:
            week_limit = self.cfg.capital * self.cfg.max_weekly_loss_pct / 100.0
            if s.week_realised + realised + open_pnl <= -week_limit and not s.halted:
                s.halted, s.halt_reason = True, (f"weekly drawdown limit Rs {week_limit:,.0f} "
                                                 f"({self.cfg.max_weekly_loss_pct:g}% of capital) reached")
        if self.cfg.max_consecutive_losses > 0 and s.consecutive_losses >= self.cfg.max_consecutive_losses:
            if not s.halted:  # like the daily loss limit: no new entries for the rest of the day
                s.halted, s.halt_reason = True, (f"{s.consecutive_losses} losing trades in a row "
                                                 f"(max {self.cfg.max_consecutive_losses})")

    def can_open(self, index: str, direction: Optional[str] = None) -> Tuple[bool, str]:
        s, c = self.state, self.cfg
        if self.external_block:
            return False, self.external_block
        if s.halted:
            return False, s.halt_reason
        if s.open_positions >= c.max_open_positions:
            return False, f"{s.open_positions} positions open (max {c.max_open_positions})"
        if c.max_same_direction > 0 and direction is not None:
            same = sum(1 for d in s.open_directions if d == direction)
            if same >= c.max_same_direction:
                return False, f"{same} {direction} positions already open across indices (max {c.max_same_direction})"
        if s.trades_total >= c.max_trades_per_session:
            return False, f"{s.trades_total} trades today (max {c.max_trades_per_session})"
        if s.trades_by_index.get(index, 0) >= c.max_trades_per_index:
            return False, f"{index}: {s.trades_by_index.get(index)} trades today (max {c.max_trades_per_index})"
        return True, ""

    def size(self, c: OptionCandidate, mult: float = 1.0) -> Tuple[int, float, str]:
        """Whole lots so that (entry - stop + costs) x qty <= risk budget (x mult), then reduced (never
        increased) to fit the portfolio limits. 0 lots = skip."""
        lots, risk_rs, note = size_in_lots(c, self.cfg, mult)
        if lots < 1:
            return lots, risk_rs, note
        per_lot = risk_rs / lots
        capped, why = self.portfolio_cap(c, lots, per_lot)
        if capped < 1:
            return 0, per_lot, why
        return capped, capped * per_lot, ""

    def portfolio_cap(self, c: OptionCandidate, lots: int, risk_per_lot: float) -> Tuple[int, str]:
        """Largest lot count <= lots that keeps every portfolio limit. NIFTY and SENSEX are treated as one
        correlated book for same-direction risk (correlated_risk_mult)."""
        cfg, open_ = self.cfg, self.state.open_details
        lot_size = max(int(c.lot_size or 0), 1)
        premium_per_lot = (c.entry_premium if finite(c.entry_premium) else 0.0) * lot_size
        reasons = []
        allowed = lots
        if cfg.max_open_premium_pct > 0 and premium_per_lot > 0:
            room = cfg.capital * cfg.max_open_premium_pct / 100.0 - sum(d.get("premium", 0.0) for d in open_)
            fit = int(max(0.0, room) // premium_per_lot)
            if fit < allowed:
                allowed = fit
                reasons.append(f"open premium limit {cfg.max_open_premium_pct:g}% of capital")
        if cfg.max_lots_per_expiry > 0 and c.expiry:
            used = sum(int(d.get("lots", 0)) for d in open_ if str(d.get("expiry")) == str(c.expiry))
            fit = max(0, cfg.max_lots_per_expiry - used)
            if fit < allowed:
                allowed = fit
                reasons.append(f"{used} lots already open on expiry {c.expiry} (max {cfg.max_lots_per_expiry})")
        if cfg.correlated_risk_mult > 0 and risk_per_lot > 0:
            same = sum(d.get("risk", 0.0) for d in open_ if d.get("direction") == c.option_type)
            room = cfg.correlated_risk_mult * self.cfg.capital * self.cfg.risk_per_trade_pct / 100.0 - same
            fit = int(max(0.0, room) // risk_per_lot)
            if fit < allowed:
                allowed = fit
                reasons.append(f"same-direction risk across NIFTY + SENSEX would exceed "
                               f"{cfg.correlated_risk_mult:g}x one trade's budget")
        return allowed, "; ".join(reasons)

    def record_open(self, index: str, direction: Optional[str] = None):
        self.state.trades_total += 1
        self.state.trades_by_index[index] = self.state.trades_by_index.get(index, 0) + 1
        self.state.open_positions += 1
        if direction is not None:
            self.state.open_directions.append(direction)

    def snapshot(self) -> dict:
        s = self.state
        limit = self.daily_limit()
        return {"day": s.day.isoformat() if s.day else None, "realised": round(s.realised, 0),
                "week_realised": round(s.week_realised, 0),
                "open_premium": round(sum(d.get("premium", 0.0) for d in s.open_details), 0),
                "open_risk_ce": round(sum(d.get("risk", 0.0) for d in s.open_details if d.get("direction") == CE), 0),
                "open_risk_pe": round(sum(d.get("risk", 0.0) for d in s.open_details if d.get("direction") == PE), 0),
                "open_pnl": round(s.open_pnl, 0), "day_pnl": round(s.realised + s.open_pnl, 0),
                "limit": limit, "used_pct": round(100 * max(0.0, -(s.realised + s.open_pnl)) / limit, 1) if limit else None,
                "trades_total": s.trades_total, "trades_by_index": dict(s.trades_by_index),
                "open_positions": s.open_positions, "consecutive_losses": s.consecutive_losses,
                "open_ce": s.open_directions.count(CE), "open_pe": s.open_directions.count(PE),
                "halted": s.halted or bool(self.external_block),
                "halt_reason": self.external_block or s.halt_reason,
                "entries_allowed": not (s.halted or self.external_block)}


# =============================================================================
# Session gates (time, expiry day, gaps, events, VIX)
# =============================================================================
def session_block(now: pd.Timestamp, cfg: PBConfig, *, is_expiry_day: bool = False, gap_atr_mult: float = np.nan,
                  vix: float = np.nan, event: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """(watch_block, entry_block): reasons, or None when allowed."""
    clock = now.time()
    watch_block = None
    entry_block = None
    if clock < cfg.watch_start:
        watch_block = entry_block = f"before {cfg.watch_start:%H:%M}"
    if clock >= cfg.last_entry:
        entry_block = f"no new entries after {cfg.last_entry:%H:%M}"
    if clock >= cfg.square_off:
        watch_block = "session closing"
    if is_expiry_day and clock >= cfg.expiry_day_cutoff:
        entry_block = entry_block or f"expiry day: no entries after {cfg.expiry_day_cutoff:%H:%M}"
    if finite(gap_atr_mult) and abs(gap_atr_mult) > cfg.gap_atr:
        gate = (datetime.combine(now.date(), time(9, 15)) + timedelta(minutes=cfg.opening_range_minutes + 15)).time()
        if clock < gate:
            entry_block = entry_block or f"gap of {gap_atr_mult:+.1f} daily ATR: wait until {gate:%H:%M}"
    if finite(vix) and vix > cfg.max_vix:
        entry_block = entry_block or f"India VIX {vix:.1f} > {cfg.max_vix}"
    if event:
        entry_block = entry_block or f"event block: {event}"
    return watch_block, entry_block
