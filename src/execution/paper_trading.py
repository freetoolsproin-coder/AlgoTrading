"""
Paper trading: positions opened from ENTRY SETUP alerts at the ask (+ slippage) and closed at the bid
(- slippage) on premium stop, targets, underlying invalidation, time stop or square-off. Net of charges.

There is NO live order code anywhere in this application.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from src.config.instruments import CE
from src.config.settings import PBConfig
from src.scanner.signals import Alert
from src.utils.numbers import finite, fnum


def build_ladder(premiums: List[float], qty: int, lot: int, cfg: PBConfig) -> List[dict]:
    """Ladder steps for a position of `qty` units: each step books floor(pct% x qty / lot) whole lots."""
    try:
        steps, book = cfg.ladder_steps()
    except (AttributeError, ValueError):
        return []
    out = []
    if len(premiums) != len(steps):
        return []
    for r, pct, prem in zip(steps, book, premiums):
        if not finite(fnum(prem)):
            return []
        lots = int((pct / 100.0) * qty // max(lot, 1))
        out.append({"r": r, "pct": pct, "premium": float(prem), "qty": lots * lot, "done": False})
    return out


# =============================================================================
# Paper positions
# =============================================================================
@dataclass
class PaperPosition:
    id: str
    index: str
    direction: str
    symbol: str
    instrument_key: Optional[str]
    strike: float
    expiry: Optional[str]
    lot_size: int
    lots: int
    qty_open: int
    entry_time: str
    entry_price: float
    sl: float
    t1: float
    t2: float
    underlying_stop: float
    underlying_entry: float
    setup_key: str
    t1_done: bool = False
    realised_gross: float = 0.0
    charges: float = 0.0
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    last_price: float = np.nan
    exits: List[dict] = field(default_factory=list)
    mock: bool = False
    peak_price: float = np.nan     # best sellable price seen since entry (for the trailing stop)
    profile: str = "classic"       # classic | ladder ([TRADE_PLAN] exit_profile when the position opened)
    ladder: List[dict] = field(default_factory=list)   # [{r, premium, qty, done}]
    und_trail: float = np.nan      # underlying trailing stop (trail_mode atr / swing), only ever tightened
    und_best: float = np.nan       # best underlying close since entry in the trade direction
    early: bool = False            # opened by an early (pre-breakout) entry

    @property
    def closed(self) -> bool:
        return self.qty_open <= 0

    def open_pnl(self) -> float:
        if self.closed or not finite(self.last_price):
            return 0.0
        return (self.last_price - self.entry_price) * self.qty_open

    def net_pnl(self) -> float:
        return self.realised_gross - self.charges + self.open_pnl()

    def to_dict(self):
        out = asdict(self)
        out["open_pnl"] = round(self.open_pnl(), 2)
        out["net_pnl"] = round(self.net_pnl(), 2)
        return out


class PaperBook:
    def __init__(self, cfg: PBConfig, costs_fn: Callable[[str, float, int, str], float]):
        self.cfg = cfg
        self.costs_fn = costs_fn
        self.positions: List[PaperPosition] = []

    def open(self, alert: Alert, now_iso: str, exchange: str, mock: bool = False) -> Optional[PaperPosition]:
        o = alert.option or {}
        lots = int(o.get("lots") or 0)
        lot = int(o.get("lot_size") or 0)
        entry = fnum(o.get("entry_premium"))
        if lots < 1 or lot < 1 or not finite(entry):
            return None
        if any(not p.closed and p.index == alert.index for p in self.positions):
            return None  # one position per index: no averaging, no pyramiding
        qty = lots * lot
        pos = PaperPosition(id=f"{alert.setup_key}|{now_iso}", index=alert.index, direction=alert.direction,
                            symbol=o.get("symbol"), instrument_key=o.get("instrument_key"), strike=fnum(o.get("strike")),
                            expiry=o.get("expiry"), lot_size=lot, lots=lots, qty_open=qty, entry_time=now_iso,
                            entry_price=entry, sl=fnum(o.get("sl_premium")), t1=fnum(o.get("t1_premium")),
                            t2=fnum(o.get("t2_premium")), underlying_stop=fnum(alert.invalidation),
                            underlying_entry=fnum(alert.price), setup_key=alert.setup_key, last_price=entry, mock=mock,
                            profile=self.cfg.exit_profile, early="early" in (alert.message or "").lower())
        if pos.profile == "ladder":
            pos.ladder = build_ladder(o.get("ladder") or [], qty, lot, self.cfg)
            if not pos.ladder:
                pos.profile = "classic"   # no ladder premiums: fall back to the classic exits
        pos.charges += self.costs_fn("BUY", entry, qty, exchange)
        self.positions.append(pos)
        return pos

    def mark(self, pos: PaperPosition, *, bid: float, underlying: float, now: pd.Timestamp, exchange: str,
             bar_close_underlying: Optional[float] = None, check_underlying: bool = True,
             ctx: Optional[dict] = None) -> List[dict]:
        """Update one open position with the latest SELLABLE price (bid, or LTP minus half the spread).
        `ctx` (optional, from the last COMPLETED signal bar): atr, swing_low, swing_high, vwap (index scale),
        momentum_exhausted_ce / _pe - used by the trailing, VWAP and momentum exits."""
        events = []
        if pos.closed or not finite(bid):
            return events
        cfg = self.cfg
        pos.last_price = bid
        pos.peak_price = bid if not finite(pos.peak_price) else max(pos.peak_price, bid)
        sign = 1.0 if pos.direction == CE else -1.0
        held = (now - pd.Timestamp(pos.entry_time)).total_seconds() / 60.0
        if pos.profile == "ladder":
            return self._mark_ladder(pos, bid, underlying, now, exchange, bar_close_underlying, check_underlying,
                                     ctx or {}, sign, held, events)

        def sell(qty, price, reason):
            price = max(0.05, price * (1 - cfg.slippage_pct / 100.0))
            pos.realised_gross += (price - pos.entry_price) * qty
            pos.charges += self.costs_fn("SELL", price, qty, exchange)
            pos.qty_open -= qty
            rec = {"time": now.isoformat(), "qty": qty, "price": round(price, 2), "reason": reason}
            pos.exits.append(rec)
            if pos.closed:
                pos.exit_time, pos.exit_reason = now.isoformat(), reason
            events.append(rec)

        if pos.t1_done and cfg.trail_after_t1_pct > 0 and finite(pos.peak_price):
            # the stop only ever moves up: never below breakeven, never widened
            pos.sl = max(pos.sl, pos.peak_price * (1 - cfg.trail_after_t1_pct / 100.0))
        und_close = bar_close_underlying if bar_close_underlying is not None else underlying
        if check_underlying and finite(und_close) and finite(pos.underlying_stop) \
                and sign * (und_close - pos.underlying_stop) <= 0:
            sell(pos.qty_open, bid, "underlying invalidation")
        elif finite(pos.sl) and bid <= pos.sl:
            sell(pos.qty_open, bid, "premium stop" if not pos.t1_done else
                 ("trailing stop" if pos.sl > pos.entry_price + 1e-9 else "breakeven stop"))
        elif finite(pos.t2) and bid >= pos.t2:
            sell(pos.qty_open, bid, "target 2")
        elif finite(pos.t1) and bid >= pos.t1 and not pos.t1_done:
            half = (pos.lots // 2) * pos.lot_size
            pos.t1_done = True
            pos.sl = max(pos.sl, pos.entry_price)  # move the stop to breakeven
            if half >= pos.lot_size:
                sell(half, bid, "target 1 (half)")
        if not pos.closed and pos.t1_done and ctx and check_underlying:
            reason = self._structural_exit(pos, und_close, ctx, sign)
            if reason:
                sell(pos.qty_open, bid, reason)
        if not pos.closed and (held >= cfg.max_hold_minutes or now.time() >= cfg.square_off):
            sell(pos.qty_open, bid, "time stop" if held >= cfg.max_hold_minutes else "square-off")
        return events

    # ---------------------------------------------------------------- ladder profile
    def _sell(self, pos, qty, price, reason, now, exchange, events):
        cfg = self.cfg
        price = max(0.05, price * (1 - cfg.slippage_pct / 100.0))
        pos.realised_gross += (price - pos.entry_price) * qty
        pos.charges += self.costs_fn("SELL", price, qty, exchange)
        pos.qty_open -= qty
        rec = {"time": now.isoformat(), "qty": qty, "price": round(price, 2), "reason": reason}
        pos.exits.append(rec)
        if pos.closed:
            pos.exit_time, pos.exit_reason = now.isoformat(), reason
        events.append(rec)

    def _structural_exit(self, pos, und_close, ctx, sign) -> Optional[str]:
        """Underlying-based exits after the first scale-out: ATR / swing trailing stop, VWAP, momentum."""
        cfg = self.cfg
        if not finite(und_close):
            return None
        pos.und_best = und_close if not finite(pos.und_best) else (max(pos.und_best, und_close) if sign > 0
                                                                   else min(pos.und_best, und_close))
        atr_v = fnum(ctx.get("atr"))
        trail = np.nan
        if cfg.trail_mode == "atr" and finite(atr_v) and finite(pos.und_best):
            trail = pos.und_best - sign * cfg.trail_atr_mult * atr_v
        elif cfg.trail_mode == "swing":
            trail = fnum(ctx.get("swing_low" if sign > 0 else "swing_high"))
        if finite(trail):
            # only ever tightened, and never looser than the original invalidation
            base = pos.und_trail if finite(pos.und_trail) else pos.underlying_stop
            pos.und_trail = trail if not finite(base) else (max(base, trail) if sign > 0 else min(base, trail))
        if finite(pos.und_trail) and sign * (und_close - pos.und_trail) <= 0:
            return f"trailing stop ({cfg.trail_mode})"
        vw = fnum(ctx.get("vwap"))
        if cfg.vwap_exit and finite(vw) and sign * (und_close - vw) < 0:
            return "VWAP exit"
        if cfg.momentum_exit and ctx.get("momentum_exhausted_ce" if sign > 0 else "momentum_exhausted_pe"):
            return "momentum exhaustion"
        return None

    def _mark_ladder(self, pos, bid, underlying, now, exchange, bar_close_underlying, check_underlying, ctx, sign,
                     held, events):
        """Partial booking at each ladder step (1R / 1.5R / 2R / 3R by default, % of the ORIGINAL quantity),
        breakeven after the first step, then the remainder trails on the underlying (trail_mode) and exits on
        the stop, the VWAP / momentum exits, the time stop or the square-off."""
        cfg = self.cfg
        und_close = bar_close_underlying if bar_close_underlying is not None else underlying
        if pos.t1_done and finite(pos.peak_price) and cfg.trail_after_t1_pct > 0:
            pos.sl = max(pos.sl, pos.peak_price * (1 - cfg.trail_after_t1_pct / 100.0))
        if check_underlying and finite(und_close) and finite(pos.underlying_stop) \
                and sign * (und_close - pos.underlying_stop) <= 0:
            self._sell(pos, pos.qty_open, bid, "underlying invalidation", now, exchange, events)
            return events
        if finite(pos.sl) and bid <= pos.sl:
            self._sell(pos, pos.qty_open, bid, "premium stop" if not pos.t1_done else
                       ("trailing stop" if pos.sl > pos.entry_price + 1e-9 else "breakeven stop"), now, exchange, events)
            return events
        for step in pos.ladder:
            if step["done"] or pos.closed:
                continue
            if bid < step["premium"]:
                break
            step["done"] = True
            qty = min(int(step["qty"]), pos.qty_open)
            if not pos.t1_done:
                pos.t1_done = True
                pos.sl = max(pos.sl, pos.entry_price)          # breakeven after the first step
            if qty >= pos.lot_size:
                self._sell(pos, qty, bid, f"target {step['r']:g}R ({step['pct']:g}%)", now, exchange, events)
        if not pos.closed and pos.t1_done and check_underlying:
            reason = self._structural_exit(pos, und_close, ctx, sign)
            if reason:
                self._sell(pos, pos.qty_open, bid, reason, now, exchange, events)
        if not pos.closed and (held >= cfg.max_hold_minutes or now.time() >= cfg.square_off):
            self._sell(pos, pos.qty_open, bid, "time stop" if held >= cfg.max_hold_minutes else "square-off", now,
                       exchange, events)
        return events

    def close(self, pos: PaperPosition, *, bid: float, now: pd.Timestamp, exchange: str, reason: str) -> List[dict]:
        """Square off at the bid (risk halt / kill switch)."""
        if pos.closed or not finite(bid):
            return []
        price = max(0.05, bid * (1 - self.cfg.slippage_pct / 100.0))
        qty = pos.qty_open
        pos.last_price = bid
        pos.realised_gross += (price - pos.entry_price) * qty
        pos.charges += self.costs_fn("SELL", price, qty, exchange)
        pos.qty_open = 0
        pos.exit_time, pos.exit_reason = now.isoformat(), reason
        rec = {"time": now.isoformat(), "qty": qty, "price": round(price, 2), "reason": reason}
        pos.exits.append(rec)
        return [rec]

    def realised_today(self, day: date) -> float:
        return sum(p.realised_gross - p.charges for p in self.positions if pd.Timestamp(p.entry_time).date() == day)

    def open_pnl(self) -> float:
        return sum(p.open_pnl() for p in self.positions if not p.closed)

    def open_positions(self) -> List[PaperPosition]:
        return [p for p in self.positions if not p.closed]

    def risk_inputs(self, day: date) -> dict:
        """Extra RiskManager.update() inputs: open sides and today's closed trades in exit order."""
        closed = sorted((p for p in self.positions if p.closed and pd.Timestamp(p.entry_time).date() == day),
                        key=lambda p: p.exit_time or "")
        return {"open_directions": [p.direction for p in self.open_positions()],
                "closed_nets": [p.realised_gross - p.charges for p in closed],
                "open_details": [{"index": p.index, "direction": p.direction, "expiry": p.expiry,
                                  "lots": p.qty_open // max(p.lot_size, 1), "premium": p.entry_price * p.qty_open,
                                  "risk": max(0.0, (p.entry_price - p.sl) * p.qty_open) if finite(p.sl) else 0.0}
                                 for p in self.open_positions()]}

    def find_open(self, setup_key: str) -> Optional[PaperPosition]:
        return next((p for p in self.open_positions() if p.setup_key == setup_key), None)
