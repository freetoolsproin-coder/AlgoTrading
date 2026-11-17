"""
One option position and its exit engine. The SAME class manages paper positions (one quote per cycle) and
backtest positions (one premium bar per step), so live paper results and backtests follow identical rules.

Order of checks inside one step (conservative: when a bar touches both the stop and a target, the stop wins):
    1. forced exit (kill switch, square-off)            -> at the close / bid
    2. gap through the stop (open <= stop)              -> at the open
    3. stop touched (low <= stop)                       -> at the stop
    4. targets T1, T2, T3 touched (high >= target)      -> scale out at the target; stop to entry after T1
    5. underlying closes beyond its invalidation level  -> at the close
    6. trailing stop on the underlying (ATR or swing)   -> at the close
    7. confirmed opposite signal, time stop             -> at the close
Market-type exits (everything except target limits) are filled `slippage_pct` worse than the trigger price.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

import pandas as pd

from src.execution.costs import CostConfig, order_charges
from src.fno.contracts import Plan, plan_split
from src.utils.numbers import finite


@dataclass
class Position:
    id: str
    symbol: str
    strategy: str
    side: str
    instrument: str
    key: str
    segment: str
    lot: int
    qty: int
    entry: float
    stop: float
    initial_stop: float
    r: float
    targets: List[float]
    split: List[int]                      # lots booked at T1 / T2 / T3
    und_entry: float
    und_stop: float
    opened: str
    confidence: float = 0.0
    expiry: str = ""
    strike: float = 0.0
    qty_open: int = 0
    stage: int = 0                        # targets reached so far
    und_trail: Optional[float] = None
    status: str = "OPEN"
    exit_reason: str = ""
    closed: str = ""
    realized: float = 0.0                 # gross of charges
    charges: float = 0.0
    last_price: float = 0.0
    max_price: float = 0.0
    min_price: float = 0.0
    price_source: str = "live quote"
    mock: bool = False
    fills: List[dict] = field(default_factory=list)
    plan: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def open(cls, plan: Plan, cfg, costs: CostConfig, when, price_source="live quote", mock=False,
             fill_price: Optional[float] = None) -> "Position":
        entry = plan.entry if fill_price is None else fill_price
        split = plan_split(plan.lots, cfg)
        shift = entry - plan.entry                      # keep R when the fill differs from the planned entry
        p = cls(id=uuid.uuid4().hex[:12], symbol=plan.symbol, strategy=plan.strategy, side=plan.side,
                instrument=plan.instrument, key=plan.key, segment=plan.segment, lot=plan.lot, qty=plan.qty,
                entry=entry, stop=round(plan.stop + shift, 2), initial_stop=round(plan.stop + shift, 2), r=plan.r,
                targets=[round(t + shift, 2) for t in (plan.t1, plan.t2, plan.t3)], split=split,
                und_entry=plan.und_entry, und_stop=plan.und_stop, opened=str(when), confidence=plan.confidence,
                expiry=plan.expiry, strike=plan.strike, qty_open=plan.qty, last_price=entry, max_price=entry,
                min_price=entry, price_source=price_source, mock=mock, plan=plan.to_dict())
        p.charges = order_charges("BUY", entry, plan.qty, plan.segment, costs)
        p.fills.append({"time": str(when), "side": "BUY", "qty": plan.qty, "price": entry, "reason": "ENTRY"})
        return p

    # ------------------------------------------------------------------ accounting
    @property
    def net(self) -> float:
        return self.realized - self.charges

    @property
    def unrealized(self) -> float:
        return (self.last_price - self.entry) * self.qty_open if self.status == "OPEN" else 0.0

    @property
    def r_multiple(self) -> float:
        risk = self.r * self.qty
        return self.net / risk if risk > 0 else 0.0

    def _sell(self, qty: int, price: float, reason: str, when, costs: CostConfig):
        qty = min(qty, self.qty_open)
        if qty <= 0:
            return
        price = max(round(price, 2), 0.05)
        self.realized += (price - self.entry) * qty
        self.charges += order_charges("SELL", price, qty, self.segment, costs)
        self.qty_open -= qty
        self.fills.append({"time": str(when), "side": "SELL", "qty": qty, "price": price, "reason": reason})
        if self.qty_open == 0:
            self.status, self.exit_reason, self.closed = "CLOSED", reason, str(when)

    def close_all(self, price: float, reason: str, when, costs: CostConfig, slippage_pct: float = 0.0):
        self._sell(self.qty_open, price * (1 - slippage_pct / 100.0), reason, when, costs)

    # ------------------------------------------------------------------ the exit engine
    def step(self, when, o: float, h: float, l: float, c: float, cfg, costs: CostConfig, und_close: float = float("nan"),
             atr: float = float("nan"), swing_lo: float = float("nan"), swing_hi: float = float("nan"),
             opposite: bool = False, force: str = "", slippage_pct: float = 0.0, touch_buffer: float = 0.0,
             minutes_open: Optional[float] = None) -> List[dict]:
        """Advance by one premium bar (or one quote: o = h = l = c). Returns the fills made in this step."""
        if self.status != "OPEN" or not finite(c):
            return []
        before = len(self.fills)
        up = self.side == "CE"
        self.last_price = c
        self.max_price, self.min_price = max(self.max_price, h), min(self.min_price, l)
        slip = 1 - slippage_pct / 100.0
        stop_name = "TRAIL" if self.stage >= cfg.trail_after and self.stop > self.entry else \
            "BREAKEVEN" if self.stage >= 1 and self.stop >= self.entry else "SL"
        if force:
            self._sell(self.qty_open, c * slip, force, when, costs)
        elif o <= self.stop:
            self._sell(self.qty_open, o * slip, f"{stop_name} (gap)", when, costs)
        elif l <= self.stop:
            self._sell(self.qty_open, self.stop * slip, stop_name, when, costs)
        else:
            while self.status == "OPEN" and self.stage < 3 and h >= self.targets[self.stage] * (1 + touch_buffer):
                lots = self.split[self.stage]
                self.stage += 1
                if self.stage == 3 or sum(self.split[self.stage:]) == 0:
                    lots = self.qty_open // self.lot        # nothing is scheduled later: everything goes here
                if lots:
                    self._sell(lots * self.lot, self.targets[self.stage - 1], f"T{self.stage}", when, costs)
                if self.status == "OPEN" and self.stage >= 1 and cfg.breakeven_after_t1:
                    self.stop = max(self.stop, self.entry)
            if self.status == "OPEN" and finite(und_close):
                if (und_close < self.und_stop) if up else (und_close > self.und_stop):
                    self._sell(self.qty_open, c * slip, "UNDERLYING INVALIDATED", when, costs)
                elif self.stage >= cfg.trail_after and cfg.trail_mode in ("atr", "swing"):
                    level = None
                    if cfg.trail_mode == "atr" and finite(atr):
                        level = und_close - cfg.trail_atr_mult * atr if up else und_close + cfg.trail_atr_mult * atr
                    elif cfg.trail_mode == "swing":
                        sw = swing_lo if up else swing_hi
                        level = sw if finite(sw) else None
                    if level is not None:
                        if self.und_trail is None:
                            self.und_trail = level
                        else:
                            self.und_trail = max(self.und_trail, level) if up else min(self.und_trail, level)
                    if self.und_trail is not None and ((und_close < self.und_trail) if up else (und_close > self.und_trail)):
                        self._sell(self.qty_open, c * slip, "TRAIL (underlying)", when, costs)
            if self.status == "OPEN" and opposite and cfg.exit_on_opposite:
                self._sell(self.qty_open, c * slip, "OPPOSITE SIGNAL", when, costs)
            if self.status == "OPEN" and minutes_open is not None and minutes_open >= cfg.max_hold_minutes:
                self._sell(self.qty_open, c * slip, "TIME STOP", when, costs)
        return self.fills[before:]

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(net=round(self.net, 2), unrealized=round(self.unrealized, 2), r_multiple=round(self.r_multiple, 2),
                 realized=round(self.realized, 2), charges=round(self.charges, 2))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})
