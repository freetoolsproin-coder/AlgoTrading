"""
Risk engine of the equity service: charges, position sizing and the entry gate.

Position size (cash equity, whole shares, no leverage):

    quantity = floor( capital x risk_per_trade% / risk per share )
    risk per share = |entry - stop| + slippage on both sides (+ gap allowance x ATR for swing trades)

then capped so that quantity x entry stays within max_position_pct of capital and within the free cash.

A new entry is refused (with the reason) when: the kill switch is on, the day's net loss reached max_daily_loss_pct,
the daily trade limit or the consecutive-loss limit is hit, the module's open-position limit is reached, the symbol
already has a position (so nothing is ever averaged down), the planned reward:risk is below 1:min_rr, or reward:risk
after charges and slippage is below 1:min_net_rr.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

from src.config import settings
from src.equity.config import RiskConfig
from src.execution.costs import EquityCostConfig, equity_intraday_charges

KILL_FILE = settings.PROJECT_ROOT / "KILL_SWITCH"       # shared with the other services


@dataclass(frozen=True)
class DeliveryCostConfig:
    brokerage_pct: float = 0.1
    brokerage_cap: float = 20.0
    stt_pct: float = 0.1                # delivery STT, both sides
    nse_txn_pct: float = 0.00297
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.015
    gst_pct: float = 18.0
    dp_charge_sell: float = 15.34       # depository charge per scrip on the sell day (incl. GST)

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, float]]) -> "DeliveryCostConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in (values or {}).items() if k in known})


def delivery_charges(side: str, price: float, qty: int, c: DeliveryCostConfig) -> float:
    if price <= 0 or qty <= 0:
        return 0.0
    turnover = price * qty
    brokerage = min(turnover * c.brokerage_pct / 100.0, c.brokerage_cap)
    exchange = turnover * c.nse_txn_pct / 100.0
    sebi = turnover * c.sebi_per_crore / 1e7
    stamp = turnover * c.stamp_buy_pct / 100.0 if side == "BUY" else 0.0
    dp = c.dp_charge_sell if side == "SELL" else 0.0
    return brokerage + exchange + sebi + turnover * c.stt_pct / 100.0 + stamp + dp + (brokerage + exchange + sebi) * c.gst_pct / 100.0


class Costs:
    """charge(module, side, price, qty) -> rupees for ONE executed order."""

    def __init__(self, intraday: Optional[Mapping[str, float]] = None, delivery: Optional[Mapping[str, float]] = None):
        self.intraday = EquityCostConfig.from_mapping(intraday)
        self.delivery = DeliveryCostConfig.from_mapping(delivery)

    def charge(self, module: str, side: str, price: float, qty: int) -> float:
        if module == "swing":
            return delivery_charges(side, float(price), int(qty), self.delivery)
        return equity_intraday_charges(side, price, qty, self.intraday)

    def round_trip(self, module: str, side: str, entry: float, exit_: float, qty: int) -> float:
        first, second = ("BUY", "SELL") if side == "BUY" else ("SELL", "BUY")
        return self.charge(module, first, entry, qty) + self.charge(module, second, exit_, qty)


@dataclass
class Sizing:
    qty: int
    risk_per_share: float
    risk_rupees: float
    value: float
    charges: float           # estimated round trip at target 2
    rr_gross: float
    rr_net: float
    note: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class RiskManager:
    def __init__(self, cfg: RiskConfig, costs: Optional[Costs] = None, kill_file: Path = KILL_FILE):
        self.cfg = cfg
        self.costs = costs or Costs()
        self.kill_file = Path(kill_file)
        self.kill_flag = False                  # the dashboard's switch (persisted by the engine)

    # ------------------------------------------------------------------ switches / budgets
    def killed(self) -> bool:
        return self.kill_flag or self.kill_file.exists()

    def risk_budget(self) -> float:
        return self.cfg.capital * self.cfg.risk_per_trade_pct / 100.0

    def daily_loss_limit(self) -> float:
        return self.cfg.capital * self.cfg.max_daily_loss_pct / 100.0

    def slip(self, price: float) -> float:
        return price * self.cfg.slippage_bps / 1e4

    # ------------------------------------------------------------------ sizing
    def size(self, module: str, side: str, entry: float, stop: float, target: float, atr: float = 0.0,
             free_cash: Optional[float] = None) -> Sizing:
        dist = abs(entry - stop)
        if not (entry > 0 and dist > 0 and math.isfinite(dist)):
            return Sizing(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "entry / stop not valid")
        per_share = dist + 2 * self.slip(entry)
        if module == "swing" and atr and math.isfinite(atr):
            per_share += self.cfg.gap_risk_atr * atr
        qty = int(self.risk_budget() // per_share)
        cap = self.cfg.capital * self.cfg.max_position_pct / 100.0
        if free_cash is not None:
            cap = min(cap, max(free_cash, 0.0))
        note = ""
        if qty * entry > cap:
            qty, note = int(cap // entry), "capped by position value (cash market, no leverage)"
        if qty < 1:
            return Sizing(0, per_share, 0.0, 0.0, 0.0, 0.0, 0.0,
                          "one share risks more than the budget" if not note else "not enough free cash for one share")
        charges = self.costs.round_trip(module, side, entry, target, qty)
        reward = abs(target - entry) * qty
        risk = dist * qty
        slip = 2 * self.slip(entry) * qty
        rr_net = (reward - charges - slip) / (risk + charges + slip)
        return Sizing(qty, per_share, per_share * qty, qty * entry, charges, reward / risk, rr_net, note)

    # ------------------------------------------------------------------ entry gate
    def check_entry(self, module: str, symbol: str, sizing: Sizing, book: dict) -> Tuple[bool, List[str]]:
        """`book`: {"open": [positions], "realized_today": net rupees, "trades_today": n, "loss_streak": n}."""
        c, why = self.cfg, []
        if self.killed():
            why.append("kill switch is ON")
        if book.get("realized_today", 0.0) <= -self.daily_loss_limit():
            why.append(f"daily loss limit reached ({c.max_daily_loss_pct:g}% of capital)")
        if book.get("trades_today", 0) >= c.daily_trade_limit:
            why.append(f"daily trade limit reached ({c.daily_trade_limit})")
        if c.max_consecutive_losses and book.get("loss_streak", 0) >= c.max_consecutive_losses:
            why.append(f"{book.get('loss_streak')} losing trades in a row - no new entries today")
        open_ = book.get("open", [])
        limit = c.max_open_swing if module == "swing" else c.max_open_intraday
        if sum(1 for p in open_ if p.get("module") == module) >= limit:
            why.append(f"maximum open {module} positions reached ({limit})")
        if any(p.get("symbol") == symbol for p in open_):
            why.append("a position in this stock is already open (no averaging)")
        if sizing.qty < 1:
            why.append(f"position size is zero: {sizing.note}")
        else:
            why += self.rr_reasons(sizing)
        return not why, why

    def rr_reasons(self, sizing: Sizing) -> List[str]:
        c, why = self.cfg, []
        if sizing.rr_gross < c.min_rr - 1e-9:
            why.append(f"reward:risk 1:{sizing.rr_gross:.2f} is below 1:{c.min_rr:g}")
        elif sizing.rr_net < c.min_net_rr:
            why.append(f"reward:risk after charges 1:{sizing.rr_net:.2f} is below 1:{c.min_net_rr:g} (stop too tight for the costs)")
        return why

    def limits_breached(self, book: dict) -> List[str]:
        """Account-level blocks only (used for the banner and the 'risk limit reached' alert)."""
        ok, why = self.check_entry("intraday", "", Sizing(1, 1, 1, 1, 0, 9, 9), {**book, "open": []})
        return why

    def to_dict(self, book: Optional[dict] = None) -> dict:
        c = self.cfg
        out = {"capital": c.capital, "risk_per_trade_pct": c.risk_per_trade_pct, "risk_budget": round(self.risk_budget(), 2),
               "max_daily_loss_pct": c.max_daily_loss_pct, "daily_loss_limit": round(self.daily_loss_limit(), 2),
               "max_open_intraday": c.max_open_intraday, "max_open_swing": c.max_open_swing, "min_rr": c.min_rr, "min_net_rr": c.min_net_rr,
               "daily_trade_limit": c.daily_trade_limit, "max_consecutive_losses": c.max_consecutive_losses,
               "max_position_pct": c.max_position_pct, "slippage_bps": c.slippage_bps, "kill_switch": self.killed(),
               "kill_file": self.kill_file.exists()}
        if book is not None:
            out.update(realized_today=round(book.get("realized_today", 0.0), 2), trades_today=book.get("trades_today", 0),
                       loss_streak=book.get("loss_streak", 0), blocked=self.limits_breached(book))
        return out
