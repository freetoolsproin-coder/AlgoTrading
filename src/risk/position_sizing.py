"""
Position sizing in whole lots from the premium stop, the real lot size and the risk budget.

    budget       capital x risk_per_trade_pct %
    1-lot risk   (entry premium - premium stop + costs per unit) x lot size
    lots         floor(budget / 1-lot risk), also capped by max_premium_pct_of_capital
    0 lots       the trade is skipped (one lot already risks more than the budget): never rounded up
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from src.utils.numbers import finite

if TYPE_CHECKING:  # imported for type hints only (avoids an import cycle with options.selection)
    from src.config.settings import PBConfig
    from src.options.selection import OptionCandidate


def risk_budget(cfg: "PBConfig") -> float:
    return cfg.capital * cfg.risk_per_trade_pct / 100.0


def size_in_lots(c: "OptionCandidate", cfg: "PBConfig", mult: float = 1.0) -> Tuple[int, float, str]:
    """(lots, rupees at risk, reason when 0 lots). `mult` scales the budget (early entries: early_risk_mult).
    Quantity = floor(budget / risk per lot) x lot size - never rounded up, never increased after a loss."""
    budget = risk_budget(cfg) * max(0.0, min(1.0, mult))
    if not finite(c.entry_premium) or not finite(c.sl_premium) or c.lot_size <= 0:
        return 0, np.nan, "no premium / lot size"
    per_unit = c.entry_premium - c.sl_premium + (c.est_costs / max(c.lot_size, 1) if finite(c.est_costs) else 0)
    one_lot = per_unit * c.lot_size
    if one_lot <= 0:
        return 0, np.nan, "invalid stop"
    lots = int(budget // one_lot)
    cap_lots = int((cfg.capital * cfg.max_premium_pct_of_capital / 100.0) // (c.entry_premium * c.lot_size))
    lots = min(lots, cap_lots)
    if lots < 1:
        return 0, one_lot, (f"1 lot risks Rs {one_lot:,.0f} > budget Rs {budget:,.0f}" if one_lot > budget
                            else "premium budget too small for 1 lot")
    return lots, lots * one_lot, ""
