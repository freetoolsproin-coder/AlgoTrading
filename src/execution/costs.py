"""
Indian index-option charges for paper trades (buying options, premium turnover).

Per executed order:
  brokerage   brokerage_per_order (flat)
  exchange    premium turnover x nse_txn_pct (NSE_FO, NIFTY) or bse_txn_pct (BSE_FO, SENSEX)
  SEBI fee    premium turnover x sebi_per_crore / 1 crore
  STT         sell side only: premium turnover x stt_sell_pct
  stamp duty  buy side only:  premium turnover x stamp_buy_pct
  GST         gst_pct x (brokerage + exchange + SEBI fee)

Rates come from the [COSTS] section of config/scanner.ini (the defaults below if it is missing).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Mapping, Optional

CostFn = Callable[..., float]   # (side, price, qty, exchange="NSE_FO") -> rupees


@dataclass(frozen=True)
class CostConfig:
    brokerage_per_order: float = 20.0
    stt_sell_pct: float = 0.15           # options premium, sell side: 0.15% from 1 April 2026 (0.1% before)
    nse_txn_pct: float = 0.03503
    bse_txn_pct: float = 0.0325
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.003
    gst_pct: float = 18.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, float]]) -> "CostConfig":
        values = values or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in values.items() if k in known})


def order_charges(side: str, price: float, qty: int, exchange: str, costs: CostConfig) -> float:
    """Charges of ONE executed order. side = 'BUY' or 'SELL'."""
    try:
        price, qty = float(price), int(qty)
    except (TypeError, ValueError):
        return 0.0
    if price <= 0 or qty <= 0:
        return 0.0
    turnover = price * qty
    txn_pct = costs.bse_txn_pct if exchange == "BSE_FO" else costs.nse_txn_pct
    brokerage = costs.brokerage_per_order
    exchange_fee = turnover * txn_pct / 100.0
    sebi = turnover * costs.sebi_per_crore / 1e7
    stt = turnover * costs.stt_sell_pct / 100.0 if side == "SELL" else 0.0
    stamp = turnover * costs.stamp_buy_pct / 100.0 if side == "BUY" else 0.0
    gst = (brokerage + exchange_fee + sebi) * costs.gst_pct / 100.0
    return brokerage + exchange_fee + sebi + stt + stamp + gst


def make_cost_fn(values: Optional[Mapping[str, float]] = None) -> CostFn:
    costs = CostConfig.from_mapping(values)

    def fn(side, price, qty, exchange="NSE_FO"):
        return order_charges(side, price, qty, exchange, costs)
    return fn


# =============================================================================
# Intraday equity (MIS) charges - used by the stock pre-breakout planner
# =============================================================================
@dataclass(frozen=True)
class EquityCostConfig:
    """NSE intraday equity. Rates from [COSTS_EQUITY] of config/scanner.ini (check them with your broker)."""
    brokerage_pct: float = 0.1          # % of turnover per order ...
    brokerage_cap: float = 20.0         # ... capped at Rs 20 per order
    stt_sell_pct: float = 0.025         # intraday STT, sell side
    nse_txn_pct: float = 0.00297
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.003
    gst_pct: float = 18.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, float]]) -> "EquityCostConfig":
        values = values or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in values.items() if k in known})


def equity_intraday_charges(side: str, price: float, qty: int, costs: EquityCostConfig) -> float:
    try:
        price, qty = float(price), int(qty)
    except (TypeError, ValueError):
        return 0.0
    if price <= 0 or qty <= 0:
        return 0.0
    turnover = price * qty
    brokerage = min(turnover * costs.brokerage_pct / 100.0, costs.brokerage_cap)
    exchange_fee = turnover * costs.nse_txn_pct / 100.0
    sebi = turnover * costs.sebi_per_crore / 1e7
    stt = turnover * costs.stt_sell_pct / 100.0 if side == "SELL" else 0.0
    stamp = turnover * costs.stamp_buy_pct / 100.0 if side == "BUY" else 0.0
    gst = (brokerage + exchange_fee + sebi) * costs.gst_pct / 100.0
    return brokerage + exchange_fee + sebi + stt + stamp + gst


def make_equity_cost_fn(values: Optional[Mapping[str, float]] = None) -> CostFn:
    costs = EquityCostConfig.from_mapping(values)

    def fn(side, price, qty):
        return equity_intraday_charges(side, price, qty, costs)
    return fn
