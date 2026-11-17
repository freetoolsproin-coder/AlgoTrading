"""
Option contract selection from the CURRENT option chain, and the trade plan on the traded premium.

select_contract   filters every strike of the chosen side (two-sided quote, spread, premium, OI, volume, IV,
                  delta band), then picks by delta or by moneyness (ATM / ITM / OTM +- strike_distance steps).
                  Premiums are never predicted from the underlying: the entry is the live ask of a real row.
build_plan        entry, stop-loss and T1 / T2 / T3 on the option premium, the underlying invalidation level,
                  whole-lot quantity for the risk budget, maximum estimated loss and R:R after charges.

R = entry premium - stop premium (the initial risk per unit). Targets are multiples of R.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.execution.costs import CostConfig, order_charges
from src.fno.config import FnoConfig
from src.fno.universe import Underlying
from src.stocks.analytics import bs_greeks
from src.utils.numbers import finite, fnum

TZ = "Asia/Kolkata"


def years_to_expiry(expiry: date, now: pd.Timestamp) -> float:
    end = pd.Timestamp(datetime.combine(expiry, time(15, 30))).tz_localize(TZ)
    return max((end - now).total_seconds() / (365.0 * 86400.0), 1.0 / (365.0 * 24.0))


def round_tick(price: float, tick: float, up: bool = False) -> float:
    tick = tick if tick and tick > 0 else 0.05
    n = price / tick
    return round((math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)) * tick, 2)


@dataclass
class Contract:
    symbol: str
    side: str                 # CE | PE
    strike: float
    expiry: date
    key: str
    lot: int
    segment: str
    ltp: float
    bid: float
    ask: float
    spread_pct: float
    oi: float
    volume: float
    iv: float
    delta: float
    gamma: float
    greeks_estimated: bool
    spot: float
    moneyness: str            # ITM | ATM | OTM
    px_chg_pct: float = float("nan")
    tick: float = 0.05
    theta: float = float("nan")   # premium lost per day, per unit
    vega: float = float("nan")    # premium change for one IV point, per unit

    @property
    def label(self) -> str:
        strike = int(self.strike) if float(self.strike).is_integer() else self.strike
        return f"{self.symbol} {self.expiry:%d %b %y} {strike} {self.side}".upper()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["expiry"] = self.expiry.isoformat()
        d["label"] = self.label
        return d


def select_contract(chain: Optional[pd.DataFrame], spot: float, side: str, und: Underlying, expiry: date,
                    cfg: FnoConfig, now: pd.Timestamp) -> Tuple[Optional[Contract], List[str]]:
    """(contract, reasons). `reasons` explains a rejection, or lists what was filtered out on the way."""
    if chain is None or chain.empty:
        return None, ["no option chain"]
    if not finite(spot):
        return None, ["no underlying price"]
    lot = und.lot_size(expiry)
    if lot <= 0:
        return None, [f"no lot size for {expiry}"]
    s = side.lower()
    step = und.strike_step(expiry, spot)
    strikes = chain["strike"].to_numpy(dtype=float)
    atm = float(strikes[np.argmin(np.abs(strikes - spot))])
    years = years_to_expiry(expiry, now)
    rows, dropped = [], {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for _, r in chain.iterrows():
        strike, key = float(r["strike"]), r.get(f"{s}_key")
        bid, ask, ltp = fnum(r.get(f"{s}_bid")), fnum(r.get(f"{s}_ask")), fnum(r.get(f"{s}_ltp"))
        if not key or not isinstance(key, str):
            continue
        if not (finite(bid) and finite(ask) and bid > 0 and ask >= bid):
            drop("no two-sided quote")
            continue
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / mid * 100.0
        if spread > cfg.max_spread_pct:
            drop(f"spread > {cfg.max_spread_pct:g}%")
            continue
        if ask < cfg.min_premium or (cfg.max_premium > 0 and ask > cfg.max_premium):
            drop("premium outside limits")
            continue
        oi, vol = fnum(r.get(f"{s}_oi"), 0.0), fnum(r.get(f"{s}_volume"), 0.0)
        if oi / lot < cfg.min_oi:
            drop(f"OI < {cfg.min_oi:g} lots")
            continue
        if vol / lot < cfg.min_volume:
            drop(f"volume < {cfg.min_volume:g} lots")
            continue
        iv = fnum(r.get(f"{s}_iv"))
        if finite(iv) and iv > 0 and not (cfg.min_iv <= iv <= cfg.max_iv):
            drop("IV outside limits")
            continue
        delta, gamma, estimated = fnum(r.get(f"{s}_delta")), fnum(r.get(f"{s}_gamma")), False
        if not (finite(delta) and delta != 0 and finite(gamma)):
            if not (finite(iv) and iv > 0):
                drop("no greeks and no IV")
                continue
            g = bs_greeks(spot, strike, years, iv / 100.0, side)
            delta, gamma, estimated = g["delta"], g["gamma"], True
        theta, vega = fnum(r.get(f"{s}_theta")), fnum(r.get(f"{s}_vega"))
        if not (finite(theta) and finite(vega)) and finite(iv) and iv > 0:
            g = bs_greeks(spot, strike, years, iv / 100.0, side)
            theta = theta if finite(theta) else g["theta"]
            vega = vega if finite(vega) else g["vega"]
        if cfg.selection == "delta" and not (cfg.delta_min <= abs(delta) <= cfg.delta_max):
            drop("delta outside band")
            continue
        close = fnum(r.get(f"{s}_close"))
        diff = (strike - atm) / step if finite(step) and step > 0 else 0.0
        money = "ATM" if abs(diff) < 0.5 else ("OTM" if (diff > 0) == (side == "CE") else "ITM")
        rows.append(Contract(symbol=und.symbol, side=side, strike=strike, expiry=expiry, key=key, lot=lot,
                             segment=und.segment, ltp=ltp, bid=bid, ask=ask, spread_pct=round(spread, 2), oi=oi,
                             volume=vol, iv=iv, delta=float(delta), gamma=float(gamma), greeks_estimated=estimated,
                             spot=float(spot), moneyness=money, tick=und.tick, theta=float(theta), vega=float(vega),
                             px_chg_pct=(ltp / close - 1) * 100 if finite(close) and close > 0 and finite(ltp) else float("nan")))
    notes = [f"{n} strikes: {why}" for why, n in sorted(dropped.items(), key=lambda kv: -kv[1])]
    if not rows:
        return None, ["no liquid contract passes the filters"] + notes
    if cfg.selection == "delta":
        def score(c):
            return abs(abs(c.delta) - cfg.target_delta) + 0.02 * c.spread_pct
        return min(rows, key=score), notes
    sign = {"ATM": 0, "OTM": 1, "ITM": -1}.get(cfg.moneyness.upper(), 0) * (1 if side == "CE" else -1)
    target = atm + sign * cfg.strike_distance * (step if finite(step) else 0.0)
    best = min(rows, key=lambda c: (abs(c.strike - target), c.spread_pct))
    if finite(step) and step > 0 and abs(best.strike - target) > 2 * step:
        return None, [f"no liquid contract within 2 strikes of {target:g}"] + notes
    return best, notes


# =============================================================================
# Trade plan
# =============================================================================
@dataclass
class Plan:
    symbol: str
    instrument: str
    side: str
    strike: float
    expiry: str
    key: str
    segment: str
    lot: int
    entry: float
    stop: float
    t1: float
    t2: float
    t3: float
    r: float                      # initial risk per unit (premium points)
    rr: float                     # reward : risk at T2 after charges
    qty: int
    lots: int
    max_loss: float               # rupees, including charges and exit slippage
    capital_used: float
    und_entry: float
    und_stop: float               # underlying invalidation level
    und_t1: float
    stop_basis: str
    confidence: float
    strategy: str
    strategy_name: str
    timestamp: str
    exits: List[str] = field(default_factory=list)
    contract: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    direction: str = ""           # Bullish | Bearish
    timeframe: str = ""           # e.g. 5-minute
    signal_status: str = "PAPER TRADE"
    plan_mode: str = "r"          # r | percent
    stop_pct: float = 0.0         # stop and targets as a % of the entry premium
    t1_pct: float = 0.0
    t2_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def premium_risk(delta: float, gamma: float, und_distance: float) -> float:
    """Premium lost by a long option when the underlying moves `und_distance` against it (delta + gamma)."""
    d = abs(delta) * und_distance - 0.5 * abs(gamma) * und_distance ** 2
    return max(d, 0.3 * abs(delta) * und_distance)


def lots_split(lots: int, book_pct: tuple, single_lot_exit: int = 2) -> List[int]:
    """Whole lots booked at T1 / T2 / T3."""
    if lots <= 0:
        return [0, 0, 0]
    if lots == 1:
        out = [0, 0, 0]
        out[min(max(single_lot_exit, 1), 3) - 1] = 1
        return out
    total = float(sum(book_pct)) or 1.0
    a = max(1, int(round(lots * book_pct[0] / total)))
    b = max(1, int(round(lots * book_pct[1] / total))) if lots - a >= 2 or book_pct[2] == 0 else max(0, lots - a - 1)
    b = min(b, lots - a)
    return [a, b, lots - a - b]


def plan_split(lots: int, cfg) -> List[int]:
    """Lots booked at T1 / T2 / T3 for the configured plan. Percent plans have two targets: nothing is left for T3."""
    if str(getattr(cfg, "plan_mode", "r")).lower() == "percent":
        return lots_split(lots, (cfg.book_pct[0], cfg.book_pct[1] + cfg.book_pct[2], 0.0), min(cfg.single_lot_exit, 2))
    return lots_split(lots, cfg.book_pct, cfg.single_lot_exit)


def build_plan(signal: dict, und: Underlying, contract: Contract, bar: pd.Series, cfg: FnoConfig, capital: float,
               now: pd.Timestamp, risk_mult: float = 1.0, premium_atr: float = float("nan"),
               fixed_lots: int = 0) -> Tuple[Optional[Plan], str]:
    """(plan, "") or (None, reason). `bar` is the last completed base-timeframe row of the enriched frame."""
    side, spot, atr = contract.side, contract.spot, fnum(bar.get("atr"))
    if not finite(atr) or atr <= 0:
        return None, "ATR not ready"
    up = side == "CE"
    dist, basis = cfg.sl_atr_mult * atr, f"{cfg.sl_atr_mult:g} x ATR"
    if cfg.use_structure_stop:
        swing = fnum(bar.get("swing_lo" if up else "swing_hi"))
        if finite(swing):
            s_dist = (spot - swing if up else swing - spot) + 0.1 * atr
            if s_dist > 0:
                dist, basis = s_dist, "beyond the last confirmed swing " + ("low" if up else "high")
    lo, hi = cfg.min_sl_atr * atr, cfg.max_sl_atr * atr
    if dist < lo:
        dist, basis = lo, f"{basis}, widened to {cfg.min_sl_atr:g} x ATR"
    elif dist > hi:
        dist, basis = hi, f"{basis}, capped at {cfg.max_sl_atr:g} x ATR"
    und_stop = spot - dist if up else spot + dist

    entry = round_tick(contract.ask * (1 + cfg.slippage_pct / 100.0), contract.tick, up=True)
    risk = premium_risk(contract.delta, contract.gamma, dist)
    notes = []
    percent = str(cfg.plan_mode).lower() == "percent"
    floor = entry * cfg.min_premium_sl_pct / 100.0
    if finite(premium_atr) and premium_atr > 0:
        floor = max(floor, cfg.premium_atr_mult * premium_atr)
    if risk < floor:
        risk = floor
        notes.append("premium stop widened to the minimum (premium volatility)")
    cap = entry * cfg.max_premium_sl_pct / 100.0
    if risk > cap:
        risk = cap
        notes.append(f"premium stop capped at {cfg.max_premium_sl_pct:g}% of the premium (tighter than the underlying level)")
    if percent:                                               # fixed % of the live premium: SL / T1 / T2
        notes = [f"stop-loss {cfg.sl_pct:g}% below, targets {cfg.target_pct[0]:g}% and {cfg.target_pct[1]:g}% above the entry premium"]
        risk = entry * cfg.sl_pct / 100.0
    stop = round_tick(entry - risk, contract.tick)
    r = round(entry - stop, 2)
    if r <= 0 or stop <= 0:
        return None, "premium too small for a valid stop"
    if percent:
        t1, t2 = (round_tick(entry * (1 + k / 100.0), contract.tick) for k in cfg.target_pct)
        t3 = t2
        if not entry < t1 < t2:
            return None, "premium too small for separate targets"
    else:
        t1, t2, t3 = (round_tick(entry + k * r, contract.tick) for k in cfg.target_r)
    costs = CostConfig.from_mapping(cfg.costs)
    lot = contract.lot

    def round_trip(qty, exit_price, exits=1):
        return order_charges("BUY", entry, qty, contract.segment, costs) + \
            exits * order_charges("SELL", exit_price, max(qty // exits, 1), contract.segment, costs)

    exit_slip = stop * cfg.slippage_pct / 100.0
    per_lot = (r + exit_slip) * lot + round_trip(lot, stop)
    budget = capital * cfg.risk_per_trade_pct / 100.0 * risk_mult
    lots = fixed_lots or int(budget // per_lot)
    if lots < 1:
        if cfg.skip_if_one_lot_exceeds_risk:
            return None, f"one lot risks Rs {per_lot:,.0f}, above the Rs {budget:,.0f} risk budget"
        lots = 1
        notes.append("one lot exceeds the risk budget")
    if und.freeze_qty and lots * lot > und.freeze_qty:
        lots = int(und.freeze_qty // lot)
    qty = lots * lot
    max_loss = (r + exit_slip) * qty + round_trip(qty, stop)
    reward = (t2 - entry) * qty - round_trip(qty, t2, 2)
    rr = reward / max_loss if max_loss > 0 else 0.0
    if rr < cfg.min_rr_after_costs:
        return None, f"reward:risk after charges {rr:.2f} < {cfg.min_rr_after_costs:g}"
    split = plan_split(lots, cfg)
    und_t1 = spot + ((t1 - entry) / max(abs(contract.delta), 0.05)) * (1 if up else -1)
    exits = [f"stop-loss at premium {stop:g} (initial risk R = {r:g})",
             (f"book {split[0]} lot(s) at T1 {t1:g} and the remaining {split[1]} at T2 {t2:g}" if percent else
              f"book {split[0]} / {split[1]} / {split[2]} lot(s) at T1 {t1:g} / T2 {t2:g} / T3 {t3:g}"),
             ("move the stop to entry after T1" if cfg.breakeven_after_t1 else "stop stays at its level after T1"),
             {"atr": f"after T{cfg.trail_after}: trail {cfg.trail_atr_mult:g} x ATR behind the underlying's best close",
              "swing": f"after T{cfg.trail_after}: trail behind the last confirmed swing of the underlying",
              "off": "no trailing"}.get(cfg.trail_mode, "no trailing"),
             f"exit if the underlying closes {'below' if up else 'above'} {und_stop:.2f} (invalidation)",
             f"time stop after {cfg.max_hold_minutes} min; square-off at {cfg.square_off:%H:%M}"]
    if cfg.exit_on_opposite:
        exits.append("exit on a confirmed opposite signal")
    plan = Plan(symbol=und.symbol, instrument=contract.label, side=side, strike=contract.strike,
                expiry=contract.expiry.isoformat(), key=contract.key, segment=contract.segment, lot=lot, entry=entry,
                stop=stop, t1=t1, t2=t2, t3=t3, r=r, rr=round(rr, 2), qty=qty, lots=lots, max_loss=round(max_loss, 2),
                capital_used=round(entry * qty, 2), und_entry=round(spot, 2), und_stop=round(und_stop, 2),
                und_t1=round(und_t1, 2), stop_basis=basis, confidence=float(signal.get("confidence", 0.0)),
                strategy=signal["strategy"], strategy_name=signal.get("strategy_name", signal["strategy"]),
                timestamp=now.isoformat(), exits=exits, contract=contract.to_dict(), notes=notes,
                direction="Bullish" if up else "Bearish", timeframe=f"{cfg.base_tf}-minute", plan_mode="percent" if percent else "r",
                stop_pct=round((entry - stop) / entry * 100.0, 1), t1_pct=round((t1 - entry) / entry * 100.0, 1),
                t2_pct=round((t2 - entry) / entry * 100.0, 1))
    return plan, ""
