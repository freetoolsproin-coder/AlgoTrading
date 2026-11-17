"""
Option contract selection for a CE / PE signal: candidate strikes around ATM from the live option chain,
filters (delta, premium, spread, volume, OI), ranking, and the premium plan (stop / targets / R:R after costs).

The index itself is never treated as tradable: every candidate is a real option contract row from the
chain, carrying its own Upstox instrument key. Greeks missing from the feed are estimated with
Black-Scholes and labelled "estimated".
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.instruments import CE
from src.config.settings import PBConfig
from src.scanner.strategies import UnderlyingPlan
from src.utils.numbers import finite, fnum, r2


@dataclass
class OptionCandidate:
    symbol: str
    instrument_key: Optional[str]
    strike: float
    option_type: str
    expiry: Optional[str]
    ltp: float
    bid: float
    ask: float
    spread_pct: float
    volume: float
    oi: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    greeks_source: str            # 'feed' or 'estimated'
    lot_size: int
    rank_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    # premium plan (filled by option_plan)
    entry_premium: float = np.nan
    sl_premium: float = np.nan
    t1_premium: float = np.nan
    t2_premium: float = np.nan
    rr: float = np.nan
    lots: int = 0
    risk_rupees: float = np.nan
    est_costs: float = np.nan
    ladder: List[float] = field(default_factory=list)   # premium at each [TRADE_PLAN] ladder_r step
    moneyness: str = ""                                 # ITM / ATM / OTM by delta

    def to_dict(self):
        out = {}
        for k, v in asdict(self).items():
            out[k] = r2(v) if isinstance(v, float) else v
        return out



def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_greeks(spot, strike, years, sigma, is_call, rate=0.065):
    """Black-Scholes price, delta, gamma, theta (per day), vega (per 1 vol point)."""
    years = max(years, 1.0 / (365 * 24 * 60))
    sigma = max(sigma, 0.01)
    root = sigma * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * years) / root
    d2 = d1 - root
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    disc = math.exp(-rate * years)
    if is_call:
        price = spot * _ncdf(d1) - strike * disc * _ncdf(d2)
        delta = _ncdf(d1)
        theta = (-spot * pdf * sigma / (2 * math.sqrt(years)) - rate * strike * disc * _ncdf(d2)) / 365.0
    else:
        price = strike * disc * _ncdf(-d2) - spot * _ncdf(-d1)
        delta = _ncdf(d1) - 1.0
        theta = (-spot * pdf * sigma / (2 * math.sqrt(years)) + rate * strike * disc * _ncdf(-d2)) / 365.0
    gamma = pdf / (spot * root)
    vega = spot * pdf * math.sqrt(years) / 100.0
    return price, delta, gamma, theta, vega


def years_to_expiry(now: pd.Timestamp, expiry: Optional[date]) -> float:
    if expiry is None:
        return 7 / 365.0
    end = pd.Timestamp(datetime.combine(expiry, time(15, 30))).tz_localize(now.tz) if now.tz else \
        pd.Timestamp(datetime.combine(expiry, time(15, 30)))
    minutes = max((end - now).total_seconds() / 60.0, 1.0)
    return minutes / (365 * 24 * 60)


def chain_candidates(chain: pd.DataFrame, spot: float, direction: str, step: float, lot_size: int,
                     expiry: Optional[date], now: pd.Timestamp, cfg: PBConfig, iv_fallback: float = np.nan,
                     symbol_prefix: str = "") -> List[OptionCandidate]:
    """chain = market_analysis.chain_frame() layout: one row per strike, ce_*/pe_* columns."""
    if chain is None or chain.empty or not finite(spot):
        return []
    side = "ce" if direction == CE else "pe"
    if not finite(step) or step <= 0:
        strikes = chain["strike"].drop_duplicates().sort_values()
        step = float(strikes.diff().dropna().median()) if len(strikes) > 1 else 50.0
    atm = round(spot / step) * step
    window = chain[(chain["strike"] >= atm - cfg.strikes_each_side * step - 1e-6) &
                   (chain["strike"] <= atm + cfg.strikes_each_side * step + 1e-6)]
    years = years_to_expiry(now, expiry)
    out = []
    for _, row in window.iterrows():
        strike = float(row["strike"])
        ltp = fnum(row.get(f"{side}_ltp"))
        bid, ask = fnum(row.get(f"{side}_bid")), fnum(row.get(f"{side}_ask"))
        iv = fnum(row.get(f"{side}_iv"))
        delta, gamma = fnum(row.get(f"{side}_delta")), fnum(row.get(f"{side}_gamma"))
        theta, vega = fnum(row.get(f"{side}_theta")), fnum(row.get(f"{side}_vega"))
        source = "feed"
        if not (finite(delta) and delta != 0):
            sigma = (iv / 100.0) if finite(iv) and iv > 0 else (iv_fallback / 100.0 if finite(iv_fallback) else 0.15)
            _, delta, gamma, theta, vega = bs_greeks(spot, strike, years, sigma, direction == CE)
            source = "estimated"
        mid = (bid + ask) / 2 if finite(bid) and finite(ask) and bid > 0 and ask > 0 else np.nan
        spread_pct = (ask - bid) / mid * 100 if finite(mid) and mid > 0 else np.nan
        sym = f"{symbol_prefix} {expiry:%d%b%y}".upper() if expiry else symbol_prefix
        out.append(OptionCandidate(
            symbol=f"{sym} {int(strike) if strike == int(strike) else strike} {direction}".strip(),
            instrument_key=row.get(f"{side}_key"), strike=strike, option_type=direction,
            expiry=expiry.isoformat() if expiry else None, ltp=ltp, bid=bid, ask=ask, spread_pct=spread_pct,
            volume=fnum(row.get(f"{side}_volume"), 0.0), oi=fnum(row.get(f"{side}_oi"), 0.0), iv=iv,
            delta=delta, gamma=gamma, theta=theta, vega=vega, greeks_source=source, lot_size=int(lot_size or 0)))
    return out


def rank_options(cands: List[OptionCandidate], cfg: PBConfig, atm_iv: float = np.nan) -> List[OptionCandidate]:
    """Filter and rank. OTM is never preferred just because it is cheaper: delta closeness to the target
    and the spread carry most of the weight."""
    ranked = []
    for c in cands:
        price = c.ask if finite(c.ask) and c.ask > 0 else c.ltp
        d = abs(c.delta) if finite(c.delta) else np.nan
        if not finite(price) or price <= 0:
            c.rejected.append("no price")
        if finite(price) and price < cfg.min_premium:
            c.rejected.append(f"premium {price:.1f} < {cfg.min_premium}")
        if not finite(d) or d < cfg.delta_min or d > cfg.delta_max:
            c.rejected.append(f"delta {d:.2f} outside {cfg.delta_min}-{cfg.delta_max}" if finite(d) else "no delta")
        if not finite(c.spread_pct):
            c.rejected.append("no bid/ask")
        elif c.spread_pct > cfg.max_spread_pct and (c.ask - c.bid) > cfg.max_spread_points:
            c.rejected.append(f"spread {c.spread_pct:.1f}% too wide")
        if c.volume < cfg.min_volume:
            c.rejected.append(f"volume {c.volume:,.0f} < {cfg.min_volume:,.0f}")
        if c.oi < cfg.min_oi:
            c.rejected.append(f"OI {c.oi:,.0f} < {cfg.min_oi:,.0f}")
        delta_fit = 1.0 - min(1.0, abs((d if finite(d) else 0) - cfg.delta_target) / 0.25)
        spread_fit = 1.0 - min(1.0, (c.spread_pct if finite(c.spread_pct) else 10) / max(cfg.max_spread_pct, 0.1))
        liq_fit = min(1.0, math.log10(max(c.volume, 1)) / 6.0) * 0.5 + min(1.0, math.log10(max(c.oi, 1)) / 7.0) * 0.5
        theta_burden = abs(c.theta) / price if finite(c.theta) and finite(price) and price > 0 else 0.1
        theta_fit = 1.0 - min(1.0, theta_burden / 0.25)
        iv_fit = 0.5
        if finite(c.iv) and finite(atm_iv) and atm_iv > 0:
            iv_fit = 1.0 - min(1.0, max(0.0, c.iv / atm_iv - 1.0) / 0.3)
        c.rank_score = round(100 * (0.35 * delta_fit + 0.25 * spread_fit + 0.20 * liq_fit + 0.10 * theta_fit
                                    + 0.10 * iv_fit), 1)
        c.reasons = [f"delta {d:.2f}" if finite(d) else "delta n/a",
                     f"spread {c.spread_pct:.1f}%" if finite(c.spread_pct) else "spread n/a",
                     f"vol {c.volume:,.0f} / OI {c.oi:,.0f}",
                     f"theta {c.theta:.1f}/day" if finite(c.theta) else "theta n/a",
                     f"greeks {c.greeks_source}"]
        ranked.append(c)
    ranked.sort(key=lambda x: (len(x.rejected) > 0, -x.rank_score))
    return ranked


def option_plan(c: OptionCandidate, plan: UnderlyingPlan, cfg: PBConfig, costs_fn: Callable[[str, float, int], float]):
    """Map the underlying plan to premiums with delta + gamma, minus theta for the expected hold."""
    raw = c.ask if finite(c.ask) and c.ask > 0 else (c.ltp * (1 + cfg.assumed_spread_pct / 200.0) if finite(c.ltp) else np.nan)
    if not finite(raw):
        return c
    entry = raw * (1 + cfg.slippage_pct / 100.0)
    d, g = abs(c.delta), (c.gamma if finite(c.gamma) else 0.0)
    th = abs(c.theta) if finite(c.theta) else 0.0
    decay = th * cfg.expected_hold_minutes / 375.0

    def premium_at(move):  # move in the favourable direction (points); long option: + delta*m + gamma*m^2/2
        return max(0.05, entry + d * move + 0.5 * g * move * move - decay)

    sl_est = premium_at(-plan.risk_points)
    sl_floor = entry * (1 - cfg.max_premium_loss_pct / 100.0)
    c.entry_premium = entry
    c.sl_premium = max(sl_est, sl_floor)
    c.t1_premium = premium_at(abs(plan.t1 - plan.entry))
    c.t2_premium = premium_at(abs(plan.t2 - plan.entry))
    try:
        steps, _ = cfg.ladder_steps()
    except (AttributeError, ValueError):
        steps = []
    c.ladder = [round(premium_at(r * plan.risk_points), 2) for r in steps]
    c.moneyness = "ATM" if 0.45 <= d <= 0.55 else ("ITM" if d > 0.55 else "OTM")
    qty = max(c.lot_size, 1)
    c.est_costs = costs_fn("BUY", entry, qty) + costs_fn("SELL", c.t1_premium, qty)
    per_unit_cost = c.est_costs / qty
    risk_u = entry - c.sl_premium + per_unit_cost
    reward_u = c.t1_premium - entry - per_unit_cost
    c.rr = reward_u / risk_u if risk_u > 0 else np.nan
    return c


def select_option(chain: pd.DataFrame, spot: float, direction: str, plan: UnderlyingPlan, step: float,
                  lot_size: int, expiry: Optional[date], now: pd.Timestamp, cfg: PBConfig,
                  costs_fn: Callable[[str, float, int], float], *, iv_fallback: float = np.nan,
                  symbol_prefix: str = "") -> Tuple[Optional[OptionCandidate], List[OptionCandidate], str]:
    """Shortlist for one setup: candidates around ATM, ranked, each with a premium plan. Returns (best, all, note)."""
    cands = chain_candidates(chain, spot, direction, step, lot_size, expiry, now, cfg, iv_fallback, symbol_prefix)
    if not cands:
        return None, [], "option chain unavailable"
    side = "ce" if direction == CE else "pe"
    atm_iv = np.nan
    if chain is not None and not chain.empty and f"{side}_iv" in chain:
        near = chain.iloc[(chain["strike"] - spot).abs().argsort()[:1]]
        atm_iv = fnum(near[f"{side}_iv"].iloc[0]) if len(near) else np.nan
    ranked = rank_options(cands, cfg, atm_iv)
    for c in ranked:
        option_plan(c, plan, cfg, costs_fn)
        if finite(c.rr) and c.rr < cfg.min_rr:
            c.rejected.append(f"R:R {c.rr:.2f} < {cfg.min_rr}")
    ranked.sort(key=lambda x: (len(x.rejected) > 0, -x.rank_score))
    best = ranked[0] if ranked and not ranked[0].rejected else None
    note = "" if best else "no strike passed: " + "; ".join(sorted({r for c in ranked for r in c.rejected})[:4])
    if any(c.greeks_source == "estimated" for c in ranked):
        note = (note + "; " if note else "") + "greeks estimated (Black-Scholes) - feed had none"
    return best, ranked, note
