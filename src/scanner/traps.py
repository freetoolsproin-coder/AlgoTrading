"""
False-breakout and trap detection (v3 layer). For one side of one completed signal bar it lists the trap
conditions present, each with penalty points:

    breakout without volume       futures relative volume below [CONFIRMATION] min_rvol on the breakout bar
    breakout with weak momentum   MACD histogram or RSI slope against the breakout
    immediate rejection           the previous bar closed beyond the level and this one closed back inside
    into higher-timeframe level   the next zone (PDH / PWH / swing / OR ...) is within 0.5 ATR beyond the level
    liquidity sweep against       the opposite side's sweep module fired in the last bars
    failed VWAP reclaim / break   accepted on the trade side of VWAP within 3 bars, back on the wrong side now
    IV spike                      ATM IV up more than iv_spike_pct in ~30 minutes (premium already expensive)
    extended                      stretched > stretched_atr ATR from EMA20, or > 2 VWAP deviations
    poor reward-to-risk           room to the next level < min_rr x the planned underlying risk
    low-liquidity options         ATM spread wider than max_spread_pct
    divergence against            regular RSI divergence against the trade

[STRATEGY_V3] trap_action: report (listed only) / penalize (points subtracted from the v3 score; an entry is
refused once the total reaches trap_block_points) / block (any trap refuses the entry).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.config.instruments import CE
from src.utils.numbers import finite


@dataclass
class Trap:
    name: str
    points: float
    note: str

    def to_dict(self):
        return {"name": self.name, "points": self.points, "note": self.note}


def detect(direction: str, ev, d, ext, modules: Dict[str, object], deriv, cfg) -> List[Trap]:
    from src.scanner import strategies as strat   # local import: strategies imports nothing from here
    up = direction == CE
    s = 1.0 if up else -1.0
    pos, ff = ev.position, ext.ff
    a = ff.arrays
    traps: List[Trap] = []
    atr_v, close = ev.atr, ev.close
    lvl = d.level_price
    rvol = ff.v("f_rvol", pos)
    hist, rsi_s = ff.v("macd_hist", pos), ff.v("rsi_slope", pos)
    if d.breakout:
        if ff.has_participation and finite(rvol) and rvol < cfg.min_rvol:
            traps.append(Trap("breakout without volume", 8, f"futures volume {rvol:.1f}x on the breakout bar"))
        if (finite(hist) and s * hist <= 0) or (finite(rsi_s) and s * rsi_s < 0):
            traps.append(Trap("breakout with weak momentum", 6, "MACD histogram / RSI slope against the break"))
    if finite(lvl) and pos >= 1:
        pc = a["close"][pos - 1]
        if s * (pc - lvl) > 0 and s * (close - lvl) < 0:
            traps.append(Trap("immediate rejection", 12, f"closed back inside {lvl:,.1f} one bar after breaking it"))
    if d.next_level is not None and finite(d.next_level) and finite(lvl) and finite(atr_v) and atr_v > 0:
        gap = s * (d.next_level - lvl) / atr_v
        if 0 <= gap < 0.5:
            traps.append(Trap("into higher-timeframe level", 8,
                              f"next level {d.next_level:,.1f} only {gap:.2f} ATR beyond the breakout level"))
    opp = "pe" if up else "ce"
    sweep_mod = modules.get("Liquidity sweep")
    if sweep_mod is not None:
        other = getattr(sweep_mod, opp)
        if other.evidence and any("swept" in e for e in other.evidence):
            traps.append(Trap("liquidity sweep against", 10, other.evidence[0]))
    acc_now = ext.v("vwap_accept", pos)
    if finite(acc_now) and np.sign(acc_now) == -s:
        prior = [ext.v("vwap_accept", p) for p in range(max(0, pos - 3), pos)]
        if any(finite(x) and np.sign(x) == s and abs(x) >= 1 for x in prior):
            traps.append(Trap("failed VWAP " + ("reclaim" if up else "breakdown"), 8,
                              "was on the trade side of VWAP within 3 bars, back on the wrong side now"))
    if deriv is not None and getattr(deriv, "available", False) and finite(deriv.atm_iv_change_pct) \
            and deriv.atm_iv_change_pct > cfg.iv_spike_pct:
        traps.append(Trap("IV spike", 8, f"ATM IV +{deriv.atm_iv_change_pct:.0f}% in ~30 min"))
    e20 = ff.v("ema20", pos)
    vw, price, dev = ext.v("vwap", pos), ext.v("vwap_price", pos), ext.v("vwap_dev", pos)
    if finite(e20) and finite(atr_v) and atr_v > 0 and s * (close - e20) / atr_v > cfg.stretched_atr:
        traps.append(Trap("extended", 10, f"{s * (close - e20) / atr_v:.1f} ATR from EMA20"))
    elif finite(vw) and finite(dev) and dev > 0 and s * (price - vw) / dev > 2.0:
        traps.append(Trap("extended", 10, f"{s * (price - vw) / dev:.1f} deviations beyond VWAP"))
    if finite(lvl) and d.next_level is not None and finite(d.next_level):
        plan = strat.underlying_plan(direction, d, close, atr_v, cfg)
        room = s * (d.next_level - close)
        if plan.risk_points > 0 and room / plan.risk_points < cfg.min_rr:
            traps.append(Trap("poor reward-to-risk", 8,
                              f"room {room:,.0f} pts to the next level vs risk {plan.risk_points:,.0f} pts"))
    if deriv is not None and getattr(deriv, "available", False) and finite(deriv.atm_spread_pct) \
            and deriv.atm_spread_pct > cfg.max_spread_pct:
        traps.append(Trap("low-liquidity options", 6, f"ATM spread {deriv.atm_spread_pct:.1f}%"))
    mom = modules.get("Momentum")
    if mom is not None and any("divergence against" in x for x in mom.side(direction).against):
        traps.append(Trap("divergence against", 5, "regular RSI divergence against the trade"))
    return traps


def verdict(traps: List[Trap], cfg) -> (float, Optional[str]):
    """(points to subtract, veto reason or None) according to [STRATEGY_V3] trap_action."""
    total = float(sum(t.points for t in traps))
    if not traps or cfg.trap_action == "report":
        return 0.0, None
    names = ", ".join(t.name for t in traps)
    if cfg.trap_action == "block":
        return total, f"trap: {names}"
    veto = f"trap points {total:.0f} >= {cfg.trap_block_points:.0f} ({names})" if total >= cfg.trap_block_points else None
    return total, veto
