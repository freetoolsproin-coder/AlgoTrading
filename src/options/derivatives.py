"""
Derivatives intelligence from the LIVE option chain (Strategy 7 of the v3 layer).

What is measured (observed from the chain of the traded expiry)
    pcr_oi            total put OI / total call OI over ATM +/- `width` strikes
    pcr_oi_change     put OI added / call OI added today over the same strikes (None when neither changed)
    call_wall         strike with the largest call OI at or above spot; put_wall: largest put OI at or below
    *_dist_atr        distance from spot to those walls in 5-minute ATR
    atm_iv            mean of the ATM call and put IV; atm_spread_pct; atm_volume
    atm_ce_flow       price change vs OI change of the ATM call (and atm_pe_flow for the put):
                      "price up, OI up" / "price up, OI down" / ... - described, NOT labelled as who traded

How it is used (cautiously)
    Public OI cannot tell who opened a position or why. A large call-OI strike is commonly *read* as resistance
    (call writers defending it) - that reading is an assumption. So this module only adds small amounts of
    evidence, never decides on its own, and every note says "suggests", not "shows".
    IV percentile needs a history of daily ATM IV. It is kept in data/prebreakout.db (iv_daily) and reported as
    unavailable until at least 20 sessions exist - it is never invented.

Not available in the backtest: there are no historical option chains in this project, so the family is marked
"no data" there and the v3 score is rescaled to the other families (the same rule as v1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.numbers import finite, fnum


@dataclass
class DerivSnapshot:
    available: bool = False
    note: str = ""
    spot: float = np.nan
    pcr_oi: float = np.nan
    pcr_oi_change: float = np.nan
    call_wall: float = np.nan
    put_wall: float = np.nan
    call_wall_oi: float = np.nan
    put_wall_oi: float = np.nan
    call_wall_dist_atr: float = np.nan
    put_wall_dist_atr: float = np.nan
    atm_strike: float = np.nan
    atm_iv: float = np.nan
    atm_iv_change_pct: float = np.nan     # vs ~30 minutes earlier (filled by the engine from its own IV history)
    iv_percentile: float = np.nan          # vs daily ATM IV history (>= 20 sessions) or NaN
    iv_history_days: int = 0
    atm_spread_pct: float = np.nan
    atm_volume: float = np.nan
    atm_ce_flow: str = ""
    atm_pe_flow: str = ""

    def to_dict(self):
        out = {}
        for k, v in asdict(self).items():
            out[k] = (round(v, 4) if isinstance(v, float) and np.isfinite(v) else (None if isinstance(v, float) else v))
        return out


def _flow(px_chg: float, oi_chg: float) -> str:
    if not (finite(px_chg) and finite(oi_chg)) or (px_chg == 0 and oi_chg == 0):
        return ""
    return f"price {'up' if px_chg > 0 else 'down'}, OI {'up' if oi_chg > 0 else 'down'}"


def analyse_chain(chain: Optional[pd.DataFrame], spot: float, atr_v: float, step: float = np.nan,
                  width: int = 10) -> DerivSnapshot:
    snap = DerivSnapshot(spot=spot)
    if chain is None or chain.empty or not finite(spot):
        snap.note = "option chain unavailable"
        return snap
    c = chain.copy()
    c["strike"] = pd.to_numeric(c["strike"], errors="coerce")
    c = c.dropna(subset=["strike"]).sort_values("strike")
    if c.empty:
        snap.note = "option chain empty"
        return snap
    if not finite(step) or step <= 0:
        diffs = c["strike"].diff().dropna()
        step = float(diffs.median()) if len(diffs) else 50.0
    atm = float(c.iloc[(c["strike"] - spot).abs().argsort().iloc[0]]["strike"])
    win = c[(c["strike"] >= atm - width * step - 1e-6) & (c["strike"] <= atm + width * step + 1e-6)]
    col = lambda name: pd.to_numeric(win[name], errors="coerce") if name in win else pd.Series(np.nan, index=win.index)
    ce_oi, pe_oi = col("ce_oi"), col("pe_oi")
    if ce_oi.fillna(0).sum() <= 0 or pe_oi.fillna(0).sum() <= 0:
        snap.note = "chain has no open interest"
        return snap
    snap.available = True
    snap.atm_strike = atm
    snap.pcr_oi = float(pe_oi.sum() / ce_oi.sum())
    ce_chg, pe_chg = col("ce_oi_chg"), col("pe_oi_chg")
    if ce_chg.notna().any() and pe_chg.notna().any():
        add_c, add_p = float(ce_chg.clip(lower=0).sum()), float(pe_chg.clip(lower=0).sum())
        if add_c > 0:
            snap.pcr_oi_change = add_p / add_c
    above = win[win["strike"] >= spot]
    below = win[win["strike"] <= spot]
    if len(above) and pd.to_numeric(above["ce_oi"], errors="coerce").notna().any():
        r = above.loc[pd.to_numeric(above["ce_oi"], errors="coerce").idxmax()]
        snap.call_wall, snap.call_wall_oi = float(r["strike"]), fnum(r["ce_oi"])
    if len(below) and pd.to_numeric(below["pe_oi"], errors="coerce").notna().any():
        r = below.loc[pd.to_numeric(below["pe_oi"], errors="coerce").idxmax()]
        snap.put_wall, snap.put_wall_oi = float(r["strike"]), fnum(r["pe_oi"])
    if finite(atr_v) and atr_v > 0:
        if finite(snap.call_wall):
            snap.call_wall_dist_atr = (snap.call_wall - spot) / atr_v
        if finite(snap.put_wall):
            snap.put_wall_dist_atr = (spot - snap.put_wall) / atr_v
    row = c[(c["strike"] - atm).abs() < 1e-6].iloc[0]
    ivs = [fnum(row.get("ce_iv")), fnum(row.get("pe_iv"))]
    ivs = [v for v in ivs if finite(v) and v > 0]
    snap.atm_iv = float(np.mean(ivs)) if ivs else np.nan
    spreads = []
    for side in ("ce", "pe"):
        bid, ask = fnum(row.get(f"{side}_bid")), fnum(row.get(f"{side}_ask"))
        if finite(bid) and finite(ask) and bid > 0 and ask > 0:
            spreads.append((ask - bid) / ((ask + bid) / 2) * 100)
    snap.atm_spread_pct = float(np.mean(spreads)) if spreads else np.nan
    snap.atm_volume = fnum(row.get("ce_volume"), 0.0) + fnum(row.get("pe_volume"), 0.0)
    snap.atm_ce_flow = _flow(fnum(row.get("ce_px_chg")), fnum(row.get("ce_oi_chg")))
    snap.atm_pe_flow = _flow(fnum(row.get("pe_px_chg")), fnum(row.get("pe_oi_chg")))
    return snap


def evidence(snap: Optional[DerivSnapshot], up: bool, iv_spike_pct: float = 10.0):
    """(fraction 0..1, supporting, contradicting, notes) for one side. Small, cautious contributions only."""
    sup: List[str] = []
    con: List[str] = []
    notes: List[str] = []
    if snap is None or not snap.available:
        return None, sup, con, [snap.note if snap is not None else "option chain unavailable"]
    pts = 0.0
    wall_d = snap.call_wall_dist_atr if up else snap.put_wall_dist_atr
    back_d = snap.put_wall_dist_atr if up else snap.call_wall_dist_atr
    wall_name = "call-OI wall" if up else "put-OI wall"
    wall = snap.call_wall if up else snap.put_wall
    if finite(wall_d):
        if wall_d >= 1.0:
            pts += 0.30
            sup.append(f"largest {wall_name} {wall:,.0f} is {wall_d:.1f} ATR away (room to move)")
        elif wall_d < 0.5:
            con.append(f"{wall_name} {wall:,.0f} only {wall_d:.1f} ATR away (OI suggests resistance)" if up else
                       f"{wall_name} {wall:,.0f} only {wall_d:.1f} ATR away (OI suggests support)")
        else:
            pts += 0.10
    if finite(back_d) and back_d <= 1.5:
        pts += 0.15
        notes.append(("put" if up else "call") + f"-OI concentration {back_d:.1f} ATR behind price")
    chg = snap.pcr_oi_change
    if finite(chg):
        if (up and chg >= 1.2) or (not up and chg <= 1 / 1.2):
            pts += 0.20
            sup.append(f"OI added today: put/call {chg:.2f} (suggests " + ("support building)" if up else "resistance building)"))
        elif (up and chg <= 1 / 1.2) or (not up and chg >= 1.2):
            con.append(f"OI added today: put/call {chg:.2f} leans the other way")
        else:
            pts += 0.05
    flow = snap.atm_ce_flow if up else snap.atm_pe_flow
    if flow == "price up, OI up":
        pts += 0.15
        sup.append(("ATM call" if up else "ATM put") + ": price up with OI up (fresh positions; who opened them is unknown)")
    elif flow == "price up, OI down":
        pts += 0.10
        notes.append(("ATM call" if up else "ATM put") + ": price up, OI down (closing / covering)")
    if finite(snap.atm_iv_change_pct):
        if snap.atm_iv_change_pct > iv_spike_pct:
            con.append(f"ATM IV +{snap.atm_iv_change_pct:.0f}% in ~30 min (premium already expensive)")
        else:
            pts += 0.20
    else:
        pts += 0.10
        notes.append("IV change: not enough intraday IV history yet")
    if finite(snap.iv_percentile):
        notes.append(f"ATM IV percentile {snap.iv_percentile:.0f} ({snap.iv_history_days} sessions)")
        if snap.iv_percentile >= 85:
            con.append(f"ATM IV in the top {100 - snap.iv_percentile:.0f}% of {snap.iv_history_days} sessions")
    else:
        notes.append(f"IV percentile unavailable ({snap.iv_history_days} of 20 sessions recorded)")
    return min(1.0, pts), sup, con, notes
