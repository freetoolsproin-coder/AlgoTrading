"""
Score v2: 5-minute context + 1-minute confirmation, seven components (your weights).

    Component                               Max   Timeframe / source
    Price vs VWAP and VWAP slope             20   futures 1-min session VWAP (index spot has no volume)
    EMA 9/20 alignment                       15   5-min EMA 9 vs 20 and EMA20 slope + 1-min close vs 1-min EMA9
    RSI directional zone                     10   5-min RSI zone + 1-min RSI side and direction
    Higher lows / lower highs                20   5-min confirmed swings + 5-min lows/highs + 1-min lows/highs
    Futures price-volume-OI confirmation     15   futures: price with volume (rel. volume) and OI build-up
    Constituents breadth                     10   % of the index's constituents above their own VWAP / advancing
    Cross-index alignment                    10   the other indices' bias
                                            ---
                                            100

It runs ALONGSIDE the original score (v1) on the same levels and the same breakout tests, with its own
stage tracker and its own paper trades, so the two can be compared on live paper data. It never sends
Telegram messages unless [SCORE_V2] push = true.

A component without data (no futures volume, no breadth feed, single-index backtest) is marked
"no data" and the score is rescaled to the components that have data - the same rule as v1.
All inputs are causal: the 1-minute values are taken from minutes that ended at or before the
5-minute bar's close.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import instruments as inst
from src.scanner import indicators as ind
from src.scanner import strategies as strat
from src.utils import numbers as num

WEIGHTS = {"VWAP": 20, "EMA 9/20": 15, "RSI zone": 10, "Structure": 20, "Futures P-V-OI": 15,
           "Breadth": 10, "Cross-index": 10}


@dataclass
class MinuteFrame:
    """1-minute spot + futures features, indexed by minute START. Built once per cycle / backtest."""
    frame: pd.DataFrame
    ends: pd.DatetimeIndex  # minute end times

    def upto(self, t_end: pd.Timestamp) -> int:
        """Number of minutes that ended at or before t_end (DatetimeIndex search: no unit mix-ups)."""
        return int(self.ends.searchsorted(t_end, side="right"))


def build_minutes(spot: pd.DataFrame, fut: Optional[pd.DataFrame]) -> MinuteFrame:
    f = pd.DataFrame(index=spot.index)
    f["close"] = spot["close"]
    f["low"] = spot["low"]
    f["high"] = spot["high"]
    f["ema9"] = ind.ema(spot["close"], 9)
    f["rsi"] = ind.rsi(spot["close"], 14)
    f["rsi_slope"] = f["rsi"] - f["rsi"].shift(3)
    if fut is not None and len(fut) and ind.has_volume(fut.get("volume")):
        fut = fut[~fut.index.duplicated(keep="last")].sort_index()
        vwap = ind.session_vwap(fut)
        g = pd.DataFrame({"f_close": fut["close"], "f_vwap": vwap, "f_volume": fut["volume"], "f_oi": fut["oi"]})
        g = g.reindex(spot.index)
        f["f_close"], f["f_vwap"] = g["f_close"], g["f_vwap"]
        f["f_vwap_slope"] = g["f_vwap"] - g["f_vwap"].shift(5)
        f["f_rvol"] = ind.relative_volume(g["f_volume"], 30)
        f["f_rvol5"] = f["f_rvol"].rolling(5, min_periods=3).mean()
        f["f_oi_chg"] = g["f_oi"] - g["f_oi"].shift(15) if g["f_oi"].notna().any() else np.nan
        f["f_price_chg"] = g["f_close"] - g["f_close"].shift(15)
    else:
        for c in ("f_close", "f_vwap", "f_vwap_slope", "f_rvol", "f_rvol5", "f_oi_chg", "f_price_chg"):
            f[c] = np.nan
    return MinuteFrame(frame=f, ends=spot.index + pd.Timedelta(minutes=1))


def _v(frame: pd.DataFrame, i: int, col: str) -> float:
    if i < 0 or i >= len(frame):
        return np.nan
    x = frame[col].iat[i]
    return float(x) if pd.notna(x) else np.nan


def _clip(x):
    return float(min(1.0, max(0.0, x)))


def score_direction(direction: str, ev: strat.BarEval, ff: strat.FeatureFrame, mf: MinuteFrame, ctx: dict) -> strat.DirectionEval:
    """v2 DirectionEval: same level / breakout fields as v1, new families and score."""
    up = direction == inst.CE
    s = 1.0 if up else -1.0
    base = ev.side(direction)
    pos = ev.position
    i = mf.upto(ev.close_time) - 1            # last 1-minute bar inside (or ending at) this 5-min close
    m = mf.frame
    sup: List[str] = []
    con: List[str] = []
    fams: List[strat.Family] = []

    # 1. VWAP (futures)
    f = strat.Family("VWAP", 0.0, WEIGHTS["VWAP"])
    fc, fv, fslope = _v(m, i, "f_close"), _v(m, i, "f_vwap"), _v(m, i, "f_vwap_slope")
    if not (num.finite(fc) and num.finite(fv)):
        f.available = False
        f.notes.append("no futures VWAP")
    else:
        atr1 = ev.atr / 2.2 if num.finite(ev.atr) else np.nan  # ~1-min scale
        dist = s * (fc - fv)
        pts = 0.0
        if dist > 0:
            pts += 12 * _clip(0.5 + dist / (2 * atr1)) if num.finite(atr1) and atr1 > 0 else 12
            sup.append("futures " + ("above" if up else "below") + " VWAP")
        else:
            con.append("futures " + ("below" if up else "above") + " VWAP")
        if num.finite(fslope):
            if s * fslope > 0:
                pts += 8
                sup.append("VWAP sloping " + ("up" if up else "down"))
            elif s * fslope < 0:
                con.append("VWAP sloping against")
        f.points = min(f.max_points, pts)
        f.notes.append(f"fut {fc:,.1f} vs VWAP {fv:,.1f}")
    fams.append(f)

    # 2. EMA 9/20 (5m) + 1m close vs 1m EMA9
    f = strat.Family("EMA 9/20", 0.0, WEIGHTS["EMA 9/20"])
    e9, e20, sl20 = ff.v("ema9", pos), ff.v("ema20", pos), ff.v("ema20_slope", pos)
    pts = 0.0
    if num.finite(e9) and num.finite(e20) and s * (e9 - e20) > 0:
        pts += 8
        sup.append("5m EMA9 " + (">" if up else "<") + " EMA20")
    elif num.finite(e9) and num.finite(e20):
        con.append("5m EMA9/20 against")
    if num.finite(sl20) and s * sl20 > 0:
        pts += 4
    c1, ema1 = _v(m, i, "close"), _v(m, i, "ema9")
    if num.finite(c1) and num.finite(ema1) and s * (c1 - ema1) > 0:
        pts += 3
        sup.append("1m close beyond 1m EMA9")
    f.points = pts
    fams.append(f)

    # 3. RSI zone
    f = strat.Family("RSI zone", 0.0, WEIGHTS["RSI zone"])
    r5, r1, r1s = ff.v("rsi", pos), _v(m, i, "rsi"), _v(m, i, "rsi_slope")
    pts = 0.0
    if num.finite(r5):
        lo, hi = (55, 70) if up else (30, 45)
        if lo <= r5 <= hi:
            pts += 6
            sup.append(f"5m RSI {r5:.0f} in zone")
        elif (up and r5 > 75) or (not up and r5 < 25):
            con.append(f"5m RSI {r5:.0f} stretched")
        elif (up and r5 < 45) or (not up and r5 > 55):
            con.append(f"5m RSI {r5:.0f} against")
    if num.finite(r1) and s * (r1 - 50) > 0 and (not num.finite(r1s) or s * r1s >= 0):
        pts += 4
    f.points = pts
    fams.append(f)

    # 4. Structure
    f = strat.Family("Structure", 0.0, WEIGHTS["Structure"])
    st = ev.structure or {}
    pts = 0.0
    if (up and st.get("hl")) or (not up and st.get("lh")):
        pts += 8
        sup.append("5m swing " + ("higher low" if up else "lower high"))
    if (up and st.get("ll")) or (not up and st.get("hh")):
        con.append("5m swing " + ("lower low" if up else "higher high"))
    w5 = ff.bars.iloc[max(0, pos - 5):pos + 1]
    slope5 = ind.rising_lows(w5) if up else ind.falling_highs(w5)
    if slope5 > 0.2:
        pts += 6
    if i >= 10:
        w1 = m.iloc[i - 9:i + 1]
        slope1 = ind.rising_lows(w1, 10) if up else ind.falling_highs(w1, 10)
        if slope1 > 0.2:
            pts += 6
            sup.append("1m " + ("higher lows" if up else "lower highs"))
    f.points = pts
    fams.append(f)

    # 5. Futures price-volume-OI
    f = strat.Family("Futures P-V-OI", 0.0, WEIGHTS["Futures P-V-OI"])
    rv, oi, pc = _v(m, i, "f_rvol5"), _v(m, i, "f_oi_chg"), _v(m, i, "f_price_chg")
    if not (num.finite(rv) and num.finite(pc)):
        f.available = False
        f.notes.append("no futures volume / OI")
    else:
        pts = 0.0
        if s * pc > 0 and rv >= 1.2:
            pts += 6
            sup.append(f"futures move on {rv:.1f}x volume")
        elif s * pc > 0 and rv >= 1.0:
            pts += 3
        if num.finite(oi):
            if oi > 0 and s * pc > 0:
                pts += 9
                sup.append("long build-up" if up else "short build-up")
            elif oi < 0 and s * pc > 0:
                pts += 4
                f.notes.append("short covering" if up else "long unwinding")
            elif oi > 0 and s * pc < 0:
                con.append("short build-up" if up else "long build-up")
        f.points = min(f.max_points, pts)
    fams.append(f)

    # 6. Breadth
    f = strat.Family("Breadth", 0.0, WEIGHTS["Breadth"])
    b = (ctx or {}).get("breadth") or {}
    above = b.get("pct_above_vwap")
    adv = b.get("pct_advancing")
    metric = above if num.finite(above) else adv
    if not num.finite(metric):
        f.available = False
        f.notes.append("no constituents feed")
    else:
        side = metric if up else 100 - metric
        f.points = f.max_points * _clip((side - 50) / 15.0)
        label = "above VWAP" if num.finite(above) else "advancing"
        f.notes.append(f"{metric:.0f}% {label} ({b.get('count', '?')} stocks{', proxy' if b.get('proxy') else ''})")
        if side >= 60:
            sup.append(f"breadth {side:.0f}% with the move")
        elif side <= 40:
            con.append(f"breadth {side:.0f}% against")
    fams.append(f)

    # 7. Cross-index
    f = strat.Family("Cross-index", 0.0, WEIGHTS["Cross-index"])
    agree = (ctx or {}).get("agree_ce" if up else "agree_pe")
    if agree is None:
        f.available = False
        f.notes.append("no other indices")
    else:
        f.points = f.max_points * _clip(agree / 0.67)
        if agree >= 0.66:
            sup.append("other indices agree")
        elif agree == 0:
            con.append("no other index agrees")
    fams.append(f)

    score = sum(x.points for x in fams)
    avail = sum(x.max_points for x in fams if x.available)
    if 0 < avail < 100:
        score = score * 100 / avail
    out = copy.copy(base)
    out.families = fams
    out.score = round(float(min(100.0, max(0.0, score))), 1)
    out.supporting = sup
    out.contradicting = con
    return out


def rescore(ev: strat.BarEval, ff: strat.FeatureFrame, mf: MinuteFrame, ctx: dict) -> strat.BarEval:
    """A copy of the v1 BarEval with v2 scores (levels, breakout tests and zones are shared)."""
    out = copy.copy(ev)
    out.ce = score_direction(inst.CE, ev, ff, mf, ctx)
    out.pe = score_direction(inst.PE, ev, ff, mf, ctx)
    return out
