"""
Independent strategy modules of the v3 layer. Each one looks at ONE completed signal bar and returns, for each
side (CE = bullish, PE = bearish):

    confidence    0..1, how strongly that module's own evidence points that way (heuristic, NOT a probability)
    evidence      the observations behind it (supporting) and against it
    invalidation  the underlying price at which that module's idea is wrong

    1  Volatility compression & expansion   Bollinger / Keltner squeeze, BB-width + ATR percentiles, NR4 / NR7,
                                           inside bars, range contraction, distance to the level, pressure
    2  Price action & market structure     HH / HL / LH / LL, break of structure, change of character, rising
                                           lows under resistance / falling highs over support, level retests
    3  Momentum acceleration               RSI zone and slope (never overbought / oversold alone), MACD histogram
                                           acceleration, ADX / DI, short- vs medium-term ROC, persistence,
                                           divergence
    4  VWAP & participation acceptance     price vs VWAP with acceptance (consecutive closes), reclaim /
                                           rejection, VWAP slope, deviation bands, VWAP anchored at the last swing
    5  Opening range                       5 / 15 / 30-minute ranges: compression under / over them, break and
                                           hold, retest, failed opening-range break
    6  Liquidity sweep & reversal          PDH / PDL / opening range / equal highs-lows / swing sweeps that close
                                           back inside, with a rejection wick; only in regimes that allow reversals
    7  Derivatives intelligence            src/options/derivatives.py (live option chain only)

Modules never trade on their own. score_v3.py combines them into the family scores, so correlated inputs
(e.g. ATR contraction and Bollinger width) are inside ONE module and counted once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.config.instruments import CE, PE
from src.options import derivatives as deriv_mod
from src.scanner import features_ext as fx
from src.scanner import regime as reg
from src.utils.numbers import finite


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass
class Side:
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    against: List[str] = field(default_factory=list)
    invalidation: float = np.nan

    def to_dict(self):
        return {"confidence": round(self.confidence, 2), "evidence": self.evidence, "against": self.against,
                "invalidation": round(self.invalidation, 2) if finite(self.invalidation) else None}


@dataclass
class ModuleSignal:
    name: str
    ce: Side
    pe: Side
    available: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def direction(self) -> str:
        if not self.available:
            return "N/A"
        if self.ce.confidence >= self.pe.confidence + 0.15 and self.ce.confidence >= 0.35:
            return CE
        if self.pe.confidence >= self.ce.confidence + 0.15 and self.pe.confidence >= 0.35:
            return PE
        return "NEUTRAL"

    def side(self, direction: str) -> Side:
        return self.ce if direction == CE else self.pe

    def to_dict(self):
        return {"name": self.name, "available": self.available, "direction": self.direction,
                "ce": self.ce.to_dict(), "pe": self.pe.to_dict(), "notes": self.notes}


@dataclass
class ModuleInput:
    ext: fx.ExtFrame
    ev: object                    # strategies.BarEval (v1: levels, zones, breakout tests)
    pos: int
    cfg: object
    regime: reg.Regime
    deriv: Optional[deriv_mod.DerivSnapshot] = None

    @property
    def atr(self) -> float:
        return self.ev.atr

    @property
    def close(self) -> float:
        return self.ev.close


# =============================================================================
# 1. Volatility compression & expansion
# =============================================================================
def volatility(mi: ModuleInput) -> ModuleSignal:
    ext, pos, ff = mi.ext, mi.pos, mi.ext.ff
    parts, notes = [], []
    bbp, atr_pct = ff.v("bb_width_pct", pos), ext.v("atr_pct", pos)
    if finite(bbp):
        parts.append(_clip((0.5 - bbp) / 0.4))
    if finite(atr_pct):
        parts.append(_clip((0.5 - atr_pct) / 0.4))
    rr = ff.v("range_ratio", pos)
    if finite(rr):
        parts.append(_clip((1.0 - rr) / 0.35))
    squeeze, sq_bars = ext.v("squeeze_on", pos), ext.v("squeeze_bars", pos)
    narrow = [n for n, c in (("NR7", "nr7"), ("NR4", "nr4"), ("inside bar", "inside_bar"))
              if any(ext.v(c, p) == 1 for p in range(max(0, pos - 2), pos + 1))]
    comp = float(np.mean(parts)) if parts else 0.0
    if squeeze:
        comp = min(1.0, comp + 0.15 + min(0.15, 0.02 * sq_bars))
        notes.append(f"Bollinger inside Keltner for {sq_bars:.0f} bars")
    if narrow:
        comp = min(1.0, comp + 0.1)
        notes.append("narrow-range bar: " + ", ".join(narrow))
    if finite(bbp):
        notes.append(f"BB width pct {bbp:.0%}, ATR pct {atr_pct:.0%}" if finite(atr_pct) else f"BB width pct {bbp:.0%}")
    fired = ext.v("squeeze_fired", pos)
    out = ModuleSignal("Volatility compression", Side(), Side(), available=bool(parts), notes=notes)
    if not parts:
        out.notes.append("not enough bars for volatility percentiles")
        return out
    a = ff.arrays
    lo_i = max(0, pos - 11)
    win_h, win_l = np.nanmax(a["high"][lo_i:pos + 1]), np.nanmin(a["low"][lo_i:pos + 1])
    mid = (win_h + win_l) / 2
    roc = ext.v("roc_fast", pos)
    for direction, side in ((CE, out.ce), (PE, out.pe)):
        s = 1.0 if direction == CE else -1.0
        d = mi.ev.side(direction)
        pressure = 0.5
        if finite(mi.close) and win_h > win_l:
            pressure += 0.25 * s * (mi.close - mid) / ((win_h - win_l) / 2)
        if finite(roc):
            pressure += 0.15 * np.sign(roc) * s
        near = finite(d.distance_atr) and 0 <= d.distance_atr <= mi.cfg.watch_max_atr
        loc = 1.0 if near else 0.4
        side.confidence = _clip(comp * _clip(pressure) * 1.4 * loc)
        if comp >= 0.6:
            side.evidence.append("volatility compressed" + (" near the level" if near else ""))
        if fired and s * (mi.close - ff.v("open", pos)) > 0:
            side.evidence.append("squeeze released in this direction")
            side.confidence = _clip(side.confidence + 0.15)
        if _clip(pressure) < 0.4:
            side.against.append("pressure points the other way inside the range")
        side.invalidation = win_l if direction == CE else win_h
    return out


# =============================================================================
# 2. Price action & market structure
# =============================================================================
def structure(mi: ModuleInput) -> ModuleSignal:
    ext, pos, ff = mi.ext, mi.pos, mi.ext.ff
    st = mi.ev.structure or {}
    trend = ext.v("struct_trend", pos)
    recent = range(max(0, pos - 5), pos + 1)
    bos_up = any(ext.v("bos_up", p) for p in recent)
    bos_dn = any(ext.v("bos_dn", p) for p in recent)
    ch_up = any(ext.v("choch_up", p) for p in recent)
    ch_dn = any(ext.v("choch_dn", p) for p in recent)
    out = ModuleSignal("Market structure", Side(), Side(), notes=[str(st.get("label", "unclear"))])
    a_ = ff.arrays
    lows6, highs6 = a_["low"][max(0, pos - 5):pos + 1], a_["high"][max(0, pos - 5):pos + 1]
    for direction, side in ((CE, out.ce), (PE, out.pe)):
        up = direction == CE
        s = 1 if up else -1
        d = mi.ev.side(direction)
        pts = 0.0
        if trend == s:
            pts += 0.25
            side.evidence.append("structure trend " + ("up" if up else "down"))
        elif trend == -s:
            side.against.append("structure trend " + ("down" if up else "up"))
        if (bos_up if up else bos_dn):
            pts += 0.2
            side.evidence.append("break of structure " + ("up" if up else "down") + " in the last 6 bars")
        if (ch_up if up else ch_dn):
            pts += 0.15
            side.evidence.append("change of character " + ("up" if up else "down"))
        if (ch_dn if up else ch_up):
            side.against.append("change of character against")
        if st.get("hl" if up else "lh"):
            pts += 0.15
            side.evidence.append("higher low" if up else "lower high")
        if st.get("hh" if up else "ll"):
            pts += 0.1
        steps = np.diff(lows6 if up else highs6)        # same measure as indicators.rising_lows / falling_highs
        slope = float(((steps > 0) if up else (steps < 0)).mean() * 2 - 1) if len(steps) >= 2 else 0.0
        near = finite(d.distance_atr) and 0 <= d.distance_atr <= mi.cfg.watch_max_atr
        if slope > 0.2 and near:
            pts += 0.15
            side.evidence.append(("higher lows under resistance" if up else "lower highs over support"))
        touches = (d.level or {}).get("touches", 0) or 0
        if near and touches >= 2:
            pts += 0.1
            side.evidence.append(f"level tested {touches}x")
        # retest of a level broken in the last 6 bars that held (price came back within 0.25 ATR and closed beyond)
        lvl = d.level_price
        if finite(lvl) and finite(mi.atr):
            a = ff.arrays
            prior = [a["close"][p] for p in range(max(0, pos - 6), pos)]
            broke = any(s * (c - lvl) > 0 for c in prior)
            touched = (a["low"][pos] - lvl) <= 0.25 * mi.atr if up else (lvl - a["high"][pos]) <= 0.25 * mi.atr
            if broke and touched and s * (mi.close - lvl) > 0:
                pts += 0.15
                side.evidence.append("retest of the broken level held")
        side.confidence = _clip(pts)
        side.invalidation = ext.v("last_swing_low", pos) if up else ext.v("last_swing_high", pos)
    return out


# =============================================================================
# 3. Momentum acceleration
# =============================================================================
def momentum(mi: ModuleInput) -> ModuleSignal:
    ext, pos, ff, cfg = mi.ext, mi.pos, mi.ext.ff, mi.cfg
    rsi_v, rsi_s = ff.v("rsi", pos), ff.v("rsi_slope", pos)
    hist, hist_s = ff.v("macd_hist", pos), ff.v("macd_hist_slope", pos)
    adx_v, pdi, mdi = ff.v("adx", pos), ff.v("plus_di", pos), ff.v("minus_di", pos)
    rf, rs = ext.v("roc_fast", pos), ext.v("roc_slow", pos)
    persist = ext.v("persistence", pos)
    div = fx.divergence(ext, pos)
    out = ModuleSignal("Momentum", Side(), Side(), available=finite(rsi_v) and finite(hist), notes=div["notes"][:])
    for direction, side in ((CE, out.ce), (PE, out.pe)):
        up = direction == CE
        s = 1.0 if up else -1.0
        pts = 0.0
        lo, hi = (cfg.rsi_ce_min, cfg.rsi_ce_max) if up else (cfg.rsi_pe_min, cfg.rsi_pe_max)
        if finite(rsi_v):
            if lo <= rsi_v <= hi and finite(rsi_s) and s * rsi_s >= 0:
                pts += 0.2
                side.evidence.append(f"RSI {rsi_v:.0f} in the {direction} zone and improving")
            elif (up and rsi_v < 45) or (not up and rsi_v > 55):
                side.against.append(f"RSI {rsi_v:.0f} against")
        if finite(hist):
            if s * hist > 0 and finite(hist_s) and s * hist_s > 0:
                pts += 0.25
                side.evidence.append("MACD histogram accelerating")
            elif s * hist > 0:
                pts += 0.1
            elif finite(hist_s) and s * hist_s > 0:
                pts += 0.1
                side.evidence.append("MACD histogram turning")
        if finite(adx_v) and finite(pdi) and finite(mdi):
            if sign_ok(s * (pdi - mdi)) and adx_v >= cfg.adx_min:
                pts += 0.2
                side.evidence.append(f"ADX {adx_v:.0f}, {'+DI' if up else '-DI'} leading")
            elif s * (pdi - mdi) < 0 and adx_v >= cfg.adx_min:
                side.against.append(f"{'-DI' if up else '+DI'} leading")
        if finite(rf) and finite(rs):
            if s * rf > 0 and abs(rf) * 4 > abs(rs) and s * rs >= 0:
                pts += 0.15
                side.evidence.append("short-term momentum faster than medium-term")
            elif s * rf > 0:
                pts += 0.05
        if finite(persist):
            p = persist if up else 1 - persist
            if p >= 0.67:
                pts += 0.1
                side.evidence.append(f"{p:.0%} of the last 6 closes with the move")
        if div["bear" if up else "bull"]:
            pts -= 0.15
            side.against.append("momentum divergence against")
        if div["bull" if up else "bear"]:
            pts += 0.1
            side.evidence.append("momentum divergence with")
        side.confidence = _clip(pts)
        side.invalidation = mi.close - s * mi.atr if finite(mi.atr) else np.nan
    return out


def sign_ok(x) -> bool:
    return finite(x) and x > 0


# =============================================================================
# 4. VWAP & price acceptance
# =============================================================================
def vwap(mi: ModuleInput) -> ModuleSignal:
    ext, pos = mi.ext, mi.pos
    vw, price = ext.v("vwap", pos), ext.v("vwap_price", pos)
    dev, slope, acc = ext.v("vwap_dev", pos), ext.v("vwap_slope", pos), ext.v("vwap_accept", pos)
    proxy = ext.vwap_source != "futures"
    out = ModuleSignal("VWAP & acceptance", Side(), Side(), available=finite(vw) and finite(price),
                       notes=["VWAP: " + ("index TWAP proxy (no futures volume) - weighted down" if proxy
                                          else "futures session VWAP")])
    if not out.available:
        out.notes.append("no VWAP")
        return out
    prev_acc = ext.v("vwap_accept", pos - 1)
    a = ext.ff.arrays
    for direction, side in ((CE, out.ce), (PE, out.pe)):
        up = direction == CE
        s = 1.0 if up else -1.0
        pts = 0.0
        if s * (price - vw) > 0:
            pts += 0.2
            n = abs(acc) if np.sign(acc) == s else 0
            if n >= 3:
                pts += 0.2
                side.evidence.append(f"{n:.0f} closes {'above' if up else 'below'} VWAP (acceptance)")
            if finite(prev_acc) and np.sign(prev_acc) == -s:
                pts += 0.15
                side.evidence.append("VWAP " + ("reclaimed" if up else "lost") + " on this bar")
        else:
            side.against.append("price " + ("below" if up else "above") + " VWAP")
            # rejection of VWAP from the wrong side (wick through, close back) supports the other side - handled there
        if finite(slope):
            if s * slope > 0:
                pts += 0.15
                side.evidence.append("VWAP sloping " + ("up" if up else "down"))
            elif s * slope < 0:
                side.against.append("VWAP sloping against")
        # rejection: this bar's extreme went through VWAP from the trade side and closed back on it.
        # Only on the index price scale (the TWAP proxy); a futures VWAP is on the futures scale.
        if ext.vwap_source == "twap-proxy":
            h, l = a["high"][pos], a["low"][pos]
            rejected = (l < vw < mi.close) if up else (h > vw > mi.close)
            if rejected:
                pts += 0.1
                side.evidence.append("VWAP rejection in the trade direction")
        if finite(dev) and dev > 0:
            z = s * (price - vw) / dev
            if z > 2.0:
                pts -= 0.2
                side.against.append(f"{z:.1f} deviations beyond VWAP (extended)")
        swing = fx.last_swing_pos(ext, pos, "L" if up else "H")
        if swing is not None:
            av = fx.anchored_vwap(ext, pos, swing)
            if finite(av) and s * (mi.close - av) > 0:
                pts += 0.15
                side.evidence.append(f"above VWAP anchored at the last swing low ({av:,.1f})" if up else
                                     f"below VWAP anchored at the last swing high ({av:,.1f})")
        conf = _clip(pts)
        side.confidence = conf * (0.6 if proxy else 1.0)
        vw_spot = mi.close - (price - vw)       # VWAP translated to the index scale (futures basis removed)
        side.invalidation = vw_spot - s * 0.25 * mi.atr if finite(mi.atr) else np.nan
    return out


# =============================================================================
# 5. Opening range (5 / 15 / 30 minutes)
# =============================================================================
def opening_range(mi: ModuleInput) -> ModuleSignal:
    ext, pos = mi.ext, mi.pos
    a = ext.ff.arrays
    out = ModuleSignal("Opening range", Side(), Side())
    windows = [w for w in ext.or_windows if finite(ext.opening_range(pos, w)[0])]
    if not windows:
        out.available = False
        out.notes.append("opening ranges not complete yet")
        return out
    comp = ext.v("squeeze_on", pos) or (finite(mi.ext.ff.v("bb_width_pct", pos)) and mi.ext.ff.v("bb_width_pct", pos) <= 0.3)
    today = [p for p in range(max(0, pos - 6), pos) if ext.ff.dates[p] == ext.ff.dates[pos]]
    for direction, side in ((CE, out.ce), (PE, out.pe)):
        up = direction == CE
        s = 1.0 if up else -1.0
        best = 0.0
        for w in windows:
            hi, lo = ext.opening_range(pos, w)
            edge, opp = (hi, lo) if up else (lo, hi)
            width = hi - lo
            dist = s * (edge - mi.close) / mi.atr if finite(mi.atr) and mi.atr > 0 else np.nan
            pts = 0.0
            if finite(dist) and 0 <= dist <= 0.5:
                pts = 0.35 + (0.15 if comp else 0.0)
                side.evidence.append(f"{w}-min OR {'high' if up else 'low'} {edge:,.1f} {dist:.2f} ATR away")
            elif finite(dist) and dist < 0:
                held = all(s * (a["close"][p] - edge) > 0 for p in today[-2:]) if today else False
                pts = 0.45 + (0.15 if held else 0.0)
                if held:
                    side.evidence.append(f"holding beyond the {w}-min OR {'high' if up else 'low'}")
                if finite(mi.atr) and abs(dist) > 1.5:
                    pts -= 0.25
                    side.against.append(f"{abs(dist):.1f} ATR beyond the {w}-min OR (extended)")
                retest = (a["low"][pos] - edge) <= 0.2 * mi.atr if up else (edge - a["high"][pos]) <= 0.2 * mi.atr
                if retest and s * (mi.close - edge) > 0:
                    pts += 0.15
                    side.evidence.append(f"{w}-min OR retest held")
            # failed break of the OTHER edge in the last 3 bars supports this side
            failed = any(s * (opp - a["close"][p]) > 0 for p in today[-3:]) and s * (mi.close - opp) > 0 if today else False
            if failed:
                pts += 0.2
                side.evidence.append(f"failed {w}-min OR {'breakdown' if up else 'breakout'}")
            if width > 0 and finite(mi.atr) and width < 1.0 * mi.atr:
                side.evidence.append(f"{w}-min OR narrow ({width / mi.atr:.1f} ATR)")
            best = max(best, pts)
        side.confidence = _clip(best)
        w0 = windows[min(1, len(windows) - 1)]
        hi, lo = ext.opening_range(pos, w0)
        side.invalidation = (hi + lo) / 2
    return out


# =============================================================================
# 6. Liquidity sweep & reversal
# =============================================================================
def liquidity_sweep(mi: ModuleInput) -> ModuleSignal:
    ext, pos, cfg = mi.ext, mi.pos, mi.cfg
    lv = mi.ev.levels or {}
    levels = [(lv.get("pdh"), "PDH"), (lv.get("pdl"), "PDL"), (lv.get("orh"), "ORH"), (lv.get("orl"), "ORL"),
              (ext.v("last_swing_high", pos), "swing high"), (ext.v("last_swing_low", pos), "swing low")]
    ib_h, ib_l = ext.initial_balance(pos)
    levels += [(ib_h, "IB high"), (ib_l, "IB low")]
    pools = fx.liquidity_pools(ext, pos, mi.atr)
    levels += [(p, "equal highs") for p in pools["equal_highs"]] + [(p, "equal lows") for p in pools["equal_lows"]]
    levels = [(float(p), n) for p, n in levels if p is not None and finite(p)]
    sw = fx.sweep(ext, pos, levels, mi.atr, cfg.sweep_lookback)
    out = ModuleSignal("Liquidity sweep", Side(), Side(),
                       notes=[f"pools: {len(pools['equal_highs'])} equal-high, {len(pools['equal_lows'])} equal-low"])
    allowed = mi.regime.name in reg.REVERSAL_ALLOWED or cfg.allow_reversal_in_trend
    a = ext.ff.arrays
    for direction, side, key in ((CE, out.ce, "bull"), (PE, out.pe, "bear")):
        hits = sw[key]
        if not hits:
            continue
        lvl, label, back = min(hits, key=lambda h: h[2])
        i = pos - back
        rng = max(a["high"][i] - a["low"][i], 1e-9)
        wick = (min(a["open"][i], a["close"][i]) - a["low"][i]) / rng if direction == CE else \
               (a["high"][i] - max(a["open"][i], a["close"][i])) / rng
        mid = (a["high"][i] + a["low"][i]) / 2
        follow = (mi.close > mid) if direction == CE else (mi.close < mid)
        pts = 0.4 + (0.25 if wick >= 0.5 else 0.0) + (0.2 if follow and back > 0 else 0.0)
        side.evidence.append(f"{label} {lvl:,.1f} swept {back} bar(s) ago and reclaimed"
                             + (", rejection wick" if wick >= 0.5 else ""))
        if not allowed:
            side.against.append(f"reversal not taken in a {mi.regime.name.lower()} regime")
            pts = 0.0
        side.confidence = _clip(pts)
        side.invalidation = (a["low"][i] - 0.1 * mi.atr) if direction == CE else (a["high"][i] + 0.1 * mi.atr)
    return out


# =============================================================================
# 7. Derivatives intelligence
# =============================================================================
def derivatives(mi: ModuleInput) -> ModuleSignal:
    out = ModuleSignal("Derivatives", Side(), Side())
    snap = mi.deriv
    if snap is None or not snap.available:
        out.available = False
        out.notes.append((snap.note if snap is not None else "option chain unavailable") + " - not scored")
        return out
    for direction, side in ((CE, out.ce), (PE, out.pe)):
        conf, sup, con, notes = deriv_mod.evidence(snap, direction == CE, mi.cfg.iv_spike_pct)
        side.confidence = _clip(conf or 0.0)
        side.evidence, side.against = sup, con
        side.invalidation = snap.put_wall if direction == CE else snap.call_wall
        if direction == CE:
            out.notes += notes
    out.notes.append(f"PCR(OI) {snap.pcr_oi:.2f}" if finite(snap.pcr_oi) else "PCR n/a")
    return out


ALL_MODULES = (volatility, structure, momentum, vwap, opening_range, liquidity_sweep, derivatives)


def run_all(mi: ModuleInput) -> Dict[str, ModuleSignal]:
    out = {}
    for fn in ALL_MODULES:
        sig = fn(mi)
        out[sig.name] = sig
    return out
