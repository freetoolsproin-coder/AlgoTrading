"""
Index market direction: causal feature frame, 0-100 setup score per side (CE = bullish, PE = bearish),
breakout tests on completed signal bars, and the underlying (index) trade plan.

This module answers "what is the INDEX doing?" only. Choosing an option contract is in
src/options/selection.py; turning evaluations into WATCH / CONFIRMED / ENTRY signals is in
src/scanner/signals.py.

What the numbers are
    * observed     prices, volumes, OI from the feed
    * calculated   indicators (EMA, RSI, MACD, ADX, ...), levels, distances
    * heuristic    the 0-100 setup score. It is NOT a probability: no calibration has been done.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.instruments import CE, OPTION_TYPES as DIRECTIONS, PE
from src.config.settings import PBConfig
from src.scanner import indicators as ind
from src.utils.numbers import finite, fnum, r2


# =============================================================================
# Features: one causal frame per index (5-minute bars + futures participation)
# =============================================================================
@dataclass
class FeatureFrame:
    bars: pd.DataFrame          # spot signal bars with indicators
    pivots: pd.DataFrame        # swing pivots with known_at
    daily: pd.DataFrame         # daily bars (spot)
    minutes: pd.DataFrame       # 1-minute spot (for opening range)
    has_participation: bool
    day_levels: Dict[date, ind.DailyLevels] = field(default_factory=dict)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    dates: Optional[np.ndarray] = None
    opening: Dict[date, tuple] = field(default_factory=dict)   # day -> (orh, orl, known_from)
    hod: Optional[np.ndarray] = None   # high of the day BEFORE each bar (nan for the first 3 bars of a day)
    lod: Optional[np.ndarray] = None

    def levels_for(self, day: date) -> ind.DailyLevels:
        if day not in self.day_levels:
            self.day_levels[day] = ind.daily_levels(self.daily, day)
        return self.day_levels[day]

    def v(self, column: str, pos: int) -> float:
        arr = self.arrays.get(column)
        if arr is None or pos < 0 or pos >= len(arr):
            return np.nan
        value = arr[pos]
        return float(value) if value == value else np.nan

    def prepare(self, cfg: "PBConfig"):
        bars = self.bars
        self.arrays = {c: pd.to_numeric(bars[c], errors="coerce").to_numpy(dtype=float) for c in bars.columns}
        self.dates = np.array([t.date() for t in bars.index]) if len(bars) else np.array([])
        # opening range per day from 1-minute data; known once the window has closed
        self.opening = {}
        m = self.minutes
        if m is not None and not m.empty:
            clock = m.index.time
            m = m[(clock >= ind.SESSION_OPEN) & (clock < ind.SESSION_CLOSE)]
            minute_of_day = (m.index.hour * 60 + m.index.minute) - (9 * 60 + 15)
            window = m[minute_of_day < cfg.opening_range_minutes]
            if not window.empty:
                g = window.groupby(window.index.date)
                for day, high, low, count in zip(g["high"].max().index, g["high"].max().values, g["low"].min().values,
                                                 g["high"].count().values):
                    if count >= max(1, cfg.opening_range_minutes // 2):
                        start = pd.Timestamp(datetime.combine(day, time(9, 15)))
                        start = start.tz_localize(m.index.tz) if m.index.tz is not None else start
                        self.opening[day] = (float(high), float(low),
                                             start + pd.Timedelta(minutes=cfg.opening_range_minutes))
        # running high / low of the day before each bar
        n = len(bars)
        self.hod, self.lod = np.full(n, np.nan), np.full(n, np.nan)
        if n:
            highs, lows = self.arrays["high"], self.arrays["low"]
            run_h = run_l = np.nan
            count = 0
            for i in range(n):
                if i == 0 or self.dates[i] != self.dates[i - 1]:
                    run_h, run_l, count = np.nan, np.nan, 0
                if count >= 3:
                    self.hod[i], self.lod[i] = run_h, run_l
                run_h = highs[i] if run_h != run_h else max(run_h, highs[i])
                run_l = lows[i] if run_l != run_l else min(run_l, lows[i])
                count += 1
        p = self.pivots
        self.piv_known = p["known_at"].to_numpy() if len(p) else np.array([], dtype=int)
        self.piv_pos = p["pos"].to_numpy() if len(p) else np.array([], dtype=int)
        self.piv_kind = p["kind"].to_numpy() if len(p) else np.array([], dtype=object)
        self.piv_price = p["price"].to_numpy(dtype=float) if len(p) else np.array([], dtype=float)
        return self


def build_features(spot_minutes: pd.DataFrame, cfg: PBConfig, fut_minutes: Optional[pd.DataFrame] = None,
                   daily: Optional[pd.DataFrame] = None) -> FeatureFrame:
    bars = ind.resample(spot_minutes, cfg.signal_minutes)
    bars = ind.add_indicators(bars, cfg.ind)
    has_part = False
    if fut_minutes is not None and not fut_minutes.empty and ind.has_volume(fut_minutes.get("volume")):
        fut = ind.resample(fut_minutes, cfg.signal_minutes)
        fut["f_rvol"] = ind.relative_volume(fut["volume"], cfg.ind.volume_ma)
        fut["f_vwap"] = ind.session_vwap(fut)
        fut["f_close"] = fut["close"]
        fut["f_oi_change"] = fut["oi"] - fut["oi"].shift(6) if fut["oi"].notna().any() else np.nan
        fut["f_volume"] = fut["volume"]
        bars = bars.join(fut[["f_rvol", "f_vwap", "f_close", "f_oi_change", "f_volume"]], how="left")
        has_part = bars["f_rvol"].notna().any()
    for column in ("f_rvol", "f_vwap", "f_close", "f_oi_change", "f_volume"):
        if column not in bars:
            bars[column] = np.nan
    pivots = ind.swing_pivots(bars, cfg.ind.swing_left, cfg.ind.swing_right) if len(bars) else \
        pd.DataFrame(columns=["pos", "known_at", "kind", "price", "time"])
    if daily is None or daily.empty:
        daily = ind.daily_from_minutes(spot_minutes)
    ff = FeatureFrame(bars=bars, pivots=pivots, daily=daily, minutes=spot_minutes, has_participation=bool(has_part))
    return ff.prepare(cfg)


# =============================================================================
# Evaluation of one bar
# =============================================================================
@dataclass
class Family:
    name: str
    points: float
    max_points: float
    notes: List[str] = field(default_factory=list)
    available: bool = True


@dataclass
class DirectionEval:
    direction: str
    score: float
    families: List[Family]
    supporting: List[str]
    contradicting: List[str]
    level: Optional[dict]          # zone dict: price/low/high/sources/strength
    level_price: float             # the price to break (zone high for CE, zone low for PE)
    distance_atr: float            # (level - close)/ATR for CE; (close - level)/ATR for PE; <0 = beyond
    breakout: bool                 # this bar closed beyond level + buffer
    breakout_quality: List[str]    # failed quality checks (empty = good)
    extension_atr: float           # how far beyond the level the close is
    next_level: Optional[float]    # the following zone in the trade direction (target cap)
    entry_veto: Optional[str] = None   # set by a model layer (v3: regime / trap engine) to refuse an entry
    extras: dict = field(default_factory=dict)   # model-specific detail (v3: modules, traps, invalidation)

    @property
    def compression_pct(self) -> float:
        for f in self.families:
            if f.name == "Compression" and f.max_points:
                return 100.0 * f.points / f.max_points
        return 0.0

    def breakdown(self) -> List[dict]:
        return [{"family": f.name, "points": round(f.points, 1), "max": f.max_points, "notes": f.notes,
                 "available": f.available} for f in self.families]


@dataclass
class BarEval:
    index: str
    time: pd.Timestamp            # bar START
    close_time: pd.Timestamp      # bar END (when it became known)
    close: float
    atr: float
    position: int
    ce: DirectionEval
    pe: DirectionEval
    indicators: dict
    levels: dict
    zones: List[dict]
    structure: dict
    extras: dict = field(default_factory=dict)   # v3: regime, module report, derivatives snapshot

    def side(self, direction: str) -> DirectionEval:
        return self.ce if direction == CE else self.pe

    @property
    def bias(self) -> str:
        if self.ce.score >= self.pe.score + 10 and self.ce.score >= 40:
            return CE
        if self.pe.score >= self.ce.score + 10 and self.pe.score >= 40:
            return PE
        return "NEUTRAL"


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _row(bars: pd.DataFrame, pos: int, col: str) -> float:
    return fnum(bars[col].iat[pos]) if col in bars and 0 <= pos < len(bars) else np.nan


def evaluate_bar(ff: FeatureFrame, pos: int, cfg: PBConfig, index: str = "",
                 context: Optional[dict] = None) -> Optional[BarEval]:
    """Score the completed bar at position `pos` using data up to and including that bar only."""
    bars = ff.bars
    if pos < max(1, cfg.min_history_bars) or pos >= len(bars):  # not enough history = no evaluation (NO TRADE)
        return None
    t = bars.index[pos]
    close = ff.v("close", pos)
    atr_v = ff.v("atr", pos)
    if not finite(close) or not finite(atr_v) or atr_v <= 0:
        return None
    day = ff.dates[pos]
    prev_close = ff.v("close", pos - 1)
    bar_end = t + pd.Timedelta(minutes=cfg.signal_minutes)
    dl = ff.levels_for(day)
    orh = orl = np.nan
    if day in ff.opening and bar_end >= ff.opening[day][2]:
        orh, orl = ff.opening[day][0], ff.opening[day][1]
    hod, lod = ff.hod[pos], ff.lod[pos]

    mask = (ff.piv_known <= pos) & (ff.piv_pos >= pos - 75)  # pivots confirmed by now, ~ last session
    level_list = [(dl.pdh, "PDH"), (dl.pdl, "PDL"), (dl.pdc, "PDC"), (dl.pwh, "PWH"), (dl.pwl, "PWL"),
                  (orh, "ORH"), (orl, "ORL"), (hod, "HOD"), (lod, "LOD")]
    level_list += [(price, f"swing{kind}") for price, kind in zip(ff.piv_price[mask], ff.piv_kind[mask])]
    zones = ind.build_zones(level_list, atr_v, cfg.merge_atr)
    lo_i = max(0, pos - 40)
    highs, lows = ff.arrays["high"][lo_i:pos], ff.arrays["low"][lo_i:pos]  # bars BEFORE this one
    tol = 0.15 * atr_v
    for z in zones:
        touched = (((highs >= z.low - tol) & (highs <= z.high + tol)) | ((lows <= z.high + tol) & (lows >= z.low - tol)))
        if len(touched):
            z.touches = int(touched[0]) + int(np.sum(touched[1:] & ~touched[:-1]))
    known_mask = ff.piv_known <= pos
    known = ff.pivots[known_mask] if len(ff.pivots) else ff.pivots
    struct = ind.structure(known.tail(8) if not known.empty else known)
    ctx = context or {}

    evals = {}
    for direction in DIRECTIONS:
        evals[direction] = _evaluate_direction(direction, bars, pos, close, prev_close, atr_v, zones, struct, ff,
                                               cfg, ctx)
    indicators = {k: r2(ff.v(k, pos)) for k in
                  ("ema9", "ema20", "ema50", "ema200", "rsi", "macd", "macd_signal", "macd_hist", "adx", "plus_di",
                   "minus_di", "atr", "bb_up", "bb_low", "bb_width_pct", "atr_ratio", "f_rvol", "f_vwap", "f_close",
                   "f_oi_change")}
    levels = dl.to_dict()
    levels.update({"orh": r2(orh), "orl": r2(orl), "hod": r2(hod), "lod": r2(lod)})
    extras = {"high": ff.v("high", pos), "low": ff.v("low", pos), "prev_high": ff.v("high", pos - 1),
              "prev_low": ff.v("low", pos - 1)}
    return BarEval(index=index, time=t, close_time=bar_end, close=close, atr=atr_v, position=pos,
                   ce=evals[CE], pe=evals[PE], indicators=indicators, levels=levels,
                   zones=[z.to_dict() for z in zones], structure={k: (r2(v) if isinstance(v, float) else v)
                                                                  for k, v in struct.items()}, extras=extras)


def _evaluate_direction(direction, bars, pos, close, prev_close, atr_v, zones, struct, ff: FeatureFrame,
                        cfg: PBConfig, ctx: dict) -> DirectionEval:
    up = direction == CE
    sign = 1.0 if up else -1.0
    supporting: List[str] = []
    contradicting: List[str] = []
    w = cfg.weights
    buffer = max(cfg.buffer_atr * atr_v, close * cfg.buffer_pct / 100.0)

    # ---- the level: the first zone beyond the PREVIOUS close (so a zone broken on this bar is still "the" level)
    ref = prev_close if finite(prev_close) else close
    if up:
        cands = [z for z in zones if z.high >= ref]
        cands.sort(key=lambda z: z.high)
    else:
        cands = [z for z in zones if z.low <= ref]
        cands.sort(key=lambda z: -z.low)
    cands = [z for z in cands if z.strength >= cfg.min_zone_strength] or cands[:0]
    zone = cands[0] if cands else None
    nxt = cands[1] if len(cands) > 1 else None
    level_price = (zone.high if up else zone.low) if zone else np.nan
    if zone is not None:
        distance = sign * (level_price - close) / atr_v
        extension = -distance
    else:
        distance, extension = np.nan, np.nan

    families: List[Family] = []

    # ---- 1. location
    f = Family("Location", 0.0, w["location"])
    if zone is None:
        f.notes.append("no meaningful level within reach")
    elif distance < 0:
        f.points = w["location"]
        f.notes.append(f"closed {abs(distance):.2f} ATR beyond {zone.label()} {level_price:,.1f}")
    elif distance <= cfg.watch_max_atr:
        frac = 1.0 if distance <= cfg.near_atr else 1.0 - (distance - cfg.near_atr) / (cfg.watch_max_atr - cfg.near_atr)
        f.points = w["location"] * _clip01(frac)
        f.notes.append(f"{distance:.2f} ATR {'below' if up else 'above'} {zone.label()} {level_price:,.1f}")
        supporting.append(f"approaching {zone.label()} ({distance:.2f} ATR)")
    else:
        f.notes.append(f"{zone.label()} is {distance:.1f} ATR away (> {cfg.watch_max_atr})")
    families.append(f)

    # ---- 2. level quality
    f = Family("Level quality", 0.0, w["level"])
    if zone is not None:
        f.points = min(w["level"], zone.strength * w["level"] / 8.0)
        f.notes.append(f"{zone.label()} strength {zone.strength}, tested {zone.touches}x")
        if zone.touches >= 2:
            supporting.append(f"level tested {zone.touches} times")
    families.append(f)

    # ---- 3. compression (one family: Bollinger width, ATR contraction and narrowing ranges are correlated)
    f = Family("Compression", 0.0, w["compression"])
    bbp = ff.v("bb_width_pct", pos)
    atr_ratio = ff.v("atr_ratio", pos)
    rr = ff.v("range_ratio", pos)
    parts = []
    if finite(bbp):
        parts.append(1.0 if bbp <= 0.2 else 0.6 if bbp <= 0.4 else 0.0)
    if finite(atr_ratio):
        parts.append(1.0 if atr_ratio <= 0.8 else 0.5 if atr_ratio <= 0.95 else 0.0)
    if finite(rr):
        parts.append(1.0 if rr <= 0.75 else 0.5 if rr <= 0.9 else 0.0)
    if parts:
        f.points = w["compression"] * float(np.mean(parts))
        f.notes.append(f"BB width pct {bbp:.0%}" if finite(bbp) else "BB width n/a")
        if f.points >= 0.6 * w["compression"]:
            supporting.append("price compression")
    families.append(f)

    # ---- 4. structure
    f = Family("Structure", 0.0, w["structure"])
    window = bars.iloc[max(0, pos - 5):pos + 1]
    slope = ind.rising_lows(window) if up else ind.falling_highs(window)
    s = 0.0
    if slope > 0.2:
        s += 0.4
        supporting.append("higher lows" if up else "lower highs")
    if up:
        s += 0.3 if struct.get("hl") else 0.0
        s += 0.3 if struct.get("hh") else 0.0
        if struct.get("ll"):
            contradicting.append("last swing low was lower (LL)")
    else:
        s += 0.3 if struct.get("lh") else 0.0
        s += 0.3 if struct.get("ll") else 0.0
        if struct.get("hh"):
            contradicting.append("last swing high was higher (HH)")
    f.points = w["structure"] * _clip01(s)
    f.notes.append(str(struct.get("label")))
    families.append(f)

    # ---- 5. trend (EMA stack + slope + higher timeframe) - one family
    f = Family("Trend", 0.0, w["trend"])
    e9, e20, e50, e200 = (ff.v(c, pos) for c in ("ema9", "ema20", "ema50", "ema200"))
    slope20 = ff.v("ema20_slope", pos)
    t = 0.0
    if finite(e9) and finite(e20) and sign * (e9 - e20) > 0:
        t += 0.25
    if finite(e20) and finite(e50) and sign * (e20 - e50) > 0:
        t += 0.25
    if finite(slope20) and sign * slope20 > 0:
        t += 0.2
    if finite(e50) and sign * (close - e50) > 0:
        t += 0.15
    if finite(e200) and sign * (close - e200) > 0:
        t += 0.15
    dl = ff.levels_for(ff.dates[pos])
    if finite(dl.daily_ema20) and sign * (close - dl.daily_ema20) < 0:
        contradicting.append(f"price on the wrong side of the daily EMA20 ({dl.daily_ema20:,.0f})")
        t -= 0.15
    f.points = w["trend"] * _clip01(t)
    f.notes.append("EMA 9/20/50 aligned" if t >= 0.7 else "EMAs mixed")
    if t >= 0.7:
        supporting.append("EMA alignment")
    elif t <= 0.2:
        contradicting.append("EMAs point the other way")
    stretched = finite(e20) and sign * (close - e20) / atr_v > cfg.stretched_atr
    families.append(f)

    # ---- 6. momentum (RSI, MACD, ADX/DI)
    f = Family("Momentum", 0.0, w["momentum"])
    rsi_v, rsi_s = ff.v("rsi", pos), ff.v("rsi_slope", pos)
    hist, hist_s = ff.v("macd_hist", pos), ff.v("macd_hist_slope", pos)
    adx_v, pdi, mdi = ff.v("adx", pos), ff.v("plus_di", pos), ff.v("minus_di", pos)
    m = 0.0
    lo, hi = (cfg.rsi_ce_min, cfg.rsi_ce_max) if up else (cfg.rsi_pe_min, cfg.rsi_pe_max)
    if finite(rsi_v):
        if lo <= rsi_v <= hi and finite(rsi_s) and sign * rsi_s >= 0:
            m += 0.4
            supporting.append(f"RSI {rsi_v:.0f} improving")
        elif (up and rsi_v > hi + 5) or (not up and rsi_v < lo - 5):
            contradicting.append(f"RSI {rsi_v:.0f} {'overbought' if up else 'oversold'}")
        elif (up and rsi_v < 45) or (not up and rsi_v > 55):
            contradicting.append(f"RSI {rsi_v:.0f} against")
    if finite(hist):
        if sign * hist > 0 and finite(hist_s) and sign * hist_s > 0:
            m += 0.3
            supporting.append("MACD histogram building")
        elif sign * hist > 0 or (finite(hist_s) and sign * hist_s > 0):
            m += 0.15
    if finite(adx_v) and finite(pdi) and finite(mdi):
        if adx_v >= cfg.adx_min and sign * (pdi - mdi) > 0:
            m += 0.3
            supporting.append(f"ADX {adx_v:.0f} with {'+DI' if up else '-DI'} leading")
        elif sign * (pdi - mdi) < 0 and adx_v >= cfg.adx_min:
            contradicting.append(f"{'-DI' if up else '+DI'} leading (ADX {adx_v:.0f})")
    f.points = w["momentum"] * _clip01(m)
    families.append(f)

    # ---- 7. participation (futures volume, VWAP, OI build-up). Index spot has no volume.
    f = Family("Participation", 0.0, w["participation"])
    rvol = ff.v("f_rvol", pos)
    fvwap, fclose, foi = ff.v("f_vwap", pos), ff.v("f_close", pos), ff.v("f_oi_change", pos)
    if not ff.has_participation or not finite(rvol):
        f.available = False
        f.notes.append("futures volume / OI unavailable: not scored, score rescaled to the other families")
    else:
        p = 0.0
        if rvol >= cfg.rvol_good:
            p += 0.4
            supporting.append(f"futures volume {rvol:.1f}x")
        elif rvol >= 1.0:
            p += 0.2
        if finite(fvwap) and finite(fclose):
            if sign * (fclose - fvwap) > 0:
                p += 0.3
                supporting.append("futures " + ("above" if up else "below") + " VWAP")
            else:
                contradicting.append("futures " + ("below" if up else "above") + " VWAP")
        if finite(foi):
            price_move = close - ff.v("close", max(0, pos - 6))
            if foi > 0 and sign * price_move > 0:
                p += 0.3
                supporting.append("long build-up (OI up, price up)" if up else "short build-up (OI up, price down)")
            elif foi < 0 and sign * price_move > 0:
                p += 0.1
                f.notes.append("short covering" if up else "long unwinding")
        f.points = w["participation"] * _clip01(p)
    families.append(f)

    # ---- 8. context (other indices, VIX)
    f = Family("Context", 0.0, w["context"])
    agree = ctx.get("agree_ce" if up else "agree_pe")
    vix_chg = ctx.get("vix_change_pct")
    c = 0.0
    if agree is not None:
        c += 0.6 * _clip01(agree)
        if agree >= 0.66:
            supporting.append("other indices agree")
        elif agree <= 0.0:
            contradicting.append("other indices disagree")
    else:
        c += 0.3
    if finite(vix_chg):
        if vix_chg > 8:
            contradicting.append(f"India VIX +{vix_chg:.0f}% today")
        else:
            c += 0.4
    else:
        c += 0.2
    f.points = w["context"] * _clip01(c)
    families.append(f)

    score = sum(fm.points for fm in families)
    available_max = sum(fm.max_points for fm in families if fm.available)
    total_max = sum(fm.max_points for fm in families)
    if 0 < available_max < total_max:
        # a family with no data (e.g. index without futures volume) is not scored as a failure:
        # the score is rescaled to the evidence that exists, and the breakdown says so
        score = score * total_max / available_max
    if stretched:
        score -= 10
        contradicting.append(f"price stretched > {cfg.stretched_atr} ATR from EMA20")
    score = float(min(100.0, max(0.0, score)))

    # ---- breakout test on this completed bar
    breakout = bool(zone is not None and sign * (close - level_price) > buffer and
                    (not finite(prev_close) or sign * (prev_close - level_price) <= buffer))
    quality: List[str] = []
    if breakout:
        o, h, l_ = ff.v("open", pos), ff.v("high", pos), ff.v("low", pos)
        rng = max(h - l_, 1e-9)
        body = abs(close - o) / rng
        close_pos = (close - l_) / rng if up else (h - close) / rng
        if sign * (close - o) <= 0:
            quality.append("breakout bar closed against the direction")
        if body < cfg.min_body_ratio:
            quality.append(f"small body ({body:.0%} of range)")
        if close_pos < cfg.min_close_position:
            quality.append("weak close inside the bar (wick)")
        if ff.has_participation and finite(rvol) and rvol < cfg.min_rvol:
            quality.append(f"futures volume only {rvol:.1f}x")
        if cfg.require_vwap and finite(fvwap) and finite(fclose) and sign * (fclose - fvwap) <= 0:
            quality.append("futures on the wrong side of VWAP")
        if cfg.require_trend and finite(slope20) and sign * slope20 < 0:
            quality.append("EMA20 sloping against")
    level_dict = zone.to_dict() if zone else None
    next_level = (nxt.high if up else nxt.low) if nxt else None
    return DirectionEval(direction=direction, score=round(score, 1), families=families, supporting=supporting,
                         contradicting=contradicting, level=level_dict, level_price=level_price,
                         distance_atr=distance, breakout=breakout, breakout_quality=quality,
                         extension_atr=extension if finite(extension) else np.nan, next_level=next_level)


# =============================================================================
# Trade plan (underlying) and option selection
# =============================================================================
@dataclass
class UnderlyingPlan:
    direction: str
    entry: float
    entry_zone: Tuple[float, float]
    stop: float
    t1: float
    t2: float
    risk_points: float
    next_level: Optional[float]
    notes: List[str] = field(default_factory=list)


def underlying_plan(direction: str, ev: DirectionEval, close: float, atr_v: float, cfg: PBConfig,
                    swing: float = np.nan, early: bool = False) -> UnderlyingPlan:
    """Underlying (index) stop and targets. [TRADE_PLAN] stop_model:
        level   the broken level - sl_atr x ATR                                    (Model A, structural: level)
        swing   the last confirmed swing low (CE) / high (PE) - 0.1 ATR            (Model A, structural: swing)
        atr     entry - atr_stop_mult x ATR                                        (Model B)
    Every stop is capped at max_sl_atr x ATR from the entry. Early entries (before the level breaks) cannot
    use the level stop - it would sit on the wrong side - so they use the swing stop, else the ATR stop."""
    up = direction == CE
    sign = 1.0 if up else -1.0
    level = ev.level_price
    notes = []
    if not finite(swing):
        swing = (getattr(ev, "extras", None) or {}).get("structural_stop", np.nan)
    model = cfg.stop_model
    if early and model == "level":
        model = "swing"
    if model == "swing" and not (finite(swing) and sign * (close - swing) > 0.2 * atr_v):
        notes.append("no usable swing for the stop: ATR stop used")
        model = "atr"
    if model == "swing":
        stop = swing - sign * 0.1 * atr_v
        notes.append("stop beyond the last swing")
    elif model == "atr":
        stop = close - sign * cfg.atr_stop_mult * atr_v
        notes.append(f"stop {cfg.atr_stop_mult} ATR from entry")
    else:
        stop = level - sign * cfg.sl_atr * atr_v
    max_stop = close - sign * cfg.max_sl_atr * atr_v
    if sign * (stop - max_stop) < 0:  # stop further than allowed -> tighten
        stop = max_stop
        notes.append(f"stop capped at {cfg.max_sl_atr} ATR")
    if sign * (close - stop) < 0.2 * atr_v:  # a stop on (or almost on) the entry is no stop
        stop = close - sign * max(0.2, min(cfg.atr_stop_mult, cfg.max_sl_atr)) * atr_v
        notes.append("stop too close to entry: widened to the ATR stop")
    risk = abs(close - stop)
    t1 = close + sign * cfg.t1_r * risk
    t2 = close + sign * cfg.t2_r * risk
    if ev.next_level is not None and finite(ev.next_level) and sign * (ev.next_level - t1) < 0:
        notes.append(f"next level {ev.next_level:,.1f} sits before T1")
    zone = (level, level + sign * cfg.max_extension_atr * atr_v)
    return UnderlyingPlan(direction=direction, entry=close, entry_zone=(min(zone), max(zone)), stop=stop, t1=t1,
                          t2=t2, risk_points=risk, next_level=ev.next_level, notes=notes)
