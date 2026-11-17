"""
Pre-breakout system: indicators, levels and price-structure features.

Every function here is CAUSAL: the value on a bar uses only that bar and earlier bars.
That is what lets the live service (last row only) and the backtest (all rows at once)
share one code path without look-ahead. tests/test_strategy.py checks it by recomputing on
truncated data.

    resample(minutes_1m, 5)              completed 5-minute bars (right-labelled = bar close time)
    add_indicators(bars, p)              EMA, RSI, MACD, ADX/DI, ATR, Bollinger, relative volume
    session_vwap(bars)                   VWAP that resets every session (needs real volume)
    daily_levels(daily, day)             previous-day H/L/C, previous-week H/L (no same-day data)
    opening_range(day_bars, minutes)     opening-range high/low once the range is complete
    swing_pivots(bars, left, right)      pivot highs/lows, known only `right` bars later
    build_zones(levels, atr, merge_atr)  cluster nearby levels into support / resistance zones
    compression(bars, p)                 Bollinger-width percentile, ATR contraction, narrowing ranges
    structure(bars, pivots)              HH / HL / LH / LL labels from confirmed pivots
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


# =============================================================================
# Bars
# =============================================================================
def resample(minutes: pd.DataFrame, minutes_per_bar: int) -> pd.DataFrame:
    """1-minute OHLCV (index = minute START, tz-aware) -> N-minute bars labelled by START,
    restricted to the session. The caller decides which bars are complete."""
    if minutes is None or minutes.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi"])
    frame = minutes
    clock = frame.index.time
    frame = frame[(clock >= SESSION_OPEN) & (clock < SESSION_CLOSE)]
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi"])
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in frame:
        agg["volume"] = "sum"
    if "oi" in frame:
        agg["oi"] = "last"
    # origin at 09:15 so bars are 09:15, 09:20, ...
    out = frame.resample(f"{minutes_per_bar}min", origin="start_day", offset="15min", label="left",
                         closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["minutes_in_bar"] = frame["close"].resample(f"{minutes_per_bar}min", origin="start_day", offset="15min",
                                                    label="left", closed="left").count().reindex(out.index)
    for column in ("volume", "oi"):
        if column not in out:
            out[column] = np.nan
    return out


def completed_bars(bars: pd.DataFrame, now: pd.Timestamp, minutes_per_bar: int, grace_seconds: int = 5) -> pd.DataFrame:
    """Only bars whose end time (start + N min + grace) has passed. Never trade on a forming bar."""
    if bars.empty:
        return bars
    ends = bars.index + pd.Timedelta(minutes=minutes_per_bar) + pd.Timedelta(seconds=grace_seconds)
    return bars[ends <= now]


# =============================================================================
# Classic indicators (Wilder smoothing where the original definition uses it)
# =============================================================================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0.0), length)
    loss = wilder((-delta).clip(lower=0.0), length)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(loss != 0.0, 100.0)
    return out.where(gain.notna())


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev = bars["close"].shift()
    return pd.concat([bars["high"] - bars["low"], (bars["high"] - prev).abs(), (bars["low"] - prev).abs()],
                     axis=1).max(axis=1)


def atr(bars: pd.DataFrame, length: int = 14) -> pd.Series:
    return wilder(true_range(bars), length)


def adx(bars: pd.DataFrame, length: int = 14):
    up = bars["high"].diff()
    down = -bars["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = wilder(true_range(bars), length)
    plus_di = 100.0 * wilder(pd.Series(plus_dm, index=bars.index), length) / tr
    minus_di = 100.0 * wilder(pd.Series(minus_dm, index=bars.index), length) / tr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return wilder(dx, length), plus_di, minus_di


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def has_volume(volume: Optional[pd.Series]) -> bool:
    if volume is None or len(volume) == 0:
        return False
    values = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    return bool((values > 0).mean() > 0.8)


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Typical-price VWAP, reset each session. NaN when the bars carry no real volume (index spot)."""
    if not has_volume(bars.get("volume")):
        return pd.Series(np.nan, index=bars.index)
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    pv = typical * bars["volume"]
    day = pd.Series(bars.index.date, index=bars.index)
    return pv.groupby(day).cumsum() / bars["volume"].groupby(day).cumsum().replace(0.0, np.nan)


def relative_volume(volume: pd.Series, length: int = 20) -> pd.Series:
    """Volume vs. the mean of the PREVIOUS `length` bars (the current bar is not in its own baseline)."""
    if not has_volume(volume):
        return pd.Series(np.nan, index=volume.index if volume is not None else None)
    base = volume.shift(1).rolling(length, min_periods=max(5, length // 2)).mean()
    return volume / base.replace(0.0, np.nan)


@dataclass
class IndicatorParams:
    ema_periods: tuple = (9, 20, 50, 200)
    rsi_length: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_length: int = 14
    atr_length: int = 14
    bb_length: int = 20
    bb_std: float = 2.0
    volume_ma: int = 20
    compression_lookback: int = 100   # bars for the Bollinger-width percentile
    swing_left: int = 3
    swing_right: int = 3


def add_indicators(bars: pd.DataFrame, p: IndicatorParams) -> pd.DataFrame:
    out = bars.copy()
    close = out["close"]
    for length in p.ema_periods:
        out[f"ema{length}"] = ema(close, length)
    out["rsi"] = rsi(close, p.rsi_length)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
    out["adx"], out["plus_di"], out["minus_di"] = adx(out, p.adx_length)
    out["atr"] = atr(out, p.atr_length)
    mid = close.rolling(p.bb_length, min_periods=p.bb_length).mean()
    std = close.rolling(p.bb_length, min_periods=p.bb_length).std(ddof=0)
    out["bb_mid"], out["bb_up"], out["bb_low"] = mid, mid + p.bb_std * std, mid - p.bb_std * std
    out["bb_width"] = (out["bb_up"] - out["bb_low"]) / mid
    # percentile of the current width within the trailing window (0 = narrowest seen)
    out["bb_width_pct"] = out["bb_width"].rolling(p.compression_lookback, min_periods=30).apply(
        lambda w: float((w[:-1] < w[-1]).mean()) if len(w) > 1 else np.nan, raw=True)
    out["atr_fast"] = atr(out, 5)
    out["atr_ratio"] = out["atr_fast"] / out["atr"]
    rng = out["high"] - out["low"]
    out["range_ratio"] = rng.rolling(6, min_periods=6).mean() / rng.shift(6).rolling(6, min_periods=6).mean()
    out["rvol"] = relative_volume(out["volume"], p.volume_ma) if "volume" in out else np.nan
    out["vwap"] = session_vwap(out)
    out["ema20_slope"] = out["ema20"] - out["ema20"].shift(3)
    out["ema50_slope"] = out["ema50"] - out["ema50"].shift(3)
    out["rsi_slope"] = out["rsi"] - out["rsi"].shift(3)
    out["macd_hist_slope"] = out["macd_hist"] - out["macd_hist"].shift(2)
    if "oi" in out and out["oi"].notna().any():
        out["oi_change"] = out["oi"] - out["oi"].shift(6)
    else:
        out["oi_change"] = np.nan
    return out


# =============================================================================
# Reference levels
# =============================================================================
@dataclass
class DailyLevels:
    pdh: float = np.nan
    pdl: float = np.nan
    pdc: float = np.nan
    pwh: float = np.nan
    pwl: float = np.nan
    daily_ema20: float = np.nan
    daily_ema50: float = np.nan
    daily_ema200: float = np.nan
    daily_atr: float = np.nan

    def to_dict(self):
        return {k: (None if not np.isfinite(v) else round(float(v), 2)) for k, v in self.__dict__.items()}


def daily_from_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    if minutes is None or minutes.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    clock = minutes.index.time
    session = minutes[(clock >= SESSION_OPEN) & (clock < SESSION_CLOSE)]
    grouped = session.groupby(session.index.date)
    daily = pd.DataFrame({"open": grouped["open"].first(), "high": grouped["high"].max(),
                          "low": grouped["low"].min(), "close": grouped["close"].last()})
    daily.index = pd.to_datetime(daily.index)
    return daily


def daily_levels(daily: pd.DataFrame, day: date) -> DailyLevels:
    """Levels known BEFORE `day` opens: only rows strictly earlier than `day` are used."""
    if daily is None or daily.empty:
        return DailyLevels()
    idx = pd.to_datetime(daily.index)
    past = daily[idx.date < day] if hasattr(idx, "date") else daily
    past = past.copy()
    past.index = pd.to_datetime(past.index)
    if past.empty:
        return DailyLevels()
    last = past.iloc[-1]
    levels = DailyLevels(pdh=float(last["high"]), pdl=float(last["low"]), pdc=float(last["close"]))
    # previous ISO week (the full week before the week containing `day`)
    week_start = day - timedelta(days=day.weekday())
    prev_week = past[(past.index.date < week_start) & (past.index.date >= week_start - timedelta(days=7))]
    if not prev_week.empty:
        levels.pwh, levels.pwl = float(prev_week["high"].max()), float(prev_week["low"].min())
    close = past["close"]
    for length, name in ((20, "daily_ema20"), (50, "daily_ema50"), (200, "daily_ema200")):
        if len(close) >= length:
            setattr(levels, name, float(ema(close, length).iloc[-1]))
    if len(past) >= 15:
        levels.daily_atr = float(atr(past, 14).iloc[-1])
    return levels


def opening_range(day_bars_1m: pd.DataFrame, minutes: int, now: pd.Timestamp):
    """(high, low) of the first `minutes` of the session, once that window has closed; else (nan, nan)."""
    if day_bars_1m is None or day_bars_1m.empty:
        return np.nan, np.nan
    start = day_bars_1m.index[0].normalize() + pd.Timedelta(hours=9, minutes=15)
    end = start + pd.Timedelta(minutes=minutes)
    if now < end:
        return np.nan, np.nan
    window = day_bars_1m[(day_bars_1m.index >= start) & (day_bars_1m.index < end)]
    if window.empty:
        return np.nan, np.nan
    return float(window["high"].max()), float(window["low"].min())


# =============================================================================
# Swing pivots and price structure
# =============================================================================
def swing_pivots(bars: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """
    Pivot high at bar i: high[i] is the strict max of [i-left, i+right]. It can only be KNOWN at
    bar i+right, so each pivot carries `known_at` (the index position where it becomes usable).
    Returns rows: pos, known_at, kind ('H'/'L'), price, time.
    """
    highs, lows = bars["high"].to_numpy(), bars["low"].to_numpy()
    n = len(bars)
    rows = []
    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            rows.append((i, i + right, "H", float(highs[i]), bars.index[i]))
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            rows.append((i, i + right, "L", float(lows[i]), bars.index[i]))
    return pd.DataFrame(rows, columns=["pos", "known_at", "kind", "price", "time"])


def pivots_known(pivots: pd.DataFrame, position: int) -> pd.DataFrame:
    if pivots.empty:
        return pivots
    return pivots[pivots["known_at"] <= position]


def structure(pivots_now: pd.DataFrame) -> Dict[str, object]:
    """Labels from the last two confirmed swing highs and lows."""
    out = {"hh": False, "hl": False, "lh": False, "ll": False, "label": "unclear",
           "last_high": np.nan, "last_low": np.nan}
    if pivots_now is None or pivots_now.empty:
        return out
    hs = pivots_now[pivots_now["kind"] == "H"]["price"].tail(2).to_list()
    ls = pivots_now[pivots_now["kind"] == "L"]["price"].tail(2).to_list()
    if hs:
        out["last_high"] = hs[-1]
    if ls:
        out["last_low"] = ls[-1]
    if len(hs) == 2:
        out["hh"], out["lh"] = hs[1] > hs[0], hs[1] < hs[0]
    if len(ls) == 2:
        out["hl"], out["ll"] = ls[1] > ls[0], ls[1] < ls[0]
    if out["hh"] and out["hl"]:
        out["label"] = "HH-HL (uptrend)"
    elif out["lh"] and out["ll"]:
        out["label"] = "LH-LL (downtrend)"
    elif out["hl"] and out["lh"]:
        out["label"] = "LH-HL (contracting)"
    elif out["hh"] and out["ll"]:
        out["label"] = "HH-LL (expanding)"
    return out


def rising_lows(bars: pd.DataFrame, count: int = 6) -> float:
    """Slope sign of the last `count` lows, scaled to [-1, 1] by the fraction of up-steps."""
    lows = bars["low"].tail(count).to_numpy()
    if len(lows) < 3:
        return 0.0
    steps = np.diff(lows)
    return float((steps > 0).mean() * 2 - 1)


def falling_highs(bars: pd.DataFrame, count: int = 6) -> float:
    highs = bars["high"].tail(count).to_numpy()
    if len(highs) < 3:
        return 0.0
    steps = np.diff(highs)
    return float((steps < 0).mean() * 2 - 1)


# =============================================================================
# Support / resistance zones
# =============================================================================
@dataclass
class Zone:
    price: float
    low: float
    high: float
    sources: List[str] = field(default_factory=list)
    touches: int = 0

    @property
    def strength(self) -> int:
        named = {s for s in self.sources if not s.startswith("swing")}
        swings = sum(1 for s in self.sources if s.startswith("swing"))
        return len(named) * 2 + swings + min(self.touches, 4)

    def label(self) -> str:
        names = sorted({s.split(":")[0] for s in self.sources})
        return "/".join(names) if names else "level"

    def to_dict(self):
        return {"price": round(self.price, 2), "low": round(self.low, 2), "high": round(self.high, 2),
                "sources": self.label(), "touches": self.touches, "strength": self.strength}


def build_zones(levels: List[tuple], atr_value: float, merge_atr: float = 0.25) -> List[Zone]:
    """levels = [(price, source)]. Levels closer than merge_atr x ATR merge into one zone."""
    clean = sorted((float(p), s) for p, s in levels if p is not None and np.isfinite(p))
    if not clean:
        return []
    tol = max(atr_value * merge_atr, 1e-9) if np.isfinite(atr_value) else 0.0
    groups: List[List[tuple]] = []
    for price, source in clean:
        if groups and price - groups[-1][-1][0] <= tol:
            groups[-1].append((price, source))
        else:
            groups.append([(price, source)])
    zones = []
    for group in groups:
        prices = [p for p, _ in group]
        named = [p for p, s in group if not s.startswith("swing")]
        zones.append(Zone(price=float(np.mean(named)) if named else float(np.mean(prices)),
                          low=min(prices), high=max(prices), sources=[s for _, s in group]))
    return zones


def count_touches(bars: pd.DataFrame, zone: Zone, tolerance: float, lookback: int = 40) -> int:
    """How many of the last `lookback` bars came within `tolerance` of the zone without closing through it."""
    recent = bars.tail(lookback)
    if recent.empty:
        return 0
    near_top = (recent["high"] >= zone.low - tolerance) & (recent["high"] <= zone.high + tolerance)
    near_bottom = (recent["low"] <= zone.high + tolerance) & (recent["low"] >= zone.low - tolerance)
    touched = near_top | near_bottom
    # count separate visits, not consecutive bars
    visits = (touched & ~touched.shift(1, fill_value=False)).sum()
    return int(visits)


def nearest_above(zones: List[Zone], price: float) -> Optional[Zone]:
    above = [z for z in zones if z.low > price]
    return min(above, key=lambda z: z.low - price) if above else None


def nearest_below(zones: List[Zone], price: float) -> Optional[Zone]:
    below = [z for z in zones if z.high < price]
    return min(below, key=lambda z: price - z.high) if below else None
