"""
Causal indicator frames for the equity strategies.

`enrich_intraday` (5-minute bars) and `enrich_daily` (daily bars) add every column the strategies read. Each value on
a bar uses that bar and earlier bars only, and higher-timeframe values are attached to a bar only once the higher
bar has CLOSED - so the live engine (last completed bar) and the backtest (all bars at once) run the same code and
nothing repaints. tests/test_equity.py recomputes the frames on truncated data and requires identical values.

Frames are indexed by bar START (tz-aware IST) with open / high / low / close / volume and hold completed bars only.
The classic indicators are the ones the index scanner and the F&O engine already use.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.fno.indicators import (bollinger, daily_table, day_key, opening_range, prev_high, prev_low, session_vwap,
                                supertrend, swings)
from src.scanner.indicators import adx, atr, ema, macd, relative_volume, resample, rsi

TZ = "Asia/Kolkata"
OHLCV = ["open", "high", "low", "close", "volume"]


def resample_bars(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Finer intraday bars -> `minutes` bars labelled by start, anchored at 09:15 (15m, 30m, 60m, 240m)."""
    return resample(bars[OHLCV], minutes)[OHLCV]


def weekly_bars(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily -> weekly bars indexed by the Monday of the week."""
    key = daily.index.normalize() - pd.to_timedelta(daily.index.weekday, unit="D")
    g = daily.groupby(key)
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum()})


def trend_state(bars: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """+1 close > fast EMA > slow EMA, -1 the mirror, else 0 (also 0 while the EMAs warm up)."""
    f, s, c = ema(bars["close"], fast), ema(bars["close"], slow), bars["close"]
    return pd.Series(np.where((c > f) & (f > s), 1.0, np.where((c < f) & (f < s), -1.0, 0.0)), index=bars.index)


def known_at_close(values: pd.Series, bar_minutes: int, base: pd.DatetimeIndex, base_minutes: int) -> pd.Series:
    """Values of higher-timeframe bars, as known at the close of each base bar (completed higher bars only)."""
    if values.empty:
        return pd.Series(np.nan, index=base)
    known = pd.Series(values.to_numpy(), index=values.index + pd.Timedelta(minutes=bar_minutes))
    closes = base + pd.Timedelta(minutes=base_minutes)
    out = known[~known.index.duplicated(keep="last")].reindex(known.index.union(closes)).ffill().reindex(closes)
    return pd.Series(out.to_numpy(), index=base)


def htf_trend(base: pd.DataFrame, base_minutes: int, tf_minutes: int) -> pd.Series:
    bars = resample_bars(base, tf_minutes)
    return known_at_close(trend_state(bars), tf_minutes, base.index, base_minutes).fillna(0.0)


# =============================================================================
# Intraday (5-minute base)
# =============================================================================
def enrich_intraday(bars: pd.DataFrame, daily: Optional[pd.DataFrame] = None, nifty: Optional[pd.DataFrame] = None,
                    base_minutes: int = 5, orb_minutes: int = 30) -> pd.DataFrame:
    """`bars`: 5-minute OHLCV of several sessions. `daily`: broker daily candles (older sessions, for the previous
    day's levels of the first session). `nifty`: a frame returned by `enrich_index` for the same period."""
    f = bars[OHLCV].copy()
    if f.empty:
        return f
    c, h, l, o, v = f["close"], f["high"], f["low"], f["open"], f["volume"]
    day = day_key(f.index)
    f["bar_no"] = f.groupby(day).cumcount()
    for n in (9, 20, 21, 50):
        f[f"ema{n}"] = ema(c, n)
    f["vwap"], f["vwap_sd"] = session_vwap(f)
    f["rsi"] = rsi(c, 14)
    f["macd"], f["macd_sig"], f["macd_hist"] = macd(c)
    f["atr"] = atr(f, 14)
    f["st_line"], f["st_dir"] = supertrend(f, 10, 3.0)
    f["bb_mid"], f["bb_up"], f["bb_lo"] = bollinger(c, 20, 2.0)
    f["bb_width"] = (f["bb_up"] - f["bb_lo"]) / f["bb_mid"]
    f["adx"], f["pdi"], f["mdi"] = adx(f, 14)
    f["vol_ma"] = v.shift(1).rolling(20, min_periods=10).mean()
    f["rvol"] = relative_volume(v, 20)
    slot = pd.Series(f.index.time, index=f.index)                    # same time of day, previous 10 sessions
    tod = v.groupby(slot).transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    f["rvol_tod"] = v / tod.replace(0.0, np.nan)
    f["hh12"], f["ll12"] = prev_high(f, 12), prev_low(f, 12)
    f["hh6"], f["ll6"] = prev_high(f, 6), prev_low(f, 6)

    # previous session's levels (a session's own row is never used inside that session)
    d = daily_table(f, daily).shift(1)
    pdh, pdl, pdc = (d[k].reindex(day).to_numpy() for k in ("high", "low", "close"))
    f["pdh"], f["pdl"], f["pdc"] = pdh, pdl, pdc
    f["pivot"] = (pdh + pdl + pdc) / 3.0
    f["r1"], f["s1"] = 2 * f["pivot"] - pdl, 2 * f["pivot"] - pdh
    f["r2"], f["s2"] = f["pivot"] + (pdh - pdl), f["pivot"] - (pdh - pdl)
    bc = (pdh + pdl) / 2.0
    tc = 2 * f["pivot"] - bc
    f["cpr_top"], f["cpr_bot"] = np.maximum(tc, bc), np.minimum(tc, bc)
    f["day_open"] = o.groupby(day).transform("first")
    f["day_pct"] = (c / f["pdc"] - 1.0) * 100.0

    f["orb_hi"], f["orb_lo"] = opening_range(f, orb_minutes, base_minutes)
    f["f15_hi"], f["f15_lo"] = opening_range(f, 15, base_minutes)

    # session extremes and where they were made (this bar included)
    f["run_hi"], f["run_lo"] = h.groupby(day).cummax(), l.groupby(day).cummin()
    f["hi_bar"] = f["bar_no"].where(h >= f["run_hi"]).groupby(day).ffill()
    f["lo_bar"] = f["bar_no"].where(l <= f["run_lo"]).groupby(day).ffill()

    sw = swings(f, 3, 3)
    f["swing_hi"], f["swing_lo"] = sw["swing_hi"], sw["swing_lo"]

    f["h15"] = htf_trend(f, base_minutes, 15)
    f["h30"] = htf_trend(f, base_minutes, 30)
    f["h60"] = htf_trend(f, base_minutes, 60)

    if nifty is not None and len(nifty):
        n = nifty.reindex(f.index, method="ffill")
        f["nifty_trend"] = n["h15"].fillna(0.0)
        f["nifty_pct"] = n["day_pct"]
        f["rs_day"] = f["day_pct"] - f["nifty_pct"]
    else:
        f["nifty_trend"], f["nifty_pct"], f["rs_day"] = 0.0, np.nan, np.nan
    return f


def enrich_index(bars: pd.DataFrame, daily: Optional[pd.DataFrame] = None, base_minutes: int = 5) -> pd.DataFrame:
    """NIFTY 50 index, 5-minute bars: trend on 15m / 60m and the day's change (the index has no volume)."""
    f = bars[OHLCV].copy()
    if f.empty:
        return f
    day = day_key(f.index)
    d = daily_table(f, daily).shift(1)
    f["pdc"] = d["close"].reindex(day).to_numpy()
    f["day_pct"] = (f["close"] / f["pdc"] - 1.0) * 100.0
    f["ema20"], f["ema50"] = ema(f["close"], 20), ema(f["close"], 50)
    f["rsi"] = rsi(f["close"], 14)
    f["h15"] = htf_trend(f, base_minutes, 15)
    f["h60"] = htf_trend(f, base_minutes, 60)
    return f


# =============================================================================
# Daily (swing)
# =============================================================================
def _divergence(f: pd.DataFrame, sw: pd.DataFrame, right: int = 3, max_gap: int = 40):
    """RSI divergence, flagged on the bar where the second pivot is confirmed (`right` bars after it)."""
    n = len(f)
    bull, bear = np.zeros(n, bool), np.zeros(n, bool)
    r = f["rsi"].to_numpy()
    for col, out, sign in (("pl", bull, 1), ("ph", bear, -1)):
        last = None                                           # (position, price, rsi) of the previous pivot
        vals = sw[col].to_numpy()
        for i in np.flatnonzero(np.isfinite(vals)):
            pos = i - right
            cur = (pos, vals[i], r[pos] if pos >= 0 else np.nan)
            if last is not None and np.isfinite(cur[2]) and np.isfinite(last[2]) and pos - last[0] <= max_gap:
                if sign == 1 and cur[1] < last[1] and cur[2] > last[2] + 2 and cur[2] < 45:
                    out[i] = True
                if sign == -1 and cur[1] > last[1] and cur[2] < last[2] - 2 and cur[2] > 55:
                    out[i] = True
            last = cur
    return bull, bear


def _cup_handle(f: pd.DataFrame, window: int = 60):
    """(resistance, handle low) of a cup-and-handle base in the `window` bars BEFORE each bar, else NaN.
    Heuristic: left rim in the first 25 bars, a rounded 8-35% deep cup at least 7 bars each side, the right rim
    back within 4% of the left one, then a handle of 3+ bars that gives back less than half of the cup."""
    n = len(f)
    res, low = np.full(n, np.nan), np.full(n, np.nan)
    hi, lo = f["high"].to_numpy(), f["low"].to_numpy()
    for i in range(window, n):
        hw, lw = hi[i - window:i], lo[i - window:i]
        a = int(np.argmax(hw[:25]))
        left = hw[a]
        b = a + int(np.argmin(lw[a:50]))
        bottom = lw[b]
        depth = (left - bottom) / left if left > 0 else 0
        if not (0.08 <= depth <= 0.35) or b - a < 7 or b >= window - 10:
            continue
        cpos = b + int(np.argmax(hw[b:]))
        right = hw[cpos]
        if cpos - b < 7 or cpos > window - 4 or not (left * 0.96 <= right <= left * 1.03):
            continue
        handle_low = lw[cpos + 1:].min()
        if right - handle_low > 0.5 * (right - bottom):
            continue
        res[i], low[i] = max(left, right), handle_low
    return res, low


def enrich_daily(daily: pd.DataFrame, nifty: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """`daily`: completed daily bars. `nifty`: NIFTY 50 daily bars (index = session date) for relative strength."""
    f = daily[OHLCV].copy()
    if f.empty:
        return f
    c, h, l, o, v = f["close"], f["high"], f["low"], f["open"], f["volume"]
    for n in (9, 20, 21, 50, 200):
        f[f"ema{n}"] = ema(c, n)
    f["rsi"] = rsi(c, 14)
    f["macd"], f["macd_sig"], f["macd_hist"] = macd(c)
    f["atr"] = atr(f, 14)
    f["st_line"], f["st_dir"] = supertrend(f, 10, 3.0)
    f["adx"], f["pdi"], f["mdi"] = adx(f, 14)
    f["bb_mid"], f["bb_up"], f["bb_lo"] = bollinger(c, 20, 2.0)
    f["bb_width"] = (f["bb_up"] - f["bb_lo"]) / f["bb_mid"]
    f["bb_rank"] = f["bb_width"].rolling(120, min_periods=60).rank(pct=True)
    f["don_hi"], f["don_lo"] = prev_high(f, 40), prev_low(f, 40)
    f["hi20"], f["lo20"] = prev_high(f, 20), prev_low(f, 20)
    f["hi10"], f["hi5"] = prev_high(f, 10), prev_high(f, 5)
    f["vol_ma"] = v.shift(1).rolling(20, min_periods=10).mean()
    f["vol_ratio"] = v / f["vol_ma"].replace(0.0, np.nan)
    f["vol5_ratio"] = v.rolling(5, min_periods=5).mean() / f["vol_ma"].replace(0.0, np.nan)
    up = v.where(c > c.shift(1), 0.0).rolling(10, min_periods=10).sum()
    dn = v.where(c < c.shift(1), 0.0).rolling(10, min_periods=10).sum()
    f["ud_ratio"] = up / dn.replace(0.0, np.nan)
    f["obv"] = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    f["obv_ema"] = ema(f["obv"], 20)
    f["roc5"], f["roc20"], f["roc60"] = c.pct_change(5) * 100, c.pct_change(20) * 100, c.pct_change(60) * 100

    sw = swings(f, 3, 3)
    for k in ("swing_hi", "swing_lo", "prev_swing_hi", "prev_swing_lo", "hi_pos", "lo_pos"):
        f[k] = sw[k]
    f["struct"] = np.where((f["swing_hi"] > f["prev_swing_hi"]) & (f["swing_lo"] > f["prev_swing_lo"]), 1.0,
                           np.where((f["swing_hi"] < f["prev_swing_hi"]) & (f["swing_lo"] < f["prev_swing_lo"]), -1.0, 0.0))
    f["bull_div"], f["bear_div"] = _divergence(f, sw)
    f["cup_res"], f["cup_low"] = _cup_handle(f)

    # weekly context: every day of a week sees the PREVIOUS completed week
    w = weekly_bars(f)
    w["trend"] = trend_state(w, 10, 20)
    wk = w.shift(1)
    key = f.index.normalize() - pd.to_timedelta(f.index.weekday, unit="D")
    f["wk_trend"] = wk["trend"].reindex(key).fillna(0.0).to_numpy()
    f["pw_high"], f["pw_low"] = wk["high"].reindex(key).to_numpy(), wk["low"].reindex(key).to_numpy()

    if nifty is not None and len(nifty):
        n = nifty.reindex(f.index, method="ffill")
        nc = n["close"]
        f["rs20"] = f["roc20"] - nc.pct_change(20) * 100
        f["rs60"] = f["roc60"] - nc.pct_change(60) * 100
        f["nifty_trend"] = trend_state(n).to_numpy()
    else:
        f["rs20"], f["rs60"], f["nifty_trend"] = np.nan, np.nan, 0.0
    f["rs_rank"], f["sector_rs"], f["sector_rank"] = np.nan, np.nan, np.nan       # add_cross_section fills these
    return f


def add_cross_section(frames: Dict[str, pd.DataFrame], sectors: Dict[str, str]) -> None:
    """Relative-strength rank of each stock across the universe (0-100) and of its sector, per day. In place.
    Uses each day's own closes only, so the value on a day never depends on later days."""
    if not frames:
        return
    blend = pd.DataFrame({s: 0.6 * f["rs20"] + 0.4 * f["rs60"].fillna(f["rs20"]) for s, f in frames.items() if len(f)})
    rank = blend.rank(axis=1, pct=True) * 100.0
    rs20 = pd.DataFrame({s: f["rs20"] for s, f in frames.items() if len(f)})
    sec = pd.Series({s: sectors.get(s) or "Other" for s in rs20.columns})
    sector_rs = rs20.T.groupby(sec).mean().T                         # equal-weighted 20-day RS of each sector
    sector_rank = sector_rs.rank(axis=1, ascending=False)
    for s, f in frames.items():
        if not len(f):
            continue
        f["rs_rank"] = rank[s].reindex(f.index)
        f["sector_rs"] = sector_rs[sec[s]].reindex(f.index)
        f["sector_rank"] = sector_rank[sec[s]].reindex(f.index)


# =============================================================================
# Scores (work on a whole frame or on one row; every component is returned so the number can be explained)
# =============================================================================
def _a(x):
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)


def intraday_components(f, side: str) -> Dict[str, np.ndarray]:
    """Signal strength 0-100 = trend 25 + volume 20 + momentum 20 + ADX 10 + market 10 + relative strength 15."""
    s = 1.0 if side == "BUY" else -1.0
    trend = 10.0 * (_a(f["h60"]) == s) + 10.0 * (_a(f["h15"]) == s) + 5.0 * (np.sign(_a(f["ema9"]) - _a(f["ema21"])) == s)
    volume = np.clip(_a(f["rvol"]) / 2.0, 0, 1) * 20.0
    momentum = np.clip(s * (_a(f["rsi"]) - 50.0) / 20.0, 0, 1) * 12.0 * (_a(f["rsi"]) > 0) \
        + 8.0 * (np.sign(_a(f["macd_hist"])) == s)
    strength = np.clip(_a(f["adx"]) / 40.0, 0, 1) * 10.0
    nt = _a(f["nifty_trend"])
    market = np.where(nt == s, 10.0, np.where(nt == 0, 5.0, 0.0))
    rs = np.clip(0.5 + s * _a(f["rs_day"]) / 2.0, 0, 1) * 15.0
    return {"trend": trend, "volume": volume, "momentum": momentum, "adx": strength, "market": market, "rel_strength": rs}


def swing_components(f, side: str) -> Dict[str, np.ndarray]:
    """Confidence 0-100 = trend 25 + momentum 20 + relative strength 20 + volume 15 + structure 10 + market 10."""
    s = 1.0 if side == "BUY" else -1.0
    trend = 10.0 * (_a(f["wk_trend"]) == s) + 10.0 * (np.sign(_a(f["ema20"]) - _a(f["ema50"])) == s) \
        + 5.0 * ((np.sign(_a(f["close"]) - _a(f["ema200"])) == s) & (_a(f["ema200"]) > 0))
    momentum = np.clip(s * (_a(f["rsi"]) - 50.0) / 20.0 + 0.25, 0, 1) * 12.0 * (_a(f["rsi"]) > 0) \
        + 8.0 * (np.sign(_a(f["macd_hist"])) == s)
    rank = np.asarray(f["rs_rank"], dtype=float)
    rank = np.where(np.isfinite(rank), rank, 50.0)
    rs = (rank if s > 0 else 100.0 - rank) / 100.0 * 20.0
    volume = np.clip(_a(f["vol_ratio"]) / 2.0, 0, 1) * 15.0
    structure = 10.0 * (_a(f["struct"]) == s)
    nt = _a(f["nifty_trend"])
    market = np.where(nt == s, 10.0, np.where(nt == 0, 5.0, 0.0))
    return {"trend": trend, "momentum": momentum, "rel_strength": rs, "volume": volume, "structure": structure,
            "market": market}


def total(components: Dict[str, np.ndarray]) -> np.ndarray:
    return sum(components.values())
