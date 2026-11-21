"""
Vectorised, CAUSAL indicators and session levels for the F&O engine.

Every column on a bar uses only that bar and earlier bars, so the live engine (last completed bar) and the
backtest (all bars at once) share one code path. tests/test_fno.py recomputes every strategy on truncated
data and requires identical answers.

Frames are indexed by bar START time (tz-aware IST) and contain completed bars only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.scanner.indicators import adx, atr, ema, has_volume, macd, relative_volume, resample, rsi

TZ = "Asia/Kolkata"


# ----------------------------------------------------------------------------- small helpers
def xup(a: pd.Series, b) -> pd.Series:
    """a crosses above b on this bar."""
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)


def xdn(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return ((a < b) & (a.shift(1) >= b.shift(1))).fillna(False)


def event(cond: pd.Series) -> pd.Series:
    """True on the first bar of a run of True."""
    cond = cond.fillna(False).astype(bool)
    return cond & ~cond.shift(1, fill_value=False)


def day_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.normalize()


def first_of_day(cond: pd.Series) -> pd.Series:
    cond = cond.fillna(False).astype(bool)
    return cond & (cond.groupby(day_key(cond.index)).cumsum() == 1)


def bars_for(m1: pd.DataFrame, tf: int) -> pd.DataFrame:
    return resample(m1, tf)[["open", "high", "low", "close", "volume", "oi"]]


def prev_high(f: pd.DataFrame, n: int) -> pd.Series:
    return f["high"].shift(1).rolling(n, min_periods=n).max()


def prev_low(f: pd.DataFrame, n: int) -> pd.Series:
    return f["low"].shift(1).rolling(n, min_periods=n).min()


# ----------------------------------------------------------------------------- indicators
def supertrend(f: pd.DataFrame, length: int = 10, mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """(line, direction): direction +1 up-trend, -1 down-trend, NaN during warm-up."""
    n = len(f)
    a = atr(f, length).to_numpy()
    hl2 = ((f["high"] + f["low"]) / 2.0).to_numpy()
    close = f["close"].to_numpy()
    upper, lower = hl2 + mult * a, hl2 - mult * a
    line, direction = np.full(n, np.nan), np.full(n, np.nan)
    fu = fl = np.nan
    d = 1
    for i in range(n):
        if not np.isfinite(a[i]):
            continue
        if not np.isfinite(fu):
            fu, fl = upper[i], lower[i]
            d = 1 if close[i] >= hl2[i] else -1
        else:
            pu, pl = fu, fl
            fu = upper[i] if (upper[i] < pu or close[i - 1] > pu) else pu
            fl = lower[i] if (lower[i] > pl or close[i - 1] < pl) else pl
            if d == -1 and close[i] > pu:
                d = 1
            elif d == 1 and close[i] < pl:
                d = -1
        line[i] = fl if d == 1 else fu
        direction[i] = d
    return pd.Series(line, index=f.index), pd.Series(direction, index=f.index)


def stoch_rsi(close: pd.Series, length: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    r = rsi(close, length)
    lo, hi = r.rolling(length, min_periods=length).min(), r.rolling(length, min_periods=length).max()
    raw = 100.0 * (r - lo) / (hi - lo).replace(0.0, np.nan)
    k = raw.rolling(smooth_k, min_periods=smooth_k).mean()
    return k, k.rolling(smooth_d, min_periods=smooth_d).mean()


def ichimoku(f: pd.DataFrame, tenkan: int = 9, kijun: int = 26, span_b: int = 52):
    """Tenkan, Kijun and the cloud that is PLOTTED at the current bar (spans computed `kijun` bars ago)."""
    def mid(n):
        return (f["high"].rolling(n, min_periods=n).max() + f["low"].rolling(n, min_periods=n).min()) / 2.0
    t, k = mid(tenkan), mid(kijun)
    a, b = ((t + k) / 2.0).shift(kijun), mid(span_b).shift(kijun)
    return t, k, a, b


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0):
    mid = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    return mid, mid + mult * sd, mid - mult * sd


def session_vwap(f: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """(VWAP, volume-weighted standard deviation), reset each session; NaN without real volume."""
    if not has_volume(f.get("volume")):
        nan = pd.Series(np.nan, index=f.index)
        return nan, nan
    day = day_key(f.index)
    typical = (f["high"] + f["low"] + f["close"]) / 3.0
    vol = f["volume"].fillna(0.0)
    cv = vol.groupby(day).cumsum().replace(0.0, np.nan)
    vwap = (typical * vol).groupby(day).cumsum() / cv
    var = ((typical ** 2) * vol).groupby(day).cumsum() / cv - vwap ** 2
    return vwap, np.sqrt(var.clip(lower=0.0))


def swings(f: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Confirmed swing pivots, placed on the bar where they become KNOWN (`right` bars after the pivot).
    Columns: ph / pl (pivot price on its confirmation bar, else NaN), swing_hi / swing_lo (latest confirmed),
    prev_swing_hi / prev_swing_lo, hi_pos / lo_pos (bar number of the latest pivot itself)."""
    w = left + right + 1
    h, l = f["high"], f["low"]
    hs, ls = h.shift(right), l.shift(right)
    ph = hs.where((hs == h.rolling(w, min_periods=w).max()) & (h.rolling(w, min_periods=w).apply(np.argmax, raw=True) == left))
    pl = ls.where((ls == l.rolling(w, min_periods=w).min()) & (l.rolling(w, min_periods=w).apply(np.argmin, raw=True) == left))
    pos = pd.Series(np.arange(len(f), dtype=float), index=f.index)
    out = pd.DataFrame({"ph": ph, "pl": pl}, index=f.index)
    out["swing_hi"], out["swing_lo"] = ph.ffill(), pl.ffill()
    out["prev_swing_hi"] = ph.dropna().shift(1).reindex(f.index).where(ph.notna()).ffill()
    out["prev_swing_lo"] = pl.dropna().shift(1).reindex(f.index).where(pl.notna()).ffill()
    out["hi_pos"] = (pos - right).where(ph.notna()).ffill()
    out["lo_pos"] = (pos - right).where(pl.notna()).ffill()
    return out


def daily_table(f: pd.DataFrame, daily_ext: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per session (index = midnight IST): OHLCV from the intraday bars, preceded by older rows of
    `daily_ext` (broker daily candles) when given. Consumers must shift(1) before using a row intraday."""
    day = day_key(f.index)
    g = f.groupby(day)
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum()})
    if daily_ext is not None and len(daily_ext) and len(d):
        ext = daily_ext.copy()
        if "datetime" in ext.columns:
            ext = ext.set_index("datetime")
        idx = pd.DatetimeIndex(ext.index)
        idx = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
        ext.index = idx.normalize()
        ext = ext[["open", "high", "low", "close", "volume"]]
        ext = ext[ext.index < d.index[0]]
        d = pd.concat([ext[~ext.index.duplicated(keep="last")].sort_index(), d])
    return d


def opening_range(f: pd.DataFrame, minutes: int, tf: int) -> Tuple[pd.Series, pd.Series]:
    """High / low of the first `minutes` of each session, available only once that window has closed."""
    n = max(1, int(np.ceil(minutes / tf)))
    day = day_key(f.index)
    bar_no = f.groupby(day).cumcount()
    inside = bar_no < n
    hi = f["high"].where(inside).groupby(day).transform("max").where(~inside)
    lo = f["low"].where(inside).groupby(day).transform("min").where(~inside)
    return hi, lo


def volume_profile(f: pd.DataFrame, bins: int = 24, value_area: float = 0.7) -> pd.DataFrame:
    """POC / VAH / VAL of each completed session (index = session date). Needs real volume."""
    rows = {}
    if not has_volume(f.get("volume")):
        return pd.DataFrame(columns=["poc", "vah", "val"])
    typical = ((f["high"] + f["low"] + f["close"]) / 3.0).to_numpy()
    vol = f["volume"].fillna(0.0).to_numpy()
    day = day_key(f.index)
    codes, uniques = pd.factorize(day)
    for code, d in enumerate(uniques):
        m = codes == code
        p, v = typical[m], vol[m]
        if v.sum() <= 0 or p.max() <= p.min():
            continue
        hist, edges = np.histogram(p, bins=bins, weights=v)
        mids = (edges[:-1] + edges[1:]) / 2.0
        i = lo = hi = int(hist.argmax())
        got, total = hist[i], hist.sum()
        while got < value_area * total and (lo > 0 or hi < bins - 1):
            below = hist[lo - 1] if lo > 0 else -1.0
            above = hist[hi + 1] if hi < bins - 1 else -1.0
            if above >= below:
                hi += 1
                got += hist[hi]
            else:
                lo -= 1
                got += hist[lo]
        rows[d] = (mids[i], edges[hi + 1], edges[lo])
    return pd.DataFrame.from_dict(rows, orient="index", columns=["poc", "vah", "val"])


def kalman_trend(close: pd.Series, process_var: float = 1e-3, measurement_var: float = 1e-1) -> Tuple[pd.Series, pd.Series]:
    """1-D linear Kalman filter of the close (from algo_options_engine): smoothed price and its per-bar velocity.
    Causal: bar i uses closes 0..i only."""
    c = close.to_numpy(dtype=float)
    n = len(c)
    est, vel = np.full(n, np.nan), np.full(n, np.nan)
    p, x = 1.0, np.nan
    for i in range(n):
        if not np.isfinite(c[i]):
            continue
        if not np.isfinite(x):
            x = c[i]
        p_pred = p + process_var
        k = p_pred / (p_pred + measurement_var)
        prev = x
        x = x + k * (c[i] - x)
        p = (1.0 - k) * p_pred
        est[i], vel[i] = x, x - prev
    return pd.Series(est, index=close.index), pd.Series(vel, index=close.index)


def rolling_value_area(f: pd.DataFrame, window: int = 75, bins: int = 24, value_area: float = 0.7) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """POC / VAH / VAL of the previous `window` bars (volume-weighted typical price), known before each bar."""
    n = len(f)
    poc, vah, val = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    hi, lo = f["high"].to_numpy(float), f["low"].to_numpy(float)
    tp = ((f["high"] + f["low"] + f["close"]) / 3.0).to_numpy(float)
    vol = f["volume"].to_numpy(float) if "volume" in f else np.zeros(n)
    for i in range(window, n):
        s = slice(i - window, i)
        v = vol[s]
        top, bot = np.nanmax(hi[s]), np.nanmin(lo[s])
        if not np.isfinite(v).any() or np.nansum(v) <= 0 or top <= bot:
            continue
        edges = np.linspace(bot, top, bins + 1)
        hist, _ = np.histogram(tp[s], bins=edges, weights=np.nan_to_num(v))
        k = int(np.argmax(hist))
        target, cum, left, right = hist.sum() * value_area, hist[k], k, k
        while cum < target and (left > 0 or right < bins - 1):
            lv = hist[left - 1] if left > 0 else -1.0
            rv = hist[right + 1] if right < bins - 1 else -1.0
            if rv >= lv:
                right += 1
                cum += rv
            else:
                left -= 1
                cum += lv
        poc[i], vah[i], val[i] = (edges[k] + edges[k + 1]) / 2, edges[right + 1], edges[left]
    idx = f.index
    return pd.Series(poc, index=idx), pd.Series(vah, index=idx), pd.Series(val, index=idx)


def tf_state(m1: pd.DataFrame, tf: int, closes: pd.DatetimeIndex, fast: int = 20, slow: int = 50) -> pd.Series:
    """Trend of timeframe `tf` as known at each time in `closes` (bar CLOSE times of the base frame):
    +1 close > EMA(fast) > EMA(slow), -1 the mirror, 0 mixed, NaN while warming up. Only bars of `tf` that
    have already closed at that time are used."""
    b = resample(m1, tf)
    if b.empty:
        return pd.Series(np.nan, index=closes)
    e1, e2 = ema(b["close"], fast), ema(b["close"], slow)
    state = pd.Series(np.where((b["close"] > e1) & (e1 > e2), 1.0, np.where((b["close"] < e1) & (e1 < e2), -1.0, 0.0)),
                      index=b.index).where(e2.notna())
    state.index = state.index + pd.Timedelta(minutes=tf)
    left = pd.DataFrame({"t": closes})
    right = pd.DataFrame({"t": state.index, "v": state.to_numpy()})
    return pd.Series(pd.merge_asof(left, right, on="t")["v"].to_numpy(), index=closes)


# ----------------------------------------------------------------------------- enrichment
def enrich(bars: pd.DataFrame, tf: int, daily_ext: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add every shared column the strategies and the confluence engine read."""
    f = bars.copy()
    c, h, l = f["close"], f["high"], f["low"]
    for n in (9, 20, 21, 50, 200):
        f[f"ema{n}"] = ema(c, n)
    f["rsi"] = rsi(c, 14)
    f["macd"], f["macd_sig"], f["macd_hist"] = macd(c)
    f["adx"], f["pdi"], f["mdi"] = adx(f, 14)
    f["atr"] = atr(f, 14)
    f["bb_mid"], f["bb_up"], f["bb_low"] = bollinger(c)
    f["kc_up"], f["kc_low"] = f["ema20"] + 1.5 * f["atr"], f["ema20"] - 1.5 * f["atr"]
    if "vwap" not in f.columns:                      # an index frame arrives with a futures-derived VWAP
        f["vwap"], f["vwap_sd"] = session_vwap(f)
    elif "vwap_sd" not in f.columns:
        f["vwap_sd"] = np.nan
    f["rvol"] = relative_volume(f["volume"], 20) if has_volume(f.get("volume")) else np.nan
    f["st_line"], f["st_dir"] = supertrend(f, 10, 3.0)

    day = day_key(f.index)
    g = f.groupby(day)
    f["bar_no"] = g.cumcount()
    f["day_open"] = g["open"].transform("first")
    f["day_hi_prev"] = g["high"].cummax().groupby(day).shift(1)       # session extreme BEFORE this bar
    f["day_lo_prev"] = g["low"].cummin().groupby(day).shift(1)
    d = daily_table(f, daily_ext)
    prev = d.shift(1)
    for col, name in (("high", "pd_high"), ("low", "pd_low"), ("close", "pd_close")):
        f[name] = prev[col].reindex(day).to_numpy()
    d_tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(), (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    d_atr = atr(d, 14).fillna(d_tr.rolling(14, min_periods=5).mean())       # short history: simple mean of 5+ sessions
    f["d_atr"] = d_atr.shift(1).reindex(day).to_numpy()
    de20, de50 = ema(d["close"], 20), ema(d["close"], 50)
    dtrend = pd.Series(np.where((d["close"] > de20) & (de20 > de50), 1.0,
                                np.where((d["close"] < de20) & (de20 < de50), -1.0, 0.0)), index=d.index).where(de50.notna())
    f["d_trend"] = dtrend.shift(1).reindex(day).to_numpy()
    p = (f["pd_high"] + f["pd_low"] + f["pd_close"]) / 3.0
    f["pp"], f["r1"], f["s1"] = p, 2 * p - f["pd_low"], 2 * p - f["pd_high"]
    f["r2"], f["s2"] = p + (f["pd_high"] - f["pd_low"]), p - (f["pd_high"] - f["pd_low"])
    bc = (f["pd_high"] + f["pd_low"]) / 2.0
    tc = 2 * p - bc
    f["cpr_top"], f["cpr_bot"] = np.maximum(tc, bc), np.minimum(tc, bc)
    f["cpr_width_pct"] = (f["cpr_top"] - f["cpr_bot"]) / p * 100.0
    f["gap_pct"] = (f["day_open"] / f["pd_close"] - 1.0) * 100.0
    f["day_ret_pct"] = (c / f["day_open"] - 1.0) * 100.0

    sw = swings(f, 3, 3)
    for col in sw.columns:
        f[col] = sw[col]
    hh, hl = f["swing_hi"] > f["prev_swing_hi"], f["swing_lo"] > f["prev_swing_lo"]
    lh, ll = f["swing_hi"] < f["prev_swing_hi"], f["swing_lo"] < f["prev_swing_lo"]
    f["structure"] = np.where(hh & hl, 1.0, np.where(lh & ll, -1.0, 0.0))

    vp = volume_profile(f)
    if len(vp):
        vp_prev = vp.reindex(d.index).shift(1)
        for col in ("poc", "vah", "val"):
            f[col] = vp_prev[col].reindex(day).to_numpy()
    else:
        f["poc"] = f["vah"] = f["val"] = np.nan
    rng = (h - l).replace(0.0, np.nan)
    f["close_pos"] = ((c - l) / rng).fillna(0.5)                      # 1 = closed at the high
    f["body_pct"] = ((c - f["open"]).abs() / rng).fillna(0.0)
    return f
