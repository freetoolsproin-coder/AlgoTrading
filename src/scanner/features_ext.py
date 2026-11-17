"""
Extended causal features for the v3 strategy layer (src/scanner/modules.py, regime.py, traps.py, score_v3.py).

Everything here is CAUSAL: the value on signal bar `pos` uses only that bar and earlier data, and swing
pivots are used only from the bar at which they become known (`known_at`). tests/test_v3.py checks this by
recomputing on truncated data.

Vectorised columns (one value per signal bar, computed once per FeatureFrame)
    Keltner channel + squeeze      kc_up / kc_low (EMA20 +/- kc_mult x ATR); squeeze_on = Bollinger inside Keltner
    squeeze_bars                   consecutive bars in a squeeze; squeeze_fired = squeeze ended on this bar
    nr4 / nr7                      this bar's range is the narrowest of the last 4 / 7 bars
    inside_bar                     high <= previous high and low >= previous low
    atr_pct                        percentile of ATR/close within the trailing window (0 = calmest seen)
    rv / rv_pct                    realised volatility (annualised, from log returns) and its percentile
    roc_fast / roc_slow            rate of change over 6 and 24 bars (30 min / 2 h on 5-min bars)
    persistence                    share of the last 6 closes that moved up (0..1; 0.5 = no persistence)
    vwap / vwap_source             futures session VWAP when real volume exists, else an equal-weighted
                                   session TWAP of the index, labelled "twap-proxy" (lower weight)
    vwap_dev / vwap_slope          volume-weighted deviation band width, VWAP change over 3 bars
    vwap_accept                    signed count of consecutive closes on one side of VWAP (+ above, - below)
    struct_trend / bos / choch     break of structure / change of character from confirmed pivots only
    htf15 / htf30                  15- and 30-minute trend (+1 / -1 / 0) from HTF bars COMPLETED by this bar's close

Per-day levels
    opening ranges for every window in [STRATEGY_V3] or_windows (default 5, 15, 30 minutes), the initial balance
    (first ib_minutes), and the day open - each known only once its window has closed.

Per-bar helpers (cheap; called by the modules for one bar)
    liquidity_pools(ext, pos)       equal highs / equal lows among recent confirmed pivots
    sweep(ext, pos, levels)         a wick through a level that closed back inside it (stop run)
    anchored_vwap(ext, pos, start)  VWAP anchored at an earlier bar (e.g. the last swing low)
    divergence(ext, pos)            price vs RSI at the last two confirmed swing highs / lows
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.scanner import indicators as ind
from src.utils.numbers import finite

BARS_PER_YEAR_5M = 75 * 252


def _pct_rank(series: pd.Series, lookback: int, min_periods: int = 30) -> pd.Series:
    """Percentile of the current value among the PREVIOUS values of the window (0 = lowest)."""
    return series.rolling(lookback, min_periods=min_periods).apply(
        lambda w: float((w[:-1] < w[-1]).mean()) if len(w) > 1 and np.isfinite(w[-1]) else np.nan, raw=True)


def _ints(text, default) -> Tuple[int, ...]:
    if isinstance(text, (tuple, list)):
        return tuple(int(x) for x in text)
    try:
        values = tuple(int(float(v)) for v in str(text).split(",") if v.strip())
        return values or tuple(default)
    except ValueError:
        return tuple(default)


@dataclass
class DayLevels:
    day_open: float = np.nan
    ranges: Dict[int, tuple] = field(default_factory=dict)   # minutes -> (high, low, known_from)
    ib: Optional[tuple] = None                               # (high, low, known_from)


@dataclass
class ExtFrame:
    ff: object                                   # strategies.FeatureFrame
    cols: Dict[str, np.ndarray]
    vwap_source: str                             # "futures" | "twap-proxy"
    days: Dict[date, DayLevels]
    ends: pd.DatetimeIndex                       # signal bar end times
    or_windows: Tuple[int, ...]
    ib_minutes: int
    eq_tol_atr: float

    def v(self, name: str, pos: int) -> float:
        arr = self.cols.get(name)
        if arr is None or pos < 0 or pos >= len(arr):
            return np.nan
        x = arr[pos]
        return float(x) if x == x else np.nan

    def day_levels(self, pos: int) -> Tuple[DayLevels, pd.Timestamp]:
        day = self.ff.dates[pos]
        return self.days.get(day, DayLevels()), self.ends[pos]

    def opening_range(self, pos: int, minutes: int) -> Tuple[float, float]:
        dl, end = self.day_levels(pos)
        r = dl.ranges.get(minutes)
        if r is None or end < r[2]:
            return np.nan, np.nan
        return r[0], r[1]

    def initial_balance(self, pos: int) -> Tuple[float, float]:
        dl, end = self.day_levels(pos)
        if dl.ib is None or end < dl.ib[2]:
            return np.nan, np.nan
        return dl.ib[0], dl.ib[1]


# =============================================================================
# Build
# =============================================================================
def build(ff, cfg) -> ExtFrame:
    """Compute every vectorised column once for a FeatureFrame (strategies.build_features output)."""
    bars = ff.bars
    n = len(bars)
    a = ff.arrays
    kc_mult = float(getattr(cfg, "kc_mult", 1.5))
    or_windows = _ints(getattr(cfg, "or_windows", "5,15,30"), (5, 15, 30))
    ib_minutes = int(getattr(cfg, "ib_minutes", 60))
    eq_tol = float(getattr(cfg, "eq_tol_atr", 0.10))
    cols: Dict[str, np.ndarray] = {}
    if n == 0:
        return ExtFrame(ff, cols, "none", {}, pd.DatetimeIndex([]), or_windows, ib_minutes, eq_tol)

    close = bars["close"].astype(float)
    high, low, open_ = bars["high"].astype(float), bars["low"].astype(float), bars["open"].astype(float)
    atr_s = bars["atr"].astype(float)
    ema20 = bars["ema20"].astype(float)
    day = pd.Series(ff.dates, index=bars.index)

    # ---- volatility: Keltner squeeze, NR4/NR7, inside bars, ATR / realised-vol percentiles
    kc_up, kc_low = ema20 + kc_mult * atr_s, ema20 - kc_mult * atr_s
    squeeze = (bars["bb_up"] < kc_up) & (bars["bb_low"] > kc_low)
    squeeze = squeeze.fillna(False).to_numpy()
    sq_bars = np.zeros(n)
    fired = np.zeros(n, dtype=bool)
    for i in range(n):
        sq_bars[i] = sq_bars[i - 1] + 1 if (i and squeeze[i]) else (1.0 if squeeze[i] else 0.0)
        fired[i] = bool(i and squeeze[i - 1] and not squeeze[i])
    rng = (high - low)
    cols["kc_up"], cols["kc_low"] = kc_up.to_numpy(), kc_low.to_numpy()
    cols["squeeze_on"] = squeeze.astype(float)
    cols["squeeze_bars"] = sq_bars
    cols["squeeze_fired"] = fired.astype(float)
    cols["nr4"] = (rng <= rng.rolling(4, min_periods=4).min()).astype(float).to_numpy()
    cols["nr7"] = (rng <= rng.rolling(7, min_periods=7).min()).astype(float).to_numpy()
    cols["inside_bar"] = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(float).to_numpy()
    cols["atr_pct"] = _pct_rank(atr_s / close, 300).to_numpy()
    logret = np.log(close / close.shift(1))
    rv = logret.rolling(20, min_periods=15).std(ddof=0) * np.sqrt(BARS_PER_YEAR_5M)
    cols["rv"] = rv.to_numpy()
    cols["rv_pct"] = _pct_rank(rv, 300).to_numpy()
    cols["range_atr"] = (rng / atr_s.shift(1)).to_numpy()

    # ---- momentum
    cols["roc_fast"] = (close / close.shift(6) - 1.0).to_numpy() * 100
    cols["roc_slow"] = (close / close.shift(24) - 1.0).to_numpy() * 100
    ups = (close.diff() > 0).astype(float).where(close.diff().notna())
    cols["persistence"] = ups.rolling(6, min_periods=6).mean().to_numpy()

    # ---- VWAP: futures with real volume, else an equal-weighted session TWAP proxy of the index
    fvwap = bars["f_vwap"].astype(float) if "f_vwap" in bars else pd.Series(np.nan, index=bars.index)
    fclose = bars["f_close"].astype(float) if "f_close" in bars else pd.Series(np.nan, index=bars.index)
    fvol = bars["f_volume"].astype(float) if "f_volume" in bars else pd.Series(np.nan, index=bars.index)
    if ff.has_participation and fvwap.notna().mean() > 0.5:
        source = "futures"
        price, vwap, weight = fclose, fvwap, fvol.fillna(0.0)
    else:
        source = "twap-proxy"
        price = close
        typical = (high + low + close) / 3.0
        vwap = typical.groupby(day).cumsum() / typical.groupby(day).cumcount().add(1)
        weight = pd.Series(1.0, index=bars.index)
    dev2 = (weight * (price - vwap) ** 2).groupby(day).cumsum() / weight.groupby(day).cumsum().replace(0.0, np.nan)
    cols["vwap"] = vwap.to_numpy()
    cols["vwap_price"] = price.to_numpy()
    cols["vwap_dev"] = np.sqrt(dev2).to_numpy()
    cols["vwap_slope"] = (vwap - vwap.groupby(day).shift(3)).to_numpy()
    side = np.sign((price - vwap).to_numpy())
    acc = np.zeros(n)
    dates = ff.dates
    for i in range(n):
        s = side[i] if np.isfinite(side[i]) else 0.0
        if i and dates[i] == dates[i - 1] and s != 0 and np.sign(acc[i - 1]) == s:
            acc[i] = acc[i - 1] + s
        else:
            acc[i] = s
    cols["vwap_accept"] = acc

    # ---- break of structure / change of character from pivots in the order they became known
    _structure_events(ff, cols, n)

    # ---- higher timeframes (15 / 30 minutes) mapped onto completed HTF bars only
    ends = bars.index + pd.Timedelta(minutes=cfg.signal_minutes)
    for minutes in _ints(getattr(cfg, "htf_minutes", "15,30"), (15, 30)):
        cols[f"htf{minutes}"] = _htf_trend(ff.minutes, minutes, ends, cfg)

    # ---- per-day opening ranges, initial balance and day open (known once their window closes)
    days = _day_levels(ff.minutes, or_windows, ib_minutes)
    return ExtFrame(ff, cols, source, days, ends, or_windows, ib_minutes, eq_tol)


def get(ff, cfg) -> ExtFrame:
    """Cached ExtFrame for a FeatureFrame (built on first use)."""
    ext = getattr(ff, "_ext", None)
    if ext is None or len(ext.ends) != len(ff.bars):
        ext = build(ff, cfg)
        try:
            ff._ext = ext
        except AttributeError:
            pass
    return ext


def _structure_events(ff, cols: Dict[str, np.ndarray], n: int):
    """Sequential pass: at each bar, the last confirmed swing high / low known by then. A close through the
    last swing high is a BOS up when the structure was already up, a CHoCH up when it was down (and the mirror
    for lows). struct_trend is +1 / -1 / 0 after the latest event."""
    closes = ff.arrays["close"]
    order = np.argsort(ff.piv_known, kind="stable") if len(ff.piv_known) else np.array([], dtype=int)
    k = 0
    last_h = last_l = np.nan
    trend = 0
    bos_up, bos_dn = np.zeros(n), np.zeros(n)
    ch_up, ch_dn = np.zeros(n), np.zeros(n)
    tr = np.zeros(n)
    since = np.full(n, np.nan)
    last_hi_arr, last_lo_arr = np.full(n, np.nan), np.full(n, np.nan)
    last_event = -10_000
    broken_h = broken_l = False
    for i in range(n):
        while k < len(order) and ff.piv_known[order[k]] <= i:
            j = order[k]
            if ff.piv_kind[j] == "H":
                last_h, broken_h = float(ff.piv_price[j]), False
            else:
                last_l, broken_l = float(ff.piv_price[j]), False
            k += 1
        c, pc = closes[i], closes[i - 1] if i else np.nan
        if finite(last_h) and not broken_h and c > last_h and (not finite(pc) or pc <= last_h):
            if trend >= 0:
                bos_up[i] = 1
            else:
                ch_up[i] = 1
            trend, broken_h, last_event = 1, True, i
        elif finite(last_l) and not broken_l and c < last_l and (not finite(pc) or pc >= last_l):
            if trend <= 0:
                bos_dn[i] = 1
            else:
                ch_dn[i] = 1
            trend, broken_l, last_event = -1, True, i
        tr[i] = trend
        since[i] = i - last_event if last_event >= 0 else np.nan
        last_hi_arr[i], last_lo_arr[i] = last_h, last_l
    cols.update(bos_up=bos_up, bos_dn=bos_dn, choch_up=ch_up, choch_dn=ch_dn, struct_trend=tr,
                bars_since_struct=since, last_swing_high=last_hi_arr, last_swing_low=last_lo_arr)


def _htf_trend(minutes: pd.DataFrame, htf: int, ends: pd.DatetimeIndex, cfg) -> np.ndarray:
    """+1 / -1 / 0 per signal bar from the latest HTF bar that had CLOSED by the signal bar's end."""
    out = np.zeros(len(ends))
    if minutes is None or minutes.empty:
        return out
    hb = ind.resample(minutes, htf)
    if len(hb) < 25:
        return out
    hb = hb[hb["minutes_in_bar"] >= max(1, htf - 1)] if "minutes_in_bar" in hb else hb
    c = hb["close"]
    e20, e50 = ind.ema(c, 20), ind.ema(c, 50)
    adx_v, pdi, mdi = ind.adx(hb, 14)
    score = (np.sign(c - e20).fillna(0) + np.sign(e20 - e50).fillna(0) + np.sign(pdi - mdi).fillna(0)) / 3.0
    trend = np.where(score >= 0.66, 1.0, np.where(score <= -0.66, -1.0, 0.0))
    trend = np.where(e50.notna().to_numpy(), trend, 0.0)
    hb_ends = hb.index + pd.Timedelta(minutes=htf)
    idx = hb_ends.searchsorted(ends, side="right") - 1
    valid = idx >= 0
    out[valid] = trend[idx[valid]]
    return out


def _day_levels(minutes: pd.DataFrame, windows: Tuple[int, ...], ib_minutes: int) -> Dict[date, DayLevels]:
    out: Dict[date, DayLevels] = {}
    if minutes is None or minutes.empty:
        return out
    clock = minutes.index.time
    m = minutes[(clock >= ind.SESSION_OPEN) & (clock < ind.SESSION_CLOSE)]
    if m.empty:
        return out
    tz = m.index.tz
    for day, g in m.groupby(m.index.date):
        g_mod = (g.index.hour * 60 + g.index.minute) - (9 * 60 + 15)
        start = pd.Timestamp(datetime.combine(day, time(9, 15)))
        start = start.tz_localize(tz) if tz is not None else start
        dl = DayLevels(day_open=float(g["open"].iloc[0]))
        for w in windows:
            win = g[g_mod < w]
            if len(win) >= max(1, w // 2):
                dl.ranges[w] = (float(win["high"].max()), float(win["low"].min()), start + pd.Timedelta(minutes=w))
        win = g[g_mod < ib_minutes]
        if len(win) >= max(1, ib_minutes // 2):
            dl.ib = (float(win["high"].max()), float(win["low"].min()), start + pd.Timedelta(minutes=ib_minutes))
        out[day] = dl
    return out


# =============================================================================
# Per-bar helpers
# =============================================================================
def known_pivots(ext: ExtFrame, pos: int, lookback: int = 75):
    ff = ext.ff
    if not len(ff.piv_known):
        return np.array([]), np.array([]), np.array([])
    mask = (ff.piv_known <= pos) & (ff.piv_pos >= pos - lookback)
    return ff.piv_pos[mask], ff.piv_kind[mask], ff.piv_price[mask]


def liquidity_pools(ext: ExtFrame, pos: int, atr_v: float, lookback: int = 75) -> Dict[str, List[float]]:
    """Equal highs / equal lows: two or more confirmed swing highs (lows) within eq_tol_atr x ATR of each other.
    Resting stops are commonly assumed to sit just beyond such levels - an assumption, not an observation."""
    pp, kk, px = known_pivots(ext, pos, lookback)
    out = {"equal_highs": [], "equal_lows": []}
    if not len(pp) or not finite(atr_v) or atr_v <= 0:
        return out
    tol = ext.eq_tol_atr * atr_v
    for kind, key, agg in (("H", "equal_highs", max), ("L", "equal_lows", min)):
        prices = sorted(float(p) for p, k in zip(px, kk) if k == kind)
        i = 0
        while i < len(prices):
            j = i
            while j + 1 < len(prices) and prices[j + 1] - prices[i] <= tol:
                j += 1
            if j > i:
                out[key].append(agg(prices[i:j + 1]))
            i = j + 1
    return out


def sweep(ext: ExtFrame, pos: int, levels: List[Tuple[float, str]], atr_v: float, lookback: int = 3) -> dict:
    """A stop run within the last `lookback` completed bars: the bar's wick went beyond a level by at least
    0.05 ATR but the bar CLOSED back inside it.
        sweep of a level above (buy-side liquidity)   -> bearish evidence (PE reversal candidate)
        sweep of a level below (sell-side liquidity)  -> bullish evidence (CE reversal candidate)
    Returns {"bull": [(level, label, bars_ago)], "bear": [...]}."""
    a = ext.ff.arrays
    out = {"bull": [], "bear": []}
    if not finite(atr_v) or atr_v <= 0:
        return out
    eps = 0.05 * atr_v
    for back in range(lookback):
        i = pos - back
        if i < 1 or ext.ff.dates[i] != ext.ff.dates[pos]:
            break
        h, l, c = a["high"][i], a["low"][i], a["close"][i]
        for lvl, label in levels:
            if not finite(lvl):
                continue
            # the level must have been intact before this bar (previous close on the near side)
            pc = a["close"][i - 1]
            if h > lvl + eps and c < lvl and pc < lvl:
                out["bear"].append((float(lvl), label, back))
            if l < lvl - eps and c > lvl and pc > lvl:
                out["bull"].append((float(lvl), label, back))
    return out


def anchored_vwap(ext: ExtFrame, pos: int, start: int) -> float:
    """VWAP of typical price from bar `start` to `pos` inclusive (futures volume as weight when available,
    equal weights otherwise)."""
    if start is None or start < 0 or start > pos:
        return np.nan
    a = ext.ff.arrays
    tp = (a["high"][start:pos + 1] + a["low"][start:pos + 1] + a["close"][start:pos + 1]) / 3.0
    if ext.vwap_source == "futures" and "f_volume" in a:
        w = np.nan_to_num(a["f_volume"][start:pos + 1], nan=0.0)
        if w.sum() > 0:
            return float((tp * w).sum() / w.sum())
    return float(tp.mean())


def divergence(ext: ExtFrame, pos: int, lookback: int = 75) -> dict:
    """Regular divergence on the last two confirmed swings: price higher high with RSI lower high = bearish;
    price lower low with RSI higher low = bullish. Returns {"bull": bool, "bear": bool, "notes": [...]}"""
    pp, kk, px = known_pivots(ext, pos, lookback)
    rsi = ext.ff.arrays.get("rsi")
    out = {"bull": False, "bear": False, "notes": []}
    if rsi is None or not len(pp):
        return out
    for kind in ("H", "L"):
        sel = [(int(p), float(v)) for p, k, v in zip(pp, kk, px) if k == kind]
        if len(sel) < 2:
            continue
        (p1, v1), (p2, v2) = sel[-2], sel[-1]
        r1, r2 = rsi[p1], rsi[p2]
        if not (finite(r1) and finite(r2)):
            continue
        if kind == "H" and v2 > v1 and r2 < r1 - 2:
            out["bear"] = True
            out["notes"].append(f"bearish divergence: high {v2:,.0f} > {v1:,.0f}, RSI {r2:.0f} < {r1:.0f}")
        if kind == "L" and v2 < v1 and r2 > r1 + 2:
            out["bull"] = True
            out["notes"].append(f"bullish divergence: low {v2:,.0f} < {v1:,.0f}, RSI {r2:.0f} > {r1:.0f}")
    return out


def last_swing_pos(ext: ExtFrame, pos: int, kind: str, lookback: int = 75) -> Optional[int]:
    pp, kk, _ = known_pivots(ext, pos, lookback)
    sel = [int(p) for p, k in zip(pp, kk) if k == kind]
    return sel[-1] if sel else None


def exit_context(ff, ev, cfg) -> dict:
    """Inputs for the trailing / VWAP / momentum exits, from the last COMPLETED signal bar only."""
    pos = ev.position
    out = {"atr": ev.atr, "swing_low": (ev.structure or {}).get("last_low"),
           "swing_high": (ev.structure or {}).get("last_high")}
    try:
        ext = get(ff, cfg)
        vw, price = ext.v("vwap", pos), ext.v("vwap_price", pos)
        if finite(vw) and finite(price):
            out["vwap"] = ev.close - (price - vw)          # on the index scale
    except Exception:
        pass
    rsi, rsi_s, hist_s = ff.v("rsi", pos), ff.v("rsi_slope", pos), ff.v("macd_hist_slope", pos)
    if finite(rsi) and finite(rsi_s) and finite(hist_s):
        out["momentum_exhausted_ce"] = bool(rsi >= 65 and rsi_s < 0 and hist_s < 0)
        out["momentum_exhausted_pe"] = bool(rsi <= 35 and rsi_s > 0 and hist_s > 0)
    return out
