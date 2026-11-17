"""
The rest of the strategy library (src/movement/strategies.py holds the first fifteen and the engine).

Every function here has the same contract as the original fifteen:

  * input: a Context of COMPLETED bars (signal timeframe `b5`, 15-min `b15`, 1-min `m1`, futures `f5` / `f1`,
    daily candles, option chain, India VIX samples, the other index's bars for the pair strategies);
  * output: Result(direction "UP" / "DOWN" / None, reason, values). UP = buy a CE, DOWN = buy a PE;
  * a signal is an EVENT on the last completed bar (a cross, a first close beyond a level), not a state, so
    a running trend does not fire on every bar;
  * missing data gives no signal with a reason starting "no data:" or "warming up:", never a guess.

Stop-loss, target, breakeven, trailing stop, time stop, daily trade / loss limits are NOT decided here: every
strategy is its own paper book with those settings in [STRATEGIES] / [STRATEGY_<KEY>] (premium %, the same
exit engine as every other book). Every parameter below is an untested starting value.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.movement.strategies import (DOWN, UP, Context, Result, Spec, _cross_dn, _cross_up, _need, _pivot_lows,
                                     _rvol, _sample_at, _sig, supertrend)
from src.scanner import indicators as ind
from src.utils.numbers import finite, hhmm


# ============================================================================================ helpers
def _first(cond: pd.Series) -> bool:
    """True on the bar where `cond` becomes true (false on the previous bar)."""
    return len(cond) >= 2 and bool(cond.iloc[-1]) and not bool(cond.iloc[-2])


def _first_today(cond: pd.Series) -> bool:
    """True when the last bar is the first bar of the series (today's bars) on which `cond` holds."""
    return len(cond) >= 1 and bool(cond.iloc[-1]) and not bool(cond.iloc[:-1].any())


def range_break(b: pd.DataFrame, n: int) -> tuple:
    """(first close above the previous n-bar high, first close below the previous n-bar low) on the last bar."""
    if len(b) < n + 2:
        return False, False
    hi = b["high"].shift(1).rolling(n).max()
    lo = b["low"].shift(1).rolling(n).min()
    return _first(b["close"] > hi), _first(b["close"] < lo)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).mean()


def prev_daily(ctx: Context) -> pd.DataFrame:
    """Daily candles strictly BEFORE today (from the snapshot, else built from the 1-min history)."""
    d = ctx.daily
    if d is None or len(d) == 0 or not {"high", "low", "close"} <= set(d.columns):
        d = ind.daily_from_minutes(ctx.m1)
    if d is None or len(d) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    idx = pd.to_datetime(d.index)
    return d[np.asarray([x.date() for x in idx]) < ctx.today]


def fut_session(ctx: Context) -> Optional[pd.DataFrame]:
    """Today's futures bars with session VWAP, EMA9/21, ATR, or None without futures volume."""
    f = ctx.today_bars(ctx.f5)
    if f is None or len(f) < 3 or not ind.has_volume(f["volume"]):
        return None
    f = f.copy()
    f["vwap"] = ind.session_vwap(f)
    full = ctx.f5
    f["atr"] = ind.atr(full, 14).reindex(f.index)
    f["ema9"] = ind.ema(full["close"], 9).reindex(f.index)
    f["ema21"] = ind.ema(full["close"], 21).reindex(f.index)
    return f


def stochastic(b: pd.DataFrame, k: int = 14, smooth: int = 3, d: int = 3):
    hh, ll = b["high"].rolling(k).max(), b["low"].rolling(k).min()
    raw = 100 * (b["close"] - ll) / (hh - ll).replace(0, np.nan)
    ks = raw.rolling(smooth).mean()
    return ks, ks.rolling(d).mean()


def cci(b: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (b["high"] + b["low"] + b["close"]) / 3
    mean = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - mean) / (0.015 * mad.replace(0, np.nan))


def williams_r(b: pd.DataFrame, n: int = 14) -> pd.Series:
    hh, ll = b["high"].rolling(n).max(), b["low"].rolling(n).min()
    return -100 * (hh - b["close"]) / (hh - ll).replace(0, np.nan)


def parabolic_sar(b: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """+1 / -1 trend of Wilder's Parabolic SAR."""
    h, l = b["high"].to_numpy(float), b["low"].to_numpy(float)
    n = len(b)
    trend = np.full(n, np.nan)
    if n < 3:
        return pd.Series(trend, index=b.index)
    up = h[1] >= h[0]
    sar = l[0] if up else h[0]
    ep = h[1] if up else l[1]
    af = step
    trend[1] = 1.0 if up else -1.0
    for i in range(2, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[i - 2])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
        trend[i] = 1.0 if up else -1.0
    return pd.Series(trend, index=b.index)


def ichimoku(b: pd.DataFrame, t: int = 9, k: int = 26, s: int = 52) -> pd.DataFrame:
    mid = lambda n: (b["high"].rolling(n).max() + b["low"].rolling(n).min()) / 2
    tenkan, kijun = mid(t), mid(k)
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun,
                         "span_a": ((tenkan + kijun) / 2).shift(k),     # plotted k bars ahead: known today
                         "span_b": mid(s).shift(k)}, index=b.index)


def keltner(b: pd.DataFrame, n: int = 20, mult: float = 1.5):
    mid = ind.ema(b["close"], n)
    a = ind.atr(b, n)
    return mid, mid + mult * a, mid - mult * a


def bollinger(b: pd.DataFrame, n: int = 20, k: float = 2.0):
    mid = sma(b["close"], n)
    sd = b["close"].rolling(n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def floor_pivots(d: pd.Series) -> Dict[str, float]:
    h, l, c = float(d["high"]), float(d["low"]), float(d["close"])
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + (h - l), "S2": p - (h - l)}


def _pair_frame(ctx: Context) -> Optional[tuple]:
    """(this index closes, other index closes, other name) on common completed signal bars, or None."""
    others = {k: v for k, v in (ctx.peers or {}).items() if k != ctx.index}
    if not others:
        return None
    name, other = sorted(others.items())[0]
    a, b = ctx.b5["close"], other.b5["close"] if other.b5 is not None else None
    if b is None or len(a) == 0 or len(b) == 0:
        return None
    common = a.index.intersection(b.index)
    return a.reindex(common), b.reindex(common), name


# ============================================================================================ trend
def ma_crossover(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, int(p["slow"]) + 2)
    if miss:
        return miss
    f, s = sma(b["close"], p["fast"]), sma(b["close"], p["slow"])
    vals = dict(fast=f.iloc[-1], slow=s.iloc[-1])
    if _cross_up(f, s):
        return _sig(UP, f"SMA{p['fast']} crossed above SMA{p['slow']}", b.index[-1], **vals)
    if _cross_dn(f, s):
        return _sig(DOWN, f"SMA{p['fast']} crossed below SMA{p['slow']}", b.index[-1], **vals)
    return _sig(None, f"no SMA{p['fast']}/{p['slow']} cross", b.index[-1], **vals)


def ema_9_21(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, int(p["slow"]) + 5)
    if miss:
        return miss
    f, s = ind.ema(b["close"], p["fast"]), ind.ema(b["close"], p["slow"])
    c = b["close"].iloc[-1]
    vals = dict(ema_fast=f.iloc[-1], ema_slow=s.iloc[-1])
    if _cross_up(f, s) and c > f.iloc[-1]:
        return _sig(UP, f"EMA{p['fast']} crossed above EMA{p['slow']}, close above both", b.index[-1], **vals)
    if _cross_dn(f, s) and c < f.iloc[-1]:
        return _sig(DOWN, f"EMA{p['fast']} crossed below EMA{p['slow']}, close below both", b.index[-1], **vals)
    return _sig(None, f"no EMA{p['fast']}/{p['slow']} cross", b.index[-1], **vals)


def ema_20_50_200(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 205)
    if miss:
        return miss
    c = b["close"]
    e20, e50, e200 = ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
    up = (e20 > e50) & (e50 > e200) & (c > e20)
    dn = (e20 < e50) & (e50 < e200) & (c < e20)
    vals = dict(ema20=e20.iloc[-1], ema50=e50.iloc[-1], ema200=e200.iloc[-1])
    if _first(up):
        return _sig(UP, "EMA20 > EMA50 > EMA200 with the close above EMA20 (new alignment / pull-back resumed)",
                    b.index[-1], **vals)
    if _first(dn):
        return _sig(DOWN, "EMA20 < EMA50 < EMA200 with the close below EMA20 (new alignment / pull-back resumed)",
                    b.index[-1], **vals)
    state = "aligned up" if up.iloc[-1] else "aligned down" if dn.iloc[-1] else "not aligned"
    return _sig(None, f"EMAs {state}, no new event", b.index[-1], **vals)


def donchian(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["length"])
    miss = _need(b, n + 2)
    if miss:
        return miss
    hi, lo = b["high"].shift(1).rolling(n).max(), b["low"].shift(1).rolling(n).min()
    vals = dict(upper=hi.iloc[-1], lower=lo.iloc[-1])
    if _first(b["close"] > hi):
        return _sig(UP, f"close above the {n}-bar Donchian high {hi.iloc[-1]:.1f}", b.index[-1], **vals)
    if _first(b["close"] < lo):
        return _sig(DOWN, f"close below the {n}-bar Donchian low {lo.iloc[-1]:.1f}", b.index[-1], **vals)
    return _sig(None, "inside the Donchian channel", b.index[-1], **vals)


def ichimoku_cloud(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 80)
    if miss:
        return miss
    ic = ichimoku(b, int(p["tenkan"]), int(p["kijun"]), int(p["senkou"]))
    top = ic[["span_a", "span_b"]].max(axis=1)
    bottom = ic[["span_a", "span_b"]].min(axis=1)
    c = b["close"]
    vals = dict(tenkan=ic["tenkan"].iloc[-1], kijun=ic["kijun"].iloc[-1], cloud_top=top.iloc[-1],
                cloud_bottom=bottom.iloc[-1])
    if not finite(top.iloc[-1]):
        return Result(reason="warming up: Ichimoku cloud")
    if _cross_up(ic["tenkan"], ic["kijun"]) and c.iloc[-1] > top.iloc[-1]:
        return _sig(UP, "Tenkan crossed above Kijun with the close above the cloud", b.index[-1], **vals)
    if _cross_dn(ic["tenkan"], ic["kijun"]) and c.iloc[-1] < bottom.iloc[-1]:
        return _sig(DOWN, "Tenkan crossed below Kijun with the close below the cloud", b.index[-1], **vals)
    return _sig(None, "no TK cross on the right side of the cloud", b.index[-1], **vals)


def psar(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 30)
    if miss:
        return miss
    t = parabolic_sar(b, float(p["step"]), float(p["max_step"]))
    if t.iloc[-1] > 0 > t.iloc[-2]:
        return _sig(UP, "Parabolic SAR flipped below price", b.index[-1])
    if t.iloc[-1] < 0 < t.iloc[-2]:
        return _sig(DOWN, "Parabolic SAR flipped above price", b.index[-1])
    return _sig(None, f"SAR {'below' if t.iloc[-1] > 0 else 'above'} price, no flip", b.index[-1])


# ============================================================================================ momentum
def rsi_reversal(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 30)
    if miss:
        return miss
    r = ind.rsi(b["close"], int(p["length"]))
    last, prev = b.iloc[-1], b.iloc[-2]
    vals = dict(rsi=r.iloc[-1])
    if _cross_up(r, p["oversold"]) and last["close"] > prev["high"]:
        return _sig(UP, f"RSI back above {p['oversold']:g} with a close above the previous high", b.index[-1], **vals)
    if _cross_dn(r, p["overbought"]) and last["close"] < prev["low"]:
        return _sig(DOWN, f"RSI back below {p['overbought']:g} with a close below the previous low", b.index[-1],
                    **vals)
    return _sig(None, f"RSI {r.iloc[-1]:.1f}, no reversal", b.index[-1], **vals)


def macd_crossover(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 40)
    if miss:
        return miss
    m, s, h = b["macd"], b["macd_sig"], b["macd_hist"]
    vals = dict(macd=m.iloc[-1], signal=s.iloc[-1], hist=h.iloc[-1])
    zero = bool(int(p["zero_filter"]))
    if _cross_up(m, s) and (not zero or m.iloc[-1] < 0):
        return _sig(UP, "MACD crossed above its signal line" + (" below zero" if zero else ""), b.index[-1], **vals)
    if _cross_dn(m, s) and (not zero or m.iloc[-1] > 0):
        return _sig(DOWN, "MACD crossed below its signal line" + (" above zero" if zero else ""), b.index[-1], **vals)
    return _sig(None, "no MACD cross", b.index[-1], **vals)


def stochastic_osc(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 30)
    if miss:
        return miss
    k, d = stochastic(b, int(p["k"]), int(p["smooth"]), int(p["d"]))
    vals = dict(k=k.iloc[-1], d=d.iloc[-1])
    if _cross_up(k, d) and min(k.iloc[-2], d.iloc[-2]) <= p["oversold"]:
        return _sig(UP, f"%K crossed above %D from below {p['oversold']:g}", b.index[-1], **vals)
    if _cross_dn(k, d) and max(k.iloc[-2], d.iloc[-2]) >= p["overbought"]:
        return _sig(DOWN, f"%K crossed below %D from above {p['overbought']:g}", b.index[-1], **vals)
    return _sig(None, "no %K/%D cross in an extreme zone", b.index[-1], **vals)


def roc_momentum(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["length"])
    miss = _need(b, n + 3)
    if miss:
        return miss
    roc = b["close"] / b["close"].shift(n) * 100 - 100
    vals = dict(roc_pct=roc.iloc[-1])
    if _cross_up(roc, p["threshold_pct"]):
        return _sig(UP, f"{n}-bar rate of change rose through +{p['threshold_pct']:g}%", b.index[-1], **vals)
    if _cross_dn(roc, -p["threshold_pct"]):
        return _sig(DOWN, f"{n}-bar rate of change fell through -{p['threshold_pct']:g}%", b.index[-1], **vals)
    return _sig(None, f"ROC {roc.iloc[-1]:+.2f}%", b.index[-1], **vals)


def cci_momentum(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["length"])
    miss = _need(b, n + 5)
    if miss:
        return miss
    c = cci(b, n)
    vals = dict(cci=c.iloc[-1])
    if _cross_up(c, p["level"]):
        return _sig(UP, f"CCI rose through +{p['level']:g}", b.index[-1], **vals)
    if _cross_dn(c, -p["level"]):
        return _sig(DOWN, f"CCI fell through -{p['level']:g}", b.index[-1], **vals)
    return _sig(None, f"CCI {c.iloc[-1]:.0f}", b.index[-1], **vals)


def williams(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["length"])
    miss = _need(b, n + 3)
    if miss:
        return miss
    w = williams_r(b, n)
    vals = dict(williams_r=w.iloc[-1])
    if _cross_up(w, p["upper"]) and b["close"].iloc[-1] > b["ema20"].iloc[-1]:
        return _sig(UP, f"%R rose into the top zone ({p['upper']:g}) above EMA20", b.index[-1], **vals)
    if _cross_dn(w, p["lower"]) and b["close"].iloc[-1] < b["ema20"].iloc[-1]:
        return _sig(DOWN, f"%R fell into the bottom zone ({p['lower']:g}) below EMA20", b.index[-1], **vals)
    return _sig(None, f"%R {w.iloc[-1]:.0f}", b.index[-1], **vals)


# ============================================================================================ price action
def sr_breakout(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n, k = int(p["lookback"]), int(p["pivot"])
    miss = _need(b, n + 20)
    if miss:
        return miss
    w = b.iloc[-n - 1:-1]                       # levels from bars BEFORE the signal bar
    atr_v = b["atr"].iloc[-1]
    if not finite(atr_v) or atr_v <= 0:
        return Result(reason="warming up: ATR")
    tol = p["touch_atr"] * atr_v
    highs = [float(w["high"].iloc[i]) for i in _pivot_lows(-w["high"], k)]
    lows = [float(w["low"].iloc[i]) for i in _pivot_lows(w["low"], k)]

    def tested(levels):
        return [lv for lv in levels if sum(abs(x - lv) <= tol for x in levels) >= int(p["min_touches"])]

    res, sup = tested(highs), tested(lows)
    last, prev = b.iloc[-1], b.iloc[-2]
    r = max(res) if res else np.nan
    s = min(sup) if sup else np.nan
    vals = dict(resistance=r, support=s)
    if finite(r) and last["close"] > r + p["buffer_atr"] * atr_v and prev["close"] <= r:
        return _sig(UP, f"broke resistance {r:.1f} (tested {int(p['min_touches'])}+ times)", b.index[-1], **vals)
    if finite(s) and last["close"] < s - p["buffer_atr"] * atr_v and prev["close"] >= s:
        return _sig(DOWN, f"broke support {s:.1f} (tested {int(p['min_touches'])}+ times)", b.index[-1], **vals)
    return _sig(None, "no break of a tested level", b.index[-1], **vals)


def pdh_pdl_breakout(ctx: Context, p: dict) -> Result:
    d = prev_daily(ctx)
    b = ctx.today_bars(ctx.b5)
    if len(d) == 0:
        return Result(reason="no data: previous day candle")
    if len(b) == 0:
        return Result(reason="waiting for today's first bar")
    pdh, pdl = float(d["high"].iloc[-1]), float(d["low"].iloc[-1])
    atr_v = ctx.b5["atr"].iloc[-1]
    buf = p["buffer_atr"] * atr_v if finite(atr_v) else 0.0
    vals = dict(pdh=pdh, pdl=pdl)
    if ctx.now.time() > hhmm(p["cutoff"]):
        return _sig(None, f"after the cut-off {p['cutoff']}", b.index[-1], **vals)
    if _first_today(b["close"] > pdh + buf):
        return _sig(UP, f"first close above the previous day high {pdh:.1f}", b.index[-1], **vals)
    if _first_today(b["close"] < pdl - buf):
        return _sig(DOWN, f"first close below the previous day low {pdl:.1f}", b.index[-1], **vals)
    return _sig(None, f"no first break of {pdl:.1f}-{pdh:.1f}", b.index[-1], **vals)


def inside_bar(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 20)
    if miss:
        return miss
    mother, inner, last = b.iloc[-3], b.iloc[-2], b.iloc[-1]
    is_inside = inner["high"] <= mother["high"] and inner["low"] >= mother["low"]
    small = (mother["high"] - mother["low"]) <= p["max_mother_atr"] * last["atr"] if finite(last["atr"]) else True
    vals = dict(mother_high=mother["high"], mother_low=mother["low"])
    if not (is_inside and small):
        return _sig(None, "no inside bar before this bar", b.index[-1], **vals)
    if last["close"] > mother["high"]:
        return _sig(UP, f"inside-bar breakout above {mother['high']:.1f}", b.index[-1], **vals)
    if last["close"] < mother["low"]:
        return _sig(DOWN, f"inside-bar breakdown below {mother['low']:.1f}", b.index[-1], **vals)
    return _sig(None, "inside bar not broken", b.index[-1], **vals)


def engulfing(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["trend_bars"])
    miss = _need(b, n + 25)
    if miss:
        return miss
    a, z = b.iloc[-2], b.iloc[-1]
    body = abs(z["close"] - z["open"])
    big = finite(z["atr"]) and body >= p["min_body_atr"] * z["atr"]
    falling = b["close"].iloc[-2] < b["close"].iloc[-2 - n]
    rising = b["close"].iloc[-2] > b["close"].iloc[-2 - n]
    vals = dict(body_atr=body / z["atr"] if finite(z["atr"]) and z["atr"] > 0 else np.nan)
    if big and falling and a["close"] < a["open"] and z["close"] > z["open"] \
            and z["close"] >= a["open"] and z["open"] <= a["close"]:
        return _sig(UP, f"bullish engulfing after a {n}-bar decline", b.index[-1], **vals)
    if big and rising and a["close"] > a["open"] and z["close"] < z["open"] \
            and z["close"] <= a["open"] and z["open"] >= a["close"]:
        return _sig(DOWN, f"bearish engulfing after a {n}-bar rise", b.index[-1], **vals)
    return _sig(None, "no engulfing candle after a move", b.index[-1], **vals)


def pin_bar(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["lookback"])
    miss = _need(b, n + 20)
    if miss:
        return miss
    z = b.iloc[-1]
    rng = z["high"] - z["low"]
    if rng <= 0 or not finite(z["atr"]) or rng < p["min_range_atr"] * z["atr"]:
        return _sig(None, "bar too small for a pin bar", b.index[-1])
    lower = min(z["open"], z["close"]) - z["low"]
    upper = z["high"] - max(z["open"], z["close"])
    before = b.iloc[-n - 1:-1]
    vals = dict(lower_wick=lower / rng, upper_wick=upper / rng)
    if lower / rng >= p["wick_ratio"] and z["low"] <= before["low"].min() and min(z["open"], z["close"]) >= \
            z["low"] + 0.6 * rng:
        return _sig(UP, f"bullish pin bar rejecting a new {n}-bar low", b.index[-1], **vals)
    if upper / rng >= p["wick_ratio"] and z["high"] >= before["high"].max() and max(z["open"], z["close"]) <= \
            z["high"] - 0.6 * rng:
        return _sig(DOWN, f"bearish pin bar rejecting a new {n}-bar high", b.index[-1], **vals)
    return _sig(None, "no pin bar at an extreme", b.index[-1], **vals)


def vwap_breakout(ctx: Context, p: dict) -> Result:
    f = fut_session(ctx)
    if f is None:
        return Result(reason="no data: futures volume needed for VWAP")
    if len(f) < 4:
        return Result(reason="warming up: VWAP")
    rvol = _rvol(ctx.f5)
    vals = dict(vwap=f["vwap"].iloc[-1], fut=f["close"].iloc[-1], rvol=rvol)
    if not (finite(rvol) and rvol >= p["min_rvol"]):
        return _sig(None, f"volume not confirming (rvol {rvol:.2f} < {p['min_rvol']})" if finite(rvol)
                    else "no volume history", f.index[-1], **vals)
    if _cross_up(f["close"], f["vwap"]):
        return _sig(UP, "futures closed above VWAP on rising volume", f.index[-1], **vals)
    if _cross_dn(f["close"], f["vwap"]):
        return _sig(DOWN, "futures closed below VWAP on rising volume", f.index[-1], **vals)
    return _sig(None, "no VWAP cross", f.index[-1], **vals)


# ============================================================================================ volatility
def bollinger_breakout(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 40)
    if miss:
        return miss
    mid, up, lo = bollinger(b, int(p["length"]), float(p["k"]))
    width = (up - lo) / mid
    expanding = width.iloc[-1] > width.iloc[-2]
    vals = dict(bb_up=up.iloc[-1], bb_lo=lo.iloc[-1], width_pct=width.iloc[-1] * 100)
    if expanding and _first(b["close"] > up):
        return _sig(UP, "close above the upper Bollinger band, bands widening", b.index[-1], **vals)
    if expanding and _first(b["close"] < lo):
        return _sig(DOWN, "close below the lower Bollinger band, bands widening", b.index[-1], **vals)
    return _sig(None, "no band breakout", b.index[-1], **vals)


def bollinger_squeeze(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 50)
    if miss:
        return miss
    mid, up, lo = bollinger(b, 20, 2.0)
    _kmid, kup, klo = keltner(b, 20, float(p["kc_mult"]))
    squeeze = (up < kup) & (lo > klo)
    recent = squeeze.iloc[-int(p["min_squeeze_bars"]) - 1:-1]
    released = len(recent) and bool(recent.all()) and not bool(squeeze.iloc[-1])
    mom = b["close"].iloc[-1] - mid.iloc[-1]
    vals = dict(squeeze=bool(squeeze.iloc[-1]), momentum=mom)
    if released and mom > 0 and b["close"].iloc[-1] > b["close"].iloc[-2]:
        return _sig(UP, "Bollinger squeeze released upward", b.index[-1], **vals)
    if released and mom < 0 and b["close"].iloc[-1] < b["close"].iloc[-2]:
        return _sig(DOWN, "Bollinger squeeze released downward", b.index[-1], **vals)
    return _sig(None, "squeeze on" if squeeze.iloc[-1] else "no squeeze release", b.index[-1], **vals)


def keltner_channel(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 40)
    if miss:
        return miss
    _mid, up, lo = keltner(b, int(p["length"]), float(p["mult"]))
    vals = dict(kc_up=up.iloc[-1], kc_lo=lo.iloc[-1])
    if _first(b["close"] > up):
        return _sig(UP, "close above the upper Keltner channel", b.index[-1], **vals)
    if _first(b["close"] < lo):
        return _sig(DOWN, "close below the lower Keltner channel", b.index[-1], **vals)
    return _sig(None, "inside the Keltner channel", b.index[-1], **vals)


def volatility_expansion(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["narrow_bars"])
    miss = _need(b, n + 20)
    if miss:
        return miss
    rng = b["high"] - b["low"]
    prev = b.iloc[-2]
    narrow = rng.iloc[-2] <= rng.iloc[-n - 1:-1].min()
    z = b.iloc[-1]
    wide = finite(z["atr"]) and rng.iloc[-1] >= p["expansion_atr"] * z["atr"]
    vals = dict(range_atr=rng.iloc[-1] / z["atr"] if finite(z["atr"]) and z["atr"] > 0 else np.nan)
    if narrow and wide and z["close"] > prev["high"]:
        return _sig(UP, f"range expansion up after the narrowest bar of {n}", b.index[-1], **vals)
    if narrow and wide and z["close"] < prev["low"]:
        return _sig(DOWN, f"range expansion down after the narrowest bar of {n}", b.index[-1], **vals)
    return _sig(None, "no narrow-range bar followed by expansion", b.index[-1], **vals)


def india_vix(ctx: Context, p: dict) -> Result:
    mem = ctx.memory.vix
    if not mem:
        return Result(reason="no data: India VIX")
    b = ctx.b5
    miss = _need(b, 25)
    if miss:
        return miss
    now_v = mem[-1][1]
    old = _sample_at(mem, ctx.now - pd.Timedelta(minutes=p["minutes"]))
    if old is None or not finite(old[1]) or old[1] <= 0:
        return Result(reason=f"warming up: needs {p['minutes']:g} min of VIX samples", values={"vix": now_v})
    chg = now_v / old[1] * 100 - 100
    n = int(p["lookback"])
    hi, lo = b["high"].iloc[-n - 1:-1].max(), b["low"].iloc[-n - 1:-1].min()
    brk_up, brk_dn = range_break(b, n)
    vals = dict(vix=now_v, vix_change_pct=chg, high_n=hi, low_n=lo)
    if chg <= -p["min_change_pct"] and brk_up and now_v <= p["max_vix_for_ce"]:
        return _sig(UP, f"VIX {chg:+.1f}% in {p['minutes']:g} min (fear easing), close above the {n}-bar high",
                    b.index[-1], **vals)
    if chg >= p["min_change_pct"] and brk_dn:
        return _sig(DOWN, f"VIX {chg:+.1f}% in {p['minutes']:g} min (fear rising), close below the {n}-bar low",
                    b.index[-1], **vals)
    return _sig(None, f"VIX {now_v:.2f} ({chg:+.1f}%), no confirmed move", b.index[-1], **vals)


# ============================================================================================ mean reversion
def rsi_mean_reversion(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 40)
    if miss:
        return miss
    r = ind.rsi(b["close"], int(p["length"]))
    last = b.iloc[-1]
    vals = dict(rsi=r.iloc[-1], adx=last["adx"])
    if last["adx"] > p["adx_max"]:
        return _sig(None, f"trending (ADX {last['adx']:.1f} > {p['adx_max']:g}): mean reversion off", b.index[-1],
                    **vals)
    if _cross_up(r, p["oversold"]) and last["close"] > last["open"]:
        return _sig(UP, f"RSI({int(p['length'])}) left the extreme below {p['oversold']:g} in a range", b.index[-1],
                    **vals)
    if _cross_dn(r, 100 - p["oversold"]) and last["close"] < last["open"]:
        return _sig(DOWN, f"RSI({int(p['length'])}) left the extreme above {100 - p['oversold']:g} in a range",
                    b.index[-1], **vals)
    return _sig(None, "no RSI extreme exit", b.index[-1], **vals)


def vwap_mean_reversion(ctx: Context, p: dict) -> Result:
    f = fut_session(ctx)
    if f is None:
        return Result(reason="no data: futures volume needed for VWAP")
    if len(f) < 12:
        return Result(reason="warming up: needs 12 bars of today's VWAP")
    dev = f["close"] - f["vwap"]
    sd = dev.expanding(min_periods=10).std(ddof=0)
    z = dev / sd.replace(0, np.nan)
    vals = dict(vwap=f["vwap"].iloc[-1], z=z.iloc[-1])
    k = p["z_entry"]
    if _cross_up(z, -k):
        return _sig(UP, f"futures back inside -{k:g} sd of VWAP", f.index[-1], **vals)
    if _cross_dn(z, k):
        return _sig(DOWN, f"futures back inside +{k:g} sd of VWAP", f.index[-1], **vals)
    return _sig(None, f"VWAP deviation {z.iloc[-1]:+.2f} sd", f.index[-1], **vals)


def zscore_reversion(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["length"])
    miss = _need(b, n + 5)
    if miss:
        return miss
    c = b["close"]
    z = (c - sma(c, n)) / c.rolling(n).std(ddof=0).replace(0, np.nan)
    vals = dict(z=z.iloc[-1])
    if _cross_up(z, -p["z_entry"]):
        return _sig(UP, f"z-score back above -{p['z_entry']:g}", b.index[-1], **vals)
    if _cross_dn(z, p["z_entry"]):
        return _sig(DOWN, f"z-score back below +{p['z_entry']:g}", b.index[-1], **vals)
    return _sig(None, f"z-score {z.iloc[-1]:+.2f}", b.index[-1], **vals)


def ma_reversion(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["length"])
    miss = _need(b, n + 5)
    if miss:
        return miss
    m = sma(b["close"], n)
    dist = (b["close"] - m) / b["atr"]
    last, prev = b.iloc[-1], b.iloc[-2]
    vals = dict(sma=m.iloc[-1], dist_atr=dist.iloc[-1])
    already_up = b["close"].iloc[-2] > b["high"].iloc[-3]
    already_dn = b["close"].iloc[-2] < b["low"].iloc[-3]
    if dist.iloc[-2] <= -p["stretch_atr"] and last["close"] > prev["high"] and not already_up:
        return _sig(UP, f"{-dist.iloc[-2]:.1f} ATR below SMA{n}, reversal bar", b.index[-1], **vals)
    if dist.iloc[-2] >= p["stretch_atr"] and last["close"] < prev["low"] and not already_dn:
        return _sig(DOWN, f"{dist.iloc[-2]:.1f} ATR above SMA{n}, reversal bar", b.index[-1], **vals)
    return _sig(None, f"{dist.iloc[-1]:+.1f} ATR from SMA{n}", b.index[-1], **vals)


# ============================================================================================ combinations
def supertrend_rsi(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 30)
    if miss:
        return miss
    st = b["st"]
    r = b["rsi"]
    up_flip, dn_flip = st.iloc[-1] > 0 > st.iloc[-2], st.iloc[-1] < 0 < st.iloc[-2]
    vals = dict(supertrend="up" if st.iloc[-1] > 0 else "down", rsi=r.iloc[-1])
    if st.iloc[-1] > 0 and r.iloc[-1] >= p["rsi_min"] and (up_flip or _cross_up(r, p["rsi_min"])):
        return _sig(UP, f"Supertrend up with RSI {r.iloc[-1]:.0f} >= {p['rsi_min']:g}", b.index[-1], **vals)
    if st.iloc[-1] < 0 and r.iloc[-1] <= 100 - p["rsi_min"] and (dn_flip or _cross_dn(r, 100 - p["rsi_min"])):
        return _sig(DOWN, f"Supertrend down with RSI {r.iloc[-1]:.0f} <= {100 - p['rsi_min']:g}", b.index[-1], **vals)
    return _sig(None, "Supertrend and RSI not newly agreeing", b.index[-1], **vals)


def macd_adx(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 50)
    if miss:
        return miss
    last = b.iloc[-1]
    vals = dict(adx=last["adx"], macd=last["macd"])
    if last["adx"] < p["adx_min"]:
        return _sig(None, f"ADX {last['adx']:.1f} < {p['adx_min']:g}", b.index[-1], **vals)
    if _cross_up(b["macd"], b["macd_sig"]) and last["pdi"] > last["mdi"]:
        return _sig(UP, f"MACD cross up confirmed by ADX {last['adx']:.1f} and +DI > -DI", b.index[-1], **vals)
    if _cross_dn(b["macd"], b["macd_sig"]) and last["mdi"] > last["pdi"]:
        return _sig(DOWN, f"MACD cross down confirmed by ADX {last['adx']:.1f} and -DI > +DI", b.index[-1], **vals)
    return _sig(None, "no MACD cross in the ADX direction", b.index[-1], **vals)


def vwap_ema(ctx: Context, p: dict) -> Result:
    f = fut_session(ctx)
    if f is None:
        return Result(reason="no data: futures volume needed for VWAP")
    if len(f) < 4:
        return Result(reason="warming up: VWAP")
    up = (f["close"] > f["vwap"]) & (f["ema9"] > f["ema21"])
    dn = (f["close"] < f["vwap"]) & (f["ema9"] < f["ema21"])
    vals = dict(vwap=f["vwap"].iloc[-1], ema9=f["ema9"].iloc[-1], ema21=f["ema21"].iloc[-1])
    if _first(up):
        return _sig(UP, "futures above VWAP and EMA9 > EMA21 (both now agree)", f.index[-1], **vals)
    if _first(dn):
        return _sig(DOWN, "futures below VWAP and EMA9 < EMA21 (both now agree)", f.index[-1], **vals)
    return _sig(None, "VWAP and EMAs not newly aligned", f.index[-1], **vals)


def fibonacci(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["lookback"])
    miss = _need(b, n + 20)
    if miss:
        return miss
    w = b.iloc[-n - 1:-1]
    atr_v = b["atr"].iloc[-1]
    if not finite(atr_v) or atr_v <= 0:
        return Result(reason="warming up: ATR")
    i_hi, i_lo = int(np.argmax(w["high"].to_numpy())), int(np.argmin(w["low"].to_numpy()))
    hi, lo = float(w["high"].iloc[i_hi]), float(w["low"].iloc[i_lo])
    swing = hi - lo
    last, prev = b.iloc[-1], b.iloc[-2]
    a, z = p["zone_from"], p["zone_to"]
    if swing < p["min_swing_atr"] * atr_v:
        return _sig(None, f"swing {swing / atr_v:.1f} ATR too small", b.index[-1])
    if i_lo < i_hi:                                         # up impulse, then a pull-back
        z_hi, z_lo = hi - a * swing, hi - z * swing
        after = w.iloc[i_hi:]
        touched = after["low"].min() <= z_hi and after["low"].min() >= z_lo - 0.1 * swing
        vals = dict(swing_high=hi, swing_low=lo, zone_top=z_hi, zone_bottom=z_lo)
        if touched and last["close"] > prev["high"] and last["close"] > z_hi:
            return _sig(UP, f"pull-back held the {a:.1%}-{z:.1%} retracement, resumed up", b.index[-1], **vals)
        return _sig(None, "up swing: waiting for a retracement into the zone and a resumption bar", b.index[-1],
                    **vals)
    z_lo, z_hi = lo + a * swing, lo + z * swing
    after = w.iloc[i_lo:]
    touched = after["high"].max() >= z_lo and after["high"].max() <= z_hi + 0.1 * swing
    vals = dict(swing_high=hi, swing_low=lo, zone_top=z_hi, zone_bottom=z_lo)
    if touched and last["close"] < prev["low"] and last["close"] < z_lo:
        return _sig(DOWN, f"bounce failed in the {a:.1%}-{z:.1%} retracement, resumed down", b.index[-1], **vals)
    return _sig(None, "down swing: waiting for a retracement into the zone and a resumption bar", b.index[-1], **vals)


def pivot_points(ctx: Context, p: dict) -> Result:
    d = prev_daily(ctx)
    b = ctx.today_bars(ctx.b5)
    if len(d) == 0:
        return Result(reason="no data: previous day candle")
    if len(b) < 2:
        return Result(reason="waiting for today's bars")
    pv = floor_pivots(d.iloc[-1])
    vals = {k: v for k, v in pv.items()}
    last, prev = b.iloc[-1], b.iloc[-2]
    if str(p["mode"]).lower() == "bounce":
        if last["low"] <= pv["S1"] < last["close"] and last["close"] > last["open"] and prev["close"] > pv["S1"]:
            return _sig(UP, f"bounced off S1 {pv['S1']:.1f}", b.index[-1], **vals)
        if last["high"] >= pv["R1"] > last["close"] and last["close"] < last["open"] and prev["close"] < pv["R1"]:
            return _sig(DOWN, f"rejected at R1 {pv['R1']:.1f}", b.index[-1], **vals)
        return _sig(None, "no S1 bounce / R1 rejection", b.index[-1], **vals)
    if _first_today(b["close"] > pv["R1"]):
        return _sig(UP, f"first close above R1 {pv['R1']:.1f}", b.index[-1], **vals)
    if _first_today(b["close"] < pv["S1"]):
        return _sig(DOWN, f"first close below S1 {pv['S1']:.1f}", b.index[-1], **vals)
    return _sig(None, f"between S1 {pv['S1']:.1f} and R1 {pv['R1']:.1f}", b.index[-1], **vals)


def cpr_breakout(ctx: Context, p: dict) -> Result:
    d = prev_daily(ctx)
    b = ctx.today_bars(ctx.b5)
    if len(d) == 0:
        return Result(reason="no data: previous day candle")
    if len(b) == 0:
        return Result(reason="waiting for today's first bar")
    y = d.iloc[-1]
    pivot = (y["high"] + y["low"] + y["close"]) / 3
    bc = (y["high"] + y["low"]) / 2
    tc = 2 * pivot - bc
    top, bottom = max(tc, bc), min(tc, bc)
    width = (top - bottom) / pivot * 100
    vals = dict(cpr_top=top, cpr_bottom=bottom, width_pct=width)
    if width > p["max_width_pct"]:
        return _sig(None, f"wide CPR ({width:.2f}% > {p['max_width_pct']:g}%): range day expected", b.index[-1], **vals)
    if ctx.now.time() > hhmm(p["cutoff"]):
        return _sig(None, f"after the cut-off {p['cutoff']}", b.index[-1], **vals)
    if _first_today(b["close"] > top):
        return _sig(UP, f"narrow CPR, first close above its top {top:.1f}", b.index[-1], **vals)
    if _first_today(b["close"] < bottom):
        return _sig(DOWN, f"narrow CPR, first close below its bottom {bottom:.1f}", b.index[-1], **vals)
    return _sig(None, "no first close outside the CPR", b.index[-1], **vals)


# ============================================================================================ two indices
def relative_strength(ctx: Context, p: dict) -> Result:
    pair = _pair_frame(ctx)
    if pair is None:
        return Result(reason="no data: needs the other index (NIFTY and SENSEX both on)")
    a, b_, other = pair
    n = int(p["lookback"])
    if len(a) < n + 2:
        return Result(reason=f"warming up: needs {n + 2} common bars")
    ra, rb = a.iloc[-1] / a.iloc[-1 - n] * 100 - 100, b_.iloc[-1] / b_.iloc[-1 - n] * 100 - 100
    edge = ra - rb
    bars = ctx.b5
    brk_up, brk_dn = range_break(bars, n)
    vals = dict(own_pct=ra, other_pct=rb, edge_pct=edge, other=other)
    if edge >= p["min_edge_pct"] and ra > 0 and brk_up:
        return _sig(UP, f"stronger than {other} by {edge:+.2f}% over {n} bars, new {n}-bar high", bars.index[-1],
                    **vals)
    if edge <= -p["min_edge_pct"] and ra < 0 and brk_dn:
        return _sig(DOWN, f"weaker than {other} by {edge:+.2f}% over {n} bars, new {n}-bar low", bars.index[-1],
                    **vals)
    return _sig(None, f"relative strength {edge:+.2f}% vs {other}", bars.index[-1], **vals)


def _spread_signal(ctx: Context, p: dict, z: pd.Series, label: str, extra: dict) -> Result:
    bar = ctx.b5.index[-1]
    vals = dict(z=z.iloc[-1], **extra)
    k = p["z_entry"]
    # this index rich vs the other (z high) and turning back -> this index should lag: PE; cheap -> CE
    if _cross_dn(z, k):
        return _sig(DOWN, f"{label}: spread z {z.iloc[-2]:+.2f} -> {z.iloc[-1]:+.2f}, this index rich, reverting",
                    bar, **vals)
    if _cross_up(z, -k):
        return _sig(UP, f"{label}: spread z {z.iloc[-2]:+.2f} -> {z.iloc[-1]:+.2f}, this index cheap, reverting",
                    bar, **vals)
    return _sig(None, f"{label}: spread z {z.iloc[-1]:+.2f}", bar, **vals)


def pair_trading(ctx: Context, p: dict) -> Result:
    pair = _pair_frame(ctx)
    if pair is None:
        return Result(reason="no data: needs the other index (NIFTY and SENSEX both on)")
    a, b_, other = pair
    n = int(p["lookback"])
    if len(a) < n + 2:
        return Result(reason=f"warming up: needs {n + 2} common bars")
    ratio = np.log(a) - np.log(b_)
    z = (ratio - ratio.rolling(n).mean()) / ratio.rolling(n).std(ddof=0).replace(0, np.nan)
    return _spread_signal(ctx, p, z, f"pair vs {other}", {"other": other})


def stat_arb(ctx: Context, p: dict) -> Result:
    pair = _pair_frame(ctx)
    if pair is None:
        return Result(reason="no data: needs the other index (NIFTY and SENSEX both on)")
    a, b_, other = pair
    n = int(p["lookback"])
    if len(a) < n + 2:
        return Result(reason=f"warming up: needs {n + 2} common bars")
    a, b_ = a.iloc[-(2 * n + 2):], b_.iloc[-(2 * n + 2):]
    la, lb = np.log(a.to_numpy(float)), np.log(b_.to_numpy(float))
    resid = np.full(len(la), np.nan)
    beta_now = np.nan
    for i in range(n - 1, len(la)):                       # rolling OLS hedge ratio, only past bars
        x, y = lb[i - n + 1:i + 1], la[i - n + 1:i + 1]
        vx = x.var()
        if vx <= 0:
            continue
        beta = ((x - x.mean()) * (y - y.mean())).mean() / vx
        alpha = y.mean() - beta * x.mean()
        resid[i] = la[i] - alpha - beta * lb[i]
        beta_now = beta
    r = pd.Series(resid, index=a.index)
    z = (r - r.rolling(n).mean()) / r.rolling(n).std(ddof=0).replace(0, np.nan)
    rr = r.dropna().iloc[-n:]
    half_life = np.nan
    if len(rr) > 10:
        dr, lag = np.diff(rr.to_numpy()), rr.to_numpy()[:-1] - rr.to_numpy()[:-1].mean()
        slope = (lag * dr).sum() / (lag * lag).sum() if (lag * lag).sum() > 0 else 0
        half_life = -math.log(2) / slope if slope < 0 else np.inf
    extra = {"other": other, "beta": beta_now, "half_life_bars": half_life}
    if not (finite(half_life) and half_life <= p["max_half_life"]):
        return _sig(None, f"spread not mean-reverting fast enough (half-life "
                          f"{half_life if finite(half_life) else 'inf'} bars > {p['max_half_life']:g})",
                    ctx.b5.index[-1], z=z.iloc[-1] if len(z) else np.nan, **extra)
    return _spread_signal(ctx, p, z, f"stat-arb vs {other}", extra)


# ============================================================================================ options / composite
def gamma_iv(ctx: Context, p: dict) -> Result:
    chain = ctx.chain
    if chain is None or chain.empty or not {"ce_gamma", "pe_gamma", "ce_oi", "pe_oi"} <= set(chain.columns):
        return Result(reason="no data: option chain with gamma and OI")
    c = chain.apply(pd.to_numeric, errors="coerce")
    gex = float((c["ce_gamma"] * c["ce_oi"] - c["pe_gamma"] * c["pe_oi"]).sum())
    if not finite(gex) or (c["ce_oi"].fillna(0).sum() <= 0):
        return Result(reason="no data: chain has no OI / gamma")
    mem = ctx.memory.iv
    if not mem:
        return Result(reason="no data: ATM IV samples")
    old = _sample_at(mem, ctx.now - pd.Timedelta(minutes=p["minutes"]))
    if old is None or not finite(old[1]) or old[1] <= 0:
        return Result(reason=f"warming up: needs {p['minutes']:g} min of IV samples")
    iv_chg = mem[-1][1] / old[1] * 100 - 100
    b = ctx.b5
    miss = _need(b, 25)
    if miss:
        return miss
    brk_up, brk_dn = range_break(b, int(p["lookback"]))
    n = int(p["lookback"])
    vals = dict(net_gamma_oi=gex, atm_iv=mem[-1][1], iv_change_pct=iv_chg)
    if gex >= 0:
        return _sig(None, "positive net gamma (writers dampen moves)", b.index[-1], **vals)
    if iv_chg < p["min_iv_rise_pct"]:
        return _sig(None, f"negative gamma but IV {iv_chg:+.1f}% (needs +{p['min_iv_rise_pct']:g}%)", b.index[-1],
                    **vals)
    if brk_up:
        return _sig(UP, f"negative gamma, IV {iv_chg:+.1f}%, close above the {n}-bar high", b.index[-1], **vals)
    if brk_dn:
        return _sig(DOWN, f"negative gamma, IV {iv_chg:+.1f}%, close below the {n}-bar low", b.index[-1], **vals)
    return _sig(None, "negative gamma and rising IV, waiting for a range break", b.index[-1], **vals)


def ensemble(ctx: Context, p: dict) -> Result:
    """Rule-based vote of seven independent indicator states (not a learned model)."""
    b = ctx.b5
    miss = _need(b, 60)
    if miss:
        return miss

    def votes(i: int) -> int:
        r = b.iloc[i]
        v = [np.sign(r["ema20"] - r["ema50"]), np.sign(r["st"]), np.sign(r["macd"] - r["macd_sig"]),
             np.sign(r["rsi"] - 50), np.sign(r["pdi"] - r["mdi"]) if r["adx"] >= 20 else 0,
             np.sign(r["close"] - r["bb_mid"]), np.sign(r["close"] - r["ema9"])]
        return int(np.nansum(v))

    now_v, prev_v = votes(-1), votes(-2)
    k = int(p["min_votes"])
    vals = dict(votes=now_v, previous=prev_v, of=7)
    if now_v >= k > prev_v:
        return _sig(UP, f"{now_v} of 7 indicators bullish (net), was {prev_v}", b.index[-1], **vals)
    if now_v <= -k < prev_v:
        return _sig(DOWN, f"{-now_v} of 7 indicators bearish (net), was {-prev_v}", b.index[-1], **vals)
    return _sig(None, f"net vote {now_v:+d} (needs +/-{k})", b.index[-1], **vals)


def ml_signal(ctx: Context, p: dict) -> Result:
    from src.movement import ml_signal as ml
    return ml.evaluate(ctx, p)


# ============================================================================================ registry
def _spec(key, title, fn, params, category, ce, pe, data="spot index bars", per_bar=True, needs_chain=False):
    return Spec(key, title, fn, params, needs_chain=needs_chain, category=category, ce=ce, pe=pe, data=data,
                per_bar=per_bar)


SPECS = [
    # trend following
    _spec("ma_crossover", "Moving Average Crossover", ma_crossover, {"fast": 20, "slow": 50}, "Trend following",
          "SMA fast crosses above SMA slow", "SMA fast crosses below SMA slow"),
    _spec("ema_9_21", "EMA 9/21", ema_9_21, {"fast": 9, "slow": 21}, "Trend following",
          "EMA9 crosses above EMA21, close above EMA9", "EMA9 crosses below EMA21, close below EMA9"),
    _spec("ema_20_50_200", "EMA 20/50/200", ema_20_50_200, {}, "Trend following",
          "EMA20 > EMA50 > EMA200 and close > EMA20 becomes true", "EMA20 < EMA50 < EMA200 and close < EMA20 becomes true",
          data="spot index bars (205+ bars of the timeframe)"),
    _spec("donchian", "Donchian Channel Breakout", donchian, {"length": 20}, "Trend following",
          "first close above the previous N-bar high", "first close below the previous N-bar low"),
    _spec("ichimoku", "Ichimoku Cloud", ichimoku_cloud, {"tenkan": 9, "kijun": 26, "senkou": 52}, "Trend following",
          "Tenkan crosses above Kijun, close above the cloud", "Tenkan crosses below Kijun, close below the cloud"),
    _spec("psar", "Parabolic SAR", psar, {"step": 0.02, "max_step": 0.2}, "Trend following",
          "SAR flips below price", "SAR flips above price"),
    # momentum
    _spec("rsi_reversal", "RSI Reversal", rsi_reversal, {"length": 14, "oversold": 30.0, "overbought": 70.0},
          "Momentum", "RSI crosses back above oversold + close above the previous high",
          "RSI crosses back below overbought + close below the previous low"),
    _spec("macd_crossover", "MACD Crossover", macd_crossover, {"zero_filter": 0}, "Momentum",
          "MACD crosses above signal (below zero if zero_filter = 1)",
          "MACD crosses below signal (above zero if zero_filter = 1)"),
    _spec("stochastic", "Stochastic Oscillator", stochastic_osc,
          {"k": 14, "smooth": 3, "d": 3, "oversold": 20.0, "overbought": 80.0}, "Momentum",
          "%K crosses above %D from the oversold zone", "%K crosses below %D from the overbought zone"),
    _spec("roc", "Rate of Change", roc_momentum, {"length": 10, "threshold_pct": 0.3}, "Momentum",
          "N-bar ROC rises through +threshold", "N-bar ROC falls through -threshold"),
    _spec("cci", "CCI Momentum", cci_momentum, {"length": 20, "level": 100.0}, "Momentum",
          "CCI rises through +level", "CCI falls through -level"),
    _spec("williams_r", "Williams %R", williams, {"length": 14, "upper": -20.0, "lower": -80.0}, "Momentum",
          "%R rises through the upper level, close above EMA20", "%R falls through the lower level, close below EMA20"),
    # price action
    _spec("sr_breakout", "Support / Resistance Breakout", sr_breakout,
          {"lookback": 60, "pivot": 3, "touch_atr": 0.25, "min_touches": 2, "buffer_atr": 0.1}, "Price action",
          "close breaks a resistance tested min_touches times", "close breaks a support tested min_touches times"),
    _spec("pdh_pdl_breakout", "Previous Day High/Low Breakout", pdh_pdl_breakout,
          {"buffer_atr": 0.1, "cutoff": "14:00"}, "Price action",
          "first close above yesterday's high (+buffer)", "first close below yesterday's low (-buffer)",
          data="spot index bars + previous day candle"),
    _spec("inside_bar", "Inside Bar Breakout", inside_bar, {"max_mother_atr": 2.0}, "Price action",
          "close above the mother bar after an inside bar", "close below the mother bar after an inside bar"),
    _spec("engulfing", "Engulfing Candlestick", engulfing, {"trend_bars": 5, "min_body_atr": 0.6}, "Price action",
          "bullish engulfing after an N-bar decline", "bearish engulfing after an N-bar rise"),
    _spec("pin_bar", "Pin Bar Reversal", pin_bar, {"lookback": 10, "wick_ratio": 0.6, "min_range_atr": 0.8},
          "Price action", "long lower wick rejecting a new N-bar low", "long upper wick rejecting a new N-bar high"),
    _spec("vwap_breakout", "VWAP Breakout", vwap_breakout, {"min_rvol": 1.3}, "Price action",
          "futures close crosses above VWAP with rvol >= min_rvol",
          "futures close crosses below VWAP with rvol >= min_rvol", data="index futures bars with volume"),
    # volatility
    _spec("bollinger_breakout", "Bollinger Bands Breakout", bollinger_breakout, {"length": 20, "k": 2.0},
          "Volatility", "first close above the upper band, bands widening",
          "first close below the lower band, bands widening"),
    _spec("bollinger_squeeze", "Bollinger Squeeze", bollinger_squeeze, {"kc_mult": 1.5, "min_squeeze_bars": 6},
          "Volatility", "bands leave the Keltner channel upward after a squeeze",
          "bands leave the Keltner channel downward after a squeeze"),
    _spec("keltner", "Keltner Channel", keltner_channel, {"length": 20, "mult": 1.5}, "Volatility",
          "first close above the upper channel", "first close below the lower channel"),
    _spec("volatility_expansion", "Volatility Expansion", volatility_expansion,
          {"narrow_bars": 7, "expansion_atr": 1.3}, "Volatility",
          "wide bar closing above a narrowest-of-N bar", "wide bar closing below a narrowest-of-N bar"),
    _spec("india_vix", "India VIX Strategy", india_vix,
          {"minutes": 30.0, "min_change_pct": 3.0, "lookback": 12, "max_vix_for_ce": 25.0}, "Volatility",
          "VIX down >= min_change_pct in `minutes` and a new N-bar high",
          "VIX up >= min_change_pct in `minutes` and a new N-bar low", data="spot index bars + India VIX",
          per_bar=False),
    # mean reversion
    _spec("rsi_mean_reversion", "RSI Mean Reversion", rsi_mean_reversion,
          {"length": 2, "oversold": 10.0, "adx_max": 22.0}, "Mean reversion",
          "RSI(2) leaves the oversold extreme in a range (ADX low)", "RSI(2) leaves the overbought extreme in a range"),
    _spec("vwap_mean_reversion", "VWAP Mean Reversion", vwap_mean_reversion, {"z_entry": 2.0}, "Mean reversion",
          "futures back above -z sd from VWAP", "futures back below +z sd from VWAP",
          data="index futures bars with volume"),
    _spec("zscore_reversion", "Z-Score Reversion", zscore_reversion, {"length": 30, "z_entry": 2.0},
          "Mean reversion", "z-score crosses back above -z_entry", "z-score crosses back below +z_entry"),
    _spec("ma_reversion", "Moving Average Reversion", ma_reversion, {"length": 50, "stretch_atr": 2.5},
          "Mean reversion", "stretched below SMA by stretch_atr ATR, then a close above the previous high",
          "stretched above SMA by stretch_atr ATR, then a close below the previous low"),
    # advanced / combinations
    _spec("supertrend_rsi", "Supertrend + RSI", supertrend_rsi, {"rsi_min": 55.0}, "Combination",
          "Supertrend up and RSI >= rsi_min (newly)", "Supertrend down and RSI <= 100 - rsi_min (newly)"),
    _spec("macd_adx", "MACD + ADX Confirmation", macd_adx, {"adx_min": 22.0}, "Combination",
          "MACD cross up with ADX >= adx_min and +DI > -DI", "MACD cross down with ADX >= adx_min and -DI > +DI"),
    _spec("vwap_ema", "VWAP + EMA Confluence", vwap_ema, {}, "Combination",
          "futures above VWAP and EMA9 > EMA21 (newly both)", "futures below VWAP and EMA9 < EMA21 (newly both)",
          data="index futures bars with volume"),
    _spec("fibonacci", "Fibonacci Retracement", fibonacci,
          {"lookback": 40, "min_swing_atr": 4.0, "zone_from": 0.5, "zone_to": 0.618}, "Combination",
          "up swing, pull-back into the 50-61.8% zone, close above the previous high",
          "down swing, bounce into the 50-61.8% zone, close below the previous low"),
    _spec("pivot_points", "Pivot Point Trading", pivot_points, {"mode": "breakout"}, "Combination",
          "breakout: first close above R1 / bounce: rejection of S1",
          "breakout: first close below S1 / bounce: rejection of R1", data="spot index bars + previous day candle"),
    _spec("cpr_breakout", "CPR Breakout", cpr_breakout, {"max_width_pct": 0.25, "cutoff": "13:30"}, "Combination",
          "narrow CPR day, first close above the CPR top", "narrow CPR day, first close below the CPR bottom",
          data="spot index bars + previous day candle"),
    _spec("relative_strength", "Relative Strength Rotation", relative_strength,
          {"lookback": 12, "min_edge_pct": 0.15}, "Two indices",
          "this index outperforms the other by min_edge_pct and makes a new N-bar high",
          "this index underperforms by min_edge_pct and makes a new N-bar low", data="NIFTY and SENSEX bars",
          per_bar=False),
    _spec("pair_trading", "Pair Trading", pair_trading, {"lookback": 60, "z_entry": 2.0}, "Two indices",
          "log-price spread vs the other index crosses back above -z (this index cheap)",
          "spread crosses back below +z (this index rich)", data="NIFTY and SENSEX bars", per_bar=False),
    _spec("stat_arb", "Statistical Arbitrage", stat_arb, {"lookback": 120, "z_entry": 2.0, "max_half_life": 30.0},
          "Two indices", "rolling-OLS residual z crosses back above -z, half-life <= max_half_life",
          "residual z crosses back below +z, half-life <= max_half_life", data="NIFTY and SENSEX bars", per_bar=False),
    # options / AI
    _spec("gamma_iv", "Gamma and IV Options Strategy", gamma_iv,
          {"minutes": 30.0, "min_iv_rise_pct": 3.0, "lookback": 12}, "Options",
          "net gamma (CE gamma x OI - PE gamma x OI) negative, ATM IV rising, new N-bar high",
          "same, new N-bar low", data="live option chain with gamma and OI", per_bar=False, needs_chain=True),
    _spec("ensemble", "Multi-Indicator Ensemble", ensemble, {"min_votes": 5}, "AI / ensemble",
          "net bullish vote of 7 indicators reaches min_votes", "net bearish vote reaches min_votes"),
    _spec("ml_signal", "Machine Learning Signal Classification", ml_signal,
          {"threshold": 0.0, "model": "auto"}, "AI / ensemble",
          "model P(up move) crosses its threshold (0 = the model's own, set on the training period)",
          "model P(down move) crosses its threshold",
          data="spot index bars + a trained model (python -m src.movement.ml_signal --train)"),
]


def _register():
    """Add SPECS to the engine's REGISTRY (works whichever of the two modules is imported first)."""
    from src.movement import strategies
    for spec in SPECS:
        if strategies.REGISTRY.get(spec.key, spec) is not spec:
            raise RuntimeError(f"duplicate strategy key {spec.key}")
        strategies.REGISTRY[spec.key] = spec


_register()
