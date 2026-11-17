"""
Fifteen rule-based intraday strategies for NIFTY / SENSEX, each a pure function of COMPLETED bars:

    breakout_momentum   close above the 20-bar high / below the low, big body, expanding range or volume, RSI
    ema_pullback        EMA20 > EMA50 trend, pull-back to EMA20, close back above the previous bar's high
    vwap_reversal       futures stretched >= k ATR from VWAP, then a reversal bar back toward VWAP, RSI turning
    orb                 first 5-min close beyond the 15-min opening range (+ buffer), before the cut-off time
    rsi_macd_divergence lower low with a higher RSI low (or the mirror), MACD histogram turning in the new direction
    adx_trend           +DI / -DI cross with ADX >= threshold and rising, price on the right side of EMA50
    bollinger_reversion in a range (ADX low): close back inside the band after a close outside it
    volume_profile      futures volume profile (previous + current session): rejection at VAL / VAH, or
                        acceptance (2 closes) beyond the value area; previous-day high / low as extra levels
    oi_buildup          futures price and OI both rising (long build-up) / price falling, OI rising (short build-up)
    pcr                 option-chain PCR rising / falling by a set amount in 30 min, with price agreeing (EMA20)
    supertrend          Supertrend(10, 3) flips
    first_hour          after 10:15: first close beyond the first-hour high / low in the first hour's direction
    atr_breakout        close beyond today's open +/- k x daily ATR
    mtf_confirmation    15-min trend + 5-min trigger (EMA9 cross or MACD cross) + 1-min momentum, all agreeing
    premium_momentum    ATM CE premium up and PE premium down (or the mirror) by set % in 10 min, same strikes

src/movement/library.py adds the rest of the library (trend, momentum, price action, volatility, mean
reversion, combinations, two-index pair strategies, gamma / IV, ensemble and the ML classifier):
python -m src.movement.strategies --list prints every strategy with its CE / PE rule and the data it needs.

A strategy returns Result(direction "UP" / "DOWN" / None, reason, values). Signals are EVENTS (something changed on
the last completed bar), not states, so a trend does not fire on every bar. Missing data gives no signal with the
reason "no data: ...", never a guess. All parameters are untested starting values (STRATEGY_<KEY> in scanner.ini).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field, replace
from datetime import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.options.derivatives import analyse_chain
from src.scanner import indicators as ind
from src.utils.numbers import finite, fnum, hhmm

UP, DOWN = "UP", "DOWN"
ONE_MIN = pd.Timedelta(minutes=1)


@dataclass
class Result:
    direction: Optional[str] = None
    reason: str = ""
    values: Dict[str, object] = field(default_factory=dict)
    bar: Optional[str] = None          # start time of the bar the result is based on (dedup key)


@dataclass
class Memory:
    """Per-index samples the chain-based strategies need over time (kept by the engine)."""
    pcr: collections.deque = field(default_factory=lambda: collections.deque(maxlen=600))
    premiums: collections.deque = field(default_factory=lambda: collections.deque(maxlen=600))
    vix: collections.deque = field(default_factory=lambda: collections.deque(maxlen=600))
    iv: collections.deque = field(default_factory=lambda: collections.deque(maxlen=600))


@dataclass
class Context:
    index: str
    now: pd.Timestamp
    price: float
    m1: pd.DataFrame                      # spot 1-min, completed (previous sessions + today)
    b5: pd.DataFrame                      # spot signal-timeframe bars, completed, with indicators
    b15: pd.DataFrame                     # spot 15-min bars, completed, with indicators
    f5: pd.DataFrame                      # futures signal-timeframe bars, completed (volume / OI / VWAP)
    f1: pd.DataFrame                      # futures 1-min, completed
    daily: pd.DataFrame
    chain: Optional[pd.DataFrame]
    step: float
    memory: Memory
    tf: int = 5
    peers: Optional[Dict[str, "Context"]] = None    # every index's context at the same timeframe (pair strategies)

    @property
    def today(self):
        return self.now.date()

    def today_bars(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame.index.date == self.today] if len(frame) else frame


# ---------------------------------------------------------------------------------------------- indicators
def supertrend(bars: pd.DataFrame, length: int = 10, mult: float = 3.0) -> pd.Series:
    """+1 / -1 trend of the classic Supertrend (ATR bands that only tighten, flip on a close through them)."""
    if len(bars) < length + 2:
        return pd.Series(np.nan, index=bars.index)
    atr_v = ind.atr(bars, length).to_numpy()
    hl2 = ((bars["high"] + bars["low"]) / 2).to_numpy()
    close = bars["close"].to_numpy()
    upper, lower = hl2 + mult * atr_v, hl2 - mult * atr_v
    trend = np.full(len(bars), np.nan)
    fu, fl = upper.copy(), lower.copy()
    for i in range(1, len(bars)):
        if not np.isfinite(atr_v[i]):
            continue
        if np.isfinite(fu[i - 1]) and np.isfinite(atr_v[i - 1]):
            fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        prev = trend[i - 1] if np.isfinite(trend[i - 1]) else 1.0
        if prev > 0:
            trend[i] = -1.0 if close[i] < fl[i] else 1.0
        else:
            trend[i] = 1.0 if close[i] > fu[i] else -1.0
    return pd.Series(trend, index=bars.index)


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or len(bars) == 0:
        return bars
    b = bars.copy()
    c = b["close"]
    for n in (9, 20, 50):
        b[f"ema{n}"] = ind.ema(c, n)
    b["rsi"] = ind.rsi(c, 14)
    b["atr"] = ind.atr(b, 14)
    b["macd"], b["macd_sig"], b["macd_hist"] = ind.macd(c)
    b["adx"], b["pdi"], b["mdi"] = ind.adx(b, 14)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    b["bb_mid"], b["bb_up"], b["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    b["st"] = supertrend(b)
    return b


def completed(frame: pd.DataFrame, now: pd.Timestamp, minutes: int) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return frame
    return frame[frame.index + pd.Timedelta(minutes=minutes) <= now]


def _f(v, d=2):
    v = fnum(v)
    return round(float(v), d) if finite(v) else None


def _need(b: pd.DataFrame, n: int, what: str = "bars") -> Optional[Result]:
    if b is None or len(b) < n:
        return Result(reason=f"warming up: needs {n} {what}")
    return None


def _cross_up(a: pd.Series, b, i=-1) -> bool:
    bb = b if np.isscalar(b) else b.iloc[i - 1]
    bn = b if np.isscalar(b) else b.iloc[i]
    return bool(a.iloc[i - 1] <= bb and a.iloc[i] > bn)


def _cross_dn(a: pd.Series, b, i=-1) -> bool:
    bb = b if np.isscalar(b) else b.iloc[i - 1]
    bn = b if np.isscalar(b) else b.iloc[i]
    return bool(a.iloc[i - 1] >= bb and a.iloc[i] < bn)


def _rvol(f: pd.DataFrame, n: int = 20) -> float:
    if f is None or len(f) < n + 1 or not ind.has_volume(f["volume"].tail(n + 1)):
        return np.nan
    base = f["volume"].iloc[-n - 1:-1].mean()
    return float(f["volume"].iloc[-1] / base) if base > 0 else np.nan


def _sig(direction, reason, bar, **values) -> Result:
    return Result(direction, reason, {k: _f(v) if isinstance(v, (float, np.floating)) else v for k, v in values.items()},
                  bar.isoformat() if bar is not None else None)


# ---------------------------------------------------------------------------------------------- strategies
def breakout_momentum(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    n = int(p["lookback"])
    miss = _need(b, n + 20)
    if miss:
        return miss
    last, prev = b.iloc[-1], b.iloc[-n - 1:-1]
    hi, lo = prev["high"].max(), prev["low"].min()
    rng = last["high"] - last["low"]
    body = abs(last["close"] - last["open"]) / rng if rng > 0 else 0
    rvol = _rvol(ctx.f5)
    expansion = rng / last["atr"] if last["atr"] > 0 else np.nan
    active = (rvol >= p["min_rvol"]) if finite(rvol) else (expansion >= p["min_range_atr"])
    vals = dict(high_n=hi, low_n=lo, body=body, rvol=rvol, range_atr=expansion, rsi=last["rsi"])
    if not (body >= p["min_body"] and active):
        return _sig(None, f"no impulse bar (body {body:.2f}, {'rvol' if finite(rvol) else 'range/ATR'} "
                          f"{(rvol if finite(rvol) else expansion):.2f})", b.index[-1], **vals)
    if last["close"] > hi and last["close"] > last["open"] and last["rsi"] >= p["rsi_up"]:
        return _sig(UP, f"close {last['close']:.1f} broke the {n}-bar high {hi:.1f}", b.index[-1], **vals)
    if last["close"] < lo and last["close"] < last["open"] and last["rsi"] <= 100 - p["rsi_up"]:
        return _sig(DOWN, f"close {last['close']:.1f} broke the {n}-bar low {lo:.1f}", b.index[-1], **vals)
    return _sig(None, f"inside the {n}-bar range {lo:.1f}-{hi:.1f}", b.index[-1], **vals)


def ema_pullback(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 60)
    if miss:
        return miss
    last, prev = b.iloc[-1], b.iloc[-2]
    tol = p["touch_atr"] * last["atr"]
    vals = dict(ema20=last["ema20"], ema50=last["ema50"], adx=last["adx"])
    if last["ema20"] > last["ema50"] and last["close"] > last["ema50"]:
        touched = prev["low"] <= prev["ema20"] + tol and prev["close"] >= prev["ema50"]
        if touched and last["close"] > prev["high"] and last["close"] > last["ema20"]:
            return _sig(UP, "up-trend pull-back to EMA20, resumed above the previous high", b.index[-1], **vals)
        return _sig(None, "up-trend, waiting for a pull-back to EMA20 and a resumption bar", b.index[-1], **vals)
    if last["ema20"] < last["ema50"] and last["close"] < last["ema50"]:
        touched = prev["high"] >= prev["ema20"] - tol and prev["close"] <= prev["ema50"]
        if touched and last["close"] < prev["low"] and last["close"] < last["ema20"]:
            return _sig(DOWN, "down-trend pull-back to EMA20, resumed below the previous low", b.index[-1], **vals)
        return _sig(None, "down-trend, waiting for a pull-back to EMA20 and a resumption bar", b.index[-1], **vals)
    return _sig(None, "no EMA20/EMA50 trend", b.index[-1], **vals)


def vwap_reversal(ctx: Context, p: dict) -> Result:
    f = ctx.today_bars(ctx.f5)
    if f is None or len(f) < 6 or not ind.has_volume(f["volume"]):
        return Result(reason="no data: futures volume needed for VWAP")
    f = f.copy()
    f["vwap"] = ind.session_vwap(f)
    f["atr"] = ind.atr(ctx.f5, 14).reindex(f.index)
    f["rsi"] = ind.rsi(ctx.f5["close"], 14).reindex(f.index)
    last, prev = f.iloc[-1], f.iloc[-2]
    if not (finite(last["atr"]) and last["atr"] > 0 and finite(last["vwap"])):
        return Result(reason="warming up: ATR / VWAP")
    stretch_prev = (prev["vwap"] - prev["low"]) / last["atr"], (prev["high"] - prev["vwap"]) / last["atr"]
    vals = dict(vwap=last["vwap"], dist_atr=(last["close"] - last["vwap"]) / last["atr"], rsi=last["rsi"])
    k = p["stretch_atr"]
    if stretch_prev[0] >= k and last["close"] > prev["high"] and last["close"] < last["vwap"] \
            and last["rsi"] > prev["rsi"] and prev["rsi"] <= p["rsi_low"]:
        return _sig(UP, f"stretched {stretch_prev[0]:.1f} ATR below VWAP, reversal bar", f.index[-1], **vals)
    if stretch_prev[1] >= k and last["close"] < prev["low"] and last["close"] > last["vwap"] \
            and last["rsi"] < prev["rsi"] and prev["rsi"] >= 100 - p["rsi_low"]:
        return _sig(DOWN, f"stretched {stretch_prev[1]:.1f} ATR above VWAP, reversal bar", f.index[-1], **vals)
    return _sig(None, f"no stretch + reversal (stretch {max(stretch_prev):.1f} ATR, needs {k})", f.index[-1], **vals)


def orb(ctx: Context, p: dict) -> Result:
    day = ctx.today_bars(ctx.m1)
    minutes = int(p["range_minutes"])
    hi, lo = ind.opening_range(day, minutes, ctx.now)
    if not finite(hi):
        return Result(reason=f"opening range ({minutes} min) not complete")
    b = ctx.today_bars(ctx.b5)
    end = day.index[0].normalize() + pd.Timedelta(hours=9, minutes=15 + minutes)
    b = b[b.index >= end]
    if len(b) == 0:
        return Result(reason="waiting for the first bar after the range")
    if ctx.now.time() > hhmm(p["cutoff"]):
        return _sig(None, f"after the ORB cut-off {p['cutoff']}", b.index[-1], or_high=hi, or_low=lo)
    atr_v = ctx.b5["atr"].iloc[-1]
    buf = p["buffer_atr"] * atr_v if finite(atr_v) else 0
    closes = b["close"]
    above, below = closes > hi + buf, closes < lo - buf
    vals = dict(or_high=hi, or_low=lo, width=hi - lo)
    if above.iloc[-1] and not above.iloc[:-1].any():
        return _sig(UP, f"first close above the opening-range high {hi:.1f}", b.index[-1], **vals)
    if below.iloc[-1] and not below.iloc[:-1].any():
        return _sig(DOWN, f"first close below the opening-range low {lo:.1f}", b.index[-1], **vals)
    return _sig(None, "no first break of the opening range on this bar", b.index[-1], **vals)


def _pivot_lows(s: pd.Series, k: int) -> List[int]:
    v = s.to_numpy()
    return [i for i in range(k, len(v) - k) if v[i] == v[i - k:i + k + 1].min()]


def rsi_macd_divergence(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 60)
    if miss:
        return miss
    w = b.iloc[-int(p["lookback"]):]
    k = int(p["pivot"])
    hist = b["macd_hist"]
    turn_up = hist.iloc[-1] > hist.iloc[-2] and hist.iloc[-2] <= hist.iloc[-3]
    turn_dn = hist.iloc[-1] < hist.iloc[-2] and hist.iloc[-2] >= hist.iloc[-3]
    lows, highs = _pivot_lows(w["low"], k), _pivot_lows(-w["high"], k)
    recent = len(w) - 1 - int(p["max_age"])
    if len(lows) >= 2 and lows[-1] >= recent:
        a, c = lows[-2], lows[-1]
        if w["low"].iloc[c] < w["low"].iloc[a] and w["rsi"].iloc[c] > w["rsi"].iloc[a] + p["rsi_gap"] and turn_up:
            return _sig(UP, f"bullish divergence: lower low, RSI {w['rsi'].iloc[a]:.0f}->{w['rsi'].iloc[c]:.0f}, "
                            "MACD histogram turning up", b.index[-1])
    if len(highs) >= 2 and highs[-1] >= recent:
        a, c = highs[-2], highs[-1]
        if w["high"].iloc[c] > w["high"].iloc[a] and w["rsi"].iloc[c] < w["rsi"].iloc[a] - p["rsi_gap"] and turn_dn:
            return _sig(DOWN, f"bearish divergence: higher high, RSI {w['rsi'].iloc[a]:.0f}->{w['rsi'].iloc[c]:.0f}, "
                              "MACD histogram turning down", b.index[-1])
    return _sig(None, "no RSI divergence with a MACD turn", b.index[-1], rsi=b["rsi"].iloc[-1], hist=hist.iloc[-1])


def adx_trend(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 60)
    if miss:
        return miss
    last, prev = b.iloc[-1], b.iloc[-2]
    vals = dict(adx=last["adx"], pdi=last["pdi"], mdi=last["mdi"])
    strong = last["adx"] >= p["adx_min"] and last["adx"] > prev["adx"]
    if not strong:
        return _sig(None, f"ADX {last['adx']:.1f} below {p['adx_min']} or falling", b.index[-1], **vals)
    di_up = _cross_up(b["pdi"], b["mdi"]) or (last["pdi"] > last["mdi"] and prev["adx"] < p["adx_min"])
    di_dn = _cross_up(b["mdi"], b["pdi"]) or (last["mdi"] > last["pdi"] and prev["adx"] < p["adx_min"])
    if di_up and last["close"] > last["ema50"]:
        return _sig(UP, f"trend start: +DI over -DI, ADX {last['adx']:.1f} rising", b.index[-1], **vals)
    if di_dn and last["close"] < last["ema50"]:
        return _sig(DOWN, f"trend start: -DI over +DI, ADX {last['adx']:.1f} rising", b.index[-1], **vals)
    return _sig(None, "trend already running (no new DI cross / ADX cross)", b.index[-1], **vals)


def bollinger_reversion(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 40)
    if miss:
        return miss
    last, prev = b.iloc[-1], b.iloc[-2]
    vals = dict(bb_up=last["bb_up"], bb_lo=last["bb_lo"], adx=last["adx"])
    if last["adx"] > p["adx_max"]:
        return _sig(None, f"trending (ADX {last['adx']:.1f} > {p['adx_max']}): mean reversion off", b.index[-1], **vals)
    if prev["close"] < prev["bb_lo"] and last["close"] > last["bb_lo"] and last["close"] > last["open"]:
        return _sig(UP, "closed back inside the lower Bollinger band", b.index[-1], **vals)
    if prev["close"] > prev["bb_up"] and last["close"] < last["bb_up"] and last["close"] < last["open"]:
        return _sig(DOWN, "closed back inside the upper Bollinger band", b.index[-1], **vals)
    return _sig(None, "no close back inside a band", b.index[-1], **vals)


def volume_profile_levels(f1: pd.DataFrame, bins: int = 60, value_pct: float = 0.70) -> dict:
    """POC / VAH / VAL of 1-min futures bars (volume spread evenly over each bar's high-low)."""
    if f1 is None or len(f1) < 30 or not ind.has_volume(f1["volume"]):
        return {}
    lo, hi = float(f1["low"].min()), float(f1["high"].max())
    if hi <= lo:
        return {}
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for h, l, v in zip(f1["high"].to_numpy(), f1["low"].to_numpy(), f1["volume"].to_numpy()):
        a, z = np.searchsorted(edges, l, "right") - 1, np.searchsorted(edges, h, "right") - 1
        a, z = max(0, min(a, bins - 1)), max(0, min(z, bins - 1))
        vol[a:z + 1] += v / (z - a + 1)
    poc = int(vol.argmax())
    lo_i = hi_i = poc
    total, inside = vol.sum(), vol[poc]
    while inside < value_pct * total and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1
            inside += vol[hi_i]
        else:
            lo_i -= 1
            inside += vol[lo_i]
    mid = (edges[:-1] + edges[1:]) / 2
    return {"poc": float(mid[poc]), "vah": float(edges[hi_i + 1]), "val": float(edges[lo_i])}


def volume_profile(ctx: Context, p: dict) -> Result:
    f1, f5 = ctx.f1, ctx.today_bars(ctx.f5)
    if f1 is None or len(f5) < 3:
        return Result(reason="no data: futures bars")
    days = sorted(set(f1.index.date))
    use = f1[f1.index.date >= days[-2]] if len(days) >= 2 else f1
    vp = volume_profile_levels(use, int(p["bins"]))
    if not vp:
        return Result(reason="no data: futures volume needed for the volume profile")
    atr_v = ind.atr(ctx.f5, 14).iloc[-1]
    tol = p["touch_atr"] * atr_v if finite(atr_v) else 0
    daily = ind.daily_from_minutes(f1)
    prev = daily[daily.index.date < ctx.today]
    pdh, pdl = (float(prev["high"].iloc[-1]), float(prev["low"].iloc[-1])) if len(prev) else (np.nan, np.nan)
    last, before = f5.iloc[-1], f5.iloc[-2]
    vals = dict(poc=vp["poc"], vah=vp["vah"], val=vp["val"], pdh=pdh, pdl=pdl, fut=last["close"])
    supports = [x for x in (vp["val"], pdl) if finite(x)]
    resists = [x for x in (vp["vah"], pdh) if finite(x)]
    for s in supports:
        if last["low"] <= s + tol and last["close"] > s and last["close"] > last["open"] and before["close"] > s - tol:
            return _sig(UP, f"rejected support {s:.1f} (futures)", f5.index[-1], **vals)
    for r in resists:
        if last["high"] >= r - tol and last["close"] < r and last["close"] < last["open"] and before["close"] < r + tol:
            return _sig(DOWN, f"rejected resistance {r:.1f} (futures)", f5.index[-1], **vals)
    if last["close"] > vp["vah"] and before["close"] > vp["vah"] and f5["close"].iloc[-3] <= vp["vah"]:
        return _sig(UP, f"accepted above the value area high {vp['vah']:.1f}", f5.index[-1], **vals)
    if last["close"] < vp["val"] and before["close"] < vp["val"] and f5["close"].iloc[-3] >= vp["val"]:
        return _sig(DOWN, f"accepted below the value area low {vp['val']:.1f}", f5.index[-1], **vals)
    return _sig(None, "no rejection or acceptance at the value area / previous-day levels", f5.index[-1], **vals)


def _oi_class(f: pd.DataFrame, bars: int, p: dict) -> Optional[str]:
    if len(f) < bars + 1:
        return None
    px = f["close"].iloc[-1] / f["close"].iloc[-1 - bars] * 100 - 100
    o0, o1 = f["oi"].iloc[-1 - bars], f["oi"].iloc[-1]
    if not (finite(o0) and finite(o1) and o0 > 0):
        return None
    oi = o1 / o0 * 100 - 100
    if px >= p["min_price_pct"] and oi >= p["min_oi_pct"]:
        return "long"
    if px <= -p["min_price_pct"] and oi >= p["min_oi_pct"]:
        return "short"
    return "none"


def oi_buildup(ctx: Context, p: dict) -> Result:
    f = ctx.today_bars(ctx.f5)
    bars = int(p["bars"])
    if f is None or len(f) < bars + 2 or not f["oi"].notna().any():
        return Result(reason="no data: futures OI")
    now_c, before_c = _oi_class(f, bars, p), _oi_class(f.iloc[:-1], bars, p)
    px = f["close"].iloc[-1] / f["close"].iloc[-1 - bars] * 100 - 100
    oi = f["oi"].iloc[-1] / f["oi"].iloc[-1 - bars] * 100 - 100
    vals = dict(price_pct=px, oi_pct=oi)
    if now_c is None:
        return Result(reason="no data: futures OI")
    if now_c == "long" and before_c != "long":
        return _sig(UP, f"long build-up: futures {px:+.2f}%, OI {oi:+.2f}% in {bars} bars", f.index[-1], **vals)
    if now_c == "short" and before_c != "short":
        return _sig(DOWN, f"short build-up: futures {px:+.2f}%, OI {oi:+.2f}% in {bars} bars", f.index[-1], **vals)
    return _sig(None, f"no new build-up (futures {px:+.2f}%, OI {oi:+.2f}%)", f.index[-1], **vals)


def _sample_at(samples, when):
    older = [s for s in samples if s[0] <= when]
    return older[-1] if older else None


def pcr(ctx: Context, p: dict) -> Result:
    if not ctx.memory.pcr:
        return Result(reason="no data: option chain PCR")
    now_t, now_pcr = ctx.memory.pcr[-1]
    old = _sample_at(ctx.memory.pcr, ctx.now - pd.Timedelta(minutes=p["minutes"]))
    if old is None:
        return Result(reason=f"warming up: needs {p['minutes']:g} min of PCR samples", values={"pcr": _f(now_pcr)})
    b = ctx.b5
    if len(b) < 25:
        return Result(reason="warming up: bars")
    change = now_pcr - old[1]
    last = b.iloc[-1]
    vals = dict(pcr=now_pcr, pcr_change=change, ema20=last["ema20"])
    if change >= p["min_change"] and now_pcr >= p["bull_level"] and last["close"] > last["ema20"]:
        return _sig(UP, f"PCR rose {change:+.2f} to {now_pcr:.2f} (puts written), price above EMA20", b.index[-1], **vals)
    if change <= -p["min_change"] and now_pcr <= p["bear_level"] and last["close"] < last["ema20"]:
        return _sig(DOWN, f"PCR fell {change:+.2f} to {now_pcr:.2f} (calls written), price below EMA20", b.index[-1],
                    **vals)
    return _sig(None, f"PCR {now_pcr:.2f}, {change:+.2f} in {p['minutes']:g} min", b.index[-1], **vals)


def supertrend_flip(ctx: Context, p: dict) -> Result:
    b = ctx.b5
    miss = _need(b, 30)
    if miss:
        return miss
    st = supertrend(b, int(p["length"]), float(p["mult"])) if (p["length"], p["mult"]) != (10, 3.0) else b["st"]
    if st.iloc[-1] > 0 and st.iloc[-2] < 0:
        return _sig(UP, "Supertrend flipped up", b.index[-1])
    if st.iloc[-1] < 0 and st.iloc[-2] > 0:
        return _sig(DOWN, "Supertrend flipped down", b.index[-1])
    return _sig(None, f"Supertrend {'up' if st.iloc[-1] > 0 else 'down'}, no flip", b.index[-1])


def first_hour(ctx: Context, p: dict) -> Result:
    day = ctx.today_bars(ctx.m1)
    end = ctx.now.normalize() + pd.Timedelta(hours=10, minutes=15)
    if ctx.now < end + pd.Timedelta(minutes=ctx.tf) or len(day) < 30:
        return Result(reason="first hour (09:15-10:15) not complete")
    fh = day[day.index < end]
    hi, lo = fh["high"].max(), fh["low"].min()
    move = fh["close"].iloc[-1] / fh["open"].iloc[0] * 100 - 100
    vals = dict(fh_high=hi, fh_low=lo, fh_move_pct=move)
    b = ctx.today_bars(ctx.b5)
    b = b[b.index >= end]
    if ctx.now.time() > hhmm(p["cutoff"]):
        return _sig(None, f"after the cut-off {p['cutoff']}", b.index[-1] if len(b) else None, **vals)
    if len(b) == 0:
        return Result(reason="waiting for a bar after 10:15", values=vals)
    above, below = b["close"] > hi, b["close"] < lo
    if move >= p["min_move_pct"] and above.iloc[-1] and not above.iloc[:-1].any():
        return _sig(UP, f"first hour {move:+.2f}%, first close above its high {hi:.1f}", b.index[-1], **vals)
    if move <= -p["min_move_pct"] and below.iloc[-1] and not below.iloc[:-1].any():
        return _sig(DOWN, f"first hour {move:+.2f}%, first close below its low {lo:.1f}", b.index[-1], **vals)
    return _sig(None, f"first hour {move:+.2f}% (needs +/-{p['min_move_pct']}%) / no first break", b.index[-1], **vals)


def atr_breakout(ctx: Context, p: dict) -> Result:
    lv = ind.daily_levels(ctx.daily, ctx.today)
    day = ctx.today_bars(ctx.m1)
    b = ctx.today_bars(ctx.b5)
    if not finite(lv.daily_atr) or len(day) == 0 or len(b) < 1:
        return Result(reason="no data: daily ATR (needs 15 daily candles)")
    o = float(day["open"].iloc[0])
    up_l, dn_l = o + p["k"] * lv.daily_atr, o - p["k"] * lv.daily_atr
    vals = dict(open=o, upper=up_l, lower=dn_l, daily_atr=lv.daily_atr)
    above, below = b["close"] > up_l, b["close"] < dn_l
    if above.iloc[-1] and not above.iloc[:-1].any():
        return _sig(UP, f"first close above open + {p['k']} x daily ATR ({up_l:.1f})", b.index[-1], **vals)
    if below.iloc[-1] and not below.iloc[:-1].any():
        return _sig(DOWN, f"first close below open - {p['k']} x daily ATR ({dn_l:.1f})", b.index[-1], **vals)
    return _sig(None, f"inside {dn_l:.1f}-{up_l:.1f}", b.index[-1], **vals)


def mtf_confirmation(ctx: Context, p: dict) -> Result:
    b, h = ctx.b5, ctx.b15
    if len(b) < 60 or len(h) < 55 or len(ctx.m1) < 25:
        return Result(reason="warming up: 15-min / 5-min / 1-min history")
    hl, last = h.iloc[-1], b.iloc[-1]
    htf_up = hl["ema20"] > hl["ema50"] and hl["close"] > hl["ema20"] and hl["st"] > 0
    htf_dn = hl["ema20"] < hl["ema50"] and hl["close"] < hl["ema20"] and hl["st"] < 0
    trig_up = _cross_up(b["close"], b["ema9"]) or _cross_up(b["macd"], b["macd_sig"])
    trig_dn = _cross_dn(b["close"], b["ema9"]) or _cross_dn(b["macd"], b["macd_sig"])
    m = ctx.m1["close"]
    e20 = ind.ema(m, 20)
    mom_up = m.iloc[-1] > e20.iloc[-1] and m.iloc[-1] > m.iloc[-4]
    mom_dn = m.iloc[-1] < e20.iloc[-1] and m.iloc[-1] < m.iloc[-4]
    vals = dict(htf="up" if htf_up else "down" if htf_dn else "mixed", rsi5=last["rsi"])
    if htf_up and trig_up and mom_up:
        return _sig(UP, "15-min up-trend + 5-min trigger + 1-min momentum", b.index[-1], **vals)
    if htf_dn and trig_dn and mom_dn:
        return _sig(DOWN, "15-min down-trend + 5-min trigger + 1-min momentum", b.index[-1], **vals)
    return _sig(None, f"timeframes not aligned (15-min {vals['htf']})", b.index[-1], **vals)


def premium_momentum(ctx: Context, p: dict) -> Result:
    mem = ctx.memory.premiums
    if not mem:
        return Result(reason="no data: option chain premiums")
    now_t, now_s = mem[-1]
    old = _sample_at(mem, ctx.now - pd.Timedelta(minutes=p["minutes"]))
    if old is None:
        return Result(reason=f"warming up: needs {p['minutes']:g} min of premium samples")
    strike = now_s.get("atm")
    a, z = old[1].get("legs", {}).get(strike), now_s.get("legs", {}).get(strike)
    if not a or not z or not all(finite(x) and x > 0 for x in (a[0], a[1], z[0], z[1])):
        return Result(reason="no data: the ATM strike has no earlier premium sample")
    ce = z[0] / a[0] * 100 - 100
    pe = z[1] / a[1] * 100 - 100
    vals = dict(strike=strike, ce_pct=ce, pe_pct=pe)
    bar = ctx.b5.index[-1] if len(ctx.b5) else None
    if ce >= p["min_rise_pct"] and pe <= -p["min_fall_pct"]:
        return _sig(UP, f"ATM {strike:g} CE {ce:+.1f}%, PE {pe:+.1f}% in {p['minutes']:g} min", bar, **vals)
    if pe >= p["min_rise_pct"] and ce <= -p["min_fall_pct"]:
        return _sig(DOWN, f"ATM {strike:g} PE {pe:+.1f}%, CE {ce:+.1f}% in {p['minutes']:g} min", bar, **vals)
    return _sig(None, f"ATM CE {ce:+.1f}% / PE {pe:+.1f}% in {p['minutes']:g} min", bar, **vals)


@dataclass(frozen=True)
class Spec:
    key: str
    title: str
    fn: Callable[[Context, dict], Result]
    params: Dict[str, object]
    needs_chain: bool = False
    category: str = ""
    ce: str = ""                 # when it buys a CE (UP)
    pe: str = ""                 # when it buys a PE (DOWN)
    data: str = "spot index bars"
    per_bar: bool = True         # result can only change when a new signal-timeframe bar completes


TIMEFRAMES = (1, 3, 5, 10, 15, 30, 60)


_META = {
    "breakout_momentum": ("Momentum", "close above the N-bar high on an impulse bar (body, RVOL or range/ATR, RSI)",
                          "close below the N-bar low on an impulse bar", "spot bars (+ futures volume if any)", True),
    "ema_pullback": ("Trend following", "EMA20 > EMA50, pull-back to EMA20, close above the previous high",
                     "EMA20 < EMA50, pull-back to EMA20, close below the previous low", "spot index bars", True),
    "vwap_reversal": ("Price action", "futures stretched below VWAP, reversal bar", "stretched above VWAP, reversal bar",
                      "index futures bars with volume", True),
    "orb": ("Price action", "first close above the opening range", "first close below the opening range",
            "spot index bars", True),
    "rsi_macd_divergence": ("Momentum", "bullish RSI divergence + MACD histogram turning up",
                            "bearish RSI divergence + MACD histogram turning down", "spot index bars", True),
    "adx_trend": ("Trend following", "+DI over -DI, ADX >= adx_min and rising, above EMA50",
                  "-DI over +DI, ADX rising, below EMA50", "spot index bars", True),
    "bollinger_reversion": ("Mean reversion", "close back inside the lower band in a range",
                            "close back inside the upper band in a range", "spot index bars", True),
    "volume_profile": ("Combination", "rejection of VAL / PDL or acceptance above VAH",
                       "rejection of VAH / PDH or acceptance below VAL", "index futures bars with volume", True),
    "oi_buildup": ("Options", "long build-up (futures price and OI up)", "short build-up (price down, OI up)",
                   "index futures bars with OI", True),
    "pcr": ("Options", "PCR up by min_change to >= bull_level, price above EMA20",
            "PCR down by min_change to <= bear_level, price below EMA20", "live option chain OI", False),
    "supertrend": ("Trend following", "Supertrend flips up", "Supertrend flips down", "spot index bars", True),
    "first_hour": ("Momentum", "after 10:15, first close above the first-hour high of an up first hour",
                   "first close below the first-hour low of a down first hour", "spot index bars", True),
    "atr_breakout": ("Volatility", "first close above open + k x daily ATR", "first close below open - k x daily ATR",
                     "spot index bars + daily candles", True),
    "mtf_confirmation": ("Combination", "15-min up-trend + 5-min trigger + 1-min momentum",
                         "15-min down-trend + 5-min trigger + 1-min momentum", "spot index bars", False),
    "premium_momentum": ("Options", "ATM CE premium up and PE down by set % in N min", "the mirror",
                         "live option chain premiums", False),
}

REGISTRY: Dict[str, Spec] = {s.key: s for s in [
    Spec("breakout_momentum", "Breakout Momentum", breakout_momentum,
         {"lookback": 20, "min_body": 0.55, "min_rvol": 1.5, "min_range_atr": 1.3, "rsi_up": 55}),
    Spec("ema_pullback", "EMA Pullback", ema_pullback, {"touch_atr": 0.15}),
    Spec("vwap_reversal", "VWAP Reversal", vwap_reversal, {"stretch_atr": 1.5, "rsi_low": 35}),
    Spec("orb", "Opening Range Breakout", orb, {"range_minutes": 15, "buffer_atr": 0.1, "cutoff": "11:30"}),
    Spec("rsi_macd_divergence", "RSI + MACD Divergence", rsi_macd_divergence,
         {"lookback": 40, "pivot": 3, "max_age": 8, "rsi_gap": 2.0}),
    Spec("adx_trend", "ADX Trend Following", adx_trend, {"adx_min": 25.0}),
    Spec("bollinger_reversion", "Bollinger Band Mean Reversion", bollinger_reversion, {"adx_max": 20.0}),
    Spec("volume_profile", "Volume Profile + S/R", volume_profile, {"bins": 60, "touch_atr": 0.2}),
    Spec("oi_buildup", "OI Buildup", oi_buildup, {"bars": 3, "min_price_pct": 0.15, "min_oi_pct": 0.5}),
    Spec("pcr", "PCR Strategy", pcr, {"minutes": 30.0, "min_change": 0.10, "bull_level": 1.0, "bear_level": 0.9},
         needs_chain=True),
    Spec("supertrend", "Supertrend", supertrend_flip, {"length": 10, "mult": 3.0}),
    Spec("first_hour", "First-Hour Momentum", first_hour, {"min_move_pct": 0.4, "cutoff": "13:00"}),
    Spec("atr_breakout", "ATR Volatility Breakout", atr_breakout, {"k": 0.5}),
    Spec("mtf_confirmation", "Multi-Timeframe Confirmation", mtf_confirmation, {}),
    Spec("premium_momentum", "Options Premium Momentum", premium_momentum,
         {"minutes": 10.0, "min_rise_pct": 8.0, "min_fall_pct": 5.0}, needs_chain=True),
]}
REGISTRY = {k: replace(s, category=_META[k][0], ce=_META[k][1], pe=_META[k][2], data=_META[k][3],
                       per_bar=_META[k][4]) for k, s in REGISTRY.items()}


def coerce_params(key: str, raw: Dict[str, str]) -> dict:
    spec = REGISTRY[key]
    out = dict(spec.params)
    raw = dict(raw)
    if "timeframe" in raw:
        tf = int(float(raw.pop("timeframe")))
        if tf not in TIMEFRAMES:
            raise ValueError(f"[STRATEGY_{key.upper()}] timeframe must be one of {', '.join(map(str, TIMEFRAMES))}")
        out["timeframe"] = tf
    for name, value in raw.items():
        if name not in spec.params:
            raise ValueError(f"[STRATEGY_{key.upper()}] unknown setting {name}")
        default = spec.params[name]
        out[name] = (int(float(value)) if isinstance(default, int) and not isinstance(default, bool)
                     else float(value) if isinstance(default, float) else str(value).strip())
    if "cutoff" in out:
        hhmm(out["cutoff"])
    return out


# ---------------------------------------------------------------------------------------------- engine
class StrategyEngine:
    """Builds the bars once per new bar and evaluates every enabled strategy for every index.

    Each strategy runs on its own timeframe (`timeframe` in its [STRATEGY_<KEY>] section, else the
    [STRATEGIES] default). Bars of a timeframe are rebuilt only when one of its bars completes; a strategy whose
    result can only change with a new bar (Spec.per_bar) is re-evaluated only then."""

    def __init__(self, enabled: List[str], params: Dict[str, dict], tf: int = 5):
        unknown = [k for k in enabled if k not in REGISTRY]
        if unknown:
            raise ValueError(f"[STRATEGIES] unknown strategy: {', '.join(unknown)}")
        if tf not in TIMEFRAMES:
            raise ValueError(f"[STRATEGIES] timeframe must be one of {', '.join(map(str, TIMEFRAMES))}")
        self.enabled = list(enabled)
        self.tf = tf
        self.params, self.tfs = {}, {}
        for k in self.enabled:
            p = dict(params.get(k, REGISTRY[k].params))
            self.tfs[k] = int(p.pop("timeframe", tf))
            self.params[k] = p
        self.memory: Dict[str, Memory] = collections.defaultdict(Memory)
        self.fired: Dict[tuple, str] = {}
        self.last: Dict[str, Dict[str, dict]] = {}
        self._bars: Dict[tuple, tuple] = {}      # (index, tf) -> (key, bars, futures bars)
        self._results: Dict[tuple, tuple] = {}   # (strategy, index) -> (bars key, Result)

    @property
    def needs_chain(self) -> bool:
        return any(REGISTRY[k].needs_chain for k in self.enabled)

    def remember_chain(self, isnap, now: pd.Timestamp):
        chain = isnap.chain
        if chain is None or chain.empty or not finite(fnum(isnap.ltp)):
            return
        mem = self.memory[isnap.name]
        stamp = isnap.chain_time or now
        if mem.premiums and mem.premiums[-1][0] == stamp:
            return    # same (cached) chain as last cycle
        d = analyse_chain(chain, float(isnap.ltp), np.nan, isnap.step)
        if d.available and finite(d.pcr_oi):
            mem.pcr.append((stamp, float(d.pcr_oi)))
        if d.available and finite(d.atm_iv):
            mem.iv.append((stamp, float(d.atm_iv)))
        legs = {float(r["strike"]): (fnum(r.get("ce_ltp")), fnum(r.get("pe_ltp"))) for _, r in chain.iterrows()}
        atm = float(chain.iloc[(chain["strike"] - float(isnap.ltp)).abs().argsort().iloc[0]]["strike"])
        mem.premiums.append((stamp, {"atm": atm, "legs": legs}))

    def remember_vix(self, name: str, vix, now: pd.Timestamp):
        v = fnum(vix)
        mem = self.memory[name].vix
        if finite(v) and v > 0 and (not mem or mem[-1][0] != now):
            mem.append((now, float(v)))

    @staticmethod
    def _bar_key(frame: Optional[pd.DataFrame], now: pd.Timestamp, tf: int) -> tuple:
        """Changes only when a bar of `tf` minutes completes (or late 1-min data arrives for it)."""
        if frame is None or len(frame) == 0:
            return (None, 0)
        open_ = now.normalize() + pd.Timedelta(hours=9, minutes=15)
        boundary = open_ + pd.Timedelta(minutes=tf * max(0, int((now - open_) / pd.Timedelta(minutes=tf))))
        return (frame.index[0], int(np.searchsorted(frame.index.values, boundary.to_datetime64())), boundary)

    def _signal_bars(self, isnap, m1, f1, now, tf):
        key = (self._bar_key(m1, now, tf), self._bar_key(f1, now, tf))
        cached = self._bars.get((isnap.name, tf))
        if cached is None or cached[0] != key:
            depth = max(3000, tf * 400)
            b = add_indicators(completed(ind.resample(m1.iloc[-depth:], tf), now, tf))
            f = completed(ind.resample(f1.iloc[-depth:], tf), now, tf) if f1 is not None and len(f1) else \
                pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi"])
            cached = (key, b, f)
            self._bars[(isnap.name, tf)] = cached
        return cached

    def context(self, isnap, now: pd.Timestamp, tf: Optional[int] = None) -> Context:
        tf = tf or self.tf
        m1 = completed(isnap.spot_minutes, now, 1)
        f1 = completed(isnap.fut_minutes, now, 1) if isnap.fut_minutes is not None else None
        _, b, f = self._signal_bars(isnap, m1, f1, now, tf)
        _, b15, _ = self._signal_bars(isnap, m1, f1, now, 15) if tf != 15 else (None, b, None)
        return Context(index=isnap.name, now=now, price=fnum(isnap.ltp), m1=m1, b5=b, b15=b15, f5=f, f1=f1,
                       daily=isnap.daily, chain=isnap.chain, step=isnap.step, memory=self.memory[isnap.name], tf=tf)

    def _key_of(self, ctx: Context) -> tuple:
        b = ctx.b5
        return (ctx.tf, b.index[-1] if len(b) else None, len(b), len(ctx.f5), ctx.today)

    def evaluate(self, snap, now: pd.Timestamp, stale: Optional[set] = None) -> List[dict]:
        """Run every enabled strategy; returns NEW signals (once per strategy / index / direction / bar)."""
        stale = stale or set()
        new, table = [], {}
        contexts: Dict[int, Dict[str, Context]] = {}
        for name, isnap in snap.indices.items():
            self.remember_chain(isnap, now)
            self.remember_vix(name, getattr(snap, "vix", None), now)
        for tf in sorted(set(self.tfs.values())):
            per_tf = {name: self.context(isnap, now, tf) for name, isnap in snap.indices.items()}
            for c in per_tf.values():
                c.peers = per_tf
            contexts[tf] = per_tf
        for name in snap.indices:
            for key in self.enabled:
                ctx = contexts[self.tfs[key]][name]
                spec = REGISTRY[key]
                cache_key = self._key_of(ctx) if spec.per_bar else None
                hit = self._results.get((key, name))
                if cache_key is not None and hit is not None and hit[0] == cache_key:
                    r = hit[1]
                else:
                    try:
                        r = spec.fn(ctx, self.params[key])
                    except Exception as error:   # one broken strategy never stops the others
                        r = Result(reason=f"error: {type(error).__name__}: {error}"[:160])
                    if cache_key is not None:
                        self._results[(key, name)] = (cache_key, r)
                table.setdefault(key, {})[name] = {"direction": r.direction, "reason": r.reason, "values": r.values,
                                                   "bar": r.bar}
                if r.direction and r.bar and name not in stale:
                    dedup = (key, name, r.direction)
                    if self.fired.get(dedup) != r.bar:
                        self.fired[dedup] = r.bar
                        new.append({"strategy": key, "title": spec.title, "index": name,
                                    "direction": r.direction, "reason": r.reason, "values": r.values, "bar": r.bar,
                                    "timeframe": self.tfs[key]})
        self.last = table
        return new

    def table(self) -> dict:
        return {k: {"title": REGISTRY[k].title, "category": REGISTRY[k].category, "timeframe": self.tfs[k],
                    "indices": self.last.get(k, {})} for k in self.enabled}


# ---------------------------------------------------------------------------------------------- config
PROFILE_DEFAULTS = {"entry_windows": ["SIGNAL"], "min_quality": "CONFIRMED", "stop_pct": 20.0, "target_pct": 35.0,
                    "breakeven_pct": 15.0, "trail_pct": 10.0, "fade_exit": 0.0, "max_hold_minutes": 45.0,
                    "max_trades": 4, "max_daily_loss": 5000.0, "reentry_minutes": 10.0, "loss_wait_minutes": 15.0}


def load_config(path=None):
    """(StrategyEngine or None, {strategy key: paper Profile}) from [STRATEGIES] and [STRATEGY_<KEY>]."""
    from dataclasses import fields as dc_fields

    from src.config import settings
    from src.movement.paper import Profile

    parser = settings.read_ini(path or settings.CONFIG_FILE)
    if not parser.has_section("STRATEGIES"):
        return None, {}
    sec = dict(parser["STRATEGIES"])
    raw_enabled = str(sec.pop("enabled", "")).strip()
    if raw_enabled.lower() in ("", "none", "off", "false"):
        return None, {}
    enabled = list(REGISTRY) if raw_enabled.lower() == "all" else [k.strip() for k in raw_enabled.split(",") if k.strip()]
    tf = int(float(sec.pop("timeframe", 5)))
    if tf not in TIMEFRAMES:
        raise ValueError(f"[STRATEGIES] timeframe must be one of {', '.join(map(str, TIMEFRAMES))}")
    names = {f.name for f in dc_fields(Profile)} - {"name", "triggers", "enabled"}
    unknown = sorted(set(sec) - names)
    if unknown:
        raise ValueError(f"[STRATEGIES] unknown setting(s): {', '.join(unknown)}")

    def apply(profile, items, section):
        for key, raw in items.items():
            current = getattr(profile, key)
            if isinstance(current, bool):
                value = str(raw).strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, list):
                value = [v.strip() for v in str(raw).split(",") if v.strip()] if isinstance(raw, str) else list(raw)
            elif isinstance(current, int):
                value = int(float(raw))
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = str(raw).strip()
            setattr(profile, key, value)

    params, profiles = {}, {}
    for key in enabled:
        if key not in REGISTRY:
            raise ValueError(f"[STRATEGIES] unknown strategy {key}; known: {', '.join(REGISTRY)}")
        p = Profile(name=key)
        apply(p, PROFILE_DEFAULTS, "defaults")
        apply(p, sec, "STRATEGIES")
        own = dict(parser[f"STRATEGY_{key.upper()}"]) if parser.has_section(f"STRATEGY_{key.upper()}") else {}
        if str(own.pop("enabled", "true")).strip().lower() in ("0", "false", "no", "off"):
            continue
        apply(p, {k: v for k, v in own.items() if k in names}, f"STRATEGY_{key.upper()}")
        p.entry_windows = ["SIGNAL"]
        p.fade_exit = 0.0
        profiles[key] = p.validate()
        params[key] = coerce_params(key, {k: v for k, v in own.items() if k not in names})
    if not profiles:
        return None, {}
    return StrategyEngine(list(profiles), params, tf), profiles


def to_trigger(signal: dict, price: float, mock: bool = False) -> dict:
    """A strategy signal in the shape PaperTrader.try_open() takes."""
    return {"index": signal["index"], "direction": signal["direction"], "window": "SIGNAL", "quality": "CONFIRMED",
            "ref_price": price, "price": price, "change": None, "change_pct": None, "threshold_ratio": 1.0,
            "mock": mock, "message": f"{signal['title']}: {signal['reason']}"}


# ---------------------------------------------------------------------------------------------- the rest
if __name__ != "__main__":      # python -m src.movement.strategies: the imported copy below registers it
    from src.movement import library as _library  # noqa: E402,F401  (adds its SPECS to REGISTRY)


def catalogue(path=None) -> List[dict]:
    """Every strategy with its rules, data, timeframe and paper exit settings (from the ini when present)."""
    rows = []
    try:
        engine, profiles = load_config(path)
    except Exception:          # a broken ini still lists the library
        engine, profiles = None, {}
    for key, spec in REGISTRY.items():
        prof = profiles.get(key)
        rows.append({"key": key, "title": spec.title, "category": spec.category, "buy_ce_when": spec.ce,
                     "buy_pe_when": spec.pe, "data": spec.data, "params": dict(spec.params),
                     "enabled": prof is not None,
                     "timeframe": engine.tfs.get(key) if engine is not None and key in engine.tfs else None,
                     "stop_pct": prof.stop_pct if prof else None, "target_pct": prof.target_pct if prof else None,
                     "trail_pct": prof.trail_pct if prof else None,
                     "reward_risk": round(prof.target_pct / prof.stop_pct, 2) if prof else None})
    return rows


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="The strategy library")
    ap.add_argument("--list", action="store_true", help="print every strategy, its CE / PE rule and data")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    rows = catalogue(args.config)
    by_cat: Dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    print(f"{len(rows)} strategies ({sum(r['enabled'] for r in rows)} enabled in the ini)\n")
    for cat, items in by_cat.items():
        print(f"== {cat} ==")
        for r in items:
            exits = (f"tf {r['timeframe']}m, stop {r['stop_pct']:g}% / target {r['target_pct']:g}% "
                     f"(R:R {r['reward_risk']:g}), trail {r['trail_pct']:g}%") if r["enabled"] else "disabled"
            print(f"  {r['key']:<22}{r['title']}  [{exits}]")
            print(f"  {'':<22}CE: {r['buy_ce_when']}")
            print(f"  {'':<22}PE: {r['buy_pe_when']}")
            print(f"  {'':<22}data: {r['data']}")
        print()


if __name__ == "__main__":
    from src.movement.strategies import main as _main    # the module copy with the full library registered
    _main()
