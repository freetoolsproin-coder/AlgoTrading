"""
The strategies of the F&O engine: 35 library strategies, Supertrend + RSI + EMA confirmation (36), option-chain confluence (37), multi-factor scanner (38), and VWAP breakout, pivot point breakout, Supertrend + MACD, VWAP + RSI and
option volume / OI divergence (39-43).

Each strategy is a pure function  fn(f, p, ctx) -> (up, down)  over the enriched base-timeframe frame `f`
(see indicators.enrich). `up` / `down` are boolean Series marking the completed bar on which the strategy's
own trigger EVENT happens: up = candidate BUY CE, down = candidate BUY PE. A trigger alone never becomes a
trade: analysis.py requires the confluence checks (structure, momentum, VWAP / EMA location, volume, higher
timeframes, candle, lower timeframe) and the plan requires option-premium confirmation.

Every default below is an untested starting value. `grid` lists the few values the walk-forward test may
choose between; keep grids small, a large grid is an invitation to overfit.

    python -m src.fno strategies      prints every strategy with its CE / PE / exit rules and settings
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.fno import indicators as ind
from src.fno.indicators import event, first_of_day, prev_high, prev_low, xdn, xup
from src.scanner.indicators import adx as _adx, atr as _atr, ema, macd as _macd, rsi as _rsi

COMMON_EXIT = ("premium stop-loss, T1 / T2 / T3 scale-out, breakeven after T1, ATR or swing trail, underlying "
               "invalidation level, time stop, square-off")


@dataclass
class Ctx:
    """Data a strategy may use besides the frame. Series are aligned to the frame's index and causal."""
    symbol: str = ""
    tf: int = 5
    htf: Dict[object, pd.Series] = field(default_factory=dict)    # 15 / 30 / 60 / "D" -> +1 / 0 / -1
    ltf: Optional[pd.Series] = None                              # lower-timeframe momentum, +1 / 0 / -1
    bench_ret: Optional[pd.Series] = None                        # benchmark % change from its day open
    chain: Optional[pd.DataFrame] = None                         # option-chain series (see data.chain_series)


@dataclass
class Strategy:
    key: str
    number: int
    name: str
    category: str               # trend | breakout | momentum | reversal | confluence | options
    kind: str                   # continuation | breakout | reversal
    ce: str
    pe: str
    exit: str
    fn: Callable
    defaults: Dict[str, float]
    grid: Dict[str, list]
    needs: tuple = ()           # volume | chain | benchmark | htf

    def params(self, overrides: Optional[dict] = None) -> dict:
        p = dict(self.defaults)
        for k, v in (overrides or {}).items():
            if k in p:
                try:
                    p[k] = type(p[k])(float(v)) if not isinstance(p[k], str) else str(v)
                except (TypeError, ValueError):
                    pass
        return p

    def enabled(self, overrides: Optional[dict] = None) -> bool:
        return str((overrides or {}).get("enabled", "true")).strip().lower() in ("1", "true", "yes", "on")

    def signals(self, f: pd.DataFrame, ctx: Ctx, overrides: Optional[dict] = None) -> pd.Series:
        """+1 on a CE trigger bar, -1 on a PE trigger bar, else 0."""
        up, down = self.fn(f, self.params(overrides), ctx)
        up = pd.Series(up, index=f.index).fillna(False).astype(bool)
        down = pd.Series(down, index=f.index).fillna(False).astype(bool)
        return (up & ~down).astype(np.int8) - (down & ~up).astype(np.int8)

    def describe(self, overrides: Optional[dict] = None) -> dict:
        return {"key": self.key, "number": self.number, "name": self.name, "category": self.category,
                "kind": self.kind, "ce": self.ce, "pe": self.pe, "exit": self.exit, "needs": list(self.needs),
                "enabled": self.enabled(overrides), "params": self.params(overrides), "grid": self.grid}


REGISTRY: Dict[str, Strategy] = {}


def strategy(key, number, name, category, kind, ce, pe, defaults=None, grid=None, needs=(), exit=None):
    def wrap(fn):
        REGISTRY[key] = Strategy(key, number, name, category, kind, ce, pe,
                                 exit or f"opposite {name} trigger; {COMMON_EXIT}", fn, defaults or {}, grid or {}, needs)
        return fn
    return wrap


def all_strategies() -> List[Strategy]:
    return sorted(REGISTRY.values(), key=lambda s: s.number)


def _none(f):
    no = pd.Series(False, index=f.index)
    return no, no


def _bull_candle(f, pos=0.6):
    return (f["close"] > f["open"]) & (f["close_pos"] >= pos)


def _bear_candle(f, pos=0.6):
    return (f["close"] < f["open"]) & (f["close_pos"] <= 1 - pos)


# =============================================================================  trend following (1-7)
@strategy("ema_9_21", 1, "EMA 9/21 crossover", "trend", "continuation",
          "EMA9 crosses above EMA21 and the bar closes above both", "EMA9 crosses below EMA21 and the bar closes below both",
          {"fast": 9, "slow": 21}, {"fast": [8, 9, 12], "slow": [21, 26]})
def ema_9_21(f, p, ctx):
    a, b = ema(f["close"], int(p["fast"])), ema(f["close"], int(p["slow"]))
    return xup(a, b) & (f["close"] > a), xdn(a, b) & (f["close"] < a)


@strategy("ema_20_50", 2, "EMA 20/50 crossover", "trend", "continuation",
          "EMA20 crosses above EMA50, EMA50 not falling", "EMA20 crosses below EMA50, EMA50 not rising",
          {"fast": 20, "slow": 50, "slope_bars": 5}, {"fast": [18, 20], "slow": [50, 55]})
def ema_20_50(f, p, ctx):
    a, b = ema(f["close"], int(p["fast"])), ema(f["close"], int(p["slow"]))
    n = int(p["slope_bars"])
    return xup(a, b) & (b >= b.shift(n)), xdn(a, b) & (b <= b.shift(n))


@strategy("supertrend", 3, "Supertrend", "trend", "continuation",
          "Supertrend flips up with the close above EMA50", "Supertrend flips down with the close below EMA50",
          {"length": 10, "mult": 3.0}, {"length": [7, 10], "mult": [2.0, 3.0]},
          exit="Supertrend flips back; " + COMMON_EXIT)
def supertrend(f, p, ctx):
    _, d = ind.supertrend(f, int(p["length"]), float(p["mult"]))
    return ((d == 1) & (d.shift(1) == -1) & (f["close"] > f["ema50"]),
            (d == -1) & (d.shift(1) == 1) & (f["close"] < f["ema50"]))


@strategy("adx_dmi", 4, "ADX + DMI", "trend", "continuation",
          "+DI crosses above -DI with ADX >= adx_min and rising, or ADX rises through adx_min while +DI > -DI",
          "-DI crosses above +DI with ADX >= adx_min and rising, or ADX rises through adx_min while -DI > +DI",
          {"length": 14, "adx_min": 22.0}, {"adx_min": [20.0, 25.0]})
def adx_dmi(f, p, ctx):
    a, pdi, mdi = _adx(f, int(p["length"]))
    strong = (a >= p["adx_min"]) & (a > a.shift(2))
    through = xup(a, p["adx_min"])
    return (xup(pdi, mdi) & strong) | (through & (pdi > mdi)), (xup(mdi, pdi) & strong) | (through & (mdi > pdi))


@strategy("vwap_trend", 5, "VWAP trend continuation", "trend", "continuation",
          "price held above a rising VWAP for hold_bars bars, then closes above the previous lookback-bar high",
          "price held below a falling VWAP for hold_bars bars, then closes below the previous lookback-bar low",
          {"hold_bars": 6, "lookback": 6}, {"hold_bars": [4, 6, 9]}, needs=("volume",),
          exit="a close on the other side of VWAP; " + COMMON_EXIT)
def vwap_trend(f, p, ctx):
    k, n = int(p["hold_bars"]), int(p["lookback"])
    v, c = f["vwap"], f["close"]
    above = (c > v).rolling(k, min_periods=k).min() == 1
    below = (c < v).rolling(k, min_periods=k).min() == 1
    return above & (v > v.shift(k)) & xup(c, prev_high(f, n)), below & (v < v.shift(k)) & xdn(c, prev_low(f, n))


@strategy("macd_cross", 6, "MACD crossover", "trend", "continuation",
          "MACD line crosses above its signal line with the histogram rising",
          "MACD line crosses below its signal line with the histogram falling",
          {"fast": 12, "slow": 26, "signal": 9, "zero_filter": 0}, {"fast": [8, 12], "slow": [21, 26]})
def macd_cross(f, p, ctx):
    line, sig, hist = _macd(f["close"], int(p["fast"]), int(p["slow"]), int(p["signal"]))
    zf = int(p["zero_filter"]) == 1
    up = xup(line, sig) & (hist > hist.shift(1)) & ((line < 0) | (not zf))
    dn = xdn(line, sig) & (hist < hist.shift(1)) & ((line > 0) | (not zf))
    return up, dn


@strategy("ichimoku", 7, "Ichimoku Cloud", "trend", "continuation",
          "Tenkan crosses above Kijun with the close above the cloud, or the close breaks above the cloud with Tenkan > Kijun",
          "Tenkan crosses below Kijun with the close below the cloud, or the close breaks below the cloud with Tenkan < Kijun",
          {"tenkan": 9, "kijun": 26, "span_b": 52}, {})
def ichimoku(f, p, ctx):
    t, k, a, b = ind.ichimoku(f, int(p["tenkan"]), int(p["kijun"]), int(p["span_b"]))
    top, bot, c = np.maximum(a, b), np.minimum(a, b), f["close"]
    return ((xup(t, k) & (c > top)) | (xup(c, top) & (t > k)), (xdn(t, k) & (c < bot)) | (xdn(c, bot) & (t < k)))


# =============================================================================  breakout (8-14)
def _range_break(f, p, ctx, minutes):
    hi, lo = ind.opening_range(f, minutes, ctx.tf)
    buf = float(p.get("buffer_atr", 0.0)) * f["atr"].fillna(0.0)
    return first_of_day(f["close"] > hi + buf), first_of_day(f["close"] < lo - buf)


@strategy("orb", 8, "Opening Range Breakout", "breakout", "breakout",
          "first close above the high of the opening range (default 30 minutes)",
          "first close below the low of the opening range",
          {"minutes": 30, "buffer_atr": 0.1}, {"minutes": [15, 30, 45]},
          exit="a close back inside the opening range; " + COMMON_EXIT)
def orb(f, p, ctx):
    return _range_break(f, p, ctx, int(p["minutes"]))


@strategy("pdh_pdl", 9, "Previous Day High/Low breakout", "breakout", "breakout",
          "first close of the day above the previous day's high (+ buffer)",
          "first close of the day below the previous day's low (- buffer)",
          {"buffer_atr": 0.1}, {"buffer_atr": [0.0, 0.1, 0.25]},
          exit="a close back through the broken level; " + COMMON_EXIT)
def pdh_pdl(f, p, ctx):
    buf = float(p["buffer_atr"]) * f["atr"].fillna(0.0)
    return first_of_day(f["close"] > f["pd_high"] + buf), first_of_day(f["close"] < f["pd_low"] - buf)


@strategy("first_15", 10, "First 15-minute candle breakout", "breakout", "breakout",
          "first close above the high of the 09:15-09:30 candle", "first close below the low of the 09:15-09:30 candle",
          {"buffer_atr": 0.05}, {"buffer_atr": [0.0, 0.05, 0.15]},
          exit="a close back inside the first candle; " + COMMON_EXIT)
def first_15(f, p, ctx):
    return _range_break(f, p, ctx, 15)


@strategy("bb_squeeze", 11, "Bollinger Band squeeze", "breakout", "breakout",
          "Bollinger Bands were inside the Keltner channel within the last squeeze_bars bars and the close breaks above the upper band",
          "the same squeeze, and the close breaks below the lower band",
          {"squeeze_bars": 6}, {"squeeze_bars": [4, 6, 10]})
def bb_squeeze(f, p, ctx):
    sq = ((f["bb_up"] < f["kc_up"]) & (f["bb_low"] > f["kc_low"])).astype(float)
    recent = sq.shift(1).rolling(int(p["squeeze_bars"]), min_periods=1).max() == 1
    return recent & xup(f["close"], f["bb_up"]), recent & xdn(f["close"], f["bb_low"])


@strategy("donchian", 12, "Donchian Channel breakout", "breakout", "breakout",
          "close crosses above the highest high of the previous N bars", "close crosses below the lowest low of the previous N bars",
          {"length": 20}, {"length": [20, 30, 40]},
          exit="close below the opposite half-channel; " + COMMON_EXIT)
def donchian(f, p, ctx):
    n = int(p["length"])
    return xup(f["close"], prev_high(f, n)), xdn(f["close"], prev_low(f, n))


@strategy("cpr", 13, "CPR breakout", "breakout", "breakout",
          "narrow CPR day (width <= max_width_pct of the pivot): first close above the top of the CPR after min_bar bars",
          "narrow CPR day: first close below the bottom of the CPR after min_bar bars",
          {"max_width_pct": 0.25, "min_bar": 3}, {"max_width_pct": [0.15, 0.25, 0.4]},
          exit="a close back inside the CPR; " + COMMON_EXIT)
def cpr(f, p, ctx):
    ok = (f["cpr_width_pct"] <= p["max_width_pct"]) & (f["bar_no"] >= int(p["min_bar"]))
    return (first_of_day(ok & (f["close"] > f["cpr_top"]) & (f["open"] <= f["cpr_top"] * 1.01)),
            first_of_day(ok & (f["close"] < f["cpr_bot"]) & (f["open"] >= f["cpr_bot"] * 0.99)))


@strategy("inside_bar", 14, "Inside Bar breakout", "breakout", "breakout",
          "the previous bar is inside its mother bar and this bar closes above the mother bar's high",
          "the previous bar is inside its mother bar and this bar closes below the mother bar's low",
          {"min_mother_atr": 0.8}, {"min_mother_atr": [0.5, 0.8, 1.2]},
          exit="a close beyond the other side of the mother bar; " + COMMON_EXIT)
def inside_bar(f, p, ctx):
    h, l = f["high"], f["low"]
    inside = (h.shift(1) <= h.shift(2)) & (l.shift(1) >= l.shift(2)) & (f["bar_no"] >= 2)
    big = (h.shift(2) - l.shift(2)) >= p["min_mother_atr"] * f["atr"].shift(2)
    return inside & big & (f["close"] > h.shift(2)), inside & big & (f["close"] < l.shift(2))


# =============================================================================  momentum (15-20)
@strategy("rsi_ema", 15, "RSI + EMA momentum", "momentum", "continuation",
          "RSI rises through rsi_bull while close > EMA20 > EMA50", "RSI falls through rsi_bear while close < EMA20 < EMA50",
          {"rsi_bull": 60.0, "rsi_bear": 40.0}, {"rsi_bull": [55.0, 60.0], "rsi_bear": [40.0, 45.0]})
def rsi_ema(f, p, ctx):
    c = f["close"]
    return (xup(f["rsi"], p["rsi_bull"]) & (c > f["ema20"]) & (f["ema20"] > f["ema50"]),
            xdn(f["rsi"], p["rsi_bear"]) & (c < f["ema20"]) & (f["ema20"] < f["ema50"]))


@strategy("stoch_rsi", 16, "Stochastic RSI", "momentum", "continuation",
          "%K crosses above %D from below the oversold level in an up-trend (close > EMA50)",
          "%K crosses below %D from above the overbought level in a down-trend (close < EMA50)",
          {"length": 14, "oversold": 20.0, "overbought": 80.0}, {"oversold": [15.0, 20.0, 30.0]})
def stoch_rsi(f, p, ctx):
    k, d = ind.stoch_rsi(f["close"], int(p["length"]))
    return (xup(k, d) & (k.shift(1) < p["oversold"]) & (f["close"] > f["ema50"]),
            xdn(k, d) & (k.shift(1) > p["overbought"]) & (f["close"] < f["ema50"]))


@strategy("macd_rsi", 17, "MACD + RSI confirmation", "momentum", "continuation",
          "MACD crosses above signal while RSI > rsi_mid and rising", "MACD crosses below signal while RSI < rsi_mid and falling",
          {"rsi_mid": 50.0, "rsi_bars": 2}, {"rsi_mid": [50.0, 55.0]})
def macd_rsi(f, p, ctx):
    r, n = f["rsi"], int(p["rsi_bars"])
    return (xup(f["macd"], f["macd_sig"]) & (r > p["rsi_mid"]) & (r > r.shift(n)),
            xdn(f["macd"], f["macd_sig"]) & (r < 100 - p["rsi_mid"]) & (r < r.shift(n)))


@strategy("rvol_breakout", 18, "Relative Volume breakout", "momentum", "breakout",
          "close crosses above the previous N-bar high on volume >= min_rvol x the 20-bar average",
          "close crosses below the previous N-bar low on volume >= min_rvol x the 20-bar average",
          {"length": 10, "min_rvol": 2.0}, {"min_rvol": [1.5, 2.0, 3.0]}, needs=("volume",))
def rvol_breakout(f, p, ctx):
    n, big = int(p["length"]), f["rvol"] >= p["min_rvol"]
    return big & xup(f["close"], prev_high(f, n)), big & xdn(f["close"], prev_low(f, n))


@strategy("pv_surge", 19, "Price-volume surge", "momentum", "continuation",
          "wide bullish bar (range >= range_atr x ATR, body >= 60%, close near the high) on volume >= min_rvol",
          "wide bearish bar (range >= range_atr x ATR, body >= 60%, close near the low) on volume >= min_rvol",
          {"range_atr": 1.5, "min_rvol": 2.5}, {"range_atr": [1.2, 1.5, 2.0], "min_rvol": [2.0, 2.5]}, needs=("volume",))
def pv_surge(f, p, ctx):
    rng = f["high"] - f["low"]
    wide = (rng >= p["range_atr"] * f["atr"].shift(1)) & (f["body_pct"] >= 0.6) & (f["rvol"] >= p["min_rvol"])
    return wide & _bull_candle(f, 0.75), wide & _bear_candle(f, 0.75)


@strategy("momentum_pullback", 20, "Momentum pullback", "momentum", "continuation",
          "EMA20 > rising EMA50, a pull-back touched EMA20 in the last 3 bars, RSI stayed above 45, close above the previous high",
          "EMA20 < falling EMA50, a pull-back touched EMA20 in the last 3 bars, RSI stayed below 55, close below the previous low",
          {"touch_bars": 3, "rsi_floor": 45.0}, {"touch_bars": [2, 3, 5]})
def momentum_pullback(f, p, ctx):
    n, c = int(p["touch_bars"]), f["close"]
    up_t = (f["ema20"] > f["ema50"]) & (f["ema50"] > f["ema50"].shift(5))
    dn_t = (f["ema20"] < f["ema50"]) & (f["ema50"] < f["ema50"].shift(5))
    touch_up = (f["low"] <= f["ema20"]).astype(float).shift(1).rolling(n, min_periods=1).max() == 1
    touch_dn = (f["high"] >= f["ema20"]).astype(float).shift(1).rolling(n, min_periods=1).max() == 1
    up = up_t & touch_up & (c > f["high"].shift(1)) & (c > f["ema20"]) & (f["rsi"] > p["rsi_floor"])
    dn = dn_t & touch_dn & (c < f["low"].shift(1)) & (c < f["ema20"]) & (f["rsi"] < 100 - p["rsi_floor"])
    return event(up), event(dn)


# =============================================================================  reversal (21-27)
@strategy("rsi_divergence", 21, "RSI divergence", "reversal", "reversal",
          "a confirmed swing low below the previous swing low while RSI at that low is higher (and below rsi_max), then a bullish bar",
          "a confirmed swing high above the previous swing high while RSI at that high is lower (and above 100 - rsi_max), then a bearish bar",
          {"rsi_max": 45.0, "window": 3}, {"rsi_max": [40.0, 45.0, 50.0]})
def rsi_divergence(f, p, ctx):
    right = 3                                                   # indicators.swings(..., right=3)
    r_at = f["rsi"].shift(right)
    rl, rh = r_at.where(f["pl"].notna()), r_at.where(f["ph"].notna())
    prev_rl = rl.dropna().shift(1).reindex(f.index)
    prev_rh = rh.dropna().shift(1).reindex(f.index)
    prev_pl = f["pl"].dropna().shift(1).reindex(f.index)
    prev_ph = f["ph"].dropna().shift(1).reindex(f.index)
    bull = (f["pl"] < prev_pl) & (rl > prev_rl) & (rl < p["rsi_max"])
    bear = (f["ph"] > prev_ph) & (rh < prev_rh) & (rh > 100 - p["rsi_max"])
    w = int(p["window"])
    live_bull = bull.astype(float).rolling(w, min_periods=1).max() == 1
    live_bear = bear.astype(float).rolling(w, min_periods=1).max() == 1
    return event(live_bull & _bull_candle(f, 0.55)), event(live_bear & _bear_candle(f, 0.55))


@strategy("bb_reversal", 22, "Bollinger Band reversal", "reversal", "reversal",
          "the previous close was below the lower band, this close is back inside, RSI was below rsi_low, ADX <= adx_max (range)",
          "the previous close was above the upper band, this close is back inside, RSI was above 100 - rsi_low, ADX <= adx_max",
          {"rsi_low": 35.0, "adx_max": 28.0}, {"rsi_low": [30.0, 35.0], "adx_max": [22.0, 28.0]},
          exit="the middle band (20-SMA) is the first objective; " + COMMON_EXIT)
def bb_reversal(f, p, ctx):
    c, quiet = f["close"], f["adx"] <= p["adx_max"]
    up = (c.shift(1) < f["bb_low"].shift(1)) & (c > f["bb_low"]) & (f["rsi"].shift(1) < p["rsi_low"]) & quiet
    dn = (c.shift(1) > f["bb_up"].shift(1)) & (c < f["bb_up"]) & (f["rsi"].shift(1) > 100 - p["rsi_low"]) & quiet
    return up & (c > f["open"]), dn & (c < f["open"])


@strategy("vwap_rejection", 23, "VWAP rejection", "reversal", "reversal",
          "price was above VWAP, dips to it (low within tol_atr x ATR) and closes back above with a bullish rejection bar",
          "price was below VWAP, rallies to it and closes back below with a bearish rejection bar",
          {"tol_atr": 0.15, "hold_bars": 4}, {"tol_atr": [0.1, 0.15, 0.3]}, needs=("volume",),
          exit="a close through VWAP; " + COMMON_EXIT)
def vwap_rejection(f, p, ctx):
    v, k, tol = f["vwap"], int(p["hold_bars"]), p["tol_atr"] * f["atr"]
    was_above = (f["close"] > v).astype(float).shift(1).rolling(k, min_periods=k).min() == 1
    was_below = (f["close"] < v).astype(float).shift(1).rolling(k, min_periods=k).min() == 1
    up = was_above & (f["low"] <= v + tol) & (f["close"] > v) & _bull_candle(f)
    dn = was_below & (f["high"] >= v - tol) & (f["close"] < v) & _bear_candle(f)
    return up, dn


def _reject(f, levels_lo, levels_hi, tol):
    up = pd.Series(False, index=f.index)
    dn = pd.Series(False, index=f.index)
    for lv in levels_lo:
        up |= (f["low"] <= lv + tol) & (f["close"] > lv) & (f["open"] > lv - tol)
    for lv in levels_hi:
        dn |= (f["high"] >= lv - tol) & (f["close"] < lv) & (f["open"] < lv + tol)
    return up & _bull_candle(f), dn & _bear_candle(f)


@strategy("sr_reversal", 24, "Support/resistance reversal", "reversal", "reversal",
          "a bullish rejection bar at support: the last two confirmed swing lows or the previous day's low",
          "a bearish rejection bar at resistance: the last two confirmed swing highs or the previous day's high",
          {"tol_atr": 0.2}, {"tol_atr": [0.1, 0.2, 0.35]},
          exit="a close through the level that was defended; " + COMMON_EXIT)
def sr_reversal(f, p, ctx):
    return _reject(f, (f["swing_lo"], f["prev_swing_lo"], f["pd_low"]),
                   (f["swing_hi"], f["prev_swing_hi"], f["pd_high"]), p["tol_atr"] * f["atr"])


@strategy("fibonacci", 25, "Fibonacci retracement", "reversal", "reversal",
          "up leg (swing low then swing high, >= min_leg_atr ATR): pull-back into the 38.2-61.8% zone, then a close above the previous high",
          "down leg: rally into the 38.2-61.8% zone, then a close below the previous low",
          {"min_leg_atr": 3.0, "shallow": 0.382, "deep": 0.618}, {"min_leg_atr": [2.0, 3.0, 4.0]},
          exit="a close beyond the 78.6% retracement; " + COMMON_EXIT)
def fibonacci(f, p, ctx):
    hi, lo, c = f["swing_hi"], f["swing_lo"], f["close"]
    leg = hi - lo
    ok = leg >= p["min_leg_atr"] * f["atr"]
    up_leg, dn_leg = ok & (f["hi_pos"] > f["lo_pos"]), ok & (f["lo_pos"] > f["hi_pos"])
    in_up = (f["low"] <= hi - p["shallow"] * leg) & (c >= hi - p["deep"] * leg)
    in_dn = (f["high"] >= lo + p["shallow"] * leg) & (c <= lo + p["deep"] * leg)
    recent_up = in_up.astype(float).rolling(3, min_periods=1).max() == 1
    recent_dn = in_dn.astype(float).rolling(3, min_periods=1).max() == 1
    return (event(up_leg & recent_up & (c > f["high"].shift(1)) & (c > f["open"])),
            event(dn_leg & recent_dn & (c < f["low"].shift(1)) & (c < f["open"])))


@strategy("pivot_reversal", 26, "Pivot Point reversal", "reversal", "reversal",
          "a bullish rejection bar at S1, S2 or the daily pivot (from above)", "a bearish rejection bar at R1, R2 or the daily pivot (from below)",
          {"tol_atr": 0.2}, {"tol_atr": [0.1, 0.2, 0.35]},
          exit="next pivot level is the first objective; " + COMMON_EXIT)
def pivot_reversal(f, p, ctx):
    return _reject(f, (f["s1"], f["s2"], f["pp"]), (f["r1"], f["r2"], f["pp"]), p["tol_atr"] * f["atr"])


@strategy("supertrend_reversal", 27, "Supertrend reversal", "reversal", "reversal",
          "Supertrend flips up after at least min_bars bars of down-trend during which RSI reached rsi_extreme or lower",
          "Supertrend flips down after at least min_bars bars of up-trend during which RSI reached 100 - rsi_extreme or higher",
          {"length": 10, "mult": 2.0, "min_bars": 12, "rsi_extreme": 32.0}, {"mult": [2.0, 3.0], "min_bars": [8, 12, 20]},
          exit="Supertrend flips back; " + COMMON_EXIT)
def supertrend_reversal(f, p, ctx):
    _, d = ind.supertrend(f, int(p["length"]), float(p["mult"]))
    n = int(p["min_bars"])
    long_down = (d.shift(1) == -1).astype(float).rolling(n, min_periods=n).min() == 1
    long_up = (d.shift(1) == 1).astype(float).rolling(n, min_periods=n).min() == 1
    was_low = f["rsi"].shift(1).rolling(n, min_periods=n).min() <= p["rsi_extreme"]
    was_high = f["rsi"].shift(1).rolling(n, min_periods=n).max() >= 100 - p["rsi_extreme"]
    return (d == 1) & long_down.shift(0) & was_low, (d == -1) & long_up & was_high


# =============================================================================  confluence (28-33)
@strategy("mtf_confluence", 28, "Multi-timeframe confluence", "confluence", "continuation",
          "15m, 60m (and daily, when known) trends are up; on the base timeframe the close crosses above EMA20 with MACD histogram > 0",
          "15m, 60m (and daily) trends are down; the close crosses below EMA20 with MACD histogram < 0",
          {"allow_flat_daily": 1}, {}, needs=("htf",))
def mtf_confluence(f, p, ctx):
    a, b = ctx.htf.get(15), ctx.htf.get(60)
    if a is None or b is None:
        return _none(f)
    d = ctx.htf.get("D")
    d = pd.Series(0.0, index=f.index) if d is None else d.fillna(0.0)
    flat = int(p["allow_flat_daily"]) == 1
    up_ok = (a == 1) & (b == 1) & ((d == 1) | (flat & (d == 0)))
    dn_ok = (a == -1) & (b == -1) & ((d == -1) | (flat & (d == 0)))
    return (up_ok & xup(f["close"], f["ema20"]) & (f["macd_hist"] > 0),
            dn_ok & xdn(f["close"], f["ema20"]) & (f["macd_hist"] < 0))


@strategy("vp_vwap", 29, "Volume Profile + VWAP", "confluence", "breakout",
          "close crosses above the previous session's value-area high while above VWAP (acceptance above value)",
          "close crosses below the previous session's value-area low while below VWAP",
          {"min_bar": 2}, {}, needs=("volume",),
          exit="a close back inside the previous value area; " + COMMON_EXIT)
def vp_vwap(f, p, ctx):
    late = f["bar_no"] >= int(p["min_bar"])
    return (late & xup(f["close"], f["vah"]) & (f["close"] > f["vwap"]),
            late & xdn(f["close"], f["val"]) & (f["close"] < f["vwap"]))


@strategy("bos_choch", 30, "BOS / CHoCH market structure", "confluence", "breakout",
          "close breaks the last confirmed swing high: BOS in an up-structure (HH + HL), CHoCH after a down-structure",
          "close breaks the last confirmed swing low: BOS in a down-structure (LH + LL), CHoCH after an up-structure",
          {"allow_choch": 1, "min_break_atr": 0.1}, {"min_break_atr": [0.0, 0.1, 0.25]},
          exit="a close back through the broken swing; " + COMMON_EXIT)
def bos_choch(f, p, ctx):
    buf = p["min_break_atr"] * f["atr"]
    choch = int(p["allow_choch"]) == 1
    s = f["structure"]
    up = xup(f["close"], f["swing_hi"] + buf) & ((s == 1) | (choch & (s == -1)))
    dn = xdn(f["close"], f["swing_lo"] - buf) & ((s == -1) | (choch & (s == 1)))
    return up, dn


@strategy("atr_breakout", 31, "ATR volatility breakout", "confluence", "breakout",
          "first close above the day's open + k x daily ATR (14, from prior sessions)",
          "first close below the day's open - k x daily ATR",
          {"k": 0.5}, {"k": [0.4, 0.5, 0.7]},
          exit="a close back through the day's open; " + COMMON_EXIT)
def atr_breakout(f, p, ctx):
    d = p["k"] * f["d_atr"]
    return first_of_day(f["close"] > f["day_open"] + d), first_of_day(f["close"] < f["day_open"] - d)


@strategy("gap", 32, "Gap-up / gap-down trading", "confluence", "breakout",
          "gap up of min_gap_pct..max_gap_pct: first close above the first-15-minute high (gap and go)",
          "gap down of the same size: first close below the first-15-minute low",
          {"min_gap_pct": 0.4, "max_gap_pct": 3.0}, {"min_gap_pct": [0.3, 0.4, 0.7]},
          exit="the gap fills (price back to the previous close); " + COMMON_EXIT)
def gap(f, p, ctx):
    hi, lo = ind.opening_range(f, 15, ctx.tf)
    g = f["gap_pct"]
    return (first_of_day((g >= p["min_gap_pct"]) & (g <= p["max_gap_pct"]) & (f["close"] > hi)),
            first_of_day((g <= -p["min_gap_pct"]) & (g >= -p["max_gap_pct"]) & (f["close"] < lo)))


@strategy("relative_strength", 33, "Relative strength against NIFTY", "confluence", "continuation",
          "today's % change beats NIFTY's by min_edge_pct, NIFTY is not weak, and the close makes a new session high",
          "today's % change lags NIFTY's by min_edge_pct, NIFTY is not strong, and the close makes a new session low",
          {"min_edge_pct": 0.6, "bench_floor_pct": -0.15, "min_bar": 6}, {"min_edge_pct": [0.4, 0.6, 1.0]},
          needs=("benchmark",))
def relative_strength(f, p, ctx):
    if ctx.bench_ret is None:
        return _none(f)
    edge = f["day_ret_pct"] - ctx.bench_ret
    late = f["bar_no"] >= int(p["min_bar"])
    up = late & (edge >= p["min_edge_pct"]) & (ctx.bench_ret >= p["bench_floor_pct"]) & (f["close"] > f["day_hi_prev"])
    dn = late & (edge <= -p["min_edge_pct"]) & (ctx.bench_ret <= -p["bench_floor_pct"]) & (f["close"] < f["day_lo_prev"])
    return event(up), event(dn)


# =============================================================================  options (34-35)
@strategy("premium_momentum", 34, "Options premium momentum", "options", "continuation",
          "ATM CE premium up >= rise_pct and ATM PE premium down >= fall_pct over the chain look-back, underlying above EMA20",
          "ATM PE premium up >= rise_pct and ATM CE premium down >= fall_pct, underlying below EMA20",
          {"rise_pct": 8.0, "fall_pct": 4.0}, {"rise_pct": [6.0, 8.0, 12.0]}, needs=("chain",))
def premium_momentum(f, p, ctx):
    ch = ctx.chain
    if ch is None or "ce_mom_pct" not in ch:
        return _none(f)
    up = (ch["ce_mom_pct"] >= p["rise_pct"]) & (ch["pe_mom_pct"] <= -p["fall_pct"]) & (f["close"] > f["ema20"])
    dn = (ch["pe_mom_pct"] >= p["rise_pct"]) & (ch["ce_mom_pct"] <= -p["fall_pct"]) & (f["close"] < f["ema20"])
    return event(up), event(dn)


@strategy("oi_price_action", 35, "Option chain OI + price action", "options", "breakout",
          "put OI near the money builds (>= build_pct) while call OI does not (<= call_cap_pct) and the close breaks the previous N-bar high",
          "call OI near the money builds while put OI does not and the close breaks the previous N-bar low",
          {"build_pct": 3.0, "cap_pct": 1.0, "length": 6}, {"build_pct": [2.0, 3.0, 5.0]}, needs=("chain",))
def oi_price_action(f, p, ctx):
    ch = ctx.chain
    if ch is None or "pe_oi_chg_pct" not in ch:
        return _none(f)
    n = int(p["length"])
    put_build = (ch["pe_oi_chg_pct"] >= p["build_pct"]) & (ch["ce_oi_chg_pct"] <= p["cap_pct"])
    call_build = (ch["ce_oi_chg_pct"] >= p["build_pct"]) & (ch["pe_oi_chg_pct"] <= p["cap_pct"])
    return put_build & xup(f["close"], prev_high(f, n)), call_build & xdn(f["close"], prev_low(f, n))


# =============================================================================  requested combination (36)
@strategy("st_rsi_ema", 36, "Supertrend + RSI + EMA confirmation", "confluence", "continuation",
          "Supertrend turns bullish and RSI crosses above rsi_bull (both within `window` bars; the signal is the bar that "
          "completes the pair), with EMA9 > EMA21, close above VWAP, volume >= min_rvol x average, 15-minute trend not "
          "bearish, and the market not sideways (ADX >= adx_min, Bollinger Bands outside the Keltner channel)",
          "Supertrend turns bearish and RSI crosses below rsi_bear (within `window` bars), with EMA9 < EMA21, close below "
          "VWAP, volume >= min_rvol x average, 15-minute trend not bullish, and the market not sideways",
          {"length": 10, "mult": 3.0, "rsi_bull": 55.0, "rsi_bear": 45.0, "window": 5, "min_rvol": 1.2, "adx_min": 18.0,
           "strict_htf": 0},
          {"mult": [2.0, 3.0], "window": [2, 3, 5], "adx_min": [15.0, 18.0, 22.0]}, needs=("volume", "htf15"),
          exit="Supertrend flips back; " + COMMON_EXIT)
def st_rsi_ema(f, p, ctx):
    h15 = ctx.htf.get(15)
    if h15 is None:
        return _none(f)
    w = int(p["window"])
    _, d = ind.supertrend(f, int(p["length"]), float(p["mult"]))

    def recent(x):
        return x.astype(float).rolling(w, min_periods=1).max() == 1
    flip_up, flip_dn = (d == 1) & (d.shift(1) == -1), (d == -1) & (d.shift(1) == 1)
    rsi_up, rsi_dn = xup(f["rsi"], p["rsi_bull"]), xdn(f["rsi"], p["rsi_bear"])
    e9, e21 = ema(f["close"], 9), ema(f["close"], 21)
    trending = (f["adx"] >= p["adx_min"]) & ~((f["bb_up"] < f["kc_up"]) & (f["bb_low"] > f["kc_low"]))   # sideways filter
    volume = recent(f["rvol"] >= p["min_rvol"])               # the surge may come on either bar of the pair
    strict = int(p["strict_htf"]) == 1
    up15 = (h15 == 1) if strict else (h15 >= 0)
    dn15 = (h15 == -1) if strict else (h15 <= 0)
    up = recent(flip_up) & recent(rsi_up) & (d == 1) & (f["rsi"] > 50) & (e9 > e21) & (f["close"] > f["vwap"]) & volume & trending & up15
    dn = recent(flip_dn) & recent(rsi_dn) & (d == -1) & (f["rsi"] < 50) & (e9 < e21) & (f["close"] < f["vwap"]) & volume & trending & dn15
    return event(up), event(dn)


# =============================================================================  option chain (37)
@strategy("option_chain", 37, "Option chain confluence", "options", "breakout",
          "bullish chain evidence >= min_score with bearish <= max_opposite (OI change read together with price: call "
          "buying, put writing, short covering; premiums; PCR change; IV change; volume), AND the underlying closes above "
          "the previous N-bar high, above EMA20 and VWAP, above the put-OI support, with room_atr x ATR before the call-OI "
          "wall. ATM spread must be <= max_atm_spread_pct. No signal on expiry day unless expiry_day = 1",
          "bearish chain evidence >= min_score with bullish <= max_opposite, AND the underlying closes below the previous "
          "N-bar low, below EMA20 and VWAP, below the call-OI resistance, with room before the put-OI wall",
          {"min_score": 55.0, "max_opposite": 35.0, "length": 6, "max_atm_spread_pct": 1.5, "room_atr": 0.5, "expiry_day": 0},
          {"min_score": [50.0, 55.0, 65.0], "length": [4, 6, 10]}, needs=("chain",),
          exit="a close back through the OI wall that supported the trade; " + COMMON_EXIT)
def option_chain(f, p, ctx):
    ch = ctx.chain
    if ch is None or "bull_score" not in ch:
        return _none(f)
    c, n, v = f["close"], int(p["length"]), f["vwap"]
    liquid = ch["atm_spread_pct"].isna() | (ch["atm_spread_pct"] <= p["max_atm_spread_pct"])
    expiry_ok = (ch["dte"] != 0) | (int(p["expiry_day"]) == 1)
    room = p["room_atr"] * f["atr"]
    sup, res = ch["support"], ch["resistance"]
    pa_up = (c > prev_high(f, n)) & (c > f["ema20"]) & (v.isna() | (c > v))
    pa_dn = (c < prev_low(f, n)) & (c < f["ema20"]) & (v.isna() | (c < v))
    up = (ch["bull_score"] >= p["min_score"]) & (ch["bear_score"] <= p["max_opposite"]) & pa_up & liquid & expiry_ok & \
        (sup.isna() | (c > sup)) & (res.isna() | (res - c >= room) | (c > res))
    dn = (ch["bear_score"] >= p["min_score"]) & (ch["bull_score"] <= p["max_opposite"]) & pa_dn & liquid & expiry_ok & \
        (res.isna() | (c < res)) & (sup.isna() | (c - sup >= room) | (c < sup))
    return event(up), event(dn)


# =============================================================================  stock scanner (38)
SCAN_FACTORS = ("EMA", "SUPERTREND", "RSI", "MACD", "VWAP", "RVOL", "ATR", "BREAKOUT", "S/R", "REL STRENGTH")


def scan_factors(f: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Ten factors per bar, each +1 (bullish), -1 (bearish), 0 (neutral) or NaN (no data), and the two scores
    bull / bear = share of the factors WITH data that point that way (0-100)."""
    c = f["close"]
    x = pd.DataFrame(index=f.index)
    sign = lambda up, dn: pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)), index=f.index)      # noqa: E731
    x["EMA"] = sign((f["ema9"] > f["ema21"]) & (f["ema20"] > f["ema50"]), (f["ema9"] < f["ema21"]) & (f["ema20"] < f["ema50"])).where(f["ema50"].notna())
    x["SUPERTREND"] = f["st_dir"]
    x["RSI"] = sign(f["rsi"] > 55, f["rsi"] < 45).where(f["rsi"].notna())
    rising = f["macd_hist"] > f["macd_hist"].shift(1)
    x["MACD"] = sign((f["macd"] > f["macd_sig"]) & rising, (f["macd"] < f["macd_sig"]) & ~rising).where(f["macd_sig"].notna())
    x["VWAP"] = sign(c > f["vwap"], c < f["vwap"]).where(f["vwap"].notna())
    busy = f["rvol"] >= 1.5
    x["RVOL"] = sign(busy & (c > f["open"]), busy & (c < f["open"])).where(f["rvol"].notna())
    expanding = f["atr"] > 1.1 * f["atr"].rolling(20, min_periods=10).mean()
    x["ATR"] = sign(expanding & (c > c.shift(3)), expanding & (c < c.shift(3))).where(f["atr"].notna())
    x["BREAKOUT"] = sign((c > prev_high(f, 20)) | (c > f["day_hi_prev"]), (c < prev_low(f, 20)) | (c < f["day_lo_prev"]))
    x["S/R"] = sign((c > f["pd_high"]) | (c > f["swing_hi"]), (c < f["pd_low"]) | (c < f["swing_lo"])).where(f["pd_high"].notna())
    if ctx.bench_ret is not None:
        edge = f["day_ret_pct"] - ctx.bench_ret
        x["REL STRENGTH"] = sign(edge >= 0.5, edge <= -0.5).where(edge.notna())
    else:
        x["REL STRENGTH"] = np.nan
    known = x[list(SCAN_FACTORS)].notna().sum(axis=1).replace(0, np.nan)
    x["bull"] = (x[list(SCAN_FACTORS)] == 1).sum(axis=1) / known * 100.0
    x["bear"] = (x[list(SCAN_FACTORS)] == -1).sum(axis=1) / known * 100.0
    return x


@strategy("multi_factor", 38, "Multi-factor scanner", "confluence", "continuation",
          "the bullish score (EMA, Supertrend, RSI, MACD, VWAP, relative volume, ATR expansion, breakout, support/resistance, "
          "relative strength vs NIFTY) rises through min_score while the bearish score is <= max_opposite",
          "the bearish score rises through min_score while the bullish score is <= max_opposite",
          {"min_score": 70.0, "max_opposite": 20.0}, {"min_score": [60.0, 70.0, 80.0]}, needs=("volume", "benchmark"))
def multi_factor(f, p, ctx):
    x = scan_factors(f, ctx)
    return (xup(x["bull"], p["min_score"]) & (x["bear"] <= p["max_opposite"]),
            xup(x["bear"], p["min_score"]) & (x["bull"] <= p["max_opposite"]))


# =============================================================================  added set (39-43)
@strategy("vwap_breakout", 39, "VWAP breakout", "breakout", "breakout",
          "price closed below VWAP for hold_bars bars, then a bullish bar closes above VWAP on volume >= min_rvol x average",
          "price closed above VWAP for hold_bars bars, then a bearish bar closes below VWAP on volume >= min_rvol x average",
          {"hold_bars": 4, "min_rvol": 1.3, "min_bar": 3}, {"hold_bars": [3, 4, 6], "min_rvol": [1.0, 1.3, 1.8]}, needs=("volume",),
          exit="a close back through VWAP; " + COMMON_EXIT)
def vwap_breakout(f, p, ctx):
    v, c, k = f["vwap"], f["close"], int(p["hold_bars"])
    was_below = (c < v).astype(float).shift(1).rolling(k, min_periods=k).min() == 1
    was_above = (c > v).astype(float).shift(1).rolling(k, min_periods=k).min() == 1
    ok = (f["rvol"] >= p["min_rvol"]) & (f["bar_no"] >= int(p["min_bar"]))
    return was_below & (c > v) & ok & _bull_candle(f), was_above & (c < v) & ok & _bear_candle(f)


@strategy("pivot_breakout", 40, "Pivot point breakout", "breakout", "breakout",
          "close crosses above R1 or R2 (+ buffer_atr x ATR) with a bullish bar, after min_bar bars of the session",
          "close crosses below S1 or S2 (- buffer_atr x ATR) with a bearish bar, after min_bar bars of the session",
          {"buffer_atr": 0.1, "min_bar": 2}, {"buffer_atr": [0.0, 0.1, 0.25]},
          exit="a close back through the broken pivot level (next pivot level is the first objective); " + COMMON_EXIT)
def pivot_breakout(f, p, ctx):
    c, buf, late = f["close"], float(p["buffer_atr"]) * f["atr"].fillna(0.0), f["bar_no"] >= int(p["min_bar"])
    up = (xup(c, f["r1"] + buf) | xup(c, f["r2"] + buf)) & late & (c > f["open"])
    dn = (xdn(c, f["s1"] - buf) | xdn(c, f["s2"] - buf)) & late & (c < f["open"])
    return up, dn


@strategy("supertrend_macd", 41, "Supertrend + MACD", "trend", "continuation",
          "Supertrend turns up and MACD crosses above its signal line (both within `window` bars; the signal is the bar "
          "that completes the pair), Supertrend still up and the MACD histogram positive",
          "Supertrend turns down and MACD crosses below its signal line (within `window` bars), Supertrend still down and "
          "the MACD histogram negative",
          {"length": 10, "mult": 3.0, "window": 5}, {"mult": [2.0, 3.0], "window": [3, 5, 8]},
          exit="Supertrend flips back or MACD crosses back; " + COMMON_EXIT)
def supertrend_macd(f, p, ctx):
    w = int(p["window"])
    _, d = ind.supertrend(f, int(p["length"]), float(p["mult"]))

    def recent(x):
        return x.astype(float).rolling(w, min_periods=1).max() == 1
    flip_up, flip_dn = (d == 1) & (d.shift(1) == -1), (d == -1) & (d.shift(1) == 1)
    up = recent(flip_up) & recent(xup(f["macd"], f["macd_sig"])) & (d == 1) & (f["macd_hist"] > 0)
    dn = recent(flip_dn) & recent(xdn(f["macd"], f["macd_sig"])) & (d == -1) & (f["macd_hist"] < 0)
    return event(up), event(dn)


@strategy("vwap_rsi", 42, "VWAP + RSI", "momentum", "continuation",
          "RSI rises through rsi_bull while the close is above a VWAP that is not falling (over slope_bars bars)",
          "RSI falls through rsi_bear while the close is below a VWAP that is not rising",
          {"rsi_bull": 55.0, "rsi_bear": 45.0, "slope_bars": 3}, {"rsi_bull": [55.0, 60.0], "rsi_bear": [40.0, 45.0]},
          needs=("volume",), exit="a close on the other side of VWAP; " + COMMON_EXIT)
def vwap_rsi(f, p, ctx):
    v, c, n = f["vwap"], f["close"], int(p["slope_bars"])
    return (xup(f["rsi"], p["rsi_bull"]) & (c > v) & (v >= v.shift(n)),
            xdn(f["rsi"], p["rsi_bear"]) & (c < v) & (v <= v.shift(n)))


@strategy("vol_oi_divergence", 43, "Option volume / OI divergence", "options", "breakout",
          "calls trade heavily (volume >= min_turnover_pct of call OI over the chain look-back) while call OI FALLS by "
          "drop_pct or more and the ATM call premium rises: call writers are covering, not new sellers arriving. The "
          "underlying must close above the previous N-bar high",
          "puts trade heavily while put OI FALLS and the ATM put premium rises: put writers are covering. The underlying "
          "must close below the previous N-bar low",
          {"min_turnover_pct": 2.0, "drop_pct": 1.0, "length": 6}, {"min_turnover_pct": [1.0, 2.0, 4.0], "drop_pct": [0.5, 1.0, 2.0]},
          needs=("chain",))
def vol_oi_divergence(f, p, ctx):
    ch = ctx.chain
    if ch is None or "ce_vol_pct" not in ch:
        return _none(f)
    n = int(p["length"])
    ce_cover = (ch["ce_vol_pct"] >= p["min_turnover_pct"]) & (ch["ce_oi_chg_pct"] <= -p["drop_pct"]) & (ch["ce_mom_pct"] > 0)
    pe_cover = (ch["pe_vol_pct"] >= p["min_turnover_pct"]) & (ch["pe_oi_chg_pct"] <= -p["drop_pct"]) & (ch["pe_mom_pct"] > 0)
    return event(ce_cover & (f["close"] > prev_high(f, n))), event(pe_cover & (f["close"] < prev_low(f, n)))


assert len(REGISTRY) == 43 and sorted(s.number for s in REGISTRY.values()) == list(range(1, 44))
