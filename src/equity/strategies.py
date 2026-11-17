"""
The 25 intraday and 25 swing strategies.

Every strategy is one small vectorised function over an enriched frame (src/equity/indicators.py):

    fn(f) -> Sig(long, short, stop_long, stop_short, why_long, why_short, cols)

`long` / `short` are boolean Series, True on the COMPLETED bar that triggers the setup. `stop_*` is the price the
setup is wrong at (the planner clamps its distance by ATR), `why_*` is a sentence with {column} placeholders that is
filled from the trigger bar, and `cols` are the indicator values shown with the signal. Because the functions see
only causal columns, the backtest (whole history) and the live engine (last completed bar) get identical answers.

    python -m src.equity.strategies        list every strategy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from src.fno.indicators import event, first_of_day, xdn, xup


class Sig(NamedTuple):
    long: Optional[pd.Series] = None
    short: Optional[pd.Series] = None
    stop_long: Optional[pd.Series] = None
    stop_short: Optional[pd.Series] = None
    why_long: str = ""
    why_short: str = ""
    cols: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    module: str          # intraday | swing
    kind: str            # breakout | trend | momentum | reversal | pattern | volume
    fn: Callable[[pd.DataFrame], Sig]
    about: str

    def run(self, f: pd.DataFrame) -> Sig:
        s = self.fn(f)
        false = pd.Series(False, index=f.index)
        fix = (lambda m: false if m is None else m.fillna(False).astype(bool))
        return s._replace(long=fix(s.long), short=fix(s.short))

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "module": self.module, "kind": self.kind, "about": self.about}


INTRADAY: Dict[str, Strategy] = {}
SWING: Dict[str, Strategy] = {}


def _reg(table, module):
    def deco(sid, name, kind, about):
        def wrap(fn):
            table[sid] = Strategy(sid, name, module, kind, fn, about)
            return fn
        return wrap
    return deco


intraday = _reg(INTRADAY, "intraday")
swing = _reg(SWING, "swing")


def _bull(f):
    return f["close"] > f["open"]


def _bear(f):
    return f["close"] < f["open"]


def _engulf(f, up: bool) -> pd.Series:
    o, c, po, pc = f["open"], f["close"], f["open"].shift(1), f["close"].shift(1)
    if up:
        return (pc < po) & (c > o) & (c >= po) & (o <= pc) & ((c - o) > (po - pc))
    return (pc > po) & (c < o) & (c <= po) & (o >= pc) & ((o - c) > (pc - po))


# =============================================================================
# Intraday (5-minute bars)
# =============================================================================
@intraday("vwap_breakout", "VWAP Breakout", "breakout", "5m close crosses VWAP with above-average volume, in the direction of the 20 EMA.")
def _i_vwap_breakout(f):
    c, vw = f["close"], f["vwap"]
    return Sig(xup(c, vw) & _bull(f) & (f["rvol"] >= 1.3) & (c > f["ema20"]),
               xdn(c, vw) & _bear(f) & (f["rvol"] >= 1.3) & (c < f["ema20"]),
               vw - 0.3 * f["atr"], vw + 0.3 * f["atr"],
               "5m close {close:.2f} crossed above VWAP {vwap:.2f} on {rvol:.1f}x volume, above the 20 EMA",
               "5m close {close:.2f} crossed below VWAP {vwap:.2f} on {rvol:.1f}x volume, below the 20 EMA", ("vwap", "rvol", "ema20"))


@intraday("vwap_reversal", "VWAP Reversal", "reversal", "Price stretched 1.5+ VWAP deviations away, then a reversal bar back toward VWAP.")
def _i_vwap_reversal(f):
    c, vw, sd = f["close"], f["vwap"], f["vwap_sd"]
    lo1, hi1 = f["low"].shift(1), f["high"].shift(1)
    return Sig((lo1 < (vw - 1.5 * sd).shift(1)) & (c > hi1) & _bull(f) & (c < vw) & (f["rsi"].shift(1) < 38),
               (hi1 > (vw + 1.5 * sd).shift(1)) & (c < lo1) & _bear(f) & (c > vw) & (f["rsi"].shift(1) > 62),
               f["low"].rolling(3).min(), f["high"].rolling(3).max(),
               "stretched 1.5 sd below VWAP {vwap:.2f} with RSI {rsi:.0f}, bullish bar closed above the previous high",
               "stretched 1.5 sd above VWAP {vwap:.2f} with RSI {rsi:.0f}, bearish bar closed below the previous low", ("vwap", "vwap_sd", "rsi"))


@intraday("ema_9_21", "9 EMA / 21 EMA Crossover", "trend", "9 EMA crosses the 21 EMA on 5m with ADX above 18 and price on the right side of VWAP.")
def _i_ema_9_21(f):
    return Sig(xup(f["ema9"], f["ema21"]) & (f["close"] > f["vwap"]) & (f["adx"] > 18),
               xdn(f["ema9"], f["ema21"]) & (f["close"] < f["vwap"]) & (f["adx"] > 18),
               f["ema21"] - 0.5 * f["atr"], f["ema21"] + 0.5 * f["atr"],
               "9 EMA {ema9:.2f} crossed above 21 EMA {ema21:.2f}, price above VWAP, ADX {adx:.0f}",
               "9 EMA {ema9:.2f} crossed below 21 EMA {ema21:.2f}, price below VWAP, ADX {adx:.0f}", ("ema9", "ema21", "adx", "vwap"))


@intraday("ema20_momentum", "20 EMA Momentum", "momentum", "Rising 20 EMA, price above it and a new 6-bar high with RSI 55-75.")
def _i_ema20_momentum(f):
    c, e = f["close"], f["ema20"]
    up = (c > e) & (e > e.shift(3)) & (c > f["hh6"]) & f["rsi"].between(55, 75)
    dn = (c < e) & (e < e.shift(3)) & (c < f["ll6"]) & f["rsi"].between(25, 45)
    return Sig(event(up), event(dn), e - 0.3 * f["atr"], e + 0.3 * f["atr"],
               "rising 20 EMA {ema20:.2f}, close {close:.2f} at a new 6-bar high, RSI {rsi:.0f}",
               "falling 20 EMA {ema20:.2f}, close {close:.2f} at a new 6-bar low, RSI {rsi:.0f}", ("ema20", "rsi"))


@intraday("orb", "Opening Range Breakout (ORB)", "breakout", "First 5m close beyond the opening range (first 30 minutes) with volume.")
def _i_orb(f):
    c = f["close"]
    early = f["bar_no"] <= 45                                            # before ~13:00
    return Sig(first_of_day((c > f["orb_hi"]) & (f["rvol"] >= 1.2) & early),
               first_of_day((c < f["orb_lo"]) & (f["rvol"] >= 1.2) & early),
               (f["orb_hi"] + f["orb_lo"]) / 2.0, (f["orb_hi"] + f["orb_lo"]) / 2.0,
               "5m close {close:.2f} above the opening-range high {orb_hi:.2f} on {rvol:.1f}x volume",
               "5m close {close:.2f} below the opening-range low {orb_lo:.2f} on {rvol:.1f}x volume", ("orb_hi", "orb_lo", "rvol"))


@intraday("pdh_breakout", "Previous Day High Breakout", "breakout", "5m close crosses above yesterday's high with volume (long only).")
def _i_pdh(f):
    return Sig(long=xup(f["close"], f["pdh"]) & (f["rvol"] >= 1.3) & _bull(f), stop_long=f["pdh"] - 0.6 * f["atr"],
               why_long="5m close {close:.2f} broke the previous day's high {pdh:.2f} on {rvol:.1f}x volume", cols=("pdh", "rvol"))


@intraday("pdl_breakdown", "Previous Day Low Breakdown", "breakout", "5m close crosses below yesterday's low with volume (short only).")
def _i_pdl(f):
    return Sig(short=xdn(f["close"], f["pdl"]) & (f["rvol"] >= 1.3) & _bear(f), stop_short=f["pdl"] + 0.6 * f["atr"],
               why_short="5m close {close:.2f} broke the previous day's low {pdl:.2f} on {rvol:.1f}x volume", cols=("pdl", "rvol"))


@intraday("first15_breakout", "First 15-Minute High/Low Breakout", "breakout", "First 5m close beyond the high / low of the first 15 minutes.")
def _i_first15(f):
    c = f["close"]
    return Sig(first_of_day((c > f["f15_hi"]) & (c > f["vwap"])), first_of_day((c < f["f15_lo"]) & (c < f["vwap"])),
               f["f15_lo"], f["f15_hi"],
               "5m close {close:.2f} above the first 15-minute high {f15_hi:.2f}",
               "5m close {close:.2f} below the first 15-minute low {f15_lo:.2f}", ("f15_hi", "f15_lo", "vwap"))


@intraday("rsi_momentum", "RSI Momentum", "momentum", "RSI(14) crosses 60 (40 for shorts) with price beyond the 20 EMA and VWAP.")
def _i_rsi(f):
    c = f["close"]
    return Sig(xup(f["rsi"], 60) & (c > f["ema20"]) & (c > f["vwap"]), xdn(f["rsi"], 40) & (c < f["ema20"]) & (c < f["vwap"]),
               why_long="RSI crossed above 60 ({rsi:.0f}) with price above the 20 EMA and VWAP",
               why_short="RSI crossed below 40 ({rsi:.0f}) with price below the 20 EMA and VWAP", cols=("rsi", "ema20", "vwap"))


@intraday("macd_cross", "MACD Crossover", "momentum", "MACD line crosses its signal line in the direction of the 50 EMA.")
def _i_macd(f):
    return Sig(xup(f["macd"], f["macd_sig"]) & (f["close"] > f["ema50"]), xdn(f["macd"], f["macd_sig"]) & (f["close"] < f["ema50"]),
               why_long="MACD {macd:.2f} crossed above its signal {macd_sig:.2f}, price above the 50 EMA",
               why_short="MACD {macd:.2f} crossed below its signal {macd_sig:.2f}, price below the 50 EMA", cols=("macd", "macd_sig", "ema50"))


@intraday("supertrend", "Supertrend Strategy", "trend", "Supertrend(10,3) flips direction with price on the same side of VWAP.")
def _i_supertrend(f):
    d = f["st_dir"]
    return Sig((d == 1) & (d.shift(1) == -1) & (f["close"] > f["vwap"]), (d == -1) & (d.shift(1) == 1) & (f["close"] < f["vwap"]),
               f["st_line"], f["st_line"],
               "Supertrend flipped up (line {st_line:.2f}), price above VWAP",
               "Supertrend flipped down (line {st_line:.2f}), price below VWAP", ("st_line", "vwap"))


@intraday("bb_breakout", "Bollinger Band Breakout", "breakout", "Close outside the 20,2 band while the bands widen and volume expands.")
def _i_bb(f):
    c, wide = f["close"], f["bb_width"] > f["bb_width"].shift(1)
    return Sig(xup(c, f["bb_up"]) & wide & (f["rvol"] >= 1.3), xdn(c, f["bb_lo"]) & wide & (f["rvol"] >= 1.3),
               f["bb_mid"], f["bb_mid"],
               "close {close:.2f} outside the upper band {bb_up:.2f}, bands widening, {rvol:.1f}x volume",
               "close {close:.2f} outside the lower band {bb_lo:.2f}, bands widening, {rvol:.1f}x volume", ("bb_up", "bb_lo", "rvol"))


@intraday("volume_breakout", "Volume Breakout", "volume", "Bar volume 2x its 20-bar average on a close beyond the last hour's range.")
def _i_volume(f):
    big = f["volume"] >= 2.0 * f["vol_ma"]
    return Sig(big & (f["close"] > f["hh12"]) & _bull(f), big & (f["close"] < f["ll12"]) & _bear(f),
               why_long="volume {rvol:.1f}x the 20-bar average on a close above the last hour's high {hh12:.2f}",
               why_short="volume {rvol:.1f}x the 20-bar average on a close below the last hour's low {ll12:.2f}", cols=("rvol", "hh12", "ll12"))


@intraday("rvol_spike", "Relative Volume Spike", "volume", "Volume 2.5x the same time of day over the last 10 sessions on a wide directional bar.")
def _i_rvol(f):
    rng = (f["high"] - f["low"]).replace(0.0, np.nan)
    pos = (f["close"] - f["low"]) / rng
    spike = (f["rvol_tod"] >= 2.5) & (rng > f["atr"])
    return Sig(spike & (pos >= 0.7) & (f["close"] > f["vwap"]), spike & (pos <= 0.3) & (f["close"] < f["vwap"]),
               why_long="{rvol_tod:.1f}x the usual volume for this time of day, wide bar closing near its high above VWAP",
               why_short="{rvol_tod:.1f}x the usual volume for this time of day, wide bar closing near its low below VWAP", cols=("rvol_tod", "vwap"))


@intraday("adx_trend", "ADX Trend Strength", "trend", "ADX crosses above 25 with +DI / -DI and the 20 EMA agreeing.")
def _i_adx(f):
    x = xup(f["adx"], 25)
    return Sig(x & (f["pdi"] > f["mdi"]) & (f["close"] > f["ema20"]), x & (f["mdi"] > f["pdi"]) & (f["close"] < f["ema20"]),
               why_long="ADX crossed above 25 ({adx:.0f}) with +DI {pdi:.0f} over -DI {mdi:.0f}",
               why_short="ADX crossed above 25 ({adx:.0f}) with -DI {mdi:.0f} over +DI {pdi:.0f}", cols=("adx", "pdi", "mdi"))


@intraday("sr_breakout", "Support and Resistance Breakout", "breakout", "Close through the latest confirmed swing high / low with volume.")
def _i_sr(f):
    hi, lo = f["swing_hi"].shift(1), f["swing_lo"].shift(1)
    return Sig(xup(f["close"], hi) & (f["rvol"] >= 1.2), xdn(f["close"], lo) & (f["rvol"] >= 1.2),
               hi - 0.6 * f["atr"], lo + 0.6 * f["atr"],
               "close {close:.2f} through swing resistance {swing_hi:.2f} on {rvol:.1f}x volume",
               "close {close:.2f} through swing support {swing_lo:.2f} on {rvol:.1f}x volume", ("swing_hi", "swing_lo", "rvol"))


@intraday("pivot_breakout", "Pivot Point Breakout", "breakout", "Close crosses the floor pivot R1 (long) or S1 (short).")
def _i_pivot(f):
    return Sig(xup(f["close"], f["r1"]) & (f["close"] > f["vwap"]), xdn(f["close"], f["s1"]) & (f["close"] < f["vwap"]),
               f["pivot"], f["pivot"],
               "close {close:.2f} crossed above R1 {r1:.2f} (pivot {pivot:.2f})",
               "close {close:.2f} crossed below S1 {s1:.2f} (pivot {pivot:.2f})", ("pivot", "r1", "s1"))


@intraday("cpr_breakout", "CPR Breakout", "breakout", "Close leaves the Central Pivot Range with price on the right side of VWAP.")
def _i_cpr(f):
    return Sig(xup(f["close"], f["cpr_top"]) & (f["close"] > f["vwap"]), xdn(f["close"], f["cpr_bot"]) & (f["close"] < f["vwap"]),
               f["cpr_bot"], f["cpr_top"],
               "close {close:.2f} broke above the CPR ({cpr_bot:.2f}-{cpr_top:.2f})",
               "close {close:.2f} broke below the CPR ({cpr_bot:.2f}-{cpr_top:.2f})", ("cpr_top", "cpr_bot", "vwap"))


@intraday("fib_retracement", "Fibonacci Retracement", "reversal", "Pullback into the 38.2-61.8% zone of the session's leg, then a bar closing back with the leg.")
def _i_fib(f):
    hi, lo = f["run_hi"].shift(1), f["run_lo"].shift(1)
    leg = hi - lo
    same_day = f["bar_no"] > 6
    up_leg = (f["hi_bar"].shift(1) > f["lo_bar"].shift(1)) & (leg > 2.0 * f["atr"]) & same_day
    dn_leg = (f["lo_bar"].shift(1) > f["hi_bar"].shift(1)) & (leg > 2.0 * f["atr"]) & same_day
    depth_up, depth_dn = (hi - f["low"]) / leg, (f["high"] - lo) / leg
    return Sig(up_leg & depth_up.between(0.382, 0.66) & _bull(f) & (f["close"] > f["high"].shift(1)),
               dn_leg & depth_dn.between(0.382, 0.66) & _bear(f) & (f["close"] < f["low"].shift(1)),
               hi - 0.786 * leg, lo + 0.786 * leg,
               "pulled back into the 38-62% zone of today's up-leg and closed above the previous bar's high",
               "bounced into the 38-62% zone of today's down-leg and closed below the previous bar's low", ("run_hi", "run_lo"))


@intraday("price_action", "Price Action Momentum", "momentum", "Three consecutive strong bars (higher highs, higher lows, bodies over half the range).")
def _i_price_action(f):
    body = (f["close"] - f["open"]).abs() >= 0.5 * (f["high"] - f["low"])
    up = _bull(f) & body & (f["high"] > f["high"].shift(1)) & (f["low"] > f["low"].shift(1))
    dn = _bear(f) & body & (f["low"] < f["low"].shift(1)) & (f["high"] < f["high"].shift(1))
    up3, dn3 = up & up.shift(1, fill_value=False) & up.shift(2, fill_value=False), dn & dn.shift(1, fill_value=False) & dn.shift(2, fill_value=False)
    return Sig(event(up3) & (f["close"] > f["vwap"]), event(dn3) & (f["close"] < f["vwap"]),
               f["low"].shift(1), f["high"].shift(1),
               "three strong bullish bars in a row above VWAP", "three strong bearish bars in a row below VWAP", ("vwap",))


@intraday("pullback", "Pullback Entry", "trend", "Trend intact (9 > 21 > 50 EMA), dip to the 21 EMA, then a close above the previous high.")
def _i_pullback(f):
    c, e9, e21, e50 = f["close"], f["ema9"], f["ema21"], f["ema50"]
    touched_up = (f["low"].shift(1) <= e21.shift(1) * 1.001) | (f["low"] <= e21 * 1.001)
    touched_dn = (f["high"].shift(1) >= e21.shift(1) * 0.999) | (f["high"] >= e21 * 0.999)
    return Sig((e9 > e21) & (e21 > e50) & (c > f["vwap"]) & touched_up & (c > f["high"].shift(1)) & (c > e9),
               (e9 < e21) & (e21 < e50) & (c < f["vwap"]) & touched_dn & (c < f["low"].shift(1)) & (c < e9),
               f["low"].rolling(3).min(), f["high"].rolling(3).max(),
               "up-trend (9>21>50 EMA) pulled back to the 21 EMA {ema21:.2f} and closed above the previous high",
               "down-trend (9<21<50 EMA) rallied to the 21 EMA {ema21:.2f} and closed below the previous low", ("ema9", "ema21", "ema50"))


@intraday("inside_bar", "Inside Bar Breakout", "pattern", "Close beyond an inside bar's range in the direction of the 20 EMA.")
def _i_inside(f):
    h, l = f["high"], f["low"]
    inside = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
    return Sig(inside & (f["close"] > h.shift(1)) & (f["close"] > f["ema20"]), inside & (f["close"] < l.shift(1)) & (f["close"] < f["ema20"]),
               l.shift(1), h.shift(1),
               "close {close:.2f} above the inside bar's high, above the 20 EMA",
               "close {close:.2f} below the inside bar's low, below the 20 EMA", ("ema20",))


@intraday("bullish_engulfing", "Bullish Engulfing", "pattern", "Bullish engulfing bar after a dip (RSI under 45 or price under the 20 EMA), volume above average.")
def _i_bull_engulf(f):
    dip = (f["rsi"].shift(1) < 45) | (f["close"].shift(1) < f["ema20"].shift(1))
    return Sig(long=_engulf(f, True) & dip & (f["rvol"] >= 1.0), stop_long=f["low"].rolling(2).min(),
               why_long="bullish engulfing bar after a dip (RSI {rsi:.0f}) on {rvol:.1f}x volume", cols=("rsi", "rvol"))


@intraday("bearish_engulfing", "Bearish Engulfing", "pattern", "Bearish engulfing bar after a pop (RSI over 55 or price over the 20 EMA), volume above average.")
def _i_bear_engulf(f):
    pop = (f["rsi"].shift(1) > 55) | (f["close"].shift(1) > f["ema20"].shift(1))
    return Sig(short=_engulf(f, False) & pop & (f["rvol"] >= 1.0), stop_short=f["high"].rolling(2).max(),
               why_short="bearish engulfing bar after a pop (RSI {rsi:.0f}) on {rvol:.1f}x volume", cols=("rsi", "rvol"))


@intraday("mtf_confirmation", "Multi-Timeframe Confirmation", "trend", "15m and 1h trends agree; 5m turns the same way (9>21 EMA, MACD histogram, close over the 20 EMA).")
def _i_mtf(f):
    up = (f["h60"] == 1) & (f["h15"] == 1) & (f["close"] > f["ema20"]) & (f["ema9"] > f["ema21"]) & (f["macd_hist"] > 0)
    dn = (f["h60"] == -1) & (f["h15"] == -1) & (f["close"] < f["ema20"]) & (f["ema9"] < f["ema21"]) & (f["macd_hist"] < 0)
    return Sig(event(up), event(dn), f["ema21"] - 0.5 * f["atr"], f["ema21"] + 0.5 * f["atr"],
               "1h and 15m trends up, 5m aligned (9>21 EMA, MACD histogram positive)",
               "1h and 15m trends down, 5m aligned (9<21 EMA, MACD histogram negative)", ("h15", "h60", "macd_hist"))


# =============================================================================
# Swing (daily bars; SELL = exit / avoid in the cash market)
# =============================================================================
@swing("ema_20_50", "20 EMA / 50 EMA Trend Following", "trend", "20 EMA crosses the 50 EMA with ADX above 15.")
def _s_ema_20_50(f):
    return Sig(xup(f["ema20"], f["ema50"]) & (f["close"] > f["ema20"]) & (f["adx"] > 15),
               xdn(f["ema20"], f["ema50"]) & (f["close"] < f["ema20"]) & (f["adx"] > 15),
               f["ema50"] - 0.5 * f["atr"], f["ema50"] + 0.5 * f["atr"],
               "20 EMA {ema20:.2f} crossed above the 50 EMA {ema50:.2f}, ADX {adx:.0f}",
               "20 EMA {ema20:.2f} crossed below the 50 EMA {ema50:.2f}, ADX {adx:.0f}", ("ema20", "ema50", "adx"))


@swing("ema50_pullback", "50 EMA Pullback", "trend", "Up-trend (20 > 50 EMA, 50 rising) dips to the 50 EMA and closes above the previous day's high.")
def _s_ema50_pullback(f):
    e50 = f["ema50"]
    touched = (f["low"].rolling(3).min() <= e50 * 1.01)
    return Sig(long=(f["ema20"] > e50) & (e50 > e50.shift(5)) & touched & (f["close"] > f["high"].shift(1)) & _bull(f) & (f["close"] > e50),
               stop_long=np.minimum(f["low"].rolling(3).min(), e50) - 0.2 * f["atr"],
               why_long="up-trend pulled back to the rising 50 EMA {ema50:.2f} and closed above the previous day's high", cols=("ema20", "ema50"))


@swing("ema_9_21", "9 EMA / 21 EMA Crossover", "trend", "9 EMA crosses the 21 EMA on the daily chart, on the right side of the 50 EMA.")
def _s_ema_9_21(f):
    return Sig(xup(f["ema9"], f["ema21"]) & (f["close"] > f["ema50"]), xdn(f["ema9"], f["ema21"]) & (f["close"] < f["ema50"]),
               f["ema21"] - 0.7 * f["atr"], f["ema21"] + 0.7 * f["atr"],
               "9 EMA {ema9:.2f} crossed above the 21 EMA {ema21:.2f}, price above the 50 EMA",
               "9 EMA {ema9:.2f} crossed below the 21 EMA {ema21:.2f}, price below the 50 EMA", ("ema9", "ema21", "ema50"))


@swing("macd_bullish", "MACD Bullish Crossover", "momentum", "Daily MACD crosses above its signal line with price above the 50 EMA (bearish mirror = exit).")
def _s_macd(f):
    return Sig(xup(f["macd"], f["macd_sig"]) & (f["close"] > f["ema50"]), xdn(f["macd"], f["macd_sig"]) & (f["close"] < f["ema50"]),
               why_long="MACD {macd:.2f} crossed above its signal {macd_sig:.2f}, price above the 50 EMA",
               why_short="MACD {macd:.2f} crossed below its signal {macd_sig:.2f}, price below the 50 EMA", cols=("macd", "macd_sig", "ema50"))


@swing("rsi_momentum", "RSI Momentum", "momentum", "Daily RSI crosses 60 in an up-trend (crossing below 40 in a down-trend = exit).")
def _s_rsi(f):
    return Sig(xup(f["rsi"], 60) & (f["close"] > f["ema20"]) & (f["ema20"] > f["ema50"]),
               xdn(f["rsi"], 40) & (f["close"] < f["ema20"]) & (f["ema20"] < f["ema50"]),
               why_long="RSI crossed above 60 ({rsi:.0f}) with 20 EMA over 50 EMA",
               why_short="RSI crossed below 40 ({rsi:.0f}) with 20 EMA under 50 EMA", cols=("rsi", "ema20", "ema50"))


@swing("rsi_divergence", "RSI Divergence", "reversal", "Price makes a lower swing low while RSI makes a higher low (confirmed 3 bars after the pivot).")
def _s_rsi_div(f):
    return Sig(f["bull_div"] & (f["close"] > f["swing_lo"]), f["bear_div"] & (f["close"] < f["swing_hi"]),
               f["swing_lo"] - 0.3 * f["atr"], f["swing_hi"] + 0.3 * f["atr"],
               "bullish RSI divergence: lower swing low {swing_lo:.2f} with a higher RSI low",
               "bearish RSI divergence: higher swing high {swing_hi:.2f} with a lower RSI high", ("rsi", "swing_lo", "swing_hi"))


@swing("supertrend", "Supertrend Trend Following", "trend", "Daily Supertrend(10,3) flips up above the 50 EMA (flip down = exit).")
def _s_supertrend(f):
    d = f["st_dir"]
    return Sig((d == 1) & (d.shift(1) == -1) & (f["close"] > f["ema50"]), (d == -1) & (d.shift(1) == 1),
               f["st_line"], f["st_line"],
               "daily Supertrend flipped up (line {st_line:.2f}), price above the 50 EMA",
               "daily Supertrend flipped down (line {st_line:.2f})", ("st_line", "ema50"))


@swing("adx_trend", "ADX Trend Strength", "trend", "Daily ADX crosses above 25 with +DI over -DI and price above the 20 EMA.")
def _s_adx(f):
    x = xup(f["adx"], 25)
    return Sig(x & (f["pdi"] > f["mdi"]) & (f["close"] > f["ema20"]), x & (f["mdi"] > f["pdi"]) & (f["close"] < f["ema20"]),
               why_long="ADX crossed above 25 ({adx:.0f}), +DI {pdi:.0f} over -DI {mdi:.0f}",
               why_short="ADX crossed above 25 ({adx:.0f}), -DI {mdi:.0f} over +DI {pdi:.0f}", cols=("adx", "pdi", "mdi"))


@swing("bb_squeeze", "Bollinger Band Squeeze", "breakout", "Band width in its lowest fifth of 120 days, then a close outside the band on volume.")
def _s_bb_squeeze(f):
    squeeze = f["bb_rank"].shift(1) <= 0.2
    return Sig(squeeze & (f["close"] > f["bb_up"]) & (f["vol_ratio"] >= 1.2), squeeze & (f["close"] < f["bb_lo"]) & (f["vol_ratio"] >= 1.2),
               f["bb_mid"], f["bb_mid"],
               "squeeze (band width in its lowest 20% of 120 days) resolved up: close {close:.2f} over {bb_up:.2f} on {vol_ratio:.1f}x volume",
               "squeeze resolved down: close {close:.2f} under {bb_lo:.2f} on {vol_ratio:.1f}x volume", ("bb_up", "bb_lo", "vol_ratio"))


@swing("donchian", "Donchian Channel Breakout", "breakout", "Close above the 40-day Donchian high (below the 40-day low = exit).")
def _s_donchian(f):
    return Sig(xup(f["close"], f["don_hi"]), xdn(f["close"], f["don_lo"]),
               why_long="close {close:.2f} above the 40-day channel high {don_hi:.2f}",
               why_short="close {close:.2f} below the 40-day channel low {don_lo:.2f}", cols=("don_hi", "don_lo"))


@swing("high_20d", "20-Day High Breakout", "breakout", "Close above the highest high of the previous 20 sessions on 1.5x volume.")
def _s_high20(f):
    return Sig(xup(f["close"], f["hi20"]) & (f["vol_ratio"] >= 1.5), xdn(f["close"], f["lo20"]) & (f["vol_ratio"] >= 1.5),
               why_long="close {close:.2f} above the 20-day high {hi20:.2f} on {vol_ratio:.1f}x volume",
               why_short="close {close:.2f} below the 20-day low {lo20:.2f} on {vol_ratio:.1f}x volume", cols=("hi20", "lo20", "vol_ratio"))


@swing("prev_week_high", "Previous Week High Breakout", "breakout", "Daily close crosses above last week's high, above the 20 EMA.")
def _s_pw_high(f):
    return Sig(xup(f["close"], f["pw_high"]) & (f["close"] > f["ema20"]), xdn(f["close"], f["pw_low"]) & (f["close"] < f["ema20"]),
               f["pw_high"] - 1.0 * f["atr"], f["pw_low"] + 1.0 * f["atr"],
               "close {close:.2f} above last week's high {pw_high:.2f}",
               "close {close:.2f} below last week's low {pw_low:.2f}", ("pw_high", "pw_low", "ema20"))


@swing("volume_accumulation", "Volume Accumulation", "volume", "Up-day volume 1.5x down-day volume over 10 sessions, OBV above its average, new 5-day closing high.")
def _s_accumulation(f):
    acc = (f["ud_ratio"] >= 1.5) & (f["obv"] > f["obv_ema"]) & (f["close"] > f["ema20"]) & (f["close"] > f["hi5"])
    dist = (f["ud_ratio"] <= 0.67) & (f["obv"] < f["obv_ema"]) & (f["close"] < f["ema20"])
    return Sig(event(acc), event(dist),
               why_long="accumulation: up-day volume {ud_ratio:.1f}x down-day volume over 10 sessions, OBV rising, 5-day high",
               why_short="distribution: up/down volume ratio {ud_ratio:.2f} over 10 sessions, OBV falling", cols=("ud_ratio", "vol_ratio"))


@swing("rs_vs_nifty", "Relative Strength vs Nifty 50", "momentum", "RS rank moves into the top 20% of the index with the stock above its 20 and 50 EMA.")
def _s_rs(f):
    strong = (f["rs_rank"] >= 80) & (f["rs20"] > 0) & (f["close"] > f["ema20"]) & (f["ema20"] > f["ema50"])
    weak = (f["rs_rank"] <= 20) & (f["rs20"] < 0) & (f["close"] < f["ema20"]) & (f["ema20"] < f["ema50"])
    return Sig(event(strong), event(weak),
               why_long="relative-strength rank {rs_rank:.0f}/100: {rs20:+.1f}% vs NIFTY over 20 days, above the 20 and 50 EMA",
               why_short="relative-strength rank {rs_rank:.0f}/100: {rs20:+.1f}% vs NIFTY over 20 days, below the 20 and 50 EMA", cols=("rs_rank", "rs20"))


@swing("sector_rs", "Sector Relative Strength", "momentum", "Stock out-performing inside one of the 3 strongest sectors, breaking a 5-day high.")
def _s_sector(f):
    lead = (f["sector_rank"] <= 3) & (f["sector_rs"] > 0) & (f["rs20"] > 0) & (f["close"] > f["ema20"]) & (f["close"] > f["hi5"])
    return Sig(long=event(lead),
               why_long="sector ranked {sector_rank:.0f} by 20-day relative strength ({sector_rs:+.1f}%), stock {rs20:+.1f}% vs NIFTY, new 5-day high",
               cols=("sector_rank", "sector_rs", "rs20"))


@swing("support_bounce", "Support Bounce", "reversal", "Low within half an ATR of the last confirmed swing low, bullish close above the previous high.")
def _s_support(f):
    sup = f["swing_lo"].shift(1)
    near = (f["low"].rolling(2).min() <= sup + 0.5 * f["atr"]) & (f["close"] > sup)
    return Sig(long=near & _bull(f) & (f["close"] > f["high"].shift(1)) & (f["close"] > f["ema50"]),
               stop_long=sup - 0.5 * f["atr"],
               why_long="bounced off swing support {swing_lo:.2f} and closed above the previous day's high", cols=("swing_lo", "ema50"))


@swing("resistance_breakout", "Resistance Breakout", "breakout", "Close through the last confirmed swing high on 1.3x volume (through swing support = exit).")
def _s_resistance(f):
    hi, lo = f["swing_hi"].shift(1), f["swing_lo"].shift(1)
    return Sig(xup(f["close"], hi) & (f["vol_ratio"] >= 1.3), xdn(f["close"], lo) & (f["vol_ratio"] >= 1.3),
               hi - 1.0 * f["atr"], lo + 1.0 * f["atr"],
               "close {close:.2f} through swing resistance {swing_hi:.2f} on {vol_ratio:.1f}x volume",
               "close {close:.2f} through swing support {swing_lo:.2f} on {vol_ratio:.1f}x volume", ("swing_hi", "swing_lo", "vol_ratio"))


@swing("fib_retracement", "Fibonacci Retracement", "reversal", "Pullback into the 38.2-61.8% zone of the last confirmed up-leg, then a close above the previous high.")
def _s_fib(f):
    leg = f["swing_hi"] - f["swing_lo"]
    up_leg = (f["hi_pos"] > f["lo_pos"]) & (leg > 2.0 * f["atr"])
    depth = (f["swing_hi"] - f["low"]) / leg
    return Sig(long=up_leg & depth.between(0.382, 0.66) & _bull(f) & (f["close"] > f["high"].shift(1)) & (f["close"] < f["swing_hi"]),
               stop_long=f["swing_hi"] - 0.786 * leg,
               why_long="retraced into the 38-62% zone of the leg {swing_lo:.2f} -> {swing_hi:.2f} and closed above the previous high",
               cols=("swing_hi", "swing_lo"))


@swing("bullish_engulfing", "Bullish Engulfing", "pattern", "Daily bullish engulfing after a pullback inside an up-trend (bearish engulfing after a rally = exit).")
def _s_engulf(f):
    return Sig(_engulf(f, True) & (f["close"] > f["ema50"]) & (f["rsi"].shift(1) < 55) & (f["vol_ratio"] >= 1.0),
               _engulf(f, False) & (f["rsi"].shift(1) > 60) & (f["vol_ratio"] >= 1.0),
               f["low"].rolling(2).min() - 0.2 * f["atr"], f["high"].rolling(2).max() + 0.2 * f["atr"],
               "daily bullish engulfing above the 50 EMA on {vol_ratio:.1f}x volume",
               "daily bearish engulfing after RSI {rsi:.0f} on {vol_ratio:.1f}x volume", ("rsi", "vol_ratio", "ema50"))


@swing("morning_star", "Morning Star Pattern", "pattern", "Long red day, small-bodied day, then a green day closing above the first day's midpoint.")
def _s_morning_star(f):
    o, c, a = f["open"], f["close"], f["atr"]
    b1, b2 = (o.shift(2) - c.shift(2)), (c.shift(1) - o.shift(1)).abs()
    star = (b1 > 0.6 * a.shift(2)) & (b2 < 0.3 * a.shift(1)) & (np.maximum(o.shift(1), c.shift(1)) < c.shift(2) + 0.2 * a) \
        & (c > o) & (c > (o.shift(2) + c.shift(2)) / 2.0)
    b1e = (c.shift(2) - o.shift(2))
    evening = (b1e > 0.6 * a.shift(2)) & (b2 < 0.3 * a.shift(1)) & (np.minimum(o.shift(1), c.shift(1)) > c.shift(2) - 0.2 * a) \
        & (c < o) & (c < (o.shift(2) + c.shift(2)) / 2.0)
    return Sig(star & (f["rsi"].shift(1) < 50), evening & (f["rsi"].shift(1) > 55),
               f["low"].rolling(3).min() - 0.2 * a, f["high"].rolling(3).max() + 0.2 * a,
               "morning star: long red day, small star, green close above the first day's midpoint",
               "evening star: long green day, small star, red close below the first day's midpoint", ("rsi",))


@swing("cup_handle", "Cup and Handle", "pattern", "Close above the rim of a 60-day cup-and-handle base (heuristic shape test) on 1.2x volume.")
def _s_cup(f):
    return Sig(long=xup(f["close"], f["cup_res"]) & (f["vol_ratio"] >= 1.2), stop_long=f["cup_low"] - 0.2 * f["atr"],
               why_long="close {close:.2f} above the cup-and-handle rim {cup_res:.2f} on {vol_ratio:.1f}x volume", cols=("cup_res", "cup_low", "vol_ratio"))


@swing("hh_hl", "Higher High / Higher Low", "trend", "Swing structure turns to higher highs and higher lows (lower highs / lower lows = exit).")
def _s_hh_hl(f):
    return Sig(event(f["struct"] == 1) & (f["close"] > f["ema20"]), event(f["struct"] == -1) & (f["close"] < f["ema20"]),
               f["swing_lo"] - 0.3 * f["atr"], f["swing_hi"] + 0.3 * f["atr"],
               "structure turned to higher highs / higher lows (swing low {swing_lo:.2f})",
               "structure turned to lower highs / lower lows (swing high {swing_hi:.2f})", ("swing_hi", "swing_lo"))


@swing("ma_confluence", "Moving Average Confluence", "trend", "9 / 20 / 50 EMAs within 2% of each other and a bullish close up through all of them.")
def _s_ma_confluence(f):
    e = pd.concat([f["ema9"], f["ema20"], f["ema50"]], axis=1)
    top, bot = e.max(axis=1), e.min(axis=1)
    tight = (top - bot) / f["close"] <= 0.02
    return Sig(tight & (f["close"] > top) & (f["low"] <= top) & _bull(f) & (f["close"].shift(1) <= top.shift(1)),
               tight & (f["close"] < bot) & (f["high"] >= bot) & _bear(f) & (f["close"].shift(1) >= bot.shift(1)),
               bot - 0.5 * f["atr"], top + 0.5 * f["atr"],
               "9 / 20 / 50 EMAs bunched within 2% and price closed up through all three",
               "9 / 20 / 50 EMAs bunched within 2% and price closed down through all three", ("ema9", "ema20", "ema50"))


@swing("mtf_trend", "Multi-Timeframe Trend Confirmation", "trend", "Last completed week in an up-trend, daily 20 > 50 EMA, daily momentum turns up (close over the 9 EMA, MACD histogram positive).")
def _s_mtf(f):
    up = (f["wk_trend"] == 1) & (f["ema20"] > f["ema50"]) & (f["close"] > f["ema9"]) & (f["macd_hist"] > 0)
    dn = (f["wk_trend"] == -1) & (f["ema20"] < f["ema50"]) & (f["close"] < f["ema9"]) & (f["macd_hist"] < 0)
    return Sig(event(up), event(dn), f["ema20"] - 0.5 * f["atr"], f["ema20"] + 0.5 * f["atr"],
               "weekly trend up, daily 20>50 EMA and daily momentum turned up",
               "weekly trend down, daily 20<50 EMA and daily momentum turned down", ("wk_trend", "ema20", "ema50", "macd_hist"))


@swing("price_volume_momentum", "Price and Volume Momentum", "momentum", "5-day gain over 4% with 5-day volume 1.3x the 20-day average, closing near the 20-day high.")
def _s_pv(f):
    up = (f["roc5"] > 4) & (f["vol5_ratio"] >= 1.3) & (f["close"] >= 0.97 * f["hi20"]) & (f["close"] > f["ema20"])
    dn = (f["roc5"] < -4) & (f["vol5_ratio"] >= 1.3) & (f["close"] <= 1.03 * f["lo20"]) & (f["close"] < f["ema20"])
    return Sig(event(up), event(dn),
               why_long="{roc5:+.1f}% in 5 days on {vol5_ratio:.1f}x volume, closing near the 20-day high",
               why_short="{roc5:+.1f}% in 5 days on {vol5_ratio:.1f}x volume, closing near the 20-day low", cols=("roc5", "vol5_ratio", "hi20"))


REGISTRY = {"intraday": INTRADAY, "swing": SWING}


def get(module: str) -> Dict[str, Strategy]:
    return REGISTRY[module]


def describe(module: Optional[str] = None) -> List[dict]:
    return [s.to_dict() for m, table in REGISTRY.items() if module in (None, m) for s in table.values()]


if __name__ == "__main__":
    for row in describe():
        print(f"{row['module']:8} {row['id']:24} {row['name']:38} {row['kind']:9} {row['about']}")
