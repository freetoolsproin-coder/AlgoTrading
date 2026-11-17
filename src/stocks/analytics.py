"""
Pure analytics for the stock-options service (no I/O, easy to test).

    supertrend(bars, 10, 3)          line + direction (+1 up / -1 down), causal
    classic_pivots(h, l, c)          P, R1-R3, S1-S3 from the previous session
    chain_stats(chain, spot)         PCR, max pain, CE / PE OI walls, ATM IV, OI change
    returns(daily)                   1W / 1M / 3M / 6M / 1Y % returns
    hist_vol(daily), beta(a, b)      annualised volatility, beta vs NIFTY
    bs_price / bs_greeks             Black-Scholes (used by the mock data and as a greeks fallback)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.scanner.indicators import atr, ema, wilder
from src.utils.numbers import finite


# =============================================================================
# Trend indicators
# =============================================================================
def supertrend(bars: pd.DataFrame, length: int = 10, mult: float = 3.0):
    """Classic Supertrend. Returns (line, direction) Series; direction +1 = up-trend, -1 = down-trend."""
    n = len(bars)
    line = pd.Series(np.nan, index=bars.index)
    direction = pd.Series(np.nan, index=bars.index)
    if n == 0:
        return line, direction
    a = atr(bars, length).to_numpy()
    hl2 = ((bars["high"] + bars["low"]) / 2.0).to_numpy()
    close = bars["close"].to_numpy()
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    fu, fl = np.full(n, np.nan), np.full(n, np.nan)
    d = 1
    for i in range(n):
        if not np.isfinite(a[i]):
            continue
        if i == 0 or not np.isfinite(fu[i - 1]):
            fu[i], fl[i] = upper[i], lower[i]
            d = 1 if close[i] >= hl2[i] else -1
        else:
            fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
            if d == -1 and close[i] > fu[i - 1]:
                d = 1
            elif d == 1 and close[i] < fl[i - 1]:
                d = -1
        line.iloc[i] = fl[i] if d == 1 else fu[i]
        direction.iloc[i] = d
    return line, direction


def adx_value(bars: pd.DataFrame, length: int = 14) -> float:
    from src.scanner.indicators import adx
    if len(bars) < length * 2:
        return float("nan")
    value = adx(bars, length)[0].iloc[-1]
    return float(value) if finite(value) else float("nan")


def day_vwap(bars_today: pd.DataFrame) -> float:
    if bars_today is None or bars_today.empty:
        return float("nan")
    vol = pd.to_numeric(bars_today["volume"], errors="coerce").fillna(0.0)
    if vol.sum() <= 0:
        return float("nan")
    typical = (bars_today["high"] + bars_today["low"] + bars_today["close"]) / 3.0
    return float((typical * vol).sum() / vol.sum())


# =============================================================================
# Levels
# =============================================================================
def classic_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    p = (high + low + close) / 3.0
    return {"P": p, "R1": 2 * p - low, "S1": 2 * p - high, "R2": p + (high - low), "S2": p - (high - low),
            "R3": high + 2 * (p - low), "S3": low - 2 * (high - p)}


def swing_levels(daily: pd.DataFrame, lookback: int = 120, side: int = 3) -> Dict[str, List[float]]:
    """Confirmed daily swing highs / lows of the last `lookback` sessions (most recent first)."""
    d = daily.tail(lookback)
    highs, lows = [], []
    h, l = d["high"].to_numpy(), d["low"].to_numpy()
    for i in range(side, len(d) - side):
        if h[i] == max(h[i - side:i + side + 1]):
            highs.append(float(h[i]))
        if l[i] == min(l[i - side:i + side + 1]):
            lows.append(float(l[i]))
    return {"highs": highs[::-1], "lows": lows[::-1]}


# =============================================================================
# Option chain statistics
# =============================================================================
def chain_stats(chain: Optional[pd.DataFrame], spot: float) -> dict:
    if chain is None or chain.empty or not finite(spot):
        return {}
    c = chain.dropna(subset=["strike"]).copy()
    ce_oi, pe_oi = float(c["ce_oi"].sum()), float(c["pe_oi"].sum())
    out = {"pcr": round(pe_oi / ce_oi, 2) if ce_oi > 0 else None, "ce_oi_total": ce_oi, "pe_oi_total": pe_oi}
    above, below = c[c["strike"] >= spot], c[c["strike"] <= spot]
    if not above.empty and above["ce_oi"].max() > 0:
        out["ce_wall"] = float(above.loc[above["ce_oi"].idxmax(), "strike"])     # resistance
    if not below.empty and below["pe_oi"].max() > 0:
        out["pe_wall"] = float(below.loc[below["pe_oi"].idxmax(), "strike"])     # support
    strikes = c["strike"].to_numpy()
    if len(strikes) and (ce_oi + pe_oi) > 0:
        pain = [float(((s - strikes).clip(min=0) * c["ce_oi"].to_numpy()).sum()
                      + ((strikes - s).clip(min=0) * c["pe_oi"].to_numpy()).sum()) for s in strikes]
        out["max_pain"] = float(strikes[int(np.argmin(pain))])
    atm = c.iloc[(c["strike"] - spot).abs().argsort()[:1]]
    if not atm.empty:
        row = atm.iloc[0]
        ivs = [v for v in (row.get("ce_iv"), row.get("pe_iv")) if finite(v) and v > 0]
        out["atm_strike"] = float(row["strike"])
        out["atm_iv"] = round(float(np.mean(ivs)), 2) if ivs else None
        straddle = [v for v in (row.get("ce_ltp"), row.get("pe_ltp")) if finite(v)]
        out["atm_straddle"] = round(float(sum(straddle)), 2) if len(straddle) == 2 else None
    for side in ("ce", "pe"):
        column = f"{side}_oi_chg"
        chg = pd.to_numeric(c[column], errors="coerce") if column in c else None
        out[f"{side}_oi_change"] = float(chg.sum()) if chg is not None and chg.notna().any() else None
    return out


# =============================================================================
# Performance / risk
# =============================================================================
def returns(daily: pd.DataFrame) -> Dict[str, Optional[float]]:
    closes = daily["close"].dropna()
    out = {}
    for label, n in (("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 248)):
        out[label] = round(float(closes.iloc[-1] / closes.iloc[-1 - n] - 1) * 100, 2) if len(closes) > n else None
    return out


def hist_vol(daily: pd.DataFrame, n: int = 20) -> Optional[float]:
    r = np.log(daily["close"]).diff().dropna().tail(n)
    return round(float(r.std() * math.sqrt(248) * 100), 2) if len(r) >= 10 else None


def beta(stock_daily: pd.DataFrame, index_daily: Optional[pd.DataFrame], n: int = 248) -> Optional[float]:
    if index_daily is None or index_daily.empty:
        return None
    a = stock_daily.set_index(stock_daily["datetime"].dt.date)["close"].pct_change()
    b = index_daily.set_index(index_daily["datetime"].dt.date)["close"].pct_change()
    j = pd.concat([a, b], axis=1, join="inner").dropna().tail(n)
    if len(j) < 30 or j.iloc[:, 1].var() == 0:
        return None
    return round(float(j.cov().iloc[0, 1] / j.iloc[:, 1].var()), 2)


def daily_indicators(daily: pd.DataFrame) -> dict:
    c = daily["close"]
    from src.scanner.indicators import macd, rsi
    line, sig, hist = macd(c)
    out = {"close": float(c.iloc[-1]),
           "ema20": float(ema(c, 20).iloc[-1]) if len(c) >= 20 else None,
           "ema50": float(ema(c, 50).iloc[-1]) if len(c) >= 50 else None,
           "ema200": float(ema(c, 200).iloc[-1]) if len(c) >= 200 else None,
           "rsi": float(rsi(c).iloc[-1]) if len(c) >= 15 else None,
           "atr": float(wilder(_tr(daily), 14).iloc[-1]) if len(c) >= 15 else None,
           "macd": float(line.iloc[-1]) if finite(line.iloc[-1]) else None,
           "macd_signal": float(sig.iloc[-1]) if finite(sig.iloc[-1]) else None,
           "macd_hist": float(hist.iloc[-1]) if finite(hist.iloc[-1]) else None,
           "adx": adx_value(daily)}
    return {k: (None if v is None or not finite(v) else round(v, 2)) for k, v in out.items()}


def _tr(bars):
    prev = bars["close"].shift()
    return pd.concat([bars["high"] - bars["low"], (bars["high"] - prev).abs(), (bars["low"] - prev).abs()],
                     axis=1).max(axis=1)


# =============================================================================
# Black-Scholes (r = 6.5%, no dividends)
# =============================================================================
RATE = 0.065


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_price(spot, strike, years, vol, kind="CE", rate=RATE) -> float:
    if years <= 0 or vol <= 0:
        return max(spot - strike, 0.0) if kind == "CE" else max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    if kind == "CE":
        return spot * _ncdf(d1) - strike * math.exp(-rate * years) * _ncdf(d2)
    return strike * math.exp(-rate * years) * _ncdf(-d2) - spot * _ncdf(-d1)


def bs_greeks(spot, strike, years, vol, kind="CE", rate=RATE) -> dict:
    years = max(years, 1e-4)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    gamma = _npdf(d1) / (spot * vol * math.sqrt(years))
    vega = spot * _npdf(d1) * math.sqrt(years) / 100.0
    if kind == "CE":
        delta = _ncdf(d1)
        theta = (-spot * _npdf(d1) * vol / (2 * math.sqrt(years)) - rate * strike * math.exp(-rate * years) * _ncdf(d2)) / 365
    else:
        delta = _ncdf(d1) - 1.0
        theta = (-spot * _npdf(d1) * vol / (2 * math.sqrt(years)) + rate * strike * math.exp(-rate * years) * _ncdf(-d2)) / 365
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "iv": vol * 100}
