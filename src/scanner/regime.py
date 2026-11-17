"""
Market regime classification on each completed signal bar (causal; uses features_ext columns).

    STRONG_BULL        ADX >= 25, +DI leads, EMA 9 > 20 > 50, close above EMA50, structure up, persistent closes
    STRONG_BEAR        the mirror image
    COMPRESSION        Bollinger inside Keltner (squeeze) or Bollinger width AND ATR both in their lowest 25-30%
    EXPANSION          ATR(5) / ATR(14) >= 1.3, or a squeeze released on a bar of >= 1.5 ATR
    HIGH_VOL_REVERSAL  volatility in its top 20% with a change of character or a liquidity sweep in the last bars
    LOW_VOL_RANGE      ADX < 18 and ATR in its lowest 40%
    SIDEWAYS           ADX < 20, no trend structure
    UNCERTAIN          anything else, or not enough data -> no new v3 entries ([STRATEGY_V3] block_uncertain_regime)

The thresholds are starting hypotheses, not fitted values. The backtest reports results per regime
(python -m src.backtest --model v3) so they can be checked.

How the v3 strategy behaves per regime (REGIME_PROFILE below):
    trend continuation (structure, momentum, higher timeframes) gets more weight in strong trends;
    compression is weighted up in compression / low-volatility regimes, down in trends and expansions;
    liquidity-sweep reversals are only allowed in range, sideways and high-volatility-reversal regimes;
    UNCERTAIN blocks new v3 entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.utils.numbers import finite

STRONG_BULL = "Strong bullish trend"
STRONG_BEAR = "Strong bearish trend"
SIDEWAYS = "Sideways"
COMPRESSION = "Volatility compression"
EXPANSION = "Volatility expansion"
HIGH_VOL_REVERSAL = "High-volatility reversal"
LOW_VOL_RANGE = "Low-volatility range"
UNCERTAIN = "Uncertain / no trade"

ALL = (STRONG_BULL, STRONG_BEAR, SIDEWAYS, COMPRESSION, EXPANSION, HIGH_VOL_REVERSAL, LOW_VOL_RANGE, UNCERTAIN)

# family weight multipliers per regime (applied to the [STRATEGY_V3] w3_* weights, then renormalised to 100)
REGIME_PROFILE: Dict[str, Dict[str, float]] = {
    STRONG_BULL: {"Market structure": 1.2, "Momentum": 1.2, "Multi-timeframe": 1.3, "Volatility compression": 0.6},
    STRONG_BEAR: {"Market structure": 1.2, "Momentum": 1.2, "Multi-timeframe": 1.3, "Volatility compression": 0.6},
    COMPRESSION: {"Volatility compression": 1.4, "Market structure": 1.1, "Momentum": 0.8},
    EXPANSION: {"Momentum": 1.3, "Volume & participation": 1.2, "Volatility compression": 0.5},
    HIGH_VOL_REVERSAL: {"Market structure": 1.3, "VWAP & acceptance": 1.2, "Multi-timeframe": 0.6},
    LOW_VOL_RANGE: {"Volatility compression": 1.3, "VWAP & acceptance": 1.1, "Momentum": 0.8},
    SIDEWAYS: {"VWAP & acceptance": 1.2, "Market structure": 1.1, "Multi-timeframe": 0.8},
    UNCERTAIN: {},
}
REVERSAL_ALLOWED = (SIDEWAYS, LOW_VOL_RANGE, HIGH_VOL_REVERSAL, COMPRESSION)


@dataclass
class Regime:
    name: str
    confidence: float                    # 0..1, how clearly the rules matched (not a probability)
    direction: int = 0                   # +1 bullish / -1 bearish / 0 none
    evidence: List[str] = field(default_factory=list)

    def to_dict(self):
        return {"name": self.name, "confidence": round(self.confidence, 2), "direction": self.direction,
                "evidence": self.evidence}


def classify(ext, pos: int) -> Regime:
    ff = ext.ff
    v = ff.v
    adx_v, pdi, mdi = v("adx", pos), v("plus_di", pos), v("minus_di", pos)
    e9, e20, e50 = v("ema9", pos), v("ema20", pos), v("ema50", pos)
    close = v("close", pos)
    atr_pct, rv_pct = ext.v("atr_pct", pos), ext.v("rv_pct", pos)
    bbp, atr_ratio = v("bb_width_pct", pos), v("atr_ratio", pos)
    squeeze, fired = ext.v("squeeze_on", pos), ext.v("squeeze_fired", pos)
    range_atr = ext.v("range_atr", pos)
    trend = ext.v("struct_trend", pos)
    since = ext.v("bars_since_struct", pos)
    persist = ext.v("persistence", pos)
    choch_recent = any(ext.v("choch_up", p) or ext.v("choch_dn", p) for p in range(max(0, pos - 5), pos + 1))
    if not (finite(adx_v) and finite(e50) and finite(atr_pct)):
        return Regime(UNCERTAIN, 0.0, 0, ["not enough history for regime indicators"])
    ev: List[str] = [f"ADX {adx_v:.0f}", f"ATR pct {atr_pct:.0%}"]
    vol_hi = max(atr_pct, rv_pct if finite(rv_pct) else 0) >= 0.8

    # high-volatility reversal: a CHoCH (or a sweep, set by the modules) while volatility is extreme
    if vol_hi and choch_recent:
        d = 1 if any(ext.v("choch_up", p) for p in range(max(0, pos - 5), pos + 1)) else -1
        return Regime(HIGH_VOL_REVERSAL, 0.7, d, ev + ["change of character in the last 6 bars"])
    if (finite(atr_ratio) and atr_ratio >= 1.3) or (fired and finite(range_atr) and range_atr >= 1.5):
        d = int(np.sign(close - e20)) if finite(e20) else 0
        why = f"ATR(5)/ATR(14) {atr_ratio:.2f}" if finite(atr_ratio) else "squeeze released on a wide bar"
        return Regime(EXPANSION, min(1.0, 0.5 + (atr_ratio - 1.3 if finite(atr_ratio) else 0)), d, ev + [why])
    for sign, name in ((1, STRONG_BULL), (-1, STRONG_BEAR)):
        checks = [adx_v >= 25, finite(pdi) and finite(mdi) and sign * (pdi - mdi) > 0,
                  finite(e9) and finite(e20) and sign * (e9 - e20) > 0 and sign * (e20 - e50) > 0,
                  sign * (close - e50) > 0, trend == sign,
                  finite(persist) and (persist >= 0.6 if sign > 0 else persist <= 0.4)]
        if sum(bool(c) for c in checks) >= 5 and checks[0]:
            return Regime(name, sum(bool(c) for c in checks) / 6.0, sign,
                          ev + ["EMA 9/20/50 stacked", "structure " + ("up" if sign > 0 else "down")])
    if squeeze or (finite(bbp) and bbp <= 0.25 and atr_pct <= 0.3):
        return Regime(COMPRESSION, 0.8 if squeeze else 0.6, 0,
                      ev + ["Bollinger inside Keltner" if squeeze else f"BB width pct {bbp:.0%}"])
    if adx_v < 18 and atr_pct <= 0.4:
        return Regime(LOW_VOL_RANGE, 0.6, 0, ev + ["low ADX and low ATR"])
    if adx_v < 20:
        return Regime(SIDEWAYS, 0.5, 0, ev + ["ADX below 20"])
    d = int(trend) if finite(trend) else 0
    return Regime(UNCERTAIN, 0.3, d, ev + ["mixed trend and volatility readings"])


def adapted_weights(base: Dict[str, float], regime: Regime, adapt: bool = True) -> Dict[str, float]:
    if not adapt:
        return dict(base)
    mult = REGIME_PROFILE.get(regime.name, {})
    raw = {k: w * mult.get(k, 1.0) for k, w in base.items()}
    total = sum(raw.values())
    return {k: (100.0 * w / total if total > 0 else w) for k, w in raw.items()}
