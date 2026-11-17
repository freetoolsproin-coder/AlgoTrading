"""
Score v3: the multi-factor pre-breakout score, built from the independent strategy modules
(src/scanner/modules.py), adapted to the market regime (regime.py) and penalised by the trap engine (traps.py).

    Family                      Weight   From
    Market structure              20     module 2 (70%) + the stronger of opening range / liquidity sweep (30%)
    Volatility compression        15     module 1
    Momentum                      15     module 3
    VWAP & acceptance             15     module 4
    Volume & participation        10     futures relative volume, OI build-up, unusual participation
    Derivatives                   15     module 7 (live option chain only)
    Multi-timeframe               10     15-min and 30-min trend, daily EMA20 side, 5-min structure trend
                                 ---
                                 100     ([STRATEGY_V3] w3_* - research starting weights, not fitted values)

The weights are multiplied by the regime profile and renormalised to 100 ([STRATEGY_V3] regime_adapt).
A family without data (no futures volume, no option chain in a backtest) is marked "no data" and the score is
rescaled to the families that have data - the same rule as v1 and v2.

What the score is: a 0-100 setup-strength score. It is NOT a probability of winning - nothing here is calibrated.
The ML filter (src/scanner/ml_filter.py) is the only place a probability is estimated, and it is validated out
of sample before it may act.

v3 keeps v1's levels, zones and breakout tests (so the three models are compared on the same events) and adds:
    entry_veto   regime UNCERTAIN, a strong trend against the trade, or traps over the limit -> no entry
    extras       regime, every module's confidence / evidence / invalidation, traps, the derivatives snapshot
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np

from src.config import instruments as inst
from src.scanner import features_ext as fx
from src.scanner import modules as mods
from src.scanner import regime as reg
from src.scanner import strategies as strat
from src.scanner import traps as trap_mod
from src.utils.numbers import finite


def _clip(x):
    return float(min(1.0, max(0.0, x)))


def _participation(ff, pos: int, up: bool, cfg):
    s = 1.0 if up else -1.0
    rvol, oi = ff.v("f_rvol", pos), ff.v("f_oi_change", pos)
    if not ff.has_participation or not finite(rvol):
        return None, [], [], ["futures volume / OI unavailable"]
    sup, con = [], []
    pts = 0.0
    move = ff.v("close", pos) - ff.v("close", max(0, pos - 6))
    if rvol >= 2.0 and s * move > 0:
        pts += 0.5
        sup.append(f"unusual futures volume {rvol:.1f}x with the move")
    elif rvol >= cfg.rvol_good and s * move > 0:
        pts += 0.35
        sup.append(f"futures volume {rvol:.1f}x")
    elif rvol >= 1.0:
        pts += 0.15
    if finite(oi):
        if oi > 0 and s * move > 0:
            pts += 0.4
            sup.append("futures OI rising with price (build-up)" if up else "futures OI rising as price falls (build-up)")
        elif oi < 0 and s * move > 0:
            pts += 0.2
            sup.append("short covering" if up else "long unwinding")
        elif oi > 0 and s * move < 0:
            con.append("futures OI build-up against")
    return _clip(pts), sup, con, [f"futures rvol {rvol:.1f}x"]


def _mtf(ext, ev, pos: int, up: bool):
    s = 1 if up else -1
    sup, con, parts = [], [], []
    for minutes in fx._ints(getattr(ext, "htf_list", None) or "15,30", (15, 30)):
        t = ext.v(f"htf{minutes}", pos)
        if not finite(t):
            continue
        parts.append(1.0 if t == s else 0.5 if t == 0 else 0.0)
        if t == s:
            sup.append(f"{minutes}-min trend " + ("up" if up else "down"))
        elif t == -s:
            con.append(f"{minutes}-min trend against")
    dl = ev.levels or {}
    de = dl.get("daily_ema20")
    if de:
        parts.append(1.0 if s * (ev.close - de) > 0 else 0.0)
        if s * (ev.close - de) <= 0:
            con.append("wrong side of the daily EMA20")
    st = ext.v("struct_trend", pos)
    if finite(st):
        parts.append(1.0 if st == s else 0.5 if st == 0 else 0.0)
    if not parts:
        return None, sup, con
    return float(np.mean(parts)), sup, con


def score_direction(direction: str, ev, ext, regime: reg.Regime, modules: Dict[str, mods.ModuleSignal], deriv,
                    cfg, weights: Dict[str, float]) -> strat.DirectionEval:
    up = direction == inst.CE
    base = ev.side(direction)
    pos = ev.position
    fams: List[strat.Family] = []
    sup: List[str] = []
    con: List[str] = []

    def fam(name, conf, notes, available=True, s_=None, c_=None):
        f = strat.Family(name, 0.0, round(weights[name], 2), notes=list(notes or []), available=available)
        if available and conf is not None:
            f.points = weights[name] * _clip(conf)
        elif conf is None:
            f.available = False
        fams.append(f)
        sup.extend(s_ or [])
        con.extend(c_ or [])

    m_struct, m_orb, m_sweep = modules["Market structure"], modules["Opening range"], modules["Liquidity sweep"]
    extra = max(m_orb.side(direction).confidence if m_orb.available else 0.0, m_sweep.side(direction).confidence)
    sd = m_struct.side(direction)
    fam("Market structure", 0.7 * sd.confidence + 0.3 * extra, m_struct.notes,
        s_=sd.evidence + m_orb.side(direction).evidence[:2] + m_sweep.side(direction).evidence[:1],
        c_=sd.against + m_sweep.side(direction).against[:1])
    def module_family(name):
        m = modules[name]
        sd = m.side(direction)
        fam(name, sd.confidence if m.available else None, m.notes[:3], available=m.available,
            s_=sd.evidence, c_=sd.against)

    for name in ("Volatility compression", "Momentum", "VWAP & acceptance"):
        module_family(name)
    conf, s_, c_, notes = _participation(ext.ff, pos, up, cfg)
    fam("Volume & participation", conf, notes, available=conf is not None, s_=s_, c_=c_)
    module_family("Derivatives")
    conf, s_, c_ = _mtf(ext, ev, pos, up)
    fam("Multi-timeframe", conf, [], available=conf is not None, s_=s_, c_=c_)

    score = sum(f.points for f in fams)
    avail = sum(f.max_points for f in fams if f.available)
    total = sum(f.max_points for f in fams)
    if 0 < avail < total:
        score = score * total / avail

    traps = trap_mod.detect(direction, ev, base, ext, modules, deriv, cfg)
    penalty, veto = trap_mod.verdict(traps, cfg)
    for t in traps:
        con.append(f"trap: {t.name} ({t.note})")
    score -= penalty
    if cfg.block_uncertain_regime and regime.name == reg.UNCERTAIN:
        veto = veto or "market regime uncertain: no new v3 entries"
    against_trend = (regime.name == reg.STRONG_BEAR and up) or (regime.name == reg.STRONG_BULL and not up)
    if against_trend:
        con.append(f"against a {regime.name.lower()}")
        veto = veto or f"against a {regime.name.lower()}"

    out = copy.copy(base)
    out.families = fams
    out.score = round(float(min(100.0, max(0.0, score))), 1)
    out.supporting = list(dict.fromkeys(sup))
    out.contradicting = list(dict.fromkeys(con))
    out.entry_veto = veto
    out.extras = {"traps": [t.to_dict() for t in traps], "trap_points": penalty,
                  "invalidation": {m.name: m.side(direction).to_dict()["invalidation"] for m in modules.values()},
                  "structural_stop": modules["Market structure"].side(direction).invalidation}
    return out


def rescore(ev, ff, cfg, ctx: Optional[dict] = None, deriv=None) -> strat.BarEval:
    """A copy of the v1 BarEval with v3 scores. `deriv` = options.derivatives.DerivSnapshot or None."""
    ext = fx.get(ff, cfg)
    ext.htf_list = cfg.htf_minutes
    regime = reg.classify(ext, ev.position)
    mi = mods.ModuleInput(ext=ext, ev=ev, pos=ev.position, cfg=cfg, regime=regime, deriv=deriv)
    modules = mods.run_all(mi)
    weights = reg.adapted_weights(cfg.weights_v3, regime, cfg.regime_adapt)
    out = copy.copy(ev)
    out.ce = score_direction(inst.CE, ev, ext, regime, modules, deriv, cfg, weights)
    out.pe = score_direction(inst.PE, ev, ext, regime, modules, deriv, cfg, weights)
    out.extras = {"regime": regime.to_dict(), "modules": {k: m.to_dict() for k, m in modules.items()},
                  "weights": {k: round(v, 1) for k, v in weights.items()}, "vwap_source": ext.vwap_source,
                  "derivatives": deriv.to_dict() if deriv is not None else None}
    return out
