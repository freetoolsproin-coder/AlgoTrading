"""
From strategy triggers to explainable signals.

    raw_signals      every trigger of the enabled strategies on an enriched frame, with stop, targets, the strength /
                     confidence score and its components, the indicator values and a sentence saying why
    opportunities    one row per stock and side (duplicate setups merged, strategy confluence counted), ranked
    snapshot_*       the per-stock rows of the market scanner and the screens (volume, momentum, trend)

Levels
    stop     the strategy's own invalidation price, with its distance from the trigger close kept between
             stop_atr_min and stop_atr_max ATRs (and at least min_stop_pct of price intraday)
    targets  target1_r and target2_r times the risk, measured from the ACTUAL entry (so reward:risk is always known)
    entry    allowed only inside the entry zone around the trigger close - a price that has run away is not chased

Scores are heuristic sums of the listed components (0-100). They are NOT win probabilities.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.equity.indicators import intraday_components, swing_components, total
from src.equity.strategies import Strategy

TICK = 0.05
INTRADAY_METHOD = ("Strength = trend 25 (1h, 15m, 5m EMAs) + volume 20 (relative volume) + momentum 20 (RSI, MACD) + "
                   "ADX 10 + market 10 (NIFTY 15m trend) + relative strength 15 (day change vs NIFTY). "
                   "A heuristic score, not a win probability.")
SWING_METHOD = ("Confidence = trend 25 (weekly, 20/50 EMA, 200 EMA) + momentum 20 (RSI, MACD) + relative strength 20 "
                "(rank in the NIFTY 50) + volume 15 + swing structure 10 + market 10 (NIFTY trend). "
                "A heuristic score, not a statistically proven win probability.")


def tick(x: float) -> float:
    return round(round(float(x) / TICK) * TICK, 2)


def num(x, digits: int = 2):
    """float for JSON, None when the value is missing (missing is never shown as zero)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, digits) if math.isfinite(x) else None


def stop_distance(side: str, close, stop_ref, atr, cfg, intraday: bool = True):
    """Risk per share from the trigger close (array or scalar). NaN where ATR is not available yet."""
    s = 1.0 if side == "BUY" else -1.0
    close, atr = np.asarray(close, dtype=float), np.asarray(atr, dtype=float)
    ref = np.full_like(close, np.nan) if stop_ref is None else np.asarray(stop_ref, dtype=float)
    dist = (close - ref) * s
    dist = np.where(np.isfinite(dist) & (dist > 0), dist, 1.2 * atr)
    dist = np.clip(dist, cfg.stop_atr_min * atr, cfg.stop_atr_max * atr)
    if intraday:
        dist = np.maximum(dist, close * cfg.min_stop_pct / 100.0)
    return np.where(np.isfinite(atr) & (atr > 0), dist, np.nan)


def levels(side: str, entry: float, stop: float, cfg) -> dict:
    s = 1.0 if side == "BUY" else -1.0
    risk = (entry - stop) * s
    return {"entry": tick(entry), "stop": tick(stop), "t1": tick(entry + s * cfg.target1_r * risk),
            "t2": tick(entry + s * cfg.target2_r * risk), "risk": round(risk, 2), "rr": round(cfg.target2_r, 2),
            "rr_t1": round(cfg.target1_r, 2)}


def entry_zone(side: str, close: float, dist: float, atr: float, max_gap_atr: Optional[float] = None) -> List[float]:
    """[low, high] prices at which the signal may still be entered."""
    s = 1.0 if side == "BUY" else -1.0
    chase = 0.35 * dist if max_gap_atr is None else min(max_gap_atr * atr, 0.5 * dist)
    a, b = close - s * 0.5 * dist, close + s * chase
    return [tick(min(a, b)), tick(max(a, b))]


def in_zone(price: float, zone: List[float]) -> bool:
    return zone[0] <= price <= zone[1]


def _fmt(template: str, row: dict) -> str:
    try:
        return template.format(**row).replace("nan", "n/a")
    except (KeyError, ValueError, IndexError):
        return template


def raw_signals(f: pd.DataFrame, strategies: Iterable[Strategy], module: str, cfg, symbol: str, name: str = "",
                start: int = 0, bar_minutes: int = 5, max_gap_atr: Optional[float] = None) -> List[dict]:
    """Every trigger on rows >= `start` of the enriched frame `f`."""
    if f is None or len(f) == 0:
        return []
    intraday = module == "intraday"
    comp_fn = intraday_components if intraday else swing_components
    comps = {side: comp_fn(f, side) for side in ("BUY", "SELL")}
    score = {side: total(c) for side, c in comps.items()}
    close, atr = f["close"].to_numpy(), f["atr"].to_numpy()
    out = []
    for st in strategies:
        sig = st.run(f)
        for side, mask, ref, why in (("BUY", sig.long, sig.stop_long, sig.why_long), ("SELL", sig.short, sig.stop_short, sig.why_short)):
            hits = np.flatnonzero(mask.to_numpy()[start:]) + start
            if not len(hits):
                continue
            dist = stop_distance(side, close, None if ref is None else ref.to_numpy(), atr, cfg, intraday)
            for i in hits:
                if not np.isfinite(dist[i]):
                    continue
                row = f.iloc[i]
                s = 1.0 if side == "BUY" else -1.0
                c = float(close[i])
                lv = levels(side, c, c - s * float(dist[i]), cfg)
                bar = f.index[i]
                when = bar + pd.Timedelta(minutes=bar_minutes) if intraday else bar
                vol_key = "rvol" if intraday else "vol_ratio"
                vol = num(row.get(vol_key))
                item = {
                    "id": f"{module[0]}|{symbol}|{st.id}|{side}|{bar.strftime('%Y%m%d%H%M')}",
                    "module": module, "symbol": symbol, "name": name, "side": side, "strategy": st.id,
                    "strategy_name": st.name, "kind": st.kind, "bar": bar.isoformat(), "time": when.isoformat(),
                    "day": bar.date().isoformat(), "price": num(c), **lv, "atr": num(atr[i]),
                    "zone": entry_zone(side, c, float(dist[i]), float(atr[i]), None if intraday else max_gap_atr),
                    "score": num(score[side][i], 1), "components": {k: num(v[i], 1) for k, v in comps[side].items()},
                    "volume_ok": bool(vol is not None and vol >= 1.2), "volume_x": vol,
                    "indicators": {k: num(row.get(k)) for k in sig.cols},
                    "reason": f"{st.name}: " + _fmt(why, row.to_dict()),
                }
                if not intraday:
                    item.update(swing_extras(row, side, lv, cfg))
                out.append(item)
    out.sort(key=lambda x: x["time"])
    return out


def swing_extras(row, side: str, lv: dict, cfg) -> dict:
    s = 1.0 if side == "BUY" else -1.0
    atr = float(row.get("atr") or np.nan)
    days = cfg.max_hold_days
    if math.isfinite(atr) and atr > 0:               # sessions to target 1 at ~0.8 ATR of progress a day
        days = int(min(cfg.max_hold_days, max(cfg.min_hold_days, round(abs(lv["t1"] - lv["entry"]) / (0.8 * atr)))))
    wk, d = float(row.get("wk_trend") or 0), float(np.sign((row.get("ema20") or 0) - (row.get("ema50") or 0)))
    trend = "UP" if wk + d >= 1 else "DOWN" if wk + d <= -1 else "SIDEWAYS"
    exits = [f"stop-loss {lv['stop']:.2f} (a gap through it exits at the open)",
             f"target 1 {lv['t1']:.2f}: book half, move the stop to cost", f"target 2 {lv['t2']:.2f}: exit the rest",
             f"daily close {'below' if s > 0 else 'above'} the 20 EMA after day {cfg.min_hold_days}: exit next open",
             f"time exit at the close of session {cfg.max_hold_days}"]
    return {"hold_days": days, "hold_text": f"{cfg.min_hold_days}-{cfg.max_hold_days} sessions (about {days})",
            "trend": trend, "rs_score": num(row.get("rs_rank"), 0), "rs20": num(row.get("rs20")),
            "sector_rank": num(row.get("sector_rank"), 0), "exits": exits, "signal_date": row.name.date().isoformat(),
            "setup": None}


def opportunities(signals: List[dict], validated: Optional[set] = None, perf: Optional[Dict[str, dict]] = None,
                  window_minutes: Optional[int] = None) -> List[dict]:
    """One row per (symbol, side): the same setup reported by several strategies is counted, not repeated.
    Confluence = strategies that agree within `window_minutes` before the latest signal (same day when None).
    rank = score + 5 per extra agreeing strategy (max +15) + 5 when the lead strategy is validated out-of-sample."""
    validated, perf = validated or set(), perf or {}
    groups: Dict[tuple, List[dict]] = {}
    for s in signals:
        groups.setdefault((s["symbol"], s["side"]), []).append(s)
    out = []
    for group in groups.values():
        latest = max(g["time"] for g in group)
        floor = latest[:10] if window_minutes is None else (pd.Timestamp(latest) - pd.Timedelta(minutes=window_minutes)).isoformat()
        recent = [g for g in group if g["time"] >= floor]
        ids = list(dict.fromkeys(g["strategy"] for g in recent))
        lead = max(recent, key=lambda g: (g["time"], g["strategy"] in validated,
                                          (perf.get(g["strategy"]) or {}).get("oos_pf") or 0, g["score"] or 0))
        item = dict(lead)
        item["confluence"] = len(ids)
        item["strategies"] = [{"id": g["strategy"], "name": g["strategy_name"], "time": g["time"],
                               "validated": g["strategy"] in validated} for g in sorted(recent, key=lambda x: x["time"], reverse=True)]
        item["validated"] = lead["strategy"] in validated
        item["rank"] = round((lead["score"] or 0) + min(15, 5 * (len(ids) - 1)) + (5 if item["validated"] else 0), 1)
        out.append(item)
    out.sort(key=lambda x: (x.get("fresh", False), x["rank"]), reverse=True)
    return out


def _trend_word(v) -> str:
    return "up" if v == 1 else "down" if v == -1 else "flat"


def snapshot_intraday(f: pd.DataFrame) -> dict:
    """Indicator snapshot of the last completed 5-minute bar (for the market scanner and the screens)."""
    if f is None or len(f) == 0:
        return {}
    r = f.iloc[-1]
    sl, ss = (float(total({k: v[-1:] for k, v in intraday_components(f.iloc[-1:], side).items()})[0]) for side in ("BUY", "SELL"))
    stack = 1 if r["ema9"] > r["ema21"] > r["ema50"] else -1 if r["ema9"] < r["ema21"] < r["ema50"] else 0
    return {"bar": f.index[-1].isoformat(), "close": num(r["close"]), "day_pct": num(r["day_pct"]), "vwap": num(r["vwap"]),
            "rsi": num(r["rsi"], 1), "adx": num(r["adx"], 1), "atr": num(r["atr"]), "rvol": num(r["rvol"]),
            "rvol_tod": num(r["rvol_tod"]), "macd_hist": num(r["macd_hist"], 3), "h15": int(r["h15"]), "h30": int(r["h30"]),
            "h60": int(r["h60"]), "st_dir": num(r["st_dir"], 0), "ema_stack": stack, "rs_day": num(r["rs_day"]),
            "above_vwap": bool(r["close"] > r["vwap"]) if np.isfinite(r["vwap"]) else None,
            "strength_buy": round(sl, 1), "strength_sell": round(ss, 1), "pdh": num(r["pdh"]), "pdl": num(r["pdl"]),
            "orb_hi": num(r["orb_hi"]), "orb_lo": num(r["orb_lo"])}


def snapshot_swing(f: pd.DataFrame) -> dict:
    if f is None or len(f) == 0:
        return {}
    r = f.iloc[-1]
    cl, cs = (float(total({k: v[-1:] for k, v in swing_components(f.iloc[-1:], side).items()})[0]) for side in ("BUY", "SELL"))
    return {"date": f.index[-1].date().isoformat(), "close": num(r["close"]), "rsi": num(r["rsi"], 1), "adx": num(r["adx"], 1),
            "atr": num(r["atr"]), "roc5": num(r["roc5"]), "roc20": num(r["roc20"]), "vol_ratio": num(r["vol_ratio"]),
            "rs_rank": num(r["rs_rank"], 0), "rs20": num(r["rs20"]), "sector_rank": num(r["sector_rank"], 0),
            "sector_rs": num(r["sector_rs"]), "wk_trend": int(r["wk_trend"]), "struct": int(r["struct"]),
            "st_dir": num(r["st_dir"], 0), "ema20": num(r["ema20"]), "ema50": num(r["ema50"]), "ema200": num(r["ema200"]),
            "trend": int(np.sign(r["ema20"] - r["ema50"])) if np.isfinite(r["ema50"]) else 0,
            "confidence_buy": round(cl, 1), "confidence_sell": round(cs, 1)}


def no_trade_reason(snap: dict, cfg, weak: Optional[dict] = None) -> str:
    """Why a stock has no actionable intraday signal right now."""
    if not snap:
        return "no completed bars yet"
    ctx = (f"15m {_trend_word(snap.get('h15'))}, 1h {_trend_word(snap.get('h60'))}, RSI {snap.get('rsi') if snap.get('rsi') is not None else 'n/a'}, "
           f"relative volume {snap.get('rvol') if snap.get('rvol') is not None else 'n/a'}x")
    if weak:
        return (f"{weak['strategy_name']} triggered ({weak['side']}) but strength {weak['score']:.0f} is below "
                f"{cfg.min_strength:g}: {ctx}")
    return f"no strategy triggered on the last completed 5m bar ({ctx})"
