"""
Movement metrics and triggers for one index, from 1-minute candles (index spot + near-month futures).

For every window W (5, 10, 15 min by default), measured at time `now`:

    change        last price - the close of the last 1-min candle that had ended by now - W   (points and %)
    range         high - low of the window (candles inside it + the last price)
    velocity      points per minute
    strength      |change| / (1-min ATR(14) x sqrt(W)): ~1 is an ordinary W-minute move, 2+ is unusual
    consistency   share of the window's 1-min candles that closed in the move's direction
    rvol          futures volume per minute in the window / per minute in the 30 minutes before it
    oi change     futures OI change over the window (%), read with the price as
                  long build-up / short covering / short build-up / long unwinding
plus index-level values: day change %, RSI(14) on 1-min, futures price vs session VWAP.

A window is only evaluated once its reference candle is from today (no false "move" across the overnight gap).
Missing data stays missing (None); a check that has no data is reported as "no data", never as passed.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.broker.market_data import empty_minutes
from src.movement.rules import MovementRules
from src.scanner import indicators as ind

ONE_MIN = pd.Timedelta(minutes=1)
UP, DOWN = "UP", "DOWN"
RVOL_BASE_MINUTES = 30
MIN_STATS_MINUTES = 3     # consistency / rvol need a few finished candles: 1- and 2-min windows use the last 3


def _f(value, digits: int = 2) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _ends(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return frame.index + ONE_MIN


def price_at(minutes: pd.DataFrame, when: pd.Timestamp, column: str = "close") -> tuple:
    """(value, candle start) of the last 1-min candle that had ended by `when`, else (nan, None)."""
    if minutes is None or minutes.empty or column not in minutes:
        return np.nan, None
    i = _ends(minutes).searchsorted(when, side="right") - 1
    if i < 0:
        return np.nan, None
    return float(minutes[column].iloc[i]), minutes.index[i]


def window_slice(minutes: pd.DataFrame, start: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    """1-min candles that started at or after `start` and had ended by `now`."""
    if minutes is None or minutes.empty:
        return minutes
    return minutes[(minutes.index >= start) & (_ends(minutes) <= now)]


def oi_reading(price_change: float, oi_change: float) -> Optional[str]:
    if not (np.isfinite(price_change) and np.isfinite(oi_change)) or price_change == 0 or oi_change == 0:
        return None
    if price_change > 0:
        return "Long build-up" if oi_change > 0 else "Short covering"
    return "Short build-up" if oi_change > 0 else "Long unwinding"


def last_price(snap) -> float:
    ltp = float(snap.ltp) if snap.ltp is not None else np.nan
    if np.isfinite(ltp):
        return ltp
    m = snap.spot_minutes
    return float(m["close"].iloc[-1]) if m is not None and len(m) else np.nan


def index_metrics(snap, now: pd.Timestamp, rules: MovementRules) -> dict:
    """All window metrics for one IndexSnapshot (src/broker/market_data.py)."""
    spot = snap.spot_minutes if snap.spot_minutes is not None else empty_minutes()
    fut = snap.fut_minutes
    price = last_price(snap)
    spot_done = spot[_ends(spot) <= now] if len(spot) else spot
    atr1 = np.nan
    rsi_value = np.nan
    if len(spot_done) >= 20:
        tail = spot_done.iloc[-200:]
        atr1 = float(ind.atr(tail, 14).iloc[-1])
        rsi_value = float(ind.rsi(tail["close"], 14).iloc[-1])
    fut_done = fut[_ends(fut) <= now] if fut is not None and len(fut) else None
    fut_today = fut_done[fut_done.index.date == now.date()] if fut_done is not None else None
    vwap_dist = np.nan
    fut_price = np.nan
    if fut_today is not None and len(fut_today):
        vwap = ind.session_vwap(fut_today)
        fut_price = float(fut_today["close"].iloc[-1])
        if np.isfinite(vwap.iloc[-1]) and vwap.iloc[-1] > 0:
            vwap_dist = (fut_price / float(vwap.iloc[-1]) - 1) * 100
    prev_close = float(snap.prev_close) if snap.prev_close is not None else np.nan
    day_change = (price / prev_close - 1) * 100 if np.isfinite(prev_close) and prev_close > 0 else np.nan
    # exchange trade time can be a second ahead of the local clock
    age = max(0.0, (now - snap.ltp_time).total_seconds()) if snap.ltp_time is not None else None
    windows = {w: window_metrics(spot_done, fut_done, price, now, w, atr1) for w in rules.windows}
    return {"index": snap.name, "price": _f(price), "time": now.isoformat(), "data_age_seconds": _f(age, 0),
            "stale": age is None or age > rules.max_data_age_seconds, "day_change_pct": _f(day_change, 3),
            "prev_close": _f(prev_close), "atr_1m": _f(atr1), "rsi_1m": _f(rsi_value, 1),
            "fut_price": _f(fut_price), "fut_vwap_dist_pct": _f(vwap_dist, 3), "mock": bool(snap.mock),
            "errors": list(snap.errors or []), "windows": windows}


def window_metrics(spot: pd.DataFrame, fut: Optional[pd.DataFrame], price: float, now: pd.Timestamp,
                   window: int, atr1: float) -> dict:
    start = now - pd.Timedelta(minutes=window)
    ref, ref_start = price_at(spot, start)
    out = {"window": window, "ready": False}
    if ref_start is None or ref_start.date() != now.date() or not np.isfinite(price) or not np.isfinite(ref):
        out["reason"] = "warming up (needs a full window of today's candles)"
        return out
    change = price - ref
    direction = UP if change > 0 else DOWN if change < 0 else None
    inside = window_slice(spot, start, now)
    stats_start = now - pd.Timedelta(minutes=max(window, MIN_STATS_MINUTES))
    stats = window_slice(spot, stats_start, now)
    highs = list(inside["high"]) + [price, ref]
    lows = list(inside["low"]) + [price, ref]
    strength = abs(change) / (atr1 * math.sqrt(window)) if np.isfinite(atr1) and atr1 > 0 else np.nan
    consistency = np.nan
    if len(stats) and direction:
        body = stats["close"] - stats["open"]
        consistency = float(((body > 0) if direction == UP else (body < 0)).mean())
    rvol = oi_change = np.nan
    if fut is not None and len(fut):
        f_in = window_slice(fut, stats_start, now)
        before = fut[(fut.index < stats_start) & (fut.index >= stats_start - pd.Timedelta(minutes=RVOL_BASE_MINUTES))]
        if ind.has_volume(f_in["volume"]) and len(before) >= 10 and len(f_in):
            base = float(before["volume"].mean())
            if base > 0:
                rvol = float(f_in["volume"].mean()) / base
        oi_ref, _ = price_at(fut, start, "oi")
        oi_now = float(fut["oi"].iloc[-1]) if "oi" in fut else np.nan
        if np.isfinite(oi_ref) and oi_ref > 0 and np.isfinite(oi_now):
            oi_change = (oi_now / oi_ref - 1) * 100
    out.update(ready=True, ref_price=_f(ref), ref_time=(ref_start + ONE_MIN).isoformat(), change=_f(change),
               change_pct=_f(change / ref * 100, 3), direction=direction, high=_f(max(highs)), low=_f(min(lows)),
               range=_f(max(highs) - min(lows)), velocity=_f(change / window, 2), strength=_f(strength),
               consistency=_f(consistency), rvol=_f(rvol), oi_change_pct=_f(oi_change, 3),
               oi_reading=oi_reading(change, oi_change), candles=int(len(inside)))
    return out


# ---------------------------------------------------------------------------------------------- triggers
def checks(m: dict, rules: MovementRules) -> Dict[str, Optional[bool]]:
    """Confirmation checks that are switched on: True / False, or None when the data is missing."""
    out = {}

    def at_least(name, value, minimum):
        if minimum > 0:
            out[name] = None if value is None else value >= minimum

    at_least("strength", m.get("strength"), rules.min_strength)
    at_least("rvol", m.get("rvol"), rules.min_rvol)
    at_least("consistency", m.get("consistency"), rules.min_consistency)
    if rules.oi_confirm:
        reading = m.get("oi_reading")
        want = "Long build-up" if m.get("direction") == UP else "Short build-up"
        out["oi"] = None if reading is None else reading == want
    return out


def threshold_hit(index: str, m: dict, rules: MovementRules, frac: float = 1.0) -> Optional[float]:
    """How many times its threshold the move is (largest of the % and points ratios), or None if below frac."""
    w = m["window"]
    ratios = []
    pct, pts = rules.threshold_pct(w), rules.threshold_points(index, w)
    if pct > 0 and m.get("change_pct") is not None:
        ratios.append(abs(m["change_pct"]) / pct)
    if pts > 0 and m.get("change") is not None:
        ratios.append(abs(m["change"]) / pts)
    if not ratios or max(ratios) < frac:
        return None
    return max(ratios)


def quality(ratio: float, results: Dict[str, Optional[bool]], rules: MovementRules) -> str:
    if any(v is not True for v in results.values()):
        return "UNCONFIRMED"
    return "STRONG" if ratio >= rules.strong_mult else "CONFIRMED"


def evaluate(metrics: dict, rules: MovementRules) -> List[dict]:
    """Triggers for one index (no cooldown here; the monitor applies it)."""
    if metrics.get("stale"):
        return []
    index = metrics["index"]
    found = []
    ready = [m for m in metrics["windows"].values() if m.get("ready")]
    for m in ready:
        ratio = threshold_hit(index, m, rules)
        if ratio is None or not m.get("direction"):
            continue
        results = checks(m, rules)
        found.append(_trigger(metrics, m, str(m["window"]), ratio, results, quality(ratio, results, rules), rules))
    if rules.alignment and len(rules.windows) > 1 and len(ready) == len(rules.windows):
        dirs = {m.get("direction") for m in ready}
        ratios = [threshold_hit(index, m, rules, rules.alignment_frac) for m in ready]
        if len(dirs) == 1 and None not in dirs and all(r is not None for r in ratios):
            longest = max(ready, key=lambda m: m["window"])
            results = checks(longest, rules)
            found.append(_trigger(metrics, longest, "ALL", min(ratios), results,
                                  quality(min(ratios), results, rules), rules))
    return found


def _trigger(metrics: dict, m: dict, window: str, ratio: float, results: dict, grade: str,
             rules: MovementRules) -> dict:
    index, w = metrics["index"], m["window"]
    label = "all windows aligned" if window == "ALL" else f"{w}-min move"
    sign = "+" if m["direction"] == UP else ""
    return {"index": index, "window": window, "direction": m["direction"], "quality": grade,
            "time": metrics["time"], "price": metrics["price"], "ref_price": m["ref_price"],
            "change": m["change"], "change_pct": m["change_pct"], "threshold_ratio": _f(ratio),
            "threshold_pct": rules.threshold_pct(w), "threshold_points": rules.threshold_points(index, w),
            "strength": m.get("strength"), "rvol": m.get("rvol"), "consistency": m.get("consistency"),
            "oi_change_pct": m.get("oi_change_pct"), "oi_reading": m.get("oi_reading"), "range": m.get("range"),
            "velocity": m.get("velocity"), "rsi_1m": metrics.get("rsi_1m"),
            "fut_vwap_dist_pct": metrics.get("fut_vwap_dist_pct"), "day_change_pct": metrics.get("day_change_pct"),
            "checks": {k: ("no data" if v is None else "pass" if v else "fail") for k, v in results.items()},
            "mock": metrics.get("mock", False),
            "message": f"{index} {m['direction']} {sign}{m['change']:.1f} pts ({sign}{m['change_pct']:.2f}%) "
                       f"- {label} - {grade}"}


def vix_triggers(samples: List[tuple], now: pd.Timestamp, rules: MovementRules, mock: bool = False) -> tuple:
    """samples: [(time, vix)] collected by the monitor. Returns (metrics by window, triggers)."""
    out, found = {}, []
    if not samples:
        return out, found
    current = samples[-1][1]
    for w, limit in sorted(rules.vix_move_pct.items()):
        start = now - pd.Timedelta(minutes=w)
        older = [v for t, v in samples if t <= start]
        if not older or not (np.isfinite(current) and np.isfinite(older[-1]) and older[-1] > 0):
            out[w] = {"window": w, "ready": False}
            continue
        change = (current / older[-1] - 1) * 100
        out[w] = {"window": w, "ready": True, "change_pct": _f(change, 2)}
        if limit > 0 and abs(change) >= limit:
            direction = UP if change > 0 else DOWN
            found.append({"index": "INDIAVIX", "window": str(w), "direction": direction,
                          "quality": "STRONG" if abs(change) >= limit * rules.strong_mult else "CONFIRMED",
                          "time": now.isoformat(), "price": _f(current), "ref_price": _f(older[-1]),
                          "change": _f(current - older[-1]), "change_pct": _f(change, 2),
                          "threshold_ratio": _f(abs(change) / limit), "threshold_pct": limit, "checks": {},
                          "mock": mock,
                          "message": f"India VIX {'jumped' if change > 0 else 'dropped'} {change:+.2f}% in {w} min"})
    return out, found
