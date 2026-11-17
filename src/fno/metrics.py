"""
Performance statistics computed from trade records (paper or backtest). Nothing is estimated: a statistic that
needs more data than exists is returned as None (Sharpe needs 20 trading days, for example).

A trade record needs: net (rupees after charges), r_multiple, opened, closed, strategy, symbol, side.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _r(x, n=2):
    return None if x is None or not math.isfinite(float(x)) else round(float(x), n)


def summary(trades: List[dict], signals: Optional[int] = None) -> dict:
    nets = np.array([float(t.get("net") or 0.0) for t in trades])
    rs = np.array([float(t.get("r_multiple") or 0.0) for t in trades])
    wins, losses = nets[nets > 0], nets[nets < 0]
    out = {"signals": signals if signals is not None else len(trades), "trades": int(len(nets)),
           "wins": int(len(wins)), "losses": int(len(losses)),
           "win_rate": _r(100.0 * len(wins) / len(nets), 1) if len(nets) else None,
           "profit_factor": _r(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
           "net_pnl": _r(nets.sum()) if len(nets) else 0.0,
           "avg_profit": _r(wins.mean()) if len(wins) else None, "avg_loss": _r(losses.mean()) if len(losses) else None,
           "expectancy": _r(nets.mean()) if len(nets) else None, "avg_r": _r(rs.mean(), 3) if len(rs) else None,
           "total_r": _r(rs.sum()) if len(rs) else 0.0, "max_drawdown": None, "sharpe": None, "days": 0}
    if not len(nets):
        return out
    order = sorted(trades, key=lambda t: str(t.get("closed") or t.get("opened")))
    equity = np.cumsum([float(t.get("net") or 0.0) for t in order])
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    out["max_drawdown"] = _r(float((peak - equity).max()))
    daily = defaultdict(float)
    for t in order:
        daily[str(t.get("closed") or t.get("opened"))[:10]] += float(t.get("net") or 0.0)
    out["days"] = len(daily)
    values = np.array(list(daily.values()))
    if len(values) >= 20 and values.std(ddof=1) > 0:
        out["sharpe"] = _r(values.mean() / values.std(ddof=1) * math.sqrt(252))
    return out


def breakdown(trades: List[dict], key: Callable[[dict], str]) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for t in trades:
        groups[str(key(t))].append(t)
    rows = [dict(summary(g), group=name) for name, g in groups.items()]
    return sorted(rows, key=lambda r: -(r["net_pnl"] or 0.0))


def time_bucket(t: dict) -> str:
    ts = pd.Timestamp(t.get("opened"))
    return f"{ts.hour:02d}:{'00' if ts.minute < 30 else '30'}"


def equity_curve(trades: List[dict]) -> List[dict]:
    order = sorted((t for t in trades if t.get("closed")), key=lambda t: str(t["closed"]))
    total, out = 0.0, []
    for t in order:
        total += float(t.get("net") or 0.0)
        out.append({"t": str(t["closed"]), "equity": round(total, 2)})
    return out


def report(trades: List[dict], signal_counts: Optional[Dict[str, int]] = None) -> dict:
    """Everything the performance page shows."""
    closed = [t for t in trades if t.get("status", "CLOSED") == "CLOSED"]
    signal_counts = signal_counts or {}
    by_strategy = breakdown(closed, lambda t: t.get("strategy"))
    for row in by_strategy:
        row["signals"] = signal_counts.get(row["group"], row["trades"])
    seen = {r["group"] for r in by_strategy}
    for name, n in signal_counts.items():
        if name not in seen:
            by_strategy.append(dict(summary([], n), group=name))
    return {"overall": summary(closed, sum(signal_counts.values()) if signal_counts else None),
            "by_strategy": by_strategy, "by_symbol": breakdown(closed, lambda t: t.get("symbol")),
            "by_side": breakdown(closed, lambda t: t.get("side")),
            "by_time": sorted(breakdown(closed, time_bucket), key=lambda r: r["group"]),
            "by_exit": breakdown(closed, lambda t: t.get("exit_reason")),
            "equity": equity_curve(closed)}
