"""
Robustness statistics for any backtest's trade list (no strategy logic here).

    bootstrap_ci(nets)            95% confidence intervals for expectancy, profit factor and win rate,
                                  by resampling trades with replacement.
    monte_carlo_drawdown(nets)    the same trades in random order, many times: how bad the drawdown and the
                                  losing streak could have been with the same edge but different luck.
    segment_table(rows, key_fn)   expectancy / PF / win rate per group (direction, time of day, days to expiry,
                                  VIX regime, year ...).
    split_halves(trades)          first 70% / last 30% of the history, in time order.

All functions take plain numbers so the ORB, swing and F&O backtests can use them as well. A fixed seed keeps
reports reproducible. These numbers describe uncertainty in a backtest; they do not make it predictive.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


def _pf(nets: np.ndarray) -> float:
    gains = nets[nets > 0].sum()
    losses = -nets[nets <= 0].sum()
    if losses <= 0:
        return np.nan
    return float(gains / losses)


def summary(nets: Sequence[float]) -> dict:
    a = np.asarray(list(nets), dtype=float)
    if len(a) == 0:
        return {"trades": 0, "expectancy": None, "profit_factor": None, "win_rate": None, "net": 0.0}
    pf = _pf(a)
    return {"trades": int(len(a)), "expectancy": float(a.mean()),
            "profit_factor": None if not np.isfinite(pf) else pf,
            "win_rate": float((a > 0).mean() * 100), "net": float(a.sum())}


def bootstrap_ci(nets: Sequence[float], n_iter: int = 2000, level: float = 0.95, seed: int = 7) -> dict:
    """Percentile bootstrap. Returns {metric: (low, high)} plus 'p_expectancy_le_0' (share of resamples <= 0)."""
    a = np.asarray(list(nets), dtype=float)
    if len(a) < 10:
        return {"n": int(len(a)), "expectancy": (None, None), "profit_factor": (None, None),
                "win_rate": (None, None), "p_expectancy_le_0": None, "note": "fewer than 10 trades"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_iter, len(a)))
    samples = a[idx]
    means = samples.mean(axis=1)
    wins = (samples > 0).mean(axis=1) * 100
    gains = np.where(samples > 0, samples, 0).sum(axis=1)
    losses = -np.where(samples <= 0, samples, 0).sum(axis=1)
    pfs = np.where(losses > 0, gains / np.where(losses > 0, losses, 1), np.nan)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100

    def band(x):
        x = x[np.isfinite(x)]
        return (float(np.percentile(x, lo)), float(np.percentile(x, hi))) if len(x) else (None, None)

    return {"n": int(len(a)), "expectancy": band(means), "profit_factor": band(pfs), "win_rate": band(wins),
            "p_expectancy_le_0": float((means <= 0).mean())}


def _max_drawdown(path: np.ndarray, capital: float) -> float:
    equity = capital + np.cumsum(path)
    peak = np.maximum.accumulate(np.concatenate([[capital], equity]))[1:]
    return float((equity - peak).min())


def _longest_losing_run(path: np.ndarray) -> int:
    best = run = 0
    for x in path:
        run = run + 1 if x < 0 else 0
        best = max(best, run)
    return best


def monte_carlo_drawdown(nets: Sequence[float], capital: float, n_iter: int = 2000, seed: int = 11,
                         loss_limit: Optional[float] = None) -> dict:
    """Shuffle the trade order (same trades, different luck). Drawdowns in rupees, negative numbers."""
    a = np.asarray(list(nets), dtype=float)
    if len(a) < 10:
        return {"n": int(len(a)), "note": "fewer than 10 trades"}
    rng = np.random.default_rng(seed)
    dds = np.empty(n_iter)
    streaks = np.empty(n_iter)
    for k in range(n_iter):
        path = rng.permutation(a)
        dds[k] = _max_drawdown(path, capital)
        streaks[k] = _longest_losing_run(path)
    out = {"n": int(len(a)), "actual_max_dd": _max_drawdown(a, capital),
           "actual_losing_streak": _longest_losing_run(a),
           "dd_median": float(np.median(dds)), "dd_p95": float(np.percentile(dds, 5)),   # 5th pct = worst 5%
           "dd_worst": float(dds.min()), "streak_median": float(np.median(streaks)),
           "streak_p95": float(np.percentile(streaks, 95))}
    if loss_limit:
        out["p_dd_beyond_limit"] = float((dds <= -abs(loss_limit)).mean())
        out["loss_limit"] = float(abs(loss_limit))
    return out


def segment_table(rows: Iterable, key_fn: Callable, net_fn: Callable = lambda r: r.net,
                  order: Optional[List[str]] = None) -> List[dict]:
    groups: Dict[str, List[float]] = {}
    for r in rows:
        key = key_fn(r)
        if key is None:
            continue
        groups.setdefault(str(key), []).append(float(net_fn(r)))
    keys = order if order else sorted(groups)
    out = []
    for k in keys:
        if k in groups:
            out.append({"group": k, **summary(groups[k])})
    return out


def split_halves(trades: List, frac: float = 0.7, time_fn: Callable = lambda t: t.entry_time):
    """First `frac` of the trades in time order, and the rest (the same 70 / 30 split used in earlier reports)."""
    ordered = sorted(trades, key=time_fn)
    cut = int(round(len(ordered) * frac))
    return ordered[:cut], ordered[cut:]


# ---------- standard buckets for intraday option trades ----------
def bucket_time(entry_time: str) -> Optional[str]:
    try:
        hh, mm = int(entry_time[11:13]), int(entry_time[14:16])
    except (ValueError, TypeError, IndexError):
        return None
    m = hh * 60 + mm
    if m < 10 * 60 + 30:
        return "09:15-10:30"
    if m < 12 * 60:
        return "10:30-12:00"
    if m < 13 * 60 + 30:
        return "12:00-13:30"
    return "13:30-close"


TIME_ORDER = ["09:15-10:30", "10:30-12:00", "12:00-13:30", "13:30-close"]


def bucket_dte(days: Optional[int]) -> Optional[str]:
    if days is None:
        return None
    if days <= 1:
        return "0-1 day"
    if days <= 3:
        return "2-3 days"
    return "4+ days"


DTE_ORDER = ["0-1 day", "2-3 days", "4+ days"]


def bucket_vix(vix: Optional[float]) -> Optional[str]:
    if vix is None or not np.isfinite(vix):
        return None
    if vix < 13:
        return "VIX < 13"
    if vix < 17:
        return "VIX 13-17"
    return "VIX 17+"


VIX_ORDER = ["VIX < 13", "VIX 13-17", "VIX 17+"]


def risk_adjusted(trades: List, capital: float, sessions: int, day_fn: Callable = lambda t: t.day,
                  net_fn: Callable = lambda t: t.net) -> dict:
    """Daily-P&L based ratios over `sessions` trading days (days without trades count as 0):
        Sharpe   mean / std of daily returns x sqrt(252)
        Sortino  mean / downside deviation x sqrt(252)
        Calmar   annualised return / max drawdown (both % of capital)
        Recovery net profit / |max drawdown|
    Plus gross profit / loss, average win / loss, longest losing streak. Not meaningful below ~30 trades."""
    out = {"sharpe": None, "sortino": None, "calmar": None, "recovery_factor": None, "gross_profit": 0.0,
           "gross_loss": 0.0, "avg_win": None, "avg_loss": None, "max_consecutive_losses": 0}
    if not trades or sessions <= 0:
        return out
    nets = np.array([float(net_fn(t)) for t in trades])
    daily: Dict[str, float] = {}
    for t in trades:
        k = str(day_fn(t))
        daily[k] = daily.get(k, 0.0) + float(net_fn(t))
    rets = np.array(list(daily.values()) + [0.0] * max(0, sessions - len(daily))) / capital
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    dd_sd = np.sqrt((downside ** 2).sum() / len(rets)) if len(rets) else 0.0
    ordered = [float(net_fn(t)) for t in sorted(trades, key=lambda t: str(getattr(t, "entry_time", day_fn(t))))]
    equity = capital + np.cumsum(ordered)
    peak = np.maximum.accumulate(np.concatenate([[capital], equity]))[1:]
    max_dd = float((equity - peak).min()) if len(equity) else 0.0
    years = sessions / 252.0
    ann = nets.sum() / capital / years if years > 0 else np.nan
    out.update(sharpe=float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else None,
               sortino=float(rets.mean() / dd_sd * np.sqrt(252)) if dd_sd > 0 else None,
               calmar=float(ann / (abs(max_dd) / capital)) if max_dd < 0 else None,
               recovery_factor=float(nets.sum() / abs(max_dd)) if max_dd < 0 else None,
               gross_profit=float(nets[nets > 0].sum()), gross_loss=float(nets[nets <= 0].sum()),
               avg_win=float(nets[nets > 0].mean()) if (nets > 0).any() else None,
               avg_loss=float(nets[nets <= 0].mean()) if (nets <= 0).any() else None,
               max_consecutive_losses=_longest_losing_run(np.array(ordered)))
    return out
