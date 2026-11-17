"""
Performance analytics computed only from recorded trades. Pure Python, deterministic.

Conventions
  * Only CLOSED trades with a recorded net P&L count; open positions are reported separately.
  * net = gross - charges (brokerage, STT, exchange, SEBI, stamp duty, GST). Slippage is already inside the fill
    prices; the slippage figure shown is an estimate of how much of the result it explains.
  * Drawdown is measured on the cumulative net P&L added to the configured capital.
  * Ratios that need a reasonable sample (Sharpe, Sortino, VaR, the confidence interval) are None until there are
    enough observations, and the result says why. Small samples are flagged, never extrapolated.
"""
from __future__ import annotations

import math
import random
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, Optional

MIN_TRADES_MEANINGFUL = 30
MIN_DAYS_FOR_RATIOS = 20


def _r(x, nd=2):
    return None if x is None else round(x, nd)


def max_drawdown(nets: List[float], capital: float) -> dict:
    equity, peak, worst, worst_pct = capital, capital, 0.0, 0.0
    start = peak_i = trough_i = 0
    longest, cur_len = 0, 0
    for i, n in enumerate(nets, 1):
        equity += n
        if equity >= peak:
            peak, start, cur_len = equity, i, 0
        else:
            cur_len += 1
            longest = max(longest, cur_len)
            dd = peak - equity
            if dd > worst:
                worst, peak_i, trough_i = dd, start, i
                worst_pct = dd / peak * 100 if peak > 0 else 0.0
    return {"amount": round(worst, 2), "pct": round(worst_pct, 2), "peak_trade": peak_i, "trough_trade": trough_i,
            "longest_underwater_trades": longest}


def losing_streak(nets: List[float]) -> int:
    best = cur = 0
    for n in nets:
        cur = cur + 1 if n < 0 else 0
        best = max(best, cur)
    return best


def bootstrap_mean_ci(nets: List[float], n_iter: int = 1000, level: float = 0.95, seed: int = 7) -> Optional[dict]:
    if len(nets) < MIN_TRADES_MEANINGFUL:
        return None
    rng = random.Random(seed)
    k = len(nets)
    means = sorted(sum(rng.choice(nets) for _ in range(k)) / k for _ in range(n_iter))
    lo = means[int((1 - level) / 2 * n_iter)]
    hi = means[int((1 + level) / 2 * n_iter) - 1]
    return {"low": round(lo, 2), "high": round(hi, 2), "level": level, "excludes_zero": lo > 0 or hi < 0}


def daily_series(trades) -> "OrderedDict[str, float]":
    d: Dict[str, float] = defaultdict(float)
    for t in trades:
        d[t.day] += t.net
    return OrderedDict(sorted(d.items()))


def _ratios(daily: List[float], capital: float) -> dict:
    out = {"sharpe": None, "sortino": None, "var95": None, "ratio_note": ""}
    if len(daily) < MIN_DAYS_FOR_RATIOS:
        out["ratio_note"] = f"needs {MIN_DAYS_FOR_RATIOS}+ trading days with trades (have {len(daily)})"
        return out
    rets = [d / capital for d in daily] if capital > 0 else daily
    mean = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
    downside = [min(0.0, r) for r in rets]
    dsd = math.sqrt(sum(r * r for r in downside) / (len(rets) - 1))
    out["sharpe"] = round(mean / sd * math.sqrt(252), 2) if sd > 0 else None
    out["sortino"] = round(mean / dsd * math.sqrt(252), 2) if dsd > 0 else None
    srt = sorted(daily)
    out["var95"] = round(-srt[max(0, int(0.05 * len(srt)) - 1)], 2)
    out["ratio_note"] = "annualised from days that had trades (zero-trade days are not included)"
    return out


def compute(trades: Iterable, capital: float = 500000.0) -> dict:
    trades = list(trades)
    closed = sorted([t for t in trades if not t.open and t.net is not None], key=lambda t: (t.exit_time or t.entry_time or ""))
    open_ = [t for t in trades if t.open]
    nets = [t.net for t in closed]
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n < 0]
    gross = sum(t.gross for t in closed if t.gross is not None)
    charges = sum(t.charges for t in closed if t.charges is not None)
    slip_vals = [t.slippage for t in closed if t.slippage is not None]
    net = sum(nets)
    n = len(nets)
    sum_w, sum_l = sum(wins), -sum(losses)
    daily = daily_series(closed)
    res = {
        "trades": n, "open_positions": len(open_), "open_pnl": _r(sum(t.open_pnl or 0 for t in open_)),
        "wins": len(wins), "losses": len(losses), "flat": n - len(wins) - len(losses),
        "win_rate": _r(len(wins) / n * 100, 1) if n else None,
        "gross": _r(gross), "charges": _r(charges), "net": _r(net),
        "slippage": _r(sum(slip_vals)) if slip_vals else None,
        "slippage_trades": len(slip_vals),
        "charges_pct_of_gross": _r(charges / abs(gross) * 100, 1) if gross else None,
        "avg_trade": _r(net / n) if n else None,
        "avg_win": _r(sum_w / len(wins)) if wins else None,
        "avg_loss": _r(-sum_l / len(losses)) if losses else None,
        "payoff": _r((sum_w / len(wins)) / (sum_l / len(losses))) if wins and losses else None,
        "profit_factor": _r(sum_w / sum_l) if sum_l > 0 else None,
        "profit_factor_note": "no losing trades yet" if n and sum_l == 0 else "",
        "best_trade": _r(max(nets)) if nets else None, "worst_trade": _r(min(nets)) if nets else None,
        "losing_streak": losing_streak(nets),
        "drawdown": max_drawdown(nets, capital), "capital": capital,
        "return_pct": _r(net / capital * 100) if capital else None,
        "days": len(daily), "best_day": _r(max(daily.values())) if daily else None,
        "worst_day": _r(min(daily.values())) if daily else None,
        "first_day": next(iter(daily), None), "last_day": next(reversed(daily), None) if daily else None,
        "mean_ci": bootstrap_mean_ci(nets),
        "sample_ok": n >= MIN_TRADES_MEANINGFUL,
        "sample_note": "" if n >= MIN_TRADES_MEANINGFUL else
        (f"{n} closed trade{'s' if n != 1 else ''}: too few to judge the strategy (need {MIN_TRADES_MEANINGFUL}+)"
         if n else "no closed trades recorded"),
    }
    res.update(_ratios(list(daily.values()), capital))
    eq, cum = [], 0.0
    for t in closed:
        cum += t.net
        eq.append({"t": t.exit_time or t.entry_time, "v": round(cum, 2)})
    res["equity"] = eq
    res["daily"] = [{"day": k, "net": round(v, 2)} for k, v in daily.items()]
    return res


def breakdown(trades: Iterable, key, capital: float = 500000.0) -> List[dict]:
    groups: Dict[str, list] = defaultdict(list)
    for t in trades:
        groups[key(t)].append(t)
    rows = []
    for k, ts in groups.items():
        m = compute(ts, capital)
        rows.append({"key": k, "trades": m["trades"], "win_rate": m["win_rate"], "net": m["net"], "charges": m["charges"],
                     "profit_factor": m["profit_factor"], "avg_trade": m["avg_trade"],
                     "max_dd": m["drawdown"]["amount"], "sample_ok": m["sample_ok"], "open": m["open_positions"]})
    rows.sort(key=lambda r: (-(r["trades"] or 0), r["key"]))
    return rows


def monthly(trades: Iterable) -> List[dict]:
    d: Dict[str, list] = defaultdict(list)
    for t in trades:
        if not t.open and t.net is not None and t.day:
            d[t.day[:7]].append(t.net)
    return [{"month": k, "trades": len(v), "net": round(sum(v), 2), "win_rate": round(sum(1 for x in v if x > 0) / len(v) * 100, 1)}
            for k, v in sorted(d.items())]
