"""
Backtest and validation of every equity strategy, on real Upstox history.

    python -m src.equity.backtest --module swing                 3 years of daily bars (config [BACKTEST])
    python -m src.equity.backtest --module intraday --years 2    2 years of 5-minute bars (about 1,300 API calls
                                                                 the first time; cached on disk afterwards)
    python -m src.equity.backtest --module swing --symbols RELIANCE,TCS --strategies donchian,high_20d

How a trade is simulated (identical rules to the live engine, same strategy functions, same risk manager)
    signal    on a COMPLETED bar; the strength / confidence filter and the entry window apply
    entry     next bar's open, worse by the slippage, only inside the entry zone (swing: not after a large gap)
    stop      fixed at the signal; a gap through it fills at the open; if a bar touches both the stop and a target the
              STOP is assumed first
    exits     target 1 books half and moves the stop to cost, target 2 exits the rest; intraday square-off bar; swing
              20-EMA / exit-signal trend exit at the next open and a time exit after max_hold_days sessions
    size      quantity = capital x risk% / (stop distance + slippage [+ gap allowance]), capped by position value;
              trades below min_rr (planned) or min_net_rr (after charges) are skipped; brokerage, STT, exchange, SEBI,
              stamp duty, GST and DP charges are deducted from every fill
    book      per strategy: one position per stock, the module's open-position limit, the daily trade limit, the
              daily-loss stop and the consecutive-loss stop

Validation (a strategy is never activated on in-sample numbers)
    out-of-sample    the last (100 - in_sample_pct)% of the period is never looked at when judging the first part
    walk-forward     anchored folds: judge on everything before a fold, measure on the fold
    validated        profitable in-sample AND out-of-sample (trades >= min_oos_trades, profit factor >=
                     min_oos_profit_factor, net > 0 in each part) and the share of profitable walk-forward folds
                     >= min_wf_positive_folds

Known limits, stated in every result: the universe is TODAY's NIFTY 50 (survivorship bias - supply
config/nifty50_history.csv with symbol,from,to rows to restrict each stock to its membership period); bars are
5-minute / daily, so the order of events inside one bar is unknown (handled conservatively); partial fills, market
impact and circuit limits are not modelled; candles are used as Upstox serves them.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import settings
from src.equity import strategies as strat
from src.equity.config import EquityConfig, enabled_ids, hhmm
from src.equity.data import INDEX, EquityData, now_ist
from src.equity.indicators import (add_cross_section, enrich_daily, enrich_index, enrich_intraday, intraday_components,
                                   swing_components, total)
from src.equity.risk import Costs, RiskManager
from src.equity.signals import entry_zone, stop_distance

logger = logging.getLogger("scanner.equity.backtest")
MEMBERSHIP_FILE = settings.CONFIG_DIR / "nifty50_history.csv"
BIG = 10 ** 9
NOTES = [
    "Universe = today's NIFTY 50 constituents: survivorship bias is present for earlier years unless "
    "config/nifty50_history.csv (symbol,from,to) is supplied.",
    "Bars are 5-minute (intraday) or daily (swing). Inside one bar the stop is assumed to be hit before a target.",
    "Charges and slippage are estimates; partial fills, market impact and circuit limits are not modelled.",
    "Past results do not predict future results. Scores and win rates here are not probabilities of profit.",
]


def _first(mask: np.ndarray) -> int:
    """Index of the first True, BIG when there is none."""
    hit = np.flatnonzero(mask)
    return int(hit[0]) if len(hit) else BIG


def load_membership(path=MEMBERSHIP_FILE) -> Dict[str, List[tuple]]:
    out: Dict[str, List[tuple]] = {}
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(line for line in handle if line.strip() and not line.startswith("#")):
                try:
                    a = date.fromisoformat((row.get("from") or "1990-01-01").strip())
                    b = date.fromisoformat((row.get("to") or "2999-12-31").strip() or "2999-12-31")
                except ValueError:
                    continue
                out.setdefault((row.get("symbol") or "").strip().upper(), []).append((a, b))
    except OSError:
        pass
    return out


# =============================================================================
# One trade
# =============================================================================
def simulate_trade(side: str, entry: float, stop: float, t1: float, t2: float, qty: int, o, h, l, c, a: int, b: int,
                   final_price: float, final_reason: str, charge: Callable[[str, float, int], float], slip_bps: float) -> dict:
    """Bars a..b (inclusive) are walked for the stop and the targets; whatever is left exits at `final_price`.
    Returns the fills as (bar index, qty, price, reason) with slippage applied, plus gross / charges."""
    s = 1.0 if side == "BUY" else -1.0
    lo, hi, op = l[a:b + 1], h[a:b + 1], o[a:b + 1]
    adverse, favour = (lo, hi) if s > 0 else (hi, lo)
    worse = (lambda p: p * (1 - s * slip_bps / 1e4))          # exit fills are worse by the slippage
    fills = []
    half = qty // 2
    i_stop = _first(s * (adverse - stop) <= 0)
    i_t1 = _first(s * (favour - t1) >= 0)
    left = qty
    if i_stop <= i_t1:                                        # stop first (or the same bar: assume the stop)
        if i_stop < BIG:
            gap = op[i_stop] if s * (op[i_stop] - stop) <= 0 else stop
            fills.append((a + i_stop, left, worse(gap), "STOP"))
            left = 0
    else:
        if half >= 1:
            fills.append((a + i_t1, half, worse(t1), "TARGET1"))
            left -= half
        i_t2 = _first(s * (favour[i_t1:] - t2) >= 0) + i_t1
        i_be = _first(s * (adverse[i_t1 + 1:] - entry) <= 0) + i_t1 + 1          # stop at cost from the next bar on
        if i_be <= i_t2 and i_be < BIG:
            gap = op[i_be] if s * (op[i_be] - entry) <= 0 else entry
            fills.append((a + i_be, left, worse(gap), "STOP_AT_COST"))
            left = 0
        elif i_t2 < BIG:
            fills.append((a + i_t2, left, worse(t2), "TARGET2"))
            left = 0
    if left > 0:
        fills.append((b if final_reason != "TREND_EXIT" else b + 1, left, worse(final_price), final_reason))
    exit_side = "SELL" if s > 0 else "BUY"
    gross = sum((p - entry) * s * q for _, q, p, _ in fills)
    charges = charge(side, entry, qty) + sum(charge(exit_side, p, q) for _, q, p, _ in fills)
    return {"fills": fills, "gross": gross, "charges": charges, "exit_idx": fills[-1][0], "reason": fills[-1][3],
            "exit": sum(p * q for _, q, p, _ in fills) / qty}


# =============================================================================
# One stock, all strategies
# =============================================================================
def trades_for_stock(f: pd.DataFrame, module: str, symbol: str, strategies, cfg: EquityConfig, risk: RiskManager,
                     membership: Optional[List[tuple]] = None, first_day: Optional[date] = None) -> List[dict]:
    if f is None or len(f) < 30:
        return []
    intraday = module == "intraday"
    mcfg = cfg.intraday if intraday else cfg.swing
    o, h, l, c, atr = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "atr"))
    n = len(f)
    days = f.index.normalize()
    dates = np.array([d.date() for d in f.index])
    ok = np.ones(n, bool)
    if first_day is not None:
        ok &= dates >= first_day
    if membership:
        ok &= np.array([any(a <= d <= b for a, b in membership) for d in dates])
    comp = intraday_components if intraday else swing_components
    score = {side: total(comp(f, side)) for side in ("BUY", "SELL")}
    min_score = mcfg.min_strength if intraday else mcfg.min_confidence
    slip = risk.cfg.slippage_bps

    def charge(side, price, qty):
        return risk.costs.charge(module, side, price, qty)

    if intraday:
        clock = np.array([t.hour * 60 + t.minute for t in f.index.time])
        start_m, end_m, sq_m = (x.hour * 60 + x.minute for x in (hhmm(mcfg.entry_start), hhmm(mcfg.entry_end), hhmm(mcfg.square_off)))
        ok &= (clock + 5 >= start_m) & (clock + 5 <= end_m)                 # the signal bar CLOSES inside the window
        day_id = pd.factorize(days)[0]
        idx = np.arange(n)
        sq_idx = pd.Series(np.where(clock + 5 <= sq_m, idx, -1)).groupby(day_id).transform("max").to_numpy()
        exit_any = None
    else:
        ema20 = f["ema20"].to_numpy(dtype=float)
        exit_any = np.zeros(n, bool)                                         # any strategy's SELL = exit signal
        for st in strat.SWING.values():
            exit_any |= st.run(f).short.to_numpy()
        trend_exit = exit_any | (np.isfinite(ema20) & (c < ema20))

    out = []
    for st in strategies:
        sig = st.run(f)
        sides = (("BUY", sig.long, sig.stop_long), ("SELL", sig.short, sig.stop_short)) if intraday and mcfg.allow_short \
            else (("BUY", sig.long, sig.stop_long),)
        free = -1
        events = []
        for side, mask, ref in sides:
            m = mask.to_numpy() & ok & (score[side] >= min_score)
            if intraday:                                  # one signal per setup and side a day, as in the live engine
                m &= pd.Series(m).groupby(day_id).cumsum().to_numpy() == 1
            if not m.any():
                continue
            dist = stop_distance(side, c, None if ref is None else ref.to_numpy(dtype=float), atr, mcfg, intraday)
            events += [(int(i), side, float(dist[i])) for i in np.flatnonzero(m) if np.isfinite(dist[i])]
        for i, side, dist in sorted(events):
            if i <= free or i + 1 >= n:
                continue
            s = 1.0 if side == "BUY" else -1.0
            e_open, stop = o[i + 1], c[i] - s * dist
            zone = entry_zone(side, c[i], dist, atr[i], None if intraday else cfg.risk.max_entry_gap_atr)
            if not (zone[0] <= e_open <= zone[1]) or s * (e_open - stop) <= 0:
                continue
            entry = e_open * (1 + s * slip / 1e4)
            r = (entry - stop) * s
            t1, t2 = entry + s * mcfg.target1_r * r, entry + s * mcfg.target2_r * r
            sizing = risk.size(module, side, entry, stop, t2, atr[i])
            if sizing.qty < 1 or risk.rr_reasons(sizing):
                continue
            if intraday:
                b = int(sq_idx[i])
                if day_id[i + 1] != day_id[i] or b < i + 1:
                    continue
                res = simulate_trade(side, entry, stop, t1, t2, sizing.qty, o, h, l, c, i + 1, b, c[b], "SQUARE_OFF", charge, slip)
            else:
                a, last = i + 1, i + mcfg.max_hold_days
                if last >= n:
                    continue                                                 # still open at the end of the data
                te = _first(trend_exit[a + mcfg.min_hold_days - 1:last]) + a + mcfg.min_hold_days - 1
                if te < last:                                                # flagged on bar te -> exit at the next open
                    res = simulate_trade(side, entry, stop, t1, t2, sizing.qty, o, h, l, c, a, te, o[te + 1], "TREND_EXIT", charge, slip)
                else:
                    res = simulate_trade(side, entry, stop, t1, t2, sizing.qty, o, h, l, c, a, last, c[last], "TIME_EXIT", charge, slip)
            free = res["exit_idx"]
            net = res["gross"] - res["charges"]
            out.append({"strategy": st.id, "symbol": symbol, "side": side, "signal_time": f.index[i], "entry_time": f.index[i + 1],
                        "exit_time": f.index[min(res["exit_idx"], n - 1)], "entry": round(entry, 2), "exit": round(res["exit"], 2),
                        "stop": round(stop, 2), "qty": sizing.qty, "gross": round(res["gross"], 2), "charges": round(res["charges"], 2),
                        "net": round(net, 2), "r": round(net / (r * sizing.qty), 3), "reason": res["reason"],
                        "bars": res["exit_idx"] - i, "score": round(float(score[side][i]), 1)})
    return out


def apply_book_rules(trades: List[dict], cfg: EquityConfig, module: str) -> List[dict]:
    """Per strategy, in time order: open-position limit, daily trade limit, daily-loss stop, loss-streak stop."""
    limit = cfg.risk.max_open_swing if module == "swing" else cfg.risk.max_open_intraday
    loss_cap = cfg.risk.capital * cfg.risk.max_daily_loss_pct / 100.0
    kept: List[dict] = []
    by: Dict[str, List[dict]] = {}
    for t in trades:
        by.setdefault(t["strategy"], []).append(t)
    for rows in by.values():
        rows.sort(key=lambda t: (t["entry_time"], -t["score"]))
        open_: List[dict] = []
        day, taken, done = None, 0, []
        for t in rows:
            d = t["entry_time"].date()
            if d != day:
                day, taken, done = d, 0, []
            open_ = [x for x in open_ if x["exit_time"] > t["entry_time"]]
            closed = sorted((x for x in done if x["exit_time"] <= t["entry_time"]), key=lambda x: x["exit_time"])
            streak = 0
            for x in reversed(closed):
                if x["net"] < 0:
                    streak += 1
                else:
                    break
            if len(open_) >= limit or taken >= cfg.risk.daily_trade_limit or sum(x["net"] for x in closed) <= -loss_cap \
                    or (cfg.risk.max_consecutive_losses and streak >= cfg.risk.max_consecutive_losses):
                continue
            kept.append(t)
            open_.append(t)
            done.append(t)
            taken += 1
    kept.sort(key=lambda t: t["entry_time"])
    return kept


# =============================================================================
# Metrics
# =============================================================================
def metrics(trades: List[dict], capital: float, days: Optional[pd.DatetimeIndex] = None) -> dict:
    n = len(trades)
    if not n:
        return {"trades": 0, "win_rate": None, "profit_factor": None, "net": 0.0, "net_return_pct": 0.0, "avg_win": None,
                "avg_loss": None, "avg_trade": None, "payoff": None, "avg_r": None, "max_dd": 0.0, "max_dd_pct": 0.0,
                "sharpe": None, "charges": 0.0}
    net = np.array([t["net"] for t in trades])
    wins, losses = net[net > 0], net[net <= 0]
    gp, gl = wins.sum(), -losses.sum()
    order = np.argsort([t["exit_time"].value for t in trades], kind="stable")
    curve = np.cumsum(net[order])
    dd = float((np.maximum.accumulate(np.maximum(curve, 0.0)) - curve).max())
    sharpe = None
    if days is not None and len(days) > 20:
        daily = pd.Series(net, index=pd.DatetimeIndex([t["exit_time"].normalize() for t in trades])).groupby(level=0).sum()
        ret = daily.reindex(days, fill_value=0.0) / capital
        sd = ret.std(ddof=1)
        sharpe = round(float(ret.mean() / sd * math.sqrt(252)), 2) if sd > 0 else None
    return {"trades": n, "win_rate": round(100.0 * len(wins) / n, 1),
            "profit_factor": round(float(gp / gl), 2) if gl > 0 else None, "net": round(float(net.sum()), 0),
            "net_return_pct": round(float(net.sum()) / capital * 100, 2),
            "avg_win": round(float(wins.mean()), 0) if len(wins) else None,
            "avg_loss": round(float(losses.mean()), 0) if len(losses) else None, "avg_trade": round(float(net.mean()), 0),
            "payoff": round(float(wins.mean() / -losses.mean()), 2) if len(wins) and len(losses) and losses.mean() < 0 else None,
            "avg_r": round(float(np.mean([t["r"] for t in trades])), 3), "max_dd": round(dd, 0),
            "max_dd_pct": round(dd / capital * 100, 2), "sharpe": sharpe,
            "charges": round(float(sum(t["charges"] for t in trades)), 0)}


def _pf_ok(m: dict, min_trades: int, min_pf: float) -> bool:
    return m["trades"] >= min_trades and m["net"] > 0 and (m["profit_factor"] is None or m["profit_factor"] >= min_pf)


def evaluate(trades: List[dict], days: pd.DatetimeIndex, cfg: EquityConfig, module: str, strategy_ids: List[str]) -> dict:
    """Per-strategy metrics for the whole period, in-sample, out-of-sample and the walk-forward folds."""
    bt, capital = cfg.backtest, cfg.risk.capital
    days = pd.DatetimeIndex(sorted(set(days)))
    split = days[min(len(days) - 1, int(len(days) * bt.in_sample_pct / 100.0))]
    k = max(2, int(bt.walk_forward_folds))
    edges = [days[int(len(days) * j / (k + 1))] for j in range(1, k + 1)] + [days[-1] + pd.Timedelta(days=1)]
    table = strat.get(module)
    by: Dict[str, List[dict]] = {sid: [] for sid in strategy_ids}
    for t in trades:
        by.setdefault(t["strategy"], []).append(t)
    rows, validated, matrix = [], [], {}
    for sid in strategy_ids:
        ts = by.get(sid, [])
        day_of = (lambda t: t["entry_time"].normalize())
        ins = [t for t in ts if day_of(t) < split]
        oos = [t for t in ts if day_of(t) >= split]
        folds = []
        for j in range(k):
            train = [t for t in ts if day_of(t) < edges[j]]
            test = [t for t in ts if edges[j] <= day_of(t) < edges[j + 1]]
            mt, ms = metrics(train, capital), metrics(test, capital)
            folds.append({"test_from": edges[j].date().isoformat(), "test_to": (edges[j + 1] - pd.Timedelta(days=1)).date().isoformat(),
                          "train_trades": mt["trades"], "train_pf": mt["profit_factor"],
                          "selected": _pf_ok(mt, bt.min_oos_trades, bt.min_oos_profit_factor),
                          "test_trades": ms["trades"], "test_pf": ms["profit_factor"], "test_net": ms["net"]})
        traded = [f for f in folds if f["test_trades"] >= 3]
        wf_pos = round(sum(1 for f in traded if f["test_net"] > 0) / len(traded), 2) if traded else 0.0
        m_all, m_in, m_oos = metrics(ts, capital, days), metrics(ins, capital, days[days < split]), metrics(oos, capital, days[days >= split])
        ok = _pf_ok(m_in, bt.min_oos_trades, bt.min_oos_profit_factor) and _pf_ok(m_oos, bt.min_oos_trades, bt.min_oos_profit_factor) \
            and wf_pos >= bt.min_wf_positive_folds
        why = []
        if not _pf_ok(m_in, bt.min_oos_trades, bt.min_oos_profit_factor):
            why.append("not profitable in-sample" if m_in["trades"] >= bt.min_oos_trades else f"only {m_in['trades']} in-sample trades")
        if m_oos["trades"] < bt.min_oos_trades:
            why.append(f"only {m_oos['trades']} out-of-sample trades (need {bt.min_oos_trades})")
        elif m_oos["net"] <= 0:
            why.append("out-of-sample net is not positive")
        elif m_oos["profit_factor"] is not None and m_oos["profit_factor"] < bt.min_oos_profit_factor:
            why.append(f"out-of-sample profit factor {m_oos['profit_factor']} below {bt.min_oos_profit_factor:g}")
        if wf_pos < bt.min_wf_positive_folds:
            why.append(f"profitable in {wf_pos:.0%} of walk-forward folds (need {bt.min_wf_positive_folds:.0%})")
        if ok:
            validated.append(sid)
        reasons: Dict[str, int] = {}
        for t in ts:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        rows.append({"id": sid, "name": table[sid].name, "kind": table[sid].kind, "all": m_all, "in_sample": m_in,
                     "out_of_sample": m_oos, "walk_forward": folds, "wf_positive": wf_pos, "validated": ok,
                     "verdict": "validated in-sample and out-of-sample" if ok else "not validated: " + "; ".join(why), "exits": reasons})
        for t in ts:
            cell = matrix.setdefault(t["symbol"], {}).setdefault(sid, [0, 0.0])
            cell[0] += 1
            cell[1] = round(cell[1] + t["net"], 0)
    by_stock = []
    for symbol, cells in matrix.items():
        ts = [t for t in trades if t["symbol"] == symbol]
        m = metrics(ts, capital)
        by_stock.append({"symbol": symbol, "trades": m["trades"], "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
                         "net": m["net"], "by_strategy": cells})
    by_stock.sort(key=lambda r: r["net"], reverse=True)
    # walk-forward of the selection rule itself: trade only what looked good BEFORE each fold
    wf = []
    for j in range(k):
        chosen = [r["id"] for r in rows if r["walk_forward"][j]["selected"]]
        test = [t for t in trades if t["strategy"] in chosen and edges[j] <= t["entry_time"].normalize() < edges[j + 1]]
        m = metrics(test, capital)
        wf.append({"test_from": edges[j].date().isoformat(), "test_to": (edges[j + 1] - pd.Timedelta(days=1)).date().isoformat(),
                   "selected": len(chosen), "trades": m["trades"], "net": m["net"], "profit_factor": m["profit_factor"], "win_rate": m["win_rate"]})
    return {"strategies": rows, "validated": validated, "by_stock": by_stock, "walk_forward": wf,
            "split": split.date().isoformat(), "days": len(days)}


# =============================================================================
# Run
# =============================================================================
def run(module: str, cfg: Optional[EquityConfig] = None, data: Optional[EquityData] = None, store=None,
        years: Optional[float] = None, symbols: Optional[List[str]] = None, strategy_ids: Optional[List[str]] = None,
        progress: Optional[Callable[[str], None]] = None, now: Optional[pd.Timestamp] = None, save: bool = True) -> dict:
    cfg = cfg or EquityConfig.from_ini()
    data = data or EquityData(requests_per_second=cfg.requests_per_second, per_minute=cfg.max_requests_per_minute,
                              per_30min=cfg.max_requests_per_30min)
    now = now or now_ist()
    say = progress or (lambda text: None)
    intraday = module == "intraday"
    years = float(years or (cfg.backtest.intraday_years if intraday else cfg.backtest.swing_years))
    first_day = now.date() - timedelta(days=int(years * 365.25))
    table = strat.get(module)
    ids = [i for i in enabled_ids(cfg.strategies_intraday if intraday else cfg.strategies_swing, list(table))
           if not strategy_ids or i in strategy_ids]
    strategies = [table[i] for i in ids]
    stocks = [s for s in data.universe() if not symbols or s.symbol in symbols]
    risk = RiskManager(cfg.risk, Costs(cfg.costs_intraday, cfg.costs_delivery), kill_file=settings.DATA_DIR / "__no_kill__")
    membership = load_membership()
    trades: List[dict] = []
    all_days: set = set()
    daily_from = first_day - timedelta(days=420)                # EMA 200 / 120-day ranks need a long warm-up

    say("loading NIFTY 50 index history")
    nifty_daily = data.history(INDEX.symbol, INDEX.key, "1d", daily_from, now)
    if intraday:
        warm = first_day - timedelta(days=12)
        nf = enrich_index(data.history(INDEX.symbol, INDEX.key, "5m", warm, now), nifty_daily)
        for n, stock in enumerate(stocks, 1):
            say(f"{n}/{len(stocks)} {stock.symbol}: 5-minute history and simulation")
            bars = data.history(stock.symbol, stock.key, "5m", warm, now)
            if len(bars) < 500:
                continue
            f = enrich_intraday(bars, data.history(stock.symbol, stock.key, "1d", daily_from, now), nf, 5, cfg.intraday.orb_minutes)
            all_days.update(f.index[f.index.date >= first_day].normalize())
            trades += trades_for_stock(f, module, stock.symbol, strategies, cfg, risk, membership.get(stock.symbol), first_day)
            data.forget("5m")
    else:
        frames = {}
        for n, stock in enumerate(stocks, 1):
            say(f"{n}/{len(stocks)} {stock.symbol}: daily history")
            d = data.history(stock.symbol, stock.key, "1d", daily_from, now)
            if len(d) >= 120:
                frames[stock.symbol] = enrich_daily(d, nifty_daily)
        add_cross_section(frames, {s.symbol: s.industry for s in stocks})
        for n, (symbol, f) in enumerate(frames.items(), 1):
            say(f"{n}/{len(frames)} {symbol}: simulation")
            all_days.update(f.index[f.index.date >= first_day])
            trades += trades_for_stock(f, module, symbol, strategies, cfg, risk, membership.get(symbol), first_day)

    say("portfolio rules and validation")
    raw_count = len(trades)
    trades = apply_book_rules(trades, cfg, module)
    days = pd.DatetimeIndex(sorted(all_days))
    if not len(days):
        raise RuntimeError("no history was loaded - check the Upstox token (python -m src.broker.auth)")
    result = evaluate(trades, days, cfg, module, ids)
    m = metrics(trades, cfg.risk.capital, days)
    result["summary"] = {"module": module, "from": days[0].date().isoformat(), "to": days[-1].date().isoformat(), "years": years,
                         "stocks": len(stocks), "strategies": len(ids), "signals_simulated": raw_count,
                         "capital": cfg.risk.capital, "risk_per_trade_pct": cfg.risk.risk_per_trade_pct,
                         "slippage_bps": cfg.risk.slippage_bps, "min_rr": cfg.risk.min_rr, "in_sample_until": result["split"],
                         "membership_file": bool(membership), "all_strategies_combined": m,
                         "note": "combined = every strategy trading its own book side by side; not one account"}
    result["notes"] = NOTES if not membership else NOTES[1:]
    result["recent_trades"] = [{**t, **{k: t[k].isoformat() for k in ("signal_time", "entry_time", "exit_time")}} for t in trades[-150:]]
    if save and store is not None:
        store.insert("backtests", module=module, created=now.isoformat(timespec="seconds"),
                     params={"years": years, "symbols": symbols, "strategies": ids}, result=result)
        if not symbols and not strategy_ids:           # only a full run may change what is allowed to trade
            validated = dict(store.get("validated", {}) or {})
            validated[module] = result["validated"]
            store.set("validated", validated)
            perf = dict(store.get("perf", {}) or {})
            perf[module] = {r["id"]: {"trades": r["all"]["trades"], "win": r["all"]["win_rate"], "pf": r["all"]["profit_factor"],
                                      "net": r["all"]["net"], "max_dd_pct": r["all"]["max_dd_pct"], "sharpe": r["all"]["sharpe"],
                                      "avg_r": r["all"]["avg_r"], "oos_trades": r["out_of_sample"]["trades"],
                                      "oos_win": r["out_of_sample"]["win_rate"], "oos_pf": r["out_of_sample"]["profit_factor"],
                                      "oos_net": r["out_of_sample"]["net"], "wf_positive": r["wf_positive"],
                                      "validated": r["validated"], "verdict": r["verdict"]} for r in result["strategies"]}
            store.set("perf", perf)
    say("done")
    return result


def print_result(result: dict):
    s = result["summary"]
    print(f"\n{s['module'].upper()} backtest  {s['from']} -> {s['to']}  ({s['stocks']} stocks, {s['strategies']} strategies)  "
          f"in-sample until {s['in_sample_until']}")
    print(f"{'strategy':26} {'trades':>6} {'win%':>6} {'PF':>5} {'net':>10} {'ret%':>6} {'maxDD%':>7} {'Sharpe':>6} {'avgR':>6} |"
          f" {'OOS n':>5} {'OOS PF':>6} {'OOS net':>9} {'WF+':>4}  verdict")
    f = (lambda v, w, d=2: f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
    for r in sorted(result["strategies"], key=lambda r: r["all"]["net"], reverse=True):
        a, o = r["all"], r["out_of_sample"]
        print(f"{r['id']:26} {a['trades']:>6} {f(a['win_rate'], 6, 1)} {f(a['profit_factor'], 5)} {f(a['net'], 10, 0)} "
              f"{f(a['net_return_pct'], 6)} {f(a['max_dd_pct'], 7)} {f(a['sharpe'], 6)} {f(a['avg_r'], 6)} | {o['trades']:>5} "
              f"{f(o['profit_factor'], 6)} {f(o['net'], 9, 0)} {r['wf_positive']:>4.0%}  {r['verdict']}")
    print("\nWalk-forward (trade only what was good before each fold):")
    for w in result["walk_forward"]:
        print(f"  {w['test_from']} -> {w['test_to']}: {w['selected']} strategies selected, {w['trades']} trades, net {w['net']:,.0f}, PF {w['profit_factor']}")
    print("\n" + "\n".join("* " + n for n in result["notes"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.equity.backtest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--module", choices=("intraday", "swing"), required=True)
    p.add_argument("--years", type=float)
    p.add_argument("--symbols", help="comma-separated subset (the result is then not used for validation)")
    p.add_argument("--strategies", help="comma-separated strategy ids")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args(argv)
    from src.equity.store import Store
    from src.utils.logging_config import setup_logging
    settings.ensure_dirs()
    setup_logging(logging.INFO, log_file=settings.LOG_DIR / "equity.log")
    cfg = EquityConfig.from_ini()
    store = None if a.no_save else Store(cfg.data_dir / "equity.db")
    split = (lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None)
    try:
        result = run(a.module, cfg, store=store, years=a.years, symbols=[s.upper() for s in split(a.symbols) or []] or None,
                     strategy_ids=split(a.strategies), progress=lambda t: print("  " + t, flush=True), save=not a.no_save)
    except Exception as error:      # noqa: BLE001
        print(f"Backtest failed: {type(error).__name__}: {error}")
        return 1
    print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
