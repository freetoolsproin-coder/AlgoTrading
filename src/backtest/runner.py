"""
Backtest of the pre-breakout CE / PE system on 1-minute index history.

    python -m src.backtest                          NIFTY, cached data/history/ (2023-01 onwards)
    python -m src.backtest --instrument SENSEX      downloads missing Upstox history into data/history/
                                                    (needs an Upstox access token), then runs
    python -m src.backtest --start 2024-01-01 --end 2026-09-30
    (VS Code: Run and Debug -> "Backtest NIFTY" / "Backtest SENSEX")

Only NIFTY and SENSEX are accepted (src/config/instruments.py).
Writes output/prebreakout_backtest_report[_SENSEX].html and output/prebreakout_backtest_trades[_SENSEX].csv.

How it works (no look-ahead)
  * The SAME engine as the live service: prebreakout_engine.evaluate_bar() on every completed 5-minute
    bar, then StageTracker.step(). Indicators are causal, swing pivots are used only once confirmed,
    daily levels come from earlier sessions only (tests check this).
  * Fill: an ENTRY SETUP at a bar's close is bought at the open of the next 1-minute bar.
  * Exits walk every later 1-minute bar as open -> adverse extreme -> favourable extreme -> close,
    so a bar that touches both stop and target counts as the stop.
  * Options are MODELLED: Black-Scholes on the index price with IV = India VIX at that minute
    (no historical option prices / bid-ask exist in this project). A bid-ask spread of
    assumed_spread_pct and slippage_pct are charged on both legs, plus brokerage, STT, exchange,
    SEBI, stamp duty and GST (src/execution/costs.py, rates from [COSTS] in config/scanner.ini).
  * Index spot has no volume: the participation family (futures volume / OI / VWAP) is NOT available
    in this backtest, so the maximum reachable score is 90, and the VWAP / volume confirmation rules are
    skipped. The live system does use futures data. The report says so.
  * Not modelled: IV smile and IV crush, liquidity limits, other indices' agreement (context family
    gets its neutral value), gap risk beyond what the 1-minute data shows.

Validation
  * Walk-forward: the entry threshold and the extension limit are chosen on a 6-month train window and
    applied to the next 3 months; only the out-of-sample months are reported as "walk-forward".
  * Sensitivity grid (in-sample, for robustness, NOT for picking values).
  * Baselines: a simple previous-day high / low breakout through the same option model and exits,
    and buy-and-hold NIFTY.
  * Signal quality: forward move after confirmations vs. all bars, false breakouts, missed moves.
  * Robustness (backtest_stats.py): bootstrap 95% confidence intervals for expectancy / PF / win rate,
    Monte Carlo reshuffles of the trade order (drawdown and losing-streak ranges), results by direction,
    time of day, days to expiry, VIX regime and year, and an exit-rule check on the first 70% / last 30%.
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest import history as hist
from src.backtest import stats as bs
from src.config import instruments as inst
from src.config import settings
from src.config.instruments import SUPPORTED_INDICES, get_instrument
from src.execution import paper_trading as paper_mod
from src.execution.costs import make_cost_fn
from src.options import selection as opt_sel
from src.risk import manager as riskm
from src.scanner import signals as sig
from src.scanner import strategies as strat
from src.scanner.features_ext import exit_context
from src.utils import numbers as num

OUTPUT_DIR = str(settings.OUTPUT_DIR)
CONFIG_FILE = str(settings.CONFIG_FILE)

# Lot size in force on a date (NSE circulars). Sizing is risk-based, so this only changes rounding.
LOT_HISTORY = {
    "NIFTY": [(date(2000, 1, 1), 50), (date(2024, 11, 20), 75), (date(2026, 1, 6), 65)],
    "SENSEX": [(date(2000, 1, 1), 10), (date(2024, 11, 20), 20)],
}


def lot_on(name: str, day: date) -> int:
    lot = hist.DEFAULT_LOT_SIZE[get_instrument(name).name]
    for start, size in LOT_HISTORY.get(name, []):
        if day >= start:
            lot = size
    return lot


# =============================================================================
# Data (src/backtest/history.py: cached monthly CSVs, downloaded when missing)
# =============================================================================
def load_history(name: str, start: date, end: date, cache: str) -> tuple:
    return hist.load_history(name, start, end, cache)


# =============================================================================
# Option model (synthetic chain in the market_analysis.chain_frame layout)
# =============================================================================
class ChainModel:
    def __init__(self, name: str, vix: pd.DataFrame, cfg: settings.PBConfig, default_iv: float = 14.0):
        self.name = name
        self.cfg = cfg
        self.step = hist.DEFAULT_STRIKE_INTERVAL[get_instrument(name).name]
        self.vix = vix["close"] if vix is not None and not vix.empty else None
        self.default_iv = default_iv

    def iv_at(self, now) -> float:
        if self.vix is None:
            return self.default_iv
        value = self.vix.asof(pd.Timestamp(now) - pd.Timedelta(seconds=1))
        return float(value) if pd.notna(value) and value > 0 else self.default_iv

    def expiry_for(self, now: pd.Timestamp) -> date:
        day = now.date()
        expiry = hist.expiry_for(day, self.name)
        hours_left = (pd.Timestamp(datetime.combine(expiry, time(15, 30))).tz_localize(now.tz) - now).total_seconds() / 3600
        if (expiry == day and not self.cfg.allow_0dte) or hours_left < self.cfg.min_hours_to_expiry:
            expiry = hist.expiry_for(day + timedelta(days=1), self.name)
            if expiry <= day:
                expiry = hist.expiry_for(day + timedelta(days=2), self.name)
        return expiry

    def is_expiry_day(self, day: date) -> bool:
        return hist.expiry_for(day, self.name) == day

    def price(self, spot, strike, now, expiry, is_call, iv):
        years = opt_sel.years_to_expiry(pd.Timestamp(now), expiry)
        return opt_sel.bs_greeks(spot, strike, years, iv / 100.0, is_call)

    def chain(self, spot: float, now: pd.Timestamp, expiry: date) -> pd.DataFrame:
        iv = self.iv_at(now)
        atm = round(spot / self.step) * self.step
        rows = []
        half = self.cfg.assumed_spread_pct / 200.0
        for k in range(-self.cfg.strikes_each_side - 1, self.cfg.strikes_each_side + 2):
            strike = atm + k * self.step
            row = {"strike": strike}
            for side, is_call in (("ce", True), ("pe", False)):
                price, delta, gamma, theta, vega = self.price(spot, strike, now, expiry, is_call, iv)
                spread = max(0.05, price * half)
                row.update({f"{side}_ltp": price, f"{side}_bid": max(0.05, price - spread),
                            f"{side}_ask": price + spread, f"{side}_iv": iv, f"{side}_delta": delta,
                            f"{side}_gamma": gamma, f"{side}_theta": theta, f"{side}_vega": vega,
                            f"{side}_volume": 1e6, f"{side}_oi": 1e7, f"{side}_key": None})
            rows.append(row)
        return pd.DataFrame(rows)

    def bid(self, spot, strike, now, expiry, is_call) -> float:
        price = self.price(spot, strike, now, expiry, is_call, self.iv_at(now))[0]
        return max(0.05, price - max(0.05, price * self.cfg.assumed_spread_pct / 200.0))

    def ask(self, spot, strike, now, expiry, is_call) -> float:
        price = self.price(spot, strike, now, expiry, is_call, self.iv_at(now))[0]
        return price + max(0.05, price * self.cfg.assumed_spread_pct / 200.0)


# =============================================================================
# Simulation
# =============================================================================
@dataclass
class Trade:
    strategy: str
    day: date
    direction: str
    setup_key: str
    entry_time: str
    exit_time: str
    strike: float
    expiry: str
    lots: int
    qty: int
    entry_price: float
    exit_reason: str
    gross: float
    charges: float
    net: float
    risk_rupees: float
    r: float
    score: float
    regime: str = ""
    entry_type: str = ""
    hold_minutes: float = float("nan")


@dataclass
class RunResult:
    trades: List[Trade]
    alerts: List[sig.Alert]
    stats: Dict[str, float] = field(default_factory=dict)


def precompute_evals(ff: strat.FeatureFrame, cfg: settings.PBConfig, name: str, vix: Optional[pd.DataFrame],
                     progress: bool = True) -> List[strat.BarEval]:
    vix_close = vix["close"] if vix is not None and not vix.empty else None
    vix_daily = vix_close.groupby(vix_close.index.date).last() if vix_close is not None else None
    evals = []
    n = len(ff.bars)
    for pos in range(30, n):
        ctx = {}
        if vix_close is not None:
            t_end = ff.bars.index[pos] + pd.Timedelta(minutes=cfg.signal_minutes)
            now_v = vix_close.asof(t_end - pd.Timedelta(seconds=1))
            prev_days = vix_daily[vix_daily.index < ff.dates[pos]]
            if len(prev_days) and pd.notna(now_v):
                ctx["vix"] = float(now_v)
                ctx["vix_change_pct"] = (float(now_v) / float(prev_days.iloc[-1]) - 1) * 100
        ev = strat.evaluate_bar(ff, pos, cfg, name, ctx)
        if ev is not None:
            ev.ctx = ctx
            evals.append(ev)
        if progress and pos % 5000 == 0:
            print(f"  evaluated {pos:,}/{n:,} bars", flush=True)
    return evals


def run(evals: List[strat.BarEval], ff: strat.FeatureFrame, minutes: pd.DataFrame, model: ChainModel,
        cfg: settings.PBConfig, name: str, costs_fn, days: Optional[set] = None) -> RunResult:
    tracker = sig.StageTracker(name, cfg)
    risk = riskm.RiskManager(cfg)
    book = paper_mod.PaperBook(cfg, costs_fn)
    exchange = "BSE_FO" if name == "SENSEX" else "NSE_FO"
    trades: List[Trade] = []
    alerts: List[sig.Alert] = []
    m_index = minutes.index
    m_open, m_high, m_low, m_close = (minutes[c].to_numpy() for c in ("open", "high", "low", "close"))
    open_meta = {}
    last_day = None
    exit_ctx = None                      # from the previous COMPLETED bar only (no look-ahead into this bar)
    for ev in evals:
        day = ev.time.date()
        if days is not None and day not in days:
            continue
        if day != last_day:
            risk.roll(day)
            last_day = day
        dl = ff.levels_for(day)
        day_open = ff.v("open", int(np.searchsorted(ff.dates, day)))
        gap = (day_open - dl.pdc) / dl.daily_atr if num.finite(dl.pdc) and num.finite(dl.daily_atr) and dl.daily_atr > 0 else np.nan
        watch_block, entry_block = riskm.session_block(ev.close_time, cfg, is_expiry_day=model.is_expiry_day(day),
                                                     gap_atr_mult=gap, vix=getattr(ev, "ctx", {}).get("vix", np.nan))
        now = ev.close_time

        def resolver(direction, plan, bar_ev, _now=now):
            expiry = model.expiry_for(_now)
            chain = model.chain(bar_ev.close, _now, expiry)
            return opt_sel.select_option(chain, bar_ev.close, direction, plan, model.step, lot_on(name, day), expiry, _now,
                                     cfg, lambda s, p, q: costs_fn(s, p, q, exchange), symbol_prefix=name)

        # positions are managed minute by minute up to this bar's close before the new signal is processed
        _manage(book, open_meta, trades, m_index, m_open, m_high, m_low, m_close, model, now, exchange, ev, ff, cfg,
                exit_ctx)
        exit_ctx = exit_context(ff, ev, cfg)
        risk.update(book.realised_today(day), book.open_pnl(), len(book.open_positions()), **book.risk_inputs(day))
        out = tracker.step(ev, watch_block=watch_block, entry_block=entry_block, data_ok=True, data_reason="",
                           risk=risk, resolver=resolver)
        for a in out:
            alerts.append(a)
            if a.kind == "invalidated" and cfg.exit_on_failed_breakout:
                pos = book.find_open(a.setup_key)
                i = int(np.searchsorted(m_index, now, side="left"))
                if pos is not None and i < len(m_index) and m_index[i].date() == day:
                    info = open_meta[pos.id]
                    bid = model.bid(m_open[i], pos.strike, m_index[i], info["expiry"], pos.direction == inst.CE)
                    book.close(pos, bid=bid, now=m_index[i], exchange=exchange, reason="failed breakout")
                    _record_trade(pos, info, trades)
                continue
            if a.kind != "entry":
                continue
            # fill at the next 1-minute bar's open
            i = int(np.searchsorted(m_index, now, side="left"))
            if i >= len(m_index) or m_index[i].date() != day:
                continue
            o = dict(a.option)
            expiry = date.fromisoformat(o["expiry"])
            fill = model.ask(m_open[i], o["strike"], m_index[i], expiry, a.direction == inst.CE) * (1 + cfg.slippage_pct / 100)
            shift = fill - o["entry_premium"]
            o["entry_premium"] = fill
            for k in ("sl_premium", "t1_premium", "t2_premium"):
                o[k] = o[k] + shift  # keep the planned distances
            if o.get("ladder"):
                o["ladder"] = [p + shift for p in o["ladder"]]
            a2 = sig.Alert(**{**a.__dict__, "option": o})
            pos = book.open(a2, m_index[i].isoformat(), exchange)
            if pos is not None:
                risk.record_open(name, a.direction)
                open_meta[pos.id] = {"expiry": expiry, "risk": float(o.get("risk_rupees") or np.nan),
                                     "score": a.score or 0.0, "next_i": i, "first_i": i,
                                     "regime": a.regime or "", "entry_type": a.entry_type or ""}
    # close anything left at the end
    return RunResult(trades=trades, alerts=alerts, stats={"suppressed_duplicates": tracker.suppressed_duplicates,
                                                          "silent_missed": tracker.silent_missed})


def _manage(book: paper_mod.PaperBook, meta, trades, m_index, m_open, m_high, m_low, m_close, model: ChainModel, until,
            exchange, ev: strat.BarEval, ff, cfg, exit_ctx=None):
    for pos in book.open_positions():
        info = meta[pos.id]
        i = info["next_i"]
        is_call = pos.direction == inst.CE
        while i < len(m_index) and m_index[i] < until:
            t = m_index[i]
            # a signal bar closed just before this minute? Underlying invalidation uses completed signal closes only.
            minute_of_session = (t.hour * 60 + t.minute) - (9 * 60 + 15)
            signal_close = None
            if minute_of_session % cfg.signal_minutes == 0 and i > info["first_i"] and m_index[i - 1].date() == t.date():
                signal_close = m_close[i - 1]
            path = (m_open[i], m_low[i], m_high[i], m_close[i]) if is_call else (m_open[i], m_high[i], m_low[i], m_close[i])
            stamp = t + pd.Timedelta(seconds=59)
            for k, spot in enumerate(path):
                bid = model.bid(spot, pos.strike, stamp, info["expiry"], is_call)
                book.mark(pos, bid=bid, underlying=spot, now=stamp, exchange=exchange,
                          bar_close_underlying=signal_close, check_underlying=(k == 0 and signal_close is not None),
                          ctx=exit_ctx)
                if pos.closed:
                    break
            i += 1
            if pos.closed:
                break
        info["next_i"] = i
        if pos.closed:
            _record_trade(pos, info, trades)


def _record_trade(pos: paper_mod.PaperPosition, info: dict, trades: List[Trade]):
    net = pos.realised_gross - pos.charges
    risk_rs = info["risk"]
    trades.append(Trade(strategy="prebreakout", day=pd.Timestamp(pos.entry_time).date(), direction=pos.direction,
                        setup_key=pos.setup_key, entry_time=pos.entry_time, exit_time=pos.exit_time or "",
                        strike=pos.strike, expiry=str(info["expiry"]), lots=pos.lots,
                        qty=pos.lots * pos.lot_size, entry_price=pos.entry_price,
                        exit_reason=pos.exit_reason or "", gross=pos.realised_gross, charges=pos.charges,
                        net=net, risk_rupees=risk_rs, r=net / risk_rs if risk_rs and risk_rs > 0 else np.nan,
                        score=info["score"], regime=info.get("regime", ""), entry_type=info.get("entry_type", ""),
                        hold_minutes=((pd.Timestamp(pos.exit_time) - pd.Timestamp(pos.entry_time)).total_seconds() / 60
                                      if pos.exit_time else np.nan)))


# =============================================================================
# Baseline: plain previous-day high / low breakout, same option model and exits
# =============================================================================
def run_simple_breakout(ff: strat.FeatureFrame, minutes, model: ChainModel, cfg: settings.PBConfig, name, costs_fn,
                        days: Optional[set] = None) -> RunResult:
    exchange = "BSE_FO" if name == "SENSEX" else "NSE_FO"
    book = paper_mod.PaperBook(cfg, costs_fn)
    trades, meta = [], {}
    m_index = minutes.index
    arrays = [minutes[c].to_numpy() for c in ("open", "high", "low", "close")]
    bars = ff.bars
    done_day = {}
    for pos in range(30, len(bars)):
        t = bars.index[pos]
        day = ff.dates[pos]
        if days is not None and day not in days:
            continue
        close_t = t + pd.Timedelta(minutes=cfg.signal_minutes)
        dummy = None
        _manage(book, meta, trades, m_index, *arrays, model, close_t, exchange, dummy, ff, cfg)
        if done_day.get(day) or close_t.time() < cfg.watch_start or close_t.time() >= cfg.last_entry:
            continue
        dl = ff.levels_for(day)
        c, a = ff.v("close", pos), ff.v("atr", pos)
        prev = ff.v("close", pos - 1)
        for direction, level in ((inst.CE, dl.pdh), (inst.PE, dl.pdl)):
            if not num.finite(level) or not num.finite(a):
                continue
            sign = 1 if direction == inst.CE else -1
            if sign * (c - level) > 0 and sign * (prev - level) <= 0:
                plan = strat.UnderlyingPlan(direction, c, (c, c), level - sign * cfg.sl_atr * a, 0, 0, 0, None)
                plan.risk_points = abs(c - plan.stop)
                plan.t1, plan.t2 = c + sign * cfg.t1_r * plan.risk_points, c + sign * cfg.t2_r * plan.risk_points
                expiry = model.expiry_for(close_t)
                chain = model.chain(c, close_t, expiry)
                best, _, _ = opt_sel.select_option(chain, c, direction, plan, model.step, lot_on(name, day), expiry, close_t,
                                               cfg.replace(min_rr=0.0), lambda s, p, q: costs_fn(s, p, q, exchange))
                if best is None:
                    continue
                rm = riskm.RiskManager(cfg)
                lots, risk_rs, _ = rm.size(best)
                if lots < 1:
                    continue
                best.lots = lots
                i = int(np.searchsorted(m_index, close_t))
                if i >= len(m_index) or m_index[i].date() != day:
                    continue
                fill = model.ask(arrays[0][i], best.strike, m_index[i], expiry, direction == inst.CE) * (1 + cfg.slippage_pct / 100)
                shift = fill - best.entry_premium
                o = best.to_dict()
                o.update(entry_premium=fill, sl_premium=best.sl_premium + shift, t1_premium=best.t1_premium + shift,
                         t2_premium=best.t2_premium + shift, lots=lots)
                alert = sig.Alert(time=close_t.isoformat(), index=name, direction=direction, stage=sig.ENTRY,
                                  kind="entry", price=c, level=level, distance_atr=0, score=0, option=o,
                                  invalidation=plan.stop, setup_key=f"simple|{day}|{direction}")
                p = book.open(alert, m_index[i].isoformat(), exchange)
                if p is not None:
                    meta[p.id] = {"expiry": expiry, "risk": risk_rs, "score": 0.0, "next_i": i, "first_i": i}
                    done_day[day] = True
                break
    for t in trades:
        t.strategy = "simple PDH/PDL breakout"
    return RunResult(trades=trades, alerts=[])


# =============================================================================
# Metrics
# =============================================================================
def metrics(trades: List[Trade], capital: float, months: float) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": None, "expectancy": None, "expectancy_r": None, "profit_factor": None,
                "net": 0.0, "max_dd": 0.0, "max_dd_pct": 0.0, "avg_win_r": None, "avg_loss_r": None,
                "per_month": 0.0, "return_pct": 0.0, "charges": 0.0}
    nets = np.array([t.net for t in trades])
    rs = np.array([t.r for t in trades if num.finite(t.r)])
    wins, losses = nets[nets > 0], nets[nets <= 0]
    equity = capital + np.cumsum(nets)
    peak = np.maximum.accumulate(np.concatenate([[capital], equity]))[1:]
    dd = equity - peak
    return {
        "trades": len(trades), "win_rate": float((nets > 0).mean() * 100), "expectancy": float(nets.mean()),
        "expectancy_r": float(rs.mean()) if len(rs) else None,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else None,
        "net": float(nets.sum()), "max_dd": float(dd.min()), "max_dd_pct": float((dd / peak).min() * 100),
        "avg_win_r": float(rs[rs > 0].mean()) if (rs > 0).any() else None,
        "avg_loss_r": float(rs[rs <= 0].mean()) if (rs <= 0).any() else None,
        "per_month": len(trades) / max(months, 1e-9), "return_pct": float(nets.sum() / capital * 100),
        "charges": float(sum(t.charges for t in trades)),
    }


def signal_quality(evals: List[strat.BarEval], alerts: List[sig.Alert], ff: strat.FeatureFrame, cfg: settings.PBConfig,
                   horizon_bars: int = 6) -> dict:
    """Model-free: signed forward move (in ATR) of the underlying after each alert type, from the alert bar close."""
    pos_by_time = {ev.close_time.isoformat(): ev.position for ev in evals}
    closes = ff.arrays["close"]
    atrs = ff.arrays["atr"]
    highs, lows = ff.arrays["high"], ff.arrays["low"]

    def fwd(pos, direction):
        if pos + horizon_bars >= len(closes) or ff.dates[pos + horizon_bars] != ff.dates[pos]:
            return None
        sign = 1 if direction == inst.CE else -1
        return sign * (closes[pos + horizon_bars] - closes[pos]) / atrs[pos]

    out = {}
    for kind in ("early_watch", "high_watch", "strong_watch", "confirmed", "missed", "entry"):
        moves = [fwd(pos_by_time[a.time], a.direction) for a in alerts if a.kind == kind and a.time in pos_by_time]
        moves = [m for m in moves if m is not None and np.isfinite(m)]
        out[kind] = {"count": len([a for a in alerts if a.kind == kind]),
                     "fwd_mean_atr": float(np.mean(moves)) if moves else None,
                     "fwd_pos_pct": float(np.mean(np.array(moves) > 0) * 100) if moves else None}
    base = []
    for ev in evals[::7]:
        for d in (inst.CE, inst.PE):
            m = fwd(ev.position, d)
            if m is not None and np.isfinite(m):
                base.append(m)
    out["all_bars"] = {"count": len(base), "fwd_mean_atr": float(np.mean(base)) if base else None,
                       "fwd_pos_pct": float(np.mean(np.array(base) > 0) * 100) if base else None}
    # false breakouts: confirmed then invalidated (closed back inside) within 3 bars
    conf = {a.setup_key: a for a in alerts if a.kind == "confirmed"}
    inval = {a.setup_key: a for a in alerts if a.kind == "invalidated"}
    quick_fail = 0
    for key, a in conf.items():
        b = inval.get(key)
        if b and key in conf:
            p1, p2 = pos_by_time.get(a.time), pos_by_time.get(b.time)
            if p1 is not None and p2 is not None and p2 - p1 <= 3:
                quick_fail += 1
    out["false_breakouts_3bars"] = quick_fail
    out["false_breakout_pct"] = quick_fail / len(conf) * 100 if conf else None
    # missed moves: a breakout of a strength>=3 level that ran >= 2 ATR within the horizon, without a confirmation
    alerted = {(a.direction, pos_by_time.get(a.time)) for a in alerts if a.kind in ("confirmed", "entry")}
    big, big_alerted = 0, 0
    for ev in evals:
        for d in (inst.CE, inst.PE):
            de = ev.side(d)
            if not de.breakout:
                continue
            p = ev.position
            if p + horizon_bars >= len(closes) or ff.dates[p + horizon_bars] != ff.dates[p]:
                continue
            seg = highs[p + 1:p + horizon_bars + 1] if d == inst.CE else lows[p + 1:p + horizon_bars + 1]
            run_ = (seg.max() - closes[p]) / atrs[p] if d == inst.CE else (closes[p] - seg.min()) / atrs[p]
            if run_ >= 2.0:
                big += 1
                if (d, p) in alerted:
                    big_alerted += 1
    out["big_moves"] = big
    out["big_moves_alerted"] = big_alerted
    return out


def walk_forward(evals, ff, minutes, model, cfg, name, costs_fn, grid, train_months=6, test_months=3):
    days = sorted({ev.time.date() for ev in evals})
    if not days:
        return [], []
    start = date(days[0].year, days[0].month, 1)
    windows = []
    cursor = start
    while True:
        train_end = _add_months(cursor, train_months)
        test_end = _add_months(train_end, test_months)
        if train_end > days[-1]:
            break
        windows.append((cursor, train_end, test_end))
        cursor = _add_months(cursor, test_months)
    oos_trades, rows = [], []
    for train_start, train_end, test_end in windows:
        train_days = {d for d in days if train_start <= d < train_end}
        test_days = {d for d in days if train_end <= d < test_end}
        if not test_days:
            continue
        best, best_key = None, None
        for params in grid:
            c = cfg.replace(**params)
            res = run(evals, ff, minutes, model, c, name, costs_fn, train_days)
            m = metrics(res.trades, cfg.capital, train_months)
            if m["trades"] < 15:
                continue
            key = m["expectancy"]
            if best_key is None or key > best_key:
                best, best_key = params, key
        chosen = best or {}
        res = run(evals, ff, minutes, model, cfg.replace(**chosen), name, costs_fn, test_days)
        mt = metrics(res.trades, cfg.capital, test_months)
        rows.append({"train": f"{train_start:%Y-%m} - {train_end - timedelta(days=1):%Y-%m}",
                     "test": f"{train_end:%Y-%m} - {test_end - timedelta(days=1):%Y-%m}",
                     "chosen": chosen or "default (no combo had 15 train trades)", "test_trades": mt["trades"],
                     "test_net": mt["net"], "test_pf": mt["profit_factor"]})
        oos_trades += res.trades
    return oos_trades, rows


def _add_months(d: date, n: int) -> date:
    month = d.month - 1 + n
    return date(d.year + month // 12, month % 12 + 1, 1)


# =============================================================================
# Robustness
# =============================================================================
def robustness(trades: List[Trade], model: ChainModel, cfg: settings.PBConfig) -> dict:
    nets = [t.net for t in trades]

    def dte(t):
        try:
            return (date.fromisoformat(t.expiry) - t.day).days
        except ValueError:
            return None

    return {
        "ci": bs.bootstrap_ci(nets),
        "mc": bs.monte_carlo_drawdown(nets, cfg.capital, loss_limit=cfg.max_daily_loss_rupees),
        "segments": {
            "Direction": bs.segment_table(trades, lambda t: t.direction, order=[inst.CE, inst.PE]),
            "Entry time": bs.segment_table(trades, lambda t: bs.bucket_time(t.entry_time), order=bs.TIME_ORDER),
            "Days to expiry": bs.segment_table(trades, lambda t: bs.bucket_dte(dte(t)), order=bs.DTE_ORDER),
            "VIX at entry": bs.segment_table(trades, lambda t: bs.bucket_vix(model.iv_at(t.entry_time)),
                                             order=bs.VIX_ORDER),
            "Year": bs.segment_table(trades, lambda t: t.day.year),
        },
    }


def exit_variants(cfg: settings.PBConfig) -> List[tuple]:
    """The current config, then each optional exit rule flipped relative to it (one change per row)."""
    fb = not cfg.exit_on_failed_breakout
    out = [("current config", {}),
           (f"failed-breakout exit {'ON' if fb else 'OFF'}", {"exit_on_failed_breakout": fb})]
    for pct in (10.0, 20.0, 30.0):
        if abs(cfg.trail_after_t1_pct - pct) > 1e-9:
            out.append((f"trail {pct:.0f}% after T1", {"trail_after_t1_pct": pct}))
    if cfg.trail_after_t1_pct > 0:
        out.append(("no trail (breakeven after T1)", {"trail_after_t1_pct": 0.0}))
    return out


def exit_check(evals, ff, minutes, model, cfg, name, costs_fn) -> List[dict]:
    """Each exit rule on the first 70% / last 30% of the sessions. A rule earns its place only if it helps in both."""
    days = sorted({ev.time.date() for ev in evals})
    if not days:
        return []
    cut = days[int(len(days) * 0.7)]
    first = {d for d in days if d < cut}
    last = {d for d in days if d >= cut}
    rows = []
    for label, change in exit_variants(cfg):
        c = cfg.replace(**change)
        a = run(evals, ff, minutes, model, c, name, costs_fn, first).trades
        b = run(evals, ff, minutes, model, c, name, costs_fn, last).trades
        rows.append({"variant": label, "first": bs.summary([t.net for t in a]), "last": bs.summary([t.net for t in b])})
    return rows


def _robust_html(rb: dict, title: str) -> str:
    ci, mc = rb["ci"], rb["mc"]

    def band(pair, pat):
        return "-" if pair[0] is None else f"{pat.format(pair[0])} to {pat.format(pair[1])}"

    out = f"<h3>{html.escape(title)}</h3>"
    if ci.get("note"):
        return out + f"<p>{html.escape(ci['note'])}</p>"
    out += ("<table><tr><th>95% bootstrap interval</th><th>Range</th></tr>"
            f"<tr><td>Expectancy Rs / trade</td><td>{band(ci['expectancy'], '{:,.0f}')}</td></tr>"
            f"<tr><td>Profit factor</td><td>{band(ci['profit_factor'], '{:.2f}')}</td></tr>"
            f"<tr><td>Win rate %</td><td>{band(ci['win_rate'], '{:.1f}')}</td></tr>"
            f"<tr><td>Chance the true expectancy is &le; 0</td><td>{ci['p_expectancy_le_0'] * 100:.0f}%</td></tr></table>")
    if not mc.get("note"):
        out += ("<table><tr><th>Monte Carlo (2,000 reshuffles of the same trades)</th><th>Actual order</th>"
                "<th>Median</th><th>Worst 5%</th><th>Worst</th></tr>"
                f"<tr><td>Max drawdown Rs</td><td>{mc['actual_max_dd']:,.0f}</td><td>{mc['dd_median']:,.0f}</td>"
                f"<td>{mc['dd_p95']:,.0f}</td><td>{mc['dd_worst']:,.0f}</td></tr>"
                f"<tr><td>Longest losing streak</td><td>{mc['actual_losing_streak']}</td><td>{mc['streak_median']:.0f}</td>"
                f"<td>{mc['streak_p95']:.0f}</td><td>-</td></tr></table>")
        if mc.get("loss_limit"):
            out += (f"<p>Share of reshuffles whose drawdown went past Rs {mc['loss_limit']:,.0f} (the daily loss limit, "
                    f"here measured over the whole test): <b>{mc['p_dd_beyond_limit'] * 100:.0f}%</b>.</p>")
    for label, rows in rb["segments"].items():
        body = "".join(f"<tr><td>{html.escape(r['group'])}</td><td>{r['trades']}</td><td>{_fmt(r['win_rate'], '{:.1f}')}</td>"
                       f"<td>{_fmt(r['expectancy'])}</td><td>{_fmt(r['profit_factor'], '{:.2f}')}</td>"
                       f"<td>{_fmt(r['net'])}</td></tr>" for r in rows)
        out += (f"<table><tr><th>{html.escape(label)}</th><th>Trades</th><th>Win %</th><th>Expectancy</th><th>PF</th>"
                f"<th>Net</th></tr>{body}</table>")
    return out


def _exit_check_html(rows: List[dict]) -> str:
    if not rows:
        return "<p>Skipped (--no-exit-check).</p>"
    body = ""
    for r in rows:
        f, l = r["first"], r["last"]
        body += (f"<tr><td>{html.escape(r['variant'])}</td><td>{f['trades']}</td><td>{_fmt(f['profit_factor'], '{:.2f}')}</td>"
                 f"<td>{_fmt(f['net'])}</td><td>{l['trades']}</td><td>{_fmt(l['profit_factor'], '{:.2f}')}</td>"
                 f"<td>{_fmt(l['net'])}</td></tr>")
    return ("<table><tr><th>Exit rule</th><th>Trades (first 70%)</th><th>PF</th><th>Net</th>"
            f"<th>Trades (last 30%)</th><th>PF</th><th>Net</th></tr>{body}</table>")


# =============================================================================
# Report
# =============================================================================
def _fmt(v, pattern="{:,.0f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return pattern.format(v)


def _metrics_rows(columns: Dict[str, dict]) -> str:
    keys = [("trades", "Trades", "{:,.0f}"), ("per_month", "Trades / month", "{:.1f}"), ("win_rate", "Win rate %", "{:.1f}"),
            ("expectancy", "Expectancy Rs / trade", "{:,.0f}"), ("expectancy_r", "Expectancy (R)", "{:.2f}"),
            ("avg_win_r", "Avg win (R)", "{:.2f}"), ("avg_loss_r", "Avg loss (R)", "{:.2f}"),
            ("profit_factor", "Profit factor", "{:.2f}"), ("net", "Net P&L Rs", "{:,.0f}"),
            ("return_pct", "Return % of capital", "{:.1f}"), ("max_dd", "Max drawdown Rs", "{:,.0f}"),
            ("max_dd_pct", "Max drawdown %", "{:.1f}"), ("charges", "Charges paid Rs", "{:,.0f}")]
    head = "<tr><th></th>" + "".join(f"<th>{html.escape(n)}</th>" for n in columns) + "</tr>"
    body = ""
    for key, label, pat in keys:
        body += f"<tr><td>{label}</td>" + "".join(f"<td>{_fmt(m.get(key), pat)}</td>" for m in columns.values()) + "</tr>"
    return f"<table>{head}{body}</table>"


def _equity_svg(series: Dict[str, List[Trade]], capital: float, width=900, height=260) -> str:
    lines = []
    all_pts = []
    for trades in series.values():
        eq = capital + np.cumsum([t.net for t in trades]) if trades else np.array([capital])
        all_pts.append(eq)
    lo = min(min(e) for e in all_pts + [np.array([capital])])
    hi = max(max(e) for e in all_pts + [np.array([capital])])
    span = (hi - lo) or 1
    colours = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    for k, (label, eq) in enumerate(zip(series.keys(), all_pts)):
        n = max(len(eq) - 1, 1)
        pts = " ".join(f"{40 + i / n * (width - 60):.1f},{height - 25 - (v - lo) / span * (height - 50):.1f}"
                       for i, v in enumerate(eq))
        lines.append(f'<polyline fill="none" stroke="{colours[k % 4]}" stroke-width="1.6" points="{pts}"/>')
        lines.append(f'<text x="{50 + 230 * k}" y="16" fill="{colours[k % 4]}" font-size="12">{html.escape(label)}</text>')
    base_y = height - 25 - (capital - lo) / span * (height - 50)
    lines.append(f'<line x1="40" x2="{width - 20}" y1="{base_y:.1f}" y2="{base_y:.1f}" stroke="#999" stroke-dasharray="4"/>')
    lines.append(f'<text x="2" y="{height - 25}" font-size="11" fill="#666">{lo:,.0f}</text>')
    lines.append(f'<text x="2" y="30" font-size="11" fill="#666">{hi:,.0f}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Equity curves">{"".join(lines)}</svg>'


def write_report(path: str, ctx: dict):
    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111;background:#fafafa;max-width:1150px}"
           "table{border-collapse:collapse;margin:8px 0 18px;background:#fff}td,th{border:1px solid #ddd;padding:5px 9px;"
           "font-size:13px;text-align:right}th{background:#f1f5f9}td:first-child{text-align:left}"
           ".warn{background:#fff7ed;border-left:4px solid #f97316;padding:10px 14px;margin:12px 0}"
           ".ok{background:#f0fdf4;border-left:4px solid #16a34a;padding:10px 14px;margin:12px 0}"
           "h2{margin-top:28px}code{background:#eef;padding:1px 4px}")
    sq = ctx["quality"]
    qrows = "".join(
        f"<tr><td>{k}</td><td>{v['count']:,}</td><td>{_fmt(v['fwd_mean_atr'], '{:+.3f}')}</td>"
        f"<td>{_fmt(v['fwd_pos_pct'], '{:.1f}')}</td></tr>"
        for k, v in sq.items() if isinstance(v, dict))
    wf_rows = "".join(f"<tr><td>{html.escape(r['train'])}</td><td>{html.escape(r['test'])}</td>"
                      f"<td>{html.escape(str(r['chosen']))}</td><td>{r['test_trades']}</td><td>{_fmt(r['test_net'])}</td>"
                      f"<td>{_fmt(r['test_pf'], '{:.2f}')}</td></tr>" for r in ctx["wf_rows"])
    grid_rows = "".join(f"<tr><td>{html.escape(str(p))}</td><td>{m['trades']}</td><td>{_fmt(m['win_rate'], '{:.1f}')}</td>"
                        f"<td>{_fmt(m['expectancy'])}</td><td>{_fmt(m['profit_factor'], '{:.2f}')}</td>"
                        f"<td>{_fmt(m['net'])}</td><td>{_fmt(m['max_dd_pct'], '{:.1f}')}</td></tr>"
                        for p, m in ctx["grid"])
    exit_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in ctx["exit_reasons"].items())
    verdict_cls = "ok" if ctx["profitable_oos"] else "warn"
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Pre-breakout backtest {html.escape(ctx['name'])}</title>
<style>{css}</style></head><body>
<h1>Pre-breakout CE / PE backtest - {html.escape(ctx['name'])}</h1>
<p>{ctx['start']} to {ctx['end']} &middot; {ctx['days']} sessions &middot; capital Rs {ctx['capital']:,.0f} &middot;
risk {ctx['risk_pct']}% per trade &middot; generated {ctx['generated']}</p>
<div class="{verdict_cls}"><b>Verdict:</b> {html.escape(ctx['verdict'])}</div>
<div class="warn"><b>Model limits.</b> Option prices are Black-Scholes on the index with IV = India VIX
(no historical option prices or bid/ask). Spread {ctx['spread']}% and slippage {ctx['slip']}% per leg are charged,
plus all statutory charges. <b>Futures volume / OI / VWAP are not in this history</b>, so the participation family
(10 points) and the volume / VWAP confirmation checks were not used: the live system is stricter than this test.
Scores are heuristic setup-strength scores, not probabilities.</div>
<h2>Results</h2>
{_metrics_rows(ctx['columns'])}
{_equity_svg(ctx['equity'], ctx['capital'])}
<p>Buy-and-hold {html.escape(ctx['name'])} over the same period: <b>{ctx['buy_hold']:+.1f}%</b> (index points, no costs;
a different risk profile, shown for scale only).</p>
<h2>Walk-forward (out-of-sample only)</h2>
<p>Each window picks <code>entry_min_score</code> and <code>max_extension_atr</code> on the previous 6 months
(highest expectancy with at least 15 trades) and trades the next 3 months with them.</p>
<table><tr><th>Train</th><th>Test</th><th>Chosen</th><th>Test trades</th><th>Test net Rs</th><th>Test PF</th></tr>{wf_rows}</table>
<h2>Signal quality (model-free)</h2>
<p>Signed move of the index {ctx['horizon']} bars ({ctx['horizon'] * ctx['signal_minutes']} min) after each alert,
in ATR, from the alert bar's close. "all bars" is the same measure on every bar in both directions (the no-skill baseline).</p>
<table><tr><th>Alert</th><th>Count</th><th>Mean fwd move (ATR)</th><th>% positive</th></tr>{qrows}</table>
<p>False breakouts (confirmed, then closed back inside within 3 bars): <b>{sq['false_breakouts_3bars']}</b>
({_fmt(sq['false_breakout_pct'], '{:.1f}')}% of confirmations).
Big moves (a level break that ran 2+ ATR within the horizon): {sq['big_moves']}, of which alerted as CONFIRMED / ENTRY:
{sq['big_moves_alerted']}. Duplicate alerts suppressed by the de-duplication: {ctx['dups']}.
Weak, unwatched extended breaks recorded silently: {ctx['silent_missed']}.</p>
<h2>Exit reasons (system, full period)</h2><table><tr><th>Reason</th><th>Trades</th></tr>{exit_rows}</table>
<h2>Robustness</h2>
<p>How much of the result could be luck. The bootstrap resamples the trades; the Monte Carlo keeps the same trades and
shuffles their order. A system whose expectancy interval includes zero has not shown an edge yet.</p>
{_robust_html(ctx['robust_wf'], 'Walk-forward out-of-sample trades') if ctx.get('robust_wf') else ''}
{_robust_html(ctx['robust_sys'], 'System, full period')}
<h2>Exit-rule check (first 70% vs last 30% of sessions)</h2>
<p>Each row changes one exit setting from the current config. Keep a rule only if it helps in both halves.</p>
{_exit_check_html(ctx.get('exit_check', []))}
<h2>Sensitivity (full period, in-sample - for robustness only, do not pick values from it)</h2>
<table><tr><th>Parameters</th><th>Trades</th><th>Win %</th><th>Expectancy</th><th>PF</th><th>Net</th><th>Max DD %</th></tr>{grid_rows}</table>
<p>Trades: <code>{html.escape(os.path.basename(ctx['trades_csv']))}</code></p>
</body></html>"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(doc)


def write_trades(path: str, sets: Dict[str, List[Trade]]):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["set", "strategy", "day", "direction", "entry_time", "exit_time", "strike", "expiry", "lots",
                         "qty", "entry_price", "exit_reason", "gross", "charges", "net", "risk_rupees", "r", "score",
                         "setup_key"])
        for label, trades in sets.items():
            for t in trades:
                writer.writerow([label, t.strategy, t.day, t.direction, t.entry_time, t.exit_time, t.strike, t.expiry,
                                 t.lots, t.qty, round(t.entry_price, 2), t.exit_reason, round(t.gross, 2),
                                 round(t.charges, 2), round(t.net, 2), round(t.risk_rupees, 2) if num.finite(t.risk_rupees) else "",
                                 round(t.r, 3) if num.finite(t.r) else "", t.score, t.setup_key])


# =============================================================================
# Main
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.backtest", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instrument", default="NIFTY", type=str.upper, choices=list(SUPPORTED_INDICES))
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cache", default=str(settings.HISTORY_DIR))
    p.add_argument("--config", default=CONFIG_FILE)
    p.add_argument("--no-walk-forward", action="store_true")
    p.add_argument("--no-exit-check", action="store_true", help="skip the exit-rule comparison (saves time)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = parse_args(argv)
    cfg = settings.PBConfig.from_ini(args.config)
    name = args.instrument
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    print(f"Loading {name} 1-minute history {start} .. {end} ...", flush=True)
    minutes, vix = load_history(name, start, end, args.cache)
    if minutes.empty:
        print("No data.")
        return 1
    print(f"  {len(minutes):,} minutes, {len(set(minutes.index.date))} sessions; VIX rows {len(vix):,}", flush=True)
    costs_fn = make_cost_fn(settings.load_app_settings(args.config).costs)
    model = ChainModel(name, vix, cfg)
    ff = strat.build_features(minutes, cfg)
    print(f"  {len(ff.bars):,} {cfg.signal_minutes}-minute bars; evaluating ...", flush=True)
    evals = precompute_evals(ff, cfg, name, vix)
    days_all = sorted({ev.time.date() for ev in evals})
    months = max(len(days_all) / 21.0, 1)

    print("Running the system ...", flush=True)
    base = run(evals, ff, minutes, model, cfg, name, costs_fn)
    print("Running score v2 (5m + 1m; only its EMA / RSI / structure components exist in this history) ...", flush=True)
    from src.scanner import score_v2 as v2
    mf = v2.build_minutes(minutes, None)
    evals_v2 = [v2.rescore(ev, ff, mf, getattr(ev, "ctx", {})) for ev in evals]
    base_v2 = run(evals_v2, ff, minutes, model, cfg, name, costs_fn)
    print("Running the simple PDH/PDL breakout baseline ...", flush=True)
    simple = run_simple_breakout(ff, minutes, model, cfg, name, costs_fn)
    grid = [{"entry_min_score": s, "max_extension_atr": x} for s in (60, 65, 70, 75) for x in (0.5, 0.8)]
    grid_results = []
    print("Sensitivity grid ...", flush=True)
    for params in grid:
        res = run(evals, ff, minutes, model, cfg.replace(**params), name, costs_fn)
        grid_results.append((params, metrics(res.trades, cfg.capital, months)))
    wf_trades, wf_rows = ([], [])
    if not args.no_walk_forward:
        print("Walk-forward ...", flush=True)
        wf_trades, wf_rows = walk_forward(evals, ff, minutes, model, cfg, name, costs_fn, grid)
    wf_months = len({t.day for t in wf_trades}) and len({d for d in days_all if wf_trades and d >= min(t.day for t in wf_trades)}) / 21.0
    quality = signal_quality(evals, base.alerts, ff, cfg)

    m_sys = metrics(base.trades, cfg.capital, months)
    m_simple = metrics(simple.trades, cfg.capital, months)
    m_v2 = metrics(base_v2.trades, cfg.capital, months)
    m_wf = metrics(wf_trades, cfg.capital, wf_months or 1)
    first, last = minutes["close"].iloc[0], minutes["close"].iloc[-1]
    buy_hold = (last / first - 1) * 100
    robust_sys = robustness(base.trades, model, cfg)
    robust_wf = robustness(wf_trades, model, cfg) if wf_trades else None
    exit_rows = [] if args.no_exit_check else (print("Exit-rule check ...", flush=True) or
                                               exit_check(evals, ff, minutes, model, cfg, name, costs_fn))
    # a real edge, not a coin flip after costs: PF >= 1.2 on >= 100 out-of-sample trades, and the bootstrap
    # 95% interval of the out-of-sample expectancy must lie above zero
    wf_ci_low = (robust_wf or {}).get("ci", {}).get("expectancy", (None, None))[0]
    profitable_oos = bool(m_wf["trades"] >= 100 and (m_wf["profit_factor"] or 0) >= 1.2 and m_wf["net"] > 0
                          and wf_ci_low is not None and wf_ci_low > 0)
    if profitable_oos:
        verdict = (f"Walk-forward out-of-sample was positive after modelled costs ({m_wf['trades']} trades, PF "
                   f"{m_wf['profit_factor']:.2f}). Still a modelled result: paper-trade it on live option prices before "
                   f"trusting it.")
    else:
        verdict = (f"NOT shown to be profitable: walk-forward out-of-sample PF {m_wf['profit_factor'] or 0:.2f} on "
                   f"{m_wf['trades']} trades (needs >= 1.2 on >= 100 trades after modelled costs, with the 95% "
                   f"interval of expectancy above zero). Treat the alerts "
                   f"as a watchlist and keep it on paper.")
    exit_reasons: Dict[str, int] = {}
    for t in base.trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    suffix = "" if name == "NIFTY" else f"_{name}"
    trades_csv = os.path.join(OUTPUT_DIR, f"prebreakout_backtest_trades{suffix}.csv")
    report = os.path.join(OUTPUT_DIR, f"prebreakout_backtest_report{suffix}.html")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_trades(trades_csv, {"system": base.trades, "walk_forward": wf_trades, "score_v2": base_v2.trades,
                              "simple": simple.trades})
    write_report(report, {
        "name": name, "start": days_all[0], "end": days_all[-1], "days": len(days_all), "capital": cfg.capital,
        "risk_pct": cfg.risk_per_trade_pct, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "verdict": verdict, "profitable_oos": profitable_oos, "spread": cfg.assumed_spread_pct, "slip": cfg.slippage_pct,
        "columns": {"System (current config, full period)": m_sys, "System walk-forward (OOS)": m_wf,
                    "Score v2 (spot-only parts), full period": m_v2, "Simple PDH/PDL breakout": m_simple},
        "equity": {"system": base.trades, "walk-forward OOS": wf_trades, "simple breakout": simple.trades},
        "buy_hold": buy_hold, "wf_rows": wf_rows, "quality": quality, "horizon": 6, "signal_minutes": cfg.signal_minutes,
        "dups": base.stats.get("suppressed_duplicates", 0), "silent_missed": base.stats.get("silent_missed", 0),
        "grid": grid_results, "exit_reasons": exit_reasons, "trades_csv": trades_csv,
        "robust_sys": robust_sys, "robust_wf": robust_wf, "exit_check": exit_rows,
    })
    print()
    for label, m in (("system", m_sys), ("walk-forward OOS", m_wf), ("score v2", m_v2), ("simple breakout", m_simple)):
        print(f"{label:>18}: trades {m['trades']:4d}  win {_fmt(m['win_rate'], '{:.1f}')}%  "
              f"exp Rs {_fmt(m['expectancy'])}  PF {_fmt(m['profit_factor'], '{:.2f}')}  net Rs {_fmt(m['net'])}  "
              f"maxDD {_fmt(m['max_dd_pct'], '{:.1f}')}%")
    print(f"buy & hold: {buy_hold:+.1f}%")
    for label, rb in (("system", robust_sys), ("walk-forward OOS", robust_wf)):
        if rb and rb["ci"].get("expectancy", (None,))[0] is not None:
            lo, hi = rb["ci"]["expectancy"]
            print(f"{label:>18}: expectancy 95% CI Rs {lo:,.0f} .. {hi:,.0f}; "
                  f"Monte Carlo worst-5% drawdown Rs {rb['mc']['dd_p95']:,.0f}")
    for r in exit_rows:
        print(f"  exit check {r['variant']:>32}: PF first 70% {_fmt(r['first']['profit_factor'], '{:.2f}')}, "
              f"last 30% {_fmt(r['last']['profit_factor'], '{:.2f}')}")
    print(verdict)
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
