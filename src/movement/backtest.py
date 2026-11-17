"""
Historical backtest of the movement service's two paper-trading books (momentum and scalp).

    python -m src.movement.backtest --instrument NIFTY
    python -m src.movement.backtest --instrument SENSEX --book scalp
    python -m src.movement.backtest --instrument NIFTY --start 2025-01-01 --end 2026-10-31
    python -m src.movement.backtest --instrument NIFTY --set PAPER_SCALP.stop_pct=10 --set MOVEMENT.min_strength=2
    python -m src.movement.backtest --instrument NIFTY --strategies all
    python -m src.movement.backtest --instrument SENSEX --strategies supertrend,ema_9_21 --set STRATEGY_SUPERTREND.mult=2

Writes output/movement_backtest_<INDEX>[_<book>].txt (the report) and ..._trades.csv (one row per trade).
With --strategies: output/strategy_backtest_<INDEX>.txt / _trades.csv, one paper book per strategy (the
[STRATEGIES] / [STRATEGY_<KEY>] exits), a summary table ranked on the out-of-sample 30%, and a full report per
strategy (Sharpe / Sortino on daily P&L, consecutive wins / losses, Monte Carlo drawdown).

How it works
  * The SAME code as the live service: every minute of the 1-minute history goes through
    MovementMonitor.run_once() (detector.index_metrics / evaluate, the per-window cooldown, the scalp trigger
    rules, the strongest-trigger-per-book choice, opposite-trigger exits) and its PaperTrader (filters, entry
    blocks, stops / target / breakeven / trailing / fade / time stop / square-off, daily loss limit).
    Nothing here re-implements a rule; this file only supplies the data and the option prices.
  * Clock: the monitor runs at each minute boundary with the last price = the close of the candle that just
    ended. Between boundaries open trades are marked twice more on the candle's adverse extreme and then its
    favourable extreme (CE: low then high, PE: high then low), so a candle that touches both the stop and the
    target counts as the stop.
  * Options are MODELLED: there is no historical option-chain data in this project. Premiums are Black-Scholes
    on the index price with IV = India VIX at that minute (src/options/selection.py bs_greeks via the
    backtest's ChainModel), nearest weekly expiry (next one on expiry day / with < min_hours_to_expiry left,
    as [OPTIONS] says), a strike chosen by the live [OPTIONS] delta / premium filters, a bid-ask spread of
    assumed_spread_pct around the model price, slippage_pct on both legs, and [COSTS] charges
    (src/execution/costs.py). Not modelled: IV smile / skew, IV moves within the day other than VIX,
    real spreads (wider on fast moves - exactly when these books trade), liquidity, latency of a 10 s poll.
  * Index spot history has no futures volume / OI, so the rvol check cannot be evaluated. With it switched on
    (the ini default) every trigger is UNCONFIRMED and neither book would ever trade, so the backtest turns
    it off ([MOVEMENT] min_rvol and [PAPER_SCALP] trigger_min_rvol = 0) unless --keep-rvol is given.
    OI confirmation is off in the ini already. The report says so.
  * Days are independent (every trade is squared off the same day, cooldowns are minutes), so months run in
    parallel worker processes.
  * Strategies (--strategies) run in the live StrategyEngine with STRATEGY_HISTORY_SESSIONS previous sessions of
    1-minute candles and daily candles built from them (only days BEFORE the replayed day). What cannot be
    tested on index history: futures volume / OI (VWAP, volume profile, OI build-up strategies give no
    signal), real option-chain OI (PCR, gamma / IV), and the two-index pair strategies (one index is replayed
    at a time). The report lists every strategy that had no trades and why.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import logging
import math
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest import history as hist
from src.backtest.runner import ChainModel, lot_on
from src.broker.market_data import IndexSnapshot, MarketSnapshot, empty_minutes
from src.config import settings
from src.config.instruments import SUPPORTED_INDICES, VIX_KEY, get_instrument
from src.config.settings import PBConfig
from src.execution.costs import make_cost_fn
from src.movement import paper as paper_mod
from src.movement.monitor import Broadcaster, MovementMonitor
from src.movement.rules import from_ini
from src.scanner import indicators as ind
from src.utils.numbers import finite

HISTORY_DIRS = (settings.HISTORY_DIR, settings.PROJECT_ROOT / "orb_data")   # orb_data: older NIFTY / VIX cache
NO_FUTURES = {"MOVEMENT": {"min_rvol": "0"}, "PAPER_SCALP": {"trigger_min_rvol": "0"}}
CHAIN_STRIKES_EACH_SIDE = 15     # quotes for open trades stay available after the index has moved
TRAIN_SHARE = 0.7
# the detector reads the last 200 finished candles (ATR / RSI) and at most a 120-min window: more is unused
DETECTOR_CANDLES = 260
CHAIN_OI_STRATEGIES = {"pcr", "gamma_iv"}   # read real option-chain OI: the modelled chain has a flat placeholder
STRATEGY_HISTORY_SESSIONS = 6     # previous sessions of 1-min candles the strategy engine sees (EMA200 on 5-min)


# =============================================================================
# Data
# =============================================================================
def load_minutes(key: str, start: date, end: date, dirs=HISTORY_DIRS) -> pd.DataFrame:
    """1-minute session candles for one instrument from the monthly CSV caches (first directory wins)."""
    parts = [hist.read_cached(key, start, end, d) for d in dirs if Path(d).exists()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return empty_minutes()
    frame = pd.concat(parts[::-1]).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def load_data(name: str, start: date, end: date, download: bool = False) -> tuple:
    key = hist.index_key(name)
    spot = load_minutes(key, start, end)
    vix = load_minutes(VIX_KEY, start - timedelta(days=10), end)
    if (spot.empty or vix.empty) and download:
        return hist.load_history(name, start, end, settings.HISTORY_DIR)
    return spot, vix


# =============================================================================
# Settings: the ini with the backtest's overrides, parsed by the live loaders
# =============================================================================
def backtest_settings(config=settings.CONFIG_FILE, overrides: Optional[Dict[str, Dict[str, str]]] = None,
                      keep_rvol: bool = False) -> tuple:
    """(rules, profiles, cfg, costs) from the ini plus overrides {section: {key: value}}."""
    parser = settings.read_ini(config)
    changes = {} if keep_rvol else {s: dict(v) for s, v in NO_FUTURES.items()}
    for section, values in (overrides or {}).items():
        changes.setdefault(section, {}).update(values)
    for section, values in changes.items():
        if not parser.has_section(section):
            parser.add_section(section)
        for key, value in values.items():
            parser[section][key] = str(value)
    fd, path = tempfile.mkstemp(suffix=".ini", prefix="movement_backtest_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            parser.write(f)
        rules, profiles, cfg = from_ini(path), paper_mod.load_profiles(path), PBConfig.from_ini(path)
        costs = settings.load_app_settings(path).costs
    finally:
        os.unlink(path)
    return rules, profiles, cfg, costs


def strategy_settings(config=settings.CONFIG_FILE, overrides: Optional[Dict[str, Dict[str, str]]] = None,
                      keys: Optional[List[str]] = None) -> tuple:
    """(StrategyEngine, {key: Profile}) for `keys` ("all" or a list) from the ini plus overrides."""
    from src.movement import strategies as strategies_mod
    parser = settings.read_ini(config)
    if not parser.has_section("STRATEGIES"):
        parser.add_section("STRATEGIES")
    for section, values in (overrides or {}).items():
        if not parser.has_section(section):
            parser.add_section(section)
        for key, value in values.items():
            parser[section][key] = str(value)
    if keys:
        wanted = list(strategies_mod.REGISTRY) if keys == ["all"] else keys
        unknown = [k for k in wanted if k not in strategies_mod.REGISTRY]
        if unknown:
            raise ValueError(f"unknown strategy: {', '.join(unknown)}")
        parser["STRATEGIES"]["enabled"] = ", ".join(wanted)
        for k in wanted:                       # chosen on the command line = run it, even if the ini disables it
            sec = f"STRATEGY_{k.upper()}"
            if parser.has_section(sec) and "enabled" in parser[sec]:
                parser[sec]["enabled"] = "true"
    fd, path = tempfile.mkstemp(suffix=".ini", prefix="strategy_backtest_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            parser.write(f)
        engine, profiles = strategies_mod.load_config(path)
    finally:
        os.unlink(path)
    if engine is None:
        raise ValueError("no strategy enabled")
    return engine, profiles


def parse_overrides(items: List[str]) -> Dict[str, Dict[str, str]]:
    """['PAPER_SCALP.stop_pct=10'] -> {'PAPER_SCALP': {'stop_pct': '10'}}."""
    out: Dict[str, Dict[str, str]] = {}
    for item in items or []:
        name, eq, value = item.partition("=")
        section, dot, key = name.partition(".")
        if not (eq and dot and section and key):
            raise ValueError(f"--set {item!r} should look like SECTION.key=value, e.g. PAPER_SCALP.stop_pct=10")
        out.setdefault(section.strip().upper(), {})[key.strip()] = value.strip()
    return out


# =============================================================================
# Replay
# =============================================================================
class _Store:
    """In-memory stand-in for the event store: counts the triggers the monitor fires."""

    def __init__(self):
        self.events: List[dict] = []

    def add(self, e: dict) -> int:
        self.events.append({k: e.get(k) for k in ("index", "window", "direction", "quality", "time")})
        return len(self.events)

    def recent(self, *args, **kwargs):
        return []


class _Quiet(Broadcaster):
    def publish(self, kind, data):
        pass


class ReplayProvider:
    """Serves the monitor one historical minute at a time, with a modelled option chain."""
    mock = False

    def __init__(self, name: str, spot: pd.DataFrame, vix: pd.DataFrame, cfg: PBConfig):
        self.name = get_instrument(name).name
        self.spot = spot
        self.vix = vix["close"] if vix is not None and not vix.empty else None
        self.model = ChainModel(self.name, vix, replace(cfg, strikes_each_side=CHAIN_STRIKES_EACH_SIDE))
        self.exchange = "BSE_FO" if self.name == "SENSEX" else "NSE_FO"
        self.t: Optional[pd.Timestamp] = None
        self.window = spot            # the candles the detector may see today (previous session + today)
        self.ends = 0                 # candles of `window` that had ended by self.t
        self.prev_close = np.nan
        self.history_candles = DETECTOR_CANDLES
        self.daily_all = ind.daily_from_minutes(spot)
        self.daily = self.daily_all.iloc[:0]

    def set_day(self, window: pd.DataFrame, prev_close: float):
        self.window, self.prev_close = window, prev_close
        today = window.index[-1].date()
        self.daily = self.daily_all[self.daily_all.index.date < today]      # only finished days

    def now(self):
        return self.t

    def ltp(self) -> float:
        return float(self.window["close"].iloc[self.ends - 1])

    def vix_at(self, now) -> float:
        if self.vix is None:
            return np.nan
        value = self.vix.asof(now - pd.Timedelta(seconds=1))
        return float(value) if pd.notna(value) else np.nan

    def key(self, strike: float, expiry, side: str) -> str:
        return f"{self.name}|{expiry}|{strike:g}|{side.upper()}"

    def chain(self, spot: float, now: pd.Timestamp, expiry: date) -> pd.DataFrame:
        chain = self.model.chain(spot, now, expiry)
        for side in ("ce", "pe"):
            chain[f"{side}_key"] = [self.key(k, expiry.isoformat(), side) for k in chain["strike"]]
        return chain

    def snapshot(self, now=None, want_chain=None) -> MarketSnapshot:
        now = now or self.t
        price = self.ltp()
        isnap = IndexSnapshot(name=self.name, spot_minutes=self.window.iloc[max(0, self.ends - self.history_candles):self.ends],
                              fut_minutes=None, daily=self.daily, ltp=price, ltp_time=now, prev_close=self.prev_close,
                              lot_size=lot_on(self.name, now.date()), step=self.model.step, exchange=self.exchange)
        if want_chain is None or self.name in want_chain:      # None = every index (live provider semantics)
            isnap.expiry = isnap.nearest_expiry = self.model.expiry_for(now)
            isnap.chain, isnap.chain_time, isnap.chain_source = self.chain(price, now, isnap.expiry), now, "model"
        return MarketSnapshot(time=now, indices={self.name: isnap}, vix=self.vix_at(now))

    def bid(self, t: paper_mod.Trade, spot: float, now: pd.Timestamp) -> float:
        return self.model.bid(spot, t.strike, now, date.fromisoformat(t.expiry), t.side == "CE")


def _intraminute_marks(trader: paper_mod.PaperTrader, provider: ReplayProvider, candle, start: pd.Timestamp):
    """Mark open trades on the candle's adverse extreme, then its favourable one (CE and PE separately, so the
    fade check sees the matching underlying)."""
    low, high = float(candle["low"]), float(candle["high"])
    for offset, (ce_spot, pe_spot) in ((20, (low, high)), (40, (high, low))):
        stamp = start + pd.Timedelta(seconds=offset)
        for side, spot in (("CE", ce_spot), ("PE", pe_spot)):
            trades = [t for t in trader.open_trades() if t.side == side]
            if trades:
                quotes = {t.instrument_key: {"bid": provider.bid(t, spot, stamp)} for t in trades}
                trader.mark(quotes, {provider.name: spot}, stamp)


def simulate(name: str, spot: pd.DataFrame, vix: pd.DataFrame, rules, profiles, cfg: PBConfig, costs: dict,
             days: Optional[List[date]] = None, engine=None) -> dict:
    """Replay `days` (default: every session in `spot`) through the monitor and paper trader (and the strategy
    engine, if given). Returns {"trades": [trade dicts], "events": [momentum triggers], "days": [days replayed],
    "strategy_status": {strategy: {reason kind: checks}}} (why a strategy could not trade)."""
    provider = ReplayProvider(name, spot, vix, cfg)
    sessions_back = 1
    if engine is not None:
        sessions_back = STRATEGY_HISTORY_SESSIONS
        provider.history_candles = 400 * (STRATEGY_HISTORY_SESSIONS + 1)
    trader = paper_mod.PaperTrader(profiles, cfg, make_cost_fn(costs), store=None, mock=False,
                                   kill_switch_file=None)
    store = _Store()

    # the monitor reads rules_store.rules; a backtest has no web-page overrides
    monitor = MovementMonitor(provider, SimpleNamespace(rules=rules), store, broadcaster=_Quiet(), push_enabled=False,
                              trader=trader, strategies=engine)
    sessions = sorted(set(spot.index.date))
    wanted = set(days) if days is not None else set(sessions)
    done_trades: List[dict] = []
    replayed = []
    status: Dict[str, Counter] = defaultdict(Counter)
    for i, day in enumerate(sessions):
        if day not in wanted or i == 0:   # the first session only provides yesterday's candles
            continue
        first_day = sessions[max(0, i - sessions_back)]
        window = spot[(spot.index.date >= first_day) & (spot.index.date <= day)]
        today_start = int(np.searchsorted(window.index.date, day))
        provider.set_day(window, float(window["close"].iloc[today_start - 1]))
        replayed.append(day)
        for k in range(today_start, len(window)):
            start = window.index[k]
            provider.t, provider.ends = start + pd.Timedelta(minutes=1), k + 1
            monitor.run_once(provider.t)
            if engine is not None and k % 30 == 0:
                for key, per in engine.last.items():
                    reason = (per.get(name) or {}).get("reason") or ""
                    status[key][_reason_kind(reason)] += 1
            if k + 1 < len(window) and trader.open_trades():
                _intraminute_marks(trader, provider, window.iloc[k + 1], provider.t)
        for t in trader.open_trades():     # the data ended before square-off (a short session)
            trader._close(t, provider.t, "end of data")
        done_trades += [t.to_dict() for t in trader.trades]
        trader.trades = []
    return {"trades": done_trades, "events": store.events, "days": replayed,
            "strategy_status": {k: dict(v) for k, v in status.items()}}


def _reason_kind(reason: str) -> str:
    """Collapse a strategy's reason to why it could not trade, for the report."""
    for prefix in ("no data:", "warming up", "error:", "inside the model's training period"):
        if reason.startswith(prefix):
            return reason.split(" (")[0][:90]
    return "evaluated"


def _month_job(args) -> dict:
    name, start, end, config, overrides, keep_rvol, strategy_keys = args
    logging.getLogger("scanner").setLevel(logging.ERROR)    # the monitor logs every trigger and trade
    rules, profiles, cfg, costs = backtest_settings(config, overrides, keep_rvol)
    engine = None
    if strategy_keys:
        engine, strategy_profiles = strategy_settings(config, overrides, strategy_keys)
        profiles = {**profiles, **strategy_profiles}
    lead = 10 if not strategy_keys else 10 + 2 * STRATEGY_HISTORY_SESSIONS
    spot, vix = load_data(name, start - timedelta(days=lead), end)
    days = [d for d in sorted(set(spot.index.date)) if start <= d <= end]
    if not days:
        return {"trades": [], "events": [], "days": [], "strategy_status": {}}
    return simulate(name, spot, vix, rules, profiles, cfg, costs, days, engine=engine)


def run_backtest(name: str, start: date, end: date, config=settings.CONFIG_FILE,
                 overrides: Optional[dict] = None, keep_rvol: bool = False, books: Optional[List[str]] = None,
                 workers: Optional[int] = None, strategies: Optional[List[str]] = None) -> dict:
    """Run month by month in parallel. books: keep only these books enabled. strategies: run these strategy
    books ("all" or keys) instead of the momentum / scalp books."""
    overrides = {s: dict(v) for s, v in (overrides or {}).items()}
    for book in paper_mod.PROFILES:
        if strategies or (books and book not in books):
            overrides.setdefault(f"PAPER_{book.upper()}", {})["enabled"] = "false"
    months, m = [], date(start.year, start.month, 1)
    while m <= end:
        nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
        months.append((name, max(m, start), min(nxt - timedelta(days=1), end), str(config), overrides, keep_rvol,
                       strategies))
        m = nxt
    workers = workers or max(1, min(len(months), (os.cpu_count() or 2) - 1))
    out = {"trades": [], "events": [], "days": [], "strategy_status": defaultdict(Counter)}
    if workers == 1:
        results = map(_month_job, months)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_month_job, months)
    for n, r in enumerate(results, 1):
        for k in ("trades", "events", "days"):
            out[k] += r[k]
        for key, counts in r.get("strategy_status", {}).items():
            out["strategy_status"][key].update(counts)
        print(f"  {months[n - 1][1]:%Y-%m}: {len(r['days'])} days, {len(r['trades'])} trades", flush=True)
    if workers != 1:
        pool.shutdown()
    out["trades"].sort(key=lambda t: t["entry_time"])
    out["strategy_status"] = {k: dict(v) for k, v in out["strategy_status"].items()}
    return out


# =============================================================================
# Statistics and report
# =============================================================================
def _runs(nets: np.ndarray, win: bool) -> int:
    best = cur = 0
    for n in nets:
        cur = cur + 1 if ((n > 0) if win else (n <= 0)) else 0
        best = max(best, cur)
    return best


def daily_ratios(trades: List[dict], days: List[date]) -> dict:
    """Sharpe / Sortino of daily net P&L (days without trades count as 0), annualised with 252 sessions."""
    if len(days) < 20:
        return {"sharpe": None, "sortino": None, "note": "fewer than 20 days"}
    per_day = defaultdict(float)
    for t in trades:
        per_day[t["entry_time"][:10]] += t["net"]
    pnl = np.array([per_day.get(d.isoformat(), 0.0) for d in sorted(days)])
    sd = pnl.std(ddof=1)
    down = pnl[pnl < 0]
    dsd = math.sqrt((down ** 2).sum() / len(pnl)) if len(down) else 0.0
    return {"sharpe": float(pnl.mean() / sd * math.sqrt(252)) if sd > 0 else None,
            "sortino": float(pnl.mean() / dsd * math.sqrt(252)) if dsd > 0 else None,
            "best_day": float(pnl.max()), "worst_day": float(pnl.min())}


def stats(trades: List[dict]) -> dict:
    """Closed-trade statistics on net P&L (after spread, slippage and charges)."""
    nets = np.array([t["net"] for t in trades], dtype=float)
    if not len(nets):
        return {"trades": 0}
    wins, losses = nets[nets > 0], nets[nets <= 0]
    equity = np.cumsum(nets)
    drawdown = float(np.max(np.maximum.accumulate(np.r_[0.0, equity])[1:] - equity))
    gross_loss = -losses.sum()
    return {"trades": int(len(nets)), "win_rate": 100.0 * len(wins) / len(nets),
            "avg_win": float(wins.mean()) if len(wins) else 0.0, "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else math.inf,
            "net": float(nets.sum()), "expectancy": float(nets.mean()), "max_drawdown": drawdown,
            "charges": float(sum(t["charges"] for t in trades)),
            "gross_profit": float(wins.sum()), "gross_loss": float(losses.sum()),
            "reward_risk": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0
            else math.inf if len(wins) else 0.0,
            "max_consecutive_wins": _runs(nets, True), "max_consecutive_losses": _runs(nets, False),
            "avg_hold": float(np.mean([(pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 60
                                       for t in trades]))}


def split_days(days: List[date], share: float = TRAIN_SHARE) -> tuple:
    """(first `share` of the trading days, the rest), chronologically."""
    days = sorted(days)
    cut = int(round(len(days) * share))
    return set(days[:cut]), set(days[cut:])


def _fmt_row(label: str, s: dict) -> str:
    if not s.get("trades"):
        return f"  {label:<22}{0:>7}"
    pf = "inf" if math.isinf(s["profit_factor"]) else f"{s['profit_factor']:.2f}"
    return (f"  {label:<22}{s['trades']:>7}{s['win_rate']:>7.1f}%{s['avg_win']:>10,.0f}{s['avg_loss']:>10,.0f}"
            f"{pf:>7}{s['expectancy']:>9,.0f}{s['net']:>12,.0f}{s['max_drawdown']:>11,.0f}")


HEADER = (f"  {'':<22}{'trades':>7}{'win':>8}{'avg win':>10}{'avg loss':>10}{'PF':>7}{'per tr':>9}"
          f"{'net Rs':>12}{'max DD':>11}")


def book_report(book: str, trades: List[dict], days: List[date]) -> List[str]:
    mine = [t for t in trades if t["book"] == book]
    lines = [f"== {book.upper()} book ==", HEADER, _fmt_row("all", stats(mine))]
    if not mine:
        return lines + ["  no trades", ""]
    train, test = split_days(days)
    first = [t for t in mine if date.fromisoformat(t["entry_time"][:10]) in train]
    last = [t for t in mine if date.fromisoformat(t["entry_time"][:10]) in test]
    lines += ["", "  out-of-sample check (chronological, by trading day)",
              _fmt_row(f"first 70% ({min(train):%Y-%m}..)", stats(first)) if train else "",
              _fmt_row(f"last 30% ({min(test):%Y-%m}..)", stats(last)) if test else ""]
    for title, keyfn in (("by year", lambda t: t["entry_time"][:4]),
                         ("by entry window", lambda t: str(t["trigger"].get("window"))),
                         ("by side", lambda t: t["side"]),
                         ("by trigger quality", lambda t: str(t["trigger"].get("quality"))),
                         ("by exit reason", lambda t: t["exit_reason"]),
                         ("by entry hour", lambda t: t["entry_time"][11:13] + ":00")):
        groups = defaultdict(list)
        for t in mine:
            groups[keyfn(t)].append(t)
        lines += ["", f"  {title}"] + [_fmt_row(k, stats(v)) for k, v in sorted(groups.items())]
    s = stats(mine)
    lines += ["", f"  charges paid Rs {s['charges']:,.0f} ({s['charges'] / len(mine):,.0f} per trade), "
                  f"average hold {s['avg_hold']:.1f} min, {len(mine) / max(len(days), 1):.2f} trades per day", ""]
    return lines


def report(name: str, result: dict, start: date, end: date, books: List[str], keep_rvol: bool,
           overrides: dict) -> str:
    days = sorted(result["days"])
    events = Counter((e["window"], e["quality"]) for e in result["events"] if e["index"] == name)
    lines = [f"Movement paper-trading backtest - {name}",
             f"{days[0] if days else start} .. {days[-1] if days else end}: {len(days)} trading days of 1-minute data",
             "",
             "APPROXIMATIONS (read before trusting any number below)",
             "  * Option premiums are MODELLED, not historical: Black-Scholes on the index with IV = India VIX,",
             "    nearest weekly expiry, strike by the [OPTIONS] delta / premium filters, a bid-ask spread of",
             "    assumed_spread_pct around the model price, slippage on both legs, [COSTS] charges.",
             "    No IV smile, no IV expansion on fast moves, no real spreads or liquidity limits.",
             "  * " + ("rvol check kept ON: with no futures volume in the history every trigger is UNCONFIRMED."
                       if keep_rvol else
                       "rvol check OFF (min_rvol = 0): spot history has no futures volume, so the live default "
                       "(rvol >= 1.2) cannot be tested; live trades will be fewer than here."),
             "  * Exits are checked 3 times a minute (boundary, adverse extreme, favourable extreme); live polls every"
             " 10 s.",
             ]
    if overrides:
        lines.append("  * Settings changed for this run: " + "; ".join(
            f"{s}.{k}={v}" for s, kv in overrides.items() for k, v in kv.items()))
    lines += ["", "Momentum triggers fired ([MOVEMENT] rules, after the cooldown): " +
              (", ".join(f"{w}/{q}: {n}" for (w, q), n in sorted(events.items())) or "none"), ""]
    for book in books:
        lines += book_report(book, result["trades"], days)
    return "\n".join(lines)


def strategy_report(name: str, result: dict, start: date, end: date, keys: List[str], overrides: dict,
                    capital: float = 100000.0) -> str:
    """Summary table of every strategy book (ranked on the out-of-sample 30%) + a full report per strategy."""
    from src.backtest.stats import monte_carlo_drawdown
    from src.movement.strategies import REGISTRY
    days = sorted(result["days"])
    train, test = split_days(days) if days else (set(), set())
    lines = [f"Strategy backtest - {name}",
             f"{days[0] if days else start} .. {days[-1] if days else end}: {len(days)} trading days of 1-minute data, "
             f"{len(keys)} strategies, each its own paper book",
             "",
             "APPROXIMATIONS (read before trusting any number below)",
             "  * Option premiums are MODELLED (Black-Scholes, IV = India VIX, nearest weekly expiry, assumed spread,",
             "    slippage both legs, [COSTS] charges). No IV smile, no IV expansion on fast moves, no liquidity limits.",
             "  * Index spot history has no futures volume / OI and no real option-chain OI: strategies that need them",
             "    give no signal here (see 'not testable' below). Pair strategies need both indices: not replayed.",
             "  * Every parameter is an untested starting value. The OUT-OF-SAMPLE column is the last 30% of days;",
             "    nothing here was tuned on it, but if you change settings after reading it, it stops being one.",
             "  * Sharpe / Sortino use daily net P&L; Monte Carlo reshuffles the trade order 2000 times "
             f"(capital Rs {capital:,.0f} for the drawdown %).",
             ]
    if overrides:
        lines.append("  * Settings changed for this run: " + "; ".join(
            f"{s}.{k}={v}" for s, kv in overrides.items() for k, v in kv.items()))
    by_book = defaultdict(list)
    for t in result["trades"]:
        by_book[t["book"]].append(t)
    rows = []
    for k in keys:
        mine = by_book.get(k, [])
        s_all = stats(mine)
        s_oos = stats([t for t in mine if date.fromisoformat(t["entry_time"][:10]) in test])
        s_is = stats([t for t in mine if date.fromisoformat(t["entry_time"][:10]) in train])
        rows.append((k, s_all, s_is, s_oos))
    rows.sort(key=lambda r: (-(r[3].get("net") or -1e18) if r[3].get("trades") else 1e18, r[0]))
    lines += ["", "SUMMARY (sorted by out-of-sample net P&L; Rs after charges)",
              f"  {'strategy':<22}{'trades':>7}{'win%':>7}{'PF':>6}{'net':>11}{'max DD':>10}"
              f"{'IS net':>11}{'OOS tr':>8}{'OOS PF':>8}{'OOS net':>11}"]
    fmt_pf = lambda s: "-" if not s.get("trades") else ("inf" if math.isinf(s["profit_factor"]) else f"{s['profit_factor']:.2f}")
    for k, a, i, o in rows:
        if not a.get("trades"):
            lines.append(f"  {k:<22}{0:>7}")
            continue
        lines.append(f"  {k:<22}{a['trades']:>7}{a['win_rate']:>6.1f}%{fmt_pf(a):>6}{a['net']:>11,.0f}"
                     f"{a['max_drawdown']:>10,.0f}{(i.get('net') or 0):>11,.0f}{o.get('trades', 0):>8}{fmt_pf(o):>8}"
                     f"{(o.get('net') or 0):>11,.0f}")
    status = result.get("strategy_status", {})
    silent = [k for k, a, _i, _o in rows if not a.get("trades")]
    if silent:
        lines += ["", "NO TRADES - why (share of the checks during the replay)"]
        for k in silent:
            counts = status.get(k, {})
            total = sum(counts.values()) or 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
            why = "; ".join(f"{reason} ({100 * n / total:.0f}%)" for reason, n in top) or "no checks recorded"
            if k in CHAIN_OI_STRATEGIES:
                why = "not testable: the modelled chain has placeholder OI (live only)"
            lines.append(f"  {k:<22}{why}")
    for k, a, _i, _o in rows:
        mine = by_book.get(k, [])
        if not mine:
            continue
        spec = REGISTRY.get(k)
        lines += ["", f"== {k} - {spec.title if spec else k} ==",
                  f"  CE when: {spec.ce if spec else '-'}", f"  PE when: {spec.pe if spec else '-'}", HEADER,
                  _fmt_row("all", a)]
        first = [t for t in mine if date.fromisoformat(t["entry_time"][:10]) in train]
        last = [t for t in mine if date.fromisoformat(t["entry_time"][:10]) in test]
        if train:
            lines.append(_fmt_row(f"first 70% ({min(train):%Y-%m}..)", stats(first)))
        if test:
            lines.append(_fmt_row(f"last 30% ({min(test):%Y-%m}..)", stats(last)))
        r = daily_ratios(mine, days)
        mc = monte_carlo_drawdown([t["net"] for t in mine], capital)
        rr = "inf" if math.isinf(a["reward_risk"]) else f"{a['reward_risk']:.2f}"
        lines += [f"  gross profit Rs {a['gross_profit']:,.0f}, gross loss Rs {a['gross_loss']:,.0f}, "
                  f"avg win / avg loss {rr}, consecutive wins {a['max_consecutive_wins']}, "
                  f"consecutive losses {a['max_consecutive_losses']}",
                  "  Sharpe " + (f"{r['sharpe']:.2f}" if r.get("sharpe") is not None else "-") +
                  ", Sortino " + (f"{r['sortino']:.2f}" if r.get("sortino") is not None else "-") +
                  (f" ({r['note']})" if r.get("note") else f", best day Rs {r['best_day']:,.0f}, "
                                                          f"worst day Rs {r['worst_day']:,.0f}")]
        if "dd_p95" in mc:
            lines.append(f"  Monte Carlo max drawdown: median Rs {-mc['dd_median']:,.0f}, worst 5% Rs "
                         f"{-mc['dd_p95']:,.0f}, losing streak 95th pct {mc['streak_p95']:.0f}")
        for title, keyfn in (("by side", lambda t: t["side"]), ("by exit reason", lambda t: t["exit_reason"]),
                             ("by year", lambda t: t["entry_time"][:4])):
            groups = defaultdict(list)
            for t in mine:
                groups[keyfn(t)].append(t)
            lines += [f"  {title}"] + [_fmt_row(g, stats(v)) for g, v in sorted(groups.items())]
    return "\n".join(lines)


def write_trades(path: Path, trades: List[dict]):
    cols = ["day", "book", "index", "side", "symbol", "strike", "expiry", "qty", "entry_time", "entry_price",
            "underlying_entry", "exit_time", "exit_price", "exit_reason", "gross", "charges", "net", "pnl_pct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["trigger_window", "trigger_quality", "trigger_change_pct", "book_equity", "reason"])
        equity = defaultdict(float)
        for t in trades:
            equity[t["book"]] += float(t.get("net") or 0)
            trig = t.get("trigger") or {}
            w.writerow([t.get(c) if c != "day" else str(t.get("entry_time", ""))[:10] for c in cols] + [trig.get("window"), trig.get("quality"), trig.get("change_pct"),
                                                   round(equity[t["book"]], 2), trig.get("message")])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest the movement service's paper-trading books on 1-min history")
    ap.add_argument("--instrument", default="NIFTY", type=str.upper, choices=list(SUPPORTED_INDICES))
    ap.add_argument("--book", choices=list(paper_mod.PROFILES), help="only this book (default: both)")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--config", default=str(settings.CONFIG_FILE))
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.key=value",
                    help="change an ini setting for this run only, e.g. PAPER_SCALP.stop_pct=10 (repeatable)")
    ap.add_argument("--keep-rvol", action="store_true", help="keep the rvol check (no futures volume: no trades)")
    ap.add_argument("--download", action="store_true", help="fetch missing history from Upstox (needs a token)")
    ap.add_argument("--workers", type=int, help="worker processes (default: CPUs - 1)")
    ap.add_argument("--strategies", metavar="KEYS",
                    help="backtest strategy books instead: 'all' or comma-separated keys "
                         "(python -m src.movement.strategies --list)")
    args = ap.parse_args(argv)
    name = get_instrument(args.instrument).name
    overrides = parse_overrides(args.set)
    backtest_settings(args.config, overrides, args.keep_rvol)          # fail fast on a bad --set
    if args.strategies:
        keys = ["all"] if args.strategies.strip().lower() == "all" else \
            [k.strip() for k in args.strategies.split(",") if k.strip()]
        engine, _ = strategy_settings(args.config, overrides, keys)    # fail fast on a bad key / setting
        if args.download:
            load_data(name, args.start, args.end, download=True)
        print(f"Replaying {name} {args.start} .. {args.end} through {len(engine.enabled)} strategies ...", flush=True)
        result = run_backtest(name, args.start, args.end, args.config, overrides, args.keep_rvol, None, args.workers,
                              strategies=engine.enabled)
        if not result["days"]:
            raise SystemExit(f"No {name} history in {', '.join(str(d) for d in HISTORY_DIRS)} for that range "
                             f"(use --download with an Upstox token).")
        text = strategy_report(name, result, args.start, args.end, engine.enabled, overrides)
        print("\n" + text)
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"strategy_backtest_{name}"
        (settings.OUTPUT_DIR / f"{stem}.txt").write_text(text, encoding="utf-8")
        write_trades(settings.OUTPUT_DIR / f"{stem}_trades.csv", result["trades"])
        print(f"\nWrote {settings.OUTPUT_DIR / (stem + '.txt')} and {stem}_trades.csv")
        return
    if args.download:
        load_data(name, args.start, args.end, download=True)
    books = [args.book] if args.book else list(paper_mod.PROFILES)
    print(f"Replaying {name} {args.start} .. {args.end} ({', '.join(books)}) ...", flush=True)
    result = run_backtest(name, args.start, args.end, args.config, overrides, args.keep_rvol, books, args.workers)
    if not result["days"]:
        raise SystemExit(f"No {name} history in {', '.join(str(d) for d in HISTORY_DIRS)} for that range "
                         f"(use --download with an Upstox token).")
    text = report(name, result, args.start, args.end, books, args.keep_rvol, overrides)
    print("\n" + text)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"movement_backtest_{name}" + (f"_{args.book}" if args.book else "")
    (settings.OUTPUT_DIR / f"{stem}.txt").write_text(text, encoding="utf-8")
    write_trades(settings.OUTPUT_DIR / f"{stem}_trades.csv", result["trades"])
    print(f"\nWrote {settings.OUTPUT_DIR / (stem + '.txt')} and {stem}_trades.csv")


if __name__ == "__main__":
    main()
