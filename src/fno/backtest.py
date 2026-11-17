"""
Per-strategy backtest of the F&O engine on historical 1-minute candles.

    python -m src.fno backtest --symbols NIFTY,RELIANCE --from 2026-08-01 --to 2026-10-31
    python -m src.fno backtest --symbols NIFTY --strategies orb,supertrend --walk-forward --download
    python -m src.fno backtest --symbols NIFTY --premium real        # traded option candles only

How it avoids fooling itself
    look-ahead      indicators are causal; a signal on bar i is entered at the OPEN of bar i+1; exits use the
                    same Position engine as paper trading, and a bar that touches both stop and target is a stop.
    execution       entry = premium + half the assumed spread + slippage; market exits lose half the spread +
                    slippage; a target fills only when the premium trades through it by half the spread;
                    every order pays brokerage, STT, exchange, SEBI, stamp duty and GST ([COSTS]).
    option prices   REAL      traded 1-minute candles of the actual contract from Upstox (contracts still in the
                              instrument master; expired ones through the expired-instruments API when the
                              account has it). Spread is still an assumption: history has no bid / ask.
                    MODELLED  Black-Scholes, IV = previous day's India VIX (indices, when cached) or realised
                              volatility x bt_model_iv_mult. No smile, no IV change. Labelled in every report.
    out-of-sample   the last bt_oos_fraction of the days is reported separately and never used to choose anything.
    walk-forward    parameters are chosen on a training window from a SMALL declared grid and judged only on the
                    following test window; the report shows how much the result shrinks out of sample.
    survivorship    today's F&O list excludes stocks that left it. Supply config/fno_membership.csv
                    (symbol,from,to) for point-in-time membership; otherwise the report carries a warning.

Results are a filter for bad ideas, not a forecast. Nothing here is known to be profitable.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import re
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.config import settings
from src.execution.costs import CostConfig
from src.fno import metrics
from src.fno.analysis import Analysis, attach_futures, evaluate, prepare
from src.fno.config import FnoConfig
from src.fno.contracts import Contract, build_plan, years_to_expiry
from src.fno.position import Position
from src.fno.store import Store, chain_series
from src.fno.strategies import REGISTRY, all_strategies
from src.fno.universe import Underlying
from src.stocks.analytics import bs_greeks, bs_price
from src.utils import market_calendar

logger = logging.getLogger("scanner.fno.backtest")
TZ = "Asia/Kolkata"
INDEX_SYMBOLS = ("NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX")


def out_dir() -> Path:
    path = settings.OUTPUT_DIR / "fno_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")


# =============================================================================
# History (1-minute candles, monthly cache)
# =============================================================================
def _read(path: Path) -> pd.DataFrame:
    from src.broker.market_data import to_minutes
    return to_minutes(pd.read_csv(path))


def load_minutes(key: str, start: date, end: date, client=None, download: bool = False) -> pd.DataFrame:
    """Cached months from data/history/fno (and the index files in data/history); missing months are fetched
    from Upstox only with `download` and a client."""
    from src.broker.market_data import empty_minutes, to_minutes
    cache = settings.HISTORY_DIR / "fno"
    cache.mkdir(parents=True, exist_ok=True)
    frames, month = [], date(start.year, start.month, 1)
    while month <= end:
        nxt = date(month.year + month.month // 12, month.month % 12 + 1, 1)
        name = f"{_safe(key)}_1m_{month:%Y-%m}"
        mine, legacy = cache / f"{name}.csv.gz", settings.HISTORY_DIR / f"{name}.csv"
        complete = nxt <= date.today()
        if mine.exists() and (complete or not download):
            frames.append(_read(mine))
        elif legacy.exists() and (complete or not download):
            frames.append(_read(legacy))
        elif download and client is not None:
            frame = to_minutes(client.minute_candles(key, month, min(nxt - timedelta(days=1), date.today())))
            if len(frame):
                frame.reset_index().rename(columns={"index": "datetime"}).to_csv(mine, index=False)
                frames.append(frame)
        month = nxt
    if not frames:
        return empty_minutes()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[(out.index.date >= start) & (out.index.date <= end)]


def membership(path: Optional[Path] = None) -> Dict[str, List[tuple]]:
    path = Path(path or settings.CONFIG_DIR / "fno_membership.csv")
    out: Dict[str, List[tuple]] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(line for line in handle if line.strip() and not line.startswith("#")):
            try:
                a = date.fromisoformat((row.get("from") or "1900-01-01").strip())
                b = date.fromisoformat((row.get("to") or "2999-12-31").strip() or "2999-12-31")
            except ValueError:
                continue
            out.setdefault((row.get("symbol") or "").strip().upper(), []).append((a, b))
    return out


# =============================================================================
# Premium sources
# =============================================================================
def model_expiry(symbol: str, day: date) -> date:
    """Expiry assumed by the MODELLED source (weekday rules as remembered from exchange circulars; holidays
    move it to the previous trading day). Real backtests take expiries from the exchange data instead."""
    def last_weekday(y, m, wd):
        d = date(y + m // 12, m % 12 + 1, 1) - timedelta(days=1)
        while d.weekday() != wd:
            d -= timedelta(days=1)
        return market_calendar.expiry_on_or_before(d)
    new_rule = day >= date(2025, 9, 1)
    if symbol in ("NIFTY", "SENSEX"):
        wd = (1 if new_rule else 3) if symbol == "NIFTY" else (3 if new_rule else 1 if day >= date(2025, 1, 1) else 4)
        d = day
        while True:
            while d.weekday() != wd:
                d += timedelta(days=1)
            e = market_calendar.expiry_on_or_before(d)
            if e >= day:
                return e
            d += timedelta(days=1)
    wd = 1 if new_rule else 3
    e = last_weekday(day.year, day.month, wd)
    if e < day:
        m = day.month % 12 + 1
        e = last_weekday(day.year + (day.month == 12), m, wd)
    return e


class ModelPremium:
    source = "MODELLED"

    def __init__(self, cfg: FnoConfig, daily_close: pd.Series, vix_daily: Optional[pd.Series] = None, index: bool = False):
        rets = np.log(daily_close / daily_close.shift(1))
        rv = (rets.rolling(20, min_periods=10).std() * math.sqrt(252)).shift(1) * cfg.bt_model_iv_mult
        self.iv = rv.fillna(0.15 if index else 0.30)
        if index and vix_daily is not None and len(vix_daily):
            self.iv = (vix_daily.shift(1) / 100.0).reindex(self.iv.index).fillna(self.iv)
        self.cfg = cfg

    def vol(self, t: pd.Timestamp) -> float:
        v = self.iv.get(t.normalize())
        return float(v) if v is not None and np.isfinite(v) and v > 0.03 else 0.2

    def contract(self, und: Underlying, t: pd.Timestamp, spot: float, side: str) -> Optional[dict]:
        cfg = self.cfg
        min_days = cfg.stock_min_days_to_expiry if und.kind == "EQUITY" else cfg.min_days_to_expiry
        if not cfg.allow_expiry_day_contract:
            min_days = max(min_days, 1)
        expiry = model_expiry(und.symbol, t.date() + timedelta(days=min_days))
        ref = und.pick_expiry(date.today()) or (und.expiries[0] if und.expiries else None)
        step = und.strike_step(ref, None) if ref else float("nan")
        if not np.isfinite(step) or step <= 0:
            return None
        sign = {"ATM": 0, "OTM": 1, "ITM": -1}.get(cfg.moneyness.upper(), 0) * (1 if side == "CE" else -1)
        strike = round(spot / step) * step + sign * cfg.strike_distance * step
        return {"strike": float(strike), "expiry": expiry, "lot": und.lot_size(ref), "key": f"MODEL|{und.symbol}{strike:g}{side}",
                "side": side, "iv": self.vol(t), "source": self.source}

    def bars(self, c: dict, und_bars: pd.DataFrame, tf: int) -> Optional[pd.DataFrame]:
        hi, lo = ("high", "low") if c["side"] == "CE" else ("low", "high")
        rows = []
        for t, b in zip(und_bars.index, und_bars.itertuples()):
            y0, y1 = years_to_expiry(c["expiry"], t), years_to_expiry(c["expiry"], t + pd.Timedelta(minutes=tf))
            px = lambda s, y: max(bs_price(float(s), c["strike"], y, c["iv"], c["side"]), 0.05)   # noqa: E731
            rows.append((px(b.open, y0), px(getattr(b, hi), y0), px(getattr(b, lo), y1), px(b.close, y1)))
        return pd.DataFrame(rows, index=und_bars.index, columns=["open", "high", "low", "close"])


class RealPremium:
    """Traded option candles from Upstox. Returns None (the trade is skipped and counted) when no data exists."""
    source = "REAL"

    def __init__(self, cfg: FnoConfig, client, cache: Optional[Path] = None):
        self.cfg, self.client = cfg, client
        self.cache = Path(cache or settings.HISTORY_DIR / "fno_options")
        self.cache.mkdir(parents=True, exist_ok=True)
        self._expired: Dict[str, List[date]] = {}
        self._contracts: Dict[tuple, dict] = {}
        self.expired_note = ""

    def _expired_expiries(self, und: Underlying) -> List[date]:
        if und.symbol not in self._expired:
            out = []
            try:
                data = self.client.get("/expired-instruments/expiries", {"instrument_key": und.key}).get("data") or []
                out = sorted(date.fromisoformat(str(x)[:10]) for x in data)
            except Exception as error:      # a 401 here means the account has no expired-instruments access
                self.expired_note = (f"Expired-contract candles are not available on this Upstox account ({type(error).__name__}); "
                                     "only contracts still listed today have REAL prices, older trades fall back to MODELLED.")
                self.expired_ok = False
            self._expired[und.symbol] = out
        return self._expired[und.symbol]

    def _expired_contracts(self, und: Underlying, expiry: date) -> dict:
        key = (und.symbol, expiry)
        if key not in self._contracts:
            rows = {}
            try:
                data = self.client.get("/expired-instruments/option/contract",
                                       {"instrument_key": und.key, "expiry_date": expiry.isoformat()}).get("data") or []
                for r in data:
                    rows[(float(r.get("strike_price")), str(r.get("instrument_type")).upper())] = \
                        (r.get("instrument_key"), int(float(r.get("lot_size") or 0)))
            except Exception as error:
                logger.warning("[%s] expired contracts of %s unavailable: %s", und.symbol, expiry, error)
            self._contracts[key] = rows
        return self._contracts[key]

    def contract(self, und: Underlying, t: pd.Timestamp, spot: float, side: str) -> Optional[dict]:
        cfg, day = self.cfg, t.date()
        min_days = cfg.stock_min_days_to_expiry if und.kind == "EQUITY" else cfg.min_days_to_expiry
        if not cfg.allow_expiry_day_contract:
            min_days = max(min_days, 1)
        expiries = sorted(set(self._expired_expiries(und)) | set(und.expiries))
        live = [e for e in expiries if (e - day).days >= min_days]
        if not live:
            return None
        expiry = live[0]
        if expiry in und.lots:
            table = {(s, k): (key, und.lot_size(expiry)) for (e, s, k), key in und.contracts.items() if e == expiry}
            expired = False
        else:
            table, expired = self._expired_contracts(und, expiry), True
        strikes = sorted({s for (s, k) in table if k == side})
        if len(strikes) < 3:
            return None
        i = int(np.argmin(np.abs(np.array(strikes) - spot)))
        sign = {"ATM": 0, "OTM": 1, "ITM": -1}.get(cfg.moneyness.upper(), 0) * (1 if side == "CE" else -1)
        strike = strikes[min(max(i + sign * cfg.strike_distance, 0), len(strikes) - 1)]
        key, lot = table[(strike, side)]
        if not key or lot <= 0:
            return None
        return {"strike": float(strike), "expiry": expiry, "lot": lot, "key": key, "side": side, "iv": float("nan"),
                "source": self.source, "expired": expired}

    def _candles(self, c: dict, day: date) -> pd.DataFrame:
        from src.broker.client import candles_frame
        from src.broker.market_data import to_minutes
        path = self.cache / f"{_safe(c['key'])}_{day.isoformat()}.csv.gz"
        if path.exists():
            return _read(path)
        try:
            if c.get("expired"):
                url = f"/expired-instruments/historical-candle/{quote(c['key'], safe='')}/1minute/{day.isoformat()}/{day.isoformat()}"
                frame = to_minutes(candles_frame(self.client.get(url).get("data", {}).get("candles", [])))
            else:
                frame = to_minutes(self.client.minute_candles(c["key"], day, day))
        except Exception as error:
            if "Auth" in type(error).__name__:
                raise
            logger.debug("no option candles for %s on %s: %s", c["key"], day, error)
            return pd.DataFrame()
        if len(frame):
            frame.reset_index().rename(columns={"index": "datetime"}).to_csv(path, index=False)
        return frame

    def bars(self, c: dict, und_bars: pd.DataFrame, tf: int) -> Optional[pd.DataFrame]:
        from src.scanner.indicators import resample
        m1 = self._candles(c, und_bars.index[0].date())
        if m1 is None or m1.empty:
            return None
        b = resample(m1, tf)[["open", "high", "low", "close", "volume"]].reindex(und_bars.index)
        if pd.isna(b["open"].iloc[0]) or b["volume"].fillna(0).sum() <= 0:
            return None                                     # the contract did not trade at the entry bar
        close = b["close"].ffill()
        for col in ("open", "high", "low"):
            b[col] = b[col].fillna(close)
        b["close"] = close
        return b


# =============================================================================
# Simulation of one strategy on one underlying
# =============================================================================
def simulate(und: Underlying, a: Analysis, key: str, cfg: FnoConfig, sources: list, costs: CostConfig,
             stats: Optional[dict] = None) -> List[dict]:
    f, sig, tf = a.frame, a.signals[key].to_numpy(), a.tf
    stats = stats if stats is not None else {}
    idx = f.index
    days = idx.normalize()
    close_t = idx + pd.Timedelta(minutes=tf)
    clock = np.array([t.time() for t in close_t])
    windows = cfg.windows()
    half = cfg.bt_assumed_spread_pct / 2.0
    sim = replace(cfg, slippage_pct=cfg.bt_slippage_pct, skip_if_one_lot_exceeds_risk=False)
    trades, busy_until, per_day = [], -1, {}
    for i in np.flatnonzero(sig != 0):
        if i <= busy_until or i + 1 >= len(f) or days[i + 1] != days[i]:
            continue
        t = clock[i]
        if t < cfg.first_entry or t >= cfg.last_entry or any(a_ <= t < b_ for a_, b_ in windows):
            continue
        if per_day.get(days[i], 0) >= cfg.bt_max_trades_per_day:
            continue
        side = "CE" if sig[i] == 1 else "PE"
        j, t_in = i + 1, idx[i + 1]
        spot = float(f["open"].iloc[j])
        end = j
        while end + 1 < len(f) and days[end + 1] == days[j]:
            end += 1
        und_bars = f.iloc[j:end + 1]
        c = pbars = None
        for src in sources:
            c = src.contract(und, t_in, spot, side)
            pbars = src.bars(c, und_bars, tf) if c else None
            if pbars is not None:
                break
        if pbars is None or c is None:
            stats["no_option_data"] = stats.get("no_option_data", 0) + 1
            continue
        years = years_to_expiry(c["expiry"], t_in)
        iv = c["iv"] if np.isfinite(c.get("iv", np.nan)) else 0.2
        g = bs_greeks(spot, c["strike"], years, iv, side)
        px = float(pbars["open"].iloc[0])
        if px < cfg.min_premium:
            stats["premium_too_small"] = stats.get("premium_too_small", 0) + 1
            continue
        contract = Contract(symbol=und.symbol, side=side, strike=c["strike"], expiry=c["expiry"], key=c["key"], lot=c["lot"],
                            segment=und.segment, ltp=px, bid=px * (1 - half / 100), ask=px * (1 + half / 100),
                            spread_pct=cfg.bt_assumed_spread_pct, oi=0.0, volume=0.0, iv=iv * 100, delta=g["delta"],
                            gamma=g["gamma"], greeks_estimated=True, spot=spot, moneyness=cfg.moneyness, tick=und.tick)
        signal = {"strategy": key, "strategy_name": REGISTRY[key].name, "confidence": float(a.gates[key]["confidence"].iloc[i])}
        plan, why = build_plan(signal, und, contract, f.iloc[i], sim, cfg.capital, t_in, fixed_lots=1)
        if plan is None:
            stats[f"no plan: {why.split(' ')[0]}"] = stats.get(f"no plan: {why.split(' ')[0]}", 0) + 1
            continue
        pos = Position.open(plan, sim, costs, t_in, c["source"])
        for k in range(j, end + 1):
            b = pbars.iloc[k - j]
            last = k == end or clock[k] >= cfg.square_off
            o = pos.entry if k == j else float(b["open"])
            pos.step(close_t[k], o, max(float(b["high"]), o), min(float(b["low"]), o), float(b["close"]), sim, costs,
                     und_close=float(f["close"].iloc[k]), atr=float(f["atr"].iloc[k]), swing_lo=float(f["swing_lo"].iloc[k]),
                     swing_hi=float(f["swing_hi"].iloc[k]), opposite=bool(sig[k] == -sig[i]) and k > j,
                     force="SQUARE-OFF" if last else "", slippage_pct=cfg.bt_slippage_pct + half,
                     touch_buffer=half / 100.0, minutes_open=(k - j + 1) * tf)
            if pos.status == "CLOSED":
                busy_until = k
                break
        per_day[days[i]] = per_day.get(days[i], 0) + 1
        d = pos.to_dict()
        d.pop("plan", None)
        dte = (c["expiry"] - t_in.date()).days
        d.update(signal_bar=str(idx[i]), dte=dte, dte_bucket="0" if dte == 0 else "1" if dte == 1 else "2-5" if dte <= 5 else "6+",
                 hold_bars=busy_until - j + 1, strategy_name=REGISTRY[key].name)
        trades.append(d)
    return trades


# =============================================================================
# Reports
# =============================================================================
def split_date(days: List[date], oos_fraction: float) -> Optional[date]:
    days = sorted(set(days))
    if len(days) < 10:
        return None
    return days[int(len(days) * (1 - oos_fraction))]


def strategy_report(trades: List[dict], cut: Optional[date]) -> dict:
    d = lambda t: pd.Timestamp(t["opened"]).date()      # noqa: E731
    ins = [t for t in trades if cut is None or d(t) < cut]
    oos = [t for t in trades if cut is not None and d(t) >= cut]
    return {"all": metrics.summary(trades), "in_sample": metrics.summary(ins), "out_of_sample": metrics.summary(oos),
            "by_side": metrics.breakdown(trades, lambda t: t["side"]),
            "by_exit": metrics.breakdown(trades, lambda t: t["exit_reason"]),
            "by_time": sorted(metrics.breakdown(trades, metrics.time_bucket), key=lambda r: r["group"]),
            "by_dte": sorted(metrics.breakdown(trades, lambda t: t["dte_bucket"]), key=lambda r: r["group"]),
            "by_symbol": metrics.breakdown(trades, lambda t: t["symbol"]),
            "by_source": metrics.breakdown(trades, lambda t: t["price_source"])}


def walk_forward(key: str, runs: Dict[tuple, List[dict]], days: List[date], cfg: FnoConfig) -> dict:
    """runs: parameter combination (tuple of (name, value)) -> all trades of that combination."""
    days = sorted(set(days))
    train, test = cfg.bt_wf_train_days, cfg.bt_wf_test_days
    default = next(iter(runs))
    windows, picked = [], []
    k = 0
    while k + train + test <= len(days):
        a, b, c = days[k], days[k + train], days[min(k + train + test, len(days) - 1)]
        last = k + train + test >= len(days)

        def part(trades, lo, hi, closed_end=False):
            return [t for t in trades if lo <= pd.Timestamp(t["opened"]).date() and
                    (pd.Timestamp(t["opened"]).date() <= hi if closed_end else pd.Timestamp(t["opened"]).date() < hi)]
        best, best_score = default, None
        for combo, trades in runs.items():
            tr = part(trades, a, b)
            if len(tr) < max(5, cfg.bt_min_trades // 4):
                continue
            score = float(np.mean([t["r_multiple"] for t in tr]))
            if best_score is None or score > best_score:
                best, best_score = combo, score
        tested = part(runs[best], b, c, closed_end=last)
        picked += tested
        windows.append({"train": f"{a} .. {days[k + train - 1]}", "test": f"{b} .. {c}", "params": dict(best),
                        "train_avg_r": None if best_score is None else round(best_score, 3),
                        "test_trades": len(tested),
                        "test_avg_r": round(float(np.mean([t["r_multiple"] for t in tested])), 3) if tested else None})
        k += test
    tr_r = [w["train_avg_r"] for w in windows if w["train_avg_r"] is not None]
    return {"combinations_tried": len(runs), "windows": windows, "out_of_sample": metrics.summary(picked),
            "mean_train_avg_r": round(float(np.mean(tr_r)), 3) if tr_r else None,
            "note": "the out-of-sample row is built only from test windows, each traded with parameters chosen on the "
                    "window before it" if windows else "not enough days for one train + test window"}


def run(cfg: FnoConfig, underlyings: List[Underlying], start: date, end: date, strategies: Optional[List[str]] = None,
        premium: str = "auto", do_walk_forward: bool = False, client=None, download: bool = False,
        minutes: Optional[Dict[str, pd.DataFrame]] = None, chain_store: Optional[Store] = None,
        progress=None, save: bool = True) -> dict:
    """`minutes` (symbol -> 1-minute frame) bypasses the history cache (tests, mock)."""
    keys = [s.key for s in all_strategies() if (strategies is None and s.enabled(cfg.strategies.get(s.key)))
            or (strategies and s.key in strategies)]
    costs = CostConfig.from_mapping(cfg.costs)
    members = membership()
    warnings, all_trades, skipped, unavailable = [], {k: [] for k in keys}, {}, {}
    wf_runs: Dict[str, Dict[tuple, List[dict]]] = {k: {} for k in keys}
    all_days: List[date] = []
    real = RealPremium(cfg, client) if premium in ("auto", "real") and client is not None else None
    if premium == "real" and real is None:
        raise ValueError("--premium real needs an Upstox token (python -m src.broker.auth)")
    bench = (minutes or {}).get(cfg.benchmark)
    if bench is None and any(u.symbol != cfg.benchmark for u in underlyings):
        bkey = next((u.key for u in underlyings if u.symbol == cfg.benchmark), "NSE_INDEX|Nifty 50")
        bench = load_minutes(bkey, start, end, client, download)
    vix = load_minutes("NSE_INDEX|India VIX", start - timedelta(days=5), end) if minutes is None else pd.DataFrame()
    vix_daily = vix["close"].groupby(vix.index.normalize()).last() if len(vix) else None
    for n, und in enumerate(underlyings):
        if progress:
            progress(f"{und.symbol} ({n + 1}/{len(underlyings)})")
        m1 = minutes[und.symbol] if minutes is not None else load_minutes(und.key, start, end, client, download)
        if len(m1) < 400:
            warnings.append(f"{und.symbol}: no 1-minute history for {start} .. {end} (run with --download and a token)")
            continue
        if und.symbol in members:
            ok = np.zeros(len(m1), dtype=bool)
            for a_, b_ in members[und.symbol]:
                ok |= (m1.index.date >= a_) & (m1.index.date <= b_)
            m1 = m1[ok]
        series = None
        if chain_store is not None:
            snaps = chain_store.chain_snapshots(und.symbol, start.isoformat(), (end + timedelta(days=1)).isoformat())
            series = chain_series(snaps, cfg.premium_lookback_minutes * 2) if snaps else None
        prep = prepare(und.symbol, m1, cfg, None, bench if bench is not None and len(bench) else None, series)
        base = evaluate(prep, cfg, keys)
        for k, why in base.unavailable.items():
            unavailable.setdefault(k, {})[und.symbol] = why
        daily_close = prep.frame.groupby(prep.frame.index.normalize())["close"].last()
        model = ModelPremium(cfg, daily_close, vix_daily, und.kind == "INDEX")
        sources = [real] if premium == "real" else [model] if premium == "model" or real is None else [real, model]
        all_days += sorted(set(prep.frame.index.date))
        for key in keys:
            if key in base.unavailable:
                continue
            st = skipped.setdefault(key, {})
            all_trades[key] += simulate(und, base, key, cfg, sources, costs, st)
            if do_walk_forward and REGISTRY[key].grid:
                s = REGISTRY[key]
                names = list(s.grid)
                combos = [tuple(zip(names, vals)) for vals in itertools.product(*(s.grid[n_] for n_ in names))][:12]
                default = tuple((n_, s.params(cfg.strategies.get(key))[n_]) for n_ in names)
                for combo in [default] + [c for c in combos if c != default]:
                    if combo == default:
                        trades = [t for t in all_trades[key] if t["symbol"] == und.symbol]
                    else:
                        alt = evaluate(prep, cfg, [key], {key: dict(cfg.strategies.get(key, {}), **dict(combo))}, base.events)
                        trades = simulate(und, alt, key, cfg, sources, costs, {})
                    wf_runs[key].setdefault(combo, []).extend(trades)
    cut = split_date(all_days, cfg.bt_oos_fraction)
    results = []
    for key in keys:
        s = REGISTRY[key]
        row = {"key": key, "number": s.number, "name": s.name, "category": s.category,
               "params": s.params(cfg.strategies.get(key)), "skipped": skipped.get(key, {}),
               "unavailable": unavailable.get(key, {})}
        row.update(strategy_report(all_trades[key], cut))
        if row["all"]["trades"] < cfg.bt_min_trades:
            row["caution"] = f"only {row['all']['trades']} trades: too few to judge (bt_min_trades = {cfg.bt_min_trades})"
        if do_walk_forward and wf_runs[key]:
            row["walk_forward"] = walk_forward(key, wf_runs[key], all_days, cfg)
        results.append(row)
    results.sort(key=lambda r: -((r["out_of_sample"]["avg_r"] if r["out_of_sample"]["trades"] else None) or -9))
    trades = [t for k in keys for t in all_trades[k]]
    sources_used = sorted({t["price_source"] for t in trades})
    if "MODELLED" in sources_used:
        warnings.append("Some or all option premiums are MODELLED (Black-Scholes), not traded prices: no IV smile, no IV "
                        "change on fast moves. Treat rupee results as indicative only.")
    if real is not None and real.expired_note:
        warnings.append(real.expired_note)
    warnings.append(f"Bid/ask spread is an assumption ({cfg.bt_assumed_spread_pct:g}% of the premium) and slippage "
                    f"{cfg.bt_slippage_pct:g}% per side; history has no quotes. Each trade is ONE lot at today's lot size.")
    if any(u.kind == "EQUITY" for u in underlyings) and not members:
        warnings.append("Survivorship bias: the stock list is today's F&O list. Stocks that left F&O are missing. Add "
                        "config/fno_membership.csv (symbol,from,to) for point-in-time membership.")
    if any("volume" in why for v in unavailable.values() for why in v.values()):
        warnings.append("Index history has no traded volume: volume / VWAP strategies and the volume check cannot be tested on it.")
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}"
    report = {"id": run_id, "created": datetime.now().isoformat(timespec="seconds"), "from": str(start), "to": str(end),
              "symbols": [u.symbol for u in underlyings], "premium": premium, "price_sources": sources_used,
              "oos_starts": str(cut) if cut else None, "days": len(set(all_days)), "walk_forward": do_walk_forward,
              "warnings": warnings, "overall": metrics.summary(trades), "strategies": results,
              "settings": {k: v for k, v in cfg.public().items() if k not in ("host", "port")},
              "equity": metrics.equity_curve(trades)[-2000:]}
    if save:
        path = out_dir() / f"{run_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, default=str), encoding="utf-8")
        os.replace(tmp, path)
        if trades:
            cols = ["symbol", "strategy", "side", "instrument", "opened", "closed", "entry", "initial_stop", "r", "qty",
                    "exit_reason", "net", "r_multiple", "charges", "price_source", "dte", "signal_bar", "confidence"]
            with open(out_dir() / f"{run_id}_trades.csv", "w", newline="", encoding="utf-8") as handle:
                w = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(trades)
        report["file"] = str(path)
    report["trades"] = trades
    return report


def list_reports() -> List[dict]:
    out = []
    for path in sorted(out_dir().glob("*.json"), reverse=True)[:60]:
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({k: r.get(k) for k in ("id", "created", "from", "to", "symbols", "price_sources", "days", "walk_forward")}
                   | {"trades": (r.get("overall") or {}).get("trades"), "net_pnl": (r.get("overall") or {}).get("net_pnl")})
    return out


def load_report(run_id: str) -> Optional[dict]:
    if not re.fullmatch(r"[0-9_]{15}", run_id or ""):
        return None
    path = out_dir() / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def print_report(r: dict):
    print(f"\nBacktest {r['id']}  {r['from']} .. {r['to']}  {', '.join(r['symbols'][:8])}{' ...' if len(r['symbols']) > 8 else ''}"
          f"  days {r['days']}  price source: {', '.join(r['price_sources']) or '-'}  out-of-sample from {r['oos_starts']}")
    print(f"{'#':>2} {'strategy':32} {'trades':>6} {'win%':>6} {'PF':>5} {'net Rs':>10} {'avgR':>7} {'maxDD':>9} | "
          f"{'IS avgR':>8} {'OOS n':>5} {'OOS avgR':>8}")
    f = lambda v, w, p=2: f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"   # noqa: E731
    for s in r["strategies"]:
        a, i, o = s["all"], s["in_sample"], s["out_of_sample"]
        note = "  (" + next(iter(s["unavailable"].values())) + ")" if s["unavailable"] and not a["trades"] else ""
        print(f"{s['number']:>2} {s['name'][:32]:32} {a['trades']:>6} {f(a['win_rate'], 6, 1)} {f(a['profit_factor'], 5)} "
              f"{f(a['net_pnl'], 10, 0)} {f(a['avg_r'], 7, 3)} {f(a['max_drawdown'], 9, 0)} | {f(i['avg_r'], 8, 3)} "
              f"{o['trades']:>5} {f(o['avg_r'], 8, 3)}{note}")
    for w in r["warnings"]:
        print("NOTE:", w)
    if r.get("file"):
        print("Saved:", r["file"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.fno backtest", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="NIFTY", help="comma list, or 'indices', or 'all'")
    p.add_argument("--from", dest="start", default=None)
    p.add_argument("--to", dest="end", default=None)
    p.add_argument("--strategies", default="all", help="comma list of strategy keys, or all")
    p.add_argument("--premium", choices=("auto", "real", "model"), default="auto")
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--download", action="store_true", help="fetch missing 1-minute history from Upstox")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    from src.fno.universe import load_universe
    from src.utils.logging_config import setup_logging
    settings.ensure_dirs()
    setup_logging(log_file=settings.LOG_DIR / "fno.log")
    cfg = FnoConfig.load(args.config)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=90)
    uni = load_universe(cfg.indices)
    if args.symbols.lower() == "all":
        unds = uni.select(cfg.indices, "all")
    elif args.symbols.lower() == "indices":
        unds = uni.indices
    else:
        names = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        unds = [uni.get(s) for s in names if uni.get(s)]
        missing = [s for s in names if not uni.get(s)]
        if missing:
            print("Not in today's F&O list:", ", ".join(missing))
    if not unds:
        print("No underlying selected.")
        return 1
    client = None
    try:
        from src.broker.client import UpstoxClient
        client = UpstoxClient(requests_per_second=cfg.requests_per_second)
        client.tokens.token()
    except Exception as error:
        client = None
        print(f"No Upstox token ({error}): using cached history and MODELLED premiums only.")
    keys = None if args.strategies == "all" else [s.strip().lower() for s in args.strategies.split(",")]
    report = run(cfg, unds, start, end, keys, args.premium if client else "model", args.walk_forward, client,
                 args.download and client is not None, chain_store=Store() if (settings.DATA_DIR / "fno" / "fno.db").exists() else None,
                 progress=lambda m: print("  ..", m, flush=True))
    print_report(report)
    return 0
