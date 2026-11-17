"""
Stock-options engine: scans the NIFTY 50 every `refresh_seconds`, builds CE / PE plans and paper-tracks the
VALID ones (one trade per stock per day, never real orders).

Per cycle:
  1. one quotes call for all 50 stocks, 15-minute candles per stock (history cached per day)
  2. a first score WITHOUT the option chain; stocks with |score| >= watch_score get the chain (nearest monthly
     expiry with >= min_days_to_expiry days left) and a full plan
  3. a VALID plan inside the entry window opens a paper trade with the plan's frozen levels
  4. open trades are marked to the option LTP: stop / T1 (stop moves to cost) / T2 / square-off

The paper book is saved to data/stocks/book_<date>.json, so a restart keeps today's trades.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time as clock
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.stocks.data import is_market_open
from src.stocks.precision import PrecisionConfig, evaluate, nifty_context, select
from src.stocks.signals import StockConfig, _t, build_plan, r2

logger = logging.getLogger("scanner.stocks.engine")


def choose_expiry(expiries: List[date], today: date, min_days: int) -> Optional[date]:
    live = sorted(e for e in expiries if e >= today)
    for e in live:
        if (e - today).days >= min_days:
            return e
    return live[-1] if live else None


class PaperBook:
    def __init__(self, folder: Path, cost_fn, mock: bool):
        self.folder = Path(folder)
        self.cost = cost_fn
        self.mock = mock
        self.day: Optional[date] = None
        self.trades: Dict[str, dict] = {}
        self.lock = threading.Lock()

    def _file(self, day: date) -> Path:
        return self.folder / f"book_{'mock_' if self.mock else ''}{day.isoformat()}.json"

    def roll(self, day: date):
        if self.day == day:
            return
        self.day, self.trades = day, {}
        f = self._file(day)
        if f.exists():
            try:
                self.trades = {t["symbol"]: t for t in json.loads(f.read_text(encoding="utf-8"))}
            except (OSError, ValueError, KeyError):
                logger.warning("paper book %s unreadable; starting empty", f.name)

    def save(self):
        if self.day is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        f = self._file(self.day)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(self.trades.values()), indent=1, default=str), encoding="utf-8")
        os.replace(tmp, f)

    def open(self, plan: dict, now: pd.Timestamp) -> Optional[dict]:
        opt = plan.get("option")
        if plan["symbol"] in self.trades or not opt:
            return None
        trade = {"symbol": plan["symbol"], "kind": opt["kind"], "contract": opt["contract"], "key": opt["key"],
                 "lot_size": opt["lot_size"], "entry": opt["entry"], "stop": opt["stop"], "initial_stop": opt["stop"],
                 "t1": opt["t1"], "t2": opt["t2"], "spot_entry": plan["spot"]["entry"], "spot_stop": plan["spot"]["stop"],
                 "spot_t1": plan["spot"]["t1"], "spot_t2": plan["spot"]["t2"], "score": plan["score"],
                 "opened": now.isoformat(), "status": "OPEN", "ltp": opt["entry"], "exit": None, "closed": None,
                 "pnl": 0.0, "events": [f"{now:%H:%M} opened at {opt['entry']}"],
                 "rank": (plan.get("precision") or {}).get("rank")}
        self.trades[plan["symbol"]] = trade
        return trade

    def _close(self, t, price, status, now):
        t.update(status=status, exit=r2(price), closed=now.isoformat())
        t["events"].append(f"{now:%H:%M} {status.replace('_', ' ').lower()} at {price:.2f}")
        self._pnl(t, price)

    def _pnl(self, t, price):
        q = t["lot_size"]
        t["pnl"] = r2((price - t["entry"]) * q - self.cost("BUY", t["entry"], q) - self.cost("SELL", price, q))

    def mark(self, ltps: Dict[str, float], spots: Dict[str, float], now: pd.Timestamp, square_off: str):
        for t in self.trades.values():
            if t["status"] not in ("OPEN", "T1_HIT"):
                continue
            ltp = ltps.get(t["key"])
            spot = spots.get(t["symbol"])
            if ltp is None:
                continue
            t["ltp"] = r2(ltp)
            sign = 1 if t["kind"] == "CE" else -1
            spot_stopped = spot is not None and t["status"] == "OPEN" and sign * (spot - t["spot_stop"]) <= 0
            if now.time() >= _t(square_off):
                self._close(t, ltp, "TIME_EXIT", now)
            elif ltp <= t["stop"] or spot_stopped:
                self._close(t, min(ltp, t["stop"]) if ltp <= t["stop"] else ltp,
                            "STOP_HIT" if t["status"] == "OPEN" else "EXIT_AT_COST", now)
            elif ltp >= t["t2"]:
                self._close(t, t["t2"], "T2_HIT", now)
            elif t["status"] == "OPEN" and ltp >= t["t1"]:
                t["status"], t["stop"] = "T1_HIT", t["entry"]
                t["events"].append(f"{now:%H:%M} T1 {t['t1']} reached - stop moved to cost {t['entry']}")
                self._pnl(t, ltp)
            else:
                self._pnl(t, ltp)

    def summary(self) -> dict:
        trades = sorted(self.trades.values(), key=lambda t: t["opened"])
        closed = [t for t in trades if t["closed"]]
        return {"trades": trades, "open": sum(1 for t in trades if not t["closed"]), "closed": len(closed),
                "wins": sum(1 for t in closed if t["pnl"] > 0), "pnl": r2(sum(t["pnl"] for t in trades))}


class StockEngine:
    def __init__(self, source, cfg: StockConfig = StockConfig(), cost_fn=None, book_dir: Optional[Path] = None,
                 breakout_cfg=None, eq_cost_fn=None, pusher=None, precision_cfg: Optional[PrecisionConfig] = None):
        from src.config.settings import DATA_DIR, OUTPUT_DIR
        from src.execution.costs import make_equity_cost_fn
        from src.stocks.breakout import BreakoutConfig, BreakoutScanner
        self.source = source
        self.cfg = cfg
        self.precision = precision_cfg if precision_cfg is not None else PrecisionConfig()
        self.nifty: dict = {}
        self.cost_fn = cost_fn or (lambda side, price, qty, exchange="NSE_FO": 0.0)
        self.book = PaperBook(book_dir or (DATA_DIR / "stocks"), self.cost_fn, source.mock)
        bcfg = breakout_cfg if breakout_cfg is not None else BreakoutConfig()
        self.breakout = BreakoutScanner(
            bcfg, cfg, self.cost_fn, eq_cost_fn or make_equity_cost_fn(), book_dir or (DATA_DIR / "stocks"),
            source.mock, pusher=pusher,
            alerts_csv=None if (book_dir is not None or source.mock) else OUTPUT_DIR / "stock_breakout_alerts.csv",
        ) if bcfg.enabled else None
        self._inputs: Dict[str, dict] = {}
        self.plans: Dict[str, dict] = {}
        self.errors: Dict[str, str] = {}
        self.status = "starting"
        self.generated_at: Optional[str] = None
        self.cycles = 0
        self.duration = 0.0
        self.lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ one stock
    def option_inputs(self, stock, today: date):
        expiries, lots = self.source.contracts(stock.key)
        expiry = choose_expiry(expiries, today, self.cfg.min_days_to_expiry)
        if expiry is None:
            return None, None, None
        return self.source.chain(stock.key, expiry), expiry, lots.get(expiry)

    def analyse(self, stock, now: pd.Timestamp, spot: float, live: bool, with_options: Optional[bool] = None) -> dict:
        bars = self.source.bars15(stock.key, now)
        daily = self.source.daily(stock.key)
        plan = build_plan(stock.symbol, bars, daily, spot, now, cfg=self.cfg, cost_fn=self.cost_fn, live=live)
        wants = with_options if with_options is not None else abs(plan["score"]) >= self.cfg.watch_score
        chain, expiry, lot = None, None, None
        if wants and stock.nifty50 is not False:
            try:
                chain, expiry, lot = self.option_inputs(stock, now.date())
            except Exception as error:
                if "Auth" in type(error).__name__:
                    raise
                chain, expiry, lot = None, None, None
                plan["notes"].append(f"option chain unavailable: {error}")
            if chain is not None:
                plan = build_plan(stock.symbol, bars, daily, spot, now, chain=chain, expiry=expiry, lot_size=lot,
                                  cfg=self.cfg, cost_fn=self.cost_fn, live=live)
        plan["name"], plan["industry"] = stock.name, stock.industry
        self._inputs[stock.symbol] = {"stock": stock, "bars": bars, "daily": daily, "chain": None, "expiry": None,
                                      "lot": None, "spot": spot}
        if wants and stock.nifty50 is not False and chain is not None:
            from src.stocks.analytics import chain_stats
            self._inputs[stock.symbol].update(chain=chain, expiry=expiry, lot=lot,
                                              stats=chain_stats(chain, spot) if spot == spot else {})
        return plan

    # ------------------------------------------------------------------ cycle
    def refresh(self) -> None:
        started = clock.monotonic()
        src = self.source
        now = src.now()
        live = src.mock or is_market_open(now)
        stocks = src.stocks()
        quotes = src.quotes([s.key for s in stocks])
        plans, errors = {}, {}
        index_bars = self._index_bars(now)
        nifty = nifty_context(index_bars, now)
        for s in stocks:
            q = quotes.get(s.key) or {}
            try:
                plan = self.analyse(s, now, q.get("ltp", float("nan")), live)
            except Exception as error:
                if "Auth" in type(error).__name__:
                    raise
                errors[s.symbol] = f"{type(error).__name__}: {error}"
                logger.warning("%s: %s", s.symbol, error)
                continue
            if q:
                prev = q.get("prev_close")
                plan["change_pct"] = r2((q["ltp"] / prev - 1) * 100) if prev and prev == prev and prev > 0 else None
                plan["day"] = {"open": r2(q.get("open")), "high": r2(q.get("high")), "low": r2(q.get("low")),
                               "prev_close": r2(prev)}
            if self.precision.enabled and plan["status"] == "VALID":
                x = self._inputs.get(s.symbol) or {}
                try:
                    plan["precision"] = evaluate(plan, x.get("bars"), nifty, now, self.precision)
                except Exception as error:
                    logger.warning("%s precision check failed: %s", s.symbol, error)
            plans[s.symbol] = plan
        self._run_breakout(now, stocks, quotes, live, index_bars)
        with self.book.lock:
            self.book.roll(now.date())
            if live:
                for p in self._to_open(plans):
                    t = self.book.open(p, now)
                    if t:
                        logger.info("paper BUY %s at %s (SL %s, T1 %s, T2 %s)", t["contract"],
                                    t["entry"], t["stop"], t["t1"], t["t2"])
                keys = [t["key"] for t in self.book.trades.values() if not t["closed"] and t["key"]]
                if keys:
                    ltps = {k: v.get("ltp") for k, v in src.quotes(keys).items() if v.get("ltp") == v.get("ltp")}
                    spots = {p["symbol"]: p.get("ltp") for p in plans.values()}
                    self.book.mark(ltps, spots, now, self.cfg.square_off)
                self.book.save()
        with self.lock:
            self.plans, self.errors, self.nifty = plans, errors, nifty
            self.generated_at = now.isoformat()
            self.status = "live" if live else "market closed - last session analysis"
            self.cycles += 1
            self.duration = round(clock.monotonic() - started, 1)

    def _index_bars(self, now):
        from src.stocks.universe import NIFTY_INDEX_KEY
        try:
            return self.source.bars15(NIFTY_INDEX_KEY, now)
        except Exception as error:
            if "Auth" in type(error).__name__:
                raise
            logger.warning("NIFTY bars unavailable (alignment / relative strength): %s", error)
            return None

    def _to_open(self, plans: Dict[str, dict]) -> List[dict]:
        """Plans that open a paper trade this cycle. Precision mode: only the best-ranked picks, inside the daily and
        open-position caps. Otherwise every VALID plan (the original behaviour)."""
        if not self.precision.enabled:
            return [p for p in plans.values() if p["status"] == "VALID"]
        trades = self.book.trades
        return select(list(plans.values()), len(trades), sum(1 for t in trades.values() if not t["closed"]),
                      set(trades), self.precision)

    def _run_breakout(self, now, stocks, quotes, live, index_bars=None):
        if self.breakout is None:
            return
        inputs = []
        for s in stocks:
            x = self._inputs.get(s.symbol)
            if x is None:
                continue
            q = quotes.get(s.key) or {}
            x = dict(x, spot=q.get("ltp", x.get("spot")))
            inputs.append(x)
        try:
            self.breakout.run(now, inputs, index_bars, live,
                              get_option=lambda stock: self.option_inputs(stock, now.date()),
                              quotes=self.source.quotes)
        except Exception as error:
            if "Auth" in type(error).__name__:
                raise
            logger.exception("pre-breakout scan failed (stock plans continue): %s", error)

    def state(self) -> dict:
        with self.lock:
            plans = list(self.plans.values())
            order = {"VALID": 0, "WATCH": 1, "NO_TRADE": 2}
            plans.sort(key=lambda p: (order.get(p["status"], 3), not (p.get("precision") or {}).get("ok"),
                                      -((p.get("precision") or {}).get("rank") or 0), -abs(p.get("score") or 0)))
            out = {"generated_at": self.generated_at, "status": self.status, "mock": self.source.mock,
                   "cycles": self.cycles, "duration": self.duration, "errors": dict(self.errors),
                   "counts": {k: sum(1 for p in plans if p["status"] == k) for k in order},
                   "plans": plans, "nifty": self.nifty,
                   "picks": [p["symbol"] for p in plans if (p.get("precision") or {}).get("ok")],
                   "precision": {"enabled": self.precision.enabled, "max_picks_per_day": self.precision.max_picks_per_day,
                                 "max_open": self.precision.max_open, "last_entry": self.precision.last_entry},
                   "config": {"min_score": self.cfg.min_score, "entry_start": self.cfg.entry_start,
                                              "entry_end": self.cfg.entry_end, "square_off": self.cfg.square_off,
                                              "refresh_seconds": self.cfg.refresh_seconds}}
        with self.book.lock:
            out["book"] = self.book.summary()
        out["breakout"] = self.breakout.state() if self.breakout is not None else None
        return out

    def plan_for(self, symbol: str) -> Optional[dict]:
        with self.lock:
            return self.plans.get(symbol)

    # ------------------------------------------------------------------ loop
    def run_forever(self):
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception as error:
                self.status = f"error: {type(error).__name__}: {error}"
                logger.error("stock engine cycle failed: %s", error)
            self._wake.wait(self.cfg.refresh_seconds)
            self._wake.clear()

    def start(self):
        self._thread = threading.Thread(target=self.run_forever, name="stock-engine", daemon=True)
        self._thread.start()

    def poke(self):
        self._wake.set()

    def stop(self):
        self._stop.set()
        self._wake.set()
