"""
The F&O engine: one cycle = quotes for the whole universe -> analyse the indices and a budgeted rotation of
stocks -> confluence-gated signals -> option contract + plan -> risk checks -> paper position -> manage exits.

PAPER ONLY. There is no order-placement code in this package; FnoConfig refuses any mode other than "paper".

Signal status (and dashboard colour):
    BUY CE   green    every check passed and a paper position was opened
    BUY PE   red
    WAIT     yellow   a trigger exists but one confirmation (often the option premium) is still missing
    NO TRADE grey     blocked by a risk rule, or no liquid contract / no valid plan (the reason is shown)
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time as clock
from collections import deque
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.execution.costs import CostConfig
from src.fno import metrics
from src.fno.analysis import Analysis, analyze, latest, trend_table
from src.fno.config import FnoConfig
from src.fno.contracts import build_plan, select_contract
from src.fno.data import completed_minutes
from src.fno.position import Position
from src.fno.risk import RiskManager
from src.fno.store import Store, chain_explain, chain_payload, chain_series
from src.fno.strategies import REGISTRY, all_strategies
from src.scanner.indicators import atr as atr_fn
from src.utils.numbers import finite, fnum

logger = logging.getLogger("scanner.fno.engine")


class Engine:
    def __init__(self, cfg: FnoConfig, feed, store: Optional[Store] = None, risk: Optional[RiskManager] = None):
        self.cfg, self.feed = cfg, feed
        self.store = store or Store(mock=feed.mock)
        self.risk = risk or RiskManager(cfg)
        self.costs = CostConfig.from_mapping(cfg.costs)
        self.positions: List[Position] = [Position.from_dict(d) for d in self.store.trades(status="OPEN")]
        self.closed: List[Position] = []
        self.rows: Dict[str, dict] = {}                 # symbol -> scanner row
        self.analysis: Dict[str, Analysis] = {}
        self.last_scan: Dict[str, float] = {}
        self.last_bar: Dict[str, pd.Timestamp] = {}
        self.quotes: Dict[str, dict] = {}
        self.signals: deque = deque(maxlen=400)
        self.errors: deque = deque(maxlen=50)
        self.cycle_no = 0
        self.last_cycle: Optional[pd.Timestamp] = None
        self.duration = 0.0
        self.status = "starting"
        self.ws = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._state: dict = {"status": "starting"}
        self._day: Optional[date] = None
        self.listeners: List = []
        self._m1: Dict[str, pd.DataFrame] = {}
        self._iv_hist: Dict[str, tuple] = {}             # symbol -> (session start, ATM IV history before it)
        self._idea_cache: Dict[str, tuple] = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="fno-engine", daemon=True)
        self._thread.start()
        if self.cfg.websocket and not self.feed.mock:
            threading.Thread(target=self._run_ws, name="fno-websocket", daemon=True).start()

    def stop(self):
        self._stop.set()
        if self.ws is not None and getattr(self, "_ws_loop", None) is not None:
            try:
                asyncio.run_coroutine_threadsafe(self.ws.stop(), self._ws_loop)
            except RuntimeError:
                pass
        if self._thread:
            self._thread.join(timeout=15)

    def _run_ws(self):
        """Upstox market-data WebSocket (last prices of the indices and of open option positions). It
        re-authorises and reconnects with back-off by itself; the engine keeps working on REST without it."""
        try:
            from src.broker.websocket_feed import WebSocketFeed
            self._ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._ws_loop)
            self.ws = WebSocketFeed(self.feed.client, self._ws_keys)
            self._ws_loop.run_until_complete(self.ws.run())
        except Exception as error:                       # the feed is optional: never take the engine down
            logger.warning("WebSocket feed stopped: %s", error)

    def _ws_keys(self) -> List[str]:
        try:
            keys = [u.key for u in self.feed.universe().indices]
        except Exception:
            return []
        return keys + [p.key for p in self.positions]

    def _loop(self):
        while not self._stop.is_set():
            started = clock.monotonic()
            try:
                self.cycle()
            except Exception as error:
                self._error("cycle", error)
                self.status = "no token" if "Auth" in type(error).__name__ else "error"
                self._publish()
            wait = max(1.0, (3.0 if self.feed.mock else self.cfg.cycle_seconds) - (clock.monotonic() - started))
            self._stop.wait(wait)

    def _error(self, where: str, error: Exception):
        text = f"{where}: {type(error).__name__}: {error}"[:300]
        self.errors.appendleft({"time": str(pd.Timestamp.now(tz='Asia/Kolkata'))[:19], "error": text})
        logger.warning(text)

    # ------------------------------------------------------------------ one cycle
    def cycle(self, now: Optional[pd.Timestamp] = None):
        with self._lock:
            started = clock.monotonic()
            now = now or self.feed.now()
            if self._day != now.date():
                self._day = now.date()
                self.closed = [Position.from_dict(d) for d in self.store.trades(status="CLOSED", day=now.date().isoformat())]
                self.store.prune_chain()
            uni = self.feed.universe()
            unds = uni.select(self.cfg.indices, self.cfg.stocks)
            keys = [u.key for u in unds] + [p.key for p in self.positions]
            try:
                self.quotes = self.feed.quotes(list(dict.fromkeys(keys)))
            except Exception as error:
                if "Auth" in type(error).__name__:
                    raise
                self._error("quotes", error)
            market_open = self.feed.mock or (self.risk.session_block(now) not in ("market holiday",) and
                                             pd.Timestamp("09:15").time() <= now.time() < pd.Timestamp("15:30").time())
            if market_open:
                for und in self._scan_list(unds, now):
                    try:
                        self._scan(und, now, self._m1.get(self.cfg.benchmark))
                    except Exception as error:
                        if "Auth" in type(error).__name__:
                            raise
                        self._error(und.symbol, error)
                self._ideas(unds, now)
                self._manage(now)
            self.cycle_no += 1
            self.last_cycle, self.duration = now, round(clock.monotonic() - started, 2)
            self.status = "running" if market_open else "market closed"
            self._publish(unds)

    def _scan_list(self, unds, now) -> list:
        held = {p.symbol for p in self.positions}
        first = [u for u in unds if u.kind == "INDEX" or u.symbol in held]
        first.sort(key=lambda u: u.symbol != self.cfg.benchmark)          # the benchmark is analysed first
        due = []
        t = clock.monotonic()
        for u in unds:
            if u in first:
                continue
            if t - self.last_scan.get(u.symbol, -1e9) >= (0 if self.feed.mock else self.cfg.stock_refresh_seconds):
                q = self.quotes.get(u.key) or {}
                ltp, prev = fnum(q.get("ltp")), fnum(q.get("prev_close"))
                move = abs(ltp / prev - 1) if finite(ltp) and finite(prev) and prev > 0 else 0.0
                due.append((self.last_scan.get(u.symbol) is not None, -move, u.symbol, u))
        due.sort(key=lambda x: x[:3])                                     # never-scanned first, then biggest movers
        return first + [x[3] for x in due[:self.cfg.max_stock_scans_per_cycle]]

    def _scan(self, und, now, bench_m1) -> Optional[Analysis]:
        self.last_scan[und.symbol] = clock.monotonic()
        m1 = completed_minutes(self.feed.minutes(und, now), now)
        if len(m1) < 120:
            self.rows[und.symbol] = {"symbol": und.symbol, "note": "not enough candles yet"}
            return None
        self._m1[und.symbol] = m1
        chain = expiry = None
        series = None
        if und.kind == "INDEX":                                           # chains of the indices are recorded every cycle
            expiry = self._expiry(und, now)
            if expiry:
                try:
                    chain = self.feed.chain(und, expiry)
                    payload = chain_payload(chain, float(m1["close"].iloc[-1]), expiry, today=now.date())
                    if payload:
                        self.store.add_chain(now, und.symbol, payload)
                    day_start = now.normalize().isoformat()
                    if self._iv_hist.get(und.symbol, (None,))[0] != day_start:      # once per session and index
                        self._iv_hist[und.symbol] = (day_start, self.store.iv_history(und.symbol, day_start,
                                                                                     self.cfg.iv_pctile_days))
                    series = chain_series(self.store.chain_snapshots(und.symbol, start=day_start),
                                          self.cfg.premium_lookback_minutes * 2, self._iv_hist[und.symbol][1])
                except Exception as error:
                    if "Auth" in type(error).__name__:
                        raise
                    self._error(f"{und.symbol} chain", error)
        a = analyze(und.symbol, m1, self.cfg, self.feed.daily(und), bench_m1, series)
        self.analysis[und.symbol] = a
        sigs = latest(a, self.cfg)
        bar_time = a.frame.index[-1]
        new_bar = self.last_bar.get(und.symbol) != bar_time
        self.last_bar[und.symbol] = bar_time
        last = a.frame.iloc[-1]
        q = self.quotes.get(und.key) or {}
        row = {"symbol": und.symbol, "kind": und.kind, "ltp": fnum(q.get("ltp"), float(last["close"])),
               "bar_time": str(bar_time), "trend": trend_table(a), "rsi": fnum(last["rsi"]), "atr": fnum(last["atr"]),
               "rvol": fnum(last["rvol"]), "vwap": fnum(last["vwap"]),
               "checks_up": int(a.comp["up_n"].iloc[-1]), "checks_dn": int(a.comp["dn_n"].iloc[-1]),
               "checks_avail": int(a.comp["avail_n"].iloc[-1]),
               "status": "NO TRADE", "side": None, "strategies": [], "unavailable": a.unavailable}
        if sigs:
            top = sigs[0]
            row.update(status="WAIT" if top["status"] == "WAIT" else f"BUY {top['side']}", side=top["side"],
                       confidence=top["confidence"], strategies=[s["strategy_name"] for s in sigs if s["side"] == top["side"]])
        scan = a.scan.iloc[-1]
        if pd.notna(scan["bull"]):
            row.update(bull=round(float(scan["bull"]), 1), bear=round(float(scan["bear"]), 1),
                       factors={k: (None if pd.isna(scan[k]) else int(scan[k])) for k in a.scan.columns if k not in ("bull", "bear")})
        if a.chain is not None and len(a.chain):
            last_chain = a.chain.iloc[-1]
            row["chain"] = {k: (None if pd.isna(last_chain.get(k)) else round(float(last_chain.get(k)), 2))
                            for k in ("pcr", "support", "resistance", "atm_iv", "iv_chg", "iv_pctile", "ce_vol_pct", "pe_vol_pct",
                                      "bull_score", "bear_score", "atm_spread_pct")}
        old = self.rows.get(und.symbol) or {}
        row.update({k: old[k] for k in ("idea", "scan_side", "scan_note") if k in old})
        self.rows[und.symbol] = row
        if new_bar and sigs:
            self._handle(und, a, sigs, chain, expiry, now)
        return a

    def _expiry(self, und, now) -> Optional[date]:
        min_days = self.cfg.stock_min_days_to_expiry if und.kind == "EQUITY" else self.cfg.min_days_to_expiry
        e = und.pick_expiry(now.date(), self.cfg.expiry, min_days)
        if e == now.date() and not self.cfg.allow_expiry_day_contract:
            e = und.pick_expiry(now.date(), self.cfg.expiry, max(min_days, 1))
        return e

    # ------------------------------------------------------------------ ranked scanner
    SCAN_MIN, SCAN_EDGE, SCAN_TOP, SCAN_TTL = 60.0, 30.0, 3, 300.0

    def _verdict(self, row: dict):
        """(side, reason): which side a scanned row qualifies for, or why it is NO TRADE."""
        bull, bear = row.get("bull"), row.get("bear")
        if bull is None or bear is None:
            return None, "not analysed yet"
        side, score = ("CE", bull) if bull >= bear else ("PE", bear)
        if score < self.SCAN_MIN:
            return None, f"conditions insufficient (best score {score:.0f} < {self.SCAN_MIN:.0f})"
        if abs(bull - bear) < self.SCAN_EDGE:
            return None, f"conflicting signals (bullish {bull:.0f} vs bearish {bear:.0f})"
        return side, ""

    def _ideas(self, unds, now):
        """For the best few stocks on each side: a liquid contract and a full plan from the live chain (cached)."""
        by_symbol = {u.symbol: u for u in unds if u.kind == "EQUITY"}
        ranked = {"CE": [], "PE": []}
        for sym, row in self.rows.items():
            if sym not in by_symbol:
                continue
            side, why = self._verdict(row)
            row["scan_side"], row["scan_note"] = side, why
            if side:
                ranked[side].append((row["bull"] if side == "CE" else row["bear"], sym))
        t = clock.monotonic()
        for side, items in ranked.items():
            for score, sym in sorted(items, reverse=True)[:self.SCAN_TOP]:
                cached = self._idea_cache.get(sym)
                if cached and cached[1] == side and t - cached[0] < self.SCAN_TTL:
                    self.rows[sym]["idea"] = cached[2]
                    continue
                und, a = by_symbol[sym], self.analysis.get(sym)
                idea = {"note": "no analysis"}
                try:
                    expiry = self._expiry(und, now)
                    if a is not None and expiry is not None:
                        q = self.quotes.get(und.key) or {}
                        spot = fnum(q.get("ltp"), float(a.frame["close"].iloc[-1]))
                        contract, notes = select_contract(self.feed.chain(und, expiry), spot, side, und, expiry, self.cfg, now)
                        if contract is None:
                            idea = {"note": notes[0]}
                        else:
                            sig = {"strategy": "multi_factor", "strategy_name": "Multi-factor scanner", "confidence": round(score, 1)}
                            plan, why = build_plan(sig, und, contract, a.frame.iloc[-1], self.cfg, self.cfg.capital, now,
                                                   self.risk.risk_mult(now, expiry))
                            idea = plan.to_dict() if plan is not None else {"note": why}
                except Exception as error:
                    if "Auth" in type(error).__name__:
                        raise
                    idea = {"note": f"{type(error).__name__}: {error}"[:120]}
                self._idea_cache[sym] = (t, side, idea)
                self.rows[sym]["idea"] = idea

    def _ranked(self) -> dict:
        out = {"bullish": [], "bearish": [], "no_trade": 0}
        for row in self.rows.values():
            if row.get("kind") != "EQUITY" or row.get("bull") is None:
                continue
            side = row.get("scan_side")
            if side is None:
                out["no_trade"] += 1
                continue
            item = {k: row.get(k) for k in ("symbol", "ltp", "bull", "bear", "factors", "idea", "bar_time")}
            item["score"] = row["bull"] if side == "CE" else row["bear"]
            item["side"] = side
            out["bullish" if side == "CE" else "bearish"].append(item)
        for key in ("bullish", "bearish"):
            out[key].sort(key=lambda r: -r["score"])
            for n, r in enumerate(out[key], 1):
                r["rank"] = n
        return out

    # ------------------------------------------------------------------ signal -> plan -> paper position
    def _handle(self, und, a: Analysis, sigs: List[dict], chain, expiry, now):
        for side in ("CE", "PE"):
            mine = [s for s in sigs if s["side"] == side]
            if not mine:
                continue
            buys = [s for s in mine if s["status"] == "BUY"]
            sig = dict((buys or mine)[0])
            sig.update(symbol=und.symbol, timestamp=now.isoformat(), bar_time=str(sig["bar_time"]), mock=self.feed.mock,
                       also=[s["strategy_name"] for s in (buys or mine)[1:]])
            sig["why"] = [f"{sig['strategy_name']}: {sig['rule']}"]
            if a.chain is not None and len(a.chain) and "chain" in REGISTRY[sig["strategy"]].needs:
                sig["why"] += chain_explain(a.chain.iloc[-1], side)
            if (und.symbol, sig["strategy"], side, sig["bar_time"]) in self.risk.seen_bars:
                continue
            if not buys:
                sig["reason"] = "waiting for one more confirmation"
                self._record(sig, und, now)
                continue
            self._enter(und, a, sig, chain, expiry, now)

    def _enter(self, und, a: Analysis, sig: dict, chain, expiry, now):
        side = sig["side"]
        expiry = expiry or self._expiry(und, now)
        if expiry is None:
            return self._record(dict(sig, status="NO TRADE", reason="no tradable expiry"), und, now)
        if chain is None:
            chain = self.feed.chain(und, expiry)
        if self.cfg.max_iv_pctile > 0 and a.chain is not None and len(a.chain):
            pct = fnum(a.chain["iv_pctile"].iloc[-1]) if "iv_pctile" in a.chain else float("nan")
            if pct == pct and pct > self.cfg.max_iv_pctile:
                return self._record(dict(sig, status="NO TRADE", reason=f"IV percentile {pct:.0f} > {self.cfg.max_iv_pctile:g}: "
                                                                        "options are expensive to buy"), und, now)
        q = self.quotes.get(und.key) or {}
        bar = a.frame.iloc[-1]
        spot = fnum(q.get("ltp"), float(bar["close"]))
        contract, notes = select_contract(chain, spot, side, und, expiry, self.cfg, now)
        if contract is None:
            return self._record(dict(sig, status="NO TRADE", reason=notes[0], notes=notes), und, now)
        premium_atr, confirmed, detail = float("nan"), None, "no premium candles"
        try:
            pm = completed_minutes(self.feed.option_minutes(contract.key), now)
            pm = pm[pm.index.date == now.date()].dropna(subset=["close"])
            n = self.cfg.premium_lookback_minutes
            if len(pm) > n:
                change = float(pm["close"].iloc[-1] / pm["close"].iloc[-1 - n] - 1) * 100
                confirmed = change > 0 and float(pm["close"].iloc[-1]) >= float(pm["open"].iloc[-1])
                detail = f"premium {change:+.1f}% over {n} min"
                if len(pm) >= 15:
                    premium_atr = fnum(atr_fn(pm, 14).iloc[-1])
        except Exception as error:
            if "Auth" in type(error).__name__:
                raise
            self._error(f"{und.symbol} premium candles", error)
        sig["premium"] = detail
        sig["checks"] = dict(sig.get("checks") or {}, PREMIUM=confirmed)
        if self.cfg.require_premium_confirmation and not confirmed:
            return self._record(dict(sig, status="WAIT", reason=f"option premium not confirming ({detail})",
                                     contract=contract.to_dict()), und, now, mark=False)
        sig["confidence"] = round(min(100.0, sig["confidence"] + (5.0 if confirmed else 0.0)), 1)
        plan, why = build_plan(sig, und, contract, bar, self.cfg, self.cfg.capital, now,
                               self.risk.risk_mult(now, expiry), premium_atr)
        if plan is None:
            return self._record(dict(sig, status="NO TRADE", reason=why, contract=contract.to_dict()), und, now)
        trade_time = q.get("trade_time")
        age = None
        if not self.feed.mock and trade_time is not None:
            from src.broker.market_data import trade_time as parse_time
            ts = parse_time(trade_time)
            age = (now - ts).total_seconds() if ts is not None else None
        blocks = self.risk.check_entry(now, und.symbol, side, sig["strategy"], sig["bar_time"], self.cfg.capital,
                                       self.closed, self.positions, expiry, contract.spread_pct, age)
        sig["plan"] = plan.to_dict()
        if blocks:
            return self._record(dict(sig, status="NO TRADE", reason="; ".join(blocks)), und, now)
        pos = Position.open(plan, self.cfg, self.costs, now, "mock" if self.feed.mock else "live quote", self.feed.mock)
        self.positions.append(pos)
        self.store.save_trade(pos)
        sig.update(status="BUY", position_id=pos.id, reason="paper position opened")
        self._record(sig, und, now)
        logger.info("PAPER BUY %s x%d @ %.2f SL %.2f T1 %.2f T2 %.2f T3 %.2f (%s, confidence %.0f)", plan.instrument,
                    plan.qty, plan.entry, plan.stop, plan.t1, plan.t2, plan.t3, plan.strategy_name, plan.confidence)

    def _record(self, sig: dict, und, now, mark: bool = True):
        if mark:
            if sig["status"] == "BUY":
                self.risk.mark_signal(now, und.symbol, sig["side"], sig["strategy"], sig["bar_time"])
            else:
                self.risk.seen_bars.add((und.symbol, sig["strategy"], sig["side"], str(sig["bar_time"])))
        elif any(s.get("symbol") == sig["symbol"] and s.get("strategy") == sig["strategy"] and
                 s.get("bar_time") == sig["bar_time"] and s.get("status") == sig["status"] for s in list(self.signals)[:40]):
            return
        row = self.rows.get(und.symbol)
        if row is not None and row.get("side") == sig["side"]:      # the watchlist shows the FINAL decision
            row["status"] = f"BUY {sig['side']}" if sig["status"] == "BUY" else sig["status"]
            row["note"] = sig.get("reason", "")
        self.signals.appendleft(sig)
        self.store.add_signal(sig)
        for fn in list(self.listeners):
            try:
                fn(sig)
            except Exception:
                pass

    # ------------------------------------------------------------------ open positions
    def _manage(self, now):
        killed = self.risk.killed()
        for pos in list(self.positions):
            q = self.quotes.get(pos.key) or {}
            price = fnum(q.get("bid"))
            if not finite(price) or price <= 0:
                price = fnum(q.get("ltp"))
            if self.ws is not None:
                tick = self.ws.latest(pos.key)
                if tick is not None and not finite(price):
                    price = tick[0]
            if not finite(price) or price <= 0:
                continue                                              # never exit on a missing quote
            a = self.analysis.get(pos.symbol)
            und_close = atr = lo = hi = float("nan")
            opposite = False
            if a is not None and len(a.frame):
                bar, bar_time = a.frame.iloc[-1], a.frame.index[-1]
                if getattr(pos, "_bar", None) != bar_time and bar_time + pd.Timedelta(minutes=a.tf) > pd.Timestamp(pos.opened):
                    pos._bar = bar_time                               # underlying rules act once per completed bar
                    und_close, atr = float(bar["close"]), fnum(bar["atr"])
                    lo, hi = fnum(bar["swing_lo"]), fnum(bar["swing_hi"])
                    want = -1 if pos.side == "CE" else 1
                    opposite = bool((a.signals.iloc[-1] == want).any())
            force = "KILL SWITCH" if killed else "SQUARE-OFF" if now.time() >= self.cfg.square_off else ""
            minutes = (now - pd.Timestamp(pos.opened)).total_seconds() / 60.0
            fills = pos.step(now, price, price, price, price, self.cfg, self.costs, und_close, atr, lo, hi, opposite,
                             force, self.cfg.slippage_pct, 0.0, minutes)
            if fills or self.cycle_no % 4 == 0:
                self.store.save_trade(pos)
            if pos.status == "CLOSED":
                self.positions.remove(pos)
                self.closed.append(pos)
                self.store.save_trade(pos)
                logger.info("PAPER EXIT %s: %s, net Rs %.0f (%.2f R)", pos.instrument, pos.exit_reason, pos.net, pos.r_multiple)

    def close_position(self, pos_id: str, reason: str = "MANUAL") -> bool:
        with self._lock:
            for pos in self.positions:
                if pos.id == pos_id:
                    pos.close_all(pos.last_price, reason, self.feed.now(), self.costs, self.cfg.slippage_pct)
                    self.positions.remove(pos)
                    self.closed.append(pos)
                    self.store.save_trade(pos)
                    return True
        return False

    # ------------------------------------------------------------------ state for the dashboard
    def _publish(self, unds=None):
        now = self.last_cycle or self.feed.now()
        stats = self.risk.day_stats(self.closed, self.positions)
        lim = self.risk.limits(self.cfg.capital)
        watch = []
        for u in unds or []:
            q = self.quotes.get(u.key) or {}
            ltp, prev = fnum(q.get("ltp")), fnum(q.get("prev_close"))
            if self.ws is not None and u.kind == "INDEX":
                tick = self.ws.latest(u.key)
                if tick is not None:
                    ltp = tick[0]
            row = dict(self.rows.get(u.symbol) or {"symbol": u.symbol, "status": "NOT SCANNED YET"})
            row.update(symbol=u.symbol, kind=u.kind, ltp=ltp if finite(ltp) else row.get("ltp"),
                       change_pct=(ltp / prev - 1) * 100 if finite(ltp) and finite(prev) and prev > 0 else None,
                       day_high=fnum(q.get("high")), day_low=fnum(q.get("low")), lot_size=u.lot_size(u.pick_expiry(now.date()) or date.max),
                       expiry=str(u.pick_expiry(now.date()) or ""), strike_step=u.strike_step(u.pick_expiry(now.date()) or date.max, ltp if finite(ltp) else None))
            watch.append(row)
        state = {
            "status": self.status, "mock": self.feed.mock, "mode": "PAPER", "time": str(now)[:19], "cycle": self.cycle_no,
            "duration": self.duration, "capital": self.cfg.capital,
            "watchlist": watch, "indices": [w for w in watch if w["kind"] == "INDEX"],
            "signals": list(self.signals)[:120], "scanner": self._ranked(),
            "positions": [p.to_dict() for p in self.positions],
            "closed_today": [p.to_dict() for p in sorted(self.closed, key=lambda p: p.closed, reverse=True)],
            "risk": dict(stats, **lim, kill=self.risk.killed(), max_trades=self.cfg.max_trades_per_day,
                         max_open=self.cfg.max_open_positions, max_consecutive_losses=self.cfg.max_consecutive_losses,
                         session=self.risk.session_block(now)),
            "health": {"feed": self.feed.status(), "websocket": self.ws.status() if self.ws is not None else
                       {"connected": False, "note": "off in mock mode" if self.feed.mock else "not started"},
                       "errors": list(self.errors)[:15], "universe": len(watch),
                       "scanned": sum(1 for w in watch if w.get("bar_time"))},
        }
        self._state = state

    def state(self) -> dict:
        return self._state

    def strategies(self) -> List[dict]:
        counts = self.store.signal_counts()
        return [dict(s.describe(self.cfg.strategies.get(s.key)), signals=counts.get(s.key, 0)) for s in all_strategies()]

    def performance(self) -> dict:
        return metrics.report(self.store.trades(), self.store.signal_counts())
