"""
Scanner engine: one evaluation cycle for NIFTY and SENSEX (called by the application controller every
loop_seconds, in a worker thread so the event loop stays responsive).

Orders: the engine itself never calls a broker. When [EXECUTION] mode is paper (default) or live, the paper
book of one lane ([EXECUTION] lane) is mirrored through src/execution/order_manager.py - simulated by default,
real only when src/execution/live_gate.py reports live trading armed for today.

Every cycle:
  1. data      index + futures 1-minute candles, option chain (bid/ask, OI, IV, greeks), India VIX, breadth
  2. quality   spot / candle / chain freshness; a stale feed = NO TRADE for that index + a "data stale" alert
  3. direction each newly completed signal bar -> strategies.evaluate_bar()  (index market direction)
  4. signals   signals.StageTracker.step() -> WATCH / CONFIRMED / ENTRY SETUP / INVALIDATED / MISSED
  5. options   for confirmations: options.selection ranks real CE / PE contracts, premium stop / targets,
               R:R after costs; risk.position_sizing sizes whole lots
  6. paper     ENTRY SETUPs open a paper position at the ask (+ slippage); exits at the bid
  7. outputs   output/prebreakout_state.json, output/prebreakout_dashboard.html, data/prebreakout.db,
               output/prebreakout_alerts.csv, Telegram / desktop popups

Restart-safe: today's bars are replayed silently on start-up to rebuild the setups, and alerts already
sent today (from the database) are never sent twice.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.broker.market_data import IndexSnapshot, MarketSnapshot
from src.config import instruments as inst
from src.config import settings
from src.config.settings import AppSettings, PBConfig
from src.dashboard import view
from src.execution import paper_trading as paper_mod
from src.execution.costs import make_cost_fn
from src.execution.order_manager import Execution
from src.options import derivatives as deriv_mod
from src.notifications.service import Notifier
from src.options import selection as opt_sel
from src.risk import manager as riskm
from src.scanner import ml_filter, score_v2, score_v3
from src.scanner.features_ext import exit_context
from src.scanner import signals as sig
from src.scanner import strategies as strat
from src.scanner.store import Store, json_default, trade_model
from src.utils import market_calendar
from src.utils import numbers as num

SESSION_START, SESSION_END = time(9, 15), time(15, 30)
logger = logging.getLogger("scanner.engine")


class Lane:
    """One scoring model's trackers, risk manager and paper book (v1 = main, v2 = shadow comparison)."""

    def __init__(self, model: str, names, cfg: settings.PBConfig, costs, push: bool):
        self.model, self.push = model, push
        self.trackers = {n: sig.StageTracker(n, cfg, tag="" if model == "v1" else model) for n in names}
        self.risk = riskm.RiskManager(cfg)
        self.book = paper_mod.PaperBook(cfg, costs)
        self.latest: Dict[str, Optional[strat.BarEval]] = {n: None for n in names}

    def apply_config(self, cfg):
        for t in self.trackers.values():
            t.cfg = cfg
        self.risk.cfg = cfg
        self.book.cfg = cfg


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return round(v, 4) if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    return obj


def atomic_write(path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# =============================================================================
# Service
# =============================================================================
class ScannerEngine:
    def __init__(self, cfg: PBConfig, app: AppSettings, provider, notifier: Notifier, store: Store,
                 push: bool = True, feed_status: Optional[Callable[[], dict]] = None):
        self.cfg, self.app, self.provider, self.notifier, self.store = cfg, app, provider, notifier, store
        self.feed_status = feed_status
        self.mock = bool(getattr(provider, "mock", False))
        self.push = push
        self.costs = make_cost_fn(app.costs)
        self.names = list(inst.validate_indices(app.indices))
        self.trackers = {n: sig.StageTracker(n, cfg) for n in self.names}
        self.last_pos: Dict[str, Optional[pd.Timestamp]] = {n: None for n in self.names}
        self.latest: Dict[str, Optional[strat.BarEval]] = {n: None for n in self.names}
        self.feed: Dict[str, dict] = {n: {"ok": False, "reason": "starting"} for n in self.names}
        self.stale_alerted: Dict[str, bool] = {n: False for n in self.names}
        self.biases: Dict[str, str] = {}
        self.risk = riskm.RiskManager(cfg)
        self.book = paper_mod.PaperBook(cfg, self.costs)
        self.v2_on = bool(app.v2_enabled)
        self.v2_push = bool(app.v2_push)
        self.v2: Optional[Lane] = Lane("v2", self.names, cfg, self.costs, self.v2_push) if self.v2_on else None
        # v3: multi-factor modules + regime + trap engine, with its own settings overrides ([SCORE_V3])
        self.v3_on = bool(getattr(app, "v3_enabled", False))
        self.v3_push = bool(getattr(app, "v3_push", False))
        self.v3_overrides = dict(getattr(app, "v3_overrides", {}) or {})
        self.cfg_v3 = cfg.with_overrides(self.v3_overrides)
        self.v3: Optional[Lane] = Lane("v3", self.names, self.cfg_v3, self.costs, self.v3_push) if self.v3_on else None
        self.deriv: Dict[str, deriv_mod.DerivSnapshot] = {}
        self.iv_hist: Dict[str, list] = {n: [] for n in self.names}
        self.exit_ctx: Dict[str, dict] = {}
        self.execution: Optional[Execution] = None
        try:
            ex = Execution(getattr(app, "execution", {}) or {}, cfg.capital, self.mock, alert_fn=self._exec_alert)
            self.execution = ex if ex.enabled else None
        except ValueError as error:
            logger.error("Execution disabled: %s", error)
        self.ml = ml_filter.MLGate(str(app.ml_model_file), app.ml_allow_block) if app.ml_enabled else None
        self._attach_ml()
        self.day: Optional[date] = None
        self.events = market_calendar.load_events(app.events_file)
        self.recent: List[dict] = []
        self.snapshots: Dict[str, IndexSnapshot] = {}
        self.vix = np.nan
        self.vix_change = np.nan
        self.warm = {n: False for n in self.names}
        self.status = "starting"
        self.cycle_count = 0
        self.last_cycle_error = ""
        self.last_state: Optional[dict] = None

    def apply_config(self, cfg: PBConfig):
        self.cfg = cfg
        for t in self.trackers.values():
            t.cfg = cfg
        self.risk.cfg = cfg
        self.book.cfg = cfg
        if self.v2 is not None:
            self.v2.apply_config(cfg)
        try:
            self.cfg_v3 = cfg.with_overrides(self.v3_overrides)
        except ValueError as error:
            logger.error("[SCORE_V3] overrides rejected, previous v3 settings kept: %s", error)
        if self.v3 is not None:
            self.v3.apply_config(self.cfg_v3)

    def _exec_alert(self, level: str, message: str):
        now = self.provider.now()
        self._emit({"time": now.isoformat(), "index": "", "direction": "", "stage": "EXECUTION", "kind": "data",
                    "message": f"Execution: {message}", "setup_key": f"exec|{now:%H%M%S}|{hash(message) % 10000}",
                    "mock": self.mock}, push=level == "critical")

    def _attach_ml(self):
        if self.ml is not None:
            for t in self.trackers.values():
                t.ml_gate = self.ml.gate

    def _books(self):
        """(lane name, book, risk, push) for every paper book."""
        out = [("v1", self.book, self.risk, True)]
        if self.v2 is not None:
            out.append(("v2", self.v2.book, self.v2.risk, self.v2_push))
        if self.v3 is not None:
            out.append(("v3", self.v3.book, self.v3.risk, self.v3_push))
        return out

    def _lane_book(self, model: str):
        for name, book, risk, push in self._books():
            if name == model:
                return book, risk, push
        return self.book, self.risk, True

    # ---------- day handling ----------
    def _roll_day(self, day: date):
        if self.day == day:
            return
        self.day = day
        self.risk.roll(day)
        self.trackers = {n: sig.StageTracker(n, self.cfg) for n in self.names}
        self._attach_ml()
        if self.v2_on:
            self.v2 = Lane("v2", self.names, self.cfg, self.costs, self.v2_push)
            self.v2.risk.roll(day)
        if self.v3_on:
            self.v3 = Lane("v3", self.names, self.cfg_v3, self.costs, self.v3_push)
            self.v3.risk.roll(day)
        self.iv_hist = {n: [] for n in self.names}
        if self.execution is not None:
            try:
                self.execution.ensure(day)
            except Exception:
                logger.exception("execution could not start; continuing without it")
        self.last_pos = {n: None for n in self.names}
        self.warm = {n: False for n in self.names}
        self.stale_alerted = {n: False for n in self.names}
        self.book = paper_mod.PaperBook(self.cfg, self.costs)
        # reload today's paper positions (open and closed) after a restart
        for d in self.store.trades(mock=self.mock, day=day):
            try:
                pos = paper_mod.PaperPosition(**{k: v for k, v in d.items() if k in paper_mod.PaperPosition.__dataclass_fields__})
                pos.last_price = num.fnum(pos.last_price)
                pos.peak_price = num.fnum(pos.peak_price)
                model = trade_model(pos.setup_key)
                lane = {"v2": self.v2, "v3": self.v3}.get(model)
                if model in ("v2", "v3"):
                    if lane is None:
                        continue
                    book, risk = lane.book, lane.risk
                else:
                    book, risk = self.book, self.risk
                book.positions.append(pos)
                risk.record_open(pos.index)
                if pos.closed:
                    risk.state.open_positions -= 1
            except Exception:
                logger.exception("Could not reload a paper trade")
        self._week_realised = self._week_before(day)
        for model, book, risk, _ in self._books():  # open sides and today's losing streak survive a restart
            risk.update(book.realised_today(day), book.open_pnl(), len(book.open_positions()), **book.risk_inputs(day),
                        week_realised=self._week_realised.get(model, 0.0))
        self.recent = [a for a in self.store.recent_alerts(100, self.mock) if str(a.get("time", ""))[:10] == day.isoformat()]

    def _week_before(self, day: date) -> Dict[str, float]:
        """Realised net P&L per lane from Monday of this week up to (not including) `day`."""
        monday = day.toordinal() - day.weekday()
        out: Dict[str, float] = {}
        try:
            for t in self.store.trades(mock=self.mock):
                d = str(t.get("entry_time", ""))[:10]
                if not d or not t.get("exit_time"):
                    continue
                o = date.fromisoformat(d).toordinal()
                if monday <= o < day.toordinal():
                    m = trade_model(t.get("setup_key", ""))
                    out[m] = out.get(m, 0.0) + float(t.get("net_pnl") or 0.0)
        except Exception:
            logger.exception("weekly P&L unavailable")
        return out

    # ---------- one cycle ----------
    def cycle(self, now: Optional[pd.Timestamp] = None) -> dict:
        now = now or self.provider.now()
        day = now.date()
        trading_day = market_calendar.is_trading_day(day) or self.mock
        in_session = SESSION_START <= now.time() <= SESSION_END
        self._roll_day(day)
        self.cycle_count += 1
        if not trading_day:
            self.status = f"Market closed: {market_calendar.holiday_name(day)}"
            return self._publish(now, market_open=False)
        snap = self.provider.snapshot(now, want_chain=None)
        self.vix, self.vix_change = snap.vix, snap.vix_change_pct
        self.snapshots = snap.indices
        self.status = "Market open" if in_session else ("Pre-open" if now.time() < SESSION_START else "Market closed (after hours)")
        for name in self.names:
            isnap = snap.indices.get(name)
            if isnap is None:
                self.feed[name] = {"ok": False, "reason": "no data"}
                continue
            try:
                self._process_index(name, isnap, snap, now, in_session)
            except Exception as error:
                logger.exception("[%s] cycle failed", name)
                self.feed[name] = {"ok": False, "reason": f"error: {type(error).__name__}"}
        self._manage_paper(now)
        self._kill_switch(now)
        wk = getattr(self, "_week_realised", {}) or {}
        for model, book, risk, _ in self._books():
            risk.update(book.realised_today(day), book.open_pnl(), len(book.open_positions()), **book.risk_inputs(day),
                        week_realised=wk.get(model, 0.0))
        self._poll_execution(now)
        return self._publish(now, market_open=in_session)

    def _freshness(self, isnap: IndexSnapshot, now: pd.Timestamp, in_session: bool) -> dict:
        cfg = self.cfg
        spot_age = (now - isnap.ltp_time).total_seconds() if isnap.ltp_time is not None else None
        today = isnap.spot_minutes[isnap.spot_minutes.index.date == now.date()] if len(isnap.spot_minutes) else isnap.spot_minutes
        candle_lag = ((now - (today.index[-1] + pd.Timedelta(minutes=1))).total_seconds() / 60) if len(today) else None
        chain_age = (now - isnap.chain_time).total_seconds() if isnap.chain_time is not None else None
        problems = []
        if spot_age is None or spot_age > cfg.max_spot_age_seconds:
            problems.append(f"spot price {('%.0f s' % spot_age) if spot_age is not None else 'missing'} old")
        if candle_lag is None or candle_lag > cfg.max_candle_lag_minutes:
            problems.append(f"candles {('%.0f min' % candle_lag) if candle_lag is not None else 'missing'} behind")
        chain_ok = chain_age is not None and chain_age <= cfg.max_chain_age_seconds and isnap.chain is not None and not isnap.chain.empty
        if isnap.errors:
            problems.append(isnap.errors[0][:80])
        warmup = in_session and now.time() < time(9, 18)
        ok = not problems or warmup
        text = (f"spot {spot_age:.0f}s" if spot_age is not None else "spot n/a") + \
               (f", candles {candle_lag:.0f}m" if candle_lag is not None else ", candles n/a") + \
               (f", chain {chain_age:.0f}s" if chain_age is not None else ", chain n/a") + \
               f" ({isnap.sources.get('spot', '?')}/{isnap.sources.get('futures', 'no futures')}/{isnap.chain_source or 'no chain'})"
        return {"ok": ok and in_session, "reason": "; ".join(problems) if problems else ("" if in_session else "outside session"),
                "spot_age_s": spot_age, "candle_lag_min": candle_lag, "chain_age_s": chain_age, "chain_ok": chain_ok,
                "text": text, "sources": dict(isnap.sources, chain=isnap.chain_source)}

    def _process_index(self, name, isnap: IndexSnapshot, snap: MarketSnapshot, now, in_session):
        cfg = self.cfg
        fresh = self._freshness(isnap, now, in_session)
        self.feed[name] = fresh
        if in_session and now.time() >= time(9, 20):
            if not fresh["ok"] and not self.stale_alerted[name]:
                self.stale_alerted[name] = True
                self._emit({"time": now.isoformat(), "index": name, "direction": "", "stage": sig.NO_TRADE, "kind": "data",
                            "message": f"Data feed stale or disconnected: {fresh['reason']}", "freshness": fresh["text"],
                            "setup_key": f"data|{name}|{now:%H%M}", "mock": self.mock}, push=True)
            elif fresh["ok"] and self.stale_alerted[name]:
                self.stale_alerted[name] = False
                self._emit({"time": now.isoformat(), "index": name, "direction": "", "stage": "DATA OK", "kind": "data",
                            "message": "Data feed fresh again", "freshness": fresh["text"],
                            "setup_key": f"dataok|{name}|{now:%H%M}", "mock": self.mock}, push=True)
        if isnap.spot_minutes is None or len(isnap.spot_minutes) < 50:
            return
        ff = strat.build_features(isnap.spot_minutes, cfg, isnap.fut_minutes, isnap.daily if len(isnap.daily) else None)
        bars = ff.bars
        if bars.empty:
            return
        ends = bars.index + pd.Timedelta(minutes=cfg.signal_minutes)
        complete = (ends + pd.Timedelta(seconds=8) <= now) & ((bars["minutes_in_bar"] >= cfg.signal_minutes) |
                                                               (ends + pd.Timedelta(seconds=90) <= now))
        positions = np.nonzero(complete.to_numpy())[0]
        if len(positions) == 0:
            return
        tracker = self.trackers[name]
        tracker2 = self.v2.trackers[name] if self.v2 is not None else None
        tracker3 = self.v3.trackers[name] if self.v3 is not None else None
        deriv = self._derivatives(name, isnap, ff, int(positions[-1]), now) if tracker3 is not None else None
        today_positions = [p for p in positions if ff.dates[p] == now.date()]
        last_done = self.last_pos[name]
        todo = [p for p in today_positions if last_done is None or bars.index[p] > last_done]
        ctx = self._context(name)
        ctx["breadth"] = snap.breadth.get(name) or {}
        mf = score_v2.build_minutes(isnap.spot_minutes, isnap.fut_minutes) if tracker2 is not None else None
        if not self.warm[name]:
            # restart: replay today's earlier bars silently, then never resend what was already sent
            gate = tracker.ml_gate
            tracker.ml_gate = None
            for p in todo[:-1]:
                ev = strat.evaluate_bar(ff, p, cfg, name, ctx)
                if ev is not None:
                    wb, eb = self._blocks(name, isnap, ev, fresh_ok=True)
                    tracker.step(ev, watch_block=wb, entry_block=eb, data_ok=True, data_reason="", risk=None, resolver=None)
                    if tracker2 is not None:
                        tracker2.step(score_v2.rescore(ev, ff, mf, ctx), watch_block=wb, entry_block=eb, data_ok=True,
                                      data_reason="", risk=None, resolver=None)
                    if tracker3 is not None:
                        tracker3.step(score_v3.rescore(ev, ff, self.cfg_v3, ctx, None), watch_block=wb, entry_block=eb,
                                      data_ok=True, data_reason="", risk=None, resolver=None)
            tracker.ml_gate = gate
            sent = self.store.sent_today(now.date(), self.mock)
            tracker.sent |= sent
            if tracker2 is not None:
                tracker2.sent |= sent
            if tracker3 is not None:
                tracker3.sent |= sent
            self.warm[name] = True
            todo = todo[-1:]
        for p in todo:
            ev = strat.evaluate_bar(ff, p, cfg, name, ctx)
            self.last_pos[name] = bars.index[p]
            if ev is None:
                continue
            self.latest[name] = ev
            self.biases[name] = ev.bias
            try:
                self.store.add_bar(ev, self.mock)
            except sqlite3.Error:
                pass
            if not in_session:
                continue
            ev2 = score_v2.rescore(ev, ff, mf, ctx) if tracker2 is not None else None
            if ev2 is not None:
                self.v2.latest[name] = ev2
            ev3 = score_v3.rescore(ev, ff, self.cfg_v3, ctx, deriv) if tracker3 is not None else None
            if ev3 is not None:
                self.v3.latest[name] = ev3
            if self.ml is not None:
                self.ml.ctx = ctx
                if ev2 is not None:
                    self.ml.v2[(name, inst.CE)], self.ml.v2[(name, inst.PE)] = ev2.ce, ev2.pe
            wb, eb = self._blocks(name, isnap, ev, fresh_ok=fresh["ok"])
            chain_ok = fresh["chain_ok"]
            stale = (lambda *a: (None, [], "option chain stale or missing"))
            resolver = self._resolver(name, isnap, now) if chain_ok else stale
            lanes = [(tracker, ev, self.book, self.risk, True, resolver, "v1")]
            if tracker2 is not None:
                lanes.append((tracker2, ev2, self.v2.book, self.v2.risk, self.v2_push, resolver, "v2"))
            if tracker3 is not None:
                lanes.append((tracker3, ev3, self.v3.book, self.v3.risk, self.v3_push,
                              self._resolver(name, isnap, now, self.cfg_v3) if chain_ok else stale, "v3"))
            for trk, e, book, risk, push, lane_resolver, model in lanes:
                new = trk.step(e, watch_block=wb, entry_block=eb, data_ok=fresh["ok"], data_reason=fresh["reason"],
                               risk=risk, resolver=lane_resolver, freshness=fresh["text"])
                for a in new:
                    d = a.to_dict()
                    d["mock"] = self.mock
                    late = (now - pd.Timestamp(a.time)).total_seconds() > 600
                    self._emit(d, push=push and not late)
                    if a.kind == "confirmed" and trk is tracker:
                        self._log_ml_sample(ev, ev2, a, ctx)
                    if a.kind == "invalidated" and cfg.exit_on_failed_breakout:
                        self._exit_failed_breakout(book, a, now, push)
                    if a.kind == "entry" and self.app.paper_trading:   # alert-only mode: no paper position
                        pos = book.open(a, now.isoformat(), isnap.exchange, mock=self.mock)
                        if pos is not None:
                            risk.record_open(name, a.direction)
                            self.store.save_trade(pos)
                            logger.warning("[%s] PAPER BUY (%s) %s x%d @ %.2f", name, a.model, pos.symbol, pos.qty_open,
                                           pos.entry_price)
                            self._exec_entry(model, pos, isnap, now, fresh["ok"] and in_session)
        if self.latest[name] is None and len(positions):
            ev = strat.evaluate_bar(ff, positions[-1], cfg, name, ctx)
            self.latest[name] = ev
        if self.v2 is not None and self.v2.latest.get(name) is None and self.latest[name] is not None and mf is not None:
            self.v2.latest[name] = score_v2.rescore(self.latest[name], ff, mf, ctx)
        if self.v3 is not None and self.v3.latest.get(name) is None and self.latest[name] is not None:
            self.v3.latest[name] = score_v3.rescore(self.latest[name], ff, self.cfg_v3, ctx, deriv)
        if self.latest[name] is not None:
            self.exit_ctx[name] = exit_context(ff, self.latest[name], self.cfg)
        self._label_ml_samples(name, isnap)
        # intrabar: live price beyond a watched level before the bar closes (shown, alert optional)
        self._intrabar(name, isnap, now)

    def _log_ml_sample(self, ev: strat.BarEval, ev2, alert: sig.Alert, ctx: dict):
        """Every CONFIRMED setup becomes a training example; its label is filled in once the outcome is known."""
        try:
            d = ev.side(alert.direction)
            plan = strat.underlying_plan(alert.direction, d, ev.close, ev.atr, self.cfg)
            feats = ml_filter.features(ev, d, ctx, ev2.side(alert.direction) if ev2 is not None else None)
            self.store.add_ml_sample(alert.setup_key, ev, alert.direction, ev.close, plan.stop, feats, self.mock)
        except Exception:
            logger.exception("Could not log an ML sample")

    def _label_ml_samples(self, name, isnap):
        try:
            for sid, t, direction, entry, stop in self.store.pending_ml_samples(name, self.mock):
                lab = ml_filter.label_outcome(isnap.spot_minutes, pd.Timestamp(t), direction, entry, stop,
                                           self.cfg.max_hold_minutes, self.cfg.square_off)
                if lab is not None:
                    self.store.label_ml_sample(sid, lab[0], lab[1])
        except Exception:
            logger.exception("Could not label ML samples")

    def _context(self, name) -> dict:
        others = [b for n, b in self.biases.items() if n != name]
        ctx = {"vix": self.vix, "vix_change_pct": self.vix_change}
        if others:
            ctx["agree_ce"] = sum(1 for b in others if b == inst.CE) / len(others)
            ctx["agree_pe"] = sum(1 for b in others if b == inst.PE) / len(others)
        return ctx

    def _blocks(self, name, isnap, ev: strat.BarEval, fresh_ok: bool):
        dl = ev.levels
        gap = np.nan
        if num.finite(isnap.day_open) and dl.get("pdc") and dl.get("daily_atr"):
            gap = (isnap.day_open - dl["pdc"]) / dl["daily_atr"]
        return riskm.session_block(ev.close_time, self.cfg, is_expiry_day=isnap.is_expiry_day, gap_atr_mult=gap,
                                 vix=self.vix, event=market_calendar.event_now(self.events, ev.close_time))

    def _resolver(self, name, isnap: IndexSnapshot, now, cfg: Optional[PBConfig] = None):
        cfg = cfg or self.cfg

        def resolve(direction, plan, ev):
            return opt_sel.select_option(isnap.chain, ev.close if not num.finite(isnap.ltp) else isnap.ltp, direction, plan,
                                     isnap.step, isnap.lot_size, isnap.expiry, now, cfg,
                                     lambda s, p, q: self.costs(s, p, q, isnap.exchange),
                                     iv_fallback=self.vix, symbol_prefix=name)
        return resolve

    def _intrabar(self, name, isnap, now):
        tracker = self.trackers[name]
        ev = self.latest.get(name)
        if ev is None or not num.finite(isnap.ltp):
            return
        for direction, setup in tracker.active.items():
            if setup is None or setup.stage != sig.WATCH:
                continue
            buffer = max(self.cfg.buffer_atr * ev.atr, isnap.ltp * self.cfg.buffer_pct / 100)
            sign = 1 if direction == inst.CE else -1
            if sign * (isnap.ltp - setup.level) > buffer:
                setup.avoid_reason = "INTRABAR break - not confirmed until the bar closes"
                if self.cfg.intrabar_alerts and (setup.key, "intrabar") not in tracker.sent:
                    tracker.sent.add((setup.key, "intrabar"))
                    self._emit({"time": now.isoformat(), "index": name, "direction": direction, "stage": "INTRABAR",
                                "kind": "intrabar", "price": isnap.ltp, "level": setup.level, "score": setup.score,
                                "message": "Intrabar break (unconfirmed: wait for the candle close)",
                                "setup_key": setup.key, "mock": self.mock}, push=True)

    def _manage_paper(self, now):
        for model, book, _, push in self._books():
            self._manage_book(now, book, push, model)

    # ---------- derivatives / execution helpers ----------
    def _derivatives(self, name, isnap: IndexSnapshot, ff, pos: int, now) -> deriv_mod.DerivSnapshot:
        if not self.feed.get(name, {}).get("chain_ok"):
            snap = deriv_mod.DerivSnapshot(note="option chain stale or missing")
            self.deriv[name] = snap
            return snap
        spot = isnap.ltp if num.finite(isnap.ltp) else ff.v("close", pos)
        snap = deriv_mod.analyse_chain(isnap.chain, spot, ff.v("atr", pos), isnap.step)
        if snap.available and num.finite(snap.atm_iv):
            hist = self.iv_hist.setdefault(name, [])
            hist.append((now, snap.atm_iv))
            del hist[:-120]
            old = [iv for t, iv in hist if (now - t).total_seconds() >= 25 * 60]
            if old:
                snap.atm_iv_change_pct = (snap.atm_iv / old[-1] - 1.0) * 100.0
            try:
                self.store.set_iv_daily(now.date(), name, snap.atm_iv, self.mock)
                past = self.store.iv_daily(name, now.date(), self.mock)
                snap.iv_history_days = len(past)
                if len(past) >= 20:
                    snap.iv_percentile = 100.0 * sum(1 for v in past if v < snap.atm_iv) / len(past)
            except sqlite3.Error:
                pass
        self.deriv[name] = snap
        return snap

    def _chain_quote(self, index: str, strike: float, direction: str):
        isnap = self.snapshots.get(index)
        if isnap is None or isnap.chain is None or isnap.chain.empty:
            return np.nan, np.nan, None
        side = "ce" if direction == inst.CE else "pe"
        row = isnap.chain[(isnap.chain["strike"] - strike).abs() < 1e-6]
        if row.empty:
            return np.nan, np.nan, None
        return num.fnum(row[f"{side}_bid"].iloc[0]), num.fnum(row[f"{side}_ask"].iloc[0]), isnap.chain_time

    def _exec_manager(self, model: str, now):
        if self.execution is None or model != self.execution.s.lane:
            return None
        try:
            return self.execution.ensure(now.date())
        except Exception:
            logger.exception("execution unavailable")
            return None

    def _exec_entry(self, model, pos, isnap, now, data_ok: bool):
        mgr = self._exec_manager(model, now)
        if mgr is None:
            return
        try:
            quote = self._chain_quote(pos.index, pos.strike, pos.direction)
            ok, why = mgr.on_entry(pos, quote, now, data_ok=data_ok, market_open=data_ok,
                                   last_entry=self.cfg.last_entry, live_open_limit=self.cfg.max_open_positions)
            if not ok:
                logger.warning("[%s] %s order for %s not sent: %s", pos.index, mgr.broker.name, pos.symbol, why)
        except Exception:
            logger.exception("[%s] execution of the entry failed (paper position unaffected)", pos.index)

    def _exec_exit(self, model, pos, qty, qty_before, now, reason):
        mgr = self._exec_manager(model, now)
        if mgr is None:
            return
        try:
            bid, ask, _ = self._chain_quote(pos.index, pos.strike, pos.direction)
            mgr.on_exit(pos, qty, qty_before, (bid, ask), now, reason)
        except Exception:
            logger.exception("[%s] execution of an exit failed - check the broker positions", pos.index)

    def _poll_execution(self, now):
        if self.execution is None:
            return
        try:
            mgr = self.execution.ensure(now.date())
            if mgr is None:
                return
            quotes = {}
            for lp in mgr.positions.values():
                if lp.closed:
                    continue
                pos = next((p for _, b, _, _ in self._books() for p in b.positions if p.id == lp.position_id), None)
                if pos is not None:
                    bid, ask, _ = self._chain_quote(pos.index, pos.strike, pos.direction)
                    quotes[lp.instrument_key] = (bid, ask)
            mgr.poll(now, quotes)
        except Exception:
            logger.exception("execution poll failed")

    def _manage_book(self, now, book, push, model="v1"):
        for pos in book.open_positions():
            isnap = self.snapshots.get(pos.index)
            if isnap is None or isnap.chain is None or isnap.chain.empty:
                continue
            side = "ce" if pos.direction == inst.CE else "pe"
            row = isnap.chain[(isnap.chain["strike"] - pos.strike).abs() < 1e-6]
            if row.empty:
                continue
            bid = num.fnum(row[f"{side}_bid"].iloc[0])
            if not (num.finite(bid) and bid > 0):
                ltp = num.fnum(row[f"{side}_ltp"].iloc[0])
                bid = ltp * (1 - self.cfg.assumed_spread_pct / 200) if num.finite(ltp) else np.nan
            fresh = self.feed.get(pos.index, {})
            if not num.finite(bid) or not fresh.get("chain_ok"):
                continue  # never mark or exit on a stale option price
            ev = self.latest.get(pos.index)
            qty_before, sl_before = pos.qty_open, pos.sl
            events = book.mark(pos, bid=bid, underlying=isnap.ltp, now=now, exchange=isnap.exchange,
                               bar_close_underlying=ev.close if ev is not None else None,
                               ctx=self.exit_ctx.get(pos.index))
            for e in events:
                self._exec_exit(model, pos, e["qty"], qty_before, now, e["reason"])
                qty_before -= e["qty"]
            if not pos.closed and num.finite(pos.sl) and pos.sl != sl_before:
                mgr = self._exec_manager(model, now)
                if mgr is not None:
                    mgr.update_stop(pos, now)
            for e in events:
                net_trade = pos.realised_gross - pos.charges if pos.closed else None
                self._emit({"time": now.isoformat(), "index": pos.index, "direction": pos.direction, "stage": sig.EXIT,
                            "kind": "exit", "price": isnap.ltp, "level": None, "score": None,
                            "message": f"{e['reason']}: SELL {e['qty']} {pos.symbol} @ {e['price']:.2f}"
                                       + (f" | trade net Rs {net_trade:,.0f}" if net_trade is not None else ""),
                            "setup_key": f"{pos.id}|{e['reason']}", "option": {"symbol": pos.symbol, "ltp": bid},
                            "mock": self.mock, "model": trade_model(pos.setup_key)}, push=push)
            self.store.save_trade(pos)

    def _chain_bid(self, pos) -> float:
        """Sellable price of a paper position from the latest chain (bid, else LTP minus half the spread)."""
        isnap = self.snapshots.get(pos.index)
        if isnap is None or isnap.chain is None or isnap.chain.empty:
            return np.nan
        side = "ce" if pos.direction == inst.CE else "pe"
        row = isnap.chain[(isnap.chain["strike"] - pos.strike).abs() < 1e-6]
        if row.empty:
            return np.nan
        bid = num.fnum(row[f"{side}_bid"].iloc[0])
        if not (num.finite(bid) and bid > 0):
            ltp = num.fnum(row[f"{side}_ltp"].iloc[0])
            bid = ltp * (1 - self.cfg.assumed_spread_pct / 200) if num.finite(ltp) else np.nan
        return bid

    def _exit_failed_breakout(self, book, alert: sig.Alert, now, push):
        """The breakout behind an open paper position closed back inside its level: sell at the bid."""
        pos = book.find_open(alert.setup_key)
        if pos is None:
            return
        if not self.feed.get(pos.index, {}).get("chain_ok"):
            logger.warning("[%s] failed breakout but the option chain is stale: exit left to the next fresh mark", pos.index)
            return  # never exit on a stale option price; the underlying stop / time stop still apply
        bid = self._chain_bid(pos)
        if not num.finite(bid):
            return
        isnap = self.snapshots.get(pos.index)
        qty_before = pos.qty_open
        for e in book.close(pos, bid=bid, now=now, exchange=isnap.exchange, reason="failed breakout"):
            self._exec_exit(trade_model(pos.setup_key), pos, e["qty"], qty_before, now, e["reason"])
            self._emit({"time": now.isoformat(), "index": pos.index, "direction": pos.direction, "stage": sig.EXIT,
                        "kind": "exit", "price": isnap.ltp, "level": None, "score": None,
                        "message": f"{e['reason']}: SELL {e['qty']} {pos.symbol} @ {e['price']:.2f}"
                                   f" | trade net Rs {pos.realised_gross - pos.charges:,.0f}",
                        "setup_key": f"{pos.id}|failed", "option": {"symbol": pos.symbol, "ltp": bid},
                        "mock": self.mock, "model": trade_model(pos.setup_key)}, push=push)
        self.store.save_trade(pos)

    def _kill_switch_reason(self) -> Optional[str]:
        """A file named KILL_SWITCH in the project root (see [SERVICE] kill_switch_file) blocks new entries and
        squares off every open paper position at the bid. Delete the file to allow entries again."""
        if self.mock:
            return None
        return "kill switch is on" if Path(self.app.kill_switch_file).exists() else None

    def _kill_switch(self, now):
        reason = self._kill_switch_reason()
        if reason != self.risk.external_block:
            if reason:
                logger.critical("New entries blocked: %s", reason)
            else:
                logger.warning("Kill switch cleared: entries allowed again.")
        for _, _, risk, _ in self._books():
            risk.external_block = reason
        if not reason:
            return
        mgr = self.execution.manager if self.execution is not None else None
        if mgr is not None and any(not p.closed and p.qty > 0 for p in mgr.positions.values()):
            try:
                mgr.close_all(now, reason)
            except Exception:
                logger.exception("emergency close at the broker failed - CHECK THE BROKER NOW")
        for pos in [p for _, book, _, _ in self._books() for p in book.open_positions()]:  # square off, like the engine
            isnap = self.snapshots.get(pos.index)
            bid = self._chain_bid(pos)
            if not num.finite(bid):
                bid = pos.last_price  # last known price, as the engine does when flattening
            exchange = isnap.exchange if isnap is not None else "NSE_FO"
            book = self._lane_book(trade_model(pos.setup_key))[0]
            for e in book.close(pos, bid=bid, now=now, exchange=exchange, reason=f"risk: {reason}"):
                self._emit({"time": now.isoformat(), "index": pos.index, "direction": pos.direction, "stage": sig.EXIT,
                            "kind": "exit", "price": isnap.ltp if isnap is not None else None, "level": None,
                            "score": None, "message": f"{e['reason']}: SELL {e['qty']} {pos.symbol} @ {e['price']:.2f}"
                            f" | trade net Rs {pos.realised_gross - pos.charges:,.0f}",
                            "setup_key": f"{pos.id}|risk", "option": {"symbol": pos.symbol}, "mock": self.mock,
                            "model": trade_model(pos.setup_key)}, push=True)
            self.store.save_trade(pos)

    def _emit(self, alert: dict, push: bool):
        alert.setdefault("model", "v1")
        alert = _clean(alert)
        try:
            self.store.add_alert(alert)
        except sqlite3.Error:
            logger.exception("Could not store an alert")
        self.recent.insert(0, alert)
        self.recent = self.recent[:200]
        self.notifier.emit(alert, push=push and self.push)

    # ---------- outputs ----------
    def _index_state(self, name) -> dict:
        ev = self.latest.get(name)
        isnap = self.snapshots.get(name)
        tracker = self.trackers[name]
        feed = self.feed.get(name, {})
        st = {"name": name, "feed": feed, "price": None, "change_pct": None, "bias": "NEUTRAL", "state": sig.NO_TRADE,
              "state_reason": feed.get("reason") or "", "ce": None, "pe": None, "levels": {}, "indicators": {},
              "structure": {}, "zones": [], "setups": tracker.snapshot(), "shortlist": [], "expiry": None,
              "lot_size": None, "step": None, "bar_time": None}
        if isnap is not None:
            st.update(price=num.r2(isnap.ltp), expiry=isnap.expiry.isoformat() if isnap.expiry else None,
                      nearest_expiry=isnap.nearest_expiry.isoformat() if isnap.nearest_expiry else None,
                      expiry_day=isnap.is_expiry_day, lot_size=isnap.lot_size or None, step=num.r2(isnap.step),
                      chain_source=isnap.chain_source, errors=isnap.errors[:3])
            if num.finite(isnap.ltp) and num.finite(isnap.prev_close) and isnap.prev_close:
                st["change_pct"] = round((isnap.ltp / isnap.prev_close - 1) * 100, 2)
        if ev is not None:
            st.update(bias=ev.bias, levels=ev.levels, indicators=ev.indicators, structure=ev.structure,
                      bar_time=ev.close_time.isoformat(),
                      zones=sorted(ev.zones, key=lambda z: abs(z["price"] - ev.close))[:8])
            for direction in inst.OPTION_TYPES:
                d = ev.side(direction)
                st[direction.lower()] = {"score": d.score, "tier": self.cfg.tier(d.score), "breakdown": d.breakdown(),
                                         "supporting": d.supporting, "contradicting": d.contradicting,
                                         "level": d.level, "level_price": num.r2(d.level_price),
                                         "distance_atr": num.r2(d.distance_atr), "breakout": d.breakout,
                                         "quality": d.breakout_quality}
        # headline state
        active = [s for s in tracker.active.values() if s is not None]
        order = {sig.ENTRY: 4, sig.CONFIRMED: 3, sig.WATCH: 2}
        if not feed.get("ok"):
            st["state"] = sig.NO_TRADE
        elif self.risk.state.halted:
            st["state"], st["state_reason"] = sig.NO_TRADE, self.risk.state.halt_reason
        elif active:
            top = max(active, key=lambda s: order.get(s.stage, 0))
            st["state"] = top.stage
            st["state_reason"] = f"{top.direction} {top.level_label} {top.level:,.1f}" + \
                                 (f" | {top.avoid_reason}" if top.avoid_reason else "")
            if top.option:
                st["shortlist"] = top.option.get("shortlist", [])
        else:
            st["state"], st["state_reason"] = "IDLE", "no setup near a level"
        # a shortlist for a live watch, so the option is ready before the break
        if not st["shortlist"] and ev is not None and isnap is not None and isnap.chain is not None and feed.get("chain_ok"):
            watch = [s for s in active if s.stage == sig.WATCH]
            if watch:
                s = watch[0]
                d = ev.side(s.direction)
                if num.finite(d.level_price):
                    plan = strat.underlying_plan(s.direction, d, ev.close, ev.atr, self.cfg)
                    _, ranked, _ = opt_sel.select_option(isnap.chain, isnap.ltp if num.finite(isnap.ltp) else ev.close, s.direction,
                                                     plan, isnap.step, isnap.lot_size, isnap.expiry, self.provider.now(),
                                                     self.cfg, lambda a, b, c: self.costs(a, b, c, isnap.exchange),
                                                     iv_fallback=self.vix, symbol_prefix=name)
                    st["shortlist"] = [c.to_dict() for c in ranked[:5]]
                    st["shortlist_note"] = "prepared for the watch (not an entry)"
        if self.v2 is not None:
            ev2 = self.v2.latest.get(name)
            tr2 = self.v2.trackers[name]
            active2 = [x for x in tr2.active.values() if x is not None]
            v2s = {"setups": tr2.snapshot(), "state": "IDLE", "state_reason": ""}
            if active2:
                top2 = max(active2, key=lambda x: order.get(x.stage, 0))
                v2s["state"] = top2.stage
                v2s["state_reason"] = f"{top2.direction} {top2.level_label} {top2.level:,.1f}" + \
                                      (f" | {top2.avoid_reason}" if top2.avoid_reason else "")
            if ev2 is not None:
                for direction in inst.OPTION_TYPES:
                    d2 = ev2.side(direction)
                    v2s[direction.lower()] = {"score": d2.score, "breakdown": d2.breakdown(), "supporting": d2.supporting,
                                              "contradicting": d2.contradicting}
            st["v2"] = v2s
        if self.v3 is not None:
            ev3 = self.v3.latest.get(name)
            tr3 = self.v3.trackers[name]
            active3 = [x for x in tr3.active.values() if x is not None]
            v3s = {"setups": tr3.snapshot(), "state": "IDLE", "state_reason": ""}
            if active3:
                top3 = max(active3, key=lambda x: order.get(x.stage, 0))
                v3s["state"] = top3.stage
                v3s["state_reason"] = f"{top3.direction} {top3.level_label} {top3.level:,.1f}" + \
                                      (f" | {top3.avoid_reason}" if top3.avoid_reason else "")
            if ev3 is not None:
                v3s.update(regime=ev3.extras.get("regime"), modules=ev3.extras.get("modules"),
                           weights=ev3.extras.get("weights"), vwap_source=ev3.extras.get("vwap_source"),
                           derivatives=ev3.extras.get("derivatives"))
                for direction in inst.OPTION_TYPES:
                    d3 = ev3.side(direction)
                    v3s[direction.lower()] = {"score": d3.score, "breakdown": d3.breakdown(), "supporting": d3.supporting,
                                              "contradicting": d3.contradicting, "veto": d3.entry_veto,
                                              "traps": d3.extras.get("traps", []),
                                              "status": ("TRIGGERED" if any(x.stage == sig.ENTRY and x.direction == direction for x in active3)
                                                         else "READY" if any(x.stage == sig.CONFIRMED and x.direction == direction for x in active3)
                                                         else "WATCH" if any(x.stage == sig.WATCH and x.direction == direction for x in active3)
                                                         else "-")}
            st["v3"] = v3s
            st["regime"] = (ev3.extras.get("regime") if ev3 is not None else None)
        return st

    def _paper_state(self, model: str = "v1") -> dict:
        lane = {"v2": self.v2, "v3": self.v3}.get(model)
        book = self.book if model == "v1" else (lane.book if lane is not None else paper_mod.PaperBook(self.cfg, self.costs))
        today = [p.to_dict() for p in book.positions]
        all_trades = self.store.trades(mock=self.mock, model=model)
        closed = [t for t in all_trades if t.get("exit_time")]
        nets = [t.get("net_pnl") or 0 for t in closed]
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n <= 0]
        return {"today": today, "summary": {
            "trades": len(closed), "net": round(sum(nets), 0), "win_rate": round(len(wins) / len(nets) * 100, 1) if nets else None,
            "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else None,
            "avg": round(float(np.mean(nets)), 0) if nets else None},
            "history": closed[-100:]}

    def _event_status(self, now) -> dict:
        """Upcoming entries of config/events.csv. With none, event-risk filtering is reported as inactive."""
        try:
            upcoming = [(d, t, n) for d, t, n in (self.events or []) if 0 <= (d - now.date()).days <= 7]
        except Exception:
            upcoming = []
        if not upcoming:
            return {"active": False, "note": "event calendar has no entries for the next 7 days (config/events.csv): "
                                              "event-risk filtering is NOT active", "upcoming": []}
        return {"active": True, "note": "", "upcoming": [f"{d} {t or 'all day'} {n}" for d, t, n in upcoming[:5]]}

    def _execution_note(self) -> str:
        if self.execution is None:
            return "Alerts and paper trading only - no orders are placed."
        if self.execution.armed:
            return f"LIVE ORDERS ARMED for today (lane {self.execution.s.lane}, max {self.execution.s.max_live_lots} lot(s))."
        return f"Orders are simulated (paper broker, lane {self.execution.s.lane}) - no real orders are placed."

    def _publish(self, now: pd.Timestamp, market_open: bool) -> dict:
        state = {
            "generated_at": now.isoformat(), "mode": "MOCK" if self.mock else "LIVE", "mock": self.mock,
            "trading_mode": "paper" if self.app.paper_trading else "alerts only",
            "status": self.status, "market_open": market_open,
            "trading_day": market_calendar.is_trading_day(now.date()) or self.mock,
            "holiday": market_calendar.holiday_name(now.date()) if not self.mock else None,
            "vix": num.r2(self.vix), "vix_change_pct": num.r2(self.vix_change),
            "risk": self.risk.snapshot(), "thresholds": self.cfg.thresholds_dict(),
            "channels": self.notifier.channels(), "indices": {n: self._index_state(n) for n in self.names},
            "recent_alerts": [a for a in self.recent if (a.get("model") or "v1") == "v1"][:60],
            "recent_alerts_v2": [a for a in self.recent if a.get("model") == "v2"][:30],
            "paper": self._paper_state("v1"), "paper_v2": self._paper_state("v2") if self.v2 is not None else None,
            "risk_v2": self.v2.risk.snapshot() if self.v2 is not None else None,
            "recent_alerts_v3": [a for a in self.recent if a.get("model") == "v3"][:30],
            "paper_v3": self._paper_state("v3") if self.v3 is not None else None,
            "risk_v3": self.v3.risk.snapshot() if self.v3 is not None else None,
            "v3_settings": {k: v for k, v in self.v3_overrides.items()},
            "events": self._event_status(now),
            "execution": self.execution.snapshot() if self.execution is not None else {"mode": "off"},
            "ml": self.ml.status() if self.ml is not None else {"loaded": False, "note": "ML disabled in config/scanner.ini"},
            "websocket": self.feed_status() if self.feed_status is not None else None,
            "cycle": {"count": self.cycle_count, "last_error": self.last_cycle_error},
            "notes": [self._execution_note(),
                      "Scores are setup-strength scores (heuristic), not probabilities.",
                      "Backtest: python -m src.backtest (output/prebreakout_backtest_report.html)."],
        }
        state = _clean(state)
        try:
            atomic_write(settings.STATE_FILE, json.dumps(state, default=json_default, indent=1))
            atomic_write(settings.HTML_FILE, view.render_html(state))
        except OSError as error:
            logger.warning("Could not write the state / dashboard files: %s", error)
        self.last_state = state
        return state
