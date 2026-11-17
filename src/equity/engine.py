"""
The equity engine: one background thread that keeps both modules current.

    quote cycle   every [SERVICE] quote_seconds while the market is open: ONE batched quote call for the 50 stocks and
                  the index -> live prices, pending entries, stops / targets / square-off, live order reconciliation
    scan cycle    every scan_seconds while the market is open (once after the close, once before the open):
                  today's 1-minute candles of every stock -> completed 5m bars (15m / 30m / 1h derived from them) ->
                  the 25 intraday strategies on today's bars; daily bars -> the 25 swing strategies on the last
                  COMPLETED session; hourly bars -> 1h / 4h trend used to time swing entries

Only completed bars are evaluated. A signal is ACTIVE (alerted, handed to the trader) only while it is fresh:
intraday within signal_valid_minutes of its bar close inside the entry window; swing on the session after its bar.
Everything else is stored with the reason it was not traded (LATE, WEAK, BLOCKED ...).
"""
from __future__ import annotations

import logging
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.equity import strategies as strat
from src.equity.alerts import Alerter
from src.equity.config import RUNTIME_KEYS, EquityConfig, coerce, enabled_ids, hhmm
from src.equity.data import CLOSE, INDEX, OPEN, TZ, EquityData, is_trading_day, last_session, market_open, now_ist
from src.equity.execution import HELD, Trader, sessions_held
from src.equity.indicators import (add_cross_section, enrich_daily, enrich_index, enrich_intraday, resample_bars, trend_state,
                                   weekly_bars)
from src.equity.risk import Costs, RiskManager
from src.equity.signals import (INTRADAY_METHOD, SWING_METHOD, no_trade_reason, num, opportunities, raw_signals,
                                snapshot_intraday, snapshot_swing)
from src.equity.store import Store
from src.scanner.indicators import ema
from src.fno.indicators import session_vwap, supertrend

logger = logging.getLogger("scanner.equity.engine")


class EquityEngine:
    def __init__(self, cfg: Optional[EquityConfig] = None, data: Optional[EquityData] = None, store: Optional[Store] = None,
                 clock=now_ist, push: bool = True, broker=None, gate=None):
        self.cfg = cfg or EquityConfig.from_ini()
        self.clock = clock
        self.store = store or Store(self.cfg.data_dir / "equity.db")
        for k, v in (self.store.get("risk", {}) or {}).items():          # dashboard overrides win over the ini file
            if k in RUNTIME_KEYS:
                setattr(self.cfg.risk, k, coerce(getattr(self.cfg.risk, k), v))
        self.data = data or EquityData(requests_per_second=self.cfg.requests_per_second,
                                       per_minute=self.cfg.max_requests_per_minute, per_30min=self.cfg.max_requests_per_30min)
        self.risk = RiskManager(self.cfg.risk, Costs(self.cfg.costs_intraday, self.cfg.costs_delivery))
        self.alerts = Alerter(self.cfg.alerts, self.store, push=push, clock=clock)
        self.trader = Trader(self.cfg, self.store, self.risk, self.alerts, broker=broker, gate=gate)
        self.strats_i = [strat.INTRADAY[i] for i in enabled_ids(self.cfg.strategies_intraday, list(strat.INTRADAY))]
        self.strats_s = [strat.SWING[i] for i in enabled_ids(self.cfg.strategies_swing, list(strat.SWING))]
        self.lock = threading.RLock()
        self.quotes: Dict[str, dict] = {}
        self.m1: Dict[str, pd.DataFrame] = {}
        self.f5: Dict[str, pd.DataFrame] = {}
        self.fd: Dict[str, pd.DataFrame] = {}
        self.h1: Dict[str, pd.DataFrame] = {}
        self.snap_i: Dict[str, dict] = {}
        self.snap_s: Dict[str, dict] = {}
        self.htf: Dict[str, dict] = {}                 # symbol -> {"h1": trend, "h4": trend}
        self.nifty: dict = {}
        self.weak: Dict[str, dict] = {}                # symbol -> latest intraday trigger that was too weak to trade
        self.sig_i: Optional[List[dict]] = None        # today's stored intraday signals (refreshed by every scan)
        self.sig_s: Optional[List[dict]] = None
        self.errors: Dict[str, str] = {}
        self.status, self.status_text = "starting", "first scan not finished"
        self.cycles = self.scans = 0
        self.last_scan: Optional[pd.Timestamp] = None
        self.last_quote: Optional[pd.Timestamp] = None
        self.scan_seconds = None
        self._swing_day = None
        self.data_through = None                       # date of the newest intraday bar that is loaded
        self._hourly_at: Optional[pd.Timestamp] = None
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.job = {"running": False, "module": None, "progress": "", "error": None, "finished": None}
        old = (self.clock().date() - timedelta(days=15)).isoformat()      # untraded intraday signals are kept 15 days
        with self.store.lock:
            self.store.db.execute("DELETE FROM signals WHERE module='intraday' AND status='LATE' AND day<?", (old,))

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="equity-engine")
            self._thread.start()
            if self.cfg.websocket:
                try:
                    self.data.start_feed([s.key for s in self.data.universe()])
                except Exception as error:      # noqa: BLE001 - the REST quotes keep working
                    logger.warning("WebSocket feed not started: %s", error)

    def stop(self):
        self._stop.set()
        self._wake.set()
        try:
            self.data.stop_feed()
        except Exception:      # noqa: BLE001
            pass

    def poke(self):
        self._force = True
        self._wake.set()

    def _scan_due(self, now: pd.Timestamp) -> bool:
        if self.last_scan is None or getattr(self, "_force", False):
            return True
        if market_open(now):
            return (now - self.last_scan).total_seconds() >= self.cfg.scan_seconds
        close = pd.Timestamp(datetime.combine(now.date(), CLOSE)).tz_localize(TZ) + pd.Timedelta(minutes=1)
        if self.last_scan < close <= now and last_session(now) == now.date():
            return True                                 # once after the close: the final bars and the daily bar
        return self.last_scan.date() != now.date() and now.time() >= time(8, 45)

    def _loop(self):
        next_quote = 0.0
        while not self._stop.is_set():
            now = self.clock()
            if self._scan_due(now):
                self._force = False
                self._safe(self.scan_cycle, now)
            if _time.monotonic() >= next_quote:
                self._safe(self.quote_cycle, self.clock())
                next_quote = _time.monotonic() + (self.cfg.quote_seconds if market_open(now) else 120.0)
            self._wake.wait(1.0)
            self._wake.clear()

    def _safe(self, fn, now):
        try:
            fn(now)
            if self.status in ("starting", "error", "token"):
                self.status, self.status_text = "ok", ""
        except Exception as error:      # noqa: BLE001 - the loop must survive anything
            auth = "Auth" in type(error).__name__
            self.status = "token" if auth else "error"
            self.status_text = ("Upstox token missing or expired: run python -m src.broker.auth" if auth
                                else f"{type(error).__name__}: {str(error)[:200]}")
            if self.last_scan is None:
                self.last_scan = now                    # do not hammer the API while the token is missing
            (logger.error if auth else logger.exception)("%s failed: %s", fn.__name__, self.status_text)

    # ------------------------------------------------------------------ quotes
    def quote_cycle(self, now: pd.Timestamp):
        stocks = self.data.universe()
        by_key = {s.key: s.symbol for s in stocks}
        by_key[INDEX.key] = INDEX.symbol
        raw = self.data.quotes(list(by_key))
        quotes = {by_key[k]: q for k, q in raw.items() if k in by_key}
        with self.lock:
            self.quotes.update(quotes)
            self.last_quote = now
            self.cycles += 1
        prices = {s: q["ltp"] for s, q in quotes.items() if np.isfinite(q.get("ltp", np.nan))}
        self.trader.on_quotes(prices, now, market_open(now), self.htf)
        if self.trader.mode == "live":
            self.trader.reconcile(now)

    # ------------------------------------------------------------------ scan
    def _load(self, stock, now):
        try:
            bars = self.data.intraday_bars(stock, now, self.cfg.intraday.history_days)
            daily = self.data.daily_bars(stock, now, self.cfg.swing.history_days, bars["1m"])
            return stock, bars, daily, None
        except Exception as error:      # noqa: BLE001
            if "Auth" in type(error).__name__:
                raise
            return stock, None, None, f"{type(error).__name__}: {str(error)[:120]}"

    def scan_cycle(self, now: pd.Timestamp):
        started = _time.monotonic()
        stocks = self.data.universe()
        nb = self.data.intraday_bars(INDEX, now, self.cfg.intraday.history_days)
        nd = self.data.daily_bars(INDEX, now, self.cfg.swing.history_days, nb["1m"])
        nf = enrich_index(nb["5m"], nd)
        with ThreadPoolExecutor(max_workers=4) as pool:
            loaded = list(pool.map(lambda s: self._load(s, now), stocks))
        errors, f5, m1, daily = {}, {}, {}, {}
        for stock, bars, d, err in loaded:
            if err:
                errors[stock.symbol] = err
                continue
            m1[stock.symbol], daily[stock.symbol] = bars["1m"], d
            if len(bars["5m"]):
                f5[stock.symbol] = enrich_intraday(bars["5m"], d, nf, 5, self.cfg.intraday.orb_minutes)
        session = last_session(now)
        self.data_through = max((f.index[-1].date() for f in f5.values()), default=None)
        with self.lock:
            self.m1, self.f5, self.errors = m1, f5, errors
            self.snap_i = {s: snapshot_intraday(f) for s, f in f5.items()}
            self.nifty = self._nifty_view(nf, nd)
        self._scan_intraday(stocks, f5, now, session)
        self._scan_swing(stocks, daily, nd, now)
        self._scan_hourly(stocks, m1, now)
        self.last_scan, self.scans = now, self.scans + 1
        self.scan_seconds = round(_time.monotonic() - started, 1)
        logger.info("scan %d: %d stocks, %d errors, %.1fs", self.scans, len(f5), len(errors), self.scan_seconds)

    def _nifty_view(self, nf: pd.DataFrame, nd: pd.DataFrame) -> dict:
        if nf is None or not len(nf):
            return {}
        r = nf.iloc[-1]
        d_trend = int(trend_state(nd).iloc[-1]) if len(nd) > 50 else 0
        return {"close": num(r["close"]), "day_pct": num(r["day_pct"]), "h15": int(r["h15"]), "h60": int(r["h60"]),
                "rsi": num(r["rsi"], 1), "daily_trend": d_trend, "bar": nf.index[-1].isoformat()}

    def _validated(self, module: str) -> set:
        return set((self.store.get("validated", {}) or {}).get(module, []))

    def _scan_intraday(self, stocks, f5, now, session):
        cfg = self.cfg.intraday
        start_t, end_t = hhmm(cfg.entry_start), hhmm(cfg.entry_end)
        validated = self._validated("intraday")
        is_open = market_open(now)
        day = session.isoformat()
        seen = {(s["symbol"], s["strategy"], s["side"]) for s in self.store.signals("intraday", day=day)}
        weak_now: Dict[str, dict] = {}
        for stock in stocks:
            f = f5.get(stock.symbol)
            if f is None or f.index[-1].date() != session:
                continue
            i0 = int(np.searchsorted(f.index.values, pd.Timestamp(session.isoformat(), tz=TZ).to_datetime64()))
            for sig in raw_signals(f, self.strats_i, "intraday", cfg, stock.symbol, stock.name, start=i0):
                when = pd.Timestamp(sig["time"])
                if not (start_t <= when.time() <= end_t):
                    continue
                if (sig["score"] or 0) < cfg.min_strength:
                    weak_now[stock.symbol] = sig          # NO TRADE: shown with its reason, not stored
                    continue
                dup = (stock.symbol, sig["strategy"], sig["side"])
                if dup in seen:                           # one signal per stock, setup and side a day
                    continue
                seen.add(dup)
                sig["key"], sig["industry"] = stock.key, stock.industry
                age = (now - when).total_seconds()
                fresh = is_open and 0 <= age <= cfg.signal_valid_minutes * 60
                status = "ACTIVE" if fresh else "LATE"
                if not self.store.add_signal(sig, status, "" if fresh else "seen after its entry window closed"):
                    continue
                if status == "ACTIVE":
                    self._announce(sig)
                    q = self.quotes.get(stock.symbol) or {}
                    ltp = q.get("ltp") if q and np.isfinite(q.get("ltp", np.nan)) else None
                    st, why = self.trader.take(sig, ltp, now, sig["strategy"] in validated)
                    self.store.signal_status(sig["id"], st if st != "ANALYSIS" else "ACTIVE", why)
        with self.lock:
            self.weak = weak_now
            self.sig_i = self.store.signals("intraday", day=day)

    def _announce(self, sig: dict):
        kind = "buy_signal" if sig["side"] == "BUY" else "sell_signal"
        label = "score" if sig["module"] == "intraday" else "confidence"
        text = (f"{sig['module'].upper()} {sig['side']} {sig['symbol']} @ {sig['price']:.2f} · SL {sig['stop']:.2f} · T1 {sig['t1']:.2f} "
                f"· T2 {sig['t2']:.2f} · {label} {sig['score']:.0f}\n{sig['reason']}")
        # one alert per stock and side: per bar intraday, per day swing (other strategies on it are confluence)
        key = f"{sig['module']}|{sig['symbol']}|{sig['side']}|{sig['time'] if sig['module'] == 'intraday' else sig['day']}"
        self.alerts.emit(kind, text, sig["symbol"], sig["module"], {"signal": sig["id"]}, key=key)
        extra = {"reversal": "trend_reversal", "volume": "volume_breakout"}.get(sig["kind"])
        if extra:
            self.alerts.emit(extra, f"{sig['symbol']}: {sig['reason']}", sig["symbol"], sig["module"], key=f"{extra}|{key}")

    def _scan_swing(self, stocks, daily, nd, now):
        ready = {s: d for s, d in daily.items() if d is not None and len(d) >= 60}
        if not ready:
            return
        # the session most stocks have reached: while Upstox is still publishing a day's history only part of the
        # universe has it, and relative-strength ranks over a partial universe would be wrong
        ends = pd.Series([d.index[-1] for d in ready.values()])
        last_day = ends.mode().max()
        if (ends >= last_day).mean() < 0.9:
            last_day = ends.min()
        if self._swing_day != last_day:                  # a new completed session: rebuild the daily frames
            frames = {s: enrich_daily(d[d.index <= last_day], nd[nd.index <= last_day]) for s, d in ready.items()
                      if d.index[-1] >= last_day}
            add_cross_section(frames, {s.symbol: s.industry for s in stocks})
            with self.lock:
                self.fd = frames
                self.snap_s = {s: snapshot_swing(f) for s, f in frames.items()}
            self._swing_day = last_day
        cfg = self.cfg.swing
        validated = self._validated("swing")
        by_symbol = {s.symbol: s for s in stocks}
        # a daily signal can be entered on the session after its bar (today, if the bar is yesterday's)
        # ... and only if that bar really is the latest completed session (history can lag behind by some hours)
        session = last_session(now)
        expected = session
        if session == now.date() and now.time() < CLOSE:        # today's session is still running
            expected -= timedelta(days=1)
            while not is_trading_day(expected):
                expected -= timedelta(days=1)
        tradable = last_day.date() == expected
        for symbol, f in self.fd.items():
            stock = by_symbol.get(symbol)
            if stock is None:
                continue
            # a bar that is not the latest completed session (history still being published) gives no new signals
            for sig in (raw_signals(f, self.strats_s, "swing", cfg, symbol, stock.name, start=len(f) - 1,
                                    max_gap_atr=self.cfg.risk.max_entry_gap_atr) if tradable else []):
                sig["key"], sig["industry"] = stock.key, stock.industry
                sell = sig["side"] == "SELL"
                if (sig["score"] or 0) < cfg.min_confidence:
                    continue                              # NO TRADE: confidence below the minimum
                status = "ALERT" if sell else "ACTIVE"
                if not self.store.add_signal(sig, status, "exit / avoid alert" if sell else ""):
                    continue
                if sell:
                    self._announce(sig)
                    self.trader.flag_exit(symbol, "TREND_EXIT", sig["reason"], now)
                elif status == "ACTIVE":
                    self._announce(sig)
                    st, why = self.trader.take(sig, None, now, sig["strategy"] in validated)
                    self.store.signal_status(sig["id"], st if st != "ANALYSIS" else "ACTIVE", why)
            r = f.iloc[-1]
            if np.isfinite(r["ema20"]) and r["close"] < r["ema20"]:
                self.trader.flag_exit(symbol, "TREND_EXIT", f"daily close {r['close']:.2f} below the 20 EMA {r['ema20']:.2f}", now)
        with self.lock:
            self.sig_s = self.store.signals("swing", since=(last_day.date() - timedelta(days=10)).isoformat())

    def _scan_hourly(self, stocks, m1, now):
        """1h and 4h trend of every stock (completed bars), refreshed when a new hourly bar can exist."""
        if self._hourly_at is not None and (now - self._hourly_at).total_seconds() < 1800:
            return
        htf, h1 = {}, {}
        for stock in stocks:
            try:
                h = self.data.hourly_bars(stock, now, 80, m1.get(stock.symbol))
            except Exception as error:      # noqa: BLE001
                if "Auth" in type(error).__name__:
                    raise
                continue
            if len(h) < 30:
                continue
            h4 = resample_bars(h, 240)
            h1[stock.symbol] = h
            htf[stock.symbol] = {"h1": int(trend_state(h).iloc[-1]), "h4": int(trend_state(h4).iloc[-1]) if len(h4) > 50 else 0}
        with self.lock:
            self.htf, self.h1 = htf, h1
        self._hourly_at = now

    # ------------------------------------------------------------------ settings from the dashboard
    def update_risk(self, values: dict) -> dict:
        limits = {"capital": (10000, 1e9), "risk_per_trade_pct": (0.05, 5), "max_daily_loss_pct": (0.1, 20), "min_net_rr": (0.5, 10),
                  "max_open_intraday": (1, 50), "max_open_swing": (1, 50), "min_rr": (1, 10), "daily_trade_limit": (1, 200),
                  "max_consecutive_losses": (0, 50), "max_position_pct": (1, 100), "slippage_bps": (0, 100)}
        saved = dict(self.store.get("risk", {}) or {})
        for k, v in (values or {}).items():
            if k not in RUNTIME_KEYS:
                continue
            x = float(v)
            lo, hi = limits[k]
            if not (lo <= x <= hi):
                raise ValueError(f"{k} must be between {lo:g} and {hi:g}")
            setattr(self.cfg.risk, k, coerce(getattr(self.cfg.risk, k), x))
            saved[k] = getattr(self.cfg.risk, k)
        self.store.set("risk", saved)
        return self.risk.to_dict()

    # ------------------------------------------------------------------ backtest job
    def run_backtest(self, module: str, years: Optional[float] = None) -> bool:
        if self.job["running"] or module not in ("intraday", "swing"):
            return False
        self.job = {"running": True, "module": module, "progress": "starting", "error": None, "finished": None}

        def work():
            try:
                from src.equity import backtest
                backtest.run(module, self.cfg, self.data, self.store, years=years,
                             progress=lambda text: self.job.update(progress=text))
            except Exception as error:      # noqa: BLE001
                logger.exception("backtest failed")
                self.job["error"] = f"{type(error).__name__}: {str(error)[:200]}"
            finally:
                self.job.update(running=False, finished=self.clock().isoformat(timespec="seconds"))
        threading.Thread(target=work, daemon=True, name="equity-backtest").start()
        return True

    # ------------------------------------------------------------------ views
    def _positions_view(self, rows: List[dict], today) -> List[dict]:
        out = []
        for p in rows:
            s = 1.0 if p["side"] == "BUY" else -1.0
            ltp = self.trader.ltp.get(p["symbol"]) or (self.quotes.get(p["symbol"]) or {}).get("ltp")
            ltp = ltp if ltp is not None and np.isfinite(ltp) else None
            risk0 = abs((p["entry"] or 0) - (p["data"].get("initial_stop") or p["stop"] or 0))
            unreal = (ltp - p["entry"]) * s * p["qty_open"] if ltp and p["status"] in ("OPEN", "EXITING") else None
            net = (p["gross"] or 0) - (p["charges"] or 0)
            sig = p["data"].get("signal", {})
            out.append({"id": p["id"], "module": p["module"], "mode": p["mode"], "symbol": p["symbol"], "side": p["side"],
                        "strategy": p["strategy"], "strategy_name": sig.get("strategy_name"), "qty": p["qty"], "qty_open": p["qty_open"],
                        "entry": p["entry"], "stop": p["stop"], "t1": p["t1"], "t2": p["t2"], "status": p["status"],
                        "created": p["created"], "opened": p["opened"], "closed": p["closed"], "exit_price": p["exit_price"],
                        "exit_reason": p["exit_reason"], "ltp": num(ltp), "unrealized": num(unreal), "realized": num(net),
                        "charges": num(p["charges"]), "t1_done": bool(p["data"].get("t1_done")), "zone": sig.get("zone"),
                        "r": num(((ltp - p["entry"]) * s / risk0) if ltp and risk0 and p["status"] in ("OPEN", "EXITING") else
                                 (net / (risk0 * p["qty"]) if risk0 and p["qty"] and p["status"] == "CLOSED" else None)),
                        "to_stop_pct": num((ltp - p["stop"]) * s / ltp * 100) if ltp and p["status"] == "OPEN" else None,
                        "to_t1_pct": num((p["t1"] - ltp) * s / ltp * 100) if ltp and p["status"] == "OPEN" else None,
                        "to_t2_pct": num((p["t2"] - ltp) * s / ltp * 100) if ltp and p["status"] == "OPEN" else None,
                        "sessions": sessions_held(p["opened"], today) if p["opened"] and p["module"] == "swing" else None,
                        "hold_text": sig.get("hold_text"), "exit_flag": p["data"].get("exit_flag"), "value": num((p["entry"] or 0) * (p["qty_open"] or p["qty"] or 0))})
        return out

    def _strategy_table(self, module: str) -> List[dict]:
        table = strat.get(module)
        enabled = {s.id for s in (self.strats_i if module == "intraday" else self.strats_s)}
        validated = self._validated(module)
        perf = (self.store.get("perf", {}) or {}).get(module, {})
        live: Dict[str, dict] = {}
        for p in self.store.positions(("CLOSED",), module=module):
            a = live.setdefault(p["strategy"], {"trades": 0, "wins": 0, "net": 0.0})
            net = (p["gross"] or 0) - (p["charges"] or 0)
            a["trades"] += 1
            a["wins"] += 1 if net > 0 else 0
            a["net"] += net
        out = []
        for s in table.values():
            row = s.to_dict()
            row.update(enabled=s.id in enabled, validated=s.id in validated, backtest=perf.get(s.id), paper=live.get(s.id))
            out.append(row)
        return out

    def state(self) -> dict:
        now = self.clock()
        today, day = now.date(), now.date().isoformat()
        session = last_session(now)
        is_open = market_open(now)
        with self.lock:
            quotes, snap_i, snap_s, htf, nifty = dict(self.quotes), dict(self.snap_i), dict(self.snap_s), dict(self.htf), dict(self.nifty)
        stocks = self.data._universe or []
        book = self.trader.book(now)
        validated_i, validated_s = self._validated("intraday"), self._validated("swing")
        perf = self.store.get("perf", {}) or {}
        with self.lock:
            cached_i, cached_s, weak_latest = self.sig_i, self.sig_s, dict(self.weak)
        sig_i = [dict(s) for s in (cached_i if cached_i is not None else self.store.signals("intraday", day=session.isoformat()))]
        valid_s = self.cfg.intraday.signal_valid_minutes * 60
        for s in sig_i:
            s["fresh"] = is_open and s["status"] in ("ACTIVE", "TAKEN", "PROPOSED", "BLOCKED") and \
                0 <= (now - pd.Timestamp(s["time"])).total_seconds() <= valid_s
        opp_i = opportunities(sig_i, validated_i, perf.get("intraday"), window_minutes=30)
        fresh_by_symbol = {o["symbol"]: o for o in opp_i if o.get("fresh")}
        latest_by_symbol: Dict[str, dict] = {}
        for o in sorted(opp_i, key=lambda x: x["time"]):
            latest_by_symbol[o["symbol"]] = o

        scanner = []
        for st in stocks:
            q, sn = quotes.get(st.symbol) or {}, snap_i.get(st.symbol) or {}
            ltp = num(q.get("ltp"))
            prev = q.get("prev_close")
            fresh, last_sig = fresh_by_symbol.get(st.symbol), latest_by_symbol.get(st.symbol)
            bid, ask = num(q.get("bid")), num(q.get("ask"))
            tb, ts = q.get("total_buy_qty"), q.get("total_sell_qty")
            scanner.append({
                "symbol": st.symbol, "name": st.name, "industry": st.industry, "ltp": ltp,
                "day_pct": num((q["ltp"] / prev - 1) * 100) if ltp and prev and np.isfinite(prev) and prev > 0 else sn.get("day_pct"),
                "volume": num(q.get("volume"), 0), "vwap": num(q.get("vwap")) or sn.get("vwap"), "bid": bid, "ask": ask,
                "spread_pct": num((ask - bid) / ltp * 100, 3) if bid and ask and ltp else None,
                "book_ratio": num(tb / ts) if tb and ts and np.isfinite(tb) and np.isfinite(ts) and ts > 0 else None,
                **{k: sn.get(k) for k in ("rsi", "adx", "rvol", "rvol_tod", "h15", "h60", "st_dir", "ema_stack", "rs_day",
                                          "strength_buy", "strength_sell", "atr", "above_vwap")},
                "signal": fresh["side"] if fresh else "NO TRADE",
                "reason": (fresh["reason"] if fresh else no_trade_reason(sn, self.cfg.intraday, weak_latest.get(st.symbol)
                           if weak_latest.get(st.symbol, {}).get("time", "") >= (sn.get("bar") or "") else None)),
                "last_signal": {k: last_sig.get(k) for k in ("side", "strategy_name", "time", "score", "status")} if last_sig else None,
                "error": self.errors.get(st.symbol)})

        sig_s = [dict(s) for s in (cached_s if cached_s is not None else self.store.signals("swing", since=(session - timedelta(days=10)).isoformat()))]
        last_swing_day = max((s["day"] for s in sig_s), default=None)
        for s in sig_s:
            s["fresh"] = s["day"] == last_swing_day and s["status"] in ("ACTIVE", "PENDING", "TAKEN", "PROPOSED", "BLOCKED", "ALERT")
            s["htf"] = htf.get(s["symbol"])
            s["ltp"] = num((quotes.get(s["symbol"]) or {}).get("ltp"))
        opp_s = opportunities([s for s in sig_s if s["day"] == last_swing_day], validated_s, perf.get("swing"))
        for o in opp_s:
            o["setup"] = " + ".join(x["name"] for x in o["strategies"][:3])
            bt = (perf.get("swing") or {}).get(o["strategy"]) or {}
            o["history"] = {k: bt.get(k) for k in ("oos_trades", "oos_win", "oos_pf")} if bt else None

        held = self.store.positions(HELD)
        closed_today = self.store.positions(("CLOSED",), since_day=day)
        closed_recent = self.store.positions(("CLOSED", "CANCELLED"), since_day=(today - timedelta(days=14)).isoformat())
        pos_view = self._positions_view(held, today)

        def pnl(module):
            fills = [f for f in self.store.fills(day=day) if (self.store.position(f["position_id"]) or {}).get("module") == module]
            realized = sum((f["pnl"] or 0) - (f["charges"] or 0) for f in fills)
            unreal = sum(p["unrealized"] or 0 for p in pos_view if p["module"] == module)
            return {"realized": round(realized, 2), "unrealized": round(unreal, 2), "total": round(realized + unreal, 2)}

        sectors: Dict[str, dict] = {}
        for st in stocks:
            sn, q = snap_s.get(st.symbol) or {}, quotes.get(st.symbol) or {}
            a = sectors.setdefault(st.industry or "Other", {"sector": st.industry or "Other", "n": 0, "day": [], "roc5": [], "roc20": [],
                                                            "rs": sn.get("sector_rs"), "rank": sn.get("sector_rank"), "members": []})
            a["n"] += 1
            a["members"].append(st.symbol)
            prev = q.get("prev_close")
            if q.get("ltp") and prev and np.isfinite(prev) and np.isfinite(q["ltp"]) and prev > 0:
                a["day"].append((q["ltp"] / prev - 1) * 100)
            for k in ("roc5", "roc20"):
                if sn.get(k) is not None:
                    a[k].append(sn[k])
            if a["rs"] is None:
                a["rs"], a["rank"] = sn.get("sector_rs"), sn.get("sector_rank")
        sector_rows = [{"sector": a["sector"], "stocks": a["n"], "members": a["members"], "rs20": a["rs"], "rank": a["rank"],
                        **{k: (round(sum(a[k]) / len(a[k]), 2) if a[k] else None) for k in ("day", "roc5", "roc20")}}
                       for a in sectors.values()]
        sector_rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0))

        adv = sum(1 for r in scanner if (r["day_pct"] or 0) > 0)
        dec = sum(1 for r in scanner if (r["day_pct"] or 0) < 0)
        above = sum(1 for r in scanner if r.get("above_vwap"))
        votes = (nifty.get("h15") or 0) + (nifty.get("h60") or 0) + (1 if adv > 1.5 * max(dec, 1) else -1 if dec > 1.5 * max(adv, 1) else 0)
        nq = quotes.get(INDEX.symbol) or {}
        market = {**nifty, "ltp": num(nq.get("ltp")) or nifty.get("close"),
                  "day_pct": num((nq["ltp"] / nq["prev_close"] - 1) * 100) if nq.get("ltp") and nq.get("prev_close") and np.isfinite(nq["prev_close"]) else nifty.get("day_pct"),
                  "advancing": adv, "declining": dec, "above_vwap": above, "stocks": len(scanner),
                  "trend": "BULLISH" if votes >= 2 else "BEARISH" if votes <= -2 else "NEUTRAL"}

        def screen(rows, key, n=10, reverse=True, cond=None):
            rows = [r for r in rows if r.get(key) is not None and (cond is None or cond(r))]
            return sorted(rows, key=lambda r: r[key], reverse=reverse)[:n]
        swing_rows = [{"symbol": s, **v, "ltp": num((quotes.get(s) or {}).get("ltp")), "htf": htf.get(s)} for s, v in snap_s.items()]
        breakout = [o for o in opp_i if o["kind"] == "breakout"][:12] + [o for o in opp_s if o["kind"] == "breakout"][:12]
        scanners = {
            "intraday_buy": [o for o in opp_i if o["side"] == "BUY"][:15], "intraday_sell": [o for o in opp_i if o["side"] == "SELL"][:15],
            "swing_buy": [o for o in opp_s if o["side"] == "BUY"][:15], "swing_sell": [o for o in opp_s if o["side"] == "SELL"][:15],
            "breakout": breakout, "high_volume": screen(scanner, "rvol_tod", cond=lambda r: r["rvol_tod"] >= 1.2),
            "momentum": screen(scanner, "strength_buy", 6) + screen(scanner, "strength_sell", 6),
            "strong_trend": screen(scanner, "adx", cond=lambda r: r["adx"] >= 25 and r.get("ema_stack") in (1, -1)),
            "swing_momentum": screen(swing_rows, "roc20"), "swing_trend": screen(swing_rows, "adx", cond=lambda r: r["adx"] >= 25 and r.get("trend") != 0)}

        armed, missing = self.trader.gate(today)
        bt = {}
        for module in ("intraday", "swing"):
            row = self.store.latest_backtest(module)
            if row:
                bt[module] = {"created": row["created"], "params": row["params"], **{k: row["result"].get(k) for k in
                              ("summary", "strategies", "walk_forward", "notes", "by_stock", "validated")}}
        return {
            "status": self.status, "status_text": self.status_text, "time": now.isoformat(timespec="seconds"),
            "market_open": is_open, "session": session.isoformat(), "mode": self.trader.mode,
            "data_through": self.data_through.isoformat() if self.data_through else None,
            "last_scan": self.last_scan.isoformat(timespec="seconds") if self.last_scan is not None and self.scans else None,
            "last_quote": self.last_quote.isoformat(timespec="seconds") if self.last_quote is not None else None,
            "scan_seconds": self.scan_seconds, "scans": self.scans, "cycles": self.cycles, "errors": self.errors,
            "data": self.data.status(), "live": {"armed": armed, "missing": missing, "mismatches": self.trader.mismatches,
                                                 "confirm_each_order": self.cfg.live_confirm_each_order,
                                                 "auto_exits": self.cfg.live_auto_exits},
            "require_validation": self.cfg.require_validation, "paper_unvalidated": self.cfg.paper_unvalidated,
            "risk": self.risk.to_dict(book), "market": market,
            "intraday": {"enabled": self.cfg.intraday.enabled, "scanner": scanner, "buy": scanners["intraday_buy"],
                         "sell": scanners["intraday_sell"], "signals_today": len(sig_i),
                         "counts": {k: sum(1 for s in sig_i if s["status"] == k) for k in sorted({s["status"] for s in sig_i})},
                         "positions": [p for p in pos_view if p["module"] == "intraday"],
                         "closed": self._positions_view([p for p in closed_today if p["module"] == "intraday"], today),
                         "pnl": pnl("intraday"), "strategies": self._strategy_table("intraday"), "method": INTRADAY_METHOD,
                         "window": [self.cfg.intraday.entry_start, self.cfg.intraday.entry_end, self.cfg.intraday.square_off]},
            "swing": {"enabled": self.cfg.swing.enabled, "buy": scanners["swing_buy"], "sell": scanners["swing_sell"],
                      "signal_date": last_swing_day, "watchlist": sorted(swing_rows, key=lambda r: r.get("confidence_buy") or 0, reverse=True)[:20],
                      "recent": [s for s in sig_s if s["day"] != last_swing_day][-40:],
                      "positions": [p for p in pos_view if p["module"] == "swing"],
                      "closed": self._positions_view([p for p in closed_recent if p["module"] == "swing"], today),
                      "pnl": pnl("swing"), "sectors": sector_rows, "strategies": self._strategy_table("swing"), "method": SWING_METHOD,
                      "hold": [self.cfg.swing.min_hold_days, self.cfg.swing.max_hold_days]},
            "scanners": scanners,
            "orders": {"pending": self.store.orders(("PROPOSED",)), "today": self.store.orders(day=day)[-60:]},
            "alerts": self.store.alerts(day=day, limit=80),
            "backtest": {"job": self.job, **bt},
        }

    # ------------------------------------------------------------------ chart
    def chart(self, symbol: str, tf: str = "5m", bars: int = 240) -> Optional[dict]:
        symbol = symbol.strip().upper()
        with self.lock:
            f5, m1, fd, h1 = self.f5.get(symbol), self.m1.get(symbol), self.fd.get(symbol), self.h1.get(symbol)
        intraday = tf in ("1m", "5m", "15m", "30m", "1h")
        if tf == "1m":
            b = m1
        elif tf == "5m":
            b = f5
        elif tf in ("15m", "30m"):
            b = resample_bars(f5, int(tf[:-1])) if f5 is not None else None
        elif tf == "1h":
            b = h1 if h1 is not None else (resample_bars(f5, 60) if f5 is not None else None)
        elif tf == "4h":
            b = resample_bars(h1, 240) if h1 is not None else None
        elif tf == "1d":
            b = fd
        elif tf == "1w":
            b = weekly_bars(fd) if fd is not None else None
        else:
            return None
        if b is None or not len(b):
            return {"symbol": symbol, "tf": tf, "candles": [], "lines": {}, "levels": None,
                    "note": "no data for this timeframe yet"}
        b = b[["open", "high", "low", "close", "volume"]].copy()
        lines = {"EMA 9": ema(b["close"], 9), "EMA 21": ema(b["close"], 21)} if intraday else \
            {"EMA 20": ema(b["close"], 20), "EMA 50": ema(b["close"], 50)}
        if tf in ("1m", "5m", "15m"):
            lines["VWAP"] = session_vwap(b)[0]
        lines["Supertrend"] = supertrend(b, 10, 3.0)[0]
        b = b.iloc[-bars:]
        stamp = [int(t.timestamp()) for t in b.index]
        module = "intraday" if intraday else "swing"
        levels = None
        pos = [p for p in self.store.positions(HELD, module=module) if p["symbol"] == symbol]
        if pos:
            p = pos[-1]
            levels = {"side": p["side"], "entry": p["entry"], "stop": p["stop"], "t1": p["t1"], "t2": p["t2"], "source": f"{p['status'].lower()} position"}
        else:
            day = last_session(self.clock()).isoformat()
            sigs = [s for s in (self.store.signals("intraday", day=day) if intraday else self.store.signals("swing", since=(pd.Timestamp(day) - pd.Timedelta(days=10)).date().isoformat()))
                    if s["symbol"] == symbol]
            if sigs:
                s = sigs[-1]
                levels = {"side": s["side"], "entry": s["price"], "stop": s["stop"], "t1": s["t1"], "t2": s["t2"],
                          "source": f"{s['strategy_name']} {s['time'][:16].replace('T', ' ')}"}
        q = self.quotes.get(symbol) or {}
        return {"symbol": symbol, "tf": tf, "ltp": num(q.get("ltp")),
                "candles": [[stamp[i], num(r.open), num(r.high), num(r.low), num(r.close), num(r.volume, 0)] for i, r in enumerate(b.itertuples())],
                "lines": {k: [num(x) for x in v.reindex(b.index).to_numpy()] for k, v in lines.items()}, "levels": levels,
                "depth": q.get("depth"), "note": None}
