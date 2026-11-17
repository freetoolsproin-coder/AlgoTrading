"""
Execution of equity signals in three modes.

    analysis   signals only, nothing is opened
    paper      simulated fills at the live price with slippage and charges (DEFAULT)
    live       real Upstox orders. Possible only when ALL of these hold:
                 1. mode = live (config/equity.ini or the dashboard switch)
                 2. .env  LIVE_TRADING_CONFIRM=I ACCEPT REAL-MONEY RISK
                 3. armed for TODAY:  python -m src.equity --arm   (typed confirmation, expires at midnight)
                 4. the order is CONFIRMED on the dashboard ([EXECUTION] live_confirm_each_order = true)

Live order rules
    * An entry is first a PROPOSED order. Confirming it sends one LIMIT order; its fill is followed automatically by
      a protective SL-M stop order at the exchange. Confirming an entry therefore also authorises its stop order,
      the intraday square-off, a software stop if the exchange stop is missing, and the kill switch exits. Target and
      time exits are PROPOSED too unless live_auto_exits = true.
    * Every order has a unique tag. A placement is sent once; after a timeout it is looked up by tag, never resent.
    * reconcile() polls working orders, applies fills, re-places a missing stop (delivery stops are day orders) and
      compares the broker's positions with this book.
    * Kill switch (dashboard button or the KILL_SWITCH file): no new entries, proposals cancelled, working entries
      cancelled, every open position exited.

Position management is the same in every mode: stop -> exit; target 1 -> book half and move the stop to cost;
target 2 -> exit the rest; intraday square-off at [INTRADAY] square_off; swing time exit after max_hold_days sessions.
There is no averaging: one position per stock, and a stop is never widened.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, time, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.config import settings
from src.equity.config import DATA_DIR, MODES, EquityConfig, hhmm
from src.equity.data import is_trading_day
from src.equity.risk import RiskManager
from src.equity.signals import in_zone, levels, tick
from src.execution.broker_api import CANCELLED, FILLED, OPEN, PENDING, REJECTED, BrokerError, OrderRequest

logger = logging.getLogger("scanner.equity.execution")
ARM_FILE = DATA_DIR / "LIVE_ARMED"
HELD = ("PROPOSED", "PENDING", "ENTERING", "OPEN", "EXITING")          # states that occupy a position slot
WORKING = ("SENDING", "SENT", "OPEN", "PENDING", "UNKNOWN")            # orders that may still fill
PROTECTIVE = ("STOP", "STOP_AT_COST", "SQUARE_OFF", "KILL", "MANUAL")  # exits that never wait for a click
EXIT_ALERT = {"STOP": "stop", "STOP_AT_COST": "stop", "TARGET1": "target", "TARGET2": "target", "TIME_EXIT": "swing_exit",
              "TREND_EXIT": "swing_exit", "SQUARE_OFF": "entry", "KILL": "risk_limit", "MANUAL": "entry"}


def sessions_held(opened: str, today: date) -> int:
    """Trading sessions from the entry day to `today`, both included."""
    d, n = date.fromisoformat(opened[:10]), 0
    while d <= today:
        n += 1 if is_trading_day(d) else 0
        d += timedelta(days=1)
    return n


class Trader:
    def __init__(self, cfg: EquityConfig, store, risk: RiskManager, alerts, broker=None,
                 gate: Optional[Callable[[date], Tuple[bool, List[str]]]] = None):
        self.cfg, self.store, self.risk, self.alerts = cfg, store, risk, alerts
        self._broker = broker
        self._gate = gate
        self.lock = threading.RLock()
        self.ltp: Dict[str, float] = {}
        self.mismatches: List[str] = []
        self._last_recon = 0.0
        self._stop_retry: Dict[int, pd.Timestamp] = {}
        self.risk.kill_flag = bool(store.get("kill", False))
        self._kill_done = False

    # ------------------------------------------------------------------ mode / gate
    @property
    def mode(self) -> str:
        m = self.store.get("mode", self.cfg.mode)
        return m if m in MODES else "paper"

    def gate(self, today: date) -> Tuple[bool, List[str]]:
        if self._gate is not None:
            return self._gate(today)
        from src.execution import live_gate
        return live_gate.status({"mode": "live"}, today, path=ARM_FILE)

    def set_mode(self, mode: str, today: date) -> Tuple[bool, str]:
        mode = str(mode).strip().lower()
        if mode not in MODES:
            return False, "mode must be analysis, paper or live"
        if mode == "live":
            armed, missing = self.gate(today)
            if not armed:
                return False, "live trading is not armed: " + "; ".join(missing).replace("src.execution.live_gate", "src.equity")
        self.store.set("mode", mode)
        self.alerts.emit("order", f"Execution mode is now {mode.upper()}", push=False)
        return True, mode

    @property
    def broker(self):
        if self._broker is None:
            from src.equity.broker import UpstoxEquityBroker
            self._broker = UpstoxEquityBroker(requests_per_second=self.cfg.requests_per_second, algo_name="")
        return self._broker

    # ------------------------------------------------------------------ book
    def book(self, now: pd.Timestamp) -> dict:
        day = now.date().isoformat()
        fills = self.store.fills(day=day)
        realized = sum((f["pnl"] or 0.0) - (f["charges"] or 0.0) for f in fills)
        held = self.store.positions(HELD)
        opened_today = self.store.rows("SELECT COUNT(*) AS n FROM positions WHERE substr(opened,1,10)=?", (day,))[0]["n"]
        streak = 0
        for p in reversed(self.store.rows("SELECT gross, charges FROM positions WHERE status='CLOSED' AND substr(closed,1,10)=? ORDER BY closed", (day,))):
            if (p["gross"] or 0) - (p["charges"] or 0) < 0:
                streak += 1
            else:
                break
        return {"open": held, "realized_today": realized, "trades_today": opened_today, "loss_streak": streak,
                "deployed": sum((p["entry"] or 0) * (p["qty_open"] or p["qty"] or 0) for p in held)}

    def free_cash(self, book: dict) -> float:
        return self.risk.cfg.capital - book["deployed"]

    # ------------------------------------------------------------------ entries
    def take(self, sig: dict, ltp: Optional[float], now: pd.Timestamp, validated: bool = True) -> Tuple[str, str]:
        """Act on one fresh signal. Returns (signal status, note)."""
        with self.lock:
            return self._take(sig, ltp, now, validated)

    def _take(self, sig, ltp, now, validated):
        mode, module, side = self.mode, sig["module"], sig["side"]
        mcfg = self.cfg.swing if module == "swing" else self.cfg.intraday
        if mode == "analysis":
            return "ANALYSIS", "analysis mode: signal only"
        if not mcfg.enabled:
            return "BLOCKED", f"{module} module is switched off"
        if module == "swing" and side == "SELL":
            return "ALERT", "exit / avoid alert (the cash market has no carry-forward short)"
        if module == "intraday" and side == "SELL" and not self.cfg.intraday.allow_short:
            return "BLOCKED", "intraday short selling is switched off"
        if not validated and ((mode == "live" and self.cfg.require_validation) or (mode == "paper" and not self.cfg.paper_unvalidated)):
            return "BLOCKED", "strategy has not passed in-sample + out-of-sample validation (see the backtest)"
        if self.store.one("SELECT id FROM positions WHERE signal_id=?", (sig["id"],)):
            return "TAKEN", "already acted on"
        if module == "intraday":
            if ltp is None or not ltp > 0:
                return "BLOCKED", "no live price"
            if not in_zone(ltp, sig["zone"]):
                return "BLOCKED", f"price {ltp:.2f} is outside the entry zone {sig['zone'][0]:.2f}-{sig['zone'][1]:.2f}"
            entry = ltp
        else:
            entry = sig["price"]
        lv = levels(side, entry, sig["stop"], mcfg)
        book = self.book(now)
        sizing = self.risk.size(module, side, entry, lv["stop"], lv["t2"], sig.get("atr") or 0.0, self.free_cash(book))
        ok, why = self.risk.check_entry(module, sig["symbol"], sizing, book)
        if not ok:
            return "BLOCKED", "; ".join(why)
        key = sig.get("key") or ""
        data = {"signal": {k: sig.get(k) for k in ("id", "strategy_name", "reason", "score", "zone", "atr", "time", "name",
                                                   "hold_text", "exits", "price", "kind")},
                "sizing": sizing.to_dict(), "t1_done": False, "initial_stop": lv["stop"]}
        base = dict(signal_id=sig["id"], module=module, mode=mode, symbol=sig["symbol"], key=key, side=side,
                    strategy=sig["strategy"], qty=sizing.qty, qty_open=0, entry=lv["entry"], stop=lv["stop"], t1=lv["t1"],
                    t2=lv["t2"], created=now.isoformat(timespec="seconds"))
        if module == "swing":
            data["valid_until"] = self._valid_until(now, self.cfg.swing.entry_valid_sessions).isoformat()
        if mode == "paper":
            if module == "swing":
                pid = self.store.insert("positions", status="PENDING", data=data, **base)
                return "PENDING", f"waiting for the entry zone {sig['zone'][0]:.2f}-{sig['zone'][1]:.2f} (position {pid})"
            pid = self.store.insert("positions", status="PENDING", data=data, **base)
            self._fill_entry(self.store.position(pid), entry, sizing.qty, now)
            return "TAKEN", f"paper {side} {sizing.qty} @ {entry:.2f}"
        # live: a proposal that waits for the user's click
        data["valid_until"] = data.get("valid_until") or (now + pd.Timedelta(minutes=self.cfg.intraday.signal_valid_minutes)).isoformat()
        pid = self.store.insert("positions", status="PROPOSED", data=data, **base)
        oid = self._order(pid, "ENTRY", sig["symbol"], key, side, sizing.qty, "LIMIT", self._limit(side, entry), 0.0,
                          "D" if module == "swing" else "I", "PROPOSED", now, {"reason": "ENTRY", "zone": sig["zone"]})
        self.alerts.emit("order", f"CONFIRM LIVE ORDER #{oid}: {side} {sizing.qty} {sig['symbol']} @ {entry:.2f} · SL {lv['stop']:.2f} "
                                  f"· T1 {lv['t1']:.2f} · T2 {lv['t2']:.2f}\n{sig['strategy_name']}", sig["symbol"], module)
        if not self.cfg.live_confirm_each_order:
            ok, msg = self.confirm(oid, now)
            return ("TAKEN", f"live order sent ({msg})") if ok else ("BLOCKED", msg)
        return "PROPOSED", f"live order #{oid} waits for confirmation on the dashboard"

    def _valid_until(self, now: pd.Timestamp, sessions: int) -> pd.Timestamp:
        """15:30 of the `sessions`-th session that can still be traded (today counts while the market is open)."""
        d = now.date()
        if not (is_trading_day(d) and now.time() < time(15, 30)):
            d += timedelta(days=1)
        left = sessions
        while True:
            if is_trading_day(d):
                left -= 1
                if left <= 0:
                    break
            d += timedelta(days=1)
        return pd.Timestamp(d.isoformat() + " 15:30", tz=now.tz)

    def _limit(self, side: str, price: float, room: float = 0.001) -> float:
        """Marketable limit: a little through the price so it fills, with a hard cap on the slippage accepted."""
        return tick(price * (1 + room) if side == "BUY" else price * (1 - room))

    def _fill_entry(self, pos: dict, price: float, qty: int, now: pd.Timestamp, slip: bool = True):
        s = 1.0 if pos["side"] == "BUY" else -1.0
        mcfg = self.cfg.swing if pos["module"] == "swing" else self.cfg.intraday
        fill = tick(price + s * self.risk.slip(price)) if slip else float(price)
        lv = levels(pos["side"], fill, pos["stop"], mcfg)
        charges = self.risk.costs.charge(pos["module"], pos["side"], fill, qty)
        self.store.insert("fills", position_id=pos["id"], time=now.isoformat(timespec="seconds"), day=now.date().isoformat(),
                          side=pos["side"], qty=qty, price=fill, charges=charges, pnl=0.0, reason="ENTRY")
        self.store.update("positions", pos["id"], status="OPEN", qty=qty, qty_open=qty, entry=fill, t1=lv["t1"], t2=lv["t2"],
                          opened=now.isoformat(timespec="seconds"), charges=charges)
        self.store.signal_status(pos["signal_id"], "TAKEN", f"{pos['mode']} {pos['side']} {qty} @ {fill:.2f}")
        self.alerts.emit("entry", f"{pos['mode'].upper()} ENTRY {pos['side']} {qty} {pos['symbol']} @ {fill:.2f}\n"
                                  f"SL {pos['stop']:.2f} · T1 {lv['t1']:.2f} · T2 {lv['t2']:.2f} · {pos['data'].get('signal', {}).get('strategy_name', '')}",
                         pos["symbol"], pos["module"])

    # ------------------------------------------------------------------ price updates
    def on_quotes(self, prices: Dict[str, float], now: pd.Timestamp, market_is_open: bool, ctx: Optional[Dict[str, dict]] = None):
        """Manage pending entries and open positions with the latest prices. `ctx[symbol]["h1"]` = 1h trend (swing entries)."""
        with self.lock:
            self.ltp.update({k: v for k, v in prices.items() if v is not None and v > 0})
            if self.risk.killed() and not self._kill_done:
                self._kill_actions(now)
            if not self.risk.killed():
                self._kill_done = False
            for pos in self.store.positions(("PENDING", "PROPOSED")):
                self._pending(pos, now, market_is_open, ctx or {})
            if market_is_open:
                for pos in self.store.positions(("OPEN",)):
                    ltp = self.ltp.get(pos["symbol"])
                    if ltp:
                        self._manage(pos, ltp, now)
            book = self.book(now)
            for reason in self.risk.limits_breached(book):
                self.alerts.emit("risk_limit", f"Risk limit: {reason}. No new entries.", key=f"risk|{now.date()}|{reason}")

    def _pending(self, pos, now, market_is_open, ctx):
        until = pos["data"].get("valid_until")
        if until and now > pd.Timestamp(until):
            self._cancel_position(pos, "entry zone not reached in time" if pos["status"] == "PENDING" else "proposal expired", now)
            return
        if pos["status"] != "PENDING" or not market_is_open or now.time() < time(9, 16):
            return
        ltp = self.ltp.get(pos["symbol"])
        zone = pos["data"].get("signal", {}).get("zone")
        if not ltp or not zone:
            return
        s = 1.0 if pos["side"] == "BUY" else -1.0
        if s * (ltp - pos["stop"]) <= 0:
            self._cancel_position(pos, f"price {ltp:.2f} reached the stop before entry (gap)", now)
            return
        if not in_zone(ltp, zone):
            return
        if (ctx.get(pos["symbol"]) or {}).get("h1") == -s:
            return                                       # the 1h trend is against the trade: wait
        book = self.book(now)
        book["open"] = [p for p in book["open"] if p["id"] != pos["id"]]
        mcfg = self.cfg.swing if pos["module"] == "swing" else self.cfg.intraday
        lv = levels(pos["side"], ltp, pos["stop"], mcfg)
        cash = self.free_cash(book) + (pos["entry"] or 0) * (pos["qty"] or 0)
        sizing = self.risk.size(pos["module"], pos["side"], ltp, pos["stop"], lv["t2"], pos["data"].get("signal", {}).get("atr") or 0.0, cash)
        ok, why = self.risk.check_entry(pos["module"], pos["symbol"], sizing, book)
        if not ok:
            self._cancel_position(pos, "; ".join(why), now)
            return
        self.alerts.emit("entry", f"Entry price reached: {pos['symbol']} {ltp:.2f} is inside the entry zone {zone[0]:.2f}-{zone[1]:.2f}",
                         pos["symbol"], pos["module"], push=False)
        self._fill_entry(pos, ltp, sizing.qty, now)

    def _cancel_position(self, pos, why: str, now):
        self.store.update("positions", pos["id"], status="CANCELLED", exit_reason=why, closed=now.isoformat(timespec="seconds"))
        for o in self.store.orders(("PROPOSED",), position_id=pos["id"]):
            self.store.update("orders", o["id"], status="CANCELLED", message=why, updated=now.isoformat(timespec="seconds"))
        self.store.signal_status(pos["signal_id"], "EXPIRED", why)

    def _manage(self, pos, ltp, now):
        s = 1.0 if pos["side"] == "BUY" else -1.0
        d = pos["data"]
        if pos["mode"] == "live" and self.store.orders(WORKING + ("PROPOSED",), position_id=pos["id"]):
            working = self.store.orders(WORKING + ("PROPOSED",), position_id=pos["id"])
            if any(o["purpose"] == "EXIT" for o in working):
                if not (pos["module"] == "intraday" and now.time() >= hhmm(self.cfg.intraday.square_off)):
                    return                                # an exit is already proposed / working
        if s * (ltp - pos["stop"]) <= 0:
            return self._exit(pos, pos["qty_open"], ltp, "STOP_AT_COST" if d.get("t1_done") else "STOP", now)
        if pos["module"] == "intraday" and now.time() >= hhmm(self.cfg.intraday.square_off):
            return self._exit(pos, pos["qty_open"], ltp, "SQUARE_OFF", now)      # before the targets: it never waits
        if s * (ltp - pos["t2"]) >= 0:
            return self._exit(pos, pos["qty_open"], ltp, "TARGET2", now)
        if not d.get("t1_done") and s * (ltp - pos["t1"]) >= 0:
            half = pos["qty_open"] // 2
            if half >= 1:
                return self._exit(pos, half, ltp, "TARGET1", now, then_stop=pos["entry"])
            d["t1_done"] = True                           # a single share cannot be split: only trail the stop
            self.store.update("positions", pos["id"], stop=pos["entry"], data=d)
            self._replace_stop(pos, pos["entry"], now)
            return None
        if pos["module"] == "swing":
            if d.get("exit_flag"):
                return self._exit(pos, pos["qty_open"], ltp, d["exit_flag"], now)
            if sessions_held(pos["opened"], now.date()) >= self.cfg.swing.max_hold_days and now.time() >= time(15, 15):
                return self._exit(pos, pos["qty_open"], ltp, "TIME_EXIT", now)
        return None

    def flag_exit(self, symbol: str, reason: str, text: str, now: pd.Timestamp):
        """Swing: mark an open position for exit at the next quote of an open market (trend exit after the close)."""
        with self.lock:
            for pos in self.store.positions(("OPEN",), module="swing"):
                if pos["symbol"] == symbol and not pos["data"].get("exit_flag") \
                        and sessions_held(pos["opened"], now.date()) >= self.cfg.swing.min_hold_days:
                    d = pos["data"]
                    d["exit_flag"] = reason
                    self.store.update("positions", pos["id"], data=d)
                    self.alerts.emit("swing_exit", f"SWING EXIT {symbol}: {text}. Exit at the next open.", symbol, "swing")

    # ------------------------------------------------------------------ exits
    def _exit(self, pos, qty: int, price: float, reason: str, now, then_stop: Optional[float] = None):
        if qty < 1:
            return
        if pos["mode"] == "live":
            return self._live_exit(pos, qty, price, reason, now, then_stop)
        s = 1.0 if pos["side"] == "BUY" else -1.0
        self._apply_exit(pos, qty, tick(price - s * self.risk.slip(price)), reason, now, then_stop)

    def _apply_exit(self, pos, qty: int, fill: float, reason: str, now, then_stop: Optional[float] = None):
        s = 1.0 if pos["side"] == "BUY" else -1.0
        side = "SELL" if pos["side"] == "BUY" else "BUY"
        charges = self.risk.costs.charge(pos["module"], side, fill, qty)
        pnl = (fill - pos["entry"]) * s * qty
        self.store.insert("fills", position_id=pos["id"], time=now.isoformat(timespec="seconds"), day=now.date().isoformat(),
                          side=side, qty=qty, price=fill, charges=charges, pnl=pnl, reason=reason)
        left = pos["qty_open"] - qty
        d = pos["data"]
        upd = {"qty_open": left, "gross": (pos["gross"] or 0) + pnl, "charges": (pos["charges"] or 0) + charges}
        if reason == "TARGET1":
            d["t1_done"] = True
        if then_stop is not None and left > 0:
            upd["stop"] = tick(then_stop)
        upd["data"] = d
        if left <= 0:
            upd.update(status="CLOSED", exit_price=fill, exit_reason=reason, closed=now.isoformat(timespec="seconds"))
        else:
            upd["status"] = "OPEN"
        self.store.update("positions", pos["id"], **upd)
        net = upd["gross"] - upd["charges"]
        self.alerts.emit(EXIT_ALERT.get(reason, "entry"),
                         f"{pos['mode'].upper()} {reason.replace('_', ' ')} {pos['symbol']}: {side} {qty} @ {fill:.2f} "
                         f"(entry {pos['entry']:.2f}), P&L {pnl - charges:+,.0f}" + (f" · position closed, net {net:+,.0f}" if left <= 0 else f" · {left} left, stop {upd.get('stop', pos['stop']):.2f}"),
                         pos["symbol"], pos["module"])

    def close(self, position_id: int, now: pd.Timestamp, reason: str = "MANUAL") -> Tuple[bool, str]:
        with self.lock:
            pos = self.store.position(position_id)
            if not pos:
                return False, "position not found"
            if pos["status"] in ("PENDING", "PROPOSED"):
                self._cancel_position(pos, "cancelled by the user", now)
                return True, "cancelled"
            if pos["status"] != "OPEN":
                return False, f"position is {pos['status']}"
            ltp = self.ltp.get(pos["symbol"])
            if not ltp:
                return False, "no live price for this stock yet"
            self._exit(pos, pos["qty_open"], ltp, reason, now)
            return True, "exit sent" if pos["mode"] == "live" else "closed"

    # ------------------------------------------------------------------ kill switch
    def kill(self, on: bool, now: pd.Timestamp) -> None:
        with self.lock:
            self.risk.kill_flag = bool(on)
            self.store.set("kill", bool(on))
            self.alerts.emit("risk_limit", "KILL SWITCH ON: no new entries, open positions are being closed" if on
                             else "Kill switch released (the KILL_SWITCH file, if present, still blocks)", push=on)
            if on:
                self._kill_actions(now)

    def _kill_actions(self, now):
        self._kill_done = True
        for pos in self.store.positions(("PENDING", "PROPOSED")):
            self._cancel_position(pos, "kill switch", now)
        for o in self.store.orders(WORKING):
            if o["mode"] == "live" and o["purpose"] == "ENTRY" and o["broker_id"]:
                self._cancel_at_broker(o, now, "kill switch")
        for pos in self.store.positions(("OPEN",)):
            ltp = self.ltp.get(pos["symbol"])
            if ltp:
                self._exit(pos, pos["qty_open"], ltp, "KILL", now)
            else:
                self._kill_done = False          # try again when a price arrives

    # ------------------------------------------------------------------ live orders
    def _order(self, pid, purpose, symbol, key, side, qty, order_type, price, trigger, product, status, now, data=None) -> int:
        seq = len(self.store.orders(position_id=pid)) + 1
        stamp = now.isoformat(timespec="seconds")
        return self.store.insert("orders", position_id=pid, tag=f"EQ{pid}{purpose[0]}{seq}", mode="live", purpose=purpose,
                                 symbol=symbol, key=key, side=side, qty=int(qty), order_type=order_type, price=float(price),
                                 trigger=float(trigger), product=product, status=status, created=stamp, updated=stamp, data=data or {})

    def confirm(self, order_id: int, now: pd.Timestamp) -> Tuple[bool, str]:
        """The user's click: send one PROPOSED order."""
        with self.lock:
            o = self.store.order(order_id)
            if not o or o["status"] != "PROPOSED":
                return False, "order is not waiting for confirmation"
            if self.mode != "live":
                return False, "execution mode is not live"
            armed, missing = self.gate(now.date())
            if not armed:
                return False, "live trading is not armed: " + "; ".join(missing)
            pos = self.store.position(o["position_id"])
            ltp = self.ltp.get(o["symbol"])
            if o["purpose"] == "ENTRY":
                if self.risk.killed():
                    return False, "kill switch is ON"
                zone = o["data"].get("zone")
                if not ltp or (zone and not in_zone(ltp, zone)):
                    return False, f"price {ltp if ltp else 'n/a'} is outside the entry zone - not sent"
                book = self.book(now)
                book["open"] = [p for p in book["open"] if p["id"] != pos["id"]]
                sizing = self.risk.size(pos["module"], pos["side"], ltp, pos["stop"], pos["t2"], 0.0, self.free_cash(book) + pos["entry"] * pos["qty"])
                sizing.qty = min(sizing.qty, o["qty"])
                ok, why = self.risk.check_entry(pos["module"], pos["symbol"], sizing, book)
                if not ok:
                    return False, "; ".join(why)
                self.store.update("orders", o["id"], price=self._limit(o["side"], ltp), qty=sizing.qty)
                self.store.update("positions", pos["id"], qty=sizing.qty)
                o = self.store.order(order_id)
            elif ltp:
                self.store.update("orders", o["id"], price=self._limit(o["side"], ltp, 0.003))
                o = self.store.order(order_id)
                self._cancel_stop(pos, now)
            ok, msg = self._send(o, now)
            if ok and pos["status"] == "PROPOSED":
                self.store.update("positions", pos["id"], status="ENTERING")
            elif ok and o["purpose"] == "EXIT":
                self.store.update("positions", pos["id"], status="EXITING")
            return ok, msg

    def reject(self, order_id: int, now: pd.Timestamp) -> Tuple[bool, str]:
        with self.lock:
            o = self.store.order(order_id)
            if not o or o["status"] != "PROPOSED":
                return False, "order is not waiting for confirmation"
            self.store.update("orders", o["id"], status="CANCELLED", message="rejected by the user", updated=now.isoformat(timespec="seconds"))
            pos = self.store.position(o["position_id"])
            if pos and pos["status"] == "PROPOSED":
                self._cancel_position(pos, "rejected by the user", now)
            return True, "rejected"

    def _send(self, o: dict, now) -> Tuple[bool, str]:
        """Send ONE order exactly once. A timeout leaves it UNKNOWN; reconcile() then looks it up by tag."""
        if o["status"] not in ("PROPOSED", "READY"):
            return False, f"order {o['id']} is {o['status']}: not sent again"
        stamp = now.isoformat(timespec="seconds")
        self.store.update("orders", o["id"], status="SENDING", updated=stamp)
        req = OrderRequest(instrument_key=o["key"], side=o["side"], quantity=o["qty"], order_type=o["order_type"],
                           price=o["price"] or 0.0, trigger_price=o["trigger"] or 0.0, product=o["product"], tag=o["tag"])
        req.closing = o["purpose"] != "ENTRY"
        try:
            ids = self.broker.place_order(req)
        except BrokerError as error:
            status = "UNKNOWN" if error.maybe_sent else "REJECTED"
            self.store.update("orders", o["id"], status=status, message=str(error)[:200], updated=stamp)
            self.alerts.emit("order", f"LIVE ORDER #{o['id']} {o['purpose']} {o['symbol']}: {status} - {str(error)[:150]}", o["symbol"])
            if status == "REJECTED":
                self._order_dead(self.store.order(o["id"]), now)
            return False, str(error)[:200]
        except Exception as error:      # noqa: BLE001 - e.g. token expired: nothing was sent
            self.store.update("orders", o["id"], status="REJECTED", message=f"{type(error).__name__}: {str(error)[:150]}", updated=stamp)
            self._order_dead(self.store.order(o["id"]), now)
            return False, f"{type(error).__name__}: {str(error)[:150]}"
        self.store.update("orders", o["id"], status="SENT", broker_id=ids[0], updated=stamp)
        self.alerts.emit("order", f"LIVE ORDER SENT #{o['id']}: {o['purpose']} {o['side']} {o['qty']} {o['symbol']} "
                                  f"{o['order_type']} {o['price'] or o['trigger']:.2f}", o["symbol"], push=False)
        return True, f"order {ids[0]} sent"

    def _order_dead(self, o: dict, now):
        """An order that will never fill: put its position back into a consistent state."""
        pos = self.store.position(o["position_id"])
        if not pos:
            return
        if o["purpose"] == "ENTRY" and pos["status"] in ("PROPOSED", "ENTERING") and not (o["filled_qty"] or 0):
            self._cancel_position(pos, f"entry order {o['status'].lower()}: {o['message']}", now)
        elif o["purpose"] == "EXIT" and pos["status"] == "EXITING":
            self.store.update("positions", pos["id"], status="OPEN")

    def _place_stop(self, pos, now):
        if any(o["purpose"] == "STOP" for o in self.store.orders(WORKING, position_id=pos["id"])):
            return                                        # duplicate protection
        side = "SELL" if pos["side"] == "BUY" else "BUY"
        oid = self._order(pos["id"], "STOP", pos["symbol"], pos["key"], side, pos["qty_open"], "SL-M", 0.0, pos["stop"],
                          "D" if pos["module"] == "swing" else "I", "READY", now, {"reason": "STOP"})
        self._send(self.store.order(oid), now)

    def _cancel_at_broker(self, o, now, why: str):
        try:
            self.broker.cancel_order(o["broker_id"])
            self.store.update("orders", o["id"], message=f"cancel requested: {why}", updated=now.isoformat(timespec="seconds"))
        except Exception as error:      # noqa: BLE001 - reconcile will show the real state
            logger.warning("cancel of order %s failed: %s", o["id"], type(error).__name__)

    def _cancel_stop(self, pos, now):
        for o in self.store.orders(WORKING, position_id=pos["id"]):
            if o["purpose"] == "STOP" and o["broker_id"]:
                self._cancel_at_broker(o, now, "replaced by an exit")
                self.store.update("orders", o["id"], status="CANCELLED", updated=now.isoformat(timespec="seconds"))

    def _replace_stop(self, pos, new_stop: float, now):
        if pos["mode"] != "live":
            return
        self._cancel_stop(pos, now)
        self._place_stop(self.store.position(pos["id"]), now)

    def _live_exit(self, pos, qty, price, reason, now, then_stop):
        side = "SELL" if pos["side"] == "BUY" else "BUY"
        existing = [o for o in self.store.orders(WORKING + ("PROPOSED",), position_id=pos["id"]) if o["purpose"] == "EXIT"]
        auto = reason in PROTECTIVE or self.cfg.live_auto_exits
        if existing:
            if not auto:
                return
            for o in existing:                           # a protective exit replaces a waiting proposal
                if o["status"] == "PROPOSED":
                    self.store.update("orders", o["id"], status="CANCELLED", message=f"replaced by {reason}")
                else:
                    return
        if reason in ("STOP", "STOP_AT_COST") and any(o["purpose"] == "STOP" for o in self.store.orders(WORKING, position_id=pos["id"])):
            return                                        # the exchange stop order is working: it handles this
        oid = self._order(pos["id"], "EXIT", pos["symbol"], pos["key"], side, qty, "LIMIT", self._limit(side, price, 0.003), 0.0,
                          "D" if pos["module"] == "swing" else "I", "PROPOSED", now, {"reason": reason, "then_stop": then_stop})
        if auto:
            self._cancel_stop(pos, now)
            ok, _ = self._send(self.store.order(oid), now)
            if ok:
                self.store.update("positions", pos["id"], status="EXITING")
        else:
            self.alerts.emit("order", f"CONFIRM LIVE EXIT #{oid}: {reason.replace('_', ' ')} - {side} {qty} {pos['symbol']} near {price:.2f}",
                             pos["symbol"], pos["module"])

    def reconcile(self, now: pd.Timestamp, force: bool = False) -> None:
        """Poll working live orders, apply fills, keep a stop order behind every live position, compare books."""
        with self.lock:
            working = [o for o in self.store.orders(WORKING) if o["mode"] == "live"]
            live_open = [p for p in self.store.positions(("OPEN", "ENTERING", "EXITING")) if p["mode"] == "live"]
            if not working and not live_open:
                self.mismatches = []
                return
            for o in working:
                try:
                    st = self.broker.order_status(order_id=o["broker_id"]) if o["broker_id"] else self.broker.order_status(tag=o["tag"])
                except Exception as error:      # noqa: BLE001 - try again next cycle
                    logger.warning("order status failed for %s: %s", o["id"], type(error).__name__)
                    continue
                stamp = now.isoformat(timespec="seconds")
                if st is None:
                    if o["status"] == "UNKNOWN" and (now - pd.Timestamp(o["updated"])).total_seconds() > 60:
                        self.store.update("orders", o["id"], status="REJECTED", message="never arrived at the broker", updated=stamp)
                        self._order_dead(self.store.order(o["id"]), now)
                    continue
                new_fill = (st.filled_quantity or 0) - (o["filled_qty"] or 0)
                status = st.status if st.status in (FILLED, CANCELLED, REJECTED, OPEN, PENDING) else o["status"]
                self.store.update("orders", o["id"], status=status, broker_id=st.order_id or o["broker_id"],
                                  filled_qty=st.filled_quantity, avg_price=st.average_price, message=st.message or o["message"], updated=stamp)
                pos = self.store.position(o["position_id"])
                if new_fill > 0 and pos and status in (FILLED, CANCELLED):
                    self._live_fill(pos, self.store.order(o["id"]), st.filled_quantity, st.average_price, now)
                elif status in (CANCELLED, REJECTED):
                    self._order_dead(self.store.order(o["id"]), now)
                    if o["purpose"] != "ENTRY":
                        self.alerts.emit("order", f"LIVE {o['purpose']} order #{o['id']} {o['symbol']} was {status}: {st.message}", o["symbol"])
                elif o["purpose"] == "ENTRY" and (now - pd.Timestamp(o["created"])).total_seconds() > self.cfg.intraday.signal_valid_minutes * 60 \
                        and pos and pos["module"] == "intraday" and o["broker_id"]:
                    self._cancel_at_broker(o, now, "not filled in time")
            for pos in [p for p in self.store.positions(("OPEN",)) if p["mode"] == "live"]:
                orders = self.store.orders(WORKING, position_id=pos["id"])
                last = self._stop_retry.get(pos["id"])
                if not orders and (last is None or (now - last).total_seconds() > 60):
                    self._stop_retry[pos["id"]] = now
                    self._place_stop(pos, now)
            if force or now.timestamp() - self._last_recon > 60:
                self._last_recon = now.timestamp()
                self._compare(now)

    def _live_fill(self, pos, o, qty: int, avg: float, now):
        if o["purpose"] == "ENTRY":
            self._fill_entry(pos, avg, qty, now, slip=False)
            self._place_stop(self.store.position(pos["id"]), now)
        else:
            reason = o["data"].get("reason") or o["purpose"]
            if o["purpose"] == "STOP" and pos["data"].get("t1_done"):
                reason = "STOP_AT_COST"
            self._apply_exit(pos, min(qty, pos["qty_open"]), avg, reason, now, o["data"].get("then_stop"))
            left = self.store.position(pos["id"])
            if left["status"] == "OPEN" and left["qty_open"] > 0:
                self._place_stop(left, now)

    def _compare(self, now):
        """Broker positions / holdings vs this book (live positions only)."""
        try:
            at_broker = {p.instrument_key: p.quantity for p in self.broker.positions() if p.quantity}
            if hasattr(self.broker, "holdings"):
                for p in self.broker.holdings():
                    at_broker.setdefault(p.instrument_key, p.quantity)
        except Exception as error:      # noqa: BLE001
            self.mismatches = [f"broker positions could not be read ({type(error).__name__})"]
            return
        out = []
        for pos in [p for p in self.store.positions(("OPEN", "EXITING")) if p["mode"] == "live"]:
            want = pos["qty_open"] * (1 if pos["side"] == "BUY" else -1)
            have = at_broker.get(pos["key"], 0)
            if pos["module"] == "intraday" and have != want:
                out.append(f"{pos['symbol']}: this book has {want}, the broker shows {have}")
            if pos["module"] == "swing" and have < want:
                out.append(f"{pos['symbol']}: this book holds {want}, the broker shows {have}")
        self.mismatches = out
