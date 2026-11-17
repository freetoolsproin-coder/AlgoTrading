"""
Notifications for the pre-breakout system.

Every alert is (1) logged, (2) stored in data/prebreakout.db and output/prebreakout_alerts.csv, (3) shown on
the dashboard; and, when its type is switched on in [NOTIFY] of config/scanner.ini, (4) sent to Telegram and
(5) shown as a Windows popup with a sound.

Telegram credentials: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env or environment variables only. The
token is never written to a log or a file by this module.

A channel failing never stops the scanner: every send is best-effort, rate-limited, and done by ONE
background worker thread (Telegram HTTP calls and Windows message boxes block, so they must not run on the
scanner's event loop). close() drains the queue and stops the worker on shutdown.
"""
from __future__ import annotations

import csv
import logging
import queue
import sys
import threading
import time as clock
from pathlib import Path
from typing import Optional

from src.config import instruments as inst
from src.config.settings import ALERTS_CSV, get_secret
from src.utils import numbers as num

logger = logging.getLogger("scanner.notify")
_STOP = object()

KIND_TO_SETTING = {
    "early_watch": "notify_early_watch", "high_watch": "notify_high_watch", "strong_watch": "notify_strong_watch",
    "confirmed": "notify_confirmed", "missed": "notify_missed", "entry": "notify_entry",
    "invalidated": "notify_invalidated", "exit": "notify_exit", "target": "notify_exit", "stop": "notify_exit",
    "data": "notify_data", "intrabar": "notify_high_watch", "movement": "notify_movement",
    "paper_trade": "notify_paper_trade",
}
STAGE_ICON = {"early_watch": "👀", "high_watch": "🔶", "strong_watch": "🔥", "confirmed": "✅", "missed": "⛔",
              "entry": "🎯", "invalidated": "❌", "exit": "🏁", "data": "⚠️", "intrabar": "⚡",
              "movement": "📈", "paper_trade": "💰"}
CSV_FIELDS = ("time", "index", "direction", "stage", "kind", "price", "level", "distance_atr", "score", "message",
              "option", "premium", "spread_pct", "rr", "invalidation", "targets", "avoid_reason", "freshness", "mock")


def _truthy(value, default=True):
    if value is None:
        return default
    return str(value).strip().strip("'\"").lower() in ("1", "true", "yes", "on")


def _p(value, digits=1):
    return f"{float(value):,.{digits}f}" if num.finite(value) else "-"


def _distance(value) -> str:
    if not num.finite(value):
        return "Distance to level: -"
    if float(value) < 0:
        return f"Distance to level: {abs(float(value)):.2f} ATR beyond the level"
    return f"Distance to level: {float(value):.2f} ATR before the level"


def format_alert(a: dict) -> tuple:
    """(title, body) with every field the spec asks for. Missing data is said, not invented."""
    kind = a.get("kind", "")
    icon = STAGE_ICON.get(kind, "•")
    mock = "[MOCK DATA] " if a.get("mock") else ""
    if a.get("model") == "v2":
        mock += "[v2 score - shadow] "
    if a.get("model") == "v3":
        mock += "[v3 multi-factor - shadow] "
    stage = a.get("stage", "")
    status = f" [{a['signal_status']}]" if a.get("signal_status") else ""
    title = f"{mock}{icon} {a.get('index')} {a.get('direction', '')} | {stage}{status} | {a.get('message', '')}".strip()
    if kind == "movement":
        return title, format_movement(a)
    if kind == "paper_trade":
        return title, format_paper_trade(a)
    if kind == "data":
        body = [f"Index: {a.get('index')}", f"Time: {a.get('time', '')[:19].replace('T', ' ')} IST",
                f"Problem: {a.get('message')}", f"Data freshness: {a.get('freshness') or '-'}",
                "State: NO TRADE until the feed is fresh again"]
        return title, "\n".join(body)
    opt = a.get("option") or {}
    lines = [
        f"Index: {a.get('index')}",
        f"Timestamp: {a.get('time', '')[:19].replace('T', ' ')} Asia/Kolkata",
        f"Direction: {a.get('direction')}",
        f"Stage: {stage}" + (f" - signal status {a['signal_status']}" if a.get("signal_status") else ""),
    ] + ([f"Market regime: {a['regime']}"] if a.get("regime") else []) + [
        f"Current underlying price: {_p(a.get('price'), 2)}",
        f"{'Breakout' if a.get('direction') == inst.CE else 'Breakdown'} level: {_p(a.get('level'), 2)}",
        _distance(a.get("distance_atr")),
        f"Setup score: {_p(a.get('score'), 0)}/100 (setup strength, not a probability)",
        "Supporting signals: " + (", ".join(a.get("supporting") or []) or "-"),
        "Contradicting signals: " + (", ".join(a.get("contradicting") or []) or "none"),
    ]
    if opt:
        lines += [f"Selected option contract: {opt.get('symbol')}",
                  f"Option premium and bid-ask spread: LTP {_p(opt.get('ltp'))}, bid {_p(opt.get('bid'))} / ask "
                  f"{_p(opt.get('ask'))} ({_p(opt.get('spread_pct'))}%)"
                  + (" [greeks estimated]" if opt.get("greeks_source") == "estimated" else "")]
    else:
        lines += ["Selected option contract: not selected at this stage", "Option premium and bid-ask spread: -"]
    if a.get("entry_conditions"):
        lines.append(f"Entry conditions: {a['entry_conditions']}")
    if a.get("invalidation") is not None:
        lines.append(f"Underlying invalidation level: {_p(a.get('invalidation'), 2)}")
    if a.get("option_risk"):
        lines.append(f"Option risk-control method: {a['option_risk']}")
    targets = a.get("targets") or []
    if targets:
        und = ", ".join(_p(t, 1) for t in targets[:2])
        prem = ", ".join(_p(t, 1) for t in targets[2:4])
        lines.append(f"Targets: underlying {und}" + (f"; premium {prem}" if prem else ""))
    if a.get("ml_prob") is not None:
        lines.append(f"ML win chance (index reaches 1.5R before the stop): {float(a['ml_prob']):.0%} - a model estimate")
    if a.get("rr") is not None:
        lines.append(f"Estimated risk-reward: {_p(a.get('rr'), 2)} (to T1, after costs)")
    if opt.get("lots"):
        lines.append(f"Size: {opt.get('lots')} lot(s) x {opt.get('lot_size')}, risk Rs {_p(opt.get('risk_rupees'), 0)}")
    if a.get("expiry_info"):
        lines.append(f"Expiry and liquidity information: {a['expiry_info']}")
    lines.append(f"Data freshness: {a.get('freshness') or '-'}")
    if a.get("avoid_reason"):
        lines.append(f"Reason to avoid entry: {a['avoid_reason']}")
    if a.get("entry_type"):
        lines.append(f"Entry type: {a['entry_type']}" + (" (before the breakout, reduced size)" if a["entry_type"] == "early" else ""))
    if kind != "entry":
        lines.append("Alert only - this message is not an order.")
    return title, "\n".join(lines)


def format_movement(a: dict) -> str:
    """Body of a 5 / 10 / 15-minute movement trigger (src/movement)."""
    window = "all windows aligned" if a.get("window") == "ALL" else f"{a.get('window')} minutes"
    limits = []
    if a.get("threshold_pct"):
        limits.append(f"{_p(a.get('threshold_pct'), 2)}%")
    if a.get("threshold_points"):
        limits.append(f"{_p(a.get('threshold_points'), 0)} pts")
    checks = ", ".join(f"{k} {v}" for k, v in (a.get("checks") or {}).items()) or "none switched on"
    lines = [f"Index: {a.get('index')}", f"Time: {str(a.get('time', ''))[:19].replace('T', ' ')} IST",
             f"Window: {window}", f"Direction: {a.get('direction')}  Quality: {a.get('quality')}",
             f"Price: {_p(a.get('ref_price'), 2)} -> {_p(a.get('price'), 2)} "
             f"({_p(a.get('change'), 1)} pts, {_p(a.get('change_pct'), 2)}%)",
             f"Threshold: {' or '.join(limits) or '-'} (move = {_p(a.get('threshold_ratio'), 2)}x)"]
    if a.get("index") != "INDIAVIX":
        lines += [f"Strength: {_p(a.get('strength'), 2)}x normal  RVOL: {_p(a.get('rvol'), 2)}  "
                  f"Consistency: {_p((a.get('consistency') or 0) * 100 if a.get('consistency') is not None else None, 0)}%",
                  f"Futures OI: {_p(a.get('oi_change_pct'), 2)}% {a.get('oi_reading') or ''}".rstrip(),
                  f"RSI(1m): {_p(a.get('rsi_1m'), 1)}  Fut vs VWAP: {_p(a.get('fut_vwap_dist_pct'), 2)}%  "
                  f"Day: {_p(a.get('day_change_pct'), 2)}%",
                  f"Checks: {checks}"]
    lines.append("Alert only - this message is not an order.")
    return "\n".join(lines)


def format_paper_trade(a: dict) -> str:
    """Body of a movement paper-trade entry / exit (src/movement/paper.py)."""
    trig = a.get("trigger") or {}
    lines = [f"Book: {a.get('book')}  Index: {a.get('index')}  Side: {a.get('side')}",
             f"Contract: {a.get('symbol')}  Qty: {a.get('qty')} ({a.get('qty', 0) // max(a.get('lot_size') or 1, 1)} lot)",
             f"Entry: {_p(a.get('entry_price'), 2)} at {str(a.get('entry_time', ''))[11:19]}  "
             f"Stop: {_p(a.get('stop'), 2)}  Target: {_p(a.get('target'), 2)}",
             f"Trigger: {trig.get('message') or '-'}"]
    if a.get("action") == "close":
        lines += [f"Exit: {_p(a.get('exit_price'), 2)} at {str(a.get('exit_time', ''))[11:19]} - {a.get('exit_reason')}",
                  f"Net P&L after charges: Rs {_p(a.get('net'), 0)} ({_p(a.get('pnl_pct'), 1)}% on premium)"]
    lines.append("PAPER TRADE - simulated, no order was placed.")
    return "\n".join(lines)


class Notifier:
    def __init__(self, notify_section: Optional[dict] = None, csv_path: Path = ALERTS_CSV,
                 telegram_sender=None, popup=None, enabled: bool = True):
        sec = notify_section or {}
        self.settings = {k: _truthy(sec.get(k), True) for k in set(KIND_TO_SETTING.values())}
        self.settings["notify_early_watch"] = _truthy(sec.get("notify_early_watch"), False)
        self.telegram_on = _truthy(sec.get("telegram"), True) and enabled
        self.popup_on = _truthy(sec.get("desktop_popup"), True) and enabled
        self.sound = _truthy(sec.get("sound"), True)
        self.popup_seconds = int(float(sec.get("popup_seconds", 30) or 30))
        self.gap = float(sec.get("min_seconds_between_messages", 1.5) or 1.5)
        self.csv_path = Path(csv_path)
        self.token, self.chat = get_secret("TELEGRAM_BOT_TOKEN"), get_secret("TELEGRAM_CHAT_ID")
        if self.telegram_on and not (self.token and self.chat):
            logger.warning("Telegram is switched on in [NOTIFY] but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set "
                           "in .env: Telegram messages are off (alerts are still logged, stored and shown).")
        self.telegram_on = self.telegram_on and bool(self.token and self.chat)
        self._send = telegram_sender or self._telegram_post
        self._popup = popup or self._windows_popup
        self._queue: "queue.Queue[tuple]" = queue.Queue(maxsize=500)
        self._warned = False
        self._worker = threading.Thread(target=self._run, name="prebreakout-notify", daemon=True)
        self._worker.start()

    def channels(self):
        out = ["log", "csv", "db", "dashboard"]
        if self.popup_on and sys.platform.startswith("win"):
            out.append("desktop popup")
        if self.telegram_on:
            out.append("Telegram")
        return out

    def wants(self, kind: str) -> bool:
        return self.settings.get(KIND_TO_SETTING.get(kind, ""), True)

    def emit(self, alert: dict, push: bool = True):
        """Log + CSV always; Telegram / popup when the type is enabled and push is True."""
        try:
            title, body = format_alert(alert)
            logger.warning("\n%s\n  PRE-BREAKOUT  %s\n  %s\n%s", "=" * 78, title, body.replace("\n", "\n  "), "=" * 78)
            self._csv(alert)
            if push and self.wants(alert.get("kind", "")):
                try:
                    self._queue.put_nowait((title, body, alert.get("kind", "")))
                except queue.Full:
                    logger.warning("Notification queue full: dropped a push (the alert is still stored).")
        except Exception:
            logger.exception("Alert formatting failed (the service continues).")

    def _csv(self, a: dict):
        opt = a.get("option") or {}
        row = {k: a.get(k) for k in CSV_FIELDS}
        row.update(option=opt.get("symbol"), premium=opt.get("entry_premium") or opt.get("ltp"),
                   spread_pct=opt.get("spread_pct"), targets=";".join(str(t) for t in (a.get("targets") or [])))
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.csv_path.exists()
            with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
                if new:
                    writer.writeheader()
                writer.writerow(row)
        except OSError:
            logger.warning("Could not write %s", self.csv_path)

    def _run(self):
        last = 0.0
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            title, body, kind = item
            wait = last + self.gap - clock.monotonic()
            if wait > 0:
                clock.sleep(wait)
            last = clock.monotonic()
            if self.telegram_on:
                try:
                    self._send(self.token, self.chat, f"{title}\n{body}")
                except Exception as exc:  # never log the token or the URL
                    if not self._warned:
                        self._warned = True
                        logger.warning("Telegram send failed (%s); other channels continue.", type(exc).__name__)
            if self.popup_on and kind not in ("early_watch",):
                threading.Thread(target=self._desktop, args=(title, body, kind), daemon=True).start()

    def _desktop(self, title, body, kind):
        try:
            if self.sound and sys.platform.startswith("win"):
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION if kind in ("entry", "data") else winsound.MB_OK)
            self._popup(title, body, self.popup_seconds)
        except Exception:
            pass

    @staticmethod
    def _windows_popup(title, text, seconds):
        if not sys.platform.startswith("win"):
            return
        import ctypes
        flags = 0x40 | 0x40000 | 0x10000 | 0x1000
        user32 = ctypes.windll.user32
        try:
            user32.MessageBoxTimeoutW(None, text, title, flags, 0, int(seconds) * 1000)
        except AttributeError:
            user32.MessageBoxW(None, text, title, flags)

    @staticmethod
    def _telegram_post(token, chat_id, message):
        import requests
        response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                 data={"chat_id": chat_id, "text": message[:4000]}, timeout=8)
        response.raise_for_status()

    def flush(self, timeout: float = 5.0):
        end = clock.monotonic() + timeout
        while not self._queue.empty() and clock.monotonic() < end:
            clock.sleep(0.05)

    def close(self, timeout: float = 5.0):
        """Send what is queued (up to `timeout` s), then stop the worker thread. Safe to call twice."""
        if not self._worker.is_alive():
            return
        self.flush(timeout)
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            logger.warning("Notification queue still full at shutdown: unsent pushes dropped (alerts are stored).")
            return
        self._worker.join(timeout=max(1.0, timeout))
