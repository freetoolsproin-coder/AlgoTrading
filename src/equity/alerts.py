"""
Alerts of the equity service. Every alert is stored and shown on the dashboard; the push channels are configurable
in [ALERTS] of config/equity.ini:

    desktop_popup   Windows message box with a sound (several alerts of one cycle share one popup)
    telegram        TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env
    webhook         JSON POST to EQUITY_WEBHOOK_URL from .env

Kinds: buy_signal, sell_signal, entry, stop, target, trend_reversal, volume_breakout, swing_exit, risk_limit, order.
`notify_<kind> = false` keeps a kind on the dashboard only.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, List, Optional

import pandas as pd

from src.config.settings import get_secret
from src.equity.config import AlertConfig

logger = logging.getLogger("scanner.equity.alerts")
KINDS = ("buy_signal", "sell_signal", "entry", "stop", "target", "trend_reversal", "volume_breakout", "swing_exit",
         "risk_limit", "order")


class Alerter:
    def __init__(self, cfg: AlertConfig, store=None, push: bool = True, clock: Optional[Callable[[], pd.Timestamp]] = None):
        self.cfg, self.store, self.push = cfg, store, push
        self.clock = clock or (lambda: pd.Timestamp.now(tz="Asia/Kolkata"))
        self.recent: List[dict] = []
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._seen = set()
        if push:
            threading.Thread(target=self._worker, daemon=True, name="equity-alerts").start()

    def wants(self, kind: str) -> bool:
        return bool(getattr(self.cfg, f"notify_{kind}", True))

    def emit(self, kind: str, text: str, symbol: str = "", module: str = "", data: Optional[dict] = None,
             key: Optional[str] = None, push: bool = True) -> Optional[dict]:
        """`key` makes an alert once-only (e.g. the daily risk-limit alert)."""
        if key is not None:
            if key in self._seen:
                return None
            self._seen.add(key)
        now = self.clock()
        alert = {"time": now.isoformat(timespec="seconds"), "day": now.date().isoformat(), "kind": kind, "module": module,
                 "symbol": symbol, "text": text, "data": data or {}}
        if self.store is not None and self.cfg.dashboard:
            alert["id"] = self.store.insert("alerts", **alert)
        self.recent = (self.recent + [alert])[-300:]
        logger.info("ALERT %s %s %s", kind, symbol, text.replace("\n", " | "))
        if self.push and push and self.wants(kind):
            self._q.put(alert)
        return alert

    # ------------------------------------------------------------------ channels
    def _worker(self):
        while True:
            batch = [self._q.get()]
            try:
                while len(batch) < 8:
                    batch.append(self._q.get(timeout=1.0))
            except queue.Empty:
                pass
            for send in (self._telegram, self._webhook, self._popup):      # the popup blocks until it closes: last
                try:
                    send(batch)
                except Exception as error:      # noqa: BLE001 - a channel must never stop the engine
                    logger.warning("alert channel %s failed (%s)", send.__name__, type(error).__name__)

    def _popup(self, batch):
        if not self.cfg.desktop_popup:
            return
        from src.desk.notify import popup
        title = "NIFTY 50 equity: " + (batch[0]["kind"].replace("_", " ") if len(batch) == 1 else f"{len(batch)} alerts")
        popup(title, "\n\n".join(a["text"] for a in batch), int(self.cfg.popup_seconds))

    def _telegram(self, batch):
        token, chat = get_secret("TELEGRAM_BOT_TOKEN"), get_secret("TELEGRAM_CHAT_ID")
        if not (self.cfg.telegram and token and chat):
            return
        import requests
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": "\n\n".join(a["text"] for a in batch)[:4000]}, timeout=8).raise_for_status()

    def _webhook(self, batch):
        url = get_secret("EQUITY_WEBHOOK_URL")
        if not (self.cfg.webhook and url):
            return
        import requests
        requests.post(url, json={"alerts": batch}, timeout=8).raise_for_status()
