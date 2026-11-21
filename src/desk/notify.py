"""
Desktop notification when a trade triggers. Runs inside the trading desk (python -m src.desk): every few
seconds it reads the stock service's and the F&O engine's state and shows a Windows popup with a sound for anything new:

    - a stock option paper trade opened                 (a VALID plan inside the entry window)
    - a breakout paper trade opened                     (the 15-minute bar closed beyond the level)
    - a breakout moved to TRIGGERED / CONFIRMED         (price went through the trigger)
    - an F&O engine paper position opened               (a strategy signal passed every check: BUY CE / BUY PE)

The index scanner (python -m src.main) already shows its own popups for NIFTY / SENSEX, so index trades are not
repeated here. What is already in the books when the desk starts (or when a service comes up) is not announced
again. Several events in the same check are shown in one popup.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("desk.notify")

TRIGGER_KINDS = ("triggered", "confirmed")


def popup(title: str, text: str, seconds: int = 30):
    """Topmost Windows message box that closes itself (same style as the index scanner's popups)."""
    if not sys.platform.startswith("win"):
        return
    import ctypes
    import winsound
    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    flags = 0x40 | 0x40000 | 0x10000 | 0x1000      # info icon, topmost, foreground, system modal
    user32 = ctypes.windll.user32
    try:
        user32.MessageBoxTimeoutW(None, text, title, flags, 0, int(seconds) * 1000)
    except AttributeError:
        user32.MessageBoxW(None, text, title, flags)


def _n(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return "-"


def events(stocks: Optional[dict]) -> List[Tuple[str, str]]:
    """[(key, text)] for every pre-breakout trade / trigger in the stock state. Stock-option plans (the stock
    service's own `book`) are no longer shown on the desk, so they raise no popup either. Keys are stable per event."""
    out: List[Tuple[str, str]] = []
    if not stocks:
        return out
    b = stocks.get("breakout") or {}
    for t in (b.get("book") or {}).get("trades") or []:
        line = (f"BREAKOUT TRADE OPENED  {t.get('side', '')} {t.get('symbol', '')}\n"
                f"Entry {_n(t.get('entry_spot'))} · SL {_n(t.get('stop'))} · T1 {_n(t.get('t1'))} · T2 {_n(t.get('t2'))}")
        if t.get("contract"):
            line += f"\nOption: {t['contract']}"
        out.append((f"brk:{t.get('symbol')}:{t.get('side')}:{t.get('opened')}", line))
    for a in b.get("alerts") or []:
        if a.get("kind") not in TRIGGER_KINDS:
            continue
        line = (f"BREAKOUT {str(a.get('kind')).upper()}  {a.get('side', '')} {a.get('symbol', '')}\n"
                f"Trigger {_n(a.get('trigger'))} · SL {_n(a.get('stop'))} · T1 {_n(a.get('t1'))} · T2 {_n(a.get('t2'))}")
        if a.get("contract"):
            line += f"\nOption: {a['contract']}"
        out.append((f"alert:{a.get('key') or (a.get('kind'), a.get('symbol'), a.get('side'), a.get('time'))}", line))
    return out


def fno_events(fno: Optional[dict]) -> List[Tuple[str, str]]:
    """[(key, text)] for every paper position of the F&O engine (open, or opened and closed today)."""
    out: List[Tuple[str, str]] = []
    for p in ((fno or {}).get("positions") or []) + ((fno or {}).get("closed_today") or []):
        plan = p.get("plan") or {}
        pct = (lambda v, sign: f" ({sign}{v:g}%)" if v else "")
        out.append((f"fno:{p.get('id')}",
                    f"F&O TRADE OPENED  BUY {p.get('side', '')}  {p.get('instrument', '')}\n"
                    f"Entry {_n(p.get('entry'))} · SL {_n(plan.get('stop'))}{pct(plan.get('stop_pct'), '-')} · "
                    f"T1 {_n(plan.get('t1'))}{pct(plan.get('t1_pct'), '+')} · T2 {_n(plan.get('t2'))}{pct(plan.get('t2_pct'), '+')}\n"
                    f"{plan.get('strategy_name') or p.get('strategy', '')} · confidence {_n(plan.get('confidence'))} · "
                    f"R:R 1:{_n(plan.get('rr'))} · {p.get('qty', '-')} qty"))
    return out


READERS = {"stocks": events, "fno": fno_events}


class TradeWatcher:
    """`fetch` is one callable (the stock service) or {"stocks": callable, "fno": callable}."""

    def __init__(self, fetch, show: Callable[[str, str, int], None] = popup, interval: float = 10.0, seconds: int = 30):
        self.sources: Dict[str, Callable[[], Optional[dict]]] = fetch if isinstance(fetch, dict) else {"stocks": fetch}
        self.show, self.interval, self.seconds = show, interval, seconds
        self.baselines: Dict[str, set] = {}       # per source; missing until its first successful read
        self._stop = threading.Event()

    @property
    def seen(self) -> Optional[set]:
        return self.baselines.get("stocks")

    def check(self) -> List[str]:
        """One pass: returns the texts that were announced."""
        fresh: List[str] = []
        for name, fetch in self.sources.items():
            state = fetch()
            if not state:
                continue
            found = READERS[name](state)
            if name not in self.baselines:        # first read of this service: what is there is not news
                self.baselines[name] = {k for k, _ in found}
                continue
            fresh += [text for key, text in found if key not in self.baselines[name]]
            self.baselines[name].update(k for k, _ in found)
        if fresh:
            title = "Trade triggered" if len(fresh) == 1 else f"{len(fresh)} trades triggered"
            body = "\n\n".join(fresh[:6]) + (f"\n\n+ {len(fresh) - 6} more" if len(fresh) > 6 else "")
            body += "\n\nPaper trading only."
            threading.Thread(target=self._show, args=(title, body), daemon=True).start()
        return fresh

    def _show(self, title, body):
        try:
            self.show(title, body, self.seconds)
        except Exception:
            logger.exception("desktop notification failed")

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="desk-notify").start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.check()
            except Exception:
                logger.exception("trade watcher pass failed")
            self._stop.wait(self.interval)
