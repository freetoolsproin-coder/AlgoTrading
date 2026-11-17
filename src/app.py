"""
Application controller: builds every component once, runs them as asyncio tasks in ONE process, and
shuts them down cleanly.

    component            how it runs                                      started
    market data          LiveProvider (Upstox REST) or MockProvider       built once
    WebSocket feed       asyncio task, one connection, auto-reconnect     once (live mode, [SERVICE] use_websocket)
    scanner engine       asyncio task; each cycle runs in a worker        once
                         thread (pandas + REST calls are blocking)
    notifications        one background worker thread (Telegram/popups)   once
    dashboard server     asyncio server on 127.0.0.1:8502                 once ([DASHBOARD] enabled)

Guarantees
  * one process: a lock file (data/scanner.lock) stops a second instance before it starts anything
  * each service is started exactly once; start() refuses a second call
  * exceptions in a cycle are logged with a traceback and the scanner keeps going; an expired token
    waits for a new one instead of crashing
  * Ctrl+C / stop in the IDE: the scanner loop stops, the WebSocket closes, the dashboard server stops,
    queued notifications are sent (5 s max), logs are flushed, the lock is released
No subprocesses, no extra terminals, no browser windows.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Optional

from src.broker.client import TokenSource, UpstoxAuthError, UpstoxClient
from src.broker.market_data import LiveProvider, now_ist
from src.broker.mock_data import MockProvider
from src.broker.websocket_feed import WebSocketFeed
from src.config import settings
from src.config.settings import AppSettings, PBConfig
from src.dashboard.app import DashboardServer
from src.notifications.service import Notifier
from src.scanner.engine import SESSION_END, SESSION_START, ScannerEngine
from src.scanner.store import Store
from src.utils import market_calendar, single_instance
from src.utils.numbers import hhmm

logger = logging.getLogger("scanner.app")


@dataclass
class RunOptions:
    mock: bool = False
    speed: float = 60.0
    start: str = "09:15"
    notify: bool = False          # mock mode: also push Telegram / popups
    once: bool = False
    dashboard: bool = True
    websocket: bool = True
    config: Path = settings.CONFIG_FILE


class Application:
    def __init__(self, options: RunOptions):
        self.options = options
        self.app: Optional[AppSettings] = None
        self.cfg: Optional[PBConfig] = None
        self.engine: Optional[ScannerEngine] = None
        self.notifier: Optional[Notifier] = None
        self.feed: Optional[WebSocketFeed] = None
        self.dashboard: Optional[DashboardServer] = None
        self.mock = False
        self.exit_code = 0
        self._stop: Optional[asyncio.Event] = None
        self._tasks: dict = {}
        self._lock = None
        self._started = False
        self._shut_down = False
        self._cfg_mtime = None

    # ------------------------------------------------------------------ build
    def build(self):
        settings.ensure_dirs()
        self.app = settings.load_app_settings(self.options.config)
        self.cfg = PBConfig.from_ini(self.options.config)
        self._cfg_mtime = Path(self.options.config).stat().st_mtime
        self.mock = self.options.mock or self.app.mode == "mock"
        names = list(self.app.indices)
        if self.mock:
            provider = MockProvider(self.cfg, names, speed=self.options.speed, start_clock=hhmm(self.options.start))
            push = self.options.notify
        else:
            client = UpstoxClient(requests_per_second=self.app.requests_per_second)
            provider = LiveProvider(names, self.cfg, client=client)
            if self.app.use_websocket and self.options.websocket:
                self.feed = WebSocketFeed(client, provider.subscription_keys)
                provider.live_feed = self.feed
            push = True
        self.notifier = Notifier(self.app.notify, enabled=push)
        self.engine = ScannerEngine(self.cfg, self.app, provider, self.notifier, Store(), push=push,
                                    feed_status=self.feed.status if self.feed else None)
        if self.app.dashboard_enabled and self.options.dashboard:
            self.dashboard = DashboardServer(lambda: self.engine.last_state, self.app.dashboard_host,
                                             self.app.dashboard_port)
        self._preflight(names)

    def _preflight(self, names):
        logger.info("NIFTY & SENSEX pre-breakout scanner | %s | indices: %s | %s | no live orders",
                    "MOCK DATA" if self.mock else "LIVE DATA", ", ".join(names),
                    "paper trading" if self.app.paper_trading else "alert-only mode")
        if not market_calendar.has_year(date.today().year):
            logger.warning("config/nse_holidays.csv has no %d holidays: add the exchange's list.", date.today().year)
        if self.mock:
            return
        try:
            TokenSource().token()
        except UpstoxAuthError as error:
            logger.error("%s The scanner will start and wait for a token.", error)
        logger.info("Notification channels: %s", ", ".join(self.notifier.channels()))

    # ------------------------------------------------------------------ run
    async def run(self) -> int:
        self._stop = asyncio.Event()
        self._lock = single_instance.acquire(settings.LOCK_FILE)
        if self._lock is None:
            logger.error("Another scanner is already running (lock: %s). Stop it first; not starting a duplicate.",
                         settings.LOCK_FILE)
            return 2
        self._install_signal_handlers()
        try:
            self.build()
            await self.start()
            await self._stop.wait()
        except asyncio.CancelledError:   # Ctrl+C on Windows / IDE stop: asyncio.run cancels this task
            logger.info("Stop requested.")
        except (FileNotFoundError, ValueError) as error:   # missing / invalid config/scanner.ini
            logger.error("Cannot start: %s", error)
            self.exit_code = 1
        except Exception:
            logger.exception("Fatal error during start-up")
            self.exit_code = 1
        finally:
            await self.shutdown()
        return self.exit_code

    async def start(self):
        if self._started:
            raise RuntimeError("Application already started")
        self._started = True
        if self.dashboard is not None:
            await self.dashboard.start()
        if self.feed is not None:
            self._tasks["websocket"] = asyncio.create_task(self.feed.run(), name="websocket-feed")
        self._tasks["scanner"] = asyncio.create_task(self._scanner_loop(), name="scanner-loop")
        logger.info("Running. Press Ctrl+C (or stop in your IDE) to shut down cleanly.")

    def request_stop(self):
        if self._stop is not None and not self._stop.is_set():
            logger.info("Shutting down ...")
            self._stop.set()

    def _install_signal_handlers(self):
        loop = asyncio.get_running_loop()
        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except (NotImplementedError, RuntimeError):
                pass  # Windows: Ctrl+C arrives as KeyboardInterrupt -> task cancellation, handled in run()

    # ------------------------------------------------------------------ scanner loop
    async def _scanner_loop(self):
        engine = self.engine
        loop_seconds = 5 if self.mock else self.app.loop_seconds
        failures = 0
        while not self._stop.is_set():
            started = asyncio.get_running_loop().time()
            self._reload_config()
            wait = loop_seconds
            try:
                state = await asyncio.to_thread(engine.cycle)
                engine.last_cycle_error = ""
                failures = 0
                if self.options.once:
                    print_summary(state)
                    self.request_stop()
                    break
                if self.mock and engine.provider.now().time() > time(15, 31):
                    logger.info("Mock session finished. Dashboard file: %s", settings.HTML_FILE)
                    self.request_stop()
                    break
            except UpstoxAuthError as error:
                engine.status = "Upstox token missing or expired - run: python -m src.broker.auth"
                engine.last_cycle_error = str(error)
                logger.error("%s Waiting 60 s for a new token.", error)
                await asyncio.to_thread(self._publish_safely)
                wait = 60
                if self.options.once:
                    self.exit_code = 1
                    self.request_stop()
                    break
            except Exception as error:
                failures += 1
                engine.last_cycle_error = f"{type(error).__name__}: {error}"[:300]
                logger.exception("Cycle failed (%d in a row); the scanner continues.", failures)
                wait = min(loop_seconds * failures, 300)
                if self.options.once:
                    self.exit_code = 1
                    self.request_stop()
                    break
            if not self.mock and not (SESSION_START <= now_ist().time() <= SESSION_END):
                wait = max(wait, 120)
            elapsed = asyncio.get_running_loop().time() - started
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=max(2.0, wait - elapsed))
            except asyncio.TimeoutError:
                pass

    def _publish_safely(self):
        try:
            self.engine._publish(now_ist(), market_open=False)
        except Exception:
            logger.exception("Could not publish the dashboard state")

    def _reload_config(self):
        """Strategy / risk thresholds edited in config/scanner.ini apply without a restart."""
        try:
            mtime = Path(self.options.config).stat().st_mtime
        except OSError:
            return
        if mtime == self._cfg_mtime:
            return
        self._cfg_mtime = mtime
        try:
            cfg = PBConfig.from_ini(self.options.config)
            app = settings.load_app_settings(self.options.config)
        except Exception as error:
            logger.error("config/scanner.ini changed but is invalid (%s); keeping the previous settings.", error)
            return
        self.engine.apply_config(cfg)
        self.cfg = cfg
        if tuple(app.indices) != tuple(self.app.indices) or app.dashboard_port != self.app.dashboard_port:
            logger.warning("Index list / dashboard port changes need a restart to take effect.")
        logger.warning("config/scanner.ini changed: new strategy and risk settings applied.")

    # ------------------------------------------------------------------ shutdown
    async def shutdown(self):
        if self._shut_down:
            return
        self._shut_down = True
        if self._stop is not None:
            self._stop.set()
        scanner = self._tasks.get("scanner")
        if scanner is not None and not scanner.done():
            try:  # let a running cycle finish (it is in a worker thread and cannot be interrupted safely)
                await asyncio.wait_for(asyncio.shield(scanner), timeout=30)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                scanner.cancel()
            except Exception:
                logger.exception("Scanner loop ended with an error")
        if self.feed is not None:
            await self.feed.stop()
            task = self._tasks.get("websocket")
            if task is not None:
                try:
                    await asyncio.wait_for(task, timeout=10)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
                except Exception:
                    logger.exception("WebSocket task ended with an error")
        if self.dashboard is not None:
            await self.dashboard.stop()
        if self.notifier is not None:
            await asyncio.to_thread(self.notifier.close, 5.0)
        single_instance.release(self._lock)
        self._lock = None
        logger.info("Stopped cleanly.")


def print_summary(state: dict):
    print(f"\n{state['mode']} | {state['status']} | VIX {state.get('vix')}")
    for name, s in state["indices"].items():
        ce, pe = s.get("ce") or {}, s.get("pe") or {}
        print(f"  {name:<7} {s.get('price')}  bias {s.get('bias'):<7} CE {ce.get('score')}  PE {pe.get('score')}  "
              f"state {s.get('state')}  {str(s.get('state_reason', ''))[:80]}")
    r = state["risk"]
    print(f"  risk: day P&L {r['day_pnl']}, trades {r['trades_total']}, halted {r['halted']}")
    print(f"  dashboard file: {settings.HTML_FILE}")
    sys.stdout.flush()
