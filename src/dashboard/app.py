"""
Built-in dashboard server (asyncio, standard library only). Started by the application controller in
the same process as the scanner; it never opens a browser. Open the address it logs yourself:

    http://127.0.0.1:8502/             HTML dashboard (refreshes every 30 s)
    http://127.0.0.1:8502/api/state    the same data as JSON
    http://127.0.0.1:8502/health       {"status": "ok", ...} for a quick check

It only reads the latest state the scanner published, so a slow browser can never slow the scanner.
It binds to 127.0.0.1 by default ([DASHBOARD] host in config/scanner.ini). If the port is already in use
(another scanner or program), the dashboard is skipped with a clear log line and the scanner continues;
output/prebreakout_dashboard.html is still written every cycle.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

from src.dashboard.view import render_html
from src.scanner.store import json_default

logger = logging.getLogger("scanner.dashboard")

WAITING_PAGE = ("<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='5'>"
                "<title>NIFTY SENSEX Scanner</title></head><body style='font-family:system-ui;padding:24px'>"
                "<h1>Scanner starting</h1><p>The first cycle has not finished yet. This page refreshes itself.</p>"
                "</body></html>")


class DashboardServer:
    def __init__(self, state_fn: Callable[[], Optional[dict]], host: str = "127.0.0.1", port: int = 8502):
        self.state_fn = state_fn
        self.host, self.port = host, port
        self._server: Optional[asyncio.AbstractServer] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def running(self) -> bool:
        return self._server is not None

    async def start(self) -> bool:
        if self._server is not None:
            logger.warning("Dashboard server already running at %s; not starting a second one.", self.url)
            return True
        try:
            self._server = await asyncio.start_server(self._handle, self.host, self.port)
        except OSError as error:
            logger.error("Dashboard not started: %s:%s is in use or not allowed (%s). Another scanner may be "
                         "running, or change [DASHBOARD] port in config/scanner.ini.", self.host, self.port, error)
            return False
        logger.info("Dashboard: %s (open it in your browser)", self.url)
        return True

    async def stop(self):
        if self._server is None:
            return
        self._server.close()
        try:
            await asyncio.wait_for(self._server.wait_closed(), timeout=3)
        except asyncio.TimeoutError:
            logger.debug("Dashboard connections did not close within 3 s")
        self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request = await asyncio.wait_for(reader.readline(), timeout=5)
            while True:  # drain the headers
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if line in (b"\r\n", b"\n", b""):
                    break
            parts = request.decode("latin-1").split()
            method, path = (parts[0], parts[1].split("?")[0]) if len(parts) >= 2 else ("GET", "/")
            status, ctype, body = self._route(method, path)
            head = (f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n"
                    "Cache-Control: no-store\r\nConnection: close\r\n\r\n").encode("latin-1")
            writer.write(head + (body if method != "HEAD" else b""))
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError):
            pass
        except Exception:
            logger.exception("Dashboard request failed")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    def _route(self, method: str, path: str):
        if method not in ("GET", "HEAD"):
            return "405 Method Not Allowed", "text/plain", b"read-only"
        state = self.state_fn()
        if path in ("/", "/index.html"):
            html = render_html(state) if state else WAITING_PAGE
            return "200 OK", "text/html; charset=utf-8", html.encode("utf-8")
        if path == "/api/state":
            return "200 OK", "application/json", json.dumps(state or {}, default=json_default).encode("utf-8")
        if path == "/health":
            payload = {"status": "ok", "has_state": bool(state),
                       "generated_at": (state or {}).get("generated_at"), "mode": (state or {}).get("mode")}
            return "200 OK", "application/json", json.dumps(payload).encode("utf-8")
        return "404 Not Found", "text/plain", b"not found"
