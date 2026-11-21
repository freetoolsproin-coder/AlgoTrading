"""
Trading desk: ONE page that combines the dashboards.

    NIFTY / SENSEX index scanner      python -m src.main      http://127.0.0.1:8502
    NIFTY 50 stock options service    python -m src.stocks    http://127.0.0.1:8504
    F&O multi-strategy engine         python -m src.fno       http://127.0.0.1:8520   (optional)

    python -m src.desk                      page on http://127.0.0.1:8500
    python -m src.desk --port 8600          another port
    python -m src.desk --no-notify          without the desktop popup when a stock trade triggers (src/desk/notify.py)
    python -m src.desk --test-notify        show one sample popup and exit
    python -m src.desk --index-url http://127.0.0.1:8502 --stocks-url http://127.0.0.1:8504
    python -m src.desk --with-equity        also host the NIFTY 50 cash-equity service (off by default; its tabs were removed from the page)

It is a front end (standard library only): it reads /api/state of the services and serves one page.
The services keep running exactly as before and their own pages still work. If one of them is not running,
its part of the page says so and the other parts keep working.

With --with-equity the desk also runs the NIFTY 50 cash-equity service (off by default): unless something already answers on its port, the desk
hosts it in this same process (a background thread, stopped with the desk), so that everything is reachable from
this one address with one command. Its API is passed through under /api/eq/, POST included; POSTs are accepted from
this computer only.

    GET /              the combined page
    GET /api/all       {"index": <8502 state>, "stocks": <8504 state>, "fno": <8520 state, trimmed>, "errors": {...}, "links": {...}}
    GET /api/search    stock search suggestions (passed on to the stock service)
    GET|POST /api/eq/* the equity service's /api/* (state, chart, mode, kill switch, order confirmation, backtest ...)
    GET /health
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"
ASSETS = {"/static/equity.js": "text/javascript; charset=utf-8", "/static/equity.css": "text/css; charset=utf-8"}
TIMEOUT = 6
_pool = ThreadPoolExecutor(max_workers=8)


def _clean(o):
    """NaN / Infinity are not valid JSON for a browser."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def fetch_json(url: str):
    """(data, error): the JSON of one upstream service, or a short reason why it could not be read."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return _clean(json.loads(r.read().decode("utf-8"))), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return None, f"not reachable ({getattr(e, 'reason', e)})"
    except ValueError:
        return None, "answer was not JSON"


FNO_KEYS = ("status", "mock", "mode", "time", "cycle", "capital", "indices", "positions", "closed_today", "risk")


def fno_trim(state):
    """The part of the F&O engine's state the page shows (its watchlist and stock ranking stay on its own page)."""
    if not state:
        return None
    out = {k: state.get(k) for k in FNO_KEYS}
    out["signals"] = (state.get("signals") or [])[:60]
    out["errors"] = ((state.get("health") or {}).get("errors") or [])[:5]
    return out


def forward(method: str, url: str, body: bytes = b"", timeout: float = 20):
    """(status, body bytes) of one request passed on to the equity service."""
    req = urllib.request.Request(url, data=body if method == "POST" else None, method=method,
                                 headers={"Content-Type": "application/json"} if method == "POST" else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or json.dumps({"ok": False, "error": f"HTTP {e.code}"}).encode()
    except (urllib.error.URLError, OSError) as e:
        return 503, json.dumps({"ok": False, "offline": True,
                                "error": f"equity service not reachable ({getattr(e, 'reason', e)})"}).encode()


def start_equity(url: str):
    """Host the equity service (engine + its Flask API) inside this process on the port of `url`, unless something
    already answers there. Returns the engine, or None when a separate `python -m src.equity` is running."""
    parts = urllib.parse.urlsplit(url)
    if parts.hostname not in ("127.0.0.1", "localhost") or fetch_json(url + "/health")[1] is None:
        return None
    import logging

    port = parts.port or 8530
    try:
        from src.config import settings
        from src.equity.config import EquityConfig
        from src.equity.engine import EquityEngine
        from src.equity.web import create_app
        from src.utils.logging_config import setup_logging
    except ImportError:
        # The desk itself needs only the standard library, so it also starts under a Python that lacks the equity
        # service's packages (plain `python` instead of the project's .venv). Run the service with the .venv then.
        root = Path(__file__).resolve().parents[2]
        venv = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not venv.exists() or venv.resolve() == Path(sys.executable).resolve():
            raise
        import atexit
        import subprocess
        child = subprocess.Popen([str(venv), "-m", "src.equity", "--port", str(port)], cwd=str(root))
        atexit.register(child.terminate)
        return child
    settings.ensure_dirs()
    setup_logging(logging.INFO, log_file=settings.LOG_DIR / "equity.log")
    engine = EquityEngine(EquityConfig.from_ini())
    app = create_app(engine)

    def serve():
        try:
            from waitress import serve as run
        except ImportError:
            app.run(host="127.0.0.1", port=port, threaded=True)
        else:
            run(app, host="127.0.0.1", port=port, threads=8, _quiet=True)
    threading.Thread(target=serve, daemon=True, name="equity-web").start()
    engine.start()
    return engine


def make_handler(index_url: str, stocks_url: str, fno_url: str, equity_url: str = "http://127.0.0.1:8530"):
    class Handler(BaseHTTPRequestHandler):
        server_version = "desk"

        def log_message(self, *args):  # quiet: one line per request is noise here
            pass

        def _send(self, status: int, ctype: str, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, payload, status: int = 200):
            self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

        def do_HEAD(self):
            self.do_GET()

        def _equity(self, method: str, parsed, body: bytes = b""):
            rest = parsed.path[len("/api/eq/"):]
            if not all(ch.isalnum() or ch in "/_%&.-" for ch in rest) or ".." in rest:      # already percent-encoded
                return self._json({"ok": False, "error": "bad path"}, 404)
            url = f"{equity_url}/api/{rest}" + (f"?{parsed.query}" if parsed.query else "")
            status, data = forward(method, url, body)
            self._send(status, "application/json", data)

        def do_POST(self):
            parsed = urllib.parse.urlsplit(self.path)
            if not parsed.path.startswith("/api/eq/"):
                return self._send(404, "text/plain", b"not found")
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                return self._json({"ok": False, "error": "accepted from this computer only"}, 403)
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 65536)
            except ValueError:
                length = 0
            self._equity("POST", parsed, self.rfile.read(length) if length else b"{}")

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            if path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", (STATIC / "index.html").read_bytes())
            if path == "/api/all":
                fi = _pool.submit(fetch_json, index_url + "/api/state")
                fs = _pool.submit(fetch_json, stocks_url + "/api/state")
                ff = _pool.submit(fetch_json, fno_url + "/api/state")
                (index, ie), (stocks, se), (fno, fe) = fi.result(), fs.result(), ff.result()
                return self._json({"index": index or None, "stocks": stocks or None, "fno": fno_trim(fno),
                                   "errors": {"index": ie or (None if index else "first scan not finished"), "stocks": se,
                                              "fno": fe or (None if fno else "first cycle not finished")},
                                   "links": {"index": index_url, "stocks": stocks_url, "fno": fno_url}})
            if path.startswith("/api/eq/"):
                return self._equity("GET", parsed)
            if path in ASSETS:
                return self._send(200, ASSETS[path], (STATIC / path.rsplit("/", 1)[1]).read_bytes())
            if path == "/api/search":
                q = urllib.parse.parse_qs(parsed.query).get("q", [""])[0][:40]
                data, err = fetch_json(stocks_url + "/api/search?" + urllib.parse.urlencode({"q": q}))
                return self._json(data if err is None else [])
            if path == "/health":
                return self._json({"status": "ok", "index": index_url, "stocks": stocks_url, "fno": fno_url, "equity": equity_url})
            self._send(404, "text/plain", b"not found")

    return Handler


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.desk", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8500)
    p.add_argument("--index-url", default="http://127.0.0.1:8502", help="index scanner (python -m src.main)")
    p.add_argument("--stocks-url", default="http://127.0.0.1:8504", help="stock options service (python -m src.stocks)")
    p.add_argument("--fno-url", default="http://127.0.0.1:8520", help="F&O multi-strategy engine (python -m src.fno)")
    p.add_argument("--equity-url", default="http://127.0.0.1:8530", help="NIFTY 50 cash-equity service (python -m src.equity)")
    p.add_argument("--with-equity", action="store_true", help="also host the NIFTY 50 cash-equity service (API only, under /api/eq/)")
    p.add_argument("--no-notify", action="store_true", help="no desktop popup when a stock or F&O trade triggers")
    p.add_argument("--test-notify", action="store_true", help="show one sample popup (5 s) and exit")
    a = p.parse_args(argv)
    from src.desk.notify import TradeWatcher, popup
    if a.test_notify:
        popup("Trade triggered", "BREAKOUT TRADE OPENED  CE SAMPLE\nEntry 1,000.00 · SL 990.00 · T1 1,015.00 · T2 1,025.00"
              "\n\nThis is a test popup from the trading desk.", 5)
        return 0
    try:
        urls = [u.rstrip("/") for u in (a.index_url, a.stocks_url, a.fno_url, a.equity_url)]
        server = ThreadingHTTPServer((a.host, a.port), make_handler(*urls))
    except OSError as error:
        print(f"Trading desk not started: {a.host}:{a.port} is in use or not allowed ({error}). Try --port 8600.")
        return 1
    print(f"Trading desk: http://{a.host}:{a.port}/   (index {a.index_url} + stocks {a.stocks_url} + F&O {a.fno_url})",
          flush=True)
    if a.with_equity:
        try:
            hosted = start_equity(urls[3])
            where = ("already running" if hosted is None else
                     f"started with the project's .venv on {urls[3]} (this Python lacks its packages), log logs/equity.log"
                     if hasattr(hosted, "terminate") else f"running inside the desk on {urls[3]}, log logs/equity.log")
            print("Equity service (API under /api/eq/): " + where, flush=True)
        except Exception as error:      # noqa: BLE001 - the rest of the desk must still come up
            print(f"Equity service not started ({type(error).__name__}: {error}); run python -m src.equity yourself.", flush=True)
    if not a.no_notify:
        TradeWatcher({"stocks": lambda: fetch_json(urls[1] + "/api/state")[0],
                      "fno": lambda: fetch_json(urls[2] + "/api/state")[0]}).start()
        print("Desktop popup when a stock or F&O trade triggers: on (--no-notify to switch off)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
