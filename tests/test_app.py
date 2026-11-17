"""
Application lifecycle: single entry point, no duplicate services, dashboard server, Ctrl+C / cancellation
shutdown, logging redaction.

    python -m unittest tests.test_app
"""
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import unittest
import urllib.request

from src.app import Application, RunOptions
from src.config import settings
from src.dashboard.app import DashboardServer
from src.utils import single_instance
from src.utils.logging_config import RedactingFilter

ROOT = settings.PROJECT_ROOT


def run_entry(args, timeout=180, env_extra=None):
    env = dict(os.environ, PYTHONUNBUFFERED="1", **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "src.main", *args], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout, encoding="utf-8", errors="replace")


class EntryPointTests(unittest.TestCase):
    def test_single_command_starts_and_stops(self):
        r = run_entry(["--mock", "--once", "--start", "11:30", "--no-dashboard"])
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-2000:])
        self.assertIn("NIFTY", out)
        self.assertIn("SENSEX", out)
        self.assertIn("Stopped cleanly", out)
        self.assertNotIn("BANKNIFTY", out)
        self.assertTrue(settings.STATE_FILE.exists())

    def test_help_and_bad_config(self):
        self.assertEqual(run_entry(["--help"]).returncode, 0)
        r = run_entry(["--mock", "--once", "--config", "does-not-exist.ini"])
        self.assertEqual(r.returncode, 1)
        self.assertIn("Configuration file not found", r.stdout + r.stderr)

    @unittest.skipIf(os.name == "nt", "POSIX signal test; Windows Ctrl+C is covered by the cancellation test")
    def test_ctrl_c_shuts_down_cleanly(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.Popen([sys.executable, "-m", "src.main", "--mock", "--speed", "30", "--no-dashboard"], cwd=ROOT,
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.time() + 60
            while time.time() < deadline and not settings.STATE_FILE.exists():
                time.sleep(0.5)
            time.sleep(2)
            p.send_signal(signal.SIGINT)
            out, _ = p.communicate(timeout=60)
        finally:
            if p.poll() is None:
                p.kill()
        self.assertEqual(p.returncode, 0, out[-2000:])
        self.assertIn("Stopped cleanly", out)
        handle = single_instance.acquire(settings.LOCK_FILE)   # the lock was released
        self.assertIsNotNone(handle)
        single_instance.release(handle)


class DuplicateServiceTests(unittest.TestCase):
    def test_second_instance_is_refused(self):
        handle = single_instance.acquire(settings.LOCK_FILE)
        self.assertIsNotNone(handle)
        try:
            self.assertIsNone(single_instance.acquire(settings.LOCK_FILE))
            r = run_entry(["--mock", "--once", "--no-dashboard"], timeout=60)
            self.assertEqual(r.returncode, 2)
            self.assertIn("already running", r.stdout + r.stderr)
        finally:
            single_instance.release(handle)

    def test_services_start_once_and_cancellation_shuts_down(self):
        """What Ctrl+C does on Windows (asyncio.run cancels the main task) and what an IDE stop does."""
        async def scenario():
            app = Application(RunOptions(mock=True, speed=60, start="10:00", dashboard=True))
            task = asyncio.create_task(app.run())
            for _ in range(200):
                await asyncio.sleep(0.05)
                if app.engine is not None and app.engine.last_state is not None:
                    break
            self.assertTrue(app._started)
            with self.assertRaises(RuntimeError):
                await app.start()                         # a second start is refused
            self.assertEqual(sorted(app._tasks), ["scanner"])  # mock: no WebSocket task
            self.assertIsNotNone(app.dashboard)
            self.assertTrue(await app.dashboard.start())  # second start: no second server
            port = app.dashboard._server.sockets[0].getsockname()[1]
            self.assertIsNone(single_instance.acquire(settings.LOCK_FILE))   # held while running
            task.cancel()
            code = await asyncio.wait_for(task, timeout=60)
            return app, code, port

        app, code, port = asyncio.run(_with_free_port(scenario))
        self.assertEqual(code, 0)
        self.assertTrue(app._shut_down)
        self.assertFalse(app.notifier._worker.is_alive())
        self.assertIsNone(app.dashboard._server)
        handle = single_instance.acquire(settings.LOCK_FILE)   # released after shutdown
        self.assertIsNotNone(handle)
        single_instance.release(handle)


async def _with_free_port(scenario):
    """Run the scenario with the dashboard on a free port (never the real 8502)."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    original = settings.load_app_settings

    def patched(path=settings.CONFIG_FILE):
        a = original(path)
        a.dashboard_port = port
        return a
    settings.load_app_settings = patched
    try:
        return await scenario()
    finally:
        settings.load_app_settings = original


class DashboardTests(unittest.TestCase):
    def test_routes(self):
        async def scenario():
            state = {"generated_at": "2026-10-16T11:00:00+05:30", "mode": "MOCK", "mock": True, "status": "Market open",
                     "risk": {}, "indices": {}, "thresholds": {}}
            srv = DashboardServer(lambda: state, "127.0.0.1", 0)
            self.assertTrue(await srv.start())
            port = srv._server.sockets[0].getsockname()[1]

            def fetch(path):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
                    return r.status, r.read().decode("utf-8")
            html = await asyncio.to_thread(fetch, "/")
            api = await asyncio.to_thread(fetch, "/api/state")
            health = await asyncio.to_thread(fetch, "/health")
            await srv.stop()
            return html, api, health
        (s1, html), (s2, api), (s3, health) = asyncio.run(scenario())
        self.assertEqual((s1, s2, s3), (200, 200, 200))
        self.assertIn("NIFTY &amp; SENSEX", html)
        self.assertEqual(json.loads(api)["mode"], "MOCK")
        self.assertEqual(json.loads(health)["status"], "ok")

    def test_port_in_use_is_reported_not_fatal(self):
        async def scenario():
            a = DashboardServer(lambda: None, "127.0.0.1", 0)
            await a.start()
            port = a._server.sockets[0].getsockname()[1]
            b = DashboardServer(lambda: None, "127.0.0.1", port)
            ok = await b.start()
            await a.stop()
            return ok
        self.assertFalse(asyncio.run(scenario()))


class LoggingTests(unittest.TestCase):
    def test_tokens_are_redacted(self):
        record = logging.LogRecord("scanner.x", logging.INFO, __file__, 1,
                                   "auth Bearer abcdefghijklmnop1234 and bot 123456789:AAEabcdefghijklmnopqrstuvwxyz012345", None, None)
        RedactingFilter().filter(record)
        msg = record.getMessage()
        self.assertNotIn("abcdefghijklmnop1234", msg)
        self.assertNotIn("AAEabcdefghijklmnopqrstuvwxyz012345", msg)


class ProjectFilesTests(unittest.TestCase):
    def test_vscode_launch_uses_the_module_entry_point(self):
        text = (ROOT / "ide" / "vscode" / "launch.json").read_text(encoding="utf-8")
        launch = json.loads("\n".join(l for l in text.splitlines() if not l.strip().startswith("//")))
        scanner = [c for c in launch["configurations"] if c.get("module") == "src.main"]
        self.assertTrue(scanner)
        self.assertEqual(scanner[0]["cwd"], "${workspaceFolder}")
        self.assertTrue(all("program" not in c for c in launch["configurations"]))

    def test_no_batch_launchers_in_the_application(self):
        self.assertEqual(list((ROOT / "src").rglob("*.bat")), [])
        self.assertNotIn("start_all", (ROOT / "ide" / "vscode" / "tasks.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
