"""
Backtest jobs started from the console. They call the existing backtest commands' main(argv) in a background
thread of the console process (no shell, no new terminal), one job at a time, with validated arguments only.
Output (print and log lines) is captured per job and shown on the Backtests page. Results are the same files the
commands write when run from a terminal (output/*_report.html, output/*_trades.csv).
"""
from __future__ import annotations

import io
import logging
import sys
import threading
import traceback
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

from src.config.instruments import SUPPORTED_INDICES

IST = timezone(timedelta(hours=5, minutes=30))


def _scanner(argv):
    from src.backtest.runner import main
    return main(argv)


def _v3(argv):
    from src.backtest.v3_report import main
    return main(argv)


def _movement(argv):
    from src.movement.backtest import main
    return main(argv)


def _strategies(argv):
    from src.movement.backtest import main
    return main(argv + ["--strategies", "all"])


def _ml(argv):
    from src.scanner.ml_filter import main
    return main(argv)


KINDS: Dict[str, dict] = {
    "scanner": {"label": "Scanner v1 backtest (walk-forward)", "fn": _scanner, "command": "python -m src.backtest"},
    "v3": {"label": "v3 vs v1 research report", "fn": _v3, "command": "python -m src.backtest.v3_report"},
    "movement": {"label": "Movement books (momentum / scalp)", "fn": _movement, "command": "python -m src.movement.backtest"},
    "strategies": {"label": "Strategy library (every strategy, own book each)", "fn": _strategies,
                   "command": "python -m src.movement.backtest --strategies all"},
    "ml": {"label": "Train ML filter (walk-forward)", "fn": _ml, "command": "python -m src.scanner.ml_filter"},
}
BACKTEST_KINDS = ("scanner", "v3", "movement", "strategies")


class JobError(ValueError):
    pass


def build_argv(kind: str, instrument: str, start: str = "", end: str = "") -> List[str]:
    if kind not in KINDS:
        raise JobError("unknown job type")
    if kind == "ml":
        names = [x.strip().upper() for x in (instrument or "").split(",") if x.strip()]
        if not names or any(n not in SUPPORTED_INDICES for n in names) or len(set(names)) != len(names):
            raise JobError(f"instruments must be from {', '.join(SUPPORTED_INDICES)}")
        argv = ["--train", "--instruments", ",".join(names)]
        if end:
            raise JobError("ML training uses all history up to yesterday; leave the end date empty")
        if start:
            try:
                d = date.fromisoformat(start)
            except ValueError as error:
                raise JobError("start must be a date like 2024-01-31") from error
            if d.year < 2015 or d >= date.today():
                raise JobError("start is out of range")
            argv += ["--start", d.isoformat()]
        return argv
    instrument = (instrument or "").upper()
    if instrument not in SUPPORTED_INDICES:
        raise JobError(f"instrument must be one of {', '.join(SUPPORTED_INDICES)}")
    argv = ["--instrument", instrument]
    for flag, value in (("--start", start), ("--end", end)):
        if value:
            try:
                d = date.fromisoformat(value)
            except ValueError as error:
                raise JobError(f"{flag[2:]} must be a date like 2024-01-31") from error
            if d.year < 2015 or d > date.today():
                raise JobError(f"{flag[2:]} is out of range")
            argv += [flag, d.isoformat()]
    if start and end and start > end:
        raise JobError("start must be before end")
    return argv


class _ThreadStdout(io.TextIOBase):
    """Routes writes from registered threads to their buffers; everything else to the real stream."""

    def __init__(self, real):
        self.real = real
        self.buffers: Dict[int, io.StringIO] = {}

    def write(self, s):
        buf = self.buffers.get(threading.get_ident())
        if buf is not None:
            if buf.tell() < 400_000:
                buf.write(s)
            return len(s)
        return self.real.write(s)

    def flush(self):
        try:
            self.real.flush()
        except Exception:   # noqa: BLE001
            pass

    def __getattr__(self, name):
        return getattr(self.real, name)


class _ThreadLogHandler(logging.Handler):
    def __init__(self, ident: int, buf: io.StringIO):
        super().__init__(logging.INFO)
        self.ident, self.buf = ident, buf
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        if record.thread == self.ident and self.buf.tell() < 400_000:
            self.buf.write(self.format(record) + "\n")


class JobRunner:
    def __init__(self, on_finish: Optional[Callable[[dict], None]] = None):
        self.jobs: List[dict] = []
        self._lock = threading.Lock()
        self.on_finish = on_finish

    def running(self) -> Optional[dict]:
        return next((j for j in self.jobs if j["status"] == "running"), None)

    def list(self) -> List[dict]:
        return [{k: v for k, v in j.items() if k != "_buf"} | {"output": j["_buf"].getvalue()[-20000:]}
                for j in reversed(self.jobs[-20:])]

    def start(self, kind: str, instrument: str, start: str = "", end: str = "", actor: str = "") -> dict:
        argv = build_argv(kind, instrument, start, end)
        with self._lock:
            if self.running():
                raise JobError(f"{self.running()['label']} is already running; wait for it to finish")
            job = {"id": uuid.uuid4().hex[:10], "kind": kind, "label": KINDS[kind]["label"],
                   "command": KINDS[kind]["command"] + " " + " ".join(argv), "argv": argv, "actor": actor,
                   "status": "running", "started": datetime.now(IST).isoformat(timespec="seconds"),
                   "finished": None, "exit_code": None, "_buf": io.StringIO()}
            self.jobs.append(job)
        t = threading.Thread(target=self._run, args=(job,), name=f"backtest-{job['id']}", daemon=True)
        t.start()
        return job

    def _run(self, job: dict):
        ident = threading.get_ident()
        if not isinstance(sys.stdout, _ThreadStdout):
            sys.stdout = _ThreadStdout(sys.stdout)
        sys.stdout.buffers[ident] = job["_buf"]
        handler = _ThreadLogHandler(ident, job["_buf"])
        root = logging.getLogger()
        scanner = logging.getLogger("scanner")
        root.addHandler(handler)
        scanner.addHandler(handler)
        try:
            code = KINDS[job["kind"]]["fn"](list(job["argv"]))
            job["exit_code"] = int(code or 0)
            job["status"] = "finished" if not code else "failed"
        except SystemExit as exit_:
            job["exit_code"] = int(exit_.code or 0) if isinstance(exit_.code, int) else 1
            job["status"] = "finished" if job["exit_code"] == 0 else "failed"
        except Exception:   # noqa: BLE001 - shown to the user, never crashes the console
            job["_buf"].write("\n" + traceback.format_exc(limit=8))
            job["status"], job["exit_code"] = "failed", 1
        finally:
            sys.stdout.buffers.pop(ident, None)
            root.removeHandler(handler)
            scanner.removeHandler(handler)
            job["finished"] = datetime.now(IST).isoformat(timespec="seconds")
            if self.on_finish:
                try:
                    self.on_finish({k: v for k, v in job.items() if k != "_buf"})
                except Exception:   # noqa: BLE001
                    pass
