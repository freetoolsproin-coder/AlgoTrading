"""
Process-wide single-instance lock (data/scanner.lock). A second `python -m src.main` exits immediately
instead of starting a duplicate scanner, WebSocket connection or dashboard server. The OS releases the
lock automatically if the process dies, so a crash never leaves a stale lock behind.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Optional


def acquire(path: Path) -> Optional[IO]:
    """Open handle holding the lock, or None when another process holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def release(handle: Optional[IO]):
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        handle.close()
