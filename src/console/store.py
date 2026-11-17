"""
The console's own database (data/console/console.db): users, login attempts and the audit log.

The audit log is append-only and hash-chained: every row stores sha256(previous hash + its own canonical
content), so editing or deleting a row breaks the chain and `verify_chain()` reports the first broken id.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

IST = timezone(timedelta(hours=5, minutes=30))
GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','viewer')),
    totp_secret TEXT, disabled INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL, password_changed TEXT,
    last_login TEXT, risk_ack TEXT);
CREATE TABLE IF NOT EXISTS login_attempts (username TEXT, ip TEXT, time TEXT, ok INTEGER);
CREATE INDEX IF NOT EXISTS login_attempts_user ON login_attempts(username, time);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, actor TEXT NOT NULL, ip TEXT, action TEXT NOT NULL,
    target TEXT, outcome TEXT NOT NULL, detail TEXT, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""


def now_iso() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def _row_hash(prev: str, time_: str, actor: str, ip: str, action: str, target: str, outcome: str, detail: str) -> str:
    canonical = json.dumps([prev, time_, actor, ip, action, target, outcome, detail], separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConsoleStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as c:
            c.executescript(SCHEMA)

    @contextlib.contextmanager
    def _connect(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode=WAL")
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    # ------------------------------------------------------------------ users
    def user(self, username: str) -> Optional[dict]:
        with self._connect() as c:
            r = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(r) if r else None

    def users(self) -> List[dict]:
        with self._connect() as c:
            return [dict(r) for r in c.execute("SELECT username, role, disabled, created, last_login, "
                                               "totp_secret IS NOT NULL AS has_2fa, risk_ack FROM users ORDER BY username")]

    def count_users(self) -> int:
        with self._connect() as c:
            return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def create_user(self, username: str, password_hash: str, role: str):
        with self._lock, self._connect() as c:
            c.execute("INSERT INTO users(username,password_hash,role,created,password_changed) VALUES(?,?,?,?,?)",
                      (username, password_hash, role, now_iso(), now_iso()))

    def update_user(self, username: str, **fields):
        allowed = {"password_hash", "role", "totp_secret", "disabled", "password_changed", "last_login", "risk_ack"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        with self._lock, self._connect() as c:
            c.execute(f"UPDATE users SET {', '.join(k + '=?' for k in sets)} WHERE username=?",
                      (*sets.values(), username))

    # ------------------------------------------------------------------ login throttling
    def record_attempt(self, username: str, ip: str, ok: bool):
        with self._lock, self._connect() as c:
            c.execute("INSERT INTO login_attempts VALUES(?,?,?,?)", (username, ip, now_iso(), int(ok)))

    def recent_failures(self, username: str, minutes: int) -> int:
        """Consecutive failures within the window (a success resets the count)."""
        since = (datetime.now(IST) - timedelta(minutes=minutes)).isoformat(timespec="seconds")
        with self._connect() as c:
            rows = c.execute("SELECT ok FROM login_attempts WHERE username=? AND time>=? ORDER BY time DESC, rowid DESC",
                             (username, since)).fetchall()
        n = 0
        for r in rows:
            if r["ok"]:
                break
            n += 1
        return n

    def recent_ip_failures(self, ip: str, minutes: int) -> int:
        since = (datetime.now(IST) - timedelta(minutes=minutes)).isoformat(timespec="seconds")
        with self._connect() as c:
            return c.execute("SELECT COUNT(*) FROM login_attempts WHERE ip=? AND ok=0 AND time>=?",
                             (ip, since)).fetchone()[0]

    # ------------------------------------------------------------------ audit
    def audit(self, actor: str, action: str, *, target: str = "", outcome: str = "ok", detail=None, ip: str = "") -> int:
        detail_s = json.dumps(detail, ensure_ascii=False, sort_keys=True, default=str) if detail is not None else ""
        with self._lock, self._connect() as c:
            last = c.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = last["hash"] if last else GENESIS
            t = now_iso()
            h = _row_hash(prev, t, actor, ip or "", action, target or "", outcome, detail_s)
            cur = c.execute("INSERT INTO audit(time,actor,ip,action,target,outcome,detail,prev_hash,hash) "
                            "VALUES(?,?,?,?,?,?,?,?,?)", (t, actor, ip or "", action, target or "", outcome, detail_s, prev, h))
            return cur.lastrowid

    def audit_rows(self, limit: int = 200, offset: int = 0, action: str = "", actor: str = "") -> List[dict]:
        q, args = "SELECT * FROM audit WHERE 1=1", []
        if action:
            q += " AND action LIKE ?"
            args.append(action + "%")
        if actor:
            q += " AND actor=?"
            args.append(actor)
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        with self._connect() as c:
            return [dict(r) for r in c.execute(q, args)]

    def verify_chain(self) -> dict:
        prev, n = GENESIS, 0
        with self._connect() as c:
            for r in c.execute("SELECT * FROM audit ORDER BY id"):
                n += 1
                expect = _row_hash(prev, r["time"], r["actor"], r["ip"] or "", r["action"], r["target"] or "",
                                   r["outcome"], r["detail"] or "")
                if r["prev_hash"] != prev or r["hash"] != expect:
                    return {"ok": False, "rows": n, "broken_at": r["id"]}
                prev = r["hash"]
        return {"ok": True, "rows": n, "broken_at": None, "head": prev}
