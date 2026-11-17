"""Movement events in SQLite (data/movement.db): one row per trigger, the full event as JSON."""
from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from src.config import settings
from src.scanner.store import json_default

MOVEMENT_DB = settings.DATA_DIR / "movement.db"


class EventStore:
    def __init__(self, path: Path = MOVEMENT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS movement_events (id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, "
                      "day TEXT, idx TEXT, window TEXT, direction TEXT, quality TEXT, change_pct REAL, "
                      "mock INTEGER, payload TEXT)")
            c.execute("CREATE INDEX IF NOT EXISTS movement_events_day ON movement_events(day)")
            c.execute("CREATE TABLE IF NOT EXISTS movement_trades (id TEXT PRIMARY KEY, day TEXT, book TEXT, idx TEXT, "
                      "side TEXT, open INTEGER, net REAL, mock INTEGER, payload TEXT)")
            c.execute("CREATE INDEX IF NOT EXISTS movement_trades_day ON movement_trades(day)")

    @contextlib.contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.path), timeout=20)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def add(self, e: dict) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO movement_events (time, day, idx, window, direction, quality, change_pct, mock, "
                            "payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (e["time"], e["time"][:10], e["index"], e["window"], e["direction"], e["quality"],
                             e.get("change_pct"), int(bool(e.get("mock"))), json.dumps(e, default=json_default)))
            return int(cur.lastrowid)

    def recent(self, limit: int = 200, mock: Optional[bool] = None, day: Optional[str] = None,
               index: Optional[str] = None) -> List[dict]:
        sql, args = "SELECT id, payload FROM movement_events WHERE 1=1", []
        if mock is not None:
            sql += " AND mock = ?"
            args.append(int(mock))
        if day:
            sql += " AND day = ?"
            args.append(day)
        if index:
            sql += " AND idx = ?"
            args.append(index.upper())
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [dict(json.loads(payload), id=row_id) for row_id, payload in rows]

    def save_trade(self, t: dict):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO movement_trades (id, day, book, idx, side, open, net, mock, payload) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (t["id"], t["entry_time"][:10], t["book"], t["index"], t["side"], int(bool(t["open"])),
                       t.get("net"), int(bool(t.get("mock"))), json.dumps(t, default=json_default)))

    def trades(self, day: Optional[str] = None, mock: Optional[bool] = None, book: Optional[str] = None,
               limit: int = 1000) -> List[dict]:
        sql, args = "SELECT payload FROM movement_trades WHERE 1=1", []
        for column, value in (("day", day), ("book", book)):
            if value:
                sql += f" AND {column} = ?"
                args.append(value)
        if mock is not None:
            sql += " AND mock = ?"
            args.append(int(mock))
        sql += " ORDER BY day DESC, id DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as c:
            return [json.loads(row[0]) for row in c.execute(sql, args).fetchall()]
