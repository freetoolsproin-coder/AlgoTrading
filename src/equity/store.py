"""
SQLite store of the equity service (data/equity/equity.db).

    signals     every signal ever generated (id is unique per stock / strategy / side / bar: no duplicates)
    positions   paper and live positions, one per taken signal
    orders      every order: proposed, confirmed, sent, filled ... (tag is unique: the idempotency key)
    fills       executed quantity with charges and gross P&L (the day's realised P&L is the sum of its fills)
    alerts      what was shown / pushed
    backtests   one row per backtest run with its full result
    kv          run-time settings (mode, kill switch, risk overrides, validated strategies)

Columns that are searched are real columns; everything else of a record lives in its JSON `data` column.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals(id TEXT PRIMARY KEY, module TEXT, day TEXT, symbol TEXT, side TEXT, strategy TEXT,
    time TEXT, status TEXT, note TEXT DEFAULT '', data TEXT);
CREATE INDEX IF NOT EXISTS ix_signals_day ON signals(module, day);
CREATE TABLE IF NOT EXISTS positions(id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id TEXT UNIQUE, module TEXT, mode TEXT,
    symbol TEXT, key TEXT, side TEXT, strategy TEXT, qty INTEGER, qty_open INTEGER, entry REAL, stop REAL, t1 REAL, t2 REAL,
    status TEXT, created TEXT, opened TEXT, closed TEXT, exit_price REAL, exit_reason TEXT, gross REAL DEFAULT 0,
    charges REAL DEFAULT 0, data TEXT);
CREATE INDEX IF NOT EXISTS ix_positions_status ON positions(status);
CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY AUTOINCREMENT, position_id INTEGER, tag TEXT UNIQUE, mode TEXT,
    purpose TEXT, symbol TEXT, key TEXT, side TEXT, qty INTEGER, order_type TEXT, price REAL, trigger REAL, product TEXT,
    status TEXT, broker_id TEXT, avg_price REAL, filled_qty INTEGER DEFAULT 0, message TEXT DEFAULT '', created TEXT,
    updated TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(status);
CREATE TABLE IF NOT EXISTS fills(id INTEGER PRIMARY KEY AUTOINCREMENT, position_id INTEGER, time TEXT, day TEXT, side TEXT,
    qty INTEGER, price REAL, charges REAL, pnl REAL, reason TEXT);
CREATE INDEX IF NOT EXISTS ix_fills_day ON fills(day);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, day TEXT, kind TEXT, module TEXT,
    symbol TEXT, text TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS backtests(id INTEGER PRIMARY KEY AUTOINCREMENT, module TEXT, created TEXT, params TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
"""
JSON_COLS = ("data", "params", "result")


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in JSON_COLS:
        if k in d:
            try:
                d[k] = json.loads(d[k]) if d[k] else {}
            except ValueError:
                d[k] = {}
    return d


class Store:
    def __init__(self, path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            if self.path != ":memory:":
                self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)

    # ------------------------------------------------------------------ generic
    def rows(self, sql: str, args: Iterable = ()) -> List[dict]:
        with self.lock:
            return [_row(r) for r in self.db.execute(sql, tuple(args)).fetchall()]

    def one(self, sql: str, args: Iterable = ()) -> Optional[dict]:
        r = self.rows(sql, args)
        return r[0] if r else None

    def insert(self, table: str, **values) -> int:
        cols = list(values)
        vals = [json.dumps(values[c], default=str) if c in JSON_COLS else values[c] for c in cols]
        with self.lock:
            cur = self.db.execute(f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)
            return cur.lastrowid

    def update(self, table: str, row_id, **values) -> None:
        if not values:
            return
        cols = list(values)
        vals = [json.dumps(values[c], default=str) if c in JSON_COLS else values[c] for c in cols]
        with self.lock:
            self.db.execute(f"UPDATE {table} SET {','.join(c + '=?' for c in cols)} WHERE id=?", vals + [row_id])

    # ------------------------------------------------------------------ kv
    def get(self, key: str, default: Any = None) -> Any:
        r = self.one("SELECT value FROM kv WHERE key=?", (key,))
        if r is None:
            return default
        try:
            return json.loads(r["value"])
        except ValueError:
            return default

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self.db.execute("INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                            (key, json.dumps(value, default=str)))

    # ------------------------------------------------------------------ signals
    def add_signal(self, sig: dict, status: str, note: str = "") -> bool:
        """False when the signal is already stored (same stock, strategy, side and bar)."""
        with self.lock:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO signals(id, module, day, symbol, side, strategy, time, status, note, data) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (sig["id"], sig["module"], sig["day"], sig["symbol"], sig["side"], sig["strategy"], sig["time"], status, note,
                 json.dumps(sig, default=str)))
            return cur.rowcount > 0

    def signal_status(self, sig_id: str, status: str, note: str = "") -> None:
        with self.lock:
            self.db.execute("UPDATE signals SET status=?, note=? WHERE id=?", (status, note, sig_id))

    def signals(self, module: str, day: Optional[str] = None, since: Optional[str] = None) -> List[dict]:
        sql, args = "SELECT * FROM signals WHERE module=?", [module]
        if day:
            sql, args = sql + " AND day=?", args + [day]
        if since:
            sql, args = sql + " AND day>=?", args + [since]
        out = []
        for r in self.rows(sql + " ORDER BY time", args):
            d = r["data"]
            d.update(status=r["status"], note=r["note"])
            out.append(d)
        return out

    # ------------------------------------------------------------------ positions / orders / fills
    def positions(self, statuses: Optional[Iterable[str]] = None, module: Optional[str] = None, since_day: Optional[str] = None) -> List[dict]:
        sql, args = "SELECT * FROM positions WHERE 1=1", []
        if statuses:
            statuses = list(statuses)
            sql, args = sql + f" AND status IN ({','.join('?' * len(statuses))})", args + statuses
        if module:
            sql, args = sql + " AND module=?", args + [module]
        if since_day:
            sql, args = sql + " AND substr(coalesce(closed, created), 1, 10)>=?", args + [since_day]
        return self.rows(sql + " ORDER BY id", args)

    def position(self, pid: int) -> Optional[dict]:
        return self.one("SELECT * FROM positions WHERE id=?", (pid,))

    def orders(self, statuses: Optional[Iterable[str]] = None, position_id: Optional[int] = None, day: Optional[str] = None) -> List[dict]:
        sql, args = "SELECT * FROM orders WHERE 1=1", []
        if statuses:
            statuses = list(statuses)
            sql, args = sql + f" AND status IN ({','.join('?' * len(statuses))})", args + statuses
        if position_id is not None:
            sql, args = sql + " AND position_id=?", args + [position_id]
        if day:
            sql, args = sql + " AND substr(created, 1, 10)=?", args + [day]
        return self.rows(sql + " ORDER BY id", args)

    def order(self, oid: int) -> Optional[dict]:
        return self.one("SELECT * FROM orders WHERE id=?", (oid,))

    def fills(self, day: Optional[str] = None, position_id: Optional[int] = None) -> List[dict]:
        sql, args = "SELECT * FROM fills WHERE 1=1", []
        if day:
            sql, args = sql + " AND day=?", args + [day]
        if position_id is not None:
            sql, args = sql + " AND position_id=?", args + [position_id]
        return self.rows(sql + " ORDER BY id", args)

    # ------------------------------------------------------------------ alerts / backtests
    def alerts(self, day: Optional[str] = None, limit: int = 200) -> List[dict]:
        if day:
            return self.rows("SELECT * FROM alerts WHERE day=? ORDER BY id DESC LIMIT ?", (day, limit))
        return self.rows("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))

    def latest_backtest(self, module: str) -> Optional[dict]:
        return self.one("SELECT * FROM backtests WHERE module=? ORDER BY id DESC LIMIT 1", (module,))

    def close(self):
        with self.lock:
            self.db.close()
