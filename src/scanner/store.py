"""
SQLite storage (data/prebreakout.db): alerts, paper trades, per-bar scores and ML training samples.
Older databases are migrated in place (columns added). WAL mode, one short connection per call.
"""
from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config.settings import DB_FILE
from src.execution import paper_trading as paper_mod
from src.scanner import strategies as strat
from src.utils import numbers as num


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


class Store:
    def __init__(self, path=DB_FILE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, day TEXT, "
                      "idx TEXT, direction TEXT, stage TEXT, kind TEXT, score REAL, price REAL, level REAL, "
                      "setup_key TEXT, mock INTEGER, payload TEXT)")
            c.execute("CREATE INDEX IF NOT EXISTS alerts_day ON alerts(day)")
            c.execute("CREATE TABLE IF NOT EXISTS paper_trades (id TEXT PRIMARY KEY, day TEXT, idx TEXT, direction TEXT, "
                      "symbol TEXT, entry_time TEXT, exit_time TEXT, entry_price REAL, qty INTEGER, net_pnl REAL, "
                      "exit_reason TEXT, closed INTEGER, mock INTEGER, payload TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS bar_scores (time TEXT, idx TEXT, close REAL, atr REAL, ce_score REAL, "
                      "pe_score REAL, ce_level REAL, pe_level REAL, bias TEXT, mock INTEGER, PRIMARY KEY(time, idx, mock))")
            c.execute("CREATE TABLE IF NOT EXISTS ml_samples (id TEXT PRIMARY KEY, time TEXT, idx TEXT, direction TEXT, "
                      "entry REAL, stop REAL, features TEXT, y INTEGER, r REAL, mock INTEGER)")
            c.execute("CREATE TABLE IF NOT EXISTS iv_daily (day TEXT, idx TEXT, atm_iv REAL, mock INTEGER, "
                      "PRIMARY KEY(day, idx, mock))")
            for table in ("alerts", "paper_trades"):  # older databases: add the model column
                cols = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
                if "model" not in cols:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN model TEXT DEFAULT 'v1'")

    @contextlib.contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.path), timeout=20)
        try:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            with conn:  # commit / rollback
                yield conn
        finally:
            conn.close()

    def add_alert(self, a: dict):
        with self._conn() as c:
            c.execute("INSERT INTO alerts (time, day, idx, direction, stage, kind, score, price, level, setup_key, mock, "
                      "payload, model) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (a.get("time"), str(a.get("time", ""))[:10], a.get("index"), a.get("direction"), a.get("stage"),
                       a.get("kind"), a.get("score"), a.get("price"), a.get("level"), a.get("setup_key"),
                       int(bool(a.get("mock"))), json.dumps(a, default=json_default), a.get("model") or "v1"))

    def sent_today(self, day: date, mock: bool) -> set:
        with self._conn() as c:
            rows = c.execute("SELECT setup_key, kind FROM alerts WHERE day = ? AND mock = ?", (day.isoformat(), int(mock))).fetchall()
        return {(r[0], r[1]) for r in rows if r[0]}

    def recent_alerts(self, limit: int = 200, mock: Optional[bool] = None) -> List[dict]:
        q = "SELECT payload FROM alerts"
        args = ()
        if mock is not None:
            q += " WHERE mock = ?"
            args = (int(mock),)
        q += " ORDER BY id DESC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(q, args + (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def save_trade(self, p: paper_mod.PaperPosition):
        d = p.to_dict()
        model = trade_model(p.setup_key)
        d["model"] = model
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO paper_trades (id, day, idx, direction, symbol, entry_time, exit_time, "
                      "entry_price, qty, net_pnl, exit_reason, closed, mock, payload, model) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (p.id, p.entry_time[:10], p.index, p.direction, p.symbol, p.entry_time, p.exit_time, p.entry_price,
                       p.lots * p.lot_size, d["net_pnl"], p.exit_reason, int(p.closed), int(p.mock),
                       json.dumps(d, default=json_default), model))

    def trades(self, mock: Optional[bool] = None, day: Optional[date] = None, model: Optional[str] = None) -> List[dict]:
        q, args = "SELECT payload FROM paper_trades WHERE 1=1", []
        if model is not None:
            q += " AND COALESCE(model, 'v1') = ?"
            args.append(model)
        if mock is not None:
            q += " AND mock = ?"
            args.append(int(mock))
        if day is not None:
            q += " AND day = ?"
            args.append(day.isoformat())
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY entry_time", args).fetchall()
        return [json.loads(r[0]) for r in rows]

    def add_ml_sample(self, sid, ev: strat.BarEval, direction, entry, stop, feats: dict, mock: bool):
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO ml_samples (id, time, idx, direction, entry, stop, features, mock) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (sid, ev.close_time.isoformat(), ev.index, direction, entry, stop,
                       json.dumps({k: (None if not np.isfinite(v) else v) for k, v in feats.items()}), int(mock)))

    def pending_ml_samples(self, idx: str, mock: bool) -> List[tuple]:
        with self._conn() as c:
            return c.execute("SELECT id, time, direction, entry, stop FROM ml_samples WHERE y IS NULL AND idx = ? AND mock = ?",
                             (idx, int(mock))).fetchall()

    def label_ml_sample(self, sid, y: int, r: float):
        with self._conn() as c:
            c.execute("UPDATE ml_samples SET y = ?, r = ? WHERE id = ?", (int(y), float(r), sid))

    def set_iv_daily(self, day: date, idx: str, atm_iv: float, mock: bool):
        """Latest ATM IV of the day (overwritten each cycle, so the last value of the session stays)."""
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO iv_daily VALUES (?,?,?,?)", (day.isoformat(), idx, float(atm_iv), int(mock)))

    def iv_daily(self, idx: str, before: date, mock: bool, limit: int = 252) -> List[float]:
        """ATM IV of earlier sessions (strictly before `before`), most recent `limit`."""
        with self._conn() as c:
            rows = c.execute("SELECT atm_iv FROM iv_daily WHERE idx = ? AND mock = ? AND day < ? ORDER BY day DESC LIMIT ?",
                             (idx, int(mock), before.isoformat(), limit)).fetchall()
        return [float(r[0]) for r in rows if r[0] is not None]

    def add_bar(self, ev: strat.BarEval, mock: bool):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO bar_scores VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (ev.close_time.isoformat(), ev.index, ev.close, ev.atr, ev.ce.score, ev.pe.score,
                       num.r2(ev.ce.level_price), num.r2(ev.pe.level_price), ev.bias, int(mock)))


def trade_model(setup_key: str) -> str:
    """'v2|2026-10-19|NIFTY|CE|24050' -> 'v2'; keys without a prefix belong to the main model v1."""
    head = str(setup_key or "").split("|", 1)[0]
    return head if head and not head[:1].isdigit() else "v1"
