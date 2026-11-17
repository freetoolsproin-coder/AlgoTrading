"""
SQLite storage of the F&O engine (WAL mode): signals, paper trades and option-chain snapshots.

    data/fno/fno.db        live market data (paper trades)
    data/fno/fno_mock.db   --mock runs: synthetic data never mixes with real results

`chain_series` turns recorded snapshots into the option-chain history that strategies 34 and 35 read:
premium momentum of the at-the-money strike (same strike, `lookback` minutes apart) and the change of
near-the-money call / put open interest. These are recorded values, never modelled ones.
"""
from __future__ import annotations

import bisect
import json
import math
import sqlite3
import threading
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.fno.config import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, day TEXT, symbol TEXT,
    strategy TEXT, side TEXT, status TEXT, confidence REAL, payload TEXT);
CREATE INDEX IF NOT EXISTS signals_day ON signals(day);
CREATE TABLE IF NOT EXISTS trades (id TEXT PRIMARY KEY, day TEXT, symbol TEXT, strategy TEXT, side TEXT,
    status TEXT, opened TEXT, closed TEXT, net REAL, r_multiple REAL, payload TEXT);
CREATE INDEX IF NOT EXISTS trades_day ON trades(day);
CREATE TABLE IF NOT EXISTS chain (ts TEXT, symbol TEXT, payload TEXT, PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


def _clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, date)):
        return o.isoformat()
    return o


def dumps(o) -> str:
    return json.dumps(_clean(o), default=str)


class Store:
    def __init__(self, path: Optional[Path] = None, mock: bool = False):
        self.path = Path(path) if path else data_dir() / ("fno_mock.db" if mock else "fno.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self):
        with self._lock:
            self._db.close()

    def _exec(self, sql, args=()):
        with self._lock:
            self._db.execute(sql, args)
            self._db.commit()

    def _rows(self, sql, args=()):
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # ---- signals
    def add_signal(self, s: dict):
        ts = str(s.get("timestamp") or s.get("bar_time"))
        self._exec("INSERT INTO signals (ts, day, symbol, strategy, side, status, confidence, payload) VALUES (?,?,?,?,?,?,?,?)",
                   (ts, ts[:10], s.get("symbol"), s.get("strategy"), s.get("side"), s.get("status"),
                    float(s.get("confidence") or 0), dumps(s)))

    def signals(self, day: Optional[str] = None, limit: int = 300) -> List[dict]:
        if day:
            rows = self._rows("SELECT payload FROM signals WHERE day = ? ORDER BY id DESC LIMIT ?", (day, limit))
        else:
            rows = self._rows("SELECT payload FROM signals ORDER BY id DESC LIMIT ?", (limit,))
        return [json.loads(r[0]) for r in rows]

    def signal_counts(self) -> Dict[str, int]:
        return {k: n for k, n in self._rows("SELECT strategy, COUNT(*) FROM signals GROUP BY strategy")}

    # ---- trades
    def save_trade(self, p):
        d = p.to_dict()
        self._exec("INSERT OR REPLACE INTO trades (id, day, symbol, strategy, side, status, opened, closed, net, r_multiple, payload) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (p.id, str(p.opened)[:10], p.symbol, p.strategy, p.side, p.status, str(p.opened), str(p.closed),
                    d["net"], d["r_multiple"], dumps(d)))

    def trades(self, status: Optional[str] = None, day: Optional[str] = None, limit: int = 5000) -> List[dict]:
        sql, args = "SELECT payload FROM trades WHERE 1=1", []
        if status:
            sql, args = sql + " AND status = ?", args + [status]
        if day:
            sql, args = sql + " AND day = ?", args + [day]
        return [json.loads(r[0]) for r in self._rows(sql + " ORDER BY opened DESC LIMIT ?", args + [limit])]

    # ---- chain snapshots
    def add_chain(self, ts: pd.Timestamp, symbol: str, payload: dict):
        self._exec("INSERT OR REPLACE INTO chain (ts, symbol, payload) VALUES (?,?,?)", (ts.isoformat(), symbol, dumps(payload)))

    def chain_snapshots(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> List[tuple]:
        sql, args = "SELECT ts, payload FROM chain WHERE symbol = ?", [symbol]
        if start:
            sql, args = sql + " AND ts >= ?", args + [start]
        if end:
            sql, args = sql + " AND ts <= ?", args + [end]
        return [(pd.Timestamp(ts).tz_convert("Asia/Kolkata"), json.loads(p)) for ts, p in self._rows(sql + " ORDER BY ts", args)]

    def iv_history(self, symbol: str, before: str, days: int = 30) -> List[float]:
        """ATM IV readings of the `days` days before `before` (ISO time), one per 10 minutes: the history the IV
        percentile is measured against."""
        start = (pd.Timestamp(before) - pd.Timedelta(days=days)).isoformat()
        rows = self._rows("SELECT MIN(ts), payload FROM chain WHERE symbol = ? AND ts >= ? AND ts < ? "
                          "GROUP BY substr(ts, 1, 15) ORDER BY 1", [symbol, start, before])
        out = []
        for _, payload in rows:
            try:
                iv = atm_iv_of(json.loads(payload))
            except (ValueError, TypeError):
                continue
            if np.isfinite(iv):
                out.append(iv)
        return out

    def prune_chain(self, keep_days: int = 120):
        cut = (pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(days=keep_days)).isoformat()
        self._exec("DELETE FROM chain WHERE ts < ?", (cut,))

    # ---- key / value
    def put(self, key: str, value):
        self._exec("INSERT OR REPLACE INTO kv (key, value) VALUES (?,?)", (key, dumps(value)))

    def get(self, key: str, default=None):
        rows = self._rows("SELECT value FROM kv WHERE key = ?", (key,))
        return json.loads(rows[0][0]) if rows else default


def _spread(bid, ask):
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(bid) and np.isfinite(ask)) or bid <= 0 or ask < bid:
        return None
    return round((ask - bid) / ((ask + bid) / 2.0) * 100.0, 3)


def chain_payload(chain: pd.DataFrame, spot: float, expiry, width: int = 10, today=None) -> Optional[dict]:
    """Compact snapshot of ATM +- `width` strikes. Per strike:
    [ce_ltp, pe_ltp, ce_oi, pe_oi, ce_volume, pe_volume, ce_iv, pe_iv, ce_spread_pct, pe_spread_pct]."""
    if chain is None or chain.empty or not np.isfinite(spot):
        return None
    strikes = chain["strike"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(strikes - spot)))
    part = chain.iloc[max(0, i - width):i + width + 1]
    rows = {f"{r.strike:g}": [r.ce_ltp, r.pe_ltp, r.ce_oi, r.pe_oi, r.ce_volume, r.pe_volume, r.ce_iv, r.pe_iv,
                              _spread(r.ce_bid, r.ce_ask), _spread(r.pe_bid, r.pe_ask)] for r in part.itertuples()}
    dte = None
    if today is not None:
        try:
            dte = (date.fromisoformat(str(expiry)[:10]) - today).days
        except ValueError:
            dte = None
    return {"spot": float(spot), "atm": float(strikes[i]), "expiry": str(expiry), "dte": dte, "rows": rows}


CHAIN_COLUMNS = ["ce_mom_pct", "pe_mom_pct", "ce_oi_chg_pct", "pe_oi_chg_pct", "pcr", "pcr_chg", "pcr_vol", "support",
                 "resistance", "atm_iv", "iv_chg", "atm_spread_pct", "atm_vol_share", "flow_bias", "flow_pct", "ce_vol_chg",
                 "pe_vol_chg", "ce_vol_pct", "pe_vol_pct", "iv_pctile", "bull_score", "bear_score", "dte", "spot"]

IV_PCTILE_MIN_READINGS = 30


def _v(row, i):
    """Field i of a snapshot row as a float (NaN when missing: old snapshots have 4 fields)."""
    try:
        v = row[i]
        return float(v) if v is not None else np.nan
    except (IndexError, TypeError, ValueError):
        return np.nan


def _walls(rows: dict, atm: float):
    """(support, resistance): the strike with the most put OI at or below the money and the strike with the
    most call OI at or above it. Writers defend these strikes, which is why they tend to act as levels."""
    sup = res = np.nan
    best_p = best_c = 0.0
    for k, r in rows.items():
        s = float(k)
        if s <= atm and _v(r, 3) > best_p:
            best_p, sup = _v(r, 3), s
        if s >= atm and _v(r, 2) > best_c:
            best_c, res = _v(r, 2), s
    return sup, res


def _flows(rows: dict, old: dict, atm: float, span: int = 3):
    """OI change near the money, split by what the PRICE did, because OI alone has no direction:
        call OI up + call price up   long build-up     bullish     put OI up + put price up     long build-up    bearish
        call OI up + call price down short build-up    bearish     put OI up + put price down   put writing      bullish
        call OI down + price up      short covering    bullish     put OI down + put price up   short covering   bearish
        call OI down + price down    long unwinding    bearish     put OI down + put price down long unwinding   bullish
    Returns (bullish OI change, bearish OI change, near-the-money OI)."""
    keys = sorted((k for k in rows if k in old), key=lambda k: abs(float(k) - atm))[:2 * span + 1]
    bull = bear = total = 0.0
    for k in keys:
        a, b = rows[k], old[k]
        for px, oi, call in ((0, 2, True), (1, 3, False)):
            d_oi, d_px = _v(a, oi) - _v(b, oi), _v(a, px) - _v(b, px)
            if np.isfinite(_v(a, oi)):
                total += _v(a, oi)
            if not (np.isfinite(d_oi) and np.isfinite(d_px)) or d_oi == 0 or d_px == 0:
                continue
            bullish = (d_px > 0) if call else (d_px < 0)
            if bullish:
                bull += abs(d_oi)
            else:
                bear += abs(d_oi)
    return bull, bear, total


def _scores(r: dict):
    """Bullish / bearish option-chain evidence, 0-100. Each part needs its own data; missing data scores nothing."""
    def ok(x):
        return x is not None and np.isfinite(x)
    bull = bear = 0.0
    bias, act = r["flow_bias"], r["flow_pct"]
    if ok(bias) and ok(act) and act >= 0.3:                       # positions actually changed
        bull += 30.0 * min(1.0, max(bias, 0.0) / 0.6)
        bear += 30.0 * min(1.0, max(-bias, 0.0) / 0.6)
    ce, pe = r["ce_mom_pct"], r["pe_mom_pct"]
    if ok(ce) and ok(pe):                                         # premium movement
        bull += 20.0 if ce > 0 and pe < 0 else 0.0
        bear += 20.0 if pe > 0 and ce < 0 else 0.0
    if ok(r["pcr_chg"]):                                          # PCR rising = puts being added (written into a rally)
        bull += 10.0 if r["pcr_chg"] > 0.01 else 0.0
        bear += 10.0 if r["pcr_chg"] < -0.01 else 0.0
    if ok(r["iv_chg"]):                                           # IV usually eases on rallies and rises on falls
        bull += 10.0 if r["iv_chg"] <= 0.25 else 0.0
        bear += 10.0 if r["iv_chg"] >= 0.0 else 0.0
    cv, pv = r["ce_vol_chg"], r["pe_vol_chg"]
    if ok(cv) and ok(pv) and ok(ce) and ok(pe) and (cv + pv) > 0:  # where the trading went, and at what price
        bull += 10.0 if cv > pv and ce > 0 else 0.0
        bear += 10.0 if pv > cv and pe > 0 else 0.0
    spot = r["spot"]
    if ok(spot) and spot > 0:                                     # room before the opposing OI wall
        if ok(r["resistance"]):
            bull += 15.0 if (r["resistance"] - spot) / spot >= 0.0015 or spot > r["resistance"] else 0.0
        if ok(r["support"]):
            bear += 15.0 if (spot - r["support"]) / spot >= 0.0015 or spot < r["support"] else 0.0
    return round(bull / 95.0 * 100.0, 1), round(bear / 95.0 * 100.0, 1)


def atm_iv_of(payload: dict) -> float:
    """ATM implied volatility of one snapshot (mean of the call and the put), NaN when it has none."""
    atm, rows = payload.get("atm"), payload.get("rows") or {}
    a = rows.get(f"{atm:g}") if atm is not None else None
    ivs = [v for v in (_v(a, 6), _v(a, 7)) if np.isfinite(v) and v > 0] if a is not None else []
    return float(np.mean(ivs)) if ivs else np.nan


def chain_series(snaps: List[tuple], lookback_minutes: int = 10, iv_history: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Snapshots [(ts, payload)] -> frame indexed by ts with CHAIN_COLUMNS. Changes compare a snapshot with the
    one `lookback_minutes` earlier on the same day and expiry, strike by strike (never across a roll).

    iv_pctile is the IV percentile: the share (0-100) of earlier ATM IV readings at or below the current one. The
    earlier readings are `iv_history` (previous sessions, see Store.iv_history) plus the snapshots before this one,
    so it is causal; it stays NaN until there are IV_PCTILE_MIN_READINGS readings to compare with.
    ce_vol_pct / pe_vol_pct are the contracts traded over the look-back as a % of that side's open interest."""
    out, times = [], [t for t, _ in snaps]
    seen_iv = sorted(float(v) for v in (iv_history or []) if v is not None and np.isfinite(v) and v > 0)
    stamps = [t.value for t in times]
    for n, (ts, now) in enumerate(snaps):
        row = {c: np.nan for c in CHAIN_COLUMNS}
        row["t"] = ts
        rows = now.get("rows") or {}
        atm = now.get("atm")
        ce_oi = sum(v for v in (_v(r, 2) for r in rows.values()) if np.isfinite(v))
        pe_oi = sum(v for v in (_v(r, 3) for r in rows.values()) if np.isfinite(v))
        ce_vol = sum(v for v in (_v(r, 4) for r in rows.values()) if np.isfinite(v))
        pe_vol = sum(v for v in (_v(r, 5) for r in rows.values()) if np.isfinite(v))
        row["pcr"] = pe_oi / ce_oi if ce_oi > 0 else np.nan
        row["pcr_vol"] = pe_vol / ce_vol if ce_vol > 0 else np.nan
        row["spot"] = now.get("spot") if now.get("spot") is not None else np.nan
        row["dte"] = now.get("dte") if now.get("dte") is not None else np.nan
        key = f"{atm:g}" if atm is not None else None
        if atm is not None:
            row["support"], row["resistance"] = _walls(rows, float(atm))
        if key in rows:
            a = rows[key]
            ivs = [v for v in (_v(a, 6), _v(a, 7)) if np.isfinite(v) and v > 0]
            row["atm_iv"] = float(np.mean(ivs)) if ivs else np.nan
            spreads = [v for v in (_v(a, 8), _v(a, 9)) if np.isfinite(v)]
            row["atm_spread_pct"] = max(spreads) if spreads else np.nan
            atm_vol = sum(v for v in (_v(a, 4), _v(a, 5)) if np.isfinite(v))
            row["atm_vol_share"] = atm_vol / (ce_vol + pe_vol) * 100.0 if (ce_vol + pe_vol) > 0 else np.nan
        if np.isfinite(row["atm_iv"]):
            if len(seen_iv) >= IV_PCTILE_MIN_READINGS:
                row["iv_pctile"] = bisect.bisect_right(seen_iv, row["atm_iv"]) / len(seen_iv) * 100.0
            bisect.insort(seen_iv, float(row["atm_iv"]))
        target = ts - pd.Timedelta(minutes=lookback_minutes)
        j = bisect.bisect_right(stamps, target.value, 0, n) - 1      # latest snapshot at or before the look-back time
        if j >= 0 and (target - times[j]) <= pd.Timedelta(minutes=lookback_minutes) and times[j].date() == ts.date() \
                and snaps[j][1].get("expiry") == now.get("expiry"):
            prev = snaps[j][1]
            old = prev.get("rows") or {}
            if key in rows and key in old:
                a, b = rows[key], old[key]
                if _v(b, 0) > 0 and _v(a, 0) > 0:
                    row["ce_mom_pct"] = (_v(a, 0) / _v(b, 0) - 1) * 100
                if _v(b, 1) > 0 and _v(a, 1) > 0:
                    row["pe_mom_pct"] = (_v(a, 1) / _v(b, 1) - 1) * 100
                iv0 = [v for v in (_v(b, 6), _v(b, 7)) if np.isfinite(v) and v > 0]
                if iv0 and np.isfinite(row["atm_iv"]):
                    row["iv_chg"] = row["atm_iv"] - float(np.mean(iv0))
            common = [k for k in rows if k in old]
            tot = lambda src, i: sum(v for v in (_v(src[k], i) for k in common) if np.isfinite(v))      # noqa: E731
            c0, p0 = tot(old, 2), tot(old, 3)
            if c0 > 0:
                row["ce_oi_chg_pct"] = (tot(rows, 2) / c0 - 1) * 100
            if p0 > 0:
                row["pe_oi_chg_pct"] = (tot(rows, 3) / p0 - 1) * 100
            if c0 > 0 and p0 > 0:
                row["pcr_chg"] = tot(rows, 3) / tot(rows, 2) - p0 / c0 if tot(rows, 2) > 0 else np.nan
            row["ce_vol_chg"], row["pe_vol_chg"] = tot(rows, 4) - tot(old, 4), tot(rows, 5) - tot(old, 5)
            if c0 > 0:
                row["ce_vol_pct"] = row["ce_vol_chg"] / c0 * 100.0
            if p0 > 0:
                row["pe_vol_pct"] = row["pe_vol_chg"] / p0 * 100.0
            if atm is not None:
                bull, bear, near = _flows(rows, old, float(atm))
                if bull + bear > 0 and near > 0:
                    row["flow_bias"] = (bull - bear) / (bull + bear)
                    row["flow_pct"] = (bull + bear) / near * 100.0
            row["bull_score"], row["bear_score"] = _scores(row)
        out.append(row)
    if not out:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    return pd.DataFrame(out).set_index("t")


def chain_explain(row, side: str) -> List[str]:
    """Plain-language reasons behind an option-chain reading (one chain_series row)."""
    def g(name):
        v = row.get(name) if hasattr(row, "get") else None
        return None if v is None or not np.isfinite(v) else float(v)
    up, why = side == "CE", []
    bias, act = g("flow_bias"), g("flow_pct")
    if bias is not None and act is not None:
        lean = "bullish (call buying / put writing / short covering)" if bias > 0 else "bearish (put buying / call writing)"
        why.append(f"Near-the-money OI changed by {act:.1f}% and {abs(bias) * 100:.0f}% of that change is {lean}, judged by "
                   "whether each option's price rose or fell with its OI")
    ce, pe = g("ce_mom_pct"), g("pe_mom_pct")
    if ce is not None and pe is not None:
        why.append(f"ATM premiums: CE {ce:+.1f}%, PE {pe:+.1f}% over the look-back")
    if g("pcr") is not None:
        chg = g("pcr_chg")
        why.append(f"PCR (OI) {g('pcr'):.2f}" + (f", {'rising' if chg > 0 else 'falling'} {abs(chg):.2f}" if chg is not None else ""))
    if g("atm_iv") is not None:
        chg = g("iv_chg")
        why.append(f"ATM IV {g('atm_iv'):.1f}%" + (f" ({chg:+.2f} pts)" if chg is not None else ""))
    if g("iv_pctile") is not None:
        pct = g("iv_pctile")
        note = "options are expensive against their recent range" if pct >= 80 else "options are cheap against their recent range" if pct <= 20 else "mid-range"
        why.append(f"IV percentile {pct:.0f} ({note})")
    for s, name in (("ce", "calls"), ("pe", "puts")):
        vol, oi = g(f"{s}_vol_pct"), g(f"{s}_oi_chg_pct")
        if vol is not None and oi is not None and vol >= 1.0:
            why.append(f"{name.capitalize()}: {vol:.1f}% of the OI traded while OI {'rose' if oi > 0 else 'fell'} {abs(oi):.1f}% "
                       f"({'new positions' if oi > 0 else 'positions being closed - volume and OI diverge'})")
    if g("support") is not None or g("resistance") is not None:
        why.append(f"OI concentration: put wall (support) {g('support') or 0:g}, call wall (resistance) {g('resistance') or 0:g}")
    if g("atm_spread_pct") is not None:
        why.append(f"ATM bid/ask spread {g('atm_spread_pct'):.2f}%, ATM strikes carry {g('atm_vol_share') or 0:.0f}% of the volume")
    if g("dte") is not None:
        why.append("Expiry today: OI changes are dominated by position closing" if g("dte") == 0 else f"{g('dte'):.0f} day(s) to expiry")
    b, s = g("bull_score"), g("bear_score")
    if b is not None and s is not None:
        why.insert(0, f"Option-chain evidence: bullish {b:.0f} / bearish {s:.0f} (out of 100), and the underlying's price action "
                      f"agrees with the {'bullish' if up else 'bearish'} side")
    return why
