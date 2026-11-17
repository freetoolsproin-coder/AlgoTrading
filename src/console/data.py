"""
Read-only access to the trading system's real records, normalised into one Trade shape with a SOURCE label.

    live      real broker fills: data/execution.db rows whose broker is not the simulated "paper" broker
    paper     simulated books: scanner lanes v1 / v2 / v3 (data/prebreakout.db) and the movement / strategy
              books (data/movement.db); fills are modelled at ask + slippage / bid - slippage
    backtest  historical simulations: output/*_trades.csv written by the backtest commands (options priced
              with Black-Scholes, IV = India VIX - no historical option quotes)

Every database is opened with SQLite's read-only URI mode, so the console can never change trading data.
Mock-mode records (synthetic data) are excluded unless asked for. Nothing here invents a value: a field the
source does not have is None and the page says "not recorded".
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("scanner.console")

LIVE, PAPER, BACKTEST = "live", "paper", "backtest"
SOURCES = (LIVE, PAPER, BACKTEST)
SOURCE_LABEL = {LIVE: "Live (broker)", PAPER: "Paper (simulated)", BACKTEST: "Backtest (modelled)"}


@dataclass
class Trade:
    source: str
    book: str                     # human label, e.g. "Scanner v1", "Movement scalp", "v3_backtest: v1 (current rules)"
    book_key: str                 # stable key for filters
    id: str
    index: str
    side: str                     # CE / PE
    symbol: str
    day: str
    entry_time: str
    exit_time: Optional[str]
    qty: int
    entry_price: Optional[float]
    exit_price: Optional[float]
    gross: Optional[float]
    charges: Optional[float]
    net: Optional[float]
    slippage: Optional[float] = None     # rupees; see slippage_basis
    slippage_basis: str = ""             # "modelled x% per leg" / "vs signal price" / ""
    exit_reason: str = ""
    open: bool = False
    open_pnl: Optional[float] = None
    mock: bool = False
    origin: str = ""

    def to_dict(self):
        return asdict(self)


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _loads(payload) -> dict:
    if not payload:
        return {}
    try:
        # the scanner writes NaN for some optional fields; Python's json accepts it
        d = json.loads(payload)
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


def connect_ro(path: Path) -> Optional[sqlite3.Connection]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        c = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
        c.row_factory = sqlite3.Row
        return c
    except sqlite3.Error as error:
        logger.warning("console: cannot open %s read-only: %s", path.name, error)
        return None


def _table_exists(c: sqlite3.Connection, name: str) -> bool:
    return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _avg_exit(exits) -> Optional[float]:
    if not isinstance(exits, list):
        return None
    qty = sum(_num(e.get("qty")) or 0 for e in exits if isinstance(e, dict))
    val = sum((_num(e.get("qty")) or 0) * (_num(e.get("price")) or 0) for e in exits if isinstance(e, dict))
    return round(val / qty, 2) if qty else None


def _modelled_slippage(entry: Optional[float], exit_: Optional[float], qty: int, pct: float) -> Optional[float]:
    if not entry or not qty or pct <= 0:
        return None
    legs = entry * qty + ((exit_ or 0) * qty)
    return round(legs * pct / 100.0, 2)


# ----------------------------------------------------------------------------- paper: scanner lanes
def scanner_paper_trades(db: Path, slippage_pct: float = 0.25, include_mock: bool = False) -> List[Trade]:
    c = connect_ro(db)
    if c is None:
        return []
    out = []
    try:
        if not _table_exists(c, "paper_trades"):
            return []
        for r in c.execute("SELECT * FROM paper_trades ORDER BY entry_time"):
            if r["mock"] and not include_mock:
                continue
            p = _loads(r["payload"])
            model = (r["model"] if "model" in r.keys() else None) or p.get("model") or "v1"
            closed = bool(r["closed"])
            qty = int(_num(p.get("lot_size")) or 0) * int(_num(p.get("lots")) or 0) or int(r["qty"] or 0)
            exit_price = _avg_exit(p.get("exits"))
            gross = _num(p.get("realised_gross"))
            charges = _num(p.get("charges"))
            net = _num(r["net_pnl"]) if closed else None
            out.append(Trade(
                source=PAPER, book=f"Scanner {model}", book_key=f"scanner-{model}", id=str(r["id"]),
                index=str(r["idx"] or p.get("index") or ""), side=str(r["direction"] or ""), symbol=str(r["symbol"] or ""),
                day=str(r["day"] or ""), entry_time=str(r["entry_time"] or ""), exit_time=r["exit_time"],
                qty=qty, entry_price=_num(r["entry_price"]), exit_price=exit_price,
                gross=gross if closed else None, charges=charges, net=net,
                slippage=_modelled_slippage(_num(r["entry_price"]), exit_price, qty, slippage_pct),
                slippage_basis=f"modelled {slippage_pct:g}% per leg (in fill prices)",
                exit_reason=str(r["exit_reason"] or ""), open=not closed,
                open_pnl=_num(p.get("open_pnl")) if not closed else None, mock=bool(r["mock"]),
                origin="data/prebreakout.db"))
    except sqlite3.Error as error:
        logger.warning("console: reading paper_trades failed: %s", error)
    finally:
        c.close()
    return out


# ----------------------------------------------------------------------------- paper: movement / strategy books
def movement_paper_trades(db: Path, slippage_pct: float = 0.25, include_mock: bool = False) -> List[Trade]:
    c = connect_ro(db)
    if c is None:
        return []
    out = []
    try:
        if not _table_exists(c, "movement_trades"):
            return []
        for r in c.execute("SELECT * FROM movement_trades ORDER BY day"):
            if r["mock"] and not include_mock:
                continue
            p = _loads(r["payload"])
            is_open = bool(r["open"])
            qty = int(_num(p.get("qty")) or 0)
            book = str(r["book"] or p.get("book") or "movement")
            entry, exit_ = _num(p.get("entry_price")), _num(p.get("exit_price"))
            gross = _num(p.get("gross"))
            out.append(Trade(
                source=PAPER, book=f"Movement {book}", book_key=f"movement-{book}", id=str(r["id"]),
                index=str(r["idx"] or ""), side=str(r["side"] or ""), symbol=str(p.get("symbol") or ""),
                day=str(r["day"] or ""), entry_time=str(p.get("entry_time") or ""), exit_time=p.get("exit_time"),
                qty=qty, entry_price=entry, exit_price=exit_, gross=None if is_open else gross,
                charges=_num(p.get("charges")), net=None if is_open else _num(r["net"]),
                slippage=_modelled_slippage(entry, exit_, qty, slippage_pct),
                slippage_basis=f"modelled {slippage_pct:g}% per leg (in fill prices)",
                exit_reason=str(p.get("exit_reason") or ""), open=is_open,
                open_pnl=_num(p.get("net")) if is_open else None, mock=bool(r["mock"]), origin="data/movement.db"))
    except sqlite3.Error as error:
        logger.warning("console: reading movement_trades failed: %s", error)
    finally:
        c.close()
    return out


# ----------------------------------------------------------------------------- order journal + live
def order_journal(db: Path, limit: int = 500) -> List[dict]:
    """Every order the execution engine sent (paper broker and live broker), newest first."""
    c = connect_ro(db)
    if c is None:
        return []
    rows = []
    try:
        if not _table_exists(c, "orders"):
            return []
        for r in c.execute("SELECT * FROM orders ORDER BY day DESC LIMIT ?", (int(limit),)):
            p = _loads(r["payload"])
            req = p.get("req") or {}
            broker = str(r["broker"] or "")
            rows.append({
                "key": r["key"], "day": r["day"], "broker": broker, "source": PAPER if broker == "paper" else LIVE,
                "purpose": p.get("purpose", ""), "status": p.get("status", ""), "side": req.get("side", ""),
                "order_type": req.get("order_type", ""), "instrument_key": req.get("instrument_key", ""),
                "quantity": req.get("quantity"), "limit_price": _num(req.get("price")),
                "trigger_price": _num(req.get("trigger_price")), "filled": p.get("filled"),
                "avg_price": _num(p.get("avg_price")), "tag": p.get("tag", ""), "order_ids": p.get("order_ids") or [],
                "created": p.get("created", ""), "updated": p.get("updated", ""), "reprices": p.get("reprices", 0),
                "message": p.get("message", ""), "position_id": p.get("position_id", "")})
    except sqlite3.Error as error:
        logger.warning("console: reading orders failed: %s", error)
    finally:
        c.close()
    rows.sort(key=lambda o: o.get("created") or "", reverse=True)
    return rows


def live_trades(db: Path, cost_fn=None, signal_prices: Optional[Dict[str, float]] = None) -> List[Trade]:
    """Positions the execution engine held at a REAL broker (broker != 'paper'), from its journal."""
    c = connect_ro(db)
    if c is None:
        return []
    out = []
    signal_prices = signal_prices or {}
    try:
        if not _table_exists(c, "live_positions"):
            return []
        orders: Dict[str, list] = {}
        if _table_exists(c, "orders"):
            for r in c.execute("SELECT * FROM orders WHERE broker != 'paper'"):
                p = _loads(r["payload"])
                orders.setdefault(p.get("position_id", ""), []).append(p)
        for r in c.execute("SELECT * FROM live_positions WHERE broker != 'paper' ORDER BY day"):
            p = _loads(r["payload"])
            pid = p.get("position_id") or r["id"]
            fills = [o for o in orders.get(pid, []) if (o.get("filled") or 0) > 0]
            buys = [o for o in fills if (o.get("req") or {}).get("side") == "BUY"]
            sells = [o for o in fills if (o.get("req") or {}).get("side") == "SELL"]
            bq = sum(o["filled"] for o in buys)
            sq = sum(o["filled"] for o in sells)
            entry = (sum(o["filled"] * (_num(o.get("avg_price")) or 0) for o in buys) / bq) if bq else _num(p.get("entry_avg"))
            exit_ = (sum(o["filled"] * (_num(o.get("avg_price")) or 0) for o in sells) / sq) if sq else None
            exch = str(p.get("instrument_key", "")).split("|")[0]
            charges = None
            if cost_fn is not None and fills:
                charges = round(sum(cost_fn((o.get("req") or {}).get("side"), _num(o.get("avg_price")) or 0,
                                            int(o["filled"]), exch) for o in fills), 2)
            closed = bool(p.get("closed"))
            gross = _num(p.get("realised"))
            sig = signal_prices.get(pid)
            slip = round((entry - sig) * bq, 2) if (sig and entry and bq) else None
            first = min((o.get("created") or "" for o in buys), default="")
            last = max((o.get("updated") or "" for o in sells), default="") or None
            out.append(Trade(
                source=LIVE, book=f"Live ({r['broker']})", book_key=f"live-{r['broker']}", id=str(pid),
                index=str(p.get("index") or ""), side=str(p.get("symbol", "")[-2:]), symbol=str(p.get("symbol") or ""),
                day=str(r["day"] or ""), entry_time=first, exit_time=last if closed else None,
                qty=int(bq or p.get("planned_qty") or 0), entry_price=round(entry, 2) if entry else None,
                exit_price=round(exit_, 2) if exit_ else None, gross=gross if closed else None, charges=charges,
                net=round(gross - (charges or 0), 2) if (closed and gross is not None) else None,
                slippage=slip, slippage_basis="entry fill vs signal price" if slip is not None else "",
                exit_reason=str(p.get("note") or ""), open=not closed, origin="data/execution.db"))
    except sqlite3.Error as error:
        logger.warning("console: reading live_positions failed: %s", error)
    finally:
        c.close()
    return out


# ----------------------------------------------------------------------------- backtest CSVs
BACKTEST_REQUIRED = {"day", "entry_time", "net"}


def backtest_files(output_dir: Path) -> List[Path]:
    return sorted(Path(output_dir).glob("*_trades.csv")) if Path(output_dir).exists() else []


def backtest_trades(output_dir: Path, files: Optional[Iterable[Path]] = None) -> List[Trade]:
    out = []
    for f in (files if files is not None else backtest_files(output_dir)):
        stem = f.stem.replace("_trades", "")
        default_index = "SENSEX" if "SENSEX" in stem.upper() else "NIFTY"
        try:
            with open(f, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not BACKTEST_REQUIRED.issubset(set(reader.fieldnames or [])):
                    logger.info("console: %s skipped (no day / entry_time / net columns)", f.name)
                    continue
                for i, row in enumerate(reader):
                    variant = row.get("set") or row.get("book") or "trades"
                    sk = row.get("setup_key") or ""
                    parts = sk.split("|")
                    index = parts[1] if len(parts) > 2 else (row.get("index") or default_index)
                    qty = int(_num(row.get("qty")) or 0)
                    out.append(Trade(
                        source=BACKTEST, book=f"{stem}: {variant}", book_key=f"bt-{stem}-{variant}",
                        id=f"{stem}#{i}", index=index, side=row.get("direction") or row.get("side") or "",
                        symbol=f"{index} {row.get('strike', '')} {row.get('direction', '')}".strip(),
                        day=row.get("day", ""), entry_time=row.get("entry_time", ""), exit_time=row.get("exit_time"),
                        qty=qty, entry_price=_num(row.get("entry_price")), exit_price=_num(row.get("exit_price")),
                        gross=_num(row.get("gross")), charges=_num(row.get("charges")), net=_num(row.get("net")),
                        slippage=None, slippage_basis="modelled in backtest fills",
                        exit_reason=row.get("exit_reason", ""), origin=f"output/{f.name}"))
        except (OSError, csv.Error, UnicodeDecodeError) as error:
            logger.warning("console: cannot read %s: %s", f.name, error)
    return out


def backtest_reports(output_dir: Path) -> List[dict]:
    out = []
    for f in sorted(Path(output_dir).glob("*.html")) if Path(output_dir).exists() else []:
        if "backtest" not in f.name:
            continue
        st = f.stat()
        out.append({"name": f.name, "size": st.st_size, "mtime": st.st_mtime})
    for f in sorted([*Path(output_dir).glob("movement_backtest_*.txt"), *Path(output_dir).glob("strategy_backtest_*.txt")]) \
            if Path(output_dir).exists() else []:
        st = f.stat()
        out.append({"name": f.name, "size": st.st_size, "mtime": st.st_mtime})
    return out


# ----------------------------------------------------------------------------- scanner state
def read_state(state_file: Path) -> Optional[dict]:
    try:
        text = Path(state_file).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


# ----------------------------------------------------------------------------- facade
@dataclass
class DataSources:
    scanner_db: Path
    execution_db: Path
    movement_db: Path
    output_dir: Path
    state_file: Path
    slippage_pct: float = 0.25
    cost_fn: object = None
    include_mock: bool = False
    _cache: dict = field(default_factory=dict, repr=False)

    def _mtimes(self):
        files = [self.scanner_db, self.execution_db, self.movement_db, *backtest_files(self.output_dir)]
        return tuple((str(f), f.stat().st_mtime if f.exists() else 0) for f in files)

    def trades(self, source: Optional[str] = None) -> List[Trade]:
        key = (self._mtimes(), self.include_mock)
        if self._cache.get("key") != key:
            paper = scanner_paper_trades(self.scanner_db, self.slippage_pct, self.include_mock) + \
                movement_paper_trades(self.movement_db, self.slippage_pct, self.include_mock)
            signal = {t.id: t.entry_price for t in paper if t.book_key.startswith("scanner-") and t.entry_price}
            live = live_trades(self.execution_db, self.cost_fn, signal)
            self._cache = {"key": key, "all": live + paper + backtest_trades(self.output_dir)}
        rows = self._cache["all"]
        return [t for t in rows if source is None or t.source == source]

    def books(self, source: Optional[str] = None) -> List[dict]:
        seen: Dict[str, dict] = {}
        for t in self.trades(source):
            seen.setdefault(t.book_key, {"key": t.book_key, "label": t.book, "source": t.source})
        return list(seen.values())

    def orders(self, limit: int = 500) -> List[dict]:
        return order_journal(self.execution_db, limit)

    def state(self) -> Optional[dict]:
        return read_state(self.state_file)
