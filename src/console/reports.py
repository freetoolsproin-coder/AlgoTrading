"""Downloadable reports: trades / orders / audit as CSV, trades as JSON. Values are written as recorded."""
from __future__ import annotations

import csv
import io
import json
from typing import Iterable, List

TRADE_COLUMNS = ["source", "book", "id", "index", "side", "symbol", "day", "entry_time", "exit_time", "qty",
                 "entry_price", "exit_price", "gross", "charges", "net", "slippage", "slippage_basis", "exit_reason",
                 "open", "open_pnl", "origin"]
ORDER_COLUMNS = ["created", "updated", "day", "source", "broker", "purpose", "side", "order_type", "instrument_key",
                 "quantity", "limit_price", "trigger_price", "status", "filled", "avg_price", "reprices", "tag",
                 "message", "position_id", "key"]
AUDIT_COLUMNS = ["id", "time", "actor", "ip", "action", "target", "outcome", "detail", "prev_hash", "hash"]


def _safe(v):
    """Neutralise spreadsheet formula injection in text cells."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") and not _is_number(v):
        return "'" + v
    return v


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def to_csv(rows: Iterable[dict], columns: List[str], preamble: List[str] = ()) -> str:
    buf = io.StringIO()
    for line in preamble:
        buf.write(f"# {line}\n")
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: _safe(r.get(k)) if not isinstance(r.get(k), (list, dict)) else json.dumps(r.get(k))
                    for k in columns})
    return buf.getvalue()


def trades_csv(trades, preamble=()) -> str:
    return to_csv((t.to_dict() for t in trades), TRADE_COLUMNS, preamble)


def trades_json(trades, meta: dict) -> str:
    return json.dumps({"meta": meta, "trades": [t.to_dict() for t in trades]}, indent=1, default=str)
