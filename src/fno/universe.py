"""
F&O universe from the Upstox instrument master (JSON, refreshed once a day).

Nothing about a contract is hard-coded: for every underlying the expiries, lot size per expiry, listed strikes
(so the strike interval), tick size, option instrument keys and futures keys are read from the master.

    indices   NIFTY, BANKNIFTY, FINNIFTY (NSE_FO) and SENSEX (BSE_FO), matched by `underlying_symbol`
    stocks    every NSE_FO underlying with underlying_type == EQUITY that has CE and PE rows today

A stock leaves this list on the day the exchange stops listing its options; the backtester warns that a
universe built today is survivorship-biased for the past (see backtest.py).
"""
from __future__ import annotations

import gzip
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.config.settings import CACHE_DIR

logger = logging.getLogger("scanner.fno.universe")
MASTER_URLS = {"NSE": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
               "BSE": "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz"}
INDEX_SEGMENT = {"NIFTY": "NSE_FO", "BANKNIFTY": "NSE_FO", "FINNIFTY": "NSE_FO", "SENSEX": "BSE_FO"}
KEEP = ("segment", "instrument_type", "instrument_key", "underlying_symbol", "underlying_key", "underlying_type",
        "expiry", "strike_price", "lot_size", "tick_size", "trading_symbol", "weekly", "name", "freeze_quantity")
VIX_NAME = "INDIA VIX"


@dataclass
class Underlying:
    symbol: str
    key: str                                  # NSE_INDEX|... or NSE_EQ|...
    kind: str                                 # INDEX | EQUITY
    segment: str                              # NSE_FO | BSE_FO
    name: str = ""
    lots: Dict[date, int] = field(default_factory=dict)
    strikes: Dict[date, List[float]] = field(default_factory=dict)
    contracts: Dict[Tuple[date, float, str], str] = field(default_factory=dict)   # (expiry, strike, CE/PE) -> key
    futures: List[Tuple[date, str]] = field(default_factory=list)
    tick: float = 0.05
    freeze_qty: float = 0.0

    @property
    def expiries(self) -> List[date]:
        return sorted(self.lots)

    def lot_size(self, expiry: date) -> int:
        return int(self.lots.get(expiry, 0))

    def strike_step(self, expiry: date, near: Optional[float] = None) -> float:
        """Most common gap between listed strikes (around `near` when given: steps widen far from the money)."""
        strikes = self.strikes.get(expiry) or []
        if near is not None and len(strikes) > 12:
            i = int(np.searchsorted(strikes, near))
            strikes = strikes[max(0, i - 6):i + 6]
        if len(strikes) < 2:
            return float("nan")
        diffs = np.round(np.diff(strikes), 4)
        values, counts = np.unique(diffs[diffs > 0], return_counts=True)
        return float(values[np.argmax(counts)]) if len(values) else float("nan")

    def near_future(self, today: date) -> Optional[str]:
        live = [k for e, k in sorted(self.futures) if e >= today]
        return live[0] if live else None

    def pick_expiry(self, today: date, rule: str = "nearest", min_days: int = 0) -> Optional[date]:
        live = [e for e in self.expiries if (e - today).days >= min_days]
        if not live:
            return None
        if rule == "next" and len(live) > 1:
            return live[1]
        if rule == "monthly":
            first = live[0]
            month = [e for e in live if (e.year, e.month) == (first.year, first.month)]
            return month[-1]
        return live[0]

    def to_dict(self, today: Optional[date] = None) -> dict:
        today = today or date.today()
        e = self.pick_expiry(today)
        return {"symbol": self.symbol, "name": self.name or self.symbol, "kind": self.kind, "segment": self.segment,
                "expiries": [x.isoformat() for x in self.expiries[:6]], "lot_size": self.lot_size(e) if e else None,
                "strike_step": self.strike_step(e) if e else None, "tick": self.tick}


def _expiry(value) -> Optional[date]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None
    # milliseconds since epoch, 23:59:59 IST of the expiry day
    return (datetime(1970, 1, 1) + timedelta(seconds=(v / 1000 if v > 1e11 else v) + 19800)).date()


class Universe:
    def __init__(self, rows: Iterable[dict], today: date, indices: Iterable[str] = tuple(INDEX_SEGMENT)):
        self.today = today
        self.items: Dict[str, Underlying] = {}
        self.vix_key = "NSE_INDEX|India VIX"
        self.equity_keys: Dict[str, str] = {}
        wanted = {i: INDEX_SEGMENT[i] for i in indices if i in INDEX_SEGMENT}
        for row in rows:
            segment = row.get("segment")
            itype = str(row.get("instrument_type") or "").upper()
            if segment == "NSE_INDEX":
                if str(row.get("trading_symbol") or row.get("name") or "").upper() == VIX_NAME:
                    self.vix_key = row.get("instrument_key") or self.vix_key
                continue
            if segment not in ("NSE_FO", "BSE_FO") or itype not in ("CE", "PE", "FUT"):
                continue
            symbol = str(row.get("underlying_symbol") or "").strip().upper()
            utype = str(row.get("underlying_type") or "").upper()
            if utype == "INDEX":
                if wanted.get(symbol) != segment:
                    continue
            elif utype != "EQUITY" or segment != "NSE_FO":
                continue
            expiry = _expiry(row.get("expiry"))
            key, ukey = row.get("instrument_key"), row.get("underlying_key")
            if expiry is None or expiry < today or not key or not ukey:
                continue
            und = self.items.get(symbol)
            if und is None:
                und = self.items[symbol] = Underlying(symbol=symbol, key=ukey, kind=utype, segment=segment,
                                                      name=str(row.get("name") or symbol))
            if itype == "FUT":
                und.futures.append((expiry, key))
                continue
            try:
                strike, lot = float(row.get("strike_price")), int(float(row.get("lot_size")))
            except (TypeError, ValueError):
                continue
            if lot <= 0 or strike <= 0:
                continue
            und.lots[expiry] = lot
            und.contracts[(expiry, strike, itype)] = key
            tick = row.get("tick_size")
            if tick:
                und.tick = float(tick) / 100.0          # the master gives paise
            und.freeze_qty = float(row.get("freeze_quantity") or und.freeze_qty or 0)
        for und in list(self.items.values()):
            for (expiry, strike, _t) in und.contracts:
                und.strikes.setdefault(expiry, []).append(strike)
            und.strikes = {e: sorted(set(s)) for e, s in und.strikes.items()}
            und.futures.sort()
            sides = {t for (_e, _s, t) in und.contracts}
            if sides != {"CE", "PE"}:
                del self.items[und.symbol]

    # ------------------------------------------------------------------
    def get(self, symbol: str) -> Optional[Underlying]:
        return self.items.get(str(symbol or "").strip().upper())

    @property
    def indices(self) -> List[Underlying]:
        order = {s: i for i, s in enumerate(INDEX_SEGMENT)}
        return sorted((u for u in self.items.values() if u.kind == "INDEX"), key=lambda u: order.get(u.symbol, 99))

    @property
    def stocks(self) -> List[Underlying]:
        return sorted((u for u in self.items.values() if u.kind == "EQUITY"), key=lambda u: u.symbol)

    def select(self, indices: Iterable[str], stocks: str = "all") -> List[Underlying]:
        out = [u for u in self.indices if u.symbol in set(indices)]
        choice = str(stocks or "none").strip().lower()
        if choice == "all":
            out += self.stocks
        elif choice != "none":
            names = {s.strip().upper() for s in str(stocks).split(",") if s.strip()}
            missing = sorted(names - {u.symbol for u in self.stocks})
            if missing:
                logger.warning("Not in today's F&O list (skipped): %s", ", ".join(missing))
            out += [u for u in self.stocks if u.symbol in names]
        return out


def load_rows(cache_dir: Path = CACHE_DIR, today: Optional[date] = None, session=None) -> List[dict]:
    """F&O + index rows of the NSE and BSE masters; downloaded at most once a day."""
    today = today or date.today()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"fno_master_{today.isoformat()}.json.gz"
    if cache.exists():
        try:
            return json.loads(gzip.decompress(cache.read_bytes()))
        except (OSError, ValueError):
            logger.warning("Cached F&O master unreadable; downloading again.")
    if session is None:
        import requests
        session = requests
    rows: List[dict] = []
    for exchange, url in MASTER_URLS.items():
        response = session.get(url, timeout=90)
        response.raise_for_status()
        for row in json.loads(gzip.decompress(response.content)):
            if row.get("segment") in ("NSE_FO", "BSE_FO", "NSE_INDEX", "BSE_INDEX"):
                rows.append({k: row.get(k) for k in KEEP if row.get(k) is not None})
    tmp = cache.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(rows).encode("utf-8")))
    os.replace(tmp, cache)
    for old in cache_dir.glob("fno_master_*.json.gz"):
        if old != cache:
            try:
                old.unlink()
            except OSError:
                pass
    logger.info("F&O instrument master downloaded: %d rows.", len(rows))
    return rows


def load_universe(indices: Iterable[str] = tuple(INDEX_SEGMENT), today: Optional[date] = None, **kw) -> Universe:
    today = today or date.today()
    return Universe(load_rows(today=today, **kw), today, indices)
