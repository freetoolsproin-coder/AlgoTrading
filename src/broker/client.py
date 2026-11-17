"""
Upstox REST client and instrument master for the NIFTY / SENSEX scanner.

Authentication
    The access token is read from (first match wins):
      1. UPSTOX_ACCESS_TOKEN (environment variable or .env)
      2. upstox_token.json in the project root, written each morning by `python -m src.broker.auth`
    It is re-read when it changes, so a new token is picked up without a restart. Tokens are never logged.

Endpoints used (current Upstox API)
    v2  /market-quote/quotes                    LTP, OHLC, depth, OI (batched, <= 450 keys per call)
    v2  /option/chain                           option chain with greeks for one expiry
    v3  /historical-candle/intraday/{key}/minutes/1          today's 1-minute candles
    v3  /historical-candle/{key}/minutes/1/{to}/{from}       earlier 1-minute candles
    v3  /historical-candle/{key}/days/1/{to}/{from}          daily candles
    v3  /feed/market-data-feed/authorize                     WebSocket URL (src/broker/websocket_feed.py)
    (The v2 historical-candle endpoints used before are superseded by the v3 ones above.)

Rate limit: one shared lock spaces every call by 1 / requests_per_second; HTTP 429 backs off and retries.

Instrument master: the Upstox NSE and BSE CSV files, downloaded at most once a day and cached (filtered) in
data/cache/. Only NIFTY / SENSEX index rows, their index options (OPTIDX) and index futures (FUTIDX), and
NSE equities (for the breadth constituents) are kept. Stock F&O rows are discarded.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time as _time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.config.instruments import INSTRUMENTS, VIX_KEY, IndexInstrument
from src.config.settings import CACHE_DIR, TOKEN_FILE, get_secret
from src.utils.numbers import finite, fnum

logger = logging.getLogger("scanner.broker")
IST = "Asia/Kolkata"
V2_BASE = "https://api.upstox.com/v2"
V3_BASE = "https://api.upstox.com/v3"
UPSTOX_MASTER_URLS = {"NSE": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz",
                      "BSE": "https://assets.upstox.com/market-quote/instruments/exchange/BSE.csv.gz"}
NSE_HEADERS = {"Accept": "text/csv,*/*", "User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}
INSTRUMENT_KEY_RE = re.compile(r"^(NSE|BSE)_(INDEX|FO|EQ)\|[A-Za-z0-9 &\-.]+$")


class UpstoxAuthError(Exception):
    """Missing / expired / rejected access token. Deliberately not swallowed by per-index error handling."""


def valid_instrument_key(key) -> bool:
    return isinstance(key, str) and bool(INSTRUMENT_KEY_RE.match(key))


# =============================================================================
# Token
# =============================================================================
class TokenSource:
    """UPSTOX_ACCESS_TOKEN first, else upstox_token.json (reloaded when the file changes)."""

    def __init__(self, token_file: Path = TOKEN_FILE):
        self.token_file = Path(token_file)
        self._mtime = None
        self._file_token: Optional[str] = None

    def token(self) -> str:
        env_token = get_secret("UPSTOX_ACCESS_TOKEN")
        if env_token:
            return env_token
        try:
            mtime = self.token_file.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != self._mtime:
            self._mtime = mtime
            self._file_token = None
            if mtime is not None:
                try:
                    data = json.loads(self.token_file.read_text(encoding="utf-8")) or {}
                    value = data.get("access_token")
                    if isinstance(value, str) and value.strip() not in ("", "None", "null"):
                        self._file_token = value.strip()
                        logger.info("Loaded the Upstox access token from %s.", self.token_file.name)
                except (OSError, ValueError, AttributeError):
                    logger.error("%s is unreadable; generate a new token with: python -m src.broker.auth",
                                 self.token_file.name)
        if not self._file_token:
            raise UpstoxAuthError("No Upstox access token. Set UPSTOX_ACCESS_TOKEN in .env, or run "
                                  "`python -m src.broker.auth` (VS Code: 'Upstox login (daily token)').")
        return self._file_token


# =============================================================================
# REST client
# =============================================================================
class UpstoxClient:
    def __init__(self, requests_per_second: float = 5.0, session=None, token_source: Optional[TokenSource] = None):
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.tokens = token_source or TokenSource()
        self.min_interval = 1.0 / max(requests_per_second, 0.5)
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._last_auth_alert = 0.0
        self.bad_keys: set = set()

    def headers(self) -> dict:
        return {"Accept": "application/json", "Authorization": f"Bearer {self.tokens.token()}"}

    def get(self, path: str, params: Optional[dict] = None, timeout: int = 20, base: str = V2_BASE) -> dict:
        for attempt in range(4):
            with self._lock:  # one shared rate limit for every caller
                wait = self._last_call + self.min_interval - _time.monotonic()
                if wait > 0:
                    _time.sleep(wait)
                self._last_call = _time.monotonic()
            response = self.session.get(f"{base}{path}", params=params, headers=self.headers(), timeout=timeout)
            status = getattr(response, "status_code", 200)
            if status == 401:
                if _time.monotonic() - self._last_auth_alert > 300:
                    self._last_auth_alert = _time.monotonic()
                    logger.critical("Upstox answered 401: the access token is expired or invalid. Generate a new one "
                                    "with `python -m src.broker.auth`; the scanner picks it up automatically.")
                raise UpstoxAuthError("Upstox token rejected (401).")
            if status == 429:
                logger.warning("Upstox rate limit (429) on %s; backing off %ss.", path.split("?")[0], 2 ** attempt)
                _time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("status") == "error":
                raise RuntimeError(f"Upstox error on {path}: {payload.get('errors')}")
            return payload
        raise RuntimeError(f"Upstox rate limit persisted for {path}")

    # ---------- market data ----------
    def quotes(self, keys: Iterable[str]) -> Dict[str, dict]:
        keys = [k for k in dict.fromkeys(keys) if k and k not in self.bad_keys]
        result: Dict[str, dict] = {}
        for start in range(0, len(keys), 450):
            self._quote_chunk(keys[start:start + 450], result)
        return result

    def _quote_chunk(self, chunk: List[str], result: Dict[str, dict]):
        """Upstox rejects a whole request for one invalid key: split the batch until the bad key is isolated."""
        try:
            data = self.get("/market-quote/quotes", {"instrument_key": ",".join(chunk)}).get("data") or {}
        except UpstoxAuthError:
            raise
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status not in (400, 404, 422):
                raise
            if len(chunk) == 1:
                self.bad_keys.add(chunk[0])
                logger.warning("Upstox rejected instrument key %s; it is skipped from now on.", chunk[0])
                return
            middle = len(chunk) // 2
            self._quote_chunk(chunk[:middle], result)
            self._quote_chunk(chunk[middle:], result)
            return
        for response_key, raw in data.items():
            q = normalize_quote(raw)
            token = raw.get("instrument_token")
            if token:
                result[token] = q
            result.setdefault(str(response_key).replace(":", "|", 1), q)

    def option_chain(self, underlying_key: str, expiry: date) -> list:
        payload = self.get("/option/chain", {"instrument_key": underlying_key, "expiry_date": expiry.isoformat()})
        return payload.get("data") or []

    def daily_candles(self, key: str, start: date, end: date) -> pd.DataFrame:
        path = f"/historical-candle/{quote(key, safe='')}/days/1/{end.isoformat()}/{start.isoformat()}"
        return candles_frame(self.get(path, base=V3_BASE).get("data", {}).get("candles", []))

    def minute_candles(self, key: str, start: date, end: date) -> pd.DataFrame:
        path = f"/historical-candle/{quote(key, safe='')}/minutes/1/{end.isoformat()}/{start.isoformat()}"
        return candles_frame(self.get(path, base=V3_BASE).get("data", {}).get("candles", []))

    def minute_candles_today(self, key: str) -> pd.DataFrame:
        path = f"/historical-candle/intraday/{quote(key, safe='')}/minutes/1"
        return candles_frame(self.get(path, base=V3_BASE).get("data", {}).get("candles", []))

    def websocket_url(self) -> str:
        data = self.get("/feed/market-data-feed/authorize", base=V3_BASE).get("data") or {}
        url = data.get("authorized_redirect_uri") or data.get("authorizedRedirectUri")
        if not url:
            raise RuntimeError("Upstox did not return a market-data WebSocket URL")
        return url


def candles_frame(raw) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "oi"])
    width = len(raw[0])
    columns = ["datetime", "open", "high", "low", "close", "volume", "oi"][:width]
    frame = pd.DataFrame(raw, columns=columns)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert(IST)
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "oi" not in frame:
        frame["oi"] = np.nan
    return frame.sort_values("datetime").reset_index(drop=True)


def normalize_quote(raw: dict) -> dict:
    ohlc = raw.get("ohlc") or {}
    depth = raw.get("depth") or {}
    buy = (depth.get("buy") or [{}])[0] or {}
    sell = (depth.get("sell") or [{}])[0] or {}
    ltp = fnum(raw.get("last_price"))
    change = fnum(raw.get("net_change"))
    prev_close = ltp - change if finite(ltp) and finite(change) else float("nan")
    return {
        "ltp": ltp, "open": fnum(ohlc.get("open")), "high": fnum(ohlc.get("high")), "low": fnum(ohlc.get("low")),
        "prev_close": prev_close, "volume": fnum(raw.get("volume"), 0.0), "vwap": fnum(raw.get("average_price")),
        "oi": fnum(raw.get("oi")), "bid": fnum(buy.get("price")), "ask": fnum(sell.get("price")),
        "trade_time": raw.get("last_trade_time"),  # not "timestamp": that is the time of the request
        "symbol": raw.get("symbol"),
    }


# =============================================================================
# Instrument master (NIFTY / SENSEX index options and futures only)
# =============================================================================
def _norm(text) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def parse_date(value) -> Optional[date]:
    if value is None or value == "" or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        v = float(value)
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v).date()
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else ts.date()


def keep_master_row(row: dict) -> bool:
    itype = str(row.get("instrument_type") or "").upper()
    if itype == "INDEX":
        return True
    if itype == "EQUITY":
        return row.get("exchange") == "NSE_EQ"
    if itype in ("OPTIDX", "FUTIDX"):
        inst = INSTRUMENTS.get(str(row.get("name") or "").strip().upper())
        return inst is not None and row.get("exchange") == inst.fo_segment
    return False


class Instruments:
    """Parsed master: index keys, NIFTY/SENSEX option expiries + lot sizes, index futures, equity keys."""

    def __init__(self, rows: List[dict], today: date):
        self.today = today
        self.equity_keys: Dict[str, str] = {}
        self.index_rows: List[dict] = []
        self.index_options: Dict[str, List[Tuple[date, float, str]]] = {}   # NIFTY -> [(expiry, lot, segment)]
        self.index_futures: Dict[str, List[Tuple[date, str, str]]] = {}     # NIFTY -> [(expiry, key, segment)]
        for row in rows:
            if not keep_master_row(row):
                continue
            itype = str(row.get("instrument_type") or "").upper()
            key = row.get("instrument_key")
            if not valid_instrument_key(key):
                continue
            if itype == "EQUITY":
                symbol = str(row.get("tradingsymbol") or "").strip().upper()
                if symbol:
                    self.equity_keys.setdefault(symbol, key)
            elif itype == "INDEX":
                self.index_rows.append(row)
            else:
                expiry = parse_date(row.get("expiry"))
                if expiry is None or expiry < today:
                    continue
                name = str(row.get("name")).strip().upper()
                if itype == "OPTIDX":
                    self.index_options.setdefault(name, []).append((expiry, fnum(row.get("lot_size")), row.get("exchange")))
                else:
                    self.index_futures.setdefault(name, []).append((expiry, key, row.get("exchange")))
        for values in list(self.index_options.values()) + list(self.index_futures.values()):
            values.sort()

    def resolve_index(self, inst: IndexInstrument) -> str:
        wanted = [_norm(n) for n in (inst.index_key.split("|", 1)[1],) + inst.aliases]
        for target in wanted:
            for row in self.index_rows:
                if row.get("exchange") == inst.spot_exchange and \
                        target in (_norm(row.get("name")), _norm(row.get("tradingsymbol"))):
                    return row.get("instrument_key")
        return inst.index_key

    def resolve_vix(self) -> str:
        for row in self.index_rows:
            if _norm(row.get("name")) == "INDIAVIX" or _norm(row.get("tradingsymbol")) == "INDIAVIX":
                return row.get("instrument_key")
        return VIX_KEY

    def option_expiries(self, inst: IndexInstrument) -> Tuple[List[date], Dict[date, int]]:
        rows = [r for r in self.index_options.get(inst.option_name, []) if r[2] == inst.fo_segment]
        return sorted({r[0] for r in rows}), {r[0]: int(r[1]) for r in rows if finite(r[1])}

    def near_future(self, inst: IndexInstrument) -> Optional[str]:
        for _expiry, key, segment in self.index_futures.get(inst.option_name, []):
            if segment == inst.fo_segment:
                return key
        return None


def load_master_rows(cache_dir: Path = CACHE_DIR, today: Optional[date] = None, session=None) -> List[dict]:
    """Filtered master rows; downloaded at most once a day (data/cache/instrument_master_<date>.json)."""
    today = today or date.today()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"instrument_master_{today.isoformat()}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Cached instrument master unreadable; downloading again.")
    if session is None:
        import requests
        session = requests
    rows: List[dict] = []
    for exchange, url in UPSTOX_MASTER_URLS.items():
        response = session.get(url, timeout=60)
        response.raise_for_status()
        frame = pd.read_csv(io.BytesIO(response.content), compression="gzip", low_memory=False)
        rows += [r for r in frame.to_dict("records") if keep_master_row(r)]
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, default=str), encoding="utf-8")
    os.replace(tmp, cache)
    for old in cache_dir.glob("instrument_master_*.json"):
        if old != cache:
            try:
                old.unlink()
            except OSError:
                pass
    logger.info("Instrument master downloaded: %d rows kept (NIFTY/SENSEX index F&O, indices, NSE equities).", len(clean))
    return clean


# =============================================================================
# NSE constituent lists (breadth) with a daily cache
# =============================================================================
def fetch_nse_csv(urls, cache_name: str, cache_dir: Path = CACHE_DIR, max_age_hours: float = 20) -> Optional[str]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / cache_name
    if cache.exists() and (_time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        return cache.read_text(encoding="utf-8")
    import requests
    for url in urls:
        try:
            response = requests.get(url, headers=NSE_HEADERS, timeout=30)
            response.raise_for_status()
            if "html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("returned HTML")
            cache.write_text(response.text, encoding="utf-8")
            return response.text
        except Exception as error:
            logger.warning("NSE download failed (%s): %s", url, error)
    if cache.exists():
        logger.warning("Using an older cached copy of %s.", cache_name)
        return cache.read_text(encoding="utf-8")
    return None


def parse_constituents(text: Optional[str]) -> List[str]:
    if not text:
        return []
    frame = pd.read_csv(io.StringIO(text))
    columns = {c.strip().lower(): c for c in frame.columns}
    symbol_col = columns.get("symbol")
    if not symbol_col:
        return []
    return [s for s in (str(v).strip().upper() for v in frame[symbol_col]) if s]
