"""
NIFTY 50 stock universe for the stock-options service.

Constituents come from the official NSE list (ind_nifty50list.csv, cached for a day in data/cache/), which
also gives the ISIN -> Upstox key "NSE_EQ|<ISIN>" and the industry. If NSE cannot be reached and there is no
cached copy, FALLBACK_SYMBOLS is used and the keys are looked up in the Upstox instrument master.

Option contracts (expiries, lot sizes, strikes) are NOT fixed here: they are read every day from Upstox
/v2/option/contract for each stock (see data.py).

This module is separate from src/config/instruments.py on purpose: the index scanner stays NIFTY / SENSEX
only, this service handles stock options only.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("scanner.stocks.universe")

NIFTY50_URLS = ("https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
                "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv")
NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"

# Used only when the NSE list cannot be downloaded and no cached copy exists. NSE changes the index
# twice a year, so the downloaded list always wins.
FALLBACK_SYMBOLS = (
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TMPV", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
)


@dataclass(frozen=True)
class Stock:
    symbol: str
    name: str
    industry: str
    key: str            # Upstox equity key, e.g. NSE_EQ|INE002A01018
    nifty50: bool = True

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "name": self.name, "industry": self.industry, "key": self.key,
                "nifty50": self.nifty50}


def parse_nifty50_csv(text: Optional[str]) -> List[Stock]:
    """NSE CSV columns: Company Name, Industry, Symbol, Series, ISIN Code."""
    if not text:
        return []
    try:
        frame = pd.read_csv(io.StringIO(text))
    except Exception:            # malformed download
        return []
    cols = {c.strip().lower(): c for c in frame.columns}
    sym, isin = cols.get("symbol"), cols.get("isin code") or cols.get("isin")
    if not sym or not isin:
        return []
    name, industry = cols.get("company name"), cols.get("industry")
    out = []
    for _, row in frame.iterrows():
        symbol = str(row[sym]).strip().upper()
        code = str(row[isin]).strip().upper()
        if not symbol or len(code) != 12:
            continue
        out.append(Stock(symbol=symbol, name=str(row[name]).strip() if name else symbol,
                         industry=str(row[industry]).strip() if industry else "",
                         key=f"NSE_EQ|{code}"))
    return out


def load_universe(fetch_csv=None, equity_keys: Optional[Dict[str, str]] = None) -> List[Stock]:
    """NIFTY 50 stocks. `fetch_csv(urls, cache_name)` returns the CSV text (src.broker.client.fetch_nse_csv);
    `equity_keys` (symbol -> NSE_EQ key from the Upstox master) is the fallback for the keys."""
    if fetch_csv is None:
        from src.broker.client import fetch_nse_csv as fetch_csv
    stocks: List[Stock] = []
    try:
        stocks = parse_nifty50_csv(fetch_csv(NIFTY50_URLS, "nifty50_constituents.csv"))
    except Exception as error:
        logger.warning("NIFTY 50 list download failed: %s", error)
    if len(stocks) >= 45:
        return sorted(stocks, key=lambda s: s.symbol)
    logger.warning("Using the built-in NIFTY 50 fallback list (NSE list unavailable).")
    if equity_keys is None:
        equity_keys = master_equity_keys()
    missing = [s for s in FALLBACK_SYMBOLS if s not in equity_keys]
    if missing:
        logger.warning("No Upstox key for %s; skipped.", ", ".join(missing))
    return [Stock(symbol=s, name=s, industry="", key=equity_keys[s]) for s in FALLBACK_SYMBOLS if s in equity_keys]


def master_equity_keys() -> Dict[str, str]:
    """symbol -> NSE_EQ key from the (daily cached) Upstox instrument master."""
    from datetime import date

    from src.broker.client import Instruments, load_master_rows
    try:
        return Instruments(load_master_rows(), date.today()).equity_keys
    except Exception as error:
        logger.warning("Upstox instrument master unavailable: %s", error)
        return {}
