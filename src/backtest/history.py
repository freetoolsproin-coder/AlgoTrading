"""
1-minute index history for the backtest and the ML trainer (NIFTY, SENSEX and India VIX only).

Upstox v3 historical candles are fetched one calendar month at a time and cached as CSV files in
data/history/ (<KEY>_1m_<YYYY-MM>.csv). Completed months are read from the cache; the current month is
refreshed. Needs an Upstox access token only when something is missing from the cache.

Historical expiry rules (used to price modelled options with the right time to expiry) are from memory
of the exchange circulars: check them before relying on expiry-day results.
"""
from __future__ import annotations

import logging
import time as clock
from datetime import date, time, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.config.instruments import VIX_KEY, get_instrument
from src.utils import market_calendar

logger = logging.getLogger("scanner.backtest")

V3_BASE = "https://api.upstox.com/v3"
# Fallbacks for the backtest only. Live trading reads strike steps / lot sizes from the chain / master.
DEFAULT_STRIKE_INTERVAL = {"NIFTY": 50, "SENSEX": 100}
DEFAULT_LOT_SIZE = {"NIFTY": 65, "SENSEX": 20}
# (first day the rule applies, cycle, weekday Mon=0); the newest rule matches today's market.
EXPIRY_RULES = {
    "NIFTY": [(date(2000, 1, 1), "weekly", 3), (date(2025, 9, 1), "weekly", 1)],
    "SENSEX": [(date(2000, 1, 1), "weekly", 4), (date(2025, 1, 1), "weekly", 1), (date(2025, 9, 1), "weekly", 3)],
}


def index_key(name: str) -> str:
    return get_instrument(name).index_key


def to_minute_frame(records) -> pd.DataFrame:
    """List of {'datetime', open, high, low, close[, volume]} dicts or a frame -> IST-indexed frame."""
    frame = pd.DataFrame(records).copy() if not isinstance(records, pd.DataFrame) else records.copy()
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if "datetime" in frame.columns:
        stamps = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
        frame = frame.assign(datetime=stamps).dropna(subset=["datetime"]).set_index("datetime")
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("Asia/Kolkata") if index.tz is None else index.tz_convert("Asia/Kolkata")
    frame.index = index
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = 0.0 if column == "volume" else np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["volume"] = frame["volume"].fillna(0.0)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return frame[["open", "high", "low", "close", "volume"]]


def session_only(frame: pd.DataFrame) -> pd.DataFrame:
    clock_times = frame.index.time
    return frame[(clock_times >= time(9, 15)) & (clock_times < time(15, 30))]


def cache_name(instrument_key: str, month: date) -> str:
    return f"{instrument_key.replace('|', '_').replace(' ', '_')}_1m_{month:%Y-%m}.csv"


def _months(start: date, end: date):
    month = date(start.year, start.month, 1)
    while month <= end:
        yield month
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)


def cache_covers(cache_dir: Path, instrument_key: str, start: date, end: date) -> bool:
    return all((Path(cache_dir) / cache_name(instrument_key, m)).exists() for m in _months(start, end))


def read_cached(instrument_key: str, start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    parts = [to_minute_frame(pd.read_csv(p)) for p in
             (Path(cache_dir) / cache_name(instrument_key, m) for m in _months(start, end)) if p.exists()]
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.concat(parts).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame[(frame.index.date >= start) & (frame.index.date <= end)]
    return session_only(frame)


def fetch_minutes(client, instrument_key: str, start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    """Download missing / current months through the rate-limited UpstoxClient (v3 endpoint), then read."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()
    for month in _months(start, end):
        next_month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
        path = cache_dir / cache_name(instrument_key, month)
        if path.exists() and next_month <= today:
            continue
        to_day = min(next_month - timedelta(days=1), today - timedelta(days=1))
        if to_day < month:
            continue
        logger.info("fetching %s %s ...", instrument_key, f"{month:%Y-%m}")
        payload = client.get(f"/historical-candle/{quote(instrument_key, safe='')}/minutes/1/"
                             f"{to_day.isoformat()}/{month.isoformat()}", base=V3_BASE)
        candles = (payload.get("data") or {}).get("candles") or []
        if candles:
            frame = to_minute_frame(pd.DataFrame([row[:6] for row in candles],
                                                 columns=["datetime", "open", "high", "low", "close", "volume"]))
            frame.reset_index().to_csv(path, index=False)
        clock.sleep(0.35)
    return read_cached(instrument_key, start, end, cache_dir)


def load_history(name: str, start: date, end: date, cache_dir: Path, client=None) -> tuple:
    """(index minutes, India VIX minutes). Uses the cache; downloads only what is missing."""
    key = index_key(name)
    vix_start = start - timedelta(days=10)
    complete = cache_covers(cache_dir, key, start, end) and cache_covers(cache_dir, VIX_KEY, vix_start, end)
    if not complete:
        if client is None:
            from src.broker.client import UpstoxClient
            client = UpstoxClient()
        return (fetch_minutes(client, key, start, end, cache_dir),
                fetch_minutes(client, VIX_KEY, vix_start, end, cache_dir))
    return read_cached(key, start, end, cache_dir), read_cached(VIX_KEY, vix_start, end, cache_dir)


def _next_weekday(day: date, weekday: int) -> date:
    return day + timedelta(days=(weekday - day.weekday()) % 7)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def expiry_for(day: date, name: str) -> date:
    """The contract a trader buys on `day`: nearest weekly / monthly expiry on or after it (holiday-shifted)."""
    cycle, weekday = "weekly", 1
    for start, rule_cycle, rule_weekday in EXPIRY_RULES[get_instrument(name).name]:
        if day >= start:
            cycle, weekday = rule_cycle, rule_weekday
    shift = market_calendar.expiry_on_or_before
    if cycle == "monthly":
        expiry = shift(_last_weekday_of_month(day.year, day.month, weekday))
        if expiry < day:
            year, month = day.year + (day.month == 12), day.month % 12 + 1
            expiry = shift(_last_weekday_of_month(year, month, weekday))
        return expiry
    expiry = shift(_next_weekday(day, weekday))
    if expiry < day:  # a holiday moved this week's expiry before today
        expiry = shift(_next_weekday(day + timedelta(days=1), weekday))
    return expiry

