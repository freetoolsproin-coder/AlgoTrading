"""
NSE / BSE trading calendar: weekends + the holidays in config/nse_holidays.csv, and the manual event
calendar (config/events.csv) that blocks new entries around RBI policy, budget, etc.

    is_trading_day(d)         regular session (09:15-15:30) on date d?
    holiday_name(d)           'Dussehra', 'Weekend' or None
    previous_trading_day(d)   last trading day strictly before d
    next_trading_day(d)       first trading day strictly after d
    expiry_on_or_before(d)    d if it is a trading day, else the trading day before (expiry shifts)
    has_year(y)               does the CSV cover year y? (preflight warns if not)

Special sessions (Budget Sunday, Muhurat) are listed for information; the engine does not trade them.
The CSV is re-read when it changes, so a running process picks up a new year's list.
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from src.config.settings import HOLIDAYS_FILE
from src.utils.numbers import hhmm

CALENDAR_FILE = str(HOLIDAYS_FILE)
logger = logging.getLogger("scanner.calendar")

_cache = {"path": None, "mtime": None, "holidays": {}, "special": {}}


def _load(path: str = CALENDAR_FILE):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    if _cache["path"] == path and _cache["mtime"] == mtime:
        return _cache["holidays"], _cache["special"]
    holidays, special = {}, {}
    if mtime is not None:
        with open(path, "r", encoding="utf-8-sig") as handle:
            rows = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
        for row in csv.DictReader(rows):
            try:
                day = date.fromisoformat((row.get("date") or "").strip())
            except ValueError:
                continue
            name = (row.get("description") or "holiday").strip()
            kind = (row.get("type") or "holiday").strip().lower()
            (special if kind == "special_session" else holidays)[day] = name
    _cache.update(path=path, mtime=mtime, holidays=holidays, special=special)
    return holidays, special


def holiday_name(day: date, path: str = CALENDAR_FILE):
    holidays, _ = _load(path)
    if day in holidays:
        return holidays[day]
    if day.weekday() >= 5:
        return "Weekend"
    return None


def is_trading_day(day: date, path: str = CALENDAR_FILE) -> bool:
    return holiday_name(day, path) is None


def special_session(day: date, path: str = CALENDAR_FILE):
    return _load(path)[1].get(day)


def previous_trading_day(day: date, path: str = CALENDAR_FILE) -> date:
    day -= timedelta(days=1)
    while not is_trading_day(day, path):
        day -= timedelta(days=1)
    return day


def next_trading_day(day: date, path: str = CALENDAR_FILE) -> date:
    day += timedelta(days=1)
    while not is_trading_day(day, path):
        day += timedelta(days=1)
    return day


def expiry_on_or_before(day: date, path: str = CALENDAR_FILE) -> date:
    return day if is_trading_day(day, path) else previous_trading_day(day, path)


def has_year(year: int, path: str = CALENDAR_FILE) -> bool:
    holidays, special = _load(path)
    return any(d.year == year for d in list(holidays) + list(special))


# =============================================================================
# Event calendar (config/events.csv): date,time,name,impact. Blank time = whole day.
# =============================================================================
def load_events(path) -> List[Tuple[date, Optional[object], str]]:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or line.lower().startswith("date,"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                try:
                    day = date.fromisoformat(parts[0])
                except ValueError:
                    logger.warning("events file: ignoring a line with an invalid date: %s", line[:60])
                    continue
                at = None
                if len(parts) > 1 and parts[1]:
                    try:
                        at = hhmm(parts[1])
                    except ValueError:
                        logger.warning("events file: invalid time on %s, treated as all day", parts[0])
                out.append((day, at, parts[2] if len(parts) > 2 else "event"))
    except FileNotFoundError:
        logger.info("No events file at %s: no event blocks.", path)
    return out


def event_now(events, now: pd.Timestamp, before_min=30, after_min=30) -> Optional[str]:
    for day, at, name in events:
        if day != now.date():
            continue
        if at is None:
            return f"{name} (all day)"
        t = pd.Timestamp(datetime.combine(day, at)).tz_localize(now.tz)
        if t - pd.Timedelta(minutes=before_min) <= now <= t + pd.Timedelta(minutes=after_min):
            return f"{name} at {at:%H:%M}"
    return None
