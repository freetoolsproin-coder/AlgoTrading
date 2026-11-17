"""
Market data service for NIFTY and SENSEX: one snapshot per scanner cycle.

LiveProvider (Upstox, no Redis / data hub needed)
    once a day   instrument master -> index keys, option expiries + lot sizes, near-month index futures;
                 daily history (levels), ~7 sessions of 1-minute history (EMA200 / swings),
                 NIFTY 50 constituents (breadth)
    every cycle  quotes (index, futures, India VIX, constituents), today's 1-minute candles (index + futures),
                 the option chain of the chosen expiry
    WebSocket    when src/broker/websocket_feed.py is running, its index / futures / VIX last prices replace
                 the REST quote price whenever they are newer
Index spot has no volume: futures supply volume, VWAP and OI. Nothing is estimated or filled in: a value
that is missing stays missing (NaN) and the strategy marks that family "no data".

MockProvider lives in src/broker/mock_data.py (synthetic, labelled MOCK everywhere).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.broker.client import (Instruments, UpstoxAuthError, UpstoxClient, fetch_nse_csv, load_master_rows,
                               parse_constituents)
from src.config.instruments import INSTRUMENTS, VIX_KEY, get_instrument
from src.config.settings import PBConfig
from src.options.chain import chain_frame, choose_expiry, strike_step
from src.utils.numbers import finite, fnum

TZ = "Asia/Kolkata"
logger = logging.getLogger("scanner.data")
EMPTY = ["open", "high", "low", "close", "volume", "oi"]


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def empty_minutes() -> pd.DataFrame:
    """An empty candle frame that still has a tz-aware time index (so date filters work on it)."""
    return pd.DataFrame(columns=EMPTY, index=pd.DatetimeIndex([], tz=TZ), dtype=float)


def to_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Any candle frame with a 'datetime' column (or a datetime index) -> tz-aware IST minute frame."""
    if frame is None or len(frame) == 0:
        return empty_minutes()
    f = frame.copy()
    if "datetime" in f.columns:
        f["datetime"] = pd.to_datetime(f["datetime"], utc=True, errors="coerce", format="ISO8601")
        f = f.dropna(subset=["datetime"]).set_index("datetime")
    if f.index.tz is None:
        f.index = f.index.tz_localize(TZ)
    else:
        f.index = f.index.tz_convert(TZ)
    for c in EMPTY:
        if c not in f:
            f[c] = np.nan
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f[EMPTY].sort_index()
    return f[~f.index.duplicated(keep="last")]


def breadth_from_quotes(symbols: Dict[str, str], quotes: Dict[str, dict]) -> dict:
    """symbols: symbol -> instrument key. % above own VWAP (average price) and % above previous close."""
    above = adv = n_vwap = n_adv = 0
    for key in symbols.values():
        q = quotes.get(key) or {}
        ltp, vwap, prev = fnum(q.get("ltp")), fnum(q.get("vwap")), fnum(q.get("prev_close"))
        if finite(ltp) and finite(vwap) and vwap > 0:
            n_vwap += 1
            above += ltp > vwap
        if finite(ltp) and finite(prev) and prev > 0:
            n_adv += 1
            adv += ltp > prev
    if max(n_vwap, n_adv) < 5:
        return {}
    return {"pct_above_vwap": 100.0 * above / n_vwap if n_vwap >= 5 else np.nan,
            "pct_advancing": 100.0 * adv / n_adv if n_adv >= 5 else np.nan, "count": max(n_vwap, n_adv)}


def trade_time(value) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    try:
        v = float(value)
        return pd.Timestamp(v / 1000 if v > 1e11 else v, unit="s", tz="UTC").tz_convert(TZ)
    except (TypeError, ValueError):
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        return None if pd.isna(ts) else ts.tz_convert(TZ)


@dataclass
class IndexSnapshot:
    name: str
    spot_minutes: pd.DataFrame
    fut_minutes: pd.DataFrame
    daily: pd.DataFrame
    ltp: float = np.nan
    ltp_time: Optional[pd.Timestamp] = None
    day_open: float = np.nan
    prev_close: float = np.nan
    chain: Optional[pd.DataFrame] = None
    chain_time: Optional[pd.Timestamp] = None
    chain_source: str = ""
    expiry: Optional[date] = None
    nearest_expiry: Optional[date] = None
    lot_size: int = 0
    step: float = np.nan
    exchange: str = "NSE_FO"
    errors: List[str] = field(default_factory=list)
    sources: Dict[str, str] = field(default_factory=dict)
    mock: bool = False

    @property
    def is_expiry_day(self) -> bool:
        return self.nearest_expiry is not None and self.ltp_time is not None and self.nearest_expiry == self.ltp_time.date()


@dataclass
class MarketSnapshot:
    time: pd.Timestamp
    indices: Dict[str, IndexSnapshot]
    vix: float = np.nan
    vix_prev_close: float = np.nan
    mock: bool = False
    errors: List[str] = field(default_factory=list)
    breadth: Dict[str, dict] = field(default_factory=dict)   # index -> {pct_above_vwap, pct_advancing, count, proxy}

    @property
    def vix_change_pct(self) -> float:
        if finite(self.vix) and finite(self.vix_prev_close) and self.vix_prev_close > 0:
            return (self.vix / self.vix_prev_close - 1) * 100
        return np.nan


class LiveProvider:
    mock = False

    def __init__(self, names, cfg: PBConfig, client: Optional[UpstoxClient] = None, live_feed=None,
                 master_loader=load_master_rows, constituents_loader=None):
        self.names = [get_instrument(n).name for n in names]
        self.cfg = cfg
        self.client = client or UpstoxClient()
        self.live_feed = live_feed            # WebSocketFeed (optional): fresher last prices
        self.master_loader = master_loader
        self.constituents_loader = constituents_loader or self._load_constituents
        self.instruments: Optional[Instruments] = None
        self.ref_day: Optional[date] = None
        self.keys: Dict[str, dict] = {}
        self.vix_key = VIX_KEY
        self.daily: Dict[str, pd.DataFrame] = {}
        self.history: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.chain_cache: Dict[Tuple[str, date], Tuple[pd.Timestamp, pd.DataFrame]] = {}
        self.intraday_cache: Dict[Tuple[str, str], Tuple[pd.Timestamp, pd.DataFrame]] = {}
        self.constituents: Dict[str, Dict[str, str]] = {}

    def now(self) -> pd.Timestamp:
        return now_ist()

    def subscription_keys(self) -> List[str]:
        """Instrument keys the WebSocket should stream (indices, near futures, VIX). Empty before day setup."""
        keys = [k["index"] for k in self.keys.values()] + [k["fut"] for k in self.keys.values() if k["fut"]]
        return keys + [self.vix_key] if keys else []

    # ---------- once a day ----------
    def refresh_reference(self, now: pd.Timestamp):
        if self.ref_day == now.date() and self.instruments is not None:
            return
        inst_master = Instruments(self.master_loader(today=now.date()), now.date())
        keys = {}
        for name in self.names:
            inst = INSTRUMENTS[name]
            expiries, lots = inst_master.option_expiries(inst)
            keys[name] = {"index": inst_master.resolve_index(inst), "fut": inst_master.near_future(inst),
                          "expiries": expiries, "lots": lots, "segment": inst.fo_segment}
            if not expiries:
                logger.error("[%s] no option expiries found in the instrument master for %s on %s: option "
                             "selection is unavailable today.", name, inst.option_name, inst.fo_segment)
        self.vix_key = inst_master.resolve_vix()
        self.constituents = self.constituents_loader(inst_master)
        daily, history = {}, {}
        for name, k in keys.items():
            try:
                daily[name] = self.client.daily_candles(k["index"], now.date() - timedelta(days=420),
                                                        now.date() - timedelta(days=1)).set_index("datetime")
            except UpstoxAuthError:
                raise
            except Exception as error:
                logger.warning("[%s] daily history failed: %s", name, error)
                daily[name] = empty_minutes()
        self.instruments, self.keys, self.daily, self.history = inst_master, keys, daily, history
        self.chain_cache, self.intraday_cache = {}, {}
        self.ref_day = now.date()
        logger.info("Reference data ready: %s", ", ".join(
            f"{n} (expiries {len(k['expiries'])}, futures {'yes' if k['fut'] else 'no'})" for n, k in keys.items()))

    def _load_constituents(self, inst_master: Instruments) -> Dict[str, Dict[str, str]]:
        out = {}
        for name in self.names:
            inst = INSTRUMENTS[name]
            source = INSTRUMENTS[inst.breadth_proxy] if inst.breadth_proxy else inst
            if source.name in out or not source.constituents_url:
                continue
            try:
                syms = parse_constituents(fetch_nse_csv(source.constituents_url, f"{source.name.lower()}_constituents.csv"))
            except Exception as error:
                logger.warning("[%s] constituent list unavailable (breadth off): %s", source.name, error)
                syms = []
            out[source.name] = {s: inst_master.equity_keys[s] for s in syms if s in inst_master.equity_keys}
        return out

    def _prior_minutes(self, name: str, kind: str, key: Optional[str], now: pd.Timestamp) -> pd.DataFrame:
        """1-minute candles of the previous ~7 sessions (EMA200 / swing levels), fetched once a day."""
        cache_key = (name, kind)
        if cache_key in self.history:
            return self.history[cache_key]
        frame = empty_minutes()
        if key:
            try:
                to_day = now.date() - timedelta(days=1)
                frame = to_minutes(self.client.minute_candles(key, to_day - timedelta(days=12), to_day))
            except UpstoxAuthError:
                raise
            except Exception as error:
                logger.warning("[%s] %s minute history failed: %s", name, kind, error)
                return frame  # not cached: retried next cycle
        self.history[cache_key] = frame
        return frame

    def _today_minutes(self, name: str, kind: str, key: Optional[str], now: pd.Timestamp) -> pd.DataFrame:
        if not key:
            return empty_minutes()
        cached = self.intraday_cache.get((name, kind))
        if cached and (now - cached[0]).total_seconds() < 25:
            return cached[1]
        frame = to_minutes(self.client.minute_candles_today(key))
        frame = frame[frame.index.date == now.date()]
        self.intraday_cache[(name, kind)] = (now, frame)
        return frame

    def chain(self, name: str, expiry: date, now: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Timestamp]:
        cached = self.chain_cache.get((name, expiry))
        if cached and (now - cached[0]).total_seconds() < 55:
            return cached[1], cached[0]
        frame = chain_frame(self.client.option_chain(self.keys[name]["index"], expiry))
        self.chain_cache[(name, expiry)] = (now, frame)
        return frame, now

    def _live_price(self, key: str, rest_ltp: float, rest_time: Optional[pd.Timestamp]):
        if self.live_feed is None or not key:
            return rest_ltp, rest_time, None
        tick = self.live_feed.latest(key)
        if tick is None:
            return rest_ltp, rest_time, None
        ltp, ts = tick
        if finite(ltp) and (rest_time is None or ts > rest_time):
            return ltp, ts, "WebSocket"
        return rest_ltp, rest_time, None

    # ---------- every cycle ----------
    def snapshot(self, now: Optional[pd.Timestamp] = None, want_chain: Optional[set] = None) -> MarketSnapshot:
        now = now or self.now()
        self.refresh_reference(now)
        keys = self.subscription_keys() + [key for syms in self.constituents.values() for key in syms.values()]
        errors = []
        quotes = {}
        try:
            quotes = self.client.quotes(keys)
        except UpstoxAuthError:
            raise
        except Exception as error:
            errors.append(f"quotes failed: {type(error).__name__}")
            logger.warning("Quote request failed: %s", error)
        out = {}
        for name, k in self.keys.items():
            snap = IndexSnapshot(name=name, spot_minutes=empty_minutes(), fut_minutes=empty_minutes(),
                                 daily=self.daily.get(name, empty_minutes()), exchange=k["segment"])
            try:
                q = quotes.get(k["index"], {})
                snap.day_open, snap.prev_close = fnum(q.get("open")), fnum(q.get("prev_close"))
                snap.ltp, snap.ltp_time, live_src = self._live_price(k["index"], fnum(q.get("ltp")),
                                                                     trade_time(q.get("trade_time")))
                spot_today = self._today_minutes(name, "spot", k["index"], now)
                snap.sources["spot"] = live_src or "Upstox REST"
                prior = self._prior_minutes(name, "spot", k["index"], now)
                snap.spot_minutes = pd.concat([prior, spot_today]).sort_index() if not prior.empty else spot_today
                snap.spot_minutes = snap.spot_minutes[~snap.spot_minutes.index.duplicated(keep="last")]
                if snap.ltp_time is None and not spot_today.empty:
                    snap.ltp_time = spot_today.index[-1] + pd.Timedelta(minutes=1)
                    snap.sources["spot_time"] = "last candle"
                if k["fut"]:
                    fut_today = self._today_minutes(name, "fut", k["fut"], now)
                    snap.sources["futures"] = "Upstox REST"
                    fprior = self._prior_minutes(name, "fut", k["fut"], now)
                    snap.fut_minutes = pd.concat([fprior, fut_today]).sort_index() if not fprior.empty else fut_today
                    snap.fut_minutes = snap.fut_minutes[~snap.fut_minutes.index.duplicated(keep="last")]
                expiry, nearest = choose_expiry(k["expiries"], now, self.cfg)
                snap.expiry, snap.nearest_expiry = expiry, nearest
                snap.lot_size = int(k["lots"].get(expiry, 0)) if expiry else 0
                if expiry and (want_chain is None or name in want_chain):
                    snap.chain, snap.chain_time = self.chain(name, expiry, now)
                    snap.chain_source = "Upstox REST"
                    snap.step = strike_step(snap.chain)
            except UpstoxAuthError:
                raise
            except Exception as error:
                snap.errors.append(f"{type(error).__name__}: {error}")
                logger.warning("[%s] data error: %s", name, error)
            out[name] = snap
        vq = quotes.get(self.vix_key, {})
        vix, _, _ = self._live_price(self.vix_key, fnum(vq.get("ltp")), trade_time(vq.get("trade_time")))
        breadth = {}
        for name in self.keys:
            inst = INSTRUMENTS[name]
            source = inst.breadth_proxy or name
            if self.constituents.get(source):
                b = breadth_from_quotes(self.constituents[source], quotes)
                if b:
                    b["proxy"] = source != name
                    breadth[name] = b
        return MarketSnapshot(time=now, indices=out, vix=vix, vix_prev_close=fnum(vq.get("prev_close")),
                              errors=errors, breadth=breadth)
