"""
Option chain handling: the Upstox chain response -> one row per strike with ce_* / pe_* columns,
dynamic expiry choice, and the strike interval read from the chain itself.

Row layout (used by selection.py, paper trading and the dashboard):
    strike, ce_ltp, ce_close, ce_oi, ce_prev_oi, ce_volume, ce_bid, ce_ask, ce_iv, ce_delta, ce_gamma,
    ce_theta, ce_vega, ce_key, ce_oi_chg, ce_px_chg, and the same for pe_*.
`*_key` is the option contract's own Upstox instrument key (NSE_FO|... / BSE_FO|...): a selected
contract is always a real chain row, never the index.
"""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import PBConfig
from src.utils.numbers import finite, fnum

TZ = "Asia/Kolkata"
_LEG_FIELDS = {"ltp": "ltp", "close": "close_price", "oi": "oi", "prev_oi": "prev_oi",
               "volume": "volume", "bid": "bid_price", "ask": "ask_price"}
_GREEK_FIELDS = {"iv": "iv", "delta": "delta", "gamma": "gamma", "theta": "theta", "vega": "vega"}
CHAIN_COLUMNS = [f"{side}_{name}" for side in ("ce", "pe") for name in list(_LEG_FIELDS) + list(_GREEK_FIELDS)]


def chain_frame(raw) -> pd.DataFrame:
    """Upstox /v2/option/chain response: [{strike_price, call_options: {...}, put_options: {...}}]."""
    rows: Dict[float, dict] = {}

    def add(strike, side, option):
        strike = fnum(strike)
        if not finite(strike) or not isinstance(option, dict):
            return
        market = option.get("market_data") or {}
        greeks = option.get("option_greeks") or {}
        record = rows.setdefault(strike, {"strike": strike})
        for name, source in _LEG_FIELDS.items():
            record[f"{side}_{name}"] = fnum(market.get(source))
        for name, source in _GREEK_FIELDS.items():
            record[f"{side}_{name}"] = fnum(greeks.get(source))
        record[f"{side}_key"] = option.get("instrument_key")

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            for source, side in (("call_options", "ce"), ("put_options", "pe")):
                if item.get(source):
                    add(item.get("strike_price"), side, item[source])
    frame = pd.DataFrame(list(rows.values()))
    if frame.empty:
        return pd.DataFrame(columns=["strike"] + CHAIN_COLUMNS)
    for column in CHAIN_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame = frame.sort_values("strike").reset_index(drop=True)
    for side in ("ce", "pe"):
        frame[f"{side}_oi"] = frame[f"{side}_oi"].fillna(0.0)
        frame[f"{side}_volume"] = frame[f"{side}_volume"].fillna(0.0)
        frame[f"{side}_oi_chg"] = frame[f"{side}_oi"] - frame[f"{side}_prev_oi"]  # NaN if no prev_oi
        frame[f"{side}_px_chg"] = frame[f"{side}_ltp"] - frame[f"{side}_close"]
    return frame


def choose_expiry(expiries: List[date], now: pd.Timestamp, cfg: PBConfig) -> Tuple[Optional[date], Optional[date]]:
    """(expiry to trade, nearest live expiry). Skips 0DTE unless allowed, and expiries with too few hours left."""
    live = sorted(e for e in expiries if e >= now.date())
    if not live:
        return None, None
    nearest = live[0]
    for e in live:
        end = pd.Timestamp(datetime.combine(e, time(15, 30))).tz_localize(TZ)
        hours = (end - now).total_seconds() / 3600
        if e == now.date() and not cfg.allow_0dte:
            continue
        if hours < cfg.min_hours_to_expiry:
            continue
        return e, nearest
    return live[-1], nearest


def strike_step(chain: Optional[pd.DataFrame]) -> float:
    if chain is None or chain.empty:
        return np.nan
    strikes = chain["strike"].drop_duplicates().sort_values()
    diffs = strikes.diff().dropna()
    return float(diffs.mode().iloc[0]) if len(diffs) else np.nan
