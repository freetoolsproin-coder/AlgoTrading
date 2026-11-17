"""
Pre-breakout scanner for the NIFTY 50 stocks: finds stocks COILING just under resistance (CE) or just above
support (PE) BEFORE the move, gives the buy-stop / sell-stop trigger, and plans the trade two ways:
  * intraday equity (MIS): quantity from a fixed rupee risk, stop, Target 1 / Target 2, charges included
  * stock option: CE or PE (delta ~0.55, monthly expiry), premium expected at the trigger, premium SL / targets

Everything is computed on COMPLETED 15-minute bars (a forming bar is never used for a stage) and is causal,
so the same code drives the live service and the replay backtest (src/stocks/breakout_backtest.py).

Parameters (all in [STOCK_BREAKOUT] of config/scanner.ini) follow widely used practitioner rules; they are
starting values, not proven edges - check them with the backtest and live paper results.
  Levels       15-minute opening range (Crabel ORB), previous-day high / low, day high / low so far,
               the 6-bar base (90 min) when it is tighter than 2 ATR, 20-day high / low. Levels closer than
               0.25 ATR merge into one zone.
  Compression  TTM squeeze: Bollinger(20, 2) inside Keltner(20, 1.5 ATR) (J. Carter); NR7 / NR4 bar
               (Crabel); tight base; volume dry-up inside the base (base volume <= 0.8x the 20 bars before,
               Minervini VCP); yesterday NR7 or inside day on the daily chart.
  Pressure     rising lows into resistance (falling highs into support), repeated tests of the level,
               close in the upper (lower) part of the base.
  Trend        price vs session VWAP, EMA 9 / 21 on 15m, daily close vs EMA 20 / 50.
  Rel. strength stock % change today minus NIFTY % change (full points at +/-0.75 %); sector breadth.
  Volume       relative volume: today's cumulative volume vs the average at the same time of day over the last
               10 sessions (full points at 1.5x).
  Penalties    RSI(14) beyond 75 / 25 (chasing), more than 2.5 ATR from VWAP, the biggest CE (PE) OI strike
               sitting just beyond the level (option writers defending it), a gap > 2 % in the trade direction.

Score 0-100 (setup strength, NOT a probability):
  Location 20, Level quality 10, Compression 20, Pressure 15, Trend 15, Relative strength 10, Volume 5,
  Market / sector 5. A family without data is left out and the score is rescaled.

Stages (per stock and side):
  WATCH      score >= 55 and within 1.0 ATR(15m) of the level
  READY      score >= 70, within 0.35 ATR, compression >= 8/20, inside the entry window - the "before breakout"
             suggestion: place the trigger, do not enter early
  TRIGGERED  live price crossed the trigger inside a forming bar (shown; not traded with entry_mode = close)
  CONFIRMED  a completed 15m bar closed beyond the level + buffer (max(0.10 ATR, 0.05 %)) with a body >= 50 %
             of its range, the close in the outer 40 % of the bar and volume >= 1.5x the 20-bar average
  BREAKING   closed beyond the level but without that quality (not traded)
  MISSED     the close is already more than 0.8 ATR beyond the level - do not chase
  FAILED     after a breakout, a close back inside the level by 0.25 ATR
A breakout is tradable only if the same level was READY first (require_prior_ready).

Trade plan (CE / long shown; PE is the mirror):
  trigger = level + buffer; stop = base low (or the low of the last 3 bars) - 0.1 ATR, kept between 0.6 and
  1.2 ATR from the trigger; T1 = 1.5 R, T2 = 2.5 R, pulled in before the next zone; T1 must give >= 1.2 R and
  fit the range the stock can still move today. After T1 the stop moves to cost. No entries after 14:30,
  square-off 15:15, at most `max_trades` paper trades a day.

Paper trading only - this module never places orders. Not investment advice.
"""
from __future__ import annotations

import configparser
import csv
import json
import logging
import math
import os
import threading
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.scanner.indicators import atr, ema, relative_volume, rsi, session_vwap
from src.utils.numbers import finite

logger = logging.getLogger("scanner.stocks.breakout")

BAR = pd.Timedelta(minutes=15)
STAGES = ("CONFIRMED", "TRIGGERED", "READY", "BREAKING", "WATCH", "MISSED", "FAILED", "NONE")
FAMILY_MAX = {"Location": 20, "Level quality": 10, "Compression": 20, "Pressure": 15, "Trend": 15,
              "Relative strength": 10, "Volume": 5, "Market / sector": 5}
LEVEL_WEIGHT = {"Opening range": 2, "Previous day": 2, "Day high/low": 1, "Base": 2, "20-day": 3}


@dataclass(frozen=True)
class BreakoutConfig:
    enabled: bool = True
    # levels / compression
    box_bars: int = 6
    max_box_atr: float = 2.0
    merge_atr: float = 0.25
    touch_atr: float = 0.15
    bb_length: int = 20
    bb_std: float = 2.0
    kc_mult: float = 1.5
    dryup_ratio: float = 0.8
    # stages
    watch_atr: float = 1.0
    ready_atr: float = 0.35
    watch_score: float = 55.0
    ready_score: float = 70.0
    min_compression: float = 8.0
    buffer_atr: float = 0.10
    buffer_pct: float = 0.05
    chase_atr: float = 0.8
    fail_atr: float = 0.25
    confirm_volume: float = 1.5
    confirm_body_pct: float = 50.0
    confirm_close_pct: float = 60.0
    require_prior_ready: bool = True
    require_vwap_side: bool = True
    # context / penalties
    rs_full_pct: float = 0.75
    rvol_full: float = 1.5
    rvol_days: int = 10
    min_atr_pct: float = 0.15
    gap_pct: float = 2.0
    gap_wait: str = "09:45"
    rsi_extreme: float = 75.0
    vwap_stretch_atr: float = 2.5
    # plan
    min_stop_atr: float = 0.6
    max_stop_atr: float = 1.2
    rr_t1: float = 1.5
    rr_t2: float = 2.5
    min_rr: float = 1.2
    # intraday equity
    capital: float = 500000.0
    risk_pct: float = 0.5
    leverage: float = 5.0
    # session
    entry_start: str = "09:30"
    entry_end: str = "14:30"
    square_off: str = "15:15"
    max_trades: int = 6
    entry_mode: str = "close"          # close = on a CONFIRMED 15m close; touch = when the trigger trades
    notify: bool = False               # Telegram for READY / CONFIRMED (TELEGRAM_BOT_TOKEN / _CHAT_ID in .env)

    @classmethod
    def from_ini(cls, path) -> "BreakoutConfig":
        parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        try:
            parser.read(path, encoding="utf-8")
        except configparser.Error:
            return cls()
        if not parser.has_section("STOCK_BREAKOUT"):
            return cls()
        sec, values = parser["STOCK_BREAKOUT"], {}
        for f in fields(cls):
            if f.name not in sec:
                continue
            raw = sec[f.name].strip()
            kind = f.type if isinstance(f.type, str) else f.type.__name__
            if kind == "bool":
                values[f.name] = raw.lower() in ("1", "true", "yes", "on")
            elif kind == "int":
                values[f.name] = int(float(raw))
            elif kind == "float":
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self):
        problems = []
        if not 0 < self.ready_atr <= self.watch_atr:
            problems.append("ready_atr must be > 0 and <= watch_atr")
        if not 0 <= self.watch_score <= self.ready_score <= 100:
            problems.append("need 0 <= watch_score <= ready_score <= 100")
        if not 0 < self.min_stop_atr <= self.max_stop_atr:
            problems.append("need 0 < min_stop_atr <= max_stop_atr")
        if not 0 < self.rr_t1 <= self.rr_t2:
            problems.append("need 0 < rr_t1 <= rr_t2")
        if self.entry_mode not in ("close", "touch"):
            problems.append("entry_mode must be close or touch")
        if self.capital <= 0 or not 0 < self.risk_pct <= 5 or self.leverage < 1:
            problems.append("capital > 0, 0 < risk_pct <= 5 and leverage >= 1 required")
        if problems:
            raise ValueError("[STOCK_BREAKOUT] invalid: " + "; ".join(problems))


def _t(text: str):
    from src.utils.numbers import hhmm
    return hhmm(text)


def r2(v):
    return round(float(v), 2) if finite(v) else None


def tick(v, size=0.05):
    return round(round(float(v) / size) * size, 2) if finite(v) else None


# =============================================================================
# Features (one row per completed 15-minute bar, causal)
# =============================================================================
def features(bars: pd.DataFrame, daily: Optional[pd.DataFrame], cfg: BreakoutConfig = BreakoutConfig(),
             index_bars: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every column at row i uses only bars <= i (and daily sessions before that bar's date)."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    b = bars[["open", "high", "low", "close", "volume"]].astype(float).copy()
    day = pd.Series(b.index.date, index=b.index)
    h, l, c, v = b["high"], b["low"], b["close"], b["volume"].fillna(0.0)
    f = b.copy()
    f["day"] = day
    f["atr"] = atr(b, 14)
    f["ema9"], f["ema21"] = ema(c, 9), ema(c, 21)
    f["rsi"] = rsi(c, 14)
    f["vwap"] = session_vwap(b)
    mid = c.rolling(cfg.bb_length).mean()
    sd = c.rolling(cfg.bb_length).std(ddof=0)
    kmid, krng = ema(c, cfg.bb_length), atr(b, cfg.bb_length) * cfg.kc_mult
    f["squeeze"] = ((mid + cfg.bb_std * sd) < (kmid + krng)) & ((mid - cfg.bb_std * sd) > (kmid - krng))
    rng = h - l
    f["nr7"] = rng <= rng.rolling(7).min()
    f["nr4"] = rng <= rng.rolling(4).min()
    f["vol_ratio"] = relative_volume(v, 20)
    n = cfg.box_bars
    f["box_hi"] = h.shift(1).rolling(n).max()
    f["box_lo"] = l.shift(1).rolling(n).min()
    base_vol = v.shift(1 + n).rolling(20, min_periods=10).mean()
    f["dryup"] = v.shift(1).rolling(n).mean() / base_vol.replace(0.0, np.nan)
    f["up_lows"] = (l.diff() > 0).astype(float).rolling(4).mean()
    f["down_highs"] = (h.diff() < 0).astype(float).rolling(4).mean()
    f["low3"] = l.rolling(3).min()
    f["high3"] = h.rolling(3).max()
    g = day.groupby(day)
    f["bar_no"] = g.cumcount()
    first_hi, first_lo = h.groupby(day).transform("first"), l.groupby(day).transform("first")
    f["or_hi"] = first_hi.where(f["bar_no"] >= 1)
    f["or_lo"] = first_lo.where(f["bar_no"] >= 1)
    f["hod"] = h.groupby(day).cummax().groupby(day).shift(1)
    f["lod"] = l.groupby(day).cummin().groupby(day).shift(1)
    f["day_open"] = b["open"].groupby(day).transform("first")
    # relative volume at the same time of day
    cumvol = v.groupby(day).cumsum()
    table = pd.DataFrame({"day": day.values, "slot": f["bar_no"].values, "cv": cumvol.values})
    piv = table.pivot_table(index="day", columns="slot", values="cv", aggfunc="last")
    base = piv.shift(1).rolling(cfg.rvol_days, min_periods=3).mean()
    lookup = base.stack()
    keys = pd.MultiIndex.from_arrays([day.values, f["bar_no"].values])
    f["rvol"] = cumvol.values / lookup.reindex(keys).to_numpy()
    f.loc[~np.isfinite(f["rvol"]), "rvol"] = np.nan
    # previous sessions (daily candles strictly before the bar's date)
    _join_daily(f, daily, b)
    f["chg_pct"] = (c / f["pdc"] - 1.0) * 100.0
    f["gap_pct"] = (f["day_open"] / f["pdc"] - 1.0) * 100.0
    # NIFTY context
    f["idx_chg"], f["idx_trend"] = np.nan, np.nan
    if index_bars is not None and not index_bars.empty:
        ib = index_bars[["open", "high", "low", "close"]].astype(float)
        iday = pd.Series(ib.index.date, index=ib.index)
        last_close = ib["close"].groupby(iday).last()
        prev_close = last_close.shift(1)
        ichg = (ib["close"] / iday.map(prev_close) - 1.0) * 100.0
        itrend = np.sign(ib["close"] - ema(ib["close"], 21))
        f["idx_chg"] = ichg.reindex(f.index).to_numpy()
        f["idx_trend"] = itrend.reindex(f.index).to_numpy()
    return f


def _join_daily(f: pd.DataFrame, daily: Optional[pd.DataFrame], b: pd.DataFrame):
    cols = ["pdh", "pdl", "pdc", "d20h", "d20l", "d_ema20", "d_ema50", "d_atr", "d_nr7", "d_inside"]
    for col in cols:
        f[col] = np.nan
    if daily is None or len(daily) < 2:
        days = f["day"]
        last = b["close"].groupby(days).last().shift(1)
        f["pdc"] = days.map(last).to_numpy()
        return
    d = daily.copy()
    d["d"] = pd.to_datetime(d["datetime"]).dt.date
    d = d.drop_duplicates("d", keep="last").sort_values("d").reset_index(drop=True)
    dc = d["close"].astype(float)
    rng = d["high"] - d["low"]
    tr = pd.concat([rng, (d["high"] - dc.shift()).abs(), (d["low"] - dc.shift()).abs()], axis=1).max(axis=1)
    row = pd.DataFrame({
        "d": d["d"], "pdh": d["high"], "pdl": d["low"], "pdc": dc,
        "d20h": d["high"].rolling(20, min_periods=5).max(), "d20l": d["low"].rolling(20, min_periods=5).min(),
        "d_ema20": ema(dc, 20), "d_ema50": ema(dc, 50),
        "d_atr": tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(),
        "d_nr7": rng <= rng.rolling(7).min(),
        "d_inside": (d["high"] <= d["high"].shift()) & (d["low"] >= d["low"].shift()),
    })
    # map each bar date to the LAST session strictly before it
    sessions = row["d"].tolist()
    import bisect
    days = f["day"].tolist()
    pos = [bisect.bisect_left(sessions, x) - 1 for x in days]
    for col in cols:
        values = row[col].to_numpy()
        f[col] = [values[p] if p >= 0 else np.nan for p in pos]
    for col in ("d_nr7", "d_inside"):
        f[col] = f[col].astype(float)


# =============================================================================
# Levels and zones
# =============================================================================
def levels_at(f: pd.DataFrame, i: int, cfg: BreakoutConfig, r=None) -> List[dict]:
    r = f.iloc[i] if r is None else r
    a = r["atr"]
    out = []

    def add(name, family, value):
        if finite(value):
            out.append({"name": name, "family": family, "value": float(value)})
    add("Opening-range high", "Opening range", r["or_hi"])
    add("Opening-range low", "Opening range", r["or_lo"])
    add("Previous-day high", "Previous day", r["pdh"])
    add("Previous-day low", "Previous day", r["pdl"])
    if r["bar_no"] >= 2:
        add("Day high", "Day high/low", r["hod"])
        add("Day low", "Day high/low", r["lod"])
    if finite(r["box_hi"]) and finite(a) and a > 0 and (r["box_hi"] - r["box_lo"]) <= cfg.max_box_atr * a:
        add(f"{cfg.box_bars}-bar base high", "Base", r["box_hi"])
        add(f"{cfg.box_bars}-bar base low", "Base", r["box_lo"])
    add("20-day high", "20-day", r["d20h"])
    add("20-day low", "20-day", r["d20l"])
    return out


def build_zones(levels: List[dict], a: float, merge_atr: float) -> List[dict]:
    zones: List[dict] = []
    for lv in sorted(levels, key=lambda x: x["value"]):
        if zones and lv["value"] - zones[-1]["top"] <= merge_atr * a:
            z = zones[-1]
            z["top"] = max(z["top"], lv["value"])
            z["members"].append(lv)
        else:
            zones.append({"bottom": lv["value"], "top": lv["value"], "members": [lv]})
    for z in zones:
        fam = {m["family"] for m in z["members"]}
        z["strength"] = sum(LEVEL_WEIGHT.get(x, 1) for x in fam)
        z["name"] = " + ".join(m["name"] for m in z["members"])
    return zones


def _touches(f: pd.DataFrame, i: int, level: float, sign: int, a: float, tol_atr: float, lookback: int = 20) -> int:
    col = "high" if sign > 0 else "low"
    values = f[col].to_numpy()[max(0, i - lookback):i]
    return int(np.sum(np.abs(values - level) <= tol_atr * a))


# =============================================================================
# Score
# =============================================================================
def score_side(f: pd.DataFrame, i: int, sign: int, zone: Optional[dict], dist_atr: Optional[float],
               cfg: BreakoutConfig, chain_stats: Optional[dict] = None, sector_up: Optional[float] = None,
               r=None) -> dict:
    r = f.iloc[i] if r is None else r
    a = float(r["atr"])
    close = float(r["close"])
    comps: List[dict] = []
    fam: Dict[str, float] = {}
    nodata = set()

    def add(family, name, points, note):
        fam[family] = fam.get(family, 0.0) + points
        comps.append({"family": family, "name": name, "points": round(points, 1), "note": note})

    # Location
    if zone is not None and dist_atr is not None:
        span = max(cfg.watch_atr - 0.25, 1e-6)
        p = 20.0 * min(max((cfg.watch_atr - max(dist_atr, 0.0)) / span, 0.0), 1.0) if dist_atr >= 0 else 20.0
        add("Location", "Distance to level", p, f"{dist_atr:+.2f} ATR from {zone['name']}")
        level = zone["top"] if sign > 0 else zone["bottom"]
        t = _touches(f, i, level, sign, a, cfg.touch_atr)
        q = min(10.0, 2.0 * zone["strength"] + 1.5 * min(t, 3))
        add("Level quality", "Zone strength + tests", q, f"strength {zone['strength']}, tested {t}x in 20 bars")
    else:
        add("Location", "Distance to level", 0.0, "no level ahead")
        add("Level quality", "Zone strength + tests", 0.0, "no level ahead")
        t = 0
    # Compression
    if bool(r["squeeze"]):
        add("Compression", "TTM squeeze (BB inside KC)", 7.0, "volatility coiled")
    if bool(r["nr7"]):
        add("Compression", "NR7 bar", 3.0, "narrowest range of 7 bars")
    elif bool(r["nr4"]):
        add("Compression", "NR4 bar", 2.0, "narrowest range of 4 bars")
    if finite(r["box_hi"]) and a > 0:
        hb = (r["box_hi"] - r["box_lo"]) / a
        p = 4.0 if hb <= 1.0 else 2.0 if hb <= 1.5 else 0.0
        if p:
            add("Compression", "Tight base", p, f"{cfg.box_bars}-bar base {hb:.2f} ATR tall")
    if finite(r["dryup"]) and r["dryup"] <= cfg.dryup_ratio:
        add("Compression", "Volume dry-up in base", 3.0, f"base volume {r['dryup']:.2f}x prior")
    if r["d_nr7"] == 1 or r["d_inside"] == 1:
        add("Compression", "Daily NR7 / inside day", 3.0, "yesterday was a narrow / inside day")
    fam.setdefault("Compression", 0.0)
    # Pressure
    steps = r["up_lows"] if sign > 0 else r["down_highs"]
    if finite(steps):
        add("Pressure", "Rising lows" if sign > 0 else "Falling highs", 8.0 * float(steps),
            f"{steps * 100:.0f}% of the last 4 steps")
    if t >= 2:
        add("Pressure", "Repeated tests", 4.0, f"level tested {t}x")
    if finite(r["box_hi"]) and r["box_hi"] > r["box_lo"]:
        pos = (close - r["box_lo"]) / (r["box_hi"] - r["box_lo"])
        pos = pos if sign > 0 else 1 - pos
        if pos >= 0.6:
            add("Pressure", "Close near the edge", 3.0,
                "close beyond the base" if pos > 1 else f"close at {pos * 100:.0f}% of the base")
    fam.setdefault("Pressure", 0.0)
    # Trend
    if finite(r["vwap"]):
        if sign * (close - r["vwap"]) > 0:
            add("Trend", "VWAP side", 5.0, f"{'above' if sign > 0 else 'below'} VWAP {r['vwap']:.2f}")
    else:
        nodata.add("vwap")
    if finite(r["ema21"]) and sign * (r["ema9"] - r["ema21"]) > 0:
        add("Trend", "EMA 9/21 (15m)", 4.0, "fast EMA leads")
    if finite(r["d_ema20"]):
        if sign * (r["pdc"] - r["d_ema20"]) > 0:
            add("Trend", "Daily close vs EMA20", 3.0, "daily trend agrees")
        if finite(r["d_ema50"]) and sign * (r["d_ema20"] - r["d_ema50"]) > 0:
            add("Trend", "Daily EMA20 vs EMA50", 3.0, "daily averages aligned")
    fam.setdefault("Trend", 0.0)
    # Relative strength vs NIFTY
    if finite(r["chg_pct"]) and finite(r["idx_chg"]):
        rs = sign * (r["chg_pct"] - r["idx_chg"])
        add("Relative strength", "vs NIFTY today", 10.0 * min(max(rs / cfg.rs_full_pct, 0.0), 1.0),
            f"{r['chg_pct'] - r['idx_chg']:+.2f}% vs NIFTY")
    else:
        nodata.add("Relative strength")
    # Volume
    if finite(r["rvol"]):
        p = 5.0 * min(max((r["rvol"] - 0.8) / max(cfg.rvol_full - 0.8, 1e-6), 0.0), 1.0)
        add("Volume", "Relative volume (time of day)", p, f"{r['rvol']:.2f}x usual by now")
    else:
        nodata.add("Volume")
    # Market / sector
    ctx = 0.0
    have = False
    if finite(r["idx_trend"]) and finite(r["idx_chg"]):
        have = True
        if r["idx_trend"] * sign > 0 and r["idx_chg"] * sign > 0:
            ctx += 3.0
            comps.append({"family": "Market / sector", "name": "NIFTY agrees", "points": 3.0,
                          "note": f"NIFTY {r['idx_chg']:+.2f}%"})
    if sector_up is not None and finite(sector_up):
        have = True
        share = sector_up if sign > 0 else 1 - sector_up
        if share >= 0.6:
            ctx += 2.0
            comps.append({"family": "Market / sector", "name": "Sector breadth", "points": 2.0,
                          "note": f"{share * 100:.0f}% of sector peers agree"})
    if have:
        fam["Market / sector"] = ctx
    else:
        nodata.add("Market / sector")
    # Penalties
    pen = 0.0
    if finite(r["rsi"]) and ((sign > 0 and r["rsi"] > cfg.rsi_extreme) or (sign < 0 and r["rsi"] < 100 - cfg.rsi_extreme)):
        pen += 10
        comps.append({"family": "Penalty", "name": "RSI stretched", "points": -10.0, "note": f"RSI {r['rsi']:.0f}"})
    if finite(r["vwap"]) and a > 0 and sign * (close - r["vwap"]) > cfg.vwap_stretch_atr * a:
        pen += 8
        comps.append({"family": "Penalty", "name": "Far from VWAP", "points": -8.0,
                      "note": f"{(close - r['vwap']) / a:+.1f} ATR from VWAP"})
    if finite(r["gap_pct"]) and sign * r["gap_pct"] > cfg.gap_pct:
        pen += 5
        comps.append({"family": "Penalty", "name": "Gap in trade direction", "points": -5.0,
                      "note": f"gap {r['gap_pct']:+.1f}%"})
    wall = (chain_stats or {}).get("ce_wall" if sign > 0 else "pe_wall")
    if zone is not None and finite(wall) and a > 0:
        level = zone["top"] if sign > 0 else zone["bottom"]
        if 0 <= sign * (wall - level) <= 0.5 * a:
            pen += 5
            comps.append({"family": "Penalty", "name": "OI wall at the level", "points": -5.0,
                          "note": f"highest {'CE' if sign > 0 else 'PE'} OI at {wall:g}"})
    families = {k: min(fam.get(k, 0.0), m) for k, m in FAMILY_MAX.items() if k not in nodata}
    avail = sum(FAMILY_MAX[k] for k in families)
    raw = sum(families.values())
    score = max(0.0, min(100.0, raw / avail * 100.0 if avail else 0.0) - pen)
    return {"score": round(score, 1), "families": {k: round(v, 1) for k, v in families.items()},
            "components": comps, "no_data": sorted(nodata), "penalty": pen}


# =============================================================================
# Stage + plan for one bar
# =============================================================================
def minutes_left(ts: pd.Timestamp) -> float:
    return max(15 * 60 + 30 - (ts.hour * 60 + ts.minute + ts.second / 60.0), 0.0)


def evaluate(f: pd.DataFrame, i: int, cfg: BreakoutConfig = BreakoutConfig(), chain_stats: Optional[dict] = None,
             sector_up: Optional[float] = None, live: bool = True, rows: Optional[list] = None) -> Dict[str, dict]:
    """Both sides for the completed bar i. Decision time = the bar's close (start + 15 min).
    `rows` = f.to_dict("records") (optional, much faster when one frame is evaluated bar by bar)."""
    r = rows[i] if rows is not None else f.iloc[i].to_dict()
    a = float(r["atr"]) if finite(r["atr"]) else float("nan")
    out = {}
    if not finite(a) or a <= 0 or i < 30:
        for side in ("CE", "PE"):
            out[side] = {"side": side, "stage": "NONE", "score": 0.0, "reason": "not enough history"}
        return out
    close = float(r["close"])
    prev_close = float(rows[i - 1]["close"]) if rows is not None else float(f["close"].iloc[i - 1])
    zones = build_zones(levels_at(f, i, cfg, r), a, cfg.merge_atr)
    decided = f.index[i] + BAR
    in_window = _t(cfg.entry_start) <= decided.time() <= _t(cfg.entry_end)
    for side, sign in (("CE", 1), ("PE", -1)):
        buf = max(cfg.buffer_atr * a, close * cfg.buffer_pct / 100.0, 0.05)
        edge = (lambda z: z["top"]) if sign > 0 else (lambda z: z["bottom"])
        broken = [z for z in zones
                  if sign * (prev_close - (edge(z) + sign * buf)) <= 0 < sign * (close - (edge(z) + sign * buf))]
        ahead = sorted([z for z in zones if sign * (edge(z) - close) > 0], key=lambda z: sign * (edge(z) - close))
        res = {"side": side, "stage": "NONE", "zone": None, "level": None, "dist_atr": None}
        if broken:
            zone = max(broken, key=lambda z: (z["strength"], -abs(edge(z) - close)))
            dist = -sign * (close - edge(zone)) / a                 # negative: beyond the level
        elif ahead:
            zone = ahead[0]
            dist = sign * (edge(zone) - close) / a
        else:
            zone, dist = None, None
        sc = score_side(f, i, sign, zone, dist, cfg, chain_stats, sector_up, r)
        res.update(score=sc["score"], families=sc["families"], components=sc["components"], no_data=sc["no_data"])
        if zone is None:
            res["reason"] = "no level ahead (price beyond every level)"
            out[side] = res
            continue
        level = edge(zone)
        res.update(zone=zone["name"], level=r2(level), strength=zone["strength"], dist_atr=round(dist, 2))
        if broken:
            beyond = sign * (close - level) / a
            rng = float(r["high"] - r["low"])
            body = abs(close - float(r["open"])) / rng * 100 if rng > 0 else 0.0
            pos = ((close - r["low"]) / rng * 100 if sign > 0 else (r["high"] - close) / rng * 100) if rng > 0 else 0.0
            vol_ok = (not finite(r["vol_ratio"])) or r["vol_ratio"] >= cfg.confirm_volume
            quality = body >= cfg.confirm_body_pct and pos >= cfg.confirm_close_pct and vol_ok and sign * (close - r["open"]) > 0
            res["confirmation"] = {"body_pct": round(body), "close_pos_pct": round(pos),
                                   "volume_x": r2(r["vol_ratio"]), "beyond_atr": round(beyond, 2)}
            if beyond > cfg.chase_atr:
                res["stage"], res["reason"] = "MISSED", f"closed {beyond:.2f} ATR beyond the level - do not chase"
            elif quality:
                res["stage"], res["reason"] = "CONFIRMED", "15m close beyond the level with body, close and volume"
            else:
                weak = [n for n, ok in (("body", body >= cfg.confirm_body_pct), ("close position", pos >= cfg.confirm_close_pct),
                                        ("volume", vol_ok)) if not ok]
                res["stage"], res["reason"] = "BREAKING", "broke the level but weak " + ", ".join(weak)
        else:
            comp = sc["families"].get("Compression", 0.0)
            if sc["score"] >= cfg.ready_score and dist <= cfg.ready_atr and comp >= cfg.min_compression and in_window:
                res["stage"], res["reason"] = "READY", f"coiled {dist:.2f} ATR from the level - set the trigger"
            elif sc["score"] >= cfg.watch_score and dist <= cfg.watch_atr:
                res["stage"] = "WATCH"
                miss = []
                if sc["score"] < cfg.ready_score:
                    miss.append(f"score {sc['score']:.0f} < {cfg.ready_score:.0f}")
                if dist > cfg.ready_atr:
                    miss.append(f"{dist:.2f} ATR away (> {cfg.ready_atr})")
                if comp < cfg.min_compression:
                    miss.append(f"compression {comp:.0f} < {cfg.min_compression:.0f}")
                if not in_window:
                    miss.append("outside entry window")
                res["reason"] = "approaching: " + ", ".join(miss) if miss else "approaching"
            else:
                res["reason"] = f"score {sc['score']:.0f}, {dist:.2f} ATR away"
        res["plan"] = spot_plan(f, i, sign, level, zones, cfg, decided, r)
        if res["stage"] == "READY" and (res["plan"]["rr1"] or 0) < cfg.min_rr:
            nxt = res["plan"].get("next_level") or {}
            res["stage"] = "WATCH"
            res["reason"] = (f"coiled, but {nxt.get('name', 'the next level')} {nxt.get('value', '')} leaves only "
                             f"{res['plan']['rr1']} R to T1")
        res["gates"] = gates(f, i, sign, cfg, decided, in_window, live, r)
        out[side] = res
    return out


def spot_plan(f: pd.DataFrame, i: int, sign: int, level: float, zones: List[dict], cfg: BreakoutConfig,
              decided: pd.Timestamp, r=None) -> dict:
    r = f.iloc[i] if r is None else r
    a = float(r["atr"])
    close = float(r["close"])
    buf = max(cfg.buffer_atr * a, close * cfg.buffer_pct / 100.0, 0.05)
    trigger = level + sign * buf
    tight = finite(r["box_hi"]) and (r["box_hi"] - r["box_lo"]) <= cfg.max_box_atr * a
    if sign > 0:
        structure = (float(r["box_lo"]) if tight else float(r["low3"])) - 0.1 * a
    else:
        structure = (float(r["box_hi"]) if tight else float(r["high3"])) + 0.1 * a
    risk = sign * (trigger - structure)
    note = "stop behind the base" if tight else "stop behind the last 3 bars"
    if risk > cfg.max_stop_atr * a:
        risk, note = cfg.max_stop_atr * a, f"stop capped at {cfg.max_stop_atr} ATR (structure is further)"
    if risk < cfg.min_stop_atr * a:
        risk, note = cfg.min_stop_atr * a, f"stop widened to {cfg.min_stop_atr} ATR (noise)"
    stop = trigger - sign * risk
    t1, t2 = trigger + sign * cfg.rr_t1 * risk, trigger + sign * cfg.rr_t2 * risk
    edge = (lambda z: z["bottom"]) if sign > 0 else (lambda z: z["top"])      # the near side of the next zone
    beyond = sorted([z for z in zones if sign * (edge(z) - trigger) > 0.15 * a], key=lambda z: sign * (edge(z) - trigger))
    t1_note = f"{cfg.rr_t1} R"
    if beyond and sign * (edge(beyond[0]) - 0.1 * a * sign - t1) < 0:
        t1 = edge(beyond[0]) - sign * 0.1 * a
        t1_note = f"before {beyond[0]['name']} {edge(beyond[0]):.2f}"
    rr1 = sign * (t1 - trigger) / risk
    further = [z for z in beyond if sign * (edge(z) - t1) > 0.25 * risk]
    if further and sign * (edge(further[0]) - sign * 0.1 * a - t2) < 0:
        t2 = edge(further[0]) - sign * 0.1 * a
    if sign * (t2 - t1) < 0.25 * risk:
        t2 = t1
    left = minutes_left(decided)
    reach = float(r["d_atr"]) * math.sqrt(max(left, 1.0) / 375.0) if finite(r["d_atr"]) else float("nan")
    hb = (r["box_hi"] - r["box_lo"]) if tight else float("nan")
    return {"trigger": tick(trigger), "stop": tick(stop), "t1": tick(t1), "t2": tick(t2), "risk": r2(risk),
            "risk_atr": r2(risk / a), "rr1": r2(rr1), "rr2": r2(sign * (t2 - trigger) / risk), "t1_note": t1_note,
            "stop_note": note, "atr15": r2(a), "reach_today": r2(reach), "buffer": r2(buf),
            "measured_move": tick(trigger + sign * hb) if finite(hb) else None, "sign": sign,
            "next_level": ({"name": beyond[0]["name"], "value": r2(edge(beyond[0]))} if beyond else None)}


def gates(f: pd.DataFrame, i: int, sign: int, cfg: BreakoutConfig, decided: pd.Timestamp, in_window: bool,
          live: bool, r=None) -> List[dict]:
    """Mandatory checks for a tradable setup (score / stage are judged separately)."""
    r = f.iloc[i] if r is None else r
    a, close = float(r["atr"]), float(r["close"])
    out = [{"name": "Entry window", "ok": in_window and live,
            "detail": f"{cfg.entry_start}-{cfg.entry_end}, square-off {cfg.square_off}" + ("" if live else " (market closed)")}]
    atr_pct = a / close * 100
    out.append({"name": "Enough movement", "ok": atr_pct >= cfg.min_atr_pct,
                "detail": f"ATR(15m) {atr_pct:.2f}% of price (min {cfg.min_atr_pct}%)"})
    if cfg.require_vwap_side and finite(r["vwap"]):
        ok = sign * (close - r["vwap"]) > 0
        out.append({"name": "VWAP side", "ok": ok, "detail": f"close {close:.2f} vs VWAP {r['vwap']:.2f}"})
    if finite(r["gap_pct"]) and abs(r["gap_pct"]) > cfg.gap_pct:
        ok = decided.time() >= _t(cfg.gap_wait)
        out.append({"name": "Gap day", "ok": ok, "detail": f"gap {r['gap_pct']:+.1f}% - no entries before {cfg.gap_wait}"})
    return out


# =============================================================================
# Intraday equity leg
# =============================================================================
def equity_leg(plan: dict, cfg: BreakoutConfig, cost_fn: Optional[Callable] = None) -> dict:
    trigger, stop, t1, t2, sign = plan["trigger"], plan["stop"], plan["t1"], plan["t2"], plan["sign"]
    per_share = abs(trigger - stop)
    cost = cost_fn or (lambda side, price, qty: 0.0)
    if not per_share or per_share <= 0:
        return {"qty": 0, "ok": False, "detail": "no risk distance"}
    budget = cfg.capital * cfg.risk_pct / 100.0
    qty = int(budget // per_share)
    cap = int(cfg.capital * cfg.leverage // trigger)
    capped = qty > cap
    qty = min(qty, cap)
    if qty <= 0:
        return {"qty": 0, "ok": False, "detail": "one share risks more than the budget"}
    side_in, side_out = ("BUY", "SELL") if sign > 0 else ("SELL", "BUY")
    c_in = cost(side_in, trigger, qty)
    risk_rs = per_share * qty + c_in + cost(side_out, stop, qty)
    rew1 = abs(t1 - trigger) * qty - c_in - cost(side_out, t1, qty)
    rew2 = abs(t2 - trigger) * qty - c_in - cost(side_out, t2, qty)
    rr = rew1 / risk_rs if risk_rs > 0 else 0.0
    return {"action": "BUY" if sign > 0 else "SELL SHORT", "order": f"{'buy' if sign > 0 else 'sell'} stop at {trigger}",
            "qty": qty, "trigger": trigger, "stop": stop, "t1": t1, "t2": t2, "notional": r2(qty * trigger),
            "margin": r2(qty * trigger / cfg.leverage), "risk_rs": r2(risk_rs), "reward1_rs": r2(rew1),
            "reward2_rs": r2(rew2), "rr": r2(rr), "ok": rr >= cfg.min_rr,
            "detail": (f"{qty} sh, risk Rs {risk_rs:,.0f} ({cfg.risk_pct}% of Rs {cfg.capital:,.0f})"
                       + (f", capped by {cfg.leverage:g}x margin" if capped else "")
                       + f", T1 {rr:.2f} R after charges")}


# =============================================================================
# Stage memory across cycles (READY must come before a tradable breakout)
# =============================================================================
class Tracker:
    def __init__(self):
        self.day: Optional[date] = None
        self.mem: Dict[str, dict] = {}

    def roll(self, day: date, saved: Optional[dict] = None):
        if self.day != day:
            self.day, self.mem = day, dict(saved or {})

    def update(self, symbol: str, res: dict, now: pd.Timestamp, atr15: float, cfg: BreakoutConfig) -> dict:
        key = f"{symbol}:{res['side']}"
        m = self.mem.get(key)
        stage, level = res["stage"], res.get("level")
        same = m is not None and level is not None and m.get("level") is not None and finite(atr15) and \
            abs(m["level"] - level) <= max(cfg.merge_atr, 0.3) * atr15
        res["ready_since"] = m.get("ready_at") if same and m else None
        res["prior_ready"] = bool(same and m and m.get("ready_at"))
        if stage in ("READY", "WATCH"):
            if not same:
                m = {"level": level, "watch_at": None, "ready_at": None, "stage": stage, "broke_at": None}
            m["watch_at"] = m.get("watch_at") or now.isoformat()
            if stage == "READY":
                m["ready_at"] = m.get("ready_at") or now.isoformat()
            m["stage"] = stage
            self.mem[key] = m
            res["ready_since"] = m.get("ready_at")
            res["prior_ready"] = bool(m.get("ready_at"))
        elif stage in ("CONFIRMED", "BREAKING", "MISSED"):
            if m is None or not same:
                m = {"level": level, "watch_at": None, "ready_at": None, "stage": stage, "broke_at": now.isoformat()}
            else:
                m["broke_at"] = m.get("broke_at") or now.isoformat()
                m["stage"] = stage
            self.mem[key] = m
        elif m is not None and m.get("stage") in ("CONFIRMED", "TRIGGERED") and m.get("ready_at") and \
                m.get("level") is not None and finite(atr15):
            sign = 1 if res["side"] == "CE" else -1
            close = res.get("close")
            if close is not None and sign * (m["level"] - close) > cfg.fail_atr * atr15:
                res["stage"], res["reason"] = "FAILED", f"back inside {m['level']} after the breakout"
                m["stage"] = "FAILED"
        return res


# =============================================================================
# Paper book (spot levels drive the exits; equity and option P&L both tracked)
# =============================================================================
class BreakoutBook:
    def __init__(self, folder: Path, opt_cost, eq_cost, mock: bool):
        self.folder = Path(folder)
        self.opt_cost, self.eq_cost = opt_cost, eq_cost
        self.mock = mock
        self.day: Optional[date] = None
        self.trades: Dict[str, dict] = {}
        self.memory: dict = {}
        self.alerts: List[dict] = []
        self.lock = threading.Lock()

    def _file(self, day: date) -> Path:
        return self.folder / f"breakout_{'mock_' if self.mock else ''}{day.isoformat()}.json"

    def roll(self, day: date):
        if self.day == day:
            return
        self.day, self.trades, self.memory, self.alerts = day, {}, {}, []
        f = self._file(day)
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self.trades = {t["symbol"]: t for t in data.get("trades", [])}
                self.memory = data.get("memory", {})
                self.alerts = data.get("alerts", [])
            except (OSError, ValueError, KeyError, AttributeError):
                logger.warning("breakout book %s unreadable; starting empty", f.name)

    def save(self, memory: dict):
        if self.day is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        f = self._file(self.day)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"trades": list(self.trades.values()), "memory": memory,
                                   "alerts": self.alerts[-300:]}, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, f)

    def open(self, s: dict, now: pd.Timestamp, spot: float) -> Optional[dict]:
        if s["symbol"] in self.trades:
            return None
        p, eq, opt = s["plan"], s.get("equity") or {}, s.get("option") or {}
        sign = p["sign"]
        qty = int(eq.get("qty") or 0)
        t = {"symbol": s["symbol"], "side": s["side"], "level": s["level"], "trigger": p["trigger"],
             "entry_spot": r2(spot), "stop": p["stop"], "initial_stop": p["stop"], "t1": p["t1"], "t2": p["t2"],
             "sign": sign, "qty": qty, "score": s["score"], "opened": now.isoformat(), "status": "OPEN",
             "spot": r2(spot), "exit_spot": None, "closed": None, "eq_pnl": 0.0,
             "contract": opt.get("contract"), "opt_key": opt.get("key"), "lot": opt.get("lot_size"),
             "opt_entry": opt.get("price_now") or opt.get("entry"), "opt_delta": opt.get("delta"),
             "opt_ltp": opt.get("price_now") or opt.get("entry"), "opt_exit": None, "opt_pnl": 0.0,
             "events": [f"{now:%H:%M} {s['stage'].lower()} - entered at {spot:.2f} "
                        f"({'buy' if sign > 0 else 'short'} {qty} sh" + (f", {opt.get('contract')}" if opt.get("contract") else "") + ")"]}
        self.trades[s["symbol"]] = t
        return t

    def _opt_price(self, t, spot, ltp):
        if finite(ltp):
            return float(ltp)
        if finite(t.get("opt_entry")) and finite(t.get("opt_delta")):
            return max(float(t["opt_entry"]) + float(t["opt_delta"]) * (spot - float(t["entry_spot"])), 0.05)
        return None

    def _pnl(self, t, spot, opt_price):
        sign, q = t["sign"], t["qty"]
        if q:
            side_in, side_out = ("BUY", "SELL") if sign > 0 else ("SELL", "BUY")
            t["eq_pnl"] = r2(sign * (spot - t["entry_spot"]) * q - self.eq_cost(side_in, t["entry_spot"], q)
                             - self.eq_cost(side_out, spot, q))
        if opt_price is not None and t.get("lot") and finite(t.get("opt_entry")):
            lot = int(t["lot"])
            t["opt_ltp"] = r2(opt_price)
            t["opt_pnl"] = r2((opt_price - t["opt_entry"]) * lot - self.opt_cost("BUY", t["opt_entry"], lot)
                              - self.opt_cost("SELL", opt_price, lot))

    def mark(self, spots: Dict[str, float], opt_ltps: Dict[str, float], now: pd.Timestamp, square_off: str):
        for t in self.trades.values():
            if t["closed"]:
                continue
            spot = spots.get(t["symbol"])
            if not finite(spot):
                continue
            sign = t["sign"]
            opt = self._opt_price(t, spot, opt_ltps.get(t.get("opt_key")))
            t["spot"] = r2(spot)
            status = None
            if now.time() >= _t(square_off):
                status = "TIME_EXIT"
            elif sign * (spot - t["stop"]) <= 0:
                status = "STOP_HIT" if t["status"] == "OPEN" else "EXIT_AT_COST"
            elif sign * (spot - t["t2"]) >= 0:
                status = "T2_HIT"
            elif t["status"] == "OPEN" and sign * (spot - t["t1"]) >= 0:
                t["status"], t["stop"] = "T1_HIT", t["entry_spot"]
                t["events"].append(f"{now:%H:%M} T1 {t['t1']} reached - stop moved to cost {t['entry_spot']}")
            self._pnl(t, spot, opt)
            if status:
                t.update(status=status, exit_spot=r2(spot), closed=now.isoformat(), opt_exit=r2(opt) if opt else None)
                t["events"].append(f"{now:%H:%M} {status.replace('_', ' ').lower()} at {spot:.2f}")

    def summary(self) -> dict:
        trades = sorted(self.trades.values(), key=lambda t: t["opened"])
        closed = [t for t in trades if t["closed"]]
        return {"trades": trades, "open": sum(1 for t in trades if not t["closed"]), "closed": len(closed),
                "wins": sum(1 for t in closed if (t["eq_pnl"] or 0) > 0),
                "eq_pnl": r2(sum(t["eq_pnl"] or 0 for t in trades)),
                "opt_pnl": r2(sum(t["opt_pnl"] or 0 for t in trades))}


# =============================================================================
# Scanner (used by the stock engine every cycle)
# =============================================================================
class BreakoutScanner:
    ACTIVE = ("CONFIRMED", "TRIGGERED", "READY", "BREAKING", "WATCH", "MISSED", "FAILED")

    def __init__(self, cfg: BreakoutConfig, stock_cfg, opt_cost, eq_cost, folder: Path, mock: bool,
                 pusher: Optional[Callable[[str], None]] = None, alerts_csv: Optional[Path] = None):
        self.cfg, self.stock_cfg = cfg, stock_cfg
        self.opt_cost, self.eq_cost = opt_cost, eq_cost
        self.tracker = Tracker()
        self.book = BreakoutBook(folder, opt_cost, eq_cost, mock)
        self.items: List[dict] = []
        self.generated_at: Optional[str] = None
        self.pusher = pusher
        self.alerts_csv = alerts_csv
        self.lock = threading.Lock()

    # ---- one cycle
    def run(self, now: pd.Timestamp, inputs: List[dict], index_bars: Optional[pd.DataFrame], live: bool,
            get_option: Callable, quotes: Callable) -> None:
        from src.stocks.signals import completed, option_leg
        cfg = self.cfg
        feats = {}
        for x in inputs:
            bars = completed(x["bars"], now) if live else x["bars"]
            if bars is None or len(bars) < 40:
                continue
            try:
                feats[x["stock"].symbol] = (x, features(bars, x["daily"], cfg, index_bars))
            except Exception as error:
                logger.warning("breakout features %s: %s", x["stock"].symbol, error)
        # sector breadth at the latest bar: share of peers up on the day
        by_sector: Dict[str, List[float]] = {}
        for sym, (x, f) in feats.items():
            chg = f["chg_pct"].iloc[-1]
            if finite(chg) and x["stock"].industry:
                by_sector.setdefault(x["stock"].industry, []).append(1.0 if chg > 0 else 0.0)
        self.book.roll(now.date())
        self.tracker.roll(now.date(), self.book.memory)
        items = []
        for sym, (x, f) in feats.items():
            i = len(f) - 1
            peers = by_sector.get(x["stock"].industry) or []
            sector = float(np.mean(peers)) if len(peers) >= 3 else None
            stats = x.get("stats") or {}
            res = evaluate(f, i, cfg, stats, sector, live)
            a = float(f["atr"].iloc[i]) if finite(f["atr"].iloc[i]) else float("nan")
            spot = x.get("spot")
            for side in ("CE", "PE"):
                rr = res[side]
                rr["close"] = float(f["close"].iloc[i])
                self.tracker.update(sym, rr, now, a, cfg)
                if rr["stage"] == "READY" and live and finite(spot) and rr.get("plan") and \
                        rr["plan"]["sign"] * (spot - rr["plan"]["trigger"]) >= 0:
                    rr["stage"], rr["reason"] = "TRIGGERED", f"price {spot:.2f} crossed the trigger (bar not closed)"
                    m = self.tracker.mem.get(f"{sym}:{side}")
                    if m is not None:
                        m["stage"] = "TRIGGERED"
            best = max((res["CE"], res["PE"]), key=lambda z: (STAGES.index(z["stage"]) * -1, z.get("score") or 0))
            if best["stage"] == "NONE":
                continue
            item = {"symbol": sym, "name": x["stock"].name, "industry": x["stock"].industry, "ltp": r2(spot),
                    "change_pct": r2(f["chg_pct"].iloc[i]), "bar": f.index[i].isoformat(), **best}
            if item.get("plan"):
                item["equity"] = equity_leg(item["plan"], cfg, self.eq_cost)
            items.append(item)
        # option legs for actionable stages (chain fetched lazily)
        for item in items:
            if item["stage"] not in ("READY", "TRIGGERED", "CONFIRMED") or not item.get("plan"):
                continue
            x = feats[item["symbol"]][0]
            try:
                chain, expiry, lot = (x.get("chain"), x.get("expiry"), x.get("lot"))
                if chain is None:
                    chain, expiry, lot = get_option(x["stock"])
                    x.update(chain=chain, expiry=expiry, lot=lot)
            except Exception as error:
                if "Auth" in type(error).__name__:
                    raise
                chain, expiry, lot = None, None, None
                item.setdefault("notes", []).append(f"option chain unavailable: {error}")
            p = item["plan"]
            spot = item["ltp"] if finite(item["ltp"]) else float(feats[item["symbol"]][1]["close"].iloc[-1])
            # before the break the premium is estimated AT the trigger; after it, the entry is the price now
            entry_spot = p["trigger"] if item["stage"] == "READY" else spot
            opt, o_checks, o_notes = option_leg(item["symbol"], item["side"], spot, entry_spot, p["stop"], p["t1"],
                                                p["t2"], x["daily"], now, chain, expiry, lot, self.stock_cfg,
                                                self.opt_cost)
            if opt is not None and chain is not None:
                row = chain.iloc[(chain["strike"] - opt["strike"]).abs().argsort()[:1]].iloc[0]
                side = item["side"].lower()
                ask, ltp = row.get(f"{side}_ask"), row.get(f"{side}_ltp")
                opt["price_now"] = r2(ask if finite(ask) and ask > 0 else ltp)
            item["option"] = opt
            item["option_checks"] = o_checks
            item.setdefault("notes", []).extend(o_notes)
        # tradable?
        for item in items:
            checks = list(item.get("gates") or [])
            p = item.get("plan") or {}
            if p:
                checks.append({"name": "Room to target", "ok": (p.get("rr1") or 0) >= cfg.min_rr,
                               "detail": f"T1 {p.get('t1_note')}: {p.get('rr1')} R (min {cfg.min_rr})"})
                if p.get("reach_today") is not None:
                    need = abs(p["t1"] - p["trigger"])
                    checks.append({"name": "Reachable today", "ok": need <= p["reach_today"],
                                   "detail": f"T1 needs {need:.2f}; range left today ~{p['reach_today']:.2f}"})
            eq = item.get("equity") or {}
            if eq:
                checks.append({"name": "Equity reward:risk", "ok": bool(eq.get("ok")), "detail": eq.get("detail", "")})
            if cfg.require_prior_ready and item["stage"] in ("CONFIRMED", "TRIGGERED", "BREAKING"):
                checks.append({"name": "READY before the break", "ok": bool(item.get("prior_ready")),
                               "detail": f"READY since {item['ready_since'][11:16]}" if item.get("ready_since")
                               else "no READY on this level first - a breakout without a setup is not taken"})
            item["checks"] = checks
            item["tradable"] = item["stage"] in ("READY", "TRIGGERED", "CONFIRMED") and all(c["ok"] for c in checks)
            item["option_ok"] = bool(item.get("option")) and all(c["ok"] for c in item.get("option_checks") or [])
        # paper entries + marking
        entry_stage = "CONFIRMED" if cfg.entry_mode == "close" else "TRIGGERED"
        with self.book.lock:
            if live:
                for item in sorted(items, key=lambda z: -(z.get("score") or 0)):
                    if item["stage"] in (entry_stage, "CONFIRMED") and item["tradable"] and \
                            len(self.book.trades) < cfg.max_trades and finite(item["ltp"]):
                        p = item["plan"]
                        # never enter past the point where T1 is less than 1 R away
                        if p["sign"] * (p["t1"] - item["ltp"]) < p["risk"]:
                            item["checks"].append({"name": "Entry price", "ok": False,
                                                   "detail": f"{item['ltp']} is too far past the trigger"})
                            item["tradable"] = False
                            continue
                        t = self.book.open(item, now, item["ltp"])
                        if t:
                            self._alert("entry", item, now)
                spots = {it["symbol"]: it["ltp"] for it in items}
                for sym, (x, f) in feats.items():
                    spots.setdefault(sym, x.get("spot"))
                keys = [t["opt_key"] for t in self.book.trades.values() if not t["closed"] and t.get("opt_key")]
                ltps = {}
                if keys:
                    try:
                        ltps = {k: v.get("ltp") for k, v in quotes(keys).items() if finite(v.get("ltp"))}
                    except Exception as error:
                        if "Auth" in type(error).__name__:
                            raise
                        logger.warning("option quotes failed: %s", error)
                self.book.mark(spots, ltps, now, cfg.square_off)
            for item in items:
                if live and (item["stage"] == "READY" or (item["stage"] == "CONFIRMED" and item.get("prior_ready"))):
                    self._alert(item["stage"].lower(), item, now)
            self.book.save(self.tracker.mem)
        order = {s: n for n, s in enumerate(STAGES)}
        items.sort(key=lambda z: (order.get(z["stage"], 9), -(z.get("score") or 0)))
        with self.lock:
            self.items = items
            self.generated_at = now.isoformat()

    # ---- alerts (deduplicated per day, stored with the book)
    def _alert(self, kind: str, item: dict, now: pd.Timestamp):
        key = f"{kind}:{item['symbol']}:{item['side']}:{item.get('level')}"
        if any(a.get("key") == key for a in self.book.alerts):
            return
        p, eq, opt = item.get("plan") or {}, item.get("equity") or {}, item.get("option") or {}
        text = (f"{'🟡 READY' if kind == 'ready' else '✅ CONFIRMED' if kind == 'confirmed' else '📈 PAPER ENTRY'} "
                f"{item['symbol']} {item['side']} - {item.get('zone')} {item.get('level')}\n"
                f"Trigger {p.get('trigger')}  SL {p.get('stop')}  T1 {p.get('t1')}  T2 {p.get('t2')}  "
                f"(score {item.get('score')})\n"
                f"Equity: {eq.get('action', '-')} {eq.get('qty', '-')} sh, risk Rs {eq.get('risk_rs', '-')}\n"
                f"Option: {opt.get('contract', '-')} @ ~{opt.get('entry', '-')} SL {opt.get('stop', '-')} "
                f"T1 {opt.get('t1', '-')} T2 {opt.get('t2', '-')}")
        rec = {"key": key, "kind": kind, "time": now.isoformat(), "symbol": item["symbol"], "side": item["side"],
               "level": item.get("level"), "trigger": p.get("trigger"), "stop": p.get("stop"), "t1": p.get("t1"),
               "t2": p.get("t2"), "score": item.get("score"), "contract": opt.get("contract"), "text": text}
        self.book.alerts.append(rec)
        logger.info("BREAKOUT %s", text.replace("\n", " | "))
        if self.alerts_csv is not None:
            try:
                self.alerts_csv.parent.mkdir(parents=True, exist_ok=True)
                new = not self.alerts_csv.exists()
                with open(self.alerts_csv, "a", newline="", encoding="utf-8") as h:
                    w = csv.DictWriter(h, fieldnames=[k for k in rec if k not in ("key", "text")], extrasaction="ignore")
                    if new:
                        w.writeheader()
                    w.writerow(rec)
            except OSError:
                pass
        if self.pusher and self.cfg.notify and not self.book.mock:
            try:
                self.pusher(text)
            except Exception:
                logger.warning("breakout push failed")

    def state(self) -> dict:
        with self.lock:
            items = list(self.items)
            gen = self.generated_at
        with self.book.lock:
            book = self.book.summary()
            alerts = list(self.book.alerts[-30:])[::-1]
        c = self.cfg
        return {"generated_at": gen, "items": items,
                "counts": {s: sum(1 for x in items if x["stage"] == s) for s in self.ACTIVE},
                "book": book, "alerts": alerts,
                "config": {"ready_score": c.ready_score, "watch_score": c.watch_score, "ready_atr": c.ready_atr,
                           "entry_mode": c.entry_mode, "capital": c.capital, "risk_pct": c.risk_pct,
                           "leverage": c.leverage, "entry_start": c.entry_start, "entry_end": c.entry_end,
                           "max_trades": c.max_trades}}


def telegram_pusher() -> Optional[Callable[[str], None]]:
    """Background, best-effort Telegram sender (None when the credentials are missing)."""
    from src.config.settings import get_secret
    token, chat = get_secret("TELEGRAM_BOT_TOKEN"), get_secret("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return None

    def push(text: str):
        def send():
            try:
                import requests
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": chat, "text": text[:4000]}, timeout=8).raise_for_status()
            except Exception as error:      # never log the token
                logger.warning("Telegram send failed (%s)", type(error).__name__)
        threading.Thread(target=send, daemon=True).start()
    return push
