"""
Analysis of one underlying: 1-minute candles -> base-timeframe frame, multi-timeframe trend table, every
enabled strategy's trigger events, the confluence checks and the confluence-gated signals.

The same function serves the live engine (it reads the last completed bar) and the backtest (it reads all
rows). Everything is causal; nothing on a bar depends on a later bar.

Confluence components (each +1 bullish, -1 bearish, 0 neutral, NaN = no data, which is left out of the count):
    STRUCTURE  HH + HL swing structure or a close above the last confirmed swing high (mirror for bearish)
    MOMENTUM   RSI > 52 with MACD histogram > 0 (RSI < 48 with histogram < 0)
    LOCATION   close above EMA20 and above VWAP when VWAP exists (below both)
    VOLUME     relative volume >= min_rvol (direction-neutral; counts for either side)
    HTF        at least htf_min_agree of the 15m / 30m / 60m / daily trends agree and none of them opposes
    CANDLE     the trigger bar itself closes in the trade direction, in the outer 40% of its range
    LTF        the 1m / 3m EMA9 vs EMA21 momentum at the bar's close agrees

A trigger becomes a signal when at least `min_confirmations` components agree (reversal strategies:
`reversal_min_confirmations`), every `required` component that has data agrees, and the confidence score
reaches `min_confidence`. The score measures how many checks line up; it is NOT a probability of profit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.fno import indicators as ind
from src.fno.config import FnoConfig
from src.fno.strategies import Ctx, REGISTRY, Strategy, all_strategies, scan_factors
from src.scanner.indicators import ema, has_volume, resample

COMPONENTS = ("STRUCTURE", "MOMENTUM", "LOCATION", "VOLUME", "HTF", "CANDLE", "LTF")
WEIGHTS = {"STRUCTURE": 1.5, "MOMENTUM": 1.0, "LOCATION": 1.5, "VOLUME": 1.0, "HTF": 1.5, "CANDLE": 1.0, "LTF": 0.5}


@dataclass
class Analysis:
    symbol: str
    tf: int
    frame: pd.DataFrame                    # enriched base frame
    events: pd.DataFrame                   # strategy key -> +1 / -1 / 0 raw trigger events
    signals: pd.DataFrame                  # strategy key -> +1 / -1 / 0 confluence-gated signals
    comp: pd.DataFrame                     # component columns + up_n / dn_n / avail_n / conf_up / conf_dn
    mtf: Dict[object, pd.Series] = field(default_factory=dict)
    unavailable: Dict[str, str] = field(default_factory=dict)   # strategy key -> why it cannot run here
    gates: Dict[str, pd.DataFrame] = field(default_factory=dict)


def attach_futures(spot_m1: pd.DataFrame, fut_m1: Optional[pd.DataFrame]) -> pd.DataFrame:
    """An index has no volume: take volume and OI from its near-month futures, minute by minute."""
    if fut_m1 is None or fut_m1.empty or not has_volume(fut_m1["volume"]):
        return spot_m1
    out = spot_m1.copy()
    out["volume"] = fut_m1["volume"].reindex(out.index).fillna(0.0)
    out["oi"] = fut_m1["oi"].reindex(out.index) if "oi" in fut_m1 else np.nan
    out["fut_close"] = fut_m1["close"].reindex(out.index)
    return out


def _base_frame(m1: pd.DataFrame, tf: int) -> pd.DataFrame:
    bars = ind.bars_for(m1, tf)
    if "fut_close" in m1.columns and has_volume(m1["volume"]):
        # VWAP of the futures, expressed on the spot scale: spot VWAP = spot close - (futures close - futures VWAP)
        fut = resample(m1.assign(open=m1["fut_close"], high=m1["fut_close"], low=m1["fut_close"],
                                 close=m1["fut_close"]).dropna(subset=["close"]), tf)
        v, sd = ind.session_vwap(fut)
        bars["vwap"] = bars["close"] - (fut["close"] - v).reindex(bars.index)
        bars["vwap_sd"] = sd.reindex(bars.index)
    return bars


def components(f: pd.DataFrame, mtf: Dict[object, pd.Series], ltf: Optional[pd.Series], cfg: FnoConfig) -> pd.DataFrame:
    c = f["close"]
    out = pd.DataFrame(index=f.index)
    out["STRUCTURE"] = np.where((f["structure"] == 1) | (c > f["swing_hi"]), 1.0,
                                np.where((f["structure"] == -1) | (c < f["swing_lo"]), -1.0, 0.0))
    out.loc[f["swing_hi"].isna() | f["swing_lo"].isna(), "STRUCTURE"] = np.nan
    out["MOMENTUM"] = np.where((f["rsi"] > 52) & (f["macd_hist"] > 0), 1.0,
                               np.where((f["rsi"] < 48) & (f["macd_hist"] < 0), -1.0, 0.0))
    out.loc[f["macd_hist"].isna(), "MOMENTUM"] = np.nan
    v = f["vwap"]
    above = (c > f["ema20"]) & (v.isna() | (c > v))
    below = (c < f["ema20"]) & (v.isna() | (c < v))
    out["LOCATION"] = np.where(above, 1.0, np.where(below, -1.0, 0.0))
    out.loc[f["ema20"].isna(), "LOCATION"] = np.nan
    out["VOLUME"] = np.where(f["rvol"].isna(), np.nan, np.where(f["rvol"] >= cfg.min_rvol, 1.0, 0.0))
    if mtf:
        m = pd.DataFrame(mtf, index=f.index)
        ups, dns, known = (m == 1).sum(axis=1), (m == -1).sum(axis=1), m.notna().sum(axis=1)
        need = np.minimum(cfg.htf_min_agree, known.clip(lower=1))
        out["HTF"] = np.where((ups >= need) & (dns == 0), 1.0, np.where((dns >= need) & (ups == 0), -1.0, 0.0))
        out.loc[known == 0, "HTF"] = np.nan
    else:
        out["HTF"] = np.nan
    out["CANDLE"] = np.where((c > f["open"]) & (f["close_pos"] >= 0.6), 1.0,
                             np.where((c < f["open"]) & (f["close_pos"] <= 0.4), -1.0, 0.0))
    out["LTF"] = ltf if ltf is not None else np.nan
    vals = out[list(COMPONENTS)]
    vol_ok = (vals["VOLUME"] == 1).astype(int)
    directional = vals.drop(columns="VOLUME")
    out["up_n"] = (directional == 1).sum(axis=1) + vol_ok
    out["dn_n"] = (directional == -1).sum(axis=1) + vol_ok
    out["avail_n"] = vals.notna().sum(axis=1)
    w = pd.Series(WEIGHTS)
    avail_w = vals.notna().mul(w, axis=1).sum(axis=1).replace(0.0, np.nan)
    up_w = (directional == 1).mul(w.drop("VOLUME"), axis=1).sum(axis=1) + vol_ok * w["VOLUME"]
    dn_w = (directional == -1).mul(w.drop("VOLUME"), axis=1).sum(axis=1) + vol_ok * w["VOLUME"]
    out["conf_up"], out["conf_dn"] = 100.0 * up_w / avail_w, 100.0 * dn_w / avail_w
    return out


def _passes(comp: pd.DataFrame, side: int, need: int, required: tuple) -> pd.Series:
    n = comp["up_n"] if side == 1 else comp["dn_n"]
    ok = n >= np.minimum(need, comp["avail_n"])          # with little data, every available check must agree
    for name in required:
        if name == "VOLUME":
            ok &= comp["VOLUME"].isna() | (comp["VOLUME"] == 1)
        elif name in comp:
            ok &= comp[name].isna() | (comp[name] == side)
    return ok & (comp["avail_n"] >= 3)


def _shift(x: np.ndarray, k: int, fill=False) -> np.ndarray:
    out = np.full(len(x), fill, dtype=x.dtype)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def gate(events: pd.Series, comp: pd.DataFrame, strat: Strategy, cfg: FnoConfig, agree: pd.DataFrame,
         score: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Confluence-gated signal, its confidence, and WAIT markers for one strategy (numpy: this runs per strategy).
    With [CONFLUENCE] confidence_mode = agreement the confidence is the strategy agreement score: the share (0-100)
    of the indicator groups behind the strategies (scan_factors: EMA, Supertrend, RSI, MACD, VWAP, relative volume,
    ATR, breakout, support / resistance, relative strength) that point the same way as the signal; `score` holds
    its bull / bear columns."""
    by_agreement = score is not None and str(cfg.confidence_mode).lower() == "agreement"
    reversal = strat.kind == "reversal"
    need = cfg.reversal_min_confirmations if reversal else cfg.min_confirmations
    required = cfg.reversal_required if reversal else cfg.required
    n = len(events)
    ev = events.to_numpy()
    day = events.index.normalize().asi8
    avail = comp["avail_n"].to_numpy()
    signal, wait, lag_out = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    confidence = np.zeros(n)
    for side, col, cnt, ag in ((1, "conf_up", "up_n", "up"), (-1, "conf_dn", "dn_n", "dn")):
        e = ev == side
        if not e.any():
            continue
        count = comp[cnt].to_numpy()
        ok = _passes(comp, side, need, required).to_numpy()
        bonus = 25.0 * np.minimum(1.0, np.clip(agree[ag].to_numpy() - 1, 0, None) / 4.0)
        conf = np.clip(0.75 * np.nan_to_num(comp[col].to_numpy()) + bonus, 0, 100)
        if by_agreement:
            conf = np.clip(np.nan_to_num(score["bull" if side == 1 else "bear"].to_numpy(dtype=float)), 0, 100)
        ok = ok & (conf >= cfg.min_confidence)
        fired = e & ok
        lags = np.zeros(n, dtype=np.int8)
        prior_ok = np.zeros(n, dtype=bool)
        for lag in range(1, cfg.confirm_bars + 1):        # the trigger was `lag` bars ago and nothing passed since
            prior_ok |= _shift(ok, lag)
            late = _shift(e, lag) & ok & ~prior_ok & (day == _shift(day, lag, -1)) & ~fired
            lags[late] = lag
            fired = fired | late
        near = e & ~fired & (count >= np.minimum(need, avail) - 1) & (conf >= cfg.min_confidence - 15)
        signal[fired], wait[near], lag_out[fired] = side, side, lags[fired]
        confidence[fired | near] = np.round(conf[fired | near], 1)
    return pd.DataFrame({"signal": signal, "confidence": confidence, "wait": wait, "lag": lag_out}, index=events.index)


@dataclass
class Prepared:
    """Everything that does not depend on strategy parameters (computed once per underlying)."""
    symbol: str
    tf: int
    frame: pd.DataFrame
    ctx: Ctx
    comp: pd.DataFrame
    mtf: Dict[object, pd.Series]
    volume_ok: bool


def prepare(symbol: str, m1: pd.DataFrame, cfg: FnoConfig, daily: Optional[pd.DataFrame] = None,
            bench_m1: Optional[pd.DataFrame] = None, chain_series: Optional[pd.DataFrame] = None) -> Prepared:
    """`m1`: completed 1-minute candles (tz-aware index of minute START times)."""
    tf = cfg.base_tf
    f = ind.enrich(_base_frame(m1, tf), tf, daily)
    closes = f.index + pd.Timedelta(minutes=tf)
    mtf: Dict[object, pd.Series] = {}
    for h in cfg.htfs:
        if h > tf:
            mtf[h] = pd.Series(ind.tf_state(m1, h, closes).to_numpy(), index=f.index)
    if f["d_trend"].notna().any():
        mtf["D"] = f["d_trend"]
    ltf = None
    if cfg.ltf < tf:
        ltf = pd.Series(ind.tf_state(m1, cfg.ltf, closes, 9, 21).to_numpy(), index=f.index)
    ctx = Ctx(symbol=symbol, tf=tf, htf=mtf, ltf=ltf)
    if bench_m1 is not None and len(bench_m1) and symbol != cfg.benchmark:
        b = ind.bars_for(bench_m1, tf)
        day_open = b.groupby(ind.day_key(b.index))["open"].transform("first")
        ctx.bench_ret = ((b["close"] / day_open - 1.0) * 100.0).reindex(f.index)
    if chain_series is not None and len(chain_series):
        left = pd.DataFrame({"t": closes})
        right = chain_series.sort_index().rename_axis("t").reset_index()
        merged = pd.merge_asof(left, right, on="t", tolerance=pd.Timedelta(minutes=3 * tf))
        ctx.chain = merged.drop(columns="t").set_index(f.index)
    return Prepared(symbol, tf, f, ctx, components(f, mtf, ltf, cfg), mtf, bool(f["rvol"].notna().any()))


def evaluate(prep: Prepared, cfg: FnoConfig, only: Optional[List[str]] = None, params: Optional[Dict[str, dict]] = None,
             base_events: Optional[pd.DataFrame] = None) -> Analysis:
    """Trigger events and confluence-gated signals. `only` limits the strategies that are (re)computed;
    `params` overrides their settings (walk-forward); `base_events` supplies the other strategies' events so
    the agreement count is the same as in a full run."""
    f, ctx, mtf = prep.frame, prep.ctx, prep.mtf
    overrides = dict(cfg.strategies)
    overrides.update(params or {})
    events, unavailable = {}, {}
    for s in all_strategies():
        o = overrides.get(s.key, {})
        if only is not None and s.key not in only:
            continue
        if only is None and not s.enabled(o):
            continue
        if "volume" in s.needs and not prep.volume_ok:
            unavailable[s.key] = "needs traded volume (none in this data)"
        elif "chain" in s.needs and ctx.chain is None:
            unavailable[s.key] = "needs option-chain history (recorded by the live engine; none for this period)"
        elif "benchmark" in s.needs and ctx.bench_ret is None:
            unavailable[s.key] = "needs the benchmark index (not applicable to the benchmark itself)"
        elif "htf15" in s.needs and 15 not in mtf:
            unavailable[s.key] = "needs the 15m trend"
        elif "htf" in s.needs and (15 not in mtf or 60 not in mtf):
            unavailable[s.key] = "needs 15m and 60m trends"
        if s.key in unavailable:
            events[s.key] = pd.Series(0, index=f.index, dtype=np.int8)
            continue
        events[s.key] = s.signals(f, ctx, o)
    ev = pd.DataFrame(events, index=f.index)
    pool = ev if base_events is None else pd.concat([base_events.drop(columns=list(ev.columns), errors="ignore"), ev], axis=1)
    k = max(1, cfg.agree_bars)
    agree = pd.DataFrame({"up": (pool == 1).rolling(k, min_periods=1).max().sum(axis=1),
                          "dn": (pool == -1).rolling(k, min_periods=1).max().sum(axis=1)}, index=f.index)
    sig = pd.DataFrame(index=f.index)
    gates = {}
    scan = scan_factors(f, ctx)
    for key in ev.columns:
        g = gate(ev[key], prep.comp, REGISTRY[key], cfg, agree, scan)
        sig[key] = g["signal"].astype(np.int8)
        gates[key] = g
    comp = prep.comp.copy()
    comp["agree_up"], comp["agree_dn"] = agree["up"], agree["dn"]
    a = Analysis(symbol=prep.symbol, tf=prep.tf, frame=f, events=ev, signals=sig, comp=comp, mtf=mtf, unavailable=unavailable)
    a.gates = gates
    a.chain = ctx.chain
    a.scan = scan
    return a


def analyze(symbol: str, m1: pd.DataFrame, cfg: FnoConfig, daily: Optional[pd.DataFrame] = None,
            bench_m1: Optional[pd.DataFrame] = None, chain_series: Optional[pd.DataFrame] = None,
            only: Optional[List[str]] = None, params: Optional[Dict[str, dict]] = None) -> Analysis:
    return evaluate(prepare(symbol, m1, cfg, daily, bench_m1, chain_series), cfg, only, params)


def trend_table(a: Analysis) -> Dict[str, Optional[int]]:
    """Latest trend per timeframe for the dashboard (1 up, -1 down, 0 mixed, None unknown)."""
    out = {}
    f = a.frame
    if len(f):
        last = f.iloc[-1]
        e1, e2 = last.get("ema20"), last.get("ema50")
        if pd.notna(e1) and pd.notna(e2):
            out[f"{a.tf}m"] = 1 if last["close"] > e1 > e2 else -1 if last["close"] < e1 < e2 else 0
    for key, series in a.mtf.items():
        v = series.iloc[-1] if len(series) else np.nan
        out["D" if key == "D" else f"{key}m"] = None if pd.isna(v) else int(v)
    return out


def latest(a: Analysis, cfg: FnoConfig) -> List[dict]:
    """Signals and WAIT states on the last completed bar."""
    if not len(a.frame):
        return []
    t = a.frame.index[-1]
    comp = a.comp.iloc[-1]
    out = []
    for key, g in a.gates.items():
        row = g.iloc[-1]
        side = int(row["signal"]) or int(row["wait"])
        if not side:
            continue
        checks = {}
        for name in COMPONENTS:
            v = comp[name]
            checks[name] = None if pd.isna(v) else bool(v == 1) if name == "VOLUME" else bool(v == side)
        s = REGISTRY[key]
        out.append({"strategy": key, "strategy_name": s.name, "category": s.category, "kind": s.kind,
                    "side": "CE" if side == 1 else "PE", "status": "BUY" if row["signal"] else "WAIT",
                    "confidence": float(row["confidence"]), "bar_time": t, "checks": checks,
                    "confidence_basis": "strategy agreement score" if str(cfg.confidence_mode).lower() == "agreement" else "checks aligned",
                    "agreeing": int(comp["agree_up"] if side == 1 else comp["agree_dn"]),
                    "confirmed_after_bars": int(row["lag"]), "rule": s.ce if side == 1 else s.pe})
    out.sort(key=lambda x: (x["status"] != "BUY", -x["confidence"]))
    return out
