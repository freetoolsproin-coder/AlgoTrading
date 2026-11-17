"""
Intraday CE / PE trade plans for NIFTY 50 stocks: direction, strike, entry, stop-loss and targets that are
checked before a plan is called VALID.

How a plan is built (all on COMPLETED 15-minute bars - a forming bar is never used):

 1. Score (bull +, bear -), max about +/-8:
      daily trend  close vs EMA20 / EMA50 (+/-1), vs EMA200 (+/-0.5)
      15m          Supertrend(10,3) (+/-1.5), price vs today's VWAP (+/-1), EMA9 vs EMA21 (+/-1),
                   RSI(14) > 55 / < 45 (+/-1, 0 when stretched beyond 75 / 25), MACD histogram (+/-0.5),
                   last bar volume >= 1.3x average in the bar's direction (+/-0.5)
      options      PCR >= 1.2 / <= 0.7 (+/-0.5), price vs max-pain is NOT used (too slow for intraday)
    score >= +min_score -> CE, <= -min_score -> PE, otherwise no trade.

 2. Underlying levels (CE shown; PE is the mirror):
      stop    = the tighter of (lowest low of the last 3 bars, Supertrend line) - 0.1 ATR,
                at least 0.6 ATR(15m) away (noise) and at most max_stop_atr ATR (else: wait for a pullback)
      T1, T2  = entry + 1.5 R and + 2.5 R, pulled in below the nearest resistance (previous-day high, today's
                high, pivots R1-R3, highest CE-OI strike, 20-day high, 52-week high). A plan whose T1 has to be
                pulled in below min_rr is rejected - the target must be reachable before resistance.
      T1 must also fit the range the stock can still move today: ATR(day) x sqrt(minutes left / 375).

 3. Option: the strike whose delta is closest to 0.55 among liquid strikes (OI > 0, bid-ask spread <= 5%),
    nearest monthly expiry with >= min_days_to_expiry days left. Premium levels are mapped from the spot
    levels with delta + gamma (Taylor), minus theta for the expected holding time on the targets; the premium
    stop is capped at max_premium_loss_pct. Reward:risk is checked again on the premium, net of charges.

 4. Status: VALID (every check passes), WATCH (direction but a check fails - reasons listed), NO_TRADE.

Everything here is a paper-trading signal, not advice. Parameters are starting values: validate them.
"""
from __future__ import annotations

import configparser
import math
from dataclasses import dataclass, fields
from datetime import date, datetime, time
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from src.scanner.indicators import atr, ema, macd, rsi
from src.stocks.analytics import bs_greeks, chain_stats, classic_pivots, day_vwap, supertrend
from src.utils.numbers import finite, fnum

TZ = "Asia/Kolkata"
BAR_MINUTES = 15
SESSION_OPEN, SESSION_CLOSE = time(9, 15), time(15, 30)


@dataclass(frozen=True)
class StockConfig:
    min_score: float = 4.5            # |score| needed for a CE / PE direction
    watch_score: float = 3.0          # below this: NO_TRADE without further checks
    rr_t1: float = 1.5                # T1 = entry + 1.5 R
    rr_t2: float = 2.5
    min_rr: float = 1.2               # minimum reward:risk to T1 (underlying and premium, net of costs)
    min_stop_atr: float = 0.6         # stop at least this many 15m ATRs away
    max_stop_atr: float = 1.8         # wider than this -> wait for a pullback
    max_premium_loss_pct: float = 35.0
    target_delta: float = 0.55
    min_delta: float = 0.35
    max_delta: float = 0.75
    max_spread_pct: float = 5.0
    min_days_to_expiry: int = 3
    entry_start: str = "09:30"
    entry_end: str = "14:30"
    square_off: str = "15:15"
    hold_hours: float = 1.5           # theta charged against the targets
    refresh_seconds: int = 180
    requests_per_second: float = 5.0

    @classmethod
    def from_ini(cls, path) -> "StockConfig":
        parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        try:
            parser.read(path, encoding="utf-8")
        except configparser.Error:
            return cls()
        if not parser.has_section("STOCK_OPTIONS"):
            return cls()
        sec, values = parser["STOCK_OPTIONS"], {}
        for f in fields(cls):
            if f.name in sec:
                raw = sec[f.name].strip()
                values[f.name] = raw if f.type in ("str", str) else (int(float(raw)) if f.type in ("int", int)
                                                                   else float(raw))
        return cls(**values)


def _t(text: str) -> time:
    hh, mm = text.split(":")
    return time(int(hh), int(mm))


def r2(v):
    return round(float(v), 2) if finite(v) else None


def tick(v, size=0.05):
    return round(round(float(v) / size) * size, 2) if finite(v) else None


# =============================================================================
# Bars helpers
# =============================================================================
def completed(bars: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if bars is None or bars.empty:
        return bars
    ends = bars.index + pd.Timedelta(minutes=BAR_MINUTES)
    return bars[ends <= now]


def minutes_left(now: pd.Timestamp) -> float:
    close = now.normalize() + pd.Timedelta(hours=15, minutes=30)
    return max((close - now).total_seconds() / 60.0, 0.0)


# =============================================================================
# Scoring
# =============================================================================
def score_components(bars: pd.DataFrame, daily: pd.DataFrame, today: date, stats: dict) -> List[dict]:
    comps = []

    def add(name, points, value, note):
        comps.append({"name": name, "points": round(points, 2), "value": value, "note": note})

    dc = daily["close"]
    last_close = float(bars["close"].iloc[-1])
    if len(dc) >= 50:
        e20, e50 = float(ema(dc, 20).iloc[-1]), float(ema(dc, 50).iloc[-1])
        p = 1.0 if last_close > e20 > e50 else -1.0 if last_close < e20 < e50 else 0.0
        add("Daily trend (EMA20/50)", p, f"{e20:.1f} / {e50:.1f}",
            "price above rising averages" if p > 0 else "price below falling averages" if p < 0 else "mixed")
    if len(dc) >= 200:
        e200 = float(ema(dc, 200).iloc[-1])
        add("Daily EMA200", 0.5 if last_close > e200 else -0.5, f"{e200:.1f}",
            "above long-term average" if last_close > e200 else "below long-term average")

    line, direction = supertrend(bars, 10, 3.0)
    if finite(direction.iloc[-1]):
        d = int(direction.iloc[-1])
        add("Supertrend 15m (10,3)", 1.5 * d, f"{line.iloc[-1]:.2f}", "up-trend" if d > 0 else "down-trend")

    today_bars = bars[bars.index.date == today]
    vwap = day_vwap(today_bars)
    if finite(vwap):
        p = 1.0 if last_close > vwap else -1.0
        add("VWAP (today)", p, f"{vwap:.2f}", "price above VWAP" if p > 0 else "price below VWAP")

    c = bars["close"]
    if len(c) >= 21:
        e9, e21 = float(ema(c, 9).iloc[-1]), float(ema(c, 21).iloc[-1])
        add("EMA 9/21 (15m)", 1.0 if e9 > e21 else -1.0, f"{e9:.1f} / {e21:.1f}",
            "fast above slow" if e9 > e21 else "fast below slow")
    if len(c) >= 15:
        r = float(rsi(c).iloc[-1])
        if r > 75:
            p, note = 0.0, "overbought - no chasing"
        elif r < 25:
            p, note = 0.0, "oversold - no chasing"
        elif r > 55:
            p, note = 1.0, "bullish momentum"
        elif r < 45:
            p, note = -1.0, "bearish momentum"
        else:
            p, note = 0.0, "neutral"
        add("RSI 14 (15m)", p, f"{r:.1f}", note)
    if len(c) >= 35:
        h = float(macd(c)[2].iloc[-1])
        add("MACD histogram (15m)", 0.5 if h > 0 else -0.5, f"{h:.2f}", "positive" if h > 0 else "negative")
    vol = pd.to_numeric(bars["volume"], errors="coerce")
    if len(vol) >= 21 and vol.iloc[-21:-1].mean() > 0:
        ratio = float(vol.iloc[-1] / vol.iloc[-21:-1].mean())
        body = float(bars["close"].iloc[-1] - bars["open"].iloc[-1])
        p = (0.5 if body > 0 else -0.5) if ratio >= 1.3 and body != 0 else 0.0
        add("Volume (last bar)", p, f"{ratio:.2f}x", "volume confirms the bar" if p else "no volume surge")
    pcr = stats.get("pcr")
    if pcr is not None:
        p = 0.5 if pcr >= 1.2 else -0.5 if pcr <= 0.7 else 0.0
        add("Put-call ratio (OI)", p, f"{pcr:.2f}",
            "put writers support" if p > 0 else "call writers cap" if p < 0 else "neutral")
    return comps


# =============================================================================
# Levels
# =============================================================================
def resistance_levels(bars, daily, today, stats, entry) -> List[tuple]:
    prev = daily[daily["datetime"].dt.date < today]
    out = []
    if not prev.empty:
        last = prev.iloc[-1]
        piv = classic_pivots(last["high"], last["low"], last["close"])
        out += [("Previous-day high", float(last["high"])), ("Pivot R1", piv["R1"]), ("Pivot R2", piv["R2"]),
                ("Pivot R3", piv["R3"]), ("Pivot P", piv["P"]),
                ("20-day high", float(prev["high"].tail(20).max())), ("52-week high", float(prev["high"].tail(248).max()))]
    tb = bars[bars.index.date == today]
    if not tb.empty:
        out.append(("Today's high", float(tb["high"].max())))
    if stats.get("ce_wall"):
        out.append(("Highest CE OI strike", stats["ce_wall"]))
    return sorted([(n, v) for n, v in out if finite(v) and v > entry], key=lambda x: x[1])


def support_levels(bars, daily, today, stats, entry) -> List[tuple]:
    prev = daily[daily["datetime"].dt.date < today]
    out = []
    if not prev.empty:
        last = prev.iloc[-1]
        piv = classic_pivots(last["high"], last["low"], last["close"])
        out += [("Previous-day low", float(last["low"])), ("Pivot S1", piv["S1"]), ("Pivot S2", piv["S2"]),
                ("Pivot S3", piv["S3"]), ("Pivot P", piv["P"]),
                ("20-day low", float(prev["low"].tail(20).min())), ("52-week low", float(prev["low"].tail(248).min()))]
    tb = bars[bars.index.date == today]
    if not tb.empty:
        out.append(("Today's low", float(tb["low"].min())))
    if stats.get("pe_wall"):
        out.append(("Highest PE OI strike", stats["pe_wall"]))
    return sorted([(n, v) for n, v in out if finite(v) and v < entry], key=lambda x: -x[1])


def underlying_levels(direction, entry, bars, daily, today, stats, a15, a_day, now, cfg: StockConfig):
    """Spot stop / targets + checks. sign = +1 CE (long), -1 PE (short)."""
    sign = 1 if direction == "CE" else -1
    checks, notes = [], []
    last3 = bars.tail(3)
    structure = float(last3["low"].min()) - 0.1 * a15 if sign > 0 else float(last3["high"].max()) + 0.1 * a15
    line, st_dir = supertrend(bars, 10, 3.0)
    stop = structure
    if finite(line.iloc[-1]) and int(st_dir.iloc[-1]) == sign:
        st_stop = float(line.iloc[-1]) - sign * 0.1 * a15
        if sign * (entry - st_stop) > 0:
            stop = max(stop, st_stop) if sign > 0 else min(stop, st_stop)   # the tighter valid stop
    risk = sign * (entry - stop)
    if risk < cfg.min_stop_atr * a15:
        risk = cfg.min_stop_atr * a15
        stop = entry - sign * risk
        notes.append(f"stop widened to {cfg.min_stop_atr} ATR to stay outside 15-minute noise")
    ok = risk <= cfg.max_stop_atr * a15
    checks.append({"name": "Stop distance", "ok": ok,
                   "detail": f"{risk:.2f} pts = {risk / a15:.2f} x ATR15 (max {cfg.max_stop_atr})" +
                             ("" if ok else " - entry is extended from structure; wait for a pullback")})

    levels = resistance_levels(bars, daily, today, stats, entry) if sign > 0 else \
        support_levels(bars, daily, today, stats, entry)
    buffer = 0.1 * a15
    want1, want2 = entry + sign * cfg.rr_t1 * risk, entry + sign * cfg.rr_t2 * risk
    blocking = [(n, v) for n, v in levels if sign * (v - entry) > 0.15 * a15]
    t1, t1_note = want1, f"{cfg.rr_t1} R"
    if blocking and sign * (blocking[0][1] - buffer - want1) < 0:
        t1, t1_note = blocking[0][1] - sign * buffer, f"just before {blocking[0][0]} {blocking[0][1]:.2f}"
    rr1 = sign * (t1 - entry) / risk
    checks.append({"name": "Room to target", "ok": rr1 >= cfg.min_rr,
                   "detail": f"T1 {t1_note}: {rr1:.2f} R (min {cfg.min_rr})" +
                             ("" if rr1 >= cfg.min_rr else f" - {blocking[0][0]} is too close")})
    beyond = [(n, v) for n, v in blocking if sign * (v - buffer - t1) > 0.25 * risk]
    t2 = want2
    if beyond and sign * (beyond[0][1] - buffer - want2) < 0:
        t2 = beyond[0][1] - sign * buffer
    if sign * (t2 - t1) < 0.25 * risk:
        t2 = t1
    left = minutes_left(now)
    reach = a_day * math.sqrt(max(left, 1.0) / 375.0) if finite(a_day) else float("nan")
    if finite(reach):
        ok = sign * (t1 - entry) <= reach
        checks.append({"name": "Reachable today", "ok": ok,
                       "detail": f"T1 needs {abs(t1 - entry):.2f} pts; expected range left today ~{reach:.2f} "
                                 f"({left:.0f} min)"})
    return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "risk": risk, "rr1": rr1,
            "rr2": sign * (t2 - entry) / risk, "levels": levels[:5]}, checks, notes


# =============================================================================
# Option selection + premium mapping
# =============================================================================
def pick_option(chain: pd.DataFrame, spot: float, kind: str, years: float, fallback_vol: float, cfg: StockConfig):
    side = kind.lower()
    rows = []
    for _, r in chain.iterrows():
        ltp = fnum(r.get(f"{side}_ltp"))
        if not finite(ltp) or ltp <= 0:
            continue
        delta, gamma, theta = fnum(r.get(f"{side}_delta")), fnum(r.get(f"{side}_gamma")), fnum(r.get(f"{side}_theta"))
        iv = fnum(r.get(f"{side}_iv"))
        estimated = False
        if not finite(delta) or delta == 0:
            vol = iv / 100 if finite(iv) and iv > 0 else fallback_vol
            g = bs_greeks(spot, float(r["strike"]), years, vol, kind)
            delta, gamma, theta, estimated = g["delta"], g["gamma"], g["theta"], True
        bid, ask = fnum(r.get(f"{side}_bid")), fnum(r.get(f"{side}_ask"))
        spread = (ask - bid) / ((ask + bid) / 2) * 100 if finite(bid) and finite(ask) and bid > 0 and ask >= bid else None
        oi = fnum(r.get(f"{side}_oi"), 0.0)
        rows.append({"strike": float(r["strike"]), "ltp": ltp, "bid": bid if finite(bid) else None,
                     "ask": ask if finite(ask) else None, "spread_pct": spread, "oi": oi,
                     "volume": fnum(r.get(f"{side}_volume"), 0.0), "delta": delta,
                     "gamma": gamma if finite(gamma) else 0.0, "theta": theta if finite(theta) else 0.0,
                     "iv": iv if finite(iv) else None, "key": r.get(f"{side}_key"), "greeks_estimated": estimated})
    if not rows:
        return None, []
    liquid = [x for x in rows if x["oi"] > 0 and (x["spread_pct"] is None or x["spread_pct"] <= cfg.max_spread_pct)
              and cfg.min_delta <= abs(x["delta"]) <= cfg.max_delta]
    pool = liquid or rows
    best = min(pool, key=lambda x: abs(abs(x["delta"]) - cfg.target_delta) + (x["spread_pct"] or 0) / 50)
    best["liquid"] = bool(liquid)
    return best, rows


def premium_at(opt: dict, spot_move: float, theta_days: float = 0.0) -> float:
    p = opt["entry"] + opt["delta"] * spot_move + 0.5 * opt["gamma"] * spot_move ** 2 - abs(opt["theta"]) * theta_days
    return max(p, 0.05)



def option_leg(symbol: str, direction: str, spot: float, entry_spot: float, stop_spot: float, t1_spot: float,
               t2_spot: float, daily: pd.DataFrame, now: pd.Timestamp, chain: Optional[pd.DataFrame],
               expiry: Optional[date], lot_size: Optional[int], cfg: StockConfig = StockConfig(),
               cost_fn: Optional[Callable] = None):
    """CE / PE leg for spot levels. Premiums are mapped from the CURRENT spot with delta + gamma, so a plan whose
    entry is a trigger above / below the spot (pre-breakout) gets the premium expected at that trigger.
    Returns (option dict or None, checks, notes)."""
    checks, notes = [], []
    if chain is not None and not chain.empty:
        years = max(((pd.Timestamp(datetime.combine(expiry, SESSION_CLOSE)).tz_localize(TZ) - now)
                     .total_seconds() / (365 * 86400)) if expiry else 20 / 365, 1 / 365)
        hv = float(np.log(daily["close"]).diff().tail(20).std() * math.sqrt(248)) if len(daily) > 21 else 0.3
        opt, _rows = pick_option(chain, spot, direction, years, hv if finite(hv) and hv > 0 else 0.3, cfg)
    else:
        opt = None
    if opt is None:
        checks.append({"name": "Option contract", "ok": False, "detail": "no option chain / premium available"})
        return None, checks, notes
    use_ask = opt["ask"] is not None and opt["ask"] > 0 and (opt["spread_pct"] or 0) <= cfg.max_spread_pct
    base = dict(opt, entry=opt["ask"] if use_ask else opt["ltp"])     # price paid now
    opt["entry"] = premium_at(base, entry_spot - spot)                # = base entry when entry_spot == spot
    sign = 1 if direction == "CE" else -1
    theta_days = cfg.hold_hours / 6.25
    sl = premium_at(base, stop_spot - spot)
    floor = opt["entry"] * (1 - cfg.max_premium_loss_pct / 100)
    if sl < floor:
        sl = floor
        notes.append(f"premium stop capped at {cfg.max_premium_loss_pct:.0f}% loss")
    t1 = premium_at(base, t1_spot - spot, theta_days)
    t2 = premium_at(base, t2_spot - spot, theta_days)
    lot = int(lot_size) if lot_size else 1
    cost = cost_fn or (lambda side, price, qty, exchange="NSE_FO": 0.0)
    entry_cost = cost("BUY", opt["entry"], lot)
    risk_rs = (opt["entry"] - sl) * lot + entry_cost + cost("SELL", sl, lot)
    reward1 = (t1 - opt["entry"]) * lot - entry_cost - cost("SELL", t1, lot)
    reward2 = (t2 - opt["entry"]) * lot - entry_cost - cost("SELL", t2, lot)
    prr = reward1 / risk_rs if risk_rs > 0 else 0.0
    liq_ok = opt.get("liquid", False)
    checks.append({"name": "Liquidity", "ok": liq_ok,
                   "detail": (f"OI {opt['oi']:,.0f}, spread {opt['spread_pct']:.1f}%" if opt["spread_pct"] is not None
                              else f"OI {opt['oi']:,.0f}, no bid/ask") +
                             ("" if liq_ok else " - no strike meets OI / spread / delta limits")})
    checks.append({"name": "Premium reward:risk", "ok": prr >= cfg.min_rr,
                   "detail": f"T1 {prr:.2f} R net of charges (min {cfg.min_rr})"})
    exp_txt = expiry.strftime("%d %b").upper() if expiry else ""
    out = {
        "kind": direction, "strike": opt["strike"], "expiry": expiry.isoformat() if expiry else None,
        "contract": f"{symbol} {exp_txt} {opt['strike']:g} {direction}".strip(), "key": opt["key"],
        "ltp": r2(opt["ltp"]), "bid": r2(opt["bid"]), "ask": r2(opt["ask"]), "entry": tick(opt["entry"]),
        "stop": tick(sl), "t1": tick(t1), "t2": tick(t2), "delta": r2(opt["delta"]), "iv": r2(opt["iv"]),
        "theta": r2(opt["theta"]), "greeks_estimated": opt["greeks_estimated"], "lot_size": lot,
        "capital": r2(opt["entry"] * lot), "risk_rs": r2(risk_rs), "reward1_rs": r2(reward1),
        "reward2_rs": r2(reward2), "rr": r2(prr), "sign": sign}
    if opt["greeks_estimated"]:
        notes.append("greeks estimated with Black-Scholes (chain had none)")
    return out, checks, notes

# =============================================================================
# Plan
# =============================================================================
def build_plan(symbol: str, bars: pd.DataFrame, daily: pd.DataFrame, spot: float, now: pd.Timestamp,
               chain: Optional[pd.DataFrame] = None, expiry: Optional[date] = None, lot_size: Optional[int] = None,
               cfg: StockConfig = StockConfig(), cost_fn: Optional[Callable] = None, live: bool = True) -> dict:
    today = now.date()
    plan = {"symbol": symbol, "time": now.isoformat(), "status": "NO_TRADE", "direction": None, "score": 0.0,
            "components": [], "checks": [], "notes": [], "spot": None, "option": None, "reason": ""}
    bars = completed(bars, now) if live else bars
    if bars is None or len(bars) < 30 or daily is None or len(daily) < 30:
        plan["reason"] = "not enough price history yet"
        return plan
    if not finite(spot):
        spot = float(bars["close"].iloc[-1])
    plan["ltp"] = r2(spot)
    stats = chain_stats(chain, spot) if chain is not None else {}
    plan["chain"] = {k: (r2(v) if isinstance(v, float) else v) for k, v in stats.items()}
    comps = score_components(bars, daily, today, stats)
    score = round(sum(c["points"] for c in comps), 2)
    plan["components"], plan["score"] = comps, score
    if abs(score) < cfg.watch_score:
        plan["reason"] = f"no clear direction (score {score:+.1f}; needs +/-{cfg.min_score})"
        return plan
    direction = "CE" if score > 0 else "PE"
    plan["direction"] = direction
    a15 = float(atr(bars, 14).iloc[-1])
    prev_daily = daily[daily["datetime"].dt.date < today]
    a_day = float(atr(prev_daily.set_index("datetime"), 14).iloc[-1]) if len(prev_daily) > 15 else float("nan")
    if not finite(a15) or a15 <= 0:
        plan["reason"] = "ATR unavailable"
        return plan
    lv, checks, notes = underlying_levels(direction, spot, bars, daily, today, stats, a15, a_day, now, cfg)
    plan["notes"] += notes
    checks.insert(0, {"name": "Signal strength", "ok": abs(score) >= cfg.min_score,
                      "detail": f"score {score:+.1f} (needs {'+' if score > 0 else '-'}{cfg.min_score})"})
    t = now.time()
    in_window = _t(cfg.entry_start) <= t <= _t(cfg.entry_end) and live
    checks.append({"name": "Entry window", "ok": in_window,
                   "detail": f"new entries {cfg.entry_start}-{cfg.entry_end}, square off {cfg.square_off}" +
                             ("" if live else " (market closed - analysis of the last session)")})
    plan["spot"] = {"entry": r2(lv["entry"]), "stop": tick(lv["stop"]), "t1": tick(lv["t1"]), "t2": tick(lv["t2"]),
                    "risk": r2(lv["risk"]), "rr1": r2(lv["rr1"]), "rr2": r2(lv["rr2"]), "atr15": r2(a15),
                    "atr_day": r2(a_day), "levels": [{"name": n, "value": r2(v)} for n, v in lv["levels"]]}

    # ---- option leg
    opt, o_checks, o_notes = option_leg(symbol, direction, spot, spot, lv["stop"], lv["t1"], lv["t2"], daily, now,
                                        chain, expiry, lot_size, cfg, cost_fn)
    checks += o_checks
    plan["notes"] += o_notes
    plan["option"] = opt
    plan["checks"] = checks
    failed = [c["name"] for c in checks if not c["ok"]]
    if abs(score) >= cfg.min_score and not failed:
        plan["status"] = "VALID"
        plan["reason"] = f"{'bullish' if direction == 'CE' else 'bearish'} setup, all checks passed"
    else:
        plan["status"] = "WATCH"
        plan["reason"] = "waiting: " + ", ".join(failed).lower()
    return plan

