"""
Precision mode for the NIFTY 50 stock CE / PE plans: a second, stricter gate on top of a VALID plan, plus a
ranking, so only a handful of the best plans per day become "top picks" (and paper trades).

Why: the first three paper days (13, 16, 17 Nov 2026) took 12-28 trades a day at a 32 % win rate. The losers
clustered in (a) CEs bought while the NIFTY itself was falling, (b) entries after 13:00, (c) the highest scores,
which mostly meant the stock had already run (stretched from VWAP / RSI extreme). Three days prove nothing:
these filters are hypotheses to paper-trade and backtest, not a proven edge.

Checks a VALID plan must also pass (CE shown; PE is the mirror):
  NIFTY alignment     at least nifty_min_votes of 3 NIFTY votes agree: day change vs previous close,
                      EMA 9 vs EMA 21 (15m), Supertrend (10,3) on 15m
  Relative strength   stock's change today beats the NIFTY's by >= min_rs_pct (PE: lags it)
  Not stretched       price within max_vwap_atr x ATR(15m) of today's VWAP, RSI(14) <= rsi_ce_max (PE >= rsi_pe_min)
  Confirmation bar    the last completed 15m bar closed in the trade's direction, close in the outer half
  Premium quality     premium >= min_premium, bid-ask spread <= max_spread_pct of the premium
  Reward : risk       premium R:R to T1 after charges >= min_rr
  Entry cut-off       no new pick after last_entry

Rank 0-100 among the plans that pass: reward:risk 30 + NIFTY alignment 20 + relative strength 20 +
closeness to VWAP 15 + spread tightness 15. Top picks = highest rank first, at most max_picks_per_day paper
trades a day and max_open at once.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass, fields
from typing import List, Optional

import pandas as pd

from src.scanner.indicators import atr, ema, rsi
from src.stocks.analytics import day_vwap, supertrend
from src.utils.numbers import finite


@dataclass(frozen=True)
class PrecisionConfig:
    enabled: bool = True
    max_picks_per_day: int = 8
    max_open: int = 5
    last_entry: str = "14:30"
    require_nifty_align: bool = True
    nifty_min_votes: int = 2
    require_relative_strength: bool = True
    min_rs_pct: float = 0.0
    max_vwap_atr: float = 1.5
    rsi_ce_max: float = 70.0
    rsi_pe_min: float = 30.0
    require_confirm_bar: bool = True
    min_premium: float = 8.0
    max_spread_pct: float = 2.0
    min_rr: float = 1.5

    @classmethod
    def from_ini(cls, path) -> "PrecisionConfig":
        parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        try:
            parser.read(path, encoding="utf-8")
        except configparser.Error:
            return cls()
        if not parser.has_section("STOCK_PRECISION"):
            return cls()
        sec, values = parser["STOCK_PRECISION"], {}
        for f in fields(cls):
            if f.name not in sec:
                continue
            raw = sec[f.name].strip()
            if f.type in ("bool", bool):
                values[f.name] = raw.lower() in ("1", "true", "yes", "on")
            elif f.type in ("int", int):
                values[f.name] = int(float(raw))
            elif f.type in ("float", float):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        return cls(**values)


def _t(text: str):
    hh, mm = text.split(":")
    return int(hh), int(mm)


def day_change_pct(bars: pd.DataFrame, today) -> Optional[float]:
    """Change of the last completed bar vs the previous session's last close, in %."""
    if bars is None or bars.empty:
        return None
    prev = bars[bars.index.date < today]
    now = bars[bars.index.date == today]
    if prev.empty or now.empty:
        return None
    pc = float(prev["close"].iloc[-1])
    return (float(now["close"].iloc[-1]) / pc - 1) * 100 if pc > 0 else None


def nifty_context(index_bars: Optional[pd.DataFrame], now: pd.Timestamp) -> dict:
    """NIFTY direction from completed 15m bars: three votes, each +1 / -1 (0 when unknown). bias = their sum."""
    out = {"bias": 0, "votes": [], "change_pct": None, "available": False}
    if index_bars is None or len(index_bars) < 25:
        return out
    bars = index_bars[index_bars.index + pd.Timedelta(minutes=15) <= now]
    if len(bars) < 25:
        return out
    votes = []
    chg = day_change_pct(bars, now.date())
    if chg is not None:
        votes.append(("Day change", 1 if chg > 0 else -1 if chg < 0 else 0, f"{chg:+.2f}%"))
    c = bars["close"].astype(float)
    e9, e21 = float(ema(c, 9).iloc[-1]), float(ema(c, 21).iloc[-1])
    votes.append(("EMA 9/21 (15m)", 1 if e9 > e21 else -1, f"{e9:.1f} / {e21:.1f}"))
    _line, direction = supertrend(bars, 10, 3.0)
    if finite(direction.iloc[-1]):
        d = int(direction.iloc[-1])
        votes.append(("Supertrend (15m)", d, "up" if d > 0 else "down"))
    out.update(bias=sum(v for _, v, _ in votes), change_pct=chg, available=True,
               votes=[{"name": n, "vote": v, "value": x} for n, v, x in votes])
    return out


def evaluate(plan: dict, bars: pd.DataFrame, nifty: dict, now: pd.Timestamp,
             cfg: PrecisionConfig = PrecisionConfig()) -> Optional[dict]:
    """Precision checks + rank for one VALID plan. Returns None for plans that are not VALID."""
    if plan.get("status") != "VALID" or not plan.get("option") or bars is None or bars.empty:
        return None
    sign = 1 if plan["direction"] == "CE" else -1
    opt = plan["option"]
    bars = bars[bars.index + pd.Timedelta(minutes=15) <= now]
    today = now.date()
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # NIFTY alignment
    bias = nifty.get("bias", 0)
    agree = sum(1 for v in nifty.get("votes", []) if v["vote"] == sign)
    if cfg.require_nifty_align:
        detail = (", ".join(f"{v['name']} {v['value']}" for v in nifty["votes"])
                  if nifty.get("available") else "NIFTY bars unavailable")
        check("NIFTY alignment", nifty.get("available") and agree >= cfg.nifty_min_votes,
              f"{agree}/3 NIFTY votes {'up' if sign > 0 else 'down'} (need {cfg.nifty_min_votes}): {detail}")

    # relative strength
    s_chg, n_chg = day_change_pct(bars, today), nifty.get("change_pct")
    rs = sign * (s_chg - n_chg) if s_chg is not None and n_chg is not None else None
    if cfg.require_relative_strength:
        check("Relative strength", rs is not None and rs >= cfg.min_rs_pct,
              f"stock {s_chg:+.2f}% vs NIFTY {n_chg:+.2f}% -> {rs:+.2f}% in the trade's favour (min {cfg.min_rs_pct})"
              if rs is not None else "day change unavailable")

    # not stretched
    a15 = float(atr(bars, 14).iloc[-1]) if len(bars) > 15 else float("nan")
    last = float(bars["close"].iloc[-1])
    vwap = day_vwap(bars[bars.index.date == today])
    dist = sign * (last - vwap) / a15 if finite(vwap) and finite(a15) and a15 > 0 else None
    check("Not stretched from VWAP", dist is not None and 0 <= dist <= cfg.max_vwap_atr,
          f"{dist:.2f} ATR on the trade's side of VWAP {vwap:.2f} (allowed 0-{cfg.max_vwap_atr})"
          if dist is not None else "VWAP unavailable")
    r = float(rsi(bars["close"].astype(float)).iloc[-1]) if len(bars) > 15 else float("nan")
    rsi_ok = finite(r) and (r <= cfg.rsi_ce_max if sign > 0 else r >= cfg.rsi_pe_min)
    check("RSI not extreme", rsi_ok, f"RSI {r:.1f} ({'max ' + str(cfg.rsi_ce_max) if sign > 0 else 'min ' + str(cfg.rsi_pe_min)})"
          if finite(r) else "RSI unavailable")

    # confirmation bar
    if cfg.require_confirm_bar:
        b = bars.iloc[-1]
        rng = float(b["high"] - b["low"])
        body = sign * float(b["close"] - b["open"])
        pos = ((float(b["close"]) - float(b["low"])) / rng if sign > 0 else (float(b["high"]) - float(b["close"])) / rng) \
            if rng > 0 else 0.0
        check("Confirmation bar", body > 0 and pos >= 0.5,
              f"last 15m bar {'green' if b['close'] > b['open'] else 'red' if b['close'] < b['open'] else 'flat'}, "
              f"close at {pos * 100:.0f}% of its range toward the trade")

    # premium quality
    prem = opt.get("entry") or 0.0
    spread = None
    if opt.get("bid") and opt.get("ask") and prem > 0:
        spread = (opt["ask"] - opt["bid"]) / prem * 100
    check("Premium and spread", prem >= cfg.min_premium and (spread is None or spread <= cfg.max_spread_pct),
          f"premium {prem:.2f} (min {cfg.min_premium}), spread " +
          (f"{spread:.1f}% (max {cfg.max_spread_pct}%)" if spread is not None else "unknown"))
    rr = opt.get("rr") or 0.0
    check("Reward:risk", rr >= cfg.min_rr, f"{rr:.2f} R to T1 after charges (min {cfg.min_rr})")

    hh, mm = _t(cfg.last_entry)
    check("Pick cut-off", (now.hour, now.minute) <= (hh, mm), f"new picks until {cfg.last_entry}")

    parts = {
        "reward_risk": 30 * min(rr / 3.0, 1.0),
        "nifty": 20 * max(sign * bias, 0) / 3.0,
        "relative_strength": 20 * min(max(rs or 0.0, 0.0) / 1.0, 1.0),
        "vwap": 15 * (1 - min(abs(dist), cfg.max_vwap_atr) / cfg.max_vwap_atr) if dist is not None and cfg.max_vwap_atr > 0 else 0.0,
        "spread": 15 * (1 - min(spread, cfg.max_spread_pct) / cfg.max_spread_pct) if spread is not None and cfg.max_spread_pct > 0 else 7.5,
    }
    rank = round(sum(parts.values()), 1)
    failed = [c["name"] for c in checks if not c["ok"]]
    return {"ok": not failed, "rank": rank, "parts": {k: round(v, 1) for k, v in parts.items()},
            "checks": checks, "failed": failed}


def select(candidates: List[dict], taken_today: int, open_now: int, held: set,
           cfg: PrecisionConfig = PrecisionConfig()) -> List[dict]:
    """Plans (with plan['precision']['ok']) that may open a paper trade now, best rank first, inside the daily and
    open-position caps. `held` = symbols already traded today."""
    room = min(cfg.max_picks_per_day - taken_today, cfg.max_open - open_now)
    if room <= 0:
        return []
    ok = [p for p in candidates if p.get("precision", {}).get("ok") and p["symbol"] not in held]
    ok.sort(key=lambda p: -p["precision"]["rank"])
    return ok[:room]
