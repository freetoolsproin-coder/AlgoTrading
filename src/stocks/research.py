"""
Research report for one stock (search bar on the stock-options page).

Sections: snapshot, performance vs NIFTY, risk (volatility, beta, ATR), daily trend + momentum, key levels
(pivots, swing support / resistance, 52-week range), options positioning (PCR, max pain, OI walls, ATM IV),
today's intraday CE / PE plan from the signal engine, an overall verdict with the reasons, and fundamentals.

Fundamentals (P/E, P/B, market cap, ROE, margins, dividend yield) need the optional `yfinance` package
(pip install yfinance) - Upstox has no fundamentals API. Without it that section says so; everything else
comes from Upstox market data.
"""
from __future__ import annotations

import logging
import threading
import time as clock
from typing import Optional

from src.stocks.analytics import (beta, chain_stats, classic_pivots, daily_indicators, hist_vol, returns,
                                  swing_levels)
from src.stocks.signals import r2
from src.utils.numbers import finite

logger = logging.getLogger("scanner.stocks.research")
_cache: dict = {}
_lock = threading.Lock()
CACHE_SECONDS = 120


def fundamentals(symbol: str) -> dict:
    try:
        import yfinance as yf          # optional
    except ImportError:
        return {"available": False, "note": "install yfinance (pip install yfinance) to add P/E, P/B, ROE, "
                                            "market cap and dividend yield"}
    try:
        info = yf.Ticker(f"{symbol.replace('&', '_')}.NS").info or {}
    except Exception as error:
        return {"available": False, "note": f"fundamentals unavailable: {error}"}
    pick = {"Market cap (Cr)": (info.get("marketCap") or 0) / 1e7 or None, "P/E (trailing)": info.get("trailingPE"),
            "P/E (forward)": info.get("forwardPE"), "P/B": info.get("priceToBook"), "EPS (TTM)": info.get("trailingEps"),
            "ROE %": (info.get("returnOnEquity") or 0) * 100 or None,
            "Profit margin %": (info.get("profitMargins") or 0) * 100 or None,
            "Debt / equity": info.get("debtToEquity"),
            "Dividend yield %": info.get("dividendYield"),
            "Revenue growth %": (info.get("revenueGrowth") or 0) * 100 or None}
    return {"available": True, "source": "Yahoo Finance via yfinance (delayed)", "sector": info.get("sector"),
            "industry": info.get("industry"), "summary": (info.get("longBusinessSummary") or "")[:700],
            "metrics": {k: r2(v) for k, v in pick.items() if finite(v)}}


def verdict(ind: dict, rets: dict, plan: Optional[dict], stats: dict) -> dict:
    pts, why = 0.0, []
    c = ind.get("close")

    def note(p, text):
        nonlocal pts
        pts += p
        why.append({"points": p, "text": text})

    if c and ind.get("ema50"):
        note(1 if c > ind["ema50"] else -1, f"price {'above' if c > ind['ema50'] else 'below'} the 50-day EMA")
    if c and ind.get("ema200"):
        note(1 if c > ind["ema200"] else -1, f"price {'above' if c > ind['ema200'] else 'below'} the 200-day EMA")
    if ind.get("ema50") and ind.get("ema200"):
        golden = ind["ema50"] > ind["ema200"]
        note(0.5 if golden else -0.5, "50-day EMA above 200-day (golden-cross zone)" if golden
             else "50-day EMA below 200-day (death-cross zone)")
    if ind.get("rsi") is not None:
        r = ind["rsi"]
        if r >= 70:
            note(-0.5, f"daily RSI {r:.0f}: overbought, upside may be limited short term")
        elif r <= 30:
            note(0.5, f"daily RSI {r:.0f}: oversold, bounce risk for shorts")
        elif r > 55:
            note(0.5, f"daily RSI {r:.0f}: positive momentum")
        elif r < 45:
            note(-0.5, f"daily RSI {r:.0f}: weak momentum")
    if ind.get("macd_hist") is not None:
        note(0.5 if ind["macd_hist"] > 0 else -0.5, f"daily MACD histogram {'positive' if ind['macd_hist'] > 0 else 'negative'}")
    if rets.get("3M") is not None:
        note(0.5 if rets["3M"] > 0 else -0.5, f"3-month return {rets['3M']:+.1f}%")
    if stats.get("pcr") is not None:
        p = stats["pcr"]
        if p >= 1.2:
            note(0.5, f"PCR {p:.2f}: put writers defending (supportive)")
        elif p <= 0.7:
            note(-0.5, f"PCR {p:.2f}: call writers active (capping)")
    view = "Bullish" if pts >= 2 else "Bearish" if pts <= -2 else "Neutral"
    intraday = None
    if plan and plan.get("direction"):
        intraday = f"{plan['status']} {plan['direction']} (score {plan['score']:+.1f})"
    elif plan:
        intraday = f"no intraday trade ({plan.get('reason')})"
    return {"positional": view, "score": round(pts, 1), "reasons": why, "intraday": intraday}


def build_report(source, engine, symbol: str, use_cache: bool = True) -> Optional[dict]:
    symbol = symbol.strip().upper()
    with _lock:
        hit = _cache.get(symbol)
        if use_cache and hit and clock.monotonic() - hit[0] < CACHE_SECONDS:
            return hit[1]
    stock = source.lookup(symbol)
    if stock is None:
        return None
    now = source.now()
    daily = source.daily(stock.key)
    if daily is None or len(daily) < 30:
        return {"symbol": symbol, "error": "not enough price history for this stock"}
    q = (source.quotes([stock.key]) or {}).get(stock.key) or {}
    ltp = q.get("ltp") if finite(q.get("ltp")) else float(daily["close"].iloc[-1])
    prev = daily[daily["datetime"].dt.date < now.date()]
    prev_close = q.get("prev_close") if finite(q.get("prev_close")) else float(prev["close"].iloc[-1])
    year = prev.tail(248)
    hi52, lo52 = float(year["high"].max()), float(year["low"].min())
    ind = daily_indicators(daily)
    rets = returns(daily)
    try:
        idx = source.index_daily()
        nifty_rets = returns(idx)
    except Exception:
        idx, nifty_rets = None, {}
    last = prev.iloc[-1]
    piv = classic_pivots(float(last["high"]), float(last["low"]), float(last["close"]))
    swings = swing_levels(prev)
    res = sorted({round(v, 2) for v in swings["highs"] if v > ltp})[:3]
    sup = sorted({round(v, 2) for v in swings["lows"] if v < ltp}, reverse=True)[:3]

    plan = engine.plan_for(symbol) if engine is not None else None
    stats, option_info = {}, None
    if plan is None and engine is not None:
        from src.stocks.data import is_market_open
        live = source.mock or is_market_open(now)
        try:
            plan = engine.analyse(stock, now, ltp, live, with_options=stock.nifty50)
        except Exception as error:
            logger.warning("plan for %s failed: %s", symbol, error)
    if plan and plan.get("chain"):
        stats = plan["chain"]
    elif stock.nifty50 and engine is not None:
        try:
            chain, expiry, lot = engine.option_inputs(stock, now.date())
            stats = chain_stats(chain, ltp) if chain is not None else {}
            option_info = {"expiry": expiry.isoformat() if expiry else None, "lot_size": lot}
        except Exception as error:
            logger.info("no option data for %s: %s", symbol, error)
    if plan and plan.get("option"):
        option_info = {"expiry": plan["option"]["expiry"], "lot_size": plan["option"]["lot_size"]}

    report = {
        "symbol": symbol, "name": stock.name, "industry": stock.industry, "nifty50": stock.nifty50,
        "generated_at": now.isoformat(), "mock": source.mock,
        "snapshot": {"ltp": r2(ltp), "change": r2(ltp - prev_close), "change_pct": r2((ltp / prev_close - 1) * 100),
                     "open": r2(q.get("open")), "high": r2(q.get("high")), "low": r2(q.get("low")),
                     "prev_close": r2(prev_close), "volume": q.get("volume"),
                     "high_52w": r2(hi52), "low_52w": r2(lo52),
                     "range_pos_pct": r2((ltp - lo52) / (hi52 - lo52) * 100) if hi52 > lo52 else None,
                     "from_high_pct": r2((ltp / hi52 - 1) * 100)},
        "performance": {"stock": rets, "nifty": nifty_rets},
        "risk": {"hist_vol_20d": hist_vol(daily), "beta_1y": beta(daily, idx),
                 "atr_14d": ind.get("atr"), "atr_pct": r2(ind["atr"] / ltp * 100) if ind.get("atr") else None},
        "technicals": ind,
        "levels": {"pivots": {k: r2(v) for k, v in piv.items()}, "resistance": res, "support": sup},
        "options": {**stats, **(option_info or {})} if (stats or option_info) else None,
        "plan": plan,
        "verdict": verdict(ind, rets, plan, stats),
        "fundamentals": fundamentals(symbol) if not source.mock else
        {"available": False, "note": "fundamentals are not shown in mock mode"},
        "disclaimer": "Generated from market data by rules for paper trading and research. Not investment advice.",
    }
    with _lock:
        _cache[symbol] = (clock.monotonic(), report)
    return report
