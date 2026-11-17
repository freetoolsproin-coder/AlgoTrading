"""
Replay backtest of the NIFTY 50 stock pre-breakout scanner (src/stocks/breakout.py) on 15-minute candles.

    python -m src.stocks.breakout_backtest                  last 6 months of Upstox history (needs today's token)
    python -m src.stocks.breakout_backtest --months 12      longer history (downloaded in 1-month pieces, cached)
    python -m src.stocks.breakout_backtest --mock           synthetic market, no token (checks the plumbing only)
    python -m src.stocks.breakout_backtest --symbols RELIANCE,TCS --entry-mode touch

The replay uses exactly the live code: features(), evaluate(), Tracker and equity_leg(), bar by bar, all 50
stocks in time order, with the same READY-before-breakout rule, entry window, trade cap and exits:
  entry   close of the CONFIRMED bar (entry_mode = close) or the trigger when a later bar trades through it
          after READY (entry_mode = touch; a gap through the trigger fills at the open)
  exits   stop / T1 (stop to cost) / T2 / square-off at the 15:00 bar close; when one bar touches both the stop
          and a target, the stop is assumed first (conservative)
  costs   intraday equity charges ([COSTS_EQUITY]) + slippage_pct on every fill

It tests the EQUITY leg only: there is no stock-option history here, so the CE / PE leg cannot be replayed
honestly. Results also show:
  * READY quality: how often the trigger traded within 4 bars, and the directional move 4 bars later (ATR)
  * a baseline: every quality breakout of the same levels without the score / READY filter
  * first 70 % vs last 30 % of the days, so a result that only holds in-sample is visible
Output: output/stock_breakout_backtest.txt and output/stock_breakout_trades.csv.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote as urlquote

import numpy as np
import pandas as pd

from src.stocks.breakout import BAR, BreakoutConfig, Tracker, equity_leg, evaluate, features
from src.utils.numbers import finite, hhmm

logger = logging.getLogger("scanner.stocks.breakout_bt")
TZ = "Asia/Kolkata"


# =============================================================================
# Data
# =============================================================================
def load_live(symbols: Optional[List[str]], months: int, cache_dir: Path):
    from src.broker.client import UpstoxClient, candles_frame
    from src.stocks.data import to_bars
    from src.stocks.universe import NIFTY_INDEX_KEY, load_universe
    client = UpstoxClient()
    stocks = load_universe()
    if symbols:
        stocks = [s for s in stocks if s.symbol in symbols]
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=int(months * 30.5))
    cache_dir.mkdir(parents=True, exist_ok=True)

    def bars15(key, tag):
        f = cache_dir / f"{tag}_15m_{start}_{end}.csv.gz"
        if f.exists():
            frame = pd.read_csv(f, index_col=0, parse_dates=True)
            frame.index = pd.DatetimeIndex(frame.index).tz_convert(TZ)
            return frame
        parts, hi = [], end
        while hi >= start:
            lo = max(start, hi - timedelta(days=28))
            raw = client.get(f"/historical-candle/{urlquote(key, safe='')}/minutes/15/{hi}/{lo}",
                             base="https://api.upstox.com/v3").get("data", {}).get("candles", [])
            parts.append(to_bars(candles_frame(raw)))
            hi = lo - timedelta(days=1)
        frame = pd.concat([p for p in parts if not p.empty]).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frame.to_csv(f)
        return frame

    def daily(key, tag):
        f = cache_dir / f"{tag}_1d_{start}_{end}.csv.gz"
        if f.exists():
            d = pd.read_csv(f, parse_dates=["datetime"])
            d["datetime"] = pd.to_datetime(d["datetime"], utc=True).dt.tz_convert(TZ)
            return d
        d = client.daily_candles(key, start - timedelta(days=400), end)
        d.to_csv(f, index=False)
        return d

    data = {}
    for n, s in enumerate(stocks, 1):
        logger.info("[%d/%d] %s", n, len(stocks), s.symbol)
        try:
            data[s.symbol] = {"stock": s, "bars": bars15(s.key, s.symbol.replace("&", "_")),
                              "daily": daily(s.key, s.symbol.replace("&", "_"))}
        except Exception as error:
            if "Auth" in type(error).__name__:
                raise
            logger.warning("%s skipped: %s", s.symbol, error)
    index_bars = bars15(NIFTY_INDEX_KEY, "NIFTY")
    return data, index_bars


def load_mock(symbols: Optional[List[str]], days: int):
    from src.stocks.data import MockSource
    from src.stocks.universe import NIFTY_INDEX_KEY
    src = MockSource(clock=pd.Timestamp(pd.Timestamp.now(tz=TZ).date() - timedelta(days=1)).tz_localize(TZ)
                     + pd.Timedelta(hours=15, minutes=30), days=days + 60)
    data = {}
    for i, s in enumerate(src.stocks()):
        if symbols and s.symbol not in symbols:
            continue
        s = s.__class__(symbol=s.symbol, name=s.name, industry=f"Sector{i % 8}", key=s.key)
        data[s.symbol] = {"stock": s, "bars": src._visible(s.key), "daily": src.daily(s.key)}
    return data, src._visible(NIFTY_INDEX_KEY)


# =============================================================================
# Simulation
# =============================================================================
def simulate_exit(f: pd.DataFrame, i: int, entry: float, sign: int, stop: float, t1: float, t2: float,
                  square_off_bar: str = "15:00"):
    """Walk the bars after i (same day). Returns (exit price, status, bars held)."""
    day = f["day"].iloc[i]
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    days = f["day"].to_numpy()
    last_t = hhmm(square_off_bar)
    cur_stop, status = stop, "OPEN"
    j = i + 1
    while j < len(f) and days[j] == day:
        adverse = l[j] if sign > 0 else h[j]
        favour = h[j] if sign > 0 else l[j]
        if sign * (o[j] - cur_stop) <= 0:                        # gapped through the stop
            return o[j], ("STOP_HIT" if status == "OPEN" else "EXIT_AT_COST"), j - i
        if sign * (adverse - cur_stop) <= 0:
            return cur_stop, ("STOP_HIT" if status == "OPEN" else "EXIT_AT_COST"), j - i
        if sign * (favour - t2) >= 0:
            return (o[j] if sign * (o[j] - t2) >= 0 else t2), "T2_HIT", j - i
        if status == "OPEN" and sign * (favour - t1) >= 0:
            status, cur_stop = "T1_HIT", entry
        if f.index[j].time() >= last_t:
            return c[j], "TIME_EXIT", j - i
        j += 1
    return c[min(j, len(f)) - 1], "TIME_EXIT", j - 1 - i


def run(data: Dict[str, dict], index_bars, cfg: BreakoutConfig, eq_cost, slippage_pct: float = 0.02,
        baseline: bool = False):
    feats = {}
    for sym, x in data.items():
        try:
            feats[sym] = features(x["bars"], x["daily"], cfg, index_bars)
        except Exception as error:
            logger.warning("%s features failed: %s", sym, error)
    if not feats:
        return [], []
    # sector breadth panel: share of peers up on the day at each bar
    chg = pd.DataFrame({s: f["chg_pct"] for s, f in feats.items()})
    sectors = {s: data[s]["stock"].industry for s in feats}
    breadth = {}
    for sec in set(sectors.values()):
        cols = [s for s in feats if sectors[s] == sec]
        if len(cols) >= 3:
            breadth[sec] = (chg[cols] > 0).mean(axis=1).where(chg[cols].notna().sum(axis=1) >= 3)
    pos = {s: pd.Series(np.arange(len(f)), index=f.index) for s, f in feats.items()}
    recs = {s: f.to_dict("records") for s, f in feats.items()}
    breadth_d = {sec: s.to_dict() for sec, s in breadth.items()}
    pos_d = {s: p.to_dict() for s, p in pos.items()}
    stamps = sorted(set().union(*[set(f.index) for f in feats.values()]))
    start_t, end_t = hhmm(cfg.entry_start), hhmm(cfg.entry_end)
    trades, readies = [], []
    cur_day, tracker, open_until, day_count, traded = None, None, {}, 0, set()
    pending: Dict[str, dict] = {}                    # touch mode: READY plans waiting for the trigger
    for ts in stamps:
        d = ts.date()
        if d != cur_day:
            cur_day, tracker, day_count, traded, pending = d, Tracker(), 0, set(), {}
            tracker.roll(d)
        decided = ts + BAR
        if not (start_t <= decided.time() <= end_t) and not (cfg.entry_mode == "touch" and pending):
            continue
        for sym, f in feats.items():
            i = pos_d[sym].get(ts)
            if i is None:
                continue
            if i < 40:
                continue
            # touch-mode fills first (the trigger trades inside this bar)
            if cfg.entry_mode == "touch" and sym in pending and sym not in traded:
                p = pending[sym]
                sign = p["sign"]
                hit = sign * ((f["high"].iloc[i] if sign > 0 else f["low"].iloc[i]) - p["trigger"]) >= 0
                if hit and day_count < cfg.max_trades and decided.time() <= end_t:
                    o = float(f["open"].iloc[i])
                    entry = o if sign * (o - p["trigger"]) >= 0 else p["trigger"]
                    trades.append(_trade(sym, f, i, p, entry, sign, eq_cost, slippage_pct, cfg, p["score"], "touch",
                                         same_bar=True))
                    traded.add(sym)
                    day_count += 1
                    pending.pop(sym, None)
                    continue
            if not (start_t <= decided.time() <= end_t):
                continue
            sec = breadth_d.get(sectors[sym])
            sector_up = sec.get(ts) if sec is not None else None
            sector_up = float(sector_up) if finite(sector_up) else None
            res = evaluate(f, i, cfg, None, sector_up, live=True, rows=recs[sym])
            a = float(recs[sym][i]["atr"])
            for side in ("CE", "PE"):
                r = res[side]
                r["close"] = float(recs[sym][i]["close"])
                was_ready = tracker.mem.get(f"{sym}:{side}", {}).get("ready_at")
                tracker.update(sym, r, ts, a, cfg)
                if r["stage"] == "READY" and not was_ready:
                    readies.append(_ready_record(sym, f, i, r))
                    if cfg.entry_mode == "touch" and all(g["ok"] for g in r.get("gates", [])):
                        eq = equity_leg(r["plan"], cfg, eq_cost)
                        if eq["ok"]:
                            pending[sym] = dict(r["plan"], score=r["score"], level=r["level"], zone=r["zone"])
                if cfg.entry_mode != "close" or sym in traded or day_count >= cfg.max_trades:
                    continue
                if r["stage"] != "CONFIRMED":
                    continue
                if not baseline:
                    if cfg.require_prior_ready and not r.get("prior_ready"):
                        continue
                    if not all(g["ok"] for g in r.get("gates", [])):
                        continue
                p = r["plan"]
                if (p["rr1"] or 0) < cfg.min_rr:
                    continue
                if p.get("reach_today") is not None and abs(p["t1"] - p["trigger"]) > p["reach_today"]:
                    continue
                entry = float(f["close"].iloc[i])
                if p["sign"] * (p["t1"] - entry) < p["risk"]:
                    continue
                if not equity_leg(p, cfg, eq_cost)["ok"]:
                    continue
                trades.append(_trade(sym, f, i, dict(p, level=r["level"], zone=r["zone"]), entry, p["sign"], eq_cost,
                                     slippage_pct, cfg, r["score"], "baseline" if baseline else "close"))
                traded.add(sym)
                day_count += 1
                break
    return trades, readies


def _trade(sym, f, i, p, entry, sign, eq_cost, slip, cfg, score, mode, same_bar=False):
    entry_fill = entry * (1 + sign * slip / 100)
    eq = equity_leg(p, cfg, eq_cost)
    qty = eq["qty"]
    if same_bar:      # touch fill inside bar i: the rest of bar i can still hit the stop (conservative)
        adverse = f["low"].iloc[i] if sign > 0 else f["high"].iloc[i]
        if sign * (adverse - p["stop"]) <= 0:
            exit_px, status, held = p["stop"], "STOP_HIT", 0
        else:
            exit_px, status, held = simulate_exit(f, i, entry, sign, p["stop"], p["t1"], p["t2"])
    else:
        exit_px, status, held = simulate_exit(f, i, entry, sign, p["stop"], p["t1"], p["t2"])
    exit_fill = exit_px * (1 - sign * slip / 100)
    side_in, side_out = ("BUY", "SELL") if sign > 0 else ("SELL", "BUY")
    charges = eq_cost(side_in, entry_fill, qty) + eq_cost(side_out, exit_fill, qty)
    gross = sign * (exit_fill - entry_fill) * qty
    risk_rs = abs(entry_fill - p["stop"]) * qty
    return {"date": f.index[i].date().isoformat(), "time": (f.index[i] + BAR).strftime("%H:%M"), "symbol": sym,
            "side": "CE/long" if sign > 0 else "PE/short", "mode": mode, "zone": p.get("zone"),
            "level": p.get("level"), "score": score, "entry": round(entry_fill, 2), "stop": p["stop"], "t1": p["t1"],
            "t2": p["t2"], "exit": round(exit_fill, 2), "status": status, "bars": held, "qty": qty,
            "gross": round(gross, 2), "charges": round(charges, 2), "net": round(gross - charges, 2),
            "r": round((gross - charges) / risk_rs, 3) if risk_rs > 0 else 0.0}


def _ready_record(sym, f, i, r):
    p, sign, a = r["plan"], r["plan"]["sign"], float(f["atr"].iloc[i])
    day = f["day"].iloc[i]
    nxt = f.iloc[i + 1:i + 5]
    nxt = nxt[nxt["day"] == day]
    hit = bool(len(nxt)) and bool(((nxt["high"] if sign > 0 else nxt["low"]) * sign >= p["trigger"] * sign).any())
    move = sign * (float(nxt["close"].iloc[-1]) - float(f["close"].iloc[i])) / a if len(nxt) else float("nan")
    return {"symbol": sym, "time": f.index[i].isoformat(), "side": r["side"], "score": r["score"], "hit": hit,
            "move4_atr": move, "zone": r["zone"]}


# =============================================================================
# Report
# =============================================================================
def stats(trades: List[dict]) -> dict:
    if not trades:
        return {"trades": 0}
    t = pd.DataFrame(trades)
    wins, losses = t[t["net"] > 0]["net"], t[t["net"] <= 0]["net"]
    eq = t["net"].cumsum()
    dd = float((eq - eq.cummax()).min())
    return {"trades": len(t), "win_pct": round(len(wins) / len(t) * 100, 1),
            "avg_r": round(float(t["r"].mean()), 3), "pf": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else math.inf,
            "net": round(float(t["net"].sum())), "gross": round(float(t["gross"].sum())),
            "charges": round(float(t["charges"].sum())), "max_dd": round(dd),
            "days": t["date"].nunique()}


def report(trades, readies, base_trades, cfg, label) -> str:
    lines = [f"NIFTY 50 stock pre-breakout backtest - {label}", "=" * 78,
             f"entry_mode {cfg.entry_mode}, ready_score {cfg.ready_score}, ready_atr {cfg.ready_atr}, "
             f"risk {cfg.risk_pct}% of Rs {cfg.capital:,.0f}, max {cfg.max_trades} trades/day, equity leg only", ""]

    def block(name, tr):
        s = stats(tr)
        if not s["trades"]:
            return [f"{name:28} no trades"]
        return [f"{name:28} trades {s['trades']:4}  win {s['win_pct']:5.1f}%  avg {s['avg_r']:+.2f} R  PF {s['pf']:5.2f}  "
                f"net Rs {s['net']:>10,}  (gross {s['gross']:,}, charges {s['charges']:,})  max DD Rs {s['max_dd']:,}"]
    lines += block("System (READY first)", trades)
    lines += block("Baseline (any breakout)", base_trades)
    if trades:
        days = sorted({t["date"] for t in trades})
        cut = days[int(len(days) * 0.7)] if len(days) > 3 else days[-1]
        lines += block("  first 70% of days", [t for t in trades if t["date"] < cut])
        lines += block("  last 30% of days", [t for t in trades if t["date"] >= cut])
        lines += block("  CE / long", [t for t in trades if t["side"].startswith("CE")])
        lines += block("  PE / short", [t for t in trades if t["side"].startswith("PE")])
        t = pd.DataFrame(trades)
        lines += ["", "Exit reasons: " + ", ".join(f"{k} {v}" for k, v in t["status"].value_counts().items())]
    if readies:
        rd = pd.DataFrame(readies)
        lines += ["", f"READY signals: {len(rd)}; trigger traded within 4 bars: {rd['hit'].mean() * 100:.1f}%; "
                      f"directional move 4 bars later: {rd['move4_atr'].mean():+.2f} ATR "
                      f"(positive {(rd['move4_atr'] > 0).mean() * 100:.0f}%)"]
        for lo, hi in ((70, 80), (80, 90), (90, 101)):
            sub = rd[(rd["score"] >= lo) & (rd["score"] < hi)]
            if len(sub):
                lines.append(f"  score {lo}-{min(hi, 100)}: {len(sub):5} signals, hit {sub['hit'].mean() * 100:5.1f}%, "
                             f"move {sub['move4_atr'].mean():+.2f} ATR")
    lines += ["", "Read this before trusting it: a positive result needs PF >= 1.2 on 100+ trades AND in the last 30%",
              "of days. The option leg is not tested (no option history). Charges and slippage are included."]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.stocks.breakout_backtest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--months", type=float, default=6)
    ap.add_argument("--days", type=int, default=60, help="mock days")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--entry-mode", choices=("close", "touch"))
    ap.add_argument("--slippage-pct", type=float, default=0.02)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    from src.config import settings
    from src.execution.costs import make_equity_cost_fn
    from src.utils.logging_config import setup_logging
    settings.ensure_dirs()
    setup_logging(logging.INFO, log_file=settings.LOG_DIR / "stocks_backtest.log")
    path = args.config or settings.CONFIG_FILE
    cfg = BreakoutConfig.from_ini(path)
    if args.entry_mode:
        cfg = replace(cfg, entry_mode=args.entry_mode)
    parser = settings.read_ini(path)
    eq_cost = make_equity_cost_fn(dict(parser["COSTS_EQUITY"]) if parser.has_section("COSTS_EQUITY") else None)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    if args.mock:
        data, index_bars = load_mock(symbols, args.days)
        label = f"MOCK data, {args.days} days (plumbing check only - not a result)"
    else:
        data, index_bars = load_live(symbols, args.months, settings.DATA_DIR / "cache" / "breakout_bt")
        label = f"Upstox history, last {args.months:g} months, {len(data)} stocks"
    trades, readies = run(data, index_bars, cfg, eq_cost, args.slippage_pct)
    base, _ = run(data, index_bars, replace(cfg, entry_mode="close"), eq_cost, args.slippage_pct, baseline=True)
    text = report(trades, readies, base, cfg, label)
    out = settings.OUTPUT_DIR
    (out / "stock_breakout_backtest.txt").write_text(text, encoding="utf-8")
    if trades:
        pd.DataFrame(trades).to_csv(out / "stock_breakout_trades.csv", index=False)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
