"""
Research backtest of the v3 multi-factor strategy against v1, on the same 1-minute index history.

    python -m src.backtest.v3_report                       NIFTY, data/history (2023-01 onwards)
    python -m src.backtest.v3_report --instrument SENSEX   (needs SENSEX history in data/history)
    python -m src.backtest.v3_report --cache orb_data      read the cached CSVs from another folder

What it does (no look-ahead, no tuning on the test period)
  1. Evaluates every completed 5-minute bar with v1, then re-scores it with v3 (modules, regime, traps).
  2. CALIBRATION on the first 70% of sessions ONLY: v3 scores sit on a different scale from v1, so its watch /
     confirm / entry thresholds are set to the v3 score with the same exceedance rate (near a level) as the v1
     threshold has on those sessions. The last 30% of sessions never influence any setting.
  3. Runs v1 and four v3 variants (calibrated close entries; + target ladder with ATR trailing; + early entries;
     + both) with the same option model, costs and risk rules, and reports in-sample (first 70%) and
     out-of-sample (last 30%) separately.
  4. Out-of-sample robustness: bootstrap intervals, Monte Carlo drawdowns, results by market regime, entry type,
     direction, time of day; a small sensitivity check of the calibrated entry threshold (+/- 5) and the trap
     limit.
Writes output/v3_backtest_report[_SENSEX].html and output/v3_backtest_trades[_SENSEX].csv.

Model limits (the same as the main backtest): options are Black-Scholes on the index with IV = India VIX; there are
no historical option chains (the Derivatives family is "no data") and no futures volume / OI (Participation is "no
data", VWAP is the index TWAP proxy). Treat every number as an estimate under those assumptions.
"""
from __future__ import annotations

import argparse
import html
import os
import sys
import time as _time
from datetime import date, datetime, timedelta
from typing import Dict, List

import numpy as np

from src.backtest import runner as R
from src.backtest import stats as bs
from src.config import settings
from src.config.instruments import SUPPORTED_INDICES
from src.execution.costs import make_cost_fn
from src.scanner import score_v3
from src.scanner import strategies as strat
from src.utils import numbers as num

THRESHOLD_KEYS = ("early_watch", "high_watch", "strong_watch", "confirm_min_score", "entry_min_score", "early_min_score")
DOMINANCE_KEYS = ("watch_min_dominance", "entry_min_dominance")


def _near_scores(evals, days, cfg, which="max"):
    """max(CE, PE) score per bar whose leading side is within watch distance of its level, on `days`."""
    out = []
    for ev in evals:
        if ev.time.date() not in days:
            continue
        for d in (ev.ce, ev.pe):
            if num.finite(d.distance_atr) and 0 <= d.distance_atr <= cfg.watch_max_atr:
                out.append(d.score)
    return np.array(out, dtype=float)


def _gaps(evals, days):
    out = []
    for ev in evals:
        if ev.time.date() in days:
            out.append(abs(ev.ce.score - ev.pe.score))
    return np.array(out, dtype=float)


def calibrate(evals_v1, evals_v3, days, cfg) -> Dict[str, float]:
    """v3 thresholds with the same exceedance rate as v1's, measured on `days` only (in-sample)."""
    s1, s3 = _near_scores(evals_v1, days, cfg), _near_scores(evals_v3, days, cfg)
    g1, g3 = _gaps(evals_v1, days), _gaps(evals_v3, days)
    out = {}
    if len(s1) < 200 or len(s3) < 200:
        return out
    for k in THRESHOLD_KEYS:
        rate = float((s1 >= getattr(cfg, k)).mean())
        out[k] = round(float(np.quantile(s3, 1 - rate)), 1) if 0 < rate < 1 else getattr(cfg, k)
    o = [out[k] for k in ("early_watch", "high_watch", "strong_watch")]
    out["high_watch"], out["strong_watch"] = max(o[1], o[0] + 1), max(o[2], max(o[1], o[0] + 1) + 1)
    for k in DOMINANCE_KEYS:
        rate = float((g1 >= getattr(cfg, k)).mean())
        out[k] = round(float(np.quantile(g3, 1 - rate)), 1) if 0 < rate < 1 else getattr(cfg, k)
    return out


def _metrics(trades, capital, sessions):
    m = R.metrics(trades, capital, max(sessions / 21.0, 1e-9))
    m.update(bs.risk_adjusted(trades, capital, sessions))
    hold = [t.hold_minutes for t in trades if num.finite(getattr(t, "hold_minutes", np.nan))]
    m["avg_hold_min"] = float(np.mean(hold)) if hold else None
    return m


def _split(trades, is_days, oos_days):
    return [t for t in trades if t.day in is_days], [t for t in trades if t.day in oos_days]


ROWS = [("trades", "Trades", "{:,.0f}"), ("net", "Net P&L Rs", "{:,.0f}"), ("gross_profit", "Gross profit Rs", "{:,.0f}"),
        ("gross_loss", "Gross loss Rs", "{:,.0f}"), ("charges", "Charges Rs", "{:,.0f}"), ("win_rate", "Win rate %", "{:.1f}"),
        ("profit_factor", "Profit factor", "{:.2f}"), ("expectancy", "Expectancy Rs / trade", "{:,.0f}"),
        ("expectancy_r", "Expectancy R", "{:+.3f}"), ("avg_win", "Average win Rs", "{:,.0f}"),
        ("avg_loss", "Average loss Rs", "{:,.0f}"), ("max_dd", "Max drawdown Rs", "{:,.0f}"),
        ("max_dd_pct", "Max drawdown %", "{:.1f}"), ("sharpe", "Sharpe (daily, ann.)", "{:.2f}"),
        ("sortino", "Sortino", "{:.2f}"), ("calmar", "Calmar", "{:.2f}"), ("recovery_factor", "Recovery factor", "{:.2f}"),
        ("max_consecutive_losses", "Longest losing streak", "{:.0f}"), ("avg_hold_min", "Avg holding (min)", "{:.0f}"),
        ("per_month", "Trades / month", "{:.1f}")]


def _f(v, p):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    try:
        return p.format(v)
    except (ValueError, TypeError):
        return html.escape(str(v))


def _table(cols: Dict[str, dict]) -> str:
    head = "".join(f"<th>{html.escape(k)}</th>" for k in cols)
    body = "".join("<tr><td>" + label + "</td>" + "".join(f"<td>{_f(c.get(key), p)}</td>" for c in cols.values())
                   + "</tr>" for key, label, p in ROWS)
    return f"<table><tr><th>Metric</th>{head}</tr>{body}</table>"


def _seg(rows: List[dict]) -> str:
    if not rows:
        return "<p>-</p>"
    body = "".join(f"<tr><td>{html.escape(str(r['group']))}</td><td>{r['trades']}</td><td>{_f(r['win_rate'], '{:.1f}')}</td>"
                   f"<td>{_f(r['expectancy'], '{:,.0f}')}</td><td>{_f(r['profit_factor'], '{:.2f}')}</td>"
                   f"<td>{_f(r['net'], '{:,.0f}')}</td></tr>" for r in rows)
    return f"<table><tr><th>Group</th><th>Trades</th><th>Win %</th><th>Expectancy</th><th>PF</th><th>Net</th></tr>{body}</table>"


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.backtest.v3_report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instrument", default="NIFTY", type=str.upper, choices=list(SUPPORTED_INDICES))
    p.add_argument("--start", default="2023-01-11")
    p.add_argument("--end", default=None)
    p.add_argument("--cache", default=str(settings.HISTORY_DIR))
    p.add_argument("--config", default=str(settings.CONFIG_FILE))
    p.add_argument("--out", default=str(settings.OUTPUT_DIR))
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    a = parse_args(argv)
    t0 = _time.time()
    cfg = settings.PBConfig.from_ini(a.config)
    app = settings.load_app_settings(a.config)
    cfg3_base = cfg.with_overrides(app.v3_overrides)
    name = a.instrument
    start = date.fromisoformat(a.start)
    end = date.fromisoformat(a.end) if a.end else date.today() - timedelta(days=1)
    print(f"Loading {name} {start} .. {end} from {a.cache} ...", flush=True)
    minutes, vix = R.load_history(name, start, end, a.cache)
    if minutes.empty:
        print("No data.")
        return 1
    costs_fn = make_cost_fn(app.costs)
    model = R.ChainModel(name, vix, cfg)
    ff = strat.build_features(minutes, cfg)
    print(f"  {len(ff.bars):,} bars; v1 evaluation ...", flush=True)
    evals = R.precompute_evals(ff, cfg, name, vix, progress=False)
    print(f"  v3 re-scoring {len(evals):,} bars ...", flush=True)
    evals3 = []
    for ev in evals:
        e3 = score_v3.rescore(ev, ff, cfg3_base, getattr(ev, "ctx", {}), None)
        e3.ctx = getattr(ev, "ctx", {})
        evals3.append(e3)
    days = sorted({ev.time.date() for ev in evals})
    cut = int(len(days) * 0.7)
    is_days, oos_days = set(days[:cut]), set(days[cut:])
    cal = calibrate(evals, evals3, is_days, cfg)
    print("  calibrated on the first 70% of sessions:", cal, flush=True)
    cfg3 = cfg3_base.replace(**cal)

    def run(evs, c):
        return R.run(evs, ff, minutes, model, c, name, costs_fn)

    variants = {"v1 (current rules)": (evals, cfg),
                "v3 close entry": (evals3, cfg3),
                "v3 + ladder / ATR trail": (evals3, cfg3.replace(exit_profile="ladder", trail_mode="atr")),
                "v3 + early entry": (evals3, cfg3.replace(entry_mode="early")),
                "v3 + early + ladder": (evals3, cfg3.replace(entry_mode="early", exit_profile="ladder", trail_mode="atr"))}
    results = {}
    for label, (evs, c) in variants.items():
        print(f"  running {label} ...", flush=True)
        results[label] = run(evs, c)
    cols_is, cols_oos = {}, {}
    for label, res in results.items():
        tr_is, tr_oos = _split(res.trades, is_days, oos_days)
        cols_is[label] = _metrics(tr_is, cfg.capital, len(is_days))
        cols_oos[label] = _metrics(tr_oos, cfg.capital, len(oos_days))
    # sensitivity of the calibrated entry threshold and the trap limit (out-of-sample only, no selection)
    sens = []
    for de in (-5, 0, 5):
        for tb in (20, 25, 35):
            c = cfg3.replace(entry_min_score=cfg3.entry_min_score + de, trap_block_points=tb)
            evs = evals3 if tb == cfg3.trap_block_points else [score_v3.rescore(ev, ff, c, getattr(ev, "ctx", {}), None)
                                                                  for ev in evals]
            res = run(evs, c)
            _, tr_oos = _split(res.trades, is_days, oos_days)
            sens.append((de, tb, _metrics(tr_oos, cfg.capital, len(oos_days))))
            print(f"    sensitivity entry {de:+d} trap {tb}: OOS trades {sens[-1][2]['trades']}", flush=True)
    main_label = "v3 close entry"
    _, oos_main = _split(results[main_label].trades, is_days, oos_days)
    best_oos = max(cols_oos, key=lambda k: (cols_oos[k]["net"] if cols_oos[k]["trades"] >= 30 else -1e18))
    _, oos_best = _split(results[best_oos].trades, is_days, oos_days)
    ci = bs.bootstrap_ci([t.net for t in oos_best])
    mc = bs.monte_carlo_drawdown([t.net for t in oos_best], cfg.capital)
    segs = {"Market regime (v3 close entry, out-of-sample)": bs.segment_table(oos_main, lambda t: t.regime or "n/a"),
            "Market regime (v3 close entry, in-sample)": bs.segment_table(
                _split(results[main_label].trades, is_days, oos_days)[0], lambda t: t.regime or "n/a"),
            "Entry type": bs.segment_table(oos_best, lambda t: t.entry_type or "n/a"),
            "Direction": bs.segment_table(oos_best, lambda t: t.direction),
            "Entry time": bs.segment_table(oos_best, lambda t: bs.bucket_time(t.entry_time), order=bs.TIME_ORDER)}
    m_best = cols_oos[best_oos]
    edge = bool(m_best["trades"] >= 100 and (m_best["profit_factor"] or 0) >= 1.2 and ci["expectancy"][0] is not None
                and ci["expectancy"][0] > 0)
    verdict = (f"Out-of-sample, the best variant ({best_oos}) made Rs {m_best['net']:,.0f} on {m_best['trades']} trades "
               f"(PF {_f(m_best['profit_factor'], '{:.2f}')}); bootstrap 95% interval of the expectancy "
               f"{_f(ci['expectancy'][0], '{:,.0f}')} .. {_f(ci['expectancy'][1], '{:,.0f}')} Rs per trade. "
               + ("That clears the bar (>= 100 trades, PF >= 1.2, interval above zero)." if edge else
                  "That does NOT show an edge after costs (the bar is >= 100 trades, PF >= 1.2 and an expectancy "
                  "interval above zero). Keep v3 on paper."))
    os.makedirs(a.out, exist_ok=True)
    suffix = "" if name == "NIFTY" else f"_{name}"
    path = os.path.join(a.out, f"v3_backtest_report{suffix}.html")
    csv_path = os.path.join(a.out, f"v3_backtest_trades{suffix}.csv")
    R.write_trades(csv_path, {k: v.trades for k, v in results.items()})
    cal_rows = "".join(f"<tr><td>{k}</td><td>{getattr(cfg, k)}</td><td>{v}</td></tr>" for k, v in cal.items())
    sens_rows = "".join(f"<tr><td>{de:+d}</td><td>{tb}</td><td>{m['trades']}</td><td>{_f(m['net'], '{:,.0f}')}</td>"
                        f"<td>{_f(m['profit_factor'], '{:.2f}')}</td><td>{_f(m['sharpe'], '{:.2f}')}</td></tr>"
                        for de, tb, m in sens)
    mc_txt = (f"median max drawdown Rs {_f(mc.get('dd_median'), '{:,.0f}')}, 95th percentile Rs "
              f"{_f(mc.get('dd_p95'), '{:,.0f}')}" if isinstance(mc, dict) and mc.get("dd_median") is not None else
              html.escape(str(mc.get("note", "-")) if isinstance(mc, dict) else "-"))
    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111;background:#fafafa;max-width:1200px}"
           "table{border-collapse:collapse;margin:8px 0 18px;background:#fff}td,th{border:1px solid #ddd;padding:5px 9px;"
           "font-size:13px;text-align:right}th{background:#f1f5f9}td:first-child{text-align:left}"
           ".warn{background:#fff7ed;border-left:4px solid #f97316;padding:10px 14px}"
           ".ok{background:#f0fdf4;border-left:4px solid #16a34a;padding:10px 14px}")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>v3 backtest {name}</title><style>{css}</style></head><body>
<h1>v3 multi-factor strategy vs v1 - {name}</h1>
<p>{days[0]} to {days[-1]} &middot; {len(days)} sessions (in-sample {len(is_days)} up to {days[cut - 1]}, out-of-sample
{len(oos_days)} from {days[cut]}) &middot; capital Rs {cfg.capital:,.0f}, risk {cfg.risk_per_trade_pct}% per trade
&middot; generated {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="{'ok' if edge else 'warn'}"><b>Verdict:</b> {html.escape(verdict)}</div>
<div class="warn"><b>Model limits.</b> Options are Black-Scholes on the index with IV = India VIX (no historical option
prices, bid/ask or chains), so the Derivatives family is "no data". There is no futures volume / OI in this history,
so Participation is "no data" and VWAP is the index TWAP proxy. Spread {cfg.assumed_spread_pct}% and slippage
{cfg.slippage_pct}% per leg plus all statutory charges are included. Scores are setup-strength scores, not
probabilities.</div>
<h2>Calibration (first 70% of sessions only)</h2>
<p>v3 thresholds set to the same exceedance rate the v1 thresholds have near a level. Put these into [SCORE_V3]
to use them live.</p><table><tr><th>Setting</th><th>v1</th><th>v3 calibrated</th></tr>{cal_rows}</table>
<h2>Out-of-sample (last 30% - never used for any setting)</h2>{_table(cols_oos)}
<h2>In-sample (first 70%)</h2>{_table(cols_is)}
<h2>Robustness of the best out-of-sample variant ({html.escape(best_oos)}); regime tables use v3 close entry</h2>
<p>Bootstrap 95% intervals: expectancy {_f(ci['expectancy'][0], '{:,.0f}')} .. {_f(ci['expectancy'][1], '{:,.0f}')} Rs,
profit factor {_f(ci['profit_factor'][0], '{:.2f}')} .. {_f(ci['profit_factor'][1], '{:.2f}')}, win rate
{_f(ci['win_rate'][0], '{:.1f}')} .. {_f(ci['win_rate'][1], '{:.1f}')}%. Monte Carlo (same trades, shuffled order):
{mc_txt}.</p>
{''.join(f'<h3>{html.escape(k)}</h3>' + _seg(v) for k, v in segs.items())}
<h2>Sensitivity (out-of-sample, for stability only - not used to choose anything)</h2>
<table><tr><th>Entry threshold change</th><th>Trap limit</th><th>Trades</th><th>Net</th><th>PF</th><th>Sharpe</th></tr>{sens_rows}</table>
<p>Trades: <code>{html.escape(os.path.basename(csv_path))}</code> &middot; run time {(_time.time() - t0) / 60:.1f} min</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(doc)
    print(f"Report: {path}\n{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
