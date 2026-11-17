"""
ML trade filter for the pre-breakout system.

What it is
  A logistic-regression model (plain numpy: no extra packages, every weight readable) that estimates the
  chance that a CONFIRMED breakout reaches its first target (1.5 R on the index) before its stop, within
  the holding time. The features are the rule engine's own measurements: score families (v1 and v2),
  RSI, ADX, compression, distance beyond the level, time of day, VIX, level strength, ...

What it may do
  * It ALWAYS shows its probability on entry setups (Telegram, dashboard).
  * It may BLOCK an entry only when the saved model is marked "approved": in walk-forward testing on data
    it was not trained on, it had to (a) rank setups better than chance (AUC >= 0.55), (b) beat the base
    rate (positive Brier skill) and (c) raise the average R of the setups it keeps by >= 0.05 R, on at least
    150 out-of-sample setups. Otherwise it is advisory only. The rules always decide first.

Training data
  * History: every CONFIRMED setup the rule engine produces on the cached 1-minute index history
    (data/history/), labelled from the actual 1-minute path. History has no futures / breadth data, so those
    features are missing there.
  * Live: every CONFIRMED setup the service sees is logged to data/prebreakout.db (ml_samples) with ALL
    features, and labelled automatically once its outcome is known. Retraining picks these up, so the
    model improves as real futures / breadth data accumulates.

    python -m src.scanner.ml_filter --train     train + walk-forward test, write data/prebreakout_ml_model.json
    python -m src.scanner.ml_filter --report    print the saved model's test results
    (VS Code: Run and Debug -> "Train ML filter")
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import instruments as inst
from src.config import settings
from src.config.instruments import SUPPORTED_INDICES, validate_indices
from src.risk import manager as riskm
from src.scanner import signals as sig
from src.scanner import strategies as strat
from src.utils import numbers as num

MODEL_FILE = str(settings.ML_MODEL_FILE)
DB_FILE = str(settings.DB_FILE)
LABEL_R_TARGET = 1.5
KEEP_QUANTILE = 0.40   # the filter drops the 40% of setups with the lowest probability (fixed, not tuned)


# =============================================================================
# Features
# =============================================================================
def features(ev: strat.BarEval, d: strat.DirectionEval, ctx: Optional[dict] = None,
             v2d: Optional[strat.DirectionEval] = None) -> Dict[str, float]:
    """Direction-neutral features (PE values are mirrored so 'good for the trade' has one sign)."""
    up = d.direction == inst.CE
    s = 1.0 if up else -1.0
    other = ev.pe if up else ev.ce
    ind = ev.indicators or {}
    f: Dict[str, float] = {}
    for fam in d.families:
        key = "v1_" + fam.name.lower().replace(" ", "_")
        f[key] = 100.0 * fam.points / fam.max_points if fam.available and fam.max_points else np.nan
    f["v1_score"] = d.score
    f["v1_dominance"] = d.score - other.score
    f["extension_atr"] = num.fnum(d.extension_atr)
    rsi = num.fnum(ind.get("rsi"))
    f["rsi_dir"] = rsi if up else (100 - rsi if num.finite(rsi) else np.nan)
    f["adx"] = num.fnum(ind.get("adx"))
    pdi, mdi = num.fnum(ind.get("plus_di")), num.fnum(ind.get("minus_di"))
    f["di_spread_dir"] = s * (pdi - mdi) if num.finite(pdi) and num.finite(mdi) else np.nan
    f["atr_ratio"] = num.fnum(ind.get("atr_ratio"))
    f["bb_width_pct"] = num.fnum(ind.get("bb_width_pct"))
    atr = ev.atr if num.finite(ev.atr) and ev.atr > 0 else np.nan
    e20 = num.fnum(ind.get("ema20"))
    f["ema20_dist_dir"] = s * (ev.close - e20) / atr if num.finite(e20) and num.finite(atr) else np.nan
    hist = num.fnum(ind.get("macd_hist"))
    f["macd_hist_dir"] = s * hist / atr if num.finite(hist) and num.finite(atr) else np.nan
    f["fut_rvol"] = num.fnum(ind.get("f_rvol"))
    t = ev.close_time
    f["minutes_from_open"] = (t.hour * 60 + t.minute) - (9 * 60 + 15)
    lvl = d.level or {}
    f["level_strength"] = num.fnum(lvl.get("strength"))
    f["level_touches"] = num.fnum(lvl.get("touches"))
    ctx = ctx or {}
    f["vix"] = num.fnum(ctx.get("vix"))
    f["vix_change_pct"] = num.fnum(ctx.get("vix_change_pct"))
    if v2d is not None:
        for fam in v2d.families:
            key = "v2_" + fam.name.lower().replace(" ", "_").replace("/", "").replace("-", "")
            f[key] = 100.0 * fam.points / fam.max_points if fam.available and fam.max_points else np.nan
        f["v2_score"] = v2d.score
    return {k: (float(v) if num.finite(v) else np.nan) for k, v in f.items()}


# =============================================================================
# Labels (model-free: from the index's own 1-minute path)
# =============================================================================
def label_outcome(minutes: pd.DataFrame, start: pd.Timestamp, direction: str, entry: float, stop: float,
                  max_minutes: float, square_off: time) -> Optional[Tuple[int, float]]:
    """(hit_target 0/1, R result) for a setup confirmed at `start`. None while the outcome is still open.
    Stop is checked before target inside a minute (pessimistic)."""
    risk = abs(entry - stop)
    if not (num.finite(risk) and risk > 0):
        return None
    sign = 1.0 if direction == inst.CE else -1.0
    target = entry + sign * LABEL_R_TARGET * risk
    end = min(start + pd.Timedelta(minutes=max_minutes),
              pd.Timestamp(datetime.combine(start.date(), square_off)).tz_localize(start.tz))
    window = minutes[(minutes.index >= start) & (minutes.index < end)]
    for _, row in window.iterrows():
        adverse = row["low"] if sign > 0 else row["high"]
        favour = row["high"] if sign > 0 else row["low"]
        if sign * (adverse - stop) <= 0:
            return 0, -1.0
        if sign * (favour - target) >= 0:
            return 1, LABEL_R_TARGET
    last_needed = end - pd.Timedelta(minutes=1)
    if len(window) == 0 or window.index[-1] < last_needed:
        if len(minutes) and minutes.index[-1] < last_needed:
            return None  # not finished yet
    if len(window) == 0:
        return None
    r = sign * (window["close"].iloc[-1] - entry) / risk
    return 0, float(max(-1.0, min(LABEL_R_TARGET, r)))


# =============================================================================
# Model: standardised L2 logistic regression (Newton / IRLS)
# =============================================================================
class Logistic:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.mean = self.std = self.w = None
        self.b = 0.0
        self.names: List[str] = []

    def _prep(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return np.where(np.isfinite(Z), Z, 0.0)  # missing -> the training mean

    def fit(self, X: np.ndarray, y: np.ndarray, names: List[str]):
        self.names = list(names)
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-9), self.std, 1.0)
        Z = self._prep(X)
        n, k = Z.shape
        A = np.hstack([np.ones((n, 1)), Z])
        beta = np.zeros(k + 1)
        p0 = min(max(y.mean(), 1e-3), 1 - 1e-3)
        beta[0] = math.log(p0 / (1 - p0))
        reg = np.eye(k + 1) * self.l2
        reg[0, 0] = 0.0
        for _ in range(50):
            p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
            W = p * (1 - p)
            g = A.T @ (y - p) - reg @ beta
            H = (A * W[:, None]).T @ A + reg
            step = np.linalg.solve(H, g)
            beta += step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.b, self.w = float(beta[0]), beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = self._prep(X) @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def to_dict(self) -> dict:
        return {"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist(), "w": self.w.tolist(),
                "b": self.b, "l2": self.l2}

    @classmethod
    def from_dict(cls, d: dict) -> "Logistic":
        m = cls(d.get("l2", 1.0))
        m.names, m.mean, m.std = d["names"], np.array(d["mean"]), np.array(d["std"])
        m.w, m.b = np.array(d["w"]), float(d["b"])
        return m


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def select_features(rows: List[dict], min_coverage: float = 0.8) -> List[str]:
    names = sorted({k for r in rows for k in r})
    keep = []
    for n in names:
        vals = np.array([r.get(n, np.nan) for r in rows], dtype=float)
        if np.isfinite(vals).mean() >= min_coverage and np.nanstd(vals) > 1e-9:
            keep.append(n)
    return keep


def matrix(rows: List[dict], names: List[str]) -> np.ndarray:
    return np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=float)


def walk_forward(samples: List[dict], min_train: int = 300, step_days: int = 30) -> dict:
    """samples: {"time", "features", "y", "r"} sorted by time. Expanding window, retrain every step_days."""
    samples = sorted(samples, key=lambda s: s["time"])
    if len(samples) < min_train + 50:
        return {"ok": False, "reason": f"only {len(samples)} labelled setups (need {min_train + 50}+)"}
    times = [pd.Timestamp(s["time"]) for s in samples]
    i = min_train
    oos_p, oos_y, oos_r, oos_keep = [], [], [], []
    while i < len(samples):
        train = samples[:i]
        names = select_features([s["features"] for s in train])
        model = Logistic(1.0).fit(matrix([s["features"] for s in train], names), np.array([s["y"] for s in train]), names)
        thr = float(np.quantile(model.predict(matrix([s["features"] for s in train], names)), KEEP_QUANTILE))
        end_time = times[i - 1] + pd.Timedelta(days=step_days)
        j = i
        while j < len(samples) and times[j] <= end_time:
            j += 1
        test = samples[i:j]
        p = model.predict(matrix([s["features"] for s in test], names))
        oos_p += p.tolist()
        oos_y += [s["y"] for s in test]
        oos_r += [s["r"] for s in test]
        oos_keep += (p >= thr).tolist()
        i = j
    p, y, r, keep = np.array(oos_p), np.array(oos_y), np.array(oos_r), np.array(oos_keep, dtype=bool)
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    res = {
        "ok": True, "oos_samples": int(len(y)), "base_rate": float(base), "auc": auc(y, p), "brier": brier,
        "brier_skill": float(1 - brier / brier_base) if brier_base > 0 else np.nan,
        "mean_r_all": float(r.mean()), "mean_r_kept": float(r[keep].mean()) if keep.any() else np.nan,
        "mean_r_dropped": float(r[~keep].mean()) if (~keep).any() else np.nan, "kept_pct": float(keep.mean() * 100),
        "hit_rate_kept": float(y[keep].mean()) if keep.any() else np.nan,
    }
    # calibration table (5 bins)
    bins = np.quantile(p, [0, .2, .4, .6, .8, 1.0])
    table = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p <= b)
        if m.any():
            table.append({"p_from": float(a), "p_to": float(b), "n": int(m.sum()), "predicted": float(p[m].mean()),
                          "actual": float(y[m].mean()), "mean_r": float(r[m].mean())})
    res["calibration"] = table
    res["approved"] = bool(res["oos_samples"] >= 150 and res["auc"] >= 0.55 and res["brier_skill"] > 0
                           and np.isfinite(res["mean_r_kept"]) and res["mean_r_kept"] - res["mean_r_all"] >= 0.05)
    return res


# =============================================================================
# Live gate
# =============================================================================
class MLGate:
    """Loads prebreakout_ml_model.json (reloaded when it changes). gate(direction, ev, d, plan) -> (p, block)."""

    def __init__(self, path: str = MODEL_FILE, allow_block: bool = True):
        self.path = path
        self.allow_block = allow_block
        self.mtime = None
        self.model: Optional[Logistic] = None
        self.meta: dict = {}
        self.ctx: dict = {}
        self.v2: Dict[Tuple[str, str], strat.DirectionEval] = {}

    def _load(self):
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self.model, self.meta, self.mtime = None, {}, None
            return
        if mtime == self.mtime:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as h:
                data = json.load(h)
            self.model = Logistic.from_dict(data["model"])
            self.meta = {k: v for k, v in data.items() if k != "model"}
            self.mtime = mtime
        except Exception:
            self.model, self.meta = None, {}

    def status(self) -> dict:
        self._load()
        if self.model is None:
            return {"loaded": False, "approved": False, "note": "no model yet - run python -m src.scanner.ml_filter --train"}
        wf = self.meta.get("walk_forward", {})
        return {"loaded": True, "approved": bool(self.meta.get("approved")), "threshold": self.meta.get("threshold"),
                "trained_at": self.meta.get("trained_at"), "samples": self.meta.get("samples"),
                "auc": wf.get("auc"), "brier_skill": wf.get("brier_skill"), "mean_r_all": wf.get("mean_r_all"),
                "mean_r_kept": wf.get("mean_r_kept"), "blocking": bool(self.meta.get("approved")) and self.allow_block}

    def probability(self, ev, d, v2d=None) -> Optional[float]:
        self._load()
        if self.model is None:
            return None
        row = features(ev, d, self.ctx, v2d)
        return float(self.model.predict(matrix([row], self.model.names))[0])

    def gate(self, direction, ev, d, plan):
        p = self.probability(ev, d, self.v2.get((ev.index, direction)))
        if p is None:
            return None, None
        if not self.meta.get("approved"):
            return None, None  # an unproven model's number is not shown (it is still logged for training)
        thr = self.meta.get("threshold")
        if self.allow_block and thr is not None and p < thr:
            return p, f"ML filter: win chance {p:.0%} < {thr:.0%} (approved model)"
        return p, None


# =============================================================================
# Training
# =============================================================================
def history_samples(instrument: str, start: date, end: date, cfg: settings.PBConfig) -> List[dict]:
    from src.backtest import runner as bt
    from src.scanner import score_v2 as v2
    minutes, vix = bt.load_history(instrument, start, end, str(settings.HISTORY_DIR))
    if minutes.empty:
        return []
    ff = strat.build_features(minutes, cfg)
    evals = bt.precompute_evals(ff, cfg, instrument, vix, progress=False)
    mf = v2.build_minutes(minutes, None)
    tracker = sig.StageTracker(instrument, cfg)
    out = []
    for ev in evals:
        wb, eb = riskm.session_block(ev.close_time, cfg)
        alerts = tracker.step(ev, watch_block=wb, entry_block=eb, data_ok=True, data_reason="", risk=None, resolver=None)
        for a in alerts:
            if a.kind != "confirmed":
                continue
            d = ev.side(a.direction)
            v2d = v2.score_direction(a.direction, ev, ff, mf, getattr(ev, "ctx", {}))
            plan = strat.underlying_plan(a.direction, d, ev.close, ev.atr, cfg)
            lab = label_outcome(minutes, ev.close_time, a.direction, ev.close, plan.stop, cfg.max_hold_minutes, cfg.square_off)
            if lab is None:
                continue
            out.append({"time": ev.close_time.isoformat(), "features": features(ev, d, getattr(ev, "ctx", {}), v2d),
                        "y": lab[0], "r": lab[1], "source": f"history {instrument}"})
    return out


def live_samples(db: str = DB_FILE) -> List[dict]:
    if not os.path.exists(db):
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT time, features, y, r FROM ml_samples WHERE y IS NOT NULL AND mock = 0").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        conn.close()
    return [{"time": t, "features": {k: (np.nan if v is None else v) for k, v in json.loads(f).items()}, "y": int(y),
             "r": float(r), "source": "live"} for t, f, y, r in rows]


def train(instruments=("NIFTY",), start=date(2023, 1, 1), end=None, out_path=MODEL_FILE, cfg=None) -> dict:
    cfg = cfg or settings.PBConfig.from_ini(settings.CONFIG_FILE)
    instruments = validate_indices(instruments)
    end = end or (date.today() - timedelta(days=1))
    samples = []
    for name in instruments:
        print(f"  building labelled setups from {name} history ...", flush=True)
        samples += history_samples(name, start, end, cfg)
    live = live_samples()
    samples += live
    print(f"  {len(samples)} labelled setups ({len(live)} live)", flush=True)
    wf = walk_forward(samples)
    if not wf.get("ok"):
        result = {"trained_at": datetime.now().isoformat(timespec="seconds"), "approved": False, "walk_forward": wf,
                  "samples": len(samples)}
        print("  not enough data:", wf.get("reason"))
        return result
    names = select_features([s["features"] for s in samples])
    X, y = matrix([s["features"] for s in samples], names), np.array([s["y"] for s in samples])
    model = Logistic(1.0).fit(X, y, names)
    threshold = float(np.quantile(model.predict(X), KEEP_QUANTILE))
    coef = sorted(zip(names, model.w.tolist()), key=lambda x: -abs(x[1]))
    result = {"trained_at": datetime.now().isoformat(timespec="seconds"), "samples": len(samples),
              "live_samples": len(live), "instruments": list(instruments), "approved": wf["approved"],
              "threshold": threshold, "label": f"index reaches {LABEL_R_TARGET} R before the stop within "
                                               f"{cfg.max_hold_minutes:.0f} min", "walk_forward": wf,
              "top_weights": [{"feature": n, "weight": round(w, 3)} for n, w in coef[:12]], "model": model.to_dict()}
    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as h:
        json.dump(result, h, indent=1, default=float)
    os.replace(tmp, out_path)
    return result


def print_report(result: dict):
    wf = result.get("walk_forward", {})
    print(f"\nML filter - trained {result.get('trained_at')} on {result.get('samples')} setups "
          f"({result.get('live_samples', 0)} live)")
    if not wf.get("ok"):
        print("  ", wf.get("reason"))
        return
    print(f"  walk-forward out-of-sample: {wf['oos_samples']} setups, base hit rate {wf['base_rate']:.1%}")
    print(f"  AUC {wf['auc']:.3f} (0.5 = chance) | Brier skill {wf['brier_skill']:+.3f} (>0 beats the base rate)")
    print(f"  mean R: all {wf['mean_r_all']:+.3f} | kept by the filter {wf['mean_r_kept']:+.3f} "
          f"({wf['kept_pct']:.0f}% kept) | dropped {wf['mean_r_dropped']:+.3f}")
    print("  calibration (predicted -> actual):")
    for row in wf.get("calibration", []):
        print(f"    {row['predicted']:.2f} -> {row['actual']:.2f}  (n={row['n']}, mean R {row['mean_r']:+.2f})")
    verdict = "APPROVED: it may block low-probability entries" if result.get("approved") else \
        "NOT approved: advisory only (shows its probability, never blocks)"
    print("  " + verdict)
    if result.get("top_weights"):
        print("  strongest features:", ", ".join(f"{w['feature']} {w['weight']:+.2f}" for w in result["top_weights"][:6]))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="python -m src.scanner.ml_filter", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--instruments", default="NIFTY", help=f"comma list of {'/'.join(SUPPORTED_INDICES)}; history is downloaded to data/history/ when missing (needs an Upstox token)")
    p.add_argument("--start", default="2023-01-01")
    a = p.parse_args(argv)
    if a.train:
        r = train(tuple(x.strip().upper() for x in a.instruments.split(",") if x.strip()), date.fromisoformat(a.start))
        print_report(r)
        return 0
    try:
        with open(MODEL_FILE, "r", encoding="utf-8") as h:
            print_report(json.load(h))
    except OSError:
        print("No model yet: python -m src.scanner.ml_filter --train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
