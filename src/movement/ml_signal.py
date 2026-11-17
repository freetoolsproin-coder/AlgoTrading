"""
Machine-learning signal classifier for NIFTY / SENSEX (strategy key `ml_signal`).

    python -m src.movement.ml_signal --train --instrument NIFTY
    python -m src.movement.ml_signal --train --instrument SENSEX --timeframe 5 --horizon 6 --holdout 0.3
    python -m src.movement.ml_signal --status

What it learns
  Two L2-regularised logistic regressions (pure numpy, no extra package) on the signal-timeframe bars:
    P(up)   = the close `horizon` bars later is >= min_move_atr x ATR higher, same session
    P(down) = ... lower
  Features (all from COMPLETED bars, nothing from the future): returns over 1 / 3 / 6 / 12 bars, RSI, MACD
  histogram / ATR, ADX and DI spread, distance to EMA9 / 20 / 50 in ATR, Bollinger %b, ATR %, bar range and
  body in ATR, Supertrend side, time of day.

How it is validated (read the report before trusting it)
  * Chronological split by trading day: the first (1 - holdout) of the days train the model, the last
    `holdout` share is never seen in training and gives the reported numbers (AUC, precision, signal count,
    average forward move of the signals vs all bars).
  * The firing threshold is chosen on the LAST 20% OF THE TRAINING PERIOD (top `signal_rate` of bars), not on
    the test period.
  * The saved model remembers the last training day. The strategy gives no signal on or before that day, so a
    backtest of `ml_signal` only ever trades days the model did not learn from.
  * Labels ignore option premiums, theta and charges. A good AUC is not a profitable options strategy: run
    python -m src.movement.backtest --strategies ml_signal for that.

Model files: data/ml_signal_<INDEX>_<tf>m.json (also trainable from the movement web page, /ml).
"""
from __future__ import annotations

import argparse
import json
import math
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import settings
from src.scanner import indicators as ind
from src.utils.numbers import finite

FEATURES = ["ret1", "ret3", "ret6", "ret12", "rsi", "macd_atr", "adx", "di", "d_ema9", "d_ema20", "d_ema50",
            "bb_pct", "atr_pct", "range_atr", "body_atr", "st", "tod_sin", "tod_cos"]
MODEL_VERSION = 1
_cache: Dict[str, tuple] = {}
_lock = threading.Lock()


def model_path(index: str, tf: int) -> Path:
    return settings.DATA_DIR / f"ml_signal_{index.upper()}_{int(tf)}m.json"


# ============================================================================================ features / labels
def features(b: pd.DataFrame) -> pd.DataFrame:
    """b: signal bars WITH src.movement.strategies.add_indicators columns."""
    c, a = b["close"], b["atr"].replace(0, np.nan)
    out = pd.DataFrame(index=b.index)
    for n in (1, 3, 6, 12):
        out[f"ret{n}"] = (c / c.shift(n) - 1) * 100
    out["rsi"] = b["rsi"] / 100 - 0.5
    out["macd_atr"] = b["macd_hist"] / a
    out["adx"] = b["adx"] / 100
    out["di"] = (b["pdi"] - b["mdi"]) / 100
    out["d_ema9"] = (c - b["ema9"]) / a
    out["d_ema20"] = (c - b["ema20"]) / a
    out["d_ema50"] = (c - b["ema50"]) / a
    width = (b["bb_up"] - b["bb_lo"]).replace(0, np.nan)
    out["bb_pct"] = (c - b["bb_lo"]) / width - 0.5
    out["atr_pct"] = a / c * 100
    out["range_atr"] = (b["high"] - b["low"]) / a
    out["body_atr"] = (c - b["open"]) / a
    out["st"] = b["st"]
    minutes = (b.index.hour * 60 + b.index.minute - (9 * 60 + 15)) / 375.0
    out["tod_sin"], out["tod_cos"] = np.sin(2 * np.pi * minutes), np.cos(2 * np.pi * minutes)
    return out[FEATURES]


def labels(b: pd.DataFrame, horizon: int, min_move_atr: float) -> pd.DataFrame:
    """Forward move `horizon` bars ahead in ATR units, only within the same session (else NaN)."""
    c = b["close"].to_numpy(float)
    a = b["atr"].to_numpy(float)
    days = np.asarray(b.index.date)
    fwd = np.full(len(c), np.nan)
    if len(c) > horizon:
        same = days[horizon:] == days[:-horizon]
        move = (c[horizon:] - c[:-horizon]) / a[:-horizon]
        fwd[:-horizon] = np.where(same, move, np.nan)
    out = pd.DataFrame({"fwd_atr": fwd}, index=b.index)
    out["up"] = np.where(np.isfinite(fwd), (fwd >= min_move_atr).astype(float), np.nan)
    out["down"] = np.where(np.isfinite(fwd), (fwd <= -min_move_atr).astype(float), np.nan)
    return out


# ============================================================================================ model
@dataclass
class Logistic:
    w: np.ndarray
    b: float

    def proba(self, x: np.ndarray) -> np.ndarray:
        z = np.clip(x @ self.w + self.b, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 30) -> Logistic:
    """Newton / IRLS with an L2 penalty on the weights (not the intercept). x must be standardised."""
    n, k = x.shape
    xa = np.c_[x, np.ones(n)]
    w = np.zeros(k + 1)
    pen = np.full(k + 1, l2)
    pen[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(xa @ w, -30, 30)))
        g = xa.T @ (p - y) + pen * w
        h = (xa * (p * (1 - p))[:, None]).T @ xa + np.diag(pen) + 1e-9 * np.eye(k + 1)
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return Logistic(w[:-1], float(w[-1]))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve (rank formula); NaN if one class is missing."""
    y = np.asarray(y, bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _evaluate_side(y: np.ndarray, prob: np.ndarray, fwd: np.ndarray, thr: float, sign: int) -> dict:
    hit = prob >= thr
    out = {"bars": int(len(y)), "base_rate": float(y.mean()) if len(y) else float("nan"), "auc": auc(y, prob),
           "signals": int(hit.sum()), "threshold": float(thr)}
    if hit.any():
        out["precision"] = float(y[hit].mean())
        out["avg_fwd_atr_signals"] = float(sign * fwd[hit].mean())
    out["avg_fwd_atr_all"] = float(sign * np.nanmean(fwd)) if len(fwd) else float("nan")
    return out


# ============================================================================================ training
def build_dataset(minutes: pd.DataFrame, tf: int, horizon: int, min_move_atr: float) -> pd.DataFrame:
    from src.movement.strategies import add_indicators
    bars = add_indicators(ind.resample(minutes, tf))
    frame = features(bars).join(labels(bars, horizon, min_move_atr))
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def train(minutes: pd.DataFrame, index: str, tf: int = 5, horizon: int = 6, min_move_atr: float = 1.0,
          holdout: float = 0.3, l2: float = 1.0, signal_rate: float = 0.05, save: bool = True,
          path: Optional[Path] = None) -> dict:
    """Fit on the first (1 - holdout) of the trading days, report on the rest, save the model JSON."""
    if not 0 <= holdout < 0.9:
        raise ValueError("holdout must be between 0 and 0.9")
    data = build_dataset(minutes, tf, horizon, min_move_atr)
    days = sorted(set(data.index.date))
    if len(days) < 20:
        raise ValueError(f"only {len(days)} trading days of data: need at least 20")
    cut = len(days) - int(round(len(days) * holdout)) if holdout > 0 else len(days)
    train_days = set(days[:cut])
    test_days = set(days[cut:])
    is_train = np.array([d in train_days for d in data.index.date])
    tr, te = data[is_train], data[~is_train]
    x_tr = tr[FEATURES].to_numpy(float)
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    std[std == 0] = 1.0
    z = lambda frame: (frame[FEATURES].to_numpy(float) - mean) / std

    # threshold from the last 20% of the TRAINING days (fit on the first 80% of those first)
    tr_days = sorted(train_days)
    vcut = tr_days[int(len(tr_days) * 0.8)]
    fit_part = np.array([d < vcut for d in tr.index.date])
    models, thresholds, report = {}, {}, {"train": {}, "test": {}}
    for side, sign in (("up", 1), ("down", -1)):
        y_tr = tr[side].to_numpy(float)
        pre = fit_logistic(z(tr[fit_part]), y_tr[fit_part], l2)
        val_p = pre.proba(z(tr[~fit_part])) if (~fit_part).any() else pre.proba(z(tr))
        thresholds[side] = float(np.quantile(val_p, 1 - signal_rate))
        models[side] = fit_logistic(z(tr), y_tr, l2)                    # final fit on all training days
        report["train"][side] = _evaluate_side(y_tr.astype(bool), models[side].proba(z(tr)),
                                               tr["fwd_atr"].to_numpy(float), thresholds[side], sign)
        if len(te):
            report["test"][side] = _evaluate_side(te[side].to_numpy(bool), models[side].proba(z(te)),
                                                  te["fwd_atr"].to_numpy(float), thresholds[side], sign)
    model = {
        "version": MODEL_VERSION, "index": index.upper(), "timeframe": int(tf), "horizon": int(horizon),
        "min_move_atr": float(min_move_atr), "l2": float(l2), "signal_rate": float(signal_rate),
        "features": FEATURES, "mean": mean.tolist(), "std": std.tolist(),
        "weights": {s: {"w": m.w.tolist(), "b": m.b} for s, m in models.items()}, "thresholds": thresholds,
        "train_start": str(min(train_days)), "train_end": str(max(train_days)),
        "test_start": str(min(test_days)) if test_days else None,
        "test_end": str(max(test_days)) if test_days else None,
        "train_days": len(train_days), "test_days": len(test_days), "report": report,
        "trained_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(timespec="seconds"),
        "top_features": {s: [FEATURES[i] for i in np.argsort(-np.abs(m.w))[:6]] for s, m in models.items()},
    }
    if save:
        target = path or model_path(index, tf)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(model, indent=1, default=float), encoding="utf-8")
        tmp.replace(target)
        model["path"] = str(target)
    return model


def load_model(path: Path) -> Optional[dict]:
    """Cached by file modification time; None if the file is missing or unreadable."""
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    key = str(path)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
        if model.get("version") != MODEL_VERSION or model.get("features") != FEATURES:
            return None
    except (OSError, ValueError):
        return None
    with _lock:
        _cache[key] = (stamp, model)
    return model


def predict(model: dict, feats: pd.DataFrame) -> pd.DataFrame:
    x = (feats[FEATURES].to_numpy(float) - np.asarray(model["mean"])) / np.asarray(model["std"])
    out = pd.DataFrame(index=feats.index)
    for side in ("up", "down"):
        w = model["weights"][side]
        out[side] = Logistic(np.asarray(w["w"]), float(w["b"])).proba(x)
    return out


def list_models() -> List[dict]:
    out = []
    for path in sorted(settings.DATA_DIR.glob("ml_signal_*m.json")):
        m = load_model(path)
        if m:
            out.append({k: m.get(k) for k in ("index", "timeframe", "horizon", "min_move_atr", "train_start",
                                              "train_end", "test_start", "test_end", "train_days", "test_days",
                                              "trained_at", "thresholds", "report", "top_features")}
                       | {"file": path.name})
    return out


# ============================================================================================ the strategy
def evaluate(ctx, p: dict):
    from src.movement.strategies import Result, _need, _sig, UP, DOWN
    path = model_path(ctx.index, ctx.tf) if str(p.get("model", "auto")).lower() == "auto" else Path(p["model"])
    model = load_model(path)
    if model is None:
        return Result(reason=f"no data: no trained model ({path.name}); train it with "
                             f"python -m src.movement.ml_signal --train --instrument {ctx.index} --timeframe {ctx.tf}")
    if int(model["timeframe"]) != ctx.tf:
        return Result(reason=f"no data: model is for {model['timeframe']}-min bars, strategy runs on {ctx.tf}-min")
    if ctx.today <= date.fromisoformat(model["train_end"]):
        return Result(reason=f"inside the model's training period (until {model['train_end']}): no signal")
    b = ctx.b5
    miss = _need(b, 60)
    if miss:
        return miss
    feats = features(b.iloc[-80:]).iloc[-2:]
    if feats.isna().any().any():
        return Result(reason="warming up: features")
    pr = predict(model, feats)
    thr = float(p.get("threshold", 0) or 0)
    t_up = thr if thr > 0 else float(model["thresholds"]["up"])
    t_dn = thr if thr > 0 else float(model["thresholds"]["down"])
    up_now, up_prev = pr["up"].iloc[-1], pr["up"].iloc[-2]
    dn_now, dn_prev = pr["down"].iloc[-1], pr["down"].iloc[-2]
    vals = dict(p_up=up_now, p_down=dn_now, threshold_up=t_up, threshold_down=t_dn)
    if up_now >= t_up > up_prev and up_now > dn_now:
        return _sig(UP, f"model P(up {model['min_move_atr']:g} ATR in {model['horizon']} bars) {up_now:.2f} "
                        f">= {t_up:.2f}", b.index[-1], **vals)
    if dn_now >= t_dn > dn_prev and dn_now > up_now:
        return _sig(DOWN, f"model P(down {model['min_move_atr']:g} ATR in {model['horizon']} bars) {dn_now:.2f} "
                          f">= {t_dn:.2f}", b.index[-1], **vals)
    return _sig(None, f"P(up) {up_now:.2f} / P(down) {dn_now:.2f}", b.index[-1], **vals)


# ============================================================================================ CLI
def load_history(index: str, start: date, end: date) -> pd.DataFrame:
    from src.backtest import history as hist
    from src.movement.backtest import load_minutes
    return load_minutes(hist.index_key(index), start, end)


def format_report(m: dict) -> str:
    lines = [f"ML signal model - {m['index']} {m['timeframe']}-min bars, label: move >= {m['min_move_atr']:g} ATR "
             f"within {m['horizon']} bars (same session)",
             f"train {m['train_start']} .. {m['train_end']} ({m['train_days']} days), "
             f"test {m['test_start']} .. {m['test_end']} ({m['test_days']} days, never seen in training)", ""]
    for part in ("train", "test"):
        for side, r in m["report"].get(part, {}).items():
            prec = f"{r['precision']:.1%}" if "precision" in r else "-"
            edge = (f"{r['avg_fwd_atr_signals']:+.2f} ATR vs {r['avg_fwd_atr_all']:+.2f} all bars"
                    if "avg_fwd_atr_signals" in r else "-")
            lines.append(f"  {part:<5} {side:<5} AUC {r['auc']:.3f}  base rate {r['base_rate']:.1%}  "
                         f"signals {r['signals']:>5} at p >= {r['threshold']:.2f}  precision {prec}  "
                         f"avg move {edge}")
    lines += ["", "  strongest features: " + "; ".join(f"{s}: {', '.join(v)}" for s, v in m["top_features"].items()),
              "", "  AUC 0.5 = no better than chance. Judge the TEST rows only, then backtest the strategy with",
              "  option costs: python -m src.movement.backtest --instrument " + m["index"] + " --strategies ml_signal"]
    return "\n".join(lines)


def main(argv=None):
    from src.config.instruments import SUPPORTED_INDICES, get_instrument
    ap = argparse.ArgumentParser(description="Train / inspect the ML signal classifier")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--status", action="store_true", help="list the trained models")
    ap.add_argument("--instrument", default="NIFTY", type=str.upper, choices=list(SUPPORTED_INDICES))
    ap.add_argument("--timeframe", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=6, help="bars ahead the label looks")
    ap.add_argument("--min-move-atr", type=float, default=1.0)
    ap.add_argument("--holdout", type=float, default=0.3, help="share of the latest days kept for the test")
    ap.add_argument("--signal-rate", type=float, default=0.05, help="share of bars the threshold lets through")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    args = ap.parse_args(argv)
    if args.status or not args.train:
        models = list_models()
        if not models:
            print("No trained models in", settings.DATA_DIR)
        for m in models:
            print(format_report(m) + "\n")
        return
    name = get_instrument(args.instrument).name
    minutes = load_history(name, args.start, args.end)
    if minutes.empty:
        raise SystemExit(f"No {name} 1-minute history for {args.start}..{args.end} in data/history or orb_data "
                         f"(python -m src.movement.backtest --download fetches it with an Upstox token).")
    model = train(minutes, name, args.timeframe, args.horizon, args.min_move_atr, args.holdout,
                  signal_rate=args.signal_rate)
    print(format_report(model))
    print(f"\nSaved {model['path']}")


if __name__ == "__main__":
    main()
