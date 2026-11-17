"""
ML lab: reads what the scanner's ML trade filter (src/scanner/ml_filter.py) has learned and how it was tested.

The filter is a logistic regression that estimates the chance a CONFIRMED breakout reaches 1.5 R on the index before
its stop. It may block entries only when it is "approved": on walk-forward (out-of-sample) data it needed
  * at least 150 out-of-sample setups,
  * AUC >= 0.55 (ranks setups better than chance),
  * Brier skill > 0 (probabilities beat the base rate),
  * mean R of the setups it keeps >= mean R of all setups + 0.05.
This module only reads the saved model (data/prebreakout_ml_model.json) and the live sample log; training is the
existing command, started from the console as a background job.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Optional

from src.console.data import connect_ro

GATE = [  # (key, label, test, requirement text)
    ("oos_samples", "Out-of-sample setups", lambda v, wf: v is not None and v >= 150, "150 or more"),
    ("auc", "AUC (ranking skill)", lambda v, wf: v is not None and v >= 0.55, "0.55 or more (0.50 = coin flip)"),
    ("brier_skill", "Brier skill (probability quality)", lambda v, wf: v is not None and v > 0, "above 0 (beats the base rate)"),
    ("r_uplift", "R uplift of kept setups", lambda v, wf: v is not None and v >= 0.05, "+0.05 R or more"),
]


def _f(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def read_model(path: Path) -> Optional[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def model_summary(path: Path) -> dict:
    data = read_model(path)
    if not data:
        return {"exists": False}
    wf = data.get("walk_forward") or {}
    all_r, kept_r = _f(wf.get("mean_r_all")), _f(wf.get("mean_r_kept"))
    values = {"oos_samples": wf.get("oos_samples"), "auc": _f(wf.get("auc")), "brier_skill": _f(wf.get("brier_skill")),
              "r_uplift": (kept_r - all_r) if (kept_r is not None and all_r is not None) else None}
    gate = [{"key": k, "label": label, "value": values[k], "pass": bool(test(values[k], wf)), "need": need}
            for k, label, test, need in GATE]
    weights = data.get("top_weights") or []
    wmax = max([abs(_f(w.get("weight")) or 0) for w in weights] or [1]) or 1
    return {
        "exists": True, "approved": bool(data.get("approved")), "trained_at": data.get("trained_at"),
        "samples": data.get("samples"), "live_samples": data.get("live_samples", 0),
        "instruments": data.get("instruments") or [], "label": data.get("label") or "",
        "threshold": _f(data.get("threshold")), "wf_ok": bool(wf.get("ok")), "wf_reason": wf.get("reason", ""),
        "base_rate": _f(wf.get("base_rate")), "auc": values["auc"], "brier": _f(wf.get("brier")),
        "brier_skill": values["brier_skill"], "oos_samples": wf.get("oos_samples"), "mean_r_all": all_r,
        "mean_r_kept": kept_r, "mean_r_dropped": _f(wf.get("mean_r_dropped")), "kept_pct": _f(wf.get("kept_pct")),
        "hit_rate_kept": _f(wf.get("hit_rate_kept")), "r_uplift": values["r_uplift"],
        "calibration": [{k: (_f(v) if k != "n" else v) for k, v in row.items()} for row in wf.get("calibration") or []],
        "gate": gate,
        "weights": [{"feature": w.get("feature"), "weight": _f(w.get("weight")),
                     "rel": abs(_f(w.get("weight")) or 0) / wmax} for w in weights],
        "has_model": "model" in data,
    }


def sample_stats(db: Path) -> dict:
    """Live setups the scanner logged for training (data/prebreakout.db, table ml_samples)."""
    c = connect_ro(db)
    if c is None:
        return {"exists": False}
    try:
        if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ml_samples'").fetchone():
            return {"exists": False}
        total, labelled, wins, mock = c.execute(
            "SELECT COUNT(*), SUM(y IS NOT NULL AND mock = 0), SUM(y = 1 AND mock = 0), SUM(mock) FROM ml_samples").fetchone()
        live_lab = c.execute("SELECT COUNT(*) FROM ml_samples WHERE y IS NOT NULL AND mock = 0").fetchone()[0]
        recent = [dict(r) for r in c.execute(
            "SELECT time, idx, direction, entry, stop, y, r FROM ml_samples WHERE mock = 0 ORDER BY time DESC LIMIT 12")]
        avg_r = c.execute("SELECT AVG(r) FROM ml_samples WHERE y IS NOT NULL AND mock = 0").fetchone()[0]
    except sqlite3.Error as error:
        return {"exists": True, "error": str(error)}
    finally:
        c.close()
    return {"exists": True, "total": total or 0, "labelled": labelled or 0, "live_labelled": live_lab or 0,
            "wins": wins or 0, "mock": mock or 0, "avg_r": _f(avg_r), "recent": recent,
            "hit_rate": (wins / labelled * 100) if labelled else None}
