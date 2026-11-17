"""Small numeric helpers shared by every module (NaN-safe conversions)."""
from __future__ import annotations

import math
from datetime import time

import numpy as np


def finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fnum(value, default=np.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def r2(value):
    return round(float(value), 2) if finite(value) else None


def hhmm(text: str) -> time:
    hh, mm = str(text).strip().split(":")
    return time(int(hh), int(mm))
