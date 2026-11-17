"""
Movement trigger rules: [MOVEMENT] in config/scanner.ini, plus the changes made from the web page.

Changes saved from the web page (PUT /api/config) go to data/movement_rules.json and override the ini values
until they are reset (POST /api/config/reset). The ini file stays the documented default.

Window maps are written "5:0.15, 10:0.25, 15:0.35" (window minutes : value). 0 switches a check off.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

from src.config import settings

logger = logging.getLogger("scanner.movement.rules")

OVERRIDE_FILE = settings.DATA_DIR / "movement_rules.json"
QUALITIES = ("UNCONFIRMED", "CONFIRMED", "STRONG")
MAX_WINDOW = 120


def parse_window_map(raw, cast=float) -> Dict[int, float]:
    """'5:0.15, 10:0.25' or {"5": 0.15} -> {5: 0.15, 10: 0.25}."""
    if isinstance(raw, dict):
        return {int(k): cast(v) for k, v in raw.items()}
    out = {}
    for part in str(raw or "").split(","):
        if not part.strip():
            continue
        key, _, value = part.partition(":")
        if not value.strip():
            raise ValueError(f"'{part.strip()}' should look like minutes:value, e.g. 5:0.15")
        out[int(key.strip())] = cast(value.strip())
    return out


def format_window_map(values: Dict[int, float]) -> str:
    return ", ".join(f"{k}:{v:g}" for k, v in sorted(values.items()))


def _bool(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class MovementRules:
    windows: List[int] = field(default_factory=lambda: [5, 10, 15])
    # price trigger: |move| >= move_pct[w] %  OR  |move| >= move_points[index][w] points (0 = off)
    move_pct: Dict[int, float] = field(default_factory=lambda: {5: 0.15, 10: 0.25, 15: 0.35})
    move_points: Dict[str, Dict[int, float]] = field(default_factory=dict)
    # confirmation checks (0 / false = off). All switched-on checks pass -> CONFIRMED.
    min_strength: float = 1.5        # move / (1-min ATR x sqrt(window)): how unusual the move is
    min_rvol: float = 1.2            # futures volume per minute in the window vs the 30 minutes before it
    min_consistency: float = 0.6     # share of 1-min candles in the window closing in the move's direction
    oi_confirm: bool = False         # futures OI must be building in the move's direction
    strong_mult: float = 1.5         # CONFIRMED and move >= strong_mult x threshold -> STRONG
    # every window moving the same way, each >= alignment_frac x its threshold -> one "ALL" event
    alignment: bool = True
    alignment_frac: float = 0.6
    vix_move_pct: Dict[int, float] = field(default_factory=lambda: {5: 3.0, 10: 4.0, 15: 5.0})
    cooldown_minutes: int = 0        # 0 = the window length
    max_data_age_seconds: int = 120
    poll_seconds: int = 10
    push: bool = True                # Telegram / popup / webhook
    push_min_quality: str = "CONFIRMED"

    # ------------------------------------------------------------------ helpers
    def threshold_pct(self, window: int) -> float:
        return float(self.move_pct.get(window, 0.0) or 0.0)

    def threshold_points(self, index: str, window: int) -> float:
        return float((self.move_points.get(index) or {}).get(window, 0.0) or 0.0)

    def cooldown(self, window: int) -> int:
        return int(self.cooldown_minutes) if self.cooldown_minutes > 0 else int(window)

    def should_push(self, quality: str) -> bool:
        return self.push and QUALITIES.index(quality) >= QUALITIES.index(self.push_min_quality)

    def validate(self):
        problems = []
        self.windows = sorted({int(w) for w in self.windows})
        if not self.windows or any(w < 1 or w > MAX_WINDOW for w in self.windows):
            problems.append(f"windows must be whole minutes between 1 and {MAX_WINDOW}")
        for w in self.windows:
            if self.threshold_pct(w) <= 0 and not any(self.threshold_points(i, w) > 0 for i in self.move_points):
                problems.append(f"window {w} min has no move_pct or move_points threshold")
        if self.push_min_quality not in QUALITIES:
            problems.append(f"push_min_quality must be one of {', '.join(QUALITIES)}")
        if not 0 <= self.min_consistency <= 1:
            problems.append("min_consistency must be between 0 and 1")
        if not 0 < self.alignment_frac <= 2:
            problems.append("alignment_frac must be between 0 and 2")
        if self.strong_mult < 1:
            problems.append("strong_mult must be at least 1")
        if self.poll_seconds < 1:
            problems.append("poll_seconds must be at least 1")
        for name in ("min_strength", "min_rvol", "cooldown_minutes", "max_data_age_seconds"):
            if getattr(self, name) < 0:
                problems.append(f"{name} cannot be negative")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["move_pct"] = {str(k): v for k, v in self.move_pct.items()}
        d["vix_move_pct"] = {str(k): v for k, v in self.vix_move_pct.items()}
        d["move_points"] = {i: {str(k): v for k, v in m.items()} for i, m in self.move_points.items()}
        return d

    def updated(self, changes: dict) -> "MovementRules":
        """A validated copy with `changes` applied (unknown keys are an error)."""
        known = {f.name for f in fields(self)}
        unknown = sorted(set(changes) - known)
        if unknown:
            raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
        new = MovementRules(**{f.name: getattr(self, f.name) for f in fields(self)})
        for key, raw in changes.items():
            setattr(new, key, _coerce(key, raw, getattr(self, key)))
        return new.validate()


def _coerce(key: str, raw, current):
    if key == "windows":
        values = raw if isinstance(raw, list) else str(raw).split(",")
        return [int(float(v)) for v in values if str(v).strip()]
    if key in ("move_pct", "vix_move_pct"):
        return parse_window_map(raw)
    if key == "move_points":
        return {str(i).upper(): parse_window_map(m) for i, m in (raw or {}).items()}
    if key == "push_min_quality":
        return str(raw).strip().upper()
    if isinstance(current, bool):
        return _bool(raw)
    if isinstance(current, int):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    return raw


def from_ini(path=settings.CONFIG_FILE) -> MovementRules:
    parser = settings.read_ini(path)
    rules = MovementRules()
    if not parser.has_section("MOVEMENT"):
        return rules.validate()
    changes, points = {}, {}
    known = {f.name for f in fields(rules)}
    for key, raw in parser["MOVEMENT"].items():
        if key.startswith("move_points_"):
            points[key[len("move_points_"):].upper()] = raw
        elif key in known and key != "move_points":
            changes[key] = raw
    if points:
        changes["move_points"] = points
    return rules.updated(changes)


class RulesStore:
    """Thread-safe current rules: ini defaults + saved web-page overrides."""

    def __init__(self, config_path=settings.CONFIG_FILE, override_file: Path = OVERRIDE_FILE):
        self.config_path = config_path
        self.override_file = Path(override_file)
        self._lock = threading.Lock()
        self.defaults = from_ini(config_path)
        self.overrides: dict = {}
        if self.override_file.exists():
            try:
                saved = json.loads(self.override_file.read_text(encoding="utf-8")) or {}
                self.defaults.updated(saved)
                self.overrides = saved
                logger.info("Movement rules: web-page overrides loaded from %s (%s).", self.override_file.name,
                            ", ".join(sorted(saved)) or "none")
            except (OSError, ValueError, TypeError) as error:
                logger.error("%s is invalid (%s); using config/scanner.ini values.", self.override_file, error)
        self._rules = self.defaults.updated(self.overrides)

    @property
    def rules(self) -> MovementRules:
        with self._lock:
            return self._rules

    def update(self, changes: dict) -> MovementRules:
        with self._lock:
            merged = {**self.overrides, **changes}
            rules = self.defaults.updated(merged)   # raises ValueError on bad input; nothing is saved then
            self.overrides, self._rules = merged, rules
            self._save()
            return rules

    def reset(self) -> MovementRules:
        with self._lock:
            self.overrides, self._rules = {}, self.defaults
            try:
                self.override_file.unlink()
            except FileNotFoundError:
                pass
            return self._rules

    def _save(self):
        self.override_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.override_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.overrides, indent=2), encoding="utf-8")
        tmp.replace(self.override_file)


def describe(rules: Optional[MovementRules] = None) -> str:
    r = rules or MovementRules()
    return (f"windows {r.windows} min, move % {format_window_map(r.move_pct)}, strength >= {r.min_strength}, "
            f"rvol >= {r.min_rvol}, consistency >= {r.min_consistency}")
