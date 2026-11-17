"""
CE / PE signal generation: one StageTracker per index turns each completed signal bar's evaluation
into stages and alerts.

    WATCH (early / high / strong)   price near a meaningful level, setup score rising
    CONFIRMED                       a completed signal bar closed beyond the level + buffer with quality
    ENTRY SETUP                     confirmed + every mandatory filter passed (score, data, option, risk, R:R)
    INVALIDATED                     closed back inside the level, or the watch fell apart
    MISSED                          the first close beyond the level was already too far: do not chase
    NO TRADE                        stale data, blocked session, risk halt, event block

A signal is never produced by an indicator crossing a threshold on its own: it needs a level, a
completed bar beyond it with quality checks, a prior watch, score dominance, an option contract that
passes its filters, and the risk manager's consent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.config.instruments import CE, OPTION_TYPES as DIRECTIONS, PE
from src.config.settings import PBConfig
from src.options.selection import OptionCandidate
from src.risk.manager import RiskManager
from src.scanner.strategies import BarEval, DirectionEval, UnderlyingPlan, underlying_plan
from src.utils.numbers import finite, r2

# stage names (also used by the dashboard colours)
WATCH = "WATCH"
CONFIRMED = "CONFIRMED"
ENTRY = "ENTRY SETUP"
INVALIDATED = "INVALIDATED"
MISSED = "MISSED"
NO_TRADE = "NO TRADE"
EXIT = "EXIT"
DATA = "DATA"


# =============================================================================
# Stage machine
# =============================================================================
@dataclass
class Setup:
    key: str
    index: str
    direction: str
    level: float
    level_label: str
    stage: str
    tier: Optional[str]
    opened_at: str
    updated_at: str
    score: float
    confirmed_at: Optional[str] = None
    confirmed_pos: Optional[int] = None
    plan: Optional[dict] = None
    option: Optional[dict] = None
    avoid_reason: Optional[str] = None
    no_prior_watch: bool = False
    bars_since_confirm: int = 0
    ml_prob: Optional[float] = None
    early: bool = False            # entered BEFORE the level broke (entry_mode = early)
    broke: bool = False            # an early setup's level has since broken

    def to_dict(self):
        return asdict(self)


# spec status words: WATCH / READY / TRIGGERED / INVALIDATED (stages stay as they were for compatibility)
STATUS = {"early_watch": "WATCH", "high_watch": "WATCH", "strong_watch": "READY", "confirmed": "READY",
          "entry": "TRIGGERED", "invalidated": "INVALIDATED", "missed": "INVALIDATED", "exit": "EXIT"}


@dataclass
class Alert:
    time: str
    index: str
    direction: str
    stage: str
    kind: str                 # early_watch / high_watch / strong_watch / confirmed / missed / entry / invalidated / exit / data
    price: float
    level: Optional[float]
    distance_atr: Optional[float]
    score: Optional[float]
    supporting: List[str] = field(default_factory=list)
    contradicting: List[str] = field(default_factory=list)
    option: Optional[dict] = None
    entry_conditions: str = ""
    invalidation: Optional[float] = None
    option_risk: str = ""
    targets: Optional[List[float]] = None
    rr: Optional[float] = None
    expiry_info: str = ""
    freshness: str = ""
    avoid_reason: str = ""
    setup_key: str = ""
    message: str = ""
    mock: bool = False
    ml_prob: Optional[float] = None
    model: str = "v1"
    signal_status: str = ""       # WATCH / READY / TRIGGERED / INVALIDATED
    regime: str = ""              # market regime on the alert bar (v3 only)
    entry_type: str = ""          # early / confirmation / retest (entries only)

    def to_dict(self):
        out = asdict(self)
        for k in ("price", "level", "distance_atr", "score", "invalidation", "rr"):
            out[k] = r2(out[k]) if out[k] is not None else None
        return out


OptionResolver = Callable[[str, UnderlyingPlan, BarEval], Tuple[Optional[OptionCandidate], List[OptionCandidate], str]]


class StageTracker:
    """One per index. Feed it each completed signal bar (BarEval) in order."""

    def __init__(self, index: str, cfg: PBConfig, tag: str = ""):
        self.index = index
        self.cfg = cfg
        self.tag = tag                  # "" for the main model, e.g. "v2" for a shadow model (keeps keys distinct)
        self.ml_gate = None             # optional callable(direction, ev, d, plan) -> (probability, block_reason)
        self.active: Dict[str, Optional[Setup]] = {CE: None, PE: None}
        self.sent: set = set()                 # (setup_key, kind) already alerted -> no duplicates
        self.cooldown: Dict[str, int] = {}     # level key -> bar position until which it is muted
        self.history: List[Setup] = []
        self.day: Optional[date] = None
        self.prev_score = {CE: 0.0, PE: 0.0}
        self.suppressed_duplicates = 0
        self.silent_missed = 0
        self.entries_today = 0

    def _key(self, direction: str, level: float, day: date) -> str:
        key = f"{day.isoformat()}|{self.index}|{direction}|{level:.0f}"
        return f"{self.tag}|{key}" if self.tag else key

    def _alert(self, out: List[Alert], setup: Setup, kind: str, ev: BarEval, d: DirectionEval, **extra) -> None:
        tag = (setup.key, kind)
        if tag in self.sent:
            self.suppressed_duplicates += 1
            return
        self.sent.add(tag)
        regime = ((getattr(ev, "extras", None) or {}).get("regime") or {}).get("name", "")
        out.append(Alert(time=ev.close_time.isoformat(), index=self.index, direction=setup.direction,
                         stage=setup.stage, kind=kind, price=ev.close, level=setup.level,
                         distance_atr=d.distance_atr if finite(d.distance_atr) else None, score=d.score,
                         supporting=list(d.supporting), contradicting=list(d.contradicting),
                         setup_key=setup.key, model=self.tag or "v1", signal_status=STATUS.get(kind, ""),
                         regime=regime, **extra))

    def step(self, ev: BarEval, *, watch_block: Optional[str], entry_block: Optional[str],
             data_ok: bool, data_reason: str, risk: Optional[RiskManager], resolver: Optional[OptionResolver],
             freshness: str = "") -> List[Alert]:
        out: List[Alert] = []
        day = ev.time.date()
        if self.day != day:
            self.day = day
            self.active = {CE: None, PE: None}
            self.sent = set()
            self.cooldown = {}
            self.entries_today = 0
        for direction in DIRECTIONS:
            d = ev.side(direction)
            try:
                self._step_direction(direction, d, ev, out, watch_block, entry_block, data_ok, data_reason, risk,
                                     resolver, freshness)
            finally:
                self.prev_score[direction] = d.score
        return out

    def _step_direction(self, direction, d: DirectionEval, ev: BarEval, out, watch_block, entry_block, data_ok,
                        data_reason, risk, resolver, freshness):
        cfg = self.cfg
        sign = 1.0 if direction == CE else -1.0
        now_s = ev.close_time.isoformat()
        setup = self.active[direction]

        # ---------- an EARLY entry (taken before the level broke): it lives until its stop or its breakout
        if setup is not None and setup.stage == ENTRY and setup.early and not setup.broke:
            setup.bars_since_confirm += 1
            buffer = max(cfg.buffer_atr * ev.atr, ev.close * cfg.buffer_pct / 100.0)
            stop = (setup.plan or {}).get("stop")
            if stop is not None and sign * (ev.close - stop) <= 0:
                setup.stage = INVALIDATED
                setup.updated_at = now_s
                self._alert(out, setup, "invalidated", ev, d, freshness=freshness,
                            message=f"Early entry invalidated: closed through the stop {stop:,.1f}")
                self._close(direction, ev)
            elif sign * (ev.close - setup.level) > buffer:
                setup.broke = True   # from now on it is managed like a confirmed breakout
            return

        # ---------- manage an existing confirmed / entry setup
        if setup is not None and setup.stage in (CONFIRMED, ENTRY):
            setup.bars_since_confirm += 1
            back_inside = sign * (ev.close - setup.level) < -cfg.invalidate_atr * ev.atr
            if back_inside:
                setup.stage = INVALIDATED
                setup.updated_at = now_s
                self._alert(out, setup, "invalidated", ev, d,
                            message=f"Closed back inside {setup.level_label} {setup.level:,.1f}: failed breakout",
                            freshness=freshness)
                self._close(direction, ev)
                return
            if setup.stage == CONFIRMED and cfg.entry_mode == "close" and setup.avoid_reason \
                    and setup.bars_since_confirm <= 2 and setup.plan:
                lo, hi = setup.plan["entry_zone"]
                if lo <= ev.close <= hi:  # e.g. the data was stale on the confirmation bar: one more try
                    self._try_entry(setup, direction, d, ev, out, entry_block, data_ok, data_reason, risk, resolver,
                                    freshness)
                return
            if setup.stage == CONFIRMED and cfg.entry_mode == "retest" and setup.avoid_reason is None:
                tol = cfg.retest_tolerance_atr * ev.atr
                touched = (ev.close - setup.level) <= tol if direction == CE else (setup.level - ev.close) <= tol
                if touched and sign * (ev.close - setup.level) > 0:
                    self._try_entry(setup, direction, d, ev, out, entry_block, data_ok, data_reason, risk, resolver,
                                    freshness, retest=True)
                elif setup.bars_since_confirm >= cfg.retest_max_bars:
                    setup.stage = INVALIDATED
                    setup.avoid_reason = "no retest within the window"
                    self._alert(out, setup, "invalidated", ev, d, message="Retest did not come: setup expired",
                                freshness=freshness)
                    self._close(direction, ev)
            return
        if setup is not None and setup.stage in (INVALIDATED, MISSED):
            self.active[direction] = None
            setup = None

        if d.level is None or not finite(d.level_price):
            if setup is not None and setup.stage == WATCH:
                self._drop_watch(setup, direction, d, ev, out, "level no longer in reach", freshness)
            return
        key = self._key(direction, d.level_price, ev.time.date())
        if setup is not None and setup.stage == WATCH and setup.key != key \
                and abs(setup.level - d.level_price) <= cfg.merge_atr * ev.atr:
            key = setup.key  # same zone, its edge moved a little (a new swing merged into it)
            setup.level = d.level_price
        if self.cooldown.get(key, -1) >= ev.position:
            return

        # ---------- breakout on this completed bar
        if d.breakout:
            pre = max(self.prev_score[direction], d.score)
            if setup is None or setup.key != key:
                setup = Setup(key=key, index=self.index, direction=direction, level=d.level_price,
                              level_label=(d.level or {}).get("sources", "level"), stage=WATCH, tier=None,
                              opened_at=now_s, updated_at=now_s, score=d.score, no_prior_watch=True)
                self.active[direction] = setup
            if watch_block:
                return
            if not data_ok:
                return
            ext = d.extension_atr
            if finite(ext) and ext > cfg.max_extension_atr:
                if setup.no_prior_watch and pre < cfg.confirm_min_score:
                    # nobody was watching this one and it was weak: record silently, no alert
                    setup.stage = MISSED
                    setup.avoid_reason = "extended break of an unwatched, weak setup"
                    self.silent_missed += 1
                    self._close(direction, ev)
                    return
                setup.stage = MISSED
                setup.updated_at = now_s
                setup.avoid_reason = f"closed {ext:.2f} ATR beyond the level (> {cfg.max_extension_atr})"
                self._alert(out, setup, "missed", ev, d, message="Missed move - do not chase",
                            avoid_reason=setup.avoid_reason, freshness=freshness)
                self._close(direction, ev)
                return
            if d.breakout_quality or pre < cfg.confirm_min_score:
                reasons = list(d.breakout_quality)
                if pre < cfg.confirm_min_score:
                    reasons.append(f"score {pre:.1f} < {cfg.confirm_min_score:.0f}")
                setup.avoid_reason = "unconfirmed break: " + "; ".join(reasons)
                setup.updated_at = now_s
                return  # an unconfirmed poke: shown on the dashboard, not alerted
            setup.stage = CONFIRMED
            setup.score = pre
            setup.confirmed_at = now_s
            setup.confirmed_pos = ev.position
            setup.bars_since_confirm = 0
            setup.avoid_reason = None
            plan = underlying_plan(direction, d, ev.close, ev.atr, cfg)
            setup.plan = _plan_dict(plan)
            self._alert(out, setup, "confirmed", ev, d,
                        message=("Bullish breakout confirmed" if direction == CE else "Bearish breakdown confirmed")
                        + (" (no prior watch)" if setup.no_prior_watch else ""),
                        invalidation=plan.stop, targets=[r2(plan.t1), r2(plan.t2)],
                        entry_conditions=_entry_text(direction, plan, cfg), freshness=freshness)
            if cfg.entry_mode == "close":
                self._try_entry(setup, direction, d, ev, out, entry_block, data_ok, data_reason, risk, resolver,
                                freshness, plan=plan)
            return

        # ---------- pre-breakout watch
        within = finite(d.distance_atr) and 0 <= d.distance_atr <= min(cfg.watch_max_atr, cfg.watch_trigger_atr)
        other = ev.side(PE if direction == CE else CE).score
        gated = (d.compression_pct >= cfg.min_compression_pct and d.score - other >= cfg.watch_min_dominance)
        tier = cfg.tier(d.score) if within and gated and not watch_block and data_ok else None
        if setup is not None and setup.stage == WATCH and setup.key != key:
            self._drop_watch(setup, direction, d, ev, out, "the watched level changed", freshness, quiet=True)
            setup = None
        if tier is None:
            if setup is not None and setup.stage == WATCH:
                far = finite(d.distance_atr) and d.distance_atr > cfg.watch_max_atr + 1.0
                weak = d.score < cfg.early_watch - 10
                if far or weak or watch_block or not data_ok:
                    why = ("moved away from the level" if far else "score faded" if weak else
                           (watch_block or data_reason or "blocked"))
                    self._drop_watch(setup, direction, d, ev, out, why, freshness)
            return
        if setup is None:
            setup = Setup(key=key, index=self.index, direction=direction, level=d.level_price,
                          level_label=(d.level or {}).get("sources", "level"), stage=WATCH, tier=tier,
                          opened_at=now_s, updated_at=now_s, score=d.score)
            self.active[direction] = setup
        order = {"early": 1, "high": 2, "strong": 3}
        if setup.tier is None or order[tier] > order.get(setup.tier, 0):
            setup.tier = tier
        setup.score, setup.updated_at = d.score, now_s
        kind = f"{tier}_watch"
        label = {"early": "Early", "high": "High-priority", "strong": "Strong"}[tier]
        word = "PRE-BREAKOUT CE WATCH" if direction == CE else "PRE-BREAKDOWN PE WATCH"
        self._alert(out, setup, kind, ev, d, message=f"{label} {word.lower()}",
                    entry_conditions=_watch_text(direction, d, ev, cfg), freshness=freshness)
        # ---------- early entry: before the break, on a micro-structure trigger, with a reduced risk budget
        if cfg.entry_mode == "early" and setup.stage == WATCH and d.score >= cfg.early_min_score \
                and finite(d.distance_atr) and 0 <= d.distance_atr <= cfg.early_max_distance_atr:
            x = getattr(ev, "extras", None) or {}
            trigger = x.get("prev_high") if direction == CE else x.get("prev_low")
            if trigger is not None and finite(trigger) and sign * (ev.close - trigger) > 0:
                self._try_entry(setup, direction, d, ev, out, entry_block, data_ok, data_reason, risk, resolver,
                                freshness, early=True)

    def _try_entry(self, setup: Setup, direction, d: DirectionEval, ev: BarEval, out, entry_block, data_ok,
                   data_reason, risk: Optional[RiskManager], resolver, freshness, plan: UnderlyingPlan = None,
                   retest=False, early=False):
        cfg = self.cfg
        plan = plan or underlying_plan(direction, d, ev.close, ev.atr, cfg, early=early)
        reasons = []
        if cfg.require_prior_watch and setup.no_prior_watch:
            reasons.append("no pre-breakout WATCH before the break (Stage A skipped)")
        if getattr(d, "entry_veto", None):
            reasons.append(d.entry_veto)
        dominance = d.score - ev.side(PE if direction == CE else CE).score
        if cfg.entry_min_dominance > 0 and dominance < cfg.entry_min_dominance:
            reasons.append(f"{direction} score leads the other side by only {dominance:.0f} (< {cfg.entry_min_dominance:.0f})")
        if setup.score < cfg.entry_min_score and d.score < cfg.entry_min_score:
            reasons.append(f"score {max(setup.score, d.score):.1f} < entry minimum {cfg.entry_min_score:.0f}")
        if entry_block:
            reasons.append(entry_block)
        if not data_ok:
            reasons.append(data_reason or "data not fresh")
        if risk is not None:
            ok, why = risk.can_open(self.index, direction)
            if not ok:
                reasons.append(why)
        setup.ml_prob = None
        if self.ml_gate is not None and not reasons:
            try:
                prob, block = self.ml_gate(direction, ev, d, plan)
            except Exception:
                prob, block = None, None  # the model can never break the rules engine
            setup.ml_prob = prob
            if block:
                reasons.append(block)
        best, shortlist, opt_note = (None, [], "no option resolver")
        if resolver is not None and not reasons:
            best, shortlist, opt_note = resolver(direction, plan, ev)
            if best is None:
                reasons.append(opt_note or "no option passed the filters")
        if best is None and not reasons:
            reasons.append(opt_note or "no option available")
        lots, risk_rs, size_note = 0, np.nan, ""
        if best is not None and not reasons:
            if finite(best.rr) and best.rr < cfg.min_rr:
                reasons.append(f"R:R {best.rr:.2f} < {cfg.min_rr} after costs")
            elif not finite(best.rr):
                reasons.append("R:R could not be estimated")
            if risk is not None and not reasons:
                lots, risk_rs, size_note = risk.size(best, cfg.early_risk_mult if early else 1.0)
                if lots < 1:
                    reasons.append(size_note + (" (early entry: reduced budget)" if early else ""))
                else:
                    best.lots, best.risk_rupees = lots, risk_rs
        setup.plan = _plan_dict(plan)
        if shortlist:
            setup.option = {"best": best.to_dict() if best else None,
                            "shortlist": [c.to_dict() for c in shortlist[:5]], "note": opt_note}
        if reasons:
            setup.avoid_reason = "; ".join(reasons)
            setup.updated_at = ev.close_time.isoformat()
            return False
        setup.stage = ENTRY
        setup.early = early
        setup.updated_at = ev.close_time.isoformat()
        self.entries_today += 1
        option_risk = (f"premium SL {best.sl_premium:.1f} (max -{cfg.max_premium_loss_pct:.0f}%), "
                       f"time stop {cfg.max_hold_minutes:.0f} min / {cfg.square_off:%H:%M}")
        if early:
            option_risk += f", early entry at {cfg.early_risk_mult:.0%} of the normal risk budget"
        targets = [r2(plan.t1), r2(plan.t2), r2(best.t1_premium), r2(best.t2_premium)]
        if cfg.exit_profile == "ladder" and best.ladder:
            steps, book = cfg.ladder_steps()
            option_risk += "; ladder " + ", ".join(f"{r:g}R@{p:.1f} ({b:g}%)" for r, p, b in zip(steps, best.ladder, book))
        kind_text = " (early: before the breakout)" if early else (" (retest)" if retest else "")
        self._alert(out, setup, "entry", ev, d,
                    message=("Entry setup: BUY " + best.symbol) + kind_text,
                    option=best.to_dict(), entry_conditions=_entry_text(direction, plan, cfg, best),
                    invalidation=plan.stop, option_risk=option_risk, targets=targets,
                    rr=best.rr, expiry_info=f"expiry {best.expiry}, lot {best.lot_size}, "
                                            f"vol {best.volume:,.0f}, OI {best.oi:,.0f}",
                    freshness=freshness, ml_prob=setup.ml_prob,
                    entry_type="early" if early else ("retest" if retest else "confirmation"))
        return True

    def _drop_watch(self, setup: Setup, direction, d, ev, out, why, freshness, quiet=False):
        setup.stage = INVALIDATED
        setup.updated_at = ev.close_time.isoformat()
        setup.avoid_reason = why
        if setup.tier in ("high", "strong") and not quiet:
            self._alert(out, setup, "invalidated", ev, d, message=f"Watch cancelled: {why}", freshness=freshness)
        self._close(direction, ev)

    def _close(self, direction, ev: BarEval):
        setup = self.active.get(direction)
        if setup is not None:
            self.history.append(setup)
            self.cooldown[setup.key] = ev.position + self.cfg.cooldown_bars
        self.active[direction] = None if setup is None or setup.stage in (INVALIDATED, MISSED) else setup

    def snapshot(self) -> dict:
        return {d: (s.to_dict() if s else None) for d, s in self.active.items()}


def _plan_dict(plan: UnderlyingPlan) -> dict:
    return {"entry": r2(plan.entry), "entry_zone": [r2(plan.entry_zone[0]), r2(plan.entry_zone[1])],
            "stop": r2(plan.stop), "t1": r2(plan.t1), "t2": r2(plan.t2), "risk_points": r2(plan.risk_points),
            "next_level": r2(plan.next_level) if plan.next_level is not None else None, "notes": plan.notes}


def _watch_text(direction, d: DirectionEval, ev: BarEval, cfg: PBConfig) -> str:
    buf = max(cfg.buffer_atr * ev.atr, ev.close * cfg.buffer_pct / 100)
    trigger = d.level_price + buf if direction == CE else d.level_price - buf
    word = "above" if direction == CE else "below"
    return (f"Wait for a {cfg.signal_minutes}-min close {word} {trigger:,.1f} "
            f"(level {d.level_price:,.1f} + buffer), no further than {cfg.max_extension_atr} ATR beyond it")


def _entry_text(direction, plan: UnderlyingPlan, cfg: PBConfig, best: Optional[OptionCandidate] = None) -> str:
    lo, hi = plan.entry_zone
    text = f"Underlying entry zone {lo:,.1f} - {hi:,.1f}; skip if price is outside it"
    if best is not None and finite(best.entry_premium):
        text += (f". Option: buy up to {best.entry_premium:.1f} (ask {best.ask:.1f}), {best.lots} lot(s) x "
                 f"{best.lot_size}")
    return text
