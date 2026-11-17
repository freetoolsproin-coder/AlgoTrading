"""
Paper trading on movement triggers: an UP trigger buys a CE, a DOWN trigger buys a PE. Simulated only.

Two independent books (each with its own settings, limits and P&L):

    momentum  [PAPER_MOMENTUM]  opens on the 5 / 10 / 15-min / ALL triggers of [MOVEMENT]
    scalp     [PAPER_SCALP]     its own fast triggers (1 / 3-min windows by default), tight target and stop,
                                short holding time, exit when the underlying gives back part of the move

Contract: chosen from the live option chain with the scanner's [OPTIONS] filters (delta, premium, spread,
volume, OI). Entry at the ask + slippage, exits at the bid - slippage, net of [COSTS] charges.

Exits, checked every cycle on a fresh option quote (never on a stale one):
    premium stop, premium target, breakeven after +breakeven_pct, trailing stop after that (trail_pct),
    underlying fade (scalp), opposite trigger (exit_on_reverse), time stop, square-off, daily loss limit,
    kill switch file, manual close from the web page.

Safety: one open position per index per book, a daily trade cap, a daily loss limit (realised + open) that
stops new entries and closes the book, a wait after a losing trade, no entries after last_entry or on stale data.
Positions are stored in data/movement.db and restored after a restart.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import settings
from src.config.instruments import CE, PE
from src.config.settings import PBConfig
from src.movement.rules import QUALITIES
from src.options.selection import chain_candidates, rank_options
from src.utils.numbers import finite, fnum, hhmm

logger = logging.getLogger("scanner.movement.paper")
PROFILES = ("momentum", "scalp")
SCALP_TRIGGER_KEYS = ("windows", "move_pct", "min_strength", "min_rvol", "min_consistency", "alignment",
                      "alignment_frac", "oi_confirm", "strong_mult")


@dataclass
class Profile:
    name: str = "momentum"
    enabled: bool = True
    sides: List[str] = field(default_factory=lambda: [CE, PE])
    entry_windows: List[str] = field(default_factory=lambda: ["5", "10", "15", "ALL"])
    min_quality: str = "CONFIRMED"
    lots: int = 1
    stop_pct: float = 20.0           # premium stop, % below entry
    target_pct: float = 40.0         # premium target, % above entry
    breakeven_pct: float = 15.0      # stop moves to entry once the bid is this % above it (0 = off)
    trail_pct: float = 10.0          # after breakeven, stop trails this % below the best bid (0 = off)
    fade_exit: float = 0.0           # exit if the underlying gives back this share of the trigger move (0 = off)
    max_hold_minutes: float = 30.0
    max_trades: int = 10             # per day
    max_daily_loss: float = 10000.0  # rupees, realised + open (0 = off)
    reentry_minutes: float = 5.0     # wait before the same index trades again in this book
    loss_wait_minutes: float = 10.0  # extra wait after a losing trade
    exit_on_reverse: bool = True
    last_entry: str = "14:45"
    square_off: str = "15:15"
    max_quote_age_seconds: float = 60.0
    triggers: Dict[str, object] = field(default_factory=dict)   # scalp: its own trigger rules

    def validate(self):
        problems = []
        self.sides = [s.strip().upper() for s in self.sides if str(s).strip()]
        self.entry_windows = [str(w).strip().upper() for w in self.entry_windows if str(w).strip()]
        if not set(self.sides) <= {CE, PE} or not self.sides:
            problems.append("sides must be CE, PE or both")
        if self.min_quality not in QUALITIES:
            problems.append(f"min_quality must be one of {', '.join(QUALITIES)}")
        if self.lots < 1:
            problems.append("lots must be at least 1")
        if not 0 < self.stop_pct < 100:
            problems.append("stop_pct must be between 0 and 100")
        if self.target_pct <= 0:
            problems.append("target_pct must be positive")
        if not 0 <= self.fade_exit <= 1:
            problems.append("fade_exit must be between 0 and 1")
        for name in ("breakeven_pct", "trail_pct", "max_hold_minutes", "max_daily_loss", "reentry_minutes",
                     "loss_wait_minutes", "max_trades"):
            if getattr(self, name) < 0:
                problems.append(f"{name} cannot be negative")
        try:
            if hhmm(self.last_entry) > hhmm(self.square_off):
                problems.append("last_entry must be before square_off")
        except (ValueError, TypeError):
            problems.append("last_entry / square_off must be HH:MM")
        if problems:
            raise ValueError(f"[PAPER_{self.name.upper()}] " + "; ".join(problems))
        return self

    def to_dict(self):
        return asdict(self)


def load_profiles(path=settings.CONFIG_FILE) -> Dict[str, Profile]:
    parser = settings.read_ini(path)
    out = {}
    for name in PROFILES:
        p = Profile(name=name)
        if name == "scalp":   # scalp defaults: fast triggers, small target, short hold
            p.entry_windows, p.stop_pct, p.target_pct, p.breakeven_pct, p.trail_pct = ["1", "3", "ALL"], 8.0, 12.0, 5.0, 4.0
            p.fade_exit, p.max_hold_minutes, p.max_trades, p.reentry_minutes = 0.5, 8.0, 20, 2.0
            p.min_quality, p.max_daily_loss, p.loss_wait_minutes = "CONFIRMED", 6000.0, 5.0
            p.triggers = {"windows": "1, 3", "move_pct": "1:0.06, 3:0.10", "min_strength": "1.8", "min_rvol": "1.3",
                          "min_consistency": "0.66", "alignment": "true", "alignment_frac": "0.8"}
        section = f"PAPER_{name.upper()}"
        if parser.has_section(section):
            known = {f.name: f for f in fields(Profile)}
            for key, raw in parser[section].items():
                if key.startswith("trigger_"):
                    sub = key[len("trigger_"):]
                    if sub not in SCALP_TRIGGER_KEYS:
                        raise ValueError(f"[{section}] unknown trigger setting {key}")
                    p.triggers[sub] = raw
                    continue
                if key not in known or key in ("name", "triggers"):
                    raise ValueError(f"[{section}] unknown setting {key}")
                current = getattr(p, key)
                if isinstance(current, bool):
                    value = str(raw).strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(current, list):
                    value = [v.strip() for v in str(raw).split(",") if v.strip()]
                elif isinstance(current, int):
                    value = int(float(raw))
                elif isinstance(current, float):
                    value = float(raw)
                else:
                    value = str(raw).strip().upper() if key == "min_quality" else str(raw).strip()
                setattr(p, key, value)
        out[name] = p.validate()
    return out


@dataclass
class Trade:
    id: str
    book: str
    index: str
    side: str
    symbol: str
    instrument_key: Optional[str]
    strike: float
    expiry: Optional[str]
    exchange: str
    lot_size: int
    qty: int
    entry_time: str
    entry_price: float
    stop: float
    target: float
    underlying_entry: float
    trigger: dict
    peak: float = np.nan
    last_price: float = np.nan
    last_quote_time: Optional[str] = None
    charges: float = 0.0
    exit_time: Optional[str] = None
    exit_price: float = np.nan
    exit_reason: Optional[str] = None
    mock: bool = False

    @property
    def open(self) -> bool:
        return self.exit_time is None

    def gross(self) -> float:
        price = self.exit_price if not self.open else self.last_price
        return (price - self.entry_price) * self.qty if finite(price) else 0.0

    def net(self) -> float:
        return self.gross() - self.charges

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(open=self.open, gross=round(self.gross(), 2), net=round(self.net(), 2),
                 pnl_pct=round((self.last_price if self.open else self.exit_price) / self.entry_price * 100 - 100, 2)
                 if finite(self.last_price if self.open else self.exit_price) else None)
        for k, v in d.items():
            if isinstance(v, float) and not np.isfinite(v):
                d[k] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        known = {f.name for f in fields(cls)}
        return cls(**{k: (np.nan if v is None and k in ("peak", "last_price", "exit_price") else v)
                      for k, v in d.items() if k in known})


class PaperTrader:
    """All books. Thread-safe: the monitor thread trades, the web server reads and may close a trade."""

    def __init__(self, profiles: Dict[str, Profile], cfg: PBConfig, costs_fn: Callable, store=None, mock: bool = False,
                 kill_switch_file=None):
        self.profiles = profiles
        self.cfg = cfg
        self.costs_fn = costs_fn
        self.store = store
        self.mock = mock
        self.kill_switch_file = kill_switch_file
        self.trades: List[Trade] = []
        self.halted: Dict[str, str] = {}          # book -> reason (daily loss limit / kill switch)
        self.skipped: List[dict] = []              # recent entries that could not be made, and why
        self._lock = threading.RLock()
        self._day = None
        if store is not None:
            try:
                today = pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()
                self.trades = [Trade.from_dict(d) for d in store.trades(day=today, mock=mock)]
                if any(t.open for t in self.trades):
                    logger.warning("Paper: restored %d open trade(s) from the database.",
                                   sum(t.open for t in self.trades))
            except Exception:
                logger.exception("Could not restore paper trades")

    # ------------------------------------------------------------------ queries
    def open_trades(self, book: Optional[str] = None) -> List[Trade]:
        with self._lock:
            return [t for t in self.trades if t.open and (book is None or t.book == book)]

    def _today(self, book: str, day) -> List[Trade]:
        return [t for t in self.trades if t.book == book and t.entry_time[:10] == day.isoformat()]

    def day_pnl(self, book: str, day) -> float:
        return sum(t.net() for t in self._today(book, day))

    def summary(self, now: Optional[pd.Timestamp] = None) -> dict:
        now = now or pd.Timestamp.now(tz="Asia/Kolkata")
        out = {}
        with self._lock:
            for name, p in self.profiles.items():
                today = self._today(name, now.date())
                closed = [t for t in today if not t.open]
                wins = [t for t in closed if t.net() > 0]
                out[name] = {"enabled": p.enabled, "halted": self.halted.get(name),
                             "open": [t.to_dict() for t in today if t.open],
                             "closed": [t.to_dict() for t in sorted(closed, key=lambda t: t.exit_time, reverse=True)],
                             "trades": len(today), "max_trades": p.max_trades,
                             "realised": round(sum(t.net() for t in closed), 2),
                             "open_pnl": round(sum(t.net() for t in today if t.open), 2),
                             "net": round(sum(t.net() for t in today), 2),
                             "win_rate": round(100 * len(wins) / len(closed), 1) if closed else None,
                             "settings": p.to_dict()}
            out["skipped"] = list(self.skipped[-20:])
        return out

    # ------------------------------------------------------------------ entries
    def wants(self, book: str, trigger: dict) -> bool:
        p = self.profiles.get(book)
        if p is None or not p.enabled or trigger.get("index") == "INDIAVIX":
            return False
        side = CE if trigger["direction"] == "UP" else PE
        return (side in p.sides and str(trigger["window"]).upper() in p.entry_windows
                and QUALITIES.index(trigger["quality"]) >= QUALITIES.index(p.min_quality))

    def entry_block(self, book: str, index: str, now: pd.Timestamp) -> Optional[str]:
        """Why a new entry is not allowed right now (None = allowed)."""
        p = self.profiles[book]
        if self.kill_switch_file is not None and self.kill_switch_file.exists():
            return "kill switch file present"
        if self.halted.get(book):
            return self.halted[book]
        if now.time() > hhmm(p.last_entry):
            return f"after last entry time {p.last_entry}"
        today = self._today(book, now.date())
        if len(today) >= p.max_trades:
            return f"daily trade cap {p.max_trades} reached"
        if any(t.open and t.index == index for t in today):
            return "position already open on this index"
        done = [t for t in today if t.index == index and not t.open]
        if done:
            last = max(done, key=lambda t: t.exit_time)
            wait = p.reentry_minutes + (p.loss_wait_minutes if last.net() < 0 else 0)
            if now - pd.Timestamp(last.exit_time) < pd.Timedelta(minutes=wait):
                return f"waiting {wait:g} min after the last {'losing ' if last.net() < 0 else ''}trade"
        return None

    def choose(self, isnap, side: str, now: pd.Timestamp):
        """(candidate, reason) from the index's option chain with the [OPTIONS] filters."""
        if isnap.chain is None or isnap.chain.empty:
            return None, "option chain unavailable"
        cands = chain_candidates(isnap.chain, float(isnap.ltp), side, isnap.step, isnap.lot_size, isnap.expiry, now,
                                 self.cfg, symbol_prefix=isnap.name)
        ranked = rank_options(cands, self.cfg)
        if not ranked:
            return None, "no strikes near ATM"
        if ranked[0].rejected:
            return None, "no strike passed the filters: " + "; ".join(sorted({r for c in ranked for r in c.rejected})[:3])
        if ranked[0].lot_size < 1:
            return None, "lot size unknown"
        return ranked[0], ""

    def try_open(self, book: str, trigger: dict, isnap, now: pd.Timestamp,
                 fresh_quote: Optional[Callable[[str], Optional[dict]]] = None) -> Optional[Trade]:
        """Open a trade for one trigger, or record why not. fresh_quote(key) -> {bid, ask, ltp} (live) or None."""
        with self._lock:
            side = CE if trigger["direction"] == "UP" else PE
            block = self.entry_block(book, trigger["index"], now)
            cand = None
            if block is None:
                cand, block = self.choose(isnap, side, now)
                block = block or None
            ask = np.nan
            if block is None:
                ask = cand.ask
                if fresh_quote is not None and cand.instrument_key:
                    q = fresh_quote(cand.instrument_key) or {}
                    if finite(fnum(q.get("ask"))) and fnum(q.get("ask")) > 0:
                        ask = fnum(q.get("ask"))
                if not (finite(ask) and ask > 0):
                    block = "no ask price for the chosen contract"
            if block is not None:
                self._skip(book, trigger, block, now)
                return None
            p = self.profiles[book]
            entry = round(ask * (1 + self.cfg.slippage_pct / 100.0), 2)
            qty = p.lots * cand.lot_size
            t = Trade(id=f"{book}|{trigger['index']}|{now.isoformat()}", book=book, index=trigger["index"], side=side,
                      symbol=cand.symbol, instrument_key=cand.instrument_key, strike=cand.strike, expiry=cand.expiry,
                      exchange=isnap.exchange, lot_size=cand.lot_size, qty=qty, entry_time=now.isoformat(),
                      entry_price=entry, stop=round(entry * (1 - p.stop_pct / 100.0), 2),
                      target=round(entry * (1 + p.target_pct / 100.0), 2), underlying_entry=float(isnap.ltp),
                      trigger={k: trigger.get(k) for k in ("window", "quality", "change", "change_pct", "ref_price",
                                                             "price", "message")},
                      peak=entry, last_price=entry, last_quote_time=now.isoformat(), mock=self.mock)
            t.charges = self.costs_fn("BUY", entry, qty, t.exchange)
            self.trades.append(t)
            self._save(t)
            return t

    def _skip(self, book, trigger, reason, now):
        self.skipped.append({"time": now.isoformat(), "book": book, "index": trigger["index"],
                             "direction": trigger["direction"], "window": trigger["window"], "reason": reason})
        del self.skipped[:-50]
        logger.info("Paper %s: no entry on %s %s (%s min): %s", book, trigger["index"], trigger["direction"],
                    trigger["window"], reason)

    # ------------------------------------------------------------------ exits
    def mark(self, quotes: Dict[str, dict], underlying: Dict[str, float], now: pd.Timestamp,
             reverse: Optional[Dict[tuple, str]] = None) -> List[Trade]:
        """Update open trades with fresh quotes {instrument_key: {bid, ltp, time}} and apply the exits.
        reverse: {(book, index): direction} of new opposite-direction triggers. Returns the trades closed now."""
        closed = []
        with self._lock:
            for t in [t for t in self.trades if t.open]:
                p = self.profiles.get(t.book) or Profile()
                q = quotes.get(t.instrument_key) or {}
                bid = fnum(q.get("bid"))
                if not (finite(bid) and bid > 0):
                    ltp = fnum(q.get("ltp"))
                    bid = ltp * (1 - self.cfg.assumed_spread_pct / 200.0) if finite(ltp) and ltp > 0 else np.nan
                if finite(bid):
                    t.last_price, t.last_quote_time = bid, now.isoformat()
                    t.peak = max(t.peak, bid) if finite(t.peak) else bid
                reason = self._exit_reason(t, p, now, underlying.get(t.index), finite(bid), reverse or {})
                if reason:
                    self._close(t, now, reason)
                    closed.append(t)
                else:
                    self._save(t)
            closed += self._loss_limits(now)
        return closed

    def _exit_reason(self, t: Trade, p: Profile, now, und, fresh: bool, reverse) -> Optional[str]:
        held = (now - pd.Timestamp(t.entry_time)).total_seconds() / 60.0
        if now.time() >= hhmm(p.square_off):
            return "square-off"
        if self.kill_switch_file is not None and self.kill_switch_file.exists():
            return "kill switch"
        if not fresh:   # never exit on a price we did not see; time-based exits still need a last price
            age = (now - pd.Timestamp(t.last_quote_time)).total_seconds() if t.last_quote_time else 1e9
            if held >= p.max_hold_minutes and age <= p.max_quote_age_seconds * 3:
                return "time stop (last quote)"
            return None
        if p.breakeven_pct > 0 and t.peak >= t.entry_price * (1 + p.breakeven_pct / 100.0):
            t.stop = max(t.stop, t.entry_price)
            if p.trail_pct > 0:
                t.stop = max(t.stop, round(t.peak * (1 - p.trail_pct / 100.0), 2))
        if t.last_price <= t.stop:
            return "stop" if t.stop < t.entry_price else ("breakeven stop" if t.stop == t.entry_price else "trailing stop")
        if t.last_price >= t.target:
            return "target"
        if p.fade_exit > 0 and und is not None and finite(und):
            move = t.underlying_entry - fnum(t.trigger.get("ref_price"))
            gave_back = (t.underlying_entry - und) if t.side == CE else (und - t.underlying_entry)
            if finite(move) and abs(move) > 0 and gave_back >= p.fade_exit * abs(move):
                return "momentum faded"
        if p.exit_on_reverse:
            d = reverse.get((t.book, t.index))
            if d and ((t.side == CE and d == "DOWN") or (t.side == PE and d == "UP")):
                return "opposite trigger"
        if held >= p.max_hold_minutes:
            return "time stop"
        return None

    def _loss_limits(self, now) -> List[Trade]:
        closed = []
        for name, p in self.profiles.items():
            if p.max_daily_loss > 0 and not self.halted.get(name) and self.day_pnl(name, now.date()) <= -p.max_daily_loss:
                self.halted[name] = f"daily loss limit Rs {p.max_daily_loss:,.0f} reached"
                logger.warning("Paper %s: %s - book closed for the day.", name, self.halted[name])
                for t in [t for t in self.trades if t.open and t.book == name]:
                    self._close(t, now, "daily loss limit")
                    closed.append(t)
        return closed

    def close_manual(self, trade_id: str, now: pd.Timestamp) -> Optional[Trade]:
        with self._lock:
            t = next((t for t in self.trades if t.id == trade_id and t.open), None)
            if t is not None:
                self._close(t, now, "closed from the web page")
            return t

    def _close(self, t: Trade, now, reason):
        price = t.last_price if finite(t.last_price) else t.entry_price
        t.exit_price = round(max(0.05, price * (1 - self.cfg.slippage_pct / 100.0)), 2)
        t.exit_time, t.exit_reason = now.isoformat(), reason
        t.charges += self.costs_fn("SELL", t.exit_price, t.qty, t.exchange)
        self._save(t)

    def new_day(self, day):
        with self._lock:
            if self._day != day:
                self._day = day
                self.halted = {}

    def _save(self, t: Trade):
        if self.store is not None:
            try:
                self.store.save_trade(t.to_dict())
            except Exception:
                logger.exception("Could not save paper trade %s", t.id)


def scalp_rules(base_rules, profile: Profile):
    """The movement rules the scalp book triggers on: the base rules with the profile's trigger overrides."""
    changes = dict(profile.triggers)
    changes.setdefault("cooldown_minutes", 0)
    changes["vix_move_pct"] = {}
    return base_rules.updated(changes)
