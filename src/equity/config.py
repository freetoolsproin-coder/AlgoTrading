"""Settings of the equity service: config/equity.ini -> EquityConfig (dataclass defaults when a key is missing)."""
from __future__ import annotations

import configparser
from dataclasses import asdict, dataclass, field, fields
from datetime import time
from pathlib import Path
from typing import Dict, List

from src.config import settings

CONFIG_FILE = settings.CONFIG_DIR / "equity.ini"
DATA_DIR = settings.DATA_DIR / "equity"
MODES = ("analysis", "paper", "live")

# keys that the dashboard may change at run time (stored in the database, they win over the ini file)
RUNTIME_KEYS = ("capital", "risk_per_trade_pct", "max_daily_loss_pct", "max_open_intraday", "max_open_swing", "min_rr", "min_net_rr",
                "daily_trade_limit", "max_consecutive_losses", "max_position_pct", "slippage_bps")


def hhmm(text: str) -> time:
    h, m = str(text).strip().split(":")[:2]
    return time(int(h), int(m))


@dataclass
class IntradayConfig:
    enabled: bool = True
    allow_short: bool = True
    entry_start: str = "09:30"
    entry_end: str = "14:45"
    square_off: str = "15:10"
    orb_minutes: int = 30
    stop_atr_min: float = 1.0
    stop_atr_max: float = 3.0
    min_stop_pct: float = 0.4
    target1_r: float = 1.5
    target2_r: float = 2.5
    min_strength: float = 55.0
    signal_valid_minutes: int = 10
    history_days: int = 20


@dataclass
class SwingConfig:
    enabled: bool = True
    min_hold_days: int = 2
    max_hold_days: int = 5
    stop_atr_min: float = 0.8
    stop_atr_max: float = 1.5
    target1_r: float = 1.5
    target2_r: float = 2.5
    min_confidence: float = 55.0
    history_days: int = 1100
    entry_valid_sessions: int = 1


@dataclass
class RiskConfig:
    capital: float = 500000.0
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_open_intraday: int = 5
    max_open_swing: int = 5
    min_rr: float = 2.0
    min_net_rr: float = 1.5
    daily_trade_limit: int = 10
    max_consecutive_losses: int = 3
    max_position_pct: float = 20.0
    slippage_bps: float = 3.0
    gap_risk_atr: float = 0.5
    max_entry_gap_atr: float = 0.75


@dataclass
class BacktestConfig:
    intraday_years: float = 2.0
    swing_years: float = 3.0
    in_sample_pct: float = 70.0
    walk_forward_folds: int = 4
    min_oos_trades: int = 20
    min_oos_profit_factor: float = 1.1
    min_wf_positive_folds: float = 0.5


@dataclass
class AlertConfig:
    dashboard: bool = True
    desktop_popup: bool = True
    sound: bool = True
    telegram: bool = False
    webhook: bool = False
    popup_seconds: int = 20
    notify_buy_signal: bool = True
    notify_sell_signal: bool = True
    notify_entry: bool = True
    notify_stop: bool = True
    notify_target: bool = True
    notify_trend_reversal: bool = False
    notify_volume_breakout: bool = False
    notify_swing_exit: bool = True
    notify_risk_limit: bool = True
    notify_order: bool = True


@dataclass
class EquityConfig:
    host: str = "127.0.0.1"
    port: int = 8530
    quote_seconds: float = 10.0
    scan_seconds: float = 120.0
    requests_per_second: float = 4.0
    max_requests_per_minute: int = 300
    max_requests_per_30min: int = 1500
    websocket: bool = False
    mode: str = "paper"
    live_confirm_each_order: bool = True
    live_auto_exits: bool = False
    require_validation: bool = True
    paper_unvalidated: bool = True
    strategies_intraday: str = "all"
    strategies_swing: str = "all"
    risk: RiskConfig = field(default_factory=RiskConfig)
    intraday: IntradayConfig = field(default_factory=IntradayConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    costs_intraday: Dict[str, float] = field(default_factory=dict)
    costs_delivery: Dict[str, float] = field(default_factory=dict)
    data_dir: Path = DATA_DIR

    @classmethod
    def from_ini(cls, path=CONFIG_FILE, scanner_ini=settings.CONFIG_FILE) -> "EquityConfig":
        cfg = cls()
        p = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        p.read(str(path), encoding="utf-8")
        _fill(cfg, p, "SERVICE")
        _fill(cfg, p, "EXECUTION")
        for section, target in (("RISK", cfg.risk), ("INTRADAY", cfg.intraday), ("SWING", cfg.swing),
                                ("BACKTEST", cfg.backtest), ("ALERTS", cfg.alerts)):
            _fill(target, p, section)
        if p.has_section("STRATEGIES"):
            cfg.strategies_intraday = p.get("STRATEGIES", "intraday", fallback="all")
            cfg.strategies_swing = p.get("STRATEGIES", "swing", fallback="all")
        if p.has_section("COSTS_DELIVERY"):
            cfg.costs_delivery = {k: float(v) for k, v in p["COSTS_DELIVERY"].items()}
        try:      # intraday charges are shared with the breakout planner ([COSTS_EQUITY] of scanner.ini)
            sp = settings.read_ini(scanner_ini)
            if sp.has_section("COSTS_EQUITY"):
                cfg.costs_intraday = {k: float(v) for k, v in sp["COSTS_EQUITY"].items()}
        except Exception:      # noqa: BLE001 - defaults apply
            pass
        cfg.mode = cfg.mode.strip().lower()
        if cfg.mode not in MODES:
            cfg.mode = "paper"
        return cfg

    def public(self) -> dict:
        d = asdict(self)
        d.pop("data_dir", None)
        return d


def _fill(target, parser: configparser.ConfigParser, section: str):
    if not parser.has_section(section):
        return
    known = {f.name: f for f in fields(target)}
    for key, raw in parser[section].items():
        f = known.get(key)
        if f is None or f.name in ("risk", "intraday", "swing", "backtest", "alerts"):
            continue
        setattr(target, key, coerce(getattr(target, key), raw))


def coerce(current, raw):
    if isinstance(current, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    return str(raw).strip()


def enabled_ids(spec: str, all_ids: List[str]) -> List[str]:
    """'all' / 'all,-x,-y' / 'a,b,c' -> the enabled strategy ids, in registry order."""
    parts = [s.strip().lower() for s in str(spec or "all").split(",") if s.strip()]
    off = {s[1:] for s in parts if s.startswith("-")}
    on = [s for s in parts if not s.startswith("-")]
    base = all_ids if (not on or "all" in on) else [i for i in all_ids if i in on]
    return [i for i in base if i not in off]
