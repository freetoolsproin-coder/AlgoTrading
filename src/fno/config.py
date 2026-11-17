"""
Settings of the F&O engine: config/fno.ini (+ data/fno/overrides.json written by the dashboard).

Every value is a starting point for testing, not a tuned or profitable setting. Charges are read from the
[COSTS] section of config/fno.ini when present, otherwise from config/scanner.ini.
"""
from __future__ import annotations

import configparser
import json
import logging
import os
from dataclasses import dataclass, field, fields
from datetime import time
from pathlib import Path
from typing import Dict, Optional

from src.config import settings
from src.utils.numbers import hhmm

logger = logging.getLogger("scanner.fno.config")
FNO_INI = settings.CONFIG_DIR / "fno.ini"
SUPPORTED_INDICES = ("NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX")
TIMEFRAMES = (1, 3, 5, 15, 30, 60)          # minutes; daily is always derived as well


def data_dir() -> Path:
    path = settings.DATA_DIR / "fno"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce(current, raw):
    value = str(raw).strip()
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, time):
        return hhmm(value)
    if isinstance(current, tuple):
        items = [v.strip() for v in value.split(",") if v.strip()]
        if current and isinstance(current[0], (int, float)) and not isinstance(current[0], bool):
            kind = type(current[0])
            return tuple(kind(float(v)) for v in items)
        return tuple(v.upper() for v in items)
    if isinstance(current, int):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    return value


@dataclass
class FnoConfig:
    # ---- [SERVICE]
    mode: str = "paper"                      # only "paper" is accepted by this engine
    host: str = "127.0.0.1"
    port: int = 8520
    cycle_seconds: int = 30
    capital: float = 500000.0
    websocket: bool = True
    requests_per_second: float = 5.0
    max_data_age_seconds: int = 180          # underlying price older than this -> no new entries
    # ---- [UNIVERSE]
    indices: tuple = SUPPORTED_INDICES
    stocks: str = "all"                      # all | none | comma-separated symbols
    max_stock_scans_per_cycle: int = 25      # candle requests are budgeted (Upstox rate limits)
    stock_refresh_seconds: int = 300
    history_days: int = 12
    benchmark: str = "NIFTY"
    # ---- [TIMEFRAMES]
    base_tf: int = 5                         # signals are evaluated on completed bars of this timeframe
    ltf: int = 1                             # lower timeframe used for entry confirmation (1 or 3)
    htfs: tuple = (15, 30, 60)               # higher timeframes for trend direction (daily is added)
    htf_min_agree: int = 2
    # ---- [CONFLUENCE]
    min_confirmations: int = 4               # of: structure, momentum, location, volume, htf, candle, ltf
    reversal_min_confirmations: int = 3
    required: tuple = ("LOCATION", "CANDLE")
    reversal_required: tuple = ("CANDLE",)
    confirm_bars: int = 2                    # a trigger may be confirmed up to this many bars later
    min_rvol: float = 1.2
    agree_bars: int = 3
    min_confidence: float = 55.0
    confidence_mode: str = "confluence"      # confluence = checks aligned (+ agreeing strategies) | agreement = strategy agreement score
    require_premium_confirmation: bool = True
    premium_lookback_minutes: int = 5
    # ---- [PLAN]
    sl_atr_mult: float = 1.2                 # underlying stop distance in ATR(base_tf)
    min_sl_atr: float = 0.6
    max_sl_atr: float = 2.5
    use_structure_stop: bool = True
    min_premium_sl_pct: float = 8.0
    max_premium_sl_pct: float = 30.0
    premium_atr_mult: float = 1.0            # stop is at least this x premium ATR when premium bars exist
    plan_mode: str = "r"                     # r = stop from the underlying level, targets in R | percent = fixed % of the premium
    sl_pct: float = 10.0                     # percent mode: stop-loss, % below the entry premium
    target_pct: tuple = (15.0, 25.0)         # percent mode: T1 and T2, % above the entry premium (everything is out at T2)
    target_r: tuple = (1.0, 2.0, 3.0)
    book_pct: tuple = (40.0, 30.0, 30.0)
    single_lot_exit: int = 2                 # with one lot: exit everything at this target
    breakeven_after_t1: bool = True
    trail_mode: str = "atr"                  # atr | swing | off
    trail_after: int = 1                     # start trailing after this target
    trail_atr_mult: float = 2.0
    exit_on_opposite: bool = True
    max_hold_minutes: int = 120
    last_entry: time = time(14, 45)
    first_entry: time = time(9, 20)
    square_off: time = time(15, 15)
    min_rr_after_costs: float = 1.2          # T2 reward / risk after charges and spread
    # ---- [CONTRACT]
    selection: str = "delta"                 # delta | moneyness
    moneyness: str = "ATM"                   # ATM | ITM | OTM
    strike_distance: int = 0                 # steps away from ATM for ITM / OTM
    target_delta: float = 0.5
    delta_min: float = 0.35
    delta_max: float = 0.7
    min_premium: float = 5.0
    max_premium: float = 0.0                 # 0 = no cap
    max_spread_pct: float = 3.0
    min_oi: float = 500.0                    # in lots
    min_volume: float = 100.0                # in lots, today
    min_iv: float = 0.0
    max_iv: float = 120.0
    max_iv_pctile: float = 0.0               # 0 = off; no option BUY when the IV percentile is above this (indices)
    iv_pctile_days: int = 30                 # history the IV percentile is measured against
    min_days_to_expiry: int = 0
    expiry: str = "nearest"                  # nearest | next | monthly
    stock_min_days_to_expiry: int = 3
    # ---- [RISK]
    risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    daily_profit_lock_pct: float = 4.0       # 0 = off; stop new entries once realised profit reaches it
    max_trades_per_day: int = 8
    max_open_positions: int = 3
    max_positions_per_underlying: int = 1
    max_consecutive_losses: int = 3
    signal_cooldown_minutes: int = 20        # same underlying + side
    no_trade_windows: str = "09:15-09:20,15:15-15:30"
    event_block_minutes: int = 30
    expiry_day_last_entry: time = time(13, 30)
    expiry_day_risk_mult: float = 0.5
    allow_expiry_day_contract: bool = False  # buy a contract that expires today
    max_slippage_pct: float = 1.0
    slippage_pct: float = 0.25               # paper fills: worse than the quote by this much
    auto_kill_on_daily_loss: bool = True
    skip_if_one_lot_exceeds_risk: bool = True
    # ---- [BACKTEST]
    bt_assumed_spread_pct: float = 1.0       # full bid/ask spread assumed on historical premiums
    bt_slippage_pct: float = 0.25
    bt_oos_fraction: float = 0.3
    bt_wf_train_days: int = 40
    bt_wf_test_days: int = 10
    bt_min_trades: int = 20
    bt_model_iv_mult: float = 1.15           # modelled premiums: IV = realised vol x this
    bt_max_trades_per_day: int = 3           # per strategy and symbol
    # ---- loaded separately
    costs: Dict[str, float] = field(default_factory=dict)
    strategies: Dict[str, dict] = field(default_factory=dict)   # key -> {"enabled": bool, param: value}

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: Optional[Path] = None, overrides: Optional[Path] = None) -> "FnoConfig":
        cfg = cls()
        parser = configparser.ConfigParser(inline_comment_prefixes=(";",), interpolation=None)
        path = Path(path or FNO_INI)
        if path.exists():
            parser.read(path, encoding="utf-8")
        known = {f.name: f for f in fields(cls)}
        for section in parser.sections():
            if section.upper().startswith("STRATEGY_"):
                cfg.strategies[section[9:].lower()] = dict(parser[section])
                continue
            if section.upper() == "COSTS":
                cfg.costs = {k: float(v) for k, v in parser[section].items()}
                continue
            for key, raw in parser[section].items():
                if key not in known or key in ("costs", "strategies"):
                    logger.warning("fno.ini: unknown setting [%s] %s ignored", section, key)
                    continue
                try:
                    setattr(cfg, key, _coerce(getattr(cfg, key), raw))
                except (ValueError, TypeError) as error:
                    logger.error("fno.ini: [%s] %s = %r is invalid (%s); default kept", section, key, raw, error)
        if not cfg.costs:
            base = configparser.ConfigParser(inline_comment_prefixes=(";",), interpolation=None)
            if Path(settings.CONFIG_FILE).exists():
                base.read(settings.CONFIG_FILE, encoding="utf-8")
            if base.has_section("COSTS"):
                cfg.costs = {k: float(v) for k, v in base["COSTS"].items()}
        cfg.apply_overrides(overrides)
        cfg.validate()
        return cfg

    def apply_overrides(self, path: Optional[Path] = None):
        path = Path(path) if path else data_dir() / "overrides.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for key, values in (data.get("strategies") or {}).items():
            self.strategies.setdefault(key, {}).update(values)

    def save_strategy_override(self, key: str, values: dict, path: Optional[Path] = None):
        path = Path(path) if path else data_dir() / "overrides.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("strategies", {}).setdefault(key, {}).update(values)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        self.strategies.setdefault(key, {}).update(values)

    def validate(self):
        if str(self.mode).lower() != "paper":
            raise ValueError("The F&O engine is paper-only: [SERVICE] mode must be 'paper'. Live orders are never "
                             "placed by this engine.")
        bad = [i for i in self.indices if i not in SUPPORTED_INDICES]
        if bad:
            raise ValueError(f"Unsupported index in [UNIVERSE] indices: {', '.join(bad)}")
        if self.base_tf not in TIMEFRAMES or self.ltf not in (1, 3) or self.ltf > self.base_tf:
            raise ValueError("[TIMEFRAMES] base_tf must be one of 1,3,5,15,30,60 and ltf 1 or 3 (<= base_tf)")
        if len(self.target_r) != 3 or len(self.book_pct) != 3 or not (self.target_r[0] < self.target_r[1] < self.target_r[2]):
            raise ValueError("[PLAN] target_r and book_pct need three values, targets ascending")
        self.plan_mode, self.confidence_mode = str(self.plan_mode).lower(), str(self.confidence_mode).lower()
        if self.plan_mode not in ("r", "percent") or self.confidence_mode not in ("confluence", "agreement"):
            raise ValueError("[PLAN] plan_mode must be r or percent, [CONFLUENCE] confidence_mode confluence or agreement")
        if len(self.target_pct) != 2 or not (0 < self.target_pct[0] < self.target_pct[1]) or not 0 < self.sl_pct < 100:
            raise ValueError("[PLAN] target_pct needs two ascending values and sl_pct must be between 0 and 100")
        if not 0 < self.risk_per_trade_pct <= 5:
            raise ValueError("[RISK] risk_per_trade_pct must be between 0 and 5")

    # ---------------------------------------------------------------- helpers
    def windows(self):
        out = []
        for part in str(self.no_trade_windows).split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                try:
                    out.append((hhmm(a), hhmm(b)))
                except ValueError:
                    logger.error("no_trade_windows: '%s' ignored", part)
        return out

    def public(self) -> dict:
        out = {}
        for f in fields(self):
            if f.name in ("costs", "strategies"):
                continue
            v = getattr(self, f.name)
            out[f.name] = v.strftime("%H:%M") if isinstance(v, time) else list(v) if isinstance(v, tuple) else v
        out["costs"] = self.costs
        return out
