"""
Settings: project paths, secrets (.env / environment), strategy configuration and application settings.

Everything tunable lives in config/scanner.ini (strategy, risk, notifications, dashboard). Secrets live
only in .env or real environment variables, never in an .ini file and never in a log line.

Paths are derived from this file's location with pathlib, so the application works from any working
directory (VS Code, PyCharm, a terminal) and on Windows.
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Dict, Optional

from src.config.instruments import SUPPORTED_INDICES, validate_indices
from src.scanner import indicators as ind
from src.utils.numbers import hhmm

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "scanner.ini"
HOLIDAYS_FILE = CONFIG_DIR / "nse_holidays.csv"
DEFAULT_EVENTS_FILE = CONFIG_DIR / "events.csv"
ENV_FILE = PROJECT_ROOT / ".env"
TOKEN_FILE = PROJECT_ROOT / "upstox_token.json"       # written by `python -m src.broker.auth` (git-ignored)

# Runtime files (database, cache, outputs, logs) live under the project root, or under SCANNER_RUNTIME_DIR
# when that environment variable is set (the tests use it so they never touch your real data).
RUNTIME_ROOT = Path(os.environ["SCANNER_RUNTIME_DIR"]).resolve() if os.environ.get("SCANNER_RUNTIME_DIR") \
    else PROJECT_ROOT
DATA_DIR = RUNTIME_ROOT / "data"
DB_FILE = DATA_DIR / "prebreakout.db"
ML_MODEL_FILE = DATA_DIR / "prebreakout_ml_model.json"
CACHE_DIR = DATA_DIR / "cache"                        # instrument master, constituent lists
HISTORY_DIR = DATA_DIR / "history"                    # backtest 1-minute history (monthly CSVs)
LOCK_FILE = DATA_DIR / "scanner.lock"

OUTPUT_DIR = RUNTIME_ROOT / "output"
STATE_FILE = OUTPUT_DIR / "prebreakout_state.json"
HTML_FILE = OUTPUT_DIR / "prebreakout_dashboard.html"
ALERTS_CSV = OUTPUT_DIR / "prebreakout_alerts.csv"

LOG_DIR = RUNTIME_ROOT / "logs"
LOG_FILE = LOG_DIR / "scanner.log"


def ensure_dirs():
    for path in (DATA_DIR, CACHE_DIR, HISTORY_DIR, OUTPUT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Secrets (.env + environment). An environment variable wins over .env.
# =============================================================================
SECRET_KEYS = ("UPSTOX_API_KEY", "UPSTOX_API_SECRET", "UPSTOX_REDIRECT_URI", "UPSTOX_ACCESS_TOKEN",
               "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "LIVE_TRADING_CONFIRM")


def read_env_file(path: Optional[Path] = None) -> Dict[str, str]:
    path = path or ENV_FILE
    values: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key.lower().startswith("export "):
                    key = key[7:].strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if key:
                    values[key] = value
    except FileNotFoundError:
        pass
    return values


def get_secret(name: str, default: Optional[str] = None, path: Optional[Path] = None) -> Optional[str]:
    """Environment variable first, then .env, then `default`. Empty strings count as missing."""
    value = os.environ.get(name)
    if value:
        return value.strip()
    value = read_env_file(path).get(name)
    if value:
        return value.strip()
    if path is None:   # last: the console's encrypted vault (data/console/vault.json), if one exists
        try:
            from src.console.vault import vault_secret
            value = vault_secret(name)
        except Exception:   # noqa: BLE001 - the vault is optional and must never break the scanner
            value = None
        if value:
            return value.strip()
    return default


def redact(secret: Optional[str]) -> str:
    """For log lines: never print a secret, only whether it is set."""
    if not secret:
        return "<not set>"
    return f"<set, {len(secret)} chars>"


def _coerce(current, raw):
    """Convert an .ini string to the type of the default value `current`."""
    value = str(raw).strip()
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, time):
        return hhmm(value)
    if isinstance(current, tuple):
        return tuple(v.strip().upper() for v in value.split(",") if v.strip())
    if isinstance(current, int):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    return value


# =============================================================================
# Strategy configuration (sections SERVICE ... RISK, STRATEGY_V3 of config/scanner.ini)
# =============================================================================
@dataclass
class PBConfig:
    # service
    mode: str = "live"
    indices: tuple = SUPPORTED_INDICES
    # data quality
    max_spot_age_seconds: int = 120
    max_chain_age_seconds: int = 180
    max_candle_lag_minutes: int = 7
    max_spread_pct: float = 3.0
    max_spread_points: float = 3.0
    # timeframe
    signal_minutes: int = 5
    opening_range_minutes: int = 15
    watch_start: time = time(9, 30)
    last_entry: time = time(14, 45)
    square_off: time = time(15, 15)
    # indicators
    ind: ind.IndicatorParams = field(default_factory=ind.IndicatorParams)
    # levels
    merge_atr: float = 0.30
    watch_max_atr: float = 1.5
    near_atr: float = 0.30
    min_zone_strength: int = 3
    # scoring
    w_location: float = 20
    w_level: float = 10
    w_compression: float = 15
    w_structure: float = 15
    w_trend: float = 15
    w_momentum: float = 10
    w_participation: float = 10
    w_context: float = 5
    rsi_ce_min: float = 52
    rsi_ce_max: float = 70
    rsi_pe_min: float = 30
    rsi_pe_max: float = 48
    adx_min: float = 18
    rvol_good: float = 1.2
    stretched_atr: float = 2.5
    # thresholds
    early_watch: float = 50
    high_watch: float = 65
    strong_watch: float = 80
    confirm_min_score: float = 55
    entry_min_score: float = 65
    # optional "dominance / compression" gates (0 = off)
    watch_trigger_atr: float = 1.5
    min_compression_pct: float = 0
    watch_min_dominance: float = 15
    entry_min_dominance: float = 20
    min_history_bars: int = 60
    # confirmation
    buffer_atr: float = 0.10
    buffer_pct: float = 0.02
    min_body_ratio: float = 0.45
    min_close_position: float = 0.60
    min_rvol: float = 1.0
    require_vwap: bool = True
    require_trend: bool = False
    require_prior_watch: bool = True
    max_extension_atr: float = 0.80
    entry_mode: str = "close"
    retest_tolerance_atr: float = 0.25
    retest_max_bars: int = 6
    invalidate_atr: float = 0.25
    cooldown_bars: int = 6
    intrabar_alerts: bool = False
    # trade plan
    sl_atr: float = 0.50
    max_sl_atr: float = 1.20
    t1_r: float = 1.5
    t2_r: float = 3.0
    min_rr: float = 1.2
    max_premium_loss_pct: float = 35
    expected_hold_minutes: float = 45
    max_hold_minutes: float = 120
    exit_on_failed_breakout: bool = False
    trail_after_t1_pct: float = 0
    # options
    strikes_each_side: int = 3
    delta_min: float = 0.35
    delta_max: float = 0.70
    delta_target: float = 0.55
    min_premium: float = 20
    min_volume: float = 1000
    min_oi: float = 5000
    allow_0dte: bool = False
    min_hours_to_expiry: float = 4
    slippage_pct: float = 0.25
    assumed_spread_pct: float = 1.0
    # risk
    capital: float = 500000
    risk_per_trade_pct: float = 1.0
    max_daily_loss_rupees: float = 60000
    max_open_positions: int = 3
    max_trades_per_index: int = 3
    max_trades_per_session: int = 8
    max_premium_pct_of_capital: float = 20
    gap_atr: float = 0.6
    expiry_day_cutoff: time = time(13, 30)
    max_vix: float = 30
    max_consecutive_losses: int = 0
    max_same_direction: int = 0
    # risk (portfolio level; 0 = off)
    max_daily_loss_pct: float = 0          # % of capital; the tighter of this and max_daily_loss_rupees applies
    max_weekly_loss_pct: float = 0         # % of capital, realised + open, Monday..today
    max_open_premium_pct: float = 0        # total premium of open positions as % of capital
    max_lots_per_expiry: int = 0           # open lots on one expiry date across both indices
    correlated_risk_mult: float = 0        # same-side open risk (NIFTY + SENSEX) <= mult x one trade's budget
    # trade plan: stop models, exit profiles, early entries
    stop_model: str = "level"              # level | swing | atr
    atr_stop_mult: float = 1.0
    exit_profile: str = "classic"          # classic (half at T1, rest at T2) | ladder
    ladder_r: str = "1.0,1.5,2.0,3.0"
    ladder_book_pct: str = "25,25,0,0"     # % of the ORIGINAL quantity booked at each ladder step; rest trails
    trail_mode: str = "none"               # none | atr | swing (underlying-based, after the first ladder step)
    trail_atr_mult: float = 1.0
    vwap_exit: bool = False                # exit when the underlying closes back across VWAP against the trade
    momentum_exit: bool = False            # exit when momentum exhausts (MACD histogram flips + RSI turns)
    early_risk_mult: float = 0.5           # early entries risk this fraction of the normal budget
    early_min_score: float = 75
    early_max_distance_atr: float = 0.35
    # strategy v3 (src/scanner/score_v3.py and its modules)
    kc_mult: float = 1.5
    or_windows: str = "5,15,30"
    ib_minutes: int = 60
    eq_tol_atr: float = 0.10
    htf_minutes: str = "15,30"
    sweep_lookback: int = 3
    w3_structure: float = 20
    w3_compression: float = 15
    w3_momentum: float = 15
    w3_vwap: float = 15
    w3_participation: float = 10
    w3_derivatives: float = 15
    w3_mtf: float = 10
    regime_adapt: bool = True
    block_uncertain_regime: bool = True
    allow_reversal_in_trend: bool = False
    trap_action: str = "penalize"          # report | penalize | block
    trap_block_points: float = 25
    iv_spike_pct: float = 10.0

    SECTIONS = ("SERVICE", "DATA_QUALITY", "TIMEFRAME", "LEVELS", "SCORING", "THRESHOLDS", "CONFIRMATION",
                "TRADE_PLAN", "OPTIONS", "RISK", "STRATEGY_V3")

    @classmethod
    def from_ini(cls, path=CONFIG_FILE) -> "PBConfig":
        return cls.from_parser(read_ini(path))

    @classmethod
    def from_parser(cls, parser: configparser.ConfigParser) -> "PBConfig":
        cfg = cls()
        defaults = cls()
        for section in cls.SECTIONS:
            if not parser.has_section(section):
                continue
            for key, raw in parser[section].items():
                if not hasattr(defaults, key) or key in ("ind",):
                    continue
                setattr(cfg, key, _coerce(getattr(defaults, key), raw))
        if parser.has_section("INDICATORS"):
            sec = parser["INDICATORS"]
            p = ind.IndicatorParams()
            if sec.get("ema_periods"):
                p.ema_periods = tuple(int(v) for v in sec.get("ema_periods").split(","))
            for key in ("rsi_length", "macd_fast", "macd_slow", "macd_signal", "adx_length", "atr_length",
                        "bb_length", "volume_ma", "swing_left", "swing_right"):
                if sec.get(key):
                    setattr(p, key, int(float(sec.get(key))))
            if sec.get("bb_std"):
                p.bb_std = float(sec.get("bb_std"))
            cfg.ind = p
        cfg.validate()
        return cfg

    def validate(self):
        problems = []
        try:
            self.indices = validate_indices(self.indices)
        except ValueError as error:
            problems.append(str(error))
        if self.mode not in ("live", "mock"):
            problems.append("mode must be 'live' or 'mock'")
        if not (0 < self.early_watch <= self.high_watch <= self.strong_watch <= 100):
            problems.append("thresholds must satisfy 0 < early <= high <= strong <= 100")
        if self.signal_minutes not in (1, 2, 3, 5, 10, 15):
            problems.append("signal_minutes must be 1, 2, 3, 5, 10 or 15")
        if not (0 < self.delta_min < self.delta_max <= 1):
            problems.append("delta_min < delta_max required")
        if self.risk_per_trade_pct <= 0 or self.risk_per_trade_pct > 5:
            problems.append("risk_per_trade_pct must be in (0, 5]")
        if self.min_rr <= 0:
            problems.append("min_rr must be > 0")
        if self.max_open_positions < 1 or self.max_trades_per_session < 1 or self.max_trades_per_index < 1:
            problems.append("max_open_positions, max_trades_per_session and max_trades_per_index must be >= 1")
        if self.max_daily_loss_rupees < 0:
            problems.append("max_daily_loss_rupees must be >= 0 (0 = off)")
        if not (0 <= self.trail_after_t1_pct < 100):
            problems.append("trail_after_t1_pct must be in [0, 100)")
        if self.max_consecutive_losses < 0 or self.max_same_direction < 0:
            problems.append("max_consecutive_losses and max_same_direction must be >= 0 (0 = off)")
        if 20 not in self.ind.ema_periods or 50 not in self.ind.ema_periods or 9 not in self.ind.ema_periods:
            problems.append("ema_periods must include 9, 20 and 50")
        if self.entry_mode not in ("close", "retest", "early"):
            problems.append("entry_mode must be 'close', 'retest' or 'early'")
        if self.stop_model not in ("level", "swing", "atr"):
            problems.append("stop_model must be 'level', 'swing' or 'atr'")
        if self.exit_profile not in ("classic", "ladder"):
            problems.append("exit_profile must be 'classic' or 'ladder'")
        if self.trail_mode not in ("none", "atr", "swing"):
            problems.append("trail_mode must be 'none', 'atr' or 'swing'")
        if self.trap_action not in ("report", "penalize", "block"):
            problems.append("trap_action must be 'report', 'penalize' or 'block'")
        try:
            steps, book = self.ladder_steps()
            if not steps or any(b < 0 for b in book) or sum(book) > 100 or sorted(steps) != list(steps):
                problems.append("ladder_r must be ascending and ladder_book_pct must be >= 0 and sum to <= 100")
        except ValueError:
            problems.append("ladder_r / ladder_book_pct must be comma-separated numbers")
        if not (0 < self.early_risk_mult <= 1):
            problems.append("early_risk_mult must be in (0, 1]")
        for k in ("max_daily_loss_pct", "max_weekly_loss_pct", "max_open_premium_pct", "correlated_risk_mult"):
            if getattr(self, k) < 0:
                problems.append(f"{k} must be >= 0 (0 = off)")
        w3 = sum(self.weights_v3.values())
        if abs(w3 - 100) > 0.5:
            problems.append(f"[STRATEGY_V3] w3_* weights must sum to 100 (they sum to {w3:g})")
        if problems:
            raise ValueError("config/scanner.ini: " + "; ".join(problems))

    def replace(self, **changes) -> "PBConfig":
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data.update(changes)
        return PBConfig(**data)

    def with_overrides(self, raw: Dict[str, str]) -> "PBConfig":
        """A copy with string overrides (e.g. the [SCORE_V3] section's per-lane settings) applied and validated."""
        changes = {}
        for key, value in (raw or {}).items():
            if key in self.__dataclass_fields__ and key not in ("ind",):
                changes[key] = _coerce(getattr(PBConfig(), key), value)
        out = self.replace(**changes)
        out.validate()
        return out

    def ladder_steps(self):
        steps = [float(x) for x in str(self.ladder_r).split(",") if x.strip()]
        book = [float(x) for x in str(self.ladder_book_pct).split(",") if x.strip()]
        book = (book + [0.0] * len(steps))[:len(steps)]
        return steps, book

    @property
    def weights_v3(self) -> Dict[str, float]:
        return {"Market structure": self.w3_structure, "Volatility compression": self.w3_compression,
                "Momentum": self.w3_momentum, "VWAP & acceptance": self.w3_vwap,
                "Volume & participation": self.w3_participation, "Derivatives": self.w3_derivatives,
                "Multi-timeframe": self.w3_mtf}

    @property
    def weights(self) -> Dict[str, float]:
        return {"location": self.w_location, "level": self.w_level, "compression": self.w_compression,
                "structure": self.w_structure, "trend": self.w_trend, "momentum": self.w_momentum,
                "participation": self.w_participation, "context": self.w_context}

    def tier(self, score: float) -> Optional[str]:
        if score >= self.strong_watch:
            return "strong"
        if score >= self.high_watch:
            return "high"
        if score >= self.early_watch:
            return "early"
        return None

    def thresholds_dict(self) -> dict:
        keys = ("early_watch", "high_watch", "strong_watch", "confirm_min_score", "entry_min_score", "buffer_atr",
                "max_extension_atr", "min_rvol", "min_rr", "entry_mode", "signal_minutes", "risk_per_trade_pct",
                "max_daily_loss_rupees", "max_open_positions", "max_trades_per_index", "max_trades_per_session")
        out = {}
        for k in keys:
            v = getattr(self, k)
            out[k] = v.strftime("%H:%M") if isinstance(v, time) else v
        return out


# =============================================================================
# Application settings (service, data feed, dashboard, notifications, costs)
# =============================================================================
@dataclass
class AppSettings:
    mode: str = "live"
    indices: tuple = SUPPORTED_INDICES
    loop_seconds: int = 30
    requests_per_second: float = 5.0
    use_websocket: bool = True
    paper_trading: bool = True        # False = alert-only mode (ENTRY alerts, no paper positions)
    events_file: Path = DEFAULT_EVENTS_FILE
    kill_switch_file: Path = PROJECT_ROOT / "KILL_SWITCH"
    notify: Dict[str, str] = field(default_factory=dict)
    v2_enabled: bool = True
    v2_push: bool = False
    ml_enabled: bool = True
    ml_model_file: Path = ML_MODEL_FILE
    ml_allow_block: bool = True
    v3_enabled: bool = True
    v3_push: bool = False
    v3_overrides: Dict[str, str] = field(default_factory=dict)   # [SCORE_V3] keys that are strategy settings
    execution: Dict[str, str] = field(default_factory=dict)      # [EXECUTION] (src/execution/order_manager.py)
    dashboard_enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8502
    costs: Dict[str, float] = field(default_factory=dict)
    config_file: Path = CONFIG_FILE


def read_ini(path=CONFIG_FILE) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None)
    parser.optionxform = str
    if not parser.read(str(path), encoding="utf-8"):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return parser


def _project_path(value: str) -> Path:
    path = Path(value.strip())
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_app_settings(path=CONFIG_FILE) -> AppSettings:
    parser = read_ini(path)
    sec = parser["SERVICE"] if parser.has_section("SERVICE") else {}
    get = (lambda k, d: sec.get(k, d)) if sec else (lambda k, d: d)
    s = AppSettings(config_file=Path(path))
    s.mode = str(get("mode", "live")).strip().lower()
    s.indices = validate_indices(v for v in str(get("indices", ",".join(SUPPORTED_INDICES))).split(",") if v.strip())
    s.loop_seconds = max(10, int(float(get("loop_seconds", 30))))
    s.requests_per_second = max(0.5, float(get("requests_per_second", 5.0)))
    s.use_websocket = str(get("use_websocket", "true")).strip().lower() in ("1", "true", "yes", "on")
    s.paper_trading = str(get("paper_trading", "true")).strip().lower() in ("1", "true", "yes", "on")
    s.events_file = _project_path(str(get("events_file", "config/events.csv")))
    s.kill_switch_file = _project_path(str(get("kill_switch_file", "KILL_SWITCH")))
    s.notify = dict(parser["NOTIFY"]) if parser.has_section("NOTIFY") else {}
    s.v2_enabled = parser.getboolean("SCORE_V2", "enabled", fallback=True)
    s.v2_push = parser.getboolean("SCORE_V2", "push", fallback=False)
    s.ml_enabled = parser.getboolean("ML", "enabled", fallback=True)
    s.ml_model_file = _project_path(parser.get("ML", "model_file", fallback="data/prebreakout_ml_model.json"))
    s.ml_allow_block = parser.getboolean("ML", "allow_block", fallback=True)
    s.v3_enabled = parser.getboolean("SCORE_V3", "enabled", fallback=True)
    s.v3_push = parser.getboolean("SCORE_V3", "push", fallback=False)
    if parser.has_section("SCORE_V3"):
        s.v3_overrides = {k: v for k, v in parser["SCORE_V3"].items() if k not in ("enabled", "push")}
        unknown = [k for k in s.v3_overrides if k not in PBConfig.__dataclass_fields__]
        if unknown:
            raise ValueError(f"config/scanner.ini: [SCORE_V3] unknown setting(s): {', '.join(unknown)}")
    s.execution = dict(parser["EXECUTION"]) if parser.has_section("EXECUTION") else {}
    s.dashboard_enabled = parser.getboolean("DASHBOARD", "enabled", fallback=True)
    s.dashboard_host = parser.get("DASHBOARD", "host", fallback="127.0.0.1").strip()
    s.dashboard_port = parser.getint("DASHBOARD", "port", fallback=8502)
    s.costs = {k: float(v) for k, v in parser["COSTS"].items()} if parser.has_section("COSTS") else {}
    if s.mode not in ("live", "mock"):
        raise ValueError("config/scanner.ini: [SERVICE] mode must be 'live' or 'mock'")
    return s
