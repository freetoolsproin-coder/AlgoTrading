"""
Console settings: [CONSOLE] in config/scanner.ini (all optional) and the console's own runtime paths.

    [CONSOLE]
    host = 127.0.0.1              ; keep it on this computer; put a TLS reverse proxy in front for remote use
    port = 8510
    session_minutes = 30          ; idle timeout
    secure_cookies = false        ; true when served over HTTPS
    max_login_failures = 5        ; then the account is locked for lockout_minutes
    lockout_minutes = 15
    broker_cache_seconds = 15     ; how long broker read-only answers are reused
    require_2fa_for_admin = false
    ai_enabled = true             ; AI assistant (Claude API, needs ANTHROPIC_API_KEY)
    ai_model = claude-sonnet-5-5
    ai_max_tokens = 1500
    ai_requests_per_hour = 30     ; per user

The console never places, modifies or cancels orders. It reads the scanner's databases read-only, reads the
broker with GET requests only, and its only write actions on the trading system are the kill switch file
(engage / release) and disarming live trading for today (the safe direction). Arming stays a terminal-only step
(python -m src.execution.live_gate --arm) with a typed confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings


@dataclass
class ConsoleConfig:
    host: str = "127.0.0.1"
    port: int = 8510
    session_minutes: int = 30
    secure_cookies: bool = False
    max_login_failures: int = 5
    lockout_minutes: int = 15
    broker_cache_seconds: int = 15
    require_2fa_for_admin: bool = False
    # AI assistant (Claude API); needs ANTHROPIC_API_KEY in the vault / .env / environment
    ai_enabled: bool = True
    ai_model: str = "claude-sonnet-5-5"
    ai_max_tokens: int = 1500
    ai_requests_per_hour: int = 30
    # paths (overridable in tests)
    data_dir: Path = field(default_factory=lambda: settings.DATA_DIR)
    output_dir: Path = field(default_factory=lambda: settings.OUTPUT_DIR)
    log_dir: Path = field(default_factory=lambda: settings.LOG_DIR)
    config_file: Path = field(default_factory=lambda: settings.CONFIG_FILE)
    kill_switch_file: Path = field(default_factory=lambda: settings.PROJECT_ROOT / "KILL_SWITCH")
    token_file: Path = field(default_factory=lambda: settings.TOKEN_FILE)

    @property
    def console_dir(self) -> Path:
        return Path(self.data_dir) / "console"

    @property
    def console_db(self) -> Path:
        return self.console_dir / "console.db"

    @property
    def vault_file(self) -> Path:
        return self.console_dir / "vault.json"

    @property
    def scanner_db(self) -> Path:
        return Path(self.data_dir) / "prebreakout.db"

    @property
    def execution_db(self) -> Path:
        return Path(self.data_dir) / "execution.db"

    @property
    def movement_db(self) -> Path:
        return Path(self.data_dir) / "movement.db"

    @property
    def ml_model_file(self) -> Path:
        return Path(self.data_dir) / "prebreakout_ml_model.json"

    @property
    def state_file(self) -> Path:
        return Path(self.output_dir) / "prebreakout_state.json"

    @property
    def arm_file(self) -> Path:
        return Path(self.data_dir) / "LIVE_ARMED"


def _bool(v, default):
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def load_console_config(path=None) -> ConsoleConfig:
    cfg = ConsoleConfig()
    path = Path(path) if path else settings.CONFIG_FILE
    cfg.config_file = path
    try:
        parser = settings.read_ini(path)
    except FileNotFoundError:
        return cfg
    try:
        app = settings.load_app_settings(path)
        cfg.kill_switch_file = app.kill_switch_file
    except Exception:   # noqa: BLE001 - an invalid scanner setting must not stop the console
        pass
    if not parser.has_section("CONSOLE"):
        return cfg
    s = parser["CONSOLE"]
    cfg.host = s.get("host", cfg.host).strip()
    cfg.port = int(s.get("port", cfg.port))
    cfg.session_minutes = max(5, int(s.get("session_minutes", cfg.session_minutes)))
    cfg.secure_cookies = _bool(s.get("secure_cookies"), cfg.secure_cookies)
    cfg.max_login_failures = max(3, int(s.get("max_login_failures", cfg.max_login_failures)))
    cfg.lockout_minutes = max(1, int(s.get("lockout_minutes", cfg.lockout_minutes)))
    cfg.broker_cache_seconds = max(5, int(s.get("broker_cache_seconds", cfg.broker_cache_seconds)))
    cfg.require_2fa_for_admin = _bool(s.get("require_2fa_for_admin"), cfg.require_2fa_for_admin)
    cfg.ai_enabled = _bool(s.get("ai_enabled"), cfg.ai_enabled)
    cfg.ai_model = (s.get("ai_model") or cfg.ai_model).strip()
    cfg.ai_max_tokens = min(8000, max(256, int(s.get("ai_max_tokens", cfg.ai_max_tokens))))
    cfg.ai_requests_per_hour = max(1, int(s.get("ai_requests_per_hour", cfg.ai_requests_per_hour)))
    return cfg
