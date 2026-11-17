"""
Encrypted credential vault (data/console/vault.json).

Each value is encrypted with Fernet (AES-128-CBC + HMAC-SHA256, from the `cryptography` package). The vault key
comes from, in order:
  1. CONSOLE_MASTER_KEY (environment variable) - a Fernet key; best for servers.
  2. data/console/vault.key - created on first use. On Windows it is protected with DPAPI (only your Windows
     user on this computer can decrypt it); elsewhere it is a file readable only by its owner.

Values are never returned to the browser: the console shows only whether a key is set, its length, the last
four characters and when it was changed. src.config.settings.get_secret falls back to this vault after the
environment and .env, so moving a secret out of .env into the vault keeps the scanner working.
"""
from __future__ import annotations

import base64
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

IST = timezone(timedelta(hours=5, minutes=30))

# Keys the console lets you manage (the same names the scanner reads).
MANAGED_KEYS = {
    "UPSTOX_API_KEY": "Upstox API key (developer console)",
    "UPSTOX_API_SECRET": "Upstox API secret",
    "UPSTOX_REDIRECT_URI": "Upstox redirect URI",
    "TELEGRAM_BOT_TOKEN": "Telegram bot token (alerts)",
    "TELEGRAM_CHAT_ID": "Telegram chat id",
    "MOVEMENT_WEBHOOK_URL": "Webhook for movement triggers",
    "MOVEMENT_API_TOKEN": "Movement service API token",
    "ANTHROPIC_API_KEY": "Anthropic API key (console AI assistant)",
}
# Never stored here: the live-trading confirmation must stay a deliberate line in .env.
FORBIDDEN_KEYS = {"LIVE_TRADING_CONFIRM"}


class VaultError(Exception):
    pass


def _fernet_cls():
    try:
        from cryptography.fernet import Fernet, InvalidToken
        return Fernet, InvalidToken
    except ImportError as error:   # pragma: no cover - cryptography is in requirements.txt
        raise VaultError("The 'cryptography' package is required for the credential vault: "
                         "python -m pip install cryptography") from error


# ----------------------------------------------------------------------------- Windows DPAPI
def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = BLOB()
    fn = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    if protect:
        ok = fn(ctypes.byref(blob_in), "trading-console", None, None, None, 0, ctypes.byref(blob_out))
    else:
        ok = fn(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out))
    if not ok:
        raise VaultError("Windows DPAPI could not " + ("protect" if protect else "unprotect") + " the vault key")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _load_or_create_key(key_file: Path) -> bytes:
    env = os.environ.get("CONSOLE_MASTER_KEY", "").strip()
    if env:
        return env.encode()
    Fernet, _ = _fernet_cls()
    key_file = Path(key_file)
    if key_file.exists():
        raw = json.loads(key_file.read_text(encoding="utf-8"))
        blob = base64.b64decode(raw["key"])
        return _dpapi(blob, protect=False) if raw.get("protection") == "dpapi" else blob
    key = Fernet.generate_key()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        record = {"protection": "dpapi", "key": base64.b64encode(_dpapi(key, protect=True)).decode()}
    else:
        record = {"protection": "file", "key": base64.b64encode(key).decode()}
    tmp = key_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(key_file)
    return key


class Vault:
    def __init__(self, path: Path, key_file: Optional[Path] = None):
        self.path = Path(path)
        self.key_file = Path(key_file) if key_file else self.path.with_name("vault.key")
        self._lock = threading.Lock()
        self._fernet = None

    def _f(self):
        if self._fernet is None:
            Fernet, _ = _fernet_cls()
            try:
                self._fernet = Fernet(_load_or_create_key(self.key_file))
            except (ValueError, KeyError) as error:
                raise VaultError(f"vault key is invalid: {type(error).__name__}") from error
        return self._fernet

    def _read(self) -> Dict[str, dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get("entries", {})
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            raise VaultError(f"vault file unreadable: {type(error).__name__}") from error

    def _write(self, entries: Dict[str, dict]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "entries": entries}, indent=1), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(self.path)

    @staticmethod
    def _check_name(name: str):
        if name in FORBIDDEN_KEYS:
            raise VaultError(f"{name} cannot be stored in the vault; it must be a deliberate line in .env")
        if name not in MANAGED_KEYS:
            raise VaultError(f"{name} is not a credential the console manages")

    def set(self, name: str, value: str, actor: str = ""):
        self._check_name(name)
        value = (value or "").strip()
        if not value or len(value) > 4096 or any(c in value for c in "\r\n\x00"):
            raise VaultError("value must be 1-4096 characters on one line")
        with self._lock:
            entries = self._read()
            entries[name] = {"ct": self._f().encrypt(value.encode()).decode(), "len": len(value),
                             "tail": value[-4:] if len(value) >= 12 else "",
                             "updated": datetime.now(IST).isoformat(timespec="seconds"), "by": actor}
            self._write(entries)

    def delete(self, name: str) -> bool:
        with self._lock:
            entries = self._read()
            if name not in entries:
                return False
            del entries[name]
            self._write(entries)
            return True

    def get(self, name: str) -> Optional[str]:
        entry = self._read().get(name)
        if not entry:
            return None
        _, InvalidToken = _fernet_cls()
        try:
            return self._f().decrypt(entry["ct"].encode()).decode()
        except InvalidToken as error:
            raise VaultError(f"{name}: ciphertext failed authentication (tampered or wrong key)") from error

    def describe(self) -> list:
        entries = self._read()
        out = []
        for name, label in MANAGED_KEYS.items():
            e = entries.get(name)
            out.append({"name": name, "label": label, "set": bool(e), "length": e.get("len") if e else None,
                        "tail": e.get("tail") if e else "", "updated": e.get("updated") if e else None,
                        "by": e.get("by") if e else None})
        return out


def default_vault() -> Vault:
    from src.config import settings
    return Vault(settings.DATA_DIR / "console" / "vault.json")


def vault_secret(name: str) -> Optional[str]:
    """Used by settings.get_secret as the last fallback. Never raises; returns None when unavailable."""
    if name not in MANAGED_KEYS:
        return None
    try:
        v = default_vault()
        if not v.path.exists():
            return None
        return v.get(name)
    except Exception:   # noqa: BLE001
        return None
