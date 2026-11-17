"""Password hashing (PBKDF2-SHA256), TOTP (RFC 6238), CSRF tokens and password policy. Standard library only."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from typing import Optional, Tuple

PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_LENGTH = 12


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_b64, dk_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iterations))
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except (ValueError, TypeError):
        return False


def password_problems(password: str, username: str = "") -> list:
    problems = []
    if len(password or "") < MIN_PASSWORD_LENGTH:
        problems.append(f"Use at least {MIN_PASSWORD_LENGTH} characters.")
    classes = sum(any(f(c) for c in password or "") for f in (str.islower, str.isupper, str.isdigit,
                                                              lambda c: not c.isalnum()))
    if classes < 3:
        problems.append("Mix at least three of: lower case, upper case, digits, symbols.")
    if username and username.lower() in (password or "").lower():
        problems.append("The password must not contain the username.")
    return problems


# ----------------------------------------------------------------------------- TOTP
def new_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _b32(secret: str) -> bytes:
    s = secret.strip().replace(" ", "").upper()
    return base64.b32decode(s + "=" * (-len(s) % 8))


def totp(secret: str, at: Optional[float] = None, step: int = 30, digits: int = 6, digest=hashlib.sha1) -> str:
    counter = int((time.time() if at is None else at) // step)
    mac = hmac.new(_b32(secret), struct.pack(">Q", counter), digest).digest()
    offset = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


def verify_totp(secret: str, code: str, at: Optional[float] = None, window: int = 1) -> bool:
    code = (code or "").strip().replace(" ", "")
    if not code.isdigit() or len(code) != 6 or not secret:
        return False
    now = time.time() if at is None else at
    return any(hmac.compare_digest(totp(secret, now + k * 30), code) for k in range(-window, window + 1))


def totp_uri(secret: str, username: str, issuer: str = "Trading Console") -> str:
    from urllib.parse import quote
    return f"otpauth://totp/{quote(issuer)}:{quote(username)}?secret={secret}&issuer={quote(issuer)}"


# ----------------------------------------------------------------------------- CSRF
def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_ok(session_token: Optional[str], form_token: Optional[str]) -> bool:
    return bool(session_token) and bool(form_token) and hmac.compare_digest(str(session_token), str(form_token))


def split_username(value: str) -> Tuple[bool, str]:
    v = (value or "").strip().lower()
    ok = 3 <= len(v) <= 32 and all(c.isalnum() or c in "._-" for c in v)
    return ok, v
