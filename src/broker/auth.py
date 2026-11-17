"""
Daily Upstox login: exchanges the authorization code for an access token and saves it to
upstox_token.json (git-ignored). The running scanner picks the new token up automatically.

    python -m src.broker.auth            (VS Code: Run and Debug -> "Upstox login (daily token)")

Needs UPSTOX_API_KEY, UPSTOX_API_SECRET and UPSTOX_REDIRECT_URI in .env (or environment variables).
Alternatively set UPSTOX_ACCESS_TOKEN in .env yourself; it takes precedence over the file.
The token is never printed or logged.
"""
from __future__ import annotations

import json
import os
import sys
from urllib.parse import parse_qs, urlencode, urlparse

from src.config.settings import ENV_FILE, TOKEN_FILE, get_secret

LOGIN_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def missing_credentials() -> list:
    return [n for n in ("UPSTOX_API_KEY", "UPSTOX_API_SECRET", "UPSTOX_REDIRECT_URI") if not get_secret(n)]


def login_url() -> str:
    return f"{LOGIN_URL}?" + urlencode({"client_id": get_secret("UPSTOX_API_KEY"),
                                       "redirect_uri": get_secret("UPSTOX_REDIRECT_URI"), "response_type": "code"})


def exchange_code(code: str, session=None) -> dict:
    if session is None:
        import requests
        session = requests
    response = session.post(TOKEN_URL, timeout=30, headers={"accept": "application/json", "Api-Version": "2.0",
                                                            "Content-Type": "application/x-www-form-urlencoded"},
                            data={"code": code, "client_id": get_secret("UPSTOX_API_KEY"),
                                  "client_secret": get_secret("UPSTOX_API_SECRET"),
                                  "redirect_uri": get_secret("UPSTOX_REDIRECT_URI"), "grant_type": "authorization_code"})
    response.raise_for_status()
    data = response.json()
    if not data.get("access_token"):
        raise RuntimeError("Upstox returned no access_token (check the API key / secret / redirect URI).")
    return data


def save_token(data: dict):
    tmp = TOKEN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, TOKEN_FILE)  # atomic: a running scanner never reads a half-written file


def main(argv=None) -> int:
    missing = missing_credentials()
    if missing:
        print(f"Missing in {ENV_FILE.name}: {', '.join(missing)}.\n"
              f"Copy .env.example to .env and fill them in (Upstox developer console -> your app).")
        return 1
    print("Upstox login\n1. Open this URL in your browser and log in:\n")
    print("   " + login_url() + "\n")
    redirected = input("2. Paste the full URL you were redirected to, then press Enter:\n> ").strip()
    code = parse_qs(urlparse(redirected).query).get("code", [None])[0]
    if not code:
        print("No 'code' parameter in that URL. Run this again and paste the whole redirected address.")
        return 1
    try:
        save_token(exchange_code(code))
    except Exception as error:
        print(f"Token exchange failed: {type(error).__name__}: {error}")
        return 1
    print(f"Saved the access token to {TOKEN_FILE.name}. A running scanner picks it up automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
