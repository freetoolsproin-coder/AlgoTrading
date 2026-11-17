"""
One-time migration of the old multi-program folder to the new single-application layout.

    python -m src.tools.migrate_layout            show what would happen (changes nothing)
    python -m src.tools.migrate_layout --apply    do it (old files go into legacy/)
    python -m src.tools.migrate_layout --apply --delete
                                                  do it and permanently delete the old files instead

What it does (without --delete it never deletes anything):
  1. Secrets: copies UPSTOX_API_KEY / UPSTOX_API_SECRET / UPSTOX_REDIRECT_URI from config.ini [UPSTOX] and
     TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from config.ini [TELEGRAM_SETTINGS] into .env, only for keys
     .env does not have yet. Values are never printed.
  2. History: moves prebreakout.db (+ -wal / -shm) and prebreakout_ml_model.json into data/, and copies
     the NIFTY, SENSEX and India VIX 1-minute history files from orb_data/ into data/history/ (backtest).
  3. VS Code: installs the run configurations from ide/vscode/ into .vscode/ (existing files are kept).
  4. Legacy: moves every other old file and folder in the project root into legacy/ - the Slope engine,
     data hub, tick streamer, market analysis, swing, ORB, regime, stock / F&O breakout and stock options
     programs, BANKNIFTY / MIDCPNIFTY state, the phone control server, every .bat launcher (start_all.bat
     included), old logs, databases and backups. Nothing in legacy/ is used by the new application.
     Reports in output/ written by those retired programs go into legacy/output/.
  5. --delete: steps 1-3 run first; if any of them fails nothing is deleted. Then every item from step 4
     (and an existing legacy/ folder) is permanently deleted instead of moved. data/, output/ files of the
     scanner, movement service and v3 backtest, logs/, .env, .venv and the code are never touched.

Stop every old program first (close the windows start_all.bat opened): Windows cannot move open files.
Files that cannot be moved are reported and left where they are; run the command again afterwards.
"""
from __future__ import annotations

import argparse
import configparser
import shutil
import sys
from pathlib import Path

from src.config.settings import DATA_DIR, ENV_FILE, HISTORY_DIR, PROJECT_ROOT, read_env_file
from src.tools import setup_vscode

KEEP = {"src", "config", "tests", "ide", ".vscode", ".venv", "venv", "data", "output", "logs", "legacy", ".git", ".idea",
        ".env", ".env.example", ".gitignore", "requirements.txt", "README.md", "upstox_token.json", "KILL_SWITCH",
        "pyproject.toml"}
TO_DATA = ["prebreakout.db", "prebreakout.db-wal", "prebreakout.db-shm", "prebreakout_ml_model.json"]
HISTORY_PREFIXES = ("NSE_INDEX_Nifty_50_1m_", "BSE_INDEX_SENSEX_1m_", "NSE_INDEX_India_VIX_1m_")
# Reports in output/ written by the retired programs (the scanner, movement service and v3 backtest files stay).
LEGACY_OUTPUT = ["fno_breakout_backtest_report.html", "market_analysis.html", "market_analysis.json",
                 "orb_backtest_all.log", "orb_backtest_report.html", "orb_tests.txt", "regime_dashboard.html",
                 "slope_global_dashboard.html", "swing_backtest_report.html", "swing_dashboard.html",
                 "trade_journal.html"]
SECRET_MAP = [("UPSTOX", "API_KEY", "UPSTOX_API_KEY"), ("UPSTOX", "API_SECRET", "UPSTOX_API_SECRET"),
              ("UPSTOX", "REDIRECT_URI", "UPSTOX_REDIRECT_URI"),
              ("TELEGRAM_SETTINGS", "telegram_token", "TELEGRAM_BOT_TOKEN"),
              ("TELEGRAM_SETTINGS", "telegram_chat_id", "TELEGRAM_CHAT_ID")]


def plan_secrets(root: Path, env_file: Path) -> list:
    ini = root / "config.ini"
    if not ini.exists():
        return []
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";", "#"))
    parser.optionxform = str
    try:
        parser.read(ini, encoding="utf-8")
    except configparser.Error:
        return []
    have = read_env_file(env_file)
    out = []
    for section, key, env_key in SECRET_MAP:
        if have.get(env_key) or not parser.has_section(section):
            continue
        lower = {k.lower(): v for k, v in parser[section].items()}
        value = str(lower.get(key.lower(), "")).strip().strip("'\"")
        if value and value.lower() not in ("none", "null"):
            out.append((env_key, value))
    return out


def _remove(path: Path) -> None:
    """Delete a file or folder; clears the read-only flag Windows puts on some files."""
    import os
    import stat

    def _retry(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if path.is_dir() and not path.is_symlink():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_retry)
        else:
            shutil.rmtree(path, onerror=_retry)
    else:
        try:
            path.unlink()
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()


def plan_moves(root: Path, delete: bool = False) -> list:
    moves = []
    for name in TO_DATA:
        src = root / name
        if src.exists() and not (DATA_DIR / name).exists():
            moves.append(("move", src, DATA_DIR / name))
    orb = root / "orb_data"
    if orb.is_dir():
        for f in sorted(orb.iterdir()):
            if f.is_file() and f.name.startswith(HISTORY_PREFIXES) and not (HISTORY_DIR / f.name).exists():
                moves.append(("copy", f, HISTORY_DIR / f.name))
    taken = {m[1].name for m in moves if m[0] == "move"}
    for item in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if item.name in KEEP or item.name in taken:
            continue
        moves.append(("delete", item, None) if delete else ("move", item, root / "legacy" / item.name))
    for name in LEGACY_OUTPUT:
        item = root / "output" / name
        if item.is_file():
            moves.append(("delete", item, None) if delete else ("move", item, root / "legacy" / "output" / name))
    if delete and (root / "legacy").exists():
        moves.append(("delete", root / "legacy", None))
    return moves


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="perform the migration (default: dry run)")
    p.add_argument("--delete", action="store_true",
                   help="permanently delete the old files instead of moving them into legacy/")
    args = p.parse_args(argv)
    root = PROJECT_ROOT
    secrets = plan_secrets(root, ENV_FILE)
    moves = plan_moves(root, delete=args.delete)
    print(f"Project: {root}")
    print(f"\n.env: {len(secrets)} secret(s) to copy from config.ini: {', '.join(k for k, _ in secrets) or 'none'}")
    print(f"\nVS Code: ide/vscode/*.json -> .vscode/ (existing files kept)")
    print(f"\n{len(moves)} item(s):")
    for kind, src, dst in moves:
        target = "PERMANENTLY DELETED" if dst is None else dst.relative_to(root)
        print(f"  {kind:<6} {src.relative_to(root)}  ->  {target}")
    if not args.apply:
        print("\nDry run: nothing changed. Run again with --apply to do it.")
        return 0
    if secrets:
        existing = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
        with open(ENV_FILE, "a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write("# copied from config.ini by src.tools.migrate_layout\n")
            for key, value in secrets:
                handle.write(f"{key}={value}\n")
        print(f"\nCopied {len(secrets)} secret(s) into .env.")
    failed = []
    save_failed = False                      # a move / copy failed: delete nothing after it
    for kind, src, dst in moves:
        if kind == "delete":
            if save_failed:
                failed.append((src, "skipped: a file that had to be saved first was not saved"))
                continue
            try:
                _remove(src)
            except OSError as error:
                failed.append((src, f"{type(error).__name__}: {error}"))
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                failed.append((src, "already exists in the destination"))
                save_failed = True
                continue
            if kind == "copy":
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))
        except OSError as error:
            failed.append((src, f"{type(error).__name__}: {error}"))
            save_failed = True
    print(f"\nDone: {len(moves) - len(failed)} of {len(moves)} item(s).")
    print("\nVS Code run configurations:")
    setup_vscode.install(root)
    for src, why in failed:
        print(f"  NOT done: {src.relative_to(root)} ({why})")
    if failed:
        print("Close the old programs (start_all.bat windows, editors holding files) and run this again.")
    if args.delete:
        print("\nThe old config.ini held the Upstox secret and Telegram token in plain text: regenerate both "
              "credentials and put the new ones in .env.")
        return 1 if failed else 0
    print("\nlegacy/config.ini still contains the old Upstox secret and Telegram token in plain text: once the "
          "scanner works from .env, delete that file and regenerate both credentials.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
