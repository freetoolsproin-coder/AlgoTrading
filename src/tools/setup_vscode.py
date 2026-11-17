"""
Install the VS Code run configurations: copies ide/vscode/*.json into .vscode/ (launch.json = F5 targets,
tasks.json, settings.json = .venv interpreter + test discovery). Existing files are kept unless --force.

    python -m src.tools.setup_vscode            (also done by: python -m src.tools.migrate_layout --apply)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.config.settings import PROJECT_ROOT

TEMPLATES = PROJECT_ROOT / "ide" / "vscode"


def install(root: Path = PROJECT_ROOT, force: bool = False) -> list:
    target = root / ".vscode"
    target.mkdir(exist_ok=True)
    done = []
    for src in sorted((root / "ide" / "vscode").glob("*.json")):
        dst = target / src.name
        if dst.exists() and not force:
            print(f"  kept existing .vscode/{src.name} (use --force to replace it)")
            continue
        shutil.copy2(src, dst)
        done.append(src.name)
        print(f"  installed .vscode/{src.name}")
    return done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.tools.setup_vscode", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="replace existing .vscode files")
    install(force=p.parse_args(argv).force)
    print("Open Run and Debug (Ctrl+Shift+D), choose 'Scanner (live)' or 'Scanner (mock data, no credentials)', press F5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
