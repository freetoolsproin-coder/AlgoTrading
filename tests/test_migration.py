"""The one-time layout migration: moves (or, with --delete, removes) old files, never prints secrets."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.tools import migrate_layout as mig


class MigrationTests(unittest.TestCase):
    def test_dry_run_then_apply(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("src", "config", ".venv", "output", "ide"):
                (root / name).mkdir()
            (root / "ide" / "vscode").mkdir()
            (root / "ide" / "vscode" / "launch.json").write_text("{}", encoding="utf-8")
            (root / "config.ini").write_text("[UPSTOX]\nAPI_KEY = key-123\nAPI_SECRET = sec-456\n"
                                             "REDIRECT_URI = https://x\n[TELEGRAM_SETTINGS]\ntelegram_token = 999:abc\n"
                                             "telegram_chat_id = 42\n", encoding="utf-8")
            (root / ".env").write_text("CONTROL_TOKEN=x\nUPSTOX_API_KEY=already\n", encoding="utf-8")
            (root / "prebreakout.db").write_text("db", encoding="utf-8")
            (root / "start_all.bat").write_text("@echo off", encoding="utf-8")
            (root / "Slope_Acceleration_Engine.py").write_text("", encoding="utf-8")
            (root / "orb_data").mkdir()
            (root / "orb_data" / "NSE_INDEX_Nifty_50_1m_2026-01.csv").write_text("a", encoding="utf-8")
            (root / "orb_data" / "NSE_INDEX_Nifty_Bank_1m_2026-01.csv").write_text("b", encoding="utf-8")
            patches = [mock.patch.object(mig, "PROJECT_ROOT", root), mock.patch.object(mig, "ENV_FILE", root / ".env"),
                       mock.patch.object(mig, "DATA_DIR", root / "data"),
                       mock.patch.object(mig, "HISTORY_DIR", root / "data" / "history")]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(mig.main([]), 0)
                self.assertTrue((root / "start_all.bat").exists())        # dry run changed nothing
                with contextlib.redirect_stdout(out):
                    self.assertEqual(mig.main(["--apply"]), 0)
            text = out.getvalue()
            for secret in ("sec-456", "999:abc", "key-123"):
                self.assertNotIn(secret, text)
            env = (root / ".env").read_text(encoding="utf-8")
            self.assertIn("UPSTOX_API_KEY=already", env)                  # existing value kept
            self.assertNotIn("UPSTOX_API_KEY=key-123", env)
            self.assertIn("UPSTOX_API_SECRET=sec-456", env)
            self.assertIn("TELEGRAM_BOT_TOKEN=999:abc", env)
            self.assertIn("TELEGRAM_CHAT_ID=42", env)
            self.assertTrue((root / "data" / "prebreakout.db").exists())
            self.assertTrue((root / "data" / "history" / "NSE_INDEX_Nifty_50_1m_2026-01.csv").exists())
            self.assertFalse((root / "data" / "history" / "NSE_INDEX_Nifty_Bank_1m_2026-01.csv").exists())
            for name in ("start_all.bat", "Slope_Acceleration_Engine.py", "config.ini", "orb_data"):
                self.assertFalse((root / name).exists(), name)
                self.assertTrue((root / "legacy" / name).exists(), name)
            for name in ("src", "config", ".venv", "output", ".env", "ide"):
                self.assertTrue((root / name).exists(), name)
            self.assertTrue((root / ".vscode" / "launch.json").exists())   # VS Code configs installed


    def test_delete_saves_first_then_removes_old_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("src", "config", ".venv", "output", "ide", "tests"):
                (root / name).mkdir()
            (root / "ide" / "vscode").mkdir()
            (root / "config.ini").write_text("[UPSTOX]\nAPI_SECRET = sec-456\n", encoding="utf-8")
            (root / ".env").write_text("CONTROL_TOKEN=x\n", encoding="utf-8")
            (root / "prebreakout_ml_model.json").write_text("{}", encoding="utf-8")
            (root / "start_all.bat").write_text("@echo off", encoding="utf-8")
            (root / "backup_2026-10-08").mkdir()
            (root / "backup_2026-10-08" / "old.py").write_text("", encoding="utf-8")
            (root / "orb_data").mkdir()
            (root / "orb_data" / "BSE_INDEX_SENSEX_1m_2026-01.csv").write_text("a", encoding="utf-8")
            (root / "output" / "swing_dashboard.html").write_text("old", encoding="utf-8")
            (root / "output" / "prebreakout_alerts.csv").write_text("keep", encoding="utf-8")
            (root / "legacy").mkdir()
            (root / "legacy" / "x.py").write_text("", encoding="utf-8")
            patches = [mock.patch.object(mig, "PROJECT_ROOT", root), mock.patch.object(mig, "ENV_FILE", root / ".env"),
                       mock.patch.object(mig, "DATA_DIR", root / "data"),
                       mock.patch.object(mig, "HISTORY_DIR", root / "data" / "history")]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(mig.main(["--delete"]), 0)       # dry run
                self.assertTrue((root / "start_all.bat").exists())
                with contextlib.redirect_stdout(out):
                    self.assertEqual(mig.main(["--apply", "--delete"]), 0)
            self.assertNotIn("sec-456", out.getvalue())
            self.assertIn("UPSTOX_API_SECRET=sec-456", (root / ".env").read_text(encoding="utf-8"))
            self.assertTrue((root / "data" / "prebreakout_ml_model.json").exists())
            self.assertTrue((root / "data" / "history" / "BSE_INDEX_SENSEX_1m_2026-01.csv").exists())
            for name in ("start_all.bat", "config.ini", "orb_data", "backup_2026-10-08", "legacy",
                         "output/swing_dashboard.html"):
                self.assertFalse((root / name).exists(), name)
            for name in ("src", "config", ".venv", "tests", ".env", "ide", "output/prebreakout_alerts.csv"):
                self.assertTrue((root / name).exists(), name)

    def test_delete_skipped_when_a_save_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "orb_data").mkdir()
            (root / "orb_data" / "NSE_INDEX_Nifty_50_1m_2026-01.csv").write_text("a", encoding="utf-8")
            (root / "start_all.bat").write_text("@echo off", encoding="utf-8")
            patches = [mock.patch.object(mig, "PROJECT_ROOT", root), mock.patch.object(mig, "ENV_FILE", root / ".env"),
                       mock.patch.object(mig, "DATA_DIR", root / "data"),
                       mock.patch.object(mig, "HISTORY_DIR", root / "data" / "history"),
                       mock.patch.object(mig.shutil, "copy2", side_effect=OSError("disk full"))]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(mig.main(["--apply", "--delete"]), 1)
            self.assertTrue((root / "orb_data").exists())
            self.assertTrue((root / "start_all.bat").exists())


if __name__ == "__main__":
    unittest.main()
