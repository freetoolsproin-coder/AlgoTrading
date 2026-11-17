"""
Instrument scope: only NIFTY and SENSEX index options (CE / PE) exist anywhere in the application.

    python -m unittest tests.test_instruments
"""
import re
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.broker.client import Instruments, keep_master_row
from src.broker.mock_data import MOCK_SPECS, MockProvider
from src.config import instruments as inst
from src.config import settings
from src.options.selection import select_option
from src.scanner.strategies import UnderlyingPlan

ROOT = settings.PROJECT_ROOT
REMOVED = ["BANKNIFTY", "MIDCPNIFTY", "MIDCAPNIFTY", "FINNIFTY", "NIFTY BANK", "NIFTY MID SELECT"]


class ScopeTests(unittest.TestCase):
    def test_only_nifty_and_sensex_are_configured(self):
        self.assertEqual(inst.SUPPORTED_INDICES, ("NIFTY", "SENSEX"))
        self.assertEqual(set(inst.INSTRUMENTS), {"NIFTY", "SENSEX"})
        self.assertEqual(settings.PBConfig.from_ini(settings.CONFIG_FILE).indices, ("NIFTY", "SENSEX"))
        self.assertEqual(settings.load_app_settings().indices, ("NIFTY", "SENSEX"))
        self.assertEqual(set(MOCK_SPECS), {"NIFTY", "SENSEX"})

    def test_banknifty_midcap_and_stocks_are_rejected(self):
        for name in ("BANKNIFTY", "MIDCPNIFTY", "MIDCAPNIFTY", "FINNIFTY", "RELIANCE", "HDFCBANK", ""):
            with self.assertRaises(inst.UnsupportedInstrumentError, msg=name):
                inst.get_instrument(name)
        with self.assertRaises(ValueError):
            inst.validate_indices(["NIFTY", "BANKNIFTY"])
        with self.assertRaises(ValueError):
            settings.PBConfig().replace(indices=("NIFTY", "MIDCPNIFTY")).validate()

    def test_config_file_with_banknifty_is_refused(self):
        text = settings.CONFIG_FILE.read_text(encoding="utf-8").replace(
            "indices = NIFTY, SENSEX", "indices = NIFTY, SENSEX, BANKNIFTY")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "scanner.ini"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                settings.load_app_settings(path)
            with self.assertRaises(ValueError):
                settings.PBConfig.from_ini(path)

    def test_lowercase_and_duplicates_are_normalised(self):
        self.assertEqual(inst.validate_indices(["nifty", " sensex", "NIFTY"]), ("NIFTY", "SENSEX"))

    def test_option_types(self):
        self.assertEqual(inst.OPTION_TYPES, ("CE", "PE"))
        self.assertEqual(inst.validate_option_type("ce"), "CE")
        with self.assertRaises(ValueError):
            inst.validate_option_type("FUT")

    def test_index_is_not_an_option_and_segments_are_right(self):
        self.assertEqual(inst.NIFTY.fo_segment, "NSE_FO")
        self.assertEqual(inst.SENSEX.fo_segment, "BSE_FO")
        self.assertTrue(inst.NIFTY.index_key.startswith("NSE_INDEX|"))
        self.assertTrue(inst.SENSEX.index_key.startswith("BSE_INDEX|"))
        self.assertFalse(inst.is_supported_option_row({"instrument_type": "INDEX", "name": "NIFTY",
                                                       "exchange": "NSE_INDEX"}))

    def test_mock_provider_refuses_other_indices(self):
        cfg = settings.PBConfig()
        with self.assertRaises(ValueError):
            MockProvider(cfg, ["NIFTY", "BANKNIFTY"], session=date(2026, 10, 16))


class InstrumentMasterTests(unittest.TestCase):
    ROWS = [
        {"instrument_type": "INDEX", "exchange": "NSE_INDEX", "instrument_key": "NSE_INDEX|Nifty 50", "name": "Nifty 50",
         "tradingsymbol": "NIFTY"},
        {"instrument_type": "INDEX", "exchange": "BSE_INDEX", "instrument_key": "BSE_INDEX|SENSEX", "name": "SENSEX",
         "tradingsymbol": "SENSEX"},
        {"instrument_type": "OPTIDX", "exchange": "NSE_FO", "instrument_key": "NSE_FO|111", "name": "NIFTY",
         "expiry": "2026-10-27", "lot_size": 65},
        {"instrument_type": "OPTIDX", "exchange": "BSE_FO", "instrument_key": "BSE_FO|222", "name": "SENSEX",
         "expiry": "2026-10-29", "lot_size": 20},
        {"instrument_type": "FUTIDX", "exchange": "NSE_FO", "instrument_key": "NSE_FO|333", "name": "NIFTY",
         "expiry": "2026-10-27", "lot_size": 65},
        # everything below must be dropped
        {"instrument_type": "OPTIDX", "exchange": "NSE_FO", "instrument_key": "NSE_FO|444", "name": "BANKNIFTY",
         "expiry": "2026-10-28", "lot_size": 30},
        {"instrument_type": "OPTIDX", "exchange": "NSE_FO", "instrument_key": "NSE_FO|555", "name": "MIDCPNIFTY",
         "expiry": "2026-10-28", "lot_size": 120},
        {"instrument_type": "OPTSTK", "exchange": "NSE_FO", "instrument_key": "NSE_FO|666", "name": "RELIANCE",
         "expiry": "2026-10-28", "lot_size": 500},
        {"instrument_type": "FUTSTK", "exchange": "NSE_FO", "instrument_key": "NSE_FO|777", "name": "RELIANCE",
         "expiry": "2026-10-28", "lot_size": 500},
        {"instrument_type": "OPTIDX", "exchange": "BSE_FO", "instrument_key": "BSE_FO|888", "name": "NIFTY",
         "expiry": "2026-10-27", "lot_size": 65},   # wrong segment for NIFTY
    ]

    def test_master_keeps_only_nifty_sensex_index_fno(self):
        kept = [r for r in self.ROWS if keep_master_row(r)]
        keys = {r["instrument_key"] for r in kept}
        self.assertEqual(keys, {"NSE_INDEX|Nifty 50", "BSE_INDEX|SENSEX", "NSE_FO|111", "BSE_FO|222", "NSE_FO|333"})
        m = Instruments(self.ROWS, date(2026, 10, 23))
        self.assertEqual(set(m.index_options), {"NIFTY", "SENSEX"})
        self.assertEqual(m.option_expiries(inst.NIFTY), ([date(2026, 10, 27)], {date(2026, 10, 27): 65}))
        self.assertEqual(m.option_expiries(inst.SENSEX), ([date(2026, 10, 29)], {date(2026, 10, 29): 20}))
        self.assertEqual(m.near_future(inst.NIFTY), "NSE_FO|333")
        self.assertIsNone(m.near_future(inst.SENSEX))
        self.assertEqual(m.resolve_index(inst.NIFTY), "NSE_INDEX|Nifty 50")
        self.assertEqual(m.resolve_index(inst.SENSEX), "BSE_INDEX|SENSEX")

    def test_expired_contracts_are_dropped(self):
        m = Instruments(self.ROWS, date(2026, 10, 28))
        self.assertEqual(m.option_expiries(inst.NIFTY)[0], [])
        self.assertEqual(m.option_expiries(inst.SENSEX)[0], [date(2026, 10, 29)])


class OptionSelectionScopeTests(unittest.TestCase):
    """NIFTY CE / PE and SENSEX CE / PE can each be selected from a chain; the result is a real contract."""

    def _select(self, name, direction):
        cfg = settings.PBConfig.from_ini(settings.CONFIG_FILE)
        prov = MockProvider(cfg, [name], session=date(2026, 10, 16))
        now = pd.Timestamp("2026-10-16 11:00", tz="Asia/Kolkata")
        snap = prov.snapshot(now).indices[name]
        spot = snap.ltp
        sign = 1 if direction == inst.CE else -1
        risk = 0.002 * spot
        plan = UnderlyingPlan(direction=direction, entry=spot, entry_zone=(spot, spot + sign * risk), stop=spot - sign * risk,
                              t1=spot + sign * 1.5 * risk * 2, t2=spot + sign * 3 * risk * 2, risk_points=risk, next_level=None)
        return select_option(snap.chain, spot, direction, plan, snap.step, snap.lot_size, snap.expiry, now,
                             cfg.replace(min_rr=0.1), lambda s, p, q: 20.0, symbol_prefix=name)

    def test_all_four_combinations(self):
        for name in ("NIFTY", "SENSEX"):
            for direction in inst.OPTION_TYPES:
                best, ranked, note = self._select(name, direction)
                self.assertIsNotNone(best, f"{name} {direction}: {note}")
                self.assertEqual(best.option_type, direction)
                self.assertTrue(best.symbol.startswith(name))
                self.assertTrue(best.symbol.endswith(direction))
                self.assertIn(f"|{name}|", best.instrument_key)      # a chain contract, not the index
                self.assertGreater(best.lot_size, 0)
                self.assertTrue(all(c.option_type == direction for c in ranked))


class NoLeftoverReferencesTests(unittest.TestCase):
    """No removed instrument and no old launcher is referenced by code, config or docs of the application."""

    def _files(self):
        for folder in ("src", "config", "ide"):   # tests mention removed names only to check they are refused
            for path in (ROOT / folder).rglob("*"):
                if path.is_file() and path.suffix in (".py", ".ini", ".csv", ".json", ".md", ".txt") \
                        and "__pycache__" not in path.parts:
                    yield path
        for name in ("README.md", "requirements.txt", ".env.example"):
            if (ROOT / name).exists():
                yield ROOT / name

    def test_removed_instruments_absent(self):
        pattern = re.compile("|".join(re.escape(w) for w in REMOVED), re.IGNORECASE)
        allowed = {ROOT / "src" / "tools" / "migrate_layout.py"}   # lists the old files it moves to legacy/
        hits = []
        for path in self._files():
            # src/fno is the separate multi-index / stock F&O engine: these names are its scope, not leftovers.
            if path in allowed or (ROOT / "src" / "fno") in path.parents or path == ROOT / "config" / "fno.ini":
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith(("#", ";")) \
                        and not any(w in line.lower() for w in ("removed", "not supported", "rejected")):
                    hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:90]}")
        self.assertEqual(hits, [], "\n".join(hits))

    def test_no_process_launching_in_src(self):
        pattern = re.compile(r"\bos\.system\(|subprocess\.|Popen\(|webbrowser\.open|start_all\.bat")
        hits = [f"{p.relative_to(ROOT)}" for p in (ROOT / "src").rglob("*.py")
                if p.name != "migrate_layout.py" and pattern.search(p.read_text(encoding="utf-8"))]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
