"""Backtest runs end to end on cached synthetic history for NIFTY and SENSEX; other indices are refused."""
import contextlib
import io
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import history as hist
from src.backtest import runner
from src.config.instruments import VIX_KEY


def write_cache(folder: Path, key: str, start="2026-08-03", days=45, base=24000.0, seed=1):
    rng = np.random.default_rng(seed)
    rows, price = [], base
    for day in pd.bdate_range(start, periods=days):
        t0 = pd.Timestamp(day.date()).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)
        for m in range(375):
            o = price
            c = price * (1 + rng.normal(0, 0.0005))
            rows.append({"datetime": (t0 + pd.Timedelta(minutes=m)).isoformat(), "open": o, "high": max(o, c) * 1.0002,
                         "low": min(o, c) * 0.9998, "close": c, "volume": 0})
            price = c
    frame = pd.DataFrame(rows)
    frame["month"] = frame["datetime"].str[:7]
    for month, part in frame.groupby("month"):
        part.drop(columns="month").to_csv(folder / hist.cache_name(key, date.fromisoformat(month + "-01")), index=False)


class BacktestTests(unittest.TestCase):
    def test_runs_for_nifty_and_sensex(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            write_cache(cache, "NSE_INDEX|Nifty 50")
            write_cache(cache, "BSE_INDEX|SENSEX", base=81000.0, seed=2)
            write_cache(cache, VIX_KEY, start="2026-07-20", days=60, base=14.0, seed=3)
            for name, suffix in (("NIFTY", ""), ("SENSEX", "_SENSEX")):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = runner.main(["--instrument", name.lower(), "--start", "2026-08-03", "--end", "2026-09-30",
                                        "--cache", str(cache), "--no-walk-forward", "--no-exit-check"])
                self.assertEqual(code, 0, out.getvalue()[-1500:])
                report = Path(runner.OUTPUT_DIR) / f"prebreakout_backtest_report{suffix}.html"
                self.assertTrue(report.exists())
                self.assertIn(name, report.read_text(encoding="utf-8"))

    def test_other_instruments_refused(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--instrument", "BANKNIFTY"])
        with self.assertRaises(ValueError):
            hist.index_key("MIDCPNIFTY")

    def test_expiry_rules(self):
        self.assertEqual(hist.expiry_for(date(2026, 10, 26), "NIFTY"), date(2026, 10, 27))    # Tuesday
        self.assertEqual(hist.expiry_for(date(2026, 10, 26), "SENSEX"), date(2026, 10, 29))   # Thursday
        # Tue 20 Oct 2026 is a holiday (Dussehra): that week's NIFTY expiry moves to Monday the 19th
        self.assertEqual(hist.expiry_for(date(2026, 10, 19), "NIFTY"), date(2026, 10, 19))


if __name__ == "__main__":
    unittest.main()
