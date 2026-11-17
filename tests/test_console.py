"""
Trading console: authentication, CSRF, roles, lockout, TOTP, vault, audit chain, analytics, source separation,
read-only broker access, safety actions and reports.

    python -m unittest tests.test_console
"""
import json
import re
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from src.console import analytics, data as data_mod, security
from src.console.broker_monitor import BrokerMonitor, ReadOnlyViolation, _GetOnlySession
from src.console.config import ConsoleConfig
from src.console.jobs import JobError, build_argv
from src.console.reports import trades_csv
from src.console.store import ConsoleStore
from src.console.vault import Vault, VaultError

PASSWORD = "Corr3ct-Horse-Battery"


def T(net, day="2026-10-01", source="paper", book="scanner-v1", open_=False, gross=None, charges=10.0, t=None):
    return data_mod.Trade(source=source, book=book, book_key=book, id=f"{day}{net}{t}", index="NIFTY", side="CE",
                          symbol="NIFTY X CE", day=day, entry_time=(t or day + "T10:00:00"), exit_time=(t or day + "T10:30:00"),
                          qty=65, entry_price=100, exit_price=100, gross=(net + charges) if gross is None else gross,
                          charges=charges, net=None if open_ else net, open=open_, open_pnl=5.0 if open_ else None)


def make_dbs(root: Path):
    """Databases with the SAME schemas the scanner, movement service and order engine create."""
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(exist_ok=True)
    c = sqlite3.connect(root / "data" / "prebreakout.db")
    c.execute("CREATE TABLE paper_trades (id TEXT PRIMARY KEY, day TEXT, idx TEXT, direction TEXT, symbol TEXT, "
              "entry_time TEXT, exit_time TEXT, entry_price REAL, qty INTEGER, net_pnl REAL, exit_reason TEXT, closed INTEGER, "
              "mock INTEGER, payload TEXT, model TEXT DEFAULT 'v1')")
    rows = [("a", "2026-10-26", 0, 0, -500.0, "v1"), ("b", "2026-10-27", 0, 0, 1500.0, "v1"),
            ("c", "2026-10-27", 0, 0, -200.0, "v3"), ("m", "2026-10-28", 1, 0, 99999.0, "v1"),   # mock: excluded
            ("o", "2026-10-29", 0, 1, None, "v1")]                                               # open
    for pid, day, mock, open_, net, model in rows:
        payload = {"lot_size": 65, "lots": 1, "realised_gross": (net or 0) + 50, "charges": 50.0,
                   "exits": [] if open_ else [{"qty": 65, "price": 110.0}], "open_pnl": 120.0, "model": model}
        c.execute("INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (pid, day, "NIFTY", "CE", "NIFTY 30OCT26 22600 CE", day + "T10:00:00+05:30",
                   None if open_ else day + "T10:30:00+05:30", 100.0, 65, net, "" if open_ else "target 2",
                   0 if open_ else 1, mock, json.dumps(payload).replace('"x"', "NaN"), model))
    c.commit(); c.close()
    c = sqlite3.connect(root / "data" / "movement.db")
    c.execute("CREATE TABLE movement_trades (id TEXT PRIMARY KEY, day TEXT, book TEXT, idx TEXT, side TEXT, open INTEGER, "
              "net REAL, mock INTEGER, payload TEXT)")
    c.execute("INSERT INTO movement_trades VALUES (?,?,?,?,?,?,?,?,?)",
              ("s1", "2026-11-03", "scalp", "SENSEX", "CE", 0, -572.02, 0,
               json.dumps({"qty": 20, "entry_price": 520.1, "exit_price": 494.76, "gross": -506.8, "charges": 65.2,
                           "entry_time": "2026-11-03T14:37:00", "exit_time": "2026-11-03T14:45:00", "exit_reason": "time stop"})))
    c.commit(); c.close()
    c = sqlite3.connect(root / "data" / "execution.db")
    c.execute("CREATE TABLE orders (key TEXT PRIMARY KEY, day TEXT, broker TEXT, payload TEXT)")
    c.execute("CREATE TABLE live_positions (id TEXT PRIMARY KEY, day TEXT, broker TEXT, payload TEXT)")
    for key, broker, side, avg in (("p|entry", "paper", "BUY", 100.0), ("L|entry", "upstox", "BUY", 101.0),
                                   ("L|exit|1", "upstox", "SELL", 111.0)):
        pid = key.split("|")[0]
        c.execute("INSERT INTO orders VALUES (?,?,?,?)", (key, "2026-10-27", broker, json.dumps(
            {"key": key, "position_id": pid, "purpose": key.split("|")[1], "status": "FILLED", "filled": 65, "avg_price": avg,
             "created": "2026-10-27T10:00:00", "updated": "2026-10-27T10:20:00",
             "req": {"side": side, "quantity": 65, "order_type": "LIMIT", "price": avg, "instrument_key": "NSE_FO|1"}})))
    c.execute("INSERT INTO live_positions VALUES (?,?,?,?)", ("p", "2026-10-27", "paper", json.dumps(
        {"position_id": "p", "closed": True, "realised": 999.0, "symbol": "NIFTY X CE"})))
    c.execute("INSERT INTO live_positions VALUES (?,?,?,?)", ("L", "2026-10-27", "upstox", json.dumps(
        {"position_id": "L", "closed": True, "realised": 650.0, "symbol": "NIFTY 30OCT26 22600 CE", "index": "NIFTY",
         "instrument_key": "NSE_FO|1", "entry_avg": 101.0})))
    c.commit(); c.close()
    (root / "output" / "v3_backtest_trades.csv").write_text(
        "set,strategy,day,direction,entry_time,exit_time,strike,expiry,lots,qty,entry_price,exit_reason,gross,charges,net,risk_rupees,r,score,setup_key\n"
        "v1,prebreakout,2023-01-30,PE,2023-01-30T10:45:00+05:30,2023-01-30T11:09:59+05:30,17600,2023-02-02,4,200,139.01,premium stop,-3934.55,93.32,-4027.87,4084.7,-0.986,77.5,2023-01-30|NIFTY|PE|17573\n",
        encoding="utf-8")


def sources(root: Path, cost_fn=None):
    return data_mod.DataSources(root / "data" / "prebreakout.db", root / "data" / "execution.db",
                                root / "data" / "movement.db", root / "output", root / "output" / "state.json", cost_fn=cost_fn)


class SecurityTests(unittest.TestCase):
    def test_password_hash_roundtrip_and_salt(self):
        h1, h2 = security.hash_password(PASSWORD, 1000), security.hash_password(PASSWORD, 1000)
        self.assertNotEqual(h1, h2)
        self.assertTrue(security.verify_password(PASSWORD, h1))
        self.assertFalse(security.verify_password(PASSWORD + "x", h1))
        self.assertFalse(security.verify_password(PASSWORD, "garbage"))

    def test_password_policy(self):
        self.assertTrue(security.password_problems("short"))
        self.assertTrue(security.password_problems("alllowercaseletters"))
        self.assertTrue(security.password_problems("Admin-Passw0rd!", "admin"))
        self.assertEqual(security.password_problems(PASSWORD, "prash"), [])

    def test_totp_rfc6238_vector(self):
        secret = __import__("base64").b32encode(b"12345678901234567890").decode()
        self.assertEqual(security.totp(secret, at=59, digits=8), "94287082")       # RFC 6238 SHA-1 vector
        code = security.totp(secret, at=1_000_000)
        self.assertTrue(security.verify_totp(secret, code, at=1_000_000 + 25))
        self.assertFalse(security.verify_totp(secret, code, at=1_000_000 + 400))
        self.assertFalse(security.verify_totp(secret, "abcdef"))


class AuditTests(unittest.TestCase):
    def test_chain_detects_tampering_and_is_append_only(self):
        with tempfile.TemporaryDirectory() as d:
            s = ConsoleStore(Path(d) / "c.db")
            for i in range(5):
                s.audit("u", "test.action", target=str(i), detail={"i": i})
            self.assertTrue(s.verify_chain()["ok"])
            c = sqlite3.connect(Path(d) / "c.db")
            with self.assertRaises(sqlite3.DatabaseError):
                c.execute("UPDATE audit SET target='x' WHERE id=3")
            with self.assertRaises(sqlite3.DatabaseError):
                c.execute("DELETE FROM audit WHERE id=3")
            # bypass the triggers (as an attacker with file access could) and check detection
            c.execute("DROP TRIGGER audit_no_update")
            c.execute("UPDATE audit SET target='tampered' WHERE id=3")
            c.commit(); c.close()
            r = s.verify_chain()
            self.assertFalse(r["ok"])
            self.assertEqual(r["broken_at"], 3)


class VaultTests(unittest.TestCase):
    def test_roundtrip_never_plaintext_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            v = Vault(Path(d) / "vault.json")
            v.set("UPSTOX_API_SECRET", "super-secret-value-1234", actor="t")
            self.assertEqual(v.get("UPSTOX_API_SECRET"), "super-secret-value-1234")
            self.assertNotIn("super-secret", (Path(d) / "vault.json").read_text())
            desc = {r["name"]: r for r in v.describe()}["UPSTOX_API_SECRET"]
            self.assertEqual((desc["length"], desc["tail"]), (23, "1234"))
            self.assertNotIn("super", json.dumps(v.describe()))
            raw = json.loads((Path(d) / "vault.json").read_text())
            ct = raw["entries"]["UPSTOX_API_SECRET"]["ct"]
            raw["entries"]["UPSTOX_API_SECRET"]["ct"] = ct[:-6] + ("A" if ct[-6] != "A" else "B") + ct[-5:]
            (Path(d) / "vault.json").write_text(json.dumps(raw))
            with self.assertRaises(VaultError):
                Vault(Path(d) / "vault.json").get("UPSTOX_API_SECRET")

    def test_live_confirmation_cannot_be_stored(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(VaultError):
                Vault(Path(d) / "v.json").set("LIVE_TRADING_CONFIRM", "I ACCEPT REAL-MONEY RISK")
            with self.assertRaises(VaultError):
                Vault(Path(d) / "v.json").set("RANDOM_KEY", "x" * 20)


class AnalyticsTests(unittest.TestCase):
    def test_metrics_known_values(self):
        nets = [1000, -500, 2000, -1000, -1000, 500]
        ts = [T(n, day=f"2026-10-0{i + 1}") for i, n in enumerate(nets)]
        m = analytics.compute(ts, capital=100000)
        self.assertEqual(m["trades"], 6)
        self.assertEqual(m["net"], 1000)
        self.assertEqual(m["win_rate"], 50.0)
        self.assertEqual(m["profit_factor"], round(3500 / 2500, 2))
        self.assertEqual(m["charges"], 60)
        self.assertEqual(m["gross"], 1060)
        # equity 101000, 100500, 102500 (peak), 101500, 100500 (trough), 101000 -> DD 2000
        self.assertEqual(m["drawdown"]["amount"], 2000)
        self.assertEqual(m["losing_streak"], 2)
        self.assertIsNone(m["sharpe"])                      # 6 days < 20: no ratio, and it says why
        self.assertIn("20", m["ratio_note"])
        self.assertFalse(m["sample_ok"])
        self.assertIn("too few", m["sample_note"])

    def test_open_positions_excluded_from_closed_stats(self):
        m = analytics.compute([T(100), T(0, open_=True)], 100000)
        self.assertEqual((m["trades"], m["open_positions"], m["open_pnl"]), (1, 1, 5.0))

    def test_no_losses_profit_factor_is_none_not_infinite(self):
        m = analytics.compute([T(100), T(200)], 100000)
        self.assertIsNone(m["profit_factor"])
        self.assertEqual(m["profit_factor_note"], "no losing trades yet")

    def test_empty(self):
        m = analytics.compute([], 100000)
        self.assertEqual(m["trades"], 0)
        self.assertIsNone(m["win_rate"])
        self.assertEqual(m["sample_note"], "no closed trades recorded")

    def test_ratios_and_ci_with_enough_data(self):
        ts = [T(100 if i % 3 else -50, day=f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}") for i in range(60)]
        m = analytics.compute(ts, 100000)
        self.assertIsNotNone(m["sharpe"])
        self.assertIsNotNone(m["var95"])
        self.assertIsNotNone(m["mean_ci"])
        self.assertLessEqual(m["mean_ci"]["low"], m["avg_trade"])
        self.assertGreaterEqual(m["mean_ci"]["high"], m["avg_trade"])


class DataSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_dbs(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sources_are_separated_and_mock_excluded(self):
        from src.execution.costs import make_cost_fn
        s = sources(self.root, make_cost_fn())
        paper, live, bt = s.trades("paper"), s.trades("live"), s.trades("backtest")
        self.assertTrue(all(t.source == "paper" for t in paper))
        self.assertNotIn("m", [t.id for t in paper])                       # mock excluded
        self.assertEqual(sorted(t.book_key for t in paper),
                         ["movement-scalp", "scanner-v1", "scanner-v1", "scanner-v1", "scanner-v3"])
        self.assertEqual([t.id for t in live], ["L"])                      # paper-broker positions are NOT live
        self.assertEqual(live[0].gross, 650.0)
        self.assertGreater(live[0].charges, 0)
        self.assertAlmostEqual(live[0].net, 650.0 - live[0].charges, places=2)
        self.assertEqual(len(bt), 1)
        self.assertEqual(bt[0].net, -4027.87)
        self.assertEqual(bt[0].index, "NIFTY")

    def test_databases_opened_read_only(self):
        c = data_mod.connect_ro(self.root / "data" / "prebreakout.db")
        with self.assertRaises(sqlite3.OperationalError):
            c.execute("DELETE FROM paper_trades")
        c.close()

    def test_missing_files_give_empty_not_errors(self):
        s = sources(self.root / "nowhere")
        self.assertEqual(s.trades(), [])
        self.assertEqual(s.orders(), [])

    def test_csv_export_escapes_formulas(self):
        t = T(10)
        t.exit_reason = "=HYPERLINK(\"http://x\")"
        text = trades_csv([t], ["source: paper"])
        self.assertIn("'=HYPERLINK", text)
        self.assertTrue(text.startswith("# source: paper"))


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code, self._p = status, payload or {"status": "success", "data": {}}

    def json(self):
        return self._p


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kw):
        self.calls.append((method, url))
        if url.endswith("positions"):
            return FakeResponse(200, {"status": "success", "data": [{"trading_symbol": "NIFTY X CE", "quantity": 65, "pnl": 10}]})
        return FakeResponse(200, {"status": "success", "data": {"user_id": "AB1", "equity": {"available_margin": 1000}}})

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, *a, **kw):   # pragma: no cover - must never be reached
        raise AssertionError("POST sent")


class BrokerMonitorTests(unittest.TestCase):
    def test_get_only_guard(self):
        s = _GetOnlySession(FakeSession())
        with self.assertRaises(ReadOnlyViolation):
            s.request("POST", "https://api-hft.upstox.com/v3/order/place")
        with self.assertRaises(ReadOnlyViolation):
            s.post("https://x")
        with self.assertRaises(ReadOnlyViolation):
            s.delete("https://x")

    def test_snapshot_uses_get_only(self):
        with tempfile.TemporaryDirectory() as d:
            tok = Path(d) / "upstox_token.json"
            tok.write_text(json.dumps({"access_token": "t" * 40}))
            fake = FakeSession()
            snap = BrokerMonitor(tok, 15, session=fake).snapshot()
            self.assertTrue(snap["connected"])
            self.assertEqual(snap["positions"]["rows"][0]["quantity"], 65)
            self.assertTrue(fake.calls)
            self.assertTrue(all(m == "GET" for m, _ in fake.calls))
            self.assertFalse(any("order/place" in u or "order/modify" in u or "order/cancel" in u for _, u in fake.calls))

    def test_no_token_reported(self):
        with tempfile.TemporaryDirectory() as d:
            snap = BrokerMonitor(Path(d) / "none.json", 15, session=FakeSession()).snapshot()
            self.assertFalse(snap["connected"])
            self.assertEqual(snap["status"], "no_token")


class JobArgTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(build_argv("v3", "nifty", "2024-01-01"), ["--instrument", "NIFTY", "--start", "2024-01-01"])
        for bad in (("x", "NIFTY"), ("v3", "FOO"), ("v3", "NIFTY", "2024-13-01"), ("v3", "NIFTY", "2024-02-01", "2024-01-01"),
                    ("v3", "NIFTY", "; rm -rf /")):
            with self.assertRaises(JobError):
                build_argv(*bad)


class JobRunnerTests(unittest.TestCase):
    def test_runs_in_thread_captures_output_one_at_a_time(self):
        import threading
        from src.console import jobs
        gate = threading.Event()

        def fake(argv):
            print("hello from backtest", argv)
            gate.wait(5)
            return 0
        saved = jobs.KINDS["v3"]["fn"]
        jobs.KINDS["v3"]["fn"] = fake
        finished = []
        try:
            r = jobs.JobRunner(on_finish=finished.append)
            r.start("v3", "NIFTY", actor="t")
            with self.assertRaises(JobError):
                r.start("v3", "SENSEX")
            gate.set()
            for _ in range(100):
                if finished:
                    break
                time.sleep(0.05)
            j = r.list()[0]
            self.assertEqual(j["status"], "finished")
            self.assertIn("hello from backtest", j["output"])
        finally:
            jobs.KINDS["v3"]["fn"] = saved


class WebTests(unittest.TestCase):
    def setUp(self):
        from src.console.web import create_app
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        make_dbs(root)
        self.cfg = ConsoleConfig(data_dir=root / "data", output_dir=root / "output", log_dir=root / "logs",
                                 kill_switch_file=root / "KILL_SWITCH", token_file=root / "upstox_token.json",
                                 max_login_failures=3)
        self.app = create_app(self.cfg, broker_session=FakeSession(), testing=True)
        self.store = self.app.extensions["console"]["store"]
        self.store.create_user("admin1", security.hash_password(PASSWORD, 1000), "admin")
        self.store.create_user("viewer1", security.hash_password(PASSWORD, 1000), "viewer")
        self.c = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def csrf(self, path="/login"):
        return re.search(r'name="csrf" value="([^"]+)"', self.c.get(path).get_data(as_text=True)).group(1)

    def login(self, user="admin1", pw=PASSWORD, ack=True):
        r = self.c.post("/login", data={"csrf": self.csrf(), "username": user, "password": pw})
        if ack and r.status_code == 302:
            self.c.post("/disclosures/accept", data={"csrf": self.csrf("/disclosures/accept"), "ack": "yes"})
        return r

    def test_pages_need_login(self):
        for p in ("/", "/performance", "/risk", "/broker", "/audit", "/admin/credentials", "/reports/trades.csv"):
            r = self.c.get(p)
            self.assertEqual(r.status_code, 302, p)
            self.assertIn("/login", r.headers["Location"])
        self.assertEqual(self.c.get("/api/summary").status_code, 401)
        self.assertEqual(self.c.get("/healthz").status_code, 200)

    def test_login_logout_and_audit(self):
        self.assertEqual(self.login().status_code, 302)
        self.assertEqual(self.c.get("/").status_code, 200)
        self.c.post("/logout", data={"csrf": self.csrf("/")})
        self.assertEqual(self.c.get("/").status_code, 302)
        actions = [r["action"] for r in self.store.audit_rows()]
        self.assertIn("auth.login", actions)
        self.assertIn("auth.logout", actions)
        self.assertIn("disclosure.accepted", actions)

    def test_risk_disclosure_required_first(self):
        self.login(ack=False)
        r = self.c.get("/")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/disclosures/accept", r.headers["Location"])

    def test_csrf_required(self):
        r = self.c.post("/login", data={"username": "admin1", "password": PASSWORD})
        self.assertEqual(r.status_code, 400)
        self.login()
        r = self.c.post("/risk/kill-switch", data={"action": "engage"})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(self.cfg.kill_switch_file.exists())

    def test_lockout_after_failures(self):
        for _ in range(3):
            self.assertEqual(self.login(pw="wrong-password-1A!").status_code, 401)
        self.assertEqual(self.login().status_code, 429)          # locked even with the right password
        self.assertIn("auth.locked", [r["action"] for r in self.store.audit_rows()])

    def test_two_factor_login(self):
        secret = security.new_totp_secret()
        self.store.update_user("admin1", totp_secret=secret)
        r = self.c.post("/login", data={"csrf": self.csrf(), "username": "admin1", "password": PASSWORD})
        self.assertIn("authenticator", r.get_data(as_text=True))
        self.assertEqual(self.c.get("/").status_code, 302)       # not signed in yet
        r = self.c.post("/login", data={"csrf": self.csrf(), "step": "2fa", "code": "000000"})
        self.assertEqual(r.status_code, 401)
        r = self.c.post("/login", data={"csrf": self.csrf(), "step": "2fa", "code": security.totp(secret)})
        self.assertEqual(r.status_code, 302)

    def test_viewer_cannot_use_admin_routes(self):
        self.login("viewer1")
        self.assertEqual(self.c.get("/admin/credentials").status_code, 403)
        self.assertEqual(self.c.get("/admin/users").status_code, 403)
        r = self.c.post("/backtests/run", data={"csrf": self.csrf("/"), "kind": "v3", "instrument": "NIFTY"})
        self.assertEqual(r.status_code, 403)

    def test_kill_switch_engage_by_viewer_release_by_admin_only(self):
        self.login("viewer1")
        self.c.post("/risk/kill-switch", data={"csrf": self.csrf("/risk"), "action": "engage", "reason": "test"})
        self.assertTrue(self.cfg.kill_switch_file.exists())
        r = self.c.post("/risk/kill-switch", data={"csrf": self.csrf("/risk"), "action": "release", "confirm": "RELEASE"})
        self.assertEqual(r.status_code, 403)
        self.assertTrue(self.cfg.kill_switch_file.exists())
        self.c.post("/logout", data={"csrf": self.csrf("/")})
        self.login("admin1")
        self.c.post("/risk/kill-switch", data={"csrf": self.csrf("/risk"), "action": "release", "confirm": "nope"})
        self.assertTrue(self.cfg.kill_switch_file.exists())     # wrong confirmation word
        self.c.post("/risk/kill-switch", data={"csrf": self.csrf("/risk"), "action": "release", "confirm": "RELEASE"})
        self.assertFalse(self.cfg.kill_switch_file.exists())
        targets = [(r["action"], r["target"], r["outcome"]) for r in self.store.audit_rows()]
        self.assertIn(("risk.kill_switch", "engage", "ok"), targets)
        self.assertIn(("risk.kill_switch", "release", "denied"), targets)
        self.assertIn(("risk.kill_switch", "release", "ok"), targets)

    def test_disarm_removes_arm_file_and_no_arm_route(self):
        self.cfg.arm_file.write_text(json.dumps({"day": "2026-01-01"}))
        self.login("viewer1")
        self.c.post("/risk/disarm", data={"csrf": self.csrf("/risk")})
        self.assertFalse(self.cfg.arm_file.exists())
        rules = [str(r) for r in self.app.url_map.iter_rules()]
        self.assertFalse([r for r in rules if "arm" in r and "disarm" not in r])

    def test_no_order_placing_code_in_console(self):
        src = Path(__file__).resolve().parents[1] / "src" / "console"
        pattern = re.compile(r"place_order|modify_order|cancel_order|UpstoxBroker|order/place|write_arm")
        hits = [p.name for p in src.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
        self.assertEqual(hits, [])

    def test_live_column_is_empty_without_live_fills_and_never_mixed(self):
        self.login()
        summary = self.c.get("/api/summary").get_json()["sources"]
        self.assertEqual(summary["live"]["trades"], 1)                 # only the upstox position
        self.assertNotIn("net", summary["paper"])                       # overlapping books are never summed
        self.assertEqual(summary["paper"]["trades"], 4)                # 3 scanner closed + 1 movement; mock/open excluded
        self.assertEqual(summary["backtest"]["trades"], 1)
        html = self.c.get("/").get_data(as_text=True)
        self.assertIn("Live (broker)", html)
        self.assertIn("never added together", html)

    def test_every_page_renders(self):
        self.login()
        for p in ("/", "/performance", "/performance?source=live", "/performance?source=backtest&book=bt-v3_backtest-v1",
                  "/trades", "/trades?source=live", "/paper", "/backtests", "/risk", "/broker", "/health", "/audit?verify=1",
                  "/reports", "/reports/performance", "/account", "/admin/users", "/admin/credentials", "/docs",
                  "/docs/live-trading", "/faq", "/disclosures", "/api/health", "/api/broker", "/api/risk",
                  "/api/performance?source=backtest"):
            r = self.c.get(p)
            self.assertEqual(r.status_code, 200, p)
        self.assertEqual(self.c.get("/docs/../../etc").status_code, 404)
        self.assertEqual(self.c.get("/backtests/reports/..%2Fdata%2Fprebreakout.db").status_code, 404)

    def test_security_headers(self):
        r = self.c.get("/login")
        self.assertIn("script-src 'self'", r.headers["Content-Security-Policy"])
        self.assertEqual(r.headers["X-Frame-Options"], "DENY")
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
        self.login()
        cookie = self.c.get_cookie("tc_session")
        self.assertTrue(cookie.http_only)

    def test_session_idle_timeout(self):
        self.login()
        with self.c.session_transaction() as s:
            s["seen"] = time.time() - 3600 * 5
        self.assertEqual(self.c.get("/").status_code, 302)

    def test_downloads_are_labelled_and_audited(self):
        self.login()
        r = self.c.get("/reports/trades.csv?source=paper")
        text = r.get_data(as_text=True)
        self.assertIn("attachment", r.headers["Content-Disposition"])
        self.assertIn("# source_label: Paper (simulated)", text)
        self.assertNotIn("99999", text)                                  # mock trade not exported
        self.assertIn("report.download", [a["action"] for a in self.store.audit_rows()])

    def test_credentials_vault_flow(self):
        self.login()
        tok = self.csrf("/admin/credentials")
        self.c.post("/admin/credentials", data={"csrf": tok, "action": "set", "name": "TELEGRAM_CHAT_ID", "value": "123456789012"})
        page = self.c.get("/admin/credentials").get_data(as_text=True)
        self.assertNotIn("123456789012", page)
        self.assertIn("…9012", page)
        r = self.c.post("/admin/credentials", data={"csrf": tok, "action": "set", "name": "LIVE_TRADING_CONFIRM", "value": "x"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("vault.set", [a["action"] for a in self.store.audit_rows()])


if __name__ == "__main__":
    unittest.main()


# ============================================================================= ML lab + AI assistant
MODEL_FIXTURE = {
    "trained_at": "2026-11-01T18:00:00", "samples": 900, "live_samples": 21, "instruments": ["NIFTY"], "approved": False,
    "threshold": 0.41, "label": "index reaches 1.5 R before the stop within 120 min",
    "walk_forward": {"ok": True, "oos_samples": 560, "base_rate": 0.38, "auc": 0.53, "brier": 0.236, "brier_skill": -0.004,
                     "mean_r_all": 0.02, "mean_r_kept": 0.04, "mean_r_dropped": -0.01, "kept_pct": 60.0, "hit_rate_kept": 0.40,
                     "calibration": [{"p_from": 0.2, "p_to": 0.3, "n": 112, "predicted": 0.27, "actual": 0.33, "mean_r": -0.05}]},
    "top_weights": [{"feature": "rsi", "weight": 0.31}, {"feature": "vix", "weight": -0.2}], "model": {}}


class MLSummaryTests(unittest.TestCase):
    def test_gate_checks_reflect_the_saved_results(self):
        from src.console import ml
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            self.assertFalse(ml.model_summary(p)["exists"])
            p.write_text(json.dumps(MODEL_FIXTURE))
            s = ml.model_summary(p)
            gate = {g["key"]: g["pass"] for g in s["gate"]}
            self.assertEqual(gate, {"oos_samples": True, "auc": False, "brier_skill": False, "r_uplift": False})
            self.assertAlmostEqual(s["r_uplift"], 0.02)
            self.assertFalse(s["approved"])
            self.assertEqual(s["weights"][0]["rel"], 1.0)

    def test_ml_job_arguments(self):
        self.assertEqual(build_argv("ml", "NIFTY,SENSEX", "2023-01-01"),
                         ["--train", "--instruments", "NIFTY,SENSEX", "--start", "2023-01-01"])
        for bad in (("ml", ""), ("ml", "NIFTY,NIFTY"), ("ml", "NIFTY", "2023-01-01", "2024-01-01"), ("ml", "FOO")):
            with self.assertRaises(JobError):
                build_argv(*bad)


class FakeClaude:
    """Plays the Claude Messages API: first asks for a tool, then answers from the tool result."""

    def __init__(self, script=None, status=200):
        self.requests, self.status = [], status
        self.script = script or [
            {"stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 20}, "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "tu1", "name": "get_performance", "input": {"source": "paper", "book": "scanner-v1"}}]},
            {"stop_reason": "end_turn", "usage": {"input_tokens": 300, "output_tokens": 40}, "model": "claude-test",
             "content": [{"type": "text", "text": "PAPER scanner-v1: **3 closed trades**, too few to judge."}]}]

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append({"url": url, "json": json, "headers": headers})
        if self.status != 200:
            return FakeResponse(self.status, {"error": {"message": "nope"}})
        return FakeResponse(200, self.script[min(len(self.requests) - 1, len(self.script) - 1)])


class AssistantTests(unittest.TestCase):
    def setUp(self):
        from src.console.web import create_app
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        make_dbs(root)
        (root / "data" / "prebreakout_ml_model.json").write_text(json.dumps(MODEL_FIXTURE))
        self.cfg = ConsoleConfig(data_dir=root / "data", output_dir=root / "output", log_dir=root / "logs",
                                 kill_switch_file=root / "KILL_SWITCH", token_file=root / "upstox_token.json",
                                 ai_requests_per_hour=3)
        self.fake = FakeClaude()
        self.app = create_app(self.cfg, broker_session=FakeSession(), ai_session=self.fake, testing=True)
        self.assistant = self.app.extensions["console"]["assistant"]
        self.assistant.api_key_fn = lambda: "sk-ant-test-key-123"
        store = self.app.extensions["console"]["store"]
        self.store = store
        store.create_user("viewer1", security.hash_password(PASSWORD, 1000), "viewer")
        store.update_user("viewer1", risk_ack="2026-11-08T00:00:00")
        self.c = self.app.test_client()
        tok = re.search(r'name="csrf" value="([^"]+)"', self.c.get("/login").get_data(as_text=True)).group(1)
        self.c.post("/login", data={"csrf": tok, "username": "viewer1", "password": PASSWORD})
        self.tok = re.search(r'name="csrf" value="([^"]+)"', self.c.get("/").get_data(as_text=True)).group(1)

    def tearDown(self):
        self.tmp.cleanup()

    def ask(self, q, token=True):
        return self.c.post("/api/assistant", json={"question": q}, headers={"X-CSRF-Token": self.tok} if token else {})

    def test_tool_loop_answers_from_real_data(self):
        r = self.ask("How is v1 doing?")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        j = r.get_json()
        self.assertIn("too few to judge", j["answer"])
        self.assertEqual([t["name"] for t in j["tools"]], ["get_performance"])
        self.assertEqual(j["usage"], {"input_tokens": 400, "output_tokens": 60})
        first, second = self.fake.requests
        self.assertEqual(first["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(first["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(first["json"]["model"], self.cfg.ai_model)
        self.assertIn("Never add sources together", first["json"]["system"])
        result = second["json"]["messages"][-1]["content"][0]
        self.assertEqual((result["type"], result["tool_use_id"]), ("tool_result", "tu1"))
        payload = json.loads(result["content"])
        self.assertEqual(payload["metrics"]["trades"], 2)          # scanner v1 closed, non-mock trades in the fixture
        self.assertIn("ai.ask", [a["action"] for a in self.store.audit_rows()])
        # the conversation is remembered for the next question
        self.ask("and v3?")
        self.assertEqual(self.fake.requests[2]["json"]["messages"][0]["content"], "How is v1 doing?")

    def test_no_secrets_and_no_write_tools_are_offered(self):
        from src.console.assistant import TOOLS
        names = {t["name"] for t in TOOLS}
        self.assertTrue(all(n.startswith(("get_", "list_")) for n in names))
        self.ask("hi")
        sent = json.dumps(self.fake.requests)
        self.assertNotIn("sk-ant-test-key-123", json.dumps([r["json"] for r in self.fake.requests]))
        self.assertNotIn("password_hash", sent)

    def test_every_tool_runs_on_real_shapes(self):
        tools = self.assistant.tools
        for name, args in (("get_overview", {}), ("list_books", {"source": "paper"}),
                           ("get_performance", {"source": "backtest"}), ("get_breakdown", {"source": "paper", "by": "hour"}),
                           ("get_breakdown", {"source": "paper", "by": "weekday"}),
                           ("get_trades", {"source": "paper", "only": "losers", "limit": 5}), ("get_risk", {}),
                           ("get_health", {}), ("get_ml_model", {})):
            out = json.loads(self.assistant._run_tool(name, args))
            self.assertNotIn("error", out if isinstance(out, dict) else {}, (name, out))
        self.assertIn("error", json.loads(self.assistant._run_tool("get_performance", {"source": "nonsense"})))
        self.assertIn("error", json.loads(self.assistant._run_tool("delete_everything", {})))
        ov = json.loads(self.assistant._run_tool("get_overview", {}))
        self.assertNotIn("net", ov["sources"]["paper"])             # overlapping books are not summed for the model either

    def test_csrf_rate_limit_and_errors(self):
        self.assertEqual(self.ask("x", token=False).status_code, 400)
        self.assertEqual(self.ask("").status_code, 400)
        for _ in range(2):
            self.assertEqual(self.ask("q").status_code, 200)
        r = self.ask("q")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.ask("q").status_code, 429)          # 3 per hour in this test

    def test_bad_key_and_missing_key(self):
        self.fake.status = 401
        r = self.ask("hello")
        self.assertEqual(r.status_code, 502)
        self.assertIn("rejected the API key", r.get_json()["error"])
        self.assistant.api_key_fn = lambda: None
        r = self.ask("hello")
        self.assertEqual(r.status_code, 503)
        self.assertIn("Credentials", r.get_json()["error"])

    def test_pages_render(self):
        for p in ("/assistant", "/ml", "/api/ml"):
            self.assertEqual(self.c.get(p).status_code, 200, p)
        html = self.c.get("/ml").get_data(as_text=True)
        self.assertIn("Advisory only", html)
        self.assertIn("AUC (ranking skill)", html)
        r = self.c.post("/ml/train", data={"csrf": self.tok, "inst_NIFTY": "1"})
        self.assertEqual(r.status_code, 403)                       # viewers cannot train
