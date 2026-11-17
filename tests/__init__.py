"""
Test package. Runtime files (database, outputs, logs, lock) go to a temporary folder for the whole test
run, so tests never touch your real data/ or output/ and can run while the scanner is running.

    python -m unittest discover -s tests -t .
"""
import os
import tempfile

os.environ.setdefault("SCANNER_RUNTIME_DIR", tempfile.mkdtemp(prefix="scanner-tests-"))
os.environ["UPSTOX_ACCESS_TOKEN"] = ""   # tests never use a real token

import logging

logging.getLogger("scanner").addHandler(logging.NullHandler())   # keep test output quiet
