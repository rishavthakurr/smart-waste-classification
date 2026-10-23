"""Real-browser test of the running Streamlit app (skipped when Playwright/Chromium are unavailable).

Setup once:  pip install playwright && python -m playwright install chromium
Uses a synthetic-data sandbox: it verifies the UI wiring (upload -> classify -> database -> history -> CSV),
never model quality.
"""
import csv
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright

from helpers import ROOT, make_sandbox, run_module
from src import database
from src.utils import pretty


@pytest.fixture(scope="module")
def live_app(tmp_path_factory):
    tmp, env = make_sandbox(tmp_path_factory.mktemp("browser"))
    run_module(["training.prepare_dataset"], env)
    run_module(["training.train_cnn", "--smoke", "--cpu"], env)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--server.port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if urllib.request.urlopen(url + "/_stcore/health", timeout=2).read() == b"ok":
                break
        except Exception:
            time.sleep(1)
    else:
        proc.kill(); pytest.fail("Streamlit server did not start")
    yield url, tmp
    proc.terminate(); proc.wait(timeout=20)


def test_upload_classify_history_csv(live_app, tmp_path):
    url, tmp = live_app
    img = sorted((tmp / "ds" / "test" / "glass").glob("*.jpg"))[0]
    try:
        pw = sync_playwright().start(); browser = pw.chromium.launch()
    except Exception as exc:
        pytest.skip(f"Chromium unavailable: {exc}")
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 1800})
        page.goto(url + "/classify_waste"); page.wait_for_selector("[data-testid=stApp]")
        page.set_input_files("input[type=file]", str(img)); page.wait_for_timeout(2000)
        assert "not available" in page.inner_text("[data-testid=stApp]").lower()   # MobileNetV2/ResNet50 untrained
        page.get_by_role("button", name="Classify Waste").click()
        page.wait_for_selector("text=Recommended Action", timeout=60000)
        shown = page.locator(".result-class").inner_text()
        rows = database.query_history(db_path=tmp / "h.db")
        assert len(rows) == 1 and pretty(rows.iloc[0]["predicted_class"]) == shown
        assert page.locator("[data-testid=stException]").count() == 0

        page.goto(url + "/history"); page.wait_for_selector("text=Matching records")
        assert re.search(r"Matching records\s*\n?\s*1", page.inner_text("[data-testid=stApp]"))
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download CSV").click()
        out = list(csv.DictReader(open(dl.value.path())))
        assert len(out) == 1 and out[0]["image_filename"] == img.name
    finally:
        browser.close(); pw.stop()
