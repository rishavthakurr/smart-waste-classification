"""Smoke tests: every page renders without raising an exception (headless Streamlit AppTest)."""
import io

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from pathlib import Path

from src import database

ROOT = Path(__file__).resolve().parents[1]

PAGES = ["pages/dashboard.py", "pages/classify_waste.py", "pages/live_detection.py",
         "pages/model_performance.py", "pages/history.py", "pages/about.py"]


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "default_db_path", lambda: tmp_path / "t.db")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    at.switch_page(page).run()
    assert not at.exception, f"{page}: {at.exception}"


def test_classify_section_invalid_file_shows_friendly_error():
    def script():
        from src.ui import classify_section
        classify_section("t", b"definitely not an image", "bad.png")
    at = AppTest.from_function(script, default_timeout=30).run()
    assert not at.exception and any("not a valid" in e.value for e in at.error)


def test_classify_section_unsupported_type_error():
    def script():
        from src.ui import classify_section
        classify_section("t", b"abc", "notes.txt")
    at = AppTest.from_function(script, default_timeout=30).run()
    assert any("Unsupported" in e.value for e in at.error)
