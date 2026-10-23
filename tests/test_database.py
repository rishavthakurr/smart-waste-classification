import pandas as pd
import pytest

from src import database as db


@pytest.fixture
def path(tmp_path):
    return tmp_path / "sub" / "h.db"


def test_init_creates_file_and_schema(path):
    db.init_db(path)
    assert path.exists() and db.count_predictions(path) == 0
    assert query_empty(path)


def query_empty(path):
    df = db.query_history(db_path=path)
    return df.empty and list(df.columns) == db.COLUMNS


def test_save_and_read_roundtrip(path):
    rid = db.add_prediction("a.jpg", "glass", 0.91, "cnn", {"glass": 0.91, "metal": 0.09}, db_path=path)
    df = db.query_history(db_path=path)
    row = df.iloc[0]
    assert rid == row["id"] and row["predicted_class"] == "glass" and row["status"] == "Confident"
    assert abs(row["confidence"] - 0.91) < 1e-9 and "metal" in row["probabilities"]


def test_low_confidence_status(path):
    db.add_prediction("b.jpg", "paper", 0.30, "cnn", db_path=path)
    assert db.query_history(db_path=path).iloc[0]["status"] == "Low confidence"


def test_filters_search_and_sort(path):
    db.add_prediction("x_one.jpg", "glass", .9, "cnn", db_path=path)
    db.add_prediction("x_two.jpg", "metal", .8, "resnet50", db_path=path)
    db.add_prediction("y.jpg", "glass", .7, "resnet50", db_path=path)
    q = lambda **k: db.query_history(db_path=path, **k)
    assert len(q(category="glass")) == 2 and len(q(model="resnet50")) == 2
    assert len(q(search="x_")) == 2 and len(q(category="glass", model="resnet50")) == 1
    assert q(newest_first=False).iloc[0]["image_filename"] == "x_one.jpg"
    assert len(q(limit=1)) == 1


def test_sql_injection_is_inert(path):
    db.add_prediction("a.jpg", "glass", .9, "cnn", db_path=path)
    db.query_history(search="'; DROP TABLE predictions; --", db_path=path)
    assert db.count_predictions(path) == 1


def test_clear_and_export_csv(path):
    db.add_prediction("a.jpg", "glass", .9, "cnn", {"glass": .9}, db_path=path)
    csv = db.export_csv(db.query_history(db_path=path)).decode()
    assert csv.splitlines()[0].startswith("id,timestamp") and "probabilities" not in csv and "a.jpg" in csv
    assert db.clear_history(path) == 1 and db.count_predictions(path) == 0
