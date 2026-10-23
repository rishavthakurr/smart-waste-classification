import pytest

from src.recommendations import DISCLAIMER, get_bin, get_recommendation, validate_coverage
from src.utils import class_names, load_config


def test_every_configured_class_has_recommendation_and_bin():
    assert validate_coverage() == []
    for c in class_names():
        rec = get_recommendation(c)
        assert rec["action"] and rec["tip"] and rec["disclaimer"] == DISCLAIMER
        assert get_bin(c)["bin"].startswith("Bin")


def test_unknown_category_raises():
    with pytest.raises(KeyError):
        get_recommendation("unobtainium")


def test_class_labels_consistent_with_config():
    assert class_names() == ["plastic", "paper", "cardboard", "glass", "metal", "general_waste"]
    cfg = load_config()
    assert set(cfg["dataset"]["label_map"].values()) == set(class_names())
    assert {m["preprocessing"] for m in cfg["models"].values()} <= {"rescale_01", "mobilenet_v2", "resnet50"}


def test_recommendations_do_not_claim_universal_recyclability():
    assert "local" in DISCLAIMER.lower()
