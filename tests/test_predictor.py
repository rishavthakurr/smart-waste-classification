import numpy as np
import pytest
from PIL import Image

from src import model_loader
from src.model_loader import ModelNotAvailableError, load_model
from src.predictor import PredictionError, predict, result_from_probabilities
from src.utils import class_names


class FakeModel:
    """Stand-in model for unit tests only. Returns fixed probabilities; NOT real inference."""
    input_shape = (None, 32, 32, 3)

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)

    def __call__(self, x, training=False):
        assert x.shape == (1, 32, 32, 3)
        return self.probs[None, :]


def test_result_structure_and_ordering():
    r = result_from_probabilities(np.array([.05, .1, .6, .15, .06, .04]), "cnn")
    assert r.predicted_class == "cardboard" and abs(r.confidence - .6) < 1e-9
    assert list(r.probabilities) == class_names() and abs(sum(r.probabilities.values()) - 1) < 1e-6
    assert [c for c, _ in r.top3] == ["cardboard", "glass", "paper"]
    assert r.low_confidence is False and set(r.as_dict()) >= {"predicted_class", "confidence", "top3"}


def test_low_confidence_flag():
    assert result_from_probabilities(np.full(6, 1 / 6), "cnn").low_confidence


@pytest.mark.parametrize("bad", [np.ones(5), np.array([np.nan] * 6)])
def test_invalid_model_output_rejected(bad):
    with pytest.raises(PredictionError):
        result_from_probabilities(bad, "cnn")


def test_predict_with_mock_model_end_to_end_preprocessing():
    img = Image.new("RGB", (80, 60), (120, 80, 40))
    r = predict(img, "cnn", FakeModel([0, 0, 0, .9, .1, 0]))
    assert r.predicted_class == "glass" and r.model_key == "cnn" and r.inference_ms >= 0


def test_inference_failure_becomes_prediction_error():
    class Broken(FakeModel):
        def __call__(self, x, training=False):
            raise RuntimeError("boom")
    with pytest.raises(PredictionError, match="Inference failed"):
        predict(Image.new("RGB", (40, 40)), "cnn", Broken([1, 0, 0, 0, 0, 0]))


def test_missing_model_file_is_reported_not_substituted(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loader, "model_path", lambda key, tag=None: tmp_path / "nope.keras")
    assert model_loader.is_available("resnet50") is False
    with pytest.raises(ModelNotAvailableError, match="not been trained"):
        load_model("resnet50")
    with pytest.raises(ModelNotAvailableError):
        load_model("not_a_model")


def test_deployment_model_none_when_nothing_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loader, "model_path", lambda key, tag=None: tmp_path / f"{key}.keras")
    key, status = model_loader.get_deployment_model()
    assert key is None and "no trained model" in status
