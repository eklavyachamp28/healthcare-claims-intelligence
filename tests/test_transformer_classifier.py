import pytest
from transformer_classifier import classify_claim


def test_reject_empty_without_model_download():
    with pytest.raises(ValueError):
        classify_claim("")


def test_reject_invalid_threshold_without_model_download():
    with pytest.raises(ValueError):
        classify_claim("test", threshold=2)
