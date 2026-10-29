"""Unit tests for data, evaluation, model wrappers, and checkpoint utilities."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from obesity.data import load_data, split_data
from obesity.evaluate import evaluate_model
from obesity.pipeline import TabPFNWrapper
from obesity.utils import ensure_model_checkpoint, get_device


def test_get_device() -> None:
    """Verify the device selection logic."""
    device: str = get_device()
    assert device in ("cuda", "cpu")


@patch("os.path.exists")
def test_ensure_model_checkpoint_exists(mock_exists: MagicMock) -> None:
    """Test that an existing model is returned without downloading."""
    mock_exists.return_value = True

    path: str = "some/local/path"
    result: str = ensure_model_checkpoint(path, "repo", "file")

    assert result == path


@patch("os.path.exists")
@patch("os.makedirs")
@patch("obesity.utils.hf_hub_download")
@patch("shutil.copy")
def test_ensure_model_checkpoint_downloads(
    mock_copy: MagicMock,
    mock_download: MagicMock,
    mock_makedirs: MagicMock,
    mock_exists: MagicMock,
) -> None:
    """Copy a downloaded checkpoint into the configured location."""
    mock_exists.return_value = False
    mock_download.return_value = "cached/path"

    path: str = "local/path"
    result: str = ensure_model_checkpoint(path, "repo", "file")

    assert result == path
    mock_download.assert_called_once_with(
        repo_id="repo",
        filename="file",
    )
    mock_copy.assert_called_once_with("cached/path", path)


@patch("obesity.data.pd.read_csv")
def test_load_data_reads_the_configured_csv(mock_read_csv: MagicMock) -> None:
    """Load a dataframe from the configured CSV path."""
    expected: pd.DataFrame = pd.DataFrame({"feature": [1, 2]})
    mock_read_csv.return_value = expected

    result: pd.DataFrame = load_data("data.csv")

    mock_read_csv.assert_called_once_with("data.csv")
    pd.testing.assert_frame_equal(result, expected)


def test_split_data_preserves_stratified_class_counts() -> None:
    """Keep the binary target balanced in both data partitions."""
    frame: pd.DataFrame = pd.DataFrame(
        {"feature": list(range(20)), "obese": [0, 1] * 10}
    )

    split: list[Any] = split_data(frame, "obese", test_size=0.3, random_state=42)

    train_target: pd.Series = split[2]
    test_target: pd.Series = split[3]
    assert train_target.value_counts().to_dict() == {0: 7, 1: 7}
    assert test_target.value_counts().to_dict() == {0: 3, 1: 3}


def test_evaluate_model_returns_roc_auc_and_accuracy() -> None:
    """Calculate ROC AUC and accuracy from classifier predictions."""
    classifier: MagicMock = MagicMock()
    features: np.ndarray = np.array([[1], [2], [3], [4]])
    target: np.ndarray = np.array([0, 0, 1, 1])
    probabilities: np.ndarray = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]]
    )
    classifier.predict_proba.return_value = probabilities
    classifier.predict.return_value = np.array([0, 1, 1, 0])

    metrics: tuple[float, float] = evaluate_model(classifier, features, target)

    assert metrics == pytest.approx((0.75, 0.5))
    classifier.predict_proba.assert_called_once_with(features)
    classifier.predict.assert_called_once_with(features)


def test_tabpfn_wrapper_captures_dataframe_feature_names() -> None:
    """Capture column names when fitting on a dataframe."""
    estimator: MagicMock = MagicMock()
    wrapper: TabPFNWrapper = TabPFNWrapper(estimator)
    features: pd.DataFrame = pd.DataFrame({"height": [1], "weight": [2]})
    target: np.ndarray = np.array([0])

    wrapper.fit(features, target)

    assert wrapper.feature_names_in_ == ["height", "weight"]
    assert wrapper.feature_names_out_ == ["height", "weight"]


def test_tabpfn_wrapper_captures_transformer_feature_names() -> None:
    """Capture names exposed by an upstream transformer."""
    estimator: MagicMock = MagicMock()
    feature_provider: MagicMock = MagicMock(spec=["get_feature_names_out"])
    feature_provider.get_feature_names_out.return_value = ["encoded_a", "encoded_b"]
    wrapper: TabPFNWrapper = TabPFNWrapper(estimator)
    target: np.ndarray = np.array([0])

    wrapper.fit(feature_provider, target)

    assert wrapper.feature_names_in_ == ["encoded_a", "encoded_b"]
    feature_provider.get_feature_names_out.assert_called_once_with()


def test_tabpfn_wrapper_delegates_transform() -> None:
    """Delegate transformation to the fitted estimator."""
    estimator: MagicMock = MagicMock()
    features: np.ndarray = np.array([[1, 2]])
    transformed: np.ndarray = np.array([[3, 4]])
    estimator.transform.return_value = transformed
    wrapper: TabPFNWrapper = TabPFNWrapper(estimator)

    wrapper.fit(features, np.array([0]))
    result: np.ndarray = wrapper.transform(features)

    np.testing.assert_array_equal(result, transformed)
    estimator.transform.assert_called_once_with(features)


def test_tabpfn_wrapper_uses_provided_feature_names() -> None:
    """Prefer feature names supplied by the preceding pipeline step."""
    estimator: MagicMock = MagicMock()
    features: pd.DataFrame = pd.DataFrame({"height": [1]})
    wrapper: TabPFNWrapper = TabPFNWrapper(estimator)

    wrapper.fit(features, np.array([0]))
    names: list[str] = wrapper.get_feature_names_out(["scaled_height"])

    assert names == ["scaled_height"]
    assert wrapper.get_feature_names_out() == ["height"]


@patch("obesity.utils.os.path.exists")
@patch("obesity.utils.os.makedirs")
@patch("obesity.utils.hf_hub_download")
def test_ensure_model_checkpoint_reraises_download_errors(
    mock_download: MagicMock,
    mock_makedirs: MagicMock,
    mock_exists: MagicMock,
) -> None:
    """Re-raise a checkpoint download error without accessing the network."""
    mock_exists.return_value = False
    mock_download.side_effect = RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        ensure_model_checkpoint("local/model.ckpt", "repo", "model.ckpt")

    mock_makedirs.assert_called_once_with("local", exist_ok=True)
