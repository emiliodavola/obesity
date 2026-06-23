"""Unit tests for the core engine."""

from unittest.mock import patch

from obesity.utils import ensure_model_checkpoint, get_device


def test_get_device():
    """Verify the device selection logic."""
    device = get_device()
    assert device in ["cuda", "cpu"]


@patch("os.path.exists")
def test_ensure_model_checkpoint_exists(mock_exists):
    """Test that an existing model is returned without downloading."""
    mock_exists.return_value = True

    path = "some/local/path"
    result = ensure_model_checkpoint(path, "repo", "file")

    assert result == path


@patch("os.path.exists")
@patch("os.makedirs")
@patch("obesity.utils.hf_hub_download")
@patch("shutil.copy")
def test_ensure_model_checkpoint_downloads(
    mock_copy, mock_download, mock_makedirs, mock_exists
):
    mock_exists.return_value = False
    mock_download.return_value = "cached/path"

    path = "local/path"
    result = ensure_model_checkpoint(path, "repo", "file")

    assert result == path
    mock_download.assert_called_once_with(
        repo_id="repo",
        filename="file",
    )
    mock_copy.assert_called_once_with("cached/path", path)