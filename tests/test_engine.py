from unittest.mock import patch
from src.engine import get_device, ensure_model_checkpoint


def test_get_device():
    # Should return "cuda" if available, else "cpu"
    device = get_device()
    assert device in ["cuda", "cpu"]


@patch("os.path.exists")
def test_ensure_model_checkpoint_exists(mock_exists):
    mock_exists.return_value = True
    path = "some/local/path"
    result = ensure_model_checkpoint(path, "repo", "file")
    assert result == path


@patch("os.path.exists")
@patch("os.makedirs")
@patch("src.engine.hf_hub_download")
@patch("shutil.copy")
def test_ensure_model_checkpoint_downloads(
    mock_copy, mock_download, mock_makedirs, mock_exists
):
    mock_exists.return_value = False
    mock_download.return_value = "cached/path"

    path = "local/path"
    result = ensure_model_checkpoint(path, "repo", "file")

    assert result == path
    mock_download.assert_called_once_with(repo_id="repo", filename="file")
    mock_copy.assert_called_once_with("cached/path", path)
