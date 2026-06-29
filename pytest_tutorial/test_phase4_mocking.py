"""Phase 4: Mocking — test code that talks to the outside world.

We test config.load_config(), which normally reads a YAML file from disk.
By mocking the file I/O we:
  1. Keep the test fast (no disk access).
  2. Keep it deterministic (no dependency on real config.yaml content).
  3. Test edge cases that would be hard to trigger with a real file.
"""

from pathlib import Path
import pytest
from cosmographic_analysis.config import load_config


def test_load_config_with_mocked_yaml(mocker):
    """Mock yaml.safe_load to return custom data without a real file.

    mocker (from pytest-mock) is a fixture that wraps unittest.mock.
    """
    fake_yaml_data = {
        "parameters": {"nside": 64, "h0f": 0.68, "q0f": -0.50, "repetitions": 10},
    }

    # Patch yaml.safe_load BEFORE calling load_config
    mock_yaml = mocker.patch("cosmographic_analysis.config.yaml.safe_load",
                             return_value=fake_yaml_data)

    # Also ensure Path.exists returns True so load_config tries to read
    mocker.patch.object(Path, "exists", return_value=True)

    # Mock builtins.open so no actual file is opened
    mocker.patch("builtins.open", mocker.mock_open())

    config = load_config("fake_path.yaml")

    # Verify the mocked values were used
    assert config.parameters.nside == 64
    assert config.parameters.h0f == 0.68
    assert config.parameters.q0f == -0.50
    assert config.parameters.repetitions == 10

    # Verify yaml.safe_load was actually called
    mock_yaml.assert_called_once()


def test_load_config_fallback_to_defaults(mocker):
    """When config file doesn't exist, load_config returns defaults."""
    # Make Path.exists return False
    mocker.patch.object(Path, "exists", return_value=False)

    config = load_config("nonexistent.yaml")

    # Should fall back to hardcoded defaults
    assert config.parameters.nside == 16
    assert config.parameters.h0f == 0.7304


def test_load_config_with_empty_yaml(mocker):
    """Edge case: YAML file exists but is empty."""
    mocker.patch.object(Path, "exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("cosmographic_analysis.config.yaml.safe_load", return_value=None)

    config = load_config("empty.yaml")

    # Should still produce a valid Config with defaults
    assert config.parameters.nside == 16
    assert config.parameters.h0f == 0.7304
