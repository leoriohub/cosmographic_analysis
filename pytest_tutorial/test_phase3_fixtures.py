"""Phase 3: Fixtures — reusable setup without boilerplate.

Tests the Config dataclass from cosmographic_analysis.config.
Fixtures (defined in conftest.py) provide pre-configured Config objects
so we don't repeat construction logic in every test.
"""

import pytest
from cosmographic_analysis.config import Config, ParametersConfig


class TestConfigDefaults:
    """Group tests that use the default_config fixture."""

    def test_nside_default(self, default_config: Config):
        assert default_config.parameters.nside == 16

    def test_h0f_default(self, default_config: Config):
        assert default_config.parameters.h0f == 0.7304

    def test_q0f_default(self, default_config: Config):
        assert default_config.parameters.q0f == -0.574

    def test_z_range_default(self, default_config: Config):
        assert default_config.parameters.zup == 0.1
        assert default_config.parameters.zdown == 0.01

    def test_output_dirs_default(self, default_config: Config):
        assert default_config.output.compilations == "compilations/"
        assert default_config.output.figures == "figures/"
        assert default_config.output.histograms == "histograms/"
        assert default_config.output.tables == "tables/"


class TestCustomConfig:
    """Group tests that use the custom_config fixture."""

    def test_custom_values(self, custom_config: Config):
        assert custom_config.parameters.nside == 32
        assert custom_config.parameters.h0f == 0.70
        assert custom_config.parameters.repetitions == 100

    def test_custom_file_paths(self, custom_config: Config):
        assert custom_config.data.lcparam == "tests/mock_data.txt"
        assert custom_config.data.cov_matrix == "tests/mock_cov.txt"


def test_modifications_do_not_leak_across_tests(default_config):
    """Each test gets a *fresh* Config — mutations in one test don't affect others."""
    cfg = default_config
    original = cfg.parameters.h0f
    cfg.parameters.h0f = 999.0  # Mutate
    assert cfg.parameters.h0f == 999.0


def test_config_is_unchanged_by_previous_test(default_config):
    """Confirms the fixture was re-created fresh for this test."""
    assert default_config.parameters.h0f == 0.7304  # Still the default!
