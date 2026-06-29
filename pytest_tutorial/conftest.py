"""Shared fixtures for the pytest tutorial.

conftest.py is automatically discovered by pytest — no import needed.
Fixtures defined here are available to all tests in this directory.
"""

import pytest
from cosmographic_analysis.config import Config, ParametersConfig, DataConfig


@pytest.fixture
def default_config():
    """Returns a Config with default parameters (no YAML file needed).

    Fixtures are injected by name into test functions. pytest handles
    setup and teardown automatically.
    """
    return Config()


@pytest.fixture
def custom_config():
    """Returns a Config with overridden analysis parameters."""
    return Config(
        data=DataConfig(
            lcparam="tests/mock_data.txt",
            cov_matrix="tests/mock_cov.txt",
        ),
        parameters=ParametersConfig(
            nside=32,
            h0f=0.70,
            q0f=-0.55,
            zup=0.15,
            zdown=0.01,
            repetitions=100,
            prefix_name="[CUSTOM]",
        ),
    )
