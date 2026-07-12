import pytest
from pathlib import Path

from cosmographic_analysis.config import (
    Config,
    DataConfig,
    OutputConfig,
    ParametersConfig,
    ensure_output_dirs,
    load_config,
)


class TestLoadConfigDefaults:
    def test_no_path_no_cwd_file_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert isinstance(config, Config)
        assert config.data == DataConfig()
        assert config.parameters == ParametersConfig()
        assert config.output == OutputConfig()

    def test_nonexistent_path_returns_defaults(self, tmp_path):
        missing = tmp_path / "missing.yaml"
        config = load_config(str(missing))
        assert config.data == DataConfig()
        assert config.parameters == ParametersConfig()
        assert config.output == OutputConfig()


class TestLoadConfigMocked:
    def test_full_yaml_overrides_all_sections(self, mocker):
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch(
            "cosmographic_analysis.config.yaml.safe_load",
            return_value={
                "data": {
                    "lcparam": "custom_lc.dat",
                    "cov_matrix": "custom_cov.cov",
                },
                "parameters": {
                    "nside": 64,
                    "h0f": 0.75,
                    "q0f": -0.5,
                    "zup": 0.2,
                    "zdown": 0.02,
                    "repetitions": 1000,
                    "prefix_name": "[TEST]",
                },
                "output": {
                    "compilations": "out/comp/",
                    "figures": "out/fig/",
                    "histograms": "out/hist/",
                    "tables": "out/tab/",
                },
            },
        )
        config = load_config("fake.yaml")
        assert config.data.lcparam == "custom_lc.dat"
        assert config.data.cov_matrix == "custom_cov.cov"
        assert config.parameters.nside == 64
        assert config.parameters.h0f == 0.75
        assert config.parameters.q0f == -0.5
        assert config.parameters.zup == 0.2
        assert config.parameters.zdown == 0.02
        assert config.parameters.repetitions == 1000
        assert config.parameters.prefix_name == "[TEST]"
        assert config.output.compilations == "out/comp/"
        assert config.output.figures == "out/fig/"
        assert config.output.histograms == "out/hist/"
        assert config.output.tables == "out/tab/"

    def test_empty_yaml_preserves_all_defaults(self, mocker):
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch(
            "cosmographic_analysis.config.yaml.safe_load",
            return_value=None,
        )
        config = load_config("fake.yaml")
        assert config.data == DataConfig()
        assert config.parameters == ParametersConfig()
        assert config.output == OutputConfig()

    def test_partial_yaml_overrides_only_provided_sections(self, mocker):
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch(
            "cosmographic_analysis.config.yaml.safe_load",
            return_value={"parameters": {"nside": 32}},
        )
        config = load_config("fake.yaml")
        assert config.parameters.nside == 32
        assert config.parameters.h0f == 0.7304
        assert config.parameters.q0f == -0.574
        assert config.parameters.zup == 0.1
        assert config.parameters.zdown == 0.01
        assert config.parameters.repetitions == 500
        assert config.parameters.prefix_name == "[SH0ES_CALIB]"
        assert config.data.lcparam == "datos/Pantheon+SH0ES.dat.txt"
        assert config.data.cov_matrix == "datos/Pantheon+SH0ES_STAT+SYS.cov.txt"
        assert config.output.compilations == "compilations/"
        assert config.output.figures == "figures/"
        assert config.output.histograms == "histograms/"
        assert config.output.tables == "tables/"


class TestEnsureOutputDirs:
    def test_creates_all_four_directories(self, tmp_path):
        config = Config(
            output=OutputConfig(
                compilations=str(tmp_path / "a"),
                figures=str(tmp_path / "b"),
                histograms=str(tmp_path / "c"),
                tables=str(tmp_path / "d"),
            )
        )
        ensure_output_dirs(config)
        assert (tmp_path / "a").is_dir()
        assert (tmp_path / "b").is_dir()
        assert (tmp_path / "c").is_dir()
        assert (tmp_path / "d").is_dir()
