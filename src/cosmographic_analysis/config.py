"""Pipeline configuration loader.

Loads parameters from config.yaml with defaults and CLI overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DataConfig:
    lcparam: str = "datos/Pantheon+SH0ES.dat.txt"
    cov_matrix: str = "datos/Pantheon+SH0ES_STAT+SYS.cov.txt"


@dataclass
class ParametersConfig:
    nside: int = 16
    h0f: float = 0.7304
    q0f: float = -0.574
    zup: float = 0.1
    zdown: float = 0.01
    repetitions: int = 500
    prefix_name: str = "[SH0ES_CALIB]"


@dataclass
class OutputConfig:
    compilations: str = "compilations/"
    figures: str = "figures/"
    histograms: str = "histograms/"
    tables: str = "tables/"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    parameters: ParametersConfig = field(default_factory=ParametersConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file, falling back to defaults."""
    config = Config()

    if config_path is None:
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent.parent.parent / "config.yaml",
        ]
        for path in candidates:
            if path.exists():
                config_path = str(path)
                break

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        if "data" in raw:
            config.data = DataConfig(**raw["data"])
        if "parameters" in raw:
            config.parameters = ParametersConfig(**raw["parameters"])
        if "output" in raw:
            config.output = OutputConfig(**raw["output"])

    return config


def ensure_output_dirs(config: Config) -> None:
    """Create output directories if they don't exist."""
    for path in [
        config.output.compilations,
        config.output.figures,
        config.output.histograms,
        config.output.tables,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
