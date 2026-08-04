from pathlib import Path
from typing import Union

import yaml


def load_config(config_path: Union[str, Path]) -> dict:
    """Load a YAML configuration file and return it as a dictionary."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must resolve to a mapping: {path}")

    return config
