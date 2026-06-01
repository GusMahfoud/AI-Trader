"""YAML config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must parse to a mapping at the top level.")
    return config


def ensure_dir(dir_path: str) -> str:
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())
