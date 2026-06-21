"""YAML config loader with deep-merge override support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overrides* into *base*, returning a new dict."""
    out = base.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    config_path: str = "config.yaml",
    override_path: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must parse to a mapping at the top level.")

    if override_path is not None:
        override_file = Path(override_path)
        if not override_file.exists():
            raise FileNotFoundError(f"Override file not found: {override_file.resolve()}")
        with override_file.open("r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
        if not isinstance(overrides, dict):
            raise ValueError("Override file must parse to a mapping at the top level.")
        config = deep_merge(config, overrides)

    return config


def ensure_dir(dir_path: str) -> str:
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())
