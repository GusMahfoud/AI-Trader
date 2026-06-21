"""DataBundle assembly — the public API for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from .features import _add_features, feature_columns
from .loader import _load_market_data
from .splits import _split_bounds


@dataclass
class DataBundle:
    frame: pd.DataFrame
    feature_columns: List[str]
    split_bounds: Dict[str, Tuple[int, int]]
    feature_mean: pd.Series
    feature_std: pd.Series


def load_featured_frame(config: Dict[str, Any]) -> pd.DataFrame:
    """Load raw OHLCV and attach features — the unnormalised, unsplit frame.

    Separated from normalisation so walk-forward can load once and re-fit
    train-split statistics per fold without re-downloading or re-engineering.
    """
    env_cfg = config.get("env", {})
    seed = int(config.get("training", {}).get("seed", 42))
    raw = _load_market_data(env_cfg, seed=seed)
    return _add_features(raw, extra_features=env_cfg.get("extra_features"))


def normalize_bundle(
    frame: pd.DataFrame,
    split_bounds: Dict[str, Tuple[int, int]],
    columns: List[str],
) -> DataBundle:
    """Z-score *columns* using train-split statistics only — no look-ahead leakage."""
    t0, t1 = split_bounds["train"]
    train_slice = frame.iloc[t0:t1]
    mu = train_slice[columns].mean()
    sigma = train_slice[columns].std().replace(0.0, 1.0).fillna(1.0)

    out = frame.copy()
    out[columns] = (out[columns] - mu) / sigma

    return DataBundle(
        frame=out,
        feature_columns=columns,
        split_bounds=split_bounds,
        feature_mean=mu,
        feature_std=sigma,
    )


def build_data_bundle(config: Dict[str, Any]) -> DataBundle:
    env_cfg = config.get("env", {})
    frame = load_featured_frame(config)
    columns = feature_columns(env_cfg)

    splits = _split_bounds(
        length=len(frame),
        train_ratio=float(env_cfg.get("train_ratio", 0.7)),
        val_ratio=float(env_cfg.get("val_ratio", 0.15)),
    )
    return normalize_bundle(frame, splits, columns)
