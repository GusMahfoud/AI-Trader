"""DataBundle assembly — the public API for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from .features import FEATURE_COLUMNS, _add_features
from .loader import _load_market_data
from .splits import _split_bounds


@dataclass
class DataBundle:
    frame: pd.DataFrame
    feature_columns: List[str]
    split_bounds: Dict[str, Tuple[int, int]]
    feature_mean: pd.Series
    feature_std: pd.Series


def build_data_bundle(config: Dict[str, Any]) -> DataBundle:
    env_cfg = config.get("env", {})
    seed = int(config.get("training", {}).get("seed", 42))

    raw = _load_market_data(env_cfg, seed=seed)
    frame = _add_features(raw)

    splits = _split_bounds(
        length=len(frame),
        train_ratio=float(env_cfg.get("train_ratio", 0.7)),
        val_ratio=float(env_cfg.get("val_ratio", 0.15)),
    )

    # Normalize using training-split statistics only — prevents look-ahead leakage.
    t0, t1 = splits["train"]
    train_slice = frame.iloc[t0:t1]
    mu = train_slice[FEATURE_COLUMNS].mean()
    sigma = train_slice[FEATURE_COLUMNS].std().replace(0.0, 1.0).fillna(1.0)

    frame = frame.copy()
    frame[FEATURE_COLUMNS] = (frame[FEATURE_COLUMNS] - mu) / sigma

    return DataBundle(
        frame=frame,
        feature_columns=FEATURE_COLUMNS,
        split_bounds=splits,
        feature_mean=mu,
        feature_std=sigma,
    )
