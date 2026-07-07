"""LightGBM LambdaRank scorer for cross-sectional stock ranking.

Learns to order stocks *within each date* by forward-return quantile bins.
Per-date grouping is load-bearing: pooled regression on the same features is
the documented failure mode. Hyperparameters are pre-committed (conservative,
regularization over capacity) per the overfitting discipline in the plan —
do not tune them against walk-forward results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.data.cross_features import RANK_COLUMNS
from ai_trader.data.labels import LABEL_BIN_COL

# Pre-committed 2026-07-07, before any walk-forward evaluation.
DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "lambdarank",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "verbose": -1,
}


class LambdaRankScorer:
    """Fit on a labeled feature panel; emit per-row ranking scores."""

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        try:
            from lightgbm import LGBMRanker
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for rank.model='lambdarank'. "
                "Install with: pip install lightgbm"
            ) from exc

        self.feature_columns = list(feature_columns or RANK_COLUMNS)
        merged = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}
        self._model = LGBMRanker(**merged)
        self._fitted = False

    def fit(self, panel: pd.DataFrame) -> "LambdaRankScorer":
        """Train on a long panel with date, feature, and label-bin columns.

        Rows with missing labels (label horizon reaching past the data) are
        dropped; groups are the per-date row counts, so the model only ever
        compares stocks against same-date peers.
        """
        frame = panel.dropna(subset=[LABEL_BIN_COL, *self.feature_columns])
        frame = frame.sort_values("date")
        if frame.empty:
            raise ValueError("No labeled rows to fit the ranker on.")

        groups = frame.groupby("date", sort=True).size().to_numpy()
        self._model.fit(
            frame[self.feature_columns],
            frame[LABEL_BIN_COL].astype(int),
            group=groups,
        )
        self._fitted = True
        return self

    def score(self, panel: pd.DataFrame) -> np.ndarray:
        """Per-row ranking scores (higher = better expected relative return)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")
        return np.asarray(self._model.predict(panel[self.feature_columns]))
