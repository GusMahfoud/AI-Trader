"""Tests for feature engineering and opt-in extra features."""

from __future__ import annotations

import numpy as np

from ai_trader.data.features import (
    BASE_FEATURES,
    EXTRA_FEATURES,
    _add_features,
    feature_columns,
)
from ai_trader.data.loader import _generate_synthetic_data


def _frame():
    return _generate_synthetic_data(length=400, seed=7)


def test_feature_columns_base_only_by_default():
    assert feature_columns({}) == BASE_FEATURES


def test_feature_columns_appends_known_extras_only():
    cfg = {"extra_features": ["price_position", "not_a_feature", "vol_regime"]}
    cols = feature_columns(cfg)
    assert cols == [*BASE_FEATURES, "price_position", "vol_regime"]


def test_base_features_present_and_finite():
    out = _add_features(_frame())
    for col in BASE_FEATURES:
        assert col in out.columns
    assert np.isfinite(out[BASE_FEATURES].to_numpy()).all()
    assert not any(extra in out.columns for extra in EXTRA_FEATURES)


def test_extra_features_added_when_requested():
    out = _add_features(_frame(), extra_features=EXTRA_FEATURES)
    for col in EXTRA_FEATURES:
        assert col in out.columns
    assert np.isfinite(out[EXTRA_FEATURES].to_numpy()).all()
    assert len(out) > 100
