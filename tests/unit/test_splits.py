"""Tests for chronological and walk-forward split bounds."""

from __future__ import annotations

import pytest

from ai_trader.data.splits import _split_bounds, walk_forward_bounds


def test_single_split_matches_fixed():
    fixed = _split_bounds(1000, 0.7, 0.15)
    wf = walk_forward_bounds(1000, 1, 0.7, 0.15)
    assert wf == [fixed]


def test_walk_forward_fold_count():
    folds = walk_forward_bounds(2000, 4, 0.7, 0.15)
    assert len(folds) == 4


def test_walk_forward_is_chronological_within_each_fold():
    folds = walk_forward_bounds(2000, 4, 0.7, 0.15)
    for b in folds:
        assert b["train"][0] < b["train"][1] == b["val"][0]
        assert b["val"][1] == b["test"][0] < b["test"][1]
        assert b["test"][1] <= 2000


def test_walk_forward_windows_slide_forward_with_contiguous_tests():
    folds = walk_forward_bounds(2000, 4, 0.7, 0.15)
    for prev, cur in zip(folds, folds[1:]):
        assert cur["train"][0] > prev["train"][0]
        # test segments tile the series: each fold's test begins where the last ended,
        # to within one row of integer-flooring slack.
        assert abs(cur["test"][0] - prev["test"][1]) <= 1


def test_walk_forward_rejects_bad_n_splits():
    with pytest.raises(ValueError):
        walk_forward_bounds(2000, 0, 0.7, 0.15)
