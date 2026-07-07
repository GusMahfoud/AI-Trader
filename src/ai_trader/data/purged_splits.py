"""Purged, forward-only walk-forward splits over trading dates.

Forward-only expanding windows: each fold trains on everything strictly before
its test block, minus a PURGE GAP of `label_horizon` dates — a training row at
date t carries a label spanning [t, t+H], so the last H training dates before
the test boundary would peek into test prices and must be dropped (Lopez de
Prado, *Advances in Financial ML*, ch. 7). No embargo is needed here because
training data never comes after a test window in forward-only splits.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Bounds = Tuple[int, int]  # half-open [lo, hi) indices into the unique-date array


def purged_walk_forward_bounds(
    n_dates: int,
    n_splits: int,
    label_horizon: int,
    test_ratio: float = 0.4,
    min_train_dates: int = 60,
) -> List[Dict[str, Bounds]]:
    """K forward-only folds whose test blocks tile the trailing ``test_ratio`` of dates.

    Returns per-fold ``{"train": (lo, hi), "test": (lo, hi)}`` index bounds where
    ``train.hi = test.lo - label_horizon`` (the purge gap).
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    test_total = int(n_dates * test_ratio)
    test_width = test_total // n_splits
    if test_width < 1:
        raise ValueError(
            f"Not enough dates ({n_dates}) for {n_splits} test folds at test_ratio={test_ratio}."
        )

    folds: List[Dict[str, Bounds]] = []
    first_test_start = n_dates - test_width * n_splits
    for k in range(n_splits):
        test_lo = first_test_start + k * test_width
        test_hi = test_lo + test_width if k < n_splits - 1 else n_dates
        train_hi = test_lo - label_horizon
        if train_hi < min_train_dates:
            raise ValueError(
                f"Fold {k}: only {train_hi} training dates after the purge gap "
                f"(need >= {min_train_dates}). Reduce n_splits/test_ratio or add data."
            )
        folds.append({"train": (0, train_hi), "test": (test_lo, test_hi)})
    return folds


def assert_no_label_overlap(
    folds: Sequence[Dict[str, Bounds]], label_horizon: int
) -> None:
    """Raise if any fold's training labels could touch its test window."""
    for k, fold in enumerate(folds):
        train_hi = fold["train"][1]
        test_lo = fold["test"][0]
        if train_hi + label_horizon > test_lo:
            raise AssertionError(
                f"Fold {k} leaks: last train label window [{train_hi - 1}, "
                f"{train_hi - 1 + label_horizon}] crosses test start {test_lo}."
            )
