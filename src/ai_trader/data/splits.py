"""Chronological train/val/test splitting with ratio validation."""

from __future__ import annotations

from typing import Dict, List, Tuple

Bounds = Dict[str, Tuple[int, int]]


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not (0.4 < train_ratio < 0.95):
        raise ValueError("env.train_ratio must be in (0.4, 0.95).")
    if not (0.0 < val_ratio < 0.4):
        raise ValueError("env.val_ratio must be in (0.0, 0.4).")
    if train_ratio + val_ratio >= 0.98:
        raise ValueError("env.train_ratio + env.val_ratio must be < 0.98.")


def _split_bounds(length: int, train_ratio: float, val_ratio: float) -> Bounds:
    _validate_ratios(train_ratio, val_ratio)

    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }


def walk_forward_bounds(
    length: int, n_splits: int, train_ratio: float, val_ratio: float
) -> List[Bounds]:
    """Rolling train/val/test windows with contiguous, non-overlapping test segments.

    Each fold keeps the same train/val/test proportions as the fixed split and slides
    forward by exactly one test-window width, so the K test segments tile the series.
    With ``n_splits == 1`` this is identical to :func:`_split_bounds`.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")
    _validate_ratios(train_ratio, val_ratio)
    if n_splits == 1:
        return [_split_bounds(length, train_ratio, val_ratio)]

    test_frac = 1.0 - train_ratio - val_ratio
    # window_len so that window + (K-1) test-width steps span the full series.
    window_len = int(length / (1.0 + (n_splits - 1) * test_frac))
    step = int(window_len * test_frac)
    if step < 1 or window_len < 10:
        raise ValueError("Series too short for the requested n_splits / ratios.")

    train_len = int(window_len * train_ratio)
    val_len = int(window_len * (train_ratio + val_ratio))

    folds: List[Bounds] = []
    for i in range(n_splits):
        s = i * step
        end = min(s + window_len, length)
        folds.append(
            {
                "train": (s, s + train_len),
                "val": (s + train_len, s + val_len),
                "test": (s + val_len, end),
            }
        )
    return folds
