"""Chronological train/val/test splitting with ratio validation."""

from __future__ import annotations

from typing import Dict, Tuple


def _split_bounds(
    length: int, train_ratio: float, val_ratio: float
) -> Dict[str, Tuple[int, int]]:
    if not (0.4 < train_ratio < 0.95):
        raise ValueError("env.train_ratio must be in (0.4, 0.95).")
    if not (0.0 < val_ratio < 0.4):
        raise ValueError("env.val_ratio must be in (0.0, 0.4).")
    if train_ratio + val_ratio >= 0.98:
        raise ValueError("env.train_ratio + env.val_ratio must be < 0.98.")

    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }
