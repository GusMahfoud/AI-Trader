"""Kelly-criterion position sizing.

Standalone building block for converting a rolling (win rate, win/loss ratio)
estimate into a position size. Not yet wired into the env — sizing changes the
trade dynamics and must be validated as its own experiment.
"""

from __future__ import annotations

import math


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Full-Kelly fraction f* = p - (1 - p) / b for win prob *p* and payoff ratio *b*.

    Returns a value that can be negative when the edge is unfavourable; callers
    typically clamp to non-negative via :func:`half_kelly_fraction`.
    """
    if win_loss_ratio <= 0.0:
        return 0.0
    p = float(win_rate)
    b = float(win_loss_ratio)
    return (p * (b + 1.0) - 1.0) / b


def half_kelly_fraction(win_rate: float, win_loss_ratio: float, cap: float = 1.0) -> float:
    """Half-Kelly fraction clamped to ``[0, cap]`` — the usual conservative default."""
    if cap < 0.0:
        raise ValueError("cap must be non-negative.")
    raw = 0.5 * kelly_fraction(win_rate, win_loss_ratio)
    return float(min(max(raw, 0.0), cap))


def kelly_position_size(
    equity: float,
    price: float,
    win_rate: float,
    win_loss_ratio: float,
    max_position: int,
    fraction_cap: float = 1.0,
) -> int:
    """Target unit count from a half-Kelly capital fraction, clamped to ``[0, max_position]``."""
    if equity <= 0.0 or price <= 0.0 or max_position <= 0:
        return 0
    fraction = half_kelly_fraction(win_rate, win_loss_ratio, cap=fraction_cap)
    units = math.floor((fraction * equity) / price)
    return int(min(max(units, 0), max_position))
