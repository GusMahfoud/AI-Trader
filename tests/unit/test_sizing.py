"""Tests for Kelly position sizing."""

from __future__ import annotations

import pytest

from ai_trader.risk.sizing import half_kelly_fraction, kelly_fraction, kelly_position_size


def test_kelly_fraction_known_value():
    # p=0.6, b=1.0 -> 0.6*2 - 1 = 0.2
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)


def test_kelly_fraction_negative_edge():
    assert kelly_fraction(0.4, 1.0) < 0.0


def test_kelly_fraction_invalid_ratio_is_zero():
    assert kelly_fraction(0.6, 0.0) == 0.0


def test_half_kelly_is_half_and_clamped_non_negative():
    assert half_kelly_fraction(0.6, 1.0) == pytest.approx(0.1)
    assert half_kelly_fraction(0.4, 1.0) == 0.0


def test_half_kelly_respects_cap():
    assert half_kelly_fraction(0.99, 50.0, cap=0.25) == 0.25


def test_position_size_clamped_to_max_position():
    size = kelly_position_size(
        equity=10_000.0, price=10.0, win_rate=0.99, win_loss_ratio=50.0, max_position=5
    )
    assert size == 5


def test_position_size_zero_when_no_edge():
    size = kelly_position_size(
        equity=10_000.0, price=10.0, win_rate=0.4, win_loss_ratio=1.0, max_position=5
    )
    assert size == 0


def test_position_size_guards_bad_inputs():
    assert kelly_position_size(0.0, 10.0, 0.6, 2.0, 5) == 0
    assert kelly_position_size(10_000.0, 0.0, 0.6, 2.0, 5) == 0
    assert kelly_position_size(10_000.0, 10.0, 0.6, 2.0, 0) == 0
