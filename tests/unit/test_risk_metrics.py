"""Tests for the risk metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trader.risk.metrics import avg_win_loss_ratio, calmar_ratio, sortino_ratio, var_cvar, win_rate


@pytest.fixture
def flat_equity():
    return np.ones(100) * 10_000.0


@pytest.fixture
def growing_equity():
    rng = np.random.default_rng(42)
    # Upward trend with noise so there are realistic minor drawdowns for Calmar.
    return np.linspace(10_000, 12_000, 252) + rng.normal(0, 30, 252)


@pytest.fixture
def declining_equity():
    rng = np.random.default_rng(42)
    return np.linspace(10_000, 8_000, 252) + rng.normal(0, 30, 252)


def test_sortino_positive_trend(growing_equity):
    assert sortino_ratio(growing_equity) > 0


def test_sortino_negative_trend(declining_equity):
    assert sortino_ratio(declining_equity) < 0


def test_sortino_flat(flat_equity):
    result = sortino_ratio(flat_equity)
    assert result == pytest.approx(0.0, abs=1e-3)


def test_sortino_no_down_days_is_finite():
    # Monotonically rising equity has zero downside observations; the ratio is
    # undefined and must return 0.0, not explode against an epsilon denominator.
    equity = np.linspace(10_000, 11_000, 50)
    assert sortino_ratio(equity) == 0.0


def test_sortino_single_down_day_is_finite():
    equity = np.linspace(10_000, 11_000, 50)
    equity[25] = equity[24] * 0.99  # exactly one down step
    result = sortino_ratio(equity)
    assert abs(result) < 1e6


def test_calmar_positive_trend(growing_equity):
    assert calmar_ratio(growing_equity) > 0


def test_calmar_flat(flat_equity):
    assert calmar_ratio(flat_equity) == pytest.approx(0.0, abs=1e-3)


def test_win_rate_all_buys_on_up_days():
    equity = np.array([10000.0, 10100.0, 10200.0, 10300.0])
    actions = [2, 2, 2]  # all buys, all up days
    assert win_rate(equity, actions) == pytest.approx(1.0)


def test_win_rate_no_trades():
    equity = np.linspace(10000, 11000, 10)
    actions = [0] * 9  # all holds
    assert win_rate(equity, actions) == pytest.approx(0.0)


def test_win_rate_empty():
    assert win_rate(np.array([]), []) == 0.0


def test_avg_win_loss_ratio_basic():
    equity = np.array([10000.0, 10100.0, 9900.0, 10050.0])
    actions = [2, 1, 2]
    result = avg_win_loss_ratio(equity, actions)
    assert result >= 0


def test_var_cvar_shape():
    equity = np.linspace(10000, 9000, 252)
    var, cvar = var_cvar(equity)
    assert isinstance(var, float)
    assert isinstance(cvar, float)


def test_cvar_leq_var():
    equity = np.linspace(10000, 9000, 252) + np.random.default_rng(0).normal(0, 50, 252)
    var, cvar = var_cvar(equity)
    # CVaR (expected shortfall) should be at least as bad as VaR
    assert cvar <= var + 1e-8


def test_var_cvar_empty():
    var, cvar = var_cvar(np.array([10000.0]))
    assert var == 0.0 and cvar == 0.0
