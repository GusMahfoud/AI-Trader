"""Position-sizing limits: shares mode is fixed, fraction mode scales with equity/price."""

from __future__ import annotations

from ai_trader.env.sizing import trade_limits


def _shares(**kw):
    return trade_limits(
        "shares", price=100.0, cash=10_000.0, position=0,
        trade_size=1, max_position=10, trade_fraction=0.25, max_exposure=1.0,
        allow_short=False, **kw,
    )


def test_shares_mode_returns_fixed_limits():
    step, max_units, min_units = _shares()
    assert step == 1
    assert max_units == 10
    assert min_units == 0


def test_shares_mode_short_allows_negative_min():
    step, max_units, min_units = trade_limits(
        "shares", price=100.0, cash=10_000.0, position=0,
        trade_size=1, max_position=10, trade_fraction=0.25, max_exposure=1.0,
        allow_short=True,
    )
    assert min_units == -10


def test_fraction_mode_step_scales_with_equity_and_price():
    # equity = 10_000, price = 100 → step = floor(0.25 * 10_000 / 100) = 25
    step, max_units, min_units = trade_limits(
        "fraction", price=100.0, cash=10_000.0, position=0,
        trade_size=1, max_position=10, trade_fraction=0.25, max_exposure=1.0,
        allow_short=False,
    )
    assert step == 25
    assert max_units == 100  # floor(1.0 * 10_000 / 100)
    assert min_units == 0


def test_fraction_mode_caps_total_exposure():
    # max_exposure 0.5 → max_units = floor(0.5 * 10_000 / 100) = 50
    _, max_units, _ = trade_limits(
        "fraction", price=100.0, cash=10_000.0, position=0,
        trade_size=1, max_position=10, trade_fraction=0.25, max_exposure=0.5,
        allow_short=False,
    )
    assert max_units == 50


def test_fraction_mode_includes_position_value_in_equity():
    # equity = cash 5_000 + position 50 * price 100 = 10_000 → max_units = 100
    _, max_units, _ = trade_limits(
        "fraction", price=100.0, cash=5_000.0, position=50,
        trade_size=1, max_position=10, trade_fraction=0.25, max_exposure=1.0,
        allow_short=False,
    )
    assert max_units == 100


def test_fraction_mode_higher_price_means_fewer_units():
    step, _, _ = trade_limits(
        "fraction", price=2_000.0, cash=10_000.0, position=0,
        trade_size=1, max_position=10, trade_fraction=0.25, max_exposure=1.0,
        allow_short=False,
    )
    assert step == 1  # floor(0.25 * 10_000 / 2_000)
