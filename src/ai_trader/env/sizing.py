"""Translate a discrete action into trade sizes — share-count or capital-fraction.

"shares" mode keeps the original fixed ``trade_size`` / ``max_position`` behaviour.
"fraction" mode sizes every trade as a fraction of *current equity*, so capital is
actually deployed and the same config generalises across tickers and price levels.
"""

from __future__ import annotations

from typing import Tuple

_EPS = 1e-8


def trade_limits(
    sizing: str,
    price: float,
    cash: float,
    position: int,
    *,
    trade_size: int,
    max_position: int,
    trade_fraction: float,
    max_exposure: float,
    allow_short: bool,
) -> Tuple[int, int, int]:
    """Return ``(step_units, max_position_units, min_position_units)`` for this state.

    In "fraction" mode the units are derived from current equity (cash + position·price):
    ``step_units = floor(trade_fraction · equity / price)`` and
    ``max_units = floor(max_exposure · equity / price)``.
    """
    if sizing == "fraction":
        equity = max(cash + position * price, _EPS)
        p = max(price, _EPS)
        step_units = int((trade_fraction * equity) / p)
        max_units = int((max_exposure * equity) / p)
        min_units = -max_units if allow_short else 0
        return step_units, max_units, min_units

    min_units = -max_position if allow_short else 0
    return trade_size, max_position, min_units
