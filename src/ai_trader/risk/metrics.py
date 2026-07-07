"""Financial risk and performance metrics computed from episode equity curves."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def sortino_ratio(equity: np.ndarray, annualize: float = 252.0) -> float:
    """Sharpe-like ratio penalising only downside deviation — more relevant for trading."""
    rets = equity[1:] / np.maximum(equity[:-1], 1e-8) - 1.0
    if rets.size == 0:
        return 0.0
    mean_ret = float(np.mean(rets))
    downside = rets[rets < 0.0]
    # With <2 down days the downside deviation is undefined; dividing by an epsilon
    # would explode the ratio to ~1e8 and poison any average it enters. Return 0.
    if downside.size < 2:
        return 0.0
    downside_std = float(np.std(downside))
    if downside_std <= 0.0:
        return 0.0
    return float((mean_ret / downside_std) * np.sqrt(annualize))


def calmar_ratio(equity: np.ndarray, annualize: float = 252.0) -> float:
    """Annualised return divided by maximum drawdown depth."""
    rets = equity[1:] / np.maximum(equity[:-1], 1e-8) - 1.0
    if rets.size == 0:
        return 0.0
    ann_return = float((1.0 + np.mean(rets)) ** annualize - 1.0)
    peaks = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - peaks) / np.maximum(peaks, 1e-8)))
    if abs(max_dd) < 1e-8:
        return 0.0
    return ann_return / abs(max_dd)


def win_rate(equity: np.ndarray, actions: List[int]) -> float:
    """Fraction of buy/sell steps that were followed by a positive portfolio move."""
    if len(equity) < 2 or not actions:
        return 0.0
    rets = equity[1:] / np.maximum(equity[:-1], 1e-8) - 1.0
    trade_rets = [rets[i] for i, a in enumerate(actions) if a in (1, 2) and i < len(rets)]
    if not trade_rets:
        return 0.0
    return float(sum(1 for r in trade_rets if r > 0) / len(trade_rets))


def avg_win_loss_ratio(equity: np.ndarray, actions: List[int]) -> float:
    """Mean winning trade return divided by mean losing trade return (absolute value)."""
    if len(equity) < 2 or not actions:
        return 0.0
    rets = equity[1:] / np.maximum(equity[:-1], 1e-8) - 1.0
    trade_rets = [rets[i] for i, a in enumerate(actions) if a in (1, 2) and i < len(rets)]
    wins = [r for r in trade_rets if r > 0]
    losses = [r for r in trade_rets if r < 0]
    if not wins or not losses:
        return 0.0
    return float(np.mean(wins) / (abs(np.mean(losses)) + 1e-8))


def equity_summary(equity: np.ndarray, annualize: float = 252.0) -> Dict[str, float]:
    """All equity-curve metrics for one deterministic backtest path."""
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2:
        return {
            "total_return": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "max_drawdown": 0.0, "var_95": 0.0, "cvar_95": 0.0,
        }
    rets = equity[1:] / np.maximum(equity[:-1], 1e-8) - 1.0
    std = float(np.std(rets))
    sharpe = float(np.mean(rets) / std * np.sqrt(annualize)) if std > 0 else 0.0
    peaks = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - peaks) / np.maximum(peaks, 1e-8)))
    var, cvar = var_cvar(equity)
    return {
        "total_return": float(equity[-1] / equity[0] - 1.0),
        "sharpe": sharpe,
        "sortino": sortino_ratio(equity, annualize),
        "calmar": calmar_ratio(equity, annualize),
        "max_drawdown": max_dd,
        "var_95": var,
        "cvar_95": cvar,
    }


def var_cvar(equity: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Historical Value-at-Risk and Conditional VaR (Expected Shortfall).

    Returns negative numbers representing losses, e.g. VaR = -0.02 means
    on the worst 5% of days the loss was at least 2%.
    """
    rets = equity[1:] / np.maximum(equity[:-1], 1e-8) - 1.0
    if rets.size < 2:
        return 0.0, 0.0
    sorted_rets = np.sort(rets)
    cutoff = max(int(np.floor((1.0 - confidence) * len(sorted_rets))), 1)
    var = float(sorted_rets[cutoff - 1])
    cvar = float(np.mean(sorted_rets[:cutoff]))
    return var, cvar
