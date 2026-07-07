"""Episode rollouts and aggregate metrics."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ai_trader.env.sizing import trade_limits
from ai_trader.risk.metrics import avg_win_loss_ratio, calmar_ratio, sortino_ratio, var_cvar, win_rate


def buy_and_hold_curve(price_history: List[float], initial_cash: float) -> List[float]:
    prices = np.asarray(price_history, dtype=float)
    shares = initial_cash / max(prices[0], 1e-8)
    return list(shares * prices)


_EMPTY_SUMMARY = {
    "final_value": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
    "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
    "num_trades": 0.0, "win_rate": 0.0, "win_loss_ratio": 0.0,
    "var_95": 0.0, "cvar_95": 0.0, "exposure": 0.0,
}


def _mean_exposure(info: Dict[str, Any], equity: np.ndarray) -> float:
    """Mean |position value| / equity over the episode — 1.0 = fully deployed."""
    positions = np.asarray(info.get("position_curve", []), dtype=float)
    prices = np.asarray(info.get("price_history", []), dtype=float)
    if positions.size != equity.size or prices.size != equity.size or not equity.size:
        return 0.0
    exposure = np.abs(positions * prices) / np.maximum(np.abs(equity), 1e-8)
    return float(np.mean(exposure))


def summarize_episode(info: Dict[str, Any], initial_cash: float) -> Dict[str, float]:
    equity = np.asarray(info.get("equity_curve", []), dtype=float)
    actions: List[int] = info.get("action_history", [])

    if equity.size == 0:
        return {**_EMPTY_SUMMARY, "final_value": initial_cash}

    rets = equity[1:] / np.maximum(equity[:-1], 1e-8) - 1.0
    mean_ret = float(np.mean(rets)) if rets.size else 0.0
    std_ret = float(np.std(rets)) if rets.size else 0.0
    # Annualise daily Sharpe with √252.
    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252.0) if std_ret > 0 else 0.0

    peaks = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - peaks) / np.maximum(peaks, 1e-8)))
    var, cvar = var_cvar(equity)

    final_value = float(equity[-1])
    return {
        "final_value": final_value,
        "total_return": (final_value / initial_cash) - 1.0,
        "max_drawdown": max_dd,
        "sharpe": float(sharpe),
        "sortino": sortino_ratio(equity),
        "calmar": calmar_ratio(equity),
        "num_trades": float(sum(1 for a in actions if a in (1, 2))),
        "win_rate": win_rate(equity, actions),
        "win_loss_ratio": avg_win_loss_ratio(equity, actions),
        "var_95": var,
        "cvar_95": cvar,
        "exposure": _mean_exposure(info, equity),
    }


def _nstep_return(
    buf: List[Tuple], gamma: float
) -> Tuple[Any, int, float, Any, bool]:
    """Compute n-step return from a list of (s, a, r, s', done) transitions."""
    G = 0.0
    for i in range(len(buf) - 1, -1, -1):
        _, _, r, _, d = buf[i]
        G = r + gamma * G * (1.0 - float(d))
    s, a = buf[0][0], buf[0][1]
    s_prime = buf[-1][3]
    any_done = any(t[4] for t in buf)
    return s, a, G, s_prime, any_done


def run_episode(
    env,
    agent,
    train: bool,
    max_steps: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    episode_reward = 0.0
    episode_losses: List[float] = []
    final_info = info
    done = False
    step = 0

    n_steps: int = getattr(agent, "n_steps", 1)
    gamma: float = getattr(agent, "gamma", 0.99)
    nstep_buf: Deque[Tuple] = deque(maxlen=n_steps)

    for step in range(max_steps):
        action = agent.choose_action(obs, explore=train)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        episode_reward += float(reward)

        if train:
            nstep_buf.append((obs, action, float(reward), next_obs, done))
            if len(nstep_buf) == n_steps:
                agent.store_transition(*_nstep_return(list(nstep_buf), gamma))
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

        obs = next_obs
        final_info = info
        if done:
            break

    # Flush remaining transitions when the episode ends before the buffer fills.
    if train and nstep_buf:
        buf = list(nstep_buf)
        for start in range(1, len(buf)):
            agent.store_transition(*_nstep_return(buf[start:], gamma))

    # If we hit the loop cap before env termination, ask the env for full history.
    if not done and hasattr(env, "_info"):
        final_info = env._info(action_masked=False, full=True)

    return {
        "episode_reward": float(episode_reward),
        "episode_steps": int(step + 1),
        "average_loss": float(np.mean(episode_losses)) if episode_losses else 0.0,
        "final_info": final_info,
    }


def _avg_summaries(summaries: List[Dict[str, float]], rewards: List[float]) -> Dict[str, float]:
    avg = lambda key: float(np.mean([x[key] for x in summaries]))
    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_total_return": avg("total_return"),
        "avg_final_value": avg("final_value"),
        "avg_max_drawdown": avg("max_drawdown"),
        "avg_sharpe": avg("sharpe"),
        "avg_sortino": avg("sortino"),
        "avg_calmar": avg("calmar"),
        "avg_num_trades": avg("num_trades"),
        "avg_win_rate": avg("win_rate"),
        "avg_win_loss_ratio": avg("win_loss_ratio"),
        "avg_var_95": avg("var_95"),
        "avg_cvar_95": avg("cvar_95"),
        "avg_exposure": avg("exposure"),
    }


def evaluate_policy(env, agent, episodes: int, max_steps: int, seed: int) -> Dict[str, Any]:
    initial_cash = float(getattr(env, "initial_cash", 10_000.0))
    rewards: List[float] = []
    summaries: List[Dict[str, float]] = []

    for ep in range(episodes):
        stats = run_episode(env, agent, train=False, max_steps=max_steps, seed=seed + ep)
        rewards.append(stats["episode_reward"])
        summaries.append(summarize_episode(stats["final_info"], initial_cash=initial_cash))

    return _avg_summaries(summaries, rewards)


def _evaluate_scripted_policy(env, make_policy, episodes: int, max_steps: int, seed: int) -> Dict[str, Any]:
    """Roll out a non-learning policy and aggregate metrics.

    *make_policy* is a zero-arg factory returning a fresh ``policy(env) -> int`` per
    episode, so stateful policies (e.g. buy-and-hold) reset between episodes.
    """
    initial_cash = float(getattr(env, "initial_cash", 10_000.0))
    rewards: List[float] = []
    summaries: List[Dict[str, float]] = []

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        policy = make_policy()
        total_reward = 0.0
        final_info: Dict[str, Any] = {}
        done = False
        for _ in range(max_steps):
            _, reward, terminated, truncated, info = env.step(policy(env))
            total_reward += float(reward)
            final_info = info
            done = terminated or truncated
            if done:
                break
        if not done and hasattr(env, "_info"):
            final_info = env._info(action_masked=False, full=True)
        rewards.append(total_reward)
        summaries.append(summarize_episode(final_info, initial_cash=initial_cash))

    return _avg_summaries(summaries, rewards)


def evaluate_random_policy(env, episodes: int, max_steps: int, seed: int) -> Dict[str, Any]:
    return _evaluate_scripted_policy(
        env, lambda: (lambda e: int(e.action_space.sample())), episodes, max_steps, seed
    )


def _buy_fits(env, price: float) -> bool:
    """True if a full sizing step would actually execute (room + affordable), mode-aware."""
    step_units, max_pos, _ = trade_limits(
        env.position_sizing, price, env._cash, env._position,
        trade_size=env.trade_size, max_position=env.max_position,
        trade_fraction=env.trade_fraction, max_exposure=env.max_exposure,
        allow_short=env.allow_short,
    )
    if step_units <= 0 or env._position + step_units > max_pos:
        return False
    cost = price * (1.0 + env.slippage) * step_units * (1.0 + env.transaction_cost)
    return env._cash >= cost


def evaluate_buy_and_hold_policy(env, episodes: int, max_steps: int, seed: int) -> Dict[str, Any]:
    """Stay maximally invested, then hold — cost-inclusive, mode-aware B&H benchmark.

    Buys one sizing step whenever another fill fits under the agent's own limits
    (``max_position`` in shares mode, ``max_exposure`` in fraction mode), so it deploys
    the same capital the DQN agent is allowed and the comparison is apples-to-apples.
    """

    def make_policy():
        def policy(e) -> int:
            price = float(e._price_at(e._cursor))
            return 2 if _buy_fits(e, price) else 0

        return policy

    return _evaluate_scripted_policy(env, make_policy, episodes, max_steps, seed)
