"""Episode rollouts and aggregate metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def buy_and_hold_curve(price_history: List[float], initial_cash: float) -> List[float]:
    prices = np.asarray(price_history, dtype=float)
    shares = initial_cash / max(prices[0], 1e-8)
    return list(shares * prices)


def summarize_episode(info: Dict[str, Any], initial_cash: float) -> Dict[str, float]:
    equity = np.asarray(info.get("equity_curve", []), dtype=float)
    if equity.size == 0:
        return {
            "final_value": initial_cash,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "num_trades": 0.0,
        }

    rets = equity[1:] / np.maximum(equity[:-1], 1e-8) - 1.0
    mean_ret = float(np.mean(rets)) if rets.size else 0.0
    std_ret = float(np.std(rets)) if rets.size else 0.0
    # Annualize daily Sharpe with √252.
    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252.0) if std_ret > 0 else 0.0

    peaks = np.maximum.accumulate(equity)
    drawdown = (equity - peaks) / np.maximum(peaks, 1e-8)
    max_dd = float(np.min(drawdown))

    actions = info.get("action_history", [])
    trades = float(sum(1 for a in actions if a in (1, 2)))

    final_value = float(equity[-1])
    return {
        "final_value": final_value,
        "total_return": (final_value / initial_cash) - 1.0,
        "max_drawdown": max_dd,
        "sharpe": float(sharpe),
        "num_trades": trades,
    }


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

    for step in range(max_steps):
        action = agent.choose_action(obs, explore=train)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        episode_reward += float(reward)

        if train:
            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

        obs = next_obs
        final_info = info
        if done:
            break

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
        "avg_num_trades": avg("num_trades"),
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


def evaluate_random_policy(env, episodes: int, max_steps: int, seed: int) -> Dict[str, Any]:
    initial_cash = float(getattr(env, "initial_cash", 10_000.0))
    rewards: List[float] = []
    summaries: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        final_info = info
        done = False
        for _ in range(max_steps):
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)
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
