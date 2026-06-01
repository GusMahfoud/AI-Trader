"""Compare Double DQN against a random baseline and buy-and-hold on a chosen split."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ai_trader.agents import build_agent
from ai_trader.env import make_env_bundle
from ai_trader.viz import plot_comparison_summary, plot_demo_dashboard

from .evaluate import (
    buy_and_hold_curve,
    evaluate_policy,
    evaluate_random_policy,
    run_episode,
)


def compare(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    split: str,
    episodes: int,
    seeds: List[int],
    out_dir: str,
) -> None:
    max_steps = int(cfg["training"]["max_steps_per_episode"])

    train_env, val_env, test_env = make_env_bundle(cfg)
    env = {"train": train_env, "val": val_env, "test": test_env}[split]

    agent = build_agent(
        cfg=cfg,
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
    )
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}")

    all_agent: List[Dict[str, float]] = []
    all_random: List[Dict[str, float]] = []
    for seed in seeds:
        all_agent.append(evaluate_policy(env, agent, episodes=episodes, max_steps=max_steps, seed=seed))
        all_random.append(evaluate_random_policy(env, episodes=episodes, max_steps=max_steps, seed=seed))

    def _avg(results: List[Dict[str, float]]) -> Dict[str, float]:
        return {k: float(np.mean([r[k] for r in results])) for k in results[0].keys()}

    ddqn_metrics = _avg(all_agent)
    random_metrics = _avg(all_random)

    print(f"=== Compare Summary (averaged over {len(seeds)} seed(s)) ===")
    for name, m in [("DoubleDQN", ddqn_metrics), ("Random", random_metrics)]:
        print(
            f"{name:10s} reward={m['avg_reward']:.5f} return={m['avg_total_return']:.3%} "
            f"sharpe={m['avg_sharpe']:.3f} drawdown={m['avg_max_drawdown']:.3%}"
        )

    # Equity-curve dashboard using the first seed.
    demo_seed = seeds[0]
    demo = run_episode(env, agent, train=False, max_steps=max_steps, seed=demo_seed)
    info = demo["final_info"]
    prices = info.get("price_history", [])
    initial_cash = float(getattr(env, "initial_cash", 10_000.0))
    bh = buy_and_hold_curve(prices, initial_cash=initial_cash)

    dashboard_path = Path(out_dir) / f"compare_dashboard_{split}.png"
    plot_demo_dashboard(
        price_history=prices,
        equity_curves={"Double DQN": info.get("equity_curve", []), "Buy&Hold": bh},
        actions=info.get("action_history", []),
        output_path=dashboard_path,
        title=f"Double DQN vs Buy&Hold ({split})",
    )
    print(f"Saved compare chart: {dashboard_path}")

    metrics_path = Path(out_dir) / f"compare_metrics_{split}.csv"
    summary_rows = [
        {"agent": "double_dqn", **ddqn_metrics},
        {"agent": "random", **random_metrics},
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved compare metrics: {metrics_path}")

    bars_path = Path(out_dir) / f"compare_bars_{split}.png"
    plot_comparison_summary(
        metrics={"Double DQN": ddqn_metrics, "Random": random_metrics},
        output_path=bars_path,
    )
    print(f"Saved comparison bar chart: {bars_path}")

    train_env.close()
    val_env.close()
    test_env.close()
