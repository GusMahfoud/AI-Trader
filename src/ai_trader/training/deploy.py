"""Run a trained DDQN checkpoint on a split and save per-seed metrics + a demo chart."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from ai_trader.agents import build_agent
from ai_trader.env import make_env_bundle
from ai_trader.utils import get_logger
from ai_trader.viz import plot_demo_dashboard

from .evaluate import buy_and_hold_curve, evaluate_policy, run_episode

logger = get_logger(__name__)


def deploy(
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
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        metrics = evaluate_policy(env, agent, episodes=episodes, max_steps=max_steps, seed=seed)
        logger.info(
            f"[Deploy seed={seed}] avg_reward={metrics['avg_reward']:.5f} "
            f"avg_return={metrics['avg_total_return']:.3%} sharpe={metrics['avg_sharpe']:.3f}"
        )
        rows.append({"seed": seed, **metrics})

    summary_path = Path(out_dir) / f"deploy_summary_{split}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved deploy summary: {summary_path}")

    demo = run_episode(env, agent, train=False, max_steps=max_steps, seed=seeds[0])
    info = demo["final_info"]
    prices = info.get("price_history", [])
    bh = buy_and_hold_curve(prices, initial_cash=float(getattr(env, "initial_cash", 10_000.0)))

    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    demo_plot_path = plots_dir / f"demo_dashboard_{split}.png"
    plot_demo_dashboard(
        price_history=prices,
        equity_curves={"Double DQN": info.get("equity_curve", []), "Buy&Hold": bh},
        actions=info.get("action_history", []),
        output_path=demo_plot_path,
        title=f"Double DQN Deployment ({split})",
    )
    logger.info(f"Saved demo chart: {demo_plot_path}")

    train_env.close()
    val_env.close()
    test_env.close()
