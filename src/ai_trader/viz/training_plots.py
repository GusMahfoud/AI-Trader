"""Training-curve plots: reward, loss, epsilon, episode length."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import C_CYAN, C_ORANGE, apply_matplotlib_style


def plot_training_curves(
    rewards: List[float],
    losses: List[float],
    output_dir: Path,
    eval_rewards: Optional[List[Optional[float]]] = None,
    epsilons: Optional[List[float]] = None,
    episode_lengths: Optional[List[int]] = None,
) -> None:
    apply_matplotlib_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _plot_metric(
        values=rewards,
        title="EPISODE REWARDS",
        ylabel="Reward",
        output_path=output_path / "reward_plot.png",
        eval_values=eval_rewards,
        color=C_CYAN,
    )
    _plot_metric(
        values=losses,
        title="Q-NETWORK LOSS",
        ylabel="Loss",
        output_path=output_path / "loss_plot.png",
        color=C_ORANGE,
    )
    if epsilons is not None:
        _plot_metric(
            values=epsilons,
            title="EXPLORATION SCHEDULE (EPSILON)",
            ylabel="Epsilon",
            output_path=output_path / "epsilon_plot.png",
            color="#a78bfa",
        )
    if episode_lengths is not None:
        _plot_metric(
            values=episode_lengths,
            title="EPISODE LENGTH",
            ylabel="Steps",
            output_path=output_path / "length_plot.png",
            color="#34d399",
        )


def _plot_metric(
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    eval_values: Optional[List[Optional[float]]] = None,
    color: str = C_CYAN,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    episodes = np.arange(1, len(values) + 1)
    ax.plot(episodes, values, linewidth=0.8, alpha=0.35, color=color, label="Raw")

    if len(values) > 20:
        window = max(10, len(values) // 50)
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window, len(values) + 1),
            smoothed,
            linewidth=2.0,
            color=color,
            label=f"Avg (window={window})",
        )

    if eval_values:
        ep_ev = [(i + 1, v) for i, v in enumerate(eval_values) if v is not None]
        if ep_ev:
            eps, evs = zip(*ep_ev)
            ax.scatter(
                eps,
                evs,
                s=30,
                alpha=0.95,
                label="Eval",
                zorder=5,
                color=C_ORANGE,
                edgecolors="white",
                linewidths=0.4,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
