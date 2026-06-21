"""Regenerate training plots from saved CSVs (no retraining required).

Usage:
    python scripts/gen_plots.py --run-dir results/20240604_143022
"""

import _bootstrap  # noqa: F401

import argparse
import csv
from pathlib import Path

from ai_trader.viz import plot_training_curves


def _read_col(path: Path, col: str):
    with path.open(encoding="utf-8") as f:
        return [float(row[col]) for row in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate training plots from a run directory.")
    parser.add_argument("--run-dir", required=True, help="Path to a results/<run_id>/ directory.")
    args = parser.parse_args()

    results_dir = Path(args.run_dir)
    if not results_dir.exists():
        raise SystemExit(f"Run directory not found: {results_dir.resolve()}")

    rewards = _read_col(results_dir / "episode_rewards.csv", "episode_reward")
    losses = _read_col(results_dir / "episode_losses.csv", "episode_loss")
    epsilons = _read_col(results_dir / "episode_epsilons.csv", "epsilon")
    lengths = _read_col(results_dir / "episode_lengths.csv", "episode_length")

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_training_curves(
        rewards=rewards,
        losses=losses,
        output_dir=plots_dir,
        epsilons=epsilons,
        episode_lengths=[int(x) for x in lengths],
    )
    print(f"Saved plots to {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
