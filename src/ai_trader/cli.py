"""Command-line entry point: train / deploy / compare."""

from __future__ import annotations

import argparse
from typing import List

from .training import compare, deploy, train
from .utils import ensure_dir, load_config, set_seed


def _parse_seeds(raw: str) -> List[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one seed is required.")
    return [int(v) for v in vals]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/deploy/compare the Double DQN trading agent.")
    parser.add_argument("--mode", choices=["train", "deploy", "compare"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for deploy/compare.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes for deploy/compare evaluation.")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["training"]["seed"]))
    out_dir = ensure_dir(cfg.get("output_dir", "results/double_dqn"))

    if args.mode == "train":
        train(cfg, out_dir=out_dir)
        return

    if args.mode == "deploy":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode deploy")
        deploy(
            cfg,
            checkpoint_path=args.checkpoint,
            split=args.split,
            episodes=int(args.episodes),
            seeds=_parse_seeds(args.seeds),
            out_dir=out_dir,
        )
        return

    if args.mode == "compare":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode compare")
        compare(
            cfg,
            checkpoint_path=args.checkpoint,
            split=args.split,
            episodes=int(args.episodes),
            seeds=_parse_seeds(args.seeds),
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
