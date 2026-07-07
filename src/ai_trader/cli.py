"""Command-line entry point: train / deploy / compare."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml

# torch and pyarrow bundle conflicting native DLLs on Windows; if torch loads
# first, the parquet cache write crashes with an access violation. Load pyarrow
# before anything that imports torch.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

from .training import compare, deploy, generate_picks, rank_backtest, train, walk_forward
from .utils import ensure_dir, get_logger, load_config, make_run_id, set_seed

logger = get_logger(__name__)


def _parse_seeds(raw: str) -> List[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one seed is required.")
    return [int(v) for v in vals]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/deploy/compare the Double DQN trading agent.")
    parser.add_argument("--mode", choices=["train", "deploy", "compare", "walk_forward", "rank_backtest", "picks"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for deploy/compare.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes for deploy/compare evaluation.")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--override", type=str, default=None, help="Path to a YAML override file deep-merged on top of --config.")
    parser.add_argument("--run-id", type=str, default=None, dest="run_id", help="Human-readable run name (auto-generated if omitted).")
    parser.add_argument("--refresh-data", action="store_true", dest="refresh_data", help="Bypass the on-disk data cache and re-download.")
    args = parser.parse_args()

    cfg = load_config(args.config, override_path=args.override)
    if args.refresh_data:
        cfg.setdefault("env", {})["refresh_data"] = True
    set_seed(int(cfg["training"]["seed"]))

    run_id = make_run_id(args.run_id)
    if Path(f"results/{run_id}").exists():
        logger.warning("results/%s already exists — its contents will be overwritten.", run_id)
    out_dir = ensure_dir(f"results/{run_id}")

    # Snapshot the fully-merged config (base + overrides) so the run dir
    # is self-contained and can be used directly with --config later.
    snapshot_path = Path(out_dir) / "config.yaml"
    with snapshot_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Run ID : {run_id}")
    logger.info(f"Out dir: {out_dir}")

    if args.mode == "train":
        train(cfg, out_dir=out_dir)
        return

    if args.mode == "walk_forward":
        walk_forward(cfg, out_dir=out_dir)
        return

    if args.mode == "rank_backtest":
        rank_backtest(cfg, out_dir=out_dir)
        return

    if args.mode == "picks":
        generate_picks(cfg, out_dir=out_dir)
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
