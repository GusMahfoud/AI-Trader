# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Quality Rules

These rules apply to every file in this repo. They are non-negotiable.

- **300-line soft limit** — if a file is approaching 300 lines, split it before adding more code.
- **1 000-line hard limit** — no file may ever exceed 1 000 lines. No exceptions.
- **Single responsibility** — each module does one thing. Avoid "utils.py" god-files; name modules by what they contain (`features.py`, `splits.py`, `per_buffer.py`).
- **No comments explaining WHAT** — only add a comment when the WHY is non-obvious (hidden constraint, subtle invariant, specific bug workaround). If removing the comment wouldn't confuse a future reader, don't write it.
- **Type hints on all public functions** — required. CI will enforce `mypy`.
- **No `print()` in library code** (`src/ai_trader/`) — use the structured logger from `utils/logging.py` once it exists; until then, raise or return instead of printing.
- **Test every new module** — each new file under `src/` must have a corresponding test file under `tests/unit/`.
- **No look-ahead bias** — feature normalization must use train-split statistics only. Never fit a scaler on val/test data.

## Folder Structure

```
AI Trader/
├── src/ai_trader/
│   ├── agents/      # One file per agent class
│   ├── data/        # loader.py, features.py, splits.py  (replaces env/data.py)
│   ├── env/         # Gymnasium environments only
│   ├── models/      # Network architectures + replay buffers
│   ├── training/    # Train / eval / compare orchestrators
│   ├── risk/        # Position sizing, stop-loss, VaR metrics
│   ├── viz/         # Charts (one file per chart family)
│   ├── api/         # FastAPI routes (future)
│   └── utils/       # Small, focused helpers
├── app/             # Streamlit dashboard as a proper package
│   ├── main.py
│   ├── pages/       # One file per dashboard page
│   └── components/  # Reusable UI widgets
├── scripts/         # Thin CLI entry-points only (< 50 lines each)
├── tests/
│   ├── unit/
│   └── integration/
├── config/          # base.yaml + experiment overrides (future)
└── docker/          # Dockerfile, compose (future)
```

New directories must fit this layout. If a new concern doesn't map to an existing directory, discuss placement in the PR description before creating an ad-hoc folder.

## Environment & install

The project targets Python 3.10 inside a conda env named `ai-trader`. Recreate it with `conda env create -f environment.yml && conda activate ai-trader`, then `pip install -e .` to register the `ai_trader` package.

You don't strictly need the editable install — `scripts/_bootstrap.py` (imported as `_bootstrap` at the top of every script under `scripts/`) prepends `src/` to `sys.path`. Same trick lives in `tests/conftest.py`, which is why pytest works from a fresh checkout.

`device: "cuda"` in `config.yaml` **fails loudly** — `get_device()` in `src/ai_trader/utils/torch_utils.py` raises `RuntimeError` if CUDA isn't available. Use `device: "auto"` if you want CPU fallback, or `device: "cpu"` to force CPU.

## Commands

```bash
# Train (5000 episodes by default — see config.yaml)
python -m ai_trader --mode train

# Evaluate a checkpoint on a split
python -m ai_trader --mode deploy  --checkpoint results/double_dqn/double_dqn_best.pt --split test --episodes 10 --seeds 42,43,44

# DDQN vs Random vs Buy & Hold side-by-side
python -m ai_trader --mode compare --checkpoint results/double_dqn/double_dqn_best.pt --split test --episodes 10 --seeds 42

# Streamlit dashboard
streamlit run scripts/demo_app.py

# Tests — offline, synthetic data, finishes in seconds
pytest
pytest tests/test_env.py::test_reset_returns_correct_shape   # single test
pytest -k replay_buffer                                      # by keyword
```

The `--config` flag overrides `config.yaml`. `--seeds` is comma-separated.

## Architecture

**Single config, single CLI.** `config.yaml` is the only source of truth for hyperparameters. `src/ai_trader/cli.py` dispatches `--mode {train,deploy,compare}` to the matching module under `src/ai_trader/training/`. There is no per-experiment config override system — fork the YAML if you need variants.

**Data pipeline (`src/ai_trader/env/data.py`).** `build_data_bundle()` is the single entry point. It loads OHLCV from `csv` / `yfinance` / `synthetic` (selected by `env.data_source`), computes ~11 engineered features (returns, log returns, SMA/EMA gaps, vol, momentum, RSI), and chronologically splits into train/val/test. **Feature normalization uses train-split mean/std only** — this is deliberate to prevent look-ahead leakage. If you add features, follow the same pattern and add them to `FEATURE_COLUMNS`. yfinance MultiIndex column flattening is handled here.

**Environment (`src/ai_trader/env/trading_env.py`).** Gymnasium env with discrete actions `0=hold, 1=sell, 2=buy`. State = flattened `(lookback_window × num_features) + 4 portfolio scalars` (pos_frac, cash_frac, exposure, unrealized). Reward = `reward_scale · step_return − risk_penalty · drawdown − position_penalty · |pos|/max_pos − inactivity_penalty · holdstreak/10`. The inactivity term is what prevents the agent from collapsing to all-holds — set the four reward-shaping coefficients in `config.yaml`, all are non-trivially load-bearing.

Two perf details that look like clutter but matter: prices/features are pre-extracted to numpy arrays in `__init__` (step() never touches pandas iloc), and `_info()` only copies full history lists when an episode terminates (`full=True`). Don't reintroduce per-step list copies.

`make_env_bundle(cfg)` builds train/val/test envs sharing one DataBundle — that's the entry point used by every training/eval module.

**Agent (`src/ai_trader/agents/double_dqn.py`).** Double DQN: online net selects argmax action, target net evaluates it (decouples selection from valuation, reduces overestimation bias). Soft target updates via `tau` (Polyak averaging in `utils/torch_utils.py`). SmoothL1 loss, Adam optimizer, gradient clipping at 1.0, and a `StepLR` that halves LR every 1500 episodes — late-stage fine-tune over oscillation. ε-greedy with multiplicative decay; `end_episode()` advances both ε and the LR scheduler.

**Replay buffer (`src/ai_trader/models/replay_buffer.py`).** Pre-allocated contiguous numpy ring buffer, not a deque of dataclasses. `sample()` is a single fancy-indexing pass. Keep this shape if you extend it (e.g., for prioritized replay) — switching to per-item Python objects is a measurable regression.

**Training loop (`src/ai_trader/training/train.py`).** Per-episode: rollout via `run_episode()` from `evaluate.py`, ε decay, and every `eval_every` episodes a val rollout. Checkpointing: `double_dqn_latest.pt` written every eval, `double_dqn_best.pt` written only when val avg_reward improves. The metric logger writes CSVs to `output_dir` on each eval and at the end; final plots come from `viz/training_plots.py`.

**Eval modules.** `evaluate.py` has `run_episode` (shared between train + eval), `summarize_episode` (Sharpe is annualized √252, drawdown is min of equity-vs-peak), `evaluate_policy` (greedy), `evaluate_random_policy` (action_space.sample). `deploy.py` and `compare.py` are thin wrappers that call these and produce CSV summaries + plots. The Streamlit demo (`scripts/demo_app.py`) reuses `summarize_episode` and `buy_and_hold_curve` from this module — keep their signatures stable.

**Viz (`src/ai_trader/viz/`).** All charts share `style.py` (dark palette, matplotlib rcParams, `plotly_dark_layout`). `training_plots.py` is for the training curves saved at end-of-train; `comparison_plots.py` covers the deployment dashboard and DDQN-vs-Random bars.

## Testing notes

`tests/conftest.py` provides a `synthetic_config` fixture — a minimal config using synthetic OHLCV, tiny replay buffer, 30-step episodes. All tests inherit it; don't add tests that hit yfinance. The test suite covers replay buffer, Q-network shapes, env reset/step contract + reward signs + chronological splits, DDQN action selection + train step + checkpoint roundtrip, and the eval helpers.
