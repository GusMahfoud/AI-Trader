# AI Trader — Double DQN Stock Trading Agent

<!-- Replace OWNER/REPO once the GitHub repo exists. -->
[![tests](https://github.com/OWNER/REPO/actions/workflows/tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Trains a **Double DQN** reinforcement-learning agent to trade AAPL using historical market data, and ships with a Streamlit dashboard for inspecting the trained policy.

Actions: `0 = hold`, `1 = sell`, `2 = buy`. Reward = scaled portfolio return − risk/position/inactivity penalties.

---

## 1. Install Miniconda

1. Download the installer: <https://www.anaconda.com/docs/getting-started/miniconda/install>
   - Windows: grab the **64-bit Python 3** `.exe`. The base Python version doesn't matter — the env file pins 3.10 separately.
2. Run the installer. On Windows: *Install for: Just Me*, skip "Add to PATH" — use **Anaconda Prompt** instead.
3. Confirm it works:
   ```bash
   conda --version
   ```

Already have Anaconda or Miniforge? Both work.

---

## 2. Setup

Run from the project root in Anaconda Prompt.

```bash
# Create + activate the environment
conda env create -f environment.yml
conda activate ai-trader

# Register the ai_trader package
pip install -e .

# Sanity check
python -c "import torch, gymnasium, streamlit; print('OK')"
```

---

## 3. Training

```bash
# Basic run — results saved to results/YYYYMMDD_HHMMSS/
python -m ai_trader --mode train

# Named run — results saved to results/my_run/
python -m ai_trader --mode train --run-id my_run

# Override specific hyperparameters without editing config.yaml
python -m ai_trader --mode train --run-id high_lr --override experiments/high_lr.yaml
```

Each run writes a self-contained directory:

```
results/my_run/
├── config.yaml          # exact config snapshot
├── checkpoints/
│   ├── double_dqn_best.pt
│   └── double_dqn_latest.pt
├── plots/
│   ├── reward_plot.png
│   └── ...
├── episode_rewards.csv
├── episode_losses.csv
├── episode_lengths.csv
└── episode_epsilons.csv
```

---

## 4. Evaluation

```bash
# Compare DDQN vs Random vs Buy & Hold on the test split
python -m ai_trader --mode compare \
  --checkpoint results/my_run/checkpoints/double_dqn_best.pt \
  --split test --episodes 10 --seeds 42

# Run the trained agent and save a dashboard plot
python -m ai_trader --mode deploy \
  --checkpoint results/my_run/checkpoints/double_dqn_best.pt \
  --split test --episodes 5 --seeds 42,43,44
```

---

## 5. Dashboard

```bash
streamlit run scripts/demo_app.py
```

Three pages: Live Agent Replay (scrub through trades with a slider), Agent vs Buy & Hold race, and an interactive Hyperparameter Explorer.

---

## 6. Configuration

All hyperparameters live in [`config.yaml`](config.yaml). Common changes:

| Key | Default | Purpose |
|---|---|---|
| `device` | `"cuda"` | `"auto"`, `"cuda"`, or `"cpu"` |
| `env.data_source` | `"yfinance"` | `"yfinance"`, `"csv"`, or `"synthetic"` |
| `env.ticker` | `"AAPL"` | Any yfinance-supported ticker |
| `training.episodes` | `5000` | Total training episodes |
| `training.eval_every` | `20` | Validate and checkpoint every N episodes |

To run a quick experiment without touching `config.yaml`, create a small override file:

```yaml
# experiments/fast_dev.yaml
env:
  data_source: "synthetic"
training:
  episodes: 100
  eval_every: 10
```

```bash
python -m ai_trader --mode train --run-id fast_dev --override experiments/fast_dev.yaml
```

---

## 7. GPU Training

The env file installs **CUDA 12.1** PyTorch by default. Verify after setup:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Driver too old (< 528.33 on Windows)?** Edit [`environment.yml`](environment.yml), change `pytorch-cuda=12.1` to `pytorch-cuda=11.8`, then:

```bash
conda deactivate
conda env remove -n ai-trader
conda env create -f environment.yml
conda activate ai-trader
pip install -e .
```

**No GPU?** Set `device: "cpu"` in `config.yaml` and remove the `pytorch-cuda` line from `environment.yml`.

---

## 8. Tests

Runs fully offline using synthetic data, finishes in seconds.

```bash
pytest
pytest tests/test_env.py          # single file
pytest -k replay_buffer           # by keyword
```

---

## 9. Project Structure

```
AI Trader/
├── config.yaml                    # all hyperparameters
├── environment.yml                # conda env spec
├── pyproject.toml                 # package metadata
│
├── src/ai_trader/
│   ├── agents/
│   │   └── double_dqn.py          # DDQN + build_agent factory
│   ├── data/                      # data pipeline
│   │   ├── loader.py              #   CSV / yfinance / synthetic loading
│   │   ├── features.py            #   RSI, SMA/EMA, momentum indicators
│   │   ├── splits.py              #   chronological train/val/test split
│   │   └── bundle.py              #   DataBundle + build_data_bundle
│   ├── env/
│   │   └── trading_env.py         # Gymnasium env + portfolio accounting
│   ├── models/
│   │   ├── q_network.py           # MLP Q-network
│   │   └── replay_buffer.py       # pre-allocated numpy ring buffer
│   ├── training/
│   │   ├── train.py               # training loop
│   │   ├── deploy.py              # single-agent eval
│   │   ├── compare.py             # DDQN vs Random vs B&H
│   │   ├── evaluate.py            # rollouts, metrics, summaries
│   │   └── logger.py              # CSV metric logger
│   ├── viz/
│   │   ├── style.py               # shared dark theme
│   │   ├── training_plots.py      # training-curve charts
│   │   └── comparison_plots.py    # deployment dashboard + comparison bars
│   └── utils/
│       ├── config.py              # YAML loader + deep_merge
│       ├── run.py                 # make_run_id
│       ├── checkpoint.py          # save/load
│       ├── seeding.py             # set_seed
│       └── torch_utils.py         # device, tensor helpers, Polyak updates
│
├── app/                           # Streamlit dashboard package
│   ├── helpers.py                 # cached loaders + episode runner
│   └── pages/
│       ├── replay.py              # Live Agent Replay page
│       ├── race.py                # Agent vs Buy & Hold page
│       └── explorer.py            # Hyperparameter Explorer page
│
├── scripts/                       # thin entry-points
│   ├── demo_app.py                # streamlit run target
│   ├── train.py
│   ├── check_actions.py
│   └── gen_plots.py               # rebuild plots: --run-dir results/my_run
│
├── results/                       # one subdirectory per run (gitignored)
│
└── tests/
    ├── unit/
    ├── integration/
    └── conftest.py
```

---

## 10. Troubleshooting

| Issue | Fix |
|---|---|
| `conda: command not found` | Use **Anaconda Prompt**, or re-run installer with *Add to PATH* |
| `ModuleNotFoundError: ai_trader` | Run `pip install -e .` inside the activated env |
| Agent only holds | Check `reward_scale` and `inactivity_penalty` in `config.yaml` are non-zero |
| `device=cuda requested but not available` | See section 7 — driver too old or `pytorch-cuda` not installed |
| yfinance download fails | Set `env.data_source: "synthetic"` for offline runs |
| Dashboard can't find checkpoints | Train first — `.pt` files are gitignored; point the sidebar to the correct run dir |
