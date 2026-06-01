# AI Trader — Double DQN Stock Trading Agent

<!-- Replace OWNER/REPO once the GitHub repo exists. -->
[![tests](https://github.com/OWNER/REPO/actions/workflows/tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Trains a **Double DQN** reinforcement-learning agent to trade AAPL using historical market data, and ships with a Streamlit dashboard for inspecting the trained policy.

Actions: `0 = hold`, `1 = sell`, `2 = buy`. Reward = scaled portfolio return − risk/position/inactivity penalties.

---

## 1. Install Conda

This project targets **Miniconda** (lightweight, ~500 MB) — the full Anaconda distribution also works but is overkill. The PyTorch channel and `python=3.10` pinned in [`environment.yml`](environment.yml) work with any Miniconda release from the last couple of years.

1. Download the Miniconda installer for your OS: <https://www.anaconda.com/docs/getting-started/miniconda/install>
   - Windows: grab the **64-bit Python 3 installer** (`.exe`). The base Python version doesn't need to match 3.10 — the env file pins that separately.
2. Run the installer. On Windows the recommended options are *Install for: Just Me* and *Register Miniconda as my default Python*. You can skip "Add to PATH" — use the **Anaconda Prompt** that ships with the installer instead.
3. Open Anaconda Prompt (or any shell where `conda --version` works) and confirm:
   ```bash
   conda --version
   ```

Already have Anaconda or Miniforge? Both work — no need to reinstall.

---

## 2. Quick Start (Full Run)

Run these from the `AISE4030/` directory inside Anaconda Prompt.

```bash
# Create + activate the environment (pulls PyTorch + all pip deps)
conda env create -f environment.yml
conda activate aise4030-trading

# Install this project as an editable package (registers `ai_trader`)
pip install -e .

# Sanity check
python -c "import torch, gymnasium, streamlit; print('OK')"

# Train (writes checkpoints + plots to results/double_dqn/)
python -m ai_trader --mode train

# Evaluate against a random baseline + buy-and-hold on the test split
python -m ai_trader --mode compare --checkpoint results/double_dqn/double_dqn_best.pt --split test --episodes 10 --seeds 42

# Launch the interactive demo
streamlit run scripts/demo_app.py
```

> The `scripts/*.py` entry points (`scripts/train.py`, `scripts/demo_app.py`, etc.) also work **without** `pip install -e .` — they bootstrap `src/` onto `sys.path` automatically.

---

## 3. CLI Modes

```bash
python -m ai_trader --mode <mode> [options]
```

| Mode | Purpose |
|---|---|
| `train` | Train the Double DQN agent using `config.yaml` |
| `deploy --checkpoint <path>` | Run a saved agent on `--split {train,val,test}` over `--episodes N --seeds S1,S2,...` |
| `compare --checkpoint <path>` | DDQN vs Random vs Buy & Hold side-by-side on the selected split |

---

## 4. Configuration

All hyperparameters live in [`config.yaml`](config.yaml). Common changes:

- `output_dir` — where checkpoints, CSVs, and plots are written (default `results/double_dqn`)
- `env.data_source` — `"yfinance"` (default), `"csv"`, or `"synthetic"` (offline)
- `training.episodes` — default `5000`
- `device` — `"auto"`, `"cuda"`, or `"cpu"`

---

## 5. Outputs

Each `output_dir` ends up with:

- `double_dqn_best.pt` / `double_dqn_latest.pt` — checkpoints (gitignored)
- `episode_rewards.csv`, `episode_losses.csv`, `episode_lengths.csv`, `episode_epsilons.csv`
- `reward_plot.png`, `loss_plot.png`, `epsilon_plot.png`, `length_plot.png`
- `deploy_summary_{split}.csv`, `demo_dashboard_{split}.png` (deploy mode)
- `compare_metrics_{split}.csv`, `compare_dashboard_{split}.png`, `compare_bars_{split}.png` (compare mode)

---

## 6. GPU Training

The env file installs **CUDA 12.1** PyTorch by default. To verify the GPU is detected after `conda env create`:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"
```

If `CUDA: True`, training will use the GPU automatically (`device: "cuda"` is set in [`config.yaml`](config.yaml)).

**If your driver is older than 528.33 on Windows** (or 525.60 on Linux), edit [`environment.yml`](environment.yml) and change `pytorch-cuda=12.1` to `pytorch-cuda=11.8`, then recreate the env:

```bash
conda deactivate
conda env remove -n aise4030-trading
conda env create -f environment.yml
conda activate aise4030-trading
pip install -e .
```

**No GPU at all?** Set `device: "cpu"` in `config.yaml` and remove the `pytorch-cuda` line from `environment.yml`.

---

## 7. Running Tests

The test suite uses pytest with synthetic OHLCV data — it runs fully offline and finishes in a few seconds.

```bash
pytest
```

What's covered: replay buffer ring + sampling, Q-network forward shapes, trading env reset/step contract + reward signs + chronological splits, DDQN action selection + train step + checkpoint roundtrip, and the evaluation helpers (`summarize_episode`, `buy_and_hold_curve`).

---

## 8. Project Structure

```
AISE4030/
├── README.md
├── pyproject.toml             # package metadata + console entry
├── environment.yml            # conda spec
├── requirements.txt           # pip alternative
├── config.yaml                # all hyperparameters
│
├── src/ai_trader/
│   ├── cli.py                 # argparse entry point
│   ├── __main__.py            # `python -m ai_trader`
│   ├── agents/
│   │   └── double_dqn.py      # DDQN + build_agent factory
│   ├── env/
│   │   ├── data.py            # data loading, features, splits
│   │   └── trading_env.py     # Gymnasium env + portfolio accounting
│   ├── models/
│   │   ├── q_network.py       # MLP (256-256-128)
│   │   └── replay_buffer.py   # pre-allocated numpy ring buffer
│   ├── training/
│   │   ├── train.py           # training loop
│   │   ├── deploy.py          # single-agent eval
│   │   ├── compare.py         # DDQN vs Random vs B&H
│   │   ├── evaluate.py        # rollouts, metrics, summaries
│   │   └── logger.py          # CSV metric logger
│   ├── viz/
│   │   ├── style.py           # shared dark palette + rcParams + plotly layout
│   │   ├── training_plots.py  # training-curve charts
│   │   └── comparison_plots.py# deployment dashboard + comparison bars
│   └── utils/
│       ├── config.py          # YAML loader + ensure_dir
│       ├── seeding.py         # set_seed
│       ├── torch_utils.py     # device selection, tensor helpers, target-net updates
│       └── checkpoint.py      # save/load
│
├── scripts/                   # thin convenience entries (work without pip install)
│   ├── train.py
│   ├── demo_app.py            # Streamlit dashboard
│   ├── check_actions.py       # sanity check action distribution
│   └── gen_plots.py           # rebuild plots from CSVs
│
├── results/
│   └── double_dqn/            # checkpoints + CSVs + plots
│
└── tests/                     # pytest suite (synthetic data, offline)
```

---

## 9. Troubleshooting

| Issue | Fix |
|---|---|
| `conda: command not found` | Open **Anaconda Prompt** instead of regular cmd/PowerShell, or re-run the installer with *Add to PATH* checked |
| `ModuleNotFoundError: ai_trader` | Run `pip install -e .` inside the activated env, or use a `scripts/*.py` entry point |
| Agent only holds | Check `reward_scale` and `inactivity_penalty` in `config.yaml` are non-zero |
| `device=cuda requested but torch.cuda.is_available() is False` | Recreate the env (see section 6) — most likely `pytorch-cuda` wasn't installed, or your NVIDIA driver is too old |
| yfinance download fails | Set `env.data_source: "synthetic"` in `config.yaml` for offline runs |
| Demo can't find checkpoints | Train first — `.pt` files are gitignored |
