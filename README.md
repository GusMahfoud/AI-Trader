# AI Trader — a 10-Best-Stocks Picker

[![tests](https://github.com/GusMahfoud/AI-Trader/actions/workflows/tests.yml/badge.svg)](https://github.com/GusMahfoud/AI-Trader/actions/workflows/tests.yml)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary%20%2F%20source--available-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A machine-learning stock ranker. Once a month it scores a fixed universe of
~112 liquid US large caps, picks the **10 best**, and tells you exactly what to
buy and in what dollar amounts. Built as a research project with an emphasis on
honest evaluation: purged walk-forward validation, transaction costs included,
and every strategy change gated on beating both an equal-weight portfolio and
SPY before adoption.

> **Not financial advice.** This is a research codebase. Backtests carry a
> documented survivorship bias (the universe is a fixed list of names that
> exist today) and past performance means nothing anyway.

## How the 10 picks are chosen

1. **Data** — daily OHLCV for every universe ticker via yfinance
   (split/dividend-adjusted), cached locally as Parquet.
2. **Features** — per stock, per day: momentum over 1/3/12 months (the 12-month
   signal skips the most recent month, the classic "12-1" convention),
   short-term reversal, realized volatility, and dollar-volume liquidity. Each
   feature is then converted to a **percentile across the universe that day**,
   so stocks are directly comparable and no scaler can leak future statistics.
3. **Model** — a LightGBM **LambdaRank** learning-to-rank model, trained to
   order stocks within each date by their next-month return quantile. Ranking
   relative order is more robust than predicting absolute returns.
4. **Portfolio** — hold the top 10 equal-weight, rebalance monthly, with a
   turnover buffer: an existing holding is kept while it stays in the top 20,
   which roughly halves trading costs at minor signal cost.
5. **Output** — `--mode picks` trains on all available history, scores the
   latest close, and prints tickers, company names, weights, and dollar/share
   amounts sized to your account value. Each run is appended to a local log.

Validation runs the same pipeline through **purged walk-forward folds**
(training rows whose forward-return label window would touch the test period
are dropped), against two benchmarks on identical dates and costs: the
equal-weight universe and SPY buy-and-hold. Signal quality is tracked with
rank information coefficient (IC), NDCG@K, and top-minus-bottom spread.

## Quickstart

```bash
conda env create -f environment.yml
conda activate ai-trader
pip install -e .

pytest                                   # offline, synthetic data, ~20s

# Today's top 10 (first run downloads ~112 tickers, then cached)
python -m ai_trader --mode picks --refresh-data --capital 10000

# Walk-forward backtest of the ranking strategy vs benchmarks
python -m ai_trader --mode rank_backtest
```

Experiment variants live in `experiments/*.yaml` and deep-merge over
`config.yaml` via `--override`. Long runs can also be queued to a Postgres
job table and executed by `python -m ai_trader.jobs.worker` (see
`.env.example`).

## Repository layout

```
src/ai_trader/
├── data/          # loaders, Parquet cache, cross-sectional features,
│                  # forward-return labels, purged walk-forward splits,
│                  # universe + sector/name metadata
├── models/        # LambdaRank scorer (+ legacy DQN networks/replay buffers)
├── training/      # rank_backtest, portfolio simulator, picks command,
│                  # DQN train/eval/walk-forward, report.json builders
├── risk/          # Sharpe/Sortino/Calmar/VaR + IC/NDCG signal metrics
├── env/           # legacy Gymnasium trading env (single-asset DQN track)
├── agents/        # legacy Double-DQN agent
├── jobs/          # Postgres-backed job queue + worker
├── api/           # config contract (pydantic) + FastAPI skeleton
├── viz/           # matplotlib/plotly chart helpers
└── utils/         # config merge, logging, seeding, run ids
app/               # Streamlit dashboard
experiments/       # one-hypothesis YAML overrides, gates documented in-file
scripts/           # thin CLI entry points (enqueue runs, demo app)
tests/             # unit + integration; all offline via synthetic fixtures
config.yaml        # single source of truth for every parameter
```

The `env/` + `agents/` DQN track is the project's first phase — a Double-DQN
agent trading a single ticker. It validated the evaluation harness but never
beat buy-and-hold after honest measurement, which is what motivated the pivot
to cross-sectional ranking. It is kept frozen as a reference.

## Development principles

- **One hypothesis per experiment**, with the hypothesis and a pre-committed
  decision gate written in the experiment file before the run.
- **No look-ahead**: features are backward-looking percentiles, labels are
  purged at fold boundaries, normalization never sees test data.
- **Costs always on**: every backtest pays transaction costs and slippage.
- **Negative results are kept**, in config comments and experiment files —
  rejected ideas stay documented so they don't get retried by accident.
- 300-line soft / 1000-line hard file limits, type hints, tests per module.

## License

**Free to use, not free to sell.** The code is source-available under a
non-commercial, attribution license: clone it, fork it, modify it, run it and
share it for free, as long as you keep the copyright notice and credit this
project as the original. You may **not** sell it, use it in a commercial
product or service (including paid signals, newsletters or advisory services),
or present it as your own work. See [LICENSE](LICENSE) and [TERMS.md](TERMS.md).
For commercial licensing, open an issue on this repository.

Copyright (c) 2026 Gus Mahfoud. All rights reserved.
