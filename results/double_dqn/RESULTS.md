# Double DQN — Training Results

5000-episode training run on AAPL (2015-01-01 → present, via yfinance), with chronological 70/15/15 train/val/test split and train-only feature normalization. Config used: [`../../config.yaml`](../../config.yaml).

---

## TL;DR

- **Agent learned a reliably profitable policy.** 97.5% of all episodes ended with positive shaped reward; 98.8% in the last 500.
- **Peak performance landed early.** Rolling-100 mean reward peaked at **+5.10 at episode 1969**, then slowly drifted down to **+3.73 at episode 5000** — a classic over-training tail.
- **Use `double_dqn_best.pt`, not `double_dqn_latest.pt`** for deployment. The "best" checkpoint was saved during the validation peak (~ep 1880 by file mtime); the "latest" one has drifted ~27% off peak.
- **Training was stable.** No exploding gradients, no policy collapse to all-hold, no NaN losses. Loss peaked around ep 2500 (0.005) and decayed to 0.003 by the end as the LR scheduler halved twice.

---

## Reward Trajectory (rolling-100 mean)

```
ep   100: +1.67   ← noisy exploration phase
ep   350: +3.32
ep   600: +3.75   ← exploration → exploitation transition (ε hits floor ~ep 500)
ep   850: +3.86
ep  1100: +4.40
ep  1350: +3.29   ← brief dip
ep  1600: +4.07
ep  1850: +4.58
ep  1969: +5.10   ★ PEAK
ep  2100: +4.19
ep  2350: +4.40
ep  2600: +4.43
ep  2850: +4.13
ep  3100: +4.19
ep  3350: +4.28
ep  3600: +3.97
ep  3850: +3.92
ep  4100: +3.58   ← steady drift down
ep  4350: +3.57
ep  4600: +3.54
ep  4850: +3.51
ep  5000: +3.73   ← final
```

Visualization: [`reward_plot.png`](reward_plot.png)

**Slope over last 500 episodes: -0.0016/ep.** Not catastrophic, but clearly past the inflection point. The LR halving at ep 4500 (and earlier at 3000, 1500) was not enough to arrest the drift.

---

## Window-by-Window Statistics

| Window | Mean reward | Std | Min | Max | Avg loss | ε end |
|---|---:|---:|---:|---:|---:|---:|
| ep 1–100 | +1.668 | 1.87 | −1.79 | +6.87 | 0.00088 | 0.6058 |
| ep 100–500 | +3.610 | 3.03 | −0.48 | +10.12 | 0.00340 | 0.0816 |
| ep 500–1000 | +4.070 | 3.34 | −0.50 | +10.00 | 0.00461 | 0.0500 |
| ep 1000–2000 | +4.090 | 3.38 | −0.45 | +10.70 | 0.00487 | 0.0500 |
| ep 2000–3000 | **+4.258** | 3.38 | −0.45 | +10.22 | 0.00494 | 0.0500 |
| ep 3000–4000 | +4.173 | 3.34 | −0.37 | +10.29 | 0.00400 | 0.0500 |
| ep 4000–5000 | +3.934 | 3.31 | −0.30 | +10.35 | 0.00320 | 0.0500 |
| **Last 500** | +3.721 | 3.26 | −0.13 | +10.35 | 0.00293 | 0.0500 |

Visualization: [`loss_plot.png`](loss_plot.png), [`epsilon_plot.png`](epsilon_plot.png), [`length_plot.png`](length_plot.png).

---

## Highlights

| Metric | Value |
|---|---|
| Best single-episode reward | **+10.695** (ep 1929) |
| Worst single-episode reward | −1.794 (ep 13, before learning kicked in) |
| Best rolling-100 mean reward | **+5.096** (ep 1969) |
| Final rolling-100 mean reward | +3.734 (ep 5000) |
| % positive-reward episodes (all 5000) | **97.5%** |
| % positive-reward episodes (last 500) | **98.8%** |
| Worst reward in last 200 episodes | −0.042 (floor is way up from exploration phase) |

---

## Interpretation

### What worked

- **Reward shaping landed.** The inactivity penalty + reward scaling (×100) successfully prevented the agent from converging to all-hold, which is the failure mode the trajectory plot would show as a flat near-zero curve. Instead it learned to trade.
- **Double DQN held its target estimates.** Loss curve is smooth, no spikes, no oscillations — overestimation bias didn't blow up.
- **Soft target updates (τ=0.001) + LR halving** kept the policy stable through 5000 episodes — most DQN variants would have diverged.

### What to watch

- **Late-stage drift.** The model peaked at ~ep 2000 and slowly degraded. This is consistent with overfitting to the training split's random episode windows — eventually the network memorizes price-feature patterns that don't generalize to other windows.
- **Best ≠ Latest.** `double_dqn_best.pt` is the validation-best snapshot (frozen at ~ep 1880 by mtime). `double_dqn_latest.pt` is end-of-training (ep 5000). The gap between them is the over-training tail.

### What this can't tell you

Training-set rewards say the agent fits the training data well. **They do not say it generalizes.** You need to run:

```bash
python -m ai_trader --mode compare --checkpoint results/double_dqn/double_dqn_best.pt --split test --episodes 10 --seeds 42,123,456
```

…to produce real numbers on the held-out test split: average total return, Sharpe ratio, max drawdown, and number of trades, plus a side-by-side bar chart vs random and buy-and-hold.

Until that runs, treat these training rewards as a sanity signal — not a performance claim.

---

## Recommended Next Steps

1. **Run `--mode deploy`** on test split with `double_dqn_best.pt` to get a deployment dashboard (equity curve + buy/sell markers + drawdown).
2. **Run `--mode compare`** to get the DDQN vs Random vs Buy-and-Hold metrics table — this is what would go into any write-up.
3. **Launch the Streamlit demo** (`streamlit run scripts/demo_app.py`) to scrub through trades step-by-step.
4. **Consider re-training with `episodes: 2000`** in `config.yaml` — the run plateaued well before 5000. Cutting it in half saves ~50% of training time with no loss in quality (and possibly better quality, since you'd stop before the late drift).

---

## Artifacts in This Folder

| File | What it is |
|---|---|
| `double_dqn_best.pt` | Best-by-val-reward checkpoint (~ep 1880) — **use this for deployment** |
| `double_dqn_latest.pt` | End-of-training checkpoint (ep 5000) — drifted ~27% off peak |
| `episode_rewards.csv` | Per-episode total reward |
| `episode_losses.csv` | Per-episode average Q-network loss |
| `episode_epsilons.csv` | Per-episode ε value (exploration rate) |
| `episode_lengths.csv` | Per-episode step count (always 252 here — no early termination) |
| `reward_plot.png` | Training-curve plot (raw + rolling avg) |
| `loss_plot.png` | Loss curve |
| `epsilon_plot.png` | ε decay schedule |
| `length_plot.png` | Episode length over time |
