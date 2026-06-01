"""DoubleDQNAgent: action selection, training step, checkpoint roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_trader.agents import DoubleDQNAgent, build_agent
from ai_trader.env import make_env_bundle


def _make_agent(synthetic_config) -> DoubleDQNAgent:
    train_env, _, _ = make_env_bundle(synthetic_config)
    return build_agent(
        synthetic_config,
        state_dim=int(train_env.observation_space.shape[0]),
        action_dim=int(train_env.action_space.n),
    )


def test_choose_action_returns_valid_action(synthetic_config):
    agent = _make_agent(synthetic_config)
    train_env, _, _ = make_env_bundle(synthetic_config)
    obs, _ = train_env.reset(seed=0)
    action = agent.choose_action(obs, explore=False)
    assert action in (0, 1, 2)


def test_train_step_no_op_until_buffer_has_batch(synthetic_config):
    agent = _make_agent(synthetic_config)
    # Buffer empty → train_step should be a no-op (returns None).
    assert agent.train_step() is None


def test_train_step_returns_loss_once_buffer_is_full(synthetic_config):
    agent = _make_agent(synthetic_config)
    train_env, _, _ = make_env_bundle(synthetic_config)
    obs, _ = train_env.reset(seed=0)

    for _ in range(agent.batch_size + 2):
        action = agent.choose_action(obs, explore=True)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = train_env.reset(seed=1)

    loss = agent.train_step()
    assert loss is not None
    assert np.isfinite(loss)


def test_end_episode_decays_epsilon_and_steps_lr(synthetic_config):
    agent = _make_agent(synthetic_config)
    eps_before = agent.epsilon
    lr_before = agent.optimizer.param_groups[0]["lr"]

    new_eps = agent.end_episode()

    assert new_eps <= eps_before
    assert new_eps >= agent.epsilon_min
    # LR scheduler doesn't drop on the first step (StepLR with step_size=1500), so LR is unchanged.
    assert agent.optimizer.param_groups[0]["lr"] == lr_before


def test_checkpoint_roundtrip(tmp_path: Path, synthetic_config):
    agent = _make_agent(synthetic_config)
    train_env, _, _ = make_env_bundle(synthetic_config)
    obs, _ = train_env.reset(seed=0)

    ckpt_path = tmp_path / "ckpt.pt"
    agent.save(str(ckpt_path))
    assert ckpt_path.exists()

    other = _make_agent(synthetic_config)
    other.load(str(ckpt_path))

    # Both agents now share weights → same greedy action.
    a1 = agent.choose_action(obs, explore=False)
    a2 = other.choose_action(obs, explore=False)
    assert a1 == a2
