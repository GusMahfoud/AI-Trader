"""Double DQN agent with ε-greedy exploration and soft target updates."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai_trader.models import QNetwork, ReplayBuffer
from ai_trader.utils import (
    get_device,
    hard_update,
    load_checkpoint,
    save_checkpoint,
    soft_update,
    to_tensor,
)


class DoubleDQNAgent:
    """Double DQN: online net picks next-action argmax, target net evaluates.

    Reduces the overestimation bias of vanilla DQN by decoupling action
    selection from value estimation in the bootstrap target.
    """

    def __init__(self, config: Dict[str, Any]):
        self.device = get_device(config.get("device", "auto"))

        self.state_dim = int(config["state_dim"])
        self.action_dim = int(config["action_dim"])
        self.gamma = float(config["gamma"])
        self.batch_size = int(config["batch_size"])
        self.epsilon = float(config["epsilon_start"])
        self.epsilon_min = float(config.get("epsilon_min", 0.05))
        self.epsilon_decay = float(config.get("epsilon_decay", 0.995))
        self.tau = float(config.get("tau", 0.001))

        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        hard_update(self.target_net, self.q_net)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=float(config["learning_rate"]))

        # Halve LR every 1500 episodes so late-training fine-tunes instead of oscillating.
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.5)

        self.replay_buffer = ReplayBuffer(int(config["buffer_size"]), self.state_dim)
        self._loss_fn = nn.SmoothL1Loss()

    def choose_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_tensor = to_tensor(state, self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = to_tensor(states, self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = to_tensor(rewards, self.device)
        next_states_t = to_tensor(next_states, self.device)
        dones_t = to_tensor(dones, self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: online net picks the argmax action, target net evaluates it.
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states_t), dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self._loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping guards against exploding updates on volatile market days.
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        soft_update(self.target_net, self.q_net, self.tau)
        return float(loss.item())

    def end_episode(self) -> float:
        """Decay ε and step the LR scheduler. Returns the new ε."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.lr_scheduler.step()
        return self.epsilon

    def save(self, filepath: str) -> None:
        save_checkpoint(self.q_net, self.optimizer, filepath)

    def load(self, filepath: str) -> None:
        load_checkpoint(self.q_net, self.optimizer, filepath, self.device)
        hard_update(self.target_net, self.q_net)


def build_agent(cfg: Dict[str, Any], state_dim: int, action_dim: int) -> DoubleDQNAgent:
    agent_cfg = {
        **cfg["agent"],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "learning_rate": float(cfg["agent"].get("learning_rate", 5e-4)),
        "device": str(cfg.get("device", "auto")).lower(),
    }
    return DoubleDQNAgent(agent_cfg)
