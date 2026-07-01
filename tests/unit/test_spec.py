"""ModelSpec contract: defaults, round-trips, and rejection of invalid configs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_trader.api.spec import ModelSpec
from ai_trader.utils import deep_merge, load_config


def test_defaults_mirror_champion() -> None:
    spec = ModelSpec()
    assert spec.agent.network == "mlp"
    assert spec.agent.replay == "per"
    assert spec.agent.n_steps == 2
    assert spec.env.position_sizing == "fraction"
    assert spec.env.risk_penalty == 0.002


def test_repo_config_with_champion_override_validates() -> None:
    cfg = load_config("config.yaml", override_path="experiments/champion.yaml")
    spec = ModelSpec.from_config(cfg)
    assert spec.agent.network == "mlp"
    assert spec.env.trade_fraction == 0.25


def test_synthetic_config_round_trips(synthetic_config) -> None:
    spec = ModelSpec.from_config(synthetic_config)
    cfg = spec.to_config()
    assert cfg["env"]["data_source"] == "synthetic"
    assert cfg["training"]["episodes"] == 2
    # Defaults are filled in for keys the fixture omits.
    assert cfg["agent"]["network"] == "mlp"
    assert ModelSpec.from_config(cfg) == spec


@pytest.mark.parametrize(
    "override",
    [
        {"agent": {"network": "transformer"}},
        {"agent": {"replay": "priority"}},
        {"agent": {"gamma": 1.5}},
        {"agent": {"n_steps": 0}},
        {"agent": {"batch_size": 512, "buffer_size": 64}},
        {"agent": {"epsilon_min": 0.9, "epsilon_start": 0.1}},
        {"env": {"data_source": "bloomberg"}},
        {"env": {"train_ratio": 0.9, "val_ratio": 0.2}},
        {"env": {"transaction_cost": -0.01}},
        {"env": {"trade_fraction": 0.8, "max_exposure": 0.5}},
        {"env": {"extra_features": ["nonexistent_feature"]}},
        {"env": {"stop_loss": 1.5}},
        {"training": {"episodes": 0}},
        {"training": {"n_splits": 1}},
        {"device": "tpu"},
    ],
)
def test_invalid_configs_rejected(synthetic_config, override) -> None:
    bad = deep_merge(synthetic_config, override)
    with pytest.raises(ValidationError):
        ModelSpec.from_config(bad)


def test_unknown_keys_rejected(synthetic_config) -> None:
    bad = deep_merge(synthetic_config, {"agent": {"learning_rat": 0.001}})
    with pytest.raises(ValidationError):
        ModelSpec.from_config(bad)
