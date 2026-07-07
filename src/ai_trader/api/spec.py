"""Pydantic ModelSpec: the validated, user-configurable model contract.

Mirrors config.yaml one-to-one so the platform can validate a config before any
compute runs. Bridges to the existing dict pipeline via ``from_config`` /
``to_config`` — the training core keeps consuming plain dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

RegimeFeature = Literal["volume_regime", "vol_regime", "price_position"]


class EnvSpec(BaseModel):
    """Data source, splits, trading dynamics, and reward shaping."""

    model_config = ConfigDict(extra="forbid")

    data_source: Literal["csv", "yfinance", "synthetic"] = "yfinance"
    data_path: str = "data/AAPL.csv"
    ticker: str = "AAPL"
    start_date: Optional[str] = "2015-01-01"
    end_date: Optional[str] = None
    synthetic_length: int = Field(default=3000, ge=100)
    refresh_data: bool = False

    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)

    extra_features: List[RegimeFeature] = Field(default_factory=list)
    universe: List[str] = Field(default_factory=list)
    auto_adjust: bool = False

    initial_cash: float = Field(default=10_000.0, gt=0.0)
    transaction_cost: float = Field(default=0.0003, ge=0.0)
    slippage: float = Field(default=0.0001, ge=0.0)
    position_sizing: Literal["shares", "fraction"] = "fraction"
    max_position: int = Field(default=10, ge=1)
    trade_size: int = Field(default=1, ge=1)
    trade_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    max_exposure: float = Field(default=1.0, gt=0.0, le=1.0)
    allow_short: bool = False
    stop_loss: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    take_profit: Optional[float] = Field(default=None, gt=0.0)

    lookback_window: int = Field(default=20, ge=2)
    random_start: bool = True
    reward_scale: float = Field(default=100.0, gt=0.0)
    risk_penalty: float = Field(default=0.002, ge=0.0)
    position_penalty: float = Field(default=0.0002, ge=0.0)
    inactivity_penalty: float = Field(default=0.005, ge=0.0)
    underexposure_penalty: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check_cross_fields(self) -> "EnvSpec":
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio + val_ratio must leave room for a test split "
                f"(got {self.train_ratio} + {self.val_ratio} >= 1)"
            )
        if self.trade_fraction > self.max_exposure:
            raise ValueError(
                f"trade_fraction ({self.trade_fraction}) cannot exceed "
                f"max_exposure ({self.max_exposure})"
            )
        if self.data_source == "csv" and not self.data_path:
            raise ValueError("data_path is required when data_source='csv'")
        return self


class AgentSpec(BaseModel):
    """Double-DQN hyperparameters: network, replay, exploration, and returns."""

    model_config = ConfigDict(extra="forbid")

    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    learning_rate: float = Field(default=0.0005, gt=0.0)
    batch_size: int = Field(default=64, ge=1)
    buffer_size: int = Field(default=100_000, ge=1)
    epsilon_start: float = Field(default=1.0, gt=0.0, le=1.0)
    epsilon_min: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.995, gt=0.0, le=1.0)
    tau: float = Field(default=0.001, gt=0.0, le=1.0)
    network: Literal["mlp", "dueling"] = "mlp"
    replay: Literal["uniform", "per"] = "per"
    n_steps: int = Field(default=2, ge=1)
    per_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    per_beta_start: float = Field(default=0.4, gt=0.0, le=1.0)
    per_beta_frames: int = Field(default=100_000, ge=1)

    @model_validator(mode="after")
    def _check_cross_fields(self) -> "AgentSpec":
        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot exceed buffer_size ({self.buffer_size})"
            )
        if self.epsilon_min > self.epsilon_start:
            raise ValueError(
                f"epsilon_min ({self.epsilon_min}) cannot exceed "
                f"epsilon_start ({self.epsilon_start})"
            )
        return self


class TrainingSpec(BaseModel):
    """Training-loop schedule: episodes, evaluation cadence, early stopping, folds."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    episodes: int = Field(default=2500, ge=1)
    max_steps_per_episode: int = Field(default=252, ge=1)
    eval_every: int = Field(default=20, ge=0)
    eval_episodes: int = Field(default=10, ge=1)
    early_stop_patience: int = Field(default=25, ge=0)
    early_stop_warmup: int = Field(default=300, ge=0)
    n_splits: int = Field(default=4, ge=2)


class RankSpec(BaseModel):
    """Cross-sectional rank-backtest settings (universe top-K strategies)."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1)
    rebalance_days: int = Field(default=21, ge=1)
    label_horizon: int = Field(default=21, ge=1)
    test_ratio: float = Field(default=0.4, gt=0.0, lt=1.0)
    score_feature: str = "mom_12_1"
    benchmark_ticker: str = "SPY"
    model: Literal["momentum", "lambdarank"] = "momentum"
    label_bins: int = Field(default=4, ge=2, le=10)
    feature_set: Literal["v1", "v2"] = "v1"
    buffer_k: int = Field(default=0, ge=0)
    weighting: Literal["equal", "inverse_vol"] = "equal"
    dd_brake: float = Field(default=0.0, ge=0.0, lt=1.0)


class ModelSpec(BaseModel):
    """Top-level model contract — the shape the frontend posts and workers consume."""

    model_config = ConfigDict(extra="forbid")

    device: Literal["auto", "cuda", "cpu"] = "auto"
    env: EnvSpec = Field(default_factory=EnvSpec)
    agent: AgentSpec = Field(default_factory=AgentSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    rank: RankSpec = Field(default_factory=RankSpec)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ModelSpec":
        """Validate a merged config dict (e.g. the output of ``load_config``)."""
        return cls.model_validate(cfg)

    def to_config(self) -> Dict[str, Any]:
        """Produce the plain dict the existing train()/walk_forward() core consumes."""
        return self.model_dump(mode="json")
