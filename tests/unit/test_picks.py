"""Tests for the live picks command (synthetic universe, offline)."""

from __future__ import annotations

import pandas as pd
import pytest

from ai_trader.training.picks import generate_picks


@pytest.fixture
def picks_config(synthetic_config):
    cfg = {**synthetic_config}
    cfg["env"] = {
        **synthetic_config["env"],
        "synthetic_length": 600,
        "universe": ["AAA", "BBB", "CCC", "DDD", "EEE"],
    }
    cfg["rank"] = {
        "top_k": 3,
        "label_horizon": 10,
        "label_bins": 3,
        "model": "momentum",
        "score_feature": "mom_21",
        "weighting": "equal",
    }
    return cfg


def test_picks_momentum(picks_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = generate_picks(picks_config, out_dir=str(tmp_path), capital=20_000)
    assert len(out) == 3
    assert out["weight"].sum() == pytest.approx(1.0)
    assert (tmp_path / "picks.csv").exists()
    # Scores must be ranked descending.
    assert out["score"].is_monotonic_decreasing
    # Position sizing: dollars = weight * capital; shares = dollars / close.
    assert out["target_dollars"].sum() == pytest.approx(20_000, rel=1e-3)
    row = out.iloc[0]
    assert row["shares"] == pytest.approx(row["target_dollars"] / row["close"], rel=1e-2)


def test_picks_appends_paper_log(picks_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate_picks(picks_config, out_dir=str(tmp_path))
    generate_picks(picks_config, out_dir=str(tmp_path))  # same as_of: replaces, not duplicates

    log = pd.read_csv(tmp_path / "paper_trading" / "picks_log.csv")
    assert len(log) == 3
    assert set(["as_of", "ticker", "weight", "target_dollars", "shares"]) <= set(log.columns)


def test_picks_lambdarank_inverse_vol(picks_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    picks_config["rank"] = {
        **picks_config["rank"],
        "model": "lambdarank",
        "weighting": "inverse_vol",
    }
    out = generate_picks(picks_config, out_dir=str(tmp_path))
    assert len(out) == 3
    assert out["weight"].sum() == pytest.approx(1.0)
    # All picks carry the same as-of date (the latest trading day).
    assert out["as_of"].nunique() == 1
    saved = pd.read_csv(tmp_path / "picks.csv")
    assert list(saved["ticker"]) == list(out["ticker"])
