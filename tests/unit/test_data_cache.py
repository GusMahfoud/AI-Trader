"""Tests for the Parquet market-data cache."""

from __future__ import annotations

import pandas as pd
import pytest

from ai_trader.data.cache import cache_key, cache_path, read_cache, write_cache


def test_cache_key_is_deterministic_and_safe():
    cfg = {"ticker": "AAPL", "start_date": "2015-01-01", "end_date": None}
    key = cache_key(cfg)
    assert key == cache_key(cfg)
    assert key == "AAPL_2015-01-01_latest"
    assert "/" not in key and "\\" not in key


def test_cache_key_distinguishes_ranges():
    a = cache_key({"ticker": "AAPL", "start_date": "2015-01-01", "end_date": "2020-01-01"})
    b = cache_key({"ticker": "AAPL", "start_date": "2015-01-01", "end_date": "2021-01-01"})
    assert a != b


def test_read_cache_missing_returns_none(tmp_path):
    assert read_cache(tmp_path / "nope.parquet") is None


def test_write_then_read_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "close": [1.0, 2.0, 3.0]})
    path = cache_path({"ticker": "TST", "start_date": "2020-01-01", "end_date": "2020-02-01"}, cache_dir=tmp_path)

    assert write_cache(path, frame) is True
    loaded = read_cache(path)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, frame)
