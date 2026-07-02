"""Worker execution: a synthetic walk-forward job runs end-to-end offline."""

from __future__ import annotations

from ai_trader.jobs import MemoryJobStore
from ai_trader.jobs.worker import process_one


def test_process_one_empty_queue_returns_false(tmp_path) -> None:
    assert process_one(MemoryJobStore(), base_cfg={}, results_root=str(tmp_path)) is False


def test_walk_forward_job_succeeds_and_stores_report(synthetic_config, tmp_path) -> None:
    synthetic_config["training"]["n_splits"] = 2
    store = MemoryJobStore()
    run_id = store.enqueue("walk_forward", synthetic_config, user_id="u1")

    assert process_one(store, base_cfg={}, results_root=str(tmp_path)) is True

    run = store.get(run_id)
    assert run.status == "succeeded"
    assert run.run_dir is not None and run_id in run.run_dir

    report = store.get_report(run_id)
    assert report is not None
    assert report["kind"] == "walk_forward"
    assert report["verdict"]["n_folds"] == 2
    assert (tmp_path / run_id / "report.json").exists()


def test_train_job_succeeds_without_report(synthetic_config, tmp_path) -> None:
    store = MemoryJobStore()
    run_id = store.enqueue("train", synthetic_config, user_id="u1")

    assert process_one(store, base_cfg={}, results_root=str(tmp_path)) is True

    run = store.get(run_id)
    assert run.status == "succeeded"
    assert store.get_report(run_id) is None
    assert (tmp_path / run_id / "checkpoints" / "double_dqn_best.pt").exists()


def test_invalid_spec_marks_run_failed(synthetic_config, tmp_path) -> None:
    bad = dict(synthetic_config)
    bad["agent"] = dict(synthetic_config["agent"], network="transformer")
    store = MemoryJobStore()
    run_id = store.enqueue("train", bad, user_id="u1")

    assert process_one(store, base_cfg={}, results_root=str(tmp_path)) is True

    run = store.get(run_id)
    assert run.status == "failed"
    assert "ValidationError" in run.error
