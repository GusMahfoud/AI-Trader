"""MemoryJobStore contract: enqueue/claim ordering, terminal states, reports."""

from __future__ import annotations

from ai_trader.jobs import MemoryJobStore


def test_enqueue_then_claim_oldest_first() -> None:
    store = MemoryJobStore()
    first = store.enqueue("train", {"a": 1}, user_id="u1")
    second = store.enqueue("walk_forward", {"b": 2}, user_id="u1")

    claimed = store.claim()
    assert claimed is not None
    assert claimed.id == first
    assert claimed.status == "running"
    assert claimed.started_at is not None

    assert store.claim().id == second
    assert store.claim() is None


def test_claim_never_returns_terminal_runs() -> None:
    store = MemoryJobStore()
    run_id = store.enqueue("train", {}, user_id="u1")
    store.claim()
    store.mark_failed(run_id, error="boom")
    assert store.claim() is None

    run = store.get(run_id)
    assert run.status == "failed"
    assert run.error == "boom"
    assert run.finished_at is not None


def test_mark_succeeded_stores_report_and_run_dir() -> None:
    store = MemoryJobStore()
    run_id = store.enqueue("walk_forward", {}, user_id="u1")
    store.claim()
    store.mark_succeeded(run_id, run_dir="results/x", report={"verdict": {"n_folds": 2}})

    run = store.get(run_id)
    assert run.status == "succeeded"
    assert run.run_dir == "results/x"
    assert store.get_report(run_id) == {"verdict": {"n_folds": 2}}


def test_list_recent_filters_by_user_and_orders_newest_first() -> None:
    store = MemoryJobStore()
    a = store.enqueue("train", {}, user_id="u1")
    store.enqueue("train", {}, user_id="u2")
    b = store.enqueue("train", {}, user_id="u1")

    runs = store.list_recent("u1")
    assert [r.id for r in runs] == [b, a]


def test_get_unknown_run_returns_none() -> None:
    store = MemoryJobStore()
    assert store.get("nope") is None
    assert store.get_report("nope") is None
