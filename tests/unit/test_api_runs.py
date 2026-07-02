"""/runs endpoints: submit, status, listing, and stored-report retrieval."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_trader.api import create_app
from ai_trader.jobs import MemoryJobStore


@pytest.fixture
def store() -> MemoryJobStore:
    return MemoryJobStore()


@pytest.fixture
def client(store, tmp_path) -> TestClient:
    return TestClient(create_app(results_dir=tmp_path, store=store, user_id="test-user"))


def test_submit_run_enqueues(client, store) -> None:
    resp = client.post("/runs", json={"kind": "walk_forward", "spec": {}})
    assert resp.status_code == 202
    payload = resp.json()
    assert payload["status"] == "queued"

    run = store.get(payload["run_id"])
    assert run.user_id == "test-user"
    assert run.kind == "walk_forward"
    # Spec is stored fully resolved (champion defaults filled in).
    assert run.spec["agent"]["network"] == "mlp"


def test_submit_run_rejects_bad_spec(client) -> None:
    resp = client.post("/runs", json={"kind": "train", "spec": {"agent": {"gamma": 2}}})
    assert resp.status_code == 422


def test_submit_run_rejects_bad_kind(client) -> None:
    resp = client.post("/runs", json={"kind": "compare", "spec": {}})
    assert resp.status_code == 422


def test_get_run_status_and_listing(client, store) -> None:
    run_id = client.post("/runs", json={"kind": "train", "spec": {}}).json()["run_id"]

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"

    listing = client.get("/runs").json()["runs"]
    assert [r["id"] for r in listing] == [run_id]

    assert client.get("/runs/does-not-exist").status_code == 404


def test_get_run_report_after_success(client, store) -> None:
    run_id = client.post("/runs", json={"kind": "walk_forward", "spec": {}}).json()["run_id"]
    assert client.get(f"/runs/{run_id}/report").status_code == 404

    store.claim()
    store.mark_succeeded(run_id, run_dir="results/x", report={"verdict": {"n_folds": 4}})

    resp = client.get(f"/runs/{run_id}/report")
    assert resp.status_code == 200
    assert resp.json()["verdict"]["n_folds"] == 4
