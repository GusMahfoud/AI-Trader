"""API skeleton: health, spec validation responses, and report retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_trader.api import create_app


@pytest.fixture
def client(tmp_path) -> TestClient:
    run_dir = tmp_path / "wf_demo"
    run_dir.mkdir()
    report = {"schema_version": 1, "run_id": "wf_demo", "verdict": {"n_folds": 4}}
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return TestClient(create_app(results_dir=tmp_path))


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_validate_accepts_valid_spec(client) -> None:
    body = {"agent": {"network": "dueling"}, "env": {"data_source": "synthetic"}}
    resp = client.post("/specs/validate", json=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["valid"] is True
    assert payload["config"]["agent"]["network"] == "dueling"
    # Omitted fields come back filled with champion defaults.
    assert payload["config"]["agent"]["replay"] == "per"


def test_validate_rejects_bad_spec(client) -> None:
    resp = client.post("/specs/validate", json={"agent": {"network": "transformer"}})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any("network" in str(err.get("loc", ())) for err in detail)


def test_get_report(client) -> None:
    resp = client.get("/reports/wf_demo")
    assert resp.status_code == 200
    assert resp.json()["verdict"]["n_folds"] == 4


def test_get_report_missing(client) -> None:
    assert client.get("/reports/nonexistent").status_code == 404


def test_get_report_traversal_guard(client) -> None:
    assert client.get("/reports/..%2Fsecrets").status_code == 404
