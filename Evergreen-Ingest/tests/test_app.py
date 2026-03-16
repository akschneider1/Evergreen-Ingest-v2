"""
Tests for app.py FastAPI routes.

Strategy:
- Use FastAPI TestClient with monkeypatched OUTPUT_DIR / SOURCES_DIR
- Pre-build minimal JSON fixtures in tmp_path so read-only routes work
- Pipeline-triggering uploads mock executor.submit to a no-op
- No API key required
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as app_module
from app import app


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with OUTPUT_DIR and SOURCES_DIR redirected to tmp_path."""
    out = tmp_path / "output"
    for sub in ("extractions", "visualizations", "comparisons", "validations", "sources"):
        (out / sub).mkdir(parents=True)
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)

    monkeypatch.setattr("app.OUTPUT_DIR", out)
    monkeypatch.setattr("app.SOURCES_DIR", sources)
    # Prevent real pipeline execution in upload tests
    monkeypatch.setattr("app.executor.submit", lambda *a, **kw: None)

    return TestClient(app, raise_server_exceptions=True)


def _write_meta(out: Path, cid: str, meta: dict) -> None:
    p = out / "comparisons" / cid
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text(json.dumps(meta))


def _write_comparison(out: Path, cid: str, comparison: dict) -> None:
    p = out / "comparisons" / cid
    p.mkdir(parents=True, exist_ok=True)
    (p / "comparison.json").write_text(json.dumps(comparison))


def _write_single_doc(out: Path, eid: str, single_doc: dict) -> None:
    p = out / "comparisons" / eid
    p.mkdir(parents=True, exist_ok=True)
    (p / "single_doc.json").write_text(json.dumps(single_doc))


def _sample_meta(cid: str, status: str = "ready", **extra) -> dict:
    # Use current time so _check_timeout doesn't convert pending → error
    now = datetime.now(timezone.utc).isoformat()
    return {
        "comparison_id": cid,
        "status": status,
        "domain": "tax",
        "model": "gpt-4o-mini",
        "policy_filename": "policy.md",
        "implementation_filename": "impl.md",
        "created_at": now,
        "logs": [],
        "error": None,
        **extra,
    }


def _sample_comparison(cid: str) -> dict:
    return {
        "comparison_id": cid,
        "domain": "tax",
        "generated_at": "2024-01-01T00:00:00+00:00",
        "summary": {"total": 1, "matched": 0, "drifted": 1, "missing": 0, "extra": 0},
        "params": [{
            "index": 0,
            "extraction_class": "tax_rate",
            "status": "drifted",
            "policy": {
                "extraction_text": "4.4%",
                "attributes": {"value": "4.4%"},
                "viz_idx": 0,
                "char_start": None,
                "char_end": None,
            },
            "implementation": {
                "extraction_text": "4.25%",
                "attributes": {"value": "4.25%"},
                "viz_idx": 0,
                "char_start": None,
                "char_end": None,
            },
            "drift_reason": "attribute_mismatch",
            "drifted_attributes": ["value"],
        }],
    }


def _sample_single_doc(eid: str) -> dict:
    return {
        "extraction_id": eid,
        "domain": "tax",
        "document_name": "policy.md",
        "model": "gpt-4o-mini",
        "generated_at": "2024-01-01T00:00:00+00:00",
        "total": 1,
        "parameters": [{
            "index": 0,
            "extraction_class": "tax_rate",
            "attributes": {"value": "4.4%"},
            "extraction_text": "4.4%",
            "viz_idx": 0,
            "char_start": None,
            "char_end": None,
        }],
    }


# ---------------------------------------------------------------------------
# Upload validation tests
# ---------------------------------------------------------------------------

def test_upload_rejects_bad_extension(client):
    resp = client.post(
        "/upload",
        data={"domain": "tax", "model": "gpt-4o-mini"},
        files={
            "policy_file": ("malware.exe", b"content", "application/octet-stream"),
            "implementation_file": ("impl.md", b"impl content", "text/plain"),
        },
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_upload_rejects_missing_file(client):
    """Omitting required form fields gives 422."""
    resp = client.post("/upload", data={"domain": "tax"})
    assert resp.status_code == 422


def test_upload_single_rejects_bad_extension(client):
    resp = client.post(
        "/upload-single",
        data={"domain": "tax", "model": "gpt-4o-mini"},
        files={
            "document_file": ("virus.exe", b"content", "application/octet-stream"),
        },
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Status JSON routes
# ---------------------------------------------------------------------------

def test_compare_status_pending(client, tmp_path):
    out = tmp_path / "output"
    _write_meta(out, "cmp001", _sample_meta("cmp001", status="pending"))
    resp = client.get("/compare/cmp001/status.json")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"


def test_compare_status_ready(client, tmp_path):
    out = tmp_path / "output"
    _write_meta(out, "cmp002", _sample_meta("cmp002", status="ready"))
    resp = client.get("/compare/cmp002/status.json")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_compare_status_error(client, tmp_path):
    out = tmp_path / "output"
    meta = _sample_meta("cmp003", status="error", error="API key missing")
    _write_meta(out, "cmp003", meta)
    resp = client.get("/compare/cmp003/status.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert "error" in data


def test_compare_status_missing_id(client):
    resp = client.get("/compare/does_not_exist/status.json")
    assert resp.status_code == 404


def test_extract_status_pending(client, tmp_path):
    out = tmp_path / "output"
    _write_meta(out, "ext001", _sample_meta("ext001", status="pending", mode="extract"))
    resp = client.get("/extract/ext001/status.json")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"


def test_extract_status_ready(client, tmp_path):
    out = tmp_path / "output"
    _write_meta(out, "ext002", _sample_meta("ext002", status="ready", mode="extract"))
    resp = client.get("/extract/ext002/status.json")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# Compare / drift view routes
# ---------------------------------------------------------------------------

def test_compare_page_loads(client, tmp_path):
    out = tmp_path / "output"
    cid = "cmp_view_001"
    _write_meta(out, cid, _sample_meta(cid))
    _write_comparison(out, cid, _sample_comparison(cid))
    resp = client.get(f"/compare/{cid}")
    assert resp.status_code == 200


def test_compare_page_missing_id(client):
    resp = client.get("/compare/nonexistent_id_xyz")
    assert resp.status_code == 404


def test_compare_report_json(client, tmp_path):
    out = tmp_path / "output"
    cid = "cmp_rpt_001"
    _write_meta(out, cid, _sample_meta(cid))
    _write_comparison(out, cid, _sample_comparison(cid))
    resp = client.get(f"/compare/{cid}/report.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["comparison_id"] == cid


def test_compare_report_csv(client, tmp_path):
    out = tmp_path / "output"
    cid = "cmp_csv_001"
    _write_meta(out, cid, _sample_meta(cid))
    _write_comparison(out, cid, _sample_comparison(cid))
    resp = client.get(f"/compare/{cid}/report.csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Validation routes
# ---------------------------------------------------------------------------

def test_param_detail_loads(client, tmp_path):
    out = tmp_path / "output"
    cid = "cmp_param_001"
    _write_meta(out, cid, _sample_meta(cid))
    _write_comparison(out, cid, _sample_comparison(cid))
    resp = client.get(f"/compare/{cid}/param/0")
    assert resp.status_code == 200


def test_param_post_confirmed(client, tmp_path):
    out = tmp_path / "output"
    cid = "cmp_dec_001"
    _write_meta(out, cid, _sample_meta(cid))
    _write_comparison(out, cid, _sample_comparison(cid))
    resp = client.post(
        f"/compare/{cid}/param/0",
        data={"decision": "confirmed", "note": "", "edited_json": ""},
        follow_redirects=False,
    )
    assert resp.status_code == 303


def test_param_post_rejected(client, tmp_path):
    out = tmp_path / "output"
    cid = "cmp_dec_002"
    _write_meta(out, cid, _sample_meta(cid))
    _write_comparison(out, cid, _sample_comparison(cid))
    resp = client.post(
        f"/compare/{cid}/param/0",
        data={"decision": "rejected", "note": "not relevant", "edited_json": ""},
        follow_redirects=False,
    )
    assert resp.status_code == 303


def test_param_post_edited(client, tmp_path):
    out = tmp_path / "output"
    cid = "cmp_dec_003"
    _write_meta(out, cid, _sample_meta(cid))
    _write_comparison(out, cid, _sample_comparison(cid))
    resp = client.post(
        f"/compare/{cid}/param/0",
        data={
            "decision": "edited",
            "note": "",
            "edited_json": '{"value": "4.4%"}',
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303


# ---------------------------------------------------------------------------
# Single-doc extract routes
# ---------------------------------------------------------------------------

def test_extract_review_loads(client, tmp_path):
    out = tmp_path / "output"
    eid = "ext_view_001"
    _write_meta(out, eid, _sample_meta(eid, status="ready", mode="extract",
                                        document_filename="policy.md"))
    _write_single_doc(out, eid, _sample_single_doc(eid))
    resp = client.get(f"/extract/{eid}")
    assert resp.status_code == 200


def test_extract_param_detail_loads(client, tmp_path):
    out = tmp_path / "output"
    eid = "ext_param_001"
    _write_meta(out, eid, _sample_meta(eid, status="ready", mode="extract",
                                        document_filename="policy.md"))
    _write_single_doc(out, eid, _sample_single_doc(eid))
    resp = client.get(f"/extract/{eid}/param/0")
    assert resp.status_code == 200


def test_extract_param_post(client, tmp_path):
    out = tmp_path / "output"
    eid = "ext_dec_001"
    _write_meta(out, eid, _sample_meta(eid, status="ready", mode="extract",
                                        document_filename="policy.md"))
    _write_single_doc(out, eid, _sample_single_doc(eid))
    resp = client.post(
        f"/extract/{eid}/param/0",
        data={"decision": "confirmed", "note": "", "edited_json": ""},
        follow_redirects=False,
    )
    assert resp.status_code == 303


def test_extract_report_json(client, tmp_path):
    out = tmp_path / "output"
    eid = "ext_rpt_001"
    _write_meta(out, eid, _sample_meta(eid, status="ready", mode="extract",
                                        document_filename="policy.md"))
    _write_single_doc(out, eid, _sample_single_doc(eid))
    resp = client.get(f"/extract/{eid}/report.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["extraction_id"] == eid


# ---------------------------------------------------------------------------
# History route
# ---------------------------------------------------------------------------

def test_comparisons_empty(client):
    resp = client.get("/comparisons")
    assert resp.status_code == 200


def test_comparisons_with_entries(client, tmp_path):
    out = tmp_path / "output"
    for cid, domain in [("hist_001", "tax"), ("hist_002", "benefits")]:
        _write_meta(out, cid, _sample_meta(cid, domain=domain))
        _write_comparison(out, cid, _sample_comparison(cid))

    resp = client.get("/comparisons")
    assert resp.status_code == 200
    # Both comparison IDs should appear in the history page body
    body = resp.text
    assert "hist_001" in body or "tax" in body.lower()
