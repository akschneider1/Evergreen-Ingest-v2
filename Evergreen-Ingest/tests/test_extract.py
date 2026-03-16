"""
Tests for extract.py — document reading and langextract integration.

Extraction tests require a valid GOOGLE_API_KEY and are skipped otherwise.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES = Path(__file__).parent / "fixtures"
REQUIRES_API = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping live API tests",
)


def test_read_markdown():
    from extract import read_document
    text = read_document(FIXTURES / "policy_tax_current.md")
    assert "14,600" in text
    assert "April 15" in text


def test_read_html():
    from extract import read_document
    text = read_document(FIXTURES / "policy_benefits_current.html")
    assert "2,500" in text
    assert "weekly benefit" in text.lower()


def test_read_unsupported_falls_back(tmp_path):
    from extract import read_document
    f = tmp_path / "test.csv"
    f.write_text("col1,col2\nval1,val2")
    text = read_document(f)
    assert "col1" in text


@REQUIRES_API
def test_extract_tax_policy(tmp_path):
    from extract import extract_document, load_extractions
    jsonl_path, viz_path, doc = extract_document(
        source_path=FIXTURES / "policy_tax_current.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id="test_tax",
        doc_slot="policy",
    )
    assert jsonl_path.exists()
    assert viz_path.exists()
    assert doc.extractions is not None
    assert len(doc.extractions) > 0

    # Verify we extracted at least one tax_rate or threshold
    classes = {e.extraction_class for e in doc.extractions}
    assert classes & {"tax_rate", "filing_threshold", "credit", "penalty_rule", "filing_deadline"}


@REQUIRES_API
def test_extract_and_roundtrip(tmp_path):
    from extract import extract_document, load_extractions
    _, _, doc = extract_document(
        source_path=FIXTURES / "policy_tax_current.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id="test_roundtrip",
        doc_slot="policy",
    )
    jsonl_path = tmp_path / "extractions" / "test_roundtrip" / "policy.jsonl"
    reloaded = load_extractions(jsonl_path)
    assert len(reloaded.extractions) == len(doc.extractions)


# ---------------------------------------------------------------------------
# Error-path tests (no API key required)
# ---------------------------------------------------------------------------

def test_read_document_missing_file():
    """Reading a non-existent path raises FileNotFoundError or OSError."""
    from extract import read_document
    with pytest.raises((FileNotFoundError, OSError)):
        read_document("/nonexistent/path/to/missing.md")


def test_extract_raises_on_empty_config(tmp_path, monkeypatch):
    """When lx.extract returns 0 extractions and fail_on_empty_extraction=True,
    extract_document should raise RuntimeError."""
    import unittest.mock as mock

    class _FakeDoc:
        extractions = []

    # Patch lx.extract to return an empty doc
    monkeypatch.setattr("extract.lx.extract", lambda **kw: _FakeDoc())
    # Patch lx.io.save_annotated_documents to be a no-op
    monkeypatch.setattr("extract.lx.io.save_annotated_documents", lambda *a, **kw: None)
    # Patch lx.visualize to return minimal HTML
    monkeypatch.setattr("extract.lx.visualize", lambda doc: "<html></html>")
    # Patch config to enable fail_on_empty_extraction
    monkeypatch.setattr("extract._CONFIG", {
        "model_id": "gpt-4o-mini",
        "extraction_passes": 1,
        "max_workers": 1,
        "max_char_buffer": 6000,
        "max_output_tokens": 512,
        "api_timeout": 30,
        "fail_on_empty_extraction": True,
    })
    # Supply a fake API key so the key check passes
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

    from extract import extract_document
    with pytest.raises(RuntimeError, match="No parameters found"):
        extract_document(
            source_path=FIXTURES / "policy_tax_current.md",
            domain_name="tax",
            output_dir=tmp_path,
            comparison_id="empty_test",
            doc_slot="policy",
        )


def test_extract_warns_but_no_raise(tmp_path, monkeypatch, caplog):
    """When lx.extract returns 0 extractions and fail_on_empty_extraction=False,
    extract_document should log a warning but NOT raise."""
    import logging
    import unittest.mock as mock

    class _FakeDoc:
        extractions = []

    monkeypatch.setattr("extract.lx.extract", lambda **kw: _FakeDoc())
    monkeypatch.setattr("extract.lx.io.save_annotated_documents", lambda *a, **kw: None)
    monkeypatch.setattr("extract.lx.visualize", lambda doc: "<html></html>")
    monkeypatch.setattr("extract._CONFIG", {
        "model_id": "gpt-4o-mini",
        "extraction_passes": 1,
        "max_workers": 1,
        "max_char_buffer": 6000,
        "max_output_tokens": 512,
        "api_timeout": 30,
        "fail_on_empty_extraction": False,
    })
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

    from extract import extract_document
    with caplog.at_level(logging.WARNING, logger="extract"):
        jsonl_path, viz_path, doc = extract_document(
            source_path=FIXTURES / "policy_tax_current.md",
            domain_name="tax",
            output_dir=tmp_path,
            comparison_id="empty_warn_test",
            doc_slot="policy",
        )
    assert any("0 extractions" in r.message for r in caplog.records)
    assert doc.extractions == []


def test_extract_missing_api_key_raises(tmp_path, monkeypatch):
    """If no API key env var is set for the chosen model, RuntimeError is raised
    before lx.extract is ever called."""
    monkeypatch.setattr("extract._CONFIG", {
        "model_id": "gpt-4o-mini",
        "extraction_passes": 1,
        "max_workers": 1,
        "max_char_buffer": 6000,
        "max_output_tokens": 512,
        "api_timeout": 30,
        "fail_on_empty_extraction": True,
    })
    # Remove the key for the model being used
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from extract import extract_document
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        extract_document(
            source_path=FIXTURES / "policy_tax_current.md",
            domain_name="tax",
            output_dir=tmp_path,
            comparison_id="no_key_test",
            doc_slot="policy",
        )
