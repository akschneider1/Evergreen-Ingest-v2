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
