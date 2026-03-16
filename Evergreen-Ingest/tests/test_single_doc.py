"""
Tests for single_doc.py — single-document view builder.

Uses the same MockExtraction/MockDoc pattern from test_compare.py.
No API key required.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from single_doc import build_single_doc_view, load_single_doc


# ---------------------------------------------------------------------------
# Minimal mock objects (mirrors test_compare.py)
# ---------------------------------------------------------------------------

class MockInterval:
    def __init__(self, start, end):
        self.start_pos = start
        self.end_pos = end


class MockExtraction:
    def __init__(self, cls, text, attrs=None, char_start=None, char_end=None):
        self.extraction_class = cls
        self.extraction_text = text
        self.attributes = attrs or {}
        self.char_interval = MockInterval(char_start, char_end) if char_start is not None else None


class MockDoc:
    def __init__(self, extractions):
        self.extractions = extractions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_empty_doc(tmp_path):
    doc = MockDoc([])
    result = build_single_doc_view(doc, "e001", "tax", "policy.md", "gpt-4o-mini", tmp_path)
    assert result["total"] == 0
    assert result["parameters"] == []


def test_build_single_parameter(tmp_path):
    ext = MockExtraction("tax_rate", "flat 4.4%", {"value": "4.4%"})
    doc = MockDoc([ext])
    result = build_single_doc_view(doc, "e002", "tax", "policy.md", "gpt-4o-mini", tmp_path)
    assert result["total"] == 1
    p = result["parameters"][0]
    assert p["index"] == 0
    assert p["extraction_class"] == "tax_rate"
    assert p["extraction_text"] == "flat 4.4%"
    assert p["attributes"]["value"] == "4.4%"


def test_build_multiple_parameters(tmp_path):
    exts = [
        MockExtraction("tax_rate", "4.4%", {"value": "4.4%"}),
        MockExtraction("filing_deadline", "April 15", {"value": "April 15"}),
        MockExtraction("threshold", "$14,600", {"value": "$14,600"}),
    ]
    doc = MockDoc(exts)
    result = build_single_doc_view(doc, "e003", "tax", "policy.md", "gpt-4o-mini", tmp_path)
    assert result["total"] == 3
    indices = [p["index"] for p in result["parameters"]]
    assert indices == [0, 1, 2]


def test_char_interval_mapped(tmp_path):
    ext = MockExtraction("tax_rate", "4.4%", {"value": "4.4%"}, char_start=100, char_end=105)
    doc = MockDoc([ext])
    result = build_single_doc_view(doc, "e004", "tax", "policy.md", "gpt-4o-mini", tmp_path)
    p = result["parameters"][0]
    assert p["char_start"] == 100
    assert p["char_end"] == 105


def test_char_interval_missing(tmp_path):
    ext = MockExtraction("tax_rate", "4.4%", {"value": "4.4%"})
    doc = MockDoc([ext])
    result = build_single_doc_view(doc, "e005", "tax", "policy.md", "gpt-4o-mini", tmp_path)
    p = result["parameters"][0]
    assert p["char_start"] is None
    assert p["char_end"] is None


def test_writes_json_to_disk(tmp_path):
    ext = MockExtraction("tax_rate", "4.4%", {"value": "4.4%"})
    doc = MockDoc([ext])
    build_single_doc_view(doc, "e006", "tax", "policy.md", "gpt-4o-mini", tmp_path)
    out = tmp_path / "comparisons" / "e006" / "single_doc.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["extraction_id"] == "e006"


def test_load_single_doc_roundtrip(tmp_path):
    ext = MockExtraction("tax_rate", "4.4%", {"value": "4.4%"})
    doc = MockDoc([ext])
    original = build_single_doc_view(doc, "e007", "tax", "policy.md", "gpt-4o-mini", tmp_path)
    loaded = load_single_doc("e007", tmp_path)
    assert loaded["extraction_id"] == original["extraction_id"]
    assert loaded["total"] == original["total"]
    assert loaded["parameters"] == original["parameters"]


def test_extraction_class_fallback(tmp_path):
    ext = MockExtraction(None, "some text", {"value": "x"})
    doc = MockDoc([ext])
    result = build_single_doc_view(doc, "e008", "tax", "policy.md", "gpt-4o-mini", tmp_path)
    assert result["parameters"][0]["extraction_class"] == "unknown"


def test_total_matches_parameters_len(tmp_path):
    exts = [MockExtraction(f"cls_{i}", f"text {i}") for i in range(5)]
    doc = MockDoc(exts)
    result = build_single_doc_view(doc, "e009", "tax", "policy.md", "gpt-4o-mini", tmp_path)
    assert result["total"] == len(result["parameters"])
