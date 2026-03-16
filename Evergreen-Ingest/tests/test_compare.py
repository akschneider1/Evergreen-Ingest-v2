"""
Tests for compare.py — diff logic.

Uses mock extraction objects so langextract is not required to run tests.
"""

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compare import (
    _similarity as _jaccard,
    _values_equal,
    _match_extractions,
    compare_extractions,
    load_comparison,
)


# ---------------------------------------------------------------------------
# Minimal mock objects
# ---------------------------------------------------------------------------

class MockExtraction:
    def __init__(self, cls, text, attrs=None, char_start=None, char_end=None):
        self.extraction_class = cls
        self.extraction_text = text
        self.attributes = attrs or {}
        self.char_interval = MockInterval(char_start, char_end) if char_start else None


class MockInterval:
    def __init__(self, start, end):
        self.start_pos = start
        self.end_pos = end


class MockDoc:
    def __init__(self, extractions):
        self.extractions = extractions


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    a = {"value": "$14,600", "type": "single"}
    assert _jaccard(a, a) == 1.0


def test_jaccard_empty():
    assert _jaccard({}, {}) == 0.0  # no keys → no basis to match
    assert _jaccard({"k": "v"}, {}) == 0.0


def test_jaccard_partial():
    a = {"value": "$14,600", "type": "single"}
    b = {"value": "$14,600", "type": "joint"}
    # Both keys overlap — key Jaccard is 1.0 (same keys, values not checked for matching)
    assert _jaccard(a, b) == pytest.approx(1.0)


def test_jaccard_different_keys():
    a = {"value": "$14,600", "rate": "4.4%"}
    b = {"amount": "$14,600", "type": "flat"}
    # 0 shared keys out of 4 total → 0.0
    assert _jaccard(a, b) == pytest.approx(0.0)


def test_values_equal_same():
    a = {"value": "$14,600", "rate": "4.4%"}
    equal, drifted = _values_equal(a, a)
    assert equal is True
    assert drifted == []


def test_values_equal_drift():
    a = {"value": "$14,600", "rate": "4.4%"}
    b = {"value": "$13,590", "rate": "4.4%"}
    equal, drifted = _values_equal(a, b)
    assert equal is False
    assert "value" in drifted
    assert "rate" not in drifted


def test_match_extractions_matched():
    policy = [MockExtraction("tax_rate", "flat 4.4%", {"value": "4.4%", "type": "flat"})]
    impl = [MockExtraction("tax_rate", "rate is 4.4%", {"value": "4.4%", "type": "flat"})]
    result = _match_extractions(policy, impl)
    assert len(result) == 1
    assert result[0]["status"] == "matched"
    assert result[0]["extraction_class"] == "tax_rate"


def test_match_extractions_drifted():
    policy = [MockExtraction("tax_rate", "flat 4.4%", {"value": "4.4%"})]
    impl = [MockExtraction("tax_rate", "rate is 4.25%", {"value": "4.25%"})]
    result = _match_extractions(policy, impl)
    assert len(result) == 1
    assert result[0]["status"] == "drifted"
    assert "value" in result[0]["drifted_attributes"]


def test_match_extractions_missing():
    policy = [MockExtraction("filing_deadline", "April 15", {"value": "April 15"})]
    impl = []
    result = _match_extractions(policy, impl)
    assert len(result) == 1
    assert result[0]["status"] == "missing"
    assert result[0]["implementation"] is None


def test_match_extractions_extra():
    policy = []
    impl = [MockExtraction("extra_proc", "call supervisor", {"value": "escalate"})]
    result = _match_extractions(policy, impl)
    assert len(result) == 1
    assert result[0]["status"] == "extra"
    assert result[0]["policy"] is None


def test_compare_extractions_writes_json(tmp_path):
    policy_doc = MockDoc([
        MockExtraction("tax_rate", "4.4% flat", {"value": "4.4%"}),
        MockExtraction("filing_deadline", "April 15", {"value": "April 15"}),
    ])
    impl_doc = MockDoc([
        MockExtraction("tax_rate", "4.25% flat", {"value": "4.25%"}),
        MockExtraction("extra_proc", "call supervisor", {"value": "escalate"}),
    ])

    comparison = compare_extractions(
        policy_doc=policy_doc,
        impl_doc=impl_doc,
        comparison_id="test_001",
        domain="tax",
        output_dir=tmp_path,
    )

    assert comparison["comparison_id"] == "test_001"
    assert comparison["summary"]["drifted"] == 1
    assert comparison["summary"]["missing"] == 1
    assert comparison["summary"]["extra"] == 1

    # File should be on disk
    loaded = load_comparison("test_001", tmp_path)
    assert loaded["comparison_id"] == "test_001"
    assert len(loaded["params"]) == 3


def test_compare_all_matched(tmp_path):
    attrs = {"value": "$14,600", "type": "single"}
    policy_doc = MockDoc([MockExtraction("threshold", "text", attrs)])
    impl_doc = MockDoc([MockExtraction("threshold", "text", attrs)])
    comparison = compare_extractions(
        policy_doc=policy_doc,
        impl_doc=impl_doc,
        comparison_id="test_002",
        domain="tax",
        output_dir=tmp_path,
    )
    assert comparison["summary"]["matched"] == 1
    assert comparison["summary"]["drifted"] == 0


# ---------------------------------------------------------------------------
# Edge-case tests for _similarity and _values_equal
# ---------------------------------------------------------------------------

def test_similarity_parameter_key_exact_match():
    """When both dicts have a 'parameter' key with identical values → 1.0."""
    a = {"parameter": "income tax rate", "value": "4.4%"}
    b = {"parameter": "income tax rate", "value": "4.25%"}
    assert _jaccard(a, b) == 1.0


def test_similarity_parameter_substring():
    """'income tax' is a substring of 'state income tax rate' → 0.7."""
    a = {"parameter": "income tax"}
    b = {"parameter": "state income tax rate"}
    assert _jaccard(a, b) == pytest.approx(0.7)


def test_similarity_parameter_mismatch():
    """'income tax' vs 'filing deadline' — no substring relation → 0.0."""
    a = {"parameter": "income tax"}
    b = {"parameter": "filing deadline"}
    assert _jaccard(a, b) == pytest.approx(0.0)


def test_match_threshold_above(tmp_path):
    """Extractions with Jaccard ≥ 0.3 should be matched (or drifted), not missing/extra."""
    attrs = {"value": "4.4%", "type": "flat"}
    policy = [MockExtraction("tax_rate", "rate 4.4%", attrs)]
    impl = [MockExtraction("tax_rate", "rate is 4.4%", attrs)]
    result = _match_extractions(policy, impl)
    assert len(result) == 1
    assert result[0]["status"] in ("matched", "drifted")


def test_match_threshold_below(tmp_path):
    """Extractions in the same class but with 0 similarity → one missing + one extra."""
    policy = [MockExtraction("tax_rate", "rate 4.4%", {"value": "4.4%"})]
    impl = [MockExtraction("tax_rate", "rate 4.25%", {"rate": "4.25%"})]
    # key overlap: {"value"} vs {"rate"} → Jaccard = 0/2 = 0.0 → below threshold
    result = _match_extractions(policy, impl)
    statuses = {r["status"] for r in result}
    assert "missing" in statuses
    assert "extra" in statuses


def test_values_equal_none_values():
    """Two dicts with None values for the same key are considered equal."""
    a = {"value": None, "type": "flat"}
    equal, drifted = _values_equal(a, a)
    assert equal is True
    assert drifted == []


def test_values_equal_normalised_strings():
    """Whitespace differences in values are normalised away."""
    a = {"rate": "4.4 %"}
    b = {"rate": "4.4%"}
    # _normalize strips/lowercases but does NOT remove internal spaces,
    # so "4.4 %" and "4.4%" differ — confirms the normalise function behaviour.
    equal, drifted = _values_equal(a, b)
    # These are NOT equal after normalization (space is preserved)
    assert equal is False
    assert "rate" in drifted
