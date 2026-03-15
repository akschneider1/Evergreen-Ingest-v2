"""Tests for validate.py — validation state management."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import validate


def test_load_state_empty(tmp_path):
    state = validate.load_state("cmp_001", tmp_path)
    assert state["comparison_id"] == "cmp_001"
    assert state["decisions"] == {}
    assert state["last_updated"] is None


def test_save_and_load_decision(tmp_path):
    state = validate.save_decision(
        comparison_id="cmp_001",
        output_dir=tmp_path,
        param_index=3,
        decision="confirmed",
        note="Looks correct",
    )
    assert "3" in state["decisions"]
    assert state["decisions"]["3"]["decision"] == "confirmed"
    assert state["decisions"]["3"]["note"] == "Looks correct"

    # Reload from disk
    reloaded = validate.load_state("cmp_001", tmp_path)
    assert reloaded["decisions"]["3"]["decision"] == "confirmed"


def test_save_edited_decision(tmp_path):
    edited_value = {"value": "$14,600", "type": "corrected"}
    state = validate.save_decision(
        comparison_id="cmp_002",
        output_dir=tmp_path,
        param_index=0,
        decision="edited",
        edited_value=edited_value,
        note="Updated to match policy",
    )
    assert state["decisions"]["0"]["edited_value"] == edited_value


def test_save_rejected_decision(tmp_path):
    state = validate.save_decision(
        comparison_id="cmp_003",
        output_dir=tmp_path,
        param_index=7,
        decision="rejected",
        note="Policy is wrong here",
    )
    assert state["decisions"]["7"]["decision"] == "rejected"


def test_invalid_decision_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="Invalid decision"):
        validate.save_decision(
            comparison_id="cmp_004",
            output_dir=tmp_path,
            param_index=0,
            decision="approved",  # invalid
        )


def test_get_decision(tmp_path):
    validate.save_decision("cmp_005", tmp_path, 2, "confirmed")
    state = validate.load_state("cmp_005", tmp_path)

    d = validate.get_decision(state, 2)
    assert d is not None
    assert d["decision"] == "confirmed"

    d_none = validate.get_decision(state, 99)
    assert d_none is None


def test_clear_decision(tmp_path):
    validate.save_decision("cmp_006", tmp_path, 1, "confirmed")
    validate.clear_decision("cmp_006", tmp_path, 1)
    state = validate.load_state("cmp_006", tmp_path)
    assert validate.get_decision(state, 1) is None


def test_multiple_decisions(tmp_path):
    validate.save_decision("cmp_007", tmp_path, 0, "confirmed")
    validate.save_decision("cmp_007", tmp_path, 1, "rejected")
    validate.save_decision("cmp_007", tmp_path, 2, "edited", {"value": "x"})

    state = validate.load_state("cmp_007", tmp_path)
    assert len(state["decisions"]) == 3

    progress = validate.validation_progress(state, total_params=10, actionable_count=5)
    assert progress["decided"] == 3
    assert progress["actionable"] == 5
    assert progress["pending"] == 2
    assert progress["percent"] == 60


def test_atomic_write_idempotent(tmp_path):
    """Overwriting an existing decision should update cleanly."""
    validate.save_decision("cmp_008", tmp_path, 0, "confirmed", note="first")
    validate.save_decision("cmp_008", tmp_path, 0, "rejected", note="changed mind")
    state = validate.load_state("cmp_008", tmp_path)
    assert state["decisions"]["0"]["decision"] == "rejected"
    assert state["decisions"]["0"]["note"] == "changed mind"
