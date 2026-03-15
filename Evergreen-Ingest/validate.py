"""
validate.py — Validation state management.

Stores expert decisions (confirm / edit / reject) per comparison in a JSON
file. All writes are atomic via os.replace to avoid corruption.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _state_path(comparison_id: str, output_dir: str | Path) -> Path:
    return Path(output_dir) / "validations" / comparison_id / "state.json"


def load_state(comparison_id: str, output_dir: str | Path) -> dict:
    """
    Load validation state for a comparison, or return an empty skeleton if
    no decisions have been recorded yet.
    """
    path = _state_path(comparison_id, output_dir)
    if not path.exists():
        return {
            "comparison_id": comparison_id,
            "last_updated": None,
            "decisions": {},
        }
    with open(path) as f:
        return json.load(f)


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically: write to a temp file, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_decision(
    comparison_id: str,
    output_dir: str | Path,
    param_index: int,
    decision: str,  # "confirmed" | "edited" | "rejected"
    edited_value: dict | None = None,
    note: str = "",
) -> dict:
    """
    Record a validation decision for a single parameter.

    Returns the updated state dict.
    Decision values:
      "confirmed" — policy is correct, flag implementation for update
      "edited"    — user supplied corrected attribute values
      "rejected"  — not a real drift or policy itself is disputed
    """
    if decision not in ("confirmed", "edited", "rejected"):
        raise ValueError(f"Invalid decision: {decision!r}")

    state = load_state(comparison_id, output_dir)
    state["decisions"][str(param_index)] = {
        "decision": decision,
        "edited_value": edited_value,
        "note": note,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    state["last_updated"] = datetime.now(timezone.utc).isoformat()

    path = _state_path(comparison_id, output_dir)
    _atomic_write(path, state)
    return state


def get_decision(state: dict, param_index: int) -> dict | None:
    """Return the decision record for a param, or None if not yet decided."""
    return state["decisions"].get(str(param_index))


def clear_decision(
    comparison_id: str,
    output_dir: str | Path,
    param_index: int,
) -> dict:
    """Remove a decision, returning the parameter to pending status."""
    state = load_state(comparison_id, output_dir)
    state["decisions"].pop(str(param_index), None)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    path = _state_path(comparison_id, output_dir)
    _atomic_write(path, state)
    return state


def validation_progress(state: dict, total_params: int, actionable_count: int) -> dict:
    """
    Compute validation progress stats.
    actionable_count = drifted + missing + extra (matched params don't need review).
    """
    decided = len(state["decisions"])
    return {
        "decided": decided,
        "actionable": actionable_count,
        "pending": max(0, actionable_count - decided),
        "percent": round(100 * decided / actionable_count) if actionable_count else 100,
    }
