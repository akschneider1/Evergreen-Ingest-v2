"""
compare.py — Diff logic for policy vs implementation extractions.

Takes two AnnotatedDocuments (or extraction lists), matches parameters by
extraction_class and attribute similarity, and categorizes each pair as:
  matched   — same class, same values
  drifted   — same class, different values
  missing   — in policy but not in implementation
  extra     — in implementation but not in policy
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize(value: Any) -> str:
    """Normalize an attribute value to a lowercase stripped string for comparison."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _similarity(attrs_a: dict, attrs_b: dict) -> float:
    """
    Match on the 'parameter' key when present (semantically correct for policy docs).
    Fall back to key-overlap (Jaccard) for extractions without a 'parameter' key.
    """
    p = (attrs_a or {}).get("parameter", "").lower().strip()
    i = (attrs_b or {}).get("parameter", "").lower().strip()
    if p and i:
        if p == i:
            return 1.0
        if p in i or i in p:
            return 0.7  # substring match
        return 0.0
    # Fallback: Jaccard on key sets
    keys_a = set(attrs_a or {})
    keys_b = set(attrs_b or {})
    if not keys_a or not keys_b:
        return 0.0
    return len(keys_a & keys_b) / len(keys_a | keys_b)


def _values_equal(attrs_a: dict, attrs_b: dict) -> tuple[bool, list[str]]:
    """
    Check whether two parameter dicts have the same values for shared keys.
    Returns (equal: bool, drifted_keys: list[str]).
    """
    if attrs_a == attrs_b:
        return True, []

    shared_keys = set(attrs_a) & set(attrs_b)
    drifted = [
        k for k in shared_keys
        if _normalize(attrs_a.get(k)) != _normalize(attrs_b.get(k))
    ]
    return len(drifted) == 0 and len(shared_keys) > 0, drifted


def _extraction_to_dict(ext, viz_idx: int | None = None) -> dict:
    """Convert a langextract Extraction object to a plain dict."""
    attrs = ext.attributes or {}
    result = {
        "extraction_text": ext.extraction_text or "",
        "attributes": dict(attrs) if attrs else {},
        "viz_idx": viz_idx,
    }
    if hasattr(ext, "char_interval") and ext.char_interval is not None:
        result["char_start"] = getattr(ext.char_interval, "start_pos", None)
        result["char_end"] = getattr(ext.char_interval, "end_pos", None)
    else:
        result["char_start"] = None
        result["char_end"] = None
    return result


def _match_extractions(
    policy_exts: list, impl_exts: list
) -> list[dict]:
    """
    Greedy best-match between policy and implementation extractions within the
    same extraction_class. Returns a flat list of parameter dicts.

    Matching strategy:
      1. Group by extraction_class.
      2. Within each class, compute pairwise similarity (parameter name, then key-overlap).
      3. Greedily pair highest-similarity pairs (>= 0.3 threshold).
      4. Determine matched vs drifted based on value equality.
      5. Unmatched policy → missing; unmatched impl → extra.
    """
    # Record each extraction's position in the original list before grouping.
    # This index matches the data-idx attribute in langextract's viz HTML.
    policy_viz: dict[int, int] = {id(ext): i for i, ext in enumerate(policy_exts)}
    impl_viz: dict[int, int] = {id(ext): i for i, ext in enumerate(impl_exts)}

    # Group by class
    policy_by_class: dict[str, list] = {}
    impl_by_class: dict[str, list] = {}

    for ext in policy_exts:
        cls = ext.extraction_class or "unknown"
        policy_by_class.setdefault(cls, []).append(ext)

    for ext in impl_exts:
        cls = ext.extraction_class or "unknown"
        impl_by_class.setdefault(cls, []).append(ext)

    all_classes = set(policy_by_class) | set(impl_by_class)
    params = []

    for cls in sorted(all_classes):
        policy_group = policy_by_class.get(cls, [])
        impl_group = impl_by_class.get(cls, [])

        if not impl_group:
            # All policy items in this class are missing from implementation
            for ext in policy_group:
                params.append({
                    "extraction_class": cls,
                    "status": "missing",
                    "policy": _extraction_to_dict(ext, policy_viz.get(id(ext))),
                    "implementation": None,
                    "drift_reason": None,
                    "drifted_attributes": [],
                })
            continue

        if not policy_group:
            # All impl items in this class are extra (not in policy)
            for ext in impl_group:
                params.append({
                    "extraction_class": cls,
                    "status": "extra",
                    "policy": None,
                    "implementation": _extraction_to_dict(ext, impl_viz.get(id(ext))),
                    "drift_reason": None,
                    "drifted_attributes": [],
                })
            continue

        # Compute similarity matrix
        used_policy = set()
        used_impl = set()
        pairs: list[tuple[float, int, int]] = []

        for pi, p_ext in enumerate(policy_group):
            p_attrs = (p_ext.attributes or {})
            for ii, i_ext in enumerate(impl_group):
                i_attrs = (i_ext.attributes or {})
                sim = _similarity(p_attrs, i_attrs)
                pairs.append((sim, pi, ii))

        # Sort descending by similarity, greedily match
        pairs.sort(key=lambda x: -x[0])
        matched_pairs: list[tuple[int, int, float]] = []
        MATCH_THRESHOLD = 0.3

        for sim, pi, ii in pairs:
            if pi in used_policy or ii in used_impl:
                continue
            if sim >= MATCH_THRESHOLD:
                matched_pairs.append((pi, ii, sim))
                used_policy.add(pi)
                used_impl.add(ii)

        # Build parameter records for matched pairs
        for pi, ii, sim in matched_pairs:
            p_ext = policy_group[pi]
            i_ext = impl_group[ii]
            p_dict = _extraction_to_dict(p_ext, policy_viz.get(id(p_ext)))
            i_dict = _extraction_to_dict(i_ext, impl_viz.get(id(i_ext)))
            p_attrs = p_dict["attributes"]
            i_attrs = i_dict["attributes"]

            equal, drifted_keys = _values_equal(p_attrs, i_attrs)
            if equal:
                status = "matched"
                drift_reason = None
            else:
                status = "drifted"
                drift_reason = "attribute_mismatch"
                # Check for text-only drift (all attributes same, text differs)
                if not drifted_keys:
                    p_text = _normalize(p_dict["extraction_text"])
                    i_text = _normalize(i_dict["extraction_text"])
                    if p_text != i_text:
                        drift_reason = "text_only_drift"

            params.append({
                "extraction_class": cls,
                "status": status,
                "policy": p_dict,
                "implementation": i_dict,
                "drift_reason": drift_reason,
                "drifted_attributes": drifted_keys,
            })

        # Unmatched policy → missing
        for pi, p_ext in enumerate(policy_group):
            if pi not in used_policy:
                params.append({
                    "extraction_class": cls,
                    "status": "missing",
                    "policy": _extraction_to_dict(p_ext, policy_viz.get(id(p_ext))),
                    "implementation": None,
                    "drift_reason": None,
                    "drifted_attributes": [],
                })

        # Unmatched impl → extra
        for ii, i_ext in enumerate(impl_group):
            if ii not in used_impl:
                params.append({
                    "extraction_class": cls,
                    "status": "extra",
                    "policy": None,
                    "implementation": _extraction_to_dict(i_ext, impl_viz.get(id(i_ext))),
                    "drift_reason": None,
                    "drifted_attributes": [],
                })

    return params


def compare_extractions(
    policy_doc,
    impl_doc,
    comparison_id: str,
    domain: str,
    output_dir: str | Path,
) -> dict:
    """
    Compare parameters between policy and implementation AnnotatedDocuments.

    Writes comparison.json to output/comparisons/<comparison_id>/ and returns
    the comparison dict.
    """
    policy_exts = list(policy_doc.extractions or [])
    impl_exts = list(impl_doc.extractions or [])

    logger.info(
        "Comparing %d policy extractions vs %d implementation extractions",
        len(policy_exts),
        len(impl_exts),
    )

    params = _match_extractions(policy_exts, impl_exts)

    # Assign stable indices
    for i, p in enumerate(params):
        p["index"] = i

    summary = {
        "total": len(params),
        "matched": sum(1 for p in params if p["status"] == "matched"),
        "drifted": sum(1 for p in params if p["status"] == "drifted"),
        "missing": sum(1 for p in params if p["status"] == "missing"),
        "extra": sum(1 for p in params if p["status"] == "extra"),
    }

    comparison = {
        "comparison_id": comparison_id,
        "domain": domain,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "params": params,
    }

    output_dir = Path(output_dir)
    comp_dir = output_dir / "comparisons" / comparison_id
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_path = comp_dir / "comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    logger.info(
        "Comparison complete: matched=%d drifted=%d missing=%d extra=%d → %s",
        summary["matched"],
        summary["drifted"],
        summary["missing"],
        summary["extra"],
        comp_path,
    )
    return comparison


def load_comparison(comparison_id: str, output_dir: str | Path) -> dict:
    """Load a previously saved comparison.json."""
    path = Path(output_dir) / "comparisons" / comparison_id / "comparison.json"
    with open(path) as f:
        return json.load(f)
