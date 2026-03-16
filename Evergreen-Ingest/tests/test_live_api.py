"""
Live API smoke tests — one extraction + one full pipeline test per provider.

Each test group is skipped when the corresponding API key is absent, so the
suite is safe to run on any machine.

Providers covered:
  OpenAI    — gpt-4o-mini              (OPENAI_API_KEY)
  Anthropic — claude-haiku-4-5-20251001 (ANTHROPIC_API_KEY)
  Google    — gemini-2.5-flash          (GOOGLE_API_KEY or GEMINI_API_KEY)

What each smoke test verifies:
  - extract_document returns > 0 extractions
  - JSONL file written to disk and reloadable with identical count
  - Visualization HTML file written and non-empty
  - At least one extraction class name contains a tax-domain keyword
    (tax, filing, credit, penalty, threshold, deduction, withholding, deadline)
  - At least one extraction has non-empty source text

What each pipeline test verifies (policy + implementation → compare):
  - Both extractions complete successfully
  - compare_extractions produces a comparison.json on disk
  - The stale training manual produces at least 1 drifted or missing parameter
  - comparison.json is re-loadable with consistent totals

Run:
    pytest tests/test_live_api.py -v
    pytest tests/test_live_api.py -v -k openai
    pytest tests/test_live_api.py -v -k anthropic
    pytest tests/test_live_api.py -v -k google
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES = Path(__file__).parent / "fixtures"

# Keywords expected in at least one extraction class name for tax-domain docs.
# Models use varying granularity (tax_rate vs flat_income_tax_rate vs filing_threshold_single)
# so we match on keywords rather than exact class names.
TAX_KEYWORDS = {"tax", "filing", "credit", "penalty", "threshold", "deduction", "withholding", "deadline"}


# ---------------------------------------------------------------------------
# Skip markers — one per provider
# ---------------------------------------------------------------------------

REQUIRES_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

REQUIRES_ANTHROPIC = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

REQUIRES_GOOGLE = pytest.mark.skipif(
    not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY / GEMINI_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Autouse fixture: bump max_output_tokens for this file
#
# The default config value (2048) is sufficient for compact providers but can
# be too small for Claude responses with 20+ parameters. We patch the
# module-level config cache in-place so the change applies only during these
# live tests and is restored afterwards.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_token_limit():
    import extract as extract_module
    cfg = extract_module._get_config()
    orig = cfg.get("max_output_tokens", 2048)
    cfg["max_output_tokens"] = 8192
    yield
    cfg["max_output_tokens"] = orig


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------

def _assert_extraction(doc, jsonl_path, viz_path):
    """Common assertions for a successful extraction."""
    assert jsonl_path.exists(), f"JSONL not written: {jsonl_path}"
    assert viz_path.exists(), f"Viz HTML not written: {viz_path}"
    assert viz_path.stat().st_size > 0, "Viz HTML is empty"

    assert doc.extractions is not None, "doc.extractions is None"
    n = len(doc.extractions)
    assert n > 0, "extract_document returned 0 extractions"

    # At least one extraction class contains a tax-domain keyword
    classes = {e.extraction_class or "" for e in doc.extractions}
    assert any(
        any(kw in cls for kw in TAX_KEYWORDS)
        for cls in classes
    ), (
        f"No class contained a tax-domain keyword.\n"
        f"Classes: {sorted(classes)}\n"
        f"Keywords: {TAX_KEYWORDS}"
    )

    # At least one extraction has non-empty text
    assert any((e.extraction_text or "").strip() for e in doc.extractions), \
        "All extractions have empty extraction_text"


def _assert_jsonl_roundtrip(jsonl_path, expected_count):
    """JSONL can be reloaded and yields the same number of extractions."""
    from extract import load_extractions
    reloaded = load_extractions(jsonl_path)
    assert len(reloaded.extractions) == expected_count, (
        f"Roundtrip mismatch: saved {expected_count}, reloaded {len(reloaded.extractions)}"
    )


def _assert_compare(comparison, cid):
    """Common assertions for a successful comparison."""
    assert comparison["comparison_id"] == cid
    summary = comparison["summary"]
    assert summary["total"] > 0, "Comparison found 0 total parameters"
    # Totals add up
    assert (
        summary["matched"] + summary["drifted"] + summary["missing"] + summary["extra"]
        == summary["total"]
    )
    # The stale training manual has 5 intentional drifts + 1 missing — at least
    # one signal of divergence should be detected
    assert summary["drifted"] + summary["missing"] > 0, (
        "Expected ≥1 drifted or missing parameter — the stale training manual "
        "intentionally diverges from the policy."
    )


# ---------------------------------------------------------------------------
# OpenAI — gpt-4o-mini
# ---------------------------------------------------------------------------

@REQUIRES_OPENAI
def test_openai_extract_smoke(tmp_path):
    """gpt-4o-mini: single document extraction, tax domain."""
    from extract import extract_document

    jsonl_path, viz_path, doc = extract_document(
        source_path=FIXTURES / "policy_tax_current.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id="live_openai_smoke",
        doc_slot="policy",
        model_id="gpt-4o-mini",
    )
    _assert_extraction(doc, jsonl_path, viz_path)
    _assert_jsonl_roundtrip(jsonl_path, len(doc.extractions))


@REQUIRES_OPENAI
def test_openai_pipeline_compare(tmp_path):
    """gpt-4o-mini: full two-doc extract + compare on the intentionally drifted tax fixture pair."""
    from extract import extract_document
    from compare import compare_extractions, load_comparison

    cid = "live_openai_pipeline"
    _, _, policy_doc = extract_document(
        source_path=FIXTURES / "policy_tax_current.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id=cid,
        doc_slot="policy",
        model_id="gpt-4o-mini",
    )
    _, _, impl_doc = extract_document(
        source_path=FIXTURES / "training_manual_tax_stale.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id=cid,
        doc_slot="implementation",
        model_id="gpt-4o-mini",
    )
    comparison = compare_extractions(
        policy_doc=policy_doc,
        impl_doc=impl_doc,
        comparison_id=cid,
        domain="tax",
        output_dir=tmp_path,
    )
    _assert_compare(comparison, cid)

    # comparison.json is on disk and reloadable
    loaded = load_comparison(cid, tmp_path)
    assert loaded["summary"]["total"] == comparison["summary"]["total"]


# ---------------------------------------------------------------------------
# Anthropic — claude-haiku-4-5-20251001
# ---------------------------------------------------------------------------

@REQUIRES_ANTHROPIC
def test_anthropic_extract_smoke(tmp_path):
    """claude-haiku-4-5-20251001: single document extraction, tax domain."""
    from extract import extract_document

    jsonl_path, viz_path, doc = extract_document(
        source_path=FIXTURES / "policy_tax_current.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id="live_anthropic_smoke",
        doc_slot="policy",
        model_id="claude-haiku-4-5-20251001",
    )
    _assert_extraction(doc, jsonl_path, viz_path)
    _assert_jsonl_roundtrip(jsonl_path, len(doc.extractions))


@REQUIRES_ANTHROPIC
def test_anthropic_pipeline_compare(tmp_path):
    """claude-haiku-4-5-20251001: full two-doc extract + compare, tax fixture pair."""
    from extract import extract_document
    from compare import compare_extractions, load_comparison

    cid = "live_anthropic_pipeline"
    _, _, policy_doc = extract_document(
        source_path=FIXTURES / "policy_tax_current.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id=cid,
        doc_slot="policy",
        model_id="claude-haiku-4-5-20251001",
    )
    _, _, impl_doc = extract_document(
        source_path=FIXTURES / "training_manual_tax_stale.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id=cid,
        doc_slot="implementation",
        model_id="claude-haiku-4-5-20251001",
    )
    comparison = compare_extractions(
        policy_doc=policy_doc,
        impl_doc=impl_doc,
        comparison_id=cid,
        domain="tax",
        output_dir=tmp_path,
    )
    _assert_compare(comparison, cid)

    loaded = load_comparison(cid, tmp_path)
    assert loaded["summary"]["total"] == comparison["summary"]["total"]


# ---------------------------------------------------------------------------
# Google — gemini-2.5-flash
# ---------------------------------------------------------------------------

@REQUIRES_GOOGLE
def test_google_extract_smoke(tmp_path):
    """gemini-2.5-flash: single document extraction, tax domain."""
    from extract import extract_document

    jsonl_path, viz_path, doc = extract_document(
        source_path=FIXTURES / "policy_tax_current.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id="live_google_smoke",
        doc_slot="policy",
        model_id="gemini-2.5-flash",
    )
    _assert_extraction(doc, jsonl_path, viz_path)
    _assert_jsonl_roundtrip(jsonl_path, len(doc.extractions))


@REQUIRES_GOOGLE
def test_google_pipeline_compare(tmp_path):
    """gemini-2.5-flash: full two-doc extract + compare, tax fixture pair."""
    from extract import extract_document
    from compare import compare_extractions, load_comparison

    cid = "live_google_pipeline"
    _, _, policy_doc = extract_document(
        source_path=FIXTURES / "policy_tax_current.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id=cid,
        doc_slot="policy",
        model_id="gemini-2.5-flash",
    )
    _, _, impl_doc = extract_document(
        source_path=FIXTURES / "training_manual_tax_stale.md",
        domain_name="tax",
        output_dir=tmp_path,
        comparison_id=cid,
        doc_slot="implementation",
        model_id="gemini-2.5-flash",
    )
    comparison = compare_extractions(
        policy_doc=policy_doc,
        impl_doc=impl_doc,
        comparison_id=cid,
        domain="tax",
        output_dir=tmp_path,
    )
    _assert_compare(comparison, cid)

    loaded = load_comparison(cid, tmp_path)
    assert loaded["summary"]["total"] == comparison["summary"]["total"]
