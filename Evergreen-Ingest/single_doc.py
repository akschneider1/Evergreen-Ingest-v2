"""
single_doc.py — Single-document extraction view builder.

Takes one AnnotatedDocument (from langextract) and normalises its extractions
into a flat indexed list suitable for the Extract & Validate review workflow.
Writes single_doc.json to output/comparisons/<extraction_id>/ and returns the
dict.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _extraction_to_param(ext, index: int, viz_idx: int | None = None) -> dict:
    """Convert a langextract Extraction object to a plain review-ready dict."""
    attrs = ext.attributes or {}
    result = {
        "index": index,
        "extraction_class": ext.extraction_class or "unknown",
        "attributes": dict(attrs) if attrs else {},
        "extraction_text": ext.extraction_text or "",
        "viz_idx": viz_idx,
        "char_start": None,
        "char_end": None,
    }
    if hasattr(ext, "char_interval") and ext.char_interval is not None:
        result["char_start"] = getattr(ext.char_interval, "start_pos", None)
        result["char_end"] = getattr(ext.char_interval, "end_pos", None)
    return result


def build_single_doc_view(
    doc,
    extraction_id: str,
    domain: str,
    document_name: str,
    model: str,
    output_dir: str | Path,
) -> dict:
    """
    Normalise extractions from a single AnnotatedDocument into a flat indexed list.

    Writes single_doc.json to output/comparisons/<extraction_id>/ and returns
    the dict with keys:
      extraction_id, domain, document_name, model, generated_at, total, parameters
    """
    extractions = list(doc.extractions or [])

    parameters = []
    for i, ext in enumerate(extractions):
        parameters.append(_extraction_to_param(ext, index=i, viz_idx=i))

    single_doc = {
        "extraction_id": extraction_id,
        "domain": domain,
        "document_name": document_name,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(parameters),
        "parameters": parameters,
    }

    output_dir = Path(output_dir)
    doc_dir = output_dir / "comparisons" / extraction_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    out_path = doc_dir / "single_doc.json"
    out_path.write_text(json.dumps(single_doc, indent=2), encoding="utf-8")

    logger.info(
        "Single-doc view built: %d parameter(s) → %s",
        len(parameters),
        out_path,
    )
    return single_doc


def load_single_doc(extraction_id: str, output_dir: str | Path) -> dict:
    """Load a previously saved single_doc.json."""
    path = Path(output_dir) / "comparisons" / extraction_id / "single_doc.json"
    with open(path) as f:
        return json.load(f)
