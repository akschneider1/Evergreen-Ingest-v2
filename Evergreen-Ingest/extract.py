"""
extract.py — langextract integration.

Takes a document path and a domain, runs langextract, saves JSONL and HTML
visualization, and returns the extraction results.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import langextract as lx
import yaml

from examples import get_domain

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def read_document(source_path: str | Path) -> str:
    """Read a document to plain text. Handles .md, .txt, .html; returns raw bytes for PDF/DOCX."""
    path = Path(source_path)
    suffix = path.suffix.lower()

    if suffix in (".md", ".txt", ".html", ".htm"):
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        try:
            import pypdf  # optional dep
            reader = pypdf.PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise RuntimeError(
                "PDF support requires pypdf: pip install pypdf"
            )

    if suffix in (".docx",):
        try:
            import docx  # optional dep
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise RuntimeError(
                "DOCX support requires python-docx: pip install python-docx"
            )

    # Fallback: try reading as UTF-8 text
    return path.read_text(encoding="utf-8", errors="replace")


def extract_document(
    source_path: str | Path,
    domain_name: str,
    output_dir: str | Path,
    comparison_id: str,
    doc_slot: str,  # "policy" or "implementation"
    extraction_passes: int | None = None,
) -> tuple[Path, Path, object]:
    """
    Run langextract on a document using domain-specific few-shot examples.

    Returns:
        (jsonl_path, viz_path, annotated_doc)
        - jsonl_path: path to saved JSONL extraction
        - viz_path: path to saved HTML visualization
        - annotated_doc: lx.data.AnnotatedDocument
    """
    config = _load_config()
    domain = get_domain(domain_name)

    output_dir = Path(output_dir)
    extraction_dir = output_dir / "extractions" / comparison_id
    viz_dir = output_dir / "visualizations" / comparison_id
    extraction_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    document_text = read_document(source_path)
    logger.info(
        "Extracting %s/%s (%d chars) with domain=%s model=%s",
        comparison_id,
        doc_slot,
        len(document_text),
        domain_name,
        config.get("model_id", "gemini-2.5-flash"),
    )

    model = config.get("model_id", "gpt-4o-mini")
    if model.startswith("gpt"):
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("LANGEXTRACT_API_KEY")
    result: lx.data.AnnotatedDocument = lx.extract(
        text_or_documents=document_text,
        prompt_description=domain.prompt,
        examples=domain.examples,
        model_id=config.get("model_id", "gemini-2.5-flash"),
        extraction_passes=extraction_passes if extraction_passes is not None else config.get("extraction_passes", 3),
        max_workers=config.get("max_workers", 4),
        max_char_buffer=config.get("max_char_buffer", 6000),
        show_progress=False,
        api_key=api_key,
        # Suppress per-chunk JSON parse errors (e.g. truncated responses hitting
        # output token limits) so the pipeline continues with partial results
        # rather than crashing entirely.
        resolver_params={"suppress_parse_errors": True},
        language_model_params={"max_output_tokens": config.get("max_output_tokens", 8192)},
    )

    # Save JSONL (immutable source record)
    jsonl_path = extraction_dir / f"{doc_slot}.jsonl"
    lx.io.save_annotated_documents(
        [result],
        output_dir=str(extraction_dir),
        output_name=f"{doc_slot}.jsonl",
        show_progress=False,
    )

    # Generate and save HTML visualization
    viz_path = viz_dir / f"{doc_slot}.html"
    try:
        html_str = lx.visualize(str(jsonl_path))
        viz_path.write_text(html_str, encoding="utf-8")
    except Exception as exc:
        logger.warning("Visualization generation failed: %s", exc)
        viz_path.write_text(
            f"<p>Visualization unavailable: {exc}</p>", encoding="utf-8"
        )

    logger.info(
        "Extraction complete: %d parameters extracted → %s",
        len(result.extractions or []),
        jsonl_path,
    )
    return jsonl_path, viz_path, result


def load_extractions(jsonl_path: str | Path) -> lx.data.AnnotatedDocument:
    """Load an AnnotatedDocument from a saved JSONL file."""
    docs = list(lx.io.load_annotated_documents_jsonl(str(jsonl_path)))
    if not docs:
        raise ValueError(f"No documents found in {jsonl_path}")
    return docs[0]
