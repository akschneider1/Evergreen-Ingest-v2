"""
extract.py — langextract integration.

Takes a document path and a domain, runs langextract, saves JSONL and HTML
visualization, and returns the extraction results.

Performance notes:
- max_char_buffer is dynamically expanded to len(doc)+1 so any document that
  fits in one chunk only makes one API call (demo fixtures are 3–5 KB).
- prompt_validation_level=OFF skips the pre-flight difflib pass.
- Policy + implementation extractions run sequentially (see app.py).

Debug mode:
- Set DEBUG_EXTRACTION=1 in your environment to enable debug=True on lx.extract,
  which prints the assembled prompt and raw model response for each chunk.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import langextract as lx
from langextract import prompt_validation as pv
import yaml

from examples import get_domain


logger = logging.getLogger(__name__)


def _load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

# Module-level cache — config.yaml is read once per process, not per extraction call.
_CONFIG: dict | None = None

def _get_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG


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


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_document(
    source_path: str | Path,
    domain_name: str,
    output_dir: str | Path,
    comparison_id: str,
    doc_slot: str,  # "policy" or "implementation"
    extraction_passes: int | None = None,
    model_id: str | None = None,
) -> tuple[Path, Path, object]:
    """
    Run langextract on a document using domain-specific few-shot examples.

    Returns:
        (jsonl_path, viz_path, annotated_doc)
        - jsonl_path: path to saved JSONL extraction
        - viz_path: path to saved HTML visualization
        - annotated_doc: lx.data.AnnotatedDocument
    """
    config = _get_config()
    domain = get_domain(domain_name)

    output_dir = Path(output_dir)
    extraction_dir = output_dir / "extractions" / comparison_id
    viz_dir = output_dir / "visualizations" / comparison_id
    extraction_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    document_text = read_document(source_path)
    passes = extraction_passes if extraction_passes is not None else config.get("extraction_passes", 1)
    model = model_id if model_id is not None else config.get("model_id", "gpt-4o-mini")

    # Expand max_char_buffer so the whole document fits in a single chunk when
    # possible — this halves API calls for documents just over the config limit.
    configured_buffer = config.get("max_char_buffer", 6000)
    effective_buffer = max(configured_buffer, len(document_text) + 1)

    logger.info(
        "Starting extraction %s/%s — %d chars, model=%s, passes=%d, chunk_buffer=%d",
        comparison_id,
        doc_slot,
        len(document_text),
        model,
        passes,
        effective_buffer,
    )

    jsonl_path = extraction_dir / f"{doc_slot}.jsonl"
    viz_path = viz_dir / f"{doc_slot}.html"

    # Resolve API key. Gemini: langextract's factory checks GEMINI_API_KEY or
    # LANGEXTRACT_API_KEY — NOT GOOGLE_API_KEY. Check GEMINI_API_KEY first, then
    # fall back to GOOGLE_API_KEY so existing .env files still work.
    if model.startswith("gpt"):
        api_key = os.environ.get("OPENAI_API_KEY")
    elif model.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    else:
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("LANGEXTRACT_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

    if not api_key:
        key_var = (
            "ANTHROPIC_API_KEY" if model.startswith("claude")
            else "OPENAI_API_KEY" if model.startswith("gpt")
            else "GEMINI_API_KEY"
        )
        raise RuntimeError(
            f"No API key found for model '{model}'. "
            f"Set {key_var} in your environment or .env file."
        )

    # Pre-build the provider for models that need a custom wrapper.
    # Gemini: uses langextract's built-in factory (model=None, model_id+api_key passed directly).
    # OpenAI: uses langextract's built-in factory (model=None, model_id+api_key passed directly).
    # Claude: uses our AnthropicLanguageModel (no built-in factory support in langextract).
    lx_model = None
    if model.startswith("claude"):
        from anthropic_provider import AnthropicLanguageModel
        lx_model = AnthropicLanguageModel(
            model_id=model,
            api_key=api_key,
            max_workers=config.get("max_workers", 10),
            client_timeout=float(config.get("api_timeout", 90)),
            max_output_tokens=config.get("max_output_tokens", 2048),
        )

    debug_mode = bool(os.environ.get("DEBUG_EXTRACTION"))
    if debug_mode:
        logger.info("DEBUG_EXTRACTION=1: debug=True will be passed to lx.extract")

    logger.info(
        "lx.extract starting — %s/%s model=%s doc_len=%d lx_model=%s",
        comparison_id, doc_slot, model, len(document_text),
        type(lx_model).__name__ if lx_model else "factory",
    )

    # Build kwargs. When lx_model is set (Claude), do NOT also pass model_id —
    # langextract ignores model_id when model= is provided, and the dual param
    # is misleading. For factory paths (Gemini, OpenAI), pass model_id + api_key
    # and let langextract create the right provider internally.
    extract_kwargs: dict = dict(
        text_or_documents=document_text,
        prompt_description=domain.prompt,
        examples=domain.examples,
        extraction_passes=passes,
        max_workers=config.get("max_workers", 10),
        batch_length=10,
        max_char_buffer=effective_buffer,
        show_progress=False,
        resolver_params={},
        language_model_params={"max_output_tokens": config.get("max_output_tokens", 2048)},
        prompt_validation_level=pv.PromptValidationLevel.OFF,
    )
    if lx_model is not None:
        extract_kwargs["model"] = lx_model
        extract_kwargs["use_schema_constraints"] = False
    else:
        extract_kwargs["model_id"] = model
        extract_kwargs["api_key"] = api_key
    if debug_mode:
        extract_kwargs["debug"] = True

    try:
        result: lx.data.AnnotatedDocument = lx.extract(**extract_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Extraction failed ({type(exc).__name__}): {exc}"
        ) from exc

    n_extractions = len(result.extractions or [])
    if n_extractions == 0:
        logger.warning(
            "lx.extract returned 0 extractions for %s/%s — "
            "set DEBUG_EXTRACTION=1 and re-run to see the raw model response",
            comparison_id, doc_slot,
        )
    else:
        logger.info(
            "lx.extract done — %s/%s: %d extractions",
            comparison_id, doc_slot, n_extractions,
        )

    # Save JSONL (immutable source record)
    lx.io.save_annotated_documents(
        [result],
        output_dir=str(extraction_dir),
        output_name=f"{doc_slot}.jsonl",
        show_progress=False,
    )

    # Generate and save HTML visualization.
    # Pass the AnnotatedDocument directly — avoids re-reading the JSONL file
    # and handles edge cases where document_id is falsy (empty JSONL).
    try:
        html_str = lx.visualize(result)
        if not isinstance(html_str, str):
            # IPython HTML object (only in Jupyter; shouldn't happen here)
            html_str = html_str.data if hasattr(html_str, "data") else str(html_str)
        viz_path.write_text(html_str, encoding="utf-8")
    except Exception as exc:
        logger.warning("Visualization generation failed for %s/%s: %s", comparison_id, doc_slot, exc)
        viz_path.write_text(
            f"<html><body style='font-family:sans-serif;padding:2rem;color:#666'>"
            f"<p><strong>Visualization unavailable</strong></p>"
            f"<p>{type(exc).__name__}: {exc}</p>"
            f"</body></html>",
            encoding="utf-8",
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
