"""
extract.py — langextract integration.

Takes a document path and a domain, runs langextract, saves JSONL and HTML
visualization, and returns the extraction results.

Performance notes:
- max_char_buffer is dynamically expanded to len(doc)+1 so any document that
  fits in one chunk only makes one API call (demo fixtures are 3–5 KB).
- prompt_validation_level=OFF skips the pre-flight difflib pass.
- Policy + implementation extractions run sequentially (see app.py).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import langextract as lx
from langextract import prompt_validation as pv
from langextract.providers.openai import OpenAILanguageModel as _LxOpenAILanguageModel
import yaml

from examples import get_domain


class _TimedOpenAILanguageModel(_LxOpenAILanguageModel):
    """Thin subclass that sets an explicit HTTP timeout on the OpenAI client.

    langextract creates openai.OpenAI() with no timeout (defaults to 600s).
    This subclass replaces self._client after super().__init__() so every
    chat.completions.create() call respects api_timeout from config.yaml.
    """
    def __init__(self, *, client_timeout: float, **kwargs):
        super().__init__(**kwargs)
        import openai as _oa
        self._client = _oa.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=client_timeout,
            max_retries=0,  # surface errors immediately — don't silently retry/backoff
        )

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

    if model.startswith("gpt"):
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("LANGEXTRACT_API_KEY")

    if not api_key:
        raise RuntimeError(
            f"No API key found for model '{model}'. "
            "Set OPENAI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
        )

    extract_timeout = config.get("api_timeout", 90)

    # For OpenAI models, pre-build the provider with an explicit HTTP timeout.
    # langextract creates openai.OpenAI() with no timeout (defaults to 600s)
    # and silently ignores 'timeout' in language_model_params.
    lx_model = None
    if model.startswith("gpt"):
        lx_model = _TimedOpenAILanguageModel(
            model_id=model,
            api_key=api_key,
            max_workers=config.get("max_workers", 10),
            client_timeout=float(extract_timeout),
            max_output_tokens=config.get("max_output_tokens", 2048),
        )

    logger.info(
        "lx.extract starting — %s/%s model=%s doc_len=%d",
        comparison_id, doc_slot, model, len(document_text),
    )

    try:
        result: lx.data.AnnotatedDocument = lx.extract(
            text_or_documents=document_text,
            prompt_description=domain.prompt,
            examples=domain.examples,
            model_id=model,          # still needed for format-type detection
            model=lx_model,          # None → factory used (Gemini); set → bypasses factory
            extraction_passes=passes,
            max_workers=config.get("max_workers", 10),
            batch_length=10,
            max_char_buffer=effective_buffer,
            show_progress=False,
            api_key=api_key,         # used by Gemini factory path; ignored for OpenAI
            resolver_params={"suppress_parse_errors": True},
            language_model_params={"max_output_tokens": config.get("max_output_tokens", 2048)},
            prompt_validation_level=pv.PromptValidationLevel.OFF,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Extraction failed ({type(exc).__name__}): {exc}"
        ) from exc

    logger.info(
        "lx.extract done — %s/%s: %d extractions",
        comparison_id, doc_slot, len(result.extractions or []),
    )

    # Save JSONL (immutable source record)
    lx.io.save_annotated_documents(
        [result],
        output_dir=str(extraction_dir),
        output_name=f"{doc_slot}.jsonl",
        show_progress=False,
    )

    # Generate and save HTML visualization
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
