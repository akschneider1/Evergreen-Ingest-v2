"""
app.py — FastAPI application for Evergreen Ingest.

All routes. Extraction runs in a ThreadPoolExecutor (langextract is sync)
and the browser polls /compare/{id}/status until ready.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import uuid
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import compare as compare_module
import extract as extract_module
import single_doc as single_doc_module
import validate as validate_module

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class _PipelineLogHandler(logging.Handler):
    """Captures INFO+ log records from extraction modules into meta['logs'] in real time.

    Installed on the 'extract' and 'anthropic_provider' loggers during a pipeline
    run so the status page can display live progress without polling stdout.
    """

    def __init__(self, meta: dict, comparison_id: str) -> None:
        super().__init__(level=logging.INFO)
        self._meta = meta
        self._comparison_id = comparison_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            level_tag = record.levelname[0]  # I / W / E
            msg = record.getMessage()
            # Strip the comparison_id prefix that extract.py includes in messages
            msg = msg.replace(self._comparison_id + "/", "")
            entry = f"[{ts}] {level_tag} {record.name}: {msg}"
            self._meta["logs"].append(entry)
            _write_meta(self._comparison_id, self._meta)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
OUTPUT_DIR = BASE_DIR / CONFIG.get("output_dir", "output")
SOURCES_DIR = BASE_DIR / CONFIG.get("sources_dir", "sources")
UPLOAD_MAX_BYTES: int = CONFIG.get("upload_max_bytes", 5 * 1024 * 1024)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_DIR.mkdir(parents=True, exist_ok=True)
for sub in ("extractions", "visualizations", "comparisons", "validations", "sources"):
    (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Evergreen Ingest")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return user-friendly HTML for 404s instead of bare JSON."""
    if exc.status_code == 404:
        return HTMLResponse(
            content=(
                f"<html><head><title>Not Found — Evergreen Ingest</title>"
                f"<link rel='stylesheet' href='/static/style.css'></head>"
                f"<body>"
                f"<header><div class='header-inner'>"
                f"<a href='/' class='logo'>Evergreen Ingest</a>"
                f"<nav class='header-nav'><a href='/comparisons'>History</a></nav>"
                f"</div></header>"
                f"<main style='padding:3rem 2rem;max-width:600px;margin:0 auto'>"
                f"<h1 style='color:#dc2626'>Not Found</h1>"
                f"<p style='color:#555'>{exc.detail}</p>"
                f"<p>The comparison may have been lost if the server restarted. "
                f"<a href='/comparisons'>View history</a> or "
                f"<a href='/'>start a new comparison</a>.</p>"
                f"</main></body></html>"
            ),
            status_code=404,
        )
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

executor = ThreadPoolExecutor(max_workers=2)

PIPELINE_TIMEOUT_SECONDS = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_comparison_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


def _meta_path(comparison_id: str) -> Path:
    return OUTPUT_DIR / "comparisons" / comparison_id / "meta.json"


def _read_meta(comparison_id: str) -> dict:
    path = _meta_path(comparison_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Comparison not found")
    with open(path) as f:
        return json.load(f)


def _write_meta(comparison_id: str, meta: dict) -> None:
    path = _meta_path(comparison_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)


def _friendly_error(exc: Exception, model_id: str = "") -> str:
    msg = str(exc)
    low = msg.lower()
    if "api_key" in low or "api key" in low or "authentication" in low or "unauthenticated" in low or "incorrect api key" in low:
        return "API key error — check that OPENAI_API_KEY (or GOOGLE_API_KEY) is set correctly in your environment."
    if "quota" in low or "resource_exhausted" in low:
        if model_id.startswith("gpt"):
            return "API quota exceeded. Wait a moment and try again, or check your OpenAI quota at platform.openai.com."
        return "API quota exceeded. Wait a moment and try again, or check your Google Cloud quota."
    if "rate" in low and ("limit" in low or "429" in msg):
        return "API rate limit hit. Wait a moment and try again."
    if "timeout" in low or "deadline" in low:
        return "Request timed out. Try again — large documents occasionally need a retry."
    if ("model" in low and ("not found" in low or "invalid" in low)) or "404" in msg:
        return f"Model not found or unavailable. Check MODEL_ID in config.yaml. Detail: {msg}"
    if "permission" in low or "403" in msg:
        return "Permission denied. Ensure your API key has access to the selected model's API."
    return msg


def _allowed_suffix(filename: str) -> bool:
    return Path(filename).suffix.lower() in {
        ".md", ".txt", ".html", ".htm", ".pdf", ".docx"
    }


# ---------------------------------------------------------------------------
# Background extraction pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    comparison_id: str,
    policy_path: Path,
    impl_path: Path,
    domain: str,
) -> None:
    """Run both extractions and comparison. Called in a thread pool."""
    meta = _read_meta(comparison_id)
    meta["logs"] = []
    meta["started_at"] = datetime.now(timezone.utc).isoformat()

    def _log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        logger.info("[%s] %s", comparison_id, msg)
        meta["logs"].append(entry)
        _write_meta(comparison_id, meta)

    # Attach live log handler so detailed extraction progress appears in the UI.
    _live_handler = _PipelineLogHandler(meta, comparison_id)
    _live_loggers = [
        logging.getLogger("extract"),
        logging.getLogger("anthropic_provider"),
    ]
    for _lgg in _live_loggers:
        _lgg.addHandler(_live_handler)

    try:
        passes = meta.get("extraction_passes", CONFIG.get("extraction_passes", 1))
        model_id = meta.get("model", CONFIG.get("model_id", "gpt-4o-mini"))

        meta["status"] = "extracting"
        _write_meta(comparison_id, meta)
        _log(
            f"Extracting policy document… "
            f"(model={model_id}, {passes} pass{'es' if passes != 1 else ''})"
        )

        # Run extractions sequentially to avoid concurrent API calls that can
        # trigger rate limits on accounts with tight per-minute quotas.
        policy_jsonl, _, policy_doc = extract_module.extract_document(
            source_path=policy_path,
            domain_name=domain,
            output_dir=OUTPUT_DIR,
            comparison_id=comparison_id,
            doc_slot="policy",
            extraction_passes=passes,
            model_id=model_id,
        )

        _log("Extracting implementation document…")
        impl_jsonl, _, impl_doc = extract_module.extract_document(
            source_path=impl_path,
            domain_name=domain,
            output_dir=OUTPUT_DIR,
            comparison_id=comparison_id,
            doc_slot="implementation",
            extraction_passes=passes,
            model_id=model_id,
        )

        n_policy = len(policy_doc.extractions or [])
        n_impl = len(impl_doc.extractions or [])
        _log(
            f"Extractions complete — policy: {n_policy} parameter(s), "
            f"implementation: {n_impl} parameter(s)."
        )

        meta["status"] = "comparing"
        _write_meta(comparison_id, meta)
        _log("Comparing parameters across both documents…")

        compare_module.compare_extractions(
            policy_doc=policy_doc,
            impl_doc=impl_doc,
            comparison_id=comparison_id,
            domain=domain,
            output_dir=OUTPUT_DIR,
        )

        _log("Comparison complete. Loading results…")
        meta["status"] = "ready"
        _write_meta(comparison_id, meta)
        logger.info("[%s] Pipeline complete.", comparison_id)

    except Exception as exc:
        logger.exception("[%s] Pipeline failed: %s", comparison_id, exc)
        meta["status"] = "error"
        meta["error"] = _friendly_error(exc, model_id=model_id)
        meta["error_detail"] = f"{type(exc).__name__}: {exc}"
        meta["logs"].append(f"[error] {type(exc).__name__}: {exc}")
        _write_meta(comparison_id, meta)
    finally:
        for _lgg in _live_loggers:
            _lgg.removeHandler(_live_handler)


# ---------------------------------------------------------------------------
# Single-document extraction pipeline
# ---------------------------------------------------------------------------

def _run_single_pipeline(
    extraction_id: str,
    source_path: Path,
    domain: str,
) -> None:
    """Run extraction on a single document. Called in a thread pool."""
    meta = _read_meta(extraction_id)
    meta["logs"] = []
    meta["started_at"] = datetime.now(timezone.utc).isoformat()

    def _log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        logger.info("[%s] %s", extraction_id, msg)
        meta["logs"].append(entry)
        _write_meta(extraction_id, meta)

    _live_handler = _PipelineLogHandler(meta, extraction_id)
    _live_loggers = [
        logging.getLogger("extract"),
        logging.getLogger("anthropic_provider"),
    ]
    for _lgg in _live_loggers:
        _lgg.addHandler(_live_handler)

    try:
        passes = meta.get("extraction_passes", CONFIG.get("extraction_passes", 1))
        model_id = meta.get("model", CONFIG.get("model_id", "gpt-4o-mini"))

        meta["status"] = "extracting"
        _write_meta(extraction_id, meta)
        _log(
            f"Extracting document… "
            f"(model={model_id}, {passes} pass{'es' if passes != 1 else ''})"
        )

        _jsonl, _viz, doc = extract_module.extract_document(
            source_path=source_path,
            domain_name=domain,
            output_dir=OUTPUT_DIR,
            comparison_id=extraction_id,
            doc_slot="document",
            extraction_passes=passes,
            model_id=model_id,
        )

        n = len(doc.extractions or [])
        _log(f"Extraction complete — {n} parameter(s) found.")

        single_doc_module.build_single_doc_view(
            doc=doc,
            extraction_id=extraction_id,
            domain=domain,
            document_name=meta.get("document_filename", source_path.name),
            model=model_id,
            output_dir=OUTPUT_DIR,
        )

        meta["status"] = "ready"
        _write_meta(extraction_id, meta)
        logger.info("[%s] Single-doc pipeline complete.", extraction_id)

    except Exception as exc:
        logger.exception("[%s] Single-doc pipeline failed: %s", extraction_id, exc)
        meta["status"] = "error"
        meta["error"] = _friendly_error(exc, model_id=model_id)
        meta["error_detail"] = f"{type(exc).__name__}: {exc}"
        meta["logs"].append(f"[error] {type(exc).__name__}: {exc}")
        _write_meta(extraction_id, meta)
    finally:
        for _lgg in _live_loggers:
            _lgg.removeHandler(_live_handler)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/comparisons", response_class=HTMLResponse)
async def history(request: Request):
    """List all prior comparisons, newest first."""
    entries = []
    for meta_path in sorted(
        (OUTPUT_DIR / "comparisons").glob("*/meta.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue

        # Attach summary counts if ready
        summary = None
        if meta.get("mode") == "extract":
            sd_path = meta_path.parent / "single_doc.json"
            if sd_path.exists():
                try:
                    with open(sd_path) as f:
                        sd = json.load(f)
                    summary = {"total": sd.get("total", 0)}
                except Exception:
                    pass
        else:
            comp_path = meta_path.parent / "comparison.json"
            if comp_path.exists():
                try:
                    with open(comp_path) as f:
                        summary = json.load(f).get("summary")
                except Exception:
                    pass

        # Derive a readable display name
        date_str = (meta.get("created_at", "") or "")[:10]
        if not meta.get("display_name"):
            if meta.get("mode") == "extract":
                doc_stem = Path(meta.get("document_filename", "document")).stem
                meta["display_name"] = f"{meta.get('domain', '').title()}: {doc_stem}"
            else:
                policy_stem = Path(meta.get("policy_filename", "policy")).stem
                impl_stem = Path(meta.get("implementation_filename", "implementation")).stem
                meta["display_name"] = f"{meta.get('domain', '').title()}: {policy_stem} vs {impl_stem}"
        meta["display_date"] = date_str
        meta["summary"] = summary
        entries.append(meta)

    return templates.TemplateResponse(
        "history.html", {"request": request, "entries": entries}
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "domains": CONFIG.get("domains", ["tax", "benefits"])},
    )


DEMO_FILES = {
    "tax_policy": BASE_DIR / "tests/fixtures/policy_tax_current.md",
    "tax_implementation": BASE_DIR / "tests/fixtures/training_manual_tax_stale.md",
    "benefits_policy": BASE_DIR / "tests/fixtures/policy_benefits_current.html",
    "benefits_implementation": BASE_DIR / "tests/fixtures/training_manual_benefits.html",
}


ALLOWED_MODELS = {
    "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6",
    "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1",
    "gemini-2.5-flash", "gemini-2.0-flash",
}


@app.post("/upload")
async def upload(
    policy_file: UploadFile = Form(...),
    implementation_file: UploadFile = Form(...),
    domain: str = Form(...),
    model: str = Form("claude-sonnet-4-6"),
    custom_prompt: str = Form(""),
    policy_demo: str = Form(""),
    implementation_demo: str = Form(""),
    extraction_passes: int = Form(1),
):
    # Substitute demo fixture if no real file was uploaded
    def _resolve(upload: UploadFile, demo_key: str) -> tuple[str, bytes]:
        if demo_key and demo_key in DEMO_FILES and (not upload.filename or upload.size == 0):
            fixture = DEMO_FILES[demo_key]
            return fixture.name, fixture.read_bytes()
        return upload.filename or "doc.txt", None  # bytes read below

    policy_name, policy_demo_bytes = _resolve(policy_file, policy_demo)
    impl_name, impl_demo_bytes = _resolve(implementation_file, implementation_demo)

    # Validate filenames
    for fname in (policy_name, impl_name):
        if not _allowed_suffix(fname):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {fname}. Accepted: .md .txt .html .pdf .docx",
            )

    comparison_id = _make_comparison_id()

    # Read and size-check uploads
    policy_bytes = policy_demo_bytes if policy_demo_bytes is not None else await policy_file.read()
    impl_bytes = impl_demo_bytes if impl_demo_bytes is not None else await implementation_file.read()

    for name, data in (
        (policy_name, policy_bytes),
        (impl_name, impl_bytes),
    ):
        if len(data) > UPLOAD_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{name} exceeds maximum size of {UPLOAD_MAX_BYTES // 1024 // 1024} MB",
            )

    # Save uploaded files
    policy_suffix = Path(policy_name).suffix.lower()
    impl_suffix = Path(impl_name).suffix.lower()

    policy_path = SOURCES_DIR / f"{comparison_id}_policy{policy_suffix}"
    impl_path = SOURCES_DIR / f"{comparison_id}_implementation{impl_suffix}"

    policy_path.write_bytes(policy_bytes)
    impl_path.write_bytes(impl_bytes)

    # Validate model
    if model not in ALLOWED_MODELS:
        model = "claude-sonnet-4-6"

    # Derive a human-readable display name for breadcrumbs and history
    policy_stem = Path(policy_name).stem
    impl_stem = Path(impl_name).stem
    date_str = datetime.now(timezone.utc).strftime("%b %-d")
    display_name = f"{domain.title()}: {policy_stem} vs {impl_stem} — {date_str}"

    # Write initial meta (status: pending)
    meta = {
        "comparison_id": comparison_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "domain": domain,
        "model": model,
        "custom_prompt": custom_prompt,
        "policy_filename": policy_name,
        "implementation_filename": impl_name,
        "policy_path": str(policy_path.relative_to(BASE_DIR)),
        "implementation_path": str(impl_path.relative_to(BASE_DIR)),
        "extraction_passes": max(1, min(5, extraction_passes)),
        "display_name": display_name,
        "error": None,
    }
    _write_meta(comparison_id, meta)

    # Kick off extraction in background thread
    executor.submit(_run_pipeline, comparison_id, policy_path, impl_path, domain)

    return RedirectResponse(
        url=f"/compare/{comparison_id}/status", status_code=303
    )


@app.post("/upload-single")
async def upload_single(
    document_file: UploadFile = Form(...),
    domain: str = Form(...),
    model: str = Form("claude-sonnet-4-6"),
    custom_prompt: str = Form(""),
    document_demo: str = Form(""),
    extraction_passes: int = Form(1),
):
    """Upload a single document and start the Extract & Validate pipeline."""
    def _resolve(upload: UploadFile, demo_key: str) -> tuple[str, bytes]:
        if demo_key and demo_key in DEMO_FILES and (not upload.filename or upload.size == 0):
            fixture = DEMO_FILES[demo_key]
            return fixture.name, fixture.read_bytes()
        return upload.filename or "doc.txt", None

    doc_name, demo_bytes = _resolve(document_file, document_demo)

    if not _allowed_suffix(doc_name):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {doc_name}. Accepted: .md .txt .html .pdf .docx",
        )

    extraction_id = _make_comparison_id()

    doc_bytes = demo_bytes if demo_bytes is not None else await document_file.read()
    if len(doc_bytes) > UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{doc_name} exceeds maximum size of {UPLOAD_MAX_BYTES // 1024 // 1024} MB",
        )

    doc_suffix = Path(doc_name).suffix.lower()
    doc_path = SOURCES_DIR / f"{extraction_id}_document{doc_suffix}"
    doc_path.write_bytes(doc_bytes)

    if model not in ALLOWED_MODELS:
        model = "claude-sonnet-4-6"

    doc_stem = Path(doc_name).stem
    date_str = datetime.now(timezone.utc).strftime("%b %-d")
    display_name = f"{domain.title()}: {doc_stem} — {date_str}"

    meta = {
        "comparison_id": extraction_id,
        "mode": "extract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "domain": domain,
        "model": model,
        "custom_prompt": custom_prompt,
        "document_filename": doc_name,
        "document_path": str(doc_path.relative_to(BASE_DIR)),
        "extraction_passes": max(1, min(5, extraction_passes)),
        "display_name": display_name,
        "error": None,
        # Populate these so status.html meta-list renders cleanly
        "policy_filename": doc_name,
        "implementation_filename": "",
    }
    _write_meta(extraction_id, meta)

    executor.submit(_run_single_pipeline, extraction_id, doc_path, domain)

    return RedirectResponse(
        url=f"/extract/{extraction_id}/status", status_code=303
    )


def _check_timeout(meta: dict) -> dict:
    """If still processing and past the timeout, mark as error."""
    if meta.get("status") not in ("ready", "error"):
        ref = meta.get("started_at") or meta.get("created_at")
        if ref:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(ref)).total_seconds()
            if elapsed > PIPELINE_TIMEOUT_SECONDS:
                meta["status"] = "error"
                meta["error"] = (
                    f"Processing timed out after {PIPELINE_TIMEOUT_SECONDS // 60} minutes. "
                    "This sometimes happens with very large documents or API delays — please try again."
                )
                _write_meta(meta["comparison_id"], meta)
    return meta


@app.get("/compare/{comparison_id}/status.json")
async def status_json(comparison_id: str):
    meta = _check_timeout(_read_meta(comparison_id))
    return JSONResponse(meta)


@app.get("/compare/{comparison_id}/status", response_class=HTMLResponse)
async def status_page(request: Request, comparison_id: str):
    meta = _check_timeout(_read_meta(comparison_id))
    if meta["status"] == "ready":
        return RedirectResponse(url=f"/compare/{comparison_id}", status_code=303)
    return templates.TemplateResponse(
        "status.html",
        {"request": request, "meta": meta, "comparison_id": comparison_id},
    )


@app.get("/compare/{comparison_id}", response_class=HTMLResponse)
async def compare_view(request: Request, comparison_id: str):
    meta = _read_meta(comparison_id)
    if meta["status"] != "ready":
        return RedirectResponse(
            url=f"/compare/{comparison_id}/status", status_code=303
        )

    comparison = compare_module.load_comparison(comparison_id, OUTPUT_DIR)
    state = validate_module.load_state(comparison_id, OUTPUT_DIR)

    # Annotate params with validation decision for display
    for p in comparison["params"]:
        p["decision"] = validate_module.get_decision(state, p["index"])

    summary = comparison["summary"]
    actionable = summary["drifted"] + summary["missing"] + summary["extra"]
    progress = validate_module.validation_progress(state, summary["total"], actionable)

    return templates.TemplateResponse(
        "compare.html",
        {
            "request": request,
            "meta": meta,
            "comparison": comparison,
            "progress": progress,
        },
    )


_VIZ_SCROLL_JS = """<script>
(function () {
  var m = location.hash.match(/[#&]idx=(\d+)/);
  if (!m) return;
  var el = document.querySelector('[data-idx="' + m[1] + '"]');
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  el.style.outline = '3px solid #f59e0b';
  el.style.borderRadius = '3px';
  el.style.backgroundColor = 'rgba(245, 158, 11, 0.2)';
})();
</script>"""


@app.get("/compare/{comparison_id}/viz/{doc_slot}", response_class=HTMLResponse)
async def viz(comparison_id: str, doc_slot: str):
    if doc_slot not in ("policy", "implementation"):
        raise HTTPException(status_code=400, detail="doc_slot must be 'policy' or 'implementation'")
    viz_path = OUTPUT_DIR / "visualizations" / comparison_id / f"{doc_slot}.html"
    if not viz_path.exists():
        return HTMLResponse(
            content=(
                "<html><body style='font-family:sans-serif;padding:2rem;color:#888'>"
                "<p><strong>Visualization not available.</strong></p>"
                "<p>The extraction visualization for this document could not be found. "
                "This can happen if the server was restarted after the extraction ran. "
                "Re-running the comparison will regenerate it.</p>"
                "</body></html>"
            ),
            status_code=200,
        )
    html = viz_path.read_text(encoding="utf-8")
    # Inject scroll-to-highlight JS so callers can link to #idx=N
    if "</body>" in html:
        html = html.replace("</body>", _VIZ_SCROLL_JS + "\n</body>", 1)
    else:
        html += _VIZ_SCROLL_JS
    return HTMLResponse(content=html)


@app.get("/compare/{comparison_id}/param/{index}", response_class=HTMLResponse)
async def param_detail(request: Request, comparison_id: str, index: int):
    meta = _read_meta(comparison_id)
    comparison = compare_module.load_comparison(comparison_id, OUTPUT_DIR)
    state = validate_module.load_state(comparison_id, OUTPUT_DIR)

    params = comparison["params"]
    if index < 0 or index >= len(params):
        raise HTTPException(status_code=404, detail="Parameter not found")

    param = params[index]
    decision = validate_module.get_decision(state, index)

    # Build prev/next navigation indices, skipping matched
    actionable = [p["index"] for p in params if p["status"] != "matched"]
    prev_idx = next_idx = None
    actionable_position = None
    if index in actionable:
        pos = actionable.index(index)
        prev_idx = actionable[pos - 1] if pos > 0 else None
        next_idx = actionable[pos + 1] if pos < len(actionable) - 1 else None
        actionable_position = pos + 1

    return templates.TemplateResponse(
        "detail.html",
        {
            "request": request,
            "meta": meta,
            "comparison": comparison,
            "param": param,
            "decision": decision,
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            "comparison_id": comparison_id,
            "actionable_position": actionable_position,
            "actionable_total": len(actionable),
        },
    )


@app.post("/compare/{comparison_id}/param/{index}")
async def submit_validation(
    request: Request,
    comparison_id: str,
    index: int,
    decision: str = Form(...),
    note: str = Form(""),
    edited_json: str = Form(""),
):
    _read_meta(comparison_id)  # existence check

    if decision not in ("confirmed", "edited", "rejected"):
        raise HTTPException(status_code=400, detail="Invalid decision value")

    edited_value = None
    if decision == "edited" and edited_json.strip():
        try:
            edited_value = json.loads(edited_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON in edited value: {exc}"
            )

    validate_module.save_decision(
        comparison_id=comparison_id,
        output_dir=OUTPUT_DIR,
        param_index=index,
        decision=decision,
        edited_value=edited_value,
        note=note,
    )

    # Redirect back to same param detail page
    return RedirectResponse(
        url=f"/compare/{comparison_id}/param/{index}", status_code=303
    )


@app.get("/compare/{comparison_id}/report", response_class=HTMLResponse)
async def report(request: Request, comparison_id: str):
    meta = _read_meta(comparison_id)
    comparison = compare_module.load_comparison(comparison_id, OUTPUT_DIR)
    state = validate_module.load_state(comparison_id, OUTPUT_DIR)

    summary = comparison["summary"]
    actionable = summary["drifted"] + summary["missing"] + summary["extra"]
    progress = validate_module.validation_progress(state, summary["total"], actionable)

    # Annotate params for report
    actionable_params = [
        {**p, "decision": validate_module.get_decision(state, p["index"])}
        for p in comparison["params"]
        if p["status"] != "matched"
    ]

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "meta": meta,
            "comparison": comparison,
            "actionable_params": actionable_params,
            "progress": progress,
        },
    )


@app.get("/compare/{comparison_id}/report.csv")
async def report_csv(comparison_id: str):
    comparison = compare_module.load_comparison(comparison_id, OUTPUT_DIR)
    state = validate_module.load_state(comparison_id, OUTPUT_DIR)

    output = io.StringIO()
    fieldnames = [
        "index", "extraction_class", "status",
        "policy_text", "policy_value",
        "impl_text", "impl_value",
        "drifted_keys", "decision", "note", "decided_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for p in comparison["params"]:
        dec = validate_module.get_decision(state, p["index"])
        writer.writerow({
            "index": p["index"] + 1,
            "extraction_class": p["extraction_class"],
            "status": p["status"],
            "policy_text": p["policy"]["extraction_text"] if p["policy"] else "",
            "policy_value": (p["policy"]["attributes"] or {}).get("value", "") if p["policy"] else "",
            "impl_text": p["implementation"]["extraction_text"] if p["implementation"] else "",
            "impl_value": (p["implementation"]["attributes"] or {}).get("value", "") if p["implementation"] else "",
            "drifted_keys": ", ".join(p.get("drifted_attributes") or []),
            "decision": dec["decision"] if dec else "",
            "note": dec["note"] if dec else "",
            "decided_at": dec["decided_at"] if dec else "",
        })

    from fastapi.responses import Response
    filename = f"drift-report-{comparison_id}.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Extract & Validate routes  (/extract/{id}/...)
# ---------------------------------------------------------------------------

@app.get("/extract/{extraction_id}/status.json")
async def extract_status_json(extraction_id: str):
    meta = _check_timeout(_read_meta(extraction_id))
    return JSONResponse(meta)


@app.get("/extract/{extraction_id}/status", response_class=HTMLResponse)
async def extract_status_page(request: Request, extraction_id: str):
    meta = _check_timeout(_read_meta(extraction_id))
    if meta["status"] == "ready":
        return RedirectResponse(url=f"/extract/{extraction_id}", status_code=303)
    return templates.TemplateResponse(
        "status.html",
        {
            "request": request,
            "meta": meta,
            "comparison_id": extraction_id,
            "mode": "extract",
        },
    )


@app.get("/extract/{extraction_id}", response_class=HTMLResponse)
async def extract_review(request: Request, extraction_id: str):
    meta = _read_meta(extraction_id)
    if meta["status"] != "ready":
        return RedirectResponse(url=f"/extract/{extraction_id}/status", status_code=303)

    single_doc = single_doc_module.load_single_doc(extraction_id, OUTPUT_DIR)
    state = validate_module.load_state(extraction_id, OUTPUT_DIR)

    for p in single_doc["parameters"]:
        p["decision"] = validate_module.get_decision(state, p["index"])

    total = single_doc["total"]
    progress = validate_module.validation_progress(state, total, total)

    return templates.TemplateResponse(
        "extract_review.html",
        {
            "request": request,
            "meta": meta,
            "single_doc": single_doc,
            "extraction_id": extraction_id,
            "progress": progress,
        },
    )


@app.get("/extract/{extraction_id}/viz", response_class=HTMLResponse)
async def extract_viz(extraction_id: str):
    viz_path = OUTPUT_DIR / "visualizations" / extraction_id / "document.html"
    if not viz_path.exists():
        return HTMLResponse(
            content=(
                "<html><body style='font-family:sans-serif;padding:2rem;color:#888'>"
                "<p><strong>Visualization not available.</strong></p>"
                "<p>The extraction visualization could not be found.</p>"
                "</body></html>"
            ),
            status_code=200,
        )
    html = viz_path.read_text(encoding="utf-8")
    if "</body>" in html:
        html = html.replace("</body>", _VIZ_SCROLL_JS + "\n</body>", 1)
    else:
        html += _VIZ_SCROLL_JS
    return HTMLResponse(content=html)


@app.get("/extract/{extraction_id}/param/{index}", response_class=HTMLResponse)
async def extract_param_detail(request: Request, extraction_id: str, index: int):
    meta = _read_meta(extraction_id)
    single_doc = single_doc_module.load_single_doc(extraction_id, OUTPUT_DIR)
    state = validate_module.load_state(extraction_id, OUTPUT_DIR)

    params = single_doc["parameters"]
    if index < 0 or index >= len(params):
        raise HTTPException(status_code=404, detail="Parameter not found")

    param = params[index]
    decision = validate_module.get_decision(state, index)

    prev_idx = index - 1 if index > 0 else None
    next_idx = index + 1 if index < len(params) - 1 else None

    return templates.TemplateResponse(
        "extract_detail.html",
        {
            "request": request,
            "meta": meta,
            "single_doc": single_doc,
            "param": param,
            "decision": decision,
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            "extraction_id": extraction_id,
            "position": index + 1,
            "total": len(params),
        },
    )


@app.post("/extract/{extraction_id}/param/{index}")
async def extract_submit_validation(
    request: Request,
    extraction_id: str,
    index: int,
    decision: str = Form(...),
    note: str = Form(""),
    edited_json: str = Form(""),
):
    _read_meta(extraction_id)

    if decision not in ("confirmed", "edited", "rejected"):
        raise HTTPException(status_code=400, detail="Invalid decision value")

    edited_value = None
    if decision == "edited" and edited_json.strip():
        try:
            edited_value = json.loads(edited_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON in edited value: {exc}"
            )

    validate_module.save_decision(
        comparison_id=extraction_id,
        output_dir=OUTPUT_DIR,
        param_index=index,
        decision=decision,
        edited_value=edited_value,
        note=note,
    )

    return RedirectResponse(
        url=f"/extract/{extraction_id}/param/{index}", status_code=303
    )


@app.get("/extract/{extraction_id}/report", response_class=HTMLResponse)
async def extract_report(request: Request, extraction_id: str):
    meta = _read_meta(extraction_id)
    single_doc = single_doc_module.load_single_doc(extraction_id, OUTPUT_DIR)
    state = validate_module.load_state(extraction_id, OUTPUT_DIR)

    total = single_doc["total"]
    progress = validate_module.validation_progress(state, total, total)

    params_with_decisions = [
        {**p, "decision": validate_module.get_decision(state, p["index"])}
        for p in single_doc["parameters"]
    ]

    return templates.TemplateResponse(
        "extract_report.html",
        {
            "request": request,
            "meta": meta,
            "single_doc": single_doc,
            "extraction_id": extraction_id,
            "params": params_with_decisions,
            "progress": progress,
        },
    )


@app.get("/extract/{extraction_id}/report.json")
async def extract_report_json(extraction_id: str):
    meta = _read_meta(extraction_id)
    single_doc = single_doc_module.load_single_doc(extraction_id, OUTPUT_DIR)
    state = validate_module.load_state(extraction_id, OUTPUT_DIR)

    records = []
    for p in single_doc["parameters"]:
        dec = validate_module.get_decision(state, p["index"])
        records.append({
            "id": f"{extraction_id}:{p['index']}",
            "domain": meta.get("domain", ""),
            "extraction_class": p["extraction_class"],
            "attributes": p["attributes"],
            "extraction_text": p["extraction_text"],
            "char_start": p.get("char_start"),
            "char_end": p.get("char_end"),
            "document": meta.get("document_filename", ""),
            "decision": dec["decision"] if dec else None,
            "corrected_value": dec.get("edited_value") if dec else None,
            "note": dec.get("note") if dec else None,
            "decided_at": dec.get("decided_at") if dec else None,
            "extraction_id": extraction_id,
            "model": meta.get("model", ""),
            "created_at": meta.get("created_at", ""),
        })

    return JSONResponse(content={
        "extraction_id": extraction_id,
        "domain": meta.get("domain", ""),
        "document": meta.get("document_filename", ""),
        "model": meta.get("model", ""),
        "total": single_doc["total"],
        "parameters": records,
    })


# ---------------------------------------------------------------------------
# Deploy routes (two-doc flow)
# ---------------------------------------------------------------------------

@app.get("/compare/{comparison_id}/deploy", response_class=HTMLResponse)
async def deploy(request: Request, comparison_id: str):
    meta = _read_meta(comparison_id)
    comparison = compare_module.load_comparison(comparison_id, OUTPUT_DIR)
    state = validate_module.load_state(comparison_id, OUTPUT_DIR)

    counts = {"confirmed": 0, "edited": 0, "rejected": 0, "matched": 0, "pending": 0}
    for p in comparison["params"]:
        dec = validate_module.get_decision(state, p["index"])
        if p["status"] == "matched":
            counts["matched"] += 1
        elif dec:
            counts[dec["decision"]] = counts.get(dec["decision"], 0) + 1
        else:
            counts["pending"] += 1

    return templates.TemplateResponse(
        "deploy.html",
        {
            "request": request,
            "meta": meta,
            "comparison": comparison,
            "counts": counts,
        },
    )


@app.post("/compare/{comparison_id}/push/evals")
async def push_evals(
    comparison_id: str,
    endpoint_url: str = Form(""),
    api_key: str = Form(""),
):
    """
    Placeholder push to an eval platform (Langfuse dataset / Evergreen Evals / promptfoo).
    Returns real record counts from this comparison. In production: POST to the
    configured endpoint with the canonical JSONL records as the request body.
    """
    _read_meta(comparison_id)
    comparison = compare_module.load_comparison(comparison_id, OUTPUT_DIR)
    state = validate_module.load_state(comparison_id, OUTPUT_DIR)

    records = [
        p for p in comparison["params"]
        if (lambda d: d and d["decision"] in ("confirmed", "edited"))(
            validate_module.get_decision(state, p["index"])
        )
    ]
    return JSONResponse({
        "status": "ok",
        "mode": "demo",
        "records_pushed": len(records),
        "endpoint": endpoint_url or "(not set)",
        "note": "Demo mode — configure a live endpoint to push to your eval platform.",
    })


@app.post("/compare/{comparison_id}/push/vectorstore")
async def push_vectorstore(
    comparison_id: str,
    endpoint_url: str = Form(""),
    api_key: str = Form(""),
):
    """
    Placeholder push to a vector store (Pinecone, Qdrant, pgvector, …).
    Returns real record counts from this comparison. In production: embed
    policy_text and upsert each record with its full metadata payload.
    """
    _read_meta(comparison_id)
    comparison = compare_module.load_comparison(comparison_id, OUTPUT_DIR)
    state = validate_module.load_state(comparison_id, OUTPUT_DIR)

    chunks = [
        p for p in comparison["params"]
        if p["status"] == "matched" or (
            lambda d: d and d["decision"] in ("confirmed", "edited")
        )(validate_module.get_decision(state, p["index"]))
    ]
    return JSONResponse({
        "status": "ok",
        "mode": "demo",
        "chunks_indexed": len(chunks),
        "endpoint": endpoint_url or "(not set)",
        "note": "Demo mode — configure a live endpoint to upsert to your vector store.",
    })


@app.get("/compare/{comparison_id}/report.jsonl")
async def report_jsonl(comparison_id: str):
    """
    Canonical per-parameter export for downstream consumers:
    vector stores, Langfuse datasets, Evergreen Evals, fine-tuning pipelines.

    One JSON object per line. Schema is stable across platform integrations:
      - Vector store: embed policy_text; all other fields as metadata/payload
      - Langfuse dataset: policy_text → input, attributes.value → expected_output, rest → metadata
      - Evergreen Evals: extraction_class + attributes → question, policy value → expected answer
      - Audit / CSV: flatten as needed
    """
    meta = _read_meta(comparison_id)
    comparison = compare_module.load_comparison(comparison_id, OUTPUT_DIR)
    state = validate_module.load_state(comparison_id, OUTPUT_DIR)

    lines = []
    for p in comparison["params"]:
        dec = validate_module.get_decision(state, p["index"])
        policy = p.get("policy") or {}
        impl = p.get("implementation") or {}

        record = {
            "id": f"{comparison_id}:{p['index']}",
            "domain": meta.get("domain", ""),
            "extraction_class": p["extraction_class"],
            "attributes": policy.get("attributes") or impl.get("attributes") or {},
            # Policy side — the authority
            "policy_text": policy.get("extraction_text", ""),
            "policy_chars": (
                [policy["char_start"], policy["char_end"]]
                if policy.get("char_start") is not None else None
            ),
            "policy_document": meta.get("policy_filename", ""),
            # Implementation side — what is in use
            "impl_text": impl.get("extraction_text", "") if impl else None,
            "impl_chars": (
                [impl["char_start"], impl["char_end"]]
                if impl and impl.get("char_start") is not None else None
            ),
            "impl_document": meta.get("implementation_filename", "") if impl else None,
            # Comparison
            "status": p["status"],
            "drifted_keys": p.get("drifted_attributes") or [],
            # Expert validation
            "decision": dec["decision"] if dec else None,
            "corrected_value": dec.get("edited_value") if dec else None,
            "note": dec.get("note") if dec else None,
            "decided_at": dec.get("decided_at") if dec else None,
            # Lineage
            "comparison_id": comparison_id,
            "model": meta.get("model", ""),
            "created_at": meta.get("created_at", ""),
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    from fastapi.responses import Response
    filename = f"evergreen-{comparison_id}.jsonl"
    return Response(
        content="\n".join(lines) + "\n",
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/compare/{comparison_id}/report.json")
async def report_json(comparison_id: str):
    comparison = compare_module.load_comparison(comparison_id, OUTPUT_DIR)
    state = validate_module.load_state(comparison_id, OUTPUT_DIR)

    merged = {
        **comparison,
        "validation_state": state,
        "params": [
            {**p, "decision": validate_module.get_decision(state, p["index"])}
            for p in comparison["params"]
        ],
    }
    return JSONResponse(content=merged)


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
