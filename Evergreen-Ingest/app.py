"""
app.py — FastAPI application for Evergreen Ingest.

All routes. Extraction runs in a ThreadPoolExecutor (langextract is sync)
and the browser polls /compare/{id}/status until ready.
"""

from __future__ import annotations

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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import compare as compare_module
import extract as extract_module
import validate as validate_module

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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
for sub in ("extractions", "visualizations", "comparisons", "validations"):
    (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Evergreen Ingest")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
    "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1",
    "gemini-2.5-flash", "gemini-2.0-flash",
}


@app.post("/upload")
async def upload(
    policy_file: UploadFile = Form(...),
    implementation_file: UploadFile = Form(...),
    domain: str = Form(...),
    model: str = Form("gpt-4o-mini"),
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
        model = "gpt-4o-mini"

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
        "error": None,
    }
    _write_meta(comparison_id, meta)

    # Kick off extraction in background thread
    executor.submit(_run_pipeline, comparison_id, policy_path, impl_path, domain)

    return RedirectResponse(
        url=f"/compare/{comparison_id}/status", status_code=303
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
        raise HTTPException(status_code=404, detail="Visualization not found")
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
    if index in actionable:
        pos = actionable.index(index)
        prev_idx = actionable[pos - 1] if pos > 0 else None
        next_idx = actionable[pos + 1] if pos < len(actionable) - 1 else None

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
