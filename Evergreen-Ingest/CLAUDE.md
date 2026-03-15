# CLAUDE.md — Evergreen Ingest

## The product

A policy-to-implementation drift detector. A government policy lead uploads two documents — the source policy and an implementation artifact (training manual, agent script, FAQ, knowledge base). They define what parameters matter. The tool extracts those parameters from both documents, shows the differences side by side, and lets the expert validate each one.

The output is a drift report: what's matched, what's drifted, what's missing, what's extra. Every finding is grounded to exact source text in both documents.

## Who uses it

A DOR tax policy lead. A CDLE unemployment program manager. Anyone responsible for keeping implementation aligned with policy — especially before that implementation becomes the source of truth for an AI system.

## The workflow

1. **Upload two documents.** Source policy (the authority) and implementation artifact (what's actually in use). Accepts PDF, DOCX, HTML, Markdown, or plain text.

2. **Pick or define parameters.** Select a domain (tax, benefits) to load preset parameter types. Or define custom ones in the UI: "Check for income thresholds, filing deadlines, penalty amounts, credit eligibility." These become the extraction schema.

3. **Extract.** Langextract runs on both documents using the same schema. Each extracted parameter is grounded to exact source text with character-level position mapping.

4. **Compare.** Side-by-side view of parameters from both documents, categorized:
   - **Matched** — same parameter, same value in both documents
   - **Drifted** — same parameter, different values (e.g., policy says $14,600, training manual says $13,590)
   - **Missing** — parameter exists in source policy but not in implementation
   - **Extra** — parameter exists in implementation but not in current policy

5. **Validate.** For each drifted, missing, or extra parameter, the expert can:
   - **Confirm** — the policy version is correct, flag implementation for update
   - **Edit** — correct the parameter (original extraction preserved alongside edit)
   - **Reject** — not a real drift, dismiss

6. **Report.** Export a drift report: what's out of sync, what was validated, what needs fixing upstream. The validated parameters are also reusable as ground truth for AI evaluation.

## Tech stack

- **Python 3.11+**
- **langextract** — structured extraction with source grounding (Google, Apache 2.0, `pip install langextract`)
- **FastAPI** + **Jinja2** — web app and UI
- **uvicorn** — server
- No database. JSON/JSONL files.

## Architecture

```
Source policy ─────┐
                   ├──→ langextract (same schema) ──→ compare ──→ validate ──→ report
Implementation ────┘
```

Langextract handles: LLM calls, chunking for extraction, source-text grounding, interactive HTML visualization. We handle: the comparison logic, the validation workflow, and the web UI.

## Directory structure

```
Evergreen-Ingest/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml
│
├── app.py                        # FastAPI app — all routes
├── extract.py                    # langextract integration
├── compare.py                    # Diff logic: matched, drifted, missing, extra
├── validate.py                   # Validation state: confirm, edit, reject
│
├── examples/                     # Few-shot extraction schemas per domain
│   ├── __init__.py
│   ├── tax.py                    # Tax policy parameter types + examples
│   └── benefits.py               # Benefits policy parameter types + examples
│
├── templates/
│   ├── index.html                # Upload two docs + pick domain
│   ├── status.html               # Polling page while extraction runs
│   ├── compare.html              # Side-by-side drift view
│   ├── detail.html               # Single parameter: source text, values, validate
│   └── report.html               # Drift report summary
│
├── static/
│   └── style.css
│
├── output/                       # All output (gitignored)
│   ├── extractions/              # langextract JSONL per document (immutable)
│   ├── visualizations/           # langextract HTML per document
│   ├── comparisons/              # Diff results JSON
│   └── validations/              # Validation state JSON
│
├── sources/                      # Uploaded documents (gitignored)
│
└── tests/
    ├── test_extract.py
    ├── test_compare.py
    ├── test_validate.py
    └── fixtures/
        ├── policy_tax_current.md           # Source policy (current)
        ├── training_manual_tax_stale.md    # Implementation (with drift)
        ├── policy_benefits_current.html
        └── training_manual_benefits.html
```

## Intentional drift in fixtures

### Tax pair
| Parameter | Policy says | Training manual says | Status |
|-----------|-------------|----------------------|--------|
| Filing threshold (single) | $14,600 | $13,590 | **Drifted** |
| State income tax rate | 4.4% | 4.25% | **Drifted** |
| Federal supplemental withholding | 22% | 20% | **Drifted** |
| Child Tax Credit | 25% of federal | 20% of federal | **Drifted** |
| PTC Rebate maximum | $1,112 | $1,000 | **Drifted** |
| Expedited service criteria | (in policy) | (absent) | **Missing** |

### Benefits pair
| Parameter | Policy says | Training manual says | Status |
|-----------|-------------|----------------------|--------|
| Maximum WBA | $781/week | $700/week | **Drifted** |
| Weekly work search contacts | 5 contacts (3 direct) | 3 contacts (2 direct) | **Drifted** |
| Initial claim filing window | 3 weeks | 2 weeks | **Drifted** |
| Minimum age requirement | None | 18 years old | **Drifted** |
| Expedited service — income threshold | (in policy) | (absent) | **Missing** |
| Expedited claim resolution window | 5 business days | (absent) | **Missing** |

## Key modules

### extract.py

Wraps langextract. Takes a document path and a domain, returns extractions.

```python
def extract_document(source_path, domain_name, output_dir, comparison_id, doc_slot) -> tuple[Path, Path, AnnotatedDocument]:
    """Run langextract on a document using domain-specific few-shot examples.
    Returns (jsonl_path, viz_path, annotated_doc)."""
```

### compare.py

Takes two sets of extractions, matches parameters by extraction_class + Jaccard attribute similarity (greedy best-match, threshold 0.3), categorizes each pair.

```python
def compare_extractions(policy_doc, impl_doc, comparison_id, domain, output_dir) -> dict:
    """Compare parameters. Returns {matched, drifted, missing, extra} dict and writes comparison.json."""
```

### validate.py

Manages validation state per comparison. Atomic writes via os.replace.

```python
def save_decision(comparison_id, output_dir, param_index, decision, edited_value, note) -> state
def load_state(comparison_id, output_dir) -> dict
def get_decision(state, param_index) -> dict | None
```

### app.py

FastAPI routes. Extraction runs in a ThreadPoolExecutor (langextract is sync). Browser polls `/compare/{id}/status` (meta-refresh every 4s) until `meta.json.status == "ready"`.

```
GET  /                                Upload form
POST /upload                          Save files, start extraction pipeline
GET  /compare/{id}/status             Polling page
GET  /compare/{id}                    Side-by-side drift view
GET  /compare/{id}/viz/{doc}          Langextract visualization iframe
GET  /compare/{id}/param/{index}      Single parameter detail + validate form
POST /compare/{id}/param/{index}      Submit validation decision
GET  /compare/{id}/report             Drift report summary
GET  /compare/{id}/report.json        Downloadable JSON report
```

## Data engineering principles

**Immutability.** Source documents in `sources/` and raw extractions in `output/extractions/` are never modified. Validation writes to `output/validations/`. If you re-run extraction with better examples or a better model, you reprocess from the original source.

**Lineage.** Every validated parameter carries: which document it came from, the exact source text, the extraction timestamp, the model used, the validation decision, and the reviewer timestamp.

**Idempotency.** Same document + same schema = same extractions. Safe to re-run.

## Running

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

pip install -r requirements.txt
python app.py
# → http://localhost:8000
```

## Testing

```bash
# Unit tests (no API key required)
pytest tests/test_compare.py tests/test_validate.py -v

# Full integration tests (requires GOOGLE_API_KEY)
pytest tests/ -v
```

## Design principles

- This is a proof of concept. Keep it simple.
- Use langextract's schema as-is. No wrapper models.
- Source grounding is non-negotiable. Every parameter links to exact text.
- Edits preserve history. The original extraction is always visible alongside the expert's version.
- Flat files, no database, no build step, no frontend framework.
- A government policy expert should be able to use this without help.
- The demo is explainable in one sentence: "Upload your policy and your training manual, see what's different."
