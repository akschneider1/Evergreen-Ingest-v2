#!/usr/bin/env python3
"""
tools/smoke_test.py — Inference layer diagnostic for Evergreen Ingest.

Calls lx.extract directly on the tax fixture using two modes:
  - vanilla: no custom model wrapper, let langextract's factory handle everything
  - wrapped: use our pre-built provider (OpenAI or Anthropic)

Prints the assembled prompt (when debug=True surfaces it) and raw model response
so you can see exactly what's being sent and returned.

Usage:
    cd Evergreen-Ingest
    python tools/smoke_test.py --provider gemini
    python tools/smoke_test.py --provider openai
    python tools/smoke_test.py --provider anthropic
    python tools/smoke_test.py --provider openai --mode wrapped
    python tools/smoke_test.py --provider anthropic --mode wrapped

Requirements:
    GEMINI_API_KEY or LANGEXTRACT_API_KEY for Gemini
    OPENAI_API_KEY for OpenAI
    ANTHROPIC_API_KEY for Anthropic
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from pathlib import Path

# Ensure we can import from the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import langextract as lx
from langextract import prompt_validation as pv

from examples import get_domain

# ---------------------------------------------------------------------------
# Logging — show DEBUG so we see raw responses from our providers
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy library loggers but keep ours
for noisy in ("httpx", "httpcore", "urllib3", "anthropic", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("smoke_test")

# ---------------------------------------------------------------------------
# Test fixture — short excerpt so we get a fast, cheap call
# ---------------------------------------------------------------------------
TEST_TEXT = textwrap.dedent("""\
    Colorado imposes a flat income tax rate of 4.4% on Colorado taxable income
    for tax year 2026. This rate applies to all filing statuses.

    Colorado residents with gross income exceeding $14,600 for single filers
    must file a state return by April 15.

    The penalty for failure to file is 5% of the unpaid tax per month,
    up to a maximum of 25%.

    Federal withholding adjustments: Employers must apply the 22% federal
    supplemental withholding rate to supplemental wages.
""")


def _get_api_key(provider: str) -> str:
    """Return the API key for the given provider, raising if missing."""
    mapping = {
        "gemini": ["GEMINI_API_KEY", "LANGEXTRACT_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
    }
    for var in mapping[provider]:
        val = os.environ.get(var)
        if val:
            return val
    raise RuntimeError(
        f"No API key found for provider '{provider}'. "
        f"Set one of: {mapping[provider]}"
    )


def run_vanilla(provider: str, model_id: str, api_key: str, domain_name: str) -> None:
    """
    Vanilla mode: pass only model_id + api_key to lx.extract.
    The factory creates the provider internally — no custom wrappers.
    This is the simplest possible call and isolates whether our wrappers are the bug.
    """
    print(f"\n{'='*60}")
    print(f"VANILLA MODE — provider={provider}  model={model_id}")
    print(f"{'='*60}\n")

    domain = get_domain(domain_name)

    logger.info("Calling lx.extract (vanilla, debug=True) ...")
    try:
        result = lx.extract(
            text_or_documents=TEST_TEXT,
            prompt_description=domain.prompt,
            examples=domain.examples,
            model_id=model_id,
            api_key=api_key,
            extraction_passes=1,
            max_workers=1,
            batch_length=10,
            max_char_buffer=len(TEST_TEXT) + 1,
            show_progress=False,
            resolver_params={},
            language_model_params={"max_output_tokens": 2048},
            prompt_validation_level=pv.PromptValidationLevel.OFF,
            debug=True,
        )
    except Exception as exc:
        print(f"\n[ERROR] lx.extract raised {type(exc).__name__}: {exc}\n")
        return

    _report(result)


def run_wrapped_openai(model_id: str, api_key: str, domain_name: str) -> None:
    """
    Wrapped mode for OpenAI: use langextract's built-in OpenAILanguageModel directly
    (no _TimedOpenAILanguageModel subclass — that's the current code smell we're diagnosing).
    """
    print(f"\n{'='*60}")
    print(f"WRAPPED MODE — provider=openai  model={model_id}")
    print(f"{'='*60}\n")

    from langextract.providers.openai import OpenAILanguageModel

    domain = get_domain(domain_name)
    lx_model = OpenAILanguageModel(
        model_id=model_id,
        api_key=api_key,
        max_workers=1,
        max_output_tokens=2048,
    )

    logger.info("Calling lx.extract (wrapped OpenAI, debug=True) ...")
    try:
        result = lx.extract(
            text_or_documents=TEST_TEXT,
            prompt_description=domain.prompt,
            examples=domain.examples,
            model=lx_model,
            extraction_passes=1,
            max_workers=1,
            batch_length=10,
            max_char_buffer=len(TEST_TEXT) + 1,
            show_progress=False,
            use_schema_constraints=False,
            resolver_params={},
            language_model_params={"max_output_tokens": 2048},
            prompt_validation_level=pv.PromptValidationLevel.OFF,
            debug=True,
        )
    except Exception as exc:
        print(f"\n[ERROR] lx.extract raised {type(exc).__name__}: {exc}\n")
        return

    _report(result)


def run_wrapped_anthropic(model_id: str, api_key: str, domain_name: str) -> None:
    """
    Wrapped mode for Anthropic: use our AnthropicLanguageModel provider.
    """
    print(f"\n{'='*60}")
    print(f"WRAPPED MODE — provider=anthropic  model={model_id}")
    print(f"{'='*60}\n")

    from anthropic_provider import AnthropicLanguageModel

    domain = get_domain(domain_name)
    lx_model = AnthropicLanguageModel(
        model_id=model_id,
        api_key=api_key,
        max_workers=1,
        max_output_tokens=2048,
    )

    logger.info("Calling lx.extract (wrapped Anthropic, debug=True) ...")
    try:
        result = lx.extract(
            text_or_documents=TEST_TEXT,
            prompt_description=domain.prompt,
            examples=domain.examples,
            model=lx_model,
            extraction_passes=1,
            max_workers=1,
            batch_length=10,
            max_char_buffer=len(TEST_TEXT) + 1,
            show_progress=False,
            use_schema_constraints=False,
            resolver_params={},
            language_model_params={"max_output_tokens": 2048},
            prompt_validation_level=pv.PromptValidationLevel.OFF,
            debug=True,
        )
    except Exception as exc:
        print(f"\n[ERROR] lx.extract raised {type(exc).__name__}: {exc}\n")
        return

    _report(result)


def _report(result: lx.data.AnnotatedDocument) -> None:
    n = len(result.extractions or [])
    if n == 0:
        print("\n[RESULT] *** 0 extractions returned ***")
        print("         Check the DEBUG log lines above for the raw model response.")
    else:
        print(f"\n[RESULT] {n} extraction(s) returned:")
        for i, ext in enumerate(result.extractions, 1):
            print(f"  {i}. [{ext.extraction_class}] {ext.extraction_text!r}")
            for k, v in (ext.attributes or {}).items():
                print(f"       {k}: {v}")


# ---------------------------------------------------------------------------
# Default model IDs per provider
# ---------------------------------------------------------------------------
DEFAULTS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test langextract inference layer."
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "anthropic"],
        default="gemini",
        help="Which provider to test (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID override (default: provider-specific default)",
    )
    parser.add_argument(
        "--mode",
        choices=["vanilla", "wrapped", "both"],
        default="both",
        help=(
            "vanilla = let factory handle everything; "
            "wrapped = use our pre-built provider; "
            "both = run both and compare (default: both)"
        ),
    )
    parser.add_argument(
        "--domain",
        default="tax",
        help="Domain preset to use for extraction schema (default: tax)",
    )
    args = parser.parse_args()

    provider = args.provider
    model_id = args.model or DEFAULTS[provider]

    try:
        api_key = _get_api_key(provider)
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)

    print(f"\nSmoke test: provider={provider}  model={model_id}  mode={args.mode}")
    print(f"Text length: {len(TEST_TEXT)} chars\n")

    if args.mode in ("vanilla", "both"):
        if provider == "anthropic":
            print(
                "[SKIP] Vanilla mode not available for Anthropic — "
                "langextract has no built-in Anthropic factory. "
                "Use --mode wrapped instead."
            )
        else:
            run_vanilla(provider, model_id, api_key, args.domain)

    if args.mode in ("wrapped", "both"):
        if provider == "gemini":
            print(
                "[SKIP] Wrapped mode not available for Gemini — "
                "no custom Gemini wrapper exists. "
                "Use --mode vanilla instead."
            )
        elif provider == "openai":
            run_wrapped_openai(model_id, api_key, args.domain)
        elif provider == "anthropic":
            run_wrapped_anthropic(model_id, api_key, args.domain)

    print("\nDone.")


if __name__ == "__main__":
    main()
