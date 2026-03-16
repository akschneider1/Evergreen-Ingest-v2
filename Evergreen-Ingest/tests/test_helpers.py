"""
Tests for pure helper functions in app.py (_friendly_error) and examples/ (domain registry).

No API key required.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# _friendly_error
# ---------------------------------------------------------------------------

from app import _friendly_error


@pytest.mark.parametrize("msg,model,expected_fragment", [
    # Claude auth errors
    ("authentication failed", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    ("unauthenticated request", "claude-opus-4-6", "ANTHROPIC_API_KEY"),
    ("incorrect api key provided", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    # OpenAI auth errors
    ("Invalid API key supplied", "gpt-4o-mini", "OPENAI_API_KEY"),
    ("api_key missing", "gpt-4o", "OPENAI_API_KEY"),
    # 529 overloaded (Claude)
    ("Anthropic API 529 error", "claude-sonnet-4-6", "overloaded"),
    # quota / resource exhausted (non-Claude)
    ("resource_exhausted quota exceeded", "gemini-2.0-flash", "quota"),
    # rate limit
    ("rate limit hit 429", "gpt-4o-mini", "rate limit"),
    # timeout
    ("timeout: connection deadline exceeded", "claude-sonnet-4-6", "timed out"),
    # no parameters passthrough
    ("No parameters found in 'policy.md'.", "claude-sonnet-4-6", "No parameters found"),
])
def test_friendly_error_parametrized(msg, model, expected_fragment):
    result = _friendly_error(Exception(msg), model_id=model)
    assert expected_fragment.lower() in result.lower(), (
        f"Expected {expected_fragment!r} in result for msg={msg!r}, model={model!r}. Got: {result!r}"
    )


def test_friendly_error_unknown_passthrough():
    """Unrecognised errors are returned as-is."""
    msg = "some totally random unexpected error xyz"
    result = _friendly_error(Exception(msg))
    assert result == msg


def test_friendly_error_no_model_defaults_to_openai_key():
    """Auth error with no model (default '') → generic key hint."""
    result = _friendly_error(Exception("authentication error"), model_id="")
    assert "API key" in result


def test_friendly_error_model_not_found():
    result = _friendly_error(Exception("model not found"), model_id="gpt-4o")
    assert "model" in result.lower()


# ---------------------------------------------------------------------------
# examples/ domain registry
# ---------------------------------------------------------------------------

from examples import get_domain, DomainConfig


def test_get_domain_tax():
    d = get_domain("tax")
    assert isinstance(d, DomainConfig)
    assert d.name == "tax"


def test_get_domain_benefits():
    d = get_domain("benefits")
    assert isinstance(d, DomainConfig)
    assert d.name == "benefits"


def test_get_domain_unknown_raises():
    with pytest.raises(ValueError, match="Unknown domain"):
        get_domain("nonexistent_xyz")


def test_domain_tax_examples_nonempty():
    d = get_domain("tax")
    assert len(d.examples) >= 1


def test_domain_tax_prompt_set():
    d = get_domain("tax")
    assert d.prompt and len(d.prompt) > 10


def test_domain_benefits_prompt_set():
    d = get_domain("benefits")
    assert d.prompt and len(d.prompt) > 10
