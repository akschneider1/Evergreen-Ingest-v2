"""
anthropic_provider.py — Anthropic Claude provider for langextract.

Implements BaseLanguageModel so Claude models can be used for extraction
in the same way as the built-in OpenAI and Gemini providers.

Key differences from OpenAI:
- `system` is a top-level parameter, not a message role
- `max_tokens` is required (not optional)
- No `response_format: json_object` → requires_fence_output=True so
  langextract's resolver parses JSON from ```json ... ``` fences
- Timeout and max_retries are set on the Anthropic client directly
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Iterator, Sequence

import anthropic as _anthropic

from langextract import exceptions
from langextract.core import types as core_types
from langextract.core.base_model import BaseLanguageModel

logger = logging.getLogger(__name__)


class AnthropicLanguageModel(BaseLanguageModel):
    """langextract provider for Anthropic Claude models.

    Usage in extract.py:
        lx_model = AnthropicLanguageModel(
            model_id="claude-sonnet-4-6",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_workers=10,
            client_timeout=200.0,
            max_output_tokens=2048,
        )
        result = lx.extract(..., model=lx_model, ...)
    """

    def __init__(
        self,
        *,
        model_id: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_workers: int = 10,
        client_timeout: float = 200.0,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        if not api_key:
            raise exceptions.InferenceConfigError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY in your environment or .env file."
            )
        self.model_id = model_id
        self.max_workers = max_workers
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client = _anthropic.Anthropic(
            api_key=api_key,
            timeout=client_timeout,
            max_retries=0,  # surface errors immediately
        )
        super().__init__(**kwargs)

    @property
    def requires_fence_output(self) -> bool:
        # Anthropic has no JSON-mode equivalent. We instruct the model via
        # system prompt and rely on langextract's fence parser to extract
        # the JSON block from the response.
        return True

    def _process_single_prompt(
        self, prompt: str, config: dict
    ) -> core_types.ScoredOutput:
        """Call the Anthropic messages API for a single prompt."""
        try:
            response = self._client.messages.create(
                model=self.model_id,
                system=(
                    "You are a helpful assistant that responds in JSON format. "
                    "Always wrap your JSON response in a ```json code fence."
                ),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.get("max_output_tokens", self.max_output_tokens),
                temperature=config.get("temperature", self.temperature),
            )
            output_text = response.content[0].text
            return core_types.ScoredOutput(score=1.0, output=output_text)
        except _anthropic.APIStatusError as exc:
            raise exceptions.InferenceRuntimeError(
                f"Anthropic API error {exc.status_code}: {exc.message}",
                original=exc,
            ) from exc
        except Exception as exc:
            raise exceptions.InferenceRuntimeError(
                f"Anthropic error ({type(exc).__name__}): {exc}",
                original=exc,
            ) from exc

    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[core_types.ScoredOutput]]:
        """Run inference on a batch of prompts, in parallel when batch > 1."""
        merged = self.merge_kwargs(kwargs)
        config = {
            k: merged[k]
            for k in ("max_output_tokens", "temperature")
            if k in merged
        }

        if len(batch_prompts) > 1 and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(batch_prompts))
            ) as pool:
                future_map = {
                    pool.submit(self._process_single_prompt, p, config.copy()): i
                    for i, p in enumerate(batch_prompts)
                }
                results: list[core_types.ScoredOutput | None] = [None] * len(batch_prompts)
                for future in concurrent.futures.as_completed(future_map):
                    idx = future_map[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        raise exceptions.InferenceRuntimeError(
                            f"Parallel inference error: {exc}", original=exc
                        ) from exc
            for result in results:
                yield [result]
        else:
            for prompt in batch_prompts:
                yield [self._process_single_prompt(prompt, config.copy())]
