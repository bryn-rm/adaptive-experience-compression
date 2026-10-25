"""OpenAI Chat Completions adapter. Only public output and usage cross this boundary."""

import os
from typing import Any

from aec.config import LLMConfig
from aec.llm import LLMResponse
from aec.schemas import TokenUsage


class OpenAIClient:
    def __init__(self, *, client: Any = None):
        if client is None:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI provider")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the optional 'openai' dependency") from exc
            # No automatic retries, alternate endpoints, or hidden conversation state.
            client = OpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0)
        self._client = client

    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=config.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            top_p=config.top_p,
            max_completion_tokens=config.max_completion_tokens,
            seed=config.seed,
            timeout=config.timeout_seconds,
            n=1, store=False,
        )
        if response.usage is None or len(response.choices) != 1:
            raise ValueError("Provider must return usage and exactly one completion")
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model_id=response.model,
            token_usage=TokenUsage(prompt_tokens=response.usage.prompt_tokens,
                                   completion_tokens=response.usage.completion_tokens),
            finish_reason=choice.finish_reason,
            response_id=response.id,
            system_fingerprint=response.system_fingerprint,
        )

    def close(self) -> None:
        self._client.close()
