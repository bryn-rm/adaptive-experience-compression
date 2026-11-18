"""Thin embedding boundary. Only vectors and provider-reported usage cross it."""

import math
import os
from typing import Any, Protocol

from pydantic import Field

from aec.config import EmbeddingConfig
from aec.schemas import Count, NonEmpty, Schema


class EmbeddingResponse(Schema):
    vectors: list[list[float]] = Field(min_length=1)
    model_id: NonEmpty
    prompt_tokens: Count


class Embedder(Protocol):
    def embed(self, texts: list[str], config: EmbeddingConfig) -> EmbeddingResponse: ...

    def close(self) -> None: ...


class OpenAIEmbedder:
    def __init__(self, *, client: Any = None):
        if client is None:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI provider")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the optional 'openai' dependency") from exc
            client = OpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0)
        self._client = client

    def embed(self, texts: list[str], config: EmbeddingConfig) -> EmbeddingResponse:
        if not texts or any(not text for text in texts):
            raise ValueError("Embedding inputs must be non-empty")
        optional = {"dimensions": config.dimensions} if config.dimensions is not None else {}
        response = self._client.embeddings.create(
            model=config.model_id, input=texts, encoding_format="float",
            timeout=config.timeout_seconds, **optional,
        )
        data = sorted(response.data, key=lambda item: item.index)
        if response.usage is None or [item.index for item in data] != list(range(len(texts))):
            raise ValueError("Provider must return usage and one embedding per input")
        return EmbeddingResponse(vectors=[list(item.embedding) for item in data],
                                 model_id=response.model,
                                 prompt_tokens=response.usage.prompt_tokens)

    def close(self) -> None:
        self._client.close()


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Embedding dimensions differ")
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        raise ValueError("Zero embedding vector")
    # Clamp floating-point overshoot so similarities stay within the schema bounds.
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / norm))
