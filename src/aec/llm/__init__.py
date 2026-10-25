"""Provider-independent generation boundary."""

from typing import Protocol

from aec.config import LLMConfig
from aec.schemas import NonEmpty, Schema, TokenUsage


class LLMResponse(Schema):
    text: str
    model_id: NonEmpty
    token_usage: TokenUsage
    finish_reason: str
    response_id: str | None = None
    system_fingerprint: str | None = None


class LLMClient(Protocol):
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse: ...
