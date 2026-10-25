import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from aec.config import LLMConfig
from aec.llm.openai import OpenAIClient


class Transport:
    def __init__(self):
        self.kwargs = None
        self.chat = NS(completions=self)
        self.closed = False
        self.response = NS(
            choices=[NS(message=NS(content="look", reasoning_content="private"), finish_reason="stop")],
            usage=NS(prompt_tokens=12, completion_tokens=3),
            model="model-snapshot", id="response-1", system_fingerprint="fingerprint-1")

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response

    def close(self):
        self.closed = True


def test_provider_conversion_and_frozen_request():
    transport = Transport()
    client = OpenAIClient(client=transport)
    config = LLMConfig(model_id="model-snapshot", temperature=0, top_p=1,
                       max_completion_tokens=32, seed=7, timeout_seconds=5)
    response = client.generate("prompt", config)
    assert transport.kwargs == dict(
        model="model-snapshot", messages=[{"role": "user", "content": "prompt"}],
        temperature=0, top_p=1, max_completion_tokens=32, seed=7, timeout=5, n=1, store=False)
    assert response.text == "look"
    assert response.token_usage.prompt_tokens == 12
    assert response.token_usage.completion_tokens == 3
    assert response.model_id == "model-snapshot"
    assert response.finish_reason == "stop"
    assert "private" not in response.model_dump_json()
    client.close()
    assert transport.closed


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_environment_credentials_used_without_persistence(monkeypatch):
    captured = {}
    transport = Transport()
    module = ModuleType("openai")
    def create(**kwargs):
        captured.update(kwargs)
        return transport
    module.OpenAI = create
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIClient()
    assert captured == dict(api_key="test-key", base_url="https://api.openai.com/v1", max_retries=0)
    response = client.generate("prompt", LLMConfig(model_id="model-snapshot"))
    assert "test-key" not in response.model_dump_json()


def test_missing_usage_rejected():
    transport = Transport()
    transport.response.usage = None
    with pytest.raises(ValueError, match="usage"):
        OpenAIClient(client=transport).generate("prompt", LLMConfig(model_id="m"))
