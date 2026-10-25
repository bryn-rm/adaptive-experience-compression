"""Validated configuration loading; experiment settings are added in later phases."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ALFWorldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_config: Path
    split: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "eval_out_of_distribution"
    max_steps: int = Field(default=50, ge=1, strict=True)
    seed: int = 0


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    provider: Literal["openai"] = "openai"
    model_id: str = Field(min_length=1)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    max_completion_tokens: int = Field(default=128, ge=1, strict=True)
    seed: int = 0
    timeout_seconds: float = Field(default=60, gt=0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_version: Literal["agent_v1"] = "agent_v1"
    llm: LLMConfig


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: ALFWorldConfig | None = None
    agent: AgentConfig | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return Config.model_validate(data)
