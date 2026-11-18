"""Validated configuration loading; experiment settings are added in later phases."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    temperature: float | None = Field(default=0, ge=0, le=2)
    top_p: float | None = Field(default=1, gt=0, le=1)
    max_completion_tokens: int = Field(default=128, ge=1, strict=True)
    seed: int | None = 0
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    timeout_seconds: float = Field(default=60, gt=0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_version: Literal["agent_v1", "agent_v2"] = "agent_v1"
    history_length: int = Field(default=0, ge=0, strict=True)
    llm: LLMConfig

    @model_validator(mode="after")
    def versioned_history(self):
        if self.prompt_version == "agent_v1" and self.history_length != 0:
            raise ValueError("agent_v1 has no interaction history")
        return self


class EmbeddingConfig(BaseModel):
    """The single fixed embedding model; retrieval comparisons must keep it identical."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai"] = "openai"
    model_id: str = Field(min_length=1)
    dimensions: int | None = Field(default=None, ge=1, strict=True)
    timeout_seconds: float = Field(default=60, gt=0)


class MemoryWriterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_version: Literal["episode_writer_v1", "skill_writer_v1"] = "episode_writer_v1"
    llm: LLMConfig


EpisodeWriterConfig = MemoryWriterConfig


class MemoryConfig(BaseModel):
    """Long-term memory policy. Absent memory config is the no-memory policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Literal["episodic", "skill"] = "episodic"
    store_path: Path
    top_k: int = Field(ge=1, strict=True)
    # Never retrieve memories derived from the task currently being attempted.
    exclude_same_task: bool = True
    # task_goal embeds only the "Your task is to:" line; task_description the full reset text.
    retrieval_key: Literal["task_goal", "task_description"] = "task_goal"
    memory_format: Literal["memory_section_v1", "skill_section_v1"] = "memory_section_v1"
    # Fixed local tokenizer for stored/injected memory token accounting.
    tokenizer: Literal["o200k_base"] = "o200k_base"
    embedding: EmbeddingConfig
    writer: MemoryWriterConfig
    # Skill policy only: group successes by ALFWorld task family; (re)synthesize a
    # family's skill from all its successes once at least `skill_min_sources` exist.
    skill_grouping: Literal["task_family"] | None = None
    skill_min_sources: int | None = Field(default=None, ge=2, strict=True)

    @model_validator(mode="after")
    def consistent_policy(self):
        expected = {"episodic": ("memory_section_v1", "episode_writer_v1", False),
                    "skill": ("skill_section_v1", "skill_writer_v1", True)}[self.policy]
        skill_fields = (self.skill_grouping, self.skill_min_sources)
        if (self.memory_format, self.writer.prompt_version) != expected[:2] or (
            (None not in skill_fields) != expected[2] or (
                any(f is not None for f in skill_fields) and not expected[2])
        ):
            raise ValueError("Memory format, writer, and skill settings must match the policy")
        return self


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
