"""Validated manifests and append-only experiment artifacts."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Annotated, Literal, TypeVar

from pydantic import ConfigDict, Field, field_validator, model_validator
import yaml

from aec.config import ALFWorldConfig, AgentConfig, MemoryConfig
from aec.schemas import (
    EpisodicMemory, NonEmpty, ProceduralSkill, RetrievalRecord, Schema, Sources, TokenUsage,
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskManifest(Schema):
    model_config = ConfigDict(extra="forbid", frozen=True)
    manifest_id: NonEmpty
    environment: Literal["alfworld"] = "alfworld"
    split: Literal["train", "eval_in_distribution", "eval_out_of_distribution"]
    task_ids: Annotated[tuple[NonEmpty, ...], Field(min_length=1)]
    task_sha256: tuple[str, ...]
    version: Literal["manifest_v1"] = "manifest_v1"
    created_at: datetime
    selection: Literal["seeded_sample_sorted_v1"] = "seeded_sample_sorted_v1"
    selection_seed: int
    candidate_count: int = Field(ge=1)
    candidate_sha256: NonEmpty

    @model_validator(mode="after")
    def valid_manifest(self):
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("Manifest task IDs must be unique")
        if len(self.task_sha256) != len(self.task_ids) or any(
            len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in self.task_sha256
        ):
            raise ValueError("Each task needs a SHA-256 content hash")
        if self.candidate_count < len(self.task_ids):
            raise ValueError("Selected more tasks than candidates")
        if self.created_at.tzinfo is None:
            raise ValueError("Manifest timestamp must have a timezone")
        return self


class BaselineConfig(Schema):
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    manifest: Path
    output_dir: Path
    environment: ALFWorldConfig
    agent: AgentConfig
    selection_count: int = Field(default=20, ge=1, strict=True)
    selection_seed: int = 0
    data_kind: Literal["real_alfworld", "fake_verification"] = "real_alfworld"
    # None is the no-memory policy; both policies share this runner and base agent.
    memory: MemoryConfig | None = None

    @model_validator(mode="after")
    def memory_needs_v2(self):
        if self.memory is not None and self.agent.prompt_version != "agent_v2":
            raise ValueError("Memory policies require agent_v2")
        return self


def load_baseline_config(path: str | Path) -> BaselineConfig:
    return BaselineConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def create_manifest(config: BaselineConfig, candidates: tuple[str, ...]) -> TaskManifest:
    ordered = sorted(candidates)
    if len(set(ordered)) != len(ordered) or config.selection_count > len(ordered):
        raise ValueError("Need unique candidates and enough tasks for the fixed selection")
    selected = tuple(random.Random(config.selection_seed).sample(ordered, config.selection_count))
    hashes = tuple(file_hash(task) for task in selected)
    identity = dict(environment="alfworld", split=config.environment.split,
                    task_ids=selected, task_sha256=hashes)
    manifest = TaskManifest(
        manifest_id="alfworld-" + digest(identity), **identity, created_at=utc_now(),
        selection_seed=config.selection_seed, candidate_count=len(ordered),
        candidate_sha256=digest(ordered),
    )
    write_once(config.manifest, manifest.model_dump(mode="json"))
    return manifest


def write_once(path: Path, value: object) -> None:
    """Publish a complete file atomically, never replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


T = TypeVar("T", bound=Schema)


def read_rows(path: Path, schema: type[T]) -> list[T]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL; preserve and inspect {path}")
    return [schema.model_validate_json(line) for line in data.splitlines()]


def append_row(path: Path, value: Schema) -> None:
    type(value).model_validate_json(value.model_dump_json())
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                raise ValueError("Incomplete JSONL; refusing to append")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(value.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class Attempt(Schema):
    task_id: NonEmpty
    run_id: NonEmpty
    started_at: datetime


class TaskResult(Schema):
    task_id: NonEmpty
    run_id: NonEmpty
    trajectory_id: NonEmpty
    manifest_id: NonEmpty
    manifest_sha256: NonEmpty
    resolved_config_sha256: NonEmpty
    data_kind: Literal["real_alfworld", "fake_verification"]
    outcome: Literal["success", "environment_failure", "aborted", "setup_failure",
                     "provider_failure", "runtime_failure"]
    termination_reason: NonEmpty
    steps: int = Field(ge=0)
    parse_failures: int = Field(ge=0)
    normalized_actions: int = Field(default=0, ge=0)
    reward: float | None
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    usage_known: bool
    wall_time: float = Field(ge=0)
    wall_time_known: bool
    retrieved_memory_ids: list[NonEmpty] = Field(default_factory=list)


class RetrievalLog(Schema):
    """Decision-time retrieval, written before acting; join outcomes via task/run."""

    run_id: NonEmpty
    task_id: NonEmpty
    query_text: NonEmpty
    store_size: int = Field(ge=0)
    eligible_count: int = Field(ge=0)
    records: list[RetrievalRecord]
    # One provenance list per record: one trajectory per episode, all sources per skill.
    source_trajectory_ids: list[Sources]
    memory_section: str | None
    memory_tokens: int = Field(ge=0)
    embedding_model_id: NonEmpty
    embedding_tokens: int = Field(ge=0)
    created_at: datetime

    @field_validator("source_trajectory_ids", mode="before")
    @classmethod
    def phase4_episode_sources(cls, value):
        # Phase 4 logs stored one bare trajectory ID per retrieved episode.
        return [[v] if isinstance(v, str) else v for v in value] if isinstance(value, list) else value

    @model_validator(mode="after")
    def consistent(self):
        if [r.rank for r in self.records] != list(range(1, len(self.records) + 1)):
            raise ValueError("Retrieval ranks must be contiguous from one")
        if any(r.task_id != self.task_id for r in self.records):
            raise ValueError("Retrieval records belong to another task")
        if len(self.source_trajectory_ids) != len(self.records):
            raise ValueError("Each retrieved memory needs its source trajectory")
        if (self.memory_section is None) != (not self.records):
            raise ValueError("A memory section is present exactly when memories are retrieved")
        return self


class MemoryWrite(Schema):
    """Append-only memory-writing log; `stored` rows are the store's source of truth.

    `trajectory_id` is the triggering trajectory. Skill rows also list every source,
    oldest first, ending with the trigger.
    """

    run_id: NonEmpty
    task_id: NonEmpty
    trajectory_id: NonEmpty
    status: Literal["started", "stored", "rejected", "provider_failure",
                    "embedding_failure", "interrupted"]
    writer_prompt_version: NonEmpty
    prompt: str | None = None
    output: str | None = None
    error: str | None = None
    error_type: str | None = None
    response_model_id: str | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    usage_known: bool = False
    embedding_model_id: str | None = None
    embedding_tokens: int = Field(default=0, ge=0)
    memory: EpisodicMemory | None = None
    skill: ProceduralSkill | None = None
    source_trajectory_ids: list[NonEmpty] = Field(default_factory=list)
    source_task_ids: list[NonEmpty] = Field(default_factory=list)
    embedding: list[float] | None = None
    created_at: datetime

    @model_validator(mode="after")
    def consistent(self):
        stored = self.status == "stored"
        payloads = (self.memory is not None) + (self.skill is not None)
        if payloads > 1 or stored != (payloads == 1) or stored != (self.embedding is not None):
            raise ValueError("Exactly stored writes carry one memory and its embedding")
        if self.memory is not None and self.memory.source_trajectory_ids != [self.trajectory_id]:
            raise ValueError("Episode provenance differs from its source trajectory")
        if len(self.source_task_ids) != len(self.source_trajectory_ids):
            raise ValueError("Each skill source trajectory needs its task ID")
        if self.source_trajectory_ids and (self.source_trajectory_ids[-1] != self.trajectory_id
                                           or self.source_task_ids[-1] != self.task_id):
            raise ValueError("The triggering trajectory must be the newest skill source")
        if self.skill is not None and self.skill.source_trajectory_ids != self.source_trajectory_ids:
            raise ValueError("Skill provenance differs from its recorded sources")
        if len(set(self.source_trajectory_ids)) != len(self.source_trajectory_ids):
            raise ValueError("Duplicate skill source trajectory")
        return self
