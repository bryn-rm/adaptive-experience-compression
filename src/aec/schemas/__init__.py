"""JSON-compatible boundary types. Only observable interaction data is stored."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

NonEmpty = Annotated[str, Field(min_length=1)]
Count = Annotated[int, Field(ge=0)]
Sources = Annotated[list[NonEmpty], Field(min_length=1)]
Action = Literal["discard", "episode", "skill"]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TokenUsage(Schema):
    prompt_tokens: Count = 0
    completion_tokens: Count = 0


class Task(Schema):
    task_id: NonEmpty
    description: NonEmpty
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TrajectoryStep(Schema):
    observation: str
    action: str
    output: str | None = None
    reward: float
    done: bool
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class Trajectory(Schema):
    trajectory_id: NonEmpty
    task_id: NonEmpty
    steps: list[TrajectoryStep]
    success: bool
    reward: float
    termination_reason: NonEmpty
    model_id: NonEmpty
    prompt_version: NonEmpty
    seed: int
    token_usage: TokenUsage


class EpisodicMemory(Schema):
    memory_id: NonEmpty
    source_trajectory_ids: Sources
    task_summary: NonEmpty
    experience: NonEmpty
    outcome: NonEmpty
    embedding_text: NonEmpty
    token_count: Count


class ProceduralSkill(Schema):
    skill_id: NonEmpty
    source_trajectory_ids: Sources
    name: NonEmpty
    activation_conditions: list[NonEmpty]
    procedure: Annotated[list[NonEmpty], Field(min_length=1)]
    constraints: list[NonEmpty]
    failure_modes: list[NonEmpty]
    embedding_text: NonEmpty
    version: Annotated[int, Field(ge=1)]
    token_count: Count


class RetrievalRecord(Schema):
    task_id: NonEmpty
    memory_id: NonEmpty
    similarity: Annotated[float, Field(ge=-1, le=1)]
    rank: Annotated[int, Field(ge=1)]


class ExperienceDecision(Schema):
    trajectory_id: NonEmpty
    action: Action
    policy_name: NonEmpty
    feature_version: NonEmpty
    features: dict[str, JsonValue]
    confidence: Annotated[float, Field(ge=0, le=1)] | None = None


class CounterfactualOutcome(Schema):
    trajectory_id: NonEmpty
    snapshot_id: NonEmpty
    task_id: NonEmpty
    action: Action
    success: bool
    reward: float
    token_usage: TokenUsage
    memory_tokens: Count
    management_tokens: Count
