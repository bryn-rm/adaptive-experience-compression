"""Phase 2 artifact extensions; Phase 0 boundary schemas remain unchanged."""

from typing import Literal

from pydantic import Field, model_validator

from aec.config import ALFWorldConfig, AgentConfig
from aec.schemas import Count, NonEmpty, Task, Trajectory, TrajectoryStep


class AgentStep(TrajectoryStep):
    step_index: Count
    prompt: str
    next_observation: str | None
    valid_actions: list[str]
    valid_action: bool
    executed: bool
    parse_error: str | None = None
    error_type: str | None = None
    wall_time: float = Field(ge=0)
    response_model_id: str | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    environment_termination_reason: Literal["environment", "max_steps"] | None = None
    usage_known: bool

    @model_validator(mode="after")
    def consistent_action(self) -> "AgentStep":
        if self.valid_action != (bool(self.action) and self.action in self.valid_actions):
            raise ValueError("Action validity must match the recorded admissible actions")
        if self.executed and (not self.valid_action or self.next_observation is None):
            raise ValueError("Executed actions need valid actions and resulting observations")
        if self.parse_error and (self.executed or self.valid_action):
            raise ValueError("A parse failure cannot execute an action")
        return self


class AgentTrajectory(Trajectory):
    schema_version: Literal["trajectory_v1"] = "trajectory_v1"
    run_id: NonEmpty
    task: Task
    steps: list[AgentStep]
    status: Literal["completed", "aborted"]
    success_known: bool
    wall_time: float = Field(ge=0)
    total_reward: float
    prompt_tokens: Count
    completion_tokens: Count
    agent_config: AgentConfig
    environment_config: ALFWorldConfig
    environment_backend_sha256: NonEmpty

    @model_validator(mode="after")
    def consistent_metadata(self) -> "AgentTrajectory":
        if self.task_id != self.task.task_id:
            raise ValueError("Task identity mismatch")
        if self.model_id != self.agent_config.llm.model_id:
            raise ValueError("Model identity mismatch")
        if self.prompt_version != self.agent_config.prompt_version:
            raise ValueError("Prompt version mismatch")
        if self.seed != self.environment_config.seed:
            raise ValueError("Environment seed mismatch")
        if self.reward != self.total_reward:
            raise ValueError("Reward aliases must agree")
        if self.prompt_tokens != self.token_usage.prompt_tokens or (
            self.completion_tokens != self.token_usage.completion_tokens
        ):
            raise ValueError("Token aliases must agree")
        if self.prompt_tokens != sum(s.token_usage.prompt_tokens for s in self.steps) or (
            self.completion_tokens != sum(s.token_usage.completion_tokens for s in self.steps)
        ):
            raise ValueError("Trajectory usage must equal step usage totals")
        if [s.step_index for s in self.steps] != list(range(len(self.steps))):
            raise ValueError("Step indices must be contiguous and zero-based")
        if any(step.done for step in self.steps[:-1]):
            raise ValueError("No steps may follow a terminal step")
        if self.status == "completed" and (
            not self.steps or not self.steps[-1].executed or not self.steps[-1].done
            or not self.success_known
            or self.termination_reason not in ("environment", "max_steps")
        ):
            raise ValueError("Completed episodes need a known, terminal environment outcome")
        return self
