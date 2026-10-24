import pytest
from pydantic import ValidationError

from aec.schemas import (
    CounterfactualOutcome, EpisodicMemory, ExperienceDecision, ProceduralSkill,
    RetrievalRecord, Task, TokenUsage, Trajectory, TrajectoryStep,
)


CASES = [
    (Task, dict(task_id="task-1", description="Find the mug")),
    (TokenUsage, dict(prompt_tokens=12, completion_tokens=4)),
    (TrajectoryStep, dict(observation="A mug", action="take mug", reward=1, done=True)),
    (Trajectory, dict(trajectory_id="tr-1", task_id="task-1", steps=[
        dict(observation="A mug", action="take mug", reward=1, done=True)
    ], success=True, reward=1, termination_reason="completed", model_id="fixed-model",
        prompt_version="v1", seed=0, token_usage=dict(prompt_tokens=12, completion_tokens=4))),
    (EpisodicMemory, dict(memory_id="ep-1", source_trajectory_ids=["tr-1"],
        task_summary="Find mug", experience="Found mug on table", outcome="success",
        embedding_text="mug on table", token_count=10)),
    (ProceduralSkill, dict(skill_id="skill-1", source_trajectory_ids=["tr-1"],
        name="Find mug", activation_conditions=["Need a mug"], procedure=["Inspect table"],
        constraints=[], failure_modes=[], embedding_text="Find mug", version=1, token_count=8)),
    (RetrievalRecord, dict(task_id="task-1", memory_id="ep-1", similarity=0.8, rank=1)),
    (ExperienceDecision, dict(trajectory_id="tr-1", action="episode", policy_name="fixture",
        feature_version="v1", features={"success": True}, confidence=0.9)),
    (CounterfactualOutcome, dict(trajectory_id="tr-1", snapshot_id="snapshot-1",
        task_id="task-2", action="discard", success=False, reward=0,
        token_usage={}, memory_tokens=0, management_tokens=0)),
]


@pytest.mark.parametrize("schema,data", CASES)
def test_json_round_trip(schema, data):
    record = schema.model_validate(data)
    assert schema.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("schema,data", CASES)
def test_unknown_fields_rejected(schema, data):
    with pytest.raises(ValidationError):
        schema.model_validate({**data, "hidden_chain_of_thought": "not an observable field"})


@pytest.mark.parametrize("schema,data,field,value", [
    (EpisodicMemory, CASES[4][1], "source_trajectory_ids", []),
    (ProceduralSkill, CASES[5][1], "source_trajectory_ids", []),
    (ProceduralSkill, CASES[5][1], "procedure", []),
    (TokenUsage, {}, "prompt_tokens", -1),
    (ExperienceDecision, CASES[7][1], "action", "unknown"),
    (ExperienceDecision, CASES[7][1], "confidence", 1.1),
    (CounterfactualOutcome, CASES[8][1], "reward", float("nan")),
])
def test_malformed_artifacts_rejected(schema, data, field, value):
    with pytest.raises(ValidationError):
        schema.model_validate({**data, field: value})


def test_mutable_defaults_are_isolated():
    first = Task(task_id="a", description="a")
    second = Task(task_id="b", description="b")
    first.metadata["source"] = "a"
    assert second.metadata == {}
