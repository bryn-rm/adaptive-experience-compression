"""episode_writer_v1 and memory_section_v1: change versions whenever content changes."""

import json
from typing import Callable, Sequence

from aec.schemas import EpisodicMemory, Task
from aec.trajectories.schemas import AgentTrajectory

WRITER_VERSION = "episode_writer_v1"
MEMORY_FORMAT = "memory_section_v1"
FIELDS = ("task_summary", "experience", "outcome")


class EpisodeParseError(ValueError):
    pass


GOAL_MARKER = "Your task is to: "


def retrieval_text(task: Task, key: str) -> str:
    """Deterministic, decision-time retrieval key for both queries and stored episodes."""
    if key == "task_description":
        return task.description
    if key != "task_goal":
        raise ValueError(f"Unknown retrieval key {key!r}")
    if task.description.count(GOAL_MARKER) != 1:
        raise ValueError("Task description must contain exactly one goal statement")
    goal = task.description.split(GOAL_MARKER, 1)[1].strip()
    if not goal or "\n" in goal:
        raise ValueError("Goal statement must be one non-empty final line")
    return goal


def writable(trajectory: AgentTrajectory) -> bool:
    """Phase 4 stores only completed, environment-confirmed successes."""
    return trajectory.status == "completed" and trajectory.success_known and trajectory.success


def build_writer_prompt(trajectory: AgentTrajectory) -> str:
    """Uses only observable interaction data: task, observations, executed actions."""
    if not writable(trajectory):
        raise ValueError("Episodes are written only for successful trajectories")
    executed = [s for s in trajectory.steps if s.executed]
    steps = [{"action": s.action, "observation": s.next_observation} for s in executed]
    return (
        "Write a concise episodic memory of this completed household task for an agent "
        "that will later attempt related tasks.\n"
        "Use only facts present in the record. Describe the concrete actions that achieved "
        "the goal, where the needed objects and receptacles were found, and any wasted "
        "actions to avoid.\n"
        "Return only a JSON object with exactly these string fields: "
        '"task_summary" (one sentence), "experience" (at most 120 words), '
        '"outcome" (one sentence). No markdown.\n'
        f"Task: {json.dumps(trajectory.task.description, ensure_ascii=False)}\n"
        f"Initial observation: {json.dumps(executed[0].observation, ensure_ascii=False)}\n"
        f"Executed actions and resulting observations: {json.dumps(steps, ensure_ascii=False)}\n"
        f"Environment outcome: success after {len(executed)} actions\n"
    )


def parse_episode(text: str) -> dict[str, str]:
    """Strict JSON; only one exact outer ```json fence is removed. No repair."""
    body = text.strip()
    if body.startswith("```json\n") and body.endswith("\n```"):
        body = body[len("```json\n"):-len("\n```")]
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise EpisodeParseError("Writer output is not a JSON object") from exc
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise EpisodeParseError(f"Writer output must have exactly the fields {FIELDS}")
    if any(not isinstance(value[f], str) or not value[f].strip() for f in FIELDS):
        raise EpisodeParseError("Writer fields must be non-empty strings")
    return {f: value[f].strip() for f in FIELDS}


def render_episode(memory: EpisodicMemory) -> dict[str, str]:
    return {"task": memory.task_summary, "experience": memory.experience,
            "outcome": memory.outcome}


def build_episode(trajectory: AgentTrajectory, fields: dict[str, str], *, retrieval_key: str,
                  count_tokens: Callable[[str], int]) -> EpisodicMemory:
    # Retrieval keys on the source task text, never on writer output.
    draft = EpisodicMemory(
        memory_id=f"episode-{trajectory.trajectory_id}",
        source_trajectory_ids=[trajectory.trajectory_id],
        embedding_text=retrieval_text(trajectory.task, retrieval_key), token_count=0, **fields,
    )
    rendered = json.dumps(render_episode(draft), ensure_ascii=False)
    return draft.model_copy(update={"token_count": count_tokens(rendered)})


def memory_section(memories: Sequence[EpisodicMemory]) -> str | None:
    """None when nothing is retrieved, so the prompt is byte-identical to no-memory."""
    if not memories:
        return None
    items = [render_episode(memory) for memory in memories]
    return ("Relevant past experiences (episodic memories, most similar first): "
            f"{json.dumps(items, ensure_ascii=False)}\n")
