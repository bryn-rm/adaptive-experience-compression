"""skill_writer_v1 and skill_section_v1: change versions whenever content changes."""

import json
from typing import Callable, Sequence

from aec.memory.episodes import EpisodeParseError, retrieval_text, writable
from aec.schemas import ProceduralSkill, Task
from aec.trajectories.schemas import AgentTrajectory

WRITER_VERSION = "skill_writer_v1"
SECTION_FORMAT = "skill_section_v1"
LIST_FIELDS = ("activation_conditions", "procedure", "constraints", "failure_modes")
FIELDS = ("name",) + LIST_FIELDS
NON_EMPTY_LISTS = ("activation_conditions", "procedure")


class SkillParseError(EpisodeParseError):
    pass


def task_family(task: Task) -> str:
    """Grouping key recorded by the environment adapter at reset."""
    family = task.metadata.get("task_family")
    if not isinstance(family, str) or not family:
        raise ValueError("Task lacks a task family; cannot group skill sources")
    return family


def build_skill_writer_prompt(trajectories: Sequence[AgentTrajectory]) -> str:
    """Uses only observable interaction data from successful trajectories, oldest first."""
    if not trajectories or not all(writable(t) for t in trajectories):
        raise ValueError("Skills are synthesized only from successful trajectories")
    if len({task_family(t.task) for t in trajectories}) != 1:
        raise ValueError("Skill sources must share one task family")
    records = []
    for t in trajectories:
        executed = [s for s in t.steps if s.executed]
        records.append({
            "task": t.task.description, "initial_observation": executed[0].observation,
            "steps": [{"action": s.action, "observation": s.next_observation} for s in executed],
        })
    return (
        "Write one reusable procedural skill for an agent in a text household environment, "
        "generalizing from these successful completions of the same kind of task.\n"
        "Use only facts present in the records. The skill must apply in new rooms: refer to "
        "object and receptacle types, not numbered instances.\n"
        "Return only a JSON object with exactly these fields: "
        '"name" (short string), '
        '"activation_conditions" (non-empty list of strings: when the skill applies), '
        '"procedure" (non-empty ordered list of strings: the steps), '
        '"constraints" (list of strings), '
        '"failure_modes" (list of strings: wasted or failed actions seen in the records and '
        "how to avoid them). Keep the whole skill under 150 words. No markdown.\n"
        f"Successful records (oldest first): {json.dumps(records, ensure_ascii=False)}\n"
    )


def parse_skill(text: str) -> dict:
    """Strict JSON; only one exact outer ```json fence is removed. No repair."""
    body = text.strip()
    if body.startswith("```json\n") and body.endswith("\n```"):
        body = body[len("```json\n"):-len("\n```")]
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SkillParseError("Writer output is not a JSON object") from exc
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise SkillParseError(f"Writer output must have exactly the fields {FIELDS}")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise SkillParseError("Skill name must be a non-empty string")
    for field in LIST_FIELDS:
        items = value[field]
        if not isinstance(items, list) or any(not isinstance(i, str) or not i.strip() for i in items):
            raise SkillParseError(f"{field} must be a list of non-empty strings")
        if field in NON_EMPTY_LISTS and not items:
            raise SkillParseError(f"{field} must not be empty")
    return {"name": value["name"].strip(),
            **{f: [i.strip() for i in value[f]] for f in LIST_FIELDS}}


def render_skill(skill: ProceduralSkill) -> dict:
    return {"name": skill.name, "when": skill.activation_conditions,
            "procedure": skill.procedure, "constraints": skill.constraints,
            "failure_modes": skill.failure_modes}


def build_skill(trajectories: Sequence[AgentTrajectory], fields: dict, *, version: int,
                retrieval_key: str, count_tokens: Callable[[str], int]) -> ProceduralSkill:
    # Retrieval keys on the sources' task text, never on writer output.
    draft = ProceduralSkill(
        skill_id=f"skill-{task_family(trajectories[0].task)}", version=version,
        source_trajectory_ids=[t.trajectory_id for t in trajectories],
        embedding_text="\n".join(retrieval_text(t.task, retrieval_key) for t in trajectories),
        token_count=0, **fields,
    )
    rendered = json.dumps(render_skill(draft), ensure_ascii=False)
    return draft.model_copy(update={"token_count": count_tokens(rendered)})


def record_id(skill: ProceduralSkill) -> str:
    return f"{skill.skill_id}@v{skill.version}"


def skill_section(skills: Sequence[ProceduralSkill]) -> str | None:
    """None when nothing is retrieved, so the prompt is byte-identical to no-memory."""
    if not skills:
        return None
    items = [render_skill(skill) for skill in skills]
    return ("Relevant skills (procedural memories, most similar first): "
            f"{json.dumps(items, ensure_ascii=False)}\n")
