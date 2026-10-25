"""agent_v1: change the version whenever acting prompt content changes."""

import json

from aec.schemas import Task

PROMPT_VERSION = "agent_v1"


def build_prompt(task: Task, observation: str, valid_actions: list[str]) -> str:
    return (
        "Choose one available action to complete the task.\n"
        "Return only the exact action on one line, without quotes or explanation.\n"
        f"Task: {json.dumps(task.description, ensure_ascii=False)}\n"
        f"Observation: {json.dumps(observation, ensure_ascii=False)}\n"
        f"Available actions: {json.dumps(sorted(valid_actions), ensure_ascii=False)}\n"
    )
