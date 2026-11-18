"""agent_v2: bounded observable working history and deterministic action formatting."""

from dataclasses import dataclass
import json
from typing import Sequence

from aec.agents.actions import ActionParseError, parse_action
from aec.schemas import Task

PROMPT_VERSION = "agent_v2"


@dataclass(frozen=True)
class ObservationAction:
    """Observation before a confirmed action; contains no provider reasoning/output."""

    observation: str
    action: str


def build_prompt_v2(task: Task, observation: str, valid_actions: list[str],
                    history: Sequence[ObservationAction], *, history_length: int) -> str:
    if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 0:
        raise ValueError("history_length must be a non-negative integer")
    recent = history[-history_length:] if history_length else []
    pairs = [{"observation": pair.observation, "action": pair.action} for pair in recent]
    return (
        "Choose one available action to complete the task.\n"
        "Return only the exact action on one line, without quotes or explanation.\n"
        f"Task: {json.dumps(task.description, ensure_ascii=False)}\n"
        "Recent completed observation/action pairs (oldest first): "
        f"{json.dumps(pairs, ensure_ascii=False)}\n"
        f"Observation: {json.dumps(observation, ensure_ascii=False)}\n"
        f"Available actions: {json.dumps(sorted(valid_actions), ensure_ascii=False)}\n"
    )


@dataclass(frozen=True)
class ParsedAction:
    action: str
    normalized: bool


def parse_action_v2(text: str, valid_actions: list[str]) -> ParsedAction:
    """Trim whitespace, one exact `Action: ` prefix, and one pair of straight quotes.

    No recursive unwrapping, case folding, fuzzy matching, retries, or fallback.
    The flag includes whitespace trimming. Failed parses never return an action.
    """
    action = text.strip()
    if not action or "\n" in action or "\r" in action:
        raise ActionParseError("Expected exactly one non-empty action line")
    if action not in valid_actions:
        if action.startswith("Action: "):
            action = action[len("Action: "):].strip()
        if len(action) >= 2 and action[0] in ('"', "'") and action[-1] == action[0]:
            action = action[1:-1]
    action = parse_action(action, valid_actions)
    return ParsedAction(action=action, normalized=action != text)
