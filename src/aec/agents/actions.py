"""Strict action parsing: whitespace trimming is the only normalization."""


class ActionParseError(ValueError):
    pass


def parse_action(text: str, valid_actions: list[str]) -> str:
    action = text.strip()
    if not action or "\n" in action or "\r" in action:
        raise ActionParseError("Expected exactly one non-empty action line")
    if not valid_actions:
        raise ActionParseError("No admissible actions available for validation")
    if action not in valid_actions:
        raise ActionParseError("Output does not exactly match an admissible action")
    return action
