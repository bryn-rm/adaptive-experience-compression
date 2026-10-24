"""Single-episode adapter for ALFWorld's batched text environment."""

from pathlib import Path
from typing import Any, Literal

import yaml

from aec.config import ALFWorldConfig
from aec.schemas import NonEmpty, Schema, Task, TrajectoryStep


class ResetResult(Schema):
    task: Task
    observation: str


class StepResult(TrajectoryStep):
    task_id: NonEmpty
    termination_reason: Literal["environment", "max_steps"] | None = None
    success: bool | None = None


def _one(values: Any) -> Any:
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise ValueError("ALFWorld must return a batch of exactly one")
    return values[0]


def _create_backend(config: ALFWorldConfig) -> Any:
    # Optional dependencies stay inside the integration layer, including imports.
    try:
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    except ImportError as exc:
        raise RuntimeError("Install the optional 'alfworld' dependency to run text episodes") from exc

    with config.backend_config.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("ALFWorld backend config must be a YAML mapping")
    if settings.get("env", {}).get("type") != "AlfredTWEnv":
        raise ValueError("Only AlfredTWEnv text mode is supported")
    # DQN selects the non-expert wrapper path; no learning code is run.
    settings.setdefault("general", {})["training_method"] = "dqn"
    settings["env"]["domain_randomization"] = False
    # Let the adapter enforce the limit before the backend's own timeout.
    settings.setdefault("rl", {}).setdefault("training", {})[
        "max_nb_steps_per_episode"
    ] = config.max_steps + 1
    owner = AlfredTWEnv(settings, train_eval=config.split)
    if not owner.game_files:
        raise ValueError("No ALFWorld games found; check task data paths and split")
    owner.game_files.sort()
    backend = owner.init_env(batch_size=1)
    backend.seed(config.seed)
    return backend


class ALFWorldAdapter:
    """Owns one text backend. ``backend`` permits offline fixture injection.

    Rewards are passed through unchanged (TextWorld reports game scores).
    A backend terminal result takes precedence if it coincides with max_steps.
    """

    def __init__(self, config: ALFWorldConfig, *, backend: Any = None):
        self.config = config.model_copy(deep=True)
        self._backend = backend if backend is not None else _create_backend(self.config)
        self._task: Task | None = None
        self._actions: list[str] = []
        self._steps = 0
        self._done = True

    def _set_actions(self, infos: dict) -> None:
        actions = _one(infos["admissible_commands"]) if "admissible_commands" in infos else []
        if not isinstance(actions, (list, tuple)) or not all(isinstance(a, str) for a in actions):
            raise ValueError("Malformed ALFWorld admissible commands")
        self._actions = list(actions)

    def reset(self) -> ResetResult:
        self._task = None
        self._done = True
        self._actions = []
        observations, infos = self._backend.reset()
        observation = _one(observations)
        gamefile = _one(infos["extra.gamefile"])
        if not isinstance(gamefile, str) or not gamefile:
            raise ValueError("ALFWorld must provide extra.gamefile task identity")
        path = Path(gamefile)
        metadata = {"gamefile": gamefile, "split": self.config.split}
        # Standard layout: <family>-<objects...>/<trial>/game.tw-pddl.
        if len(path.parents) >= 2:
            metadata["task_family"] = path.parent.parent.name.split("-", 1)[0]
        self._task = Task(task_id=gamefile, description=observation, metadata=metadata)
        self._set_actions(infos)
        self._steps = 0
        self._done = False
        return ResetResult(task=self.current_task(), observation=observation)

    def step(self, action: str) -> StepResult:
        if self._task is None or self._done:
            raise RuntimeError("Call reset before stepping or after episode termination")
        if not isinstance(action, str):
            raise TypeError("Action must be a string")
        # If conversion fails after backend mutation, require reset before reuse.
        self._done = True
        observations, rewards, dones, infos = self._backend.step([action])
        self._steps += 1
        terminated = _one(dones)
        if not isinstance(terminated, bool):
            raise ValueError("Malformed ALFWorld termination flag")
        reason = "environment" if terminated else (
            "max_steps" if self._steps >= self.config.max_steps else None
        )
        result = StepResult(
            task_id=self._task.task_id, observation=_one(observations), action=action,
            reward=_one(rewards), done=reason is not None, termination_reason=reason,
            success=_one(infos["won"]) if "won" in infos else None,
        )
        self._set_actions(infos)
        self._done = result.done
        return result

    def valid_actions(self) -> list[str]:
        return [] if self._done else list(self._actions)

    def current_task(self) -> Task:
        if self._task is None:
            raise RuntimeError("Call reset before requesting the current task")
        return self._task.model_copy(deep=True)

    def close(self) -> None:
        self._done = True
        self._backend.close()
