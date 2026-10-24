import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
from pydantic import ValidationError

from aec.config import ALFWorldConfig, load_config
from aec.environments.alfworld import ALFWorldAdapter, ResetResult, StepResult
from aec.schemas import Task, TrajectoryStep


class Backend:
    def __init__(self):
        self.gamefile = "/data/pick_and_place_simple-Mug-None-Table/trial-1/game.tw-pddl"
        self.infos = {"extra.gamefile": [self.gamefile],
                      "admissible_commands": [["look", "take mug"]], "won": [False]}
        self.done = False
        self.actions = []
        self.closed = False

    def reset(self):
        return ["Your task is to put a mug on the table."], self.infos

    def step(self, actions):
        self.actions.append(actions)
        return ["You see a mug."], [1.0], [self.done], self.infos

    def close(self):
        self.closed = True


@pytest.fixture
def setup_env():
    backend = Backend()
    config = ALFWorldConfig(backend_config=Path("unused.yaml"), max_steps=2)
    return ALFWorldAdapter(config, backend=backend), backend


def test_reset_and_schema_conversion(setup_env):
    env, backend = setup_env
    with pytest.raises(RuntimeError):
        env.current_task()
    with pytest.raises(RuntimeError):
        env.step("look")
    result = env.reset()
    assert ResetResult.model_validate_json(result.model_dump_json()) == result
    assert isinstance(result.task, Task)
    assert result.task.task_id == backend.gamefile
    assert result.task.metadata == {
        "gamefile": backend.gamefile, "split": "eval_out_of_distribution",
        "task_family": "pick_and_place_simple",
    }
    result.task.metadata.clear()
    assert env.current_task().metadata
    task = env.current_task()
    task.metadata.clear()
    assert env.current_task().metadata
    actions = env.valid_actions()
    actions.clear()
    assert env.valid_actions() == ["look", "take mug"]


def test_step_conversion_and_terminal_success(setup_env):
    env, backend = setup_env
    env.reset()
    backend.done = True
    backend.infos["won"] = [True]
    result = env.step("take mug")
    assert isinstance(result, TrajectoryStep)
    assert StepResult.model_validate_json(result.model_dump_json()) == result
    assert backend.actions == [["take mug"]]
    assert result.observation == "You see a mug."
    assert result.action == "take mug"
    assert result.reward == 1
    assert result.done and result.success
    assert result.termination_reason == "environment"
    assert result.task_id == backend.gamefile
    assert env.valid_actions() == []
    with pytest.raises(RuntimeError):
        env.step("look")


def test_max_steps_reset_and_terminal_precedence(setup_env):
    env, backend = setup_env
    env.reset()
    assert env.step("look").termination_reason is None
    result = env.step("look")
    assert result.done and result.termination_reason == "max_steps"
    assert result.success is False
    with pytest.raises(RuntimeError):
        env.step("look")
    env.reset()
    assert not env.step("look").done
    backend.done = True
    result = env.step("look")
    assert result.termination_reason == "environment"
    assert result.success is False  # termination does not imply success


def test_missing_optional_infos_and_changing_actions(setup_env):
    env, backend = setup_env
    backend.infos.pop("admissible_commands")
    backend.infos.pop("won")
    env.reset()
    assert env.valid_actions() == []
    backend.infos["admissible_commands"] = [["open fridge"]]
    assert env.step("look").success is None
    assert env.valid_actions() == ["open fridge"]
    env.close()
    assert backend.closed
    with pytest.raises(RuntimeError):
        env.step("look")


@pytest.mark.parametrize("field,value", [("extra.gamefile", [""]),
    ("extra.gamefile", ["one", "two"]), ("admissible_commands", [object()])])
def test_malformed_backend_reset_fails(setup_env, field, value):
    env, backend = setup_env
    backend.infos[field] = value
    with pytest.raises(ValueError):
        env.reset()
    with pytest.raises(RuntimeError):
        env.step("look")


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_max_steps(value):
    with pytest.raises(ValidationError):
        ALFWorldConfig(backend_config="unused", max_steps=value)


def test_smoke_config_loads():
    assert load_config("configs/alfworld_smoke.yaml").environment.max_steps == 50


def test_factory_text_only_configuration(monkeypatch, tmp_path):
    backend = Backend()
    captured = {}

    class Owner:
        def __init__(self, settings, train_eval):
            captured.update(settings=settings, split=train_eval)
            self.game_files = ["b", "a"]

        def init_env(self, batch_size):
            captured.update(batch_size=batch_size, files=self.game_files.copy())
            return backend

    backend.seed = lambda seed: captured.update(seed=seed)
    module = ModuleType("alfworld.agents.environment.alfred_tw_env")
    module.AlfredTWEnv = Owner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = ALFWorldConfig(backend_config="configs/alfworld_backend.yaml", max_steps=7, seed=9)
    env = ALFWorldAdapter(config)
    assert captured["settings"]["general"]["training_method"] == "dqn"
    assert captured["settings"]["rl"]["training"]["max_nb_steps_per_episode"] == 8
    assert captured["settings"]["env"]["domain_randomization"] is False
    assert captured["batch_size"] == 1
    assert captured["files"] == ["a", "b"]
    assert captured["seed"] == 9
    env.close()
    bad = tmp_path / "thor.yaml"
    bad.write_text("env:\n  type: AlfredThorEnv\n")
    with pytest.raises(ValueError, match="text mode"):
        ALFWorldAdapter(ALFWorldConfig(backend_config=bad))


def test_three_episode_smoke_offline(setup_env, tmp_path):
    spec = importlib.util.spec_from_file_location("smoke", "scripts/smoke_alfworld.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env, backend = setup_env
    output = tmp_path / "smoke.jsonl"
    module.run_smoke(env, output, episodes=3, seed=0)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 9
    terminal = [row for row in rows if row.get("done")]
    assert len(terminal) == 3
    assert all(row["termination_reason"] == "max_steps" for row in terminal)
    for row in rows:
        assert row["task_id"] == backend.gamefile
        if row["event"] == "step":
            StepResult.model_validate({key: value for key, value in row.items()
                                       if key not in ("episode", "event")})
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.run_smoke(env, output, episodes=3, seed=0)
    assert output.read_bytes() == original
    repeat = tmp_path / "repeat.jsonl"
    module.run_smoke(env, repeat, episodes=3, seed=0)
    assert repeat.read_bytes() == original


def test_failed_step_conversion_requires_reset(setup_env):
    env, backend = setup_env
    env.reset()
    backend.infos["admissible_commands"] = [object()]
    with pytest.raises(ValueError):
        env.step("look")
    with pytest.raises(RuntimeError):
        env.step("look")


def test_direct_imports_stay_in_environment_layer():
    import ast

    for path in Path("src/aec").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            if any(name.split(".")[0] in {"alfworld", "textworld"} for name in modules):
                assert path.is_relative_to("src/aec/environments")
