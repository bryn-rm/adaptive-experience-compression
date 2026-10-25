import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aec.agents.actions import ActionParseError, parse_action
from aec.agents.prompt import PROMPT_VERSION, build_prompt
from aec.agents.runner import run_episode
from aec.config import AgentConfig, ALFWorldConfig, LLMConfig, load_config
from aec.environments.alfworld import ALFWorldAdapter
from aec.llm import LLMResponse
from aec.schemas import Task, TokenUsage
from aec.trajectories.recorder import TrajectoryRecorder
from aec.trajectories.schemas import AgentTrajectory


class Backend:
    """Deterministic text backend: cumulative score stays one across two actions."""

    def __init__(self, *, success=True):
        self.episode = 0
        self.actions = []
        self.success = success
        self.steps = 0

    def infos(self):
        return {"extra.gamefile": [f"/data/pick_and_place_simple-Mug/trial-{self.episode}/game.tw-pddl"],
                "admissible_commands": [["look", "take mug"]],
                "won": [self.success if self.steps == 2 else False]}

    def reset(self):
        self.episode += 1
        self.steps = 0
        return ["Put a mug on the table."], self.infos()

    def step(self, actions):
        self.actions.append(actions)
        self.steps += 1
        return [f"Observation {self.steps}"], [1.0], [self.steps == 2], self.infos()


class FakeLLM:
    def __init__(self, outputs=("look", "take mug")):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, prompt, config):
        self.calls.append((prompt, config))
        text = next(self.outputs)
        if isinstance(text, Exception):
            raise text
        return LLMResponse(text=text, model_id="fake-snapshot", finish_reason="stop",
                           response_id="response-1", system_fingerprint="backend-1",
                           token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2))


@pytest.fixture
def setup_episode(tmp_path):
    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("env: {type: AlfredTWEnv}\n")
    env_config = ALFWorldConfig(backend_config=backend_config, seed=7, max_steps=3)
    backend = Backend()
    env = ALFWorldAdapter(env_config, backend=backend)
    config = AgentConfig(llm=LLMConfig(model_id="fake-snapshot", seed=11))
    recorder = TrajectoryRecorder(tmp_path / "trajectories.jsonl")
    return env, backend, config, recorder


def test_prompt_v1_snapshot():
    task = Task(task_id="t", description="Put mug on table")
    assert PROMPT_VERSION == "agent_v1"
    assert build_prompt(task, "A mug", ["take mug", "look"]) == (
        "Choose one available action to complete the task.\n"
        "Return only the exact action on one line, without quotes or explanation.\n"
        'Task: "Put mug on table"\n'
        'Observation: "A mug"\n'
        'Available actions: ["look", "take mug"]\n'
    )
    assert build_prompt(task, "A mug", ["look", "take mug"]) == build_prompt(
        task, "A mug", ["take mug", "look"])


def test_parse_single_action():
    assert parse_action("  take mug\n", ["take mug"]) == "take mug"


@pytest.mark.parametrize("text,actions", [
    ("", ["look"]), ("look\ntake mug", ["look", "take mug"]),
    ('"look"', ["look"]), ("Action: look", ["look"]),
    ("```\nlook\n```", ["look"]), ("LOOK", ["look"]),
    ("invent action", ["look"]), ("look", []),
])
def test_parse_failure(text, actions):
    with pytest.raises(ActionParseError):
        parse_action(text, actions)


def test_completed_trajectory_records_observables_and_totals(setup_episode):
    env, backend, config, recorder = setup_episode
    llm = FakeLLM()
    ticks = iter([0, 1, 3, 4, 7, 9])
    trajectory = run_episode(env, llm, config, recorder, run_id="run-1", clock=lambda: next(ticks))
    assert trajectory.status == "completed"
    assert trajectory.success and trajectory.success_known
    assert trajectory.termination_reason == "environment"
    assert trajectory.reward == trajectory.total_reward == 1  # not 1 + 1
    assert trajectory.wall_time == 9
    assert [s.wall_time for s in trajectory.steps] == [2, 3]
    assert trajectory.seed == 7 and trajectory.agent_config.llm.seed == 11
    assert trajectory.prompt_tokens == 20 and trajectory.completion_tokens == 4
    assert trajectory.token_usage == TokenUsage(prompt_tokens=20, completion_tokens=4)
    assert trajectory.environment_backend_sha256 == hashlib.sha256(
        env.config.backend_config.read_bytes()).hexdigest()
    first, second = trajectory.steps
    assert first.observation == "Put a mug on the table."
    assert first.next_observation == second.observation == "Observation 1"
    assert second.next_observation == "Observation 2"
    assert first.output == "look" and first.action == "look"
    assert first.valid_action and first.executed and first.usage_known
    assert not first.done and second.done
    assert second.environment_termination_reason == "environment"
    assert first.response_model_id == "fake-snapshot"
    assert first.system_fingerprint == "backend-1"
    assert backend.actions == [["look"], ["take mug"]]
    assert all(settings == config.llm for _, settings in llm.calls)
    assert llm.calls[0][0] == first.prompt
    assert llm.calls[1][0] == second.prompt
    # No prior interaction transcript is carried into the next prompt.
    assert first.prompt not in second.prompt
    rows = recorder.path.read_text().splitlines()
    assert len(rows) == 1
    assert AgentTrajectory.model_validate_json(rows[0]) == trajectory


def test_parse_failure_logged_without_action_substitution(setup_episode, caplog):
    env, backend, config, recorder = setup_episode
    trajectory = run_episode(env, FakeLLM(["Action: look"]), config, recorder, run_id="bad")
    assert trajectory.status == "aborted" and not trajectory.success
    assert trajectory.termination_reason == "parse_error"
    assert backend.actions == []
    step, = trajectory.steps
    assert step.parse_error and not step.valid_action and not step.executed
    assert step.output == "Action: look" and step.action == ""
    assert step.token_usage.prompt_tokens == trajectory.prompt_tokens == 10
    assert "Action parse failure" in caplog.text
    assert AgentTrajectory.model_validate_json(recorder.path.read_text()) == trajectory


def test_provider_error_aborts_and_logs_without_secrets(setup_episode):
    env, backend, config, recorder = setup_episode
    trajectory = run_episode(env, FakeLLM([RuntimeError("secret message")]), config,
                             recorder, run_id="error")
    assert trajectory.termination_reason == "provider_error"
    assert trajectory.status == "aborted"
    assert trajectory.steps[0].error_type == "RuntimeError"
    assert not trajectory.steps[0].usage_known
    assert trajectory.prompt_tokens == 0
    assert not backend.actions
    assert "secret message" not in recorder.path.read_text()


def test_environment_error_logged(setup_episode):
    env, backend, config, recorder = setup_episode
    backend.step = lambda _: (_ for _ in ()).throw(ValueError("backend failed"))
    trajectory = run_episode(env, FakeLLM(), config, recorder, run_id="error")
    assert trajectory.termination_reason == "environment_error"
    assert trajectory.steps[0].valid_action
    assert not trajectory.steps[0].executed
    assert trajectory.steps[0].error_type == "ValueError"
    assert trajectory.prompt_tokens == 10


def test_no_actions_aborts_without_llm_call(setup_episode):
    env, backend, config, recorder = setup_episode
    backend.infos = lambda: {"extra.gamefile": ["/task"], "admissible_commands": [[]]}
    llm = FakeLLM()
    trajectory = run_episode(env, llm, config, recorder, run_id="no-actions")
    assert trajectory.termination_reason == "no_valid_actions"
    assert trajectory.steps == [] and llm.calls == []
    assert trajectory.status == "aborted"


def test_max_steps_and_unknown_success(setup_episode):
    env, backend, config, recorder = setup_episode
    env.config.max_steps = 1
    trajectory = run_episode(env, FakeLLM(), config, recorder, run_id="limit")
    assert trajectory.termination_reason == "max_steps"
    assert trajectory.status == "completed" and not trajectory.success
    original_infos = backend.infos
    backend.infos = lambda: {k: v for k, v in original_infos().items() if k != "won"}
    env.config.max_steps = 3
    trajectory = run_episode(env, FakeLLM(), config, recorder, run_id="unknown")
    assert trajectory.termination_reason == "missing_success"
    assert not trajectory.success_known and trajectory.status == "aborted"


def test_truncated_response_is_not_executed(setup_episode):
    env, backend, config, recorder = setup_episode
    class Truncated(FakeLLM):
        def generate(self, prompt, config):
            result = super().generate(prompt, config)
            result.finish_reason = "length"
            return result
    trajectory = run_episode(env, Truncated(), config, recorder, run_id="truncated")
    assert trajectory.termination_reason == "parse_error"
    assert backend.actions == []


def test_append_only_and_malformed_artifacts(setup_episode):
    env, _, config, recorder = setup_episode
    first = run_episode(env, FakeLLM(), config, recorder, run_id="one")
    prefix = recorder.path.read_bytes()
    second = run_episode(env, FakeLLM(), config, recorder, run_id="two")
    assert recorder.path.read_bytes().startswith(prefix)
    assert first.trajectory_id != second.trajectory_id
    assert len(recorder.path.read_text().splitlines()) == 2
    first.prompt_tokens = -1
    with pytest.raises(ValidationError):
        recorder.append(first)
    with recorder.path.open("a") as stream:
        stream.write('{"broken":')
    prior = recorder.path.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        recorder.append(second)
    assert recorder.path.read_bytes() == prior


def test_ten_task_offline_batch(setup_episode):
    env, backend, config, recorder = setup_episode
    for index in range(10):
        backend.success = index % 2 == 0
        outputs = ["look", "take mug"] if index != 9 else ["malformed"]
        run_episode(env, FakeLLM(outputs), config, recorder, run_id="smoke-10")
    artifacts = [AgentTrajectory.model_validate_json(line)
                 for line in recorder.path.read_text().splitlines()]
    assert len(artifacts) == len({a.trajectory_id for a in artifacts}) == 10
    assert len({a.task_id for a in artifacts}) == 10
    assert sum(a.success for a in artifacts) == 5
    assert sum(a.status == "completed" and not a.success for a in artifacts) == 4
    assert sum(a.termination_reason == "parse_error" for a in artifacts) == 1
    assert len({a.agent_config.model_dump_json() for a in artifacts}) == 1
    assert len({a.environment_config.model_dump_json() for a in artifacts}) == 1
    assert all(a.model_id == "fake-snapshot" and a.prompt_version == "agent_v1" for a in artifacts)


def test_config_frozen_and_example_valid():
    config = load_config("configs/agent_smoke.yaml")
    assert config.agent.prompt_version == PROMPT_VERSION
    with pytest.raises(ValidationError):
        config.agent.llm.temperature = 1
    with pytest.raises(ValidationError):
        AgentConfig(llm=config.agent.llm, prompt_version="unknown")
    with pytest.raises(ValidationError):
        LLMConfig(model_id="m", temperature=float("nan"))


@pytest.mark.parametrize("mutation", ["model", "usage", "reward", "indices", "validity", "terminal"])
def test_inconsistent_artifact_rejected(setup_episode, mutation):
    env, _, config, recorder = setup_episode
    trajectory = run_episode(env, FakeLLM(), config, recorder, run_id="schema")
    data = trajectory.model_dump(mode="json")
    if mutation == "model":
        data["model_id"] = "different-model"
    elif mutation == "usage":
        data["prompt_tokens"] += 1
    elif mutation == "reward":
        data["total_reward"] = 999
    elif mutation == "indices":
        data["steps"][0]["step_index"] = 5
    elif mutation == "validity":
        data["steps"][0]["action"] = "unavailable"
    elif mutation == "terminal":
        data["steps"][-1]["done"] = False
    with pytest.raises(ValidationError):
        AgentTrajectory.model_validate(data)
