import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aec.config import AgentConfig, ALFWorldConfig, LLMConfig
from aec.environments.alfworld import ALFWorldAdapter
from aec.experiments.artifacts import (
    BaselineConfig, TaskManifest, TaskResult, create_manifest, read_rows, write_once,
)
from aec.experiments.baseline import aggregate, run_baseline
from aec.llm import LLMResponse
from aec.schemas import TokenUsage
from aec.trajectories.schemas import AgentTrajectory


class Backend:
    def __init__(self, task_id):
        self.task_id = task_id
        self.count = 0

    def infos(self):
        return {"extra.gamefile": [self.task_id], "admissible_commands": [["look"]],
                "won": [self.count == 2 and Path(self.task_id).name != "failure"]}

    def reset(self):
        self.count = 0
        return ["Put a mug on a table"], self.infos()

    def step(self, actions):
        assert actions == ["look"]
        self.count += 1
        return ["A mug"], [float(self.infos()["won"][0])], [self.count == 2], self.infos()

    def close(self):
        pass


class LLM:
    def generate(self, prompt, config):
        return LLMResponse(text="look", model_id=config.model_id, finish_reason="stop",
                           token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2))

    def close(self):
        pass


@pytest.fixture
def experiment(tmp_path):
    backend = tmp_path / "backend.yaml"
    backend.write_text("env: {type: AlfredTWEnv}\n")
    candidates = []
    for name in ("success", "failure", "third"):
        path = tmp_path / name
        path.write_text(name)
        candidates.append(str(path))
    config = BaselineConfig(
        run_id="test", manifest=tmp_path / "manifest.json", output_dir=tmp_path / "runs",
        environment=ALFWorldConfig(backend_config=backend, max_steps=3, seed=7),
        agent=AgentConfig(llm=LLMConfig(model_id="fake", seed=11)),
        selection_count=3, data_kind="fake_verification",
    )
    manifest = create_manifest(config, tuple(candidates))
    seen = []

    def env_factory(settings, *, task_id):
        seen.append(task_id)
        return ALFWorldAdapter(settings, backend=Backend(task_id), task_id=task_id)

    return config, manifest, env_factory, seen


def artifacts(config):
    root = config.output_dir / config.run_id
    return (read_rows(root / "trajectories.jsonl", AgentTrajectory),
            read_rows(root / "results.jsonl", TaskResult))


def test_manifest_validation_and_immutability(experiment):
    config, manifest, _, _ = experiment
    assert TaskManifest.model_validate_json(config.manifest.read_text()) == manifest
    with pytest.raises(ValidationError):
        manifest.task_ids = ("x",)
    for updates in ({"task_ids": ()}, {"task_ids": ("x", "x")},
                    {"task_sha256": ("bad",)}, {"split": "unknown"},
                    {"created_at": "2026-01-01T00:00:00"}):
        with pytest.raises(ValidationError):
            TaskManifest.model_validate(manifest.model_dump() | updates)
    before = config.manifest.read_bytes()
    with pytest.raises(FileExistsError):
        create_manifest(config, manifest.task_ids)
    assert config.manifest.read_bytes() == before


def test_selection_is_seeded_and_independent_of_candidate_order(experiment, tmp_path):
    config, manifest, _, _ = experiment
    config.manifest = tmp_path / "another.json"
    repeated = create_manifest(config, tuple(reversed(manifest.task_ids)))
    assert repeated.task_ids == manifest.task_ids
    assert repeated.manifest_id == manifest.manifest_id
    config.selection_count = 4
    with pytest.raises(ValueError):
        create_manifest(config, manifest.task_ids)


def test_fixed_execution_results_metrics_and_skip(experiment):
    config, manifest, factory, seen = experiment
    metrics = run_baseline(config, env_factory=factory, llm_factory=LLM)
    assert seen == list(manifest.task_ids)
    ts, rs = artifacts(config)
    assert len(ts) == len(rs) == 3
    assert [t.task_id for t in ts] == list(manifest.task_ids)
    assert all(t.agent_config == config.agent and t.environment_config == config.environment for t in ts)
    assert all(r.manifest_id == manifest.manifest_id for r in rs)
    assert metrics | {} == json.loads((config.output_dir / config.run_id / "metrics.json").read_text())
    assert metrics["completed_tasks"] == 3 and metrics["successful_tasks"] == 2
    assert metrics["environment_failures"] == 1 and metrics["aborted_tasks"] == 0
    assert metrics["success_rate"] == 2 / 3
    assert metrics["mean_steps"] == metrics["median_steps"] == 2
    assert metrics["mean_reward"] == 2 / 3
    assert metrics["prompt_tokens"] == 60 and metrics["completion_tokens"] == 12
    assert metrics["usage_known"] and metrics["wall_time"] > 0
    assert metrics["data_kind"] == "fake_verification"
    before = {p: p.read_bytes() for p in (config.output_dir / config.run_id).glob("*.json*")}
    def forbidden(*args, **kwargs):
        pytest.fail("Resume must not initialize environment/provider for a finished pair")
    assert run_baseline(config, env_factory=forbidden, llm_factory=forbidden) == metrics
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("failure,category,known", [
    ("parse", "aborted", True), ("provider", "provider_failure", False),
    ("setup", "setup_failure", False), ("step", "runtime_failure", True),
    ("reset", "runtime_failure", False),
])
def test_failure_separation_and_unknown_usage(experiment, failure, category, known):
    config, _, factory, _ = experiment
    class BrokenLLM(LLM):
        def generate(self, prompt, settings):
            if failure == "provider":
                raise RuntimeError("secret must not persist")
            result = super().generate(prompt, settings)
            if failure == "parse":
                result.text = "Action: look"
            return result
    def broken_factory(settings, *, task_id):
        if failure == "setup":
            raise RuntimeError("secret must not persist")
        env = factory(settings, task_id=task_id)
        def fail(*args):
            raise ValueError("secret must not persist")
        if failure == "step":
            env.step = fail
        if failure == "reset":
            env.reset = fail
        return env
    metrics = run_baseline(config, env_factory=broken_factory, llm_factory=BrokenLLM)
    ts, rs = artifacts(config)
    assert all(t.status == "aborted" for t in ts)
    assert all(r.outcome == category and r.usage_known == known for r in rs)
    assert metrics["environment_failures"] == metrics["completed_tasks"] == 0
    assert metrics["aborted_tasks"] == 3 and metrics["abort_rate"] == 1
    assert metrics["success_rate"] is metrics["mean_reward"] is metrics["mean_steps"] is None
    assert metrics["usage_known"] == known
    assert metrics["parse_failures"] == (3 if failure == "parse" else 0)
    assert all("secret" not in t.model_dump_json() for t in ts)


def test_interruption_is_not_replayed_and_existing_rows_are_append_only(experiment):
    config, manifest, factory, seen = experiment
    class Interrupt(LLM):
        calls = 0
        def generate(self, prompt, settings):
            Interrupt.calls += 1
            if Interrupt.calls == 3:
                raise KeyboardInterrupt
            return super().generate(prompt, settings)
    with pytest.raises(KeyboardInterrupt):
        run_baseline(config, env_factory=factory, llm_factory=Interrupt)
    root = config.output_dir / config.run_id
    prefix = (root / "results.jsonl").read_bytes()
    metrics = run_baseline(config, env_factory=factory, llm_factory=LLM)
    assert (root / "results.jsonl").read_bytes().startswith(prefix)
    assert seen == list(manifest.task_ids)  # Interrupted task was not retried.
    ts, rs = artifacts(config)
    assert len(ts) == len(rs) == 3
    assert sum(t.termination_reason == "interrupted" for t in ts) == 1
    assert metrics["aborted_tasks"] == 1 and metrics["unknown_usage_tasks"] == 1
    assert not metrics["wall_time_known"]


def test_recover_trajectory_committed_before_result(experiment, monkeypatch):
    import aec.experiments.baseline as module
    config, _, factory, seen = experiment
    original = module.append_row
    def fail_result(path, value):
        if isinstance(value, TaskResult):
            raise OSError("disk failure")
        original(path, value)
    monkeypatch.setattr(module, "append_row", fail_result)
    with pytest.raises(OSError):
        run_baseline(config, env_factory=factory, llm_factory=LLM)
    ts, rs = artifacts(config)
    assert len(ts) == 1 and not rs
    monkeypatch.setattr(module, "append_row", original)
    run_baseline(config, env_factory=factory, llm_factory=LLM)
    assert len(seen) == 3
    assert artifacts(config)[0][0] == ts[0]


@pytest.mark.parametrize("change", ["model", "seed", "task", "backend", "manifest", "partial", "result"])
def test_resume_rejects_drift_and_corruption_without_writes(experiment, change):
    config, manifest, factory, _ = experiment
    run_baseline(config, env_factory=factory, llm_factory=LLM)
    root = config.output_dir / config.run_id
    if change == "model":
        config.agent = AgentConfig(llm=LLMConfig(model_id="different"))
    elif change == "seed":
        config.environment.seed += 1
    elif change == "task":
        Path(manifest.task_ids[0]).write_text("changed")
    elif change == "backend":
        config.environment.backend_config.write_text("env: {type: AlfredTWEnv, changed: true}")
    elif change == "manifest":
        data = manifest.model_dump(mode="json")
        data["task_ids"].reverse()
        data["task_sha256"].reverse()
        config.manifest.write_text(json.dumps(data))
    elif change == "partial":
        with (root / "results.jsonl").open("a") as f:
            f.write('{"incomplete":')
    else:
        path = root / "results.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        lines[0]["prompt_tokens"] += 1
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    before = {p: p.read_bytes() for p in root.glob("*.json*")}
    with pytest.raises(ValueError):
        run_baseline(config, env_factory=factory, llm_factory=LLM)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_deterministic_fake_execution(experiment):
    config, _, factory, _ = experiment
    run_baseline(config, env_factory=factory, llm_factory=LLM)
    first, _ = artifacts(config)
    config.run_id = "repeat"
    run_baseline(config, env_factory=factory, llm_factory=LLM)
    second, _ = artifacts(config)
    for a, b in zip(first, second):
        assert (a.task_id, a.success, a.reward, a.token_usage) == (b.task_id, b.success, b.reward, b.token_usage)
        assert [(s.prompt, s.action, s.next_observation) for s in a.steps] == [
            (s.prompt, s.action, s.next_observation) for s in b.steps]


def test_aggregate_empty_duplicate_and_unknown_partial_usage(experiment):
    config, _, factory, _ = experiment
    class Partial(LLM):
        calls = 0
        def generate(self, prompt, settings):
            Partial.calls += 1
            if Partial.calls % 2 == 0:
                raise RuntimeError()
            return super().generate(prompt, settings)
    metrics = run_baseline(config, env_factory=factory, llm_factory=Partial)
    assert metrics["prompt_tokens"] == 30 and metrics["completion_tokens"] == 6
    assert not metrics["usage_known"] and metrics["unknown_usage_tasks"] == 3
    _, rs = artifacts(config)
    with pytest.raises(ValueError):
        aggregate(rs + rs)
    assert aggregate([])["success_rate"] is None
    assert aggregate([])["abort_rate"] is None


def test_atomic_write_does_not_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    write_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_once(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_fake_factories_cannot_be_labeled_real(experiment):
    config, _, factory, _ = experiment
    config.data_kind = "real_alfworld"
    with pytest.raises(ValueError, match="fake_verification"):
        run_baseline(config, env_factory=factory, llm_factory=LLM)


def test_concurrent_writer_is_rejected(experiment):
    import fcntl
    config, _, factory, _ = experiment
    root = config.output_dir / config.run_id
    root.mkdir(parents=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run_baseline(config, env_factory=factory, llm_factory=LLM)
    assert not list(root.glob("*.jsonl"))
