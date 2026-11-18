"""One writer, one immutable run, one canonical trajectory per attempted task."""

import fcntl
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
from statistics import mean, median
from time import perf_counter
from uuid import uuid4

from aec.agents.runner import run_episode
from aec.environments.alfworld import ALFWorldAdapter, resolved_backend_settings
from aec.llm.openai import OpenAIClient
from aec.memory.embeddings import OpenAIEmbedder
from aec.schemas import Task, TokenUsage
from aec.trajectories.recorder import TrajectoryRecorder
from aec.trajectories.schemas import AgentTrajectory
from aec.experiments.artifacts import (
    Attempt, BaselineConfig, RetrievalLog, TaskManifest, TaskResult, append_row, digest,
    file_hash, read_rows, utc_now, write_once,
)
from aec.experiments.memory import EpisodicPolicy, token_counter


def resolve_run(config: BaselineConfig, manifest: TaskManifest) -> dict:
    if config.environment.split != manifest.split:
        raise ValueError("Manifest split differs from environment split")
    for task, expected in zip(manifest.task_ids, manifest.task_sha256):
        if file_hash(task) != expected:
            raise ValueError("Manifest task content has changed")
    source_root = Path(__file__).resolve().parents[1]
    return dict(
        version="baseline_v1", policy=config.memory.policy if config.memory else "no_memory",
        config=config.model_dump(mode="json"),
        manifest=manifest.model_dump(mode="json"),
        manifest_sha256=digest(manifest.model_dump(mode="json")),
        environment_backend_sha256=file_hash(config.environment.backend_config),
        environment_backend=resolved_backend_settings(config.environment),
        python=platform.python_version(),
        packages=dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
        source_sha256=digest({str(p.relative_to(source_root)): file_hash(p)
                              for p in sorted(source_root.rglob("*.py"))}),
    )


def task_result(trajectory: AgentTrajectory, resolved: dict,
                retrieval: RetrievalLog | None = None) -> TaskResult:
    t = trajectory
    if t.status == "completed":
        outcome = "success" if t.success else "environment_failure"
    else:
        outcome = {"setup_error": "setup_failure", "provider_error": "provider_failure",
                   "environment_error": "runtime_failure", "runtime_error": "runtime_failure"
                   }.get(t.termination_reason, "aborted")
    lost = t.termination_reason in {"interrupted", "runtime_error", "setup_error"}
    return TaskResult(
        task_id=t.task_id, run_id=t.run_id, trajectory_id=t.trajectory_id,
        manifest_id=resolved["manifest"]["manifest_id"],
        manifest_sha256=resolved["manifest_sha256"], resolved_config_sha256=digest(resolved),
        data_kind=resolved["config"]["data_kind"], outcome=outcome,
        termination_reason=t.termination_reason, steps=sum(s.executed for s in t.steps),
        parse_failures=sum(s.parse_error is not None for s in t.steps),
        normalized_actions=sum(s.action_normalized for s in t.steps),
        reward=t.reward if t.status == "completed" else None,
        prompt_tokens=t.prompt_tokens, completion_tokens=t.completion_tokens,
        usage_known=not lost and all(s.usage_known for s in t.steps),
        wall_time=t.wall_time, wall_time_known=t.termination_reason != "interrupted",
        retrieved_memory_ids=[r.memory_id for r in retrieval.records] if retrieval else [],
    )


def aggregate(results: list[TaskResult]) -> dict:
    """Steps/reward summarize completed outcomes; tokens are observed lower bounds."""
    if len({(r.run_id, r.task_id) for r in results}) != len(results):
        raise ValueError("Duplicate task/run results")
    if len({(r.resolved_config_sha256, r.data_kind) for r in results}) > 1:
        raise ValueError("Cannot aggregate different experiments or real/fake data")
    completed = [r for r in results if r.outcome in {"success", "environment_failure"}]
    successes = sum(r.outcome == "success" for r in results)
    unknown = sum(not r.usage_known for r in results)
    return dict(
        attempted_tasks=len(results), completed_tasks=len(completed),
        successful_tasks=successes, environment_failures=len(completed) - successes,
        aborted_tasks=len(results) - len(completed),
        setup_failures=sum(r.outcome == "setup_failure" for r in results),
        provider_failures=sum(r.outcome == "provider_failure" for r in results),
        runtime_failures=sum(r.outcome == "runtime_failure" for r in results),
        parse_failures=sum(r.parse_failures for r in results),
        task_parse_failures=sum(r.parse_failures > 0 for r in results),
        normalized_actions=sum(r.normalized_actions for r in results),
        success_rate=successes / len(completed) if completed else None,
        abort_rate=(len(results) - len(completed)) / len(results) if results else None,
        mean_steps=mean(r.steps for r in completed) if completed else None,
        median_steps=median(r.steps for r in completed) if completed else None,
        mean_reward=mean(r.reward for r in completed) if completed else None,
        prompt_tokens=sum(r.prompt_tokens for r in results),
        completion_tokens=sum(r.completion_tokens for r in results),
        usage_known=unknown == 0, unknown_usage_tasks=unknown,
        wall_time=sum(r.wall_time for r in results),
        wall_time_known=all(r.wall_time_known for r in results),
        data_kind=results[0].data_kind if results else None,
    )


def failure_trajectory(config, resolved, task_id, reason, elapsed=0.0, error_type=None):
    # The manifest supplies identity, not an observed reset or fabricated interaction.
    task = Task(task_id=task_id, description="No durable episode observations available",
                metadata={"split": config.environment.split, "observations_available": False,
                          "error_type": error_type})
    return AgentTrajectory(
        trajectory_id=str(uuid4()), run_id=config.run_id, task_id=task_id, task=task,
        steps=[], success=False, success_known=False, reward=0, total_reward=0,
        status="aborted", termination_reason=reason, wall_time=elapsed,
        model_id=config.agent.llm.model_id, prompt_version=config.agent.prompt_version,
        seed=config.environment.seed, token_usage=TokenUsage(), prompt_tokens=0,
        completion_tokens=0, agent_config=config.agent, environment_config=config.environment,
        environment_backend_sha256=resolved["environment_backend_sha256"],
    )


def _index(rows, config, manifest):
    result = {}
    for row in rows:
        if row.run_id != config.run_id or row.task_id not in manifest.task_ids:
            raise ValueError("Artifact belongs to a different task/run")
        if row.task_id in result:
            raise ValueError("Duplicate task/run artifact")
        result[row.task_id] = row
    return result


class CheckedRecorder(TrajectoryRecorder):
    """Check provenance before a trajectory can enter the canonical log."""

    def __init__(self, path, task_id, config, resolved):
        super().__init__(path)
        self.task_id, self.config, self.resolved = task_id, config, resolved

    def append(self, t):
        if (t.task_id != self.task_id or t.run_id != self.config.run_id
            or t.agent_config != self.config.agent or t.environment_config != self.config.environment
            or t.environment_backend_sha256 != self.resolved["environment_backend_sha256"]):
            raise ValueError("Trajectory provenance differs from the frozen experiment")
        super().append(t)


def run_baseline(config: BaselineConfig, *, env_factory=ALFWorldAdapter,
                 llm_factory=OpenAIClient, embedder_factory=OpenAIEmbedder,
                 count_tokens=None) -> dict:
    """Shared runner for every memory policy; `config.memory=None` is no memory."""
    config = config.model_copy(deep=True)
    if config.data_kind != "fake_verification" and (
        env_factory is not ALFWorldAdapter or llm_factory is not OpenAIClient
        or embedder_factory is not OpenAIEmbedder or count_tokens is not None
    ):
        raise ValueError("Injected environments/providers require fake_verification labeling")
    manifest = TaskManifest.model_validate_json(config.manifest.read_text())
    resolved = resolve_run(config, manifest)
    directory = config.output_dir / config.run_id
    directory.mkdir(parents=True, exist_ok=True)
    # Kernel releases this lock on crash; never unlink the lock inode.
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resolved_path = directory / "resolved_config.json"
        fresh = not resolved_path.exists()
        if not fresh:
            if json.loads(resolved_path.read_text()) != resolved:
                raise ValueError("Resolved configuration changed; use a new run ID")
        else:
            if any(directory.glob("*.jsonl")):
                raise ValueError("Raw artifacts exist without resolved configuration")
            write_once(resolved_path, resolved)
        attempts_path = directory / "attempts.jsonl"
        trajectories_path = directory / "trajectories.jsonl"
        results_path = directory / "results.jsonl"
        attempts = _index(read_rows(attempts_path, Attempt), config, manifest)
        trajectories = _index(read_rows(trajectories_path, AgentTrajectory), config, manifest)
        results = _index(read_rows(results_path, TaskResult), config, manifest)
        if not results.keys() <= trajectories.keys() <= attempts.keys():
            raise ValueError("Artifacts have missing attempt/trajectory provenance")
        if len({t.trajectory_id for t in trajectories.values()}) != len(trajectories):
            raise ValueError("Duplicate trajectory ID")
        policy = None
        if config.memory is not None:
            policy = EpisodicPolicy(
                config.memory, config.run_id, directory, manifest.task_ids, fresh=fresh,
                embedder=embedder_factory(), llm_factory=llm_factory,
                count_tokens=count_tokens or token_counter(config.memory.tokenizer),
            )
            if not policy.retrievals.keys() <= attempts.keys():
                raise ValueError("Retrieval logged without an attempt")
        retrievals = policy.retrievals if policy else {}
        for task_id, trajectory in trajectories.items():
            if (trajectory.agent_config != config.agent
                or trajectory.environment_config != config.environment
                or trajectory.environment_backend_sha256 != resolved["environment_backend_sha256"]):
                raise ValueError("Stored trajectory configuration mismatch")
            if task_id in results:
                if results[task_id] != task_result(trajectory, resolved, retrievals.get(task_id)):
                    raise ValueError("Stored task result differs from canonical trajectory")
                if policy:
                    policy.check_complete(trajectory)
        for task_id in manifest.task_ids:
            if task_id in results:
                continue
            recorder = CheckedRecorder(trajectories_path, task_id, config, resolved)
            if task_id in trajectories:
                trajectory = trajectories[task_id]  # Crash after trajectory commit.
            elif task_id in attempts:
                # Never repeat a potentially billable call or outcome-selectively retry.
                trajectory = failure_trajectory(config, resolved, task_id, "interrupted")
                recorder.append(trajectory)
            else:
                append_row(attempts_path, Attempt(task_id=task_id, run_id=config.run_id,
                                                  started_at=utc_now()))
                env = llm = None
                started = perf_counter()
                try:
                    env = env_factory(config.environment, task_id=task_id)
                    llm = llm_factory()
                except Exception as exc:
                    trajectory = failure_trajectory(config, resolved, task_id, "setup_error",
                                                    perf_counter() - started, type(exc).__name__)
                    recorder.append(trajectory)
                else:
                    # Use an in-memory recorder so storage errors cannot be mislabeled as
                    # acting failures or cause a second append after a partial write.
                    class Capture:
                        def append(self, value):
                            pass
                    try:
                        trajectory = run_episode(
                            env, llm, config.agent, Capture(), run_id=config.run_id,
                            retrieve_memory=policy.retriever(task_id) if policy else None)
                    except Exception as exc:
                        trajectory = failure_trajectory(config, resolved, task_id, "runtime_error",
                                                        perf_counter() - started, type(exc).__name__)
                    recorder.append(trajectory)
                finally:
                    for resource in (env, llm):
                        if resource is not None:
                            resource.close()
            if policy:
                policy.record_episode(trajectory)  # Finishes before the result commits.
            result = task_result(trajectory, resolved, retrievals.get(task_id))
            append_row(results_path, result)
            results[task_id] = result
            print(f"{len(results)}/{len(manifest.task_ids)} {result.outcome} "
                  f"steps={result.steps} task={task_id}", flush=True)
        metrics = aggregate(list(results.values()))
        metrics.update(manifest_id=manifest.manifest_id,
                       resolved_config_sha256=digest(resolved), run_id=config.run_id)
        if policy:
            metrics["memory"] = policy.metrics()
            policy.close()
            policy.embedder.close()
        metrics_path = directory / "metrics.json"
        if metrics_path.exists():
            if json.loads(metrics_path.read_text()) != metrics:
                raise ValueError("Stored metrics disagree with raw results")
        else:
            write_once(metrics_path, metrics)
        return metrics
