"""Gate A: does memory change behaviour or outcomes? Descriptive, paired, one seed.

Pairs no-memory, episodic, and skill runs over one immutable manifest and one acting
setup. An optional second no-memory run gives the run-to-run noise floor: the acting
model samples, so action differences alone do not show that memory caused them.
Incomparable runs (different tasks, environment, agent, or data kind) are rejected.
Weaker controls (same code, dependencies, and time window) are reported, not assumed.
"""

from datetime import datetime
from difflib import SequenceMatcher
import json
from math import comb
from pathlib import Path
from statistics import mean, median

from aec.config import ALFWorldConfig
from aec.experiments.artifacts import Attempt, read_rows
from aec.experiments.baseline import aggregate
from aec.experiments.comparison import load_run

POLICIES = {"no_memory": "no_memory", "episodic": "episodic", "skill": "skill",
            "no_memory_replicate": "no_memory"}
SHARED_RETRIEVAL = ("embedding", "top_k", "retrieval_key", "exclude_same_task", "tokenizer")
COMPARISONS = (("no_memory", "episodic"), ("no_memory", "skill"), ("episodic", "skill"),
               ("no_memory", "no_memory_replicate"))


def sign_test(reference_only: int, candidate_only: int) -> float | None:
    """Exact two-sided McNemar (binomial) p-value over discordant pairs."""
    n = reference_only + candidate_only
    if n == 0:
        return None
    tail = sum(comb(n, i) for i in range(min(reference_only, candidate_only) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def failure_mode(result) -> str | None:
    return None if result.outcome == "success" else f"{result.outcome}:{result.termination_reason}"


def executed_actions(trajectory) -> list[str]:
    return [s.action for s in trajectory.steps if s.executed]


def divergence(reference: list[str], candidate: list[str]) -> dict:
    first = next((i for i, (a, b) in enumerate(zip(reference, candidate)) if a != b), None)
    if first is None and len(reference) != len(candidate):
        first = min(len(reference), len(candidate))
    return dict(identical_actions=reference == candidate, first_divergent_step=first,
                action_similarity=SequenceMatcher(None, reference, candidate, autojunk=False).ratio())


def _window(directory: Path) -> tuple[datetime, datetime]:
    started = [a.started_at for a in read_rows(directory / "attempts.jsonl", Attempt)]
    if not started:
        raise ValueError(f"No attempts recorded in {directory}")
    return min(started), max(started)


def _load(directory: Path) -> dict:
    resolved, manifest, results, trajectories, retrievals = load_run(directory)
    metrics = json.loads((directory / "metrics.json").read_text())
    summary = aggregate(results)
    if any(metrics[key] != value for key, value in summary.items()):
        raise ValueError(f"Stored metrics disagree with raw results in {directory}")
    first, last = _window(directory)
    return dict(resolved=resolved, manifest=manifest, results=results, trajectories=trajectories,
                retrievals=retrievals, metrics=metrics, first_attempt=first, last_attempt=last)


def _check_comparable(runs: dict[str, dict]) -> None:
    reference = runs["no_memory"]["resolved"]
    for label, run in runs.items():
        resolved = run["resolved"]
        if resolved["policy"] != POLICIES[label]:
            raise ValueError(f"{label} run has policy {resolved['policy']}")
        if run["manifest"] != runs["no_memory"]["manifest"]:
            raise ValueError("Gate A requires the same immutable task manifest")
        if (ALFWorldConfig.model_validate(resolved["config"]["environment"])
                != ALFWorldConfig.model_validate(reference["config"]["environment"])
                or any(resolved[k] != reference[k]
                       for k in ("environment_backend_sha256", "environment_backend"))):
            raise ValueError("Environment settings differ")
        if resolved["config"]["agent"] != reference["config"]["agent"]:
            raise ValueError("Acting agent settings differ")
        if resolved["config"]["data_kind"] != reference["config"]["data_kind"]:
            raise ValueError("Cannot compare real performance to fake verification")
    if len({run["resolved"]["config"]["run_id"] for run in runs.values()}) != len(runs):
        raise ValueError("Each policy needs its own run")
    episodic = runs["episodic"]["resolved"]["config"]["memory"]
    skill = runs["skill"]["resolved"]["config"]["memory"]
    if any(episodic[f] != skill[f] for f in SHARED_RETRIEVAL):
        raise ValueError("Episodes and skills must share retrieval machinery")
    if episodic["writer"]["llm"] != skill["writer"]["llm"]:
        raise ValueError("Episodes and skills must share the writer LLM")
    if episodic["store_path"] == skill["store_path"]:
        raise ValueError("Episodic and skill runs must use separate stores")


def _controls(runs: dict[str, dict], max_span_hours: float) -> dict:
    def same(key):
        return len({json.dumps(r["resolved"][key], sort_keys=True) for r in runs.values()}) == 1

    start = min(r["first_attempt"] for r in runs.values())
    end = max(r["last_attempt"] for r in runs.values())
    span = (end - start).total_seconds() / 3600
    checks = dict(same_source_sha256=same("source_sha256"), same_python=same("python"),
                  same_packages=same("packages"), within_time_window=span <= max_span_hours)
    return dict(**checks, controlled=all(checks.values()), max_span_hours=max_span_hours,
                attempt_span_hours=span,
                windows={label: dict(run_id=r["resolved"]["config"]["run_id"],
                                     first_attempt=r["first_attempt"].isoformat(),
                                     last_attempt=r["last_attempt"].isoformat())
                         for label, r in runs.items()})


def _paired(reference: dict, candidate: dict, labels: tuple[str, str]) -> dict:
    pairs = list(zip(reference["results"], reference["trajectories"],
                     candidate["results"], candidate["trajectories"]))
    rows, both_steps = [], []
    for a, ta, b, tb in pairs:
        rows.append(dict(task_id=a.task_id, reference_outcome=a.outcome,
                         candidate_outcome=b.outcome, reference_failure_mode=failure_mode(a),
                         candidate_failure_mode=failure_mode(b),
                         reference_trajectory_id=a.trajectory_id,
                         candidate_trajectory_id=b.trajectory_id,
                         reference_steps=a.steps, candidate_steps=b.steps,
                         candidate_retrieved_memory_ids=b.retrieved_memory_ids,
                         **divergence(executed_actions(ta), executed_actions(tb))))
        if a.outcome == b.outcome == "success":
            both_steps.append(b.steps - a.steps)

    def select(reference_success, candidate_success):
        return [r for r in rows if (r["reference_outcome"] == "success") == reference_success
                and (r["candidate_outcome"] == "success") == candidate_success]

    lost, gained = select(True, False), select(False, True)
    n = len(rows)
    return dict(
        reference=labels[0], candidate=labels[1], tasks=n,
        both_success=len(select(True, True)), neither_success=len(select(False, False)),
        reference_only_success=len(lost), candidate_only_success=len(gained),
        # Build guide NTR when the reference is no memory: candidate fails, reference succeeds.
        negative_transfer_rate=len(lost) / n, positive_transfer_rate=len(gained) / n,
        sign_test_p=sign_test(len(lost), len(gained)),
        failure_mode_changes=sum(r["reference_failure_mode"] != r["candidate_failure_mode"]
                                 for r in rows),
        identical_action_sequences=sum(r["identical_actions"] for r in rows),
        mean_action_similarity=mean(r["action_similarity"] for r in rows),
        both_success_step_difference=dict(
            tasks=len(both_steps),
            mean=mean(both_steps) if both_steps else None,
            median=median(both_steps) if both_steps else None),
        reference_only_success_tasks=lost, candidate_only_success_tasks=gained,
        task_rows=rows,
    )


def gate_a_report(no_memory: Path, episodic: Path, skill: Path,
                  no_memory_replicate: Path | None = None, *, max_span_hours: float = 24) -> dict:
    paths = dict(no_memory=no_memory, episodic=episodic, skill=skill)
    if no_memory_replicate is not None:
        paths["no_memory_replicate"] = no_memory_replicate
    runs = {label: _load(Path(path)) for label, path in paths.items()}
    _check_comparable(runs)
    controls = _controls(runs, max_span_hours)
    n = len(runs["no_memory"]["results"])
    tasks = []
    for i, task_id in enumerate(runs["no_memory"]["manifest"].task_ids):
        trajectory = runs["no_memory"]["trajectories"][i]
        row = dict(index=i + 1, task_id=task_id,
                   task_family=trajectory.task.metadata.get("task_family"))
        for label, run in runs.items():
            result = run["results"][i]
            row[label] = dict(trajectory_id=result.trajectory_id, outcome=result.outcome,
                              failure_mode=failure_mode(result), steps=result.steps,
                              retrieved_memory_ids=result.retrieved_memory_ids)
        tasks.append(row)
    return dict(
        version="gate_a_v1",
        interpretation=(
            f"Descriptive paired comparison over {n} tasks, one seed, online memory stream "
            "(early tasks see little or no memory). Exact sign tests are exploratory. "
            + ("Runs share code, dependencies, and time window."
               if controls["controlled"] else
               "Runs are NOT fully controlled (see controls); do not sign off Gate A on them.")
            + (" The no-memory replicate gives the run-to-run noise floor."
               if "no_memory_replicate" in runs else
               " No no-memory replicate: action differences cannot be separated from sampling noise.")),
        manifest_id=runs["no_memory"]["manifest"].manifest_id,
        manifest_sha256=runs["no_memory"]["resolved"]["manifest_sha256"],
        agent=runs["no_memory"]["resolved"]["config"]["agent"],
        controls=controls,
        metrics={label: run["metrics"] for label, run in runs.items()},
        comparisons=[_paired(runs[a], runs[b], (a, b)) for a, b in COMPARISONS
                     if a in runs and b in runs],
        tasks=tasks,
    )
