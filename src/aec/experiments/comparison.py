"""Descriptive paired outcomes; never pool materially different acting setups."""

import json
from pathlib import Path

from aec.config import ALFWorldConfig
from aec.experiments.artifacts import RetrievalLog, TaskManifest, TaskResult, digest, read_rows
from aec.experiments.baseline import aggregate, task_result
from aec.trajectories.schemas import AgentTrajectory


def _load_run(directory: Path):
    resolved = json.loads((directory / 'resolved_config.json').read_text())
    manifest = TaskManifest.model_validate(resolved['manifest'])
    results = read_rows(directory / 'results.jsonl', TaskResult)
    trajectories = read_rows(directory / 'trajectories.jsonl', AgentTrajectory)
    if tuple(r.task_id for r in results) != manifest.task_ids or (
        tuple(t.task_id for t in trajectories) != manifest.task_ids
    ):
        raise ValueError('Comparison requires every manifest task exactly once, in fixed order')
    if resolved['manifest_sha256'] != digest(manifest.model_dump(mode='json')):
        raise ValueError('Manifest hash mismatch')
    retrievals = {r.task_id: r for r in read_rows(directory / 'retrievals.jsonl', RetrievalLog)}
    for result, trajectory in zip(results, trajectories):
        if result != task_result(trajectory, resolved, retrievals.get(result.task_id)):
            raise ValueError('Result does not match its canonical trajectory')
    return resolved, manifest, results


def compare_runs(previous: Path, candidate: Path) -> dict:
    before, manifest_before, old = _load_run(previous)
    after, manifest_after, new = _load_run(candidate)
    if manifest_before != manifest_after:
        raise ValueError('Comparison requires the same immutable task manifest')
    if ALFWorldConfig.model_validate(before['config']['environment']) != (
        ALFWorldConfig.model_validate(after['config']['environment'])
    ) or any(before[key] != after[key] for key in (
        'environment_backend_sha256', 'environment_backend'
    )):
        raise ValueError('Environment settings differ')
    if before['config']['data_kind'] != after['config']['data_kind']:
        raise ValueError('Cannot compare real performance to fake verification')
    if before['config']['agent'] == after['config']['agent']:
        interpretation = ('Descriptive paired comparison of memory policies with an identical acting '
                          'setup. Runs executed separately (provider drift possible), one seed: '
                          'not a causal estimate of memory benefit.')
    else:
        interpretation = ('Descriptive paired diagnostic comparison of materially different acting '
                          'setups; not a causal single-variable ablation. Do not pool these baseline '
                          'results.')
    return dict(
        interpretation=interpretation,
        previous_policy=before['policy'], candidate_policy=after['policy'],
        manifest_id=manifest_before.manifest_id,
        manifest_sha256=before['manifest_sha256'],
        previous_config_sha256=digest(before), candidate_config_sha256=digest(after),
        previous_agent=before['config']['agent'], candidate_agent=after['config']['agent'],
        previous_metrics=aggregate(old), candidate_metrics=aggregate(new),
        tasks=[dict(task_id=a.task_id, previous_run_id=a.run_id, candidate_run_id=b.run_id,
                    previous_trajectory_id=a.trajectory_id, candidate_trajectory_id=b.trajectory_id,
                    previous_outcome=a.outcome, candidate_outcome=b.outcome,
                    previous_steps=a.steps, candidate_steps=b.steps,
                    previous_parse_failures=a.parse_failures, candidate_parse_failures=b.parse_failures,
                    candidate_normalized_actions=b.normalized_actions,
                    candidate_retrieved_memory_ids=b.retrieved_memory_ids)
               for a, b in zip(old, new)],
    )
