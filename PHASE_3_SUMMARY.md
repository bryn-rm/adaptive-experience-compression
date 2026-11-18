# Phase 3 completion summary

Implemented the no-memory baseline pipeline. **The engineering work is complete; the live pilot fails the baseline suitability gate.** Phase 4 was not started.

## Runtime and live-stack validation

- Date: 2026-10-25.
- Live environment: `.venv-alfworld`, system Python **3.12.3** (`/usr/bin/python3.12`). Existing `.venv` uses Python 3.14.2.
- ALFWorld **0.4.2**, TextWorld **1.7.0**, OpenAI SDK **2.54.0**. Exact installed versions are in `configs/phase3-pilot-requirements.txt` and the resolved run artifact.
- The system lacked Python 3.12 `ensurepip`; the existing pip provisioned the separate Python 3.12 venv. No upstream libraries were patched.
- Official task JSON and generated text-game archives were extracted under `data/alfworld`; this downloaded data and the venv are ignored by Git.
- Real preflight **passed**: ALFWorld reset → `agent_v1` prompt → OpenAI `gpt-4o-mini-2024-07-18` → strict action parsing → ALFWorld step (`go to fridge 1`). Evidence: `data/results/phase3-preflight.json`. This single-cycle check is not included in baseline metrics.
- API calls required network access outside the execution sandbox. Credentials were read from the configured environment and were not stored in artifacts.

## Real ALFWorld diagnostic pilot

Run: `phase3-pilot-20`, split: `eval_out_of_distribution` (`valid_unseen`). Twenty tasks were sampled once without replacement from 134 upstream eligible tasks, using sorted IDs and selection seed 0. The fixed manifest covers all six task families. No tasks were replaced or selected based on outcomes.

| Metric | Result |
|---|---:|
| Attempted tasks | 20 |
| Completed tasks | 14 |
| Successful tasks | 0 |
| Completed environment failures | 14 |
| Aborted tasks | 6 |
| Parse failures | 6 |
| Setup/provider/runtime failures | 0 / 0 / 0 |
| Success rate over completed tasks | 0% (0/14) |
| Abort rate | 30% (6/20) |
| Mean / median executed steps, completed tasks | 50 / 50 |
| Mean terminal cumulative reward, completed tasks | 0 |
| Model calls / confirmed executed actions | 794 / 788 |
| Prompt tokens | 330,791 |
| Completion tokens | 5,895 |
| Token usage available | All 794 calls; 20/20 task totals known |
| Sum of recorded trajectory wall time | 591.371 seconds |

All 14 completed environment failures reached the configured 50-step limit. Aborts are excluded from the completed-task success denominator. All six aborts were strict parsing failures: quoted actions or commands absent from the recorded admissible-action list. No fallback action was executed. The per-call parse-failure rate is approximately 0.76%, but the task-level abort rate is an unacceptable 30% for a clean baseline.

Wall time includes reset and acting within each recorded trajectory, excluding backend/client construction, teardown, artifact writes, and the separate preflight. Rewards are last cumulative environment scores, never summed step scores. Token counts are provider-reported observed usage, with the existing `usage_known` semantics preserved.

## Inspection and validity gate

All 20 diagnostic trajectories were inspected; notes with source trajectory IDs are in `data/results/phase3-pilot-20/inspection.md`.

Repeated structures include container opening, object taking/placing, navigation between locations, and heating/light actions. Most repetition is pathological: the agent manipulates an incorrect object, undoes its placement, revisits the same locations, or repeats an appliance action. There are no successful trajectories from which to build a successful-episode memory baseline.

**The frozen `gpt-4o-mini-2024-07-18` + `agent_v1` setup is unsuitable as the current benchmark baseline.** Success is exactly 0%, not a meaningful intermediate fraction, and task aborts are high. This diagnostic does not identify the model alone as the cause: the existing prompt deliberately carries no interaction history, which plausibly contributes to loops in a partially observable environment. No demonstrated pipeline defect justified changing the acting behavior, so the prompt, parser, model, decoding settings, and trajectory semantics were preserved.

Resolve benchmark/model/acting-setup suitability explicitly before later policy comparisons, then establish a new frozen baseline under a new run ID. Retain this pilot as diagnostic data. Do not present it as final thesis data or claim that it demonstrates memory benefit.

## Implemented behavior and provenance

- Frozen, validated task manifests with unique ordered IDs, split, creation/version metadata, selection provenance, and per-task SHA-256 hashes; manifest publication never overwrites an existing file.
- Exact task targeting in the thin ALFWorld adapter, with reset identity checks and unchanged ordinary acting behavior.
- One durable attempt record, canonical `AgentTrajectory`, and task-level result per attempted task/run pair. Separate outcomes for success, environment failure, abort, and setup/provider/runtime failure.
- Resolved artifacts record the full manifest/config, model/prompt/decoding/provider seed, environment seed, effective backend settings and hash, task hashes, Python/dependency versions, and source hash.
- Append-only raw JSONL, immutable JSON publication, single-writer locking, drift/corruption rejection, and safe resume. Finished pairs—including aborts/failures—are skipped. A committed trajectory can recover its missing result without re-execution. A started attempt lacking a trajectory becomes an interrupted abort rather than being retried.
- Unknown provider usage is never interpreted as known zero cost. Aggregate token sums become lower bounds when any usage is unknown. Aborted rewards are excluded from environment metrics.
- Fake execution requires explicit `fake_verification` labeling; no mock/scripted outcomes were used as real baseline performance.
- The existing end-of-episode recorder cannot recover in-flight observations after a hard interruption. Such attempts receive an explicit zero-step interrupted artifact with unknown usage/time, not invented interactions. Malformed partial raw files require inspection and are never automatically truncated.

Manifest ID: `alfworld-26a02eb5d85c76bb8171d23d438b793fabda50667afb9cdcdd220a94265481a5`.

Resolved-config SHA-256: `bb32125b35a7de3c267c1158e39de7f7a2a5765e50c08796bfd1a90447c7572b`.

The finished real pilot was resumed successfully in the network-restricted sandbox. All canonical JSON/JSONL hashes were unchanged and no new attempt was written. Evidence: `data/results/phase3-pilot-20/resume_verification.json`.

## Verification

- `.venv-alfworld/bin/python -m pytest -q`: **106 passed**, Python 3.12.3.
- `.venv/bin/python -m pytest -q`: **106 passed**, Python 3.14.2.
- Tests require no API credentials or ALFWorld downloads. They cover manifest validation, deterministic fixed selection, exact environment task targeting, result creation, resume/skip/recovery, append-only writes, corruption/config/task drift, writer locking, abort/environment-failure separation, metrics, unknown/partial usage, and deterministic fake execution.
- Both script CLIs passed `--help`; `git diff --check` passed.
- Git diff confirms no changes to `src/aec/agents`, `src/aec/llm`, `src/aec/trajectories`, or `src/aec/config.py`. Pilot agent and environment settings equal the Phase 2 smoke configuration.

## Changed and added files

- `.gitignore`, `README.md`, `PHASE_3_SUMMARY.md`.
- `src/aec/environments/alfworld.py`.
- `src/aec/experiments/__init__.py`, `artifacts.py`, `baseline.py`.
- `scripts/run_baseline.py`, `scripts/validate_live_stack.py`.
- `configs/baseline_pilot.yaml`, `configs/phase3-pilot-requirements.txt`.
- `tests/test_alfworld.py`, `tests/test_baseline.py`.
- `data/manifests/phase3-pilot-20.json`.
- `data/results/phase3-preflight.json`.
- `data/results/phase3-pilot-20/`: `resolved_config.json`, `attempts.jsonl`, `trajectories.jsonl`, `results.jsonl`, `metrics.json`, `inspection.md`, `resume_verification.json`.

No episodic memory, skills, retrieval, oracle, controller, or Phase 4 functionality was added.
