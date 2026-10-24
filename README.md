# adaptive-experience-compression

Learning What to Remember: Adaptive Experience Compression for LLM Agents

Given a completed agent trajectory, should the experience be discarded, stored as an episodic memory, or consolidated into a procedural skill?

## Phase 0 development

Requires Python 3.11 or newer.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/aec --config configs/default.yaml
```

The CLI validates YAML configuration and initializes console logging. It does not
run experiments. The same entry point is available as `python -m aec`.

Schemas live in `src/aec/schemas` and support Pydantic's `model_dump_json()` and
`model_validate_json()`. They reject unknown fields and require source trajectory
IDs for episodes and skills. `TokenUsage` records prompt and completion counts;
trajectory-level counts are totals, while step-level counts describe that step.
Standard logging is for diagnostics; experimental JSONL recording and persistence
will be added in their designated phases.

## Phase 1: text environment

Install the optional text dependency in your virtual environment:

```sh
.venv/bin/python -m pip install -e '.[dev,alfworld]'
```

ALFWorld task data must be provisioned separately. Set `ALFWORLD_DATA` to its
root (containing `json_2.1.1` and generated `game.tw-pddl` files), or edit the
paths in `configs/alfworld_backend.yaml`. No THOR or visual extras are needed.
From the repository root, with a fresh output filename:

```sh
.venv/bin/python scripts/smoke_alfworld.py --episodes 3 --output /tmp/alfworld-smoke.jsonl
```

The seeded policy samples sorted admissible actions, falling back to `look` when
none are available. JSONL contains reset observations and per-action observations,
rewards, task IDs, success (when supplied), done flags, and termination reasons.
Existing output files are rejected. This is a smoke check, not a success baseline.

`ALFWorldAdapter` exposes `reset()`, `step(action)`, `valid_actions()`,
`current_task()`, and `close()`. Reset returns a typed task and initial observation;
step returns a `TrajectoryStep` subtype with the observation **after** the action,
task ID, success flag, and termination reason (`environment` or `max_steps`, or
`None` while active). Rewards preserve the backend score unchanged; Phase 2 must
not assume they are incremental rewards. Backend termination takes precedence
when it coincides with the configured step limit. Reset is required after either
termination. Task IDs preserve `extra.gamefile` verbatim; metadata records split,
gamefile, and the family inferred from the standard dataset directory layout.
File paths must remain consistent when comparing runs across machines.

Configuration uses the existing YAML loader with an optional `environment` section
(see `configs/alfworld_smoke.yaml`). Paths are relative to the working directory.
The adapter uses batch size one, disables domain randomization, seeds TextWorld,
and selects the upstream non-expert wrapper path. Its backend timeout is one step
beyond the adapter limit, making adapter timeouts distinguishable. It never
consumes expert plans. Always close the adapter to release backend processes.

Automated tests use an injected batch backend and require no ALFWorld installation
or downloads. The integration targets ALFWorld 0.4's batched text API. Real runtime
compatibility on this Codespace's Python 3.14 remains unverified; a compatible
ALFWorld/TextWorld installation and task data are required before Phase 2 runs.
Upstream API references:
[ALFWorld text adapter](https://github.com/alfworld/alfworld/blob/master/alfworld/agents/environment/alfred_tw_env.py)
and [installation/data instructions](https://github.com/alfworld/alfworld#quickstart).
