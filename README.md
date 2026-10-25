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

## Phase 2: frozen agent and canonical trajectories

Install the provider and text extras, provision ALFWorld text data as above, and
set `OPENAI_API_KEY` in the environment. Credentials are never config fields or
artifact fields. Run from the repository root:

```sh
.venv/bin/python -m pip install -e '.[dev,alfworld,openai]'
.venv/bin/python scripts/smoke_agent.py --config configs/agent_smoke.yaml --episodes 10 --output /tmp/agent-trajectories.jsonl
```

`configs/agent_smoke.yaml` explicitly fixes a dated model ID, decoding settings,
provider seed, prompt version, environment seed, split, and step limit. The model
ID is an example requiring account access, not a validated ALFWorld baseline.
The provider uses OpenAI Chat Completions with no SDK retries or conversation
state. It records the returned model ID and system fingerprint when supplied.
Provider seeds offer best-effort reproducibility, not guaranteed identical output;
models must support the configured Chat Completions parameters. There is no
parameter fallback or model substitution. See the
[official API reference](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).
Real provider and ALFWorld execution on Python 3.14 remains unverified here.

`LLMClient.generate(prompt, config)` returns project-owned text and usage data.
`agent_v1` sends only the task description, current observation, sorted admissible
actions, and the required exact single-line action format. It requests no
reasoning and stores no provider reasoning fields. Change the prompt version if
this content changes. No previous-task or interaction-history context is added.

Parsing trims surrounding whitespace and requires an exact admissible-action
match. Malformed, refused, or truncated responses abort the episode, log a parse
error, preserve the public output, and execute no substituted action. Missing
admissible actions also abort; this is the single fixed acting setup. Provider
and environment errors after reset produce aborted artifacts with error types;
exception messages are excluded to avoid leaking credential-bearing diagnostics.
Initialization, reset, and output-write errors fail loudly rather than fabricate
a task identity or a successful write.

Each `AgentTrajectory` in the append-only JSONL extends the existing `Trajectory`
schema. It records task metadata, UUID trajectory ID, shared smoke-run ID,
settings, backend-config SHA-256, termination, monotonic elapsed seconds, token
usage, and steps. Load each line with
`aec.trajectories.schemas.AgentTrajectory.model_validate_json(line)`. The recorder
revalidates artifacts and refuses to append after an incomplete final line. It is
intended for one writer; concurrent execution is outside this phase.

Canonical step `observation` is the input to the LLM; `next_observation` is the
confirmed post-action environment observation. `prompt` and `output` preserve
observable request/public-response text. `executed` means the environment returned
a confirmed result; after an environment exception actual execution may be
unknown. On abort, step `done` terminates the acting loop;
`environment_termination_reason` separately records confirmed environment endings.
Indices start at zero. Step wall time includes generation and environment work;
episode wall time includes reset but excludes artifact writing.

`reward` and `total_reward` both mean the last confirmed cumulative TextWorld
score, **not** the sum of step scores. Token counts are summed from per-call
provider usage and also retained in the original nested `token_usage` schema.
An unsuccessful provider call may have unknown billable usage: its counts use
zero placeholders with `usage_known: false`, so an aborted artifact's token totals
must not be treated as complete cost accounting. `success_known: false` likewise
marks missing success evidence; `success: false` is then a placeholder. Separate
`status: aborted` artifacts from completed environment outcomes in later analysis.
Environment and provider seeds are recorded separately. Keep the backend config
and task paths stable during runs; the config hash permits later mismatch checks.

The offline test suite runs ten distinct scripted tasks, validates every artifact,
and covers successful episodes, environment failures, parse aborts, token/timing
metadata, prompt versioning, provider conversion, and append-only recording. This
smoke script adds no manifest, resumability, metrics pipeline, or Phase 3 baseline.
