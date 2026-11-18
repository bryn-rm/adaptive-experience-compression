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

## Phase 3: reproducible no-memory baseline

The live stack was validated with Python **3.12.3**, ALFWorld **0.4.2**,
TextWorld **1.7.0**, and OpenAI SDK **2.54.0**. See `PHASE_3_SUMMARY.md` for
real pilot results and the suitability gate. The Phase 1/2 notes above describe
what was unverified at those earlier phases. `agent_v1`, parsing, model settings,
and the acting loop remain unchanged.

Create a Python 3.12 environment and install the recorded dependencies:

```sh
python3.12 -m venv .venv-alfworld
.venv-alfworld/bin/python -m pip install -r configs/phase3-pilot-requirements.txt
.venv-alfworld/bin/python -m pip install -e '.[dev,alfworld,openai]'
```

If the system lacks `ensurepip`, an existing pip can provision a venv created
with `python3.12 -m venv --without-pip .venv-alfworld`:
` .venv/bin/python -m pip --python .venv-alfworld/bin/python install -r configs/phase3-pilot-requirements.txt`,
followed by the same pip invocation with `install -e '.[dev,alfworld,openai]'`.
No upstream code was patched. No visual/THOR components are needed.

Download and extract these official archives into `data/alfworld`:

- [ALFWorld task JSON](https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip)
- [Generated text games](https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip)

Both archives contain `json_2.1.1/...` paths. Set `ALFWORLD_DATA` to the **absolute**
path of `data/alfworld` and configure `OPENAI_API_KEY` in the environment. Keep
those task paths stable across policy comparisons; canonical task IDs retain the
existing gamefile semantics. Credentials are never written into artifacts.

From the repository root:

```sh
export ALFWORLD_DATA="$PWD/data/alfworld"
# Optional live preflight, with a NEW output path (one model call):
.venv-alfworld/bin/python scripts/validate_live_stack.py --output /tmp/new-preflight.json
# Only for a NEW manifest path; an existing manifest is never overwritten:
.venv-alfworld/bin/python scripts/run_baseline.py manifest --config configs/baseline_pilot.yaml
# Execute, or safely resume the SAME run:
.venv-alfworld/bin/python scripts/run_baseline.py run --config configs/baseline_pilot.yaml
.venv-alfworld/bin/python -m pytest -q
```

The included pilot manifest is already fixed; do not recreate it. For a separate
experiment, copy the YAML, set a new run ID, and explicitly choose whether to reuse
the manifest or create a new one. The runner never creates/replaces/selects tasks
while acting. Manifest creation samples without replacement from sorted upstream
eligible game IDs using `selection_seed` and `selection_count`; selection happens
before outcomes. Upstream eligibility uses its distributed solvability metadata.
The immutable manifest contains IDs in execution order, split, creation/version
metadata, candidate-set hash, and per-game SHA-256 hashes. The adapter restricts
each fresh backend to exactly one requested game and checks its reset identity.

`data/results/<run_id>/` contains:

- `resolved_config.json`: frozen config, complete manifest and its hash, expanded
  backend settings and backend-file hash, Python/package versions, source-code
  hash. Nested agent settings include model ID, prompt version, decoding settings,
  provider seed, and timeout; environment settings include its separate seed.
- `attempts.jsonl`: durable task/run intents written before environment/provider
  construction; timestamps indicate when attempts began.
- `trajectories.jsonl`: one existing `AgentTrajectory` per attempted task, including
  aborted trajectories. Existing per-step response IDs/models/fingerprints remain.
- `results.jsonl`: one typed task result linked to the canonical trajectory,
  manifest, and resolved config by IDs/hashes.
- `metrics.json`: immutable aggregate derived from the final task results.

Resume validates all artifacts and rejects config, source, dependency, manifest,
backend, or task-content drift. It skips **all** committed task/run results,
including failures and aborts. A trajectory committed before its result is reused
without another call. A durable attempt with no trajectory becomes an `interrupted`
abort, with unknown usage/time; it is never automatically replayed. Setup/reset or
unexpected runtime exceptions without a durable episode produce a zero-step aborted
artifact explicitly marked as lacking observations. Their zero reward/success/usage
fields are placeholders; task results exclude them from environment scoring.
Any interactions lost in a hard interruption are unrecoverable with the unchanged
Phase 2 end-of-episode recorder. To retry, explicitly use a new run ID and retain
the original attempt for analysis.

A process lock prevents concurrent writers. JSONL is append-only and flushed to
disk. Immutable JSON files publish without replacement. Incomplete or malformed
raw files stop the run for inspection; no automatic truncation or repair occurs.
Storage failures propagate rather than becoming misleading environment outcomes.
This is a local single-writer runner, not a distributed execution system.

Metric definitions:

- `completed_tasks` includes only successful environment completion and completed
  environment failure (including the configured step limit with known success).
- `aborted_tasks` includes every other trajectory. Setup/provider/runtime failure
  counts are **subsets** of aborts, not additional tasks. `parse_failures` counts
  failed parses and performs no fallback action.
- `success_rate` is successes / completed tasks; `abort_rate` is aborts / attempted
  tasks. Undefined rates are JSON `null`.
- Mean/median steps count confirmed executed actions over completed tasks only.
  Mean reward uses the last cumulative environment score over completed tasks,
  never a sum of cumulative rewards and never an aborted reward placeholder.
- Prompt/completion tokens sum observed provider usage. When `usage_known=false`,
  these totals are **lower bounds**, not complete billing totals.
  `unknown_usage_tasks` reports how many task totals are incomplete. The existing
  provider rejects missing usage and the acting loop records a provider abort.
- Wall time sums recorded episode durations (reset and acting, excluding ordinary
  environment/client construction, teardown, artifact writes, and downtime).
  Setup/runtime failures record elapsed time until the exception. Interrupted
  attempts contribute zero with `wall_time_known=false`, making the sum incomplete.

Tests inject deterministic environments and LLMs and need no credentials or task
archive. Injected factories require `data_kind: fake_verification`; their outcomes
cannot be labeled as real ALFWorld data. There is no memory, retrieval, prompt
tuning, model fallback, or Phase 4 implementation in this pipeline.

## Phase 3b: acting setup calibration

`configs/baseline_calibration.yaml` defines a separate baseline candidate:
`agent_v2`, `gpt-5.6-terra`, low reasoning effort, and a four-pair within-episode
history window. It reuses the **exact** Phase 3 manifest, task order, environment
settings, scoring, seeds, and 50-step limit. This is a baseline redesign, not a
single-variable causal ablation. Phase 3 diagnostic data remains unchanged and
must not be pooled with the new candidate. See `PHASE_3B_SUMMARY.md` for outcomes
and the suitability decision; Phase 4 requires a separate decision.

`agent_v2` adds the most recent `history_length` completed observation/action
pairs, oldest first, to the original task, current observation, and admissible
actions. Each pair contains the observation **before** the confirmed action and
the action itself. The following pair/current observation supplies subsequent
feedback. Only executed steps enter history. Raw model output, provider reasoning,
and prior prompts are not copied into history. The window resets for each task;
there is no persistent cross-task memory or provider conversation state.

The v2 parser trims outer whitespace, removes at most one exact `Action: ` prefix,
and removes at most one matching pair of straight single/double quotes. It then
requires exact membership in the current admissible actions. It does not accept
case changes, fuzzy matches, multiline/code-fenced responses, explanatory suffixes,
recursive wrappers, fallback actions, or action retries. `action_normalized` on
each step records whether an accepted action differs from the public response,
including whitespace changes. `normalized_actions` counts these accepted actions
in task results and metrics. `task_parse_failures` counts affected tasks separately
from individual parse events. The original `agent_v1` prompt and strict parser
remain intact; old artifacts load with normalization flags defaulting to false.

The [model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-terra)
lists low reasoning effort. The existing Chat Completions adapter sends it as
`reasoning_effort`, and continues to retain only public text and usage metadata.
The live API rejected temperature 0 for this model; the calibration config
explicitly sets `temperature: null` to omit it and use the provider default.
The other retained sampling settings are `top_p: 1` and `seed: 0`. The completion
budget is 2,048 tokens to accommodate the reasoning model's internal tokens and
public action, instead of the old non-reasoning model's 128-token cap. These
model-interface adjustments are recorded as part of the redesign. There is no
silent parameter/model fallback. Completion usage includes the provider's full
completion-token accounting; no hidden reasoning content is requested or stored.

Run from the repository root with the existing Python 3.12 environment and
credentials. Do **not** regenerate the manifest:

```sh
export ALFWORLD_DATA="$PWD/data/alfworld"
# Optional one-cycle check; choose a new output path:
.venv-alfworld/bin/python scripts/validate_live_stack.py --baseline-config --config configs/baseline_calibration.yaml --output /tmp/phase3b-new-preflight.json
# Execute or resume the new candidate only:
.venv-alfworld/bin/python scripts/run_baseline.py run --config configs/baseline_calibration.yaml
# After both runs finish; output must not already exist:
.venv-alfworld/bin/python scripts/compare_baselines.py --previous data/results/phase3-pilot-20 --candidate data/results/phase3b-pilot-20 --output /tmp/phase3b-new-comparison.json
```

The comparison validates complete one-to-one task order, identical manifests and
environment settings, and canonical result/trajectory consistency. It reports
separate aggregates and paired task outcomes with both source trajectory IDs.
It explicitly labels the comparison as descriptive and non-causal. The old run's
source/config hashes intentionally differ after this redesign; the old run is
preserved for read-only comparison, not resumed using changed acting source code.

## Phase 4: episodic-memory baseline

`configs/episodic_pilot.yaml` adds a `memory` section to the frozen Phase 3b setup.
Its manifest, environment, and acting agent (`agent_v2`, model, decoding, history)
are identical to `configs/baseline_calibration.yaml`; a test enforces this. A config
without `memory` is the no-memory policy. Both policies use the same runner
(`scripts/run_baseline.py`) and the same `run_episode` acting loop.

Install the memory tokenizer (`configs/phase4-pilot-requirements.txt` records the
exact environment), then probe provider compatibility and run:

```sh
.venv-alfworld/bin/python -m pip install -e '.[dev,alfworld,openai,memory]'
export ALFWORLD_DATA="$PWD/data/alfworld"
# Optional; one embedding call and one writer call; choose a NEW output path:
.venv-alfworld/bin/python scripts/probe_memory_stack.py --source data/results/phase3b-pilot-20/trajectories.jsonl --output /tmp/new-probe.json
.venv-alfworld/bin/python scripts/run_baseline.py run --config configs/episodic_pilot.yaml
.venv-alfworld/bin/python scripts/compare_baselines.py --previous data/results/phase3b-pilot-20 --candidate data/results/phase4-episodic-20 --output /tmp/new-comparison.json
```

Online stream design: tasks run in fixed manifest order. For each task, after
reset and before the first action, the goal line (`Your task is to: ...`,
`retrieval_key: task_goal`) is embedded with the fixed `text-embedding-3-small`
model and the top-k stored episodes are retrieved by brute-force cosine similarity
(ties broken by insertion order). Episodes derived from the current task ID are
excluded. When nothing is retrievable no query call is made. Retrieved episodes
are inserted as one fixed-format line (`memory_section_v1`) directly after the
task line. **With zero retrieved episodes the prompt is byte-identical to agent_v2;**
all 442 recorded Phase 3b prompts were reproduced exactly by the modified builder.

After a completed, environment-confirmed **successful** trajectory, the
`episode_writer_v1` prompt receives only observable data (task, initial observation,
executed actions and resulting observations) and must return strict JSON with
`task_summary`, `experience`, and `outcome`. Malformed output is rejected and stored
nowhere; there is no repair or retry. The episode's `embedding_text` is the source
task's goal line, never writer output. `memory_id` is derived from, and
`source_trajectory_ids` contains, the source trajectory ID. Episode `token_count`
uses the fixed local `o200k_base` tokenizer on the rendered episode.

Additional files in `data/results/<run_id>/`:

- `retrievals.jsonl`: one decision-time row per task, written before acting: query,
  store size, eligible count, memory IDs, similarities, ranks, source trajectory
  IDs, the exact injected section, its token count, and query-embedding usage.
  `results.jsonl` repeats `retrieved_memory_ids` alongside each task outcome.
- `memory_writes.jsonl`: `started` then one terminal row (`stored`, `rejected`,
  `provider_failure`, `embedding_failure`, `interrupted`) per successful trajectory,
  with writer prompt/output/usage; `stored` rows carry the episode and its vector.
- `metrics.json` gains a `memory` block: retrieval counts, injected memory tokens,
  writer and embedding tokens (kept separate from acting tokens), write outcomes,
  stored episodes, and stored memory tokens.

The SQLite store (`memory.store_path`) persists across processes and runs and can
be opened read-only by later phases. It is bound to one embedding configuration and
one writing run ID; a fresh run refuses a pre-existing store. The append-only
`memory_writes.jsonl` is its source of truth: every start verifies the store
against it, reinserts a row lost between log append and store insert, and rejects
any other difference. A writer call interrupted mid-flight is logged as
`interrupted` and never replayed. A task result commits only after its episode
write finishes, so stream order is preserved on resume.
