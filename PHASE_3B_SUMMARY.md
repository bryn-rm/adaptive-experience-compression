# Phase 3b: Acting Setup Calibration

**Implemented and evaluated. Suitability classification: Suitable — the diagnostic gate passes.** The real candidate succeeded on **16/20 tasks (80%)**, with four environment failures and zero aborts. It provides 16 successful source trajectories spanning all six task families. Phase 4 was not started.

## Frozen candidate and preservation

- New run: `phase3b-pilot-20`; new config: `configs/baseline_calibration.yaml`; prompt version: `agent_v2`.
- Requested model: **`gpt-5.6-terra`**; reasoning effort: **low**. The actual returned model identifier was **`gpt-5.6-terra` on all 442 model calls**.
- Four most recent completed observation/action pairs, oldest first. Each observation is the input to its confirmed action. Only observable fields enter the window; no raw prior model output, hidden reasoning, prior prompts, or cross-task state is included.
- The original task, current observation, admissible actions, and action-only output instruction remain. The v1 prompt and strict parser files are byte-identical to their pre-calibration versions.
- Deterministic v2 normalization permits outer whitespace, one exact `Action: ` prefix, and one enclosing straight quote pair. The result must still exactly match a current admissible action. No fuzzy matching, action retry, invented/fallback action, or model fallback exists.
- All Phase 3 manifest/config/results/trajectories/metrics/inspection files checked by the preservation audit are byte-identical. The same manifest, split, task order, task-selection seed 0, environment seed 0, backend config/hash, and **50-step limit** were used.
- Environment: `.venv-alfworld`, **Python 3.12.3**, ALFWorld 0.4.2, TextWorld 1.7.0, OpenAI SDK 2.54.0. No dependency upgrade or upstream patch was needed.

Manifest ID: `alfworld-26a02eb5d85c76bb8171d23d438b793fabda50667afb9cdcdd220a94265481a5`.

Candidate resolved-config SHA-256: `8027806500a1760ff3a7b1127ba6664cc21df9c2883ba3d7e33e1ab3f7609303`.

## Provider compatibility and preflight

The first compatibility probe rejected `temperature=0` with HTTP 400 (`unsupported_value`), explicitly reporting that this model supports only its default temperature. The new config therefore records `temperature: null` and the adapter omits that argument. `top_p: 1`, provider seed 0, timeout 60 seconds, one completion, `store=False`, and no SDK retries remain explicit. Reasoning effort is translated to the Chat Completions `reasoning_effort` parameter only inside the adapter.

The completion budget is explicitly **2,048 tokens**, replacing the non-reasoning model's 128-token cap to accommodate internal reasoning and public action tokens. This is a model-interface adaptation included in the baseline redesign, not a silent fallback. Three live calls used more than 128 completion tokens; the maximum was 260. Token counts include provider-reported internal completion usage, but no hidden reasoning text was requested, persisted, or put in history.

The agent_v2 real preflight passed: reset → v2 prompt → `gpt-5.6-terra` call → parse → `go to fridge 1` → confirmed ALFWorld step. The compatibility probe and preflight are recorded separately in `data/results/phase3b-compatibility.json` and `data/results/phase3b-preflight.json` and excluded from pilot metrics. Low reasoning support is documented on the [official model page](https://developers.openai.com/api/docs/models/gpt-5.6-terra).

## Real ALFWorld results

| Metric | Phase 3 diagnostic | Phase 3b candidate |
|---|---:|---:|
| Attempted tasks | 20 | 20 |
| Completed tasks | 14 | 20 |
| Successful tasks | 0 | 16 |
| Completed environment failures | 14 | 4 |
| Aborted tasks | 6 | 0 |
| Task-level parse failures | 6 | 0 |
| Setup / provider / runtime failures | 0 / 0 / 0 | 0 / 0 / 0 |
| Success rate over completed tasks | 0% (0/14) | 80% (16/20) |
| Successes over attempted tasks | 0/20 | 16/20 |
| Abort rate | 30% | 0% |
| Mean / median executed steps over completed tasks | 50 / 50 | 22.1 / 18 |
| Mean cumulative reward over completed tasks | 0 | 0.8 |
| Prompt tokens | 330,791 | 272,446 |
| Completion tokens | 5,895 | 16,803 |
| Recorded trajectory wall time | 591.371 s | 551.465 s |
| Normalized actions | Not recorded under v1 | 0 |
| Token-usage availability | All calls known | All 442 calls known |

The candidate executed 442 actions in 442 model calls. Mean executed steps among its 16 successful tasks was 15.125. Every unsuccessful candidate task reached 50 steps. No parse failure, truncation, unavailable-usage event, or infrastructure failure occurred. All live outputs already matched their admissible actions exactly; normalization was exercised by offline tests, not by this pilot.

Step means/medians and rewards use completed tasks only; the old and new completed denominators differ. Wall time sums canonical episode durations (including reset/acting, excluding ordinary setup/teardown, artifact writes, and preflight). Internal reasoning makes completion-token counts different from visible action lengths; these two models' token totals are not interchangeable cost estimates.

## Paired task outcomes — descriptive, not causal

**This is a material acting-setup redesign, not a one-variable ablation.** Model/reasoning behavior, within-episode history, and parsing changed together, with documented model-interface parameter adjustments. Differences cannot be attributed to any single component. Old and new results are stored and aggregated separately.

Of the 14 old environment failures, 11 became successes and three remained failures. Of the six old aborts, five became successes and one became a completed environment failure. The full validated comparison, including exact task IDs, both trajectory IDs, and config hashes, is `data/results/phase3b-pilot-20/comparison.json`.

| Manifest row / task | Phase 3 outcome | Phase 3b outcome | Old → new executed steps |
|---|---|---|---:|
| 1. `pick_heat_then_place_in_recep-Apple-None-GarbageCan-10` | environment_failure | success | 50 → 32 |
| 2. `pick_heat_then_place_in_recep-Mug-None-Cabinet-10` | aborted | success | 43 → 19 |
| 3. `look_at_obj_in_light-CD-None-DeskLamp-308` | environment_failure | success | 50 → 6 |
| 4. `pick_clean_then_place_in_recep-SoapBar-None-Cabinet-424` | environment_failure | environment_failure | 50 → 50 |
| 5. `pick_two_obj_and_place-SoapBar-None-GarbageCan-424` | environment_failure | success | 50 → 18 |
| 6. `pick_two_obj_and_place-PepperShaker-None-Drawer-10` | environment_failure | success | 50 → 12 |
| 7. `pick_heat_then_place_in_recep-Egg-None-GarbageCan-10` | environment_failure | environment_failure | 50 → 50 |
| 8. `pick_cool_then_place_in_recep-Lettuce-None-CounterTop-10` | aborted | success | 3 → 14 |
| 9. `pick_two_obj_and_place-PepperShaker-None-Drawer-10` | environment_failure | success | 50 → 25 |
| 10. `pick_cool_then_place_in_recep-Tomato-None-Microwave-10` | environment_failure | environment_failure | 50 → 50 |
| 11. `pick_clean_then_place_in_recep-Mug-None-CoffeeMachine-10` | aborted | success | 24 → 18 |
| 12. `pick_two_obj_and_place-SoapBar-None-Cabinet-424` | aborted | success | 9 → 18 |
| 13. `pick_and_place_simple-SoapBottle-None-Toilet-424` | environment_failure | success | 50 → 4 |
| 14. `pick_clean_then_place_in_recep-Spatula-None-Drawer-10` | environment_failure | success | 50 → 28 |
| 15. `pick_and_place_simple-PepperShaker-None-Drawer-10` | aborted | environment_failure | 4 → 50 |
| 16. `pick_clean_then_place_in_recep-SoapBar-None-Cabinet-424` | environment_failure | success | 50 → 7 |
| 17. `pick_and_place_simple-Vase-None-Safe-219` | environment_failure | success | 50 → 21 |
| 18. `pick_cool_then_place_in_recep-Mug-None-Cabinet-10` | aborted | success | 5 → 10 |
| 19. `pick_and_place_simple-PepperShaker-None-Drawer-10` | environment_failure | success | 50 → 5 |
| 20. `pick_and_place_simple-Mug-None-Desk-308` | environment_failure | success | 50 → 5 |

## Qualitative inspection and gate

All 20 candidate trajectories were inspected. Per-trajectory notes with IDs are in `data/results/phase3b-pilot-20/inspection.md`. Reconstruction also verified that all 442 recorded prompts contain exactly the configured bounded observable history.

- **Recent actions/subgoals:** Successful runs generally retain the acquired object and complete cleaning/cooling/placement sequences. Two-object tasks usually place distinct objects without immediately retaking the first.
- **Incorrect objects:** No wrong object-type manipulation was observed in the candidate, unlike the earlier diagnostic.
- **Repeated search:** All four failures (manifest rows 4, 7, 10, 15) repeatedly revisit already searched containers or appliances while leaving other available locations unexplored. Several successful runs also waste steps this way.
- **Undoing subgoals:** Row 9 briefly retakes/replaces an already deposited peppershaker and relocates another correct-type instance before succeeding. Older progress is not tracked reliably.
- **Repeated appliances:** Rows 1–2 repeatedly open/close/examine the microwave before using the admissible heat action while holding the object. Both recover and succeed.
- **Planning versus missing state:** Immediate tracking is materially better, but remaining failures cannot be labeled purely planning failures independent of working state. Poor search priorities coexist with a window that omits older search coverage. This is an observable-pattern interpretation, not access to model reasoning or a causal attribution.

**Suitable** for the next controlled memory pilot: 80% success is meaningfully non-zero, 0% aborts is low, successful source trajectories cover all six task families, and four failures retain experimental headroom. This small calibration sample does not establish population-level stability or final thesis performance. Future policy comparisons must freeze this whole candidate setup, including history length, model parameters, parser, environment, and manifest. The returned model name is not a dated snapshot, and provider seeds are best-effort; continue recording exact returned identifiers.

No additional calibration change was implemented or is required to pass this diagnostic gate. No long-term memory, retrieval, episodic memory, skill, oracle, controller, or Phase 4 functionality was added.

## Verification and files

- **135 tests passed** under both Python 3.12.3 and Python 3.14.2. The final suite uses fake providers/environments and does not require credentials, pilot artifacts, or ALFWorld downloads.
- Tests cover v2 prompt construction, ordering/truncation/reset of observable history, unchanged v1 behavior, quoted/prefixed action normalization, invalid-action rejection, normalization flags/counts, model/reasoning parameters, no fallback, backward artifact compatibility, baseline resume/append-only invariants, and paired-comparison validation.
- Live candidate resume preserved all canonical file hashes without new attempts/calls. Phase 3 preservation and candidate resume evidence: `data/results/phase3b-pilot-20/preservation_verification.json`.
- `git diff --check` passed.

Phase 3b source/config/test changes:

- `src/aec/config.py`, `src/aec/llm/openai.py`.
- `src/aec/agents/v2.py`, `src/aec/agents/runner.py`.
- `src/aec/trajectories/schemas.py` (additive normalization flag; existing semantics retained).
- `src/aec/experiments/artifacts.py`, `baseline.py`, `comparison.py`.
- `configs/baseline_calibration.yaml`.
- `scripts/validate_live_stack.py`, `scripts/compare_baselines.py`.
- `tests/test_agent_v2.py`, `tests/test_comparison.py`.
- `.gitignore`, `README.md`, `PHASE_3B_SUMMARY.md`.

New evidence: the separate compatibility/preflight JSON files and `data/results/phase3b-pilot-20/` containing resolved config, attempts, canonical trajectories, task results, metrics, paired comparison, inspection notes, and preservation verification. Existing Phase 3 outputs were not rewritten.
