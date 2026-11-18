# Gate A: does memory change behaviour or outcomes?

**Recommendation: pass, with caveats.** Memory changes behaviour beyond the run-to-run
noise floor. Episodic memory does so strongly. Procedural skills do so weakly, mostly
through shorter solutions. Neither changes the success rate detectably at n=20; Gate A
does not require that. Signing the gate off is the researcher's decision.

## Runs (controlled)

Four runs over the immutable `phase3-pilot-20` manifest, started together on 2026-11-18.
All attempts fell within 11.5 minutes, from the same source hash, Python, and packages
(`controls.controlled: true` in the report). Configs: `configs/gate_a/`. Each is
identical to its pilot except run ID and store path, and a test enforces this.

| Run | Policy |
|---|---|
| `gatea-none-20` | no memory |
| `gatea-none-r2-20` | no memory, replicate (noise floor) |
| `gatea-episodic-20` | episodic |
| `gatea-skill-20` | fixed skill rule |

Report: `data/results/gate-a-report.json` (`scripts/gate_a_report.py`). A replicate is
needed because the acting model samples at its provider-default temperature, so two
no-memory runs already differ.

## Aggregates

| Metric | None | None r2 | Episodic | Skill |
|---|---:|---:|---:|---:|
| Successes / env. failures / aborts | 14 / 5 / 1 | 14 / 4 / 2 | 15 / 3 / 2 | 15 / 4 / 1 |
| Mean / median steps, completed | 23.9 / 18 | 22.1 / 16.5 | 18.0 / 13 | 18.7 / 9 |
| Acting prompt / completion tokens | 280k / 16.2k | 250k / 14.4k | 316k / 12.8k | 279k / 13.2k |
| Tasks with memory retrieved / items | — | — | 18 / 50 | 14 / 25 |
| Injected memory tokens | — | — | 7,778 | 5,379 |
| Writer prompt / completion tokens | — | — | 14.2k / 2.1k | 19.9k / 2.1k |
| Stored items / store tokens | — | — | 15 episodes / 2,141 | 5 skills (9 versions) / 836 |

No writes were rejected. No provider, setup, or runtime failures occurred.

## Paired comparisons, with no memory as the reference

| Comparison | Gained | Lost | Sign p | Failure-mode changes | Identical action seqs | Mean action similarity | Δsteps, both solved (mean / median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Replicate (floor) | 1 | 1 | 1.0 | 2 | 6 | 0.69 | +0.8 / 0 |
| Episodic | 3 | 2 | 1.0 | 5 | 2 | 0.46 | −2.0 / −1 |
| Skill | 1 | 0 | 1.0 | 1 | 4 | 0.64 | −4.0 / −4 |

"Lost" is negative transfer (NTR 0.10 episodic, 0.00 skill, 0.05 floor).

- **Behaviour.** On tasks where memory was retrieved, episodic action similarity was 0.43
  vs 0.68 for the floor on the same tasks. Skill similarity was 0.61 vs 0.63, barely
  below the floor.
- **Cleanest evidence.** The two no-memory runs took identical actions on tasks 5, 12,
  13, 14, 16, and 17. On tasks 12 and 13, both memory policies changed the **first**
  action and cut steps from 18 to 9 and from 14 to 4. Episodic memory also changed task
  14 (8 → 17 steps, still solved).
- **Outcomes.** Success differences are within noise for every pair (exact sign test
  p = 1.0). The replicate alone flipped two tasks.

## Mechanisms (traced to retrieved memories)

- **Positive transfer, task 10** (cool tomato → microwave). Every earlier run of every
  policy failed this search, and both no-memory runs and the skill run did here too. The
  episodic agent went straight to `garbagecan 1` and solved it in 7 steps. The retrieved
  episode from **task 7**, the same floorplan (`-10`), recorded that "the garbagecan
  already contained … tomato 3". There is no leakage: across all retrievals, 0 came from
  the current or a later task. This is a scene-specific fact, which skills abstract away
  by design.
- **Negative transfer, task 19** (peppershaker → drawer). No-memory and skill solved it.
  Episodic retrieved three same-scene peppershaker episodes (similarity 0.87–0.94) that
  located peppershakers on countertop 3 and cabinets 1–2. Here the target was on
  countertop 1. The agent went to countertop 3 first, then cycled cabinets until the step
  limit. The memory was specific and stale.
- **Task 20.** Episodic aborted on an inadmissible `go to drawer 3`. The no-memory
  replicate aborted the same way, so this is within noise.
- **Task 15.** Gains under both memory policies match the replicate, which also solved it
  in 7 steps. Treat this as noise.

## Corrections to earlier conclusions

- **Task 17 aborts are not memory-caused.** `PHASE_5_SUMMARY.md` said the adjacency abort
  ("3 occurrences across the two memory runs, 0 in B0") might be memory-related. Here
  **all four** runs abort task 17 at step 2, both no-memory runs included. The abort is
  an acting-setup quirk: the model issues `go to shelf 3` from an adjacent shelf.
- **Task 10 is not a pure search failure.** It is solvable from same-scene experience.

## Caveats for Phase 6

1. **Scene overlap is heavy and was not designed for.** 12 of 20 tasks share floorplan
   `10`, 5 share `424`, and 2 share `308`. Every episodic retrieval from task 6 onwards
   included a same-scene episode. Both the strongest gain and the strongest loss came
   from scene-specific facts. Phase 6's "very similar / shifted / distant" axes should
   separate same scene from same task family. Otherwise episodic results will be
   dominated by room-layout recall.
2. **The skill condition is structurally weak in an online stream.** A family needs two
   successes before any skill exists. Tasks 7 and 10 (heat, cool) saw only a cross-family
   `pick_two` skill. Phase 6's controlled design, with fixed source experience and every
   condition built from it, removes this confound.
3. **Power.** One seed, n=20, sampled acting. Outcome effects of this size cannot be
   distinguished from noise; only behavioural differences are interpretable.
4. **The task 17 / task 20 acting quirk** (inadmissible adjacent `go to`) affects every
   policy. It is a fixed property of the acting setup, not something to fix under memory
   policies.

## Verification

- 188 tests pass under Python 3.12.3 and 3.14 (7 new in `tests/test_gate_a.py`). They
  cover:
  - config equality with the pilots;
  - the exact sign test and divergence;
  - pairing, transfer counting, and provenance on fake runs;
  - rejection of mismatched policies or agents;
  - reporting an uncontrolled time window;
  - rejection of tampered results.
- A dry run on the historical Phase 3b/4/5 runs correctly reports `controlled: false`
  (575 h span, different source hash and packages).
