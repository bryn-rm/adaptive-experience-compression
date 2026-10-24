Adaptive Experience Compression — Compact Build Guide
Purpose
Build the MSc thesis system incrementally without letting coding agents over-engineer it.

Core research question:
Given a completed agent trajectory, should the experience be discarded, stored as an episodic memory, or consolidated into a procedural skill?

The acting LLM stays fixed. The thesis evaluates whether choosing the representation adaptively improves future task performance under memory and token constraints.

The detailed research rationale, hypotheses, literature, and experimental design live in the thesis plan. This file is only the build specification.
1. Research Invariants
These rules must hold unless explicitly changed for an experiment.
1. Fixed acting model. Use the same LLM, prompt, decoding settings, and environment across compared memory policies.
2. Fixed retrieval machinery. Same embedding model, similarity metric, and top-k settings unless retrieval itself is being ablated.
3. Preserve provenance. Every episode and skill must store its source trajectory IDs.
4. No hidden chain-of-thought. Store only observable prompts, actions, observations, outputs, rewards, and metadata.
5. Isolate experimental state. discard, episode, and skill counterfactual branches must never mutate each other.
6. Config over hard-coding. Experiment-changing values belong in config.
7. No future leakage. Controller features may only use information available when the D/E/S decision is made.
8. Append-only raw results. Never overwrite raw experiment outputs.
9. Deterministic scoring where possible. Prefer environment success/reward over LLM-as-a-judge.
10. Do not build future phases early.
2. Minimal Stack
Use:
- Python
- Pydantic
- pytest
- YAML config
- SQLite for persistent memory/state
- JSONL for trajectories/raw logs
- Parquet for analysis datasets
- pandas / numpy / scikit-learn / scipy / matplotlib
- one fixed sentence-embedding model
- ALFWorld text mode as the first environment
Do not add unless needed:
- LangChain/LangGraph
- vector database
- web UI
- distributed execution
- multi-agent architecture
- custom training infrastructure
For hundreds or low-thousands of memories, brute-force cosine similarity is enough.
3. Target Architecture
Keep four layers separate:
┌──────────────────────────────────┐
│ 4. EXPERIENCE POLICY             │
│ discard / episode / skill        │
│ heuristic / learned / oracle     │
├──────────────────────────────────┤
│ 3. EXPERIENCE SYSTEM             │
│ episodes, skills, retrieval      │
├──────────────────────────────────┤
│ 2. AGENT                         │
│ fixed LLM + prompt + action loop │
├──────────────────────────────────┤
│ 1. ENVIRONMENT                   │
│ ALFWorld + deterministic scoring │
└──────────────────────────────────┘
The thesis primarily manipulates layer 4.

Layers 1–3 should become stable before the main experiments.
4. Minimal Repository
adaptive-experience-compression/
├── AGENTS.md
├── BUILD_GUIDE.md
├── README.md
├── pyproject.toml
├── configs/
├── src/aec/
│   ├── schemas/
│   ├── environments/
│   ├── llm/
│   ├── agents/
│   ├── trajectories/
│   ├── memory/
│   ├── oracle/
│   ├── controller/
│   ├── experiments/
│   └── analysis/
├── scripts/
├── tests/
└── data/
    ├── trajectories/
    ├── memory/
    ├── oracle/
    └── results/
Do not scaffold every possible file at the start. Add modules only when a phase requires them.
5. Core Data Types
At minimum define:
Task
TrajectoryStep
Trajectory
EpisodicMemory
ProceduralSkill
RetrievalRecord
ExperienceDecision
CounterfactualOutcome
Important fields:
Trajectory
trajectory_id
task_id
steps
success
reward
termination_reason
model_id
prompt_version
seed
token usage
EpisodicMemory
memory_id
source_trajectory_ids
task_summary
experience
outcome
embedding_text
token_count
ProceduralSkill
skill_id
source_trajectory_ids
name
activation_conditions
procedure
constraints
failure_modes
embedding_text
version
token_count
ExperienceDecision
trajectory_id
action: discard | episode | skill
policy_name
feature_version
features
confidence
6. Build Order
Do not skip the gates.
Phase 0  Project scaffold
Phase 1  ALFWorld adapter
Phase 2  Frozen LLM agent + trajectory logging
Phase 3  No-memory baseline
Phase 4  Episodic-memory baseline
Phase 5  Procedural-skill baseline

GATE A: Does memory change behaviour/outcomes?

Phase 6  Controlled representation pilot

GATE B: Does the best representation vary?

Phase 7  Counterfactual oracle

GATE C: Is oracle adaptive headroom meaningful?

Phase 8  Controller features + dataset
Phase 9  Heuristic + learned controllers
Phase 10 Main evaluation
Phase 11 Longitudinal / OOD / ablations
7. Phase Requirements
Phase 0 — Scaffold
Build:
- package structure
- config loader
- schemas
- logging
- CLI skeleton
- pytest setup
Acceptance:
pytest
passes and schemas round-trip to/from JSON.

Do not add agent or memory logic.
Phase 1 — ALFWorld Adapter
Create a thin project-owned interface:
reset()
step(action)
valid_actions()
current_task()
Only the adapter should import ALFWorld directly.

Acceptance:
- run 3 text-mode episodes with a random/scripted policy
- log task, observation, action, reward, done
No LLM yet.
Phase 2 — Frozen Agent
Add:
- thin LLMClient
- one provider implementation
- versioned agent prompt
- action parser
- trajectory recorder
Acceptance:
- run 10 tasks
- every task produces a valid trajectory record
- malformed actions are logged
- model/prompt/settings are recorded
Do not add long-term memory.
Phase 3 — No-Memory Baseline
Add:
- immutable task manifest
- batch runner
- resumable runs
- task-level results
- aggregate metrics
This is baseline B0.

Before proceeding, manually inspect ~20 trajectories.

If the model solves almost nothing or almost everything, fix benchmark/model difficulty first.
Phase 4 — Episodic Memory
For each successful trajectory:
1. generate a concise task-specific episode
2. embed it
3. store it
4. retrieve top-k episodes for future tasks
5. insert them into a fixed memory section in the prompt
Log every retrieval:
memory ID
similarity
rank
current task
task outcome
Acceptance:
- memory persists across runs
- related tasks retrieve episodes
- none and episodic policies use the same base agent
Phase 5 — Procedural Skills
Add a reusable skill representation:
name
activation conditions
ordered procedure
constraints
failure modes
source trajectories
Initial baseline can use a simple rule:
if >= N similar successful trajectories:
    synthesize/update skill
This is only a fixed skill baseline, not the adaptive controller.

Acceptance:
- skills preserve provenance
- malformed skills are rejected
- skill retrieval uses the same embedding/retrieval machinery as episodes
8. Gate A — Does Memory Matter?
Run the same small task set with:
no memory
episodes
skills
Look for changes in:
- success
- actions
- step count
- failure mode
If memory never changes behaviour, debug retrieval/prompting before continuing.
9. Phase 6 — Controlled Representation Pilot
This is the most important early experiment.

Take the same source experience and create:
discard condition
episode condition
skill condition
Evaluate each on the same future task set.

Include:
- very similar tasks
- moderately shifted tasks
- more distant/OOD tasks
Do not build a controller yet.

Proceed only if:
1. representation changes downstream performance, and
2. one representation does not dominate almost everywhere.
If skills always win, the thesis may be better reframed around skill consolidation.

If episodes always win, investigate whether skill abstraction is too lossy.
10. Phase 7 — Counterfactual Oracle
At decision point t:
snapshot M_t
   ├── apply discard
   ├── apply episode
   └── apply skill
Run the same future task window for each branch.

Store raw outcomes:
successes
rewards
prompt tokens
completion tokens
memory tokens
management tokens
Compute utility later:
\[
U(a)=Success(a)-\lambda_T TokenCost(a)-\lambda_M MemoryCost(a)
\]
Do not bake one utility weighting permanently into raw results.

Handle ties with an epsilon.

Acceptance:
- branches are isolated
- runs are resumable
- every outcome is traceable to the original trajectory and memory snapshot
11. Gate C — Oracle Headroom
Compute:
\[
U_{fixed} = \max(U_D,U_E,U_S)
\]
for the best single fixed policy across the dataset.

Then:
\[
U_{oracle} = \sum_i \max_a U_i(a)
\]
Headroom:
\[
H = U_{oracle} - U_{fixed}
\]
If headroom is negligible, stop controller work.

Also inspect the oracle label distribution. If almost every example has the same winner, adaptive routing is not a strong research problem.
12. Phase 8 — Controller Dataset
Use only decision-time features.

Suggested v1 features:
Trajectory
- success
- reward
- steps
- invalid actions
- repeated actions
- token count
Novelty
- max similarity to episode store
- max similarity to skill store
- mean top-k similarity
Recurrence
- count of similar prior trajectories
- count of similar successful trajectories
Store state
- episode count/tokens
- skill count/tokens
Specificity
- task length
- task family frequency
- action diversity
Output one row per oracle-labelled decision:
decision_id
features...
oracle_action
utility_D
utility_E
utility_S
margin
ambiguous
No future outcome may appear as a feature.
13. Phase 9 — Controllers
Train in this order:
1. majority/fixed-action baseline
2. simple heuristic
3. multinomial logistic regression
4. gradient-boosted trees
5. optional LLM controller
Primary metrics:
accuracy
macro F1
balanced accuracy
mean utility regret
oracle utility recovered
More important than classification accuracy:
\[
Regret_i = U_i(a_i^*) - U_i(\hat a_i)
\]
A wrong prediction with near-equal utility is less serious than a large-utility mistake.
14. Phase 10 — Online Adaptive Policy
Runtime:
retrieve memory
-> run task
-> record trajectory
-> extract features
-> predict D/E/S
-> apply action
-> log decision
-> next task
All policies must use the same experiment runner:
NoMemoryPolicy
AlwaysEpisodePolicy
AlwaysSkillPolicy
HeuristicPolicy
LearnedPolicy
15. Final Experiments
Minimum final evaluation:
Main comparison
- no memory
- episodes only
- skills only
- heuristic adaptive
- learned adaptive
- oracle where feasible
Memory growth
Evaluate after approximately:
10 / 25 / 50 / 100 / 200 / 500 experiences
Measure:
- success
- memory size
- prompt tokens
- negative transfer
- retrieval quality
OOD
Compare in-distribution and held-out task variants/families.
Ablations
At minimum:
- remove recurrence features
- remove novelty features
- remove discard option
- remove skill option
- token-match memory conditions
16. Negative Transfer
Treat this as a first-class metric.

A useful definition:
\[
NTR = P(\text{memory policy fails AND no-memory succeeds})
\]
For each such case, record which memory was retrieved.

These examples may be among the most important qualitative results in the thesis.
17. Cost Accounting
Track separately:
acting prompt tokens
acting completion tokens
episode-writing tokens
skill-synthesis tokens
controller tokens
memory tokens stored
Do not call a policy "efficient" based only on acting context if it uses large memory-management calls.
18. Testing Priorities
Unit tests:
- schemas
- retrieval ordering
- prompt formatting
- utility calculation
- feature extraction
- controller decisions
Integration tests:
- full fake-LLM episode
- episode write/retrieve
- skill write/retrieve
- counterfactual branch isolation
Research-invariant tests:
- same task manifest across policies
- same model config
- same embedding model
- source trajectory equality in representation-controlled experiments
- no future leakage
- no cross-branch state mutation
Use a deterministic fake LLM for tests. Do not spend API calls on control-flow tests.
19. Coding-Agent Workflow
For each task, give Codex/Claude:
Implement [PHASE] from BUILD_GUIDE.md.

Goal:
[short goal]

Current state:
[what already works]

Required:
- ...
- ...

Invariants:
- ...

Out of scope:
- ...

Verification:
- run ...
- add tests for ...
- report changed files and test results
Keep each coding task small.

After implementation:
git diff --stat
git diff
pytest
Review research logic manually even if tests pass.
20. Review Prompt
Use this with a second coding agent:
Review the current implementation against BUILD_GUIDE.md and AGENTS.md.

Focus only on:
1. correctness bugs
2. research-invariant violations
3. data leakage
4. state contamination
5. reproducibility gaps
6. missing tests that could invalidate conclusions
7. unnecessary complexity

For each issue give:
- severity: P0/P1/P2
- file/location
- why it matters
- concrete fix
- regression test

Do not suggest cosmetic refactors or new features.
21. Scope Rules
Core thesis only needs:
ALFWorld
frozen LLM
trajectory logging
episodes
skills
discard
retrieval
counterfactual oracle
simple controller features
logistic regression
tree model
final experiments
Only add later if needed:
ScienceWorld
LLM controller
skill retirement
active labelling
cross-model transfer
Leave out unless the results demand them:
declarative rules
bandit/RL controller
multi-agent system
visual ALFWorld
learned embedding model
UI
distributed framework
22. First Prompt to Give Codex / Claude
Read BUILD_GUIDE.md and AGENTS.md.

Implement only Phase 0.

Before editing, inspect the repository and identify anything that already satisfies
the phase so you do not duplicate it.

Keep the implementation minimal and do not implement future phases.

When finished:
1. run the relevant tests,
2. report the test result,
3. list changed files,
4. note any decisions that affect later phases.
The key discipline throughout the project is:
Do not build the adaptive controller until the controlled representation experiment and oracle analysis demonstrate that there is actually something useful to adapt.