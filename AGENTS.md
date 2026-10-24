# AGENTS.md

## Project

This repository supports an MSc research project on adaptive experience compression for LLM agents.

The core decision is whether a completed agent trajectory should be:

- discarded
- stored as episodic memory
- consolidated into a procedural skill

Read `BUILD_GUIDE.md` before making substantive changes.

## Research invariants

1. Keep the acting LLM, prompt, decoding settings, and environment fixed across primary policy comparisons.
2. Keep retrieval machinery fixed unless an experiment explicitly varies it.
3. Preserve source trajectory IDs for every generated episode and skill.
4. Do not request or store hidden chain-of-thought. Store only observable interaction data.
5. Counterfactual discard / episode / skill branches must use isolated state.
6. Experiment-changing parameters belong in config, not hard-coded logic.
7. Controller features may use only information available at decision time. No future leakage.
8. Raw experimental outputs are append-only by default.
9. Prefer deterministic environment scoring over LLM-as-a-judge.
10. Do not implement later build phases before earlier validation gates are complete.

## Engineering principles

- Prefer simple, explicit research code over generic frameworks.
- Keep environment and model-provider dependencies behind thin adapters.
- Use typed schemas at component boundaries.
- Make state mutation explicit.
- Fail loudly on malformed experimental artifacts.
- Add tests for research invariants, not only software correctness.
- Avoid unnecessary abstractions, refactors, dependencies, or infrastructure.
- Do not add a web UI, vector database, multi-agent framework, or distributed system unless explicitly requested.

## Workflow

For each task:

1. Inspect the existing implementation before editing.
2. Make the smallest coherent change that satisfies the request.
3. Add or update relevant tests.
4. Run the relevant test suite.
5. Report changed files and verification results.
6. Flag any requested change that would compromise experimental validity.

When in doubt, preserve reproducibility and experimental control over implementation convenience.