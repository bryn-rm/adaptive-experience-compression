import json

import pytest

from aec.config import AgentConfig, LLMConfig
from aec.experiments.artifacts import load_baseline_config
from aec.experiments.baseline import run_baseline
from aec.experiments.gate_a import divergence, gate_a_report, sign_test
from test_memory import Embedder, count_tokens, memory_config
from test_skills import LLM, experiment, logs  # noqa: F401  (fixture)

GATE_A = {
    "no_memory": "configs/baseline_calibration.yaml",
    "no_memory_replicate": "configs/baseline_calibration.yaml",
    "episodic": "configs/episodic_pilot.yaml",
    "skill": "configs/skill_pilot.yaml",
}


def test_gate_a_configs_equal_their_pilots_except_run_identity():
    configs = {label: load_baseline_config(f"configs/gate_a/{label}.yaml") for label in GATE_A}
    for label, source in GATE_A.items():
        pilot, gate = load_baseline_config(source), configs[label]
        update = dict(run_id=gate.run_id)
        if pilot.memory is not None:
            update["memory"] = pilot.memory.model_copy(
                update=dict(store_path=gate.memory.store_path))
        assert pilot.model_copy(update=update) == gate
        assert gate.run_id != pilot.run_id
        if pilot.memory is not None:
            assert gate.memory.store_path != pilot.memory.store_path
    assert len({c.run_id for c in configs.values()}) == len(configs)
    assert len({(c.agent.model_dump_json(), c.environment.model_dump_json(), c.manifest)
                for c in configs.values()}) == 1
    assert configs["no_memory"].memory is configs["no_memory_replicate"].memory is None


def test_sign_test_is_exact_two_sided_binomial():
    assert sign_test(0, 0) is None
    assert sign_test(0, 5) == sign_test(5, 0) == 2 / 32
    assert sign_test(1, 4) == 2 * (1 + 5) / 32
    assert sign_test(3, 3) == 1.0


def test_divergence_reports_first_differing_step():
    assert divergence(["a", "b"], ["a", "b"]) == dict(
        identical_actions=True, first_divergent_step=None, action_similarity=1.0)
    assert divergence(["a", "b"], ["a", "c", "d"])["first_divergent_step"] == 1
    assert divergence(["a"], ["a", "b"])["first_divergent_step"] == 1
    assert not divergence(["a"], ["a", "b"])["identical_actions"]


class MemorySensitiveLLM(LLM):
    """Aborts whenever a skill is injected: deterministic negative transfer."""

    def generate(self, prompt, config):
        response = super().generate(prompt, config)
        if "Relevant skills (procedural memories" in prompt:
            response.text = "not an admissible action"
        return response


def run(config, factory, llm_factory=LLM):
    return run_baseline(config, env_factory=factory, llm_factory=llm_factory,
                        embedder_factory=Embedder, count_tokens=count_tokens)


def _runs(config, factory, tmp_path, skill_llm=LLM):
    skill = config.memory
    paths = {}
    for label, memory, llm in (("no_memory", None, LLM), ("no_memory_replicate", None, LLM),
                               ("episodic", memory_config(tmp_path / "ep"), LLM),
                               ("skill", skill, skill_llm)):
        config.run_id, config.memory = label, memory
        run(config, factory, llm_factory=llm)
        paths[label] = config.output_dir / label
    return paths


def test_gate_a_report_pairs_all_policies_and_counts_transfer(experiment, tmp_path):
    config, manifest, factory = experiment
    paths = _runs(config, factory, tmp_path, skill_llm=MemorySensitiveLLM)
    report = gate_a_report(paths["no_memory"], paths["episodic"], paths["skill"],
                           paths["no_memory_replicate"])
    json.dumps(report, allow_nan=False)
    assert report["controls"]["controlled"]
    assert [t["task_id"] for t in report["tasks"]] == list(manifest.task_ids)
    assert set(report["metrics"]) == {"no_memory", "no_memory_replicate", "episodic", "skill"}
    comparisons = {(c["reference"], c["candidate"]): c for c in report["comparisons"]}
    assert set(comparisons) == {("no_memory", "episodic"), ("no_memory", "skill"),
                                ("episodic", "skill"), ("no_memory", "no_memory_replicate")}
    for c in comparisons.values():
        assert c["both_success"] + c["neither_success"] + c["reference_only_success"] + (
            c["candidate_only_success"]) == c["tasks"] == len(manifest.task_ids)

    # Deterministic fake acting: the replicate is the zero noise floor.
    floor = comparisons[("no_memory", "no_memory_replicate")]
    assert floor["identical_action_sequences"] == floor["tasks"]
    assert floor["sign_test_p"] is None and floor["failure_mode_changes"] == 0

    # Every task with a retrieved skill aborts; each one no-memory solved is negative transfer.
    skill_rows = {r["task_id"]: r for r in comparisons[("no_memory", "skill")]["task_rows"]}
    _, skill_results, _, _ = logs(config)
    lost = comparisons[("no_memory", "skill")]["reference_only_success_tasks"]
    expected = [r.task_id for r in skill_results if r.retrieved_memory_ids
                and skill_rows[r.task_id]["reference_outcome"] == "success"]
    assert expected and [r["task_id"] for r in lost] == expected
    assert all(r["candidate_failure_mode"] == "aborted:parse_error"
               and r["candidate_retrieved_memory_ids"] for r in lost)
    assert comparisons[("no_memory", "skill")]["negative_transfer_rate"] == (
        len(expected) / len(manifest.task_ids))
    by_task = {r.task_id: r for r in skill_results}
    for row in report["tasks"]:
        assert row["skill"]["trajectory_id"] == by_task[row["task_id"]].trajectory_id
        assert row["no_memory"]["retrieved_memory_ids"] == []


def test_gate_a_rejects_incomparable_runs(experiment, tmp_path):
    config, _, factory = experiment
    paths = _runs(config, factory, tmp_path)
    with pytest.raises(ValueError, match="policy"):
        gate_a_report(paths["no_memory"], paths["skill"], paths["episodic"])
    with pytest.raises(ValueError, match="policy"):
        gate_a_report(paths["no_memory"], paths["episodic"], paths["skill"], paths["skill"])
    config.run_id, config.memory = "other_agent", None
    config.agent = AgentConfig(prompt_version="agent_v2", history_length=2,
                               llm=LLMConfig(model_id="fake", seed=11))
    run(config, factory)
    with pytest.raises(ValueError, match="Acting agent"):
        gate_a_report(config.output_dir / "other_agent", paths["episodic"], paths["skill"])


def test_gate_a_reports_uncontrolled_time_window(experiment, tmp_path):
    config, _, factory = experiment
    paths = _runs(config, factory, tmp_path)
    report = gate_a_report(paths["no_memory"], paths["episodic"], paths["skill"],
                           max_span_hours=0)
    assert not report["controls"]["within_time_window"]
    assert not report["controls"]["controlled"]
    assert "NOT fully controlled" in report["interpretation"]
    assert "No no-memory replicate" in report["interpretation"]


def test_gate_a_rejects_tampered_results(experiment, tmp_path):
    config, _, factory = experiment
    paths = _runs(config, factory, tmp_path)
    results = paths["skill"] / "results.jsonl"
    rows = results.read_text().splitlines()
    first = json.loads(rows[0])
    first["steps"] += 1
    results.write_text("\n".join([json.dumps(first)] + rows[1:]) + "\n")
    with pytest.raises(ValueError, match="canonical trajectory"):
        gate_a_report(paths["no_memory"], paths["episodic"], paths["skill"])
