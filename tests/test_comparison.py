import pytest

from aec.config import AgentConfig
from aec.experiments.baseline import run_baseline
from aec.experiments.comparison import compare_runs
from test_baseline import experiment, LLM


def test_comparison_keeps_setups_separate_and_checks_pairing(experiment):
    config, manifest, factory, _ = experiment
    run_baseline(config, env_factory=factory, llm_factory=LLM)
    previous = config.output_dir / config.run_id
    config.run_id = 'candidate'
    config.agent = AgentConfig(prompt_version='agent_v2', history_length=4, llm=config.agent.llm)
    run_baseline(config, env_factory=factory, llm_factory=LLM)
    candidate = config.output_dir / config.run_id
    comparison = compare_runs(previous, candidate)
    assert comparison['previous_metrics']['attempted_tasks'] == 3
    assert comparison['candidate_metrics']['attempted_tasks'] == 3
    assert comparison['previous_agent']['prompt_version'] == 'agent_v1'
    assert comparison['candidate_agent']['prompt_version'] == 'agent_v2'
    assert [r['task_id'] for r in comparison['tasks']] == list(manifest.task_ids)
    assert all(r['previous_trajectory_id'] != r['candidate_trajectory_id'] for r in comparison['tasks'])
    path = candidate / 'results.jsonl'
    path.write_text(path.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='every manifest task'):
        compare_runs(previous, candidate)


def test_normalized_actions_are_counted_in_baseline_metrics(experiment):
    config, _, factory, _ = experiment
    config.agent = AgentConfig(prompt_version='agent_v2', history_length=4, llm=config.agent.llm)
    class Formatted(LLM):
        def generate(self, prompt, settings):
            response = super().generate(prompt, settings)
            response.text = 'Action: "look"'
            return response
    metrics = run_baseline(config, env_factory=factory, llm_factory=Formatted)
    assert metrics['normalized_actions'] == 6
    assert metrics['task_parse_failures'] == metrics['parse_failures'] == 0
    assert metrics['completed_tasks'] == 3
