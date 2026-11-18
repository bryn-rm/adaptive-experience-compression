import json

import pytest
from pydantic import ValidationError

from aec.agents.actions import ActionParseError, parse_action
from aec.agents.prompt import build_prompt
from aec.agents.runner import run_episode
from aec.agents.v2 import ObservationAction, build_prompt_v2, parse_action_v2
from aec.config import AgentConfig, ALFWorldConfig, LLMConfig
from aec.environments.alfworld import ALFWorldAdapter
from aec.experiments.artifacts import load_baseline_config
from aec.llm import LLMResponse
from aec.llm.openai import OpenAIClient
from aec.schemas import Task, TokenUsage
from aec.trajectories.recorder import TrajectoryRecorder
from aec.trajectories.schemas import AgentTrajectory
from test_llm import Transport
from test_agent import setup_episode, FakeLLM as V1FakeLLM


def history_in(prompt):
    line = next(line for line in prompt.splitlines() if line.startswith('Recent completed'))
    return json.loads(line.split(': ', 1)[1])


def test_prompt_v2_snapshot_and_history_order():
    task = Task(task_id='t', description='Put mug on table')
    history = [ObservationAction('A mug', 'take mug')]
    assert build_prompt_v2(task, 'Holding mug', ['look', 'place mug'], history, history_length=4) == (
        'Choose one available action to complete the task.\n'
        'Return only the exact action on one line, without quotes or explanation.\n'
        'Task: "Put mug on table"\n'
        'Recent completed observation/action pairs (oldest first): '
        '[{"observation": "A mug", "action": "take mug"}]\n'
        'Observation: "Holding mug"\n'
        'Available actions: ["look", "place mug"]\n'
    )
    pairs = [ObservationAction(f'obs-{i}', f'action-{i}') for i in range(8)]
    prompt = build_prompt_v2(task, 'current', ['look'], pairs, history_length=4)
    assert history_in(prompt) == [dict(observation=f'obs-{i}', action=f'action-{i}') for i in range(4,8)]
    assert len(pairs) == 8
    assert history_in(build_prompt_v2(task, 'current', [], pairs, history_length=0)) == []
    for bad in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            build_prompt_v2(task, 'current', [], pairs, history_length=bad)


@pytest.mark.parametrize('output,normalized', [
    ('go to fridge 1', False), ('"go to fridge 1"', True),
    ("'go to fridge 1'", True), ('Action: go to fridge 1', True),
    ('Action: "go to fridge 1"', True), ('  go to fridge 1\n', True),
])
def test_admissible_normalization(output, normalized):
    parsed = parse_action_v2(output, ['go to fridge 1'])
    assert parsed.action == 'go to fridge 1' and parsed.normalized == normalized


@pytest.mark.parametrize('output', [
    '', 'go to fridge 2', '"go to fridge 2"', 'action: go to fridge 1',
    'ACTION: go to fridge 1', 'Action:go to fridge 1', 'go to fridge 1.',
    'Go to fridge 1', 'go  to fridge 1', 'Action: Action: go to fridge 1',
    'go to fridge 1\nlook', '```\ngo to fridge 1\n```',
    '"go to fridge 1', 'go to fridge 1 because it is cold',
])
def test_invalid_outputs_never_select_a_substitute(output):
    with pytest.raises(ActionParseError):
        parse_action_v2(output, ['go to fridge 1'])


def test_v1_preservation_and_configuration():
    config = load_baseline_config('configs/baseline_calibration.yaml')
    old = load_baseline_config('configs/baseline_pilot.yaml')
    assert config.manifest == old.manifest
    assert config.environment == old.environment
    assert config.selection_seed == old.selection_seed
    assert config.run_id != old.run_id
    assert config.agent.history_length == 4 and config.agent.prompt_version == 'agent_v2'
    assert config.agent.llm.model_id == 'gpt-5.6-terra'
    assert config.agent.llm.reasoning_effort == 'low'
    assert config.agent.llm.temperature is None
    assert old.agent.history_length == 0 and old.agent.llm.reasoning_effort is None
    assert 'Recent completed' not in build_prompt(Task(task_id='t', description='task'), 'obs', ['look'])
    for output in ['"look"', 'Action: look']:
        with pytest.raises(ActionParseError):
            parse_action(output, ['look'])
    with pytest.raises(ValidationError):
        AgentConfig(prompt_version='agent_v1', history_length=4, llm=old.agent.llm)
    with pytest.raises(ValidationError):
        LLMConfig(model_id='gpt-5.6-terra', reasoning_effort='automatic')


def test_provider_reasoning_parameters_and_no_reasoning_persistence():
    transport = Transport()
    transport.response.model = 'gpt-5.6-terra-returned-snapshot'
    client = OpenAIClient(client=transport)
    config = load_baseline_config('configs/baseline_calibration.yaml').agent.llm
    result = client.generate('prompt', config)
    assert transport.kwargs['model'] == 'gpt-5.6-terra'
    assert transport.kwargs['reasoning_effort'] == 'low'
    assert 'temperature' not in transport.kwargs
    assert transport.kwargs['seed'] == 0 and transport.kwargs['top_p'] == 1
    assert transport.kwargs['max_completion_tokens'] == 2048
    assert transport.kwargs['store'] is False
    assert result.model_id == 'gpt-5.6-terra-returned-snapshot'
    assert 'private' not in result.model_dump_json()
    assert not any('reasoning' in key for key in result.model_dump())


def test_unavailable_model_has_no_retry_or_fallback():
    transport = Transport()
    calls = []
    def unavailable(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('model unavailable')
    transport.create = unavailable
    config = load_baseline_config('configs/baseline_calibration.yaml').agent.llm
    with pytest.raises(RuntimeError, match='unavailable'):
        OpenAIClient(client=transport).generate('prompt', config)
    assert len(calls) == 1 and calls[0]['model'] == 'gpt-5.6-terra'


class Backend:
    def __init__(self):
        self.index = 0
        self.actions = []
    def infos(self):
        return {'extra.gamefile': ['task'], 'admissible_commands': [['look']],
                'won': [self.index == 7]}
    def reset(self):
        self.index = 0
        return ['original task'], self.infos()
    def step(self, actions):
        self.actions.extend(actions)
        self.index += 1
        return [f'obs-{self.index}'], [float(self.index == 7)], [self.index == 7], self.infos()


class FakeLLM:
    def __init__(self, output='Action: "look"'):
        self.prompts = []
        self.output = output
    def generate(self, prompt, config):
        self.prompts.append(prompt)
        return LLMResponse(text=self.output, model_id='gpt-5.6-terra-returned-snapshot',
                           finish_reason='stop', token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5))


def test_episode_history_is_bounded_observable_and_cleared_between_tasks(tmp_path):
    path = tmp_path/'backend.yaml'
    path.write_text('env: {type: AlfredTWEnv}')
    backend = Backend()
    env = ALFWorldAdapter(ALFWorldConfig(backend_config=path,max_steps=8), backend=backend)
    config = load_baseline_config('configs/baseline_calibration.yaml').agent
    llm = FakeLLM()
    recorder = TrajectoryRecorder(tmp_path/'trajectories.jsonl')
    trajectory = run_episode(env, llm, config, recorder, run_id='v2')
    assert trajectory.success and len(trajectory.steps) == 7
    assert history_in(llm.prompts[0]) == []
    for index, prompt in enumerate(llm.prompts):
        expected = [dict(observation=s.observation, action=s.action)
                    for s in trajectory.steps[max(0,index-4):index]]
        assert history_in(prompt) == expected
        assert 'Action:' not in prompt  # Raw provider text is never history.
        assert all(set(pair) == {'observation', 'action'} for pair in history_in(prompt))
    assert all(s.action_normalized and s.output == 'Action: "look"' for s in trajectory.steps)
    assert all(s.response_model_id == 'gpt-5.6-terra-returned-snapshot' for s in trajectory.steps)
    restored = AgentTrajectory.model_validate_json(recorder.path.read_text())
    assert restored == trajectory
    run_episode(env, llm, config, recorder, run_id='v2-second')
    assert history_in(llm.prompts[7]) == []


def test_invalid_v2_action_aborts_without_execution(tmp_path):
    path=tmp_path/'backend.yaml'
    path.write_text('env: {type: AlfredTWEnv}')
    backend=Backend()
    env=ALFWorldAdapter(ALFWorldConfig(backend_config=path), backend=backend)
    config=load_baseline_config('configs/baseline_calibration.yaml').agent
    trajectory=run_episode(env, FakeLLM('Action: invent'), config,
                           TrajectoryRecorder(tmp_path/'t.jsonl'), run_id='invalid')
    assert trajectory.status == 'aborted' and trajectory.termination_reason == 'parse_error'
    assert backend.actions == []
    assert not trajectory.steps[0].action_normalized


def test_old_phase3_artifacts_remain_readable(setup_episode):
    # An offline legacy artifact fixture: no pilot files or downloads required.
    env, _, config, recorder = setup_episode
    original = run_episode(env, V1FakeLLM(), config, recorder, run_id='legacy')
    legacy = original.model_dump(mode='json')
    legacy['agent_config'].pop('history_length')
    legacy['agent_config']['llm'].pop('reasoning_effort')
    for step in legacy['steps']:
        step.pop('action_normalized')
    restored = AgentTrajectory.model_validate_json(json.dumps(legacy))
    assert restored == original
    assert restored.prompt_version == 'agent_v1'
    assert restored.agent_config.history_length == 0
    assert not any(s.action_normalized for s in restored.steps)
