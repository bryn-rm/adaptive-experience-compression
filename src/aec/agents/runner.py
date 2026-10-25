"""One frozen episode; parse/provider/environment failures produce aborted artifacts."""

import hashlib
import logging
from time import perf_counter
from typing import Callable
from uuid import uuid4

from aec.agents.actions import ActionParseError, parse_action
from aec.agents.prompt import build_prompt
from aec.config import AgentConfig
from aec.environments.alfworld import ALFWorldAdapter
from aec.llm import LLMClient
from aec.schemas import TokenUsage
from aec.trajectories.recorder import TrajectoryRecorder
from aec.trajectories.schemas import AgentStep, AgentTrajectory

logger = logging.getLogger(__name__)


def run_episode(
    env: ALFWorldAdapter,
    llm: LLMClient,
    config: AgentConfig,
    recorder: TrajectoryRecorder,
    *,
    run_id: str,
    clock: Callable[[], float] = perf_counter,
) -> AgentTrajectory:
    if not run_id:
        raise ValueError("run_id must be non-empty")
    settings = config.model_copy(deep=True)
    environment_config = env.config.model_copy(deep=True)
    backend_hash = hashlib.sha256(environment_config.backend_config.read_bytes()).hexdigest()
    started = clock()
    initial = env.reset()
    task = initial.task
    observation = initial.observation
    steps: list[AgentStep] = []
    score = 0.0
    success = False
    success_known = False
    status = "aborted"
    reason = "no_valid_actions"
    while True:
        actions = sorted(env.valid_actions())
        if not actions:
            # This acting setup requires admissible actions; never invent a fallback.
            reason = "no_valid_actions"
            break
        prompt = build_prompt(task, observation, actions)
        step_started = clock()
        response = None
        action = ""
        parse_error = None
        error_type = None
        result = None
        try:
            response = llm.generate(prompt, settings.llm)
        except Exception as exc:
            error_type = type(exc).__name__
            reason = "provider_error"
        if response is not None:
            try:
                if response.finish_reason != "stop":
                    raise ActionParseError("Provider did not finish a complete action response")
                action = parse_action(response.text, actions)
            except ActionParseError as exc:
                parse_error = str(exc)
                reason = "parse_error"
                logger.warning("Action parse failure in run %s, step %d: %s",
                               run_id, len(steps), parse_error)
            else:
                try:
                    result = env.step(action)
                    if result.task_id != task.task_id:
                        raise ValueError("Environment changed task identity within episode")
                except Exception as exc:
                    result = None
                    error_type = type(exc).__name__
                    reason = "environment_error"
        step = AgentStep(
            step_index=len(steps), observation=observation,
            next_observation=result.observation if result else None,
            prompt=prompt, action=action, output=response.text if response else None,
            reward=result.reward if result else 0,
            done=result.done if result else True,
            valid_actions=actions, valid_action=bool(action), executed=result is not None,
            parse_error=parse_error, error_type=error_type,
            wall_time=clock() - step_started,
            token_usage=response.token_usage if response else TokenUsage(),
            usage_known=response is not None,
            response_model_id=response.model_id if response else None,
            response_id=response.response_id if response else None,
            finish_reason=response.finish_reason if response else None,
            system_fingerprint=response.system_fingerprint if response else None,
            environment_termination_reason=result.termination_reason if result else None,
        )
        steps.append(step)
        if result is None:
            break
        score = result.reward  # TextWorld scores are cumulative, not per-action increments.
        success_known = result.success is not None
        success = result.success is True
        observation = result.observation
        if result.done:
            reason = result.termination_reason or "environment"
            status = "completed" if success_known else "aborted"
            if not success_known:
                reason = "missing_success"
            break
    usage = TokenUsage(
        prompt_tokens=sum(step.token_usage.prompt_tokens for step in steps),
        completion_tokens=sum(step.token_usage.completion_tokens for step in steps),
    )
    trajectory = AgentTrajectory(
        trajectory_id=str(uuid4()), run_id=run_id, task_id=task.task_id, task=task,
        steps=steps, success=success, success_known=success_known,
        reward=score, total_reward=score, termination_reason=reason, status=status,
        model_id=settings.llm.model_id, prompt_version=settings.prompt_version,
        seed=environment_config.seed, wall_time=clock() - started,
        token_usage=usage, prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens, agent_config=settings,
        environment_config=environment_config, environment_backend_sha256=backend_hash,
    )
    recorder.append(trajectory)
    return trajectory
