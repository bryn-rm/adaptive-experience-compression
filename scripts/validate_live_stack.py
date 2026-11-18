"""One real reset/prompt/OpenAI/parse/step cycle; never baseline performance data."""

import argparse
from pathlib import Path

from aec.agents.actions import ActionParseError, parse_action
from aec.agents.prompt import build_prompt
from aec.agents.v2 import build_prompt_v2, parse_action_v2
from aec.config import load_config
from aec.environments.alfworld import ALFWorldAdapter
from aec.experiments.artifacts import TaskManifest, load_baseline_config, write_once
from aec.llm.openai import OpenAIClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agent_smoke.yaml")
    parser.add_argument("--baseline-config", action="store_true",
                        help="Load an experiment config and validate its first fixed manifest task")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preflight output already exists")
    config = (load_baseline_config if args.baseline_config else load_config)(args.config)
    if config.environment is None or config.agent is None:
        parser.error("Environment and agent configuration are required")
    task_id = None
    if args.baseline_config:
        task_id = TaskManifest.model_validate_json(config.manifest.read_text()).task_ids[0]
    env = ALFWorldAdapter(config.environment, task_id=task_id)
    llm = None
    try:
        llm = OpenAIClient()
        initial = env.reset()
        actions = sorted(env.valid_actions())
        prompt = (build_prompt_v2(initial.task, initial.observation, actions, [],
                                 history_length=config.agent.history_length)
                  if config.agent.prompt_version == "agent_v2"
                  else build_prompt(initial.task, initial.observation, actions))
        response = llm.generate(prompt, config.agent.llm)
        if response.finish_reason != "stop":
            raise ActionParseError("Provider did not finish a complete action response")
        normalized = False
        if config.agent.prompt_version == "agent_v2":
            parsed = parse_action_v2(response.text, actions)
            action, normalized = parsed.action, parsed.normalized
        else:
            action = parse_action(response.text, actions)
        result = env.step(action)
        write_once(args.output, dict(
            validation="real ALFWorld/OpenAI single cycle; not baseline data",
            config=config.model_dump(mode="json"), initial=initial.model_dump(mode="json"),
            prompt=prompt, response=response.model_dump(mode="json"), action=action,
            result=result.model_dump(mode="json"), action_normalized=normalized,
        ))
        print(f"Live cycle passed: {action}; model={response.model_id}")
    finally:
        env.close()
        if llm is not None:
            llm.close()


if __name__ == "__main__":
    main()
