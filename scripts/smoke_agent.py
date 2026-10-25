"""Small Phase 2 validation run, without manifest/resume/baseline machinery."""

import argparse
from pathlib import Path
from uuid import uuid4

from aec.agents.runner import run_episode
from aec.config import load_config
from aec.environments.alfworld import ALFWorldAdapter
from aec.llm.openai import OpenAIClient
from aec.logging import configure_logging
from aec.trajectories.recorder import TrajectoryRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agent_smoke.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.environment is None or config.agent is None:
        parser.error("Configuration must include environment and agent settings")
    if args.episodes < 1:
        parser.error("episodes must be positive")
    configure_logging(config.log_level)
    recorder = TrajectoryRecorder(args.output)
    run_id = str(uuid4())
    llm = OpenAIClient()
    try:
        env = ALFWorldAdapter(config.environment)
        try:
            for _ in range(args.episodes):
                trajectory = run_episode(env, llm, config.agent, recorder, run_id=run_id)
                print(f"{trajectory.trajectory_id}: task={trajectory.task_id} "
                      f"status={trajectory.status} success={trajectory.success} "
                      f"reason={trajectory.termination_reason}")
        finally:
            env.close()
    finally:
        llm.close()


if __name__ == "__main__":
    main()
