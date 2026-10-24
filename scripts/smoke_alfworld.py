"""Seeded random-policy text smoke run; output is exclusive-create JSONL."""

import argparse
import json
from pathlib import Path
import random

from aec.config import load_config
from aec.environments.alfworld import ALFWorldAdapter
from aec.logging import configure_logging


def run_smoke(env: ALFWorldAdapter, output: Path, *, episodes: int, seed: int) -> None:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    rng = random.Random(seed)
    # Fail if output exists: raw observations must never be overwritten.
    with output.open("x", encoding="utf-8") as stream:
        for episode in range(episodes):
            initial = env.reset()
            stream.write(json.dumps({**initial.model_dump(mode="json"),
                                     "task_id": initial.task.task_id,
                                     "episode": episode, "seed": seed, "event": "reset"}) + "\n")
            while True:
                actions = sorted(env.valid_actions())
                result = env.step(rng.choice(actions) if actions else "look")
                stream.write(json.dumps({**result.model_dump(mode="json"),
                                         "episode": episode, "event": "step"}) + "\n")
                stream.flush()
                if result.done:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alfworld_smoke.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config.log_level)
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if config.environment is None:
        parser.error("Configuration must include environment settings")
    env = ALFWorldAdapter(config.environment)
    try:
        run_smoke(env, args.output, episodes=args.episodes, seed=config.environment.seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
