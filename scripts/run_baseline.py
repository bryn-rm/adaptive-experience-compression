"""Create a fixed task manifest or execute/resume the Phase 3 no-memory baseline."""

import argparse
import json

from aec.environments.alfworld import available_task_ids
from aec.experiments.artifacts import create_manifest, load_baseline_config
from aec.experiments.baseline import run_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["manifest", "run"])
    parser.add_argument("--config", default="configs/baseline_pilot.yaml")
    args = parser.parse_args()
    config = load_baseline_config(args.config)
    if args.command == "manifest":
        if config.manifest.exists():
            parser.error("Manifest exists; it cannot be overwritten")
        manifest = create_manifest(config, available_task_ids(config.environment))
        print(manifest.model_dump_json(indent=2))
    else:
        print(json.dumps(run_baseline(config), indent=2))


if __name__ == "__main__":
    main()
