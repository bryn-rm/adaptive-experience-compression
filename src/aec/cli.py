"""CLI scaffold: load and validate configuration without running experiments."""

import argparse
import logging

from aec.config import Config, load_config
from aec.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a YAML configuration file")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else Config()
    configure_logging(config.log_level)
    logging.getLogger(__name__).info("Configuration validated; scaffold ready")
    return 0
