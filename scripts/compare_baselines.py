"""Write an immutable, descriptive task-level comparison of two completed runs."""

import argparse
from pathlib import Path

from aec.experiments.artifacts import write_once
from aec.experiments.comparison import compare_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_once(args.output, compare_runs(args.previous, args.candidate))


if __name__ == '__main__':
    main()
