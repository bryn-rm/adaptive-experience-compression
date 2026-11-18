"""Write an immutable Gate A report: no memory vs episodes vs skills on one manifest."""

import argparse
from pathlib import Path

from aec.experiments.artifacts import write_once
from aec.experiments.gate_a import gate_a_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-memory', type=Path, required=True)
    parser.add_argument('--episodic', type=Path, required=True)
    parser.add_argument('--skill', type=Path, required=True)
    parser.add_argument('--no-memory-replicate', type=Path,
                        help='second no-memory run: run-to-run noise floor')
    parser.add_argument('--max-span-hours', type=float, default=24,
                        help='all attempts must start within this window to count as controlled')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_once(args.output, gate_a_report(
        args.no_memory, args.episodic, args.skill, args.no_memory_replicate,
        max_span_hours=args.max_span_hours))


if __name__ == '__main__':
    main()
