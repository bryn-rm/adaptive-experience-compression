"""One embedding call and one episode-writer call on a recorded successful trajectory.

Checks provider compatibility before a memory run. Excluded from all run metrics.
"""

import argparse
from pathlib import Path

from aec.experiments.artifacts import load_baseline_config, utc_now, write_once
from aec.llm.openai import OpenAIClient
from aec.memory.embeddings import OpenAIEmbedder
from aec.memory.episodes import (
    EpisodeParseError, build_writer_prompt, parse_episode, retrieval_text, writable,
)
from aec.trajectories.schemas import AgentTrajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/episodic_pilot.yaml")
    parser.add_argument("--source", required=True, help="trajectories.jsonl of a finished run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    memory = load_baseline_config(args.config).memory
    trajectory = next(t for t in map(AgentTrajectory.model_validate_json,
                                     Path(args.source).read_text().splitlines()) if writable(t))
    query = retrieval_text(trajectory.task, memory.retrieval_key)
    embedder = OpenAIEmbedder()
    embedded = embedder.embed([query], memory.embedding)
    embedder.close()
    llm = OpenAIClient()
    response = llm.generate(build_writer_prompt(trajectory), memory.writer.llm)
    llm.close()
    try:
        fields, error = parse_episode(response.text), None
    except EpisodeParseError as exc:
        fields, error = None, str(exc)
    report = dict(
        created_at=utc_now().isoformat(), config=args.config,
        source_trajectory_id=trajectory.trajectory_id, query_text=query,
        embedding=dict(requested=memory.embedding.model_id, returned=embedded.model_id,
                       dimensions=len(embedded.vectors[0]), prompt_tokens=embedded.prompt_tokens),
        writer=dict(requested=memory.writer.llm.model_id, returned=response.model_id,
                    finish_reason=response.finish_reason, output=response.text,
                    parsed=fields, parse_error=error,
                    token_usage=response.token_usage.model_dump()),
    )
    write_once(args.output, report)
    print(args.output.read_text())


if __name__ == "__main__":
    main()
