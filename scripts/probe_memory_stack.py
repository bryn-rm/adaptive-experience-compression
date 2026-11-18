"""One embedding call and one writer call on recorded successful trajectories.

Episodic configs write one episode; skill configs synthesize one skill from the first
task family with at least `skill_min_sources` successes.

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
from aec.memory.skills import build_skill_writer_prompt, parse_skill, task_family
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
    successes = [t for t in map(AgentTrajectory.model_validate_json,
                                Path(args.source).read_text().splitlines()) if writable(t)]
    if memory.policy == "skill":
        families: dict[str, list[AgentTrajectory]] = {}
        for t in successes:
            families.setdefault(task_family(t.task), []).append(t)
        sources = next(g[:memory.skill_min_sources] for g in families.values()
                       if len(g) >= memory.skill_min_sources)
        prompt, parse = build_skill_writer_prompt(sources), parse_skill
    else:
        sources = successes[:1]
        prompt, parse = build_writer_prompt(sources[0]), parse_episode
    trajectory = sources[-1]
    query = retrieval_text(trajectory.task, memory.retrieval_key)
    embedder = OpenAIEmbedder()
    embedded = embedder.embed([query], memory.embedding)
    embedder.close()
    llm = OpenAIClient()
    response = llm.generate(prompt, memory.writer.llm)
    llm.close()
    try:
        fields, error = parse(response.text), None
    except EpisodeParseError as exc:
        fields, error = None, str(exc)
    report = dict(
        created_at=utc_now().isoformat(), config=args.config,
        policy=memory.policy, writer_prompt_version=memory.writer.prompt_version,
        source_trajectory_ids=[t.trajectory_id for t in sources], query_text=query,
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
