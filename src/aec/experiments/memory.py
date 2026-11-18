"""Episodic-memory policy hooks for the shared experiment runner.

Per task: retrieve after reset and log before acting; after a committed successful
trajectory, write one episode. The append-only logs are the source of truth; the
SQLite store is a persistent index that is verified against them on every start.
"""

from pathlib import Path
from typing import Callable

from aec.config import MemoryConfig
from aec.experiments.artifacts import MemoryWrite, RetrievalLog, append_row, read_rows, utc_now
from aec.memory.episodes import (
    WRITER_VERSION, EpisodeParseError, build_episode, build_writer_prompt, memory_section,
    parse_episode, retrieval_text, writable,
)
from aec.memory.store import MemoryStore
from aec.schemas import RetrievalRecord, Task
from aec.trajectories.schemas import AgentTrajectory

TERMINAL = {"stored", "rejected", "provider_failure", "embedding_failure", "interrupted"}


def token_counter(name: str) -> Callable[[str], int]:
    import tiktoken
    encoding = tiktoken.get_encoding(name)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


class EpisodicPolicy:
    def __init__(self, memory: MemoryConfig, run_id: str, directory: Path, task_ids, *,
                 fresh: bool, embedder, llm_factory, count_tokens: Callable[[str], int]):
        if fresh and memory.store_path.exists():
            raise ValueError("Memory store predates this run; use a new store path")
        self.memory, self.run_id = memory, run_id
        self.embedder, self.llm_factory, self.count_tokens = embedder, llm_factory, count_tokens
        self.retrievals_path = directory / "retrievals.jsonl"
        self.writes_path = directory / "memory_writes.jsonl"
        self.retrievals: dict[str, RetrievalLog] = {}
        for row in read_rows(self.retrievals_path, RetrievalLog):
            if row.run_id != run_id or row.task_id not in task_ids or row.task_id in self.retrievals:
                raise ValueError("Retrieval log has foreign or duplicate rows")
            self.retrievals[row.task_id] = row
        self.writes: dict[str, list[MemoryWrite]] = {}
        for row in read_rows(self.writes_path, MemoryWrite):
            rows = self.writes.setdefault(row.trajectory_id, [])
            if (row.run_id != run_id or row.task_id not in task_ids
                    or [r.status for r in rows] != ([] if row.status == "started" else ["started"])):
                raise ValueError("Memory-write log has foreign or out-of-order rows")
            rows.append(row)
        self.store = MemoryStore(memory.store_path, embedding=memory.embedding, run_id=run_id)
        self._sync_store()

    def _stored_rows(self) -> list[MemoryWrite]:
        return [r for rows in self.writes.values() for r in rows if r.status == "stored"]

    def _sync_store(self) -> None:
        logged = self._stored_rows()
        present = self.store.episodes()
        if len(present) > len(logged) or any(
            (e.memory, e.source_task_id, e.embedding) != (r.memory, r.task_id, r.embedding)
            for e, r in zip(present, logged)
        ):
            raise ValueError("Memory store differs from the append-only write log")
        for row in logged[len(present):]:  # Crash between log append and store insert.
            self.store.add(row.memory, source_task_id=row.task_id, embedding=row.embedding)

    def check_complete(self, trajectory: AgentTrajectory) -> None:
        """A committed task result implies its episode write already finished."""
        if writable(trajectory) and not any(
            r.status in TERMINAL for r in self.writes.get(trajectory.trajectory_id, [])
        ):
            raise ValueError("Committed result lacks its finished memory write")

    def _embed(self, text: str):
        response = self.embedder.embed([text], self.memory.embedding)
        if response.model_id != self.memory.embedding.model_id or len(response.vectors) != 1:
            raise ValueError("Embedding provider returned a different model or batch")
        return response

    def retriever(self, task_id: str) -> Callable[[Task], str | None]:
        def retrieve(task: Task) -> str | None:
            if task.task_id != task_id or task_id in self.retrievals:
                raise ValueError("Retrieval must happen once for the manifest task")
            query = retrieval_text(task, self.memory.retrieval_key)
            store_size = len(self.store.episodes())
            exclude = task_id if self.memory.exclude_same_task else None
            eligible = len([e for e in self.store.episodes() if e.source_task_id != exclude])
            top, tokens, model_id = [], 0, self.memory.embedding.model_id
            if eligible:  # No query call when nothing could be retrieved.
                response = self._embed(query)
                tokens, model_id = response.prompt_tokens, response.model_id
                eligible, top = self.store.retrieve(response.vectors[0], top_k=self.memory.top_k,
                                                    exclude_task_id=exclude)
            section = memory_section([e.memory for e, _ in top])
            row = RetrievalLog(
                run_id=self.run_id, task_id=task_id, query_text=query, store_size=store_size,
                eligible_count=eligible,
                records=[RetrievalRecord(task_id=task_id, memory_id=e.memory.memory_id,
                                         similarity=s, rank=i)
                         for i, (e, s) in enumerate(top, start=1)],
                source_trajectory_ids=[e.memory.source_trajectory_ids[0] for e, _ in top],
                memory_section=section,
                memory_tokens=self.count_tokens(section) if section else 0,
                embedding_model_id=model_id, embedding_tokens=tokens, created_at=utc_now(),
            )
            append_row(self.retrievals_path, row)  # Durable before the first action.
            self.retrievals[task_id] = row
            return section
        return retrieve

    def record_episode(self, trajectory: AgentTrajectory) -> None:
        if not writable(trajectory):
            return
        rows = self.writes.setdefault(trajectory.trajectory_id, [])
        base = dict(run_id=self.run_id, task_id=trajectory.task_id,
                    trajectory_id=trajectory.trajectory_id, writer_prompt_version=WRITER_VERSION)

        def log(status, **fields):
            row = MemoryWrite(status=status, created_at=utc_now(), **base, **fields)
            append_row(self.writes_path, row)
            rows.append(row)
            return row

        if any(r.status in TERMINAL for r in rows):
            return
        if rows:  # Never repeat a potentially billable writer call after a crash.
            log("interrupted")
            return
        prompt = build_writer_prompt(trajectory)
        log("started", prompt=prompt)
        llm = None
        try:
            llm = self.llm_factory()
            response = llm.generate(prompt, self.memory.writer.llm)
        except Exception as exc:
            log("provider_failure", prompt=prompt, error_type=type(exc).__name__)
            return
        finally:
            if llm is not None:
                llm.close()
        call = dict(prompt=prompt, output=response.text, response_model_id=response.model_id,
                    response_id=response.response_id, finish_reason=response.finish_reason,
                    token_usage=response.token_usage, usage_known=True)
        try:
            if response.finish_reason != "stop":
                raise EpisodeParseError("Writer did not finish a complete response")
            fields = parse_episode(response.text)
        except EpisodeParseError as exc:
            log("rejected", error=str(exc), **call)
            return
        episode = build_episode(trajectory, fields, retrieval_key=self.memory.retrieval_key,
                                count_tokens=self.count_tokens)
        try:
            embedded = self._embed(episode.embedding_text)
        except Exception as exc:
            log("embedding_failure", error_type=type(exc).__name__, **call)
            return
        row = log("stored", memory=episode, embedding=embedded.vectors[0],
                  embedding_model_id=embedded.model_id, embedding_tokens=embedded.prompt_tokens,
                  **call)
        self.store.add(episode, source_task_id=trajectory.task_id, embedding=row.embedding)

    def metrics(self) -> dict:
        retrievals = list(self.retrievals.values())
        finished = [r for rows in self.writes.values() for r in rows if r.status in TERMINAL]
        stored = self.store.episodes()

        def count(status):
            return sum(r.status == status for r in finished)

        return dict(
            policy=self.memory.policy, retrieval_logs=len(retrievals),
            tasks_with_retrieved_memory=sum(bool(r.records) for r in retrievals),
            retrieved_memories=sum(len(r.records) for r in retrievals),
            injected_memory_tokens=sum(r.memory_tokens for r in retrievals),
            query_embedding_tokens=sum(r.embedding_tokens for r in retrievals),
            write_attempts=len(finished), episodes_stored=count("stored"),
            writes_rejected=count("rejected"), writer_provider_failures=count("provider_failure"),
            embedding_failures=count("embedding_failure"), interrupted_writes=count("interrupted"),
            writer_prompt_tokens=sum(r.token_usage.prompt_tokens for r in finished),
            writer_completion_tokens=sum(r.token_usage.completion_tokens for r in finished),
            writer_usage_known=all(r.usage_known for r in finished),
            write_embedding_tokens=sum(r.embedding_tokens for r in finished),
            store_episodes=len(stored),
            store_memory_tokens=sum(e.memory.token_count for e in stored),
        )

    def close(self) -> None:
        self.store.close()
