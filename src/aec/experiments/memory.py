"""Memory-policy hooks for the shared experiment runner.

Per task: retrieve after reset and log before acting; after a committed trajectory,
write memory before the task result commits. The append-only logs are the source of
truth; each SQLite store is a persistent index verified against them on every start.
Episodes and skills share the embedding model, similarity, ranking, and top-k.
"""

from pathlib import Path
from typing import Callable

from aec.config import MemoryConfig
from aec.experiments.artifacts import MemoryWrite, RetrievalLog, append_row, read_rows, utc_now
from aec.memory import episodes, skills
from aec.memory.episodes import EpisodeParseError, retrieval_text, writable
from aec.memory.store import MemoryStore, SkillStore
from aec.schemas import RetrievalRecord, Task
from aec.trajectories.schemas import AgentTrajectory

TERMINAL = {"stored", "rejected", "provider_failure", "embedding_failure", "interrupted"}


def token_counter(name: str) -> Callable[[str], int]:
    import tiktoken
    encoding = tiktoken.get_encoding(name)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def memory_policy(memory: MemoryConfig, *args, **kwargs) -> "MemoryPolicy":
    return {"episodic": EpisodicPolicy, "skill": SkillPolicy}[memory.policy](memory, *args, **kwargs)


class MemoryPolicy:
    writer_version: str

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
                    or row.writer_prompt_version != self.writer_version
                    or [r.status for r in rows] != ([] if row.status == "started" else ["started"])):
                raise ValueError("Memory-write log has foreign or out-of-order rows")
            rows.append(row)
        self.store = self._open_store()
        self._sync_store()

    # Subclass hooks.

    def _open_store(self): ...

    def _sync_store(self) -> None: ...

    def _record_ids(self, item) -> tuple[str, list[str]]: ...

    def _section(self, items) -> str | None: ...

    def _store_metrics(self) -> dict: ...

    def check_complete(self, trajectory: AgentTrajectory) -> None:
        """Called in order for trajectories whose task result is already committed."""

    def record(self, trajectory: AgentTrajectory) -> None:
        """Called in order for each newly committed trajectory, before its result."""

    # Shared machinery.

    def _stored_rows(self) -> list[MemoryWrite]:
        return [r for rows in self.writes.values() for r in rows if r.status == "stored"]

    def _finished(self, trajectory_id: str) -> bool:
        return any(r.status in TERMINAL for r in self.writes.get(trajectory_id, []))

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
            store_size = len(self.store.candidates())
            exclude = task_id if self.memory.exclude_same_task else None
            eligible = len(self.store.candidates(exclude))
            top, tokens, model_id = [], 0, self.memory.embedding.model_id
            if eligible:  # No query call when nothing could be retrieved.
                response = self._embed(query)
                tokens, model_id = response.prompt_tokens, response.model_id
                eligible, top = self.store.retrieve(response.vectors[0], top_k=self.memory.top_k,
                                                    exclude_task_id=exclude)
            ids = [self._record_ids(item) for item, _ in top]
            section = self._section([item for item, _ in top])
            row = RetrievalLog(
                run_id=self.run_id, task_id=task_id, query_text=query, store_size=store_size,
                eligible_count=eligible,
                records=[RetrievalRecord(task_id=task_id, memory_id=memory_id, similarity=s, rank=i)
                         for i, ((memory_id, _), (_, s)) in enumerate(zip(ids, top), start=1)],
                source_trajectory_ids=[sources for _, sources in ids],
                memory_section=section,
                memory_tokens=self.count_tokens(section) if section else 0,
                embedding_model_id=model_id, embedding_tokens=tokens, created_at=utc_now(),
            )
            append_row(self.retrievals_path, row)  # Durable before the first action.
            self.retrievals[task_id] = row
            return section
        return retrieve

    def _write(self, trigger: AgentTrajectory, prompt: str, parse, build, *, kind: str,
               sources: list[AgentTrajectory] | None = None):
        """started -> writer call -> strict parse -> embed -> stored. Returns the stored row.

        Every failure is logged as a terminal row and stores nothing. A started write
        found on resume is marked interrupted and never replayed.
        """
        rows = self.writes.setdefault(trigger.trajectory_id, [])
        base = dict(run_id=self.run_id, task_id=trigger.task_id,
                    trajectory_id=trigger.trajectory_id, writer_prompt_version=self.writer_version,
                    source_trajectory_ids=[t.trajectory_id for t in sources or []],
                    source_task_ids=[t.task_id for t in sources or []])

        def log(status, **fields):
            row = MemoryWrite(status=status, created_at=utc_now(), **base, **fields)
            append_row(self.writes_path, row)
            rows.append(row)
            return row

        if any(r.status in TERMINAL for r in rows):
            return None
        if rows:  # Never repeat a potentially billable writer call after a crash.
            log("interrupted")
            return None
        log("started", prompt=prompt)
        llm = None
        try:
            llm = self.llm_factory()
            response = llm.generate(prompt, self.memory.writer.llm)
        except Exception as exc:
            log("provider_failure", prompt=prompt, error_type=type(exc).__name__)
            return None
        finally:
            if llm is not None:
                llm.close()
        call = dict(prompt=prompt, output=response.text, response_model_id=response.model_id,
                    response_id=response.response_id, finish_reason=response.finish_reason,
                    token_usage=response.token_usage, usage_known=True)
        try:
            if response.finish_reason != "stop":
                raise EpisodeParseError("Writer did not finish a complete response")
            item = build(parse(response.text))
        except (EpisodeParseError, ValueError) as exc:
            # ValueError covers schema validation of the built memory (malformed memory).
            log("rejected", error=str(exc)[:500], **call)
            return None
        try:
            embedded = self._embed(item.embedding_text)
        except Exception as exc:
            log("embedding_failure", error_type=type(exc).__name__, **call)
            return None
        return log("stored", embedding=embedded.vectors[0], embedding_model_id=embedded.model_id,
                   embedding_tokens=embedded.prompt_tokens, **{kind: item}, **call)

    def metrics(self) -> dict:
        retrievals = list(self.retrievals.values())
        finished = [r for rows in self.writes.values() for r in rows if r.status in TERMINAL]

        def count(status):
            return sum(r.status == status for r in finished)

        return dict(
            policy=self.memory.policy, retrieval_logs=len(retrievals),
            tasks_with_retrieved_memory=sum(bool(r.records) for r in retrievals),
            retrieved_memories=sum(len(r.records) for r in retrievals),
            injected_memory_tokens=sum(r.memory_tokens for r in retrievals),
            query_embedding_tokens=sum(r.embedding_tokens for r in retrievals),
            write_attempts=len(finished), writes_stored=count("stored"),
            writes_rejected=count("rejected"), writer_provider_failures=count("provider_failure"),
            embedding_failures=count("embedding_failure"), interrupted_writes=count("interrupted"),
            writer_prompt_tokens=sum(r.token_usage.prompt_tokens for r in finished),
            writer_completion_tokens=sum(r.token_usage.completion_tokens for r in finished),
            writer_usage_known=all(r.usage_known for r in finished),
            write_embedding_tokens=sum(r.embedding_tokens for r in finished),
            **self._store_metrics(),
        )

    def close(self) -> None:
        self.store.close()


class EpisodicPolicy(MemoryPolicy):
    """One episode per successful trajectory."""

    writer_version = episodes.WRITER_VERSION

    def _open_store(self):
        return MemoryStore(self.memory.store_path, embedding=self.memory.embedding,
                           run_id=self.run_id)

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

    def _record_ids(self, item):
        return item.memory.memory_id, list(item.memory.source_trajectory_ids)

    def _section(self, items):
        return episodes.memory_section([item.memory for item in items])

    def check_complete(self, trajectory: AgentTrajectory) -> None:
        if writable(trajectory) and not self._finished(trajectory.trajectory_id):
            raise ValueError("Committed result lacks its finished memory write")

    def record(self, trajectory: AgentTrajectory) -> None:
        if not writable(trajectory):
            return
        prompt = episodes.build_writer_prompt(trajectory)
        row = self._write(trajectory, prompt, episodes.parse_episode, lambda fields: (
            episodes.build_episode(trajectory, fields, retrieval_key=self.memory.retrieval_key,
                                   count_tokens=self.count_tokens)), kind="memory")
        if row is not None:
            self.store.add(row.memory, source_task_id=trajectory.task_id, embedding=row.embedding)

    def _store_metrics(self) -> dict:
        stored = self.store.episodes()
        return dict(episodes_stored=sum(r.memory is not None for r in self._stored_rows()),
                    store_episodes=len(stored),
                    store_memory_tokens=sum(e.memory.token_count for e in stored))


class SkillPolicy(MemoryPolicy):
    """Fixed rule: once a task family has >= N successful trajectories in this run,
    (re)synthesize its skill from all of them, oldest first, as a new version."""

    writer_version = skills.WRITER_VERSION

    def __init__(self, *args, **kwargs):
        self.sources: dict[str, list[AgentTrajectory]] = {}
        super().__init__(*args, **kwargs)

    def _open_store(self):
        return SkillStore(self.memory.store_path, embedding=self.memory.embedding,
                          run_id=self.run_id)

    def _sync_store(self) -> None:
        logged = self._stored_rows()
        present = self.store.skills()
        if len(present) > len(logged) or any(
            (s.skill, s.source_task_ids, s.embedding) != (r.skill, r.source_task_ids, r.embedding)
            for s, r in zip(present, logged)
        ):
            raise ValueError("Skill store differs from the append-only write log")
        for row in logged[len(present):]:  # Crash between log append and store insert.
            self.store.add(row.skill, embedding=row.embedding,
                           source_task_ids=row.source_task_ids)

    def _record_ids(self, item):
        return skills.record_id(item.skill), list(item.skill.source_trajectory_ids)

    def _section(self, items):
        return skills.skill_section([item.skill for item in items])

    def _register(self, trajectory: AgentTrajectory) -> list[AgentTrajectory] | None:
        """Add a success to its family; return the sources if synthesis is due."""
        if not writable(trajectory):
            return None
        group = self.sources.setdefault(skills.task_family(trajectory.task), [])
        group.append(trajectory)
        return list(group) if len(group) >= self.memory.skill_min_sources else None

    def check_complete(self, trajectory: AgentTrajectory) -> None:
        if self._register(trajectory) and not self._finished(trajectory.trajectory_id):
            raise ValueError("Committed result lacks its finished skill synthesis")

    def record(self, trajectory: AgentTrajectory) -> None:
        sources = self._register(trajectory)
        if sources is None:
            return
        family = skills.task_family(trajectory.task)
        versions = [s.skill.version for s in self.store.skills()
                    if s.skill.skill_id == f"skill-{family}"]
        version = max(versions, default=0) + 1
        prompt = skills.build_skill_writer_prompt(sources)
        row = self._write(
            trajectory, prompt, skills.parse_skill,
            lambda fields: skills.build_skill(sources, fields, version=version,
                                              retrieval_key=self.memory.retrieval_key,
                                              count_tokens=self.count_tokens),
            kind="skill", sources=sources)
        if row is not None:
            self.store.add(row.skill, embedding=row.embedding,
                           source_task_ids=[t.task_id for t in sources])

    def _store_metrics(self) -> dict:
        latest = self.store.latest()
        return dict(skill_versions_stored=len(self.store.skills()), store_skills=len(latest),
                    store_memory_tokens=sum(s.skill.token_count for s in latest))
