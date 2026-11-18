"""Persistent SQLite memory stores with brute-force cosine retrieval.

Each store is an insert-only index of its run's append-only memory-write log.
It is bound to one embedding configuration and one writing run. Episodes and
skills share the same ranking: similarity descending, ties by insertion order.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Sequence, TypeVar

from aec.config import EmbeddingConfig
from aec.memory.embeddings import cosine
from aec.schemas import EpisodicMemory, ProceduralSkill

EPISODE_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL UNIQUE,
    source_trajectory_id TEXT NOT NULL UNIQUE,
    source_task_id TEXT NOT NULL,
    memory TEXT NOT NULL,
    embedding TEXT NOT NULL
);
"""
SKILL_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    source_task_ids TEXT NOT NULL,
    skill TEXT NOT NULL,
    embedding TEXT NOT NULL,
    UNIQUE (skill_id, version)
);
"""


@dataclass(frozen=True)
class StoredEpisode:
    seq: int
    memory: EpisodicMemory
    source_task_id: str
    embedding: list[float]


@dataclass(frozen=True)
class StoredSkill:
    seq: int
    skill: ProceduralSkill
    source_task_ids: list[str]
    embedding: list[float]


Item = TypeVar("Item", StoredEpisode, StoredSkill)


def rank(items: Sequence[Item], query: list[float], top_k: int) -> list[tuple[Item, float]]:
    scored = sorted(((item, cosine(query, item.embedding)) for item in items),
                    key=lambda pair: (-pair[1], pair[0].seq))
    return scored[:top_k]


class _Store:
    kind: str
    schema: str

    def __init__(self, path: Path, *, embedding: EmbeddingConfig, run_id: str | None,
                 read_only: bool = False):
        """`run_id` is the only run allowed to write; read-only opens may omit it."""
        self.path = Path(path)
        self.read_only = read_only
        meta = {"embedding": embedding.model_dump_json(), "memory": self.kind}
        if read_only:
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            if not run_id:
                raise ValueError("A writable store needs its owning run ID")
            meta["owner_run_id"] = run_id
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path)
            with self._db:
                self._db.executescript(
                    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                    + self.schema)
                for key, value in meta.items():
                    self._db.execute("INSERT OR IGNORE INTO meta VALUES (?, ?)", (key, value))
        stored = dict(self._db.execute("SELECT key, value FROM meta"))
        for key, value in meta.items():
            if stored.get(key) != value:
                self._db.close()
                raise ValueError(f"Memory store {key} differs from this experiment")

    def _writable(self) -> None:
        if self.read_only:
            raise ValueError("Store opened read-only")

    def close(self) -> None:
        self._db.close()


class MemoryStore(_Store):
    kind, schema = "episodic_store_v1", EPISODE_SCHEMA

    def add(self, memory: EpisodicMemory, *, source_task_id: str, embedding: list[float]) -> None:
        self._writable()
        if len(memory.source_trajectory_ids) != 1:
            raise ValueError("Episodes have exactly one source trajectory")
        with self._db:
            self._db.execute(
                "INSERT INTO episodes (memory_id, source_trajectory_id, source_task_id, memory, "
                "embedding) VALUES (?, ?, ?, ?, ?)",
                (memory.memory_id, memory.source_trajectory_ids[0], source_task_id,
                 memory.model_dump_json(), json.dumps(embedding)),
            )

    def episodes(self) -> list[StoredEpisode]:
        rows = self._db.execute(
            "SELECT seq, memory, source_task_id, embedding FROM episodes ORDER BY seq")
        return [StoredEpisode(seq, EpisodicMemory.model_validate_json(memory), task,
                              json.loads(vector)) for seq, memory, task, vector in rows]

    def candidates(self, exclude_task_id: str | None = None) -> list[StoredEpisode]:
        return [e for e in self.episodes() if e.source_task_id != exclude_task_id]

    def retrieve(self, query: list[float], *, top_k: int,
                 exclude_task_id: str | None = None) -> tuple[int, list[tuple[StoredEpisode, float]]]:
        """Return (eligible count, top-k by similarity; ties broken by insertion order)."""
        eligible = self.candidates(exclude_task_id)
        return len(eligible), rank(eligible, query, top_k)


class SkillStore(_Store):
    """All skill versions are kept; retrieval sees only each skill's latest version."""

    kind, schema = "skill_store_v1", SKILL_SCHEMA

    def add(self, skill: ProceduralSkill, *, source_task_ids: list[str],
            embedding: list[float]) -> None:
        self._writable()
        if len(source_task_ids) != len(skill.source_trajectory_ids):
            raise ValueError("Each skill source trajectory needs its task ID")
        latest = {s.skill.skill_id: s.skill.version for s in self.latest()}
        if skill.version != latest.get(skill.skill_id, 0) + 1:
            raise ValueError("Skill versions must be contiguous")
        with self._db:
            self._db.execute(
                "INSERT INTO skills (skill_id, version, source_task_ids, skill, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (skill.skill_id, skill.version, json.dumps(source_task_ids),
                 skill.model_dump_json(), json.dumps(embedding)),
            )

    def skills(self) -> list[StoredSkill]:
        rows = self._db.execute(
            "SELECT seq, skill, source_task_ids, embedding FROM skills ORDER BY seq")
        return [StoredSkill(seq, ProceduralSkill.model_validate_json(skill), json.loads(tasks),
                            json.loads(vector)) for seq, skill, tasks, vector in rows]

    def latest(self) -> list[StoredSkill]:
        current: dict[str, StoredSkill] = {}
        for item in self.skills():
            current[item.skill.skill_id] = item
        return sorted(current.values(), key=lambda item: item.seq)

    def candidates(self, exclude_task_id: str | None = None) -> list[StoredSkill]:
        return [s for s in self.latest() if exclude_task_id not in s.source_task_ids]

    def retrieve(self, query: list[float], *, top_k: int,
                 exclude_task_id: str | None = None) -> tuple[int, list[tuple[StoredSkill, float]]]:
        eligible = self.candidates(exclude_task_id)
        return len(eligible), rank(eligible, query, top_k)
