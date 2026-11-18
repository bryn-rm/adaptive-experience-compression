"""Persistent SQLite episode store with brute-force cosine retrieval.

The store is an insert-only index of the run's append-only memory-write log.
It is bound to one embedding configuration and one writing run.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from aec.config import EmbeddingConfig
from aec.memory.embeddings import cosine
from aec.schemas import EpisodicMemory

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episodes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL UNIQUE,
    source_trajectory_id TEXT NOT NULL UNIQUE,
    source_task_id TEXT NOT NULL,
    memory TEXT NOT NULL,
    embedding TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class StoredEpisode:
    seq: int
    memory: EpisodicMemory
    source_task_id: str
    embedding: list[float]


class MemoryStore:
    def __init__(self, path: Path, *, embedding: EmbeddingConfig, run_id: str | None,
                 read_only: bool = False):
        """`run_id` is the only run allowed to write; read-only opens may omit it."""
        self.path = Path(path)
        self.read_only = read_only
        meta = {"embedding": embedding.model_dump_json(), "memory": "episodic_store_v1"}
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
                self._db.executescript(SCHEMA)
                for key, value in meta.items():
                    self._db.execute("INSERT OR IGNORE INTO meta VALUES (?, ?)", (key, value))
        stored = dict(self._db.execute("SELECT key, value FROM meta"))
        for key, value in meta.items():
            if stored.get(key) != value:
                self._db.close()
                raise ValueError(f"Memory store {key} differs from this experiment")

    def add(self, memory: EpisodicMemory, *, source_task_id: str, embedding: list[float]) -> None:
        if self.read_only:
            raise ValueError("Store opened read-only")
        if len(memory.source_trajectory_ids) != 1:
            raise ValueError("Phase 4 episodes have exactly one source trajectory")
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

    def retrieve(self, query: list[float], *, top_k: int,
                 exclude_task_id: str | None = None) -> tuple[int, list[tuple[StoredEpisode, float]]]:
        """Return (eligible count, top-k by similarity; ties broken by insertion order)."""
        eligible = [e for e in self.episodes() if e.source_task_id != exclude_task_id]
        scored = sorted(((e, cosine(query, e.embedding)) for e in eligible),
                        key=lambda pair: (-pair[1], pair[0].seq))
        return len(eligible), scored[:top_k]

    def close(self) -> None:
        self._db.close()
