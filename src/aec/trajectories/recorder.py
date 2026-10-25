"""Single-process append-only JSONL; never repair or overwrite existing raw data."""

import os
from pathlib import Path

from aec.trajectories.schemas import AgentTrajectory


class TrajectoryRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, trajectory: AgentTrajectory) -> None:
        # Revalidate even if callers have mutated an artifact since construction.
        line = AgentTrajectory.model_validate_json(trajectory.model_dump_json()).model_dump_json()
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    raise ValueError("Existing JSONL has an incomplete final line")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
