"""Validated configuration loading; experiment settings are added in later phases."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ALFWorldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_config: Path
    split: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "eval_out_of_distribution"
    max_steps: int = Field(default=50, ge=1, strict=True)
    seed: int = 0


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: ALFWorldConfig | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return Config.model_validate(data)
