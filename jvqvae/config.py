"""YAML-backed configuration.

Typos in a config file are a common and expensive way to waste a training run,
so unknown keys are rejected rather than ignored.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    val_writers: int = 12
    test_writers: int = 12


@dataclass(frozen=True)
class DataConfig:
    source: str
    image_size: int
    cache_dir: str = "data/cache"
    etl4_zip: str | None = None
    k49_dir: str = "data/k49"
    k49_max_per_class: int | None = None
    split: SplitConfig = field(default_factory=SplitConfig)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {', '.join(sorted(unknown))}"
        )
    return cls(**raw)


def load_data_config(path: str | Path) -> DataConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw["split"] = _build(SplitConfig, raw.get("split") or {})
    return _build(DataConfig, raw)
