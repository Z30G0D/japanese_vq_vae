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


def _build(cls: type, raw: Any, *, where: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(
            f"{where}: expected a mapping for {cls.__name__}, "
            f"got {type(raw).__name__}"
        )

    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) for {cls.__name__}: "
            f"{', '.join(sorted(unknown))}"
        )

    try:
        return cls(**raw)
    except TypeError as error:
        raise ValueError(f"{where}: invalid {cls.__name__}: {error}") from error


def load_data_config(path: str | Path) -> DataConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(raw).__name__}"
        )

    split_raw = raw.get("split")
    raw["split"] = _build(
        SplitConfig, {} if split_raw is None else split_raw, where=f"{path}:split"
    )
    return _build(DataConfig, raw, where=str(path))
