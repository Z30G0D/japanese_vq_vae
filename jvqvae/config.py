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
    """How to divide a source into train, validation and test.

    ``val_writers`` and ``test_writers`` apply to sources that carry writer
    identities; ``val_fraction`` is the fallback for sources that do not.
    """

    seed: int = 0
    val_writers: int = 12
    test_writers: int = 12
    val_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.val_writers < 0 or self.test_writers < 0:
            raise ValueError(
                f"writer counts must not be negative, got "
                f"{self.val_writers} val and {self.test_writers} test"
            )
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must be in [0, 1), got {self.val_fraction}"
            )


SOURCES = ("etl4", "k49")


@dataclass(frozen=True)
class DataConfig:
    """Which dataset to load and at what size.

    ``image_size`` means different things per source, because the two differ in
    native resolution: for ETL4 it is a bilinear resize target, while for K49 it
    is a zero-pad canvas around the native 28x28 and so must be at least 28.
    """

    source: str
    image_size: int
    cache_dir: str = "data/cache"
    etl4_zip: str | None = None
    k49_dir: str = "data/k49"
    k49_max_per_class: int | None = None
    split: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(
                f"source must be one of {', '.join(SOURCES)}, got {self.source!r}"
            )
        if self.image_size < 1:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.k49_max_per_class is not None and self.k49_max_per_class < 1:
            raise ValueError(
                f"k49_max_per_class must be positive when set, "
                f"got {self.k49_max_per_class}"
            )


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
