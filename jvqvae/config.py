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


@dataclass(frozen=True)
class VQVAEConfig:
    """Shape and behaviour of the VQ-VAE.

    ``num_downsample`` sets how far the encoder reduces the image, so with an
    ``image_size`` of 64 three stages give the 8x8 code grid the prior expects.
    It is a field rather than a constant so a 16x16 grid is available as an
    ablation without touching the model.
    """

    num_codes: int = 512
    embedding_dim: int = 64
    hidden_channels: int = 128
    residual_hidden: int = 32
    num_residual_blocks: int = 2
    num_downsample: int = 3
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    dead_code_steps: int = 50

    def __post_init__(self) -> None:
        positive = {
            "num_codes": self.num_codes,
            "embedding_dim": self.embedding_dim,
            "hidden_channels": self.hidden_channels,
            "residual_hidden": self.residual_hidden,
            "num_downsample": self.num_downsample,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_residual_blocks < 0:
            raise ValueError(
                f"num_residual_blocks must not be negative, "
                f"got {self.num_residual_blocks}"
            )
        if self.commitment_cost < 0:
            raise ValueError(
                f"commitment_cost must not be negative, got {self.commitment_cost}"
            )
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")
        if self.dead_code_steps < 0:
            raise ValueError(
                f"dead_code_steps must not be negative, got {self.dead_code_steps}"
            )


@dataclass(frozen=True)
class TrainConfig:
    """Optimization settings.

    ``max_steps`` caps total optimizer steps regardless of ``epochs``, which is
    what makes a smoke run cheap.
    """

    seed: int = 0
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_steps: int | None = None
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(
                f"max_steps must be positive when set, got {self.max_steps}"
            )


@dataclass(frozen=True)
class RunConfig:
    """Everything one training run needs, from one file."""

    name: str
    data: DataConfig
    model: VQVAEConfig = field(default_factory=VQVAEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        stride = 2**self.model.num_downsample
        if self.data.image_size % stride:
            raise ValueError(
                f"image_size {self.data.image_size} is not divisible by "
                f"{stride} ({self.model.num_downsample} downsampling stages), "
                "so the encoder cannot reach a whole code grid"
            )

    @property
    def latent_size(self) -> int:
        """Side length of the square code grid the encoder produces."""
        return self.data.image_size // 2**self.model.num_downsample


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


_SECTIONS = ("name", "data", "model", "train")


def _build_data_config(raw: Any, *, where: str) -> DataConfig:
    if not isinstance(raw, dict):
        raise ValueError(
            f"{where}: expected a mapping for DataConfig, got {type(raw).__name__}"
        )

    raw = dict(raw)
    split_raw = raw.get("split")
    raw["split"] = _build(
        SplitConfig, {} if split_raw is None else split_raw, where=f"{where}:split"
    )
    return _build(DataConfig, raw, where=where)


def load_run_config(path: str | Path) -> RunConfig:
    """Parse one YAML file into the full configuration for a run."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(raw).__name__}"
        )

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(
            f"{path}: unknown top-level section(s): {', '.join(sorted(unknown))}. "
            f"Expected any of: {', '.join(_SECTIONS)}"
        )
    if "data" not in raw:
        raise ValueError(f"{path}: missing the required 'data' section")

    def section(name: str) -> Any:
        # An absent section defaults, but an explicitly wrong one must still be
        # rejected, so only None is replaced -- not every falsy value.
        value = raw.get(name)
        return {} if value is None else value

    return RunConfig(
        name=raw.get("name") or path.stem,
        data=_build_data_config(raw["data"], where=f"{path}:data"),
        model=_build(VQVAEConfig, section("model"), where=f"{path}:model"),
        train=_build(TrainConfig, section("train"), where=f"{path}:train"),
    )
