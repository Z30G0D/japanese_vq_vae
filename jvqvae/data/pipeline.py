"""Config to ready-to-use splits.

This is the only place that decides how a source gets divided, which is what
keeps the writer-held-out guarantee a property of the code rather than a
convention. ETL4 carries writer identities and is always split by writer; K49
does not and ships its own test set, so its validation set is carved at random
out of the official train portion. Asking for the wrong one is not possible
from here.
"""

from pathlib import Path

from jvqvae.config import DataConfig
from jvqvae.data.dataset import CharacterData
from jvqvae.data.sources.etl4 import load_cached_etl4
from jvqvae.data.sources.k49 import load_k49
from jvqvae.data.splits import carve_validation, split_by_writer


def etl4_cache_path(config: DataConfig) -> Path:
    """Derive the ETL4 cache location so callers never spell it by hand."""
    return Path(config.cache_dir) / f"etl4_{config.image_size}.npz"


def _split_etl4(config: DataConfig) -> dict[str, CharacterData]:
    data = load_cached_etl4(etl4_cache_path(config))
    split = split_by_writer(
        data.writer_ids,
        seed=config.split.seed,
        n_val_writers=config.split.val_writers,
        n_test_writers=config.split.test_writers,
    )
    return {
        "train": data.subset(split.train),
        "val": data.subset(split.val),
        "test": data.subset(split.test),
    }


def _split_k49(config: DataConfig) -> dict[str, CharacterData]:
    def load(split: str) -> CharacterData:
        return load_k49(
            config.k49_dir,
            split=split,
            image_size=config.image_size,
            max_per_class=config.k49_max_per_class,
            seed=config.split.seed,
        )

    official_train = load("train")
    train, val = carve_validation(
        len(official_train),
        seed=config.split.seed,
        n_val=int(len(official_train) * config.split.val_fraction),
    )
    return {
        "train": official_train.subset(train),
        "val": official_train.subset(val),
        "test": load("test"),
    }


def load_splits(config: DataConfig) -> dict[str, CharacterData]:
    """Load the configured source and divide it the only way that source allows.

    Returns train, val and test as independent `CharacterData`. For ETL4 no
    writer appears in more than one of them.
    """
    if config.source == "etl4":
        return _split_etl4(config)
    return _split_k49(config)
