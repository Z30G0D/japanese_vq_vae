import numpy as np
import pytest

from jvqvae.config import DataConfig, SplitConfig
from jvqvae.data.dataset import CharacterData
from jvqvae.data.pipeline import etl4_cache_path, load_splits
from jvqvae.data.sources.etl4 import _save_cache
from jvqvae.data.splits import split_by_writer

N_CLASSES = 6
N_WRITERS = 30


def build_etl4_cache(tmp_path, image_size=8):
    """A stand-in for an ingested ETL4 cache: every writer writes every class."""
    writer_ids = np.repeat(np.arange(N_WRITERS), N_CLASSES).astype(np.int64)
    class_ids = np.tile(np.arange(N_CLASSES), N_WRITERS).astype(np.int64)
    data = CharacterData(
        images=np.zeros((len(class_ids), image_size, image_size), dtype=np.float32),
        class_ids=class_ids,
        writer_ids=writer_ids,
        classes=[chr(0x3042 + i) for i in range(N_CLASSES)],
    )
    config = DataConfig(
        source="etl4",
        image_size=image_size,
        cache_dir=str(tmp_path),
        split=SplitConfig(seed=0, val_writers=5, test_writers=5),
    )
    _save_cache(data, etl4_cache_path(config))
    return config


def test_cache_path_is_derived_from_the_config(tmp_path):
    config = DataConfig(source="etl4", image_size=64, cache_dir=str(tmp_path))
    assert etl4_cache_path(config) == tmp_path / "etl4_64.npz"


def test_etl4_splits_hold_writers_out_entirely(tmp_path):
    splits = load_splits(build_etl4_cache(tmp_path))

    writers = {
        name: set(part.writer_ids.tolist()) for name, part in splits.items()
    }
    assert writers["train"] & writers["val"] == set()
    assert writers["train"] & writers["test"] == set()
    assert writers["val"] & writers["test"] == set()
    assert len(writers["val"]) == 5
    assert len(writers["test"]) == 5
    assert len(writers["train"]) == N_WRITERS - 10


def test_etl4_splits_cover_every_sample_exactly_once(tmp_path):
    splits = load_splits(build_etl4_cache(tmp_path))
    assert sum(len(part) for part in splits.values()) == N_CLASSES * N_WRITERS


def test_every_etl4_split_contains_every_class(tmp_path):
    splits = load_splits(build_etl4_cache(tmp_path))
    for name, part in splits.items():
        assert sorted(set(part.class_ids.tolist())) == list(range(N_CLASSES)), name


def test_splitting_by_writer_is_refused_without_writer_identities():
    with pytest.raises(ValueError, match="no writer identities"):
        split_by_writer(None, seed=0, n_val_writers=1, n_test_writers=1)
