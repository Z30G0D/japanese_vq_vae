import numpy as np
import pytest

from jvqvae.config import DataConfig, PriorConfig, RunConfig, TrainConfig, VQVAEConfig
from jvqvae.data.dataset import CharacterData


@pytest.fixture
def tiny_config():
    """A model small enough that a full training step costs milliseconds.

    Every model test uses this. Never load the shipped ETL4 or K49 configs in a
    test: they describe real runs and would make the suite minutes long.
    """
    return RunConfig(
        name="tiny",
        data=DataConfig(source="k49", image_size=16, cache_dir="unused"),
        model=VQVAEConfig(
            num_codes=8,
            embedding_dim=4,
            hidden_channels=8,
            residual_hidden=4,
            num_residual_blocks=1,
            num_downsample=2,
            dead_code_steps=0,
        ),
        train=TrainConfig(epochs=1, batch_size=4, max_steps=2),
        prior=PriorConfig(
            num_layers=1, num_heads=2, width=16, epochs=1, batch_size=4, max_steps=2
        ),
    )


@pytest.fixture
def tiny_data():
    """Twelve 16x16 images across three classes, deterministic."""
    rng = np.random.default_rng(0)
    return CharacterData(
        images=rng.random((12, 16, 16), dtype=np.float32),
        class_ids=np.tile(np.arange(3), 4).astype(np.int64),
        writer_ids=np.repeat(np.arange(4), 3).astype(np.int64),
        classes=["あ", "い", "う"],
    )
