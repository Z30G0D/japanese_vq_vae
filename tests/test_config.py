from pathlib import Path

import pytest

from jvqvae.config import (
    DataConfig,
    RunConfig,
    SplitConfig,
    TrainConfig,
    VQVAEConfig,
    load_run_config,
)

REPO_ROOT = Path(__file__).parent.parent


def write(tmp_path, text):
    path = tmp_path / "run.yaml"
    path.write_text(text)
    return path


MINIMAL = "data:\n  source: k49\n  image_size: 32\n"


def test_defaults_fill_in_every_section(tmp_path):
    config = load_run_config(write(tmp_path, MINIMAL))

    assert config.data.source == "k49"
    assert config.model.num_codes == 512
    assert config.model.embedding_dim == 64
    assert config.train.epochs == 50
    assert config.data.split.seed == 0
    assert config.name == "run"


def test_sections_override_defaults(tmp_path):
    config = load_run_config(
        write(
            tmp_path,
            "name: experiment\n"
            "data:\n  source: k49\n  image_size: 32\n"
            "model:\n  num_codes: 64\n  num_downsample: 2\n"
            "train:\n  epochs: 3\n  batch_size: 8\n",
        )
    )

    assert config.name == "experiment"
    assert config.model.num_codes == 64
    assert config.train.epochs == 3
    assert config.train.batch_size == 8
    assert config.train.learning_rate == 3e-4


def test_latent_size_follows_image_size_and_downsampling(tmp_path):
    config = load_run_config(
        write(
            tmp_path,
            "data:\n  source: k49\n  image_size: 32\n"
            "model:\n  num_downsample: 2\n",
        )
    )
    assert config.latent_size == 8


def test_a_missing_data_section_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="data"):
        load_run_config(write(tmp_path, "model:\n  num_codes: 64\n"))


def test_unknown_top_level_sections_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="prior"):
        load_run_config(write(tmp_path, MINIMAL + "prior:\n  layers: 4\n"))


def test_unknown_keys_inside_a_section_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="codebook_size"):
        load_run_config(write(tmp_path, MINIMAL + "model:\n  codebook_size: 64\n"))


def test_a_non_mapping_section_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="model"):
        load_run_config(write(tmp_path, MINIMAL + "model: 0\n"))


def test_error_messages_name_the_offending_file(tmp_path):
    path = write(tmp_path, MINIMAL + "model:\n  codebook_size: 64\n")
    with pytest.raises(ValueError, match=str(path)):
        load_run_config(path)


def test_an_image_size_that_does_not_reach_the_grid_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        load_run_config(
            write(
                tmp_path,
                "data:\n  source: k49\n  image_size: 20\n"
                "model:\n  num_downsample: 3\n",
            )
        )


@pytest.mark.parametrize(
    "section,text",
    [
        ("model", "model:\n  num_codes: 0\n"),
        ("model", "model:\n  embedding_dim: -1\n"),
        ("model", "model:\n  num_downsample: 0\n"),
        ("model", "model:\n  ema_decay: 1.5\n"),
        ("model", "model:\n  commitment_cost: -0.1\n"),
        ("train", "train:\n  epochs: 0\n"),
        ("train", "train:\n  batch_size: 0\n"),
        ("train", "train:\n  learning_rate: 0\n"),
        ("train", "train:\n  max_steps: 0\n"),
    ],
)
def test_out_of_range_scalars_are_rejected(tmp_path, section, text):
    with pytest.raises(ValueError):
        load_run_config(write(tmp_path, MINIMAL + text))


def test_split_config_still_validates():
    with pytest.raises(ValueError, match="negative"):
        SplitConfig(val_writers=-1)


@pytest.mark.parametrize("name", ["etl4.yaml", "k49_dev.yaml"])
def test_shipped_configs_load_and_reach_an_eight_by_eight_grid(name):
    config = load_run_config(REPO_ROOT / "configs" / name)
    assert config.latent_size == 8
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.model, VQVAEConfig)
    assert isinstance(config.train, TrainConfig)


def test_the_etl4_config_holds_the_ninety_six_twelve_twelve_split():
    config = load_run_config(REPO_ROOT / "configs" / "etl4.yaml")
    assert config.data.source == "etl4"
    assert config.data.image_size == 64
    assert config.data.split.val_writers == 12
    assert config.data.split.test_writers == 12


def test_run_config_can_be_built_directly():
    config = RunConfig(
        name="direct",
        data=DataConfig(source="k49", image_size=32),
        model=VQVAEConfig(num_downsample=2),
        train=TrainConfig(),
    )
    assert config.latent_size == 8
