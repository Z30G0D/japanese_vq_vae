import dataclasses

import pytest

from jvqvae.config import DataConfig, SplitConfig, load_data_config


def test_defaults_apply_when_yaml_is_minimal(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("source: k49\nimage_size: 32\n")

    config = load_data_config(path)

    assert config.source == "k49"
    assert config.image_size == 32
    assert config.split == SplitConfig()
    assert config.etl4_zip is None


def test_nested_split_is_parsed(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "source: etl4\n"
        "image_size: 64\n"
        "etl4_zip: data/raw/ETL4.zip\n"
        "split:\n"
        "  seed: 7\n"
        "  val_writers: 10\n"
    )

    config = load_data_config(path)

    assert config.split.seed == 7
    assert config.split.val_writers == 10
    assert config.split.test_writers == 12  # default preserved
    assert config.etl4_zip == "data/raw/ETL4.zip"


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("source: k49\nimage_size: 32\nlearnig_rate: 0.1\n")

    with pytest.raises(ValueError, match="learnig_rate"):
        load_data_config(path)


def test_config_is_immutable():
    config = DataConfig(source="k49", image_size=32)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.source = "etl4"
