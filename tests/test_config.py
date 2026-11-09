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


def test_unknown_nested_split_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("source: k49\nimage_size: 32\nsplit:\n  sedd: 1\n")

    with pytest.raises(ValueError, match="sedd"):
        load_data_config(path)


def test_empty_file_names_the_missing_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("")

    with pytest.raises(ValueError, match="source"):
        load_data_config(path)


def test_empty_split_block_falls_back_to_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("source: k49\nimage_size: 32\nsplit:\n")

    assert load_data_config(path).split == SplitConfig()


@pytest.mark.parametrize("value", ["5", "abc", "[seed]", "0", "[]"])
def test_non_mapping_split_is_rejected(tmp_path, value):
    path = tmp_path / "c.yaml"
    path.write_text(f"source: k49\nimage_size: 32\nsplit: {value}\n")

    with pytest.raises(ValueError, match="expected a mapping"):
        load_data_config(path)


def test_top_level_must_be_a_mapping(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- source: k49\n")

    with pytest.raises(ValueError, match="top level"):
        load_data_config(path)


def test_shipped_configs_match_their_datasets():
    k49 = load_data_config("configs/k49_dev.yaml")
    etl4 = load_data_config("configs/etl4.yaml")

    assert k49.image_size == 32
    assert etl4.image_size == 64
    # ETL4 may not be redistributed, so the config must point at a local path.
    assert etl4.etl4_zip is not None and "://" not in etl4.etl4_zip
