import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from jvqvae.data.dataset import CharacterData
from jvqvae.models.vqvae import load_checkpoint
from jvqvae.train.seed import set_seed
from jvqvae.train.vqvae import train_vqvae

CPU = torch.device("cpu")


@pytest.fixture
def splits(tiny_data):
    return {"train": tiny_data, "val": tiny_data.subset(np.arange(4))}


def run(config, splits, tmp_path, **overrides):
    if overrides:
        config = replace(config, train=replace(config.train, **overrides))
    return train_vqvae(
        config, splits, device=CPU, output_root=tmp_path, log=lambda *a: None
    )


def test_set_seed_makes_torch_reproducible():
    set_seed(3)
    first = torch.randn(4)
    set_seed(3)
    assert torch.equal(torch.randn(4), first)


def test_set_seed_makes_numpy_reproducible():
    set_seed(3)
    first = np.random.rand(4)
    set_seed(3)
    assert np.array_equal(np.random.rand(4), first)


def test_training_returns_one_record_per_epoch(tiny_config, splits, tmp_path):
    record = run(tiny_config, splits, tmp_path, epochs=3, max_steps=None)

    assert len(record["epochs"]) == 3
    assert [e["epoch"] for e in record["epochs"]] == [0, 1, 2]


def test_max_steps_caps_the_run(tiny_config, splits, tmp_path):
    record = run(tiny_config, splits, tmp_path, epochs=100, max_steps=3)

    assert record["total_steps"] == 3
    assert len(record["epochs"]) < 100


def test_every_epoch_reports_the_metrics_the_blog_post_needs(
    tiny_config, splits, tmp_path
):
    record = run(tiny_config, splits, tmp_path, epochs=1, max_steps=None)
    epoch = record["epochs"][0]

    for key in (
        "train_loss",
        "train_reconstruction",
        "train_commitment",
        "val_loss",
        "perplexity",
        "num_active",
        "num_restarted",
    ):
        assert key in epoch, key
    assert epoch["num_active"] >= 1


def test_training_reduces_the_loss(tiny_config, splits, tmp_path):
    record = run(
        tiny_config, splits, tmp_path, epochs=25, max_steps=None, learning_rate=1e-2
    )
    losses = [e["train_loss"] for e in record["epochs"]]

    assert losses[-1] < losses[0]


def test_both_checkpoints_are_written(tiny_config, splits, tmp_path):
    run(tiny_config, splits, tmp_path, epochs=2, max_steps=None)
    directory = tmp_path / "checkpoints" / "tiny"

    assert (directory / "vqvae_best.pt").exists()
    assert (directory / "vqvae_last.pt").exists()


def test_the_best_checkpoint_is_loadable_and_knows_its_epoch(
    tiny_config, splits, tmp_path
):
    run(tiny_config, splits, tmp_path, epochs=2, max_steps=None)

    model, meta = load_checkpoint(
        tmp_path / "checkpoints" / "tiny" / "vqvae_best.pt", device=CPU
    )

    assert meta["config"].num_codes == tiny_config.model.num_codes
    assert 0 <= meta["epoch"] <= 1
    assert model.eval()(torch.rand(1, 1, 16, 16)).reconstruction.shape == (1, 1, 16, 16)


def test_the_best_checkpoint_tracks_the_lowest_validation_loss(
    tiny_config, splits, tmp_path
):
    record = run(tiny_config, splits, tmp_path, epochs=4, max_steps=None)

    best = min(record["epochs"], key=lambda e: e["val_loss"])
    assert record["best_epoch"] == best["epoch"]
    assert record["best_val_loss"] == pytest.approx(best["val_loss"])

    _, meta = load_checkpoint(
        tmp_path / "checkpoints" / "tiny" / "vqvae_best.pt", device=CPU
    )
    assert meta["epoch"] == best["epoch"]


def test_the_run_record_is_written_as_json(tiny_config, splits, tmp_path):
    returned = run(tiny_config, splits, tmp_path, epochs=1, max_steps=None)
    path = tmp_path / "results" / "tiny" / "vqvae_run.json"

    assert path.exists()
    assert json.loads(path.read_text()) == returned


def test_the_run_record_carries_enough_to_reproduce_the_run(
    tiny_config, splits, tmp_path
):
    record = run(tiny_config, splits, tmp_path, epochs=1, max_steps=None)

    assert record["name"] == "tiny"
    assert record["device"] == "cpu"
    assert record["config"]["model"]["num_codes"] == tiny_config.model.num_codes
    assert record["config"]["data"]["source"] == "k49"
    assert record["config"]["train"]["seed"] == tiny_config.train.seed
    assert record["num_train"] == 12
    assert record["num_val"] == 4
    assert "torch_version" in record


def test_the_same_seed_gives_the_same_losses(tiny_config, splits, tmp_path):
    first = run(tiny_config, splits, tmp_path / "a", epochs=2, max_steps=None)
    second = run(tiny_config, splits, tmp_path / "b", epochs=2, max_steps=None)

    assert [e["train_loss"] for e in first["epochs"]] == [
        e["train_loss"] for e in second["epochs"]
    ]


def test_different_seeds_give_different_losses(tiny_config, splits, tmp_path):
    first = run(tiny_config, splits, tmp_path / "a", epochs=2, max_steps=None, seed=0)
    second = run(tiny_config, splits, tmp_path / "b", epochs=2, max_steps=None, seed=1)

    assert [e["train_loss"] for e in first["epochs"]] != [
        e["train_loss"] for e in second["epochs"]
    ]


def test_training_without_a_validation_split_still_works(
    tiny_config, tiny_data, tmp_path
):
    record = train_vqvae(
        tiny_config,
        {"train": tiny_data},
        device=CPU,
        output_root=tmp_path,
        log=lambda *a: None,
    )

    assert record["epochs"][0]["val_loss"] is None
    assert record["best_val_loss"] is None
    # With nothing to select on, "best" is simply the final state.
    assert (tmp_path / "checkpoints" / "tiny" / "vqvae_best.pt").exists()


def test_an_empty_training_split_is_refused(tiny_config, tiny_data, tmp_path):
    empty = tiny_data.subset(np.arange(0))
    with pytest.raises(ValueError, match="empty"):
        train_vqvae(
            tiny_config,
            {"train": empty},
            device=CPU,
            output_root=tmp_path,
            log=lambda *a: None,
        )


def test_an_image_size_disagreeing_with_the_config_is_refused(
    tiny_config, tmp_path
):
    wrong = CharacterData(
        images=np.zeros((4, 32, 32), dtype=np.float32),
        class_ids=np.zeros(4, dtype=np.int64),
        writer_ids=None,
        classes=["あ"],
    )
    with pytest.raises(ValueError, match="image_size"):
        train_vqvae(
            tiny_config,
            {"train": wrong},
            device=CPU,
            output_root=tmp_path,
            log=lambda *a: None,
        )


def test_the_log_callback_sees_every_epoch(tiny_config, splits, tmp_path):
    lines = []
    config = replace(tiny_config, train=replace(tiny_config.train, epochs=3,
                                                max_steps=None))

    train_vqvae(config, splits, device=CPU, output_root=tmp_path, log=lines.append)

    assert len(lines) == 3
    assert "train recon" in lines[0]
    assert "val recon" in lines[0]
    assert "perplexity" in lines[0]
