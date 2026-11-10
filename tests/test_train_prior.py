import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from jvqvae.codes import CodeData
from jvqvae.models.prior import load_prior_checkpoint
from jvqvae.train.prior import train_prior

CPU = torch.device("cpu")


def _codes(n: int, *, seed: int = 0) -> CodeData:
    rng = np.random.default_rng(seed)
    return CodeData(
        codes=rng.integers(0, 8, size=(n, 4, 4), dtype=np.int64),
        class_ids=np.tile(np.arange(3), n)[:n].astype(np.int64),
        classes=["あ", "い", "う"],
        num_codes=8,
    )


@pytest.fixture
def code_splits():
    data = _codes(12)
    return {"train": _slice(data, slice(0, 8)), "val": _slice(data, slice(8, 12))}


def _slice(data: CodeData, chosen: slice) -> CodeData:
    return CodeData(
        codes=data.codes[chosen],
        class_ids=data.class_ids[chosen],
        classes=list(data.classes),
        num_codes=data.num_codes,
    )


def run(config, splits, tmp_path, **overrides):
    if overrides:
        config = replace(config, prior=replace(config.prior, **overrides))
    return train_prior(
        config, splits, device=CPU, output_root=tmp_path, log=lambda *a: None
    )


def test_training_returns_one_record_per_epoch(tiny_config, code_splits, tmp_path):
    record = run(tiny_config, code_splits, tmp_path, epochs=3, max_steps=None)

    assert len(record["epochs"]) == 3
    assert [e["epoch"] for e in record["epochs"]] == [0, 1, 2]


def test_max_steps_caps_the_run(tiny_config, code_splits, tmp_path):
    record = run(tiny_config, code_splits, tmp_path, epochs=100, max_steps=3)

    assert record["total_steps"] == 3
    assert len(record["epochs"]) < 100


def test_every_epoch_reports_train_and_val_cross_entropy(
    tiny_config, code_splits, tmp_path
):
    record = run(tiny_config, code_splits, tmp_path, epochs=1, max_steps=None)
    epoch = record["epochs"][0]

    for key in ("epoch", "train_loss", "val_loss"):
        assert key in epoch, key
    assert epoch["epoch"] == 0
    assert epoch["train_loss"] > 0
    assert epoch["val_loss"] > 0


def test_best_checkpoint_is_the_lowest_validation_loss(
    tiny_config, code_splits, tmp_path
):
    record = run(tiny_config, code_splits, tmp_path, epochs=4, max_steps=None)

    best = min(record["epochs"], key=lambda e: e["val_loss"])
    assert record["best_epoch"] == best["epoch"]
    assert record["best_val_loss"] == pytest.approx(best["val_loss"])

    _, meta = load_prior_checkpoint(
        tmp_path / "checkpoints" / "tiny" / "prior_best.pt", device=CPU
    )
    assert meta["epoch"] == best["epoch"]
    assert (tmp_path / "checkpoints" / "tiny" / "prior_last.pt").exists()


def test_a_run_without_validation_still_writes_best(
    tiny_config, code_splits, tmp_path
):
    record = train_prior(
        tiny_config,
        {"train": code_splits["train"]},
        device=CPU,
        output_root=tmp_path,
        log=lambda *a: None,
    )

    assert record["epochs"][0]["val_loss"] is None
    assert record["best_val_loss"] is None
    assert record["num_val"] == 0
    assert (tmp_path / "checkpoints" / "tiny" / "prior_best.pt").exists()
    assert (tmp_path / "checkpoints" / "tiny" / "prior_last.pt").exists()


def test_empty_train_split_is_rejected(tiny_config, tmp_path):
    empty = CodeData(
        codes=np.zeros((0, 4, 4), dtype=np.int64),
        class_ids=np.zeros((0,), dtype=np.int64),
        classes=["あ", "い", "う"],
        num_codes=8,
    )
    with pytest.raises(ValueError, match="empty"):
        train_prior(
            tiny_config,
            {"train": empty},
            device=CPU,
            output_root=tmp_path,
            log=lambda *a: None,
        )


def test_the_json_record_is_written(tiny_config, code_splits, tmp_path):
    returned = run(tiny_config, code_splits, tmp_path, epochs=1, max_steps=None)
    path = tmp_path / "results" / "tiny" / "prior_run.json"

    assert path.exists()
    assert json.loads(path.read_text()) == returned
    for key in (
        "name",
        "created",
        "device",
        "torch_version",
        "config",
        "num_train",
        "num_val",
        "total_steps",
        "best_epoch",
        "best_val_loss",
        "epochs",
    ):
        assert key in returned, key
    assert returned["name"] == "tiny"
    assert returned["device"] == "cpu"
    assert returned["num_train"] == 8
    assert returned["num_val"] == 4
    for section in ("data", "model", "train", "prior"):
        assert section in returned["config"], section
    assert returned["config"]["prior"]["learning_rate"] == tiny_config.prior.learning_rate
    assert returned["config"]["train"]["seed"] == tiny_config.train.seed


def test_training_leaves_the_model_in_eval_mode(tiny_config, code_splits, tmp_path):
    run(tiny_config, code_splits, tmp_path, epochs=1, max_steps=None)

    model, _ = load_prior_checkpoint(
        tmp_path / "checkpoints" / "tiny" / "prior_last.pt", device=CPU
    )

    assert not model.training


def test_a_few_steps_drive_the_recorded_loss_down(tiny_config, code_splits, tmp_path):
    record = run(
        tiny_config,
        code_splits,
        tmp_path,
        epochs=40,
        max_steps=None,
        learning_rate=1e-2,
        label_dropout=0.0,
    )
    losses = [e["train_loss"] for e in record["epochs"]]

    assert losses[-1] < losses[0]
