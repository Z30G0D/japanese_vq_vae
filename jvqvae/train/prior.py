"""Prior training.

Same two-checkpoint pattern as the VQ-VAE loop: best by validation
cross-entropy, and last always. With no validation split, best is the final
state.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from jvqvae.codes import CodeData, CodeTorchDataset
from jvqvae.config import RunConfig
from jvqvae.models.prior import Prior, save_prior_checkpoint
from jvqvae.train.seed import set_seed


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None


def _check(splits: dict[str, CodeData]) -> CodeData:
    train = splits.get("train")
    if train is None or len(train) == 0:
        raise ValueError("the training split is empty, so there is nothing to train on")
    return train


def _loader(data: CodeData, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        CodeTorchDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


@torch.no_grad()
def _validate(model: Prior, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for codes, class_ids in loader:
        codes = codes.to(device)
        class_ids = class_ids.to(device)
        loss = model(codes, class_ids, dropout_labels=False).loss
        total += loss.item() * len(codes)
        count += len(codes)
    return total / max(count, 1)


def train_prior(
    config: RunConfig,
    splits: dict[str, CodeData],
    *,
    device: torch.device,
    output_root: str | Path = ".",
    log: Callable[[str], None] = print,
) -> dict:
    """Train the prior and write its checkpoints and run record."""
    train = _check(splits)
    set_seed(config.train.seed)

    root = Path(output_root)
    checkpoint_dir = root / config.train.checkpoint_dir / config.name
    results_dir = root / config.train.results_dir / config.name
    results_dir.mkdir(parents=True, exist_ok=True)

    model = Prior(
        config.prior,
        num_codes=train.num_codes,
        num_classes=train.num_classes,
        latent_size=train.latent_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.prior.learning_rate)

    train_loader = _loader(train, config.prior.batch_size, shuffle=True)
    val = splits.get("val")
    val_loader = (
        _loader(val, config.prior.batch_size, shuffle=False)
        if val is not None and len(val)
        else None
    )

    history: list[EpochMetrics] = []
    best_val, best_epoch = None, None
    steps = 0

    def _save(path: Path, epoch: int, metrics: dict) -> None:
        save_prior_checkpoint(
            model,
            path,
            config=config.prior,
            num_codes=train.num_codes,
            num_classes=train.num_classes,
            latent_size=train.latent_size,
            epoch=epoch,
            metrics=metrics,
        )

    for epoch in range(config.prior.epochs):
        model.train()
        total, batches = 0.0, 0

        for codes, class_ids in train_loader:
            if config.prior.max_steps is not None and steps >= config.prior.max_steps:
                break

            codes = codes.to(device)
            class_ids = class_ids.to(device)
            optimizer.zero_grad()
            out = model(codes, class_ids, dropout_labels=True)
            out.loss.backward()
            optimizer.step()
            steps += 1
            batches += 1
            total += out.loss.item()

        if batches == 0:
            break

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=total / batches,
            val_loss=_validate(model, val_loader, device) if val_loader else None,
        )
        history.append(metrics)

        log(
            f"epoch {epoch}: train ce {metrics.train_loss:.4f}"
            + (
                f" val ce {metrics.val_loss:.4f}"
                if metrics.val_loss is not None
                else ""
            )
        )

        if metrics.val_loss is not None and (
            best_val is None or metrics.val_loss < best_val
        ):
            best_val, best_epoch = metrics.val_loss, epoch
            _save(checkpoint_dir / "prior_best.pt", epoch, asdict(metrics))

        if config.prior.max_steps is not None and steps >= config.prior.max_steps:
            break

    model.eval()
    last_epoch = history[-1].epoch if history else -1
    last_metrics = asdict(history[-1]) if history else {}
    _save(checkpoint_dir / "prior_last.pt", last_epoch, last_metrics)
    if best_epoch is None:
        _save(checkpoint_dir / "prior_best.pt", last_epoch, last_metrics)

    record = {
        "name": config.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "device": device.type,
        "torch_version": torch.__version__,
        "config": {
            "data": asdict(config.data),
            "model": asdict(config.model),
            "train": asdict(config.train),
            "prior": asdict(config.prior),
        },
        "num_train": len(train),
        "num_val": len(val) if val is not None else 0,
        "total_steps": steps,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs": [asdict(m) for m in history],
    }
    (results_dir / "prior_run.json").write_text(json.dumps(record, indent=2))
    return record
