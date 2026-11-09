"""VQ-VAE training.

Writes two checkpoints (best by validation loss, and last) plus a JSON run
record. The record exists from phase 1 even though no evaluation metrics are
computed yet, so phase 2 has somewhere to put them.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from jvqvae.config import RunConfig
from jvqvae.data.dataset import CharacterData
from jvqvae.data.torch_dataset import CharacterTorchDataset
from jvqvae.models.vqvae import VQVAE, save_checkpoint
from jvqvae.train.seed import set_seed


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_reconstruction: float
    train_commitment: float
    val_loss: float | None
    perplexity: float
    num_active: int
    num_restarted: int


def _check(config: RunConfig, splits: dict[str, CharacterData]) -> None:
    train = splits.get("train")
    if train is None or len(train) == 0:
        raise ValueError("the training split is empty, so there is nothing to train on")

    height, width = train.image_shape
    if (height, width) != (config.data.image_size, config.data.image_size):
        raise ValueError(
            f"config image_size is {config.data.image_size} but the data is "
            f"{height}x{width}; they must agree"
        )


def _loader(data: CharacterData, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        CharacterTorchDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


@torch.no_grad()
def _validate(model: VQVAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for images, _ in loader:
        images = images.to(device)
        total += model(images).reconstruction_loss.item() * len(images)
        count += len(images)
    return total / max(count, 1)


def train_vqvae(
    config: RunConfig,
    splits: dict[str, CharacterData],
    *,
    device: torch.device,
    output_root: str | Path = ".",
    log: Callable[[str], None] = print,
) -> dict:
    """Train the VQ-VAE and write its checkpoints and run record."""
    _check(config, splits)
    set_seed(config.train.seed)

    root = Path(output_root)
    checkpoint_dir = root / config.train.checkpoint_dir / config.name
    results_dir = root / config.train.results_dir / config.name
    results_dir.mkdir(parents=True, exist_ok=True)

    model = VQVAE(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    train_loader = _loader(splits["train"], config.train.batch_size, shuffle=True)
    val = splits.get("val")
    val_loader = (
        _loader(val, config.train.batch_size, shuffle=False)
        if val is not None and len(val)
        else None
    )

    history: list[EpochMetrics] = []
    best_val, best_epoch = None, None
    steps = 0

    for epoch in range(config.train.epochs):
        model.train()
        sums = {"loss": 0.0, "recon": 0.0, "commit": 0.0, "perplexity": 0.0}
        used = torch.zeros(config.model.num_codes, dtype=torch.bool, device=device)
        restarted, batches = 0, 0

        for images, _ in train_loader:
            if config.train.max_steps is not None and steps >= config.train.max_steps:
                break

            images = images.to(device)
            optimizer.zero_grad()
            out = model(images)
            out.loss.backward()
            optimizer.step()
            steps += 1
            batches += 1

            sums["loss"] += out.loss.item()
            sums["recon"] += out.reconstruction_loss.item()
            sums["commit"] += out.commitment_loss.item()
            sums["perplexity"] += out.perplexity.item()
            used.scatter_(0, out.codes.reshape(-1), True)
            restarted += out.num_restarted

        if batches == 0:
            break

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=sums["loss"] / batches,
            train_reconstruction=sums["recon"] / batches,
            train_commitment=sums["commit"] / batches,
            val_loss=_validate(model, val_loader, device) if val_loader else None,
            perplexity=sums["perplexity"] / batches,
            num_active=int(used.sum()),
            num_restarted=restarted,
        )
        history.append(metrics)

        log(
            f"epoch {epoch}: train recon {metrics.train_reconstruction:.4f}"
            f" commit {metrics.train_commitment:.4f}"
            + (
                f" val recon {metrics.val_loss:.4f}"
                if metrics.val_loss is not None
                else ""
            )
            + f" perplexity {metrics.perplexity:.1f}"
            f" active {metrics.num_active}/{config.model.num_codes}"
            + (f" restarted {metrics.num_restarted}" if metrics.num_restarted else "")
        )

        if metrics.val_loss is not None and (
            best_val is None or metrics.val_loss < best_val
        ):
            best_val, best_epoch = metrics.val_loss, epoch
            save_checkpoint(
                model,
                checkpoint_dir / "vqvae_best.pt",
                config=config.model,
                epoch=epoch,
                metrics=asdict(metrics),
            )

        if config.train.max_steps is not None and steps >= config.train.max_steps:
            break

    # Leave the model in eval mode: whatever runs next should not keep mutating
    # the codebook by accident.
    model.eval()
    last_epoch = history[-1].epoch if history else -1
    save_checkpoint(
        model,
        checkpoint_dir / "vqvae_last.pt",
        config=config.model,
        epoch=last_epoch,
        metrics=asdict(history[-1]) if history else {},
    )
    if best_epoch is None:
        # No validation split, so "best" is simply the final state.
        save_checkpoint(
            model,
            checkpoint_dir / "vqvae_best.pt",
            config=config.model,
            epoch=last_epoch,
            metrics=asdict(history[-1]) if history else {},
        )

    record = {
        "name": config.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "device": device.type,
        "torch_version": torch.__version__,
        "config": {
            "data": asdict(config.data),
            "model": asdict(config.model),
            "train": asdict(config.train),
        },
        "num_train": len(splits["train"]),
        "num_val": len(val) if val is not None else 0,
        "total_steps": steps,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs": [asdict(m) for m in history],
    }
    (results_dir / "vqvae_run.json").write_text(json.dumps(record, indent=2))
    return record
