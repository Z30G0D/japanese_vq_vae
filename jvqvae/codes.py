"""Code grids extracted from a trained VQ-VAE.

This is the hand-off to the prior: the prior never sees an image, only these
grids and their class labels.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from jvqvae.data.dataset import CharacterData
from jvqvae.models.vqvae import VQVAE


@dataclass(frozen=True)
class CodeData:
    codes: np.ndarray  # int64 (N, G, G)
    class_ids: np.ndarray  # int64 (N,)
    classes: list[str]
    num_codes: int

    def __post_init__(self) -> None:
        if self.codes.ndim != 3:
            raise ValueError(f"codes must be (N, G, G), got {self.codes.shape}")
        if self.codes.shape[1] != self.codes.shape[2]:
            raise ValueError(f"code grid must be square, got {self.codes.shape[1:]}")
        if len(self.class_ids) != len(self.codes):
            raise ValueError(
                f"length mismatch: {len(self.codes)} grids, "
                f"{len(self.class_ids)} class ids"
            )
        if len(self.codes) and (
            int(self.codes.max()) >= self.num_codes or int(self.codes.min()) < 0
        ):
            raise ValueError(
                f"codes must index a codebook of {self.num_codes}, got "
                f"[{int(self.codes.min())}, {int(self.codes.max())}]"
            )
        if len(self.class_ids) and int(self.class_ids.max()) >= len(self.classes):
            raise ValueError(
                f"class id {int(self.class_ids.max())} exceeds class table of "
                f"size {len(self.classes)}"
            )

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def latent_size(self) -> int:
        return self.codes.shape[1]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def sequence_length(self) -> int:
        """Tokens per image once the grid is flattened in raster order."""
        return self.codes.shape[1] * self.codes.shape[2]


@torch.no_grad()
def extract_codes(
    model: VQVAE,
    data: CharacterData,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> CodeData:
    """Run a trained model over a dataset and keep only the code grids."""
    was_training = model.training
    model.eval().to(device)

    grids = []
    for start in range(0, len(data), batch_size):
        images = data.images[start : start + batch_size]
        tensor = torch.from_numpy(images).unsqueeze(1).to(device)
        grids.append(model.encode(tensor).cpu().numpy())

    if was_training:
        model.train()

    codes = (
        np.concatenate(grids).astype(np.int64)
        if grids
        else np.zeros((0, 0, 0), dtype=np.int64)
    )
    return CodeData(
        codes=codes,
        class_ids=data.class_ids.astype(np.int64),
        classes=list(data.classes),
        num_codes=model.config.num_codes,
    )


def save_codes(data: CodeData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        codes=data.codes,
        class_ids=data.class_ids,
        classes=np.array(data.classes),
        num_codes=np.array(data.num_codes),
    )


def load_codes(path: str | Path) -> CodeData:
    with np.load(path, allow_pickle=False) as raw:
        return CodeData(
            codes=raw["codes"],
            class_ids=raw["class_ids"],
            classes=[str(char) for char in raw["classes"]],
            num_codes=int(raw["num_codes"]),
        )
