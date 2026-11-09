"""PyTorch adapter over `CharacterData`.

Kept deliberately thin: the numpy container stays the single source of truth,
and this only adds the channel dimension and tensor conversion.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from jvqvae.data.dataset import CharacterData


class CharacterTorchDataset(Dataset):
    def __init__(self, data: CharacterData, indices: np.ndarray | None = None) -> None:
        self._data = data
        self._indices = (
            np.arange(len(data), dtype=np.int64) if indices is None else indices
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self._indices[position])
        image = torch.from_numpy(self._data.images[index]).unsqueeze(0)
        class_id = torch.tensor(int(self._data.class_ids[index]), dtype=torch.int64)
        return image, class_id
