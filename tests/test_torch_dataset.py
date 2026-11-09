import numpy as np
import torch
from torch.utils.data import DataLoader

from jvqvae.data.dataset import CharacterData
from jvqvae.data.torch_dataset import CharacterTorchDataset


def make(n=8, n_classes=4):
    rng = np.random.default_rng(0)
    return CharacterData(
        images=rng.random((n, 8, 8)).astype(np.float32),
        class_ids=np.arange(n, dtype=np.int64) % n_classes,
        writer_ids=np.arange(n, dtype=np.int64) // n_classes,
        classes=[chr(0x3042 + i) for i in range(n_classes)],
    )


def test_item_has_a_channel_dimension():
    dataset = CharacterTorchDataset(make())
    image, class_id = dataset[0]

    assert image.shape == (1, 8, 8)
    assert image.dtype == torch.float32
    assert class_id.dtype == torch.int64
    assert class_id.item() == 0


def test_indices_restrict_the_view():
    dataset = CharacterTorchDataset(make(), indices=np.array([5, 6]))

    assert len(dataset) == 2
    assert dataset[0][1].item() == 1  # original index 5, class 5 % 4


def test_works_with_a_dataloader():
    loader = DataLoader(CharacterTorchDataset(make()), batch_size=4)
    images, class_ids = next(iter(loader))

    assert images.shape == (4, 1, 8, 8)
    assert class_ids.shape == (4,)
