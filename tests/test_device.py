import torch

from jvqvae.device import select_device


def test_explicit_preference_is_respected():
    assert select_device("cpu").type == "cpu"


def test_auto_returns_a_usable_device():
    device = select_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in {"mps", "cuda", "cpu"}
    torch.zeros(2, 2, device=device)
