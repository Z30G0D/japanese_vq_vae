import torch

from jvqvae.figures import save_reconstruction_grid
from jvqvae.models.vqvae import VQVAE

CPU = torch.device("cpu")


def test_the_figure_is_written(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model)
    path = tmp_path / "nested" / "recon.png"

    returned = save_reconstruction_grid(
        model, tiny_data, path, device=CPU, num_images=4
    )

    assert returned == path
    assert path.exists()
    assert path.stat().st_size > 0


def test_asking_for_more_images_than_exist_is_clamped(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model)
    path = tmp_path / "recon.png"

    save_reconstruction_grid(model, tiny_data, path, device=CPU, num_images=500)

    assert path.exists()


def test_the_figure_does_not_leave_the_model_training(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model).train()
    before = model.quantizer.codebook.clone()

    save_reconstruction_grid(
        model, tiny_data, tmp_path / "recon.png", device=CPU, num_images=4
    )

    assert torch.equal(model.quantizer.codebook, before)
