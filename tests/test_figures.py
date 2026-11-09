import pytest
import torch

from jvqvae.figures import (
    save_guidance_sweep,
    save_reconstruction_grid,
    save_sample_grid,
)
from jvqvae.models.prior import Prior
from jvqvae.models.vqvae import VQVAE

CPU = torch.device("cpu")


def make_prior(config, num_classes=3) -> Prior:
    return Prior(
        config.prior,
        num_codes=config.model.num_codes,
        num_classes=num_classes,
        latent_size=config.latent_size,
    )


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


def test_the_sample_grid_is_written(tiny_config, tmp_path):
    vqvae = VQVAE(tiny_config.model)
    prior = make_prior(tiny_config)
    path = tmp_path / "nested" / "samples.png"

    returned = save_sample_grid(
        prior,
        vqvae,
        path,
        class_ids=[0, 1],
        samples_per_class=2,
        device=CPU,
    )

    assert returned == path
    assert path.exists()
    assert path.stat().st_size > 0


def test_the_guidance_sweep_is_written(tiny_config, tmp_path):
    vqvae = VQVAE(tiny_config.model)
    prior = make_prior(tiny_config)
    path = tmp_path / "nested" / "guidance.png"

    returned = save_guidance_sweep(
        prior,
        vqvae,
        path,
        class_id=0,
        scales=(0.0, 1.5),
        num_samples=2,
        device=CPU,
    )

    assert returned == path
    assert path.exists()
    assert path.stat().st_size > 0


def test_sample_grid_does_not_leave_the_vqvae_training(tiny_config, tmp_path):
    vqvae = VQVAE(tiny_config.model).train()
    prior = make_prior(tiny_config).train()
    before = vqvae.quantizer.codebook.clone()

    save_sample_grid(
        prior,
        vqvae,
        tmp_path / "samples.png",
        class_ids=[0],
        samples_per_class=1,
        device=CPU,
    )

    assert vqvae.training
    assert prior.training
    assert torch.equal(vqvae.quantizer.codebook, before)


def test_an_empty_class_list_is_rejected(tiny_config, tmp_path):
    vqvae = VQVAE(tiny_config.model)
    prior = make_prior(tiny_config)

    with pytest.raises(ValueError, match="empty"):
        save_sample_grid(
            prior,
            vqvae,
            tmp_path / "samples.png",
            class_ids=[],
            device=CPU,
        )
