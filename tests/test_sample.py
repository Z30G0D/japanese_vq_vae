import torch

from jvqvae.models.prior import Prior
from jvqvae.models.vqvae import VQVAE
from jvqvae.sample import sample_images

CPU = torch.device("cpu")


def make_prior(config, num_classes=3) -> Prior:
    return Prior(
        config.prior,
        num_codes=config.model.num_codes,
        num_classes=num_classes,
        latent_size=config.latent_size,
    )


def test_sample_images_match_the_vqvae_decode_size(tiny_config, tiny_data):
    vqvae = VQVAE(tiny_config.model).eval()
    prior = make_prior(tiny_config).eval()
    images = sample_images(prior, vqvae, torch.tensor([0, 1, 2]), device=CPU)
    assert images.shape == (3, 1, 16, 16)
    assert images.device == CPU


def test_sample_images_do_not_move_the_codebook(tiny_config):
    vqvae = VQVAE(tiny_config.model).train()
    before = vqvae.quantizer.codebook.clone()
    sample_images(make_prior(tiny_config), vqvae, torch.tensor([0]), device=CPU)
    assert torch.equal(vqvae.quantizer.codebook, before)


def test_sample_images_restore_training_flags(tiny_config):
    vqvae = VQVAE(tiny_config.model).train()
    prior = make_prior(tiny_config).train()
    sample_images(prior, vqvae, torch.tensor([0]), device=CPU)
    assert vqvae.training
    assert prior.training


def test_sampled_codes_decode_the_same_as_calling_decode_yourself(tiny_config):
    vqvae = VQVAE(tiny_config.model).eval()
    prior = make_prior(tiny_config).eval()
    class_ids = torch.tensor([0, 1, 2])

    torch.manual_seed(0)
    expected = vqvae.decode(prior.generate(class_ids))

    torch.manual_seed(0)
    images = sample_images(prior, vqvae, class_ids, device=CPU)

    assert images.device == expected.device
    assert torch.allclose(images, expected, atol=1e-5)


def test_sampling_options_are_passed_through_to_generate(tiny_config):
    vqvae = VQVAE(tiny_config.model).eval()
    prior = make_prior(tiny_config).eval()
    class_ids = torch.tensor([0, 1, 2])

    torch.manual_seed(0)
    cold = sample_images(prior, vqvae, class_ids, device=CPU, temperature=1e-3)
    torch.manual_seed(0)
    hot = sample_images(prior, vqvae, class_ids, device=CPU, temperature=5.0)

    assert not torch.allclose(cold, hot, atol=1e-5)
