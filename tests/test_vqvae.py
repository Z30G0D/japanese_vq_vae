import pytest
import torch

from jvqvae.config import VQVAEConfig
from jvqvae.models.vqvae import VQVAE, load_checkpoint, save_checkpoint


def test_forward_returns_a_reconstruction_of_the_input_shape(tiny_config):
    model = VQVAE(tiny_config.model)
    images = torch.rand(2, 1, 16, 16)

    out = model(images)

    assert out.reconstruction.shape == images.shape
    assert out.codes.shape == (2, 4, 4)


def test_reconstructions_are_valid_images(tiny_config):
    model = VQVAE(tiny_config.model)
    out = model(torch.rand(2, 1, 16, 16))

    assert out.reconstruction.min() >= 0.0
    assert out.reconstruction.max() <= 1.0


def test_the_code_grid_matches_the_configured_latent_size(tiny_config):
    model = VQVAE(tiny_config.model)
    codes = model.encode(torch.rand(3, 1, 16, 16))

    assert codes.shape == (3, tiny_config.latent_size, tiny_config.latent_size)
    assert codes.dtype == torch.int64
    assert int(codes.max()) < tiny_config.model.num_codes


def test_loss_is_the_sum_of_its_reported_parts(tiny_config):
    model = VQVAE(tiny_config.model)
    out = model(torch.rand(4, 1, 16, 16))

    assert out.loss.item() == pytest.approx(
        out.reconstruction_loss.item() + out.commitment_loss.item(), rel=1e-5
    )


def test_backward_reaches_both_the_encoder_and_the_decoder(tiny_config):
    model = VQVAE(tiny_config.model)
    model(torch.rand(4, 1, 16, 16)).loss.backward()

    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    decoder_grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]

    assert encoder_grads and sum(g.abs().sum() for g in encoder_grads) > 0
    assert decoder_grads and sum(g.abs().sum() for g in decoder_grads) > 0


def test_decode_accepts_what_encode_produces(tiny_config):
    model = VQVAE(tiny_config.model).eval()
    images = torch.rand(2, 1, 16, 16)

    assert model.decode(model.encode(images)).shape == images.shape


def test_encode_and_decode_round_trip_through_the_full_forward(tiny_config):
    model = VQVAE(tiny_config.model).eval()
    images = torch.rand(2, 1, 16, 16)

    out = model(images)

    assert torch.allclose(model.decode(out.codes), out.reconstruction, atol=1e-5)


def test_a_few_steps_of_training_drive_the_loss_down(tiny_config):
    """A tiny model on four fixed images must be able to overfit them."""
    torch.manual_seed(0)
    model = VQVAE(tiny_config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    images = torch.rand(4, 1, 16, 16)

    first = model(images).reconstruction_loss.item()
    for _ in range(30):
        optimizer.zero_grad()
        out = model(images)
        out.loss.backward()
        optimizer.step()

    assert out.reconstruction_loss.item() < first


def test_checkpoint_round_trip_reproduces_identical_outputs(tiny_config, tmp_path):
    torch.manual_seed(0)
    model = VQVAE(tiny_config.model).eval()
    images = torch.rand(2, 1, 16, 16)
    expected = model(images).reconstruction

    path = tmp_path / "vqvae.pt"
    save_checkpoint(
        model, path, config=tiny_config.model, epoch=7, metrics={"val_loss": 0.5}
    )
    restored, meta = load_checkpoint(path, device=torch.device("cpu"))

    assert torch.allclose(restored.eval()(images).reconstruction, expected, atol=1e-6)
    assert meta["epoch"] == 7
    assert meta["metrics"]["val_loss"] == 0.5
    assert meta["config"] == tiny_config.model


def test_the_checkpoint_carries_the_codebook(tiny_config, tmp_path):
    model = VQVAE(tiny_config.model).train()
    model(torch.rand(8, 1, 16, 16))
    path = tmp_path / "vqvae.pt"
    save_checkpoint(model, path, config=tiny_config.model, epoch=0, metrics={})

    restored, _ = load_checkpoint(path, device=torch.device("cpu"))

    assert torch.equal(restored.quantizer.codebook, model.quantizer.codebook)


def test_a_checkpoint_rebuilds_the_model_without_being_handed_a_config(
    tiny_config, tmp_path
):
    """Plan 3 loads a checkpoint knowing only its path."""
    path = tmp_path / "vqvae.pt"
    save_checkpoint(
        VQVAE(tiny_config.model), path, config=tiny_config.model, epoch=0, metrics={}
    )

    restored, meta = load_checkpoint(path, device=torch.device("cpu"))

    assert isinstance(meta["config"], VQVAEConfig)
    assert restored.config.num_codes == tiny_config.model.num_codes


def test_a_loaded_checkpoint_comes_back_in_eval_mode(tiny_config, tmp_path):
    """Otherwise a fresh module defaults to training and keeps moving the codebook."""
    path = tmp_path / "vqvae.pt"
    save_checkpoint(
        VQVAE(tiny_config.model).train(),
        path,
        config=tiny_config.model,
        epoch=0,
        metrics={},
    )

    model, _ = load_checkpoint(path, device=torch.device("cpu"))

    assert not model.training
    before = model.quantizer.codebook.clone()
    model(torch.rand(8, 1, 16, 16))
    assert torch.equal(model.quantizer.codebook, before)


def test_loading_a_checkpoint_does_not_unpickle_arbitrary_objects(
    tiny_config, tmp_path
):
    path = tmp_path / "vqvae.pt"
    save_checkpoint(
        VQVAE(tiny_config.model), path, config=tiny_config.model, epoch=0, metrics={}
    )

    # weights_only=True must be enough to read it back.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"model", "config", "epoch", "metrics"}
