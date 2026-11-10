import pytest
import torch

from jvqvae.models.blocks import Decoder, Encoder, ResidualBlock


def test_residual_block_preserves_shape():
    block = ResidualBlock(8, 4)
    x = torch.randn(2, 8, 5, 5)
    assert block(x).shape == x.shape


def test_residual_block_actually_adds_a_residual():
    """A zeroed branch must leave the input untouched, which proves the skip."""
    block = ResidualBlock(8, 4)
    for parameter in block.parameters():
        torch.nn.init.zeros_(parameter)

    x = torch.randn(2, 8, 5, 5)
    assert torch.allclose(block(x), x)


@pytest.mark.parametrize("image_size,num_downsample", [(64, 3), (32, 2), (16, 1)])
def test_encoder_reaches_the_expected_grid(image_size, num_downsample):
    encoder = Encoder(
        1,
        hidden_channels=16,
        embedding_dim=6,
        num_downsample=num_downsample,
        num_residual_blocks=2,
        residual_hidden=4,
    )
    out = encoder(torch.rand(2, 1, image_size, image_size))

    expected = image_size // 2**num_downsample
    assert out.shape == (2, 6, expected, expected)


@pytest.mark.parametrize("grid,num_downsample", [(8, 3), (8, 2), (4, 1)])
def test_decoder_restores_the_image_size(grid, num_downsample):
    decoder = Decoder(
        6,
        hidden_channels=16,
        out_channels=1,
        num_downsample=num_downsample,
        num_residual_blocks=2,
        residual_hidden=4,
    )
    out = decoder(torch.randn(2, 6, grid, grid))

    expected = grid * 2**num_downsample
    assert out.shape == (2, 1, expected, expected)


def test_decoder_does_not_end_in_a_sigmoid():
    """A sigmoid last layer saturates MSE: logits of -8 already give ~0 gradient."""
    decoder = Decoder(
        6,
        hidden_channels=16,
        out_channels=1,
        num_downsample=2,
        num_residual_blocks=1,
        residual_hidden=4,
    )

    assert not isinstance(decoder.net[-1], torch.nn.Sigmoid)
    assert isinstance(decoder.net[-1], torch.nn.ConvTranspose2d)


def test_encoder_and_decoder_round_trip_the_shape():
    shared = dict(num_downsample=3, num_residual_blocks=2, residual_hidden=4)
    encoder = Encoder(1, hidden_channels=16, embedding_dim=6, **shared)
    decoder = Decoder(6, hidden_channels=16, out_channels=1, **shared)

    images = torch.rand(2, 1, 64, 64)
    assert decoder(encoder(images)).shape == images.shape


def test_positional_misuse_is_rejected():
    with pytest.raises(TypeError):
        Encoder(1, 16, 6, 3, 2, 4)


def test_a_batch_of_one_behaves_like_a_batch_of_many():
    """Pins the no-normalization claim: no cross-sample statistics anywhere.

    Deliberately in train mode. A BatchNorm would normalize by the batch's own
    statistics here, so one image alone would decode differently from the same
    image inside a batch. In eval mode it would use running statistics and this
    test would pass with a BatchNorm present, proving nothing.
    """
    shared = dict(num_downsample=2, num_residual_blocks=2, residual_hidden=4)
    encoder = Encoder(1, hidden_channels=16, embedding_dim=6, **shared)
    decoder = Decoder(6, hidden_channels=16, out_channels=1, **shared)

    torch.manual_seed(0)
    images = torch.rand(4, 1, 32, 32)

    assert torch.allclose(encoder(images[:1]), encoder(images)[:1], atol=1e-6)
    grid = encoder(images)
    assert torch.allclose(decoder(grid[:1]), decoder(grid)[:1], atol=1e-6)


def test_decoder_positional_misuse_is_rejected():
    with pytest.raises(TypeError):
        Decoder(6, 16, 1, 2, 2, 4)
