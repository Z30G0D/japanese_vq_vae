"""The VQ-VAE itself: encoder, quantizer, decoder, and its checkpoint format.

A VQ-VAE is not generative on its own. It learns a discrete code for an image
and how to decode one, but it holds no distribution over code grids, so a grid
sampled uniformly decodes to noise. Learning that distribution is the prior's
job, in plan 3; this model's output is the code grids it trains on.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from jvqvae.config import VQVAEConfig
from jvqvae.models.blocks import Decoder, Encoder
from jvqvae.models.quantizer import VectorQuantizerEMA


@dataclass
class VQVAEOutput:
    reconstruction: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor
    num_active: int
    num_restarted: int
    codes: torch.Tensor


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig, in_channels: int = 1) -> None:
        super().__init__()
        self.config = config
        self.in_channels = in_channels

        shared = dict(
            num_downsample=config.num_downsample,
            num_residual_blocks=config.num_residual_blocks,
            residual_hidden=config.residual_hidden,
        )
        self.encoder = Encoder(
            in_channels,
            hidden_channels=config.hidden_channels,
            embedding_dim=config.embedding_dim,
            **shared,
        )
        self.quantizer = VectorQuantizerEMA(
            config.num_codes,
            config.embedding_dim,
            commitment_cost=config.commitment_cost,
            decay=config.ema_decay,
            dead_code_steps=config.dead_code_steps,
        )
        self.decoder = Decoder(
            config.embedding_dim,
            hidden_channels=config.hidden_channels,
            out_channels=in_channels,
            **shared,
        )

    def forward(self, images: torch.Tensor) -> VQVAEOutput:
        quantized = self.quantizer(self.encoder(images))
        reconstruction = self.decoder(quantized.quantized)
        reconstruction_loss = F.mse_loss(reconstruction, images)

        return VQVAEOutput(
            reconstruction=reconstruction,
            loss=reconstruction_loss + quantized.commitment_loss,
            reconstruction_loss=reconstruction_loss,
            commitment_loss=quantized.commitment_loss,
            perplexity=quantized.perplexity,
            num_active=quantized.num_active,
            num_restarted=quantized.num_restarted,
            codes=quantized.codes,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return the grid of code indices for a batch of images."""
        return self.quantizer(self.encoder(images)).codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from a grid of code indices."""
        return self.decoder(self.quantizer.lookup(codes))


def save_checkpoint(
    model: VQVAE,
    path: str | Path,
    *,
    config: VQVAEConfig,
    epoch: int,
    metrics: dict,
) -> None:
    """Write a checkpoint that can rebuild the model from the file alone."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        # Stored as a plain dict so the file reads back with weights_only=True.
        "config": asdict(config),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[VQVAE, dict]:
    """Rebuild a model and its metadata from a checkpoint.

    The model comes back in eval mode. Training mode is not part of a
    `state_dict`, so a freshly built module would otherwise default to
    training and quietly keep updating the codebook on whoever used it.
    """
    raw = torch.load(path, map_location=device, weights_only=True)
    config = VQVAEConfig(**raw["config"])

    model = VQVAE(config)
    model.load_state_dict(raw["model"])
    model.to(device)
    model.eval()

    return model, {
        "config": config,
        "epoch": raw["epoch"],
        "metrics": raw["metrics"],
    }
