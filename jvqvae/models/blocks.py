"""Convolutional encoder and decoder around the quantizer.

Both are plain stacks with no normalization layers: at this data scale the
model is small enough that batch statistics add variance without buying
stability, and keeping them out means a batch of one behaves like a batch of
many at sampling time.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block, 3x3 then 1x1."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)


class Encoder(nn.Module):
    """Downsample an image to a grid of embedding vectors."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int,
        embedding_dim: int,
        num_downsample: int,
        num_residual_blocks: int,
        residual_hidden: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for stage in range(num_downsample):
            # Widen once at the start, then hold: the first stage sees raw
            # pixels and does not need the full width.
            out = hidden_channels // 2 if stage == 0 else hidden_channels
            layers += [
                nn.Conv2d(channels, out, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            ]
            channels = out

        layers.append(nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1))
        layers += [
            ResidualBlock(hidden_channels, residual_hidden)
            for _ in range(num_residual_blocks)
        ]
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class Decoder(nn.Module):
    """From a grid of embedding vectors back to an image.

    Undoes the encoder's stride-2 stages in reverse, so the spatial sizes match
    exactly. The channel widths are not a literal mirror: the taper holds at
    half width for every stage but the last, rather than retracing the
    encoder's widths backwards.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        hidden_channels: int,
        out_channels: int,
        num_downsample: int,
        num_residual_blocks: int,
        residual_hidden: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1)
        ]
        layers += [
            ResidualBlock(hidden_channels, residual_hidden)
            for _ in range(num_residual_blocks)
        ]
        layers.append(nn.ReLU())

        channels = hidden_channels
        for stage in range(num_downsample):
            is_last = stage == num_downsample - 1
            out = out_channels if is_last else hidden_channels // 2
            layers.append(
                nn.ConvTranspose2d(channels, out, kernel_size=4, stride=2, padding=1)
            )
            if not is_last:
                layers.append(nn.ReLU())
            channels = out

        # Linear last layer. A sigmoid saturates MSE: once logits go large and
        # negative, sigmoid'(z) ~ 0 and the decoder cannot recover. Clip when
        # drawing figures, not in the forward pass.
        self.net = nn.Sequential(*layers)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)
