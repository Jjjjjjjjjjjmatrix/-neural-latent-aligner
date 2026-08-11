"""Encodeur / décodeur convolutifs 1D (branche du haut de la Figure 1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """f_theta : x (B, E, T) -> s^x (B, d, T')."""

    def __init__(
        self,
        n_electrodes: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 3,
        kernel_size: int = 5,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = [n_electrodes] + [hidden_dim] * (n_layers - 1) + [latent_dim]
        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers += [
                nn.Conv1d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                ),
                nn.GroupNorm(8, dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.stride = stride
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """h_psi : s^x (B, d, T') -> x_hat (B, E, T)."""

    def __init__(
        self,
        n_electrodes: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 3,
        kernel_size: int = 5,
        stride: int = 2,
    ) -> None:
        super().__init__()
        dims = [latent_dim] + [hidden_dim] * (n_layers - 1) + [n_electrodes]
        layers: list[nn.Module] = []
        for i in range(n_layers):
            last = i == n_layers - 1
            layers.append(
                nn.ConvTranspose1d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    output_padding=stride - 1,
                )
            )
            if not last:
                layers += [nn.GroupNorm(8, dims[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, out_len: int | None = None) -> torch.Tensor:
        x_hat = self.net(s)
        if out_len is not None:
            x_hat = x_hat[..., :out_len]
        return x_hat


class ContentHead(nn.Module):
    """g_phi : s^x (B, d, T') -> c^x (B, k, T'), goulot d'etranglement (k < d)."""

    def __init__(self, latent_dim: int = 128, content_dim: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        assert content_dim < latent_dim, "k doit etre < d (goulot d'etranglement)"
        self.conv = nn.Conv1d(latent_dim, content_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.conv(s)
