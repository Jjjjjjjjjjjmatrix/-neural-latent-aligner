"""Modele complet Neural Latent Aligner."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import ContentHead, ConvDecoder, ConvEncoder
from .twm import TrialWarpingModule


class NeuralLatentAligner(nn.Module):
    def __init__(
        self,
        n_electrodes: int,
        latent_dim: int = 128,
        content_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 3,
        proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(n_electrodes, latent_dim, hidden_dim, n_layers)
        self.decoder = ConvDecoder(n_electrodes, latent_dim, hidden_dim, n_layers)
        self.content = ContentHead(latent_dim, content_dim)
        self.twm = TrialWarpingModule(latent_dim, proj_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> dict[str, torch.Tensor]:
        """x, x_prime : (B, E, T), deux essais du meme stimulus."""
        s_x = self.encoder(x)
        s_xp = self.encoder(x_prime)

        x_hat = self.decoder(s_x, out_len=x.size(-1))

        s_aligned, a = self.twm(s_x, s_xp)

        return {
            "s_x": s_x,
            "s_xp": s_xp,
            "s_aligned": s_aligned,
            "x_hat": x_hat,
            "c_x": self.content(s_x),
            "c_aligned": self.content(s_aligned),
            "A": a,
        }

    @torch.no_grad()
    def extract_content(self, x: torch.Tensor) -> torch.Tensor:
        """Representation utilisee en aval (decodage de parole, etc.)."""
        return self.content(self.encoder(x))
