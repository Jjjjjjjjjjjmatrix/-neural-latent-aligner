"""Trial Warping Module (TWM) : alignement temporel inter-essais.

Produit une matrice d'alignement A (B, T', T') telle que
    s^{x'->x}_t = sum_{t'} A[t, t'] * s^{x'}_{t'}
avec une penalite de monotonie/regularite Omega(A).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrialWarpingModule(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        proj_dim: int = 64,
        temperature: float | None = None,
        monotonic_prior_sigma: float = 4.0,
    ) -> None:
        super().__init__()
        self.q = nn.Conv1d(latent_dim, proj_dim, 1)
        self.k = nn.Conv1d(latent_dim, proj_dim, 1)
        self.scale = temperature or math.sqrt(proj_dim)
        self.sigma = monotonic_prior_sigma

    def _diagonal_prior(self, t: int, device: torch.device) -> torch.Tensor:
        """Biais gaussien centre sur la diagonale : favorise les warps locaux."""
        idx = torch.arange(t, device=device, dtype=torch.float32)
        d = idx[:, None] - idx[None, :]
        return -(d**2) / (2 * self.sigma**2)

    def forward(self, s_x: torch.Tensor, s_xp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: s_x, s_xp de forme (B, d, T'). Retourne (s_aligned, A)."""
        q = self.q(s_x).transpose(1, 2)          # (B, T', p)
        k = self.k(s_xp).transpose(1, 2)         # (B, T', p)
        logits = q @ k.transpose(1, 2) / self.scale
        logits = logits + self._diagonal_prior(logits.size(-1), logits.device)
        a = F.softmax(logits, dim=-1)            # (B, T', T')
        s_aligned = a @ s_xp.transpose(1, 2)     # (B, T', d)
        return s_aligned.transpose(1, 2), a

    @staticmethod
    def monotonicity_penalty(a: torch.Tensor) -> torch.Tensor:
        """Omega(A) : penalise les barycentres non croissants et non lisses."""
        t = a.size(-1)
        pos = torch.arange(t, device=a.device, dtype=a.dtype)
        center = (a * pos).sum(-1)                       # (B, T') barycentre du warp
        diff = center[:, 1:] - center[:, :-1]
        non_monotone = F.relu(-diff).pow(2).mean()       # doit etre croissant
        roughness = (diff[:, 1:] - diff[:, :-1]).pow(2).mean()
        return non_monotone + 0.1 * roughness
