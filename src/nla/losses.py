"""Fonctions de cout : reconstruction, calibration, regularisation du warp."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """L_rec = E[||x - x_hat||^2]."""
    return F.mse_loss(x_hat, x)


def calibration_loss(c_x: torch.Tensor, c_aligned: torch.Tensor) -> torch.Tensor:
    """L_cal = ||c^x - g_phi(s^{x'->x})||^2 (version L2)."""
    return F.mse_loss(c_aligned, c_x)


def contrastive_calibration_loss(
    c_x: torch.Tensor, c_aligned: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Variante InfoNCE : les negatifs sont les autres pas de temps du meme essai."""
    a = F.normalize(c_x.transpose(1, 2), dim=-1)        # (B, T', k)
    b = F.normalize(c_aligned.transpose(1, 2), dim=-1)  # (B, T', k)
    logits = a @ b.transpose(1, 2) / temperature        # (B, T', T')
    target = torch.arange(logits.size(1), device=logits.device)
    target = target.unsqueeze(0).expand(logits.size(0), -1)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


def total_loss(
    x: torch.Tensor,
    out: dict[str, torch.Tensor],
    lambda_cal: float = 1.0,
    beta_warp: float = 0.1,
    contrastive: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = L_rec + lambda * L_cal + beta * Omega(A)."""
    from .models.twm import TrialWarpingModule

    l_rec = reconstruction_loss(x, out["x_hat"])
    cal_fn = contrastive_calibration_loss if contrastive else calibration_loss
    l_cal = cal_fn(out["c_x"], out["c_aligned"])
    l_warp = TrialWarpingModule.monotonicity_penalty(out["A"])
    loss = l_rec + lambda_cal * l_cal + beta_warp * l_warp
    return loss, {
        "loss": float(loss.detach()),
        "rec": float(l_rec.detach()),
        "cal": float(l_cal.detach()),
        "warp": float(l_warp.detach()),
    }
