"""Boucle d'entrainement minimale : python -m nla.train --config configs/default.yaml"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data.ecog_dataset import PairedTrialDataset
from .losses import total_loss
from .models.nla import NeuralLatentAligner


def build(cfg: dict) -> tuple[NeuralLatentAligner, DataLoader]:
    ds = PairedTrialDataset(cfg["data"]["path"])
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 2),
        drop_last=True,
    )
    model = NeuralLatentAligner(n_electrodes=ds.n_electrodes, **cfg["model"])
    return model, dl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(cfg["train"].get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, dl = build(cfg)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running: dict[str, float] = {}
        for batch in dl:
            x = batch["x"].to(device)
            x_prime = batch["x_prime"].to(device)

            out = model(x, x_prime)
            loss, logs = total_loss(
                x,
                out,
                lambda_cal=cfg["loss"]["lambda_cal"],
                beta_warp=cfg["loss"]["beta_warp"],
                contrastive=cfg["loss"].get("contrastive", False),
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v

        n = max(len(dl), 1)
        msg = " | ".join(f"{k}={v / n:.4f}" for k, v in running.items())
        print(f"epoch {epoch:03d} | {msg}", flush=True)
        torch.save(model.state_dict(), Path(args.out) / "last.pt")


if __name__ == "__main__":
    main()
