"""Dataset ECoG : retourne des paires d'essais partageant le meme stimulus."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedTrialDataset(Dataset):
    """Attend un .npz contenant :
        trials : (N, E, T) float32  -- signal haute-gamma, deja z-scored
        labels : (N,) int/str       -- identifiant de phrase/stimulus
    """

    def __init__(self, path: str | Path, augment: bool = True) -> None:
        data = np.load(Path(path), allow_pickle=True)
        self.trials = torch.from_numpy(data["trials"].astype("float32"))
        labels = list(data["labels"])
        self.augment = augment

        self.by_label: dict[object, list[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            self.by_label[lab].append(i)
        self.labels = labels

        singles = [k for k, v in self.by_label.items() if len(v) < 2]
        if singles:
            raise ValueError(
                f"{len(singles)} stimuli n'ont qu'un seul essai : "
                "NLA a besoin d'au moins 2 repetitions par stimulus."
            )

    @property
    def n_electrodes(self) -> int:
        return int(self.trials.shape[1])

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        candidates = [j for j in self.by_label[self.labels[idx]] if j != idx]
        partner = random.choice(candidates)
        x, x_prime = self.trials[idx], self.trials[partner]
        if self.augment:
            x = x + 0.01 * torch.randn_like(x)
        return {"x": x, "x_prime": x_prime, "index": torch.tensor(idx)}

