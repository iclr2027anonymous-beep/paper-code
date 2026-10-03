"""Text dataset loading for the LSNPC text path.

``TextDataset`` mirrors ``ImageDataset``: it holds frozen
sentence embeddings (``.features``/``.images``) and integer labels
(``.targets``/``.labels``), with optional label-noise injection, so it
interoperates with the shared LSNPC stage-1 semi-supervised code.

``load_sst2`` loads SST-2 (Stanford Sentiment Treebank, binary sentiment) from
local parquet, embeds the sentences with a ``TextEncoder``, and returns
(train, val, test, feature_types) compatible with ``get_dataset``.

NOTE: SST-2's test split labels are all ``-1`` (held out), so the test split is
returned with clean labels only for use as a semi-supervised clean set where
the caller supplies an oracle; for the clean-label oracle use the train/val
split.
"""
from __future__ import annotations

from utils.paths import project_path

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_process.base import inject_label_noise

log = logging.getLogger(__name__)

_SST2_DIR = project_path('data/sst2')


class TextDataset(Dataset):
    """Frozen sentence-embedding dataset with optional label noise.

    ``features`` holds (n, embed_dim) float32 embeddings; ``targets`` holds
    integer labels. Aliases ``images``/``labels`` mirror ``ImageDataset`` so the
    shared LSNPC code reads either.
    """

    def __init__(self, features: np.ndarray | torch.Tensor,
                 targets: np.ndarray | torch.Tensor,
                 noise_rate: float = 0.0,
                 noise_type: str = "symmetric",
                 random_state: int = 42) -> None:
        self.features = torch.tensor(features) if not torch.is_tensor(features) else features
        self.features = self.features.float()
        self.targets = torch.tensor(targets, dtype=torch.long) if not torch.is_tensor(targets) else targets.long()
        if noise_rate > 0:
            self.targets = inject_label_noise(
                self.targets, noise_rate, noise_type, random_state)
        # Compatibility aliases (mirror ImageDataset).
        self.images = self.features
        self.labels = self.targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def load_sst2(encoder, data_dir: str = _SST2_DIR,
              noise_rate: float = 0.0, noise_type: str = "symmetric",
              random_state: int = 42):
    """Load SST-2, embed with ``encoder``, return (train, val, test, feature_types).

    An 85/15 train/val split mirrors the image/tabular loaders. The test split's
    ``-1`` labels are left as-is (held out) so it cannot corrupt the clean oracle.
    """
    root = Path(data_dir)
    tr = pd.read_parquet(root / "train-00000-of-00001.parquet")
    te = pd.read_parquet(root / "test-00000-of-00001.parquet")

    n_train = len(tr)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_train)
    split = int(0.85 * n_train)
    train_idx, val_idx = perm[:split], perm[split:]

    train_text = tr["sentence"].tolist()
    X_train = encoder.encode_text([train_text[i] for i in train_idx])
    X_val = encoder.encode_text([train_text[i] for i in val_idx])
    X_test = encoder.encode_text(te["sentence"].tolist())

    y_train = tr["label"].to_numpy()[train_idx]
    y_val = tr["label"].to_numpy()[val_idx]
    y_test = te["label"].to_numpy()  # all -1 (held out)

    train_ds = TextDataset(X_train, y_train, noise_rate, noise_type, random_state)
    val_ds = TextDataset(X_val, y_val, noise_rate, noise_type, random_state)
    test_ds = TextDataset(X_test, y_test, 0.0, "symmetric", random_state)
    log.info(f"SST-2: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test, "
             f"embed_dim={X_train.shape[1]}")
    return train_ds, val_ds, test_ds, None


# ── Generic parquet text loader (multiclass) ────────────────────────────────

def load_ag_news(encoder, data_dir: str = _SST2_DIR,
                 noise_rate: float = 0.0, noise_type: str = "symmetric",
                 random_state: int = 42):
    """Load AG News (4-class topic classification), embed, return splits.

    Mirrors ``load_sst2``; AG News has a real (non-held-out) test label set.
    """
    return _load_parquet_text(
        "ag_news", encoder, data_dir, text_col="text",
        noise_rate=noise_rate, noise_type=noise_type, random_state=random_state)


def _load_parquet_text(name: str, encoder, data_dir: str, text_col: str,
                       noise_rate: float, noise_type: str, random_state: int):
    """Shared loader: read {name}_*/parquet, embed, 85/15 split, build TextDatasets."""
    root = Path(data_dir)
    tr = pd.read_parquet(root / "train-00000-of-00001.parquet")
    te = pd.read_parquet(root / "test-00000-of-00001.parquet")

    n_train = len(tr)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_train)
    split = int(0.85 * n_train)
    train_idx, val_idx = perm[:split], perm[split:]

    texts = tr[text_col].tolist()
    X_train = encoder.encode_text([texts[i] for i in train_idx])
    X_val = encoder.encode_text([texts[i] for i in val_idx])
    X_test = encoder.encode_text(te[text_col].tolist())

    y_train = tr["label"].to_numpy()[train_idx]
    y_val = tr["label"].to_numpy()[val_idx]
    y_test = te["label"].to_numpy()

    train_ds = TextDataset(X_train, y_train, noise_rate, noise_type, random_state)
    val_ds = TextDataset(X_val, y_val, noise_rate, noise_type, random_state)
    test_ds = TextDataset(X_test, y_test, 0.0, "symmetric", random_state)
    n_cls = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    log.info(f"{name}: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test, "
             f"{n_cls} classes, embed_dim={X_train.shape[1]}")
    return train_ds, val_ds, test_ds, None
