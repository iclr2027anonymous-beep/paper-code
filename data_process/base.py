"""Dataset loading for the image and text experiments.

Each loader returns ``(train, val, test, feature_types, na_mask)``; both
``feature_types`` and ``na_mask`` are ``None`` for the datasets supported
here (images and frozen text embeddings).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from models.text_encoder import build_text_encoder


class SizedDataset(Protocol):
    """``torch.utils.data.Dataset`` declares no ``__len__``; this adds it."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Any: ...


class SplitsDataset(SizedDataset, Protocol):
    """What `get_dataset` returns: sized, indexable, with feature/target tensors."""

    features: torch.Tensor
    targets: torch.Tensor


class ImageDataset(Dataset):
    """Local image dataset."""

    def __init__(self, images: np.ndarray | torch.Tensor,
                 labels: np.ndarray | torch.Tensor,
                 noise_rate: float = 0.0,
                 noise_type: str = "symmetric",
                 random_state: int = 42) -> None:
        self.images = torch.tensor(images) if not torch.is_tensor(images) else images
        self.labels = torch.tensor(labels, dtype=torch.long) if not torch.is_tensor(labels) else labels
        if noise_rate > 0:
            self.labels = inject_label_noise(self.labels, noise_rate, noise_type, random_state)
        # Compatibility aliases: run_lsnpc_stage1 reads `.features`/`.targets`.
        self.features = self.images
        self.targets = self.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def build_image_splits(X_train_all, y_train_all, X_test, y_test,
                       random_state: int = 42,
                       mean: tuple = (0.5, 0.5, 0.5),
                       std: tuple = (0.5, 0.5, 0.5)):
    """Apply the 85/15 train/val split + normalization and build ImageDatasets.

    Shared by the CIFAR-10/CIFAR-100 loaders: ``y_train_all`` is the (noisy
    or clean) label array used for BOTH the train and val splits; test
    labels are always clean.  ``mean``/``std`` are the per-dataset channel
    normalization constants (PyTorch convention).

    Returns ``(train_ds, val_ds, test_ds)``.
    """
    n_train = len(X_train_all)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_train)
    split = int(0.85 * n_train)
    train_idx = perm[:split]
    val_idx = perm[split:]

    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)

    def _normalize(images: np.ndarray) -> torch.Tensor:
        """uint8 [0,255] -> float32 normalized tensor (N, 3, 32, 32)."""
        x = torch.from_numpy(images).float() / 255.0
        return (x - mean_t) / std_t

    X_train = _normalize(np.asarray(X_train_all)[train_idx])
    X_val = _normalize(np.asarray(X_train_all)[val_idx])
    y_train = torch.from_numpy(np.asarray(y_train_all)[train_idx]).long()
    y_val = torch.from_numpy(np.asarray(y_train_all)[val_idx]).long()

    X_test = _normalize(np.asarray(X_test))
    y_test = torch.from_numpy(np.asarray(y_test)).long()  # test always clean

    return (
        ImageDataset(X_train, y_train),
        ImageDataset(X_val, y_val),
        ImageDataset(X_test, y_test),
    )


def inject_label_noise(labels: torch.Tensor, noise_rate: float,
                       noise_type: str, random_state: int = 42) -> torch.Tensor:
    """Inject label noise of the specified type.

    Supported noise types:
    - ``symmetric``: uniform flip to any other class (``noise_rate`` fraction).
    - ``pairflip``: flip ``0↔1`` only (binary) or to the adjacent class
      (multi-class, cyclical: ``0→1, 1→2, …, K→0``).

    Instance-dependent noise is *not* generated here. The IDN streams this
    paper reports come from ``experiments.protocol.inject_noise_idn``, which
    scores every row against every class centroid; this helper keeps only the
    label-only generators.
    """
    rng = np.random.RandomState(random_state)
    labels = labels.clone()
    n_classes = int(labels.max().item()) + 1

    if noise_type == "symmetric" or noise_rate == 0.0:
        return _inject_symmetric(labels, noise_rate, rng, n_classes)
    elif noise_type == "pairflip":
        return _inject_pairflip(labels, noise_rate, rng, n_classes)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


def _inject_symmetric(labels: torch.Tensor, noise_rate: float,
                      rng: np.random.RandomState,
                      n_classes: int) -> torch.Tensor:
    """Uniform symmetric noise across all classes."""
    n = len(labels)
    flip_mask = rng.rand(n) < noise_rate
    flip_idx = np.where(flip_mask)[0]
    if flip_idx.size == 0:
        return labels
    new_labels = rng.randint(0, n_classes, size=flip_idx.size)
    orig = labels[flip_idx].cpu().numpy()
    same = new_labels == orig
    max_redraw = max(8, int(np.ceil(np.log2(max(2, flip_idx.size + 1)))) + 4)
    for _ in range(max_redraw):
        if not same.any():
            break
        redraw = rng.randint(0, n_classes, size=int(same.sum()))
        new_labels[same] = redraw
        same = new_labels == orig
    labels[flip_idx] = torch.as_tensor(new_labels, dtype=labels.dtype)
    return labels


def _inject_pairflip(labels: torch.Tensor, noise_rate: float,
                     rng: np.random.RandomState,
                     n_classes: int) -> torch.Tensor:
    """Pair-flip noise: 0↔1 for binary, cyclical adjacency for multi-class."""
    n = len(labels)
    flip_mask = rng.rand(n) < noise_rate
    flip_idx = np.where(flip_mask)[0]
    if flip_idx.size == 0:
        return labels

    if n_classes == 2:
        labels[flip_idx] = 1 - labels[flip_idx]
    else:
        # Cyclical: 0→1, 1→2, …, K-1→0
        flipped = (labels[flip_idx] + 1) % n_classes
        labels[flip_idx] = flipped
    return labels


def get_dataset(name: str, data_dir: str,
                noise_rate: float = 0.0,
                noise_type: str = "symmetric",
                random_state: int = 42,
                text_encoder=None,
                ) -> tuple[SplitsDataset, SplitsDataset, SplitsDataset,
                           list[str] | None,
                           np.ndarray | None]:
    """Load dataset from local cache.

    Returns ``(train, val, test, feature_types, na_mask)``; both trailing
    elements are always ``None`` for the datasets supported here.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # The loaders below are imported lazily on purpose: each of them imports
    # this module at top level, so a module-level import here would be circular.
    if name == "cifar10":
        from data_process.cifar10n import _load_cifar10
        train, val, test, ft = _load_cifar10(
            data_dir, noise_rate=noise_rate, noise_type=noise_type,
            random_state=random_state)
        return train, val, test, ft, None

    elif name == "cifar10n":
        from data_process.cifar10n import _load_cifar10n
        train, val, test, ft = _load_cifar10n(
            data_dir, noise_rate=noise_rate, noise_type=noise_type,
            random_state=random_state)
        return train, val, test, ft, None

    elif name == "cifar100":
        from data_process.cifar100n import _load_cifar100
        train, val, test, ft = _load_cifar100(
            data_dir, noise_rate=noise_rate, noise_type=noise_type,
            random_state=random_state)
        return train, val, test, ft, None

    elif name == "cifar100n":
        from data_process.cifar100n import _load_cifar100n
        train, val, test, ft = _load_cifar100n(
            data_dir, noise_rate=noise_rate, noise_type=noise_type,
            random_state=random_state)
        return train, val, test, ft, None

    elif name == "animal10n":
        from data_process.animal10n import _load_animal10n
        train, val, test, ft = _load_animal10n(
            data_dir, noise_rate=noise_rate, noise_type=noise_type,
            random_state=random_state)
        return train, val, test, ft, None

    elif name == "sst2":
        # SST-2 needs a text encoder to produce embeddings. It is supplied via
        # the encoder_kwargs convention (see get_dataset signature) or a default.
        encoder = text_encoder
        if encoder is None:
            encoder = build_text_encoder()
        from data_process.text import load_sst2
        train, val, test, ft = load_sst2(
            encoder, data_dir=data_dir, noise_rate=noise_rate,
            noise_type=noise_type, random_state=random_state)
        return train, val, test, ft, None

    elif name == "ag_news":
        encoder = text_encoder
        if encoder is None:
            encoder = build_text_encoder()
        from data_process.text import load_ag_news
        train, val, test, ft = load_ag_news(
            encoder, data_dir=data_dir, noise_rate=noise_rate,
            noise_type=noise_type, random_state=random_state)
        return train, val, test, ft, None

    else:
        raise ValueError(f"Unknown dataset '{name}'. "
                         "Available: {cifar10, cifar10n, cifar100, cifar100n, "
                         "animal10n, sst2, ag_news}")
