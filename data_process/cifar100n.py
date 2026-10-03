"""CIFAR-100N (Noisy) dataset loader.

Loads CIFAR-100 from local pickle + human-annotated noisy labels from the
CIFAR-100N dataset (Wei et al., 2022).

Sources:
- Images: data/cifar100_train.pkl, data/cifar100_test.pkl (rebuilt from the
  HuggingFace parquet; row order VERIFIED aligned with the
  human labels, fine-label agreement 1.0)
- Noisy labels: data/CIFAR-100_human.pt
  https://github.com/UCSC-REAL/cifar-10-100n

ALIGNMENT NOTE : unlike CIFAR-10's parquet (block-sorted rows,
9.8% agreement), the uoft-cs/cifar100 parquet preserves original CIFAR-100
row order. Verified: parquet fine_label vs human clean_label agreement
= 1.0000 before the pickles were written.

Label sets available:
- fine: 100 classes, clean_label / noisy_label (noise rate 40.2%)
- coarse: 20 superclasses, clean_coarse_label / noisy_coarse_label (25.6%)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from data_process.base import build_image_splits, inject_label_noise

# Normalization constants for CIFAR-100 (PyTorch convention)
_MEAN = (0.5071, 0.4867, 0.4408)
_STD = (0.2675, 0.2565, 0.2761)


def _load_cifar100_pickles(data_dir: str):
    """Load raw CIFAR-100 train/test pickles.

    Returns ``(X_train_all, y_clean_all, X_test, y_test)``: train arrays
    (50000, 3, 32, 32) uint8, test arrays (10000, 3, 32, 32) uint8.
    """
    data_path = Path(data_dir)
    train_pkl = data_path / "cifar100_train.pkl"
    test_pkl = data_path / "cifar100_test.pkl"
    if not train_pkl.exists() or not test_pkl.exists():
        raise FileNotFoundError(
            f"CIFAR-100 pickles not found in {data_dir}. "
            "Run scripts/rebuild_cifar100_from_parquet.py first.")

    with train_pkl.open("rb") as f:
        train_data = pickle.load(f)
    with test_pkl.open("rb") as f:
        test_data = pickle.load(f)

    X_train_all = train_data["images"]
    y_clean_all = train_data["labels"]
    X_test = test_data["images"]
    y_test = test_data["labels"]
    return X_train_all, y_clean_all, X_test, y_test


def _load_cifar100(data_dir: str, noise_rate: float = 0.0,
                   noise_type: str = "symmetric", random_state: int = 42):
    """Load CIFAR-100 with SYNTHETIC label noise (``cifar100`` dataset)."""

    X_train_all, y_clean_all, X_test, y_test = _load_cifar100_pickles(data_dir)
    y_clean_all = np.asarray(y_clean_all)
    y_train_all = inject_label_noise(
        torch.as_tensor(y_clean_all, dtype=torch.long),
        noise_rate, noise_type, random_state,
    ).numpy()
    actual_noise = float((y_train_all != y_clean_all).mean())
    print(f"CIFAR-100 (synthetic): noise_type={noise_type} "
          f"target={noise_rate:.2f}, actual noise rate = {actual_noise:.2%}")

    train_ds, val_ds, test_ds = build_image_splits(
        X_train_all, y_train_all, X_test, y_test, random_state,
        mean=_MEAN, std=_STD)
    return train_ds, val_ds, test_ds, None


def _load_cifar100n(data_dir: str, noise_rate: float = 0.0,
                    noise_type: str = "symmetric", random_state: int = 42):
    """Load CIFAR-100 with real human noisy labels (CIFAR-100N).

    ``noise_type`` maps to:
    - ``"symmetric"`` / ``"noisy"`` / ``"fine"`` -> fine noisy_label (~40% noise)
    - ``"coarse"``                            -> coarse noisy_coarse_label (~26% noise)
    - ``"clean"``                             -> clean labels (0% noise)
    ``noise_rate`` is IGNORED — labels are the fixed real human labels.

    Returns (train, val, test, feature_types) compatible with get_dataset().
    """

    X_train_all, y_clean_all, X_test, y_test = _load_cifar100_pickles(data_dir)

    data_path = Path(data_dir)
    cifar100n_path = data_path / "CIFAR-100_human.pt"
    if not cifar100n_path.exists():
        raise FileNotFoundError(
            f"CIFAR-100N human labels not found: {cifar100n_path}. "
            "Download from https://github.com/UCSC-REAL/cifar-10-100n")

    hn = torch.load(str(cifar100n_path), map_location="cpu", weights_only=False)
    clean_label = np.asarray(hn["clean_label"])
    noisy_label = np.asarray(hn["noisy_label"])
    clean_coarse = np.asarray(hn["clean_coarse_label"])
    noisy_coarse = np.asarray(hn["noisy_coarse_label"])

    if noise_type in ("symmetric", "noisy", "fine", "pairflip"):
        noisy = noisy_label
        clean = clean_label
        actual_noise = float((noisy != clean).mean())
    elif noise_type == "coarse":
        noisy = noisy_coarse
        clean = clean_coarse
        actual_noise = float((noisy != clean).mean())
    elif noise_type == "clean":
        noisy = clean_label
        clean = clean_label
        actual_noise = 0.0
    else:
        raise ValueError(f"Unknown noise_type for CIFAR-100N: {noise_type!r}")

    print(f"CIFAR-100N: using {noise_type} labels, actual noise rate = {actual_noise:.2%}")

    train_ds, val_ds, test_ds = build_image_splits(
        X_train_all, noisy, X_test, y_test, random_state,
        mean=_MEAN, std=_STD)
    return train_ds, val_ds, test_ds, None
