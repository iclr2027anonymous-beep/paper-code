"""CIFAR-10N (Noisy) dataset loader.

Loads CIFAR-10 from local pickle + human-annotated noisy labels
from the CIFAR-10N dataset (Wei et al., 2021).

Sources:
- Images: data/cifar10_train.pkl, data/cifar10_test.pkl (rebuilt from the
  ORIGINAL cifar-10-python.tar.gz archive)
- Noisy labels: https://github.com/UCSC-REAL/cifar-10-100n

ALIGNMENT REQUIREMENT (row-order requirement): the CIFAR-10N
human labels are keyed in ORIGINAL CIFAR-10 row order. The HuggingFace
parquet (`uoft-cs/cifar10/plain_text`) is NOT in original order (rows
shuffled, images JPEG-re-encoded), so pickles converted from the parquet
silently misalign with the human labels: image-vs-label agreement drops
to chance (~9.8%), and classifiers trained on the pairs collapse to
random accuracy. The pickles must be built from the original archive;
see scripts/rebuild_cifar_original_order.py.
"""
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import torch

from data_process.base import build_image_splits, inject_label_noise

CIFAR10N_URL = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-10_human.pt"

# Normalization constants for CIFAR-10 (PyTorch convention)
_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def _load_cifar_pickles(data_dir: str):
    """Load the raw CIFAR-10 train/test pickles (shared by cifar10/cifar10n).

    Returns ``(X_train_all, y_clean_all, X_test, y_test)`` where the train
    arrays are ``(50000, 3, 32, 32) uint8`` and the test arrays are
    ``(10000, 3, 32, 32) uint8`` (test labels always clean).
    """

    data_path = Path(data_dir)

    train_pkl = data_path / "cifar10_train.pkl"
    test_pkl = data_path / "cifar10_test.pkl"
    if not train_pkl.exists() or not test_pkl.exists():
        raise FileNotFoundError(
            f"CIFAR-10 pickles not found in {data_dir}. "
            "Run scripts/rebuild_cifar_original_order.py first.")

    with train_pkl.open("rb") as f:
        train_data = pickle.load(f)
    with test_pkl.open("rb") as f:
        test_data = pickle.load(f)

    X_train_all = train_data["images"]  # (50000, 3, 32, 32) uint8
    y_clean_all = train_data["labels"]
    X_test = test_data["images"]        # (10000, 3, 32, 32) uint8
    y_test = test_data["labels"]

    return X_train_all, y_clean_all, X_test, y_test


def _load_cifar10(data_dir: str, noise_rate: float = 0.0,
                  noise_type: str = "symmetric", random_state: int = 42):
    """Load CIFAR-10 with SYNTHETIC label noise (``cifar10`` dataset).

    Uses the ground-truth ``clean_label`` from the train pickle and injects
    symmetric/pairflip label noise into the TRAIN split only via
    ``inject_label_noise`` (symmetric works without features; instance noise
    is NOT supported for images). The 85/15 split and normalization match
    ``_load_cifar10n``.

    Returns (train, val, test, feature_types) compatible with get_dataset().
    """

    X_train_all, y_clean_all, X_test, y_test = _load_cifar_pickles(data_dir)

    # Ground-truth labels with synthetic noise injected on the train split.
    y_clean_all = np.asarray(y_clean_all)
    y_train_all = inject_label_noise(
        torch.as_tensor(y_clean_all, dtype=torch.long),
        noise_rate, noise_type, random_state,
    ).numpy()

    actual_noise = float((y_train_all != y_clean_all).mean())
    print(f"CIFAR-10 (synthetic): noise_type={noise_type} "
          f"target={noise_rate:.2f}, actual noise rate = {actual_noise:.2%}")

    train_ds, val_ds, test_ds = build_image_splits(
        X_train_all, y_train_all, X_test, y_test, random_state,
        mean=_MEAN, std=_STD)
    return train_ds, val_ds, test_ds, None


def _load_cifar10n(data_dir: str, noise_rate: float = 0.0,
                   noise_type: str = "symmetric", random_state: int = 42):
    """Load CIFAR-10 with real human noisy labels (CIFAR-10N).

    The ``noise_rate`` and ``noise_type`` arguments map to:
    - ``"symmetric"`` or ``"aggre"`` → aggre_label (~9% noise)
    - ``"worse"``                  → worse_label (~40% noise)
    - ``"clean"``                  → clean_label (0% noise)

    ``noise_rate`` is IGNORED — labels are the fixed real human labels.

    Returns (train, val, test, feature_types) compatible with get_dataset().
    """
    X_train_all, y_clean_all, X_test, y_test = _load_cifar_pickles(data_dir)

    data_path = Path(data_dir)

    # Download CIFAR-10N human labels if not present
    cifar10n_path = data_path / "CIFAR-10N_human.pt"
    if not cifar10n_path.exists():
        print(f"Downloading CIFAR-10N human labels to {cifar10n_path} ...")
        urllib.request.urlretrieve(CIFAR10N_URL, str(cifar10n_path))
        print("Done.")

    cifar10n = torch.load(str(cifar10n_path), map_location="cpu", weights_only=False)
    clean_label = np.asarray(cifar10n["clean_label"])
    aggre_label = np.asarray(cifar10n["aggre_label"])
    worse_label = np.asarray(cifar10n["worse_label"])

    # Select label set based on noise_type
    if noise_type in ("symmetric", "aggre", "pairflip"):
        noisy_label = aggre_label
        actual_noise = float((aggre_label != clean_label).mean())
    elif noise_type == "worse":
        noisy_label = worse_label
        actual_noise = float((worse_label != clean_label).mean())
    elif noise_type == "clean":
        noisy_label = clean_label
        actual_noise = 0.0
    else:
        raise ValueError(f"Unknown noise_type for CIFAR-10N: {noise_type!r}")

    print(f"CIFAR-10N: using {noise_type} labels, actual noise rate = {actual_noise:.2%}")

    train_ds, val_ds, test_ds = build_image_splits(
        X_train_all, noisy_label, X_test, y_test, random_state,
        mean=_MEAN, std=_STD)
    return train_ds, val_ds, test_ds, None
