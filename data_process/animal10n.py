"""ANIMAL-10N (Noisy) dataset loader.

Loads ANIMAL-10N (Song et al., 2019): 50K training + 5K test images,
10 animal classes, 64x64 RGB, ~8% real human label noise from five pairs
of confusing animals.

Sources:
- Available from the Activeloop hub mirror
  (hub://activeloop/animal10n-train / -test), stored as:
  data/animal10n/train_batch -> {'data': (50000,3,64,64) uint8, 'labels': (50000,)}
  data/animal10n/val_batch   -> {'data': (5000,3,64,64) uint8,  'labels': (5000,)}

NOISE NOTE: ANIMAL-10N does NOT release clean training labels (the noisy
human labels ARE the training labels). The test split is clean
(search-keyword filtered). So for this dataset the train split is always
used with its native noisy labels; ``noise_rate``/``noise_type`` are
accepted for interface compatibility and are IGNORED (a synthetic-noise
variant would need a clean-label oracle, which does not exist here).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from data_process.base import build_image_splits

# Normalization constants for ANIMAL-10N (ImageNet convention, 64x64)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _load_pickles(data_dir: str):
    """Load raw ANIMAL-10N train/test pickles.

    Returns ``(X_train_all, y_train_all, X_test, y_test)`` where train
    labels are the native noisy human labels and test labels are clean.
    """
    base = Path(data_dir) / "animal10n"
    train_pkl = base / "train_batch"
    test_pkl = base / "val_batch"
    if not train_pkl.exists() or not test_pkl.exists():
        raise FileNotFoundError(
            f"ANIMAL-10N pickles not found in {base}. "
            "Download via scripts/dl_animal10n.py (Activeloop mirror).")

    with train_pkl.open("rb") as f:
        train_data = pickle.load(f)
    with test_pkl.open("rb") as f:
        test_data = pickle.load(f)

    X_train_all = train_data["data"]     # (50000, 3, 64, 64) uint8
    y_train_all = np.asarray(train_data["labels"])
    X_test = test_data["data"]           # (5000, 3, 64, 64) uint8
    y_test = np.asarray(test_data["labels"])
    return X_train_all, y_train_all, X_test, y_test


def _load_animal10n(data_dir: str, noise_rate: float = 0.0,
                    noise_type: str = "symmetric", random_state: int = 42):
    """Load ANIMAL-10N with its native real human noisy labels.

    ``noise_rate`` and ``noise_type`` are accepted for interface
    compatibility with get_dataset() and are IGNORED (no clean training
    oracle exists for this dataset).

    Returns (train, val, test, feature_types) compatible with get_dataset().
    """

    X_train_all, y_train_all, X_test, y_test = _load_pickles(data_dir)
    train_ds, val_ds, test_ds = build_image_splits(
        X_train_all, y_train_all, X_test, y_test, random_state,
        mean=_MEAN, std=_STD)

    print(f"ANIMAL-10N: native noisy labels, {len(X_train_all)} train / {len(y_test)} test, "
          f"actual noise rate ~8% (no clean oracle)")
    return train_ds, val_ds, test_ds, None


# ---------------------------------------------------------------------------
# Row-aligned pair loader for the LSNPC image path.
#
# ANIMAL-10N releases no clean training labels, so unlike cifar10n/eurosat
# there is no train-side oracle to return. y_clean is a -1 sentinel,
# deliberately NOT a copy of the noisy labels: any oracle-dependent computation
# (per-query validity, mis-slice AUROC) is meaningless here and must fail
# loudly rather than be reported against a fabricated clean reference.
#
# The only trusted labels are the clean 5k test split. It is carved disjointly:
# the head becomes the semi-supervised clean set, the remainder the held-out
# evaluation pool. Both share one source, which the appendix must state.
# ---------------------------------------------------------------------------
ANIMAL10N_CLEAN_POOL = 500          # the clean set the pipeline carves
ANIMAL10N_PIPELINE_ROWS = 1000      # clean set + validation, carved from the test split
ANIMAL10N_ORACLE_ABSENT = -1


def _norm_u8(X: np.ndarray) -> np.ndarray:
    """uint8 (N,3,64,64) -> float32, ImageNet-normalised."""
    mean = np.asarray(_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return ((X.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)


def _clean_carve(n_test: int, seed: int = 42):
    """Split the clean test split into (clean-pool indices, held-out indices).

    A seeded random carve, not a positional head: the clean pool must not
    inherit any ordering skew of the pickle, and the complement is the
    held-out evaluation pool, disjoint by construction.
    """
    perm = np.random.default_rng(seed).permutation(n_test)
    return perm[:ANIMAL10N_CLEAN_POOL], perm[ANIMAL10N_CLEAN_POOL:]


def load_animal10n_clean_pool(data_dir: str):
    """The whole clean test split, for the pipeline to carve from.

    Returns all 5000 labelled rows; the pipeline takes the clean set and the
    validation out of it and leaves the rest as the held-out evaluation pool.
    """
    _, _, X_test, y_test = _load_pickles(data_dir)
    return _norm_u8(X_test), np.asarray(y_test, dtype=np.int64)


def load_animal10n_pair(data_dir: str, noise_rate: float = 0.0,
                        noise_type: str = "symmetric", random_state: int = 42):
    """Return (X, y_noisy, y_clean, X_test, y_test, mean, std).

    The pair loaders' contract, with this dataset's real-noise difference
    stated: noise_rate/noise_type are ignored (the human labels are the noise),
    y_clean is the -1 sentinel (no train oracle exists), and the held-out test
    is the tail of the clean test split, disjoint from the clean pool above.
    """
    X_train, y_noisy, X_test_all, y_test_all = _load_pickles(data_dir)
    n = int(X_train.shape[0])
    assert X_train.shape == (n, 3, 64, 64), X_train.shape

    X = _norm_u8(X_train)
    y_noisy = np.asarray(y_noisy, dtype=np.int64)
    y_clean = np.full(n, ANIMAL10N_ORACLE_ABSENT, dtype=np.int64)

    y_test_all = np.asarray(y_test_all, dtype=np.int64)
    # Held-out evaluation pool: the tail of the clean test split, disjoint of the
    # rows the pipeline carves for the clean set and the validation.
    held_idx = _clean_carve(len(y_test_all))[1][ANIMAL10N_PIPELINE_ROWS:]
    X_test = _norm_u8(X_test_all[held_idx])
    y_test = y_test_all[held_idx]

    print(f"ANIMAL-10N: {n} noisy train rows, {int(np.unique(y_noisy).size)} classes, "
          f"clean set {ANIMAL10N_CLEAN_POOL}, held-out test {int(y_test.size)}, "
          f"train oracle ABSENT (sentinel {ANIMAL10N_ORACLE_ABSENT})")
    return X, y_noisy, y_clean, X_test, y_test, _MEAN, _STD
