"""dopanim (Noisy Doppelganger Animals) loader for the LSNPC pixel path.

dopanim (NeurIPS 2024 Datasets & Benchmarks): 15,734 iNaturalist animal
images of 15 confusable (doppelganger) classes; 10,484 train images each
annotated by 3-10 humans with soft likelihoods over the 15 classes.
Clean oracle = iNaturalist-verified species (= class dir, verified 1.0);
noisy label = argmax of one seeded-annotator's likelihoods (~32% real
noise, the paper's rand-1 regime).

Prepared by scripts/dopanim_prep.py into data/dopanim/train_batch
({'data': (10484,3,S,S) uint8, 'labels': clean, 'noisy_labels': noisy}, where
S = the --size the prep ran at; the 224px Set-C probe rebuilds S=224).

Interface mirrors data_process/cifar100n.py so scripts/image_lsnpc.py's
load_image_pair works unchanged (pixel mode expects CIFAR-normalized
input; mean/std are the shared CIFAR constants the preprocess helper
undoes).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from data_process.base import build_image_splits

# Shared CIFAR constants: _pretrained_preprocess denormalizes by these, so
# any pixel-mode dataset must be normalized with the same values.
_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def _load_dopanim_pickle(data_dir: str):
    """Return (X (N,3,64,64) uint8, y_clean, y_noisy) row-aligned."""
    p = Path(data_dir) / "dopanim" / "train_batch"
    if not p.exists():
        raise FileNotFoundError(
            f"dopanim pickle not found at {p}. "
            "Run scripts/dopanim_prep.py first.")
    with p.open("rb") as f:
        d = pickle.load(f)
    X = np.asarray(d["data"])
    y_clean = np.asarray(d["labels"])
    y_noisy = np.asarray(d["noisy_labels"])
    assert X.ndim == 4 and X.shape[1] == 3, X.shape
    assert X.shape[0] == y_clean.shape[0] == y_noisy.shape[0]
    return X, y_clean, y_noisy


def _load_dopanim(data_dir: str, noise_rate: float = 0.0,
                  noise_type: str = "symmetric", random_state: int = 42):
    """Load dopanim; noise_type selects the label column:
    - ``"symmetric"``/``"noisy"``/``"rand"`` -> noisy_labels (single-annotator
      argmax, ~32% real human noise)
    - ``"clean"`` -> clean oracle (0% noise)
    ``noise_rate`` is IGNORED — labels are the fixed real human labels.
    """
    X_all, y_clean_all, y_noisy_all = _load_dopanim_pickle(data_dir)
    y_all = y_clean_all if noise_type == "clean" else y_noisy_all
    actual_noise = float((y_all != y_clean_all).mean())
    print(f"dopanim: noise_type={noise_type}, actual noise = {actual_noise:.2%}")

    # Fake a test pool so build_image_splits has the (X_test, y_test) slot;
    # the LSNPC data-level protocol never uses the test split (splits are
    # carved from the train pool by the runner), so pass an empty test.
    _, C, H, W = X_all.shape
    X_test = np.empty((0, C, H, W), dtype=np.uint8)
    y_test = np.empty(0, dtype=np.int64)
    train_ds, val_ds, test_ds = build_image_splits(
        X_all, y_all, X_test, y_test, random_state,
        mean=_MEAN, std=_STD)
    return train_ds, val_ds, test_ds, None


def load_clean_test_pool(data_dir: str):
    """Return the held-out CLEAN test split (X, y) — annotations exist only
    on the train split, so the test split is a guaranteed-clean source for
    the semi-supervised clean set of special real-noise datasets.

    X is (4500, 3, S, S) float32 CIFAR-normalized where S is the prep --size
    (must match the train-pool resolution; the same pixel pipeline consumes
    both).
    """
    p = Path(data_dir) / "dopanim" / "test_batch"
    if not p.exists():
        raise FileNotFoundError(
            f"dopanim test pickle not found at {p}. "
            "Run scripts/dopanim_prep.py first.")
    with p.open("rb") as f:
        d = pickle.load(f)
    X = np.asarray(d["data"])            # (4500, 3, 32, 32) uint8
    y = np.asarray(d["labels"]).astype(np.int64)
    mean_t = torch.tensor(_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(_STD).view(1, 3, 1, 1)
    x = torch.from_numpy(X).float() / 255.0
    Xn = ((x - mean_t) / std_t).numpy().astype(np.float32)  # CIFAR-normalized
    return Xn, y
