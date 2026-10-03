"""EuroSAT-RGB loader for the LSNPC image pipelines (image synthetic control).

EuroSAT (Helber et al., 2019) is a 10-class Sentinel-2 land-use benchmark of
64x64 RGB tiles. The HF parquet (``blanchon/EuroSAT_RGB``) stores each image as
PNG bytes under ``image`` with an integer ``label``.

Contract (mirrors ``data_process/cifar10n.py``):
  ``load_eurosat_pair(...)`` returns one row-aligned pool of
  ``(X, y_noisy, y_clean, X_test, y_test, mean, std)``. ``X`` is
  (N, 3, 64, 64) float32 CIFAR-normalized RGB (64 % 8 == 0, so no padding is
  needed). The training pool is the loader's
  train + val split; the held-out test split is returned for reference but is
  not used by the data-level protocol, which carves its own eval/val cleanly
  from the pool. Synthetic noise is injected ONCE over the full label array
  (row-consistent across every split carve) before the caller subsets by index.

Instance-dependent noise needs a feature matrix for the class centroids.
``experiments.protocol.inject_noise_idn`` forms an ``(n, n_classes, D)``
distance tensor, which for raw 64x64x3 pixels (D = 12288, n = 21600) would be
~10.6 GB. The loader therefore passes a 2x2 average-pooled 16x16x3 descriptor
(D = 768) to that call only; the
images the model trains on are untouched. Symmetric noise uses no features and
is unaffected.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.protocol import inject_noise, inject_noise_idn

_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)

_POOL_SPLITS = ("train.parquet", "val.parquet")
_TEST_SPLIT = "test.parquet"
_POOL_FOR_CENTROIDS = 4  # 64x64 -> 16x16, see module docstring


def _decode_rgb_bytes(b: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(b)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)  # (64, 64, 3)


def _load_split(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(root / name)
    rgb = np.stack([_decode_rgb_bytes(row.image["bytes"]) for row in df.itertuples()])
    X = np.transpose(rgb, (0, 3, 1, 2)).astype(np.uint8)  # (N, 3, 64, 64)
    return X, np.asarray(df["label"].to_numpy(dtype=np.int64))


def _normalize(X: np.ndarray, mean_t: np.ndarray, std_t: np.ndarray) -> np.ndarray:
    return ((X.astype(np.float32) / 255.0 - mean_t) / std_t).astype(np.float32)


def _centroid_features(X: np.ndarray) -> np.ndarray:
    """Pooled descriptor for the instance-noise centroids (see docstring)."""
    n, c, h, w = X.shape
    p = _POOL_FOR_CENTROIDS
    pooled = X.reshape(n, c, h // p, p, w // p, p).mean(axis=(3, 5))
    return pooled.reshape(n, -1)


def load_eurosat_pair(
    data_dir: str,
    noise_rate: float = 0.0,
    noise_type: str = "symmetric",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple, tuple]:
    """Load the EuroSAT pool with row-aligned clean/noisy labels."""
    root = Path(data_dir) / "eurosat"
    missing = [str(root / s) for s in _POOL_SPLITS if not (root / s).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing EuroSAT parquet: {missing}")

    Xs, ys = zip(*(_load_split(root, s) for s in _POOL_SPLITS))
    X = np.concatenate(Xs)
    y_clean = np.concatenate(ys)
    n = len(y_clean)
    n_classes = int(y_clean.max()) + 1
    assert X.shape == (n, 3, 64, 64), X.shape

    mean_t = np.asarray(_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std_t = np.asarray(_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    X = _normalize(X, mean_t, std_t)

    y_noisy = y_clean.copy()
    if noise_rate > 0:
        if noise_type == "idn":
            y_noisy = inject_noise_idn(
                y_clean, float(noise_rate), int(random_state), int(n_classes),
                X=_centroid_features(X))
        else:
            y_noisy = inject_noise(
                y_clean, float(noise_rate), int(random_state), int(n_classes))
    actual = float((y_noisy != y_clean).mean())
    print(f"EuroSAT: {n} rows, {n_classes} classes, noise_type={noise_type} "
          f"target={noise_rate:.2f}, actual flip rate = {actual:.3f}")

    X_test = np.empty((0, 3, 64, 64), dtype=np.float32)
    y_test = np.empty((0,), dtype=np.int64)
    if (root / _TEST_SPLIT).is_file():
        Xt, yt = _load_split(root, _TEST_SPLIT)
        X_test = _normalize(Xt, mean_t, std_t)
        y_test = yt
    return X, y_noisy, y_clean, X_test, y_test, _MEAN, _STD
