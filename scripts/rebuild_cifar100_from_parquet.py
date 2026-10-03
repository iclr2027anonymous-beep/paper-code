#!/usr/bin/env python
"""Rebuild data/cifar100_{train,test}.pkl from the HuggingFace parquet.

PROVENANCE :
- Source: `uoft-cs/cifar100` parquet (downloaded via hf-mirror, 17 MB/s)
- ALIGNMENT VERIFIED: parquet fine_label vs CIFAR-100N human clean_label
  agreement = 1.0 (and coarse 1.0). Unlike CIFAR-10's parquet (block-sorted
  rows, 9.8% agreement), the CIFAR-100 parquet preserves original row order,
  so the CIFAR-100N human labels (`CIFAR-100_human.pt`) align directly.
- Images are PNG-encoded bytes in the parquet; decoded to (N,3,32,32) uint8.

Output format matches the cifar10 pickles:
  cifar100_train.pkl -> {'images': (50000,3,32,32) uint8, 'labels': (50000,) int64}
  cifar100_test.pkl  -> {'images': (10000,3,32,32) uint8, 'labels': (10000,) int64}
(labels = fine labels; coarse labels are available in CIFAR-100_human.pt)

Usage: python -m scripts.rebuild_cifar100_from_parquet [--parquet /tmp/cifar100]
"""
from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def _decode_png(bytes_col) -> np.ndarray:
    n = len(bytes_col)
    out = np.empty((n, 3, 32, 32), dtype=np.uint8)
    for i, b in enumerate(bytes_col):
        img = Image.open(io.BytesIO(b["bytes"])).convert("RGB")
        arr = np.asarray(img)  # (32,32,3)
        out[i] = np.transpose(arr, (2, 0, 1))
    return out


def rebuild(parquet_dir: Path, data_dir: Path) -> None:
    train = pd.read_parquet(parquet_dir / "cifar100_train.parquet")
    test = pd.read_parquet(parquet_dir / "cifar100_test.parquet")
    print(f"parquet train: {len(train)} rows, test: {len(test)} rows")

    X_train = _decode_png(train["img"])
    y_train = train["fine_label"].values.astype(np.int64)
    X_test = _decode_png(test["img"])
    y_test = test["fine_label"].values.astype(np.int64)
    print(f"train images: {X_train.shape} {X_train.dtype}, labels {y_train.shape}")
    print(f"test images : {X_test.shape} {X_test.dtype}, labels {y_test.shape}")

    # 1) Alignment with CIFAR-100N human labels (the whole point).
    hn = torch.load(data_dir / "CIFAR-100_human.pt", map_location="cpu", weights_only=False)
    cl = np.asarray(hn["clean_label"])
    fine_agree = float((y_train == cl).mean())
    print(f"parquet fine vs human clean_label agree: {fine_agree:.4f}")
    if fine_agree < 0.98:
        raise RuntimeError(f"parquet labels disagree with human.pt clean_label ({fine_agree:.4f})")

    # 2) Image sanity: per-channel means near CIFAR-100 canonical
    #    [129.30, 124.08, 112.78] (uint8 scale).  Note: use axis=(0,2,3);
    #    reshape(3,-1) on CHW does NOT select channels (C-order mixing).
    mu = X_train.mean(axis=(0, 2, 3))
    print(f"train per-channel mean: {[f'{m:.1f}' for m in mu]}")
    if abs(mu[0] - 129.3) > 8 or abs(mu[2] - 112.8) > 8:
        raise RuntimeError(f"image statistics look wrong for CIFAR-100: {mu}")

    # 3) Write pickles (same layout as cifar10 pickles).
    with open(data_dir / "cifar100_train.pkl", "wb") as f:
        pickle.dump({"images": X_train, "labels": y_train}, f, protocol=4)
    with open(data_dir / "cifar100_test.pkl", "wb") as f:
        pickle.dump({"images": X_test, "labels": y_test}, f, protocol=4)
    print("wrote cifar100_train.pkl / cifar100_test.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp", help="dir with cifar100_{train,test}.parquet")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()
    rebuild(Path(args.parquet), Path(args.data))
