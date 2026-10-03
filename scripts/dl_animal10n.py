#!/usr/bin/env python3
"""Download ANIMAL-10N from Activeloop hub mirror to local files.

Outputs (matching the pipeline's expected layout):
  data/animal10n/train_batch  -> pickle {'data': (50000,3,64,64) uint8, 'labels': (50000,) int64}
  data/animal10n/val_batch    -> pickle {'data': (5000,3,64,64) uint8, 'labels': (5000,) int64}
(Format mirrors the CurBench loader: entry['data'], entry['labels'].)

NOTE: ANIMAL-10N's own labels are the noisy human labels (no clean ground
truth released); ~8% noise. Test split labels are search-keyword-based.
"""
import pickle
from pathlib import Path

import deeplake
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "animal10n"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch(split_url, out_name):
    print(f"loading {split_url} ...", flush=True)
    ds = deeplake.load(split_url, read_only=True)
    n = len(ds)
    print(f"  {n} samples, tensors={list(ds.tensors.keys())}", flush=True)
    imgs = ds.images.numpy()          # (n, H, W, C) uint8
    labs = ds.labels.numpy().ravel()  # (n,)
    print(f"  images {imgs.shape} {imgs.dtype}, labels {labs.shape} {labs.dtype}", flush=True)
    # channels-last -> channels-first (N,3,64,64)
    imgs = np.transpose(imgs, (0, 3, 1, 2)).copy()
    entry = {"data": imgs, "labels": labs}
    out = OUT_DIR / out_name
    with out.open("wb") as f:
        pickle.dump(entry, f, protocol=4)
    print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    fetch("hub://activeloop/animal10n-train", "train_batch")
    fetch("hub://activeloop/animal10n-test", "val_batch")
    print("DONE")
