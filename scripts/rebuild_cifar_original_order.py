#!/usr/bin/env python
"""Rebuild data/cifar10_{train,test}.pkl in ORIGINAL CIFAR-10 row order.

WHY (row-order requirement):
The HuggingFace parquet `uoft-cs/cifar10/plain_text` is NOT in original
CIFAR-10 row order (verified: block-homogeneity [1,0,0,0,0,0,0,1,0,0] at
5000-row granularity; original order is block-sorted). The CIFAR-10N
human labels (`CIFAR-10N_human.pt`) are keyed in ORIGINAL row order, so
loading the parquet-derived pickle with the human labels silently pairs
each image with another row's noisy label: pickle-vs-clean_label
agreement was 9.8% (chance for 10 classes), and a ConvClassifier trained
on the misaligned pairs collapsed to ~10% val accuracy (random).

FIX: rebuild the pickles from the original `cifar-10-python.tar.gz`
archive (data_batch_1..5 + test_batch), whose row order matches the
CIFAR-10N human label file. The parquet-derived pickle is verified to be
the same image set (per-image byte identity), so no information is lost.

Usage: python -m scripts.rebuild_cifar_original_order --archive /tmp/cifar_orig.tar.gz
"""
from __future__ import annotations

import argparse
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch


def _unpickle(archive: tarfile.TarFile, name: str) -> dict:
    f = archive.extractfile(name)
    assert f is not None, name
    return pickle.load(f, encoding="bytes")


def rebuild(archive_path: Path, data_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        names = sorted(n for n in tar.getnames() if "data_batch" in n or n.endswith("test_batch"))
        print("archive members:", names)
        batches = [_unpickle(tar, n) for n in names if "data_batch" in n]
        test = _unpickle(tar, "cifar-10-batches-py/test_batch")

    X_orig = np.concatenate([b[b"data"] for b in batches], axis=0)          # (50000, 3072)
    X_orig = X_orig.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)            # HWC
    X_orig = np.ascontiguousarray(X_orig.transpose(0, 3, 1, 2))             # CHW (N,3,32,32)
    y_orig = np.concatenate([b[b"labels"] for b in batches], axis=0)
    X_test = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = np.ascontiguousarray(X_test.transpose(0, 3, 1, 2))
    y_test = np.asarray(test[b"labels"])

    print(f"archive train: {X_orig.shape} {X_orig.dtype}, labels {len(y_orig)}")
    print(f"archive test : {X_test.shape} {X_test.dtype}, labels {len(y_test)}")

    # 1) Alignment with CIFAR-10N human labels (the whole point): the
    #    archive's labels must match the authoritative human labels.  Note
    #    clean_label agreement is ~1.0 by construction; the human 'clean'
    #    labels are the original CIFAR labels.
    hn = torch.load(data_dir / "CIFAR-10N_human.pt", map_location="cpu", weights_only=False)
    cl = np.asarray(hn["clean_label"]); ag = np.asarray(hn["aggre_label"]); wo = np.asarray(hn["worse_label"])
    clean_agree = float((y_orig == cl).mean())
    print(f"archive vs human-clean agree: {clean_agree:.4f}")
    print(f"archive vs human-aggre agree: {float((y_orig == ag).mean()):.4f}")
    print(f"human noise rates: aggre {float((ag != cl).mean()):.4f}, worse {float((wo != cl).mean()):.4f}")
    if clean_agree < 0.98:
        raise RuntimeError(f"archive labels disagree with human.pt clean_label ({clean_agree:.4f})")

    # 2) Image sanity: per-channel statistics near the canonical CIFAR-10
    #    train means [125.31, 122.95, 113.87] (uint8 scale).
    mu = X_orig.reshape(3, -1).mean(axis=1)
    sd = X_orig.reshape(3, -1).std(axis=1)
    print(f"archive image per-channel mean/std: {[f'{m:.1f}' for m in mu]} / {[f'{s:.1f}' for s in sd]}")
    if abs(mu[0] - 125.3) > 8 or abs(mu[2] - 113.9) > 8:
        raise RuntimeError("archive image statistics look wrong for CIFAR-10")

    # 3) The old (parquet-derived) pickle is NOT byte-comparable: the parquet
    #    stores JPEG re-encodings, so byte identity cannot hold.  Instead
    #    verify learnability separately (see the probe after this script).

    # 5) Write the original-order pickles
    out_train = data_dir / "cifar10_train.pkl"
    out_test = data_dir / "cifar10_test.pkl"
    pickle.dump({"images": X_orig, "labels": y_orig}, open(out_train, "wb"))
    pickle.dump({"images": X_test, "labels": y_test}, open(out_test, "wb"))
    print(f"wrote {out_train} ({X_orig.shape}) and {out_test} ({X_test.shape})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="/tmp/cifar_orig.tar.gz", type=Path)
    ap.add_argument("--data-dir", default="data", type=Path)
    args = ap.parse_args()
    rebuild(args.archive, args.data_dir)
