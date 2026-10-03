"""Build dopanim training pickles for the LSNPC pixel pipeline.

dopanim (NeurIPS 2024 D&B): 15 doppelganger animal classes, ~10.5k train
images each annotated by 3-10 humans with soft likelihoods over the 15
classes; clean oracle = the iNaturalist-verified species (= class dir).

Pipeline convention (mirrors CIFAR-N): images are uint8 NCHW; the clean
oracle and the noisy label are per-query rows of the SAME training pool, so
any split/permutation keeps them row-aligned.

Noise construction: hard noisy label = argmax of ONE annotator's likelihoods,
chosen deterministically per image (annotator ranked by a seeded RNG over the
image's annotators) — the paper's "rand-1" variant (~33% real noise).
Alternate: "worst" = pick the annotator whose argmax differs most from clean.

Outputs (data/dopanim/):
  train_batch -> {'data': (10484,3,SIZE,SIZE) uint8, 'labels': (10484,) int64
                  clean oracle, 'noisy_labels': (10484,) int64}
  test_batch  -> {'data': (4500,3,SIZE,SIZE) uint8, 'labels': (4500,) int64
                  clean only}  (annotations exist only on the train split)
  class_names.txt (15 lines, index order)

The image resolution is a dial (--size): Set A-era probes used 64; the
Set-C 224px probe (paper plan C-c) rebuilds with --size 224.
"""
from __future__ import annotations

from utils.paths import project_path

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(project_path('data/dopanim'))
SEED = 42
_RESAMPLE = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def _load_anns() -> dict[str, list[dict]]:
    ann = json.load(open(BASE / "annotation_data.json"))
    by_img: dict[str, list[dict]] = {}
    for a in ann.values():
        by_img.setdefault(str(a["observation_id"]), []).append(a)
    return by_img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32,
                    help="Resize side (default 32 = cifar pixel-mode recipe)")
    args = ap.parse_args()
    SIZE = args.size
    by_img = _load_anns()
    # class index order: fixed by the first annotation's likelihood keys
    first_obs = next(iter(by_img.values()))[0]
    classes = list(first_obs["likelihoods"].keys())
    cls_idx = {c: i for i, c in enumerate(classes)}
    assert len(classes) == 15
    (BASE / "class_names.txt").write_text("\n".join(classes) + "\n")

    # train images: class-dir organization = clean oracle (verified 1.0)
    train_files: list[tuple[str, str]] = []  # (path, clean_class)
    for c in os.listdir(BASE / "extracted" / "train"):
        cd = BASE / "extracted" / "train" / c
        if not cd.is_dir():
            continue
        for f in sorted(os.listdir(cd)):
            if f.endswith(".jpeg"):
                train_files.append((str(cd / f), c))
    print(f"{len(train_files)} train images")

    rng = np.random.RandomState(SEED)
    data = np.empty((len(train_files), 3, SIZE, SIZE), dtype=np.uint8)
    clean = np.empty(len(train_files), dtype=np.int64)
    noisy = np.empty(len(train_files), dtype=np.int64)
    flips = 0
    for i, (path, cls) in enumerate(train_files):
        stem = os.path.basename(path).replace(".jpeg", "")
        anns = by_img.get(stem)
        if not anns:
            raise ValueError(f"no annotations for {stem}")
        with Image.open(path) as im:
            im = im.convert("RGB").resize((SIZE, SIZE), _RESAMPLE)
            data[i] = np.asarray(im).transpose(2, 0, 1)
        clean[i] = cls_idx[cls]
        # deterministic random annotator (rand-1 regime)
        a = anns[rng.randint(len(anns))]
        noisy[i] = cls_idx[max(a["likelihoods"], key=a["likelihoods"].get)]
        flips += noisy[i] != clean[i]
    print(f"clean==dir class: 100% by construction; noisy flip rate = "
          f"{flips / len(train_files):.4f}")
    with open(BASE / "train_batch", "wb") as f:
        pickle.dump({"data": data, "labels": clean,
                     "noisy_labels": noisy}, f, protocol=4)
    print(f"wrote {BASE / 'train_batch'} "
          f"({data.nbytes / 1e6:.0f} MB, {len(train_files)} x 3x{SIZE}x{SIZE})")

    # ── Clean test split (held-out; annotations exist only on train) ──
    test_files: list[tuple[str, str]] = []
    for c in os.listdir(BASE / "extracted" / "test"):
        cd = BASE / "extracted" / "test" / c
        if not cd.is_dir():
            continue
        for f in sorted(os.listdir(cd)):
            if f.endswith(".jpeg"):
                test_files.append((str(cd / f), c))
    tdata = np.empty((len(test_files), 3, SIZE, SIZE), dtype=np.uint8)
    tclean = np.empty(len(test_files), dtype=np.int64)
    for i, (path, cls) in enumerate(test_files):
        with Image.open(path) as im:
            im = im.convert("RGB").resize((SIZE, SIZE), _RESAMPLE)
            tdata[i] = np.asarray(im).transpose(2, 0, 1)
        tclean[i] = cls_idx[cls]
    with open(BASE / "test_batch", "wb") as f:
        pickle.dump({"data": tdata, "labels": tclean}, f, protocol=4)
    print(f"wrote {BASE / 'test_batch'} "
          f"({len(test_files)} x 3x{SIZE}x{SIZE}, clean only)")


if __name__ == "__main__":
    main()
