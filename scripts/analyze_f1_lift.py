#!/usr/bin/env python3
"""Analyze F1/accuracy lift of the LSNPC corrected predictor over the base
classifier, from per-query recourse caches (rec_cache_*.json).

Base prediction   = y_noisy       (classifier argmax)
Corrected         = corrected_cls (LSNPC corrected argmax)
Truth             = y_clean

Usage:
    python scripts/analyze_f1_lift.py --dir results/recourse
    python scripts/analyze_f1_lift.py --dir A --dir B        (compare two runs)

For each cache file under each --dir, prints base vs corrected acc/F1 and the
F1 lift, plus the best corrected-predictor validation error (val_err_h) from
the run's lsnpc_history.json when present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", action="append", required=True,
                   help="Output dir to scan for rec_cache_*.json (repeatable).")
    return p.parse_args()


def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def metrics(preds, truth) -> tuple[float, float]:
    preds = np.asarray(preds)
    truth = np.asarray(truth)
    acc = float(np.mean(preds == truth))
    f1s = []
    for c in set(truth.tolist()) | set(preds.tolist()):
        tp = int(np.sum((preds == c) & (truth == c)))
        fp = int(np.sum((preds == c) & (truth != c)))
        fn = int(np.sum((preds != c) & (truth == c)))
        f1s.append(_f1(tp, fp, fn))
    return acc, float(np.mean(f1s))


def best_val_err(out_dir: Path) -> float | None:
    """Best corrected-predictor val error across run histories.

    Returns None only when there are genuinely no history files at all.
    A history file with no usable val_err_h values is an exposed error
    (missing key or all-NaN), not a silent skip: the file's path is
    reported and the caller flags the anomaly.
    """
    files = sorted(out_dir.glob("ckpt/*/lsnpc_history.json"))
    if not files:
        return None
    empty = []
    best = None
    for h in files:
        d = json.loads(h.read_text())
        if "val_err_h" not in d:
            empty.append((h, "key 'val_err_h' missing"))
            continue
        v = [x for x in d["val_err_h"] if x is not None and x == x]
        if not v:
            empty.append((h, "all val_err_h values NaN/None"))
            continue
        best = min(v) if best is None else min(best, min(v))
    for h, reason in empty:
        print(f"  WARNING: {h} has no usable val_err_h ({reason})")
    return best


def analyze_dir(out_dir: Path) -> None:
    caches = sorted(out_dir.glob("rec_cache_*.json"))
    print(f"=== {out_dir} ({len(caches)} caches) ===")
    if not caches:
        print("  (no rec_cache files)")
        return
    rows = []
    for f in caches:
        d = json.loads(f.read_text())
        if not isinstance(d, list) or not d:
            continue
        y_noisy = [r["y_noisy"] for r in d]
        corr = [r["corrected_cls"] for r in d]
        truth = [r["y_clean"] for r in d]
        ba, bf = metrics(y_noisy, truth)
        ca, cf = metrics(corr, truth)
        rows.append((f.name, len(d), ba, ca, bf, cf))
    for name, n, ba, ca, bf, cf in rows:
        print(f"  {name:<48} n={n:>6}  base acc={ba:.4f} F1={bf:.4f} | "
              f"corr acc={ca:.4f} F1={cf:.4f} | F1 lift={cf - bf:+.4f}")
    if rows:
        lifts = [cf - bf for *_, bf, cf in rows]
        print(f"  mean F1 lift = {np.mean(lifts):+.4f}  (n_cells={len(lifts)})")
    bve = best_val_err(out_dir)
    if bve is not None:
        print(f"  best val_err_h (corrected predictor) = {bve:.4f}")


def main() -> None:
    args = parse_args()
    for d in args.dir:
        analyze_dir(Path(d))


if __name__ == "__main__":
    main()
