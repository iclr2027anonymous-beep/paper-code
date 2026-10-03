"""S2a: operational evaluation of the entropy gate (selective correction).

Protocol: gate g(x) = noisy-classifier entropy. Accept the LSNPC
correction iff g(x) >= tau (uncertainty -> trust the correction), else
abstain (keep the noisy prediction yhat / flag for review).

Metrics per cell (full test):
- slice: precision@coverage for entropy-gated acceptance vs
  low-confidence(-p_max)-gated acceptance vs always-accept (= slice
  correction accuracy).
- all queries: accuracy@coverage vs always-correct and never-correct
  (always keep yhat) endpoints.
- M2 decile data: P(correct | slice, entropy decile) for the paper bar
  plot.

Run from repo root:  python -m scripts.evaluate_gate_s2a
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from data_process.base import get_dataset

ROOT = Path(__file__).resolve().parent.parent
RECOURSE_DIR = ROOT / "results" / "recourse"
DATA_DIR = ROOT / "data"

CELL_RE = re.compile(
    r"rec_cache_(?P<ds>[\w]+)_(?P<nt>symmetric|instance|worse|clean|aggre)(?P<noise>\d+)"
    r"_seed(?P<seed>\d+)_nt(?P<ntest>\d+)\.json"
)
COVERAGES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.2)


def auroc(y, s):
    if len(np.unique(y)) < 2 or len(y) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def acc_at_coverage(score, accept_ok, yhat_ok, n, cov):
    """All-query accuracy at coverage: top cov*n by score use correction,
    the rest keep yhat. accept_ok = correction correct, yhat_ok = yhat correct."""
    k = max(1, int(round(cov * n)))
    order = np.argsort(-score)[:k]
    acc = (accept_ok[order].sum() + yhat_ok[~np.isin(np.arange(n), order)].sum()) / n
    return float(acc)


def prec_at_cov(score, y, cov):
    k = max(1, int(round(cov * len(y))))
    order = np.argsort(-score)[:k]
    return float(np.mean(y[order]))


def main() -> None:
    caches = sorted(RECOURSE_DIR.glob("rec_cache_*_nt*.json"))
    cells = []
    for cp in caches:
        m = CELL_RE.match(cp.name)
        rec = json.loads(cp.read_text())
        dataset, ntype, noise, seed = (m["ds"], m["nt"], int(m["noise"]) / 100.0,
                                       int(m["seed"]))
        noise_int = int(m["noise"])
        base_tag = f"{dataset}_{ntype}{noise_int}"
        cell_dir = RECOURSE_DIR / "ckpt" / f"{dataset}_{ntype}{noise_int}_seed{seed}"

        _, _, test_noisy, _, _ = get_dataset(
            dataset, str(DATA_DIR), noise_rate=noise, noise_type=ntype,
            random_state=seed)
        X_test = test_noisy.features.cpu().numpy()[: len(rec)]
        clf = pickle.load(open(cell_dir / f"classifiers_{base_tag}.pkl", "rb"))["clf"]
        probs = clf.predict_proba(X_test)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        p_max = probs.max(axis=1)

        y_noisy = np.array([r["y_noisy"] for r in rec])
        y_clean = np.array([r["y_clean"] for r in rec])
        cc = np.array([r["corrected_cls"] for r in rec])
        correct = (cc == y_clean).astype(int)
        sl = np.array([r["is_misclassified"] for r in rec]).astype(bool)
        n = len(rec)

        yhat_ok = (y_noisy == y_clean).astype(int)

        # Slice precision@coverage (entropy desc = trust high uncertainty;
        # -p_max desc = trust low confidence; always-accept = slice acc).
        sl_prec = {
            "entropy": {f"{c:.1f}": prec_at_cov(entropy[sl], correct[sl], c)
                        for c in COVERAGES},
            "lowconf": {f"{c:.1f}": prec_at_cov(-p_max[sl], correct[sl], c)
                        for c in COVERAGES},
            "always": float(np.mean(correct[sl])),
        }
        # All-query accuracy@coverage.
        all_acc = {
            "entropy": {f"{c:.1f}": acc_at_coverage(entropy, correct, yhat_ok, n, c)
                        for c in COVERAGES},
            "lowconf": {f"{c:.1f}": acc_at_coverage(-p_max, correct, yhat_ok, n, c)
                        for c in COVERAGES},
            "never_correct": float(np.mean(yhat_ok)),
            "always_correct": float(np.mean(correct)),
        }
        # M2 decile data on the slice.
        deciles = {}
        if sl.sum() >= 100:
            edges = np.quantile(entropy[sl], np.linspace(0, 1, 11))
            for d in range(10):
                lo, hi = edges[d], edges[d + 1]
                mask = (entropy[sl] >= lo) & (entropy[sl] < hi) if d < 9 else \
                    (entropy[sl] >= lo) & (entropy[sl] <= hi)
                if mask.sum() >= 5:
                    deciles[str(d)] = {
                        "p_correct": float(np.mean(correct[sl][mask])),
                        "n": int(mask.sum()),
                        "entropy_lo": float(lo), "entropy_hi": float(hi),
                    }

        cells.append({
            "cell": f"{dataset}_{ntype}{noise_int}_seed{seed}",
            "dataset": dataset, "noise_type": ntype, "noise": noise,
            "n_test": n, "slice_size": int(sl.sum()),
            "slice_acc": float(np.mean(correct[sl])),
            "overall_acc": float(np.mean(correct)),
            "clf_acc": float(np.mean(yhat_ok)),
            "entropy_slice_auroc": auroc(correct[sl], entropy[sl]),
            "slice_prec": sl_prec, "all_acc": all_acc, "deciles": deciles,
        })

    # Print headline: median slice precision@0.5 / 0.8 and all-query acc@0.8.
    print(f"s2a: {len(cells)} cells\n")
    print(f"{'cell':<34}{'sl':>5}{'slAcc':>7}{'entAUC':>7}"
          f"{'slP@.8':>7}{'slP@.5':>7}{'acc@.8':>7}{'clfAcc':>7}{'corrAcc':>7}")
    for c in cells:
        if c["slice_size"] < 100:
            continue
        print(f"{c['cell']:<34}{c['slice_size']:>5}{c['slice_acc']:>7.3f}"
              f"{c['entropy_slice_auroc']:>7.3f}"
              f"{c['slice_prec']['entropy']['0.8']:>7.3f}"
              f"{c['slice_prec']['entropy']['0.5']:>7.3f}"
              f"{c['all_acc']['entropy']['0.8']:>7.3f}"
              f"{c['clf_acc']:>7.3f}{c['overall_acc']:>7.3f}")

    # Aggregates.
    big = [c for c in cells if c["slice_size"] >= 100]
    print("\n=== median over big-slice cells ===")
    for key, fmt in [
        ("slice_acc", "{:.3f}"), ("entropy_slice_auroc", "{:.3f}"),
        ("slP@.8", "{:.3f}"), ("slP@.5", "{:.3f}"),
        ("acc@.8", "{:.3f}"), ("clf_acc", "{:.3f}"), ("corrAcc", "{:.3f}")]:
        vals = {
            "slice_acc": [c["slice_acc"] for c in big],
            "entropy_slice_auroc": [c["entropy_slice_auroc"] for c in big],
            "slP@.8": [c["slice_prec"]["entropy"]["0.8"] for c in big],
            "slP@.5": [c["slice_prec"]["entropy"]["0.5"] for c in big],
            "acc@.8": [c["all_acc"]["entropy"]["0.8"] for c in big],
            "clf_acc": [c["clf_acc"] for c in big],
            "corrAcc": [c["overall_acc"] for c in big],
        }[key]
        print(f"  {key:<14} {fmt.format(float(np.median(vals)))}")

    # The headline comparison: gated vs always-correct vs never-correct.
    g08 = np.median([c["all_acc"]["entropy"]["0.8"] for c in big])
    nc = np.median([c["clf_acc"] for c in big])
    ac = np.median([c["overall_acc"] for c in big])
    print(f"\n  all-query acc@0.8 (entropy gate): {g08:.3f}")
    print(f"  never-correct (keep yhat)      : {nc:.3f}")
    print(f"  always-correct                 : {ac:.3f}")

    out = {"schema_version": 1, "cells": cells}
    out_path = RECOURSE_DIR / "gate_s2a.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
