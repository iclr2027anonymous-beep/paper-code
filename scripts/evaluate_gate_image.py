"""Gate evaluation for the S4 image caches (multiclass-safe v3 mirror).

Binary-only concepts from evaluate_gate_v3 (logit, p1, abs_logit) do not
generalise to 10 classes; this mirror uses p_max / entropy / margin
(multiclass-valid) plus the cache geometry features, and a logistic combo
over the multiclass-safe directional features.

Run:  python -m scripts.evaluate_gate_image
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_process.base import get_dataset

ROOT = Path(__file__).resolve().parent.parent
RECOURSE_DIR = ROOT / "results" / "recourse"
DATA_DIR = ROOT / "data"

GEOMETRY_FEATURES = [
    "lat_shift", "dec_amp", "on_manifold_err", "na_cost",
    "random_flip_rate", "mech_flip",
]
DIRECTIONAL_FEATURES = ["p_max", "entropy", "margin"]
BASELINES = ["p_max", "entropy", "margin"]

CELL_RE = re.compile(
    r"rec_cache_(?P<ds>[\w]+)_(?P<nt>symmetric|instance|worse|clean|aggre)(?P<noise>\d+)"
    r"_seed(?P<seed>\d+)_nt(?P<ntest>\d+)\.json"
)


def load_classifier(cell_dir: Path, base_tag: str):
    return pickle.load(open(cell_dir / f"classifiers_{base_tag}.pkl", "rb"))["clf"]


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(y) < 2:
        return float("nan")
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).all():
        s = np.nan_to_num(s, nan=-1.0, posinf=1.0, neginf=0.0)
    return float(roc_auc_score(y, s))


def oof_scores(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=seed))
    return cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]


def prec_at_cov(s: np.ndarray, y: np.ndarray, cov: float) -> float:
    k = max(1, int(round(cov * len(s))))
    order = np.argsort(-s)[:k]
    return float(np.mean(y[order]))


def analyze_cell(cache_path: Path) -> dict:
    rec = json.loads(cache_path.read_text())
    m = CELL_RE.match(cache_path.name)
    assert m, f"unparseable cache: {cache_path.name}"
    dataset, ntype, noise_int = m["ds"], m["nt"], int(m["noise"])
    noise = noise_int / 100.0
    seed = int(m["seed"])
    base_tag = f"{dataset}_{ntype}{noise_int}"
    cell_dir = RECOURSE_DIR / "ckpt" / f"{dataset}_{ntype}{noise_int}_seed{seed}"

    _, _, test_noisy, _, _ = get_dataset(
        dataset, str(DATA_DIR), noise_rate=noise, noise_type=ntype,
        random_state=seed)
    n_test = len(rec)
    X_test = test_noisy.images.cpu().numpy()[:n_test]   # (N,3,32,32) NCHW
    X_test = np.transpose(X_test, (0, 2, 3, 1))          # NHWC for the clf

    clf = load_classifier(cell_dir, base_tag)
    y_noisy = np.array([r["y_noisy"] for r in rec])
    agree = float(np.mean(clf.predict(X_test) == y_noisy))

    probs = clf.predict_proba(X_test)
    p_max = probs.max(axis=1)
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    entropy = np.array([scipy_entropy(p) for p in probs])

    y_clean = np.array([r["y_clean"] for r in rec])
    corrected_cls = np.array([r["corrected_cls"] for r in rec])
    correct = (corrected_cls == y_clean).astype(int)
    sl = np.array([r["is_misclassified"] for r in rec]).astype(bool)

    feat = {"p_max": p_max, "entropy": entropy, "margin": margin}
    for name in GEOMETRY_FEATURES:
        feat[name] = np.nan_to_num(
            np.array([r[name] for r in rec], dtype=float), nan=0.0)

    X_dir = np.column_stack([feat[n] for n in DIRECTIONAL_FEATURES])
    X_geo = np.column_stack([feat[n] for n in GEOMETRY_FEATURES])
    oof_dir = oof_scores(X_dir, correct)
    oof_geo = oof_scores(X_geo, correct)

    cell = {
        "dataset": dataset, "noise_type": ntype, "noise": noise,
        "seed": seed, "n_test": n_test, "agree": agree,
        "acc": float(np.mean(correct)),
        "slice_size": int(sl.sum()),
        "slice_acc": float(np.mean(correct[sl])),
        "baselines_slice": {k: auroc(correct[sl], feat[k][sl])
                            for k in BASELINES},
        "geom_slice": {k: auroc(correct[sl], feat[k][sl])
                       for k in GEOMETRY_FEATURES},
        "slice_auroc_dir": auroc(correct[sl], oof_dir[sl]),
        "slice_auroc_geo": auroc(correct[sl], oof_geo[sl]),
        "prec_cov_slice": {
            "pmax": {f"{c:.1f}": prec_at_cov(feat["p_max"][sl], correct[sl], c)
                     for c in (1.0, 0.8, 0.5)},
            "entropy": {f"{c:.1f}": prec_at_cov(feat["entropy"][sl], correct[sl], c)
                        for c in (1.0, 0.8, 0.5)},
        },
    }
    return cell


def main() -> None:
    caches = sorted(RECOURSE_DIR.glob("rec_cache_cifar10*.json"))
    assert caches, "no cifar caches found"
    print(f"gate image: {len(caches)} cells\n")

    cells = [analyze_cell(cp) for cp in caches]
    for c in cells:
        b = c["baselines_slice"]
        print(
            f"{c['dataset']:<10} {c['noise_type'][:3]} {c['noise']:.1f} "
            f"s={c['seed']} sl={c['slice_size']:>5} slAcc={c['slice_acc']:.3f} "
            f"| pMax={b['p_max']:.3f} ent={b['entropy']:.3f} "
            f"margin={b['margin']:.3f} dir={c['slice_auroc_dir']:.3f} "
            f"geo={c['slice_auroc_geo']:.3f}")

    def med(vals):
        v = [x for x in vals if isinstance(x, float) and x == x]
        return float(np.median(v)) if v else float("nan")

    print(f"\n{'score':<14}{'median_slice':>13}")
    for name in BASELINES:
        print(f"{name:<14}{med([c['baselines_slice'][name] for c in cells]):>13.3f}")
    for name in GEOMETRY_FEATURES:
        print(f"{name:<14}{med([c['geom_slice'][name] for c in cells]):>13.3f}")
    print(f"{'combo_dir':<14}{med([c['slice_auroc_dir'] for c in cells]):>13.3f}")
    print(f"{'combo_geo':<14}{med([c['slice_auroc_geo'] for c in cells]):>13.3f}")
    print(f"p_max anti-predictive (<0.5): "
          f"{sum(1 for c in cells if c['baselines_slice']['p_max'] < 0.5)}/{len(cells)}")

    out = RECOURSE_DIR / "gate_image.json"
    out.write_text(json.dumps({"cells": cells}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
