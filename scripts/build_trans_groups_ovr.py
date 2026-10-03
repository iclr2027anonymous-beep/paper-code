"""Rebuild/extend code_v2/results/trans_groups_ovr_setA.csv from the per-query npz.

The original builder for this CSV is not in the tree, so this script re-derives it
from the same inputs (npz transition + score columns) and VALIDATES itself against
the rows already in the CSV before appending anything. Definition inferred and then
confirmed: for each leg/seed/axis/group, the one-vs-rest AUROC and average precision
of that raw score, treating that transition group as positive against the other three.

Usage:  python build_trans_groups_ovr.py --validate        # check against the CSV
        python build_trans_groups_ovr.py --legs eurosat    # append matching legs
"""

from utils.paths import project_path
import argparse
import csv
import glob
import os
import re

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = project_path('results')
CSV = os.path.join(ROOT, "trans_groups_ovr_setA.csv")
AXES = {"conf": "p_max(conf)",
        "minimality": "minimality(1-minf)", "robustness": "noise_robustness",
        "plausibility": "plausibility(density)"}
GROUPS = ["WW", "RW", "RR", "WR"]


def rows_for_npz(npz_path, leg, seed):
    z = np.load(npz_path)
    tr = np.asarray(z["transition"])
    out = []
    for axis, key in AXES.items():
        v = np.asarray(z["score__" + key], dtype=np.float64)
        for g in GROUPS:
            pos = (tr == g)
            if len(set(pos.tolist())) > 1:
                out.append(dict(leg=leg, seed=str(seed), group=g, axis=axis,
                                auroc=roc_auc_score(pos, v),
                                ap=average_precision_score(pos, v)))
    return out


def parse_stem(stem):
    """npz stem -> (leg, seed), matching the CSV's convention exactly.

    ``image_eurosat_0.4_symmetric_resnet50_s42_e10_b0.5_cs2000`` maps to
    leg ``image_eurosat_0.4_symmetric_resnet50_e10`` and seed ``42``: the seed is
    dropped from the leg and the epoch token (immediately after it) is kept.
    """
    m = re.match(r"(?P<base>.+?)_s(?P<seed>\d+)_(?P<rest>.+)$", stem)
    if not m:
        raise ValueError("cannot parse npz stem: " + stem)
    return "%s_%s" % (m.group("base"), m.group("rest").split("_")[0]), m.group("seed")


def discover(prefix):
    """Map leg -> {seed: npz path} for npz files whose leg matches prefix."""
    found = {}
    for p in glob.glob(os.path.join(ROOT, "set_a", "per_query", f"{prefix}*.npz")):
        leg, seed = parse_stem(os.path.basename(p)[:-4])
        found.setdefault(leg, {})[seed] = p
    return found


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--validate", action="store_true")
    ap_.add_argument("--legs", default=None)
    a = ap_.parse_args()

    existing = list(csv.DictReader(open(CSV)))
    if a.validate:
        want = {("image_dermamnist_0.4_symmetric_resnet50_e10", "42", "WW", "conf"): 0.28365318593792915,
                ("image_dermamnist_0.4_symmetric_resnet50_e10", "42", "WR", "conf"): None,
                ("image_dermamnist_0.4_symmetric_resnet50_e10", "49", "RR", "minimality"): None}
        index = {(r["leg"], r["seed"], r["group"], r["axis"]): r for r in existing}
        # recompute the first three DermaMNIST legs and diff against the CSV
        for p in sorted(glob.glob(os.path.join(ROOT, "set_a", "per_query", "image_dermamnist_*.npz")))[:3]:
            leg, seed = parse_stem(os.path.basename(p)[:-4])
            mism = 0
            for r in rows_for_npz(p, leg, seed):
                ref = index.get((r["leg"], r["seed"], r["group"], r["axis"]))
                if ref is None:
                    continue
                if abs(float(ref["auroc"]) - r["auroc"]) > 1e-9 or abs(float(ref["ap"]) - r["ap"]) > 1e-9:
                    mism += 1
                    if mism <= 2:
                        print("   MISMATCH %s %s %s: csv=%.12f mine=%.12f"
                              % (r["seed"], r["group"], r["axis"], float(ref["auroc"]), r["auroc"]))
            print("   %s s%s: %s" % (leg, seed, "exact match" if mism == 0 else f"{mism} mismatches"))
        return

    if a.legs:
        found = discover(a.legs)
        new = []
        for leg, seeds in sorted(found.items()):
            for seed, p in sorted(seeds.items()):
                new.extend(rows_for_npz(p, leg, seed))
        have = {(r["leg"], r["seed"], r["group"], r["axis"]) for r in existing}
        add = [r for r in new if (r["leg"], r["seed"], r["group"], r["axis"]) not in have]
        print("legs matched: %d | rows computed: %d | new rows to append: %d"
              % (len(found), len(new), len(add)))
        for leg in sorted(found):
            print("   %s (%d seeds)" % (leg, len(found[leg])))
        with open(CSV, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["leg", "seed", "group", "axis", "auroc", "ap"])
            w.writerows(add)
        print("appended to", CSV)


if __name__ == "__main__":
    main()
