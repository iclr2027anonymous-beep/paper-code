#!/usr/bin/env python3
"""Set E — end-to-end analysis: paired-bootstrap deltas across configurations.

Reads every Set E cell (results/set_e/cells/<tag>.json) and computes the
pre-registered deltas of the experiment protocol §E5/E6:

  * clean-test accuracy and macro-F1 per configuration (a/b/c@<coverage>/d/e/f@<coverage>);
  * paired bootstrap deltas over the downstream seeds {42,43,44,45,46} (2,000
    resamples): Δ(c-a), Δ(b-a), Δ(g-a), Δ(d-a), Δ(c-e), Δ(g-e), Δ(c-f),
    Δ(e-a) — at the operating coverage (default 0.1) and at the sensitivity grid
    coverage ∈ {0.1, 0.25, 0.5, 0.9} where present;
  * label-vs-model sign agreement: sign of mean Δ(c-a) acc vs the sign of
    the corrector's held-out label-level net (WR - RW) recorded in the cell;
  * NoisyAG severity monotonicity (worst >= med >= best) as a report-only;
  * dopanim cells: abstention status carried through (gate accounting).

Statistical unit (pre-registered): the per-cell paired bootstrap over seeds.
No pooling across datasets into a headline average.

Usage:  python -m scripts.analyze_set_e \
            --cells results/set_e/cells --out results/set_e/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

OPERATING_Q = 0.1
N_BOOT = 2000

def paired_boot(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                seed: int = 0) -> dict:
    """Paired bootstrap of mean(b - a) over the seed population."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")],
                "ci_excludes_zero": False}
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.array([np.mean(rng.choice(d, len(d), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool((lo > 0 and hi > 0)
                                     or (lo < 0 and hi < 0))}


def cell_configs(cell: dict) -> dict[str, dict]:
    cfg = {}
    for name, c in (cell.get("configs") or {}).items():
        if "acc" in c and "f1" in c:
            cfg[name] = c
    return cfg


def analyze(cells_dir: Path, out_dir: Path, n_boot: int = N_BOOT,
            tag_glob: str = "*") -> dict:
    rows: list[dict] = []
    summary: dict = {"notes": [
        "paired bootstrap over downstream seeds (2000 resamples); "
        "no pooling over datasets",
        "deltas on clean-test accuracy and macro-F1 per configuration",
        "operating coverage = 0.1; the coverage grid is reported where present",
        "label-level net = WR - RW on the corrector's held-out rows "
        "(val+eval, out-of-sample)", "dopanim abstention cells carry (a) only",
    ], "cells": {}}
    # tag_glob isolates a cell family: the same testbed at a different epoch
    # budget or backbone carries a different tag, and mixing them silently
    # averages two correctors into one delta.
    tags = sorted(p.stem for p in cells_dir.glob(f"{tag_glob}.json"))
    if not tags:
        print(f"no Set E cells under {cells_dir}")
        return summary

    for tag in tags:
        cell = json.loads((cells_dir / f"{tag}.json").read_text())
        cfg = cell_configs(cell)
        if not cfg:
            continue
        modality = cell.get("modality", tag.split("_")[1] if tag.startswith("e_") else "?")
        testbed = cell.get("testbed", tag.split("_")[2] if tag.startswith("e_") else "?")
        abstention = bool(cell.get("dopanim_abstention", False))
        meta_cfg = {k: {
            "mean_acc": c["acc_sum"]["mean"], "sd_acc": c["acc_sum"]["sd"],
            "mean_f1": c["f1_sum"]["mean"], "sd_f1": c["f1_sum"]["sd"],
            "seeds_acc": np.asarray(c["acc"], dtype=float),
            "seeds_f1": np.asarray(c["f1"], dtype=float)}
            for k, c in cfg.items()}
        entry = {"tag": tag, "modality": modality, "testbed": testbed,
                 "abstention": abstention, "configs": {k: {
                     "mean_acc": v["mean_acc"], "sd_acc": v["sd_acc"],
                     "mean_f1": v["mean_f1"], "sd_f1": v["sd_f1"]}
                     for k, v in meta_cfg.items()},
                 "deltas": {}, "label_net": (cell.get("stream") or {}).get(
                     "heldout")}
        gate_acct = cell.get("gate_accounting")
        if gate_acct:
            entry["gate_accounting"] = gate_acct

        # Operating coverage + sensitivity grid (c/f keys carry @<coverage>).
        qs_present = sorted({float(k.split("@")[1]) for k in cfg
                             if "@" in k and k.split("@")[0] in ("c", "f")})
        op_q = OPERATING_Q if OPERATING_Q in qs_present else \
            (qs_present[0] if qs_present else None)

        def _delta(metric: str, ca: str, cb: str):
            if ca not in meta_cfg or cb not in meta_cfg:
                return None
            return paired_boot(meta_cfg[ca][f"seeds_{metric}"],
                               meta_cfg[cb][f"seeds_{metric}"], n_boot)

        # Downstream model sensitivity (families: mlp, linear, asl) — paired
        # bootstrap deltas within each family on the same operating coverage.
        models_out = {}
        for fam, blk in (cell.get("models") or {}).items():
            fcfg = blk.get("configs") or {}
            if not fcfg:
                continue
            fmeta = {k: {"acc": np.asarray(v["acc"], dtype=float),
                         "f1": np.asarray(v["f1"], dtype=float)}
                     for k, v in fcfg.items() if "acc" in v}
            if "a" not in fmeta:
                continue
            cq = next((k for k in fmeta if k.startswith("c@") and
                       float(k.split("@")[1]) == op_q), None)
            fam_entry = {"mean_acc": {k: float(v["acc"].mean())
                                      for k, v in fmeta.items()}}
            for name, other in (("c-a", cq), ("b-a", "b" if "b" in fmeta else None)):
                if other:
                    fam_entry[f"acc_{name}"] = paired_boot(
                        fmeta["a"]["acc"], fmeta[other]["acc"], n_boot)
                    fam_entry[f"f1_{name}"] = paired_boot(
                        fmeta["a"]["f1"], fmeta[other]["f1"], n_boot)
            models_out[fam] = fam_entry
        if models_out:
            entry["models"] = models_out

        deltas = {}
        for metric in ("acc", "f1"):
            dm = {}
            if op_q is not None:
                for name, c_a, c_b in (("dc-a", "a", f"c@{op_q}"),
                                       ("db-a", "a", "b"),
                                       ("dg-a", "a", f"c@{op_q}"),
                                       ("dc-e", "e", f"c@{op_q}"),
                                       ("dg-e", "e", f"c@{op_q}"),
                                       ("dc-f", "f@{op_q}", f"c@{op_q}"),
                                       ("de-a", "a", "e")):
                    if c_a in cfg and c_b in cfg:
                        r = _delta(metric, c_a, c_b)
                        if r is not None:
                            dm[name] = r
            if "d" in cfg:
                r = _delta(metric, "a", "d")
                if r is not None:
                    dm["dd-a"] = r
            deltas[metric] = dm
        entry["deltas"] = deltas
        for metric in ("acc", "f1"):
            for dk, r in deltas.get(metric, {}).items():
                rows.append({"tag": tag, "modality": modality,
                             "testbed": testbed, "metric": metric,
                             "delta": dk, "mean": r["mean"],
                             "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1],
                             "ci_excludes_zero": r["ci_excludes_zero"]})
        # label-vs-model sign agreement (accuracy delta sign at the operating coverage).
        sign_agree = None
        if op_q is not None and f"c@{op_q}" in meta_cfg and "a" in meta_cfg:
            model_d = float(np.mean(meta_cfg[f"c@{op_q}"]["seeds_acc"]
                                    - meta_cfg["a"]["seeds_acc"]))
            held = (cell.get("stream") or {}).get("heldout")
            if held and held.get("net") is not None:
                sign_agree = bool(np.sign(model_d) == np.sign(held["net"])
                                   or (abs(model_d) < 1e-12
                                       and held["net"] == 0))
        entry["sign_agreement_model_vs_label"] = sign_agree
        summary["cells"][tag] = entry
        hdr = (f"=== {tag} (modality={modality}, testbed={testbed}"
              + (", ABSTENTION dopanim)" if abstention else ")"))
        print(hdr)
        for metric in ("acc", "f1"):
            d = deltas.get(metric, {})
            if d:
                line = "  " + metric + ": " + "  ".join(
                    f"{k}={v['mean']:+.4f}[{v['ci95'][0]:+.3f},"
                    f"{v['ci95'][1]:+.3f}]" for k, v in sorted(d.items()))
                print(line)
        if sign_agree is not None:
            print(f"  sign agreement (model c-a vs label net): {sign_agree}")

    # NoisyAG severity monotonicity of Δ(c-a) acc (report-only).
    text_cells = {e["testbed"]: e for e in summary["cells"].values()
                  if e["modality"] == "text"}
    if {"noisyag_worst", "noisyag_med", "noisyag_best"} <= set(text_cells):
        deltas = {sev: text_cells[sev]["deltas"]["acc"].get("dc-a", {}).get(
            "mean") for sev in ("noisyag_worst", "noisyag_med",
                                "noisyag_best")}
        monotone = all(deltas[s] is not None for s in deltas) and \
            deltas["noisyag_worst"] >= deltas["noisyag_med"] >= \
            deltas["noisyag_best"]
        summary["severity_monotonicity"] = {
            "dc_a_mean_acc": {k: (float(v) if v is not None else None)
                              for k, v in deltas.items()},
            "monotone_worst_ge_med_ge_best": bool(monotone)}
        print("severity monotonicity dc-a acc "
              f"(report-only): {deltas} -> "
              f"{'monotone' if monotone else 'NOT monotone'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_dir / "deltas.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'deltas.csv'} and {out_dir / 'summary.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="results/set_e/cells")
    ap.add_argument("--out", default="results/set_e/analysis")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--tag-glob", default="*",
                    help="Only cells whose tag matches this glob (e.g. "
                         "'e_image_animal10n_e5_cs*'). Default: every cell.")
    args = ap.parse_args()
    analyze(Path(args.cells), Path(args.out), args.n_boot, args.tag_glob)


if __name__ == "__main__":
    main()
