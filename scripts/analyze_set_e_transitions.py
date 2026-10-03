#!/usr/bin/env python3
"""Set E — transition-group breakdown of the end-to-end streams (Table 8).

The table reports downstream clean-test accuracy only. This script decomposes the
SAME label streams into the four transitions of the repair/damage instrument

    RR  correct label, corrector left it alone   (~mis & ~edited)
    WW  wrong label, not recovered               (mis & ~succ)
    WR  wrong -> right (repair)                   (mis & succ)
    RW  right -> wrong (damage)                   (~mis & edited)

on the retrain pool (cs+tr, i.e. what the downstream heads actually train on) and
on the held-out audit slice (val+eval), and then reports what each gate operating
point the coverage does to the edit population: how much of the RW block it reverts, how much
of the WR block it discards, and the resulting stream-label accuracy.

The gate composition is NOT reimplemented here: `assemble_streams` from
scripts.retrain_e_streams is the authoritative builder (candidate set = edited
rows, text axis minimality_self, image axis noise_robustness_self + alpha=0.1
plausibility floor, no fill-in), so the c@<coverage> streams and gate@<coverage> records come from
the same code path that produced the banked cells. The admitted rows are then
recovered as the rows where the c@<coverage> stream differs from the noisy label.

Two traps this script is built to respect:
  * `edited != WR + RW` — WW absorbs the edits that stay wrong, so the edited
    count must never be reconciled against WR+RW.
  * the corpus convention is population SD (ddof=0); every mean here is
    mean ± pstdev over correfor seeds, matching how the table was built.

Read-only. Writes one JSON per invocation next to the other Set E analysis files
unless --no-write is passed.

Usage:
  python -m scripts.analyze_set_e_transitions                      # banked grid
  python -m scripts.analyze_set_e_transitions --coverages 0.1 0.25 0.5 0.7 0.8 0.9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.retrain_e_streams import assemble_streams

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "results/set_e/cells"
OUT = ROOT / "results/set_e/analysis"

# Primary settings of Table 8 (+ dopanim, whose full path is banked for seeds 42-44
# but reported in the text rather than as a table column).
SETTINGS = (
    ("e_text_noisyag_worst", (45, 46, 47, 48, 49)),
    ("e_text_noisyag_med", (45, 46, 47, 48, 49)),
    ("e_text_noisyag_best", (45, 46, 47, 48, 49)),
    ("e_image_cifar10n", (45, 46, 47, 48, 49)),
    ("e_image_cifar100n", (45, 46, 47, 48, 49)),
    ("e_image_dopanim", (42, 43, 44)),
)
FAMILIES = ("mlp", "linear", "asl")


def mean_pstd(xs) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std())


def cell_transitions(tag: str, coverages: list[float], seeds: list[int]) -> dict | None:
    """Per-cell pool/holdout composition, gate composition per coverage, accuracy per config."""
    npz = ROOT / "results/set_e" / tag / f"emissions_cs{tag.split('_cs')[-1]}.npz"
    cellf = CELLS / f"{tag}.json"
    if not (npz.exists() and cellf.exists()):
        return None
    em = dict(np.load(npz, allow_pickle=True))
    cell = json.loads(cellf.read_text())
    pool = np.where(em["is_pool"].astype(bool))[0]
    hold = np.where(em["is_holdout"].astype(bool))[0]
    modality = cell["modality"]
    clean_ceiling = bool(cell.get("clean_ceiling", True))
    streams = assemble_streams(em, pool, coverages, modality, clean_ceiling, seeds)

    y_noisy = em["y_noisy"][pool]
    mis_p = em["mis"][pool].astype(bool)
    succ_p = em["succ"][pool].astype(bool)
    edited_p = em["edited"][pool].astype(bool)

    def groups(sel: np.ndarray) -> dict:
        mis, ed, su = em["mis"][sel].astype(bool), em["edited"][sel].astype(bool), em["succ"][sel].astype(bool)
        return {"n": int(len(sel)), "RR": int((~mis & ~ed).sum()),
                "WW": int((mis & ~su).sum()), "WR": int((mis & su).sum()),
                "RW": int((~mis & ed).sum()), "edited": int(ed.sum())}

    pool_g, hold_g = groups(pool), groups(hold)
    # Gate composition: the admitted rows are exactly those whose c@<coverage> label
    # differs
    # from the noisy label, i.e. the edits the gate let through.
    gate = {}
    for cov in coverages:
        adm = streams[f"c@{cov}"] != y_noisy
        gate[str(cov)] = {
            "n_edits": int(edited_p.sum()), "n_admit": int(adm.sum()),
            "WR_adm": int((adm & mis_p & succ_p).sum()),
            "RW_adm": int((adm & (~mis_p)).sum()),
            "WW_adm": int((adm & mis_p & (~succ_p)).sum()),
            "recorded": streams[f"gate@{cov}"],
        }
    acc = {}
    for fam in FAMILIES:
        cfgs = (cell.get("models") or {}).get(fam, {}).get("configs") or {}
        acc[fam] = {k: v["acc_sum"]["mean"] for k, v in cfgs.items()}
    return {"tag": tag, "testbed": cell["testbed"], "modality": modality,
            "corr_seed": cell["corr_seed"], "pool": pool_g, "heldout": hold_g,
            "mis_frac_pool": float(mis_p.mean()), "gate": gate,
            "acc": acc, "families": list(FAMILIES)}


def aggregate(prefix: str, seeds: tuple[int, ...], coverages: list[float]) -> dict | None:
    cells = [c for c in (cell_transitions(f"{prefix}_cs{s}", coverages, list(range(42, 45)))
                         for s in seeds) if c]
    if not cells:
        return None
    agg: dict = {"seeds": [c["corr_seed"] for c in cells],
                 "n_pool": cells[0]["pool"]["n"],
                 "mis_frac": mean_pstd([c["mis_frac_pool"] for c in cells])}
    for part in ("pool", "heldout"):
        agg[part] = {g: mean_pstd([c[part][g] for c in cells])
                     for g in ("RR", "WW", "WR", "RW", "n")}
        agg[part + "_net"] = mean_pstd(
            [c[part]["WR"] - c[part]["RW"] for c in cells])
    agg["gate"] = {}
    for cov in coverages:
        e: dict[str, object] = {
            k: mean_pstd([c["gate"][str(cov)][k] for c in cells])
            for k in ("n_edits", "n_admit", "WR_adm", "RW_adm", "WW_adm")}
        wr_tot = agg["pool"]["WR"][0]
        rw_tot = agg["pool"]["RW"][0]
        rr_tot = agg["pool"]["RR"][0]
        e["net_adm"] = e["WR_adm"][0] - e["RW_adm"][0]
        e["reverts_damage"] = 1.0 - e["RW_adm"][0] / rw_tot if rw_tot else float("nan")
        e["loses_repairs"] = 1.0 - e["WR_adm"][0] / wr_tot if wr_tot else float("nan")
        # Stream-label accuracy of config (c) at this coverage: rows whose final label is clean.
        e["stream_acc"] = (rr_tot + (rw_tot - e["RW_adm"][0]) + e["WR_adm"][0]) / agg["n_pool"]
        e["coverage"] = e["n_admit"][0] / e["n_edits"][0] if e["n_edits"][0] else float("nan")
        agg["gate"][str(cov)] = e
    # a/b stream-level label accuracy (same definition, no gate)
    rr_tot, wr_tot, rw_tot = (agg["pool"][g][0] for g in ("RR", "WR", "RW"))
    agg["stream_acc_a"] = (rr_tot + rw_tot) / agg["n_pool"]
    agg["stream_acc_b"] = (rr_tot + wr_tot) / agg["n_pool"]
    agg["acc"] = {}
    for fam in FAMILIES:
        keys = ("a", "b", "d", "e") + tuple(f"c@{cov}" for cov in coverages)
        agg["acc"][fam] = {k: mean_pstd([c["acc"][fam][k] for c in cells
                                         if k in c["acc"][fam]]) for k in keys}
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.7, 0.8, 0.9])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    coverages = list(args.coverages)

    report = {"coverages": coverages, "settings": {}}
    for prefix, seeds in SETTINGS:
        agg = aggregate(prefix, seeds, coverages)
        if agg is None:
            print(f"[warn] no cells for {prefix}")
            continue
        report["settings"][prefix] = agg

    for prefix, agg in report["settings"].items():
        n = agg["n_pool"]
        print(f"\n=== {prefix} (corrector seeds {agg['seeds']}, pool n={n}) ===")
        print("  group  |   pool rows (% of pool)  |  held-out rows (% of holdout)")
        for g in ("RR", "WW", "WR", "RW"):
            p, h = agg["pool"][g], agg["heldout"][g]
            print(f"  {g}     | {p[0]:9.1f} ({p[0]/n:6.1%})      | "
                  f"{h[0]:7.1f} ({h[0]/agg['heldout']['n'][0]:6.1%})")
        print(f"  net WR-RW: pool {agg['pool_net'][0]:+.1f} ± {agg['pool_net'][1]:.1f}   "
              f"held-out {agg['heldout_net'][0]:+.1f} ± {agg['heldout_net'][1]:.1f}")
        print(f"  stream-label accuracy: a original {agg['stream_acc_a']:.2%}  "
              f"b unguarded {agg['stream_acc_b']:.2%}")
        print("  coverage admitted   WR_adm   RW_adm   WW_adm  net_adm  reverts RW  loses WR  stream acc")
        for cov in coverages:
            e = agg["gate"][str(cov)]
            print(f"  {cov:<8} {e['n_admit'][0]:8.1f} {e['WR_adm'][0]:8.1f} "
                  f"{e['RW_adm'][0]:8.1f} {e['WW_adm'][0]:8.1f} "
                  f"{e['net_adm']:+8.1f} {e['reverts_damage']:9.1%} "
                  f"{e['loses_repairs']:9.1%} {e['stream_acc']:10.2%}")
        for fam in FAMILIES:
            a = agg["acc"][fam]
            print(f"  acc[{fam:6s}] " + "  ".join(
                f"{k}={v[0]*100:.1f}±{v[1]*100:.1f}" for k, v in a.items() if v))

    if not args.no_write:
        OUT.mkdir(parents=True, exist_ok=True)
        path = args.out or OUT / f"transitions_cov{'-'.join(str(cov) for cov in coverages)}.json"
        path.write_text(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
