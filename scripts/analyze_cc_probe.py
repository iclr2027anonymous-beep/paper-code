#!/usr/bin/env python3
"""C-c: the dopanim 224 px resolution probe, recomputed from its own artifacts.

The C appendix reports this probe in prose, and the numbers in that prose had no
generator. This script is that generator: it reads the five probe cells of one
config (by default the beta=0.1 stream at 10 epochs, seeds 42-46, selected from
`results/set_c/json/*dopanim*sz224*.json`) and their per-query sidecars
(`results/set_c/per_query/*dopanim*sz224*.npz`) and prints the quantity set the
paragraph rests on:

  * the edit stream each seed emits out of the 1,000-row slice, and its net
    (repairs minus damage, label level);
  * the label-level accuracy change against the noisy stream;
  * the separation of repairs from damage by proximity, robustness, confidence
    and entropy (AUROC, from the cell's `cf_auroc`);
  * the conjunction gate of the threshold study -- proximity and robustness on
    the *worse* of the two within-candidate ranks, exactly as
    `analyze_set_b_gates.gate_streams` builds it -- and what it keeps at 10% and
    50% coverage: the share of the damaging edits reverted and of the repairs
    kept, against the random-rejection control at equal coverage (median and
    95th percentile over resamples, as in Set B).

Usage: python -m scripts.analyze_cc_probe [--cells DIR] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

SLICE = 1000
RESAMPLES = 200
COVERAGES = (0.1, 0.5)


def within_rank(v: np.ndarray) -> np.ndarray:
    """Ascending rank normalised to [0, 1], stable, ties averaged (as in Set B)."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    if len(v) > 1:
        uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        for i, c in enumerate(counts):
            if c > 1:
                m = inv == i
                ranks[m] = ranks[m].mean()
    return ranks / max(1, len(v) - 1)


def probe(cell: dict, npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    tr = np.asarray(z["transition"]).astype(str)
    n = len(tr)
    rw, wr, rr = tr == "RW", tr == "WR", tr == "RR"
    cand = rw | wr
    prox = np.asarray(z["score__minimality(1-minf)"], dtype=float)
    rob = np.asarray(z["score__noise_robustness"], dtype=float)
    g = np.full(n, -np.inf)
    g[cand] = np.minimum(within_rank(prox[cand]), within_rank(rob[cand]))
    rng = np.random.default_rng(0)
    out = {
        "seed": int(cell["seed"]),
        "n_eval": int(cell["n_eval"]),
        "edits": int(cand.sum()),
        "net": int(wr.sum() - rw.sum()),
        "d_acc_points": 100 * (cell["corr_acc"] - cell["noisy_acc"]),
        "auroc": {k: round(float(v), 4) for k, v in cell["cf_auroc"].items()},
        "coverage": {},
    }
    for q in COVERAGES:
        k = int(min(round(q * int(cand.sum())), int(cand.sum())))
        keep = np.zeros(n, dtype=bool)
        keep[np.argsort(-g, kind="stable")[:k]] = True
        gated_ok = (wr & keep).sum() + (rw & ~keep).sum() + rr.sum()
        gated_acc = gated_ok / n
        # random rejection at equal coverage, among the candidates
        idx = np.argsort(rng.random((RESAMPLES, int(cand.sum()))), axis=1)[:, :k]
        cpos = np.flatnonzero(cand)
        wr_c, rw_c = wr[cpos].astype(int), rw[cpos].astype(int)
        r_ok = (wr_c[idx].sum(1) + rw.sum() - rw_c[idx].sum(1) + rr.sum()) / n
        out["coverage"][f"{q:g}"] = {
            "kept": int(k),
            "reverts_damage_share": float((rw & ~keep).sum() / max(1, rw.sum())),
            "keeps_repairs_share": float((wr & keep).sum() / max(1, wr.sum())),
            "gated_acc": float(gated_acc),
            "random_median": float(np.median(r_ok)),
            "random_p95": float(np.percentile(r_ok, 95)),
            "beats_random": bool(gated_acc >= np.percentile(r_ok, 95)
                                 and gated_acc >= np.median(r_ok)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="results/set_c/json")
    ap.add_argument("--npz", default="results/set_c/per_query")
    ap.add_argument("--out", default="results/set_c/analysis/cc_probe.json")
    # The probe lattice is beta {0.1, 0.5} x epochs {10, 15}; the paragraph rests on
    # the beta=0.1 stream at the primary epoch count, one cell per seed. Selecting the
    # config explicitly is what makes the five-seed pin meaningful: a bare
    # *dopanim*sz224* glob also matches the other three configs and can never be five.
    ap.add_argument("--beta", default="0.1")
    ap.add_argument("--epochs", default="10")
    args = ap.parse_args()

    cells_dir, npz_dir = Path(args.cells), Path(args.npz)
    rows = []
    pattern = f"*dopanim*_e{args.epochs}_b{args.beta}_*sz224*.json"
    cells = sorted(cells_dir.glob(pattern))
    if not cells:
        raise SystemExit(f"no probe cells match {pattern} in {cells_dir}")
    for p in cells:
        side = npz_dir / (p.stem + ".npz")
        if not side.exists():
            raise SystemExit(f"missing sidecar for {p.name}")
        rows.append(probe(json.loads(p.read_text()), side))
    if len(rows) != 5:
        seeds = sorted(r["seed"] for r in rows)
        raise SystemExit(f"expected 5 probe seeds for e{args.epochs} b{args.beta}, "
                         f"found {len(rows)} {seeds}")

    # A cell can carry a partial `cf_auroc`: a ranking whose score does not vary on the
    # slice has no AUROC to report, so the columns are the union of the keys present and
    # an absent value prints as n/a, with the gap listed under the table. Reporting the
    # gap is the point -- a silently missing key would let prose quote a five-seed range
    # that two seeds actually support.
    auroc_keys = sorted({k for r in rows for k in r["auroc"]})
    col_w = max(6, max((len(k) for k in auroc_keys), default=6))
    print(f"{'seed':>4} {'edits':>6} {'net':>5} {'dAcc':>6} "
          + " ".join(f"{k:>{col_w}}" for k in auroc_keys)
          + f" {'rev%':>5} {'keep%':>6} {'beats':>5}")
    for r in rows:
        c = r["coverage"]["0.5"]
        cols = " ".join((f"{r['auroc'][k]:>{col_w}.3f}" if k in r["auroc"]
                         else f"{'n/a':>{col_w}}") for k in auroc_keys)
        print(f"{r['seed']:>4} {r['edits']:>6} {r['net']:>+5} {r['d_acc_points']:>+6.1f} "
              f"{cols} "
              f"{100*c['reverts_damage_share']:>5.0f} {100*c['keeps_repairs_share']:>6.0f} "
              f"{str(c['beats_random']):>5}")
    absent = {k: [r["seed"] for r in rows if k not in r["auroc"]] for k in auroc_keys}
    print(f"auroc keys present: {auroc_keys}")
    print(f"auroc keys absent (seeds): "
          f"{ {k: v for k, v in absent.items() if v} or 'none'}")
    summary = {
        "n_seeds": len(rows),
        "edits_range": [min(r["edits"] for r in rows), max(r["edits"] for r in rows)],
        "edited_share_of_slice": [min(r["edits"] for r in rows) / SLICE,
                                  max(r["edits"] for r in rows) / SLICE],
        "net_mean": st.mean(r["net"] for r in rows),
        "net_negative_seeds": sum(1 for r in rows if r["net"] < 0),
        "d_acc_mean_points": st.mean(r["d_acc_points"] for r in rows),
        "d_acc_negative_seeds": sum(1 for r in rows if r["d_acc_points"] < 0),
        "auroc_range": {k: [min(r["auroc"][k] for r in rows if k in r["auroc"]),
                            max(r["auroc"][k] for r in rows if k in r["auroc"])]
                        for k in sorted({k for r in rows for k in r["auroc"]})},
        "auroc_seeds_present": {k: sum(1 for r in rows if k in r["auroc"])
                                for k in sorted({k for r in rows for k in r["auroc"]})},
        "coverage": {},
    }
    for q in COVERAGES:
        k = f"{q:g}"
        summary["coverage"][k] = {
            "reverts_damage_share_mean": st.mean(r["coverage"][k]["reverts_damage_share"] for r in rows),
            "keeps_repairs_share_mean": st.mean(r["coverage"][k]["keeps_repairs_share"] for r in rows),
            "beats_random_seeds": sum(r["coverage"][k]["beats_random"] for r in rows),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_seed": rows, "summary": summary}, indent=1))
    print(f"\nwrote {out}")
    c5, c1 = summary["coverage"]["0.5"], summary["coverage"]["0.1"]
    print(f"  edits {summary['edits_range'][0]}-{summary['edits_range'][1]} of {SLICE} "
          f"({summary['edited_share_of_slice'][0]:.0%}-{summary['edited_share_of_slice'][1]:.0%})")
    print(f"  net mean {summary['net_mean']:+.0f} rows, negative on {summary['net_negative_seeds']}/5; "
          f"dAcc mean {summary['d_acc_mean_points']:+.2f} points, negative on {summary['d_acc_negative_seeds']}/5")
    print(f"  at 50% coverage reverts {100*c5['reverts_damage_share_mean']:.0f}% of the damage, "
          f"keeps {100*c5['keeps_repairs_share_mean']:.0f}% of the repairs, beats random on {c5['beats_random_seeds']}/5")
    print(f"  at 10% coverage reverts {100*c1['reverts_damage_share_mean']:.0f}% of the damage, "
          f"keeps {100*c1['keeps_repairs_share_mean']:.0f}% of the repairs, beats random on {c1['beats_random_seeds']}/5")
    print("  AUROC ranges: " + ", ".join(
        f"{k} {v[0]:.3f}-{v[1]:.3f}" for k, v in summary["auroc_range"].items()))


if __name__ == "__main__":
    main()
