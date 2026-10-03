#!/usr/bin/env python3
"""Set B — Selective correction: the operational rank gate (offline analysis).

Consumes the per-query artifacts of every per-query-capable Set A cell
(``results/set_a/json/*.json`` + ``results/set_a/per_query/*.npz``) and
computes the gate operating characteristic per (cell, gate axis):

  - the full keep-curve: gated label accuracy vs the coverage of the edit
    stream (keep the top-coverage edits by the gate score, revert the rest to the
    original noisy label), spanning never-correct (coverage->0) to blanket (coverage=1);
  - deltas vs blanket and vs never-correct at the pre-registered coverage grid
    {0.05, 0.10, 0.20, 0.30, 0.50, 1.0}, with paired bootstrap CIs at
    coverage in {0.1, 0.5};
  - the random-rejection control (R resamples of equal-size random keep
    sets; gated must beat the random median AND 95th percentile to count);
  - per-coverage gate confusion: damage removed (RW reverted) vs repair retained
    (WR kept);
  - area under the keep-curve (vs never-correct and vs blanket);
  - bottom-decile broke rate (damage share among the lowest-scored edits).

Gate axes s in {p* (conf), 1-f* (minimality), r (robustness),
pi (plausibility)} as single axes plus the pre-registered conjunctions
{(1-f*) & r, (1-f*) & r & pi-floor(alpha=0.1)}.

Threshold semantics: the coverage is a pre-committed fraction (rank cut on the
edit stream), never a fitted parameter, so nothing is tuned against the
eval slice. Under the scoring protocol, the
gate operating characteristic is the population-level coverage-keep instrument
computed on the per-query eval carve. A strict val-slice calibration would
require per-query validation dumps in the Set A cells (not archived); the
fixed coverage grid + pre-committed claim points make the eval-population curves
the honest population-level estimate here.

No GPU / no training: pure numpy over the cached Set A artifacts.

Usage:
    python -m scripts.analyze_set_b_gates \
        --json-dir results/set_a/json --npz-dir results/set_a/per_query \
        --out results/set_b [--modality text|image] [--max-cells N]

Outputs under --out:
    cell/<tag>.json     per-cell gate records (keep-curve, grid, CIs, AUCs)
    keep_curves.csv     long-format curve table for plotting/tables
    summary.json        per-axis aggregates (win shares, hierarchy table)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

GATE_COVERAGES = (0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
BOOT_COVERAGES = (0.10, 0.50)
PLAUS_FLOOR_ALPHA = 0.10
MIN_KEPT = 30          # threshold-floor rule: drop the coverage with < 30 kept rows
SCORE_KEYS = {
    "conf": "p_max(conf)",
    "minimality": "minimality(1-minf)",
    "robustness": "noise_robustness",
    "plausibility": "plausibility(density)",
}
GATE_NEEDS = {
    "conf": {"conf"},
    "minimality": {"minimality"},
    "robustness": {"robustness"},
    "plausibility": {"plausibility"},
    "minimality&robustness": {"minimality", "robustness"},
    "minimality&robustness&plausibility-floor": {
        "minimality", "robustness", "plausibility"},
}
AXIS_ORDER = ["minimality", "robustness", "conf", "plausibility",
              "minimality&robustness",
              "minimality&robustness&plausibility-floor"]

TAG_RE = {
    "text": re.compile(
        r"^text_(?P<dataset>[a-z_]+)_s(?P<seed>\d+)_n(?P<noise>[0-9.]+)_"
        r"(?P<nt>[a-z]+)_e(?P<epochs>\d+)_b(?P<beta>[0-9.]+)_cs(?P<cs>\d+)$"),
    "image": re.compile(
        r"^image_(?P<dataset>[a-z0-9]+)_(?P<view>[a-z0-9]+)_(?P<nt>[a-z0-9]+)_"
        r"(?P<backbone>[a-z0-9]+)_s(?P<seed>\d+)_e(?P<epochs>\d+)_"
        r"b(?P<beta>[0-9.]+)_cs(?P<cs>\d+)$"),
}


def parse_tag(tag: str) -> dict:
    prefix = "text_" if tag.startswith("text_") else "image_"
    m = TAG_RE[prefix[:-1]].match(tag)
    d = dict(m.groupdict()) if m else {}
    return {"modality": prefix[:-1], "tag": tag, **d}


def _candidate_gate(t, scores, gate):
    """Per-candidate gate value + eligibility mask (higher = better)."""
    wr = t == "WR"
    rw = t == "RW"
    cand = wr | rw
    g = np.full(len(t), -np.inf)
    if gate in SCORE_KEYS:
        g[cand] = scores[gate][cand]
        elig = cand
    elif gate in ("minimality&robustness",
                  "minimality&robustness&plausibility-floor"):
        # Conjunction: gate on the *worse* of the two per-axis candidate
        # ranks (equivalent to "both axes above their coverage threshold").
        r_minf = _within_rank(scores["minimality"][cand])
        r_rob = _within_rank(scores["robustness"][cand])
        g[cand] = np.minimum(r_minf, r_rob)
        elig = cand
        if gate.endswith("plausibility-floor"):
            floor = np.quantile(scores["plausibility"][cand],
                                PLAUS_FLOOR_ALPHA)
            elig = cand & (scores["plausibility"] >= floor)
    else:
        raise KeyError(gate)
    # Degenerate/abstaining scores (NaN/inf) sort last: never kept by a gate.
    g = np.where(np.isfinite(g), g, -np.inf)
    return wr, rw, cand, g, elig


def _within_rank(v: np.ndarray) -> np.ndarray:
    """Ascending rank of v normalised to [0, 1] (stable, averaged ties)."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    if len(v) > 1:
        # average the ranks inside tied groups
        uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        means = sums / cnt
        ranks = means[inv]
    return ranks / max(1.0, len(v) - 1.0)


def curve_stats(wr, rw, rr, n, g, elig, rng, r_resample, n_boot):
    """Everything for one (cell, gate): curve, grid, control, CIs, AUC.

    The gate universe is the EDIT stream (candidates = WR | RW). ``elig``
    restricts it (the plausibility-floor conjunction); at coverage ``cov`` the
    keep set is the top ``min(round(cov*m), |elig|)`` edits of the eligible
    stream ordered by the gate value (higher = better), everything else
    reverts to the original noisy label.
    """
    cand = wr | rw
    m = int(cand.sum())
    cand_pos = np.where(cand)[0]
    wr_s = wr[cand_pos].astype(np.int64)   # aligned to candidate order
    rw_s = rw[cand_pos].astype(np.int64)
    rw_tot = int(rw_s.sum())
    wr_tot = int(wr_s.sum())
    rr_n = int(rr.sum())
    noisy_acc = (rw_tot + rr_n) / n
    blanket_acc = (wr_tot + rr_n) / n

    order = np.argsort(-g[cand_pos], kind="stable")   # all candidates
    e_idx = order[elig[cand_pos]]                     # eligible, by gate value
    e_len = int(e_idx.size)

    # exact full keep-curve on the eligible lattice (k = 0..m; capped at |E|)
    e_wr = np.concatenate([[0], np.cumsum(wr_s[e_idx])])
    e_rw = np.concatenate([[0], np.cumsum(rw_s[e_idx])])
    kk = np.arange(m + 1)
    kk_c = np.minimum(kk, e_len)
    gated_ok = e_wr[kk_c] + (rw_tot - e_rw[kk_c]) + rr_n
    acc_full = gated_ok / n
    cov_full = kk / max(1, m)

    # ── grid statistics at the pre-registered coverage levels ──
    grid = {}
    for cov in GATE_COVERAGES:
        k = int(min(round(cov * m), e_len))
        top = e_idx[:k]
        wr_k = int(wr_s[top].sum())
        rw_k = int(rw_s[top].sum())
        gated_acc = (wr_k + (rw_tot - rw_k) + rr_n) / n
        rec = {
            "coverage": cov, "n_kept": k, "n_candidates": m, "n_eligible": e_len,
            "gated_acc": gated_acc,
            "d_vs_noisy": gated_acc - noisy_acc,
            "d_vs_blanket": gated_acc - blanket_acc,
            "damage_removed": (rw_tot - rw_k) / max(1, rw_tot),
            "repair_retained": wr_k / max(1, wr_tot),
            "wr_tot": int(wr_tot), "rw_tot": int(rw_tot),
            "wr_k": int(wr_k), "rw_k": int(rw_k),
            "rr_n": int(rr_n), "n": int(n),
            "dropped_threshold_floor": k < MIN_KEPT,
            "random_median": None, "random_p95": None, "random_win": False,
        }
        # Random-rejection control at equal coverage among the eligible edits.
        if 0 < cov < 1.0 and e_len > 1 and k >= MIN_KEPT:
            idx = np.argsort(rng.random((r_resample, e_len)), axis=1)[:, :k]
            rw_r = rw_s[e_idx][idx].sum(1)
            wr_r = wr_s[e_idx][idx].sum(1)
            acc_r = (wr_r + (rw_tot - rw_r) + rr_n) / n
            rec["random_median"] = float(np.median(acc_r))
            rec["random_p95"] = float(np.percentile(acc_r, 95))
            rec["random_win"] = bool(gated_acc >= rec["random_p95"] and
                                     gated_acc >= rec["random_median"])
        elif cov == 1.0:
            rec["random_median"] = rec["random_p95"] = gated_acc
        grid[str(cov)] = rec

    # ── paired bootstrap CIs on d(cov) at cov in BOOT_COVERAGES (fixed keep rule) ──
    boot = {}
    if n_boot and m > 1 and e_len > 0:
        row_ok_blanket = (wr | rr).astype(float)      # blanket-correct rows
        row_ok_noisy = (rw | rr).astype(float)        # never-correct rows
        keep_rows = np.zeros(n, dtype=bool)
        for cov in BOOT_COVERAGES:
            k = int(min(round(cov * m), e_len))
            keep_rows[:] = False
            keep_rows[cand_pos[e_idx[:k]]] = True
            row_ok_gated = (rr | (keep_rows & wr) | ((cand & ~keep_rows) & rw))
            d_noisy = row_ok_gated.astype(float) - row_ok_noisy
            d_blanket = row_ok_gated.astype(float) - row_ok_blanket
            idx = rng.integers(0, n, size=(n_boot, n))
            b_noisy = d_noisy[idx].mean(1)
            b_blank = d_blanket[idx].mean(1)
            boot[str(cov)] = {
                "d_vs_noisy": [float(np.percentile(b_noisy, 2.5)),
                               float(np.percentile(b_noisy, 97.5))],
                "d_vs_blanket": [float(np.percentile(b_blank, 2.5)),
                                 float(np.percentile(b_blank, 97.5))],
            }

    auc_vs_noisy = float(np.trapezoid(acc_full - noisy_acc, cov_full))
    auc_vs_blanket = float(np.trapezoid(acc_full - blanket_acc, cov_full))
    n_bot = max(1, m // 10)
    broke_bottom = float(rw_s[order[:n_bot]].mean())
    step = max(1, m // 200)
    return {
        "gate": None,  # filled by caller
        "n_candidates": m, "n_eligible": e_len,
        "coverage": m / n,
        "noisy_acc": noisy_acc, "blanket_acc": blanket_acc,
        "auc_vs_noisy": auc_vs_noisy, "auc_vs_blanket": auc_vs_blanket,
        "broke_bottom_decile": broke_bottom,
        "rw_base_rate": rw_tot / max(1, m),
        "grid": grid, "bootstrap_ci": boot,
        "curve": {"coverage": cov_full[::step].tolist(),
                  "acc": acc_full[::step].tolist()},
    }


def analyze_cell(json_dir: Path, npz_dir: Path, tag: str, rng,
                 r_resample: int, n_boot: int) -> dict:
    z = np.load(npz_dir / f"{tag}.npz", allow_pickle=True)
    res = json.load(open(json_dir / f"{tag}.json"))
    t = z["transition"]
    n = len(t)
    wr = t == "WR"
    rw = t == "RW"
    rr = t == "RR"
    cand = wr | rw

    scores = {a: np.asarray(z[f"score__{k}"], dtype=float)
              for a, k in SCORE_KEYS.items() if f"score__{k}" in z}

    got = {k: int((t == k).sum()) for k in ("WR", "RW", "RR", "WW")}
    if res.get("transitions") and got != res["transitions"]:
        raise ValueError(f"{tag}: npz transitions {got} != json "
                         f"{res['transitions']}")

    meta = parse_tag(tag)
    meta.update({
        "mis_frac": float(res.get("mis_frac", float("nan"))),
        "validity_mis": float(res.get("validity_mis", float("nan"))),
        "transitions": res.get("transitions"),
        "corr_acc": float(res.get("corr_acc", float("nan"))),
        "noisy_acc": float(res.get("noisy_acc", float("nan"))),
        "beta": res.get("beta"), "rob_k": res.get("rob_k"),
        "g4_mis_ok": int(wr.sum() + (t == "WW").sum()) >= 200,
        "edit_rate": float(cand.sum() / n),
        "n_eval": n,
    })

    gates = {}
    for gate in AXIS_ORDER:
        if not GATE_NEEDS[gate].issubset(scores):
            continue
        g_wr, g_rw, g_cand, g, elig = _candidate_gate(t, scores, gate)
        if not g_cand.any():
            gates[gate] = {"gate": gate, "n_candidates": 0,
                           "skipped": "no edits"}
            continue
        rec = curve_stats(g_wr, g_rw, rr, n, g, elig, rng, r_resample, n_boot)
        rec["gate"] = gate
        gates[gate] = rec
    return {"meta": meta, "gates": gates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default="results/set_a/json")
    ap.add_argument("--npz-dir", default="results/set_a/per_query")
    ap.add_argument("--out", default="results/set_b")
    ap.add_argument("--modality", choices=["text", "image"], default=None)
    ap.add_argument("--max-cells", type=int, default=0, help="debug: cap cells")
    ap.add_argument("--r-resample", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out)
    (out_dir / "cell").mkdir(parents=True, exist_ok=True)

    tags = sorted(p.stem for p in npz_dir.glob("*.npz"))
    if args.modality:
        tags = [t for t in tags if t.startswith(args.modality + "_")]
    if args.max_cells:
        tags = tags[: args.max_cells]
    if not tags:
        raise SystemExit(f"no cells found under {npz_dir}")

    rng = np.random.default_rng(args.seed)
    cells = {}
    for tag in tags:
        print(f"[set-b] {tag}", flush=True)
        cells[tag] = analyze_cell(json_dir, npz_dir, tag, rng,
                                  args.r_resample, args.n_boot)

    csv_path = out_dir / "keep_curves.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "modality", "dataset", "seed", "gate", "coverage",
                    "gated_acc", "noisy_acc", "blanket_acc"])
        for tag, cell in cells.items():
            meta = cell["meta"]
            for gate, rec in cell["gates"].items():
                if not rec.get("curve"):
                    continue
                for cov, acc in zip(rec["curve"]["coverage"], rec["curve"]["acc"]):
                    w.writerow([tag, meta["modality"], meta.get("dataset"),
                                meta.get("seed"), gate, round(cov, 4),
                                round(acc, 6), round(rec["noisy_acc"], 6),
                                round(rec["blanket_acc"], 6)])
    print(f"wrote {csv_path}")

    summary = {"notes": [
        "cells = Set A per-query eval carves (one per seed run)",
        "gate keep rule: keep the top-coverage edits by gate score, revert the rest",
        "random_win: gated_acc >= random median AND p95 at equal coverage (R resamples)",
        "dropped_threshold_floor: <30 kept rows at that coverage (plan SS0 floor)",
        "g4_mis_ok: mis-slice >= 200 on the eval carve",
        "AUC vs noisy/blanket over the exact full keep-curve (coverage in [0,1])",
        "strict val-slice calibration not archived in Set A cells; the coverage is a "
        "pre-committed keep fraction (no fitted parameter) - module docstring",
    ], "cells": {}, "axes": {}}

    for tag, cell in cells.items():
        (out_dir / "cell" / f"{tag}.json").write_text(json.dumps(cell, indent=1))
        summary["cells"][tag] = {
            "meta": cell["meta"],
            "auc_vs_noisy": {g: r.get("auc_vs_noisy")
                             for g, r in cell["gates"].items()},
            "d_at_q10": {g: (r.get("grid") or {}).get("0.1", {}).get(
                "d_vs_noisy") for g, r in cell["gates"].items()},
        }

    # Aggregation per plan Set B acceptance: ex-ante population = all
    # G4-passing cells (mis-slice >= 200), modality-separated. Text cells
    # where blanket is already constructive are scored in a no-harm bucket
    # (gated ~= blanket) and EXCLUDED from the selectivity numerator.
    # random_win = gated >= random median AND p95 at equal coverage (R >= 100).
    NO_HARM_TOL = 0.02  # constructive cells: gated >= blanket - tol => no harm
    for mod in ("text", "image"):
        mod_cells = [c for c in cells.values() if c["meta"]["modality"] == mod]
        if not mod_cells:
            continue
        pop = [c for c in mod_cells if c["meta"]["g4_mis_ok"]]
        for gate in AXIS_ORDER:
            rw_by_coverage = {cov: {"n": 0, "win": 0, "sel_num": 0, "sel_win": 0,
                        "nh_num": 0, "nh_pass": 0, "constr": 0}
                    for cov in (0.1, 0.5)}
            aucs, d50, d10, bot, coverages = [], [], [], [], []
            for c in pop:
                r = c["gates"].get(gate)
                if not r or not r.get("grid"):
                    continue
                aucs.append(r["auc_vs_noisy"])
                d50.append(r["grid"]["0.5"]["d_vs_noisy"])
                d10.append(r["grid"]["0.1"]["d_vs_noisy"])
                bot.append(r["broke_bottom_decile"])
                coverages.append(r["coverage"])
                constructive = r["blanket_acc"] > r["noisy_acc"] + 1e-9
                for cov in (0.1, 0.5):
                    gr = r["grid"][str(cov)]
                    if gr["dropped_threshold_floor"]:
                        continue
                    bucket = rw_by_coverage[cov]
                    bucket["n"] += 1
                    if constructive:
                        bucket["constr"] += 1
                        bucket["nh_num"] += 1
                        if gr["d_vs_blanket"] >= -NO_HARM_TOL:
                            bucket["nh_pass"] += 1
                        if gr["random_win"]:
                            bucket["win"] += 1
                    else:
                        bucket["sel_num"] += 1
                        if gr["random_win"] and gr["d_vs_blanket"] >= 0:
                            bucket["sel_win"] += 1
                        if gr["random_win"]:
                            bucket["win"] += 1
            out = {
                "n_g4_cells": len(pop),
                "random_win_share_coverage10": (rw_by_coverage[0.1]["win"] / rw_by_coverage[0.1]["n"]
                                         if rw_by_coverage[0.1]["n"] else None),
                "random_win_share_coverage50": (rw_by_coverage[0.5]["win"] / rw_by_coverage[0.5]["n"]
                                         if rw_by_coverage[0.5]["n"] else None),
                "selectivity_win_share_coverage10": (rw_by_coverage[0.1]["sel_win"] /
                                              rw_by_coverage[0.1]["sel_num"]
                                              if rw_by_coverage[0.1]["sel_num"] else None),
                "selectivity_win_share_coverage50": (rw_by_coverage[0.5]["sel_win"] /
                                              rw_by_coverage[0.5]["sel_num"]
                                              if rw_by_coverage[0.5]["sel_num"] else None),
                "no_harm_pass_share_coverage10": (rw_by_coverage[0.1]["nh_pass"] /
                                           rw_by_coverage[0.1]["nh_num"]
                                           if rw_by_coverage[0.1]["nh_num"] else None),
                "no_harm_pass_share_coverage50": (rw_by_coverage[0.5]["nh_pass"] /
                                           rw_by_coverage[0.5]["nh_num"]
                                           if rw_by_coverage[0.5]["nh_num"] else None),
                "mean_auc_vs_noisy": float(np.mean(aucs)) if aucs else None,
                "mean_d10_vs_noisy": float(np.mean(d10)) if d10 else None,
                "mean_d50_vs_noisy": float(np.mean(d50)) if d50 else None,
                "mean_broke_bottom_decile": float(np.mean(bot)) if bot else None,
                "mean_edit_rate": float(np.mean(coverages)) if coverages else None,
            }
            summary["axes"].setdefault(gate, {})[mod] = out
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"wrote {out_dir}/summary.json  ({len(cells)} cells)")
    for gate in AXIS_ORDER:
        for mod in ("text", "image"):
            a = summary["axes"].get(gate, {}).get(mod)
            if a:
                def _f(x, nd=4):
                    return "None" if x is None else f"{x:.{nd}f}"
                print(f"  {gate:44s} {mod:5s} "
                      f"randwin coverage10={_f(a['random_win_share_coverage10'])} "
                      f"coverage50={_f(a['random_win_share_coverage50'])} "
                      f"sel coverage10={_f(a['selectivity_win_share_coverage10'])} "
                      f"coverage50={_f(a['selectivity_win_share_coverage50'])} "
                      f"auc={_f(a['mean_auc_vs_noisy'])}")


if __name__ == "__main__":
    main()
