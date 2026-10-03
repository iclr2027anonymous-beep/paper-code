#!/usr/bin/env python3
"""Set K — candidate-extension analysis (the manuscript appendix C, app:ceiling).

Reads the per-cell ``candidates_cs<seed>.npz`` written by
``scripts/emit_candidate_posterior.py`` and answers the two questions appendix C
leaves open:

  1. **Top-K coverage.** Share of wrong rows (``l_i != y_i``) whose clean label
     falls among the K most probable classes of the corrected-label posterior
     ``p_c``.  Coverage at K=1 is the intrinsic-validity ceiling ``vbar`` of the
     single-proposal rule: no threshold over the audit scores can repair a row
     outside it.
  2. **Does widening the candidate set raise the realised repair rate above
     ``vbar``?**  For a width K, a trust floor and an agreement check we apply the
     oracle-free rule of appendix C and measure the realised repair / damage /
     wasted rates.

The rule (oracle-free; the clean label is used only to score the outcome):

    candidates are the classes of ``K_i`` in descending posterior order,
    skipping the current label ``l_i``; the FIRST candidate ``c`` that clears
    the trust floor is adopted; if none clears it, ``l_i`` is kept.
    A relabel is withheld entirely when the independent classifier is already
    confident in the current label.

Trust-floor modes (the appendix leaves the floor to experiment):

    rel    ``p_c(c) >= rho * p_c(top-1)``   (relative to the posterior mode)
    beat   ``p_c(c) > p_c(l_i)``            (the candidate must beat the status
                                              quo under the corrected posterior)
    both   the conjunction of the two

Agreement check (comparative, scale-free): with ``lam`` finite, a relabel is
withheld when ``p_probe(l_i) >= lam * p_probe(c)`` — the independent classifier
prefers the current label over the candidate.  ``lam = inf`` disables the check
(the paper's symbol for its threshold is lambda; rule keys tag it ``agree``).  An
ABSOLUTE threshold on the probe is not usable: the plain logistic head is
uncalibrated, and the share of wrong rows above 0.9 runs from 0.3% (10 classes)
to 96.8% (100 classes), which collapsed the realised repair rate to 0.010 on
CIFAR-100N.  ``consistency.probe_deg_share`` records that diagnostic per cell.

Metrics per rule (over the annotated pool; ``is_pool`` rows):

    coverage_K        share of wrong rows with y in K_i
    repair_rate       share of wrong rows ending on y       ("moved W -> R")
    damage_rate       share of correct rows ending off y
    wasted_rate       share of wrong rows ending on a different wrong label
    kept_rate         share of rows whose label is unchanged
    net               repair_rate * P(wrong) - damage_rate * P(correct)

Baselines reported alongside: K=1 (the corrector alone, IW and plug-in), the
no-edit stream, and the clean-oracle ceiling.

Usage:
    python -m scripts.analyze_candidate_extension \\
        --dir results/set_k --out results/set_k/analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Pre-chosen grid of the open extension ("K and the trust floor are to be
# chosen by experiment").  The "agreement check" is the independent-classifier
# test of the appendix, implemented as a comparative test with threshold LAM
# (the paper's symbol is lambda).  Tag "off" disables it.
KS = (1, 2, 3, 5, 10)
FLOORS = (("rel", 0.0), ("rel", 0.25), ("rel", 0.5), ("rel", 0.75),
          ("beat", 0.0), ("both", 0.25), ("both", 0.5))
AGREEMENT = (("off", float("inf")), ("1", 1.0), ("0.5", 0.5))
METRICS = ("repair_rate", "damage_rate", "wasted_rate", "kept_rate", "net")


def apply_rule(top_idx: np.ndarray, top_p: np.ndarray, y_noisy: np.ndarray,
               K: int, mode: str, rho: float, lam: float,
               p_cur_post: np.ndarray | None,
               p_cur_probe: np.ndarray | None,
               probe_top_p: np.ndarray | None) -> np.ndarray:
    """Vectorised appendix-C rule: first eligible candidate in posterior order."""
    out = y_noisy.copy()
    decided = np.zeros(len(y_noisy), dtype=bool)
    K = int(min(K, top_idx.shape[1]))
    p_top1 = top_p[:, 0]
    check = (probe_top_p is not None and p_cur_probe is not None
             and np.isfinite(lam))
    for j in range(K):
        c = top_idx[:, j]
        p = top_p[:, j]
        elig = (~decided) & (c != y_noisy)
        if mode in ("rel", "both") and rho > 0.0:
            elig &= p >= rho * p_top1
        if mode in ("beat", "both"):
            if p_cur_post is None:
                raise SystemExit("floor mode 'beat' needs p_cur_post in the npz")
            elig &= p > p_cur_post
        if check:
            elig &= p_cur_probe < lam * probe_top_p[:, j]
        out[elig] = c[elig]
        decided |= elig
    return out


def _metrics(y_noisy: np.ndarray, y_clean: np.ndarray, out: np.ndarray) -> dict:
    wrong = y_noisy != y_clean
    correct = ~wrong
    moved = out != y_noisy
    repaired = wrong & (out == y_clean)
    damaged = correct & (out != y_clean)
    wasted = wrong & moved & (out != y_clean)
    p_wrong = float(wrong.mean())
    rep = float(repaired.sum() / max(wrong.sum(), 1))
    dam = float(damaged.sum() / max(correct.sum(), 1))
    return {"n": int(len(y_noisy)), "n_wrong": int(wrong.sum()),
            "p_wrong": p_wrong, "repair_rate": rep, "damage_rate": dam,
            "wasted_rate": float(wasted.sum() / max(wrong.sum(), 1)),
            "kept_rate": float((~moved).mean()),
            "net": rep * p_wrong - dam * (1.0 - p_wrong),
            "n_repaired": int(repaired.sum()), "n_damaged": int(damaged.sum())}


def _coverage(top_idx: np.ndarray, y_noisy: np.ndarray, y_clean: np.ndarray,
              K: int) -> float:
    wrong = y_noisy != y_clean
    if wrong.sum() == 0:
        return float("nan")
    kk = np.asarray(top_idx[wrong][:, :min(K, top_idx.shape[1])], dtype=np.int64)
    return float((kk == y_clean[wrong, None]).any(axis=1).mean())


def analyze_cell(npz: Path, ks=KS, floors=FLOORS, agreement=AGREEMENT) -> dict:
    d = np.load(npz)
    if "y_clean" not in d:
        raise SystemExit(f"{npz}: no clean oracle — appendix C needs it to score "
                         "the rule (the rule itself is oracle-free).")
    m = np.asarray(d["is_pool"], dtype=bool)

    def g(key):
        return (np.asarray(d[key], dtype=np.float64)[m] if key in d else None)

    y_noisy = np.asarray(d["y_noisy"], dtype=np.int64)[m]
    y_clean = np.asarray(d["y_clean"], dtype=np.int64)[m]
    top_idx = np.asarray(d["top_idx"], dtype=np.int64)[m]
    top_p = g("top_p")
    if top_p is None:
        raise SystemExit(f"{npz}: missing top_p (re-emit with the current "
                         "scripts.emit_candidate_posterior)")
    corr_iw = np.asarray(d["corr_iw"], dtype=np.int64)[m]
    corr_pl = np.asarray(d["corr_plugin"], dtype=np.int64)[m]
    p_cur_post = g("p_cur_post")
    p_cur_probe = g("p_cur_probe")
    probe_top_p = g("probe_top_p")
    wrong = y_noisy != y_clean

    res: dict = {"n_pool": int(m.sum()),
                 "coverage": {f"K{K}": _coverage(top_idx, y_noisy, y_clean, K)
                              for K in ks},
                 "baselines": {
                     "no_edit": _metrics(y_noisy, y_clean, y_noisy),
                     "k1_corrector_iw": _metrics(y_noisy, y_clean, corr_iw),
                     "k1_corrector_plugin": _metrics(y_noisy, y_clean, corr_pl),
                     "oracle": _metrics(y_noisy, y_clean, y_clean)},
                 "rules": {}}
    for K in ks:
        for mode, rho in floors:
            for atag, lam in agreement:
                if lam < float("inf") and p_cur_probe is None:
                    continue                   # check unavailable in this cell
                if mode in ("beat", "both") and p_cur_post is None:
                    continue                      # older npz without p_cur_post
                out = apply_rule(top_idx, top_p, y_noisy, K, mode, rho, lam,
                                 p_cur_post, p_cur_probe, probe_top_p)
                res["rules"][f"K{K}_{mode}_rho{rho:g}_agree{atag}"] = _metrics(
                    y_noisy, y_clean, out)
    res["consistency"] = {
        "top1_is_argmax": float((top_idx[:, 0] == corr_iw).mean()),
        "iw_vs_plugin_top1_disagree": float((corr_iw != corr_pl).mean()),
        "probe_conf_in_current_label": (float(np.nanmean(p_cur_probe))
                                        if p_cur_probe is not None else None),
        # check pathology diagnostic: share of WRONG rows whose probe confidence
        # in the current label exceeds 0.9 (an absolute-threshold check would
        # withhold these rows entirely).
        "probe_deg_share_wrong": (float(np.nanmean(p_cur_probe[wrong] >= 0.9))
                                  if p_cur_probe is not None else None),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("results/set_k"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.dir
    out_dir = args.out or (root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[str, dict] = {}
    for npz in sorted(root.glob("k_*/candidates_cs*.npz")):
        tag, seed = npz.parent.name, int(npz.stem.split("cs")[-1])
        cells.setdefault(tag, {})[seed] = analyze_cell(npz)
        c = cells[tag][seed]
        print(f"[analyze] {tag} cs{seed}: "
              f"vbar={c['baselines']['k1_corrector_iw']['repair_rate']:.3f} "
              f"cov@K3={c['coverage']['K3']:.3f} "
              f"probe_deg={c['consistency']['probe_deg_share_wrong']}", flush=True)
    if not cells:
        raise SystemExit(f"no cells under {root}/k_*/candidates_cs*.npz")

    # ── aggregate over corrector seeds (population SD: the Set E convention) ──
    # Each (tag, seed) is one cell; tags of the same setting differ only in the
    # corrector seed, so group by setting across tags — keying on the setting
    # alone would overwrite four of five seeds and report a spurious SD of 0.
    by_setting: dict[str, dict[str, dict]] = {}
    for tag, per_seed in cells.items():
        key = tag.replace("k_text_", "").replace("k_image_", "").rsplit("_cs", 1)[0]
        for seed, res in per_seed.items():
            by_setting.setdefault(key, {})[f"{tag}:cs{seed}"] = res

    agg: dict[str, dict] = {}
    for key, per_seed in by_setting.items():
        first = next(iter(per_seed.values()))
        entry: dict = {"seeds": sorted(per_seed), "n_seeds": len(per_seed),
                       "coverage": {}, "baselines": {}, "rules": {}}

        def _agg(store_key, name, metrics):
            for metric in metrics:
                v = np.array([c[store_key][name][metric] for c in per_seed.values()
                              if name in c[store_key]], dtype=np.float64)
                if v.size:
                    entry[store_key][name][metric] = [float(np.nanmean(v)),
                                                      float(np.nanstd(v))]

        for kname in first["coverage"]:
            v = np.array([c["coverage"][kname] for c in per_seed.values()],
                         dtype=np.float64)
            entry["coverage"][kname] = [float(np.nanmean(v)), float(np.nanstd(v))]
        for bname in first["baselines"]:
            entry["baselines"][bname] = {}
            _agg("baselines", bname, METRICS)
        for rname in first["rules"]:
            entry["rules"][rname] = {}
            _agg("rules", rname, METRICS)
        agg[key] = entry

    (out_dir / "summary.json").write_text(json.dumps(
        {"cells": cells, "aggregate": agg}, indent=2, default=str))

    cols = ["repair_mean", "repair_sd", "damage_mean", "damage_sd",
            "wasted_mean", "wasted_sd", "kept_mean", "net_mean", "net_sd"]
    rows = ["setting,rule," + ",".join(cols)]
    for key, e in sorted(agg.items()):
        for rname, met in sorted(e["rules"].items()):
            vals = [met["repair_rate"][0], met["repair_rate"][1],
                    met["damage_rate"][0], met["damage_rate"][1],
                    met["wasted_rate"][0], met["wasted_rate"][1],
                    met["kept_rate"][0], met["net"][0], met["net"][1]]
            rows.append(f"{key},{rname}," + ",".join(f"{v:.5f}" for v in vals))
    (out_dir / "deltas.csv").write_text("\n".join(rows) + "\n")
    print(f"[analyze] wrote {out_dir/'summary.json'} and {out_dir/'deltas.csv'} "
          f"({len(cells)} cells)", flush=True)


if __name__ == "__main__":
    main()
