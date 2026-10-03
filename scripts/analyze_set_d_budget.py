#!/usr/bin/env python3
"""Set D — Clean-budget axis (nested, fair): offline budget-curve analysis.

Consumes the per-cell result jsons of the Set D grid
(``results/set_d/json/*.json``) and reports the budget mechanism evidence of
the experiment protocol SS4:

  - corr-acc / corr-F1 / validity / noisy-acc vs clean-set budget, per
    family = (modality, dataset, backbone), budget cs from the D tag;
  - per-curve monotonicity slope of corr-acc vs cs (each curve = one
    (family, seed, noise-config) triple of cells whose cs are nested
    prefixes of ONE seeded permutation — the causal construction);
  - mean slope with a paired bootstrap CI across curves (resampling curves,
    i.e. the seed/level population);
  - image-frozen flatness test (|slope| within seed noise, CI includes 0);
  - image-trainable positive-slope test (the C-b content of D);
  - RW damage counts across budgets (mechanism read-out: damage must stay
    flat for "extra clean rows do not change corrector behaviour");

No GPU / no training: pure numpy + json over the archived cells. Runs with

    python -m scripts.analyze_set_d_budget \
        --json-dir results/set_d/json --out results/set_d

Outputs under --out: budget_curves.csv (long table), summary.json
(per-family slopes + per-budget aggregates) and a console summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

# D tags carry every dial (sweep-copy rule): eta 0.1 is explicit, cs varies.
TEXT_TAG_RE = re.compile(
    r"^text_(?P<dataset>noisyag_med|medical_abstracts)_s(?P<seed>\d+)_"
    r"n(?P<noise>[0-9.]+)_(?P<nt>symmetric|idn)_e(?P<epochs>\d+)_"
    r"b(?P<beta>[0-9.]+)_eta0\.1_cs(?P<cs>\d+)$")
IMAGE_TAG_RE = re.compile(
    r"^image_(?P<dataset>cifar10n|cifar100n)_(?P<view>[a-z]+)_(?P<nt>[a-z]+)_"
    r"(?P<bb>resnet50|resnet50train)_s(?P<seed>\d+)_e(?P<epochs>\d+)_"
    r"b(?P<beta>[0-9.]+)_eta0\.1_cs(?P<cs>\d+)$")

def parse_tag(tag: str) -> dict | None:
    m = TEXT_TAG_RE.match(tag)
    if m:
        d = dict(m.groupdict())
        return {"modality": "text", "dataset": d["dataset"],
                "backbone": "mpnet", "view": f"{d['noise']}/{d['nt']}",
                "seed": int(d["seed"]), "cs": int(d["cs"]),
                "epochs": int(d["epochs"]), "tag": tag}
    m = IMAGE_TAG_RE.match(tag)
    if m:
        d = dict(m.groupdict())
        return {"modality": "image", "dataset": d["dataset"],
                "backbone": d["bb"], "view": d["view"],
                "seed": int(d["seed"]), "cs": int(d["cs"]),
                "epochs": int(d["epochs"]), "tag": tag}
    return None


def family_key(meta: dict) -> tuple:
    return (meta["modality"], meta["dataset"], meta["backbone"])


def _load_results(json_dir: Path, epochs: int | None = None,
                  epochs_of: dict | None = None,
                  ) -> tuple[list[dict], dict[str, dict]]:
    """Load the ladder cells, optionally restricted to one training budget.

    A store can hold more than one budget for the same configuration (the
    ladder ran ten epochs before it ran five). Those cells are not
    interchangeable: averaging them measures the budget change, not the seeds,
    so the budget is part of the configuration and the count is printed.
    """
    cells, by_tag = [], {}
    seen: dict[int, int] = {}
    for p in sorted(json_dir.glob("*.json")):
        meta = parse_tag(p.stem)
        if meta is None:
            continue
        seen[meta["epochs"]] = seen.get(meta["epochs"], 0) + 1
        want = epochs
        if epochs_of:
            want = epochs_of.get(meta["modality"], epochs)
        if want is not None and meta["epochs"] != want:
            continue
        r = json.loads(p.read_text())
        if any(k not in r for k in ("corr_acc", "noisy_acc", "corr_f1",
                                    "transitions", "validity_mis")):
            continue
        meta.update(
            corr_acc=float(r["corr_acc"]), noisy_acc=float(r["noisy_acc"]),
            corr_f1=float(r["corr_f1"]), validity_mis=float(r["validity_mis"]),
            mis_frac=float(r.get("mis_frac", np.nan)),
            WR=int(r["transitions"].get("WR", 0)),
            RW=int(r["transitions"].get("RW", 0)),
            RR=int(r["transitions"].get("RR", 0)),
            WW=int(r["transitions"].get("WW", 0)),
            n_eval=int(r.get("n_eval", 0)),
        )
        cells.append(meta)
        by_tag[p.stem] = meta
    if seen:
        counts = ", ".join(f"e{k}: {v}" for k, v in sorted(seen.items()))
        used = dict(epochs_of or {})
        if epochs is not None:
            used = {"text": epochs, "image": epochs}
        tail = (", using " + ", ".join(f"{m} e{e}" for m, e in sorted(used.items()))
                if used else "")
        print(f"[cells] training budgets present: {counts}{tail}")
    return cells, by_tag


def _curve_rows(cells: list[dict], fam: tuple) -> dict[tuple, list[dict]]:
    """Group cells into (family, seed, view) budget curves."""
    curves: dict[tuple, list[dict]] = {}
    for c in cells:
        if family_key(c) != fam:
            continue
        key = (c["dataset"], c["seed"], c["view"], c["epochs"])
        curves.setdefault(key, []).append(c)
    return curves


def _log_slope(ys: list[float], xs: list[int]) -> float:
    """OLS slope of corr-acc vs log2(cs): budget doublings, comparable steps."""
    if len(xs) < 2:
        return float("nan")
    x = np.log2(np.asarray(xs, dtype=float))
    y = np.asarray(ys, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _boot_ci(samples: np.ndarray, n_boot: int = 2000,
             alpha: float = 0.05) -> tuple:
    """Percentile CI over the sample population (bootstrap over curves)."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if len(samples) == 1:
        return (float(samples[0]), float(samples[0]), float(samples[0]))
    rng = np.random.default_rng(0)
    boots = np.array([np.mean(rng.choice(samples, len(samples), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, 100 * alpha / 2), np.percentile(
        boots, 100 * (1 - alpha / 2))
    return (float(samples.mean()), float(lo), float(hi))


def _monotone(ys: list[float]) -> bool:
    return all(np.greater_equal(ys[i + 1], ys[i] - 1e-9)
               for i in range(len(ys) - 1))


def analyze(json_dir: Path, out_dir: Path, n_boot: int = 2000,
            epochs: int | None = None, epochs_of: dict | None = None) -> dict:
    cells, _ = _load_results(json_dir, epochs=epochs, epochs_of=epochs_of)
    fams = sorted({family_key(c) for c in cells})
    long_rows: list[dict] = []
    summary: dict = {"budgets": sorted({c["cs"] for c in cells}),
                     "families": {}, "notes": [
                         "per-cell metric = eval-carve protocol result "
                         "(rows invariant across budgets by nested-prefix "
                         "construction)",
                         "slope = OLS of corr_acc vs log2(cs) per curve "
                         "(curve = family x seed x noise-config x budget)"
                         + (", cells restricted to "
                            + ", ".join(f"{m} e{e}" for m, e in sorted(
                                ({"text": epochs, "image": epochs} if epochs
                                 else (epochs_of or {})).items()))
                            if (epochs or epochs_of) else ""),
                         "paired bootstrap CI over curves (2000 resamples)",
                         "flatness: image-frozen |mean slope| within seed "
                         "noise (CI containing 0)",
                     ]}
    for fam in fams:
        curves = _curve_rows(cells, fam)
        slopes, deltas, mono_ok, mono_bad = [], [], 0, 0
        per_budget: dict[int, list[dict]] = {}
        for key, rows in sorted(curves.items()):
            rows = sorted(rows, key=lambda r: r["cs"])
            if not _monotone([r["cs"] for r in rows]):
                continue  # incomplete budget sequence; skip this curve
            for r in rows:
                per_budget.setdefault(r["cs"], []).append(r)
            ys = [r["corr_acc"] for r in rows]
            slopes.append(_log_slope(ys, [r["cs"] for r in rows]))
            deltas.append(ys[-1] - ys[0])
            mono_ok += int(all(ys[i] <= ys[i + 1] + 1e-9
                               for i in range(len(ys) - 1)))
            if len(rows) > 1 and any(ys[i] > ys[i + 1] + 1e-9
                                     for i in range(len(ys) - 1)):
                mono_bad += 1
        mean_slope, lo_s, hi_s = _boot_ci(np.asarray(slopes), n_boot)
        mean_delta, lo_d, hi_d = _boot_ci(np.asarray(deltas), n_boot)
        budget_agg = {}
        for cs, rows in sorted(per_budget.items()):
            def agg(k):
                v = np.asarray([r[k] for r in rows], dtype=float)
                return dict(mean=float(np.nanmean(v)),
                            sd=float(np.nanstd(v)) if len(v) > 1 else 0.0,
                            n=len(v))
            budget_agg[str(cs)] = {
                "corr_acc": agg("corr_acc"), "corr_f1": agg("corr_f1"),
                "validity_mis": agg("validity_mis"),
                "noisy_acc": agg("noisy_acc"), "RW": agg("RW"),
                "WR": agg("WR"), "edit_n": agg("RR")}
            for r in rows:
                long_rows.append({
                    "modality": fam[0], "dataset": fam[1], "backbone": fam[2],
                    "view": r["view"], "seed": r["seed"], "cs": r["cs"],
                    "corr_acc": r["corr_acc"], "corr_f1": r["corr_f1"],
                    "validity_mis": r["validity_mis"],
                    "noisy_acc": r["noisy_acc"], "RW": r["RW"], "WR": r["WR"],
                    "n_eval": r["n_eval"], "mis_frac": r["mis_frac"]})
        summary["families"]["/".join(fam)] = {
            "n_curves": len(slopes), "monotone_curves": mono_ok,
            "nonmonotone_curves": mono_bad,
            "slope_corr_acc_vs_log2cs": {
                "mean": mean_slope, "ci95": [lo_s, hi_s],
                "ci_excludes_zero": (lo_s > 0) if mean_slope > 0
                                    else (hi_s < 0) if mean_slope < 0
                                    else False},
            "delta_corr_acc_100pct_minus_5pct": {
                "mean": mean_delta, "ci95": [lo_d, hi_d]},
            "by_budget": budget_agg,
        }
    with open(out_dir / "budget_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys())
                           if long_rows else ["modality"])
        w.writeheader()
        w.writerows(long_rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # ── console ──
    print("=== Set D budget axis (corr_acc vs cs, per family) ===")
    print(f"{'family':52s} {'n':>4s} {'mono':>5s} {'slope/2x':>9s} "
          f"{'CI95':>18s} {'delta100-5':>10s} {'CI95':>18s}")
    for famkey, s in summary["families"].items():
        sl, ci = s["slope_corr_acc_vs_log2cs"]["mean"], \
            s["slope_corr_acc_vs_log2cs"]["ci95"]
        dl, dci = s["delta_corr_acc_100pct_minus_5pct"]["mean"], \
            s["delta_corr_acc_100pct_minus_5pct"]["ci95"]
        print(f"{famkey:52s} {s['n_curves']:4d} {s['monotone_curves']:5d} "
              f"{sl:+9.4f} [{ci[0]:+8.4f},{ci[1]:+8.4f}] "
              f"{dl:+10.4f} [{dci[0]:+8.4f},{dci[1]:+8.4f}]")
    print("\nper-budget RW (damage counts, mean across seeds):")
    for famkey, s in summary["families"].items():
        row = "  ".join(
            f"cs{c}:{s['by_budget'][str(c)]['RW']['mean']:.0f}±"
            f"{s['by_budget'][str(c)]['RW']['sd']:.1f}"
            for c in summary["budgets"] if str(c) in s["by_budget"])
        print(f"  {famkey:50s} {row}")
    print(f"\nwrote {out_dir}/budget_curves.csv and {out_dir}/summary.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default="results/set_d/json")
    ap.add_argument("--out", default="results/set_d")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=None,
                    help="restrict every modality to one training budget")
    ap.add_argument("--epochs-image", type=int, default=5,
                    help="training budget of the image ladder (5)")
    ap.add_argument("--epochs-text", type=int, default=15,
                    help="training budget of the text ladder (15)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    epochs_of = {"image": args.epochs_image, "text": args.epochs_text}
    analyze(Path(args.json_dir), out, args.n_boot, epochs=args.epochs,
            epochs_of=epochs_of)


if __name__ == "__main__":
    main()
