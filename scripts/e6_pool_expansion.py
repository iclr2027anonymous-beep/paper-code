"""E6: four-configuration pseudo-clean pool expansion (test-design mandate).

Round-0 LSNPC on AG News → score the candidate pool with M1 (minimality,
f<1 crossing) → calibrate the admission gate on the Round-0 VALIDATION split
(precision >= 95% at max achievable coverage) → run the four size-matched
configurations for a single round-1 LSNPC training:

    V0       reference: no pool expansion (clean set only)
    Gated    admit pool rows whose M1 score passes the calibrated gate
    Unguarded admit a RANDOM sample of size N_gated from all candidates
             (size-matched to the gated pool — the key control)
    Random   admit a random sample of size N_gated (same-size random control)

Every configuration runs >= 3 seeds. Primary metric: end-of-round-1
valid_clean + F1 on the eval split, per configuration per seed. Decision
delta: gated minus unguarded (selectivity over volume), with random and V0 as
auxiliary references. Falsification map and acceptance criteria per
the manuscript.

The gate is M1-only (minimality f<1 crossing) for the text path: the spec's
cross-encoder agreement needs a second text encoder which is not wired;
p_max is recorded as a reported secondary only. This is the honest
instrument set given what exists.

Usage: accelerate launch --mixed_precision bf16 -m scripts.e6_pool_expansion \
    --noise-type symmetric --epochs 15 --beta 0.5 --seeds 42 43 44
"""

from __future__ import annotations

from utils.paths import project_path

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accelerate import Accelerator

from experiments.data import _batched_lsnpc_infer
from experiments.lsnpc_stage1 import resolve_clean_rows
from experiments.protocol import (
    compute_protocol_scores,
    inject_noise,
    inject_noise_idn,
    multiclass_f1,
)
from models.text_encoder import build_text_encoder
from scripts import e_common
from scripts.e6_downstream_eval import head_eval
from scripts.text_lsnpc_sst2 import build_config, load_embeddings

DATA_ROOT = Path(project_path('data'))
CONFIG_NAMES = ("V0", "Gated", "Unguarded", "Random")


def _score_pool(lsnpc, X_pool, y_pool_clean, y_pool_noisy,
                device, bs, M, seed, tag: str):
    """All protocol scores over a candidate set (pool or validation split).

    Every pass computes the FULL score set — confidence (p_max),
    minimality (M1), robustness (P3'), plausibility — and prints the
    per-axis table (AUROC, AP, valid/invalid means) so the whole score
    picture is always visible, never just the gating score.
    """
    proto = compute_protocol_scores(
        lsnpc, X_pool, y_pool_clean, y_pool_noisy,
        device, batch_size=bs, M=M, seed=seed)
    print(f"--- scores on {tag} ---", flush=True)
    print(f"{'feature':24s} {'AUROC':>7s} {'AP':>7s} "
          f"{'valid_mean':>11s} {'invalid_mean':>13s} "
          f"{'dmg_AUROC':>9s} {'dmg_AP':>7s}")
    for name, a, ap, vm, im, da, dap in proto["rows"]:
        print(f"{name:24s} {a:7.3f} {ap:7.3f} {vm:11.4f} {im:13.4f} "
              f"{da:7.3f} {dap:7.3f}", flush=True)
    return (proto["scores"]["minimality(1-minf)"], proto["succ"],
            proto["rows"], np.asarray(proto["corr"]).astype(np.int64))


# Quarter volume is dropped deliberately: the selective set is held to three
# methods (the calibrated gate, the top-decile cut, the top-half cut), each
# with its volume-matched random baseline. Keep this list and the testbeds'
# cell sets consistent -- adding a volume here re-fits every seed.
DEFAULT_COVERAGES = (0.1, 0.5)


AG_CELLS_PATH = Path("results/set_f/e6_agnews_cells.json")


def _load_cells() -> dict:
    try:
        return json.loads(AG_CELLS_PATH.read_text())
    except Exception:
        return {}


def _assemble_pool(X_cand, y_cand_clean, y_cand_noisy, admit_idx):
    """Size-matched pool for one configuration: admitted rows only."""
    return X_cand[admit_idx], y_cand_clean[admit_idx], y_cand_noisy[admit_idx]


def run_e6(noise_type: str, epochs: int, beta: float, seeds: list[int],
           noise: float = 0.2, clean_set_size: int = 2000,
           val_size: int = 500, eval_slice: int = 1000,
           coverages=DEFAULT_COVERAGES, downstream: int = 1, only_configs: str | None = None):
    """Run the full E6 protocol for one noise type."""
    results = {c: [] for c in CONFIG_NAMES}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X, yc = load_embeddings("ag_news")
        n_classes = int(yc.max()) + 1
        n = len(X)
        eval_idx = rng.choice(n, size=eval_slice, replace=False)
        rest = np.setdiff1d(np.arange(n), eval_idx)
        cs_idx = rng.choice(rest, size=resolve_clean_rows(clean_set_size, len(rest) // 2),
                            replace=False)
        rest = np.setdiff1d(rest, cs_idx)
        n_val = min(val_size, len(rest) // 2)
        val_idx = rng.choice(rest, size=n_val, replace=False)
        tr_idx = np.setdiff1d(rest, val_idx)

        Xe, yce = X[eval_idx], yc[eval_idx]
        Xtr, yctr = X[tr_idx], yc[tr_idx]
        Xcs, ycs = X[cs_idx], yc[cs_idx]
        Xv, ycv = X[val_idx], yc[val_idx]
        if noise_type == "idn":
            yne = inject_noise_idn(yce, noise, seed, n_classes, X=Xe)
            yntr = inject_noise_idn(yctr, noise, seed, n_classes, X=Xtr)
            ynv = inject_noise_idn(ycv, noise, seed, n_classes, X=Xv)
            ycs_noisy = inject_noise_idn(ycs, noise, seed, n_classes, X=Xcs)
        else:
            yne = inject_noise(yce, noise, seed, n_classes)
            yntr = inject_noise(yctr, noise, seed, n_classes)
            ynv = inject_noise(ycv, noise, seed, n_classes)
            ycs_noisy = inject_noise(ycs, noise, seed, n_classes)

        latent_dim = int(X.shape[1])
        config = build_config(seed, epochs, beta, noise, 512, clean_set_size,
                              eval_slice, latent_dim, "ag_news", n_classes)
        accel = Accelerator()
        device = str(accel.device)
        enc = build_text_encoder(latent_dim=latent_dim)
        bs = int(config.batch_size)
        M = int(getattr(config, "iw_samples", 5))

        # ── Round-0 ────────────────────────────────────────────────
        print(f"\n=== [E6] {noise_type} seed={seed} ROUND-0 ===", flush=True)
        trainer0 = e_common.train_round(
            enc, Xtr, yntr, Xcs, ycs, ycs_noisy,
            Xv, ycv, ynv, n_classes, config, accel, device)
        ls0 = trainer0.model.eval()

        # ── Score the candidate pool (ALL protocol axes) ───────────
        # Candidates = noisy training rows NOT in the clean set (the pool
        # that gating could enlarge the clean set with). The full score
        # table is printed so every axis is visible, not just the gate.
        pool_mask = np.ones(len(tr_idx), dtype=bool)
        pool_X = Xtr[pool_mask]
        pool_y_clean = yctr[pool_mask]
        pool_y_noisy = yntr[pool_mask]
        # The full pass over the pool yields both the M1 gate axis and the
        # corrected labels that admitted rows will carry into round 1.
        m1_pool, succ_pool, _rows_pool, corr_pool = _score_pool(
            ls0, pool_X, pool_y_clean, pool_y_noisy, device, bs, M, seed,
            tag=f"{noise_type} seed={seed} candidate pool")
        # Gate calibration on the VALIDATION split (regime-matched,
        # disjoint from clean set and eval) — full score table printed.
        m1_val, succ_val, _rows_val, _corr_val = _score_pool(
            ls0, Xv, ycv, ynv, device, bs, M, seed,
            tag=f"{noise_type} seed={seed} validation split")
        curve, best = e_common.gate_curve(m1_val, succ_val)
        print("[E6] gate curve (val): " + " | ".join(
            (f"prec>={r['target']:.2f}: cov={r['coverage']:.3f} "
             f"prec={r['precision']:.3f}") if r["threshold"] is not None
            else f"prec>={r['target']:.2f}: INFEASIBLE" for r in curve),
            flush=True)
        print(f"[E6] best reachable precision={best['precision']:.3f} at "
              f"coverage={best['coverage']:.3f}", flush=True)
        reach = [r for r in curve if r["threshold"] is not None]
        if reach:
            gate_target = reach[0]["target"]
            thresh, cov, prec = (reach[0]["threshold"], reach[0]["coverage"],
                                 reach[0]["precision"])
            gated_admit = m1_pool >= thresh
            n_gated = int(gated_admit.sum())
            print(f"[E6] gate target={gate_target:.2f} "
                  f"threshold={thresh:.4f} coverage={cov:.3f} "
                  f"precision={prec:.3f} -> N_gated={n_gated}", flush=True)
        else:
            gate_target, thresh, cov, prec = None, None, None, None
            gated_admit = np.zeros(len(pool_X), dtype=bool)
            n_gated = 0
            print("[E6] gate infeasible at every declared target -> every "
                  "configuration degenerates to V0", flush=True)
        print(f"[E6] gated pool size N_gated={n_gated} "
              f"({n_gated / len(pool_X):.3f} of candidates)", flush=True)

        # ── Four size-matched configurations ───────────────────────
        X_pool = pool_X
        y_clean_pool = pool_y_clean
        y_noisy_pool = pool_y_noisy
        _configs = e_common.admission_configurations(m1_pool, gated_admit, n_gated, seed, coverages)
        if only_configs:
            keep_c = {c.strip() for c in only_configs.split(",")}
            _configs = [c for c in _configs if c[0] in keep_c]
        e_common.dump_pool_context(f"noisyag_{noise_type}", seed, corr=corr_pool, m1=m1_pool,
                     succ=succ_pool, y_clean=pool_y_clean,
                     y_noisy=pool_y_noisy, configs=_configs, eval_idx=eval_idx,
                     val_idx=val_idx, cs_idx=cs_idx, tr_idx=tr_idx)
        _cells = _load_cells()
        for cfg_name, arm_mask in _configs:
            _ckey = (f"noisyag_{noise_type}:{seed}:t{gate_target}:"
                    f"{cfg_name}:d{int(downstream)}")
            if _ckey in _cells:
                results.setdefault(cfg_name, []).append(_cells[_ckey])
                print(f"[E6] {_ckey} cached — skip", flush=True)
                continue
            if cfg_name == "V0":
                # Reference: no expansion, round-1 clean set = C0 only.
                X_c1, y_c1, y_c1_noisy = Xcs, ycs, ycs_noisy
                admit = np.zeros(len(Xtr), dtype=bool)
                admit_desc = "clean set only"
            else:
                admit = arm_mask
                admit_desc = (f"gated (N={n_gated})" if cfg_name == "Gated"
                              else f"{cfg_name} N={int(admit.sum())}")

                # Corrected labels downstream: an admitted row enters round-1
                # with the label the corrector emitted for it in the full pass.
                # This harness previously handed over y_clean_pool (the labels
                # the noise was injected from), which made every configuration an
                # oracle-labelled expansion and inflated the comparison.
                X_a, y_a, y_a_noisy = _assemble_pool(
                    X_pool, corr_pool, pool_y_noisy, admit)
                X_c1 = np.concatenate([Xcs, X_a], axis=0)
                y_c1 = np.concatenate([ycs, y_a], axis=0)
                y_c1_noisy = np.concatenate([ycs_noisy, y_a_noisy], axis=0)

            print(f"[E6] seed={seed} config={cfg_name} "
                  f"pool={admit_desc} clean_set={len(X_c1)}", flush=True)
            config2 = build_config(seed, epochs, beta, noise, 512,
                                   clean_set_size, eval_slice,
                                   latent_dim, "ag_news", n_classes)
            config2.output_dir = (
                f"/tmp/e6_{noise_type}_s{seed}_b{beta}_e{epochs}"
                f"_{cfg_name.lower()}")
            accel2 = Accelerator()
            keep = ~admit
            X_tr_kept, y_tr_kept = Xtr[keep], yntr[keep]
            n_removed = int((~keep).sum())
            trainer1 = e_common.train_round(
                enc, X_tr_kept, y_tr_kept, X_c1, y_c1, y_c1_noisy,
                Xv, ycv, ynv, n_classes, config2, accel2, str(accel2.device))
            ls1 = trainer1.model.eval()

            # ── Eval: valid_clean + F1 on the eval split ──────────
            Xt = torch.as_tensor(Xe, dtype=torch.float32, device=str(accel2.device))
            zv = torch.as_tensor(Xe, dtype=torch.float32, device=str(accel2.device))
            yh_t = torch.as_tensor(yne.astype(np.int64), dtype=torch.long,
                                   device=str(accel2.device))
            z_corr, c_corr = _batched_lsnpc_infer(
                ls1, Xt, zv, yh_t, bs, str(accel2.device), M)
            corr = np.argmax(c_corr.cpu().numpy(), axis=-1)
            succ = corr == yce
            if downstream:
                assert len(Xtr) == len(yntr), (
                    f"emission inputs/labels misaligned: {len(Xtr)} vs "
                    f"{len(yntr)}")
                r1 = compute_protocol_scores(
                    ls1, Xtr, yctr, yntr, str(accel2.device), batch_size=bs,
                    M=M, seed=seed)
                corr_r1 = np.asarray(r1["corr"]).astype(np.int64)
                ds_res = head_eval(Xtr, corr_r1, Xe, yce, n_classes, seed,
                                   str(accel2.device))
                print(f"[E6] {cfg_name} seed={seed}: downstream_acc="
                      f"{ds_res['acc']:.4f} (corrected-pool labels "
                      f"{(corr_r1 == yctr).mean():.3f} valid)", flush=True)
            mis = yne != yce
            n_adm = 0 if cfg_name == "V0" else int(admit.sum())
            # Admitted rows move out of the noisy training stream and into
            # the clean pool (never both: see the CIFAR harness note).
            adm_acc = (float((corr_pool[admit] == pool_y_clean[admit]).mean())
                       if n_adm else None)
            entry = {
                "noise_type": noise_type, "seed": seed, "config": cfg_name,
                "gate_target": gate_target,
                "pool_labels": "corrector-full-pass",
                "coverage": (float(cfg_name.split("@")[1]) if "@" in cfg_name
                      else None),
                "admitted_precision": (float(succ_pool[admit].mean())
                                       if n_adm else None),
                "admitted_label_role": ("supervision = corrector's label; "
                                        "conditioning input = the row's "
                                        "observed noisy label (matches this "
                                        "harness's clean set)"),
                "admitted_label_acc": adm_acc,
                "gate_target": gate_target, "gate_curve": curve,
                "gate_best_precision": best["precision"],
                "gate_best_coverage": best["coverage"],
                "n_admitted": n_adm,
                "n_train_removed": n_removed,
                "valid_clean": float(succ.mean()),
                **({"downstream_acc": ds_res["acc"],
                    "downstream_f1": ds_res["f1"],
                    "r1_pool_label_valid": float((corr_r1 == yctr).mean())}
                   if downstream else {}),
                "validity_mis": float(succ[mis].mean()) if mis.any() else float("nan"),
                "corr_f1": float(multiclass_f1(corr, yce)),
                "noisy_f1": float(multiclass_f1(yne, yce)),
                "corr_acc": float(succ.mean()),
            }
            results.setdefault(cfg_name, []).append(entry)
            _cells[_ckey] = entry
            AG_CELLS_PATH.parent.mkdir(parents=True, exist_ok=True)
            AG_CELLS_PATH.write_text(json.dumps(_cells, indent=2))
            print(f"[E6] {cfg_name} seed={seed}: valid_clean={entry['valid_clean']:.4f} "
                  f"validity_mis={entry['validity_mis']:.4f} "
                  f"corr_f1={entry['corr_f1']:.4f}", flush=True)

    # ── Aggregate + acceptance checks ──────────────────────────────
    out_path = Path(project_path(f"results/set_f/e6_{noise_type}_summary.json"))
    # Merge with any prior summary (earlier seed batches): per_seed arrays are
    # concatenated in run order and the moments are recomputed on the union,
    # so re-running with a fresh seed batch never clobbers earlier seeds.
    prior = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text()).get("configs", {})
        except Exception:
            prior = {}
    summary = {"noise_type": noise_type, "configs": {}}
    # every configuration actually run: the four calibrated names + the sweep
    arm_names = [c for c in CONFIG_NAMES if results.get(c)]
    arm_names += sorted(c for c in results if c not in CONFIG_NAMES)
    for cfg_name in arm_names:
        rows = results[cfg_name]
        if not rows:
            if cfg_name in prior:
                summary["configs"][cfg_name] = prior[cfg_name]
            continue
        vc = [r["valid_clean"] for r in rows]
        f1 = [r["corr_f1"] for r in rows]
        if cfg_name in prior:
            pv = prior[cfg_name]
            vc = list(pv.get("valid_clean", {}).get("per_seed", [])) + vc
            f1 = list(pv.get("corr_f1", {}).get("per_seed", [])) + f1
        summary["configs"][cfg_name] = {
            "valid_clean": {"mean": float(np.mean(vc)), "std": float(np.std(vc)),
                            "per_seed": vc},
            "corr_f1": {"mean": float(np.mean(f1)), "std": float(np.std(f1)),
                        "per_seed": f1},
        }
    # Selection contrast at each volume: gated (top-coverage by the gate axis) minus
    # its volume-matched random control, plus the calibrated pair.
    for a, b, label in (("Gated", "Unguarded", "gated - unguarded (calibrated)"),
                        ("c@0.1", "f@0.1", "c@0.1 - f@0.1"),
                        ("c@0.25", "f@0.25", "c@0.25 - f@0.25"),
                        ("c@0.5", "f@0.5", "c@0.5 - f@0.5")):
        av = summary["configs"].get(a, {}).get("valid_clean", {}).get("mean")
        bv = summary["configs"].get(b, {}).get("valid_clean", {}).get("mean")
        if av is not None and bv is not None:
            summary[f"{a}_minus_{b}"] = float(av - bv)
            print(f"=== [E6] {label} = {av - bv:+.4f} ===", flush=True)
            if a == "Gated" and b == "Unguarded":
                delta = float(av - bv)
    g = summary["configs"].get("Gated", {}).get("valid_clean", {})
    b = summary["configs"].get("Unguarded", {}).get("valid_clean", {})
    if g and b:
        delta = float(np.mean(g["per_seed"]) - np.mean(b["per_seed"]))
        summary["gated_minus_unguarded_valid_clean"] = delta
        print(f"\n=== [E6] {noise_type} gated - unguarded (valid_clean) = "
              f"{delta:+.4f} ===")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-type", default="symmetric", choices=["symmetric", "idn"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--downstream", type=int, default=1,
                    help="train the downstream head on the corrected dataset")
    ap.add_argument("--only-configs", default=None,
                    help="comma list: run only these configurations")
    ap.add_argument("--coverages", nargs="+", type=float, default=list(DEFAULT_COVERAGES),
                    help="admission fractions for the selectivity sweep")
    args = ap.parse_args()
    run_e6(args.noise_type, args.epochs, args.beta, args.seeds,
           coverages=args.coverages, downstream=args.downstream,
           only_configs=args.only_configs)


if __name__ == "__main__":
    main()
