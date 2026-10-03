"""Deployed-frame re-emission of the two label-referenced protocol scores.

The regime bank reports proximity and noise-robustness scored against the
evaluation-only clean label (the instrument frame of sec:scores), while the
end-to-end gate ranks corrections by their self-referenced variants
(``scripts/retrain_e_streams.py``: ``GATE_AXIS`` = text
``minimality(1-minf)_self``, image ``noise_robustness_self``). Both variants
are computed here on the same rows in one pass, so the two reference frames can
be compared directly rather than across campaigns.

Artifacts written into the run's own output directory:

    frame_report.json   per-axis AUROC and AP on the repair and damage
                        channels, for the instrument and the deployed frame,
                        with the mis-slice counts, the transition table and a
                        recheck against the AUROCs the run itself recorded
    frame_scores.npz    the four per-row score columns (both frames) plus
                        mis / succ / edited, so the gate study can be
                        recomputed in either frame without re-running a model

Enabled by ``EMIT_FRAMES=1`` so the default campaign path is unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.protocol import (
    compute_stream_scores, mis_set_ap, mis_set_auroc,
)

# (score axis, paper name)
FRAME_AXES = (("minimality(1-minf)", "proximity"),
              ("noise_robustness", "noise-robustness"))

REPORT_NAME = "frame_report.json"
SCORES_NAME = "frame_scores.npz"


def _auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """AUROC that names its own degenerate case instead of raising."""
    n = int(pos.size)
    if n == 0 or pos.sum() in (0, n):
        return float("nan")
    return float(mis_set_auroc(score, pos))


def _ap(score: np.ndarray, pos: np.ndarray) -> float:
    n = int(pos.size)
    if n == 0 or pos.sum() in (0, n):
        return float("nan")
    return float(mis_set_ap(score, pos))


def emit_frame_report(output_dir, lsnpc, X_eval, y_clean, y_noisy, device, *,
                      batch_size: int, M: int, seed: int, k_perturb: int,
                      z_vae: np.ndarray | None = None,
                      stored_auroc: dict | None = None) -> dict | None:
    """Score the eval slice in both reference frames and write the artifacts.

    ``stored_auroc`` is the AUROC table the run already wrote into its own
    ``result.json``; it is carried along so a reader can see that re-emission
    reproduces the recorded instrument values (the reproduction gate).
    """
    if os.environ.get("EMIT_FRAMES", "0") != "1":
        return None

    stream = compute_stream_scores(
        lsnpc, X_eval, np.asarray(y_noisy), device, batch_size=batch_size,
        M=M, seed=seed, k_perturb=k_perturb, z_vae=z_vae, y_clean=y_clean,
    )
    scores = stream["scores"]
    mis = np.asarray(stream["mis"]).astype(bool)
    succ = np.asarray(stream["succ"]).astype(bool)
    was_right = ~mis
    broke = ~succ & was_right

    report = {
        "n_eval": int(mis.size),
        "n_mis": int(mis.sum()),
        "mis_frac": float(mis.mean()),
        "rob_k": int(k_perturb),
        "validity_mis": float(succ[mis].mean()) if mis.any() else float("nan"),
        "frames": {},
        "transitions": stream.get("transitions"),
    }
    for axis, nice in FRAME_AXES:
        entry = {}
        for frame, key in (("instrument", axis), ("deployed", f"{axis}_self")):
            if key not in scores:
                continue
            s = np.asarray(scores[key], dtype=np.float64)
            entry[frame] = {
                "cf_auroc": _auroc(s[mis], succ[mis]),
                "cf_ap": _ap(s[mis], succ[mis]),
                "damage_auroc": _auroc(s[was_right], broke[was_right]),
                "damage_ap": _ap(s[was_right], broke[was_right]),
                "mean": float(s.mean()),
            }
        report["frames"][nice] = entry
    if stored_auroc:
        report["recheck"] = {
            nice: {"rerun": report["frames"][nice].get("instrument", {}).get("cf_auroc"),
                   "recorded": stored_auroc.get(axis)}
            for axis, nice in FRAME_AXES
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out_dir / SCORES_NAME,
        mis=mis, succ=succ,
        edited=np.asarray(stream["edited"]).astype(bool),
        **{f"score__{k}": np.asarray(v, dtype=np.float64)
           for k, v in scores.items() if v is not None},
    )
    print(f"[frames] wrote {out_dir / REPORT_NAME} and {out_dir / SCORES_NAME}")
    for axis, nice in FRAME_AXES:
        e = report["frames"].get(nice, {})
        if "instrument" in e and "deployed" in e:
            print(f"[frames] {nice:16s} instrument {e['instrument']['cf_auroc']:.3f} "
                  f"-> deployed {e['deployed']['cf_auroc']:.3f}")
    return report
