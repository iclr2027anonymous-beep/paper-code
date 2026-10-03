"""Evaluate the preregistered LSNPC condition-quality gate.

This consumes validated exploratory sweep manifests.  It deliberately does
not aggregate semifactual metrics: Gate 6 is about whether the corrected
predictor itself is trustworthy enough to justify the larger system sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.save import atomic_write_json

REQUIRED_METRICS = (
    "diagnostic_test_size",
    "noisy_conditioning_error",
    "corrected_conditioning_error",
    "conditioning_error_reduction",
    "noisy_ece",
    "corrected_ece",
    "noisy_brier",
    "corrected_brier",
    "corrected_prediction_hash",
)


def evaluate_gate(manifest_paths: list[Path]) -> dict:
    cells: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") not in {"validated_exploratory", "frozen"}:
            raise ValueError(
                f"manifest is not validated: {manifest_path}")
        for run in manifest["runs"]:
            result_path = manifest_path.parent / run["result_json"]
            result = json.loads(result_path.read_text())
            provenance = result["provenance"]
            config = provenance["scientific_config"]
            if provenance.get("system") != "full_lsnpc":
                # Baseline cells have no correction diagnostics and are not
                # part of this condition-quality gate.
                if provenance.get("system") == "baseline":
                    continue
                raise ValueError(
                    f"non-full LSNPC system in Gate 6: {result_path}")
            if config.get("lsnpc_loss") != "correction":
                raise ValueError(f"non-correction result in gate: {result_path}")
            method = result["results"].get("lsnpc", {})
            missing = [key for key in REQUIRED_METRICS if key not in method]
            if missing:
                raise ValueError(
                    f"missing Gate 6 metrics {missing}: {result_path}")

            identity = (config["dataset"], int(config["seed"]))
            if identity in seen:
                raise ValueError(
                    f"duplicate dataset/seed Gate 6 cell: {identity}")
            seen.add(identity)
            primary_pass = method["conditioning_error_reduction"] > 0.0
            calibration_pass = (
                method["corrected_ece"] <= method["noisy_ece"]
                or method["corrected_brier"] <= method["noisy_brier"]
            )
            cells.append({
                "dataset": config["dataset"],
                "seed": int(config["seed"]),
                "diagnostic_test_size": method["diagnostic_test_size"],
                "noisy_error": method["noisy_conditioning_error"],
                "corrected_error": method["corrected_conditioning_error"],
                "error_reduction": method["conditioning_error_reduction"],
                "noisy_ece": method["noisy_ece"],
                "corrected_ece": method["corrected_ece"],
                "noisy_brier": method["noisy_brier"],
                "corrected_brier": method["corrected_brier"],
                "primary_pass": primary_pass,
                "calibration_pass": calibration_pass,
                "cell_pass": primary_pass and calibration_pass,
                "prediction_hash": method["corrected_prediction_hash"],
                "result_json": str(result_path),
            })

    if not cells:
        raise ValueError("no LSNPC correction cells found")

    by_dataset: dict[str, dict] = {}
    for dataset in sorted({cell["dataset"] for cell in cells}):
        rows = [cell for cell in cells if cell["dataset"] == dataset]
        by_dataset[dataset] = {
            "seed_count": len(rows),
            "median_error_reduction": float(np.median(
                [row["error_reduction"] for row in rows])),
            "all_seeds_primary_pass": all(row["primary_pass"] for row in rows),
            "all_cells_pass": all(row["cell_pass"] for row in rows),
        }

    passed = all(
        summary["seed_count"] >= 2
        and summary["median_error_reduction"] > 0.0
        and summary["all_seeds_primary_pass"]
        and summary["all_cells_pass"]
        for summary in by_dataset.values()
    )
    return {
        "schema_version": 1,
        "gate": "Gate 6 condition quality",
        "status": "passed" if passed else "failed",
        "cell_count": len(cells),
        "datasets": by_dataset,
        "cells": sorted(cells, key=lambda row: (row["dataset"], row["seed"])),
        "decision": (
            "proceed_to_gate5_matrix"
            if passed
            else "stop_before_gate5_matrix_and_revise"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_gate(args.manifests)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
