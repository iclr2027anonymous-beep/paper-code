import json
from pathlib import Path

from scripts.evaluate_lsnpc_gate import evaluate_gate


def _write_cell(root: Path, dataset: str, seed: int, reduction: float,
                calibration_improves: bool) -> dict:
    result_path = root / f"{dataset}_{seed}.json"
    method = {
        "diagnostic_test_size": 100,
        "noisy_conditioning_error": 0.4,
        "corrected_conditioning_error": 0.4 - reduction,
        "conditioning_error_reduction": reduction,
        "noisy_ece": 0.1,
        "corrected_ece": 0.09 if calibration_improves else 0.11,
        "noisy_brier": 0.4,
        "corrected_brier": 0.39 if calibration_improves else 0.41,
        "corrected_prediction_hash": f"hash-{dataset}-{seed}",
    }
    result_path.write_text(json.dumps({
        "provenance": {
            "system": "full_lsnpc",
            "scientific_config": {
                "dataset": dataset,
                "seed": seed,
                "lsnpc_loss": "correction",
                "system": "full_lsnpc",
            },
        },
        "results": {"lsnpc": method},
    }))
    return {"result_json": result_path.name}


def test_gate_requires_every_seed_and_calibration_to_pass(tmp_path):
    runs = [
        _write_cell(tmp_path, "heloc", 42, 0.03, True),
        _write_cell(tmp_path, "heloc", 43, -0.01, True),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "validated_exploratory",
        "runs": runs,
    }))

    report = evaluate_gate([manifest])

    assert report["status"] == "failed"
    assert report["decision"] == "stop_before_gate5_matrix_and_revise"
    assert report["datasets"]["heloc"]["all_seeds_primary_pass"] is False


def test_gate_passes_two_consistent_seeds(tmp_path):
    runs = [
        _write_cell(tmp_path, "heloc", 42, 0.03, True),
        _write_cell(tmp_path, "heloc", 43, 0.01, True),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "validated_exploratory",
        "runs": runs,
    }))

    assert evaluate_gate([manifest])["status"] == "passed"
