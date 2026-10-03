"""Save experiment results (JSON + CSV summary) and load model checkpoints."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import ujson as json

log = logging.getLogger(__name__)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON completely before atomically replacing the destination.

    No try/except by policy: a failed result write must raise with its
    full traceback. If it fails mid-write the tmp file lingers; the next
    successful write replaces the same name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)
    return path


def _to_jsonable(v: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types.

    Recursion is load-bearing: a per-method result row carries nested
    containers (``history``, ``valid_at``, ``transitions``,
    ``corrected_confusion_matrix``), and ``ujson`` rejects the numpy scalar
    inside one of them (``np.int64``, ``np.bool_``) even though it accepts
    ``np.float64``. Without this the result write fails with a ``TypeError``
    at the very end of an otherwise complete experiment.

    Unknown types pass through untouched, so a genuinely unserializable
    payload still raises at dump time instead of being silently mangled.
    """
    if isinstance(v, dict):
        return {str(k): _to_jsonable(item) for k, item in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(item) for item in v]
    if isinstance(v, np.ndarray):
        return _to_jsonable(v.tolist())
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def save_json(out_dir: Path, run_id: str,
              args: dict, results: dict[str, dict[str, Any]],
              artifacts: dict[str, Any] | None = None,
              provenance: dict[str, Any] | None = None) -> Path:
    """Write per-method metrics + run args to a JSON file. Returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    payload = {
        "args": args,
        "results": _to_jsonable(results),
        "run_id": run_id,
    }
    if provenance:
        payload["provenance"] = provenance
    if artifacts:
        payload["artifacts"] = {
            key: _to_jsonable(value) for key, value in artifacts.items()
        }
    atomic_write_json(path, payload)
    log.info(f"Saved results to {path}")
    return path


def save_csv(out_dir: Path, run_id: str, results: dict[str, dict[str, Any]],
             args: dict) -> Path:
    """Write a one-row-per-method CSV summary via pandas. Returns the path.

    Cell contract: scalars (bool/int/float, Python or numpy) are written as-is;
    every other value -- nested ``history`` / ``valid_at`` / ``transitions``
    structures, confusion matrices, per-query arrays -- is written as a compact
    JSON string. The former rule skipped any list longer than one element, so a
    nested field was either dropped outright or written as a Python ``repr``
    that only ``ast.literal_eval`` could read back.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.csv"
    rows: list[dict[str, Any]] = []
    for name, m in results.items():
        row: dict[str, Any] = {"method": name}
        for k, v in m.items():
            if isinstance(v, (bool, np.bool_)):
                row[k] = bool(v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                row[k] = float(v)
            elif v is None or isinstance(v, str):
                row[k] = v
            else:
                row[k] = json.dumps(_to_jsonable(v), sort_keys=True)
        row["dataset"] = args.get("dataset")
        row["noise_type"] = args.get("noise_type")
        row["noise_rate"] = args.get("noise")
        row["n_test"] = args.get("n_test")
        row["timestamp"] = args.get("timestamp", "")
        row["run_id"] = run_id
        rows.append(row)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    log.info(f"Saved CSV summary to {path}")
    return path


def load_model_state(path: str | Path, model: torch.nn.Module,
                     device: str | None = None, strict: bool = True) -> None:
    """Load a raw ``state_dict`` checkpoint into ``model`` and set eval mode.

    ``device`` is forwarded as ``map_location`` (``None`` keeps torch's
    default behaviour of restoring tensors to their saved device).
    """
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=strict)
    model.eval()
