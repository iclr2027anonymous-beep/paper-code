"""Deterministic identities and provenance records for experiment runs."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.save import atomic_write_json as _write_json_atomic

log = logging.getLogger(__name__)


# These fields control where/how a run executes, but not the scientific
# configuration being evaluated. In particular, ``seed`` is intentionally
# *not* excluded.
OPERATIONAL_CONFIG_FIELDS = frozenset({
    "output_dir",
    "force_retrain",
    "formal_run",
    "lsnpc_loss_explicit",
    "system_explicit",
})


def _normalise(value: Any) -> Any:
    """Convert nested config values to stable JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _normalise(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalise(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Unsupported value in experiment config: {type(value).__name__}"
    )


def scientific_config(config: Any) -> dict[str, Any]:
    """Return the public dataclass fields that define a scientific run."""
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    return {
        field.name: _normalise(getattr(config, field.name))
        for field in fields(config)
        if field.name not in OPERATIONAL_CONFIG_FIELDS
    }


def scientific_config_hash(payload: dict[str, Any], length: int = 16) -> str:
    """Hash an already-normalised scientific config deterministically."""
    encoded = json.dumps(
        _normalise(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def config_hash(config: Any, length: int = 16) -> str:
    """Hash the complete scientific configuration deterministically."""
    return scientific_config_hash(scientific_config(config), length=length)


def checkpoint_dir(output_dir: str | Path, config: Any) -> Path:
    """Return the isolated checkpoint directory for one exact config."""
    return Path(output_dir) / "checkpoints" / config_hash(config)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint metadata must be an object: {path}")
    return payload


def checkpoint_file_hashes(directory: str | Path) -> dict[str, str]:
    """Hash model files in a checkpoint directory, excluding metadata."""
    directory = Path(directory)
    excluded = {"checkpoint_identity.json", "checkpoint_index.json"}
    return {
        str(path.relative_to(directory)): hash_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _purge_checkpoint_dir(directory: Path, reason: str) -> None:
    """Discard an incomplete/stale checkpoint directory (crash recovery).

    A killed run can leave model files behind without a completed identity
    or index. Rather than raising and blocking the next run, purge the
    garbage so the same config retrains cleanly.
    """
    log.warning(f"Purging incomplete/stale checkpoint dir {directory}: {reason}")
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            path.unlink()
    for sub in sorted(directory.rglob("*"), reverse=True):
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()


def prepare_checkpoint_dir(output_dir: str | Path, config: Any) -> Path:
    """Create or validate the checkpoint cache for one exact config.

    A directory whose identity/config hash does not match (or that has
    garbage with no identity) is treated as stale and purged.  A *same-
    config* directory from an interrupted run (model files present but no
    completed index) is kept: per-component loaders reuse the existing
    classifiers / VAEs instead of retraining everything.  A completed index
    whose file hashes no longer match is a hard error (real corruption).
    """
    directory = checkpoint_dir(output_dir, config)
    directory.mkdir(parents=True, exist_ok=True)
    expected_hash = config_hash(config)
    identity_path = directory / "checkpoint_identity.json"
    index_path = directory / "checkpoint_index.json"
    model_files = checkpoint_file_hashes(directory)

    # Purge only stale/garbage state.  Same-config files from an
    # interrupted run are reusable and are deliberately kept.
    if identity_path.exists():
        identity = _read_json(identity_path)
        if identity.get("config_hash") != expected_hash:
            _purge_checkpoint_dir(directory, "checkpoint belongs to a different config")
            identity_path = directory / "checkpoint_identity.json"
            index_path = directory / "checkpoint_index.json"
            model_files = checkpoint_file_hashes(directory)
    elif model_files or index_path.exists():
        _purge_checkpoint_dir(directory, "checkpoint identity missing")
        identity_path = directory / "checkpoint_identity.json"
        index_path = directory / "checkpoint_index.json"
        model_files = checkpoint_file_hashes(directory)

    if not identity_path.exists():
        _write_json_atomic(
            identity_path,
            {
                "schema_version": 1,
                "config_hash": expected_hash,
                "scientific_config": scientific_config(config),
            },
        )

    if index_path.exists():
        index = _read_json(index_path)
        expected_files = index.get("files")
        if not isinstance(expected_files, dict):
            raise ValueError(f"checkpoint index lacks file hashes: {index_path}")
        if model_files != expected_files:
            raise ValueError(
                f"checkpoint file hash mismatch in {directory}")
    elif model_files:
        log.warning(
            f"Checkpoint dir {directory} has model files but no completed "
            "index (interrupted run); reusing existing files.")

    return directory


def finalise_checkpoint_index(
    directory: str | Path,
    config: Any,
) -> dict[str, str]:
    """Freeze checkpoint file hashes after a successful run."""
    directory = Path(directory)
    files = checkpoint_file_hashes(directory)
    if not files:
        raise ValueError(f"no checkpoint files were produced in {directory}")
    _write_json_atomic(
        directory / "checkpoint_index.json",
        {
            "schema_version": 1,
            "config_hash": config_hash(config),
            "files": files,
        },
    )
    return files


def hash_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_arrays(named_arrays: Iterable[tuple[str, Any]]) -> str:
    """Hash named arrays including name, shape, dtype, and byte content."""
    digest = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def git_commit(repo_dir: str | Path) -> str:
    """Return the current Git commit, or ``unknown`` outside a checkout."""
    if not (Path(repo_dir) / ".git").exists():
        return "unknown"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_is_dirty(repo_dir: str | Path) -> bool | None:
    """Return whether tracked or untracked files differ from HEAD."""
    if not (Path(repo_dir) / ".git").exists():
        return None
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
