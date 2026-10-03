"""LSNPC checkpoint bundles — save/reload trained correctors for re-use.

Why: the experiment scripts (image_lsnpc.py, text_lsnpc_sst2.py) train an
LSNPC per run and previously persisted only per-query scores + result.json —
the trained weights lived and died inside /tmp. Set E (end-to-end retrain)
and the planned train-pool emission mode need the trained corrector to be
re-loadable without retraining.

A bundle is a single ``.pt`` containing everything needed to rebuild the
model for inference and to reproduce the data view it was trained on:

    {
      "lsnpc_state": model.state_dict(),        # strict-loadable
      "arch": {...},                            # build_lsnpc kwargs
      "splits": {"eval_idx", "val_idx", "cs_idx", "tr_idx"},  # np arrays
      "run": {...},                             # CLI-relevant run metadata
      "val_err_h": float,                       # best val error of the run
      "code_version": git sha | "unknown",
      "format": 1,
    }

Save location: ``results/ckpt/lsnpc/<tag>/<name>.pt`` (persistent,
git-ignored alongside the other result caches) — never /tmp.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.lsnpc import build_lsnpc

FORMAT_VERSION = 1


def git_sha(repo_root: Path | None = None) -> str:
    """Best-effort short sha of the repo; 'unknown' outside a repo."""
    root = (repo_root or Path(__file__).resolve().parents[1])
    if not (root / ".git").exists():
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[warn] git_sha unavailable ({exc}); provenance=unknown",
              file=sys.stderr)
        return "unknown"
    return out.stdout.strip() or "unknown"


def save_bundle(
    path: str | Path,
    model: torch.nn.Module,
    arch: dict,
    splits: dict[str, np.ndarray | None],
    run: dict,
    val_err_h: float | None,
) -> Path:
    """Persist a trained LSNPC + everything needed to rebuild it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format": FORMAT_VERSION,
        "lsnpc_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arch": dict(arch),
        "splits": {k: (None if v is None else np.asarray(v))
                   for k, v in splits.items()},
        "run": dict(run),
        "val_err_h": val_err_h,
        "code_version": git_sha(),
    }
    torch.save(bundle, path)
    return path


#: A bundle without this key cannot be rebuilt at all (the key predates every
#: bundle written, so its absence is corruption, not an older format).
_REQUIRED = object()


def _optional_float(value):
    """``float`` that passes ``None`` through -- the focal alpha's unset value."""
    return None if value is None else float(value)


#: The LSNPC build schema, in one place: bundle key -> (config attribute,
#: value a bundle predating the key means, cast).
#:
#: Written out per site (as it was, in the trainer, the text entry point and
#: ``make_config_shim``) this drifted twice: the shim demanded a ``decoder_type``
#: key the writer had stopped emitting, so every load raised; and the focal
#: alpha had no bundle key, so a focal run reloaded as plain cross-entropy.
#: ``default`` must reproduce the run as trained -- loss-side fields cannot
#: disturb a strict ``state_dict`` load, structural ones can.
LSNPC_BUNDLE_FIELDS: tuple[tuple[str, str, object, object], ...] = (
    # bundle key             config attribute            default    cast
    ("latent_dim",           "latent_dim",               _REQUIRED, int),
    ("hidden_dim",           "hidden_dim",               128,       int),
    ("nu0",                  "lsnpc_nu0",                2.0,       float),
    ("beta",                 "lsnpc_beta",               1.0,       float),
    ("focal_gamma",          "lsnpc_focal_gamma",        0.0,       float),
    ("focal_alpha",          "lsnpc_focal_alpha",        None,      _optional_float),
    ("image_data",           "lsnpc_image_data",         False,     bool),
    ("img_channels",         "lsnpc_img_channels",       3,         int),
    ("img_size",             "lsnpc_img_size",           96,        int),
    ("encoder_backbone",     "encoder_backbone",         "conv",    str),
    ("freeze_backbone",      "freeze_backbone",          True,      bool),
    ("head",                 "lsnpc_head",               "concat",  str),
    ("image_embed_dim",      "lsnpc_embed_dim",          128,       int),
    ("correction_cond",      "lsnpc_correction_cond",    "none",    str),
    ("shared_yhat_embed",    "lsnpc_shared_yhat_embed",  False,     bool),
    ("correction_input",     "lsnpc_correction_input",   "gated",   str),
)


def bundled_build_kwargs(config, *, x_dim: int, n_classes: int) -> dict:
    """The ``arch["build_kwargs"]`` a bundle records for this config.

    Also the contract for what a config must carry: the keys here are exactly
    the ones :func:`make_config_shim` reads back.
    """
    kwargs = {"x_dim": int(x_dim), "n_classes": int(n_classes),
              # Stored post-decrement: build_lsnpc subtracts 1 from the config's
              # block count, so the bundle holds the number the model was built
              # with and the shim adds it back.
              "n_blocks": max(1, int(getattr(config, "n_blocks", 3)) - 1)}
    for key, attr, default, cast in LSNPC_BUNDLE_FIELDS:
        value = getattr(config, attr) if default is _REQUIRED else getattr(
            config, attr, default)
        kwargs[key] = cast(value)
    return kwargs


def make_config_shim(build_kwargs: dict) -> SimpleNamespace:
    """A minimal config stand-in carrying exactly the fields ``build_lsnpc``
    reads (same attribute names as ExperimentConfig). Refactor-safe: no
    ExperimentConfig import, nothing pickled but plain kwargs."""
    fields = {attr: (build_kwargs[key] if default is _REQUIRED
                     else build_kwargs.get(key, default))
              for key, attr, default, _cast in LSNPC_BUNDLE_FIELDS}
    fields["n_blocks"] = int(build_kwargs["n_blocks"]) + 1  # build_lsnpc subtracts 1
    return SimpleNamespace(**fields)


def load_bundle(path: str | Path, device: str = "cpu") -> dict:
    """Load a bundle and rebuild the LSNPC model (strict state load).

    The model is rebuilt from the explicit ``arch["build_kwargs"]`` stored
    at save time via ``make_config_shim`` — no config object is pickled, so
    a bundle survives config/argparse refactors.
    """

    path = Path(path)
    bundle = torch.load(path, map_location=device, weights_only=False)
    if (not isinstance(bundle, dict) or bundle.get("format") != FORMAT_VERSION
            or "lsnpc_state" not in bundle or "arch" not in bundle):
        raise ValueError(f"{path} is not an LSNPC ckpt bundle "
                         f"(expected format {FORMAT_VERSION} with "
                         f"'lsnpc_state' and 'arch').")
    state = bundle["lsnpc_state"]
    if (any(k.startswith("data_decoder.") for k in state)
            and "data_decoder.mu.weight" not in state):
        raise ValueError(
            f"{path} carries the removed CNN data decoder (ShuffleResDecoder "
            f"keys), so it cannot be rebuilt against the MLP decoder. "
            f"Re-train the cell, or load it with an older code revision."
        )
    if ("noisy_encoder.logvar.weight" in state
            and "noisy_encoder.nu.weight" not in state):
        raise ValueError(
            f"{path} carries the fixed-df posterior (no learned ν head), "
            f"which this revision no longer builds: the df is now ν(x, ŷ). "
            f"Re-train the cell, or load it with an older code revision."
        )
    kw = dict(bundle["arch"]["build_kwargs"])
    x_dim = int(kw.pop("x_dim")); n_classes = int(kw.pop("n_classes"))
    # Older bundles also carry task_type/out_dim (the multilabel arm, since
    # removed); build_lsnpc no longer takes them.
    kw.pop("task_type", None); kw.pop("out_dim", None)
    shim = make_config_shim(kw)
    model = build_lsnpc(shim, x_dim=x_dim, n_classes=n_classes)
    model = model.to(device)
    model.load_state_dict(bundle["lsnpc_state"], strict=True)
    # Re-decide backbone sharing against the *loaded* weights: the two trunk
    # copies are bit-identical in a frozen bundle, but a bundle that carries
    # them fine-tuned apart must not have one forward serve both encoders.
    model.refresh_shared_trunk()
    model.eval()
    return {
        "model": model,
        "arch": bundle.get("arch"),
        "splits": bundle["splits"],
        "run": bundle["run"],
        "val_err_h": bundle.get("val_err_h"),
        "code_version": bundle.get("code_version"),
        "path": path,
    }
