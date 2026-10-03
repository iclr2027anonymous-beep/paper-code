"""Shared Set-E helpers: feature views, caches, split metadata.

Set E (end-to-end retrain on corrected streams, the experiment protocol
§Set E) is split into three stages that share data views:

  1. fit  — corrector fit (existing entry points, reused ckpts);
  2. emit — scripts/emit_lsnpc_stream.py writes per-row emissions over the
            retrain pool (+ held-out eval/val rows for the label-level sign
            check);
  3. retrain — scripts/retrain_e_streams.py fits the downstream heads
            (configs a-f) over FROZEN features of the same modality.

This module owns the data-view construction so stage 2 and 3 cannot drift:

  * text views (NoisyAG worst/med/best): cached mpnet embeddings + real
    crowd noisy labels + the per-row clean oracle (ground_truth);
  * image views (CIFAR-10N worse / CIFAR-100N fine / dopanim): the raw pool
    pickles, the published human label sets (no injection), normalized to
    the CIFAR constants every pixel pipeline uses;
  * feature caching: RN50-embedding features of pool + clean test are
    computed once per (dataset, view, resolution) and cached under
    results/set_e/_features/ so emission and retrain never re-featurize;
  * split bookkeeping: bundles carry their own splits; the only RNG replay
    needed is the dopanim clean-test carve (2000 corrector clean-set rows /
    2500 final test rows from the 4500-row clean test pool).
"""
from __future__ import annotations

from utils.paths import project_path

import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.protocol import inject_noise, inject_noise_idn

from experiments.protocol import inject_noise, inject_noise_idn

from data_process.cifar10n import _MEAN as _MEAN_C10
from data_process.cifar10n import _STD as _STD_C10
from data_process.cifar10n import _load_cifar10n, _load_cifar_pickles
from data_process.cifar100n import _MEAN as _MEAN_C100
from data_process.cifar100n import _STD as _STD_C100
from data_process.cifar100n import _load_cifar100_pickles, _load_cifar100n
from models.rn50_features import RN50FeatureEncoder
from trainers.lsnpc import LSNPCTrainer
from utils.batching import place_loader_local, tensor_loader
from trainers.plausibility_vae import ckpt_path
from trainers.plausibility_vae import get_or_train as get_or_train_vae

DATA_ROOT = Path(project_path('data'))
RESULTS_ROOT = Path(project_path('results'))
SET_E = RESULTS_ROOT / "set_e"
FEAT_DIR = SET_E / "_features"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

TEXT_TESTBEDS = ("noisyag_best", "noisyag_med", "noisyag_worst")
# Fit-time split sizes for the injected text sets (driver passes the same).
EVAL_SLICE_BY_TESTBED = {"ag_news": 2000, "sst2": 2000,
                         "medical_abstracts": 2000}
VAL_SIZE = 2500
CLEAN_SET = 0
IMAGE_TESTBEDS = ("cifar10n", "cifar100n", "dopanim", "animal10n")


def cell_tag(modality: str, testbed: str, corr_seed: int) -> str:
    """Canonical Set-E cell directory name (run-tag rule: every dial)."""
    return f"e_{modality}_{testbed}_cs{corr_seed}"


def cell_dir(tag: str) -> Path:
    d = SET_E / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── text ────────────────────────────────────────────────────────────────


def load_text_view(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X (N,768) float32, y_clean, y_noisy) for a NoisyAG variant."""
    if dataset not in TEXT_TESTBEDS:
        raise ValueError(f"{dataset}: Set-E text testbeds are {TEXT_TESTBEDS}")
    c = pickle.load(open(DATA_ROOT / "noisyag_news" / "embeddings.pkl", "rb"))
    variant = dataset.split("_", 1)[1]
    return (np.asarray(c["x"], dtype=np.float32),
            np.asarray(c["ground_truth"], dtype=np.int64),
            np.asarray(c[f"noisy_{variant}"], dtype=np.int64))


# ── injected-noise text testbeds (ag_news / sst2 / medical_abstracts) ────
# The corrector fit (scripts/text_lsnpc_sst2.py) injects noise ONCE over the
# full label array with seed = the corrector's seed, then carves splits by a
# seeded permutation with split_seed = seed. The emitter and the retrain MUST
# replay that exact stream, so both go through here with the same (dataset,
# rate, type, seed) the fit used — read from the bundle's run metadata.

INJECTED_TEXT_TESTBEDS = ("ag_news", "sst2", "medical_abstracts")
INJECTED_TEXT_CLASSES = {"ag_news": 4, "sst2": 2, "medical_abstracts": 5}


def injected_text_stream(dataset: str, noise_rate: float, noise_type: str,
                         seed: int, n_classes: int):
    """(X, y_clean, y_noisy, eval_idx, val_idx, cs_idx, tr_idx) for an
    injected-noise text testbed — a byte-exact replay of the fit-time
    construction in scripts/text_lsnpc_sst2.py (load_embeddings, the single
    injection over the full label array, and the seeded split carves).

    Any divergence here silently misaligns the emitted label stream from the
    rows the corrector was trained on, so the replay is deliberately literal.
    """
    if dataset not in INJECTED_TEXT_TESTBEDS:
        raise ValueError(
            f"{dataset}: injected text testbeds are {INJECTED_TEXT_TESTBEDS}")
    if dataset == "sst2":
        c = pickle.load(open(DATA_ROOT / "sst2" / "embeddings.pkl", "rb"))
        X, yc = c["train_x"], c["train_y"]
    elif dataset == "ag_news":
        tr = pickle.load(open(DATA_ROOT / "ag_news" / "train_emb.pkl", "rb"))
        X, yc = tr["x"], tr["y"]
    else:
        c = pickle.load(open(DATA_ROOT / "medical_abstracts" / "embeddings.pkl",
                             "rb"))
        X, yc = c["x"], c["y"]
    X = np.asarray(X, dtype=np.float32)
    yc = np.asarray(yc, dtype=np.int64).ravel()
    if noise_type == "idn":
        yn_all = inject_noise_idn(yc, noise_rate, seed, n_classes, X=X)
    else:
        yn_all = inject_noise(yc, noise_rate, seed, n_classes)
    # Split carves — identical order of rng draws as the fit path.
    n = len(X)
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(n, size=EVAL_SLICE_BY_TESTBED[dataset],
                          replace=False)
    rest = np.setdiff1d(np.arange(n), eval_idx)
    order = rng.permutation(rest)
    val_idx = order[:VAL_SIZE]
    pool = order[VAL_SIZE:]
    cs_idx = pool[:CLEAN_SET]
    tr_idx = pool[CLEAN_SET:]
    return X, yc, yn_all, eval_idx, val_idx, cs_idx, tr_idx


def load_eurosat_view(noise_rate: float, noise_type: str):
    """(X, y_noisy, y_clean, X_test, y_test) for EuroSAT at a given setting.

    Thin wrapper over the pair loader with the SAME fixed random_state=42 the
    fit path passes, so the injected label array is row-aligned with the fit.
    """
    from data_process.eurosat import load_eurosat_pair
    return load_eurosat_pair(str(DATA_ROOT), noise_rate=float(noise_rate),
                             noise_type=noise_type, random_state=42)


def load_text_test() -> tuple[np.ndarray, np.ndarray]:
    """Official clean AG-News test embeddings + labels (7600 rows)."""
    c = pickle.load(open(DATA_ROOT / "ag_news" / "test_emb.pkl", "rb"))
    return np.asarray(c["x"], dtype=np.float32), np.asarray(c["y"])


# ── image pools (row-aligned normalized views) ──────────────────────────


def _normalize(x_uint8: np.ndarray, mean, std) -> np.ndarray:
    """(N,3,H,W) uint8 -> float32 CIFAR-normalized NCHW."""
    x = torch.from_numpy(np.asarray(x_uint8)).float() / 255.0
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    return ((x - mean_t) / std_t).numpy().astype(np.float32)


def load_cifar_view(dataset: str, view: str):
    """(X_norm (N,3,H,W) float32, y_noisy, y_clean, X_test, y_test).

    Row-aligned and order-IDENTICAL to scripts.image_lsnpc.load_image_pair:
    CIFAR-N images go through build_image_splits' shared 85/15 permutation
    (random_state=42), so the "pool" the Set A/C correctors were fit on is
    the 42,500-row train split in its internal row order (NOT the raw 50k
    pickle order). y_noisy / y_clean come from the noisy and clean label
    views of that same permutation; the official CIFAR test images and
    labels are normalized with the per-dataset constants (test labels are
    always the clean benchmark labels).
    """
    if dataset == "cifar10n":
        load, mean_ds, std_ds = _load_cifar10n, _MEAN_C10, _STD_C10
    elif dataset == "cifar100n":
        load, mean_ds, std_ds = _load_cifar100n, _MEAN_C100, _STD_C100
    else:
        raise ValueError(dataset)
    tr_n, _va_n, _te_n, _ = load(str(DATA_ROOT), noise_rate=0.0,
                                 noise_type=view, random_state=42)
    tr_c, _va_c, te_c, _ = load(str(DATA_ROOT), noise_rate=0.0,
                                noise_type="clean", random_state=42)
    Xn = np.asarray(tr_n.features)
    Xc = np.asarray(tr_c.features)
    assert Xn.shape == Xc.shape and (Xn == Xc).all(), \
        "noisy/clean CIFAR views not row-aligned"
    # Official test images from the raw pickles, normalized identically.
    load_pickles = (_load_cifar_pickles if dataset == "cifar10n"
                    else _load_cifar100_pickles)
    _Xtr, _yc, X_test_raw, y_test_raw = load_pickles(str(DATA_ROOT))
    Xt = _normalize(X_test_raw, mean_ds, std_ds)
    y_test = np.asarray(te_c.targets).ravel()
    assert (np.asarray(y_test_raw).ravel() == y_test).all(), \
        "test labels disagree between raw pickle and loader view"
    return (Xn.astype(np.float32),
            np.asarray(tr_n.targets).ravel().astype(np.int64),
            np.asarray(tr_c.targets).ravel().astype(np.int64),
            Xt.astype(np.float32),
            np.asarray(y_test).astype(np.int64))


def load_dopanim_view(size: int | None = None):
    """dopanim annotated pool + clean test pool.

    Returns (X_pool_norm, y_noisy, y_clean, X_test_pool_norm, y_test_pool).
    y_clean here is the class-dir verified label the loader stores; per the
    pre-registered E2 decision the dopanim CLEAN-ORACLE CEILING is NOT run
    (labels on the disputed pool are not trusted as a train ceiling) — the
    clean test pool is the only guaranteed-clean evaluation source.
    """
    p = DATA_ROOT / "dopanim" / "train_batch"
    if not p.exists():
        raise FileNotFoundError("run scripts/dopanim_prep.py first")
    d = pickle.load(open(p, "rb"))
    Xp = _normalize(d["data"], CIFAR10_MEAN, CIFAR10_STD)
    y_noisy = np.asarray(d["noisy_labels"]).astype(np.int64)
    y_clean = np.asarray(d["labels"]).astype(np.int64)
    t = pickle.load(open(DATA_ROOT / "dopanim" / "test_batch", "rb"))
    Xt = _normalize(t["data"], CIFAR10_MEAN, CIFAR10_STD)
    yt = np.asarray(t["labels"]).astype(np.int64)
    return Xp, y_noisy, y_clean, Xt, yt


def load_animal10n_view():
    """ANIMAL-10N annotated pool + clean test pool.

    Returns (X_pool_norm, y_noisy, y_clean, X_test_norm, y_test) with
    ``y_clean = None``: the dataset releases no clean training labels, so the
    pool carries no oracle and every clean-referenced diagnostic (transitions,
    mis/succ, clean-referenced AUROCs) is undefined for it. Returning None
    rather than a sentinel array makes the emitter skip those keys instead of
    writing numbers computed against a placeholder.

    The clean test split is the only trusted source; the rows the corrector was
    not given (data_process.animal10n.ANIMAL10N_PIPELINE_ROWS onward) are the
    held-out evaluation pool, disjoint from its clean set and validation.
    """
    from data_process.animal10n import ANIMAL10N_ORACLE_ABSENT, load_animal10n_pair
    X, y_noisy, y_clean, X_test, y_test, _mean, _std = load_animal10n_pair(
        str(DATA_ROOT))
    assert (np.asarray(y_clean) == ANIMAL10N_ORACLE_ABSENT).all(), \
        "animal10n pool must carry the absent-oracle sentinel"
    return X, y_noisy, None, X_test, y_test


def dopanim_test_carve(corr_seed: int, eval_slice: int = 1000,
                       val_size: int = 2500,
                       n_pool: int = 10484, n_test_pool: int = 4500,
                       clean_set: int = 2000,
                       split_seed: int | None = None,
                       allow_large_eval: bool = False
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Replay the image_lsnpc RNG stream to recover the dopanim test carve.

    The corrector run draws eval/val from the annotated pool with
    ``rng = default_rng(split_seed or seed)``, then draws
    ``cs_perm = rng.permutation(4500)`` and takes the first 2000 rows as its
    clean set. The remaining 2500 rows of that same permutation are the
    FINAL TEST carve (pre-registered 2000/2500 split, disclosed in the
    cell json). ``split_seed`` must match the --split-seed the corrector
    fit used (index-fixed across dopanim corrector seeds: all E dopanim
    corrector seeds share the split of seed 42). Returns (clean_set_idx,
    test_idx), both indices into the 4500-row clean test pool.
    """
    ss = split_seed if split_seed is not None else corr_seed
    rng = np.random.default_rng(ss)
    ev_n = (eval_slice if allow_large_eval else min(eval_slice, n_pool // 4))
    eval_idx = rng.choice(n_pool, size=ev_n, replace=False)
    rest = np.setdiff1d(np.arange(n_pool), eval_idx)
    order = rng.permutation(rest)
    n_val = min(val_size, len(order) // 2)
    order = order[n_val:]
    del order  # pool layout irrelevant for the test-pool permutation
    cs_perm = rng.permutation(n_test_pool)
    return cs_perm[:clean_set], cs_perm[clean_set:]


# ── frozen RN50 features (embedding mode) ───────────────────────────────


def _rn50_batch(X_norm_nchw: np.ndarray, device: str,
                batch: int = 256) -> np.ndarray:
    enc = RN50FeatureEncoder().to(device).eval()
    dl = place_loader_local(
        tensor_loader(torch.as_tensor(X_norm_nchw, dtype=torch.float32),
                      batch_size=batch, shuffle=False),
        device)
    outs = []
    with torch.no_grad():
        for (xb,) in dl:
            mu, _ = enc.encode(xb)
            outs.append(mu.float().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def image_feature_cache_path(dataset: str, view: str, split: str,
                             size: int) -> Path:
    """Cached frozen-RN50 features: results/set_e/_features/<key>.npy."""
    key = f"feat_{dataset}_{view}_rn50_{split}_sz{size}"
    return FEAT_DIR / f"{key}.npy"


def rn50_features_cached(X_norm_nchw: np.ndarray, dataset: str, view: str,
                         split: str, device: str,
                         batch: int = 256) -> np.ndarray:
    """Frozen RN50 features of an image array, cached by (dataset, view,
    split, resolution). ``split`` distinguishes the pool (e.g. 'pool') from
    the clean test (e.g. 'test') so the same encoder never re-runs."""
    size = int(X_norm_nchw.shape[-1])
    path = image_feature_cache_path(dataset, view, split, size)
    if path.exists():
        return np.load(path)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    f = _rn50_batch(X_norm_nchw, device, batch)
    np.save(path, f)
    return f


# ── Set F gate / admission helpers (shared by the e6_* pool studies) ─────
#
# These lived as near-identical private copies in e6_pool_expansion.py,
# e6_cifar_pool_expansion.py and e6_destructive_pool_expansion.py. The
# copies had already drifted (_calibrate_gate and _gate_curve each existed
# in two spellings), so they live here where the three studies cannot
# diverge again.

GATE_TARGETS = (0.95, 0.90, 0.85, 0.80)


def random_mask(n: int, k: int, stream: int) -> np.ndarray:
    """Size-``k`` uniform random boolean mask, deterministic in ``stream``."""
    m = np.zeros(n, dtype=bool)
    if k > 0:
        r = np.random.default_rng(stream)
        m[r.choice(n, size=min(k, n), replace=False)] = True
    return m


def calibrate_gate(minimality_val, succ_val, target_prec=0.95):
    """Calibrate the M1 admission threshold on the validation split.

    Returns (threshold, coverage, precision): the minimality threshold that
    maximizes coverage subject to precision >= target_prec on the val split,
    the resulting coverage (fraction of candidates admitted), and the
    achieved precision. Ties are handled by scanning all distinct score
    values (score >= t admits).
    """
    scores = np.asarray(minimality_val, dtype=np.float64)
    valid = np.asarray(succ_val, dtype=bool)
    best_t, best_cov, best_prec = None, 0.0, 0.0
    for t in np.unique(scores):
        admit = scores >= t
        n_admit = int(admit.sum())
        if n_admit == 0:
            continue
        prec = float(valid[admit].mean())
        cov = n_admit / len(scores)
        if prec >= target_prec and cov > best_cov:
            best_t, best_cov, best_prec = float(t), float(cov), prec
    return best_t, best_cov, best_prec


def gate_curve(m1_val, succ_val, targets=GATE_TARGETS):
    """Reachability at each declared precision target, plus the best point.

    The declared F protocol is >=95%; when no threshold reaches it the study
    falls back to the highest target that is reachable and the curve is
    recorded, so an unreachable bar cannot silently empty the gated
    configuration.
    """
    curve = []
    for tgt in targets:
        t, cov, prec = calibrate_gate(m1_val, succ_val, target_prec=tgt)
        curve.append({"target": float(tgt), "threshold": t,
                      "coverage": float(cov) if t is not None else None,
                      "precision": float(prec) if t is not None else None})
    scores = np.asarray(m1_val, dtype=np.float64)
    valid = np.asarray(succ_val, dtype=bool)
    best = {"precision": 0.0, "coverage": 0.0, "threshold": None}
    for t in np.unique(scores):
        admit = scores >= t
        if not admit.any():
            continue
        prec, cov = float(valid[admit].mean()), float(admit.mean())
        if (prec, cov) > (best["precision"], best["coverage"]):
            best = {"precision": prec, "coverage": cov, "threshold": float(t)}
    return curve, best


def admission_configurations(m1_pool, gated_admit, n_gated, seed, coverages,
                             config_names=None):
    """Admission configurations: calibrated volume, then the ``coverage`` sweep.

    ``c@<coverage>`` keeps the top-coverage fraction by the gate axis; ``f@<coverage>`` keeps a uniform
    random sample of the same size from an independent stream. Their
    difference is what the ranking buys at that volume. ``config_names``
    restricts the two volume-matched-random baselines (the destructive study
    gates them behind CLI flags); ``None`` emits both.
    """
    n = len(m1_pool)
    out = [("V0", None), ("Gated", gated_admit)]
    for key in (("Unguarded", "Random") if config_names is None
                else config_names):
        if key in ("Unguarded", "Random"):
            out.append((key, random_mask(n, n_gated,
                                         seed if key == "Unguarded"
                                         else seed + 10_000)))
    rank = np.argsort(-np.asarray(m1_pool, dtype=np.float64), kind="stable")
    for cov in coverages:
        n_cov = max(1, int(np.ceil(cov * n)))
        m = np.zeros(n, dtype=bool)
        m[rank[:n_cov]] = True
        out.append((f"c@{cov}", m))
        out.append((f"f@{cov}", random_mask(n, n_cov, seed + 1_000_000
                                          + int(cov * 1000))))
    return out


def train_round(vae, X_train, y_noisy, X_clean, y_clean, y_clean_noisy,
                X_val, y_val_clean, y_val_noisy, n_classes, config, accel,
                device) -> LSNPCTrainer:
    """One LSNPC training round. The clean-set rows carry BOTH their clean
    label (y_clean, supervision) and their noisy label (y_clean_noisy, the
    correction-path yhat) via config._clean_set_yhat."""
    config._clean_set_yhat = np.asarray(y_clean_noisy).astype(np.int64)
    trainer = LSNPCTrainer(model=None, config=config, device=device, accel=accel)
    trainer.train(
        vae=vae, X_train=X_train, y_train_noisy=y_noisy,
        X_clean_set=X_clean, y_clean_set=y_clean, n_classes=n_classes,
        X_val=X_val, y_val_clean=y_val_clean, y_val_noisy=y_val_noisy,
        use_semi=True,
    )
    return trainer


def dump_pool_context(tag: str, seed: int, *, corr, m1, succ, y_clean, y_noisy,
                      configs, **idx) -> Path:
    """Persist what the downstream stage needs, so it never refits a corrector.

    The Set F downstream comparison trains a classifier on the dataset each
    configuration defines. That needs the corrector's labels for the pool, its
    gate scores, and the admission masks -- all of which exist only inside a
    fit. Dumping them makes the downstream stage a cheap, re-runnable pass
    (heads on frozen features) instead of a reason to repeat the fits.
    """
    out = Path("results/set_f/pool_ctx")
    out.mkdir(parents=True, exist_ok=True)
    payload = {"corr": np.asarray(corr), "m1": np.asarray(m1),
               "succ": np.asarray(succ),
               "y_clean": np.asarray(y_clean), "y_noisy": np.asarray(y_noisy)}
    for name, mask in configs:
        payload[f"mask__{name}"] = (np.zeros(len(corr), dtype=bool)
                                    if mask is None else np.asarray(mask))
    for k, v in idx.items():
        if v is not None:
            payload[k] = np.asarray(v)
    p = out / f"{tag}_s{seed}.npz"
    np.savez_compressed(p, **payload)
    print(f"[F] context dumped -> {p}", flush=True)
    return p


def pixel_vae(bundle: dict, X_train_nhwc: np.ndarray, device: str,
               latent_dim: int, size: int):
    """Load the cached conv pixel VAE used by the pixel fit (dataset view).

    Fails loudly when the cache is absent: run the matching image_lsnpc cell
    once so its pixel VAE cache exists (Set C-c dopanim / A-C pixel cells).
    """
    run = bundle["run"]
    config = SimpleNamespace(vae_ckpt=None, batch_size=256, hidden_dim=256,
                             vae_epochs=30, latent_dim=latent_dim)
    key = dict(dataset=run["dataset"], noise_type="symmetric",
               noise_rate=0.0, latent_dim=latent_dim,
               img_size=size)
    path = ckpt_path(**key, ckpt_dir=Path("results/ckpt"))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"pixel emission needs the cached pixel VAE {path} -- run the "
            f"corresponding image_lsnpc cell first (dataset {run['dataset']}, "
            f"sz{size}, latent {latent_dim}).")
    return get_or_train_vae(
        dataset=run["dataset"], noise_type="symmetric", noise_rate=0.0,
        X_train=np.ascontiguousarray(X_train_nhwc), config=config,
        device=device, latent_dim=latent_dim,
        )
