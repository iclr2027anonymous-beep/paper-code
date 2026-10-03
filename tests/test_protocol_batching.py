"""Batching in ``experiments/protocol.py`` must be bit-for-bit transparent.

The module feeds a live sweep, so a batching refactor may not move a single
number. Success criteria:

  1. ``_centroid_dists`` row-chunking equals the single-shot
     ``np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)``
     exactly (``np.array_equal``, same dtype) for chunk sizes 1 / 7 / n-rows.
  2. ``inject_noise_idn`` returns identical labels for chunk sizes 1 / 7 / n
     at a fixed seed -- the chunk only bounds the (chunk, n_classes, D)
     intermediate, it never changes the expression.
  3. The plausibility DataLoader loop reproduces the hand-rolled slice loop
     bit-for-bit. Plausibility is scored under the encoder posterior
     q(ẑ|x,ŷ) (``posterior_log_prob``); the prior-density variant was
     removed, so it is not a code path.
  4. It consumes no global RNG: the hand-rolled loop consumed none, and the
     next (unseeded) call samples corrected latents from that same stream.
  5. The P3' robustness flips are one torch draw on the model's device, off a
     private generator, and the K draws ride in the row batch (one sweep, not
     K sweeps); the global NumPy stream is no longer read at all.
"""
import numpy as np
import pytest
import torch

from experiments import protocol
from models.lsnpc import LSNPC

N_ROWS = 23  # 3 classes x 7 + 2 rows -> a short trailing batch at bs=4


def _fixed_labels(n=N_ROWS, n_classes=3):
    return np.tile(np.arange(n_classes), n // n_classes + 1)[:n]


def _fixed_X(n=N_ROWS, d=5, seed=11):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def _centroids(X, y, n_classes):
    return np.stack([X[y == c].mean(axis=0) for c in range(n_classes)])


@pytest.mark.parametrize("chunk", [1, 7, N_ROWS])
def test_centroid_dists_chunked_equals_single_shot(chunk):
    y = _fixed_labels()
    X = _fixed_X()
    centroids = _centroids(X, y, n_classes=3)
    ref = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    got = protocol._centroid_dists(X, centroids, chunk)
    assert got.dtype == ref.dtype
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("chunk", [1, 7, N_ROWS])
def test_inject_noise_idn_labels_invariant_to_chunk(chunk, monkeypatch):
    y = _fixed_labels()
    X = _fixed_X()
    rate, seed = 0.4, 5
    base = protocol.inject_noise_idn(y, rate, seed, 3, X=X)
    assert not np.array_equal(base, y)  # the flip path is actually exercised
    monkeypatch.setattr(protocol, "IDN_DIST_CHUNK", chunk)
    assert np.array_equal(
        protocol.inject_noise_idn(y, rate, seed, 3, X=X), base)


def _tiny_model(n_feat=5, n_classes=3, seed=0):
    torch.manual_seed(seed)
    # latent_dim == x_dim so the ``z_vae is None`` path (z = x) of
    # compute_protocol_scores is the one under test.
    model = LSNPC(x_dim=n_feat, latent_dim=n_feat, n_classes=n_classes,
                  hidden_dim=16, n_blocks=1, dropout=0.0)
    return model.eval()


def _protocol_inputs(n=N_ROWS, n_feat=5, seed=13):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat)).astype(np.float32)
    y_clean = _fixed_labels(n).astype(np.int64)
    y_noisy = y_clean.copy()
    y_noisy[1::3] = (y_noisy[1::3] + 1) % 3
    return X, y_clean, y_noisy


def test_plausibility_matches_the_slice_loop_bit_for_bit():
    n, bs = N_ROWS, 4
    X, y_clean, y_noisy = _protocol_inputs(n)
    model = _tiny_model()
    torch.manual_seed(0)
    out = protocol.compute_protocol_scores(
        model, X, y_clean, y_noisy, "cpu", batch_size=bs, M=2,
        which=(protocol.PLAUSIBILITY_KEY,))

    Xt = torch.as_tensor(X, dtype=torch.float32)
    yh_t = torch.as_tensor(y_noisy.astype(np.int64), dtype=torch.long)
    zc_t = torch.as_tensor(out["z_corr"], dtype=torch.float32)
    ref = torch.zeros(n, dtype=torch.float32)
    for i0 in range(0, n, bs):
        sl = slice(i0, min(i0 + bs, n))
        ref[sl] = model.posterior_log_prob(Xt[sl], yh_t[sl], zc_t[sl])
    assert np.array_equal(out["scores"][protocol.PLAUSIBILITY_KEY],
                          ref.numpy())


def test_plausibility_batching_does_not_shift_the_global_rng():
    X, y_clean, y_noisy = _protocol_inputs()
    model = _tiny_model()

    def rng_state_after(which):
        torch.manual_seed(0)
        protocol.compute_protocol_scores(
            model, X, y_clean, y_noisy, "cpu", batch_size=4, M=2, which=which)
        return torch.get_rng_state()

    without_pl = rng_state_after((protocol.CONF_KEY,))
    with_pl = rng_state_after((protocol.CONF_KEY, protocol.PLAUSIBILITY_KEY))
    assert torch.equal(without_pl, with_pl)


def test_perturbation_noise_is_one_seeded_torch_draw():
    zv = torch.randn(7, 3)
    a = protocol._perturbation_noise(zv, 4, seed=3)
    b = protocol._perturbation_noise(zv, 4, seed=3)
    c = protocol._perturbation_noise(zv, 4, seed=4)
    assert a.shape == (4, 7, 3)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    # The definition, spelled out: 0.1 sigma of the VAE latent, drawn from a
    # generator seeded with ``seed`` on the latent's own device.
    gen = torch.Generator(device=zv.device)
    gen.manual_seed(3)
    ref = zv.unsqueeze(0) + 0.1 * zv.std() * torch.randn(
        (4, *zv.shape), generator=gen)
    assert torch.equal(a, ref)


def test_robustness_flips_are_one_sweep_and_leave_the_numpy_stream_alone():
    """P3' drew its K flips one at a time with ``np.random.randn``, each
    followed by a whole-slice pass over the correction. The flips are now a
    single torch draw and the K draws ride in the row batch, so the correction
    is swept once for the ensemble rather than once per flip."""
    n, bs, K = N_ROWS, 4, 3
    X, y_clean, y_noisy = _protocol_inputs(n)
    model = _tiny_model()

    calls = {"n": 0}
    real = model.corrected_conditioning

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    model.corrected_conditioning = counting
    np.random.seed(99)
    before = np.random.get_state()
    out = protocol.compute_protocol_scores(
        model, X, y_clean, y_noisy, "cpu", batch_size=bs, M=2,
        which=(protocol.ROBUSTNESS_KEY,), k_perturb=K)
    after = np.random.get_state()

    nr = out["scores"][protocol.ROBUSTNESS_KEY]
    assert nr.shape == (n,)
    assert (nr >= 0).all() and (nr <= 1).all()
    assert np.allclose(nr * K, np.round(nr * K), atol=1e-12)
    # Main inference + the whole K-perturbation ensemble. The ensemble is ONE
    # pass (each step carrying all K draws of ``bs // K`` latent rows), so it
    # costs the same as one minimality grid point -- not K of them.
    per_sweep = -(-n // max(1, bs // K))
    assert calls["n"] == -(-n // bs) + per_sweep
    assert before[2] == after[2] and np.array_equal(before[1], after[1])


def test_perturbed_latent_labels_pairs_draws_rows_and_slots(monkeypatch):
    """The K draws fold into the row batch, so ``xb.repeat(K)`` and the
    ``(K, B, D) -> (K*B, D)`` reshape must both be draw-major and the
    ``(K*B,) -> (K, B)`` write-back must undo exactly that. Echo the latent
    through ``corrected_conditioning`` and the label at ``[k, i]`` must be the
    argmax of the k-th draw AT row i: any transposition moves a label."""
    n, n_feat, n_cls, bs, K = N_ROWS, 3, 3, 8, 4
    X, _, y_noisy = _protocol_inputs(n, n_feat)
    model = _tiny_model(n_feat, n_cls)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yh_t = torch.as_tensor(y_noisy.astype(np.int64), dtype=torch.long)
    zv_pert = protocol._perturbation_noise(Xt, K, seed=3)

    # Sampling is identity, so the latent reaching the decoder is the draw.
    monkeypatch.setattr(model, "sample_corrected_latent",
                        lambda x, z, y, M=5, x_pool=None: z)
    monkeypatch.setattr(model, "corrected_conditioning",
                        lambda x, z0, x_pool=None: z0[:, :n_cls])
    got = protocol._perturbed_latent_labels(
        model, Xt, yh_t, zv_pert, x_pool=None, M=2, batch_size=bs, device="cpu")
    assert torch.equal(got, zv_pert[:, :, :n_cls].argmax(dim=-1))


def test_perturbed_latent_labels_feeds_matching_rows_and_pool(monkeypatch):
    """Every draw of an latent row must be paired with THAT row's ``x`` and
    with THAT row's pooled features, in the same draw-major order."""
    n, n_feat, n_cls, bs, K = N_ROWS, 5, 3, 8, 4
    X, _, y_noisy = _protocol_inputs(n, n_feat)
    model = _tiny_model(n_feat, n_cls)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yh_t = torch.as_tensor(y_noisy.astype(np.int64), dtype=torch.long)
    zv_pert = protocol._perturbation_noise(Xt, K, seed=3)
    x_pool = torch.arange(n, dtype=torch.float32)[:, None]  # row-identity marks

    seen = []

    def spy(x, z, y, M=5, x_pool=None):
        seen.append((x.clone(), x_pool.clone()))
        return z

    monkeypatch.setattr(model, "sample_corrected_latent", spy)
    monkeypatch.setattr(model, "corrected_conditioning",
                        lambda x, z0, x_pool=None: z0)
    protocol._perturbed_latent_labels(model, Xt, yh_t, zv_pert, x_pool=x_pool,
                                      M=2, batch_size=bs, device="cpu")

    rows = max(1, bs // K)
    assert len(seen) == -(-n // rows)
    for (x_seen, pool_seen), i0 in zip(seen, range(0, n, rows)):
        b = min(rows, n - i0)
        assert torch.equal(x_seen, Xt[i0:i0 + b].repeat(K, 1))
        assert torch.equal(pool_seen, x_pool[i0:i0 + b].repeat(K, 1))


class _ShardingAccel:
    """An ``Accelerator`` stand-in that shards prepared loaders, as a
    multi-process real one does."""

    def __init__(self, device, num_processes=2, process_index=0):
        self.device = device
        self.num_processes = num_processes
        self.process_index = process_index

    def autocast(self):
        from contextlib import nullcontext
        return nullcontext()

    def prepare(self, loader):
        from accelerate.data_loader import prepare_data_loader
        return prepare_data_loader(loader, num_processes=self.num_processes,
                                   process_index=self.process_index,
                                   put_on_device=False)


def test_batched_infer_ignores_accel_sharding():
    """``_batched_lsnpc_infer`` reassembles batches into the caller's row
    order, so it must NOT hand its loader to ``accel.prepare``: under
    ``num_processes > 1`` that shards a map-style loader, and the function
    would then return each rank's slice of the rows as if it were all of them.
    ``accel`` stays for autocast only."""
    from experiments.data import _batched_lsnpc_infer
    from utils.batching import tensor_loader

    n, n_feat, bs = 23, 5, 4
    X, _, y_noisy = _protocol_inputs(n, n_feat)
    model = _tiny_model(n_feat, 3)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yh_t = torch.as_tensor(y_noisy.astype(np.int64), dtype=torch.long)

    # The premise of the test: this stand-in really does shard, so the
    # assertion below is about the fix and not about a no-op prepare().
    accel = _ShardingAccel("cpu")
    plain = tensor_loader(Xt, Xt, yh_t, batch_size=bs)
    sharded_rows = sum(b[0].shape[0] for b in accel.prepare(plain))
    assert sharded_rows < n

    # Seeded per call: the correction path samples, and each ``iter()`` over a
    # loader also draws a base seed, so the two runs must start level.
    torch.manual_seed(0)
    with_sharding_accel = _batched_lsnpc_infer(
        model, Xt, Xt, yh_t, bs, "cpu", 2, accel=accel)
    torch.manual_seed(0)
    without_accel = _batched_lsnpc_infer(model, Xt, Xt, yh_t, bs, "cpu", 2)
    z_a, c_a = with_sharding_accel
    z_b, c_b = without_accel
    assert z_a.shape[0] == n and c_a.shape[0] == n
    assert torch.equal(z_a, z_b) and torch.equal(c_a, c_b)
