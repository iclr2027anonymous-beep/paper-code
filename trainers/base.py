"""Trainer base class with epoch loop (`train_model`, `train_one_epoch`)
and Accelerate integration (`use_accel` / `mixed_precision="bf16"`).

Subclasses define `train_one_epoch` (per-batch forward+loss+step) and
`predict`. `train(X, y, ...)` builds model+opt+dl+scheduler then calls
`self.train_model(n_epochs, train_loader)`.

`_NullAccel` is a no-op Accelerator stand-in so every `train_one_epoch`
can call `accel.prepare(...)` / `accel.backward(loss)` unconditionally.
`_make_scheduler` / `_clip_grad` provide cosine LR + gradient clipping.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Protocol

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from utils.batching import _DeviceLoader

log = logging.getLogger(__name__)


class _NullAccel:
    """Drop-in stand-in for `accelerate.Accelerator` when Accelerate is not used.

    Lets `train()` code call `accel.prepare(...)`, `accel.backward(loss)`,
    `accel.unwrap_model(model)`, `accel.gather(t)` unconditionally —
    the no-accel path simply does the obvious single-device thing.
    """

    is_main_process = True
    num_processes = 1
    device = None  # set by Trainer.__init__ to self.device

    def __init__(self, device: str):
        self.device = device

    def prepare(self, *objs: Any) -> Any:
        # Loaders are wrapped, not moved, mirroring ``DataLoaderShard``; other
        # torch objects move to `device` and everything else passes through.
        out = []
        for o in objs:
            if isinstance(o, DataLoader):
                o = _DeviceLoader(o, self.device)
            elif hasattr(o, "to"):
                if isinstance(o, (torch.Tensor, nn.Module)):
                    o = o.to(self.device)
            out.append(o)
        return tuple(out) if len(out) > 1 else (out[0] if out else None)

    def autocast(self) -> Any:
        """No-op AMP context: the shim never applies mixed precision."""
        return nullcontext()

    def backward(self, loss: Any) -> None:
        loss.backward()

    def unwrap_model(self, model: Any) -> Any:
        return model

    def gather(self, tensor: Any) -> Any:
        return tensor

    def print(self, *args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)

    def clip_grad_norm_(self, params: Any, max_norm: float) -> float:
        """`nn.utils.clip_grad_norm_` on the (possibly wrapped) model's parameters."""
        return float(nn.utils.clip_grad_norm_(params, max_norm))


class AccelLike(Protocol):
    """The `accelerate` surface the trainers use.

    Satisfied by both `Accelerator` and the `_NullAccel` shim.
    """

    device: Any
    is_main_process: bool

    def prepare(self, *objs: Any) -> Any: ...
    def autocast(self) -> Any: ...
    def backward(self, loss: Any) -> None: ...
    def unwrap_model(self, model: Any) -> Any: ...
    def gather(self, tensor: Any) -> Any: ...
    def clip_grad_norm_(self, params: Any, max_norm: float) -> Any: ...


def _build_accel(device: str, accel: Any, use_accel: bool,
                 mixed_precision: str | None) -> AccelLike:
    """Return an `Accelerator` (or a `_NullAccel` shim) per the request."""
    if accel is not None:
        return accel

    if not use_accel:
        return _NullAccel(device)

    kwargs: dict[str, Any] = {}
    if mixed_precision is not None:
        kwargs["mixed_precision"] = mixed_precision
    return Accelerator(**kwargs)


_MP_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "no": torch.float32,
}


def process_dtype(accel: Any) -> torch.dtype:
    """Return the work dtype dictated by the accelerator's mixed-precision mode.

    Reads ``accel.state.mixed_precision`` and maps it to a PyTorch dtype
    ("bf16" → ``torch.bfloat16``, "fp16" → ``torch.float16``, else ``torch.float32``).
    ``_NullAccel`` does not have a ``state`` — we fall back to ``torch.float32``.
    """
    # Explicit hasattr guard instead of try/except AttributeError: an
    # AttributeError raised *inside* accel.state must propagate, not be
    # silently read as "no mixed precision".
    if not hasattr(accel, "state"):
        return torch.float32
    mp: str = accel.state.mixed_precision  # type: ignore[union-attr]
    return _MP_MAP.get(mp, torch.float32)


class Trainer:
    """Base class for all trainers (torch and sklearn).

    Pattern:
        trainer = MyTrainer(model, config, device, use_accel=True).train(*args)
        preds = trainer.predict(*args)
        trainer.save(path)
        trainer2 = MyTrainer.load(path, device)

    `model` is expected to expose a `predict()` method (alias of the
    model's native inference method — `sample`, `generate`, `log_prob`,
    etc.). The trainer owns the optimizer, dataloader, logging, and
    checkpointing.

    Args:
        model: the torch.nn.Module (or sklearn estimator) to train.
        config: an object with training hyperparameters (e.g. `ExperimentConfig`).
        device: target device string ("cuda", "cpu", "cuda:0", ...).
        accel: optional pre-built `accelerate.Accelerator`. Takes
            precedence over `use_accel`.
        use_accel: if True and `accel is None`, build a fresh
            `Accelerator()` (installs DDP wrapping, mixed precision, etc.).
            If False, `self.accel` becomes a `_NullAccel` shim so `train()`
            code can be written once.
        mixed_precision: "bf16", "fp16", or "no". Forwarded to
            `Accelerator(mixed_precision=...)` when `use_accel=True`.
            Ignored when `accel` is supplied directly.
    """

    def __init__(
        self,
        model: Any,
        config: Any,
        device: str,
        accel: Any = None,
        use_accel: bool = False,
        mixed_precision: str | None = None,
        max_grad_norm: float | None = 1.0,
        lr_schedule: str | None = "cosine",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.use_accel = use_accel or (accel is not None)
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm
        self.lr_schedule = lr_schedule
        self.accel = _build_accel(device, accel, use_accel, mixed_precision)
        self.process_dtype: torch.dtype = process_dtype(self.accel)
        # Make `self.accel.device` consistent (Accelerator may pick a
        # different device than the caller requested under DDP).
        if self.accel.device is not None:
            self.device = str(self.accel.device)
        self.arch_meta: dict = {}

        # Per-training-run state. Populated by `train_model` (or by a
        # subclass's `train(X, y, ...)` before delegating to `train_model`).
        self.optimizer: Any = None
        self.lr_scheduler: Any = None
        self.train_loader: Any = None
        self.val_loader: Any = None
        self.test_loader: Any = None
        # `_current_epoch` is set by `train_model` at the top of each epoch
        # so `train_one_epoch` (and subclass hooks) can reference it
        # (e.g. an ε-annealed schedule uses it to anneal a prior-mixing term).
        self._current_epoch: int = 0

    # ── Epoch loop (lsnpc pattern) ────────────────────────────────────

    def train_model(
        self,
        n_epochs: int,
        train_loader: Any,
        val_loader: Any = None,
        test_loader: Any = None,
        verbose: bool = False,
    ) -> None:
        """Canonical epoch loop with early stopping support.

        Subclasses should call this from their public `train(X, y, ...)`
        method after building the model / optimizer / dataloader / scheduler.

        Early stopping: override `eval_and_save` to set
        `self._early_stopped = True` when a stopping condition is met.
        This loop checks the flag after each epoch.

        For each epoch:
          1. `self._current_epoch = epoch`
          2. `loss = self.train_one_epoch(train_loader)`
          3. `self.eval_and_save(epoch, n_epochs, val_loader, test_loader, verbose)`
          4. (Per-epoch scheduler step is the subclass's responsibility
             if it wants per-epoch cadence; per-step stepping happens
             inside `train_one_epoch`.)
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self._early_stopped = False
        # Expose the total epoch count so progress bars / subclasses can
        # render "epoch i / n_epochs" instead of a bare epoch index.
        self._n_epochs = n_epochs

        for epoch in range(n_epochs):
            self._current_epoch = epoch
            self.train_mode()
            mean_loss = self.train_one_epoch(train_loader)
            log_interval = getattr(self, 'log_interval', 1)
            if verbose and ((epoch + 1) % log_interval == 0 or epoch == 0 or epoch == n_epochs - 1):
                log.info(f"  epoch {epoch + 1}/{n_epochs}: loss={mean_loss:.4f}")

            self.eval_and_save(epoch, n_epochs, val_loader, test_loader, verbose)
            
            if self._early_stopped:
                log.info(f"  Early stopped at epoch {epoch + 1}/{n_epochs} (best val_loss={getattr(self, '_best_val_loss', float('nan')):.4f})")
                break

    def train_one_epoch(self, train_loader: Any, **kwargs) -> float:
        """Forward+loss+backward+step for every batch in `train_loader`.

        Subclasses must override. Returns the mean loss across batches.
        The base-class contract: callers may assume `self._current_epoch`
        is set, `self.optimizer` is the prepared optimizer, and
        `self.accel` is the (possibly null) Accelerator.
        """
        raise NotImplementedError("Subclasses must implement train_one_epoch()")

    def eval_and_save(
        self,
        epoch: int,
        n_epochs: int,
        val_loader: Any = None,
        test_loader: Any = None,
        verbose: bool = False,
    ) -> None:
        """Per-epoch evaluation + checkpoint hook. Default is a no-op.

        Subclasses can override to compute validation metrics and save
        checkpoints on best metric (see lsnpc `Trainer.eval_and_save`).
        """
        pass

    # ── Public train API (subclass entry point) ──────────────────────

    def train(self, *args, **kwargs) -> "Trainer":
        """Public training entry point. Subclasses override to build the
        model + optimizer + dataloader + scheduler, then call
        `self.train_model(...)`. Returns self."""
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, *args, **kwargs) -> Any:
        """Delegate to `self.model.predict(...)`."""
        return self.model.predict(*args, **kwargs)

    # ── Model-mode toggles (lsnpc convention) ─────────────────────────

    def train_mode(self) -> None:
        """Set `self.model` to training mode."""
        if self.model is not None:
            self.model.train()

    def eval_mode(self) -> None:
        """Set `self.model` to eval mode."""
        if self.model is not None:
            self.model.eval()

    # ── Checkpointing ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save `{"state_dict", "arch"}` bundle to `path`.

        Uses `accel.unwrap_model` so the saved state_dict is the bare
        model's, not the DDP-wrapped one.
        """
        model = self.accel.unwrap_model(self.model)
        torch.save(
            {"state_dict": model.state_dict(), "arch": self.arch_meta},
            path,
        )

    @classmethod
    def load(cls, path: str, device: str, **ctor_kwargs: Any) -> "Trainer":
        """Load a trainer from a saved bundle. Subclasses must implement `_build_model`."""
        bundle = torch.load(path, map_location=device, weights_only=True)
        model = cls._build_model(bundle["arch"])
        model.load_state_dict(bundle["state_dict"])
        model = model.to(device).eval()
        return cls(model, device=device, **ctor_kwargs)

    @classmethod
    def _build_model(cls, arch: dict) -> Any:
        """Reconstruct a fresh model from the saved arch bundle. Subclasses override."""
        raise NotImplementedError

    # ── Cosine LR scheduler ───────────────────────────────────────────

    def _make_scheduler(self, opt: Any, total_steps: int) -> Any:
        """Build the cosine LR scheduler.

        Cosine decay to 0 over `total_steps`, via HF
        `transformers.get_cosine_schedule_with_warmup` (matches HF Trainer
        behaviour). No warmup: every training entry point runs with
        `warmup_steps = 0`, so the warmup variants were removed.

        Returns `None` if `self.lr_schedule is None` (no scheduling).
        Subclasses should call `scheduler.step()` once per optimizer step.
        """
        if self.lr_schedule is None:
            return None
        total_steps = max(1, int(total_steps))
        return get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=0, num_training_steps=total_steps,
        )

    # ── Gradient clipping ─────────────────────────────────────────────

    def _clip_grad(self, model: Any, max_norm: float | None = None) -> float | None:
        """Clip gradients of `model` using `accel.clip_grad_norm_` so it works under DDP.
        """ 
        limit = self.max_grad_norm if max_norm is None else max_norm
        if limit is None:
            return None  # clipping disabled
        norm = self.accel.clip_grad_norm_(model.parameters(), float(limit))
        # `Accelerator.clip_grad_norm_` returns a tensor; `_NullAccel`
        # already returns a float. Normalise so callers see a float.
        return float(norm) if hasattr(norm, "item") else norm
