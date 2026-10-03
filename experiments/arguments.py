"""Configuration management for LSNPC label-correction experiments.

Provides a clean dataclass that replaces the monolithic argparse Namespace.
Can be instantiated from command-line args or programmatically.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """All configuration for an LSNPC experiment run."""

    # Dataset
    dataset: str = "cifar100n"
    data_dir: str = "data"

    # Noise
    noise: float = 0.0
    noise_type: str = "symmetric"
    conditioning_mode: str = "posterior"
    # Source of the noisy label yhat the correction conditions on: "clf" uses
    # a pretrained classifier's prediction (paper default); "data" uses the
    # raw noisy labels from the dataset directly and trains no classifier.
    yhat_source: str = "clf"
    # Image-encoder backbone for LSNPC: "conv", "resnet50", "resnet101",
    # "swin", "swin_tiny" or "vit".
    encoder_backbone: str = "conv"
    # Freeze the pretrained image-encoder backbone (heads always train).
    # False = fine-tune the full backbone (heavier/slower, higher memory).
    # The backbone's own norm layers follow this flag: frozen keeps their
    # pretrained BatchNorm statistics, trainable fine-tunes them too.
    freeze_backbone: bool = True
    # With yhat_source="data", evaluate on a held-out slice of the TRAINING set
    # (which carries both noisy and clean labels) instead of the test set.
    # 0 = off (use the test set). Ignored unless yhat_source="data".
    eval_train_slice: int = 0
    # With yhat_source="data", cap the pseudo-clean pool (drawn from the full
    # test set) to this many rows. 0 = use all. Useful to keep a heavy encoder
    # (e.g. Swin) within a feasible runtime.
    pseudo_clean_cap: int = 0

    # Test settings
    # Eval slice for the protocol scores; -1 = use every test row. Protocol runs
    # use 100 (per-run CI ~= +-0.095), full-eval runs pass -1 (README).
    n_test: int = 100

    # Output
    output_dir: str = "results"
    seed: int = 42
    protocol_id: str = "exploratory"
    formal_run: bool = False

    # Training
    vae_epochs: int = 10  # paper: 500 (use `--vae-epochs 500` for official runs)
    early_stop_patience: int = 0  # 0 → use n_epochs/10
    batch_size: int = 256  # paper: 64, 128 (classifier)
    image_vae_beta: float = 1.0  # KL β for the image VAE
    vae_augment: bool = True  # batch-vectorized crop/flip/normalize aug for image VAE
    vae_ckpt: str | None = (
        None  # load universal plausibility VAE from this path (skip training)
    )

    # Model architecture
    latent_dim: int = 64
    hidden_dim: int = 64
    n_blocks: int = 3

    # Accelerator
    force_retrain: bool = False  # retained for back-compat; cache is always used now

    # ── v5B: LSNPC training-time label correction (paper §4 / Appendix C) ──
    # Four-system ablation matrix, one value per run:
    #   "no_lsnpc"                 System 1 baseline (no correction)
    #   "corrected_latent_only"    System 2: corrected z_0, keep h(x)
    #   "corrected_condition_only" System 3: VAE z_0, corrected h̃(x,ŷ)
    #   "full_lsnpc"               System 4: full LSNPC
    #   "lsnpc_train_only"         train LSNPC but apply no correction (wiring ablation)
    system: str = "full_lsnpc"
    system_explicit: bool = False  # CLI provenance: was a system flag supplied?
    use_semi: bool = (
        False  # enable semi-supervised LSNPC with η clean-branch on eval set
    )
    lsnpc_beta: float = 1.0  # β: KL coefficient of the closed-form correction loss (paper eq:correction_loss); replaces the former lsnpc_kl_weight
    lsnpc_eta: float | None = (
        None  # η: probability a clean-set batch takes the clean-conditioned semi-supervised pass (None = auto: 0.1 if use_semi else 0.0)
    )

    lsnpc_backbone_lr_scale: float = 1.0  # LR multiplier for the image backbone (1.0 = same as heads)
    lsnpc_loss: str = "correction"  # closed-form correction loss (sole objective)
    lsnpc_loss_explicit: bool = False  # CLI provenance: was --lsnpc-loss supplied?
    clean_set_size: int = -1  # -1 = use all available clean rows (default); 0 = no clean set (unsupervised); >0 = cap
    iw_samples: int = 5  # K importance samples for the IW-ELBO
    lsnpc_lr: float = (
        5e-5  # LSNPC (Stage 1) learning rate; 1e-3 caused seed instability 
    )
    lsnpc_eps: float = (
        1e-6  # AdamW eps for LSNPC (REQUIRED for bf16 gradient noise)
    )
    lsnpc_epochs: int = 30  # LSNPC (Stage 1) training epochs
    lsnpc_focal_gamma: float = (
        0.0  # focal-loss focusing parameter on the label cross-entropies (0 = plain CE)
    )
    # Focal-loss alpha balance of the one-vs-all terms,
    # alpha*y + (1-alpha)*(1-y). None leaves them unweighted (a balanced
    # label set wants that); 0.25 is the detection setting for a rare
    # positive class, and the two are only comparable at equal gamma.
    lsnpc_focal_alpha: float | None = None
    # Label-head structure: "concat" (original: x_feat ⊕ z through one trunk)
    # or "gate" (one trunk per input, mixed by a learned gate; exposes reliance).
    lsnpc_head: str = "concat"
    # Width of the trainable projection the pretrained backbone's pooled
    # feature goes through (RN50: 2048 -> this, via the encoder's mu/logvar
    # heads). Both the noisy encoder and the label head read this embedding.
    lsnpc_embed_dim: int = 128
    # Inputs of the correction map q(z | ẑ, ·): "none" is the paper's
    # q(z | ẑ, x) through the VAE latent, "x" adds the image embedding to the
    # gate, "x_yhat" adds the one-hot noisy label as well, and "yhat" drops x
    # entirely, blending toward a learned label embedding (q(z | ẑ, ŷ)).
    lsnpc_correction_cond: str = "none"
    # Shared label-embedding library: one learnable per-class embedding feeds
    # both the posterior's label slot (q(ẑ|x, E(ŷ))) and the blend target,
    # instead of a raw one-hot in each place.
    lsnpc_shared_yhat_embed: bool = False
    # Correction-map input: "gated" (convex blend of ẑ vs the blend target) or
    # "concat" (a trunk over cat([ẑ, x_feat])).
    lsnpc_correction_input: str = "gated"
    lsnpc_image_data: bool = False  # LSNPC consumes raw pixels, not features
    lsnpc_img_channels: int = 3  # image channels for the image encoder
    lsnpc_img_size: int = 96  # image size for the image encoder
    lsnpc_nu0: float = 2.0  # prior (generative) df for LSNPC Student-t
    iw_infer_samples: int = 5  # M importance samples for inference z_0 estimate

    def system_name(self) -> str:
        """Return the tabular system (single source of truth: ``self.system``)."""
        return self.system

    def validate(self) -> None:
        """Reject ambiguous or scientifically invalid experiment configs."""
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError(
                f"noise must lie in [0, 1] (it is the fraction of labels the "
                f"injectors flip), got {self.noise}. A rate above 1 flips "
                f"every row and silently diverges from the requested rate."
            )
        if self.lsnpc_beta <= 0.0 or self.lsnpc_beta > 4.0:
            raise ValueError(
                "lsnpc_beta (KL weight) must lie in (0, 4]. The original "
                "(0, 1] bound reflected the fractional-posterior reading; "
                "the beta-sweep protocol treats beta as the plain KL "
                "weight and explores up to 4.0 ."
            )
        if self.lsnpc_focal_gamma < 0.0:
            raise ValueError(
                "lsnpc_focal_gamma must be >= 0 (0 disables focal loss and "
                "keeps the cross-entropy baseline)."
            )
        if self.lsnpc_focal_alpha is not None and not (
                0.0 <= self.lsnpc_focal_alpha <= 1.0):
            raise ValueError(
                "lsnpc_focal_alpha must be None (unweighted terms) or within "
                f"[0, 1], got {self.lsnpc_focal_alpha!r}"
            )
        if self.lsnpc_head not in ("concat", "gate"):
            raise ValueError(
                f"lsnpc_head must be 'concat' or 'gate', got {self.lsnpc_head!r}"
            )
        if self.lsnpc_embed_dim < 1:
            raise ValueError(
                f"lsnpc_embed_dim must be >= 1, got {self.lsnpc_embed_dim}"
            )
        if self.lsnpc_correction_cond not in ("none", "x", "x_yhat", "yhat"):
            raise ValueError(
                "lsnpc_correction_cond must be 'none', 'x', 'x_yhat' or "
                f"'yhat', got {self.lsnpc_correction_cond!r}"
            )
        if self.system not in (
            "no_lsnpc",
            "corrected_latent_only",
            "corrected_condition_only",
            "full_lsnpc",
            "lsnpc_train_only",
        ):
            raise ValueError(f"unknown system {self.system!r}")
        if self.formal_run:
            if not self.protocol_id or self.protocol_id == "exploratory":
                raise ValueError("formal runs require a non-exploratory --protocol-id")
            if not self.lsnpc_loss_explicit:
                raise ValueError("formal runs require an explicit --lsnpc-loss")
            if not self.system_explicit:
                raise ValueError("formal runs require an explicit system flag")
            if self.system == "lsnpc_train_only":
                raise ValueError(
                    "lsnpc_train_only is not one of the four formal systems"
                )
            if self.system not in ("no_lsnpc", "lsnpc_train_only") and not self.use_semi:
                raise ValueError(
                    f"formal {self.system} runs require --use-semi so clean set "
                    "size and IW samples enter training"
                )
        if self.clean_set_size < -1:
            raise ValueError(
                f"clean_set_size must be -1 (use all clean rows), 0 (no "
                f"clean set) or a positive cap, got {self.clean_set_size}"
            )
        if self.yhat_source not in ("clf", "data"):
            raise ValueError(
                f"yhat_source must be 'clf' or 'data', got {self.yhat_source!r}"
            )
        if self.encoder_backbone not in (
            "conv",
            "resnet50",
            "resnet101",
            "swin",
            "swin_tiny",
            "vit",
        ):
            raise ValueError(
                f"encoder_backbone must be 'conv', 'resnet50', 'resnet101', "
                f"'swin', 'swin_tiny', or 'vit', "
                f"got {self.encoder_backbone!r}"
            )
        if self.eval_train_slice < 0:
            raise ValueError("eval_train_slice must be non-negative")
        if self.pseudo_clean_cap < 0:
            raise ValueError("pseudo_clean_cap must be non-negative")
        if self.iw_samples < 1 or self.iw_infer_samples < 1:
            raise ValueError("IW sample counts must be positive")

    @classmethod
    def from_args(cls, args: argparse.Namespace | None = None) -> ExperimentConfig:
        """Create config from parsed argparse Namespace or from sys.argv."""
        if args is None:
            args = _parse_args()
        config = cls(
            dataset=args.dataset,
            data_dir=args.data_dir,
            noise=args.noise,
            noise_type=args.noise_type,
            conditioning_mode=args.conditioning_mode,
            yhat_source=args.yhat_source,
            encoder_backbone=args.encoder_backbone,
            freeze_backbone=args.freeze_backbone,
            eval_train_slice=args.eval_train_slice,
            pseudo_clean_cap=args.pseudo_clean_cap,
            n_test=args.n_test,
            output_dir=args.output_dir,
            seed=args.seed,
            protocol_id=args.protocol_id,
            formal_run=args.formal_run,
            vae_epochs=args.vae_epochs,
            image_vae_beta=args.image_vae_beta,
            vae_augment=args.vae_augment,
            vae_ckpt=args.vae_ckpt,
            early_stop_patience=args.early_stop_patience,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_blocks,
            force_retrain=args.force_retrain,
            system=(
                args.system
                if args.system is not None
                # Legacy flag shims → system values (same precedence as the
                # former boolean resolution).
                else (
                    "corrected_latent_only"
                    if args.use_corrected_latent_only
                    else "corrected_condition_only"
                    if args.use_corrected_condition_only
                    else "lsnpc_train_only"
                    if args.lsnpc_train_only
                    else ("full_lsnpc" if args.use_lsnpc else "no_lsnpc")
                    if args.use_lsnpc is not None
                    else "full_lsnpc"
                )
            ),
            system_explicit=(
                args.system is not None
                or args.use_lsnpc is not None
                or args.use_corrected_latent_only
                or args.use_corrected_condition_only
                or args.lsnpc_train_only
            ),
            use_semi=args.use_semi,
            clean_set_size=args.clean_set_size,
            iw_samples=args.iw_samples,
            lsnpc_lr=args.lsnpc_lr,
            lsnpc_backbone_lr_scale=args.lsnpc_backbone_lr_scale,
            lsnpc_eps=args.lsnpc_eps,
            lsnpc_epochs=args.lsnpc_epochs,
            lsnpc_beta=args.lsnpc_beta,
            lsnpc_eta=args.lsnpc_eta,
            lsnpc_nu0=args.lsnpc_nu0,
            lsnpc_loss=args.lsnpc_loss or "correction",
            lsnpc_loss_explicit=args.lsnpc_loss is not None,
            lsnpc_focal_gamma=args.focal_gamma,
            lsnpc_focal_alpha=args.focal_alpha,
            lsnpc_head=args.lsnpc_head,
            lsnpc_embed_dim=args.lsnpc_embed_dim,
            lsnpc_correction_cond=args.correction_cond,
            lsnpc_shared_yhat_embed=args.shared_yhat_embed,
            lsnpc_correction_input=args.correction_input,
            lsnpc_image_data=args.lsnpc_image_data,
            lsnpc_img_channels=args.lsnpc_img_channels,
            lsnpc_img_size=args.lsnpc_img_size,
            iw_infer_samples=args.iw_infer_samples,
        )
        config.validate()
        return config


def _add_data_args(p: argparse.ArgumentParser) -> None:
    """Dataset, noise and evaluation-slice options."""
    p.add_argument(
        "--dataset",
        default="cifar100n",
        # Datasets reachable through ``data_process.base.get_dataset``, PLUS the
        # names the dedicated runners in scripts/ (image_lsnpc,
        # text_lsnpc_sst2) pass: they construct an argv and re-parse it through
        # this parser to build their ExperimentConfig, so a name they serve but
        # this harness does not must still parse here. The tabular sets went
        # with the tabular code path and are not listed.
        choices=[
            "cifar10",
            "cifar10n",
            "cifar100",
            "cifar100n",
            "animal10n",
            "dopanim",
            "eurosat",
            "sst2",
            "ag_news",
            "noisyag_best",
            "noisyag_med",
            "noisyag_worst",
            "medical_abstracts",
        ],
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--noise", type=float, default=0.0, help="Label noise rate")
    p.add_argument(
        "--noise-type",
        default="symmetric",
        choices=[
            "symmetric",
            "pairflip",
            "worse",
            "clean",
            "aggre",
            "fine",
            "coarse",
            "noisy",
        ],
    )
    p.add_argument(
        "--conditioning-mode",
        default="posterior",
        choices=["posterior", "hard", "dense_cost", "hard_path"],
        help="Conditioning signal: posterior class probabilities (default) or hard arg-max labels.",
    )
    p.add_argument(
        "--yhat-source",
        default="clf",
        choices=["clf", "data"],
        help="Source of the noisy label yhat the correction conditions on. "
        "'clf' (default) uses a pretrained classifier's prediction; "
        "'data' uses the raw noisy labels directly and trains no classifier.",
    )
    p.add_argument(
        "--encoder-backbone",
        default="conv",
        choices=["conv", "resnet50", "resnet101", "swin", "swin_tiny", "vit"],
        help="Image encoder backbone for LSNPC: 'conv' (paper ConvEncoder), "
        "'resnet50'/'resnet101' (torchvision ResNet), 'swin' (pre-"
        "downloaded Swin-Base), 'swin_tiny' (vendored Swin-Tiny), or 'vit' "
        "(pre-downloaded ViT-Base/16). All use their pretrained architecture "
        "unchanged.",
    )
    p.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze the pretrained image-encoder backbone (heads train). "
        "Pass --no-freeze-backbone to fine-tune the full backbone.",
    )
    p.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Fine-tune the full pretrained image-encoder backbone "
        "(heavier/slower, higher memory).",
    )
    p.add_argument(
        "--eval-train-slice",
        type=int,
        default=0,
        help="With --yhat-source data, evaluate on a held-out slice of this many "
        "TRAINING rows (which carry noisy + clean labels) instead of the "
        "test set. 0 = off.",
    )
    p.add_argument(
        "--pseudo-clean-cap",
        type=int,
        default=0,
        help="With --yhat-source data, cap the pseudo-clean pool (the full "
        "test set) to this many rows. 0 = use all. Use this to keep a heavy "
        "encoder (e.g. Swin) within a feasible runtime.",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=100,
        help="Eval slice for the protocol scores: -1 = every test row. "
        "Protocol runs use 100 (per-run CI ~= +-0.095); full-eval runs pass -1.",
    )


def _add_vae_args(p: argparse.ArgumentParser) -> None:
    """VAE training and architecture options."""
    p.add_argument("--vae-epochs", type=int, default=10)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Universal-VAE epochs without val improvement before stopping "
        "(0 = n_epochs/10). The LSNPC stage-1 trainer runs its full "
        "--lsnpc-epochs regardless.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--image-vae-beta",
        type=float,
        default=1.0,
        help="KL β weight for the image Conv VAE",
    )
    p.add_argument(
        "--no-vae-augment",
        dest="vae_augment",
        action="store_false",
        help="Disable image VAE batch aug (random crop + flip + normalize)",
    )
    p.add_argument(
        "--vae-ckpt",
        type=str,
        default=None,
        help="Load the universal plausibility VAE from this checkpoint path "
        "instead of training one (mutes VAE training; skip-wins over the "
        "per-cell cache and --force-retrain)",
    )
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=3)


def _add_system_args(p: argparse.ArgumentParser) -> None:
    """Correction-system options."""
    p.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore checkpoints and retrain all models from scratch",
    )
    p.add_argument(
        "--system",
        type=str,
        default=None,
        choices=[
            "no_lsnpc",
            "corrected_latent_only",
            "corrected_condition_only",
            "full_lsnpc",
            "lsnpc_train_only",
        ],
        help="Four-system ablation matrix (one per run): no_lsnpc (System 1 "
        "baseline) | corrected_latent_only (System 2) | "
        "corrected_condition_only (System 3) | full_lsnpc (System 4, "
        "default) | lsnpc_train_only (wiring ablation). Formal runs must "
        "pass --system or a legacy flag explicitly.",
    )
    p.add_argument(
        "--use-lsnpc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy shim: --use-lsnpc → --system full_lsnpc, "
        "--no-use-lsnpc → --system no_lsnpc",
    )
    p.add_argument(
        "--use-semi",
        action="store_true",
        help="Enable semi-supervised LSNPC: use eval set with η clean-branch. "
        "Otherwise train on noisy set only.",
    )
    p.add_argument(
        "--use-corrected-latent-only",
        action="store_true",
        help="Legacy shim: equivalent to --system corrected_latent_only",
    )
    p.add_argument(
        "--use-corrected-condition-only",
        action="store_true",
        help="Legacy shim: equivalent to --system corrected_condition_only",
    )
    p.add_argument(
        "--lsnpc-train-only",
        action="store_true",
        help="Legacy shim: equivalent to --system lsnpc_train_only",
    )


def _add_lsnpc_args(p: argparse.ArgumentParser) -> None:
    """LSNPC training-time label-correction options."""
    p.add_argument(
        "--clean-set-size",
        type=int,
        default=-1,
        help=(
            "Clean set size |D_clean|; -1 = use all available clean rows "
            "(default); 0 = no clean set (unsupervised correction); "
            "(default; required when no held-out clean split is provided and "
            "clean set must be drawn from training — then pass a positive size)"
        ),
    )
    p.add_argument(
        "--iw-samples",
        type=int,
        default=5,
        help="K importance samples for the IW-ELBO (Stage 1)",
    )
    p.add_argument(
        "--lsnpc-lr",
        type=float,
        default=5e-5,
        help="LSNPC (Stage 1) learning rate (1e-3 caused seed instability)",
    )
    p.add_argument(
        "--lsnpc-backbone-lr-scale",
        type=float,
        default=1.0,
        help="LR multiplier for the pretrained image backbone (the correction "
        "heads keep --lsnpc-lr). 1.0 = one group, the previous behaviour; "
        "small values (e.g. 0.02) stop a fine-tuned backbone from collapsing.",
    )
    p.add_argument(
        "--lsnpc-eps",
        type=float,
        default=1e-6,
        help="AdamW eps for LSNPC (1e-6 REQUIRED for bf16 mixed "
        "precision gradient noise; 1e-8 destabilizes bf16 runs)",
    )
    p.add_argument(
        "--lsnpc-epochs", type=int, default=30, help="LSNPC (Stage 1) training epochs"
    )
    p.add_argument(
        "--lsnpc-beta",
        type=float,
        default=1.0,
        help="KL coefficient β of the closed-form correction loss (paper eq:correction_loss); replaces the former --lsnpc-kl-weight",
    )
    p.add_argument(
        "--lsnpc-eta",
        type=float,
        default=None,
        help="Gating weight η on the semi-supervised LSNPC pass "
        "(default: 0.5 when --use-semi, else 0.0)",
    )
    p.add_argument(
        "--lsnpc-nu0",
        type=float,
        default=2.0,
        help="Prior (generative) df for LSNPC Student-t",
    )
    p.add_argument(
        "--lsnpc-loss",
        type=str,
        default=None,
        choices=["correction"],
        help="LSNPC training loss (correction only). Required explicitly for --formal-run.",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal-loss focusing parameter (Lin et al., 2017) on the LSNPC "
             "label terms: 0.0 keeps plain cross-entropy, >0 switches to the "
             "one-vs-all focal form (sum of per-class sigmoid focal terms)",
    )
    p.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Focal-loss alpha balance of the one-vs-all terms, "
             "alpha*y + (1-alpha)*(1-y); omit for unweighted terms, 0.25 is "
             "the detection setting for a rare positive class",
    )
    p.add_argument(
        "--lsnpc-head",
        choices=["concat", "gate"],
        default="concat",
        help="Label-head structure. 'concat' is the original (x_feat ⊕ z "
             "through one trunk, so the image path can be ignored). 'gate' "
             "runs a trunk per input and mixes their logits with a learned "
             "gate, making reliance on each pathway observable.",
    )
    p.add_argument(
        "--correction-cond",
        choices=["none", "x", "x_yhat", "yhat"],
        default="none",
        help="Inputs of the correction map q(z | ẑ, ·). 'none' keeps the "
             "paper's q(z | ẑ, x) through the VAE latent, 'x' adds the image "
             "embedding to its gate, 'x_yhat' adds the one-hot noisy label, "
             "and 'yhat' drops x, blending toward a learned label embedding.",
    )
    p.add_argument(
        "--shared-yhat-embed",
        action="store_true",
        help="Shared label-embedding library: one learnable per-class "
             "embedding feeds both the posterior's label slot and the "
             "blend target.",
    )
    p.add_argument(
        "--correction-input", choices=["gated", "concat"], default="gated",
        help="Correction map input: 'gated' blends ẑ toward the blend target, "
             "'concat' runs a trunk over cat([ẑ, x_feat]).")
    p.add_argument(
        "--lsnpc-embed-dim",
        type=int,
        default=128,
        help="Width of the trainable projection from the pretrained backbone's "
             "pooled feature (RN50: 2048-d) into the LSNPC image embedding "
             "shared by the noisy encoder and the label head. The backbone "
             "feature is never fed raw; this sets how narrow the projection is.",
    )
    p.add_argument(
        "--lsnpc-image-data",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Feed the LSNPC raw pixels instead of precomputed features "
             "(default: False)",
    )
    p.add_argument("--lsnpc-img-channels", type=int, default=3)
    p.add_argument("--lsnpc-img-size", type=int, default=96)
    p.add_argument(
        "--iw-infer-samples",
        type=int,
        default=5,
        help="M importance samples for the inference-time z_0 estimate (M=1 → single-sample)",
    )


def _add_run_args(p: argparse.ArgumentParser) -> None:
    """Run identity, seeding and output options."""
    p.add_argument("--output-dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--protocol-id",
        default="exploratory",
        help="Immutable protocol identifier recorded in formal run manifests.",
    )
    p.add_argument(
        "--formal-run",
        action="store_true",
        help="Enable strict reproducibility checks for paper-result runs.",
    )
def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiments."""
    p = argparse.ArgumentParser(description="Run LSNPC experiments")
    _add_data_args(p)
    _add_vae_args(p)
    _add_system_args(p)
    _add_lsnpc_args(p)
    _add_run_args(p)
    return p.parse_args()
