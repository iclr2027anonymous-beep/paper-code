"""Contracts for experiment identity and checkpoint isolation."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.arguments import ExperimentConfig
from utils.provenance import (
    checkpoint_dir,
    config_hash,
    finalise_checkpoint_index,
    hash_arrays,
    prepare_checkpoint_dir,
)


def _base_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="cifar100n",
        noise=0.4,
        noise_type="symmetric",
        seed=42,
        use_semi=True,
        clean_set_size=20,
        iw_samples=1,
        iw_infer_samples=1,
        hidden_dim=32,  # explicit: defaults moved 32->64 (beta-sweep era);
        # an implicit default would collide with the hidden_dim=64 variant.
    )


def test_scientific_axes_change_config_hash():
    base = _base_config()
    # lsnpc_loss="correction" is the default, so passing it is a no-op
    # (dropped from the variant list — the dataset already sets the value).
    variants = [
        replace(base, seed=43),
        replace(base, clean_set_size=200),
        replace(base, iw_samples=5, iw_infer_samples=5),
        replace(base, system="corrected_condition_only"),
        replace(base, hidden_dim=64),
    ]

    hashes = {config_hash(base), *(config_hash(item) for item in variants)}
    assert len(hashes) == len(variants) + 1


def test_operational_fields_do_not_change_scientific_hash():
    base = _base_config()
    assert config_hash(base) == config_hash(
        replace(
            base,
            output_dir="/different/output",
            force_retrain=True,
            formal_run=True,
            lsnpc_loss_explicit=True,
            system_explicit=True,
        )
    )


def test_checkpoint_directory_is_scoped_to_output_and_config(tmp_path):
    base = _base_config()
    other_seed = replace(base, seed=43)

    base_dir = checkpoint_dir(tmp_path, base)
    other_dir = checkpoint_dir(tmp_path, other_seed)

    assert base_dir.parent == tmp_path / "checkpoints"
    assert other_dir.parent == tmp_path / "checkpoints"
    assert base_dir != other_dir


def test_checkpoint_index_rejects_tampering(tmp_path):
    config = _base_config()
    directory = prepare_checkpoint_dir(tmp_path, config)
    model_path = directory / "lsnpc.pt"
    model_path.write_bytes(b"first model")
    hashes = finalise_checkpoint_index(directory, config)
    assert hashes["lsnpc.pt"]

    assert prepare_checkpoint_dir(tmp_path, config) == directory
    model_path.write_bytes(b"tampered model")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_checkpoint_dir(tmp_path, config)


def test_checkpoint_files_without_identity_are_purged_then_recreated(tmp_path):
    config = _base_config()
    directory = checkpoint_dir(tmp_path, config)
    directory.mkdir(parents=True)
    (directory / "lsnpc.pt").write_bytes(b"legacy checkpoint")
    # Identity-less directory is stale: prepare_checkpoint_dir purges it
    # (crash recovery) rather than raising, then writes a fresh identity.
    prepare_checkpoint_dir(tmp_path, config)
    assert not (directory / "lsnpc.pt").exists(), (
        "stale model files must be purged by prepare_checkpoint_dir"
    )
    assert (directory / "checkpoint_identity.json").exists()


def test_data_hash_captures_split_and_labels():
    features = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    original = hash_arrays([("features", features), ("labels", labels)])
    changed_label = hash_arrays(
        [
            ("features", features),
            ("labels", np.array([1, 1], dtype=np.int64)),
        ]
    )
    reordered = hash_arrays(
        [
            ("features", features[::-1]),
            ("labels", labels[::-1]),
        ]
    )

    assert original != changed_label
    assert original != reordered


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"system": "no_lsnpc", "use_semi": False}, "no_lsnpc"),
        (
            {"system": "corrected_latent_only", "use_semi": True},
            "corrected_latent_only",
        ),
        (
            {"system": "corrected_condition_only", "use_semi": True},
            "corrected_condition_only",
        ),
        ({"system": "full_lsnpc", "use_semi": True}, "full_lsnpc"),
    ],
)
def test_four_systems_have_explicit_names(updates, expected):
    config = replace(_base_config(), **updates)
    config.validate()
    assert config.system_name() == expected


def test_unknown_system_is_rejected():
    config = replace(_base_config(), system="bogus_system")
    with pytest.raises(ValueError, match="unknown system"):
        config.validate()


def test_formal_correction_system_requires_clean_set_and_explicit_protocol():
    missing_clean_set = replace(
        _base_config(),
        formal_run=True,
        protocol_id="tabular-v7-r1",
        lsnpc_loss_explicit=True,
        system_explicit=True,
        use_semi=False,
    )
    with pytest.raises(ValueError, match="require --use-semi"):
        missing_clean_set.validate()

    valid = replace(missing_clean_set, use_semi=True)
    valid.validate()

    implicit_loss = replace(valid, lsnpc_loss_explicit=False)
    with pytest.raises(ValueError, match="explicit --lsnpc-loss"):
        implicit_loss.validate()

    implicit_system = replace(valid, system_explicit=False)
    with pytest.raises(ValueError, match="explicit system flag"):
        implicit_system.validate()


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--no-use-lsnpc"], "no_lsnpc"),
        (
            [
                "--use-corrected-latent-only",
                "--use-semi",
            ],
            "corrected_latent_only",
        ),
        (
            [
                "--use-corrected-condition-only",
                "--use-semi",
            ],
            "corrected_condition_only",
        ),
        (["--use-lsnpc", "--use-semi"], "full_lsnpc"),
    ],
)
def test_formal_cli_system_mapping(monkeypatch, flags, expected):
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiments.run",
            "--formal-run",
            "--protocol-id",
            "tabular-v7-test",
            "--lsnpc-loss",
            "correction",
            *flags,
        ],
    )
    config = ExperimentConfig.from_args()
    assert config.system_name() == expected
