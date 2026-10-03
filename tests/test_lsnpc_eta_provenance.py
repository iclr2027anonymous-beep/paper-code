"""Regression test for F-06 (lsnpc_eta override).

F-06: ``lsnpc_stage1.py`` unconditionally overwrote ``config.lsnpc_eta`` with
      ``0.5 if use_semi else 0.0``, so an explicit ``--lsnpc-eta`` was ignored.
"""

import sys
import unittest

from experiments.arguments import ExperimentConfig
from experiments.lsnpc_stage1 import _resolve_lsnpc_eta


def _config(argv: list[str]) -> ExperimentConfig:
    sys.argv = ["test"] + argv
    return ExperimentConfig.from_args()


class LsnpcEtaOverrideTests(unittest.TestCase):
    def test_config_eta_default_is_none(self):
        cfg = _config([])
        self.assertIsNone(
            cfg.lsnpc_eta,
            "default --lsnpc-eta must be None so stage-1 can apply mode default",
        )

    def test_explicit_eta_preserved_in_config(self):
        for value in ("0.0", "0.1", "0.5", "0.9"):
            cfg = _config(["--lsnpc-eta", value])
            self.assertAlmostEqual(
                cfg.lsnpc_eta,
                float(value),
                msg=f"--lsnpc-eta {value} must reach the config untouched",
            )

    def test_resolve_mode_default_semi(self):
        cfg = _config([])
        self.assertAlmostEqual(_resolve_lsnpc_eta(cfg, use_semi=True), 0.1)

    def test_resolve_mode_default_unsupervised(self):
        cfg = _config([])
        self.assertAlmostEqual(_resolve_lsnpc_eta(cfg, use_semi=False), 0.0)

    def test_resolve_never_overrides_explicit_eta(self):
        for value, use_semi in (("0.1", True), ("0.0", True), ("0.9", False)):
            cfg = _config(["--lsnpc-eta", value])
            self.assertAlmostEqual(
                _resolve_lsnpc_eta(cfg, use_semi=use_semi),
                float(value),
                msg=f"explicit --lsnpc-eta {value} (use_semi={use_semi}) must not be overwritten",
            )


if __name__ == "__main__":
    unittest.main()
