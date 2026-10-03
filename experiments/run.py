"""Unified experiment runner (LSNPC label-correction pipeline).

Usage:
  python -m experiments.run --dataset cifar100n --noise 0.2

Outputs: results/ directory with per-method JSON and a summary CSV.
"""

import logging

from .arguments import ExperimentConfig
from .pipeline import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("experiments")


def main() -> None:
    """Parse args, create config, and run the experiment pipeline."""
    config = ExperimentConfig.from_args()
    run_experiment(config)


if __name__ == "__main__":
    main()
