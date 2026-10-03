"""Check whether a cached downstream cell contains the requested experiment."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def complete(cell, models, seeds, coverages, configs):
    for family in models:
        blocks = (cell.get("models", {}).get(family) or {}).get("configs", {})
        for config in configs:
            if config == "d" and cell.get("clean_ceiling") is False:
                continue
            if config in ("e", "f") and family != "mlp":
                continue
            names = [f"{config}@{c:g}" for c in coverages] if config in ("c", "f") else [config]
            for name in names:
                record = (cell.get("configs", {}) if config == "f" else blocks).get(name, {})
                actual = record.get("seeds", [])
                if len(actual) != len(set(actual)) or set(actual) != set(seeds):
                    return False
                for metric in ("acc", "f1"):
                    values = record.get(metric, [])
                    if len(values) != len(actual) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--models", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--coverages", required=True)
    p.add_argument("--configs", required=True)
    a = p.parse_args()
    try:
        cell = json.loads(a.path.read_text())
        ok = complete(cell, a.models.split(","), [int(s) for s in a.seeds.split()],
                      [float(c) for c in a.coverages.split()], a.configs.split())
    except (OSError, ValueError, TypeError, AttributeError):
        ok = False
    print(0 if ok else 1)


if __name__ == "__main__":
    main()
