"""Validate the independent corrector seeds consumed by Set B."""
import argparse
import re
from pathlib import Path


def check(directory, sidecars, seeds, modality="all"):
    groups = {}
    for p in sorted(directory.glob("*.json")):
        if modality != "all" and not p.name.startswith(modality + "_"):
            continue
        m = re.search(r"_s(\d+)_", p.stem)
        if not m:
            raise ValueError(f"Cannot identify seed in {p.name}")
        if not (sidecars / (p.stem + ".npz")).is_file():
            raise ValueError(f"Missing per-query data for {p.name}")
        key = p.stem[:m.start()] + "_sSEED_" + p.stem[m.end():]
        seen = groups.setdefault(key, set())
        seed = int(m.group(1))
        if seed in seen:
            raise ValueError(f"Duplicate seed {seed} in {key}")
        seen.add(seed)
    if not groups:
        raise ValueError("No Set A cells found. Run Set A first.")
    for key, actual in groups.items():
        if actual != set(seeds):
            raise ValueError(f"{key}: expected seeds {sorted(seeds)}, found {sorted(actual)}; use a separate directory for extensions")
    return len(groups)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--npz-dir", type=Path, required=True)
    p.add_argument("--seeds", default="42 43 44 45 46")
    p.add_argument("--modality", default="all", choices=["all", "text", "image"])
    a = p.parse_args()
    try:
        n = check(a.json_dir, a.npz_dir, [int(x) for x in a.seeds.split()], a.modality)
    except ValueError as e:
        p.exit(1, str(e) + "\n")
    print(f"Validated {n} settings with complete seed coverage")
