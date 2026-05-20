#!/usr/bin/env python3
"""Create data subsets for pairwise data efficiency curve."""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "pairwise" / "cross_source_train.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pairwise"

def main():
    random.seed(42)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    total = len(full_data)
    print(f"Full dataset: {total} pairs")

    # Shuffle deterministically
    indices = list(range(total))
    random.shuffle(indices)

    for frac_name, frac in [("25", 0.25), ("50", 0.50)]:
        n = int(total * frac)
        subset_indices = indices[:n]
        subset = [full_data[i] for i in subset_indices]

        out_path = OUTPUT_DIR / f"cross_source_train_{frac_name}pct.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)

        print(f"{frac_name}% subset: {len(subset)} pairs -> {out_path}")

    print("Done!")

if __name__ == "__main__":
    main()
