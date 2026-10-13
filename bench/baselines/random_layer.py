"""Reference baseline: predict a random LLM layer for every cell."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True,
                   help="TSV with cell_id column listing eval cells")
    p.add_argument("--n_layers", type=int, default=13)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cells = pd.read_csv(args.cells, sep="\t")
    rng = np.random.default_rng(args.seed)
    cells["predicted_best_layer"] = rng.integers(
        0, args.n_layers, size=len(cells))
    cells.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(cells)} random-layer predictions to {args.out}")


if __name__ == "__main__":
    main()
