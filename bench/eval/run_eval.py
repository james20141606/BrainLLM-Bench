"""Reference eval script for MEG-LLM-Bench submissions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def eval_best_layer(submission: pd.DataFrame, gold: pd.DataFrame
                    ) -> dict:
    """Score: Spearman ρ between submitted and gold-standard layer ranks."""
    m = submission.merge(gold, on="cell_id", suffixes=("_sub", "_gold"))
    rho, p = spearmanr(m["predicted_best_layer_sub"],
                       m["predicted_best_layer_gold"])
    return {"task": "best-layer", "spearman_rho": float(rho),
            "p_value": float(p), "n_cells": int(len(m))}


def eval_boundary_conditional(submission: pd.DataFrame) -> dict:
    """Score: fraction of cells with z_medial > z_onset (sign consistency)."""
    sub = submission.dropna(subset=["z_onset", "z_medial"])
    sub = sub.assign(delta=sub["z_medial"] - sub["z_onset"])
    pos = (sub["delta"] > 0).sum()
    return {"task": "boundary-conditional",
            "sign_consistency": float(pos) / max(1, len(sub)),
            "median_delta": float(sub["delta"].median()),
            "n_cells": int(len(sub))}


def eval_surprisal_tercile(submission: pd.DataFrame) -> dict:
    """Score: weighted-mean active layer drift (high - low tercile)."""
    pivot = submission.pivot_table(
        index=["cell_id", "layer"], columns="tercile",
        values="z_rho").reset_index()
    if "low" not in pivot or "high" not in pivot:
        return {"task": "surprisal-tercile", "error": "missing tercile columns"}

    def w_mean_layer(g, col):
        w = g[col].clip(lower=0)
        if w.sum() == 0: return float("nan")
        return float((g["layer"] * w).sum() / w.sum())

    cells = pivot["cell_id"].unique()
    drifts = []
    for c in cells:
        g = pivot[pivot["cell_id"] == c]
        wL = w_mean_layer(g, "low")
        wH = w_mean_layer(g, "high")
        if not (np.isnan(wL) or np.isnan(wH)):
            drifts.append(wH - wL)
    return {"task": "surprisal-tercile",
            "mean_layer_drift": float(np.mean(drifts)) if drifts else float("nan"),
            "n_cells": int(len(drifts))}


def eval_sae_feature_transfer(submission: pd.DataFrame,
                               anchor: pd.DataFrame) -> dict:
    """Score: top-50 overlap with anchor cohort."""
    sub_top = set(submission.nlargest(50, "z_rho")["sae_feature_idx"])
    anchor_top = set(anchor.nlargest(50, "z_rho")["sae_feature_idx"])
    overlap = sub_top & anchor_top
    return {"task": "sae-feature-transfer",
            "top50_overlap": int(len(overlap)),
            "expected_chance": 50 * 50 / 24576}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("submission", help="path to submission TSV")
    p.add_argument("--task", required=True,
                   choices=["best-layer", "boundary-conditional",
                            "surprisal-tercile", "sae-feature-transfer"])
    p.add_argument("--gold", help="path to gold standard TSV")
    p.add_argument("--anchor", help="anchor cohort submission TSV (for SAE)")
    args = p.parse_args()

    sub = pd.read_csv(args.submission, sep="\t")
    if args.task == "best-layer":
        gold = pd.read_csv(args.gold, sep="\t")
        result = eval_best_layer(sub, gold)
    elif args.task == "boundary-conditional":
        result = eval_boundary_conditional(sub)
    elif args.task == "surprisal-tercile":
        result = eval_surprisal_tercile(sub)
    elif args.task == "sae-feature-transfer":
        anchor = pd.read_csv(args.anchor, sep="\t")
        result = eval_sae_feature_transfer(sub, anchor)
    print(result)


if __name__ == "__main__":
    main()
