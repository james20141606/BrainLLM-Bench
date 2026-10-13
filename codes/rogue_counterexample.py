#!/usr/bin/env python3
"""Minimal counterexample to rank-one rogue invariance of standardized ridge.

This script is self-contained and does not read or modify the MEG repository.
It verifies two separate claims:

1. StandardScaler + Ridge is numerically invariant to positive per-coordinate
   rescaling when the same scale is applied to all samples.
2. It is not invariant to adding a sample-varying rank-one term X + s a v^T.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_SEED = 7
ROGUE_SEED = 9
CV_SEED = 11
N_SAMPLES = 240
N_FEATURES = 18
N_SPLITS = 6
RIDGE_ALPHA = 10.0
ROGUE_SHARE = 0.70
AXIS_LOG_SPREAD = 1.0


def inject_rogue(x: np.ndarray, share: float, seed: int) -> np.ndarray:
    """Match MEG/codes/invariance_intervention.py:53-65."""
    rng = np.random.default_rng(seed)
    xc = x - x.mean(axis=0, keepdims=True)
    direction = rng.normal(size=x.shape[1])
    direction /= np.linalg.norm(direction)
    sample_scores = rng.normal(size=(x.shape[0], 1))
    sample_scores -= sample_scores.mean()
    base = np.square(xc).sum()
    scale = np.sqrt(
        share / (1.0 - share)
        * base
        / max(np.square(sample_scores).sum(), 1e-12)
    )
    return xc + scale * sample_scores * direction[None, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(DATA_SEED)
    x = rng.normal(size=(N_SAMPLES, N_FEATURES))
    coefficients = rng.normal(size=N_FEATURES)
    y = np.einsum("ij,j->i", x, coefficients) + 0.3 * rng.normal(size=N_SAMPLES)
    axis_scale = np.exp(rng.normal(0.0, AXIS_LOG_SPREAD, size=N_FEATURES))
    x_rescaled = x * axis_scale[None, :]
    x_rogue = inject_rogue(x, ROGUE_SHARE, ROGUE_SEED)

    cv = KFold(N_SPLITS, shuffle=True, random_state=CV_SEED)
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA, solver="lsqr"),
    )
    pred_original = cross_val_predict(model, x, y, cv=cv)
    pred_rescaled = cross_val_predict(model, x_rescaled, y, cv=cv)
    pred_rogue = cross_val_predict(model, x_rogue, y, cv=cv)

    z_original = StandardScaler().fit_transform(x)
    z_rogue = StandardScaler().fit_transform(x_rogue)

    result = {
        "purpose": "Counterexample: standardized ridge is not invariant to additive sample-varying rank-one rogue directions.",
        "versions": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "design": {
            "data_seed": DATA_SEED,
            "rogue_seed": ROGUE_SEED,
            "cv_seed": CV_SEED,
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_splits": N_SPLITS,
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_solver": "lsqr",
            "rogue_variance_share": ROGUE_SHARE,
            "axis_log_spread": AXIS_LOG_SPREAD,
        },
        "axis_rescaling_control": {
            "max_abs_prediction_difference": float(
                np.max(np.abs(pred_original - pred_rescaled))
            ),
            "r2": float(r2_score(y, pred_rescaled)),
            "expected": "Invariant up to floating-point error.",
        },
        "rogue_counterexample": {
            "standardized_design_relative_frobenius_change": float(
                np.linalg.norm(z_rogue - z_original) / np.linalg.norm(z_original)
            ),
            "prediction_correlation_with_original": float(
                np.corrcoef(pred_original, pred_rogue)[0, 1]
            ),
            "max_abs_prediction_difference": float(
                np.max(np.abs(pred_original - pred_rogue))
            ),
            "r2_original": float(r2_score(y, pred_original)),
            "r2_rogue": float(r2_score(y, pred_rogue)),
            "expected": "Not invariant; one counterexample disproves the by-construction claim.",
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
