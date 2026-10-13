#!/usr/bin/env python3
"""Train-only alpha-grid sensitivity for the LibriBrain Qwen2.5 encoding ladder.

The original readout is replayed before the expanded grid on the same 12 outer
leave-one-session-out folds. RidgeCV chooses alpha using training data only.
One Slurm array task handles one cached model and writes one JSON receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path("<PROJECT>")
sys.path.insert(0, str(ROOT / "code"))
import p4_concordance32 as C  # noqa: E402
from region_atlas import get_canonical_neuromag306_order, get_neuromag_region_masks  # noqa: E402

MODELS = [
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
]
ORIGINAL = np.logspace(1, 6, 8)
EXPANDED = np.unique(np.concatenate([ORIGINAL, np.logspace(7, 14, 8)]))
OUT = ROOT / "results/p4_encoding/alpha_sensitivity_20260922"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score(y_true: np.ndarray, pred: np.ndarray) -> float:
    yc = y_true - y_true.mean(0, keepdims=True)
    pc = pred - pred.mean(0, keepdims=True)
    num = (yc * pc).sum(0)
    den = np.sqrt((yc ** 2).sum(0) * (pc ** 2).sum(0)) + 1e-12
    return float((num / den).mean())


def fit_one(xtr: np.ndarray, xte: np.ndarray, ytr: np.ndarray,
            yte: np.ndarray, grid: np.ndarray) -> tuple[float, float, np.ndarray]:
    model = RidgeCV(alphas=grid)
    model.fit(xtr, ytr - ytr.mean(0))
    pred = model.predict(xte) + ytr.mean(0)
    return float(model.alpha_), score(yte, pred), pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-index", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="one outer fold, 0.5B only")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    if not 0 <= args.model_index < len(MODELS):
        raise SystemExit("model-index must be 0..6")
    if args.smoke and args.model_index != 0:
        raise SystemExit("smoke is restricted to model-index 0")
    model_name = MODELS[args.model_index]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = C.DATA / "trial_metadata.tsv"
    epoch_path = C.DATA / "epochs_data.npy"
    meta = pd.read_csv(meta_path, sep="\t")
    sel = C.subsample(meta, 2500, 42, C.GROUP_COL)
    groups = meta.iloc[sel][C.GROUP_COL].to_numpy()
    if len(sel) != 30_000 or len(np.unique(groups)) != 12:
        raise RuntimeError(f"unexpected subsample: {len(sel)} words, {len(np.unique(groups))} sessions")
    ep = np.load(epoch_path, mmap_mode="r")
    mask = get_neuromag_region_masks(get_canonical_neuromag306_order())[C.REGION]
    s0 = int(round((C.WIN_MS[0] / 1000.0 - C.TMIN) * C.SFREQ))
    s1 = int(round((C.WIN_MS[1] / 1000.0 - C.TMIN) * C.SFREQ))
    pat = C.neural_patterns(ep, sel, mask, s0, s1, C.BAND)
    y = pat.reshape(len(pat), int(mask.sum()), s1 - s0).mean(axis=2).astype(np.float64)
    yscale = float(y.std())
    if yscale > 0:
        y /= yscale

    feat_path = C.mid_layer_path(model_name, C.FEAT)
    x = np.asarray(np.load(feat_path, mmap_mode="r")[sel], dtype=np.float64)
    baseline_path = args.out_dir / "concordance32_CORRECTED.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"copy the frozen corrected baseline to {baseline_path}")
    baseline = json.loads(baseline_path.read_text())
    baseline_scores = baseline["per_session_enc"][model_name]
    expected_mean = next(r["enc_mean"] for r in baseline["per_model"] if r["model"] == model_name)
    if baseline["shared_subsample"]["n_words"] != 30_000:
        raise RuntimeError("baseline used a different word subsample")

    print(f"MODEL {model_name} X={x.shape} Y={y.shape} original={list(ORIGINAL)} expanded={list(EXPANDED)}", flush=True)
    folds = []
    for fold_idx, (tr, te) in enumerate(LeaveOneGroupOut().split(x, y, groups)):
        sid = str(groups[te][0])
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
        xtr = (x[tr] - mu) / sd
        xte = (x[te] - mu) / sd
        ytr, yte = y[tr], y[te]
        a0, sc0, p0 = fit_one(xtr, xte, ytr, yte, ORIGINAL)
        match_error = sc0 - float(baseline_scores[sid])
        if abs(match_error) > 1e-6:
            raise RuntimeError(f"original-grid replay mismatch model={model_name} session={sid}: {match_error}")
        a1, sc1, p1 = fit_one(xtr, xte, ytr, yte, EXPANDED)
        rows = {
            "session": sid, "n_train": int(len(tr)), "n_test": int(len(te)),
            "original_alpha": a0, "original_score": sc0,
            "stored_original_score": float(baseline_scores[sid]),
            "original_match_error": match_error,
            "expanded_alpha": a1, "expanded_score": sc1,
            "expanded_minus_original": sc1 - sc0,
            "prediction_correlation": float(np.corrcoef(p0.ravel(), p1.ravel())[0, 1]),
            "prediction_rms_difference": float(np.sqrt(np.mean((p1 - p0) ** 2))),
        }
        folds.append(rows)
        print(f"FOLD {sid} old={a0:.4g}/{sc0:.7f} new={a1:.4g}/{sc1:.7f} delta={sc1-sc0:+.7f}", flush=True)
        if args.smoke:
            break

    old_mean = float(np.mean([r["original_score"] for r in folds]))
    new_mean = float(np.mean([r["expanded_score"] for r in folds]))
    result = {
        "model": model_name, "smoke": bool(args.smoke),
        "protocol": {
            "cohort": "LibriBrain", "region": C.REGION, "band": C.BAND,
            "window_ms": list(C.WIN_MS), "depth_frac": 0.5,
            "subsample_max_per_session": 2500, "subsample_seed": 42,
            "outer_split": "LeaveOneGroupOut(session)",
            "alpha_selection": "RidgeCV default GCV on outer training fold only",
            "original_alphas": ORIGINAL.tolist(), "expanded_alphas": EXPANDED.tolist(),
            "n_words": int(len(sel)), "n_sessions": int(len(np.unique(groups))),
            "x_shape": list(x.shape), "y_shape": list(y.shape),
        },
        "provenance": {
            "generator_sha256": sha256(Path(__file__)),
            "original_pipeline_sha256": sha256(Path(C.__file__)),
            "metadata_sha256": sha256(meta_path),
            "epochs_path": str(epoch_path),
            "epochs_size_bytes": epoch_path.stat().st_size,
            "embedding_path": str(feat_path), "embedding_sha256": sha256(feat_path),
            "baseline_sha256": sha256(baseline_path),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "original_mean": old_mean, "stored_original_mean": expected_mean,
        "original_mean_match_error": None if args.smoke else old_mean - expected_mean,
        "expanded_mean": new_mean, "expanded_minus_original_mean": new_mean - old_mean,
        "expanded_top_grid_count": sum(r["expanded_alpha"] == float(EXPANDED[-1]) for r in folds),
        "folds": folds,
    }
    suffix = "smoke" if args.smoke else "full"
    out_path = args.out_dir / f"qwen25_{args.model_index}_{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"WROTE {out_path} mean {old_mean:.7f} -> {new_mean:.7f}", flush=True)


if __name__ == "__main__":
    main()
