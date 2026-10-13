"""bench/run_ecog.py — per-subject ECoG bench runner.
ECoG metadata is per-subject; aggregates median best-layer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata


PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "bench/submissions"


def fisher_z(rho):
    return float(np.arctanh(np.clip(rho, -0.999, 0.999)))


def headline_z(meg_pat, llm_arr):
    if meg_pat.shape[0] < 50: return float("nan")
    rdm_m = np.nan_to_num(pdist(meg_pat, metric="correlation"), nan=0.0)
    rdm_l = np.nan_to_num(pdist(llm_arr, metric="correlation"), nan=0.0)
    if rdm_m.std() == 0 or rdm_l.std() == 0: return float("nan")
    return fisher_z(float(np.corrcoef(rankdata(rdm_m), rankdata(rdm_l))[0, 1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="gpt2")
    p.add_argument("--n_sample", type=int, default=800)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ecog_root = PROJ / "data/preprocessed/ecog"
    sub_dirs = sorted(ecog_root.glob("sub-*"))[:5]
    llm_base = PROJ / f"data/preprocessed/ecog_llm/{args.llm}/{args.llm}"
    if not llm_base.exists():
        print(f"[err] llm dir not found: {llm_base}")
        return

    rows = []
    for sd in sub_dirs:
        sid = sd.name
        info = json.loads((sd / "metadata.json").read_text())
        epoch_tmin = float(info["epoch_tmin"])
        epoch_tmax = float(info["epoch_tmax"])
        n_times = int(info["n_times"])
        times = np.linspace(epoch_tmin, epoch_tmax, n_times)
        # Headline window: M400c-ish (300-450ms post-onset)
        t_mask = (times >= 0.30) & (times <= 0.45)

        ep = np.load(sd / "epochs_data.npy", mmap_mode="r")
        n_total = ep.shape[0]
        n_sample = min(args.n_sample, n_total)
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n_total, n_sample, replace=False))
        E = np.asarray(ep[idx], dtype=np.float32)
        meg_pat = np.nan_to_num(E[:, :, t_mask].mean(axis=2))

        per_layer = []
        layer_files = sorted(llm_base.glob("layer_*.npy"))
        for lf in layer_files:
            L = int(lf.stem.split("_")[-1])
            arr = np.load(lf, mmap_mode="r")
            if arr.shape[0] < n_total: continue
            emb = np.asarray(arr[idx], dtype=np.float32)
            z = headline_z(meg_pat, emb)
            per_layer.append((L, z))
        if not per_layer: continue
        per_layer.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else -np.inf)
        best_L, peak = per_layer[0]
        rows.append({"subject": sid, "best_layer": int(best_L),
                     "peak_z": float(peak)})
        print(f"  {sid}: best L={best_L} z={peak*1000:+.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"ecog_{args.llm}_persubj_best_layer.tsv",
              sep="\t", index=False)
    # cohort-aggregate
    if len(df):
        median_L = int(df["best_layer"].median())
        median_z = float(df["peak_z"].median())
        agg = pd.DataFrame([{
            "cell_id": f"ecog_{args.llm}_headline",
            "predicted_best_layer": median_L,
            "peak_z_rho": median_z,
        }])
        agg.to_csv(OUT / f"ecog_{args.llm}_best_layer.tsv",
                   sep="\t", index=False)
        print(f"\nCohort median: L{median_L}, z={median_z*1000:+.2f}",
              flush=True)


if __name__ == "__main__":
    main()
