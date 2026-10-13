"""bench/run.py — runs reference baselines for MEG-LLM-Bench tasks.

Usage:
    python -m bench.run --cohort libribrain --task best-layer
    python -m bench.run --cohort libribrain --task all
"""
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


COHORT_CONFIG = {
    "libribrain": {
        "epochs": "data/preprocessed/libribrain/epochs_data.npy",
        "metadata": "data/preprocessed/libribrain/trial_metadata.tsv",
        "tmin": -0.2, "n_times": 250,
        "headline_window": (0.30, 0.45),
        "default_llm": "gpt2",
        "llm_dir_pattern": "data/preprocessed/llm_embeddings/{llm}",
    },
    "meg_masc": {
        "epochs": "data/preprocessed/meg_masc/epochs_data.npy",
        "metadata": "data/preprocessed/meg_masc/trial_metadata.tsv",
        "tmin": -0.2, "n_times": 250,
        "headline_window": (0.65, 0.85),
        "default_llm": "gpt2",
        "llm_dir_pattern": "data/preprocessed/meg_masc_llm/{llm}",
    },
    "brennan": {
        "epochs": "data/preprocessed/brennan_2019/epochs_data.npy",
        "metadata": "data/preprocessed/brennan_2019/trial_metadata.tsv",
        "tmin": -0.2, "n_times": 251,
        "headline_window": (0.40, 0.55),
        "default_llm": "gpt2",
        "llm_dir_pattern": "data/preprocessed/llm_embeddings_brennan/{llm}",
    },
    "meg_scans": {
        "epochs": "data/preprocessed/meg_scans/epochs_data.npy",
        "metadata": "data/preprocessed/meg_scans/trial_metadata.tsv",
        "tmin": -0.2, "n_times": 250,
        "headline_window": (0.50, 0.65),
        "default_llm": "dbmdz_german-gpt2",
        "llm_dir_pattern": "data/preprocessed/meg_scans_llm/{llm}",
    },
}


def fisher_z(rho):
    return float(np.arctanh(np.clip(rho, -0.999, 0.999)))


def headline_z(meg_pat, llm_arr):
    if meg_pat.shape[0] < 100: return float("nan")
    rdm_m = np.nan_to_num(pdist(meg_pat, metric="correlation"), nan=0.0)
    rdm_l = np.nan_to_num(pdist(llm_arr, metric="correlation"), nan=0.0)
    if rdm_m.std() == 0 or rdm_l.std() == 0: return float("nan")
    return fisher_z(float(np.corrcoef(rankdata(rdm_m), rankdata(rdm_l))[0, 1]))


def task_best_layer(cohort: str, llm: str = None,
                    n_sample: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Per-LLM-layer alignment scan; return best layer per (region) cell."""
    cfg = COHORT_CONFIG[cohort]
    if llm is None: llm = cfg["default_llm"]
    llm_dir = PROJ / cfg["llm_dir_pattern"].format(llm=llm)
    if not llm_dir.exists():
        raise FileNotFoundError(f"LLM dir not found: {llm_dir}")

    print(f"[best-layer] {cohort} × {llm}", flush=True)
    rng = np.random.default_rng(seed)
    epochs_mmap = np.load(PROJ / cfg["epochs"], mmap_mode="r")
    n_total = epochs_mmap.shape[0]
    n = min(n_sample, n_total)
    idx = np.sort(rng.choice(n_total, n, replace=False))
    E = np.asarray(epochs_mmap[idx], dtype=np.float32)

    n_ch = E.shape[1]; n_t = E.shape[2]
    times = np.linspace(cfg["tmin"], cfg["tmin"] + n_t / 250.0, n_t)
    tlo, thi = cfg["headline_window"]
    t_mask = (times >= tlo) & (times <= thi)
    ch = (np.arange(n_ch // 4, 3 * n_ch // 4) if n_ch >= 200
          else np.arange(n_ch))
    meg_pat = np.nan_to_num(
        E[:, ch, :][:, :, t_mask].mean(axis=2))

    layer_files = sorted(llm_dir.glob("layer_*.npy"))
    rows = []
    for lf in layer_files:
        L = int(lf.stem.split("_")[-1])
        arr = np.load(lf, mmap_mode="r")
        if arr.shape[0] < n_total: continue
        emb = np.asarray(arr[idx], dtype=np.float32)
        z = headline_z(meg_pat, emb)
        rows.append({"layer": L, "z_rho": z})
    df = pd.DataFrame(rows).sort_values("layer")
    if df.empty:
        return df
    best_layer = int(df.loc[df["z_rho"].idxmax(), "layer"])
    return pd.DataFrame([{
        "cell_id": f"{cohort}_{llm}_headline",
        "predicted_best_layer": best_layer,
        "peak_z_rho": float(df["z_rho"].max()),
    }])


def task_protocol(cohort: str, llm: str = None, *, region="temporal_L", band="broadband",
                  time_s=0.350, reliability=None, max_words=2000) -> pd.DataFrame:
    """The paper's protocol end to end: band -> region -> word-averaged RDM -> confound
    partialization -> ceiling normalization, one row per subject.

    Unlike `task_best_layer` above, which is a reference baseline, this is the statistic the
    paper reports. The RSA half is checked against the paper's stored per-(model, subject)
    numbers by `python -m bench.protocol --validate`.

    LLM features are averaged over each probe word's occurrences, matching the neural side,
    which is word-averaged for the reliability reason documented in bench/protocol.py.
    """
    from . import protocol as P

    cfg = COHORT_CONFIG[cohort]
    if llm is None:
        llm = cfg["default_llm"]
    llm_dir = PROJ / cfg["llm_dir_pattern"].format(llm=llm)
    meta = pd.read_csv(PROJ / cfg["metadata"], sep="\t")
    epochs = np.load(PROJ / cfg["epochs"], mmap_mode="r")

    wcol = next((c for c in ("word_text", "word", "stimulus") if c in meta.columns), None)
    if wcol is None:
        raise KeyError(f"no word column in {cfg['metadata']}; have {list(meta.columns)}")
    common, per_subj = P.shared_word_list(meta[wcol].tolist(), meta["subject"].tolist(),
                                          max_words)
    if not common:
        raise RuntimeError("no words shared across subjects")
    print(f"[protocol] {cohort} x {llm}: {len(common)} shared words, "
          f"{len(per_subj)} subjects, cell = {region}/{band}/{time_s*1000:.0f}ms", flush=True)

    layer_files = sorted(llm_dir.glob("layer_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    if not layer_files:
        raise FileNotFoundError(f"no layer_*.npy under {llm_dir}")
    # type-level features: average each word's occurrences, using any subject's trial rows
    any_subj = next(iter(per_subj.values()))
    rows_for = [any_subj[w] for w in common]
    hiddens = np.stack([
        np.array([np.asarray(np.load(lf, mmap_mode="r")[g], dtype=np.float64).mean(axis=0)
                  for g in rows_for])
        for lf in layer_files])

    ceiling = (P.ceiling_for(reliability, region, time_s * 1000.0) if reliability else None)
    out = []
    for subj, wmap in sorted(per_subj.items()):
        widx = [wmap[w] for w in common]
        ranks = P.neural_rdm(epochs, region=region, band=band, time_s=time_s,
                             sfreq=cfg.get("sfreq", 250.0), tmin=cfg["tmin"],
                             word_index=widx)
        r = P.score_model(ranks, hiddens, common, ceiling=ceiling)
        r.update({"cohort": cohort, "llm": llm, "subject": subj, "region": region,
                  "band": band, "time_ms": time_s * 1000.0, "n_words": len(common)})
        out.append(r)
        print(f"  sub-{subj}: L{r['best_layer']} partial={r['partial_rho']:+.5f}", flush=True)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True,
                   choices=list(COHORT_CONFIG.keys()))
    p.add_argument("--task", default="best-layer",
                   choices=["best-layer", "protocol", "all"])
    p.add_argument("--region", default="temporal_L")
    p.add_argument("--band", default="broadband")
    p.add_argument("--time_s", type=float, default=0.350)
    p.add_argument("--reliability", default=None,
                   help="reliability TSV for the per-band ceiling")
    p.add_argument("--llm", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--n_sample", type=int, default=1500)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.task == "protocol":
        out_path = (Path(args.out) if args.out
                    else OUT / f"{args.cohort}_{args.llm or 'default'}_protocol.tsv")
        df = task_protocol(args.cohort, args.llm, region=args.region, band=args.band,
                           time_s=args.time_s, reliability=args.reliability)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"  wrote {len(df)} rows to {out_path}")
        print(df)
        return

    if args.task in ("best-layer", "all"):
        out_path = (Path(args.out) if args.out
                    else OUT / f"{args.cohort}_{args.llm or 'default'}_best_layer.tsv")
        df = task_best_layer(args.cohort, args.llm, args.n_sample)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"  wrote {len(df)} rows to {out_path}")
        print(df)


if __name__ == "__main__":
    main()
