#!/usr/bin/env python
"""Which frequencies carry reproducible word-level structure? A multi-cohort test.

On LibriBrain the split-half reliability of the word-level neural dissimilarity matrix is 0.244 in
delta, 0.135 broadband and 0.107 theta, but 0.003 alpha, 0.001 beta and -0.002 gamma: two orders of
magnitude, with the high bands sitting at zero. If that holds generally it matters beyond this
benchmark, because intracranial language work routinely filters to high gamma before computing
exactly this kind of representational analysis, and a band with no reproducible word-level
structure cannot support one.

It is currently a single-subject observation on one cohort, which is why the benchmark reports it
as a preprocessing note rather than a finding. This tests it where it can be tested: Brennan (49
EEG listeners) and the rebuilt MEG-MASC (27 MEG listeners), against LibriBrain's deep
single-listener recordings.

The quantity involves no language model. For each unit and band the epochs are band-passed, each
probe word's repetitions are split in half, a dissimilarity matrix is built from each half over the
shared word set, and the two are correlated and Spearman-Brown corrected. It is a property of the
neural data alone, so it is immune to every objection that attaches to the alignment statistic --
model choice, confound battery, readout, layer.

One caveat governs the reading. Reliability depends on how many repetitions can be averaged, and
the cohorts differ sharply in depth: LibriBrain is one listener over many sessions, the others are
shallow per subject. A cohort can therefore be near zero in every band for want of averaging
rather than for want of low-frequency structure. So the test is not whether the absolute values
match across cohorts, but whether the ORDERING over bands is the same wherever any reliability
exists at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
         "gamma": (30, 80), "broadband": (1, 80)}

R = Path("<PROJECT>")
COHORTS = {
    "Brennan": {"dir": R / "data/brennan_p4_v2", "sfreq": 500.0, "tmin": -0.1,
                "win": (0.30, 0.50), "unit": "subject"},
    "MEG-MASC": {"dir": R / "data/meg_masc_merged27", "sfreq": 250.0, "tmin": -0.2,
                 "win": (0.30, 0.50), "unit": "subject"},
    "LibriBrain": {"dir": R / "data/libribrain_full", "sfreq": 250.0, "tmin": -0.2,
                   "win": (0.30, 0.50), "unit": "session"},
}
MIN_REPS, MAX_WORDS, MIN_WORDS = 4, 600, 40


def bandpass(x, lo, hi, sf):
    nyq = 0.5 * sf
    b, a = butter(4, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def spearman_brown(r):
    return 2 * r / (1 + r) if r > -1 else float("nan")


def unit_reliability(ep, idx_by_word, tmask, sf, rng):
    """Split-half reliability of the word-level RDM, per band, for one unit."""
    words = sorted(idx_by_word)
    flat = np.array([i for w in words for i in idx_by_word[w]])
    X = np.asarray(ep[np.sort(flat)], dtype=np.float64)
    order = np.argsort(np.argsort(flat))          # restore per-word grouping
    X = X[order]
    bounds = np.cumsum([len(idx_by_word[w]) for w in words])[:-1]

    out = {}
    for band, (lo, hi) in BANDS.items():
        Xb = bandpass(X, lo, hi, sf)[:, :, tmask].mean(axis=2)
        A, B = [], []
        for g in np.split(Xb, bounds):
            k = len(g) // 2
            if k < 1:
                A.append(np.full(g.shape[1], np.nan))
                B.append(np.full(g.shape[1], np.nan))
                continue
            p = rng.permutation(len(g))
            A.append(g[p[:k]].mean(axis=0))
            B.append(g[p[k:2 * k]].mean(axis=0))
        A, B = np.array(A), np.array(B)
        ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        if ok.sum() < MIN_WORDS:
            out[band] = float("nan")
            continue
        da = np.nan_to_num(pdist(A[ok], metric="correlation"), nan=0.0)
        db = np.nan_to_num(pdist(B[ok], metric="correlation"), nan=0.0)
        out[band] = spearman_brown(float(spearmanr(rankdata(da), rankdata(db)).statistic))
    return out


def group_reliability_repeated(ep, per_unit_words, common, tmask, sf, rng, n_rep=20):
    """Repeat the across-unit split many times and summarise.

    A single random half-split is one draw from a noisy statistic, and the band ORDERING is what
    the claim rests on, so it has to be shown stable rather than assumed. Reports the median per
    band and how often the high bands beat the low ones across splits.
    """
    draws = []
    for _ in range(n_rep):
        draws.append(group_reliability(ep, per_unit_words, common, tmask, sf, rng))
    out = {}
    for band in BANDS:
        v = np.array([d[band] for d in draws], float)
        out[band] = {"median": float(np.nanmedian(v)), "sd": float(np.nanstd(v)),
                     "n_draws": int(np.isfinite(v).sum())}
    hi = np.array([np.nanmedian([d["beta"], d["gamma"]]) for d in draws])
    lo = np.array([np.nanmedian([d["delta"], d["theta"]]) for d in draws])
    out["_reversal"] = {"n_draws_high_beats_low": int((hi > lo).sum()), "n_draws": len(draws),
                        "median_high_minus_low": float(np.nanmedian(hi - lo))}
    return out


def group_reliability(ep, per_unit_words, common, tmask, sf, rng):
    """Split-half reliability ACROSS units, per band.

    The right level for a shallow cohort. Brennan subjects hear the story once, so no single
    subject has the repetitions for a stable word-level matrix -- within-subject reliability is
    ~0 in every band, low and high alike, which says nothing about frequency. Averaging half the
    units against the other half is how the cohort's own signal is recoverable at all (the paper
    reports 0.69 across Brennan subjects), and it is the level at which the band question can
    actually be asked here.
    """
    units = sorted(per_unit_words)
    rng.shuffle(units)
    h1, h2 = units[: len(units) // 2], units[len(units) // 2:]
    out = {}
    for band, (lo, hi) in BANDS.items():
        halves = []
        for grp in (h1, h2):
            pats = []
            for w in common:
                acc = []
                for u in grp:
                    idx = per_unit_words[u].get(w)
                    if idx is None:
                        continue
                    X = np.asarray(ep[np.sort(idx)], dtype=np.float64)
                    acc.append(bandpass(X, lo, hi, sf)[:, :, tmask].mean(axis=2).mean(axis=0))
                pats.append(np.mean(acc, axis=0) if acc else np.full(ep.shape[1], np.nan))
            halves.append(np.array(pats))
        A, B = halves
        ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        if ok.sum() < MIN_WORDS:
            out[band] = float("nan"); continue
        da = np.nan_to_num(pdist(A[ok], metric="correlation"), nan=0.0)
        db = np.nan_to_num(pdist(B[ok], metric="correlation"), nan=0.0)
        out[band] = spearman_brown(float(spearmanr(rankdata(da), rankdata(db)).statistic))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--across-units", action="store_true",
                    help="split-half across units instead of within each unit")
    ap.add_argument("--cohort", required=True, choices=list(COHORTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-units", type=int, default=0)
    a = ap.parse_args()

    cfg = COHORTS[a.cohort]
    meta = pd.read_csv(cfg["dir"] / "trial_metadata.tsv", sep="\t")
    ep = np.load(cfg["dir"] / "epochs_data.npy", mmap_mode="r")
    times = cfg["tmin"] + np.arange(ep.shape[2]) / cfg["sfreq"]
    tmask = (times >= cfg["win"][0]) & (times <= cfg["win"][1])
    wcol = next(c for c in ("word_text", "word", "stimulus") if c in meta.columns)
    ucol = "subject" if cfg["unit"] == "subject" else (
        "sess_id" if "sess_id" in meta.columns else "session")
    rng = np.random.default_rng(0)

    if a.across_units:
        per_unit, counter = {}, {}
        for u in sorted(meta[ucol].unique()):
            sm = meta[meta[ucol] == u]
            d = {}
            for w, g in sm.groupby(sm[wcol].astype(str).str.lower()):
                if w.strip():
                    d[w] = g.index.to_numpy()
                    counter[w] = counter.get(w, 0) + 1
            per_unit[u] = d
        n_u = len(per_unit)
        common = sorted([w for w, c in counter.items() if c == n_u])[:MAX_WORDS]
        print(f"[across] {n_u} units, {len(common)} words shared by all", flush=True)
        rep = group_reliability_repeated(ep, per_unit, common, tmask, cfg["sfreq"], rng)
        r = {b: rep[b]["median"] for b in BANDS}
        out = {"cohort": a.cohort, "mode": "across_units_repeated", "n_units": n_u,
               "repeated": rep,
               "n_words": len(common), "window_s": list(cfg["win"]),
               "per_band": {b: float(r[b]) for b in BANDS}}
        lo_ = np.nanmedian([r["delta"], r["theta"]]); hi_ = np.nanmedian([r["beta"], r["gamma"]])
        out["low_median"], out["high_median"] = float(lo_), float(hi_)
        out["low_minus_high"] = float(lo_ - hi_)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print("  " + "  ".join(f"{b}={r[b]:+.4f}" for b in BANDS))
        print(f"low(delta,theta) {lo_:+.4f}  high(beta,gamma) {hi_:+.4f}  diff {lo_-hi_:+.4f}")
        print(f"wrote {a.out}")
        return

    units = sorted(meta[ucol].unique())
    if a.max_units:
        units = units[:a.max_units]
    rows = []
    for u in units:
        sm = meta[meta[ucol] == u]
        by_word = {}
        for w, g in sm.groupby(sm[wcol].astype(str).str.lower()):
            if len(g) >= MIN_REPS and w.strip():
                by_word[w] = g.index.to_numpy()
        if len(by_word) < MIN_WORDS:
            continue
        by_word = {w: by_word[w] for w in sorted(by_word)[:MAX_WORDS]}
        r = unit_reliability(ep, by_word, tmask, cfg["sfreq"], rng)
        r.update({"unit": str(u), "n_words": len(by_word)})
        rows.append(r)
        print(f"  {u}: " + "  ".join(f"{b}={r[b]:+.3f}" for b in BANDS), flush=True)

    df = pd.DataFrame(rows)
    summ = {b: {"median": float(np.nanmedian(df[b])), "mean": float(np.nanmean(df[b])),
                "n_units": int(df[b].notna().sum()),
                "n_positive": int((df[b] > 0).sum())} for b in BANDS}
    low = np.nanmedian(df[["delta", "theta"]].to_numpy())
    high = np.nanmedian(df[["beta", "gamma"]].to_numpy())
    out = {"cohort": a.cohort, "window_s": list(cfg["win"]), "n_units": int(len(df)),
           "per_band": summ, "low_median": float(low), "high_median": float(high),
           "low_minus_high": float(low - high)}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n{a.cohort}: " + "  ".join(f"{b}={summ[b]['median']:+.4f}" for b in BANDS))
    print(f"low(delta,theta) {low:+.4f}  high(beta,gamma) {high:+.4f}  diff {low-high:+.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
