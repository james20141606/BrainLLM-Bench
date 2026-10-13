#!/usr/bin/env python
"""The benchmark protocol, in one place.

Until now `bench/run.py` was a reference baseline and said so: it took a crude channel slice as
a "region", correlated two RDMs with a plain Spearman, and had no band-splitting, no confound
partialization and no ceiling normalization. Its numbers were therefore not comparable to the
paper's cells, which a reviewer rightly called the benchmark's weakest point, since a benchmark
whose protocol cannot be run is not a benchmark.

This module implements the full chain the paper actually uses:

    band-split -> region select -> time window -> neural RDM
                                                    |
    LLM layer features -> LLM RDM                   |
                                                    v
                          confound partialization (C1..C4) -> partial Spearman
                                                    |
                                                    v
                                  per-band reliability ceiling -> normalized score

The confound family is word-level, which is the correct family for a word-level RSA:

    C1  word length            |len_i - len_j|
    C2  log word frequency     |zipf_i - zipf_j|      (wordfreq; inert and flagged if absent)
    C3  orthographic           Levenshtein(w_i, w_j)
    C4  static embedding       the model's own layer-0 RDM, which isolates the CONTEXTUAL
                               contribution. Per-model, and skipped when scoring layer 0
                               itself, where it would partial a vector against itself.

Partialling means: rank both sides and every confound, residualize the neural ranks and the LLM
ranks on [1, C1_rank .. C4_rank], then Pearson the residuals. Constant confounds are dropped
rather than allowed to make the design singular.

`--validate` is the reason to trust any of this. It recomputes the stored per-(model, subject)
partial correlations from the same cached inputs and fails on any mismatch, so the port is
checked against the numbers the paper reports rather than merely looking equivalent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

BANDS = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 80.0), "broadband": (1.0, 80.0),
}

# Regions are the paper's anatomical sensor wedges (frontal / temporal / parieto-occipital x
# left / right; the atlas modules leave the midline "central" sensors unassigned), plus "all",
# taken from the same atlas modules that built the
# stored results: codes/region_atlas.py for Neuromag-306 (LibriBrain, MEG-SCANS) and
# codes/region_atlas_kit.py for KIT-208 (MEG-MASC), in each cohort's canonical channel order. The
# stored receipts record the atlas they used ("neuromag306" / "kit208"). Scalp EEG is scored on
# all channels only. (Corrected 2026-09-24: an earlier version of this module split the channel
# INDEX array in half, which is not the paper's regioning; `--validate` starts from cached neural
# RDMs and therefore never exercised this map, so its 504/504 match did not cover it.)
_CODES = Path(__file__).resolve().parent.parent / "codes"


def _anatomical_regions(n_channels):
    if str(_CODES) not in sys.path:
        sys.path.insert(0, str(_CODES))
    if n_channels == 306:
        from region_atlas import get_canonical_neuromag306_order, get_neuromag_region_masks
        masks = get_neuromag_region_masks(get_canonical_neuromag306_order())
    elif n_channels == 208:
        from region_atlas_kit import get_canonical_kit208_order, get_kit_region_masks
        masks = get_kit_region_masks(get_canonical_kit208_order())
    else:
        return {"all": np.arange(n_channels)}
    out = {r: np.flatnonzero(m) for r, m in masks.items()}
    out["all"] = np.arange(n_channels)
    return out


REGIONS = {n: None for n in (306, 208, 61, 60)}   # filled lazily from the atlas modules


# ------------------------------------------------------------------ neural side

def bandpass(data, lo, hi, sf, order=4):
    """Zero-phase Butterworth bandpass along the time axis."""
    nyq = 0.5 * sf
    hi = min(hi, nyq * 0.99)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def region_channels(n_channels, region):
    if n_channels not in REGIONS:
        raise ValueError(f"no region map for {n_channels} channels; add one to REGIONS")
    if REGIONS[n_channels] is None:
        REGIONS[n_channels] = _anatomical_regions(n_channels)
    if region not in REGIONS[n_channels]:
        raise ValueError(f"unknown region {region!r} for {n_channels} channels")
    return REGIONS[n_channels][region]


def shared_word_list(words, subjects, max_words=2000):
    """Words every subject heard, in deterministic order.

    The RDM row order has to be identical across subjects and identical to the LLM feature
    order, so the probe set is the intersection over subjects rather than each subject's own
    vocabulary.
    """
    norm = lambda w: str(w).strip().lower()
    per_subj = {}
    for i, (w, s) in enumerate(zip(words, subjects)):
        w = norm(w)
        if w:
            per_subj.setdefault(s, {}).setdefault(w, []).append(i)
    if not per_subj:
        return [], {}
    common = sorted(set.intersection(*(set(d) for d in per_subj.values())))[:max_words]
    return common, per_subj


def neural_rdm(epochs, *, region, band, time_s, sfreq, tmin, halfwidth_s=0.020,
               word_index=None):
    """Band-split, region-select and time-window a (trial, channel, time) array into a
    rank-transformed condensed RDM. The neural half of the protocol.

    `word_index` is a list of trial-index lists, one per probe word. When given, each word's
    repetitions are averaged before the RDM is formed. That averaging is not cosmetic: on these
    cohorts the single-trial RDM has essentially zero split-half reliability, so a protocol that
    skips it is correlating against noise. When omitted, rows are trials.
    """
    lo, hi = BANDS[band]
    chans = region_channels(epochs.shape[1], region)
    flat = (np.asarray([i for g in word_index for i in g]) if word_index is not None
            else np.arange(epochs.shape[0]))
    X = np.asarray(epochs[flat][:, chans, :], dtype=np.float64)
    X = bandpass(X, lo, hi, sfreq)
    n_t = X.shape[2]
    times = tmin + np.arange(n_t) / sfreq
    m = (times >= time_s - halfwidth_s) & (times <= time_s + halfwidth_s)
    if not m.any():
        raise ValueError(f"time {time_s}s outside epoch [{times[0]:.3f}, {times[-1]:.3f}]")
    feat = np.nan_to_num(X[:, :, m].mean(axis=2))
    if word_index is not None:
        bounds = np.cumsum([len(g) for g in word_index])[:-1]
        feat = np.array([g.mean(axis=0) for g in np.split(feat, bounds)])
    d = np.nan_to_num(pdist(feat, metric="correlation"), nan=0.0)
    return rankdata(d)


# ------------------------------------------------------------------ confounds

def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def length_rdm(words):
    L = np.array([len(w) for w in words], dtype=float)
    return pdist(L[:, None], metric="cityblock")


def logfreq_rdm(words):
    """(condensed RDM, source). Inert and flagged when `wordfreq` is unavailable, rather than
    silently substituting a proxy that would look like a control without being one."""
    try:
        from wordfreq import zipf_frequency
        z = np.array([zipf_frequency(w, "en") for w in words], dtype=float)
        src = "wordfreq.zipf_frequency(en)"
    except Exception:
        z = np.zeros(len(words), dtype=float)
        src = "UNAVAILABLE(wordfreq) -> C2 inert"
    return pdist(z[:, None], metric="cityblock"), src


def ortho_rdm(words):
    n = len(words)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = levenshtein(words[i], words[j])
    return squareform(D, checks=False)


def rdm_from_feat(feat):
    return np.nan_to_num(pdist(feat, metric="correlation"), nan=0.0)


# ------------------------------------------------------------------ statistic

def _resid(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def confound_basis(confound_ds, m):
    """Orthonormal basis for the confound design [1, ranked confounds].

    The design does not change across layers or subjects, so callers scoring many of either
    should build this once. Residualizing against Q is equivalent to the least-squares fit but
    avoids re-solving the same system thousands of times, which dominated the runtime.
    Constant confounds are dropped so the design cannot go singular.
    """
    cols = [np.ones(m)]
    for cd in confound_ds:
        if cd is None:
            continue
        cr = rankdata(cd)
        if np.ptp(cr) == 0:
            continue
        cols.append(cr)
    Q, _ = np.linalg.qr(np.vstack(cols).T)
    return Q


def _resid_q(y, Q):
    return y - Q @ (Q.T @ y)


def partial_spearman(neural_ranks, llm_d, confound_ds=None, *, basis=None):
    """Partial Spearman of a pre-ranked neural RDM against an unranked LLM RDM.

    Pass `basis` (from `confound_basis`) to reuse a design across calls; otherwise it is built
    from `confound_ds` each time.
    """
    y = np.asarray(neural_ranks, float)
    Q = basis if basis is not None else confound_basis(confound_ds, len(y))
    ny = _resid_q(y, Q)
    lx = _resid_q(rankdata(llm_d).astype(float), Q)
    denom = float(np.sqrt((ny @ ny) * (lx @ lx)))
    return float((ny @ lx) / denom) if denom > 0 else 0.0


def plain_spearman(neural_ranks, llm_d):
    a = np.asarray(neural_ranks, float) - np.mean(neural_ranks)
    b = rankdata(llm_d) - np.mean(rankdata(llm_d))
    denom = float(np.sqrt((a @ a) * (b @ b)))
    return float((a @ b) / denom) if denom > 0 else 0.0


def ceiling_for(reliability_tsv, region, time_ms):
    rel = pd.read_csv(reliability_tsv, sep="\t")
    cell = rel[(rel["region"] == region) & (np.isclose(rel["time_ms"], time_ms))]
    return float(cell["mean_rho"].iloc[0]) if len(cell) else float("nan")


# ------------------------------------------------------------------ end to end

def score_model(neural_ranks, hiddens, words, *, ceiling=None, confounds=None,
                layer_rdms=None, basis=None):
    """Score one model against one pre-ranked neural RDM over all its layers.

    hiddens is (n_layers+1, n_words, d). Returns the best layer by partial correlation, which is
    the paper's selection rule, along with the raw correlation at that layer and the
    ceiling-normalized score.
    """
    if confounds is None:
        c1 = length_rdm(words)
        c2, _ = logfreq_rdm(words)
        confounds = [c1, c2, ortho_rdm(words)]
    # LLM RDMs do not depend on the subject, so let callers scoring many subjects against one
    # model pass them in rather than rebuilding them per subject.
    if layer_rdms is None:
        layer_rdms = [rdm_from_feat(h) for h in hiddens]
    c4 = layer_rdms[0]
    n_layers = len(layer_rdms)
    m = len(neural_ranks)
    # Two designs only: without C4 (layer 0) and with it (every other layer). Neither depends on
    # the layer or the subject, so both are built once here.
    if basis is None:
        basis = (confound_basis(list(confounds), m),
                 confound_basis(list(confounds) + [c4], m))
    raw = np.empty(n_layers)
    par = np.empty(n_layers)
    for li in range(n_layers):
        d = layer_rdms[li]
        raw[li] = plain_spearman(neural_ranks, d)
        par[li] = partial_spearman(neural_ranks, d, basis=basis[1 if li else 0])
    bl = int(np.argmax(par))
    return {"best_layer": bl, "raw_rho": float(raw[bl]), "partial_rho": float(par[bl]),
            "ceiling": ceiling,
            "partial_norm": (float(par[bl] / ceiling)
                             if ceiling and np.isfinite(ceiling) else float("nan"))}


def selftest(seed=0, n=200, tol=1e-10):
    """Check the statistic against independent routes, on data with known answers.

    The cluster validation (`--validate`) is the strong check but needs cohort data. This runs
    anywhere, so CI can catch a regression in the partialization:

      1. an independently coded partial correlation (explicit pseudo-inverse on the raw design,
         no QR) must agree with the shipped QR path;
      2. partialling against a confound identical to the LLM RDM must give exactly zero;
      3. partialling against a constant (degenerate) confound must reduce to plain Spearman,
         since a constant column carries no information and is dropped.
    """
    rng = np.random.default_rng(seed)
    neural = rankdata(rng.normal(size=n))
    llm = rng.normal(size=n)
    c = [rng.normal(size=n), rng.normal(size=n)]

    # (1) least-squares residualization, independent of the shipped QR path
    X = np.vstack([np.ones(n)] + [rankdata(x) for x in c]).T
    ny, lx = _resid(neural.astype(float), X), _resid(rankdata(llm).astype(float), X)
    want = float((ny @ lx) / np.sqrt((ny @ ny) * (lx @ lx)))
    got = partial_spearman(neural, llm, c)
    assert abs(got - want) < tol, f"(1) partial mismatch: {got} vs {want}"

    # (2) textbook one-confound identity, which shares no code with either route:
    #     r_xy.z = (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    z = c[0]
    rxy = plain_spearman(neural, llm)
    rxz = plain_spearman(neural, z)
    ryz = float(np.corrcoef(rankdata(llm), rankdata(z))[0, 1])
    want2 = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    got2 = partial_spearman(neural, llm, [z])
    assert abs(got2 - want2) < 1e-9, f"(2) one-confound identity: {got2} vs {want2}"

    # (3) constant confound is dropped, so this is plain Spearman
    got3 = partial_spearman(neural, llm, [np.ones(n)])
    want3 = plain_spearman(neural, llm)
    assert abs(got3 - want3) < tol, f"(3) constant confound changed the result: {got3} vs {want3}"

    # basis reuse must match the from-scratch path
    b = confound_basis(c, n)
    got4 = partial_spearman(neural, llm, basis=b)
    assert abs(got4 - got) < tol, f"(4) basis reuse diverged: {got4} vs {got}"

    print(f"[selftest] OK  partial={got:+.6f}  one-confound identity={got2:+.6f}  "
          f"constant-confound==plain={want3:+.6f}")
    return True


def validate(rdm_cache, hidden_root, stored_tsv, tol=1e-9):
    """Recompute the stored per-(model, subject) numbers and fail on any mismatch."""
    rdm_cache, hidden_root = Path(rdm_cache), Path(hidden_root)
    words = (rdm_cache / "word_list_shared.txt").read_text().split()
    subs = {int(p.stem.split("-")[1]): np.load(p).astype(float)
            for p in sorted(rdm_cache.glob("sub-*.npy"))}
    ref = pd.read_csv(stored_tsv, sep="\t")
    c1 = length_rdm(words)
    c2, c2_src = logfreq_rdm(words)
    conf = [c1, c2, ortho_rdm(words)]
    print(f"[validate] {len(subs)} subjects, {len(words)} words, C2 = {c2_src}")

    n_ok = n_bad = 0
    for tag, g in ref.groupby("tag"):
        hp = hidden_root / tag / "hiddens.npy"
        if not hp.exists():
            continue
        hiddens = np.load(hp)
        if hiddens.shape[1] != len(words):
            continue
        layer_rdms = [rdm_from_feat(h) for h in hiddens]
        m = len(layer_rdms[0])
        basis = (confound_basis(list(conf), m),
                 confound_basis(list(conf) + [layer_rdms[0]], m))
        for _, r in g.iterrows():
            subj = int(r["subject"])
            if subj not in subs:
                continue
            got = score_model(subs[subj], hiddens, words, ceiling=float(r["ceiling"]),
                              confounds=conf, layer_rdms=layer_rdms, basis=basis)
            for k in ("best_layer", "raw_rho", "partial_rho"):
                exp = r[k]
                d = abs(got[k] - exp)
                if d > (0 if k == "best_layer" else tol):
                    print(f"  MISMATCH {tag} sub-{subj} {k}: got {got[k]!r} want {exp!r} "
                          f"(|d|={d:.3g})")
                    n_bad += 1
                else:
                    n_ok += 1
    print(f"\n[validate] {n_ok} values match, {n_bad} mismatch")
    return n_bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="known-answer checks on the statistic; needs no cohort data")
    ap.add_argument("--validate", action="store_true",
                    help="recompute stored numbers and exit non-zero on mismatch")
    ap.add_argument("--rdm-cache", help="dir of sub-*.npy plus word_list_shared.txt")
    ap.add_argument("--hidden-root", help="dir of <tag>/hiddens.npy")
    ap.add_argument("--stored", help="reference rsa_confound_per_model_subject.tsv")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.validate:
        if not all([a.rdm_cache, a.hidden_root, a.stored]):
            sys.exit("--validate needs --rdm-cache, --hidden-root and --stored")
        sys.exit(0 if validate(a.rdm_cache, a.hidden_root, a.stored) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
