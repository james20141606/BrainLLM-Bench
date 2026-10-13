#!/usr/bin/env python
"""Score 32 models under RSA and encoding on ONE shared word set, and decide H1.

Pre-registration: paper4_temporalalign/CONCORDANCE_PREREG.md.

What this fixes relative to the pilot (see pre-reg section 2b): the pilot's
rho = 0.60 compared an RSA computed at each model's BEST layer on a 462-word
cache against an encoding computed at the middle of an arbitrary 3-6 layer cache
on 177,127 words. Three differences rode along with the readout. Here both
readouts see:
  * the same words   -- one per-session subsample under a fixed seed
  * the same layer   -- the pre-registered 0.5 relative depth
  * the same sessions, region, band and window

Precise about what still differs, because "only the readout differs" would be an
overstatement: within that shared cell each readout uses its own conventional
neural representation -- RSA the flattened channel x time pattern
(compute_rdms.py:178), encoding the time-averaged channel vector
(p4_encoding_families_lb.py:40). That is not a confound to remove; it is part of
what each readout IS in the literature being compared. Forcing both onto one
representation would test something nobody runs. What the pilot confounded, and
this fixes, is the words, the layer rule and the cache.

Outputs results/p4_encoding/concordance32.json with the per-model scores, each
readout's split-half ranking reliability, the attenuation ceiling, and the H1
verdict computed by the pre-registered rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

R = Path("<PROJECT>")
sys.path.insert(0, str(R / "code"))
sys.path.insert(0, str(R / "code" / "paper4_nn_revision"))

from compute_rdms import FREQ_BANDS, bandpass_filter            # noqa: E402
from region_atlas import (                                       # noqa: E402
    get_canonical_neuromag306_order,
    get_neuromag_region_masks,
)
from path1_confound_rsa import (                                 # noqa: E402
    length_rdm, logfreq_rdm, ortho_rdm, rdm_from_feat, partial_spearman,
)

DATA = R / "data" / "libribrain_full"
FEAT = R / "data" / "preprocessed" / "llm_embeddings_fd"
OUT = R / "results" / "p4_encoding" / "concordance32.json"

# The pre-registered fixed cell (pre-reg section 5). Each cohort uses its OWN
# headline cell -- a per-cohort anatomical choice fixed in advance, NOT a
# per-model argmax (which is what the pilot confounded). LibriBrain: temporal_L
# ~350 ms. MEG-MASC: temporal_R, and its 208-channel KIT atlas, not Neuromag-306.
REGION = "temporal_L"
BAND = "broadband"
WIN_MS = (300.0, 400.0)      # window containing the 350 ms headline latency
ATLAS = "neuromag306"        # or "kit208" for MEG-MASC
GROUP_COL = "session"        # or "subject" for MEG-MASC
TMIN, SFREQ = -0.2, 250.0
ALPHAS = np.logspace(1, 6, 8)
SEED = 42


def load_models():
    from extract_llm_layers_multi import MODEL_SET
    return MODEL_SET


def mid_layer_path(model: str, feat_dir: Path) -> Path:
    d = feat_dir / model.replace("/", "_")
    lf = sorted(d.glob("layer_*.npy"),
                key=lambda p: int(p.stem.split("_")[1]))
    if len(lf) != 3:
        raise FileNotFoundError(f"{model}: expected 3 depths, found {len(lf)}")
    return lf[1]          # 0.25 / 0.5 / 0.75 -> index 1 is the 0.5 primary


def subsample(meta: pd.DataFrame, max_per_session: int, seed: int,
              group_col: str):
    """One shared word subsample per group, fixed seed. Both readouts use it."""
    rng = np.random.RandomState(seed)
    keep = []
    for s, idx in meta.groupby(group_col).indices.items():
        idx = np.asarray(idx)
        if len(idx) > max_per_session:
            idx = np.sort(rng.choice(idx, max_per_session, replace=False))
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def neural_patterns(ep, sel, mask, s0, s1, band):
    x = np.asarray(ep[sel], dtype=np.float32)
    if band != "broadband":
        fmin, fmax = FREQ_BANDS[band]
        x = bandpass_filter(x, SFREQ, fmin, fmax)
    x = x[:, mask, s0:s1]
    return x.reshape(len(x), -1)


def rsa_per_session(pat, feat, words, meta_sel, confounds, group_col):
    """Confound-controlled partial Spearman, per group (session or subject)."""
    out = {}
    for s in np.unique(meta_sel[group_col].values):
        m = meta_sel[group_col].values == s
        if m.sum() < 50:
            continue
        nd = dist.pdist(pat[m], metric="correlation")
        ld = dist.pdist(feat[m], metric="cosine")
        good = np.isfinite(nd) & np.isfinite(ld)
        nr = rankdata(nd[good])
        cds = [c[good] for c in confounds[s]]
        out[s] = float(partial_spearman(nr, ld[good], cds))
    return out


def encoding_per_session(X, Y, groups):
    # Ridge is not scale-invariant against a fixed alpha grid, and our neural archives differ in
    # absolute units by five orders of magnitude (rebuilt MEG-MASC ~1e-13 against the original
    # ~1e-8). At 1e-13 every alpha swamps the data, RidgeCV pins 1e6, predictions collapse to the
    # training mean, and every correlation is exactly zero. Normalising Y to unit variance makes
    # the penalty comparable across archives; the reported per-column Pearson correlation is
    # invariant to the rescaling, so well-scaled data are unaffected.
    yscale = float(Y.std())
    if yscale > 0:
        Y = Y / yscale
    logo = LeaveOneGroupOut()
    out = {}
    for tr, te in logo.split(X, Y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        ym = Y[tr].mean(0)
        m = RidgeCV(alphas=ALPHAS)
        m.fit((X[tr] - mu) / sd, Y[tr] - ym)
        P = m.predict((X[te] - mu) / sd) + ym
        Yt = Y[te]
        Yc = Yt - Yt.mean(0, keepdims=True)
        Pc = P - P.mean(0, keepdims=True)
        num = (Yc * Pc).sum(0)
        den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0)) + 1e-12
        out[groups[te][0]] = float((num / den).mean())
    return out


def splithalf_rank_reliability(per_sess: dict, n_iter=200, seed=0):
    """Split sessions in half, rank models in each half, correlate. Median + 90% CI.

    This is the r_xx that sets the attenuation ceiling. It must be computed on the
    SAME sessions for both readouts or the ceiling is not comparable.
    """
    models = sorted(per_sess)
    sess = sorted(set.intersection(*[set(per_sess[m]) for m in models]))
    if len(sess) < 4 or len(models) < 3:
        return None, None
    rng = np.random.RandomState(seed)
    rs = []
    for _ in range(n_iter):
        perm = rng.permutation(sess)
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        a = [np.mean([per_sess[m][s] for s in h1]) for m in models]
        b = [np.mean([per_sess[m][s] for s in h2]) for m in models]
        r, _ = spearmanr(a, b)
        if np.isfinite(r):
            rs.append(r)
    rs = np.array(rs)
    # Spearman-Brown: split-half underestimates full-set reliability
    sb = 2 * rs / (1 + rs)
    return float(np.median(sb)), [float(np.percentile(sb, 5)),
                                  float(np.percentile(sb, 95))]


def fisher_ci(r, n, alpha=0.05):
    from scipy.stats import norm
    if n < 4:
        return [float("nan"), float("nan")]
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return [float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_per_session", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--data_dir", default=str(DATA))
    ap.add_argument("--feat_dir", default=str(FEAT))
    ap.add_argument("--emb_idx", default=None,
                    help="npy mapping each trial to its row in the feature matrices "
                         "(for epoch sets not row-aligned to the features)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--partial_out", default=None,
                    help="write this shard's per-session results here and skip aggregation")
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--atlas", default=ATLAS,
                    choices=["neuromag306", "kit208", "eeg60"])
    ap.add_argument("--tmin", type=float, default=TMIN,
                    help="epoch start (s). Brennan EEG = -0.1 (vs MEG -0.2).")
    ap.add_argument("--sfreq", type=float, default=SFREQ,
                    help="sample rate. Brennan EEG = 500 (vs MEG 250).")
    ap.add_argument("--group_col", default=GROUP_COL,
                    help="session for LibriBrain (single subject), subject for "
                         "MEG-MASC")
    a = ap.parse_args()
    feat_dir = Path(a.feat_dir)
    data_dir = Path(a.data_dir)

    MODEL_SET = load_models()
    if a.nshards > 1:
        _all = list(MODEL_SET.items())
        MODEL_SET = dict(_all[a.shard::a.nshards])
        print(f"[shard {a.shard}/{a.nshards}] {len(MODEL_SET)} of {len(_all)} models: "
              f"{list(MODEL_SET)}", flush=True)
    meta = pd.read_csv(data_dir / "trial_metadata.tsv", sep="\t")
    sel = subsample(meta, a.max_per_session, a.seed, a.group_col)
    meta_sel = meta.iloc[sel].reset_index(drop=True)
    words = meta_sel["word_text"].astype(str).tolist()
    groups = meta_sel[a.group_col].to_numpy()
    print(f"[shared subsample] {len(sel)} words, "
          f"{len(np.unique(groups))} {a.group_col} groups", flush=True)

    if a.emb_idx:
        _embidx = np.load(a.emb_idx)
        if len(_embidx) != len(meta):
            raise SystemExit(f"--emb_idx has {len(_embidx)} rows, metadata has {len(meta)}")
        emb_rows = _embidx[sel]
        print(f"[emb_idx] indexing features through {a.emb_idx}", flush=True)
    else:
        emb_rows = sel

    ep = np.load(data_dir / "epochs_data.npy", mmap_mode="r")
    if a.atlas == "eeg60":
        # Brennan scalp EEG (easycap M10, 60 channels): no MEG sensor-space
        # region structure, so the headline cell is ALL channels.
        n_ch = int(ep.shape[1])
        mask = np.ones(n_ch, dtype=bool)
    elif a.atlas == "kit208":
        from region_atlas_kit import (get_canonical_kit208_order,
                                       get_kit_region_masks)
        order = get_canonical_kit208_order()
        masks = get_kit_region_masks(order)
        mask = masks[a.region]
    else:
        order = get_canonical_neuromag306_order()
        masks = get_neuromag_region_masks(order)
        mask = masks[a.region]
    s0 = int(round((WIN_MS[0] / 1000.0 - a.tmin) * a.sfreq))
    s1 = int(round((WIN_MS[1] / 1000.0 - a.tmin) * a.sfreq))
    pat = neural_patterns(ep, sel, mask, s0, s1, BAND)
    print(f"[neural] {pat.shape} at {a.region}/{BAND}/{WIN_MS} ms", flush=True)

    # Encoding target: same cell, mean over the window.
    Y = pat.reshape(len(pat), int(mask.sum()), s1 - s0).mean(axis=2)

    # Confound RDMs per session, built on the SAME shared subsample.
    # logfreq_rdm returns (rdm, source) and falls back to an all-zero vector when
    # wordfreq is missing -- and partial_spearman SKIPS constant confounds, so a
    # missing wordfreq would silently drop the frequency control and leave a
    # "partial" correlation that partials nothing. Assert instead.
    confounds = {}
    for s in np.unique(groups):
        m = groups == s
        w = [words[i] for i in np.where(m)[0]]
        nd_freq, freq_src = logfreq_rdm(w)
        if "UNAVAILABLE" in freq_src or np.ptp(nd_freq) == 0:
            raise RuntimeError(
                f"session {s}: frequency confound is inert ({freq_src}); "
                "partial RSA would not control for word frequency")
        confounds[s] = [length_rdm(w), nd_freq, ortho_rdm(w)]
    print(f"[confounds] built for {len(confounds)} sessions "
          f"(frequency source: {freq_src})", flush=True)

    rsa_ps, enc_ps, rows = {}, {}, []
    for model, kind in MODEL_SET.items():
        try:
            F = np.load(mid_layer_path(model, feat_dir), mmap_mode="r")
            X = np.asarray(F[emb_rows], dtype=np.float64)
            r_rsa = rsa_per_session(pat, X, words, meta_sel, confounds,
                                    a.group_col)
            r_enc = encoding_per_session(X, Y.astype(np.float64), groups)
            rsa_ps[model], enc_ps[model] = r_rsa, r_enc
            rows.append({"model": model, "kind": kind,
                         "rsa_mean": float(np.mean(list(r_rsa.values()))),
                         "enc_mean": float(np.mean(list(r_enc.values())))})
            print(f"  {model:38s} {kind:7s} RSA={rows[-1]['rsa_mean']:+.5f} "
                  f"ENC={rows[-1]['enc_mean']:+.5f}", flush=True)
        except Exception as e:
            print(f"  [FAIL] {model}: {type(e).__name__}: {e}", flush=True)

    if a.partial_out:
        import json as _json
        _ser = lambda d: {m: {str(k): float(v) for k, v in s.items()} for m, s in d.items()}
        Path(a.partial_out).write_text(_json.dumps(
            {"rows": rows, "rsa_ps": _ser(rsa_ps), "enc_ps": _ser(enc_ps)}, indent=1))
        print(f"[shard {a.shard}/{a.nshards}] wrote {len(rows)} models -> {a.partial_out}",
              flush=True)
        return

    models = [r["model"] for r in rows]
    rsa_v = [r["rsa_mean"] for r in rows]
    enc_v = [r["enc_mean"] for r in rows]

    r_xx, ci_xx = splithalf_rank_reliability(rsa_ps)
    r_yy, ci_yy = splithalf_rank_reliability(enc_ps)
    rho, p = spearmanr(rsa_v, enc_v)
    ci = fisher_ci(rho, len(models))
    ceiling = float(np.sqrt(max(r_xx, 0) * max(r_yy, 0))) if (
        r_xx and r_yy) else float("nan")

    # Pre-registered rule (section 2): H1 is supported iff the CI on the observed
    # concordance excludes the measured ceiling.
    if not np.isfinite(ceiling) or np.isnan(ci[1]):
        verdict = "INDETERMINATE (ceiling or CI not estimable)"
    elif ci[1] < ceiling:
        verdict = "H1 SUPPORTED (CI upper bound below ceiling)"
    else:
        verdict = "H1 FALSIFIED (CI includes the ceiling)"

    res = {"prereg": "paper4_temporalalign/CONCORDANCE_PREREG.md",
           "rsa_controls": ["word_length", "log_zipf_frequency", "orthographic_edit_distance"],
           "model_specific_static_embedding_control": False,
           "encoding_preprocessing": "train-fold feature standardization",
           "cell": {"region": a.region, "band": BAND, "window_ms": list(WIN_MS),
                    "depth_frac": 0.5, "atlas": a.atlas, "group_col": a.group_col},
           "shared_subsample": {"max_per_session": a.max_per_session,
                                "seed": a.seed, "n_words": int(len(sel)),
                                "n_sessions": int(len(np.unique(groups)))},
           "n_models": len(models), "models": models,
           "per_model": rows,
           "rsa_reliability_sb": r_xx, "rsa_reliability_ci90": ci_xx,
           "enc_reliability_sb": r_yy, "enc_reliability_ci90": ci_yy,
           "attenuation_ceiling": ceiling,
           "concordance_spearman": float(rho), "concordance_p": float(p),
           "concordance_ci95": ci,
           "attenuation_corrected": (float(rho / ceiling)
                                     if ceiling and ceiling > 0 else None),
           "H1_verdict": verdict,
           "per_session_rsa": {m: {str(k):v for k,v in rsa_ps[m].items()} for m in rsa_ps},
           "per_session_enc": {m: {str(k):v for k,v in enc_ps[m].items()} for m in enc_ps}}
    Path(a.out).write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 64)
    print(f"n models            : {len(models)}")
    print(f"RSA reliability     : {r_xx}  CI90 {ci_xx}")
    print(f"encoding reliability: {r_yy}  CI90 {ci_yy}")
    print(f"attenuation ceiling : {ceiling:.4f}")
    print(f"observed concordance: rho={rho:.4f}  95% CI {ci}")
    print(f"VERDICT             : {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    main()
