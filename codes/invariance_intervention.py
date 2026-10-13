#!/usr/bin/env python
"""RSA-only sensitivity to feature geometry, using a frozen reference ranking.

This analysis recomputes RSA after coordinate rescaling or additive rank-one noise.
It does not recompute encoding, and therefore cannot test encoding invariance or
exclude geometry as an explanation for cross-readout disagreement. Standardized
ridge is invariant to positive coordinate rescaling, but not generally to additive
sample-varying rank-one terms. See rogue_counterexample.py.

The original invariance_intervention.json is retained unchanged for provenance.
New outputs use rsa_geometry_sensitivity.json and explicit scope metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = Path("<PROJECT>")
sys.path.insert(0, str(R / "code"))
from protocol import (confound_basis, length_rdm, logfreq_rdm,          # noqa: E402
                      ortho_rdm, partial_spearman, rdm_from_feat)

CACHE = R / "data/libribrain_full/cache/rdm_libribrain_path1"
HID = R / "results/path1_libribrain"
DOSES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
SPREADS = [0.0, 0.5, 1.0, 2.0]


def inject_rogue(X, share, rng):
    """Add one direction carrying `share` of the total variance of the result."""
    if share <= 0:
        return X
    Xc = X - X.mean(0, keepdims=True)
    v = rng.normal(size=X.shape[1])
    v /= np.linalg.norm(v)
    a = rng.normal(size=(X.shape[0], 1))
    a -= a.mean()
    base = (Xc ** 2).sum()
    # solve for the scale s such that the added component holds `share` of total variance
    s = np.sqrt(share / (1 - share) * base / max((a ** 2).sum(), 1e-12))
    return Xc + s * a * v[None, :]


def rescale_axes(X, spread, rng):
    """Multiply each coordinate by a lognormal factor. Standardized ridge cannot see this."""
    if spread <= 0:
        return X
    return X * np.exp(rng.normal(0, spread, size=X.shape[1]))[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-frac", type=float, default=0.5)
    ap.add_argument("--out", default=str(R / "results/p4_encoding/rsa_geometry_sensitivity.json"))
    a = ap.parse_args()

    words = (CACHE / "word_list_shared.txt").read_text().split()
    subs = {int(p.stem.split("-")[1]): np.load(p).astype(float)
            for p in sorted(CACHE.glob("sub-*.npy"))}
    c2, c2src = logfreq_rdm(words)
    conf = [length_rdm(words), c2, ortho_rdm(words)]
    basis = confound_basis(conf, len(subs[next(iter(subs))]))
    enc_ref = {r["model"]: r["enc_mean"]
               for r in json.load(open(R / "results/p4_encoding/concordance32_CORRECTED.json"))["per_model"]}
    print(f"{len(subs)} groups, {len(words)} words, C2 = {c2src}", flush=True)

    feats = {}
    for d in sorted(HID.iterdir()):
        hp, mp = d / "hiddens.npy", d / "meta.json"
        if not (hp.exists() and mp.exists()):
            continue
        model = json.loads(mp.read_text())["model"]
        if model not in enc_ref:
            continue
        H = np.load(hp)
        if H.shape[1] != len(words):
            continue
        feats[model] = np.asarray(H[int(round(a.layer_frac * (H.shape[0] - 1)))], dtype=np.float64)
    models = sorted(feats)
    print(f"{len(models)} models with stored hidden states", flush=True)

    def rsa_scores(transform):
        out = []
        for m in models:
            X = transform(feats[m])
            dm = rdm_from_feat(X)
            out.append(np.mean([partial_spearman(subs[s], dm, basis=basis) for s in sorted(subs)]))
        return np.array(out)

    base_rsa = rsa_scores(lambda X: X)
    enc = np.array([enc_ref[m] for m in models])
    print(f"undistorted: RSA-vs-encoding concordance = "
          f"{spearmanr(base_rsa, enc).statistic:+.3f}", flush=True)

    res = {"n_models": len(models), "models": models,
           "baseline_concordance": float(spearmanr(base_rsa, enc).statistic),
           "rogue": [], "rescale": []}

    for share in DOSES:
        rng = np.random.default_rng(0)
        v = rsa_scores(lambda X: inject_rogue(X, share, rng))
        res["rogue"].append({
            "variance_share": share,
            "rank_corr_with_undistorted_rsa": float(spearmanr(v, base_rsa).statistic),
            "concordance_with_encoding": float(spearmanr(v, enc).statistic)})
        r = res["rogue"][-1]
        print(f"  rogue {share:.0%}: RSA leaderboard vs original "
              f"{r['rank_corr_with_undistorted_rsa']:+.3f}, "
              f"concordance {r['concordance_with_encoding']:+.3f}", flush=True)

    for spread in SPREADS:
        rng = np.random.default_rng(1)
        v = rsa_scores(lambda X: rescale_axes(X, spread, rng))
        res["rescale"].append({
            "log_spread": spread,
            "rank_corr_with_undistorted_rsa": float(spearmanr(v, base_rsa).statistic),
            "concordance_with_encoding": float(spearmanr(v, enc).statistic)})
        r = res["rescale"][-1]
        print(f"  rescale sd={spread}: RSA leaderboard vs original "
              f"{r['rank_corr_with_undistorted_rsa']:+.3f}, "
              f"concordance {r['concordance_with_encoding']:+.3f}", flush=True)

    res["analysis_scope"] = "RSA sensitivity against a frozen corrected encoding reference"
    res["encoding_recomputed"] = False
    res["encoding_reference"] = "concordance32_CORRECTED.json"
    res["interpretation"] = (
        "Coordinate standardization removes positive axis rescaling; it does not "
        "guarantee invariance to sample-varying additive rank-one terms. Encoding "
        "was not recomputed here. Correlation against its frozen ranking is a "
        "descriptive sensitivity and does not identify or exclude a mechanism.")
    cd = [r["concordance_with_encoding"] for r in res["rogue"]]
    res["concordance_range_across_doses"] = [float(min(cd)), float(max(cd))]
    res["verdict"] = "RSA-ONLY SENSITIVITY; transformed-input encoding remains untested"
    print(f"\n{res['verdict']}")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
