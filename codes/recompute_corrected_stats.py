#!/usr/bin/env python
"""Recompute every quantity downstream of the encoding arm under the corrected code.

Correcting the headline alone would leave the CI, the family-clustered interval and the depth curve
consistent-looking but no longer derived from it. This produces all of them from
`concordance32_CORRECTED.json`, using the same estimators as the originals so the corrected and
published numbers are like-for-like.

Family clustering resamples whole families rather than checkpoints, because 32 checkpoints are ~12
families and sibling checkpoints are not independent observations. The family map is taken from
codes/model_families.py where available, so this cannot drift from the paper's own definition.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "paper4_temporalalign" / "results" / "p4_encoding"


def family_map(models):
    """Prefer the repo's single source of truth; fall back to the documented rules."""
    try:
        sys.path.insert(0, str(ROOT / "codes"))
        from model_families import family              # noqa: F401
        return {m: family(m) for m in models}
    except Exception:
        hints = [("pythia", "pythia"), ("gpt-neo", "gpt-neo"), ("opt", "opt"), ("qwen", "qwen"),
                 ("gpt2", "gpt2"), ("albert", "bert"), ("distilbert", "bert"), ("bert", "bert"),
                 ("roberta", "roberta"), ("deberta", "deberta"), ("electra", "electra"),
                 ("bart", "bart"), ("t5", "t5"), ("llama", "llama")]
        out = {}
        for m in models:
            s = m.lower().split("/")[-1]
            out[m] = next((f for p, f in hints if p in s), s.split("-")[0])
        return out


def load(name):
    j = json.loads((RES / name).read_text())
    return j["per_session_rsa"], j["per_session_enc"]


def point(rsa_ps, enc_ps, models=None):
    ms = sorted(set(rsa_ps) & set(enc_ps)) if models is None else models
    sess = sorted(set.intersection(*[set(rsa_ps[m]) for m in ms]))
    a = [np.mean([rsa_ps[m][s] for s in sess]) for m in ms]
    b = [np.mean([enc_ps[m][s] for s in sess]) for m in ms]
    return spearmanr(a, b)[0], ms, sess


def fisher_ci(r, n, alpha=0.05):
    from scipy.stats import norm
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def family_clustered_ci(rsa_ps, enc_ps, n_boot=10000, seed=0):
    """Resample whole families with replacement; recompute the concordance each time."""
    rho, ms, sess = point(rsa_ps, enc_ps)
    fam = family_map(ms)
    fams = sorted(set(fam.values()))
    by_fam = {f: [m for m in ms if fam[m] == f] for f in fams}
    a = {m: np.mean([rsa_ps[m][s] for s in sess]) for m in ms}
    b = {m: np.mean([enc_ps[m][s] for s in sess]) for m in ms}
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(fams), len(fams), replace=True)
        sel = [m for i in pick for m in by_fam[fams[i]]]
        if len(set(sel)) < 4:
            continue
        r = spearmanr([a[m] for m in sel], [b[m] for m in sel])[0]
        if np.isfinite(r):
            out.append(r)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return rho, float(lo), float(hi), len(fams)


def depth_curve(rsa_ps, enc_ps, ks=(12, 10, 5, 2, 1), n_draw=400, seed=0):
    rho, ms, sess = point(rsa_ps, enc_ps)
    R = np.array([[rsa_ps[m][s] for s in sess] for m in ms])
    E = np.array([[enc_ps[m][s] for s in sess] for m in ms])
    rng = np.random.RandomState(seed)
    out = {}
    for k in ks:
        if k > len(sess):
            continue
        v = []
        for _ in range(n_draw):
            i = rng.choice(len(sess), k, replace=False)
            v.append(spearmanr(R[:, i].mean(1), E[:, i].mean(1))[0])
        out[k] = (float(np.mean(v)), float(np.std(v)))
    return out


def main():
    for tag, name in (("PUBLISHED (corrupted)", "concordance32.json"),
                      ("CORRECTED", "concordance32_CORRECTED.json")):
        rsa_ps, enc_ps = load(name)
        rho, ms, sess = point(rsa_ps, enc_ps)
        lo, hi = fisher_ci(rho, len(ms))
        frho, flo, fhi, nfam = family_clustered_ci(rsa_ps, enc_ps)
        dc = depth_curve(rsa_ps, enc_ps)
        print(f"\n=== {tag}   ({name})")
        print(f"  concordance          rho = {rho:+.4f}")
        print(f"  naive 95% CI (n={len(ms)})   [{lo:+.3f}, {hi:+.3f}]")
        print(f"  family-clustered 95% CI      [{flo:+.3f}, {fhi:+.3f}]   n_families={nfam}")
        print(f"  depth curve (mean +- sd over 400 draws):")
        for k, (m, s) in dc.items():
            print(f"      k={k:2d}   {m:+.4f} +- {s:.3f}")


if __name__ == "__main__":
    main()
