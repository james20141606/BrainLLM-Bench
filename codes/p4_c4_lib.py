#!/usr/bin/env python
"""Shared pieces of the C4 (static layer-0 embedding) control for the fixed-depth RSA route.

Pre-registration: paper4_temporalalign/PREREG_c4_static32_20260924.md (MEG commit 6e81d61).

Everything that defines the headline C1--C3 score is IMPORTED from the <cluster> copies of
p4_concordance32.py and path1_confound_rsa.py, not re-implemented: the word subsample, the neural
pattern, the confound builders and partial_spearman. The only new operation is adding one more
confound, the cosine RDM of the model's own layer-0 features on the same rows. Gate G2 of the
prereg (recomputed C1--C3 must equal the published per-session values) checks that this file did
not drift from the headline pipeline.

Storage constraint (2026-09-24): /scratch is at 99.86% of quota, so nothing large is written.
Prepared inputs (neural patterns, orthographic RDMs) and all outputs live under $HOME.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path("<PROJECT>")
sys.path.insert(0, str(R / "code"))
sys.path.insert(0, str(R / "code" / "paper4_nn_revision"))

DATA = R / "data" / "libribrain_full"
META = DATA / "trial_metadata.tsv"
FEAT_FD = R / "data" / "preprocessed" / "llm_embeddings_fd"
GPT2_L0_V2 = R / "data" / "preprocessed" / "llm_embeddings_v2" / "gpt2" / "layer_0.npy"
WORK = Path(os.environ.get("C4_WORK", "<HPC_HOME>/p4_c4_20260924"))
PREP = WORK / "prep"
OUTD = WORK / "per_model"
# The published receipt lives in the paper repo, not on /scratch; a copy is placed here and its
# SHA-256 is asserted before use.
HEADLINE = WORK / "concordance32_CORRECTED.json"
HEADLINE_SHA = "e92d79fe8c77e0f8b1a073ffc67524ae83bc2532796d9d2f1fc72753bebf24b3"

ROW_SHA = "bfe5c58de6dc807aae86ad2b4b2a59d8a7c96e57f10490ab000e2c2fcdb95346"
META_SHA = "b8e7784006b049060cb0383b5f4ad4a191674e540ed571e0aa831c1d6e7e843e"
MAX_PER_SESSION, SEED, GROUP_COL = 2500, 42, "session"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_shas() -> dict:
    names = ["p4_c4_lib.py", "p4_c4_prep.py", "p4_c4_extract_score.py", "p4_concordance32.py",
             "extract_llm_layers_multi.py", "extract_llm_layers.py",
             "extract_llm_layers_multi_ef17743.py",
             "paper4_nn_revision/path1_confound_rsa.py", "compute_rdms.py"]
    out = {}
    for n in names:
        p = R / "code" / n
        out[n] = sha256_file(p) if p.exists() else None
    return out


def selection():
    """The headline's shared subsample, via the headline's own function. G1 is asserted here."""
    import pandas as pd
    from p4_concordance32 import subsample
    meta = pd.read_csv(META, sep="\t")
    sel = subsample(meta, MAX_PER_SESSION, SEED, GROUP_COL)
    row_sha = hashlib.sha256(sel.astype(np.int64).tobytes()).hexdigest()
    meta_sha = sha256_file(META)
    if row_sha != ROW_SHA or meta_sha != META_SHA:
        raise SystemExit(f"G1 FAILED: row sha {row_sha}, meta sha {meta_sha}")
    meta_sel = meta.iloc[sel].reset_index(drop=True)
    words = meta_sel["word_text"].astype(str).tolist()
    groups = meta_sel[GROUP_COL].to_numpy()
    return meta, sel, meta_sel, words, groups, {"row_sha256": row_sha, "meta_sha256": meta_sha}


# ---------------------------------------------------------------- scoring (fork workers)
_G: dict = {}


def _score_one_session(s):
    """C1--C3 exactly as p4_concordance32.rsa_per_session, plus the same with C4 appended."""
    import scipy.spatial.distance as dist
    from scipy.stats import rankdata
    from path1_confound_rsa import length_rdm, logfreq_rdm, partial_spearman
    groups, words = _G["groups"], _G["words"]
    m = groups == s
    if m.sum() < 50:                       # same skip rule as the headline
        return s, None
    w = [words[i] for i in np.where(m)[0]]
    c2, src = logfreq_rdm(w)
    if "UNAVAILABLE" in src or np.ptp(c2) == 0:
        raise RuntimeError(f"session {s}: frequency confound inert ({src})")
    conf = [length_rdm(w), c2, _G["ortho"][str(s)].astype(float)]
    nd = dist.pdist(_G["pat"][m], metric="correlation")
    ld = dist.pdist(_G["X"][m], metric="cosine")
    good = np.isfinite(nd) & np.isfinite(ld)
    nr = rankdata(nd[good])
    c3 = float(partial_spearman(nr, ld[good], [c[good] for c in conf]))
    out = {"c3": c3, "n_pairs": int(good.sum())}
    if _G.get("L0") is not None:
        c4d = dist.pdist(_G["L0"][m], metric="cosine")
        g4 = good & np.isfinite(c4d)
        nr4 = nr if g4.sum() == good.sum() else rankdata(nd[g4])
        out["c4"] = float(partial_spearman(nr4, ld[g4], [c[g4] for c in conf] + [c4d[g4]]))
        out["n_pairs_c4"] = int(g4.sum())
        out["n_nonfinite_c4"] = int(good.sum() - g4.sum())
        # descriptive: how much of the contextual RDM the static RDM carries on its own
        out["spearman_ld_c4"] = float(np.corrcoef(rankdata(ld[g4]), rankdata(c4d[g4]))[0, 1])
    return s, out


def score_sessions(pat, X, L0, words, groups, ortho, nproc):
    import multiprocessing as mp
    _G.clear()
    _G.update(pat=pat, X=X, L0=L0, words=words, groups=groups, ortho=ortho)
    sess = list(np.unique(groups))
    ctx = mp.get_context("fork")
    with ctx.Pool(min(nproc, len(sess))) as pool:
        res = dict(pool.map(_score_one_session, sess))
    return {str(s): v for s, v in res.items() if v is not None}


def load_prep():
    z = np.load(PREP / "prep_inputs.npz", allow_pickle=False)
    pat = z["pat"]
    ortho = {k[len("ortho_"):]: z[k] for k in z.files if k.startswith("ortho_")}
    return pat, ortho


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)
