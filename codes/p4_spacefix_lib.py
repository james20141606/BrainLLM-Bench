#!/usr/bin/env python
"""Shared pieces of the space-fix re-extraction (paperA, 2026-09-25).

Pre-registration: paper4_temporalalign/PREREG_space_fix_reextract_20260925.md (MEG fed178a).

Everything that defines a published score is IMPORTED from the frozen <cluster> pipeline
(<PROJECT>/code): the word subsample, the neural pattern, the confound
builders, rsa_per_session, encoding_per_session, the context builders, and the ef17743 extractor.
This file only holds cohort settings (copied from the sbatch lines that produced the published
receipts), hashing, and I/O. Gate G2 (scorer on cached features == published per-session values)
checks that nothing here drifted from the published pipeline.

Storage: nothing large goes to /scratch. Features (0.5-depth layer, selected rows only) live under
$HOME/p4_spacefix_20260925/feat and are deleted after scoring; receipts under .../receipts.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path("<PROJECT>")
CODE = R / "code"
for _p in (str(CODE), str(CODE / "paper4_nn_revision")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WORK = Path(os.environ.get("SF_WORK", "<HPC_HOME>/p4_spacefix_20260925"))
FEAT = WORK / "feat"
PREP = WORK / "prep"
OUTD = WORK / "receipts"
PUB = WORK / "published"          # copies of the published receipts, SHA-256 asserted
PREREG = "paper4_temporalalign/PREREG_space_fix_reextract_20260925.md"
C4_PREP = Path("<HPC_HOME>/p4_c4_20260924/prep/prep_inputs.npz")
C4_PREP_SHA = "ea2dae192744f78d99f6a470012acf402f841ee8d77b21e96266d3250ff27f07"

# Settings of the published run of each cohort (sbatch lines quoted in the prereg section 5).
COHORTS = {
    "libribrain": dict(
        data_dir=R / "data/libribrain_full",
        ctx_meta=R / "data/libribrain_full/trial_metadata.tsv", builder="trial_metadata",
        feat_dir=R / "data/preprocessed/llm_embeddings_fd", emb_idx=None,
        region="temporal_L", atlas="neuromag306", group_col="session", tmin=-0.2, sfreq=250.0,
        row_sha="bfe5c58de6dc807aae86ad2b4b2a59d8a7c96e57f10490ab000e2c2fcdb95346",
        meta_sha="b8e7784006b049060cb0383b5f4ad4a191674e540ed571e0aa831c1d6e7e843e"),
    "masc27": dict(
        data_dir=R / "data/meg_masc_merged27",
        ctx_meta=R / "data/meg_masc_merged27/trial_metadata.tsv", builder="columns",
        feat_dir=R / "data/preprocessed/llm_embeddings_masc27_fd", emb_idx=None,
        region="temporal_R", atlas="kit208", group_col="subject", tmin=-0.2, sfreq=250.0),
    "brennan": dict(
        data_dir=R / "data/brennan_p4_v2",
        ctx_meta=R / "data/brennan_p4/trial_metadata.tsv", builder="columns",
        feat_dir=R / "data/preprocessed/llm_embeddings_brennan_fd",
        emb_idx=R / "data/brennan_p4_v2/emb_row_idx.npy",
        region="all", atlas="eeg60", group_col="subject", tmin=-0.1, sfreq=500.0),
}
MAX_PER_GROUP, SEED = 2500, 42


def tag(model: str) -> str:
    return model.replace("/", "_")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def code_shas() -> dict:
    here = Path(__file__).resolve().parent
    out = {}
    for n in ["p4_spacefix_lib.py", "p4_spacefix_extract.py", "p4_spacefix_prep.py",
              "p4_spacefix_score.py", "p4_spacefix_aggregate.py"]:
        p = here / n
        out[n] = sha256_file(p) if p.exists() else None
    for n in ["p4_concordance32.py", "extract_llm_layers_multi.py", "extract_llm_layers.py",
              "extract_llm_layers_multi_ef17743.py", "alpha_sensitivity_20260922.py",
              "paper4_nn_revision/path1_confound_rsa.py", "compute_rdms.py"]:
        p = CODE / n
        out["frozen/" + n] = sha256_file(p) if p.exists() else None
    return out


def _subsample_copy(meta, max_per_session: int, seed: int, group_col: str):
    """Byte-for-byte copy of p4_concordance32.subsample, used only where that module cannot be
    imported (the transformers-5 env `<env2>` has no scikit-learn). G1 asserts the row hash."""
    rng = np.random.RandomState(seed)
    keep = []
    for s, idx in meta.groupby(group_col).indices.items():
        idx = np.asarray(idx)
        if len(idx) > max_per_session:
            idx = np.sort(rng.choice(idx, max_per_session, replace=False))
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def selection(cohort: str):
    """The published shared subsample of a cohort, via p4_concordance32's own function (G1)."""
    import pandas as pd
    try:
        from p4_concordance32 import subsample
    except ImportError:
        subsample = _subsample_copy
    c = COHORTS[cohort]
    meta_path = c["data_dir"] / "trial_metadata.tsv"
    meta = pd.read_csv(meta_path, sep="\t")
    sel = subsample(meta, MAX_PER_GROUP, SEED, c["group_col"])
    meta_sel = meta.iloc[sel].reset_index(drop=True)
    words = meta_sel["word_text"].astype(str).tolist()
    groups = meta_sel[c["group_col"]].to_numpy()
    if c["emb_idx"] is not None:
        e = np.load(c["emb_idx"])
        if len(e) != len(meta):
            raise SystemExit(f"G1 FAILED: emb_idx {len(e)} rows vs metadata {len(meta)}")
        emb_rows = e[sel]
    else:
        emb_rows = sel
    h = {"row_sha256": sha256_arr(sel.astype(np.int64)),
         "meta_sha256": sha256_file(meta_path),
         "emb_rows_sha256": sha256_arr(np.asarray(emb_rows).astype(np.int64)),
         "n_words": int(len(sel)), "n_groups": int(len(np.unique(groups)))}
    if "row_sha" in c and (h["row_sha256"] != c["row_sha"] or h["meta_sha256"] != c["meta_sha"]):
        raise SystemExit(f"G1 FAILED ({cohort}): {h}")
    return meta, sel, meta_sel, words, groups, np.asarray(emb_rows), h


def contexts(cohort: str):
    """Sentence contexts with the builder the cached run of this cohort used."""
    c = COHORTS[cohort]
    cwd = os.getcwd()
    os.chdir(R)            # the LibriBrain builder resolves events TSVs relative to the repo root
    try:
        if c["builder"] == "trial_metadata":
            from extract_llm_layers import build_sentence_contexts_from_trial_metadata
            return build_sentence_contexts_from_trial_metadata(Path(c["ctx_meta"]))
        from extract_llm_layers_multi import build_contexts_from_columns
        return build_contexts_from_columns(Path(c["ctx_meta"]))
    finally:
        os.chdir(cwd)


def cached_mid(model: str, cohort: str) -> Path:
    try:
        from p4_concordance32 import mid_layer_path
    except ImportError:                                # same logic as p4_concordance32.mid_layer_path
        def mid_layer_path(model, feat_dir):
            d = Path(feat_dir) / model.replace("/", "_")
            lf = sorted(d.glob("layer_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
            if len(lf) != 3:
                raise FileNotFoundError(f"{model}: expected 3 depths, found {len(lf)}")
            return lf[1]
    return mid_layer_path(model, COHORTS[cohort]["feat_dir"])


def feat_paths(model: str, cohort: str):
    d = FEAT / cohort / tag(model)
    return d, d / "rows.npy"


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(p)


def row_cos(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-30)


def cos_stats(new, old) -> dict:
    cos = row_cos(new, old)
    return {"n_rows": int(len(cos)), "median_row_cos": float(np.median(cos)),
            "min_row_cos": float(cos.min()), "p05_row_cos": float(np.percentile(cos, 5)),
            "frac_rows_cos_ge_0.9999": float((cos >= 0.9999).mean()),
            "frac_rows_cos_ge_0.999": float((cos >= 0.999).mean()),
            "frac_rows_cos_ge_0.99": float((cos >= 0.99).mean()),
            "frac_rows_bitwise_equal": float((np.asarray(new) == np.asarray(old)).all(1).mean()),
            "max_abs_diff": float(np.abs(np.asarray(new, np.float64) - np.asarray(old, np.float64)).max())}


def store_features(arr, d: Path, stem: str) -> Path:
    """float32 features -> $HOME. If every value is an exact bf16 (low 16 bits zero), store the upper
    16 bits (lossless, half the size); else store float32 unchanged."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    u = a.view(np.uint32)
    if not (u & 0xFFFF).any():
        dst = d / f"{stem}.bf16.npy"
        np.save(dst, (u >> 16).astype(np.uint16))
        back = load_features(dst)
        if not np.array_equal(back.view(np.uint32), u):
            raise SystemExit("bf16 packing is not lossless")
        return dst
    dst = d / f"{stem}.npy"
    np.save(dst, a)
    return dst


def load_features(path) -> np.ndarray:
    path = Path(path)
    if path.name.endswith(".bf16.npy"):
        h = np.load(path)
        return (h.astype(np.uint32) << 16).view(np.float32)
    return np.load(path)
