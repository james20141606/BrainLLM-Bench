#!/usr/bin/env python
"""Space-fix re-extraction, CPU step: prep (orthographic confounds), gate G2, and scoring.

Pre-registration: paper4_temporalalign/PREREG_space_fix_reextract_20260925.md (MEG fed178a).

  --prep                 per-group orthographic RDMs (uint8, asserted integer <= 255) for one cohort,
                         computed with path1_confound_rsa.ortho_rdm; LibriBrain reuses the C4 prep
                         file (SHA-256 asserted), whose ortho arrays passed C4 gate G2.
  --source cached        gate G2: score the CACHED features and compare with the published
                         per-session values (RSA <= 1e-8, encoding <= 1e-6; alpha replay <= 1e-6).
  --source new           score the re-extracted features; optionally delete them afterwards.

Scoring is the published pipeline, imported: p4_concordance32.{subsample, neural_patterns,
rsa_per_session, encoding_per_session} with the cohort's published cell, and
alpha_sensitivity_20260922.{fit_one, ORIGINAL, EXPANDED} for the post hoc alpha grid.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

import p4_spacefix_lib as L

ALPHA_MODELS = {"Qwen/Qwen2.5-3B": 2, "Qwen/Qwen2.5-7B": 3, "Qwen/Qwen2.5-14B": 4,
                "Qwen/Qwen2.5-32B": 5, "Qwen/Qwen2.5-72B": 6}
PUBLISHED = {  # file -> which cohort's per-session values it holds
    "libribrain": ["concordance_modern_final_v2.json"],
    "masc27": ["concordance_masc27_ladder.json", "concordance32_masc27.json"],
    "brennan": ["concordance32_brennan_v2.json"],
}


def _ortho_one(args):
    from path1_confound_rsa import ortho_rdm
    s, w = args
    o = ortho_rdm(w)
    if o.max() > 255 or not np.array_equal(o, np.round(o)):
        raise SystemExit(f"ortho RDM of group {s} not uint8-representable")
    return str(s), o.astype(np.uint8)


def prep(cohort, nproc):
    meta, sel, meta_sel, words, groups, emb_rows, h = L.selection(cohort)
    out = L.PREP / f"{cohort}_ortho.npz"
    if cohort == "libribrain":
        raise SystemExit("libribrain uses the C4 prep file")
    import multiprocessing as mp
    jobs = [(s, [words[i] for i in np.where(groups == s)[0]]) for s in np.unique(groups)]
    with mp.get_context("fork").Pool(min(nproc, len(jobs))) as pool:
        res = dict(pool.map(_ortho_one, jobs))
    L.PREP.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{f"ortho_{k}": v for k, v in res.items()})
    rec = {"prereg": L.PREREG, "cohort": cohort, **h, "npz": str(out),
           "npz_sha256": L.sha256_file(out), "n_groups": len(res),
           "jid": os.environ.get("SLURM_JOB_ID"), "code_sha256": L.code_shas()}
    L.write_json(L.PREP / f"{cohort}_prep_receipt.json", rec)
    print(f"[prep] {cohort}: {len(res)} groups -> {out}", flush=True)


def load_ortho(cohort):
    if cohort == "libribrain":
        if L.sha256_file(L.C4_PREP) != L.C4_PREP_SHA:
            raise SystemExit("C4 prep file SHA-256 mismatch")
        z = np.load(L.C4_PREP)
        return {k[len("ortho_"):]: z[k] for k in z.files if k.startswith("ortho_")}, L.C4_PREP_SHA
    p = L.PREP / f"{cohort}_ortho.npz"
    z = np.load(p)
    return {k[len("ortho_"):]: z[k] for k in z.files if k.startswith("ortho_")}, L.sha256_file(p)


def build_inputs(cohort):
    """Exactly the arrays p4_concordance32.main builds for this cohort."""
    import p4_concordance32 as C
    from path1_confound_rsa import length_rdm, logfreq_rdm
    c = L.COHORTS[cohort]
    meta, sel, meta_sel, words, groups, emb_rows, h = L.selection(cohort)
    ep = np.load(c["data_dir"] / "epochs_data.npy", mmap_mode="r")
    if c["atlas"] == "eeg60":
        mask = np.ones(int(ep.shape[1]), dtype=bool)
    elif c["atlas"] == "kit208":
        from region_atlas_kit import get_canonical_kit208_order, get_kit_region_masks
        mask = get_kit_region_masks(get_canonical_kit208_order())[c["region"]]
    else:
        from region_atlas import get_canonical_neuromag306_order, get_neuromag_region_masks
        mask = get_neuromag_region_masks(get_canonical_neuromag306_order())[c["region"]]
    s0 = int(round((C.WIN_MS[0] / 1000.0 - c["tmin"]) * c["sfreq"]))
    s1 = int(round((C.WIN_MS[1] / 1000.0 - c["tmin"]) * c["sfreq"]))
    pat = C.neural_patterns(ep, sel, mask, s0, s1, C.BAND)
    Y = pat.reshape(len(pat), int(mask.sum()), s1 - s0).mean(axis=2)
    ortho, ortho_sha = load_ortho(cohort)
    confounds = {}
    for s in np.unique(groups):
        m = groups == s
        w = [words[i] for i in np.where(m)[0]]
        nd_freq, src = logfreq_rdm(w)
        if "UNAVAILABLE" in src or np.ptp(nd_freq) == 0:
            raise RuntimeError(f"group {s}: frequency confound inert ({src})")
        confounds[s] = [length_rdm(w), nd_freq, ortho[str(s)].astype(float)]
    return dict(meta_sel=meta_sel, words=words, groups=groups, emb_rows=emb_rows, sel=sel, pat=pat,
                Y=Y, confounds=confounds, hashes={**h, "ortho_sha256": ortho_sha},
                group_col=c["group_col"])


def alpha_grid(X, pat_Y, groups):
    """alpha_sensitivity_20260922's loop: original and expanded RidgeCV grids on the 12 LOGO folds."""
    import alpha_sensitivity_20260922 as A
    from sklearn.model_selection import LeaveOneGroupOut
    y = np.asarray(pat_Y, dtype=np.float64)
    ysc = float(y.std())
    if ysc > 0:
        y = y / ysc
    folds = {}
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sid = str(groups[te][0])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        xtr, xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        a0, s0, _ = A.fit_one(xtr, xte, y[tr], y[te], A.ORIGINAL)
        a1, s1, _ = A.fit_one(xtr, xte, y[tr], y[te], A.EXPANDED)
        folds[sid] = {"original_alpha": a0, "original_score": s0,
                      "expanded_alpha": a1, "expanded_score": s1}
    return {"folds": folds,
            "original_mean": float(np.mean([f["original_score"] for f in folds.values()])),
            "expanded_mean": float(np.mean([f["expanded_score"] for f in folds.values()])),
            "expanded_top_grid_count": int(sum(f["expanded_alpha"] == float(A.EXPANDED[-1])
                                               for f in folds.values()))}


def published_ps(cohort, model):
    out = {}
    for fn in PUBLISHED[cohort]:
        p = L.PUB / fn
        d = json.loads(p.read_text())
        if model in d.get("per_session_rsa", {}):
            out[fn] = {"sha256": L.sha256_file(p), "rsa": d["per_session_rsa"][model],
                       "enc": d["per_session_enc"][model]}
    return out


def score_model(model, cohort, I, source, do_alpha):
    import p4_concordance32 as C
    if source == "cached":
        F = np.load(L.cached_mid(model, cohort), mmap_mode="r")
        X = np.asarray(F[I["emb_rows"]], dtype=np.float64)
        fsha = L.sha256_file(L.cached_mid(model, cohort))
        fpath = str(L.cached_mid(model, cohort))
    else:
        d, rows_p = L.feat_paths(model, cohort)
        rec_x = json.loads((L.OUTD / "extract" / f"{cohort}__{L.tag(model)}.json").read_text())
        if rec_x.get("status") != "OK":
            raise SystemExit(f"extraction receipt status {rec_x.get('status')}")
        fpath = rec_x["feature_file"]
        fsha = L.sha256_file(fpath)
        if fsha != rec_x["feature_sha256"]:
            raise SystemExit("feature file SHA-256 differs from its extraction receipt")
        rows = np.load(rows_p)
        F = L.load_features(fpath)
        if "feature_float32_sha256" in rec_x and L.sha256_arr(F) != rec_x["feature_float32_sha256"]:
            raise SystemExit("rebuilt float32 features differ from the extraction receipt")
        pos = np.searchsorted(rows, I["emb_rows"])
        if not np.array_equal(rows[pos], I["emb_rows"]):
            raise SystemExit("selected rows missing from the re-extracted feature file")
        X = np.asarray(F[pos], dtype=np.float64)
    t0 = time.time()
    r_rsa = C.rsa_per_session(I["pat"], X, I["words"], I["meta_sel"], I["confounds"],
                              I["group_col"])
    t1 = time.time()
    r_enc = C.encoding_per_session(X, I["Y"].astype(np.float64), I["groups"])
    t2 = time.time()
    out = {"prereg": L.PREREG, "model": model, "cohort": cohort, "source": source,
           "feature_file": fpath, "feature_sha256": fsha, "x_shape": list(X.shape),
           **I["hashes"],
           "per_session_rsa": {str(k): float(v) for k, v in r_rsa.items()},
           "per_session_enc": {str(k): float(v) for k, v in r_enc.items()},
           "rsa_mean": float(np.mean(list(r_rsa.values()))),
           "enc_mean": float(np.mean(list(r_enc.values()))),
           "timing_s": {"rsa": round(t1 - t0, 1), "enc": round(t2 - t1, 1)}}
    if do_alpha:
        out["alpha_grid"] = alpha_grid(X, I["Y"], I["groups"])
        # internal consistency: the original-grid replay equals encoding_per_session
        out["alpha_grid"]["replay_vs_enc_max_abs"] = float(max(
            abs(out["alpha_grid"]["folds"][s]["original_score"] - out["per_session_enc"][s])
            for s in out["per_session_enc"]))
    if source == "cached":
        g2 = {}
        for fn, pub in published_ps(cohort, model).items():
            dr = max(abs(out["per_session_rsa"][s] - float(pub["rsa"][s])) for s in pub["rsa"])
            de = max(abs(out["per_session_enc"][s] - float(pub["enc"][s])) for s in pub["enc"])
            same_keys = set(pub["rsa"]) == set(out["per_session_rsa"])
            g2[fn] = {"sha256": pub["sha256"], "max_abs_rsa": dr, "max_abs_enc": de,
                      "same_groups": same_keys,
                      "pass": bool(dr <= 1e-8 and de <= 1e-6 and same_keys)}
        if do_alpha:
            idx = ALPHA_MODELS[model]
            pa = L.PUB / "alpha_sensitivity_20260922" / f"qwen25_{idx}_full.json"
            ref = {f["session"]: f for f in json.loads(pa.read_text())["folds"]}
            da = max(abs(out["alpha_grid"]["folds"][s]["expanded_score"] - ref[s]["expanded_score"])
                     for s in ref)
            do = max(abs(out["alpha_grid"]["folds"][s]["original_score"] - ref[s]["original_score"])
                     for s in ref)
            g2["alpha_" + pa.name] = {"sha256": L.sha256_file(pa), "max_abs_expanded": da,
                                      "max_abs_original": do, "pass": bool(max(da, do) <= 1e-6)}
        out["G2"] = g2
        out["G2_pass"] = bool(g2) and all(v["pass"] for v in g2.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(L.COHORTS))
    ap.add_argument("--models", default="", help="comma list")
    ap.add_argument("--source", choices=["cached", "new"], default="new")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--delete_after", action="store_true")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    if a.prep:
        return prep(a.cohort, a.nproc)
    t0 = time.time()
    I = build_inputs(a.cohort)
    print(f"[inputs] {a.cohort}: pat {I['pat'].shape} Y {I['Y'].shape} groups "
          f"{len(np.unique(I['groups']))} ({time.time() - t0:.0f}s)", flush=True)
    for model in [m for m in a.models.split(",") if m]:
        do_alpha = a.cohort == "libribrain" and model in ALPHA_MODELS
        try:
            out = score_model(model, a.cohort, I, a.source, do_alpha)
        except SystemExit as e:
            print(f"[SKIP] {model}: {e}", flush=True)
            continue
        out.update(jid=f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID'))}_"
                       f"{os.environ.get('SLURM_ARRAY_TASK_ID', '')}",
                   host=socket.gethostname(), code_sha256=L.code_shas())
        p = L.OUTD / "score" / f"{a.cohort}__{L.tag(model)}__{a.source}.json"
        L.write_json(p, out)
        msg = f"G2 {out['G2_pass']}" if a.source == "cached" else "scored"
        print(f"[{a.source}] {a.cohort} {model}: {msg} {out['timing_s']}", flush=True)
        if a.source == "new" and a.delete_after:
            d, rows_p = L.feat_paths(model, a.cohort)
            for f in list(d.glob("*.npy")):
                f.unlink()
            try:
                d.rmdir()
            except OSError:
                pass
            print(f"[deleted] {d}", flush=True)


if __name__ == "__main__":
    main()
