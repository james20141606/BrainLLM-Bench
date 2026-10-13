#!/usr/bin/env python
"""C4 control, step 1 (CPU): prepare the shared inputs once and run gate G2.

Pre-registration: paper4_temporalalign/PREREG_c4_static32_20260924.md.

Writes to $HOME (not /scratch, which is at 99.86% of quota):
  prep/prep_inputs.npz   neural patterns of the 30,000 scored words (float32, exactly the array
                         p4_concordance32.main builds) + per-session orthographic RDMs (uint8)
  prep/prep_receipt.json hashes, G1, and G2: C1--C3 partial RSA recomputed for all 32 models from
                         the cached 0.5-depth features, compared with the published per-session
                         values of concordance32_CORRECTED.json.
No layer-0 features are touched here, so nothing in this step reads a C4 score.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import p4_c4_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--only", default=None, help="comma list of models for a smoke")
    a = ap.parse_args()
    t0 = time.time()
    from p4_concordance32 import (neural_patterns, mid_layer_path, REGION, BAND, WIN_MS,
                                  TMIN, SFREQ)
    from region_atlas import get_canonical_neuromag306_order, get_neuromag_region_masks
    from path1_confound_rsa import ortho_rdm
    from extract_llm_layers_multi import MODEL_SET

    meta, sel, meta_sel, words, groups, hashes = L.selection()
    print(f"[G1] PASS {hashes}", flush=True)

    npz = L.PREP / "prep_inputs.npz"
    if npz.exists():
        pat, ortho = L.load_prep()
        print(f"[prep] reusing {npz}", flush=True)
    else:
        ep = np.load(L.DATA / "epochs_data.npy", mmap_mode="r")
        mask = get_neuromag_region_masks(get_canonical_neuromag306_order())[REGION]
        s0 = int(round((WIN_MS[0] / 1000.0 - TMIN) * SFREQ))
        s1 = int(round((WIN_MS[1] / 1000.0 - TMIN) * SFREQ))
        pat = neural_patterns(ep, sel, mask, s0, s1, BAND)
        print(f"[neural] {pat.shape} {pat.dtype}", flush=True)
        ortho = {}
        for s in np.unique(groups):
            w = [words[i] for i in np.where(groups == s)[0]]
            o = ortho_rdm(w)
            if o.max() > 255 or not np.array_equal(o, np.round(o)):
                raise SystemExit(f"ortho RDM of session {s} not uint8-representable")
            ortho[str(s)] = o.astype(np.uint8)
            print(f"[ortho] session {s}: {len(o)} pairs", flush=True)
        L.PREP.mkdir(parents=True, exist_ok=True)
        np.savez(npz, pat=pat, **{f"ortho_{k}": v for k, v in ortho.items()})

    if L.sha256_file(L.HEADLINE) != L.HEADLINE_SHA:
        raise SystemExit("headline receipt copy does not match the paper repo SHA-256")
    head = json.loads(L.HEADLINE.read_text())
    # The <cluster> MODEL_SET lists the ADR 0006/0008 additions between the masked and encdec blocks,
    # so its first 32 entries are not the frozen 32; the headline receipt's list is authoritative.
    models = list(head["models"])
    assert len(models) == 32 and all(m in MODEL_SET for m in models)
    assert [MODEL_SET[m] for m in models] == [r["kind"] for r in head["per_model"]]
    if a.only:
        models = [m for m in models if m in set(a.only.split(","))]
    g2 = {}
    worst = 0.0
    for m in models:
        F = np.load(mid_layer_path(m, L.FEAT_FD), mmap_mode="r")
        X = np.asarray(F[sel], dtype=np.float64)
        res = L.score_sessions(pat, X, None, words, groups, ortho, a.nproc)
        ref = head["per_session_rsa"][m]
        d = max(abs(res[s]["c3"] - ref[s]) for s in ref)
        assert set(res) == set(ref), (m, sorted(res), sorted(ref))
        worst = max(worst, d)
        g2[m] = {"max_abs_diff_vs_CORRECTED": d,
                 "per_session_c3": {s: res[s]["c3"] for s in res}}
        print(f"[G2] {m:38s} max|d|={d:.3e}", flush=True)
    rec = {"prereg": "paper4_temporalalign/PREREG_c4_static32_20260924.md",
           "step": "prep+G2", **hashes,
           "prep_npz_sha256": L.sha256_file(npz),
           "headline_receipt_sha256": L.sha256_file(L.HEADLINE),
           "code_sha256": L.code_shas(), "seed": L.SEED,
           "n_models": len(models), "G2_worst_abs_diff": worst,
           "G2_pass": bool(worst <= 1e-8 and len(models) == 32),
           "G2": g2, "jid": os.environ.get("SLURM_JOB_ID"),
           "elapsed_s": round(time.time() - t0, 1)}
    L.write_json(L.PREP / ("prep_receipt.json" if not a.only else "prep_receipt_smoke.json"), rec)
    print(f"[G2] worst {worst:.3e}  pass={rec['G2_pass']}  ({len(models)} models)", flush=True)


if __name__ == "__main__":
    main()
