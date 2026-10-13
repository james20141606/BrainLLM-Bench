#!/usr/bin/env python
"""C4 control, step 2 (GPU, one model per task): layer-0 features on the 30,000 scored rows,
validity gates G3-G5, and the C1--C3 / C1--C4 partial RSA for that model.

Pre-registration: paper4_temporalalign/PREREG_c4_static32_20260924.md.

Nothing large leaves the node. Weights are downloaded into the job's /dev/shm (HF_HOME points
there, set by the sbatch) unless the checkpoint is already in the shared HF cache, which is then
read offline. Features are written by the headline's own extractor (extract_one, <cluster> copy, batch
32, fp32) into /dev/shm and deleted with it. Only a JSON receipt (scores, gate statistics, SHA-256
of the feature files) is written, to $HOME.

Layer 0 is extracted together with the 0.5-depth layer so that G3 can compare the re-extracted
0.5-depth rows with the cached llm_embeddings_fd rows the headline was scored on: if a re-downloaded
checkpoint, tokenizer or context builder differed from the headline run, the layer-0 features
would not be that model's layer 0 and the gate says so.
"""
from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

import numpy as np

import p4_c4_lib as L


def _revisions(model: str) -> list[str]:
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ["HF_HOME"]) / "hub")
    d = hub / ("models--" + model.replace("/", "--")) / "snapshots"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def _row_cos(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shm", required=True)
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--tag_suffix", default="")
    ap.add_argument("--extractor", default="current", choices=["current", "ef17743"],
                    help="ef17743 = prereg deviation 2 (causal models only): extract_one from "
                         "extract_llm_layers_multi_ef17743.py, a verbatim copy of the extractor at git "
                         "ef17743 (JID 14100743/14103660), the version that produced the headline causal "
                         "features; G3 thresholds unchanged")
    ap.add_argument("--l0_from_v2", action="store_true",
                    help="gpt2 only (prereg deviation 1): the headline gpt2 0.5-depth features are the "
                         "llm_embeddings_v2 extraction (file-identical), so C4 is that extraction's "
                         "layer_0, read by row; no GPU, no re-extraction")
    a = ap.parse_args()
    if a.l0_from_v2:
        return v2_path(a)
    t0 = time.time()
    import torch
    from extract_llm_layers import build_sentence_contexts_from_trial_metadata, get_device
    from extract_llm_layers_multi import MODEL_SET, extract_one
    if a.extractor == "ef17743":
        # prereg deviation 2: the headline causal features were written by this version
        from extract_llm_layers_multi_ef17743 import extract_one  # noqa: F811
        if MODEL_SET[a.model] != "causal":
            raise SystemExit("deviation 2 covers causal models only")
    from p4_concordance32 import mid_layer_path

    model = a.model
    kind = MODEL_SET[model]
    tag = model.replace("/", "_")
    meta, sel, meta_sel, words, groups, hashes = L.selection()           # G1

    contexts = build_sentence_contexts_from_trial_metadata(L.META)
    if len(contexts) != len(meta):
        raise SystemExit(f"contexts {len(contexts)} != metadata rows {len(meta)}")
    ctx = [contexts[i] for i in sel]
    word_mismatch = int(sum(str(c["word"]) != w for c, w in zip(ctx, words)))

    shm = Path(a.shm)
    fdir = shm / "feat"
    device = get_device()
    t1 = time.time()
    rec = extract_one(model, kind, ctx, fdir, 32, device, 0.05, [0.0, 0.5])
    t_extract = time.time() - t1
    revisions = _revisions(model)
    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
        except (TypeError, ValueError):
            tok = AutoTokenizer.from_pretrained(model)
        padding_side = tok.padding_side
    except Exception as e:                                              # descriptive only
        padding_side = f"unknown ({type(e).__name__})"

    mdir = fdir / tag
    files = sorted(mdir.glob("layer_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    l0_path = mdir / "layer_0.npy"
    mid_new_path = files[-1]
    fd_mid = mid_layer_path(model, L.FEAT_FD)
    out = {"prereg": "paper4_temporalalign/PREREG_c4_static32_20260924.md",
           "extractor": ("extract_llm_layers_multi_ef17743.extract_one (git ef17743, verbatim)"
                         if a.extractor == "ef17743" else "extract_llm_layers_multi.extract_one"),
           "model": model, "kind": kind, **hashes,
           "hf_revisions": revisions, "hf_source": os.environ.get("C4_HF_SOURCE"),
           "tokenizer_padding_side": padding_side,
           "extractor_record": rec, "context_word_mismatch": word_mismatch,
           "layers_written": [p.name for p in files],
           "fd_mid_file": str(fd_mid.relative_to(L.R)),
           "mid_layer_name_match": fd_mid.name == mid_new_path.name,
           "jid": f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID'))}_"
                  f"{os.environ.get('SLURM_ARRAY_TASK_ID', '')}",
           "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__, "code_sha256": L.code_shas(), "seed": L.SEED}
    if "FAILED" in rec:
        out["gates"] = {"G4": False}
        L.write_json(L.OUTD / f"{tag}{a.tag_suffix}.json", out)
        raise SystemExit(f"G4 FAILED: {rec['FAILED']}")

    if a.extractor == "ef17743":
        out["deviation"] = ("prereg deviation 2: causal C4 layer 0 extracted with the ef17743 "
                            "extractor that produced the headline causal features; G3 unchanged")
        out["tokenization_audit"] = tokenization_audit(model, ctx, contexts)

    L0 = np.load(l0_path)
    MIDn = np.load(mid_new_path)
    out["l0_sha256"] = L.sha256_file(l0_path)
    out["mid_reextracted_sha256"] = L.sha256_file(mid_new_path)
    out["l0_shape"] = list(L0.shape)

    # G5 completeness
    g5 = {"rows_ok": L0.shape[0] == len(sel), "finite": bool(np.isfinite(L0).all()),
          "zero_rows": int((np.abs(L0).max(1) == 0).sum())}
    g5["pass"] = bool(g5["rows_ok"] and g5["finite"] and g5["zero_rows"] == 0)

    # G3 identity of the re-extracted 0.5-depth rows vs the cached headline features
    Fmid = np.load(fd_mid, mmap_mode="r")
    MIDo = np.asarray(Fmid[sel], dtype=np.float32)
    cos = _row_cos(MIDn.astype(np.float64), MIDo.astype(np.float64))
    rel = np.linalg.norm(MIDn - MIDo, axis=1) / np.maximum(np.linalg.norm(MIDo, axis=1), 1e-30)
    g3 = {"shape_match": MIDn.shape == MIDo.shape,
          "median_row_cos": float(np.median(cos)), "min_row_cos": float(cos.min()),
          "frac_rows_cos_ge_0.9999": float((cos >= 0.9999).mean()),
          "median_rel_l2": float(np.median(rel)), "max_abs_diff": float(np.abs(MIDn - MIDo).max()),
          "bitwise_equal": bool(np.array_equal(MIDn, MIDo))}
    g3["pass"] = bool(g3["shape_match"] and out["mid_layer_name_match"]
                      and g3["median_row_cos"] >= 0.99999
                      and g3["frac_rows_cos_ge_0.9999"] >= 0.999)
    gates = {"G3": g3, "G4": True, "G5": g5}
    if model == "gpt2":
        V2 = np.load(L.GPT2_L0_V2, mmap_mode="r")
        v2 = np.asarray(V2[sel], dtype=np.float32)
        gates["G3_gpt2_v2_layer0"] = {"max_abs_diff": float(np.abs(v2 - L0).max()),
                                      "bitwise_equal": bool(np.array_equal(v2, L0)),
                                      "v2_file_sha256_expected":
                                          "56ff513042d0a8ab22ad2584f9582235213ff4f9f5c42835501dbc68604f9a66"}
        gates["G3_gpt2_v2_layer0"]["pass"] = gates["G3_gpt2_v2_layer0"]["max_abs_diff"] <= 1e-5
    out["gates"] = gates
    del MIDn, MIDo, Fmid

    # Scoring: the headline 0.5-depth features (cached), the model's own new layer 0 as C4.
    pat, ortho = L.load_prep()
    F = np.load(mid_layer_path(model, L.FEAT_FD), mmap_mode="r")
    X = np.asarray(F[sel], dtype=np.float64)
    t2 = time.time()
    res = L.score_sessions(pat, X, np.asarray(L0, dtype=np.float64), words, groups, ortho,
                           a.nproc)
    out["per_session"] = res
    out["rsa_c3_mean"] = float(np.mean([v["c3"] for v in res.values()]))
    out["rsa_c4_mean"] = float(np.mean([v["c4"] for v in res.values()]))
    out["timing_s"] = {"extract": round(t_extract, 1), "score": round(time.time() - t2, 1),
                       "total": round(time.time() - t0, 1)}
    L.write_json(L.OUTD / f"{tag}{a.tag_suffix}.json", out)
    print(f"[done] {model}: G3 {g3['pass']} (med cos {g3['median_row_cos']:.7f}) G5 {g5['pass']} "
          f"C3 {out['rsa_c3_mean']:+.5f} C4 {out['rsa_c4_mean']:+.5f} {out['timing_s']}",
          flush=True)


def tokenization_audit(model, ctx_sel, ctx_all):
    """Descriptive, for the provenance disclosure: how often the headline causal rule (ef17743:
    tokenizer(" ".join(prefix)), default tokenizer) and the documented rule (current
    extract_llm_layers_multi: word list, is_split_into_words, add_prefix_space) give a different
    token sequence, a different final (target) token, or a different target position."""
    from transformers import AutoTokenizer
    t_old = AutoTokenizer.from_pretrained(model)
    try:
        t_new = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
    except (TypeError, ValueError):
        t_new = AutoTokenizer.from_pretrained(model)

    def one(ctxs):
        n = len(ctxs)
        d_seq = d_last = d_len = 0
        B = 2048
        for i in range(0, n, B):
            cb = ctxs[i:i + B]
            words = [list(c["sentence_words"][: c["word_position"] + 1]) for c in cb]
            o = t_old([" ".join(w) for w in words], truncation=True, max_length=512)["input_ids"]
            w = t_new(words, is_split_into_words=True, truncation=True, max_length=512)["input_ids"]
            for a_, b_ in zip(o, w):
                d_seq += a_ != b_
                d_last += a_[-1] != b_[-1]
                d_len += len(a_) != len(b_)
        return {"n_rows": n, "frac_token_sequence_differs": d_seq / n,
                "frac_target_token_id_differs": d_last / n,
                "frac_target_position_differs": d_len / n}
    return {"scored_30000": one(ctx_sel), "all_rows": one(ctx_all),
            "old_rule": "tokenizer(' '.join(prefix)), from_pretrained default",
            "new_rule": "tokenizer(prefix_words, is_split_into_words=True), add_prefix_space=True"}


def v2_path(a):
    """GPT-2 C4 from the one extraction the headline GPT-2 features actually came from.

    The smoke (18407672_0) showed the headline gpt2 features in llm_embeddings_fd were not produced
    by extract_llm_layers_multi: re-extraction gave median row cosine 0.99904 at layer 6 and a
    different layer 0 on some rows. The feasibility audit had found fd/gpt2/layer_6.npy
    file-identical to llm_embeddings_v2/gpt2/layer_6.npy, i.e. the headline gpt2 features ARE the
    v2 extraction, whose layer_0.npy is therefore the same model's layer 0 under the same target rule.
    Gate here: that file identity (SHA-256), re-checked, plus the v2 layer_0 SHA from the audit.
    """
    import torch  # noqa: F401  (version only)
    from p4_concordance32 import mid_layer_path
    model = a.model
    assert model == "gpt2"
    meta, sel, meta_sel, words, groups, hashes = L.selection()
    fd_mid = mid_layer_path(model, L.FEAT_FD)
    v2_mid = L.GPT2_L0_V2.parent / fd_mid.name
    sha_fd, sha_v2mid, sha_v2l0 = (L.sha256_file(fd_mid), L.sha256_file(v2_mid),
                                   L.sha256_file(L.GPT2_L0_V2))
    L0 = np.asarray(np.load(L.GPT2_L0_V2, mmap_mode="r")[sel], dtype=np.float32)
    g3 = {"mode": "v2_cache_file_identity", "fd_mid_file": str(fd_mid.relative_to(L.R)),
          "fd_mid_sha256": sha_fd, "v2_mid_sha256": sha_v2mid, "v2_layer0_sha256": sha_v2l0,
          "fd_mid_equals_v2_mid": sha_fd == sha_v2mid,
          "v2_layer0_sha_matches_audit": sha_v2l0 ==
          "56ff513042d0a8ab22ad2584f9582235213ff4f9f5c42835501dbc68604f9a66"}
    g3["pass"] = bool(g3["fd_mid_equals_v2_mid"] and g3["v2_layer0_sha_matches_audit"])
    g5 = {"rows_ok": L0.shape[0] == len(sel), "finite": bool(np.isfinite(L0).all()),
          "zero_rows": int((np.abs(L0).max(1) == 0).sum())}
    g5["pass"] = bool(g5["rows_ok"] and g5["finite"] and g5["zero_rows"] == 0)
    out = {"prereg": "paper4_temporalalign/PREREG_c4_static32_20260924.md",
           "deviation": "prereg deviation 1: gpt2 C4 from llm_embeddings_v2 layer_0 (see v2_path doc)",
           "model": model, "kind": "causal", **hashes, "hf_revisions": None,
           "hf_source": "none (cached features)", "extractor_record": {"fallback_frac": None},
           "l0_sha256": sha_v2l0, "l0_shape": list(L0.shape),
           "jid": f"{os.environ.get('SLURM_JOB_ID')}", "host": socket.gethostname(),
           "code_sha256": L.code_shas(), "seed": L.SEED,
           "gates": {"G3": g3, "G4": True, "G5": g5,
                     "G3_gpt2_v2_layer0": {"pass": g3["pass"], "mode": "is the v2 file"}}}
    pat, ortho = L.load_prep()
    X = np.asarray(np.load(fd_mid, mmap_mode="r")[sel], dtype=np.float64)
    res = L.score_sessions(pat, X, np.asarray(L0, dtype=np.float64), words, groups, ortho, a.nproc)
    out["per_session"] = res
    out["rsa_c3_mean"] = float(np.mean([v["c3"] for v in res.values()]))
    out["rsa_c4_mean"] = float(np.mean([v["c4"] for v in res.values()]))
    L.write_json(L.OUTD / f"{model}{a.tag_suffix}.json", out)
    print(f"[done v2] gpt2: G3 {g3['pass']} G5 {g5['pass']}", flush=True)


if __name__ == "__main__":
    main()
