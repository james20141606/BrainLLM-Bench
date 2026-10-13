#!/usr/bin/env python
"""Space-fix re-extraction, GPU step (one checkpoint per task, one or more cohorts).

Pre-registration: paper4_temporalalign/PREREG_space_fix_reextract_20260925.md (MEG fed178a).

Rule: the ef17743 extractor's extract_one, called unchanged (joined-string prefix, default tokenizer,
read-out at attention_mask.sum()-1, layer int(0.5*top+0.5)). For extension checkpoints (--mode ext)
its module-level _load is replaced by one that keeps ef17743's tokenizer lines verbatim and loads the
model exactly as the cached extension run did (cluster extractor --big branch: bf16,
device_map="auto"). For Qwen2.5-0.5B/1.5B and Llama-3.2-1B (--mode small) ef17743's _load is used as
is (fp32, one device), as for their LibriBrain headline features.

Gates computed here (prereg section 4): G1 (rows / context map), T1/T2 (tokenization, every row, before
extraction), W (documented rule re-run on 512 rows vs the cached rows; ext only), I (--identity_full:
all LibriBrain contexts in the original batch composition vs the cached file; small only), G5, D1.

Only the GPU work happens here. Features go to /dev/shm, then the 0.5-depth matrix of the selected
rows is copied to $HOME/p4_spacefix_20260925/feat/<cohort>/<tag>/ for the CPU scorer.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

import p4_spacefix_lib as L


def tokenizer_files(model: str) -> dict:
    out = {}
    try:
        from huggingface_hub import try_to_load_from_cache
        for fn in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            p = try_to_load_from_cache(model, fn)
            if isinstance(p, str) and os.path.exists(p):
                out[fn] = {"sha256": L.sha256_file(p),
                           "revision": Path(p).parent.name}
    except Exception as e:                                         # descriptive only
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def tok_gates(tok, ctx) -> dict:
    """T1/T2 on every row, tokenizing exactly the ef17743 texts (no padding needed: right padding
    does not change the real ids; padding side is asserted)."""
    texts = [" ".join(c["sentence_words"][: c["word_position"] + 1]) for c in ctx]
    special = set(tok.all_special_ids)
    n = len(texts)
    eq = space_loss = space_loss_strict = last_special = trunc = 0
    bad = []
    B = 4096
    for i in range(0, n, B):
        enc = tok(texts[i:i + B], truncation=True, max_length=512)["input_ids"]
        for j, ids in enumerate(enc):
            t = texts[i + j]
            d = tok.decode(ids, skip_special_tokens=True).strip()
            if d == t:
                eq += 1
            else:
                if d.replace(" ", "") == t.replace(" ", ""):
                    space_loss += 1
                if d.count(" ") < t.strip().count(" "):
                    space_loss_strict += 1
                if len(bad) < 20:
                    bad.append({"row": i + j, "text": t[-120:], "decode": d[-120:]})
            if ids[-1] in special:
                last_special += 1
            if len(ids) >= 512:
                trunc += 1
    g = {"n_rows": n, "decode_equal": eq, "frac_decode_equal": eq / max(n, 1),
         "space_loss_rows": space_loss, "space_loss_strict_rows": space_loss_strict, "readout_is_special_rows": last_special,
         "rows_at_512_tokens": trunc, "padding_side": tok.padding_side,
         "mismatch_examples": bad}
    g["T1_pass"] = bool(g["frac_decode_equal"] >= 0.999 and space_loss == 0)
    g["T2_pass"] = bool(last_special == 0 and tok.padding_side == "right" and trunc == 0)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohorts", required=True, help="comma list of libribrain,masc27,brennan")
    ap.add_argument("--mode", required=True, choices=["ext", "small"])
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--shm", required=True)
    ap.add_argument("--identity_full", action="store_true",
                    help="small only: gate I on all LibriBrain contexts, original batch composition")
    ap.add_argument("--no_copy", action="store_true", help="smoke: do not copy features to $HOME")
    a = ap.parse_args()
    t0 = time.time()
    import torch
    import transformers
    import tokenizers
    from transformers import AutoTokenizer, AutoModel
    import extract_llm_layers_multi_ef17743 as E            # frozen, verbatim ef17743
    import extract_llm_layers_multi as CUR                   # documented rule, for gate W only
    from extract_llm_layers import get_device

    model = a.model
    cohorts = a.cohorts.split(",")
    device = get_device()
    jid = (f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID'))}_"
           f"{os.environ.get('SLURM_ARRAY_TASK_ID', '')}")
    base = {"prereg": L.PREREG, "model": model, "mode": a.mode, "batch_size": a.bs,
            "jid": jid, "host": socket.gethostname(),
            "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "tokenizers": tokenizers.__version__},
            "python": os.path.realpath(sys.executable),
            "hf_source": os.environ.get("SF_HF_SOURCE"),
            "code_sha256": L.code_shas(), "tokenizer_files": tokenizer_files(model),
            "rule": "ef17743 extract_one verbatim; " + (
                "ef17743 _load verbatim (fp32)" if a.mode == "small" else
                "_load = ef17743 tokenizer lines + cluster --big model load (bf16, device_map=auto)")}

    # ---- tokenizers: ef17743 lines verbatim; documented-rule tokenizer for gate W
    tok_ef = AutoTokenizer.from_pretrained(model)
    if tok_ef.pad_token is None:
        tok_ef.pad_token = tok_ef.eos_token if tok_ef.eos_token else tok_ef.unk_token

    # ---- per-cohort rows and contexts (G1) and tokenization gates (T1/T2), before any forward
    plan = {}
    for coh in cohorts:
        meta, sel, meta_sel, words, groups, emb_rows, h = L.selection(coh)
        ctx_all = L.contexts(coh)
        cached = np.load(L.cached_mid(model, coh), mmap_mode="r")
        g1 = {**h, "n_contexts": len(ctx_all), "cached_rows": int(cached.shape[0]),
              "cached_mid_file": str(L.cached_mid(model, coh).relative_to(L.R)),
              "cached_mid_sha256": L.sha256_file(L.cached_mid(model, coh))}
        g1["pass"] = bool(len(ctx_all) == cached.shape[0])
        urows = np.unique(emb_rows)
        import pandas as pd
        cm = pd.read_csv(L.COHORTS[coh]["ctx_meta"], sep="\t")["word_text"].astype(str).tolist()
        g1["context_word_mismatch_selected"] = int(sum(str(ctx_all[i]["word"]) != cm[i] for i in urows))
        g1["n_unique_rows"] = int(len(urows))
        if not g1["pass"]:
            raise SystemExit(f"G1 FAILED {coh}: {g1}")
        ctx = [ctx_all[i] for i in urows]
        tg = tok_gates(tok_ef, ctx)
        print(f"[{coh}] rows {len(ctx)}  T1 {tg['T1_pass']} ({tg['decode_equal']}/{tg['n_rows']}, "
              f"space_loss {tg['space_loss_rows']})  T2 {tg['T2_pass']}", flush=True)
        plan[coh] = dict(ctx=ctx, ctx_all=ctx_all, urows=urows, g1=g1, tok=tg)
        if not (tg["T1_pass"] and tg["T2_pass"]):
            rec = {**base, "cohort": coh, "gates": {"G1": g1, "T": tg}, "status": "T_FAILED"}
            L.write_json(L.OUTD / "extract" / f"{coh}__{L.tag(model)}.json", rec)
            raise SystemExit(f"T1/T2 FAILED on {coh}")

    # ---- load the model once
    t1 = time.time()
    if a.mode == "small":
        tok_ef2, net = E._load(model, "causal", device)        # ef17743 verbatim (fp32)
        tok_ef = tok_ef2
    else:
        net = AutoModel.from_pretrained(model, output_hidden_states=True,
                                        torch_dtype=torch.bfloat16, device_map="auto")
        net.eval()
    t_load = time.time() - t1
    base["param_dtype"] = str(next(net.parameters()).dtype)
    base["hf_device_map"] = sorted({str(v) for v in getattr(net, "hf_device_map", {}).values()}) or None
    E._load = lambda *_a, **_k: (tok_ef, net)                  # extract_one body stays verbatim
    if int(transformers.__version__.split(".")[0]) >= 5:
        # transformers 5 no longer turns from_pretrained(output_hidden_states=True) into a config
        # default for these checkpoints, so ef17743's model(**inputs).hidden_states is None. Use the
        # cluster extractor's _forward, which passes output_hidden_states=True in the call -- the
        # forward the cached <env2> features were made with. Tokenization and read-out unchanged.
        E._forward = CUR._forward
        base["forward"] = "cluster _forward (output_hidden_states=True in the call; transformers>=5)"

    shm = Path(a.shm)
    for coh, P in plan.items():
        rec = {**base, "cohort": coh, "t_load_s": round(t_load, 1)}
        gates = {"G1": P["g1"], "T": P["tok"]}
        cached = np.load(L.cached_mid(model, coh), mmap_mode="r")
        cached_layer = L.cached_mid(model, coh).name

        # gate I (small, LibriBrain): all contexts, original batch composition, full file compare
        if a.identity_full and coh == "libribrain":
            if a.mode != "small":
                raise SystemExit("identity_full is for the small checkpoints")
            fdir = shm / "identity"
            t2 = time.time()
            r = E.extract_one(model, "causal", P["ctx_all"], fdir, a.bs, device, 0.05, [0.5])
            newf = fdir / L.tag(model) / f"{r['layers_kept'][0]}.npy"
            NEW = np.load(newf, mmap_mode="r")
            st = {"layer_written": newf.name, "cached_layer": cached_layer,
                  "rows": int(NEW.shape[0]), "t_extract_s": round(time.time() - t2, 1)}
            full = L.cos_stats(np.asarray(NEW), np.asarray(cached))
            selr = L.cos_stats(np.asarray(NEW[P["urows"]]), np.asarray(cached[P["urows"]]))
            st.update(full_file=full, selected_rows=selr,
                      file_bitwise_equal=bool(np.array_equal(np.asarray(NEW), np.asarray(cached))))
            st["pass"] = bool(newf.name == cached_layer and full["median_row_cos"] >= 0.99999
                              and full["frac_rows_cos_ge_0.9999"] >= 0.999)
            gates["I"] = st
            print(f"[gate I] {model}: pass {st['pass']} bitwise_rows {full['frac_rows_bitwise_equal']:.4f} "
                  f"max|d| {full['max_abs_diff']:.3g} med cos {full['median_row_cos']:.10f}", flush=True)
            rec.update(gates=gates, status="IDENTITY_ONLY")
            L.write_json(L.OUTD / "extract" / f"identity__{L.tag(model)}.json", rec)
            shutil.rmtree(fdir, ignore_errors=True)
            continue

        # extraction of the selected rows under the corrected rule
        fdir = shm / coh
        t2 = time.time()
        r = E.extract_one(model, "causal", P["ctx"], fdir, a.bs, device, 0.05, [0.5])
        t_ex = time.time() - t2
        newf = fdir / L.tag(model) / f"{r['layers_kept'][0]}.npy"
        NEW = np.load(newf)
        OLD = np.asarray(cached[P["urows"]], dtype=np.float32)
        g5 = {"shape": list(NEW.shape), "finite": bool(np.isfinite(NEW).all()),
              "zero_rows": int((np.abs(NEW).max(1) == 0).sum()),
              "layer_written": newf.name, "cached_layer": cached_layer}
        g5["pass"] = bool(NEW.shape == OLD.shape and g5["finite"] and g5["zero_rows"] == 0
                          and newf.name == cached_layer)
        gates["G5"] = g5
        rec["D1_new_vs_cached"] = L.cos_stats(NEW, OLD)
        rec["extractor_record"] = r

        # gate W: documented rule, same model object, first 512 extracted rows, vs cached rows
        if a.mode == "ext" and coh == cohorts[0]:
            try:
                tok_doc = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
            except (TypeError, ValueError):
                tok_doc = AutoTokenizer.from_pretrained(model)
            if tok_doc.pad_token is None:
                tok_doc.pad_token = tok_doc.eos_token if tok_doc.eos_token else tok_doc.unk_token
            CUR._load = lambda *_a, **_k: (tok_doc, net)
            wdir = shm / "W"
            k = min(512, len(P["ctx"]))
            rw = CUR.extract_one(model, "causal", P["ctx"][:k], wdir, a.bs, device, 0.05, [0.5],
                                 big=True)
            WN = np.load(wdir / L.tag(model) / f"{rw['layers_kept'][0]}.npy")
            w = L.cos_stats(WN, OLD[:k])
            w["layer_written"] = rw["layers_kept"][0]
            w["pass"] = bool(w["layer_written"] + ".npy" == cached_layer
                             and w["median_row_cos"] >= 0.999 and w["frac_rows_cos_ge_0.99"] >= 0.99)
            gates["W"] = w
            shutil.rmtree(wdir, ignore_errors=True)
            print(f"[gate W] {model}: pass {w['pass']} med cos {w['median_row_cos']:.6f}", flush=True)

        rec["gates"] = gates
        rec["t_extract_s"] = round(t_ex, 1)
        ok = g5["pass"] and gates.get("W", {"pass": True})["pass"]
        rec["status"] = "OK" if ok else "GATE_FAILED"
        if ok and not a.no_copy:
            d, rows_p = L.feat_paths(model, coh)
            d.mkdir(parents=True, exist_ok=True)
            np.save(rows_p, P["urows"].astype(np.int64))
            # $HOME is the only store visible to the CPU scorer and it is near quota. Features of a
            # bf16 model are float32 casts of bf16 values (low 16 mantissa bits zero), so they are
            # stored LOSSLESSLY as their upper 16 bits; the scorer rebuilds the identical float32.
            dst = L.store_features(NEW, d, newf.stem)
            rec["feature_file"] = str(dst)
            rec["feature_sha256"] = L.sha256_file(dst)
            rec["feature_float32_sha256"] = L.sha256_arr(NEW)
            rec["feature_storage"] = "bf16-exact-uint16" if dst.name.endswith(".bf16.npy") else "float32"
            rec["rows_sha256"] = L.sha256_file(rows_p)
        print(f"[{coh}] {model}: {rec['status']} D1 med cos {rec['D1_new_vs_cached']['median_row_cos']:.5f} "
              f"extract {t_ex:.0f}s", flush=True)
        L.write_json(L.OUTD / "extract" / f"{coh}__{L.tag(model)}.json", rec)
        shutil.rmtree(fdir, ignore_errors=True)
    print(f"[done] {model} total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
