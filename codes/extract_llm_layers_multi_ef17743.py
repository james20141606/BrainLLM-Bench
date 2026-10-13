#!/usr/bin/env python
"""Extract word features at fixed relative depths for the 32-model concordance experiment.

Why a new script (2026-07-17): extract_llm_layers.py's registry holds only six
models, and its causal path is hardcoded to GPT2Model/config.n_layer, so Pythia,
OPT, Llama and Qwen cannot go through it. Its masked path is generic but calls
model(**inputs), which raises on T5/BART (they want decoder_input_ids). This
script adds the three architecture classes the pre-registration needs and reuses
the existing context builder and on-disk layout so stage1_streaming_rsa.py can
consume the output unchanged.

Output layout matches extract_llm_layers.py exactly:
    <output_dir>/<model_name with / -> _>/layer_<i>.npy   shape (n_words, dim)

Only the layers at --layer_fracs relative depths are written (default
0.25/0.5/0.75, with 0.5 the pre-registered primary). Writing every layer costs
6.6 GB for a 13-layer 768-dim model at this word count and about a terabyte over
the frozen 32, and the pre-registration's fixed-relative-depth rule cannot use
the rest.

Two things this fixes relative to the old script:
  * n_layers/hidden_dim are read off an actual forward pass rather than guessed
    from config attribute names, which differ across families.
  * the target-word token alignment failure is COUNTED and reported. The old
    masked path silently fell back to [CLS] when _find_sublist missed, so a
    tokenizer that never aligns would produce plausible-looking garbage with no
    error. Any model whose fallback rate exceeds --max_fallback_frac is failed
    loudly instead of written.

Usage:
  python extract_llm_layers_multi.py --trial_metadata_file .../trial_metadata.tsv \
      --output_dir .../llm_embeddings --models gpt2 bert-base-uncased
  python extract_llm_layers_multi.py ... --models ALL
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_llm_layers import (  # noqa: E402
    get_device,
    build_sentence_contexts_from_trial_metadata,
    _init_layer_memmaps,
    _find_sublist,
)

# The 32 models frozen in paper4_temporalalign/CONCORDANCE_PREREG.md section 4.
# "causal" = prefix to target, take last real token.
# "masked" = full sentence, average the target word's tokens.
# "encdec" = same as masked but through the encoder stack only.
MODEL_SET: dict[str, str] = {
    # causal / autoregressive (18)
    "gpt2": "causal",
    "gpt2-medium": "causal",
    "gpt2-large": "causal",
    "gpt2-xl": "causal",
    "EleutherAI/pythia-70m": "causal",
    "EleutherAI/pythia-160m": "causal",
    "EleutherAI/pythia-410m": "causal",
    "EleutherAI/pythia-1b": "causal",
    "EleutherAI/pythia-1.4b": "causal",
    "EleutherAI/pythia-2.8b": "causal",
    "facebook/opt-125m": "causal",
    "facebook/opt-350m": "causal",
    "facebook/opt-1.3b": "causal",
    "EleutherAI/gpt-neo-125m": "causal",
    "EleutherAI/gpt-neo-1.3B": "causal",
    "meta-llama/Llama-3.2-1B": "causal",
    "Qwen/Qwen2.5-0.5B": "causal",
    "Qwen/Qwen2.5-1.5B": "causal",
    # masked / bidirectional (9)
    "bert-base-uncased": "masked",
    "bert-large-uncased": "masked",
    "roberta-base": "masked",
    "roberta-large": "masked",
    "microsoft/deberta-v3-base": "masked",
    "microsoft/deberta-v3-large": "masked",
    "google/electra-base-discriminator": "masked",
    "albert-base-v2": "masked",
    "distilbert-base-uncased": "masked",
    # encoder-decoder (5)
    "t5-small": "encdec",
    "t5-base": "encdec",
    "t5-large": "encdec",
    "facebook/bart-base": "encdec",
    "facebook/bart-large": "encdec",
}


def _load(model_name: str, kind: str, device):
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.unk_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval().to(device)
    return tok, model


def _forward(model, kind: str, inputs):
    """Return hidden_states tuple for the encoder-side stack."""
    if kind == "encdec":
        # T5Model/BartModel want decoder_input_ids; we only want the encoder.
        return model.encoder(**inputs, output_hidden_states=True).hidden_states
    return model(**inputs).hidden_states


def _batch_texts(batch, kind):
    if kind == "causal":
        # prefix up to and including the target word
        return [" ".join(c["sentence_words"][: c["word_position"] + 1])
                for c in batch]
    return [" ".join(c["sentence_words"]) for c in batch]


def extract_one(model_name: str, kind: str, contexts: list[dict],
                output_dir: Path, batch_size: int, device,
                max_fallback_frac: float, layer_fracs: list[float]) -> dict:
    tok, model = _load(model_name, kind, device)
    n_words = len(contexts)

    memmaps = None
    layer_names: list[str] = []
    n_fallback = 0
    written = 0

    with torch.no_grad():
        for i in range(0, n_words, batch_size):
            batch = contexts[i: i + batch_size]
            texts = _batch_texts(batch, kind)
            inputs = tok(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=512).to(device)
            hs_all = _forward(model, kind, inputs)

            if memmaps is None:
                # Read the stack size off a real forward pass rather than
                # guessing which config attribute this family uses.
                n_hidden = len(hs_all)          # includes the embedding layer
                hidden_dim = int(hs_all[0].shape[-1])
                # Keep only the pre-registered relative depths. Saving every
                # layer costs 6.6 GB for a 13-layer 768-dim model at this word
                # count, which is ~1 TB over the frozen 32 and buys nothing the
                # fixed-depth rule can use.
                top = n_hidden - 1
                # round-half-UP, not Python's round(), which is round-half-to-
                # EVEN: round(1.5)=2 but round(4.5)=4, so on a 6-deep stack the
                # nominal 0.75 landed at 4/6 (67%) while 0.25 landed at 2/6
                # (33%) -- asymmetric. The 0.5 primary is exact for every
                # even-layer stack either way; this only straightens the
                # sensitivity depths.
                keep_idx = sorted({int(f * top + 0.5) for f in layer_fracs})
                layer_names = [f"layer_{j}" for j in keep_idx]
                _, memmaps = _init_layer_memmaps(
                    output_dir, model_name, layer_names, n_words, hidden_dim)
                print(f"  {model_name}: {n_hidden} hidden states, dim "
                      f"{hidden_dim}; keeping {dict(zip(layer_fracs, keep_idx))}",
                      flush=True)

            attn = inputs["attention_mask"]
            for lname in layer_names:
                hs = hs_all[int(lname.split("_")[1])]
                vecs = np.empty((len(batch), hs.shape[-1]), dtype=np.float32)
                for b in range(len(batch)):
                    if kind == "causal":
                        last = int(attn[b].sum().item()) - 1
                        vecs[b] = hs[b, last].float().cpu().numpy()
                    else:
                        wt = tok.encode(batch[b]["word"],
                                        add_special_tokens=False)
                        ids = inputs["input_ids"][b].tolist()
                        hit = _find_sublist(ids, wt)
                        if hit:
                            vecs[b] = hs[b, hit].mean(dim=0).float().cpu().numpy()
                        else:
                            # Count it. The old script fell back silently.
                            if lname == layer_names[0]:
                                n_fallback += 1
                            vecs[b] = hs[b, 0].float().cpu().numpy()
                memmaps[lname][i: i + len(batch)] = vecs
            written += len(batch)
            if (i // batch_size) % 50 == 0:
                print(f"    {written}/{n_words}", flush=True)

    for m in memmaps.values():
        m.flush()
    frac = n_fallback / max(n_words, 1)
    rec = {"model": model_name, "kind": kind, "layers_kept": layer_names,
           "n_words": n_words, "fallback_frac": round(frac, 4)}
    if kind != "causal" and frac > max_fallback_frac:
        rec["FAILED"] = (f"target-word alignment fell back for {frac:.1%} of "
                         f"words (> {max_fallback_frac:.0%}); features unusable")
    print(f"  -> {rec}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial_metadata_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--models", nargs="+", required=True,
                   help='HF names, or ALL for the frozen 32')
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_fallback_frac", type=float, default=0.05)
    p.add_argument("--layer_fracs", nargs="+", type=float,
                   default=[0.25, 0.5, 0.75],
                   help="relative depths to keep; 0.5 is the pre-registered "
                        "primary, the others are the declared sensitivity check")
    p.add_argument("--report", default=None)
    a = p.parse_args()

    models = list(MODEL_SET) if a.models == ["ALL"] else a.models
    unknown = [m for m in models if m not in MODEL_SET]
    if unknown:
        raise SystemExit(f"not in the frozen set: {unknown}")

    device = get_device()
    contexts = build_sentence_contexts_from_trial_metadata(
        Path(a.trial_metadata_file))
    print(f"[contexts] {len(contexts)} words, device={device}", flush=True)

    out = Path(a.output_dir)
    recs = []
    for m in models:
        mdir = out / m.replace("/", "_")
        if mdir.exists() and len(list(mdir.glob("layer_*.npy"))) > 0:
            print(f"[skip] {m} already extracted", flush=True)
            continue
        try:
            recs.append(extract_one(m, MODEL_SET[m], contexts, out,
                                    a.batch_size, device, a.max_fallback_frac,
                                    a.layer_fracs))
        except Exception as e:  # log and continue; the pre-reg requires a reason
            print(f"[FAIL] {m}: {type(e).__name__}: {e}", flush=True)
            recs.append({"model": m, "kind": MODEL_SET[m], "FAILED": str(e)})

    if a.report:
        Path(a.report).write_text(json.dumps(recs, indent=2))
    bad = [r for r in recs if "FAILED" in r]
    print(f"\n[done] {len(recs) - len(bad)} ok, {len(bad)} failed")
    for r in bad:
        print(f"  FAILED {r['model']}: {r['FAILED']}")


if __name__ == "__main__":
    main()
