"""Read-only tokenizer probe for paperA (2026-09-25). Loads TOKENIZERS ONLY, no weights.
Replicates verbatim the tokenization of:
  documented rule  = extract_llm_layers_multi.py (ba9ff1f..HEAD, identical in cluster copy 2026-08-17): 
      AutoTokenizer.from_pretrained(name, add_prefix_space=True) [fallback without on TypeError/ValueError];
      pad_token fallback; tok(words, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True, max_length=512)
      readout (causal) = attn[b].sum()-1
  ef17743 rule     = AutoTokenizer.from_pretrained(name); tok(" ".join(prefix), return_tensors='pt', padding=True, truncation=True, max_length=512)
Contexts come from the extractor's own builder on the real LibriBrain trial_metadata.
"""
import os, sys, json, hashlib, glob
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
R = "<PROJECT>"
sys.path.insert(0, R + "/code")
os.chdir(R)
from pathlib import Path
import transformers, tokenizers
from transformers import AutoTokenizer
from extract_llm_layers import build_sentence_contexts_from_trial_metadata
from extract_llm_layers_multi import _batch_words   # the extractor's own helper
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__, flush=True)

ctx = build_sentence_contexts_from_trial_metadata(Path(R + "/data/libribrain_full/trial_metadata.tsv"))
print("n_contexts", len(ctx), flush=True)
BATCHES = {"first_batch_bs32": ctx[0:32]}
# also a long real sentence: the first context whose prefix has >=20 words, plus its 7 predecessors
j = next(k for k, c in enumerate(ctx) if c["word_position"] >= 19)
BATCHES["long_prefix_bs8"] = ctx[j - 7: j + 1]
for k, b in BATCHES.items():
    print(k, "prefix of last item:", " ".join(_batch_words(b, "causal")[-1]), flush=True)

HUB = "<HPC_SCRATCH>/.huggingface/hub"
NAMES = ["gpt2",
         "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-7B-Instruct",
         "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-8B-Base", "Qwen/Qwen3-14B-Base", "Qwen/Qwen3-8B",
         "Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B",
         "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"]

def sha(name, fn):
    fs = glob.glob(f"{HUB}/models--{name.replace('/', '--')}/snapshots/*/{fn}")
    return hashlib.sha256(open(fs[0], "rb").read()).hexdigest()[:12] if fs else None

SPACE_MARKS = ("Ġ", "▁")  # 'Ġ' byte-level, '▁' sentencepiece
out = {}
for name in NAMES:
    rec = {"tokenizer_json_sha": sha(name, "tokenizer.json")}
    try:
        try:
            tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True); rec["aps_accepted"] = True
        except (TypeError, ValueError) as e:
            tok = AutoTokenizer.from_pretrained(name); rec["aps_accepted"] = f"fallback: {type(e).__name__}"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token else tok.unk_token
        old = AutoTokenizer.from_pretrained(name)
        if old.pad_token is None:
            old.pad_token = old.eos_token if old.eos_token else old.unk_token
    except Exception as e:
        rec["LOAD_FAIL"] = f"{type(e).__name__}: {e}"[:300]; out[name] = rec; print(name, rec, flush=True); continue
    rec.update(cls=type(tok).__name__, is_fast=tok.is_fast, padding_side=tok.padding_side,
               padding_side_old=old.padding_side, bos=tok.bos_token, pad=tok.pad_token)
    for bname, batch in BATCHES.items():
        words = _batch_words(batch, "causal")
        enc = tok(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encO = old([" ".join(w) for w in words], return_tensors="pt", padding=True, truncation=True, max_length=512)
        n_bound = n_lost = 0; n_dec_eq = n_dec_eq_old = 0; n_same_ids = 0; n_same_readout = 0
        n_readout_last_word = 0; n_readout_last_word_old = 0; ex = None
        for b, w in enumerate(words):
            ids = enc["input_ids"][b].tolist(); am = enc["attention_mask"][b].tolist()
            wid = enc.word_ids(batch_index=b)
            real = [t for t, m in zip(ids, am) if m]
            for wi in range(1, len(w)):
                first = next((t for t, x in enumerate(wid) if x == wi), None)
                if first is None: continue
                n_bound += 1
                piece = tok.convert_ids_to_tokens(ids[first])
                dec = tok.decode([ids[first]])
                if not (piece.startswith(SPACE_MARKS) or dec.startswith(" ")): n_lost += 1
            dec_new = tok.decode(real, skip_special_tokens=True)
            n_dec_eq += dec_new.strip() == " ".join(w)
            r = int(sum(am)) - 1                      # extractor readout
            n_readout_last_word += (wid[r] == len(w) - 1)
            idsO = encO["input_ids"][b].tolist(); amO = encO["attention_mask"][b].tolist()
            realO = [t for t, m in zip(idsO, amO) if m]
            decO = old.decode(realO, skip_special_tokens=True)
            n_dec_eq_old += decO.strip() == " ".join(w)
            rO = int(sum(amO)) - 1
            # is the old readout the last real token (right padding) ?
            lastrealO = max(t for t, m in enumerate(amO) if m)
            n_readout_last_word_old += (rO == lastrealO)
            n_same_ids += real == realO
            n_same_readout += ids[r] == idsO[rO]
            if b == len(words) - 1:
                ex = {"words": " ".join(w), "new_tokens": tok.convert_ids_to_tokens(real)[:40],
                      "new_decode": dec_new[:200], "old_tokens": old.convert_ids_to_tokens(realO)[:40],
                      "old_decode": decO[:200],
                      "new_readout_tok": tok.convert_ids_to_tokens(ids[r]), "old_readout_tok": old.convert_ids_to_tokens(idsO[rO])}
        n = len(words)
        rec[bname] = dict(n_items=n, boundaries=n_bound, lost=n_lost, L=round(n_lost / max(n_bound, 1), 4),
                          decode_equal_new=f"{n_dec_eq}/{n}", decode_equal_old=f"{n_dec_eq_old}/{n}",
                          same_token_ids_new_vs_old=f"{n_same_ids}/{n}", same_readout_token=f"{n_same_readout}/{n}",
                          new_readout_in_target_word=f"{n_readout_last_word}/{n}",
                          old_readout_is_last_real_token=f"{n_readout_last_word_old}/{n}", example=ex)
    out[name] = rec
    print(json.dumps({name: rec}, ensure_ascii=False), flush=True)
json.dump(out, open("<HPC_SCRATCH>/probes/paperA_tok_20260925/probe_out.json", "w"), indent=1, ensure_ascii=False)
print("DONE", flush=True)
