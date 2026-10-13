"""Read-only tokenizer probe for the 14 masked / encoder-decoder checkpoints of paperA's 32 (2026-09-25).

DECISION RULE (written and committed before the probe is run; loads TOKENIZERS ONLY, no weights):

Object: the documented extractor's masked/encdec path (extract_llm_layers_multi.py, cluster copy =
HEAD): AutoTokenizer.from_pretrained(name, add_prefix_space=True) [fallback without it on
TypeError/ValueError]; is_fast required; pad fallback; for each item the WHOLE sentence as a word list,
tok(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512);
read-out = mean over tokens t with word_ids()[t] == word_position (fallback to token 0 if none).
Env: <env> (transformers 4.57.1), the env of the 07-17 masked extraction (JID 14106644).
Input: real LibriBrain contexts from the extractor's own builder: the first 32 contexts, plus the
8 contexts ending at the first word_position >= 19 (same two batches as the causal probe), plus a
fixed sample of 2,000 contexts (numpy RandomState(0) over all 177,127) for rates.

Per tokenizer, three measures:
 (a) boundary loss L = share of non-initial words whose first sub-token is a word CONTINUATION rather
     than a word start. Word start = byte-level BPE token beginning with 'Ġ' (RoBERTa, BART);
     SentencePiece token beginning with '▁' (T5, ALBERT, DeBERTa-v3); WordPiece token NOT beginning
     with '##' (BERT, DistilBERT, ELECTRA). A SentencePiece word whose first piece is a bare '▁' followed
     by the word's pieces counts as a word start.
 (b) same-ids rate S = share of sentences whose word-list ids equal the ids of the same tokenizer on the
     space-joined sentence (" ".join(words), default call). S = 1 means the word-list call changes
     nothing relative to ordinary spaced text.
 (c) target-span correctness T = share of items whose read-out tokens (word_ids == word_position) are
     non-empty and decode (skip special tokens, word-start markers and spaces removed) to the target
     word after the tokenizer's own normalizer (e.g. lower-casing / accent stripping for uncased BERT),
     spaces removed; F = fallback share (no token for the target).

Verdict per tokenizer:
 - NO PROBLEM: L = 0, F = 0 and T >= 0.999 (S is reported; S < 1 alone is not a problem if L = 0,
   because e.g. a leading marker on the first word is the ordinary spaced form).
 - PROBLEM (headline-level; report immediately): L >= 0.5, or T < 0.95, or F > 0.05.
 - PARTIAL / UNCERTAIN: anything else; list the offending items.
 - CANNOT DECIDE: tokenizer not loadable (report, do not substitute).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

R = "<PROJECT>"
sys.path.insert(0, R + "/code")
os.chdir(R)
from pathlib import Path  # noqa: E402

import tokenizers  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from extract_llm_layers import build_sentence_contexts_from_trial_metadata  # noqa: E402
from extract_llm_layers_multi import MODEL_SET, _batch_words  # noqa: E402

OUT = sys.argv[1]
NAMES = [m for m, k in MODEL_SET.items() if k in ("masked", "encdec")]
assert len(NAMES) == 14, NAMES
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__, flush=True)

ctx = build_sentence_contexts_from_trial_metadata(Path(R + "/data/libribrain_full/trial_metadata.tsv"))
j = next(k for k, c in enumerate(ctx) if c["word_position"] >= 19)
rs = np.random.RandomState(0)
samp = sorted(rs.choice(len(ctx), 2000, replace=False).tolist())
ITEMS = {"first_batch_bs32": ctx[0:32], "long_prefix_bs8": ctx[j - 7: j + 1],
         "random2000": [ctx[i] for i in samp]}


def family(tok):
    t = type(tok.backend_tokenizer.pre_tokenizer).__name__ if tok.backend_tokenizer.pre_tokenizer else ""
    m = type(tok.backend_tokenizer.model).__name__
    if m == "WordPiece":
        return "wordpiece"
    if "ByteLevel" in t or "ByteLevel" in json.dumps(json.loads(tok.backend_tokenizer.to_str()).get("pre_tokenizer")):
        return "bytelevel"
    return "sentencepiece"


def is_start(piece, fam, nxt=None):
    if fam == "wordpiece":
        return not piece.startswith("##")
    if fam == "bytelevel":
        return piece.startswith("Ġ")
    return piece.startswith("▁")


def clean(s, tok):
    return s.replace("▁", " ").replace("Ġ", " ").strip()


out = {}
for name in NAMES:
    rec = {}
    try:
        try:
            tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
            rec["aps_accepted"] = True
        except (TypeError, ValueError) as e:
            tok = AutoTokenizer.from_pretrained(name)
            rec["aps_accepted"] = f"fallback: {type(e).__name__}"
        if not tok.is_fast:
            raise RuntimeError("not fast")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token else tok.unk_token
    except Exception as e:
        rec["LOAD_FAIL"] = f"{type(e).__name__}: {e}"[:300]
        out[name] = rec
        print(name, rec, flush=True)
        continue
    fam = family(tok)
    fs = glob.glob(os.path.join(os.environ.get("HF_HOME", ""), "hub", "models--" + name.replace("/", "--"),
                                "snapshots", "*", "tokenizer.json"))
    rec.update(cls=type(tok).__name__, family=fam, padding_side=tok.padding_side,
               tokenizer_json_sha=(hashlib.sha256(open(fs[0], "rb").read()).hexdigest()[:12] if fs else None))
    special = set(tok.all_special_ids)
    for bname, batch in ITEMS.items():
        words_l = _batch_words(batch, "masked")          # whole sentence, as the extractor does
        enc = tok(words_l, is_split_into_words=True, padding=True, truncation=True, max_length=512)
        n_b = n_lost = n_same = n_T = n_F = 0
        bad = []
        for b, words in enumerate(words_l):
            ids = [t for t, m in zip(enc["input_ids"][b], enc["attention_mask"][b]) if m]
            wid = enc.word_ids(batch_index=b)
            pieces = tok.convert_ids_to_tokens(enc["input_ids"][b])
            for wi in range(1, len(words)):
                first = next((t for t, x in enumerate(wid) if x == wi), None)
                if first is None:
                    continue
                p = pieces[first]
                if fam == "sentencepiece" and p == "▁" and first + 1 < len(pieces) and wid[first + 1] == wi:
                    ok = True
                else:
                    ok = is_start(p, fam)
                n_b += 1
                n_lost += (not ok)
                if not ok and len(bad) < 10:
                    bad.append({"word": words[wi], "first_piece": p})
            ref = tok(" ".join(words), truncation=True, max_length=512)["input_ids"]
            n_same += (ref == ids)
            want = batch[b]["word_position"]
            span = [t for t, x in enumerate(wid) if x == want]
            if not span:
                n_F += 1
                continue
            got = clean(tok.decode([enc["input_ids"][b][t] for t in span], skip_special_tokens=True), tok)
            normzr = tok.backend_tokenizer.normalizer
            exp = normzr.normalize_str(words[want]) if normzr is not None else words[want]
            exp = clean(exp, tok)
            n_T += (got.replace(" ", "") == exp.replace(" ", "") and got != "")
            if got != exp and len(bad) < 20:
                bad.append({"target": words[want], "span_decode": got, "alone_decode": exp})
        n = len(words_l)
        rec[bname] = dict(n_items=n, boundaries=n_b, lost=n_lost, L=round(n_lost / max(n_b, 1), 5),
                          S=round(n_same / n, 4), T=round(n_T / n, 5), F=round(n_F / n, 5),
                          examples=bad[:12],
                          sample_tokens=pieces[:30] if bname == "long_prefix_bs8" else None)
    r2 = rec["random2000"]
    L_all = max(rec[k]["L"] for k in ITEMS)
    T_min = min(rec[k]["T"] for k in ITEMS)
    F_max = max(rec[k]["F"] for k in ITEMS)
    if L_all == 0 and F_max == 0 and T_min >= 0.999:
        v = "NO PROBLEM"
    elif L_all >= 0.5 or T_min < 0.95 or F_max > 0.05:
        v = "PROBLEM"
    else:
        v = "PARTIAL / UNCERTAIN"
    rec["verdict"] = v
    out[name] = rec
    print(json.dumps({name: {"family": fam, "verdict": v, "L": L_all, "T_min": T_min, "F_max": F_max,
                             "S_random": r2["S"]}}), flush=True)
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
print("DONE", flush=True)
