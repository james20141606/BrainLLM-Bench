#!/usr/bin/env python3
"""Pin the feature-extraction provenance disclosure (paperA, appendix paragraph
app:extract-provenance, reproducibility statement, bundle README) to the G3 gate receipts.

Source data (already on disk, no new computation): the per-model gate receipts of the C4 control,
results/p4_encoding/c4_static32/per_model/
  <17 causal>_wordids_G3fail.json, gpt2_reextract_smoke_G3fail.json   documented extractor vs cache
  <14 masked / encoder-decoder>.json                                   same comparison, kind != causal
Each carries G3 = row cosine of the re-extracted depth-0.5 features against the cached headline
rows on the 30,000 scored words. Recomputes every number the disclosure prints and checks it is in
the TeX (and in the README when --readme is given). No writes.

    python codes/check_extract_provenance_20260925.py
    python codes/check_extract_provenance_20260925.py --tex X --results Y --readme Z   # mutation tests
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAUSAL18 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "facebook/opt-125m",
            "facebook/opt-350m", "facebook/opt-1.3b", "EleutherAI/gpt-neo-125m",
            "EleutherAI/gpt-neo-1.3B", "meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B"]
MASKED14 = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large",
            "microsoft/deberta-v3-base", "microsoft/deberta-v3-large",
            "google/electra-base-discriminator", "albert-base-v2", "distilbert-base-uncased",
            "t5-small", "t5-base", "t5-large", "facebook/bart-base", "facebook/bart-large"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(ROOT / "paper4_temporalalign/tex/paperA_benchmark.tex"))
    ap.add_argument("--results", default=None)
    ap.add_argument("--readme", default=None)
    a = ap.parse_args()
    res = Path(a.results) if a.results else ROOT / "paper4_temporalalign/results"
    if not res.is_dir():
        res = ROOT / "results"
    D = res / "p4_encoding/c4_static32/per_model"
    fails = []

    def need(c, msg):
        if not c:
            fails.append(msg)

    g3c = {}
    for m in CAUSAL18:
        tag = m.replace("/", "_")
        f = D / (f"{tag}_reextract_smoke_G3fail.json" if m == "gpt2" else f"{tag}_wordids_G3fail.json")
        r = json.loads(f.read_text())
        need(r["model"] == m and r["kind"] == "causal", f"receipt identity {f.name}")
        need("ef17743" not in r.get("extractor", ""), f"{f.name} must be the documented extractor")
        need(r["row_sha256"].startswith("bfe5c58de6dc"), f"row hash {f.name}")
        g = r["gates"]["G3"]
        need(g["pass"] is False, f"{f.name} G3 should be a failure")
        g3c[m] = (g["median_row_cos"], 1.0 - g["frac_rows_cos_ge_0.9999"])
    g3m = {}
    for m in MASKED14:
        r = json.loads((D / (m.replace("/", "_") + ".json")).read_text())
        need(r["kind"] in ("masked", "encdec"), f"kind {m}")
        g3m[m] = r["gates"]["G3"]["median_row_cos"]
        need(r["gates"]["G3"]["pass"], f"masked G3 pass {m}")

    neo = "EleutherAI/gpt-neo-125m"
    others = [m for m in CAUSAL18 if m != neo]
    lo_other = min(g3c[m][1] for m in others)
    hi_other = max(g3c[m][1] for m in others)
    special = {"meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"}
    rest = [m for m in CAUSAL18 if m not in special]
    exp = {
        "neo share": (f"${round(100 * g3c[neo][1])}\\%$ (GPT-Neo-125M)", ),
        "other share": (f"${int(100 * lo_other)}$--${100 * hi_other:.1f}\\%$ (the other",),
        "llama": (f"${g3c['meta-llama/Llama-3.2-1B'][0]:.3f}$\n(Llama-3.2-1B)",
                  f"${g3c['meta-llama/Llama-3.2-1B'][0]:.3f}$ (Llama-3.2-1B)"),
        "qwen": (f"${g3c['Qwen/Qwen2.5-0.5B'][0]:.3f}$ and ${g3c['Qwen/Qwen2.5-1.5B'][0]:.3f}$",),
        "rest range": (f"${min(g3c[m][0] for m in rest):.4f}$--"
                       f"${max(g3c[m][0] for m in rest):.4f}$ for",),
        "masked": (f"${min(g3m.values()):.7f}$ for all $14$",),
    }
    need(len(rest) == 15, "15 others")
    tex = Path(a.tex).read_text()
    i = tex.index(r"\label{app:extract-provenance}")
    par = tex[i:i + 14000]      # 2026-09-25 14:10 rewrite is ~4.3k chars (was 2500)
    par = par[:par.index("\n\n")] if "\n\n" in par else par
    flat = re.sub(r"\s+", " ", par)
    for k, alts in exp.items():
        need(any(re.sub(r"\s+", " ", x) in flat for x in alts), f"TeX {k}: expected {alts[0]!r}")
    need("$18$ autoregressive" in flat and "($17$ by an earlier revision" in flat and "GPT-2 small" in flat,
         "TeX 18 = 17 + GPT-2 small")
    need("The $32$-checkpoint LibriBrain headline read spaced text throughout and is unchanged" in flat,
         "TeX: headline unchanged statement (R version 09-25)")
    rep = re.sub(r"\s+", " ", tex[tex.index(r"\label{reproducibility}"):tex.index(r"\appendix")])
    need(r"Appendix~\ref{app:extract-provenance}" in rep
         and "Re-extracting the autoregressive ones with the documented revision does not reproduce them" in rep
         and "$31$ of its $32$ checkpoints also by re-extraction: the $17$ autoregressive ones with the earlier extractor revision" in rep
         and "checkpoints, $18$ were extracted with the documented extractor under a library version at which" in rep,
         "reproducibility statement qualifier")

    # 2026-09-25 14:10 (acceptance #1/#1b): tokenizer probe receipt -> the space-drop facts, and the
    # collinearity disclosure in the paragraph and in the main text
    pr = json.loads((res / "p4_encoding/tokenizer_probe_20260925/probe_out.json").read_text())
    ql = [k for k in pr if k != "gpt2"]
    need(all(k.startswith(("Qwen/", "meta-llama/")) for k in ql), "probe: only Qwen/Llama besides gpt2")
    lost = {k: pr[k]["first_batch_bs32"]["lost"] + pr[k]["long_prefix_bs8"]["lost"] for k in pr}
    bnd = {k: pr[k]["first_batch_bs32"]["boundaries"] + pr[k]["long_prefix_bs8"]["boundaries"] for k in pr}
    need(lost["gpt2"] == 0, "probe: gpt2 control loses no space")
    need(len(set(bnd.values())) == 1 and all(lost[k] == bnd[k] for k in ql), "probe: every Qwen/Llama boundary lost")
    need(all(pr[k][b]["new_readout_in_target_word"].split("/")[0] == pr[k][b]["new_readout_in_target_word"].split("/")[1]
             for k in pr for b in ("first_batch_bs32", "long_prefix_bs8")), "probe: read-out stays in the target word")
    need(f"(${bnd[ql[0]]}/{bnd[ql[0]]}$ word boundaries for each of the ${len(ql)}$ Qwen and Llama tokenizers probed" in flat,
         f"TeX probe count {bnd[ql[0]]}/{bnd[ql[0]]}, {len(ql)} tokenizers")
    # 14:40 correction (prereg fed178a §2.1): Qwen3.5 x5 and one post-trained checkpoint were extracted under
    # transformers 5.15.0, where the same probe loses 0 boundaries; 18 of 25 extension checkpoints are unspaced
    pre_p = ROOT / "paper4_temporalalign/PREREG_space_fix_reextract_20260925.md"
    if not pre_p.is_file():
        pre_p = ROOT / "adr/PREREG_space_fix_reextract_20260925.md"   # release layout
    prereg = pre_p.read_text()
    need("0/128 and 0/124" in prereg and "18 checkpoints are U (12 of the 16 added base, 6 of the 9 post-trained)" in prereg,
         "prereg fed178a: v5 probe 0/252 and the 18 U count")
    need(bnd["gpt2"] == 252, "probe: 252 boundaries per tokenizer")
    for s in ["transformers $4.57.1$", "inserts no space between words", "transformers $5.15.0$", "($0/252$)",
              "extension checkpoints, $18$ were first scored on unspaced text: $12$ of the $16$ base checkpoints",
              "and $6$ of the $9$ post-trained ones", "the Qwen3.5 ladder and its twin were not",
              "rule is collinear with objective class among the $32$ checkpoints",
              "so the cross-cohort contrast also crosses the two rules",
              "$72$B-versus-$0.5$B contrast compared the two input conditions",
              "\\emph{Re-extraction.} Under a pre-registration written after the defect was found and before any re-extracted feature existed",
              "ds003020 and the MEG-MASC story, listener-half and crossed-narrative analyses were not re-run", "C4 was not rerun",
              "not on recomputed features"]:
        need(s in flat, f"TeX paragraph: {s!r}")
    tflat = re.sub(r"\s+", " ", tex)
    for bad in ["all $25$ extension checkpoints were scored on unspaced", "the Qwen3 and Qwen3.5 ladders",
                "The $25$ extension checkpoints were extracted with the documented extractor, which"]:
        need(bad.lower() not in tflat.lower(), f"superseded 25/25-unspaced wording: {bad!r}")
    # 09-25 round 3 (P1-4): the 14 masked / encoder-decoder tokenizers, probed under the word-list rule
    mp = json.loads((res / "p4_encoding/tokenizer_probe_20260925/masked_probe_out.json").read_text())
    need(sorted(mp) == sorted(MASKED14), "masked probe covers exactly the 14")
    need(all(mp[m]["verdict"] == "NO PROBLEM" and all(mp[m][b]["lost"] == 0 for b in ("first_batch_bs32", "long_prefix_bs8", "random2000"))
             and mp[m]["random2000"]["n_items"] == 2000 for m in mp), "masked probe: 0 lost boundaries, verdict NO PROBLEM")
    nb = {mp[m]["random2000"]["boundaries"] for m in mp}
    n0 = next(iter(nb))
    need(len(nb) == 1 and f"($0$ of ${n0 // 1000}{{,}}{n0 % 1000:03d}$ word boundaries each, in $2{{,}}000$ LibriBrain sentences)" in flat,
         f"TeX masked-probe sentence ({nb})")
    need("were not checked under this rule" not in flat, "superseded 'not checked' wording")
    main = re.sub(r"\s+", " ", tex[:tex.index(r"\appendix")])
    # 09-25 de-numbering: the main text keeps the 15-of-18 statement; the cosine range is pinned in the paragraph above ("rest range")
    # 09-26 rereview: 5.3 now names context direction, pooling and rule; section 6 names rule and fold
    # R1 2026-10-07: the registered reading-rule test (branch B, readrule_20260926) replaced "no test here
    # separates these from class"; 5.3 now states its result and keeps the tokenization-rule disclosure
    # (the numbers themselves are pinned by check_paperA_revision_r1_20261007.py)
    need("no test here separates these" not in main, "superseded 5.3 wording 'no test here separates these' is gone")
    for s in ["On LibriBrain class also fixes the tokenization rule, which changes $15$ of the $18$ autoregressive feature sets little (Appendix~\\ref{app:extract-provenance})",
              "and under a common reading rule it is not determined (Appendix~\\ref{app:readrule})",
              "cross-cohort contrast also crosses the tokenization rule of autoregressive features and the encoding fold (Appendices~\\ref{app:extract-provenance},~\\ref{app:depth})"]:
        need(s in main, f"TeX main-text disclosure: {s!r}")
    need(len(re.findall(r"\\label\{app:extract-provenance\}", tex)) == 1, "label unique")

    if a.readme:
        rd = re.sub(r"\s+", " ", Path(a.readme).read_text())
        for s in [f"{round(100 * g3c[neo][1])}% (GPT-Neo-125M)",
                  f"{int(100 * lo_other)}-{100 * hi_other:.1f}% (the other 17)",
                  f"{g3c['meta-llama/Llama-3.2-1B'][0]:.3f} (Llama-3.2-1B)",
                  f"{g3c['Qwen/Qwen2.5-0.5B'][0]:.3f} and {g3c['Qwen/Qwen2.5-1.5B'][0]:.3f}",
                  f"{min(g3m.values()):.7f} for all 14",
                  "codes/extract_llm_layers_multi_ef17743.py"]:
            need(s in rd, f"README: {s!r}")

    if fails:
        print("Extraction provenance pin: FAIL")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print(f"Extraction provenance pin: PASS (18 causal G3 receipts: rows below 0.9999 "
          f"{100 * g3c[neo][1]:.1f}% GPT-Neo-125M, {100 * lo_other:.1f}-{100 * hi_other:.1f}% others; "
          f"median cos min {min(v[0] for v in g3c.values()):.4f}; 14 masked/encdec min median "
          f"{min(g3m.values()):.7f})")


if __name__ == "__main__":
    main()
