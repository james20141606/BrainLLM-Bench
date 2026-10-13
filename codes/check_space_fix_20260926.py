#!/usr/bin/env python3
"""Pin paperA's space-fix (R version) text to the re-extraction receipts (prereg fed178a).

Reads results/p4_encoding/space_fix_20260925.json (summary with the mechanical outcomes), the aggregates under
space_fix_20260925/aggregate/, the per-model score receipts (alpha grid) and the two figure provenance files,
and checks every re-extracted number the TeX prints, the unspaced originals it prints for comparison, the
outcome-dependent wording, and that superseded wording is gone. No writes.

    python codes/check_space_fix_20260926.py [--tex X] [--results Y]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(ROOT / "paper4_temporalalign/tex/paperA_benchmark.tex"))
    ap.add_argument("--results", default=str(ROOT / "paper4_temporalalign/results"))
    a = ap.parse_args()
    tex = Path(a.tex).read_text()
    flat = " ".join(tex.split())
    body = " ".join(tex[:tex.index(r"\appendix")].split())
    absn = " ".join(tex[tex.index(r"\begin{abstract}"):tex.index(r"\end{abstract}")].split())
    res = Path(a.results)
    bundle = not res.is_dir()
    if bundle:
        res = ROOT / "results"          # release layout
    P = res / "p4_encoding"
    SF = P / "space_fix_20260925"
    s = json.loads((P / "space_fix_20260925.json").read_text())
    o = s["outcomes"]
    fails = []

    def need(c, msg):
        if not c:
            fails.append(msg)

    def sg(v, nd):
        x = f"{v:+.{nd}f}"
        return "+" + x[1:] if x == f"-{0:.{nd}f}" else x

    def iv(lo, hi, nd=2, plus=True):
        return f"$[{sg(lo, nd) if plus else f'{lo:.{nd}f}'}, {sg(hi, nd) if plus else f'{hi:.{nd}f}'}]$"

    # completeness and outcomes
    need(s["libribrain"]["missing_unspaced"] == [] and "no checkpoint withdrawn" in s["summary_meta"]["completeness"]
         and s["summary_meta"]["completeness"].startswith("36/36"), "36/36 re-extracted, none withdrawn")
    need(all(v == "re-extracted" for v in s["libribrain"]["status"].values()) and len(s["libribrain"]["status"]) == 25,
         "all 25 LibriBrain extension checkpoints re-extracted")
    want = {"6.1_Q2_48base": "holds", "6.2_Q1_generations": "weakens", "6.4_alpha_grid": "holds",
            "6.5_masc_ladder": "weakens", "6.6_twins": "holds", "6.7_antonello": "holds",
            "6.8_masc27_32": "holds", "6.8_brennan_32": "flips"}
    for k, v in want.items():
        need(o[k]["outcome"] == v, f"outcome {k} = {v} (receipt {o[k]['outcome']})")
    need(o["6.3_readout_dependent_qwen25"]["readout_dependent_sentence"] == "kept", "6.3 readout-dependent kept")

    # 6.1: abstract, Table 1 (table1 checker pins the row), app D comparison
    q2 = o["6.1_Q2_48base"]
    r, (lo, hi) = q2["recomputed"]["rho"], q2["recomputed"]["ci95_family_clustered"]
    # 2026-09-26 tone pass: the "re-extracted after a tokenization defect" wording moved out of the abstract;
    # the re-extraction stays disclosed in app D (pinned below) and the Section 5.3 holds sentence.
    need(f"survives the $48$ base checkpoints of the registered test under family-clustered uncertainty ($\\rho = {r:.2f}$, CI $[{lo:.2f}, {hi:.2f}]$)" in absn,
         "abstract 48-base clause (re-extracted value)")
    pr, (plo, phi) = q2["published"]["rho"], q2["published"]["ci95_family_clustered"]
    need(f"$48$-checkpoint test gave $\\rho={pr:.2f}$ (family-clustered CI $[{plo:.2f},{phi:.2f}]$; re-extracted ${r:.2f}$, $[{lo:.2f},{hi:.2f}]$)" in flat,
         "app D: 48-base original and re-extracted")
    need("with the added checkpoints re-extracted, the reversal also survives the $48$-checkpoint base set" in body
         and "so the registered falsifier that it was an artifact of older models did not fire" in body, "Section 5.3 holds sentence")

    # 6.3 / 6.4: Section 5.4 and the alpha-grid table
    rc = o["6.3_readout_dependent_qwen25"]["recomputed"]
    need(f"($72$B vs $0.5$B, paired $p={rc['enc']['top_minus_bottom_p']:.4f}$) while similarity rises ($p={rc['rsa']['top_minus_bottom_p']:.3f}$)" in body,
         "Section 5.4 p values (re-extracted)")
    ag = o["6.4_alpha_grid"]
    need(f"from ${ag['spearman_size_original']:.3f}$ to ${ag['spearman_size_expanded']:.3f}$, so that slope is regularization-sensitive" in body,
         "Section 5.4 alpha-grid correlations")
    need(f"weakens from ${ag['spearman_size_original']:.3f}$ to ${ag['spearman_size_expanded']:.3f}$" in flat, "app alpha-grid prose")
    need(f"$72$B still scores below $0.5$B by ${-ag['72B_minus_0.5B_expanded']:.4f}$ on the expanded grid" in flat
         and ag["72B_minus_0.5B_expanded"] < 0 and ag["72B_minus_0.5B_original"] < 0, "72B below 0.5B (expanded)")
    tab = tex[tex.rfind(r"\begin{table}", 0, tex.index(r"\label{tab:alpha-grid}")):]
    tab = tab[:tab.index(r"\end{table}")]
    for row in ag["rows"]:
        size = row["model"].split("-")[-1]
        want_row = f"{row['original_mean']:.5f} & {row['expanded_mean']:.5f}"
        need(re.search(rf"^{re.escape(size)}\s*&\s*\d+\s*&\s*{re.escape(want_row)}", tab, re.M) is not None,
             f"alpha table row {size}: {want_row}")
    modes = {}
    for size in ["3B", "7B", "14B", "32B", "72B"]:
        d = json.loads((SF / f"score/libribrain__Qwen_Qwen2.5-{size}__new.json").read_text())["alpha_grid"]["folds"]
        c = collections.Counter(v["expanded_alpha"] for v in d.values()).most_common(1)[0]
        modes[size] = c
        need(abs(np.mean([v["original_score"] for v in d.values()]) - next(r for r in ag["rows"] if r["model"].endswith(size))["original_mean"]) < 1e-12,
             f"alpha receipt mean {size}")
    need(modes["14B"][1] == 10 and all(modes[k][1] == 11 for k in ["3B", "7B", "32B", "72B"])
         and "every rung except $14$B ($10/12$)" in " ".join(tab.split()), "mode-alpha fold counts in caption")

    # 6.2 / 6.6: Section 5.4 labels and twins
    tw = o["6.6_twins"]["recomputed"]
    need(f"(${tw[0].split()[0]}$, ${tw[1].split()[0]}$; exploratory)" in body, f"twins counts {tw}")
    need(not any(g["recomputed"].startswith("SUPPORTED") for g in o["6.2_Q1_generations"]["per_generation"].values())
         and "in no generation does size order both readouts positively" in body, "6.2 main sentence")  # 09-26 AI-review W26: reworded, same claim
    need('registered label was ``mixed\'\' (now ``not supported\'\';' in flat, "app D: Qwen3 label change")

    # 6.5: MEG-MASC ladder
    m = o["6.5_masc_ladder"]["recomputed"]
    need(m["f3"]["rsa"]["passes"] and not m["f3"]["enc"]["passes"], "MASC F3 gate swap in receipt")
    need("the same rungs do not order similarity by size, and the encoding arm fails its rankability gate, so it is not determined" in body,
         "Section 5.4 MASC sentence")
    f3r, f3e = m["f3"]["rsa"], m["f3"]["enc"]
    qr = m["q1_masc"]["rsa"]
    for x in [f"(split-half ${f3r['split_half']:.2f}$, $90\\%$ CI $[{f3r['ci90'][0]:.2f}, {f3r['ci90'][1]:.2f}]$)",
              f"$\\rho = {sg(qr['rho'], 2)}$ with a subject-clustered CI $[{sg(qr['ci95_subject_clustered'][0], 2)}, {sg(qr['ci95_subject_clustered'][1], 2)}]$",
              f"($p = {qr['top_minus_bottom_p']:.2f}$, ${qr['n_positive_top_minus_bottom']}/27$ positive)",
              f"reliability ${f3e['split_half']:.2f}$ with a $90\\%$ CI $[{f3e['ci90'][0]:.2f}, {f3e['ci90'][1]:.2f}]$ that includes zero"]:
        need(x in flat, f"MASC appendix: {x}")
    mp = o["6.5_masc_ladder"]["published"]
    need(f"encoding passed (${mp['f3']['enc']['split_half']:.2f}$, $[{mp['f3']['enc']['ci90'][0]:.2f}, {mp['f3']['enc']['ci90'][1]:.2f}]$)" in flat
         and f"similarity failed (${mp['f3']['rsa']['split_half']:.2f}$, $[{mp['f3']['rsa']['ci90'][0]:.2f}, {mp['f3']['rsa']['ci90'][1]:.2f}]$)" in flat,
         "MASC appendix: unspaced gate outcomes")

    # 6.7: Antonello long-table row
    need(f"(encoding $\\rho={rc['enc']['rho']:.2f}$; $72$B vs $0.5$B $p={rc['enc']['top_minus_bottom_p']:.4f}$)" in flat
         and f"size orders positively ($\\rho={sg(rc['rsa']['rho'], 2)}$, $p={rc['rsa']['top_minus_bottom_p']:.3f}$)" in flat
         and "on MEG-MASC the encoding arm fails its rankability gate and is not determined" in flat
         and "& \\textbf{contradict} \\\\" in flat, "Antonello row (holds, refilled)")

    # 6.8: app D originals, flip wording, depth appendix, refuted appendix
    mm, bb = o["6.8_masc27_32"], o["6.8_brennan_32"]
    for lab, d in (("MASC", mm), ("Brennan", bb)):
        for key in ("published", "recomputed"):
            v, (l, h) = d[key]
            need(f"${sg(v, 3)}$ $[{sg(l, 2)},{sg(h, 2)}]$" in flat, f"app D {lab} {key}")
    # R1 2026-10-10 (PREREG_paperA_brennan_preproc_20261009, K2 = affected): the Brennan re-extraction values above
    # describe un-baselined epochs. They stay pinned only as that labelled history: the sentence that carries
    # them must say so and point to the baseline-corrected result; no text may present them as current.
    k2 = json.loads((res / "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json").read_text())
    need(bb["recomputed"][1][0] > 0 and bb["published"][1][0] < 0
         and f"${sg(bb['recomputed'][0], 3)}$ $[{sg(bb['recomputed'][1][0], 2)},{sg(bb['recomputed'][1][1], 2)}]$; on Brennan epochs "
             "without a baseline the interval then excluded zero, narrowly, and with one the Brennan concordance is not "
             "determined, Appendix~\\ref{app:brennan-preproc})" in flat
         and k2["b"]["branch"] == k2["c"]["branch"] == "not determined",
         "Brennan flip, stated as narrow and as un-baselined history (K2 b/c not determined)")
    need("against MEG-MASC's " + f"${sg(mm['recomputed'][0], 2)}$ (re-extracted; ${sg(mm['published'][0], 2)}$ before)" in flat
         and f"Brennan's ${sg(bb['recomputed'][0], 2)}$ in the submitted version came from epochs without a baseline; "
             f"with one it is ${sg(k2['b']['rho'], 2)}$, not determined" in flat, "depth appendix values (MEG-MASC; Brennan K2 b)")
    need(f"Brennan's ${sg(bb['recomputed'][0], 2)}$ and MEG-MASC's" not in flat, "superseded depth-appendix Brennan comparison gone")
    need(f"(${sg(mm['recomputed'][0], 3)}$ with its three unspaced checkpoints re-extracted, still outside)" in flat, "refuted appendix MASC value")

    # D1 descriptive (prereg rule not met for 6 of 7 N')
    d1 = []
    for f in sorted((SF / "extract").glob("libribrain__Qwen_Qwen3.[58]*.json")):
        d1.append(json.loads(f.read_text())["D1_new_vs_cached"]["median_row_cos"])
    below = sorted(x for x in d1 if x < 0.99)
    need(len(d1) == 7 and len(below) == 6 and f"(median row cosine ${below[0]:.3f}$--${below[-1]:.3f}$;" in flat, f"D1 N' range {below}")

    # figures are drawn from the re-extraction receipts
    fig_dir = ROOT / "paper4_temporalalign/figures_v2/integrated_20260926"
    # R1 2026-10-10: Figure 3 redrawn into integrated_20261010 with the K2 baseline (b) Brennan point in place of
    # the un-baselined concordance_spacefix_brennan_32 point (PA_FIG3_BRENNAN_K2); the scaling figure is unchanged.
    fig3_dir = ROOT / "paper4_temporalalign/figures_v2/integrated_20261010"
    need("integrated_20261010/fig03_concordance.pdf" in tex and "integrated_20260926/fig_scaling_spacefix.pdf" in tex
         and "integrated_20260926/fig03_concordance.pdf" not in tex, "TeX uses the redrawn figures (Figure 3: 20261010)")
    p3 = json.loads((fig3_dir / "fig03_concordance.provenance.json").read_text())
    ins = {Path(x["path"]).name: x["sha256"] for x in p3["inputs"]}
    for nm in ["adr0006_registered_verdicts_spacefix.json", "concordance_spacefix_masc27_32.json",
               "brennan_k2_summary_20261009.json", "concordance32_c4.json"]:
        # repo layout: the provenance hash must equal the receipt on disk. The review bundle redacts host
        # names inside receipts (bytes change; manifest.json keeps both hashes), so there only the name is checked.
        src_f = (SF / "aggregate" / nm if "spacefix" in nm else
                 res / "paperA_rebuttal_20261004" / nm if nm.startswith("brennan_k2") else P / nm)
        need(nm in ins and (bundle or ins[nm] == hashlib.sha256(src_f.read_bytes()).hexdigest()), f"Figure 3 input {nm}")
    need("concordance_spacefix_brennan_32.json" not in ins and "concordance32_brennan_v2.json" not in ins,
         "Figure 3 draws no un-baselined Brennan receipt")
    need(hashlib.sha256((fig3_dir / "fig03_concordance.pdf").read_bytes()).hexdigest() == p3["pdf_sha256"], "Figure 3 pdf = provenance")
    ps = json.loads((fig_dir / "fig_scaling_spacefix.provenance.json").read_text())
    need((bundle or ps["input_sha256"] == hashlib.sha256((SF / "aggregate/concordance_spacefix_libribrain.json").read_bytes()).hexdigest())
         and hashlib.sha256((fig_dir / "fig_scaling_spacefix.pdf").read_bytes()).hexdigest() == ps["pdf_sha256"], "scaling figure input/pdf")

    # 09-25 21:xx verification round (round3/paperA_VERIFY.md): P1/P2 wording pinned to receipts
    rep_block = " ".join(tex[tex.index(r"\label{reproducibility}"):tex.index(r"\appendix")].split())
    # R1 2026-10-07: the counts are derived from the harness registry itself (they were literals 193/33/25,
    # which went stale when the companion analysis added claims); the statement must match the registry.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bench.reproduce as _R
    _sc = collections.Counter(_R.claim_scope(c) for c, _, _ in _R.CLAIMS)
    _nsf = sum("space-fix" in v for v in _R.SUPERSEDED.values())
    need(f"(${len(_R.CLAIMS)}$ registered claims; the run marks the ${_sc['superseded']}$ that pin superseded values kept for provenance, ${_nsf}$ of them values computed on the unspaced features" in rep_block
         and f"and the ${_sc['companion']}$ that belong to a separate boundary-effect analysis" in rep_block,
         f"P1 registry counts in the reproducibility statement ({len(_R.CLAIMS)}/{_sc['superseded']}/{_nsf}/{_sc['companion']})")
    need(f"(the registry has since grown to ${len(_R.CLAIMS)}$ claims)" in flat, "P1 registry count in app N (derived from bench.reproduce, R1)")
    need("the $25$ LibriBrain extension extractions, whose weight load also produced the MEG-MASC ladder rows, passed a weight-identity gate" in flat
         and len([k for k in s["gates"] if k.startswith("W")]) >= 0, "P2-1 W gate scope")
    need("the top panel and the per-story row still include three such checkpoints" in flat
         and "nor were the repeat-removal analysis (Appendix~\\ref{app:refuted}), the Brennan segment reliabilities (Appendix~\\ref{sec:partition}) and the Brennan class gap (Appendix~\\ref{app:mechanism})" in flat
         and "except in ds003020 and other MEG-MASC and Brennan analyses" in body, "P2-2 not-re-run list")
    pg = o["6.2_Q1_generations"]["per_generation"]
    def g(gen, arm):
        st = pg[gen]["stats"][arm]
        return f"${sg(st['rho'], 2)}$ $[{sg(st['ci95_session_clustered'][0], 2)},{sg(st['ci95_session_clustered'][1], 2)}]$"
    six = (f"Qwen2.5 similarity $\\rho={sg(pg['qwen2.5']['stats']['rsa']['rho'], 2)}$ "
           f"$[{sg(pg['qwen2.5']['stats']['rsa']['ci95_session_clustered'][0], 2)},{sg(pg['qwen2.5']['stats']['rsa']['ci95_session_clustered'][1], 2)}]$ "
           f"and encoding {g('qwen2.5', 'enc')}, Qwen3 {g('qwen3', 'rsa')} and {g('qwen3', 'enc')}, Qwen3.5 {g('qwen3.5', 'rsa')} and {g('qwen3.5', 'enc')}")
    six = six.replace("$+0.00$", "$0.00$").replace(",+0.00]", ",0.00]")
    need(six in flat.replace(",0.00]", ",0.00]"), f"P2-3 per-generation CIs: {six}")
    lb = json.loads((P / "concordance32_CORRECTED.json").read_text())["concordance_spearman"]
    need(f"($1.02$ with the three checkpoints re-extracted)" in flat and f"{mm['recomputed'][0] - lb:.2f}" == "1.02", "P2-4 span 1.02")
    need("its training-selected $0.0449093$ is the unspaced counterpart of the $0.04759$" in flat, "P2-5 unspaced marking")
    # \allowbreak hints are layout only (23:xx cold-read pass added more); compare with them removed
    need("p4\\_encoding/space\\_fix\\_20260925/aggregate/adr0006\\_registered\\_verdicts\\_spacefix.json}" in rep_block.replace("\\allowbreak ", ""), "P2-6 pointer")
    ceil = {}
    for size in ["3B", "7B", "14B", "32B", "72B"]:
        d = json.loads((SF / f"score/libribrain__Qwen_Qwen2.5-{size}__new.json").read_text())["alpha_grid"]["folds"]
        ceil[size] = sum(1 for v in d.values() if v["original_alpha"] == 1e6)
    need(all(v == 11 for v in ceil.values()) and "is selected in $11/12$ sessions at every re-extracted rung from $3$B to $72$B" in flat,
         f"original-grid ceiling counts {ceil}")
    need("not confirmed by this check" in flat, "P3-c D1 wording")
    # R1 2026-10-10: the body's Brennan class gap is now the K2 baseline value (pinned to the receipt by
    # check_table1_receipts); the pre-re-extraction, un-baselined "near zero" sentence must be gone.
    need("(Appendix~\\ref{app:mechanism}; before re-extraction)" not in body and "The class gap is near zero on Brennan" not in body
         and "has LibriBrain's sign (Appendix~\\ref{app:brennan-preproc})" in body, "P3-b Brennan class gap: K2 value, old sentence gone")

    # superseded wording
    for bad in ["leave encoding flat", "($3/6$", "$-0.893$ to $-0.464$", "similarity rises ($p=0.002$)",
                "Brennan's interval includes zero", "the EEG interval including zero", "none is changed here",
                "fig03_redrawn.pdf", "MEG-MASC is null", "is again not supported: $\\rho = +0.29$"]:
        need(bad not in flat, f"superseded wording present: {bad!r}")

    if fails:
        print("Space-fix pin: FAIL")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print(f"Space-fix pin: PASS (48-base {r:+.3f} [{lo:+.3f},{hi:+.3f}]; Qwen2.5 enc {rc['enc']['rho']:+.3f}, rsa {rc['rsa']['rho']:+.3f}; "
          f"MASC-32 {mm['recomputed'][0]:+.3f}, Brennan-32 un-baselined receipt {bb['recomputed'][0]:+.3f}; alpha rows 7; D1 N' {below[0]:.3f}-{below[-1]:.3f})")


if __name__ == "__main__":
    main()
