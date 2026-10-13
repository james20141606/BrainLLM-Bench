#!/usr/bin/env python3
"""Pin every number the paperA discussion-period revision (R1, 2026-10-07) adds, at the place it is printed.

R1 integrates results that were not in the submitted PDF (MEG 6f43d2ac):
  A1  reading-rule test, branch B        results/readrule_20260926/readrule_summary.json (+ PREREG_readrule_20260926 sec. 7)
  A2  ridge-range test, branch A         results/alpha32_20260926/alpha32_summary.json
  A3  cohort family-clustered CIs        recomputed here with codes/recompute_corrected_stats.family_clustered_ci
                                         (seed 0, 10,000) from space_fix_20260925/aggregate/concordance_spacefix_*_32.json
  U*  session target scale + re-score    results/paperA_rebuttal_20261004/session9_raw/diag.json, session9_summary_20261005.json,
                                         session9_raw/rescore/*.json (per-session concordance recomputed here)
      Qwen2.5 ladder rescaled (arm C)    results/paperA_rebuttal_20261004/qwen_ladder_summary_20261005.json
  C4  seed sweep                         results/paperA_rebuttal_20261004/paperA_rebuttal_recompute_20261004.json (part D)
  A5  Brennan lineage                    exp_boundary_conditional/{brennan47_derived_20261003.json, brennan47_offset_reextract.tsv,
                                         boundary_brennan_persubject.tsv, boundary_conditional_v2.tsv}, brennan_align_raw/*.json,
                                         paperB_phase1_20261005/BRANCHES_read_20261005.json, the Fig. 9 provenance JSON
  A4  Schrimpf wording, A6 abstract +0.764/+0.961: text pins. (The AI-PROVENANCE block itself was
      deleted 2026-10-08, author decision -- the AI-use statement reverts to the submitted paragraph.)

Every value is derived from a receipt (or recomputed) and matched as a literal in context, in the section
where it is printed (abstract / body / appendix), so a digit change, a sign flip or a move to another section
fails. Superseded sentences of the submitted version are pinned as absent. Table 1 rows are pinned by
codes/check_table1_receipts.py; this script covers everything else.

Usage:  python3 codes/check_paperA_revision_r1_20261007.py [--tex PATH] [--results DIR] [--selftest]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "paper4_temporalalign/results"
if not RESULTS.is_dir():
    RESULTS = ROOT / "results"                      # release layout
DEFAULT_TEX = ROOT / "paper4_temporalalign/tex/paperA_benchmark.tex"
if not DEFAULT_TEX.exists():
    DEFAULT_TEX = ROOT / "tex/paperA_benchmark.tex"
PREREG_READRULE = [ROOT / "paper4_temporalalign/notes/PREREG_readrule_20260926.md",
                   ROOT / "adr/PREREG_readrule_20260926.md"]
sys.path.insert(0, str(HERE))
from recompute_corrected_stats import family_clustered_ci  # noqa: E402


# ------------------------------------------------------------------ formatting (as the tex prints numbers)
def f(v: float, nd: int) -> str:
    s = f"{v:+.{nd}f}"
    return "+" + s[1:] if s == f"-{0:.{nd}f}" else s


def cis(lo: float, hi: float, nd: int = 2) -> str:          # "[-0.40, +0.34]" (prose form, with a space)
    return f"[{f(lo, nd)}, {f(hi, nd)}]"


def cit(lo: float, hi: float, nd: int = 2) -> str:          # "[-0.40,+0.34]" (table form)
    return f"[{f(lo, nd)},{f(hi, nd)}]"


def norm(s: str) -> str:
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)                 # strip comments, keep \%
    return re.sub(r"\s+", " ", s)


class Pins:
    def __init__(self):
        self.n, self.fail = 0, []

    def check(self, cid: str, ok: bool, msg: str):
        self.n += 1
        print(("  ok    " if ok else "  FAIL  ") + f"{cid:30s} {msg}")
        if not ok:
            self.fail.append(cid)

    def has(self, cid: str, hay: str, needle: str, where: str):
        self.check(cid, needle in hay, f"{needle[:90]!r} in {where}")

    def hasnt(self, cid: str, hay: str, needle: str, where: str):
        self.check(cid, needle not in hay, f"superseded {needle[:80]!r} absent from {where}")


def rj(p: Path):
    return json.loads(p.read_text())


def run(tex_raw: str, R: Path) -> Pins:
    P = Pins()
    T = norm(tex_raw)
    a0, a1 = T.find(r"\begin{abstract}"), T.find(r"\end{abstract}")
    ABS, BODY, APPX = T[a0:a1], T[a1:T.find(r"\appendix")], T[T.find(r"\appendix"):]
    MAIN = BODY[: BODY.find(r"\subsection*{AI Use Statement}")]      # main text only (no statements)

    def sect(start: str, end: str, hay: str = APPX) -> str:
        i = hay.find(start)
        if i < 0:
            return ""
        j = hay.find(end, i + len(start))
        return hay[i: j if j > 0 else len(hay)]

    APP_D_RR = sect(r"\paragraph{Common reading rule (registered).}", r"The $27$-cell inference grid")
    APP_L = sect(r"\section{Literature-Inspired Hypotheses Under Changed Settings}", r"\section{Narrative Identity")
    APP_N = sect(r"\section{Numerical Failure Modes and Their Detection}", r"\section{Geometry Sensitivity")
    APP_J = sect(r"\section{Static-Embedding Control (C4)}", r"\section{Broadband versus")
    APP_B1 = sect(r"\subsection{Prior Work and Benchmark Comparison}", r"\subsection{Executed Route Matrix}")
    BRL = sect(r"\paragraph{Brennan lineage.}", r"\section{")
    S53 = sect(r"\subsection{Dependence on the Readout}", r"\subsection{Additional Findings}", BODY)
    S54 = sect(r"\subsection{Additional Findings}", r"\subsection{Fragility of Score Computation}", BODY)
    S55 = sect(r"\subsection{Fragility of Score Computation}", r"\section{Discussion and Limitations}", BODY)
    LIM = sect(r"\section{Discussion and Limitations}", r"\subsection*{AI Use Statement}", BODY)
    INTRO = sect(r"\section{Introduction}", r"\section{Related Work}", BODY)
    for name, s in (("App D reading-rule paragraph", APP_D_RR), ("App L", APP_L), ("App N", APP_N), ("App J", APP_J),
                    ("App B.1", APP_B1), ("Brennan lineage", BRL), ("5.3", S53), ("5.4", S54), ("5.5", S55),
                    ("Limitations", LIM), ("Introduction", INTRO)):
        P.check("SECT_" + name.split()[0] + name.split()[-1], bool(s), f"section located: {name}")

    # ================= A1 reading rule (registered, branch B) =================
    rr = rj(R / "readrule_20260926/readrule_summary.json")
    V = rr["variants"]
    pl, pm, fl, fm = V["prefix_last"], V["prefix_mean"], V["full_last"], V["full_mean"]
    P.check("A1_receipt", rr["status"] == "complete" and rr["primary"] == {"rule": "prefix_last", "branch": "B",
            "label": "not determined (family CI includes zero)"} and not rr["missing"] and not rr["gate_failures"],
            "readrule receipt complete, primary prefix_last branch B, no gate failures")
    nonc = [m for m in pl["per_model"] if m["kind"] != "causal"]
    n14 = len(nonc)
    rho_ind = spearmanr([m["rsa_mean"] for m in pl["per_model"]], [m["enc_mean"] for m in pl["per_model"]])[0]
    P.check("A1_rho_independent", abs(rho_ind - pl["rho"]) < 1e-12 and n14 == 14 and len(pl["per_model"]) == 32,
            f"rho_PL recomputed from per-model means {rho_ind:+.4f}; {n14} non-causal of {len(pl['per_model'])}")
    lo, hi = pl["family_ci95"]
    P.check("A1_branchB", pl["rho"] < 0 and lo <= 0 <= hi and pl["reliability"]["rsa_sb"] >= 0.70
            and pl["reliability"]["enc_sb"] >= 0.70, "branch B rule (rho<0, L<=0<=U, both reliabilities >= 0.70)")
    r3, c2 = f(pl["rho"], 3), cis(lo, hi)
    P.has("A1_abs", ABS, f"($\\rho = {r3}$, CI ${c2}$)", "abstract")
    P.has("A1_abs_14", ABS, f"when the ${n14}$ masked and encoder--decoder checkpoints are read up to the target word, as the causal ones are", "abstract")
    P.has("A1_abs_conv", ABS, "for the readouts as conventionally applied (masked models also read the words after the target)", "abstract")
    P.has("A1_intro", INTRO, f"the ${n14}$ masked and encoder--decoder checkpoints leave it not determined", "introduction")
    share = round(100 * rr["comparator_decomposition"]["between_class_share_of_abs_rho"])
    P.has("A1_intro_56", INTRO, f"under the class-specific reading rule ${share}\\%$ of it is a between-class offset", "introduction")
    P.has("A1_53_56", S53, f"under the class-specific reading rule, ${share}\\%$ of the effect is a between-class offset", "5.3")
    P.has("A1_53", S53, f"the ${n14}$ give $\\rho={r3}$ (family-clustered CI ${c2}$; registered", "5.3")
    P.has("A1_53_verbatim", S53, "and under a common reading rule it is not determined (Appendix~\\ref{app:readrule})", "5.3 (prereg sec. 7-B main-text sentence)")
    P.hasnt("A1_53_old", BODY, "no test here separates these from class", "body")
    P.has("A1_lim", LIM, f"every checkpoint read up to the target word ($\\rho={r3}$; Appendix~\\ref{{app:readrule}})", "Limitations")
    P.has("A1_disc", LIM, "not determined under a common reading rule", "Discussion")
    # registered disclosure sentence, verbatim from the prereg (sec. 7, first quoted sentence)
    pre = next((p for p in PREREG_READRULE if p.exists()), None)
    if pre is None:
        P.check("A1_prereg", False, "PREREG_readrule_20260926.md not found")
    else:
        q = re.search(r'Appendix D, after the provenance paragraph.*?\n"(.*?)"', pre.read_text(), re.S)
        disclosure = " ".join(q.group(1).split()) if q else ""
        P.check("A1_prereg", bool(disclosure), "prereg sec. 7 disclosure sentence parsed")
        P.has("A1_disclosure_verbatim", APP_D_RR, disclosure, "App D reading-rule paragraph")
    P.has("A1_D_branchB", APP_D_RR, f"gives $\\rho = {r3}$, family-clustered CI ${c2}$ including zero, so under a common reading rule the reversal is not determined.", "App D (prereg sec. 7-B, filled)")
    P.has("A1_D_timing", APP_D_RR, "computed and committed about two hours before the submission deadline and was not integrated into the submitted version", "App D")
    rel = pl["reliability"]
    P.has("A1_D_rel", APP_D_RR, f"split-half ${rel['rsa_sb']:.3f}$ for similarity, ${rel['enc_sb']:.3f}$ for encoding", "App D")
    d, dl = pl["delta_vs_comparator"], pl["delta_paired_family_ci95"]
    P.has("A1_D_delta", APP_D_RR, f"the paired change is ${f(d, 3)}$ (family-clustered CI ${cis(*dl)}$)", "App D")
    P.has("A1_D_fulllast", APP_D_RR, f"last sub-token gives ${f(fl['rho'], 3)}$ (family-clustered CI ${cis(*fl['family_ci95'])}$)", "App D (descriptive, both CIs)")
    P.has("A1_D_prefixmean", APP_D_RR, f"sub-token averaging gives ${f(pm['rho'], 3)}$ (${cis(*pm['family_ci95'])}$)", "App D (descriptive, both CIs)")
    pub = {m["model"]: m for m in pl["per_model"]}
    n_enc_up = sum(m["enc_mean"] > m["enc_mean_published"] for m in nonc)
    n_rsa_dn = sum(m["rsa_mean"] < m["rsa_mean_published"] for m in nonc)
    P.has("A1_D_counts", APP_D_RR, f"all ${n_enc_up}$ checkpoints rise in encoding and ${n_rsa_dn}$ fall in similarity", "App D")
    P.check("A1_D_counts_all", n_enc_up == n14, f"all non-causal rise in encoding ({n_enc_up}/{n14})")
    P.has("A1_D_within", APP_D_RR, f"concordance is ${pl['decomposition']['within_class_centred_rho']:+.2f}$", "App D")
    for cid, lab, v, delta in (("A1_T_pub", "whole sentence (published) & sub-token mean", fm, None),
                               ("A1_T_pl", "prefix to the target (primary) & last sub-token", pl, True),
                               ("A1_T_pm", "prefix to the target (descriptive) & sub-token mean", pm, True),
                               ("A1_T_fl", "whole sentence (descriptive) & last sub-token", fl, True)):
        cell = f"{lab} & ${f(v['rho'], 3)}$ & ${cis(*v['family_ci95'])}$ & "
        cell += (f"${f(v['delta_vs_comparator'], 3)}$ ${cis(*v['delta_paired_family_ci95'])}$" if delta else "---")
        P.has(cid, APP_D_RR, cell, "App D reading-rule table")
    P.has("A1_T_fisher", APP_D_RR, f"Fisher-$z$ interval of the primary: ${cis(*pl['fisher_ci95'])}$", "App D reading-rule table note")
    # timing claim: summary committed before the 2026-09-26 19:59 (+08:00) deadline, ~2 h before (git, if available)
    g = subprocess.run(["git", "log", "-1", "--diff-filter=A", "--format=%cI", "--", str(R / "readrule_20260926/readrule_summary.json")],
                       cwd=ROOT, capture_output=True, text=True)
    if g.returncode == 0 and g.stdout.strip():
        from datetime import datetime
        t = datetime.fromisoformat(g.stdout.strip())
        dl_t = datetime.fromisoformat("2026-09-26T19:59:00+08:00")
        P.check("A1_git_timing", 1.5 * 3600 <= (dl_t - t).total_seconds() <= 2.5 * 3600, f"readrule summary committed {t.isoformat()} (about two hours before 19:59)")
    # ================= A2 ridge range (registered, branch A) =================
    al = rj(R / "alpha32_20260926/alpha32_summary.json")
    ex, ep = al["variants"]["exp/score"], al["variants"]["exp_pt/score"]
    P.check("A2_receipt", al["status"] == "complete" and al["primary"]["branch"] == "A" and ex["family_ci95"][1] < 0 and not al["missing"],
            "alpha32 complete, branch A (family CI upper < 0)")
    top, tot = al["orig_grid_top_counts"]["checkpoint_folds_at_1e6"], al["orig_grid_top_counts"]["checkpoint_folds_total"]
    per = {m["orig_folds_at_top_1e6"] for m in al["per_model"]}
    idx = {int(np.argmin(np.array(m["orig_alphas"]) >= 1e6)) for m in al["per_model"]}
    P.check("A2_folds", top == sum(m["orig_folds_at_top_1e6"] for m in al["per_model"]) and per == {11} and idx == {8},
            f"{top}/{tot} at 1e6; per checkpoint {per}; non-top fold index {idx} (= session 9 of 1..12)")
    P.has("A2_L_folds", APP_L, f"selected in ${top}$ of the ${tot}$ checkpoint-folds, ${per.pop()}$ of $12$ for every checkpoint, and the exception is always the fold that holds session~$9$ out", "App L")
    P.has("A2_L_exp", APP_L, f"leaves the readout concordance at $\\rho = {f(ex['rho'], 3)}$ (family-clustered CI ${cis(*ex['family_ci95'])}$; paired change from $-0.573$, ${cis(*ex['delta_paired_family_ci95'])}$); per-sensor $\\alpha$ gives ${f(ep['rho'], 3)}$ (${cis(*ep['family_ci95'])}$)", "App L")
    names = {"gpt2": "GPT-2", "bert-base-uncased": "BERT-base", "roberta-base": "RoBERTa-base", "facebook/bart-large": "BART-large"}
    flagged = al["boundary_flag_exp_ge6_of_12_at_1e14"]
    nfl = {m["exp_folds_at_top_1e14"] for m in al["per_model"] if m["model"] in flagged}
    P.check("A2_flags", set(flagged) == set(names) and nfl == {11}, f"boundary-flagged {flagged} at {nfl}/12")
    P.has("A2_L_flags", APP_L, "GPT-2, BERT-base, RoBERTa-base and BART-large select the new ceiling 1e14 in $11$ of $12$ folds", "App L")
    P.hasnt("A2_untested", T, "sensitivity of the readout concordance to the ridge range is untested", "tex")
    # ================= A3 cohort family-clustered CIs =================
    SF = R / "p4_encoding/space_fix_20260925/aggregate"
    fam = {}
    for co, fn in (("MEG-MASC", "concordance_spacefix_masc27_32.json"), ("Brennan", "concordance_spacefix_brennan_32.json")):
        dd = rj(SF / fn)
        fam[co] = family_clustered_ci(dd["per_session_rsa"], dd["per_session_enc"], n_boot=10000, seed=0)
    P.check("A3_signs", fam["MEG-MASC"][1] > 0 and fam["Brennan"][1] < 0 < fam["Brennan"][2],
            f"MEG-MASC {cis(*fam['MEG-MASC'][1:3])} excludes 0; Brennan {cis(*fam['Brennan'][1:3])} includes 0")
    # 2026-10-10 Brennan K2 (PREREG_paperA_brennan_preproc_20261009): the stored Brennan epochs carry no baseline;
    # with one the Brennan concordance is not determined, so "one cohort" rests on MEG-MASC alone. The
    # un-baselined Brennan family CI above stays a receipt check (A3_signs) but is no longer printed as current.
    k2 = rj(R / "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json")
    P.has("A3_53_masc", S53, f"This rests on MEG-MASC alone. With the same band, window, depth and controls (right temporal "
          f"sensors), its $27$ listeners give two determined readouts and a \\emph{{positive}} concordance whose family-clustered "
          f"interval, the inference the headline is held to, excludes zero, narrowly (${cis(*fam['MEG-MASC'][1:3])}$)", "5.3")
    P.check("A3_k2_branch", k2["b"]["branch"] == k2["c"]["branch"] == "not determined", "K2 b/c Brennan branch 'not determined'")
    P.has("A3_53_br", S53, f"With a per-epoch baseline the Brennan concordance is ${f(k2['b']['rho'], 3)}$ (family-clustered CI "
          f"${cis(*k2['b']['family_ci95_seed0'])}$; linear detrend ${f(k2['c']['rho'], 3)}$, ${cis(*k2['c']['family_ci95_seed0'])}$), not determined", "5.3")
    P.hasnt("A3_53_br_old", BODY, f"Brennan's crosses it (${cis(*fam['Brennan'][1:3])}$)", "body")
    P.has("A3_intro", INTRO, "it does not travel to MEG-MASC (positive, with a family-clustered interval excluding zero narrowly), "
          "and on Brennan EEG, once slow drift is removed, it is not determined", "introduction")
    P.has("A3_lim", LIM, "only MEG-MASC is positive under family-clustered inference, narrowly, so the cross-cohort contrast rests "
          "on it alone; Brennan's epochs had no baseline, with one its sign is not determined", "Limitations")
    P.has("A3_setup", BODY, "claims about them are held to the clustered one, on all three cohorts.", "3 Problem Setup")
    P.hasnt("A3_old_stored", T, "only the Fisher-$z$ interval is stored", "tex")
    P.hasnt("A3_old_justexcl", BODY, "(on Brennan the interval only just excludes zero)", "body")
    # ================= U* session target scale (registered diagnostic + re-score) =================
    S9 = R / "paperA_rebuttal_20261004"
    dg = rj(S9 / "session9_raw/diag.json")
    s9 = rj(S9 / "session9_summary_20261005.json")
    rk = {k: float(v) for k, v in dg["D1_r_k"].items()}
    others = [v for k, v in rk.items() if k not in ("9", "3")]
    P.check("U_diag", dg["plants"]["all_pass"] and s9["diag_branch"] == "U*" and rk["9"] > 2 and rk["3"] > 2
            and all(0.5 <= v <= 2 for v in others), f"plants pass; U*; r9 {rk['9']:.1f}, r3 {rk['3']:.1f}, others {min(others):.2f}-{max(others):.2f}")
    sc = s9["per_session_scaled"]
    P.check("U_branch", s9["branch"] == "A" and s9["seed_sweep_0_19"]["branches"] == {"A": 20} and sc["family_ci95"][1] < 0
            and not s9["G2_fail"] and not s9["missing"], "re-score branch A in 20/20 seeds; G2 replay clean")
    ratios = [v["enc_mean_scaled"] / v["enc_mean_unscaled"] for v in s9["per_model"].values()]
    mu0 = np.mean([v["enc_mean_unscaled"] for v in s9["per_model"].values()])
    mu1 = np.mean([v["enc_mean_scaled"] for v in s9["per_model"].values()])
    zero = sum(s9["folds_at_1e6_scaled"].values()); top0 = sum(s9["folds_at_1e6_unscaled"].values())
    P.check("U_folds", zero == 0 and top0 == top and min(ratios) > 1, f"folds at 1e6: {top0} -> {zero}; every score rises ({min(ratios):.2f}-{max(ratios):.2f}x)")
    rr9, rr3 = f"{rk['9']:.1f}", f"{rk['3']:.1f}"
    P.has("U_55", S55, f"(${rr9}\\times$ and ${rr3}\\times$ the median), which lowered every encoding score (${min(ratios):.1f}$--${max(ratios):.1f}\\times$)", "5.5")
    P.has("U_55_rho", S55, f"with each session scaled separately the concordance is $\\rho={f(sc['rho'], 3)}$ (family-clustered CI ${cis(*sc['family_ci95'])}$), so the reversal does not depend on it", "5.5")
    P.has("U_55_four", S55, "Four occurred in this pipeline", "5.5")
    P.has("U_N_rk", APP_N, f"(${rr9}\\times$ and ${rr3}\\times$ the median session; the other ten lie within ${min(others):.2f}$--${max(others):.2f}\\times$)", "App N")
    P.has("U_N_folds", APP_N, f"selected in ${top0}$ of ${tot}$ checkpoint-folds", "App N")
    P.has("U_N_ratio", APP_N, f"lowered, by ${min(ratios):.1f}$--${max(ratios):.1f}\\times$ (mean over the $32$ checkpoints ${mu1:.3f}$ with each session scaled separately against ${mu0:.3f}$ as published)", "App N")
    P.has("U_N_rho", APP_N, f"$\\rho = {f(sc['rho'], 3)}$, family-clustered CI ${cis(*sc['family_ci95'])}$ (branch A in $20$ of $20$ bootstrap seeds)", "App N")
    # per-session concordance under scaling, recomputed from the per-model re-score receipts
    H = rj(R / "p4_encoding/concordance32_CORRECTED.json")["per_session_rsa"]
    enc = {}
    for p in sorted((S9 / "session9_raw/rescore").glob("*.json")):
        j = rj(p)
        enc[j["model"]] = {k: v["score"] for k, v in j["per_session_scaled"].items()}
    ms = sorted(H)
    neg = sum(spearmanr([H[m][s] for m in ms], [enc[m][s] for m in ms])[0] < 0 for s in sorted(H[ms[0]], key=int)) if set(enc) == set(ms) else -1
    P.has("U_N_sessions", APP_N, f"${neg}$ of $12$ sessions still reverse on their own", "App N")
    s9m = s9["enc_session_means_scaled"]
    P.check("U_s9_zero", abs(s9m["9"]) < 0.005 and min(v for k, v in s9m.items() if k != "9") > 0.04,
            f"session 9 scaled encoding mean {s9m['9']:.4f} near zero, others > 0.04")
    P.has("U_N_s9", APP_N, "In session $9$ the encoding score stays near zero after rescaling while similarity is ordinary; we have not identified why.", "App N")
    P.has("U_L_ceiling", APP_L, "with each session scaled separately no fold selects 1e6", "App L")
    # ================= Qwen2.5 ladder under per-session scaling (arm C, registered) =================
    q = rj(S9 / "qwen_ladder_summary_20261005.json")
    cz, c0 = q["Cz_scaled"], q["C0_unscaled"]
    P.check("Q_branch", q["branch"] == "holds" and q["arm"] == "C" and not q["G2_fail"] and not q["G3_fail"]
            and q["Q1_rule_with_Cz_enc_and_published_rsa"].startswith("NOT SUPPORTED"), "arm C holds; G2/G3 clean; Q1 unchanged")
    ratio_q = np.array(cz["per_rung_mean"]) / np.array(c0["per_rung_mean"])
    rung = cz["per_rung_mean"]
    P.check("Q_shape", int(np.argmin(rung)) == 6 and rung[3] < rung[2] and rung[3] < rung[4], "72B lowest; 7B an interior dip")
    P.has("Q_L", APP_L, f"($\\rho = {f(cz['rho'], 3)}$, session-clustered CI ${cis(*cz['ci95_session_clustered'], 3)}$; $72$B $-$ $0.5$B $= {f(cz['top_minus_bottom_mean'], 4)}$, Wilcoxon $p = {cz['top_minus_bottom_wilcoxon_p']:.4f}$", "App L")
    P.has("Q_L_shape", APP_L, f"every rung rises ${ratio_q.min():.1f}$--${ratio_q.max():.1f}\\times$, $0.5$B--$32$B are nearly level (${min(rung[:6]):.3f}$--${max(rung[:6]):.3f}$)", "App L")
    P.has("Q_L_c0", APP_L, f"unscaled one on the same features (${f(c0['top_minus_bottom_mean'], 4)}$; paired change in $\\rho$ ${f(q['T3_delta_rho'], 3)}$, CI ${cis(*q['T3_delta_ci95'])}$)", "App L")
    P.has("Q_L_c0rho", APP_L, f"(unscaled $\\rho = {f(c0['rho'], 3)}$, the unspaced value", "App L")
    P.has("Q_L_scope", APP_L, "The re-extracted features of the $3$B--$72$B rungs were not retained, so the registered-feature value is not recomputed", "App L")
    P.has("Q_54", S54, f"$\\rho={f(cz['rho'], 2)}$, with $0.5$B--$32$B nearly level and the drop carried by $72$B", "5.4")
    # ================= C4 seed sweep (registered sentence, App J) =================
    c4 = rj(S9 / "paperA_rebuttal_recompute_20261004.json")["D_c4_seed_sweep_paper_function"]
    ub = c4["upper_by_seed_0_19"]
    P.check("C4_seed", len(ub) == 20 and c4["seed0_is_max"] and abs(ub[0] - c4["stored_family_ci95"][1]) < 1e-12, "seed 0 = stored upper bound and the maximum")
    P.has("C4_J", APP_J, f"ranges from ${f(min(ub), 3)}$ to ${f(max(ub), 3)}$ and is below zero for ${sum(u < 0 for u in ub)}$ of $20$ seeds. We keep the registered verdict", "App J")
    # ================= A5 Brennan lineage =================
    EB = R / "exp_boundary_conditional"
    b47 = rj(EB / "brennan47_derived_20261003.json")
    bl1 = b47["arms"]["BERT/L1"]
    rat = [v["ratio_mean_L1_over_49"] for v in b47["archive_vs_rebuild"].values()]
    P.check("B_rule", bl1["mean"] > 0 and bl1["wilcoxon_p"] < 0.05 and b47["holm_over_three_models_L1"]["GPT-2-medium"] < 0.05,
            "same scorecard rule on the rebuild reads supports; GPT-2-medium significant after Holm")
    p_mant, p_exp = f"{bl1['wilcoxon_p']:.1e}".split("e")
    P.has("B_L_36", BRL, f"BERT ${bl1['n_positive']}$ of ${bl1['n']}$ listeners, Wilcoxon $p = {p_mant}\\times10^{{{int(p_exp)}}}$", "Brennan lineage")
    P.has("B_L_ratio", BRL, f"model means ${round(min(rat))}$--${round(max(rat))}$ times the archive's", "Brennan lineage")
    arc = {r["subject"] + "|" + r["llm"]: float(r["delta"]) for r in csv.DictReader((EB / "boundary_brennan_persubject.tsv").open(), delimiter="\t")}
    reb = {}
    for r in csv.DictReader((EB / "brennan47_offset_reextract.tsv").open(), delimiter="\t"):
        reb[(r["subject"], r["model"], r["arm"])] = float(r["delta"])
    subs = sorted({k[0] for k in reb})
    sp = spearmanr([arc[s + "|BERT"] for s in subs], [reb[(s, "BERT", "L0")] for s in subs])[0]
    P.has("B_L_spear", BRL, f"over the ${len(subs)}$ shared listeners ${f(sp, 2)}$)", "Brennan lineage")
    nine = {"S02", "S24", "S26", "S27", "S29", "S30", "S34", "S35", "S49"}
    a9 = 1e3 * np.mean([arc[s + "|BERT"] for s in subs if s in nine]); a38 = 1e3 * np.mean([arc[s + "|BERT"] for s in subs if s not in nine])
    r9_ = 1e3 * np.mean([reb[(s, "BERT", "L0")] for s in subs if s in nine]); r38 = 1e3 * np.mean([reb[(s, "BERT", "L0")] for s in subs if s not in nine])
    P.has("B_L_nine", BRL, f"archived BERT contrast of ${f(a9, 1)}$ against ${f(a38, 1)}\\times10^{{-3}}$ for the other ${len(subs) - len(nine)}$ shared listeners (rebuild ${f(r9_, 1)}$ against ${f(r38, 1)}$)", "Brennan lineage")
    bld = ROOT / "codes/build_brennan_epochs.py"; pre_b = ROOT / "codes/preprocess_brennan.py"
    if bld.exists() and pre_b.exists():
        P.check("B_code", "if len(starts)<12" in bld.read_text() and 'f"Stimulus/{stim_id}"' in pre_b.read_text(),
                "Bug-B excluded by code: v1 drops <12 markers; archive maps segments by label")
    al_r = [rj(p) for p in sorted((S9 / "brennan_align_raw").glob("*.json"))]
    pb = R / "paperB_phase1_20261005/BRANCHES_read_20261005.json"
    d2_fail = (not rj(pb)["a"]["D2_instrument"]["PC_S01_all_pass"]) if pb.exists() else None
    P.check("B_align", len(al_r) == 3 and all(x["branch"] == "E" for x in al_r) and any(x.get("highpass_hz") == 0.5 for x in al_r)
            and d2_fail in (True, None), "alignment check: branch E (registered and 0.5 Hz stage 2); paperB D2 S01 control failed")
    P.has("B_L_align", BRL, "verified on one listener (S01) only", "Brennan lineage")
    P.has("B_L_bugB", BRL, "drops every listener with fewer than $12$ segment markers, and the archive's builder maps segments by label", "Brennan lineage")
    P.has("B_L_pending", BRL, "The Feghhi/Hadidi and He Brennan cells are pending re-derivation on the rebuild", "Brennan lineage")
    fp = ROOT / "paper4_temporalalign/figures_v2/revision_r1_20261007/fig09_scorecard_r1.provenance.json"
    if fp.exists():
        cells = {(c["paper"], c["cohort"]) for c in rj(fp)["archived_cells"]}
        P.check("B_fig9", cells == {("Zou et al.", "Brennan"), ("Feghhi/Hadidi et al.", "Brennan"), ("He et al.", "Brennan")},
                f"Fig. 9 marks exactly the three archive cells {sorted(cells)}")
    P.has("B_fig9_tex", APPX, "fig09_scorecard_r1.pdf", "App L (revised figure)")
    P.has("B_fig9_cap", APPX, f"the same rule still reads ``supports'' there (BERT ${bl1['n_positive']}/{bl1['n']}$, epochs without a baseline) and its provenance is lost", "Fig. 9 caption")
    bc = list(csv.DictReader((EB / "boundary_conditional_v2.tsv").open(), delimiter="\t"))
    zc = {(r["cohort"], r["llm"], r["condition"]): float(r["z"]) for r in bc}
    cells_e = {(c, m) for c, m, _ in zc}
    k = sum(zc[(c, m, "medial")] > zc[(c, m, "onset")] for c, m in cells_e)
    P.has("B_zou_L", APP_L, f"the stored cell file gives medial above onset in ${k}$ of ${len(cells_e)}$", "App L Zou sentence")
    P.has("B_zou_13", APPX, f"(historical count; the stored cell file gives {k}/{len(cells_e)};", "Table 13 Zou row")
    for cid, s in (("B_old_four", "replicates on all four MEG/EEG cohorts"), ("B_old_4", "holds on all 4 MEG/EEG cohorts"),
                   ("B_old_list", "LibriBrain, MEG-MASC, MEG-SCANS, and Brennan"), ("B_old_gold5", "layer $\\times$ peaktime, 5 cohorts &")):
        P.hasnt(cid, T, s, "tex")
    # ================= A4 Schrimpf, A6 abstract, AI provenance =================
    for cid, s in (("S_old_neither", "can see neither"), ("S_old_none", "none reports whether"), ("S_old_single", "single-metric (encoding)")):
        P.hasnt(cid, T, s, "tex")
    P.has("S_abs", ABS, "so they usually do not report this, or whether a ranking survives a change of text", "abstract")
    P.has("S_B1", APP_B1, "($r = 0.94$ between the two Pereira2018 experiments, $r = 0.50$ between Pereira2018 and Fedorenko2016)", "App B.1")
    P.check("S_verify_comment", "% VERIFY-SCHRIMPF" in tex_raw, "VERIFY-SCHRIMPF comment left for the author (SI not opened)")
    m = re.search(r"Listener halves & \$([+-][0-9.]+)\$ & \$([+-][0-9.]+)\$ & \$([+-][0-9.]+)\$ & \$([+-][0-9.]+)\$", T)
    if m:
        cells = [float(x) for x in m.groups()]
        P.has("A6_abs", ABS, f"median Spearman $\\rho$ of ${f(min(cells), 3)}$ to ${f(max(cells), 3)}$", "abstract (Table 1 row min/max)")
    else:
        P.check("A6_abs", False, "Table 1 Listener-halves row not found")
    # AI_prov removed 2026-10-08 (author decision, reaffirmed from 09-25/09-28): the % AI-PROVENANCE ...
    # % END AI-PROVENANCE block is deleted, not kept -- the AI-use statement reverts to the single
    # paragraph submitted (iclr2027-submitted), so there is no provenance sentence left to pin.
    P.hasnt("AI_noauthors", BODY, "the authors checked", "statements")
    return P


# ------------------------------------------------------------------ selftest
TEX_MUTANTS = {
    "A1 abstract rho -0.016->-0.061": ("($\\rho = -0.016$, CI $[-0.40, +0.34]$)", "($\\rho = -0.061$, CI $[-0.40, +0.34]$)"),
    "A1 abstract CI sign flip": ("($\\rho = -0.016$, CI $[-0.40, +0.34]$)", "($\\rho = -0.016$, CI $[-0.40, -0.34]$)"),
    "A1 5.3 verbatim sentence dropped": ("and under a common reading rule it is not determined (Appendix~\\ref{app:readrule})", "(Appendix~\\ref{app:readrule})"),
    "A1 App D delta 0.557->0.575": ("the paired change is $+0.557$", "the paired change is $+0.575$"),
    "A1 App D full_last CI": ("$-0.663$ (family-clustered CI $[-0.82, -0.36]$)", "$-0.663$ (family-clustered CI $[-0.82, +0.36]$)"),
    "A1 table row swap": ("prefix to the target (primary) & last sub-token & $-0.016$", "prefix to the target (primary) & last sub-token & $-0.032$"),
    "A1 timing softened": ("and was not integrated into the submitted version", "and was integrated later"),
    "A1 counts 9->8": ("and $9$ fall in similarity", "and $8$ fall in similarity"),
    "A2 352->325": ("selected in $352$ of the $384$ checkpoint-folds", "selected in $325$ of the $384$ checkpoint-folds"),
    "A2 exp rho": ("leaves the readout concordance at $\\rho = -0.657$", "leaves the readout concordance at $\\rho = -0.567$"),
    "A3 MASC CI": ("narrowly ($[+0.03, +0.71]$)", "narrowly ($[+0.13, +0.71]$)"),
    "A3 Brennan K2 rho": ("the Brennan concordance is $-0.248$", "the Brennan concordance is $-0.428$"),
    "A3 Brennan old CI back": ("adds no second case.", "adds no second case; Brennan's crosses it ($[-0.20, +0.69]$)."),
    "B fig9 baseline scope dropped": ("(BERT $36/47$, epochs without a baseline)", "(BERT $36/47$)"),
    "A3 old sentence back": ("held to the clustered one, on all three cohorts.", "held to the clustered one; for MEG-MASC and Brennan only the Fisher-$z$ interval is stored."),
    "U 76.6->67.6 (5.5)": ("($76.6\\times$ and $5.5\\times$ the median), which", "($67.6\\times$ and $5.5\\times$ the median), which"),
    "U App N rho": ("$\\rho = -0.543$, family-clustered CI $[-0.71, -0.22]$ (branch A", "$\\rho = -0.534$, family-clustered CI $[-0.71, -0.22]$ (branch A"),
    "U 11->12 sessions": ("$11$ of $12$ sessions still reverse", "$12$ of $12$ sessions still reverse"),
    "U ratio 3.1->2.1": ("($1.7$--$3.1\\times$) and pinned", "($1.7$--$2.1\\times$) and pinned"),
    "Q rho -0.786": ("($\\rho = -0.786$, session-clustered CI", "($\\rho = -0.768$, session-clustered CI"),
    "Q 72B diff": ("$0.5$B $= -0.0145$", "$0.5$B $= -0.0154$"),
    "Q 5.4 rho": ("$\\rho=-0.79$, with $0.5$B--$32$B nearly level", "$\\rho=-0.97$, with $0.5$B--$32$B nearly level"),
    "C4 17->18": ("is below zero for $17$ of $20$ seeds", "is below zero for $18$ of $20$ seeds"),
    "B 36->38": ("BERT $36$ of $47$ listeners", "BERT $38$ of $47$ listeners"),
    "B nine split sign": ("contrast of $-1.5$ against", "contrast of $+1.5$ against"),
    "B spearman": ("shared listeners $+0.18$)", "shared listeners $+0.81$)"),
    "B Zou 5->7 of 9": ("medial above onset in $5$ of $9$", "medial above onset in $7$ of $9$"),
    "B Toneva Brennan back": ("replicates on the three MEG cohorts in its weaker form", "replicates on all four MEG/EEG cohorts in its weaker form"),
    "S neither back": ("so they usually do not report this, or whether", "so they can see neither this nor whether"),
    "S r 0.94->0.49": ("$r = 0.94$ between the two", "$r = 0.49$ between the two"),
    "A6 abstract 0.764->0.76": ("of $+0.764$ to $+0.961$", "of $+0.76$ to $+0.96$"),
}
NULL_EDIT = ("Two Brennan substrates exist.", "Two Brennan substrates exist ")      # punctuation only, touches no pin


def receipt_mutants():
    return [("readrule rho", "readrule_20260926/readrule_summary.json", ["variants", "prefix_last", "rho"], -0.05),
            ("alpha32 top count", "alpha32_20260926/alpha32_summary.json", ["orig_grid_top_counts", "checkpoint_folds_at_1e6"], 351),
            ("session9 r9", "paperA_rebuttal_20261004/session9_raw/diag.json", ["D1_r_k", "9"], 67.6),
            ("qwen Cz rho", "paperA_rebuttal_20261004/qwen_ladder_summary_20261005.json", ["Cz_scaled", "rho"], -0.70),
            ("c4 seed upper", "paperA_rebuttal_20261004/paperA_rebuttal_recompute_20261004.json",
             ["D_c4_seed_sweep_paper_function", "upper_by_seed_0_19", 11], -0.03),
            ("brennan47 n_positive", "exp_boundary_conditional/brennan47_derived_20261003.json", ["arms", "BERT/L1", "n_positive"], 38)]


def mutate(raw: str, old: str, new: str) -> str | None:
    """Apply a mutant to the RAW tex (comments kept, so the comment-based pins still see a real file);
    the anchor may wrap physical lines, so whitespace in `old` matches any whitespace run."""
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split(" ")))
    m = pat.search(raw)
    return None if m is None else raw[: m.start()] + new + raw[m.end():]


def selftest(tex_path: Path, R: Path) -> int:
    raw = tex_path.read_text()
    base = run(raw, R)
    if base.fail:
        print(f"\nSELFTEST ABORTED: baseline fails {base.fail}")
        return 1
    caught = missed = 0
    print("\n--- tex mutants ---")
    for name, (old, new) in TEX_MUTANTS.items():
        mt = mutate(raw, old, new)
        if mt is None:
            print(f"  ERROR   {name}: anchor not in tex"); missed += 1; continue
        P = run(mt, R)
        ok = bool(P.fail); caught += ok; missed += (not ok)
        print(f"  {'caught ' if ok else 'MISSED '} {name}" + (f" -> {P.fail}" if ok else ""))
    print("\n--- null edit ---")
    mt = mutate(raw, *NULL_EDIT); assert mt is not None
    Pn = run(mt, R); null_ok = not Pn.fail
    print("  ok (no false positive)" if null_ok else f"  FALSE POSITIVE {Pn.fail}")
    print("\n--- receipt mutants (temp copies) ---")
    rc = rm = 0
    for name, rel, keys, val in receipt_mutants():
        tmp = Path(tempfile.mkdtemp(prefix="paperA_r1_selftest_"))
        try:
            dst = tmp / "results"
            # copy only what run() reads, to keep the selftest fast
            for sub in ("readrule_20260926", "alpha32_20260926", "paperA_rebuttal_20261004", "exp_boundary_conditional",
                        "paperB_phase1_20261005"):
                if (R / sub).exists():
                    shutil.copytree(R / sub, dst / sub)
            (dst / "p4_encoding/space_fix_20260925/aggregate").mkdir(parents=True)
            for fn in ("concordance_spacefix_masc27_32.json", "concordance_spacefix_brennan_32.json"):
                shutil.copy2(R / "p4_encoding/space_fix_20260925/aggregate" / fn, dst / "p4_encoding/space_fix_20260925/aggregate" / fn)
            shutil.copy2(R / "p4_encoding/concordance32_CORRECTED.json", dst / "p4_encoding/concordance32_CORRECTED.json")
            t = dst / rel; j = json.loads(t.read_text()); cur = j
            for k in keys[:-1]:
                cur = cur[k]
            cur[keys[-1]] = val
            t.write_text(json.dumps(j))
            P = run(raw, dst)
            ok = bool(P.fail); rc += ok; rm += (not ok)
            print(f"  {'caught ' if ok else 'MISSED '} {name}" + (f" -> {P.fail}" if ok else ""))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    restored = not run(raw, R).fail
    print(f"\ntex mutants {len(TEX_MUTANTS)}: caught {caught}, missed {missed}; receipt mutants {rc + rm}: caught {rc}, missed {rm}; "
          f"null edit {'ok' if null_ok else 'FALSE POSITIVE'}; real results unaffected: {'ok' if restored else 'NO'}")
    ok = missed == 0 and rm == 0 and null_ok and restored
    print(f"SELFTEST={'ok' if ok else 'BLIND'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default=str(DEFAULT_TEX))
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(Path(a.tex), Path(a.results))
    P = run(Path(a.tex).read_text(), Path(a.results))
    print(f"\n{'FAILED' if P.fail else 'PASS'}: paperA R1 pins {P.n - len(P.fail)}/{P.n} hold")
    if P.fail:
        print("failed:", P.fail)
    return 1 if P.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
