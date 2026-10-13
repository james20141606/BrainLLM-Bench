#!/usr/bin/env python
"""Context-anchored, receipt-derived pins for paperA (#60477) main-text and abstract numbers.

Background (2026-10-05, BootLoops audit / MUTATION_SWEEP_20261005, paperA camera-ready TODO):
`<coordination-notes>/tools/mutation_sweep.py` injected 60 single-token numeric mutations into
`paperA_benchmark.tex` and ran it past the five existing paperA checkers. 39 of them were missed
(`<coordination-notes>/tools/mutsweep_runs/paperA_20261005/report.md` section "Missed mutations") because
the existing checkers only read Table 1's cells (`check_table1_receipts.py`) or the two fixed
C4/space-fix receipts (`check_c4_receipt_20260924.py`, `check_followup_receipts_20260923.py`) or a
narrow provenance note (`check_extract_provenance_20260925.py`) or a formatting rule
(`check_space_fix_20260926.py`) -- every OTHER number in the main text and abstract had no checker
reading it at all, so a one-digit change, a sign flip or a sentence-neighbour swap at any of those
locations would silently survive. This file adds one pin per missed mutation (37 pins covering the
39 mutations; two line-groups bundle two mutations each because both corrupt the same anchor
string -- L67's fig-caption panel markers and L198's rank-agreement formula) plus the two items
named directly in the BootLoops audit that the mutation sweep could not have found because they
sit in the appendix, outside `--scope main`:

  (a) `bench/reproduce.py`'s depth-curve claims (`c_depth_sessions_negative`, `c_depth_session_p`)
      read `concordance_depth_curve.json` (the PRE-FIX/retracted file) instead of the
      `_CORRECTED` file, because the corrected file has no `independent_per_session` key at all.
      This is silent (`n_negative=11` happens to match in both files) and is NOT fixed here --
      fixing `bench/reproduce.py` was out of scope for this task -- but Group G/H below pin the
      tex's depth-curve prose directly against `concordance_depth_curve_CORRECTED.json` and an
      independent, fully deterministic recomputation from `concordance32_CORRECTED.json`'s own
      per-session matrices, bypassing both the pre-fix file and reproduce.py entirely.
  (b) the abstract's rounded "+0.76 to +0.96" listener-halves range: Group I checks it against
      Table 1's "Listener halves" row rather than re-deriving the number from scratch (that row's
      own 200-iteration permutation resampling is already independently verified by
      `check_table1_receipts.py`; re-implementing it here would just be a second, RNG-dependent
      copy of the same computation). This is documented as a cross-location consistency check, not
      an independent receipt derivation.

Receipt-derived vs. literal-anchor-only pins: most pins both (i) require a long, sentence-length
literal substring of the tex containing the target number -- any single-digit change, sign flip or
swap with a neighbouring sentence's number breaks the substring match -- and (ii) independently
recompute the same number from a results/*.json receipt or from source code and compare. A few
numbers have NO backing receipt anywhere in the repo and are necessarily literal-anchor-only; this
is noted at each one:
  - L394 "ADR 0005": `find . -iname "*ADR*0005*"` finds no such document anywhere in this repo.
    There is no `ADR 0006` document either, only `codes/adr0006_registered_analysis.py`. Pinned as
    a literal anchor only.
  - L184/187/198/215/218/224/233 (equation and Proposition exponents/constants) and L166/169/420/
    492/499 (window size, relative depth, word count, model-size ladders): these are design
    constants, not computed statistics -- there is no receipt to recompute them against. Pinned as
    literal anchors only.
  - L519 "$97/97$ green throughout": this is a *historical* count (the claim-reproduction harness
    read 97/97 claims green under the defective epsilon guard); `python -m bench.reproduce` today
    reports a different total (198/198, after the fix and after claims were added). The two numbers
    are not supposed to agree -- that mismatch is the point of the sentence ("harness ... cannot see
    that they came from defective code"). Pinned as a literal anchor only, with no live-count cross
    check attempted.

Mutation-testing this checker (required before trusting it, per the epistemic rules this portfolio
runs on): run `--selftest`. It applies all 39 missed-mutation substitutions (one at a time, to a
tempfile copy of the tex text) plus 4 receipt-side corruptions (tempdir copy of results/) plus one
null-edit control, and reports which were CAUGHT, which were MISSED, and whether the null edit
produced a false positive. `SELFTEST=ok` means all mutants were caught and the null edit was not.

CLI: `check_paperA_main_numbers.py [--tex PATH] [--results DIR] [--selftest]`
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Repo-relative paths (2026-10-05): this file is new, so it never had the hardcoded-path bug that
# check_page_budget.py / check_tex_consistency.py had, but it follows the same resolution pattern
# (check_table1_receipts.py, check_paperB_receipts.py) so it also works on the release copy.
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper4_temporalalign/results"
if not RESULTS.is_dir():
    RESULTS = ROOT / "results"          # release layout
DEFAULT_TEX = ROOT / "paper4_temporalalign/tex/paperA_benchmark.tex"
if not DEFAULT_TEX.is_dir() and not DEFAULT_TEX.exists():
    alt = ROOT / "tex/paperA_benchmark.tex"
    if alt.exists():
        DEFAULT_TEX = alt


# ---------------------------------------------------------------- shared helpers (check_paperB_receipts.py pattern)

def norm(s: str) -> str:
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)     # strip comments, keep escaped \%
    return re.sub(r"\s+", " ", s)


def body_of(tex: str) -> str:
    a = tex.find(r"\end{abstract}")
    b = tex.find(r"\appendix")
    return tex[a: b if b > 0 else len(tex)]


def abstract_of(tex: str) -> str:
    return tex[tex.find(r"\begin{abstract}"): tex.find(r"\end{abstract}")]


class Pins:
    def __init__(self, results: Path):
        self.R = results
        self.fail: list[str] = []
        self.n = 0
        self.notes: list[str] = []

    def check(self, cid: str, ok: bool, msg: str):
        self.n += 1
        print(("  ok    " if ok else "  FAIL  ") + f"{cid:34s} {msg}")
        if not ok:
            self.fail.append(cid)

    def has(self, cid: str, hay: str, needle: str, where: str):
        self.check(cid, needle in hay, f"'{needle[:70]}' in {where}")


def _load(results: Path, rel: str):
    p = results / rel
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"__error__": f"{rel}: {e}"}


def _rget(d: dict, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------------------------------------------- the pins

def run(tex_text: str, results: Path) -> Pins:
    P = Pins(results)
    T = norm(tex_text)
    ABS = norm(abstract_of(tex_text))

    # ================= Literal context-anchor pins, one per missed mutation (report.md "Missed") =================
    # Each anchor is a long, sentence-scale literal substring of the (whitespace-normalized) tex
    # containing the mutated number. A one-digit change, sign flip, or swap with a nearby sentence's
    # number breaks the substring match. Grouped and commented by source line in paperA_benchmark.tex.

    # R1 2026-10-10 (Brennan K2): the abstract's Brennan clause now says "not determined" once drift is removed
    P.has("L38_eeg47", ABS,
          "it does not reproduce on $27$-listener MEG and is not determined on $47$-subject English EEG once slow "
          "drift is removed, and feature geometry is not excluded",
          "abstract")
    P.has("L57_families12", T,
          "Score $32$ checkpoints from $12$ families against one",
          "body (intro)")
    P.has("L59_rsa_reliability098", T,
          "returns a leaderboard with ranking reliability $0.98$; ridge encoding",
          "body (intro)")
    P.has("L61_rho_signflip", T,
          r"opposite directions ($\rho=-0.573$): similarity puts the",
          "body (intro)")
    # 2026-10-10 Figure 1 redraw (FIG1_GUIDE 4A): the schematic caption pins (L67 values4/panel3, sweep, item2)
    # retired with the schematic; the new caption is pinned here, the Brennan omission against the K2 verdict.
    P.has("F1_cap_claim", T,
          r"\textbf{A model ranking survives new listeners, but not always a new story or a new readout.}",
          "fig:overview caption (claim sentence)")
    P.has("F1_cap_ci95", T,
          r"Diamonds: readout concordance, family-clustered 95\% intervals; grey circles: reliability ceiling.",
          "fig:overview caption (interval type)")
    P.has("F1_cap_c4", T,
          "With each model's static layer-0 embedding (C4) also partialled out, the reversal is not determined, "
          "nor under a common reading rule.",
          "fig:overview caption (C4 and reading-rule disclosure)")
    P.has("F1_cap_brennan", T,
          "Brennan EEG is omitted: its epochs were not baseline-corrected, and after drift removal its "
          "concordance is not determined.",
          "fig:overview caption (Brennan omission)")
    _k2 = results / "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json"
    _k2d = json.loads(_k2.read_text()) if _k2.is_file() else {}
    P.check("F1_k2_branch", all(_k2d.get(a, {}).get("branch") == "not determined" for a in ("b", "c")),
            "brennan_k2_summary_20261009.json: baseline (b) and detrend (c) branches both 'not determined' "
            "(backs the fig:overview Brennan omission sentence)")
    # R1 2026-10-07: the sentence now continues with the registered reading-rule result (anchor extended, not dropped)
    P.has("L86_base48", T,
          "survives family-clustered inference and the $48$ base checkpoints of the registered test, but not a common reading rule",
          "body (contributions)")
    P.has("L99_years", T,
          r"readout sweep ($2019$--$2026$) under one protocol and both readouts;",
          "body (contributions)")
    P.has("L99_sweep32to57", T,
          r"a $32$-to-$57$-checkpoint readout sweep",
          "body (contributions)")
    P.has("L166_window300_400ms", T,
          r"broadband, $300$--$400$~ms), flattened over sensors and time",
          "body (setup)")
    P.has("L169_depth05", T,
          "word $i$ at relative depth $0.5$, one layer per model fixed before scoring.",
          "body (setup)")
    P.has("L184_ridge_mu", T,
          r"\tilde X = (X-\mathbf 1\mu^\top)\Sigma^{-1}",
          "body (encoding readout eq. preamble)")
    P.has("L187_frobenius_exp", T,
          r"\hat W_\alpha = \arg\min_{W}\ \|\tilde Y_{-u}-\tilde X_{-u}W\|_F^2+\alpha\|W\|_F^2",
          "eq:ridge")
    P.has("L198_rank_formula", T,
          r"\rho(\pi,\pi')=1-6\sum_m(\pi_m-\pi'_m)^2/\bigl(M(M^2-1)\bigr)",
          "body (rankings and agreement) -- covers both the '6' coefficient and the '2' exponent")
    P.has("L215_prop1i_c_gt_0", T,
          "Replacing $X$ by $cXQ$ with $c>0$ and $Q$",
          "Proposition 1(i)")
    P.has("L218_prop1ii_1b", T,
          r"$XA+\mathbf 1b^\top$ with $A$ positive diagonal leaves $\tilde X$",
          "Proposition 1(ii)")
    P.has("L224_participation_ratio_exp", T,
          r"\mathrm{PR}=(\sum_k\lambda_k)^2/\sum_k\lambda_k^2$ its participation ratio",
          "Proposition 1(iv) / discussion")
    P.has("L233_sk2_alpha", T,
          r"a weight that saturates at one once $s_k^2\gg\alpha$, whatever its variance share",
          "body (discussion after Proposition 1)")
    P.has("L249_libribrain36", T,
          r"we use its $36$ Sherlock~1 MEG recordings ($12$ sessions $\times$ three story parts)",
          "body (cohorts)")
    P.has("L251_megscans2324", T,
          "is German MEG from $23$ of its $24$ listeners",
          "body (cohorts)")
    P.has("L254_brennan47rebuilt", T,
          "($47$ in rebuilt analyses: two subjects' public files are corrupt; the rebuilt epochs carry no filter "
          "or baseline, Appendix~\\ref{app:brennan-preproc});",
          "body (cohorts)")
    P.has("L262_families12b", T,
          "The readout comparison ranks a $32$-checkpoint sweep over $12$ families and three objectives",
          "body (cohorts)")
    P.has("L308_ci95", T,
          r"give a difference with its 95\% interval.",
          "Table 1 caption")
    P.has("L312_fam12boot", T,
          r"[fam] family-clustered bootstrap over $12$ families.",
          "Table 1 caption")
    P.has("L313_registered3248", T,
          "Registered test that held ($32$ checkpoints: concordance below the",
          "Table 1 caption (also carries the '48 base' clause)")
    P.has("L322_ds003020_25x9", T,
          r"ds003020 fMRI ($25$ $\times$ $9$)",
          "Table 1 header")
    P.has("L381_megmasc27", T,
          "MEG-MASC presents four narratives and all $27$ listeners heard all four, a crossed design.",
          "body (crossed-story section)")
    P.has("L394_adr0005", T,
          "This test was registered before scoring (ADR 0005) and was",
          "body (crossed-story section) -- NO backing document exists in-repo (confirmed by "
          "`find . -iname '*ADR*0005*'` returning nothing); literal-anchor-only, see module docstring")
    # R1 2026-10-10: the main-text Brennan-segment sentence ("similarity is determined on all and encoding on
    # one") rested on un-baselined epochs and is removed (K2 #4); App M carries both readings (Group L below).
    P.check("L420_words8000", r"on eight Brennan segments of about $8\,000$ words, similarity is determined on all and" not in T
            and "encoding needs far more data than similarity" not in T,
            "retired: main-text Brennan 8-segment sentence (un-baselined) is gone")
    P.has("L443_bundle_libribrain_numbers", T,
          "\\emph{opposite} orders ($\\rho=-0.573$). Both rankings are individually reliable ($0.98$ and $0.96$ "
          "against an attenuation ceiling of $0.97$), so this is a disagreement between determined rankings.",
          "body (readout reversal section) -- bundles rho, both reliabilities and the ceiling")
    P.has("L445_units12of32", T,
          "The family-clustered $95\\%$ CI, over $12$ units rather than $32$, is wider ($[-0.79, -0.26]$) and still excludes zero",  # R1: CI printed in place (abstract gate --strict)
          "body (readout reversal section)")
    P.has("L480_brennan60channels", T,
          r"Brennan EEG ($47$ listeners, all $60$ channels) adds no second case.",
          "body (reversal belongs to one cohort)")
    P.has("L481_brennan47listeners", T,
          r"its $27$ listeners give two determined readouts and a \emph{positive} concordance whose family-clustered interval",
          "body (reversal belongs to one cohort)")
    P.has("L492_qwen_sizes", T,
          r"$2.5$ ($0.5$B--$72$B), $3$ and $3.5$, in no generation does size order",
          "body (additional findings, scaling)")
    P.has("L499_pythia70m28b", T,
          r"The six-point Pythia ladder ($70$M--$2.8$B), scored",
          "body (additional findings, scaling)")
    P.has("L516_epsilon_1e12", T,
          r"The least visible of the first three, a $10^{-12}$ guard in a correlation",  # R1: four failure modes now
          "body (fragility / limitations)")
    P.has("L519_green9797", T,
          r"Our claim-reproduction harness read $97/97$ green throughout, because it checks stored",
          "body (fragility / limitations) -- historical count, see module docstring")

    # ================= Group L: Brennan preprocessing K0/K1/K2 (R1 2026-10-10) =================
    # PREREG_paperA_brennan_preproc_20261009.md: the stored Brennan epochs carry no filter or baseline; every
    # Brennan number below is recomputed from brennan_{preproc,k2}_summary_20261009.json (baseline b primary,
    # detrend c beside it) and the un-baselined values appear only as labelled history.
    def sg(v, nd):
        s = f"{v:+.{nd}f}"
        return "+" + s[1:] if s == f"-{0:.{nd}f}" else s
    def iv(lo, hi, nd=2):
        return f"$[{sg(lo, nd)}, {sg(hi, nd)}]$"
    K2 = _load(results, "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json")
    K0 = _load(results, "paperA_rebuttal_20261004/brennan_preproc_summary_20261009.json")
    NSb = _load(results, "p4_encoding/noise_sweep/brennan_p4_v2_dose0_seed0.json")
    NSl = _load(results, "p4_encoding/noise_sweep/libribrain_full_dose0_seed0.json")
    NSm = _load(results, "p4_encoding/noise_sweep/meg_masc_merged27_dose0_seed0.json")
    RIc = _load(results, "p4_encoding/rank_identifiability_CORRECTED.json")
    DLB = _load(results, "p4_encoding/concordance_decomposition_libribrain_CORRECTED.json")
    DBR = _load(results, "p4_encoding/concordance_decomposition_brennan.json")
    errs = [d["__error__"] for d in (K2, K0, NSb, NSl, NSm, RIc, DLB, DBR) if "__error__" in d]
    P.check("GL_receipts_load", not errs, f"Brennan K0/K1/K2 and comparison receipts load {errs if errs else ''}")
    if not errs:
        b, cc = K2["b"], K2["c"]
        P.check("GL_k2_gates", b["gate_pass"] and cc["gate_pass"] and b["n_models"] == cc["n_models"] == 32
                and not b["missing"] and not cc["missing"] and K0["K1_branch"] == "affected"
                and K0["K0_branch"] == "drift mechanism supported" and K0["gate_pattern_a_identical_all"]
                and K0["n_listeners"] == 47,
                "K2 gates pass (32/32, b and c), K1 'affected', K0 'drift mechanism supported', raw gate, 47 listeners")
        fb, Fb, fc, Fc = b["fisher_ci95"], b["family_ci95_seed0"], cc["fisher_ci95"], cc["family_ci95_seed0"]
        P.check("GL_branch_rule", all(F[0] < 0 < F[1] for F in (Fb, Fc)) and b["branch"] == cc["branch"] == "not determined",
                "registered K2 reading: both family CIs include 0 -> 'not determined'")
        # Section 5.3
        P.has("GL_53_masc_alone", T, r"\textbf{The reversal belongs to one cohort.} This rests on MEG-MASC alone.", "5.3")
        P.has("GL_53_protocol", T, "Its epochs, unlike LibriBrain's and MEG-MASC's, are neither filtered nor "
              "baseline-corrected (raw FieldTrip slices)", "5.3 (identical-protocol description error fixed)")
        P.has("GL_53_drift", T, "track the distance in story time between words (median Spearman "
              f"${sg(K0['K0']['a']['spearman_Db_dt_median'], 3)}$)", "5.3")
        P.has("GL_53_k2", T, f"With a per-epoch baseline the Brennan concordance is ${sg(b['rho'], 3)}$ (family-clustered CI "
              f"{iv(*Fb)}; linear detrend ${sg(cc['rho'], 3)}$, {iv(*Fc)}), not determined", "5.3")
        gb, gc = b["decomposition"], cc["decomposition"]
        P.check("GL_gap_sign", gb["class_gap_causal_minus_other"] > 0 and DLB["class_gap_causal_minus_other"] > 0
                and DBR["class_gap_causal_minus_other"] < 0, "K2 b class gap has LibriBrain's sign; the stored one did not")
        P.has("GL_53_gap", T, f"class gap (${sg(gb['class_gap_causal_minus_other'], 1)}$ ranks, family-clustered "
              f"$p={gb['gap_p_family_clustered']:.3f}$) has LibriBrain's sign", "5.3")
        # Section 5.1 and intro / discussion / limitations
        rb = [v["ratio"] for v in b["rank_identifiability"].values()]
        encfam = b["rank_identifiability"]["concordance32_brennan_k2_b|encoding [family-level]"]["verdict"]
        P.check("GL_51_encfam", encfam.startswith("NOT IDENTIFIABLE"), f"K2 b family-level encoding verdict: {encfam[:24]}")
        P.has("GL_51", T, f"the spread is about the paired-difference error (${min(rb):.2f}$--${max(rb):.2f}\\times$) and the "
              "family-level encoding ranking is not identifiable", "5.1")
        P.has("GL_51_lb_only", T, "rankings survive participant resampling in all four LibriBrain readout $\\times$ "
              "aggregation combinations", "5.1")
        P.has("GL_intro_listen", T, "barely moves a leaderboard in all four LibriBrain readout and aggregation combinations "
              "and in both crossed cohorts.", "intro")
        P.has("GL_intro_travel", T, "it does not travel to MEG-MASC (positive, with a family-clustered interval excluding zero "
              "narrowly), and on Brennan EEG, once slow drift is removed, it is not determined;", "intro")
        P.has("GL_disc", T, "but not on MEG-MASC, and on Brennan EEG, once slow drift is removed, not determined;", "Discussion")
        P.has("GL_lim", T, "only MEG-MASC is positive under family-clustered inference, narrowly, so the cross-cohort contrast "
              "rests on it alone; Brennan's epochs had no baseline, with one its sign is not determined", "Limitations")
        # App D protocol: broadband filter description, checked against the code
        src_c = (ROOT / "codes/p4_concordance32.py").read_text() if (ROOT / "codes/p4_concordance32.py").exists() else ""
        src_r = (ROOT / "codes/compute_rdms.py").read_text() if (ROOT / "codes/compute_rdms.py").exists() else ""
        src_p = (ROOT / "codes/perband_reliability_multicohort.py").read_text() if (ROOT / "codes/perband_reliability_multicohort.py").exists() else ""
        P.check("GL_proto_code", 'if band != "broadband":' in src_c and '"broadband": (1, 150)' in src_r
                and 'if band_name == "broadband":\n            band_data = sess_epochs' in src_r and '"broadband": (1, 80)' in src_p,
                "code: concordance and tensor routes leave broadband unfiltered (FREQ_BANDS (1, 150) unused); per-band uses 1-80")
        P.has("GL_proto_tex", T, "filter the narrow bands the same way but apply no filter for broadband, which there is the "
              "stored epoch: baseline-corrected for LibriBrain and MEG-MASC, and neither filtered nor baseline-corrected for "
              "the Brennan rebuild", "App D protocol")
        # App L preprocessing paragraph
        a0 = K0["K0"]
        P.check("GL_k0_npos", a0["a"]["spearman_Db_dt_n_pos"] == K0["n_listeners"] == 47, "drift correlation positive in 47/47")
        c0 = a0["c"]["spearman_Db_dt_median"]
        P.has("GL_pp_k0", T, f"(median Spearman ${sg(a0['a']['spearman_Db_dt_median'], 3)}$, positive in all $47$ listeners); a "
              f"baseline or detrend removes this (${sg(a0['b']['spearman_Db_dt_median'], 3)}$, ${abs(c0):.3f}$), while model "
              f"dissimilarities barely track story time (median ${sg(K0['Dm_vs_time_S01']['median_spearman_Dm_dt'], 3)}$)",
              "App L preprocessing (K0)")
        P.check("GL_pp_var", a0["a"]["var_share_median"] > 0.9999, f"stored var share median {a0['a']['var_share_median']:.6f} > 99.99%")
        P.has("GL_pp_var_tex", T, "carry over $99.99\\%$ of each pattern's variance (median)", "App L preprocessing")
        P.has("GL_pp_k1", T, f"correlates ${sg(K0['S_b'], 2)}$ (baseline) and ${sg(K0['S_c'], 2)}$ (detrend) with the stored one",
              "App L preprocessing (K1)")
        up = b["family_seed_sweep"]["upper"]
        P.has("GL_pp_conc", T, f"concordance ${sg(b['rho'], 3)}$, checkpoint interval {iv(*fb)}, family-clustered {iv(*Fb)} "
              f"(upper bound ${sg(up[0], 2)}$ to ${sg(up[1], 2)}$ over seeds $0$--$19$) (${sg(cc['rho'], 3)}$, {iv(*fc)}, {iv(*Fc)}), "
              "not determined by the registered reading", "App L preprocessing (K2 concordance)")
        P.has("GL_pp_rel", T, f"split-half ranking reliability ${b['rsa_reliability_sb']:.2f}$ for similarity and "
              f"${b['enc_reliability_sb']:.2f}$ for encoding (${cc['rsa_reliability_sb']:.2f}$, ${cc['enc_reliability_sb']:.2f}$; "
              f"${NSb['rsa_rank_reliability']:.3f}$, ${NSb['enc_rank_reliability']:.3f}$)", "App L preprocessing (reliability)")
        def tr(d):
            t_ = [v["tau_median"] for v in d.values()]; r_ = [v["ratio"] for v in d.values()]
            return t_, r_
        tb_, rb_ = tr(b["rank_identifiability"]); tc_, rc_ = tr(cc["rank_identifiability"])
        ta_, ra_ = tr({k: v for k, v in RIc.items() if k.startswith("concordance32_brennan_v2|")})
        P.has("GL_pp_tau", T, f"unit-half Kendall $\\tau$ ${min(tb_):.2f}$--${max(tb_):.2f}$ with spread "
              f"${min(rb_):.2f}$--${max(rb_):.2f}\\times$ the paired-difference SE (${min(tc_):.2f}$--${max(tc_):.2f}$, "
              f"${min(rc_):.2f}$--${max(rc_):.2f}\\times$; ${min(ta_):.2f}$--${max(ta_):.2f}$, ${min(ra_):.1f}$--${max(ra_):.1f}\\times$)",
              "App L preprocessing (identifiability)")
        P.has("GL_pp_gap", T, f"class gap ${sg(gb['class_gap_causal_minus_other'], 1)}$ ranks, family-clustered "
              f"$p={gb['gap_p_family_clustered']:.3f}$ (${sg(gc['class_gap_causal_minus_other'], 1)}$, $p={gc['gap_p_family_clustered']:.3f}$; "
              f"${sg(DBR['class_gap_causal_minus_other'], 1)}$, $p={DBR['gap_p_family_clustered']:.2f}$), the sign LibriBrain has",
              "App L preprocessing (class gap)")
        P.check("GL_rdm_gate", abs(a0["a"]["rdm_reliability"] - NSb["neural_rdm_reliability"]) < 1e-6,
                "K0 raw RDM reliability reproduces the published 0.858 receipt")
        P.has("GL_pp_rdm", T, f"word-level RDM reliability ${a0['b']['rdm_reliability']:.3f}$ (${a0['c']['rdm_reliability']:.3f}$; "
              f"${a0['a']['rdm_reliability']:.3f}$)", "App L preprocessing (RDM reliability)")
        pr = K2["published_raw"]
        P.has("GL_pp_342", T, f"the comparable stored concordance is ${sg(pr['rho_v2'], 3)}$, not ${sg(pr['table1_spacefix'], 3)}$",
              "App L preprocessing (cached-feature comparator)")
        P.has("GL_lineage_scope", T, "used the rebuild's epochs as stored, without a baseline, and were not recomputed after the "
              "correction below", "App L lineage (36/47, 3-8x, +0.18 scoped, not recomputed)")
        # App F: RDM reliability range and repeat removal (cv arm, seed mean, like the published +0.140)
        rl, rm_ = NSl["neural_rdm_reliability"], NSm["neural_rdm_reliability"]
        rngb = max(rl, rm_, a0["b"]["rdm_reliability"]) - min(rl, rm_, a0["b"]["rdm_reliability"])
        rngc = max(rl, rm_, a0["c"]["rdm_reliability"]) - min(rl, rm_, a0["c"]["rdm_reliability"])
        P.has("GL_F_rel", T, f"are ${rl:.3f}$ (LibriBrain), ${rm_:.3f}$ (MEG-MASC) and ${a0['b']['rdm_reliability']:.3f}$ (Brennan with a "
              f"per-epoch baseline; ${a0['c']['rdm_reliability']:.3f}$ detrended, ${a0['a']['rdm_reliability']:.3f}$ on the stored epochs, "
              f"Appendix~\\ref{{app:brennan-preproc}}) --- a range of ${rngb:.2f}$ (${rngc:.2f}$ detrended)", "App F (reliability)")
        rp = K2["repeat_k1_b"]
        seeds = sorted(v["draw_seed"] for v in rp.values())
        cvc = float(np.mean([v["arms"]["cv"]["concordance"] for v in rp.values()]))
        cvr = float(np.mean([v["arms"]["cv"]["rsa_rank_reliability"] for v in rp.values()]))
        P.check("GL_F_rep_seeds", seeds == [0, 1, 2] and all(v["k"] == 1 for v in rp.values()), f"repeat k=1 seeds {seeds}")
        P.has("GL_F_rep", T, f"${sg(cvc, 3)}$ on Brennan (mean over draw seeds $0$--$2$; $+0.140$ on the stored epochs)",
              "App F repeat removal (cv arm; +0.140 = published cv seed mean, ELN 2026-08-03, no local receipt)")
        P.has("GL_F_rep_rel", T, f"(split-half ${sg(cvr, 2)}$ with the baseline, below $0.2$ without it)", "App F repeat removal")
        # App I depth, App M segments, App O mechanism, Figure 3
        P.has("GL_I", T, f"with one it is ${sg(b['rho'], 2)}$, not determined", "App I")
        seg = K2["cross_8seg_b"]
        nr = sum(v["rsa_rank_reliability_ci"][0] > 0 for v in seg.values())
        ne = sum(v["enc_rank_reliability_ci"][0] > 0 for v in seg.values())
        nb = sum(bool(v["rankable"]) for v in seg.values())
        rr_ = [v["rsa_rank_reliability"] for v in seg.values()]
        words = {4: "four", 5: "five", 6: "six", 3: "three"}
        P.check("GL_M_counts", len(seg) == 8 and nr == ne and nb == sum(v["rsa_rank_reliability_ci"][0] > 0 and v["enc_rank_reliability_ci"][0] > 0 for v in seg.values()),
                f"8 segments; RSA determined {nr}, encoding {ne}, both {nb}")
        P.has("GL_M", T, f"similarity reliability is ${min(rr_):.2f}$--${max(rr_):.2f}$, similarity and encoding are each determined on "
              f"{words.get(nr, nr)} segments and both on {words.get(nb, nb)}, so that reading does not survive", "App M")
        P.has("GL_O", T, f"on Brennan EEG the class gap is ${sg(gb['class_gap_causal_minus_other'], 1)}$ ranks with a per-epoch baseline",
              "App O")
        P.has("GL_fig3", T, "Brennan is the registered per-epoch-baseline value (\\texttt{brennan\\_k2\\_summary\\_20261009}", "Fig. 3 caption")
    # superseded Brennan wording (un-baselined values presented as current)
    for cid, s in (("GL_old_row", "Brennan EEG, $47$ listeners$^\\P$ & $+0.366$"),
                   ("GL_old_identical", "Under the identical protocol"),
                   ("GL_old_crosses", "Brennan's crosses it ($[-0.20, +0.69]$)"),
                   ("GL_old_neargap", "The class gap is near zero on Brennan"),
                   ("GL_old_eight", "in all eight cohort, readout and aggregation"),
                   ("GL_old_eight51", "in all eight LibriBrain and Brennan"),
                   ("GL_old_twopos", "two other cohorts give positive point estimates"),
                   ("GL_old_twocoh", "but not on two other cohorts;"),
                   ("GL_old_rep", "concordance is $+0.140$ (Brennan) and $+0.191$ (MEG-MASC)"),
                   ("GL_old_I", "Brennan's $+0.37$ and MEG-MASC's $+0.45$"),
                   ("GL_old_F", "and $0.858$ (Brennan) --- a range of $0.08$"),
                   ("GL_old_M", "$0.995$--$0.999$, while encoding is undetermined on seven"),
                   ("GL_old_O", "on Brennan EEG the class gap is $-0.4$ ranks"),
                   ("GL_old_band", "Each epoch is filtered into six bands"),
                   ("GL_old_unchanged", "The Brennan numbers are therefore unchanged"),
                   ("GL_old_abs", "not reproducing on $27$-listener MEG or $47$-subject English EEG")):
        P.check(cid, s not in T, f"superseded: {s[:60]!r} absent")

    # ================= Group F: epsilon-guard constant, checked against the actual source, not just the tex =================
    src = (ROOT / "codes/p4_concordance32.py")
    src_text = src.read_text() if src.exists() else ""
    P.check("GF_epsilon_source_is_1e-12",
            "1e-12" in src_text and "1e+12" not in src_text and "1e12" not in src_text.replace("1e-12", ""),
            f"codes/p4_concordance32.py denominator guard literal is 1e-12 (L516's '$10^{{-12}}$' claim)")

    # ================= Group E: Brennan 49/47 arithmetic identity + filename corroboration =================
    # No JSON field stores "49 listeners, 2 corrupt -> 47" directly; L254/L481/L38 are pinned above as
    # literal anchors. This is the supporting arithmetic + evidence-of-existence check.
    brennan_files = list(results.glob("**/brennan47_*"))
    P.check("GE_brennan_49_minus_2_is_47", 49 - 2 == 47,
            "49 public listeners minus 2 corrupt files = 47 (arithmetic identity backing L254/L481/L38)")
    P.check("GE_brennan47_receipt_exists", len(brennan_files) > 0,
            f"at least one results/**/brennan47_* receipt exists ({len(brennan_files)} found)")

    # ================= Group A: concordance32_CORRECTED.json headline numbers =================
    jA = _load(results, "p4_encoding/concordance32_CORRECTED.json")
    P.check("GA_n_models_32", _rget(jA, "n_models") == 32,
            f"concordance32_CORRECTED.json n_models == 32 (backs L57/L61/L443/L445/L516-519 '32')")
    P.check("GA_rsa_reliability_098", f"{_rget(jA, 'rsa_reliability_sb', default=float('nan')):.2f}" == "0.98",
            "concordance32_CORRECTED.json rsa_reliability_sb rounds to 0.98 (backs L59/L443)")
    P.check("GA_enc_reliability_096", f"{_rget(jA, 'enc_reliability_sb', default=float('nan')):.2f}" == "0.96",
            "concordance32_CORRECTED.json enc_reliability_sb rounds to 0.96 (backs L60/L443)")
    P.check("GA_ceiling_097", f"{_rget(jA, 'attenuation_ceiling', default=float('nan')):.2f}" == "0.97",
            "concordance32_CORRECTED.json attenuation_ceiling rounds to 0.97 (backs L443)")
    P.check("GA_concordance_spearman_0573", f"{_rget(jA, 'concordance_spearman', default=float('nan')):.3f}" == "-0.573",
            "concordance32_CORRECTED.json concordance_spearman rounds to -0.573 (backs L61/L443/L516-519)")

    # ================= Group B: n_families / n_scored / n_base / registered reversal rho+CI =================
    jB = _load(results, "p4_encoding/concordance32_c4.json")
    P.check("GB_n_families_12", _rget(jB, "n_families") == 12,
            "concordance32_c4.json n_families == 12 (backs L57/L262/L312 '12 families')")
    P.check("GB_n_models_32", _rget(jB, "n_models") == 32,
            "concordance32_c4.json n_models == 32")
    P.check("GB_n_sessions_12", _rget(jB, "n_sessions") == 12,
            "concordance32_c4.json n_sessions == 12")

    jC = _load(results, "p4_encoding/space_fix_20260925/aggregate/adr0006_registered_verdicts_spacefix.json")
    P.check("GB_n_scored_57", _rget(jC, "n_scored") == 57,
            "adr0006_registered_verdicts_spacefix.json n_scored == 57 (backs L67/L99/L262/L313 '57')")
    P.check("GB_q2_n_base_48", _rget(jC, "q2", "n_base") == 48,
            "adr0006_registered_verdicts_spacefix.json q2.n_base == 48 (backs L86/L313/L446 '48 base')")
    P.check("GB_q2_n_families_12", _rget(jC, "q2", "n_families") == 12,
            "adr0006_registered_verdicts_spacefix.json q2.n_families == 12")
    q2rho = _rget(jC, "q2", "rho", default=float("nan"))
    P.check("GB_q2_rho_minus059", f"{q2rho:.2f}" == "-0.59",
            "adr0006_registered_verdicts_spacefix.json q2.rho rounds to -0.59 (backs L37/L86 reversal rho)")
    ci = _rget(jC, "q2", "ci95_family_clustered", default=[float("nan"), float("nan")])
    P.check("GB_q2_ci_lo_078", f"{ci[0]:.2f}" == "-0.78", "q2 family-clustered CI lower bound rounds to -0.78")
    P.check("GB_q2_ci_hi_012", f"{ci[1]:.2f}" == "-0.12", "q2 family-clustered CI upper bound rounds to -0.12")

    # ================= Group C: ds003020 / crossed_narrative directory arithmetic (backs L322) =================
    ds_dir = results / "p4_encoding/ds003020"
    ds_files = list(ds_dir.glob("*.json")) if ds_dir.is_dir() else []
    subjects = sorted({f.name.split("_")[0] for f in ds_files if f.name.startswith("UTS")})
    P.check("GC_ds003020_9_subjects", len(subjects) == 9,
            f"ds003020/ has exactly 9 distinct UTS subject prefixes ({len(subjects)} found)")
    P.check("GC_ds003020_25_stories_per_subject", len(subjects) > 0 and len(ds_files) // max(len(subjects), 1) == 25,
            f"ds003020/ file count / subject count == 25 ({len(ds_files)}/{len(subjects)}) "
            "(backs L322 'ds003020 fMRI ($25$ x $9$)')")
    cn_dir = results / "p4_encoding/crossed_narrative"
    meg_masc_stories = list(cn_dir.glob("meg_masc_merged27_story-*_full.json")) if cn_dir.is_dir() else []
    P.check("GC_megmasc_4_stories", len(meg_masc_stories) == 4,
            f"crossed_narrative/ has exactly 4 meg_masc_merged27_story-*_full.json files "
            f"({len(meg_masc_stories)} found) (backs L322 '4 stories')")

    # ================= Group D: selection_tax_multicohort.json per-cohort unit/subject counts =================
    jD = _load(results, "selection_tax_multicohort.json")
    P.check("GD_megscans_23_subjects", _rget(jD, "MEG-SCANS", "n_subjects") == 23,
            "selection_tax_multicohort.json MEG-SCANS.n_subjects == 23 (backs L251 '23 of its 24 listeners')")
    P.check("GD_libribrain_36_units", _rget(jD, "LibriBrain", "n_units") == 36,
            "selection_tax_multicohort.json LibriBrain.n_units == 36 (backs L249 '36 Sherlock 1 MEG recordings')")

    # ================= Group G: deterministic per-session depth recompute (bypasses reproduce.py's bug) =================
    # bench/reproduce.py's c_depth_sessions_negative()/c_depth_session_p() read concordance_depth_curve.json
    # (the pre-fix file) because concordance_depth_curve_CORRECTED.json has no independent_per_session key.
    # This recomputes the same quantity directly and deterministically from concordance32_CORRECTED.json's
    # own per-session matrices (no RNG anywhere), independent of both depth-curve files and reproduce.py.
    rsa_ps, enc_ps = _rget(jA, "per_session_rsa"), _rget(jA, "per_session_enc")
    if rsa_ps and enc_ps:
        ms = sorted(set(rsa_ps) & set(enc_ps))
        sess = sorted(set.intersection(*[set(rsa_ps[m]) for m in ms])) if ms else []
        vals = []
        for s in sess:
            a = [rsa_ps[m][s] for m in ms]
            b = [enc_ps[m][s] for m in ms]
            vals.append(spearmanr(a, b)[0])
        vals = np.array(vals)
        n_neg = int((vals < 0).sum())
        P.check("GG_depth_n_sessions_12", len(sess) == 12, f"concordance32_CORRECTED.json has 12 sessions ({len(sess)} found)")
        P.check("GG_depth_11_of_12_negative", n_neg == 11,
                f"independent per-session recompute: {n_neg}/12 negative (backs L1110-1111 '11 of the 12 are negative')")
        P.check("GG_depth_mean_minus0389", f"{vals.mean():.3f}" == "-0.389",
                "independent per-session recompute mean rounds to -0.389 (backs L1111)")
        P.check("GG_depth_median_minus0421", f"{np.median(vals):.3f}" == "-0.421",
                "independent per-session recompute median rounds to -0.421 (backs L1111)")
        P.check("GG_depth_min_minus063", f"{vals.min():.2f}" == "-0.63",
                "independent per-session recompute min rounds to -0.63 (backs L1111 'values -0.63 to +0.03')")
        P.check("GG_depth_max_plus003", f"{vals.max():.2f}" == "0.03",
                "independent per-session recompute max rounds to +0.03 (backs L1111)")
    else:
        P.check("GG_depth_per_session_matrices_present", False,
                "concordance32_CORRECTED.json is missing per_session_rsa/per_session_enc -- cannot "
                "verify the depth-curve appendix claims at all")

    P.has("GG_depth_anchor_1111", T,
          "the $12$ are negative (values $-0.63$ to $+0.03$, mean $-0.389$, median $-0.421$).",
          "app:depth (Depth and the Readout Reversal)")

    # ================= Group H: depth-curve bootstrap values, read directly from the _CORRECTED file =================
    jH = _load(results, "p4_encoding/concordance_depth_curve_CORRECTED.json")
    curve = _rget(jH, "curve", default={})
    expected_k = {"12": "-0.573", "10": "-0.574", "5": "-0.556", "2": "-0.490", "1": "-0.393"}
    for k, want in expected_k.items():
        got = f"{_rget(curve, k, 'mean', default=float('nan')):.3f}" if k in curve else "n/a"
        P.check(f"GH_depth_curve_k{k}", got == want,
                f"concordance_depth_curve_CORRECTED.json curve[{k}].mean rounds to {want} (got {got})")
    P.check("GH_depth_curve_k1_sd_0181", f"{_rget(curve, '1', 'sd', default=float('nan')):.3f}" == "0.181",
            "curve['1'].sd rounds to 0.181 (backs L1107 '+-0.181 at k=1')")
    P.check("GH_depth_curve_k10_sd_0032", f"{_rget(curve, '10', 'sd', default=float('nan')):.3f}" == "0.032",
            "curve['10'].sd rounds to 0.032 (backs L1107 '+-0.032 at k=10')")
    P.check("GH_depth_curve_k1_frac_neg_92pct", f"{_rget(curve, '1', 'frac_negative', default=float('nan')) * 100:.0f}" == "92",
            "curve['1'].frac_negative rounds to 92% (backs L1105 '92% of draws are still negative')")
    P.check("GH_depth_spearman_minus090", f"{_rget(jH, 'spearman_k_vs_concordance', default=float('nan')):.2f}" == "-0.90",
            "concordance_depth_curve_CORRECTED.json spearman_k_vs_concordance rounds to -0.90 (backs L1102)")

    P.has("GH_depth_anchor_1101", T,
          r"concordance is $-0.573$ at $k=12$, $-0.574$ at $k=10$, $-0.556$ at $k=5$, $-0.490$ at $k=2$ "
          r"and $-0.393$ at $k=1$ (Spearman $\rho = -0.90$ against $k$",
          "app:depth (Depth and the Readout Reversal)")
    P.has("GH_depth_anchor_1105_92pct", T,
          r"$92\%$ of draws are still negative and the mean sits at $-0.393$",
          "app:depth (Depth and the Readout Reversal)")
    P.has("GH_depth_anchor_1107_dispersion", T,
          r"falling from $\pm 0.181$ at $k=1$ to $\pm 0.032$ at $k=10$",
          "app:depth (Depth and the Readout Reversal)")

    # ================= Group I: abstract "+0.76 to +0.96" vs Table 1's "Listener halves" row =================
    # Cross-location consistency, not an independent receipt derivation: the row's own 200-iteration
    # permutation resampling is independently verified by check_table1_receipts.py. This only checks
    # that the abstract's rounded range still matches whatever that row currently says.
    m = re.search(
        r"Listener halves & \$([+-][0-9.]+)\$ & \$([+-][0-9.]+)\$ & \$([+-][0-9.]+)\$ & \$([+-][0-9.]+)\$",
        T)
    if m:
        cells = [float(x) for x in m.groups()]
        lo, hi = min(cells), max(cells)
        # R1 2026-10-07 (plan A6): the abstract now prints the row's own 3-dp values, so the pin is exact
        # equality with the row's min/max (stricter than the former 2-dp rounding check)
        P.check("GI_abstract_listener_halves_lo", f"{lo:.3f}" == "0.764",
                f"Table 1 'Listener halves' row min is +0.764 (got {lo:.3f}); abstract claims '+0.764 to +0.961'")
        P.check("GI_abstract_listener_halves_hi", f"{hi:.3f}" == "0.961",
                f"Table 1 'Listener halves' row max is +0.961 (got {hi:.3f}); abstract claims '+0.764 to +0.961'")
        P.has("GI_abstract_equals_row", ABS, f"$+{lo:.3f}$ to $+{hi:.3f}$", "abstract (row min/max, derived)")
    else:
        P.check("GI_table1_listener_halves_row_found", False,
                "could not locate Table 1's 'Listener halves' row to cross-check against the abstract")
    P.has("GI_abstract_text", ABS,
          "disjoint listener halves agree at median Spearman $\\rho$ of $+0.764$ to $+0.961$",
          "abstract")


    # ================= Group J: second-round pins (seed-1 mutation sweep, 2026-10-05) =================
    # The first round (39 pins, Group above + Groups A-I) was built from the ORIGINAL audit's missed-
    # mutation list and caught 60/60 (100%) against a seed-0 random resample of the n=60 preset, but
    # only 32/60 (53%) against a seed-1 resample -- a different random draw from the same ~237-candidate
    # result-class pool hits numbers the first round never looked at. These 20 anchors cover all 27
    # distinct misses from that seed-1 run (report.md / MUTATION_SWEEP_20261005 second pass), literal-
    # anchor-only except where noted; none overlaps an existing Group A-I pin's exact substring.
    P.has("GJ_L43_fmri_078", ABS,
          r"a fully crossed fMRI cohort ($25$ narratives $\times$ $9$ listeners) transfers well ($+0.778$).",  # R1: Table 1's 3-dp value (was +0.78)
          "abstract (covers L43 sign_flip +0.78->-0.78)")
    P.has("GJ_L45_abstract_sweep_326years", ABS,
          r"a readout sweep extended from $32$ to $57$ checkpoints ($48$ base) spanning $2019$--$2026$.",
          "abstract (covers L45 swap_sent 32<->57<->48, both directions)")
    P.has("GJ_L60_enc_reliability096", T,
          r"ridge encoding \citep{huth2016natural} returns an equally reliable one ($0.96$).",
          "body (intro) (covers L60 digit_swap 0.96->9.06)")
    P.has("GJ_L174_distance_defs", T,
          r"$D^{\mathrm b}_{ij} = 1-\operatorname{corr}(y_i,y_j)$ and "
          r"$D^{(m)}_{ij} = 1-\langle x_i,x_j\rangle/(\|x_i\|\|x_j\|)$, the unit score is a partial Spearman",
          "body (similarity readout eq.) (covers L174 equation last_digit 1->0)")
    P.has("GJ_L222_ridge_svd_sumk", T,
          r"the training-fold fitted values are $\sum_k\frac{s_k^2}{s_k^2+\alpha}\,u_k u_k^\top\tilde Y_{-u}$. "
          r"\textbf{(iv)} If",
          "body (ridge SVD identity, between eq:ridge and Prop.1) (covers L222 equation last_digit 2->3)")
    P.has("GJ_L225_participation_frac", T,
          r"its participation ratio, the leading component "
          r"carries at least a fraction $1/\mathrm{PR}$ of the squared Frobenius norm of the centered Gram "
          r"matrix $\bar X\bar X^\top$.",
          "Proposition 1(iv) (covers L225 equation last_digit 1->2)")
    P.has("GJ_L250_megmasc_tensor27", T,
          r"is English MEG ($10$ subjects in the tensor archive, $27$ "
          r"in the raw rebuild for crossed and readout analyses)",
          "body (cohorts) (covers L250 last_digit 27->26)")
    P.has("GJ_L314_reliability_ceiling_48base", T,
          r"concordance below the "
          r"reliability ceiling; $48$ base: family-clustered interval below zero); the C4 row and the",
          "Table 1 caption (covers L314 last_digit 48->49; extends L313_registered3248's anchor rightward)")
    P.has("GJ_L322_megmasc_4stories27", T,
          r"MEG-MASC ($4$ stories $\times$ $27$)",
          "Table 1 header (covers L322 last_digit 4->3 and 27->28)")
    P.has("GJ_L382_crossed_32models", T,
          r"we score all $32$ models within each story; each story's trials span both recording sessions of",
          "body (crossed-story section) (covers L382 last_digit 32->31)")
    P.has("GJ_L442_libribrain_rank32", T,
          r"On LibriBrain the two readouts rank the $32$ checkpoints in "
          r"\emph{opposite} orders ($\rho=-0.573$).",
          "body (readout reversal section) (covers L442 last_digit 32->31; precedes L443_bundle's anchor)")
    P.has("GJ_L446_depth48_years", T,
          r"the reversal also survives "
          r"the $48$-checkpoint base set that spans $2019$--$2026$ (Table~\ref{tab:main})",
          "body (readout reversal section) (covers L446 digit_swap 2019->2109)")
    P.has("GJ_L460_megmasc_306channels", T,
          r"scoring all "
          r"$306$ channels or dropping C1--C3 leaves the concordance close to its headline value",
          "body (readout reversal section) (covers L460 digit_swap 306->036)")
    P.has("GJ_L484_libribrain_11of12_reverse", T,
          r"does not explain the difference ($11$ of $12$ LibriBrain sessions reverse on their own, "
          r"Appendix~\ref{app:depth})",
          "body (limitations) (covers L484 digit_swap 12->21 and last_digit 11->12; distinct from "
          "Group G/H's depth-curve recompute, which backs a different sentence in the appendix)")
    P.has("GJ_L497_72B_under_05B", T,
          r"regularization-sensitive, while $72$B stays below $0.5$B (Appendix Table~\ref{tab:alpha-grid})",
          "body (additional findings, scaling) (covers L497 digit_swap 0.5->5.0)")
    P.has("GJ_L505_broadband_306_213", T,
          r"broadband reaches a higher peak "
          r"alignment than $30$--$80$~Hz gamma in $206$ of $213$ recordings",
          "body (band comparison) (covers L505 structural digit_swap 80->08 and last_digit 206->205)")
    P.has("GJ_L542_543_magnitude_scale", T,
          r"Raw best-cell alignment is $10^{-3}$--$10^{-2}$ in MEG "
          r"and $\approx2\times10^{-2}$ in ECoG, and headline values select cells in-sample at a $6$--$85\%$ cost",
          "body (magnitude scale and in-sample selection) (covers L542 last_digit 2->3, digit_swap 10->01, "
          "and L543 last_digit 6->7, all in one sentence)")
    P.has("GJ_L592_504_values", T,
          r"Its validation recomputes all $504$ stored per-(model, "
          r"session) values behind the LibriBrain cell to within $10^{-9}$ from cached neural dissimilarity",
          "body (claim-reproduction harness description) (covers L592 swap_sent 504->9)")

    # Group K: the one residual seed-1 miss (round-3 valid rerun, post-commit) -- a repeated
    # table-header boilerplate number, pinned via its unique "interval or note" anchor.
    P.has("GK_L339_table2_header_95pct", T,
          r"Readout concordance on & $\rho(\pi^{\mathrm{RSA}},\pi^{\mathrm{enc}})$ & "
          r"95\% interval or note",
          "Table (readout concordance) header row (covers L339 structural digit_swap 95->59; "
          "unique anchor via \"interval or note\", distinct from the 95% intervals in the Table 1 rows above)")

    # ================= Group M: misses of the 2026-10-10 sweeps (main scope, seeds 0 and 1, HEAD c1a4332e) =================
    # Literal sentence anchors only (no receipt recomputation): one pin per missed main-scope mutation.
    P.has('GM_L52_pythia70M', T,
          'Model size did not consistently order alignment on a Pythia ladder ($70$M--$2.8$B) or across three Qwen generations (up to $72$B).',
          'abstract')
    P.has('GM_L193_ridge_inverse', T,
          '= (\\tilde X_{-u}^\\top\\tilde X_{-u}+\\alpha I)^{-1}\\tilde X_{-u}^\\top\\tilde Y_{-u},',
          'eq:ridge closed form')
    P.has('GM_L269_18of25', T,
          '$18$ of the $25$ added checkpoints were first scored on text without inter-word spaces',
          '4.1')
    P.has('GM_L281_27cell', T,
          'Inference treats the unit as the replication unit in a mixed-effects model over a $27$-cell grid fixed in advance',
          '4.2')
    P.has('GM_L291_tax', T,
          'costs $85\\%$ of the value across nine ECoG patients and $6\\%$ across the $36$ recordings of one MEG listener',
          '4.3')
    P.has('GM_L316_95', T,
          '\\emph{Bottom:} readout concordance with 95\\% intervals, [ckpt] Fisher-$z$ over checkpoints',
          'Table 1 caption')
    P.has('GM_L320_14', T,
          'common-reading-rule row ($14$ masked and encoder--decoder checkpoints read up to the target word, as causal ones are;',
          'Table 1 caption')
    P.has('GM_L370_7791', T,
          "for the encoding readout $77$--$91\\%$ of each model's marginal standard error is this shared component",
          '5.1')
    P.has('GM_L389_4x200', T,
          'listener halves (MEG-MASC, $4$ stories $\\times$ $200$ splits; ds003020, per-story medians), story pairs (MEG-MASC $6$ pairs',
          'Fig. 2 caption')
    P.has('GM_L413_gt2', T,
          'a story-to-listener dispersion ratio with threshold $>2$, came out below one for both readouts',
          '5.2')
    P.has('GM_L444_32rank1', T,
          '(a)~Similarity rank against encoding rank for the $32$ LibriBrain checkpoints (rank $1$ strongest);',
          'Fig. 3 caption')
    P.has('GM_L525_masc27', T,
          "On MEG-MASC's $27$ listeners the same rungs do not order similarity by size, and the encoding arm fails its rankability gate",
          '5.4')
    P.has('GM_L549_0678', T,
          'changed the readout concordance of an earlier analysis in this project from $-0.573$ to $-0.678$; changing only that constant recovers $-0.573$.',
          '5.5')
    P.has('GM_L624_48', T,
          'CORRECTED.txt} and, for the $48$ base checkpoints,',
          'Reproducibility')

    return P


# ---------------------------------------------------------------- selftest

def _tex_mutants() -> dict[str, tuple[str, str]]:
    """One entry per missed mutation in report.md (39 total; two anchors each bundle 2)."""
    return {
        "L38 digit_swap 47->74": ("$47$-subject English EEG", "$74$-subject English EEG"),
        "L38 not determined -> determined": ("MEG and is not determined on $47$-subject",
                                             "MEG and is determined on $47$-subject"),
        "L57 last_digit 12->11": ("Score $32$ checkpoints from $12$ families", "Score $32$ checkpoints from $11$ families"),
        "L59 last_digit 0.98->0.97": ("ranking reliability $0.98$; ridge encoding", "ranking reliability $0.97$; ridge encoding"),
        "L61 sign_flip -0.573->+0.573": (r"opposite directions ($\rho=-0.573$)", r"opposite directions ($\rho=+0.573$)"),
        "F1 cap claim listeners->stories": ("survives new listeners, but not always a new story",
                                            "survives new stories, but not always a new story"),
        "F1 cap digit_swap 95->59": ("family-clustered 95\\% intervals; grey", "family-clustered 59\\% intervals; grey"),
        "L86 digit_swap 48->84": ("the $48$ base checkpoints of the registered test,", "the $84$ base checkpoints of the registered test,"),
        "L99 digit_swap 2026->0226": (r"($2019$--$2026$) under one protocol", r"($2019$--$0226$) under one protocol"),
        "L99 digit_swap 32->23 (sweep)": ("a $32$-to-$57$-checkpoint readout sweep", "a $23$-to-$57$-checkpoint readout sweep"),
        "L166 swap_sent 400->300": (r"$300$--$400$~ms), flattened", r"$300$--$300$~ms), flattened"),
        "L169 digit_swap 0.5->5.0": ("at relative depth $0.5$, one layer", "at relative depth $5.0$, one layer"),
        "L184 last_digit 1->2 (ridge mu)": (r"\mathbf 1\mu^\top", r"\mathbf 2\mu^\top"),
        "L187 last_digit 2->3 (frobenius)": (r"\|_F^2+\alpha\|W\|_F^2", r"\|_F^3+\alpha\|W\|_F^2"),
        "L198 swap_sent 2->6 (rank formula)": (r"1-6\sum_m(\pi_m-\pi'_m)^2", r"1-6\sum_m(\pi_m-\pi'_m)^6"),
        "L198 last_digit 6->7 (rank formula coeff)": (r"\rho(\pi,\pi')=1-6\sum", r"\rho(\pi,\pi')=1-7\sum"),
        "L215 last_digit 0->1 (prop1i c>0)": ("with $c>0$ and $Q$", "with $c>1$ and $Q$"),
        "L218 last_digit 1->0 (prop1ii 1b)": (r"\mathbf 1b^\top$ with $A$", r"\mathbf 0b^\top$ with $A$"),
        "L224 swap_sent 2->1 (participation ratio)": (r"\mathrm{PR}=(\sum_k\lambda_k)^2/\sum_k\lambda_k^2",
                                                        r"\mathrm{PR}=(\sum_k\lambda_k)^1/\sum_k\lambda_k^2"),
        "L233 last_digit 2->3 (sk^2)": (r"$s_k^2\gg\alpha$, whatever", r"$s_k^3\gg\alpha$, whatever"),
        "L249 digit_swap 36->63": ("we use its $36$ Sherlock~1", "we use its $63$ Sherlock~1"),
        "L251 last_digit 24->23": ("from $23$ of its $24$ listeners", "from $23$ of its $23$ listeners"),
        "L254 swap_sent 47->27": ("($47$ in rebuilt analyses", "($27$ in rebuilt analyses"),
        "L254 no-baseline clause dropped": ("corrupt; the rebuilt epochs carry no filter or baseline,", "corrupt; the rebuilt epochs carry a baseline,"),
        "L262 last_digit 12->11 (families)": ("checkpoint sweep over $12$ families and three", "checkpoint sweep over $11$ families and three"),
        "L308 digit_swap 95->59": (r"with its 95\% interval.", r"with its 59\% interval."),
        "L312 digit_swap 12->21 (fam boot)": (r"bootstrap over $12$ families.", r"bootstrap over $21$ families."),
        "L313 last_digit 32->33": ("Registered test that held ($32$ checkpoints", "Registered test that held ($33$ checkpoints"),
        "L322 last_digit 25->26": (r"ds003020 fMRI ($25$ $\times$ $9$)", r"ds003020 fMRI ($26$ $\times$ $9$)"),
        "L381 last_digit 27->26": ("all $27$ listeners heard all four", "all $26$ listeners heard all four"),
        "L394 digit_swap 0005->0050": ("(ADR 0005) and was", "(ADR 0050) and was"),
        "L420 old 8-segment sentence restored": ("dataset properties predict it. ",
                                                 r"dataset properties predict it, and encoding needs far more data than similarity before a ranking exists. "),
        "L443 digit_swap 0.573->0.753": (r"\emph{opposite} orders ($\rho=-0.573$). Both", r"\emph{opposite} orders ($\rho=-0.753$). Both"),
        "L445 digit_swap 32->23 (units)": ("over $12$ units rather than $32$,", "over $12$ units rather than $23$,"),
        "L480 swap_sent 60->27": ("listeners, all $60$ channels) adds", "listeners, all $27$ channels) adds"),
        "L481 digit_swap 27->72": (r"its $27$ listeners give two determined", r"its $72$ listeners give two determined"),
        # ---- 2026-10-10 sweep misses (Group M) ----
        'GM L52_pythia70M': ('Pythia ladder ($70$M--$2.8$B) or across', 'Pythia ladder ($71$M--$2.8$B) or across'),
        'GM L193_ridge_inverse': ('+\\alpha I)^{-1}\\tilde X_{-u}^\\top\\tilde Y_{-u},', '+\\alpha I)^{-0}\\tilde X_{-u}^\\top\\tilde Y_{-u},'),
        'GM L269_18of25': ('$18$ of the $25$ added', '$18$ of the $26$ added'),
        'GM L281_27cell': ('over a $27$-cell grid fixed', 'over a $28$-cell grid fixed'),
        'GM L291_tax': ('across the $36$ recordings of one MEG', 'across the $63$ recordings of one MEG'),
        'GM L316_95': ('readout concordance with 95\\% intervals, [ckpt]', 'readout concordance with 94\\% intervals, [ckpt]'),
        'GM L320_14': ('common-reading-rule row ($14$ masked', 'common-reading-rule row ($41$ masked'),
        'GM L370_7791': ('readout $77$--$91\\%$ of each', 'readout $76$--$91\\%$ of each'),
        'GM L389_4x200': ('(MEG-MASC, $4$ stories $\\times$ $200$ splits;', '(MEG-MASC, $3$ stories $\\times$ $200$ splits;'),
        'GM L413_gt2': ('with threshold $>2$, came out', 'with threshold $>1$, came out'),
        'GM L444_32rank1': ('LibriBrain checkpoints (rank $1$ strongest)', 'LibriBrain checkpoints (rank $32$ strongest)'),
        'GM L525_masc27': ("On MEG-MASC's $27$ listeners the same", "On MEG-MASC's $28$ listeners the same"),
        'GM L549_0678': ('from $-0.573$ to $-0.678$; changing', 'from $-0.573$ to $-6.078$; changing'),
        'GM L624_48': ('CORRECTED.txt} and, for the $48$ base checkpoints,', 'CORRECTED.txt} and, for the $49$ base checkpoints,'),
        # ---- R1 2026-10-10: Brennan K2 (Group L), one mutant per new pin ----
        "GL 5.3 MEG-MASC alone dropped": ("This rests on MEG-MASC alone.", "This rests on two cohorts."),
        "GL 5.3 identical protocol back": ("Its epochs, unlike LibriBrain's and MEG-MASC's, are neither filtered nor baseline-corrected",
                                          "Its epochs, under the identical protocol, are neither filtered nor baseline-corrected"),
        "GL 5.3 drift 0.885->0.858": ("(median Spearman $+0.885$). With", "(median Spearman $+0.858$). With"),
        "GL 5.3 rho sign": ("the Brennan concordance is $-0.248$", "the Brennan concordance is $+0.248$"),
        "GL 5.3 fam CI": ("(family-clustered CI $[-0.62, +0.14]$; linear", "(family-clustered CI $[-0.62, -0.14]$; linear"),
        "GL 5.3 detrend": ("linear detrend $-0.047$, $[-0.53, +0.39]$), not", "linear detrend $-0.074$, $[-0.53, +0.39]$), not"),
        "GL 5.3 gap p": ("family-clustered $p=0.012$) has", "family-clustered $p=0.021$) has"),
        "GL 5.1 spread": ("error ($0.86$--$1.10\\times$) and", "error ($0.86$--$1.01\\times$) and"),
        "GL 5.1 eight back": ("in all four LibriBrain readout $\\times$", "in all eight LibriBrain readout $\\times$"),
        "GL intro travel": ("once slow drift is removed, it is not determined;", "once slow drift is removed, it is positive;"),
        "GL discussion": ("but not on MEG-MASC, and on Brennan EEG", "but not on two other cohorts, and on Brennan EEG"),
        "GL limitations": ("so the cross-cohort contrast rests on it alone;", "so the cross-cohort contrast rests on both;"),
        "GL protocol broadband filtered": ("but apply no filter for broadband, which", "and filter broadband at 1--80 Hz, which"),
        "GL pp k0 detrend": ("removes this ($+0.002$, $0.000$)", "removes this ($+0.020$, $0.000$)"),
        "GL pp var share": ("carry over $99.99\\%$ of each", "carry over $99.9\\%$ of each"),
        "GL pp k1": ("correlates $-0.05$ (baseline)", "correlates $+0.05$ (baseline)"),
        "GL pp conc seeds": ("(upper bound $+0.12$ to $+0.14$ over", "(upper bound $+0.12$ to $+0.41$ over"),
        "GL pp ckpt CI": ("checkpoint interval $[-0.55, +0.11]$", "checkpoint interval $[-0.55, +0.13]$"),
        "GL pp rel": ("$0.45$ for similarity and $0.59$ for encoding", "$0.54$ for similarity and $0.59$ for encoding"),
        "GL pp stored rel": ("$0.977$, $0.970$)", "$0.970$, $0.977$)"),
        "GL pp tau": ("Kendall $\\tau$ $0.64$--$0.70$ with", "Kendall $\\tau$ $0.64$--$0.97$ with"),
        "GL pp stored spread": ("$0.93$--$0.97$, $2.4$--$6.8\\times$)", "$0.93$--$0.97$, $2.4$--$7.4\\times$)"),
        "GL pp gap": ("class gap $+15.6$ ranks, family-clustered", "class gap $-15.6$ ranks, family-clustered"),
        "GL pp rdm": ("RDM reliability $0.677$ ($0.743$; $0.858$)", "RDM reliability $0.743$ ($0.677$; $0.858$)"),
        "GL pp 342": ("stored concordance is $+0.342$, not", "stored concordance is $+0.366$, not"),
        "GL lineage scope": ("and were not recomputed after the correction below", "and were recomputed after the correction below"),
        "GL F rel range": ("a range of $0.11$ ($0.04$ detrended)", "a range of $0.08$ ($0.04$ detrended)"),
        "GL F rep cv": ("$+0.128$ on Brennan (mean", "$-0.103$ on Brennan (mean"),
        "GL F rep rel": ("(split-half $-0.13$ with the baseline", "(split-half $+0.13$ with the baseline"),
        "GL I depth": ("with one it is $-0.25$, not determined", "with one it is $+0.25$, not determined"),
        "GL M counts": ("each determined on five segments and both on four", "each determined on eight segments and both on four"),
        "GL M range": ("similarity reliability is $0.30$--$0.91$", "similarity reliability is $0.30$--$0.99$"),
        "GL O gap": ("the class gap is $+15.6$ ranks with a per-epoch", "the class gap is $+16.5$ ranks with a per-epoch"),
        "GL fig3 caption": ("Brennan is the registered per-epoch-baseline value", "Brennan is the stored value"),
        "GL old row back": ("Brennan EEG, $47$, per-epoch baseline$^\\|$ & $-0.248$",
                            "Brennan EEG, $47$ listeners$^\\P$ & $+0.366$ & x \\\\ Brennan EEG, $47$, per-epoch baseline$^\\|$ & $-0.248$"),
        "L492 last_digit 2.5->2.6": (r"$2.5$ ($0.5$B--$72$B), $3$ and $3.5$", r"$2.6$ ($0.5$B--$72$B), $3$ and $3.5$"),
        "L499 digit_swap 70->07": (r"Pythia ladder ($70$M--$2.8$B), scored", r"Pythia ladder ($07$M--$2.8$B), scored"),
        "L516 sign_flip -12->+12": (r"a $10^{-12}$ guard in a correlation", r"a $10^{+12}$ guard in a correlation"),
        "L519 digit_swap 97->79 (first)": (r"harness read $97/97$ green", r"harness read $79/97$ green"),
        "L519 digit_swap 97->79 (second)": (r"harness read $97/97$ green", r"harness read $97/79$ green"),
        # ---- second round (seed-1 mutation sweep, 2026-10-05): one mutant per original miss ----
        "L43 sign_flip +0.78->-0.78": (
            r"a fully crossed fMRI cohort ($25$ narratives $\times$ $9$ listeners) transfers well ($+0.778$).",
            r"a fully crossed fMRI cohort ($25$ narratives $\times$ $9$ listeners) transfers well ($-0.778$)."),
        "L45 swap_sent 32->57": (
            r"a readout sweep extended from $32$ to $57$ checkpoints ($48$ base) spanning $2019$--$2026$.",
            r"a readout sweep extended from $57$ to $57$ checkpoints ($48$ base) spanning $2019$--$2026$."),
        "L45 swap_sent 57->48": (
            r"a readout sweep extended from $32$ to $57$ checkpoints ($48$ base) spanning $2019$--$2026$.",
            r"a readout sweep extended from $32$ to $48$ checkpoints ($48$ base) spanning $2019$--$2026$."),
        "L60 digit_swap 0.96->9.06": (
            r"ridge encoding \citep{huth2016natural} returns an equally reliable one ($0.96$).",
            r"ridge encoding \citep{huth2016natural} returns an equally reliable one ($9.06$)."),
        "F1 cap C4 negation dropped": ("(C4) also partialled out, the reversal is not determined, nor",
                                       "(C4) also partialled out, the reversal is determined, nor"),
        "F1 cap Brennan shown": ("Brennan EEG is omitted: its epochs", "Brennan EEG is shown: its epochs"),
        "L174 equation last_digit 1->0": (
            r"$D^{\mathrm b}_{ij} = 1-\operatorname{corr}(y_i,y_j)$ and $D^{(m)}_{ij} = 1-\langle x_i,x_j\rangle/(\|x_i\|\|x_j\|)$, the unit score is a partial Spearman",
            r"$D^{\mathrm b}_{ij} = 0-\operatorname{corr}(y_i,y_j)$ and $D^{(m)}_{ij} = 1-\langle x_i,x_j\rangle/(\|x_i\|\|x_j\|)$, the unit score is a partial Spearman"),
        "L222 equation last_digit 2->3": (
            r"the training-fold fitted values are $\sum_k\frac{s_k^2}{s_k^2+\alpha}\,u_k u_k^\top\tilde Y_{-u}$. \textbf{(iv)} If",
            r"the training-fold fitted values are $\sum_k\frac{s_k^3}{s_k^2+\alpha}\,u_k u_k^\top\tilde Y_{-u}$. \textbf{(iv)} If"),
        "L225 equation last_digit 1->2": (
            r"its participation ratio, the leading component carries at least a fraction $1/\mathrm{PR}$ of the squared Frobenius norm of the centered Gram matrix $\bar X\bar X^\top$.",
            r"its participation ratio, the leading component carries at least a fraction $2/\mathrm{PR}$ of the squared Frobenius norm of the centered Gram matrix $\bar X\bar X^\top$."),
        "L250 last_digit 27->26": (
            r"is English MEG ($10$ subjects in the tensor archive, $27$ in the raw rebuild for crossed and readout analyses)",
            r"is English MEG ($10$ subjects in the tensor archive, $26$ in the raw rebuild for crossed and readout analyses)"),
        "L314 last_digit 48->49": (
            r"concordance below the reliability ceiling; $48$ base: family-clustered interval below zero); the C4 row and the",
            r"concordance below the reliability ceiling; $49$ base: family-clustered interval below zero); the C4 row is a registered"),
        "L322 last_digit 4->3 (megmasc stories)": (
            r"MEG-MASC ($4$ stories $\times$ $27$)",
            r"MEG-MASC ($3$ stories $\times$ $27$)"),
        "L322 last_digit 27->28 (megmasc raw rebuild)": (
            r"MEG-MASC ($4$ stories $\times$ $27$)",
            r"MEG-MASC ($4$ stories $\times$ $28$)"),
        "L382 last_digit 32->31": (
            r"we score all $32$ models within each story; each story's trials span both recording sessions of",
            r"we score all $31$ models within each story; each story's trials span both recording sessions of"),
        "L442 last_digit 32->31": (
            r"On LibriBrain the two readouts rank the $32$ checkpoints in \emph{opposite} orders ($\rho=-0.573$).",
            r"On LibriBrain the two readouts rank the $31$ checkpoints in \emph{opposite} orders ($\rho=-0.573$)."),
        "L446 digit_swap 2019->2109": (
            r"the reversal also survives the $48$-checkpoint base set that spans $2019$--$2026$ (Table~\ref{tab:main})",
            r"the reversal also survives the $48$-checkpoint base set that spans $2109$--$2026$ (Table~\ref{tab:main})"),
        "L460 digit_swap 306->036": (
            r"scoring all $306$ channels or dropping C1--C3 leaves the concordance close to its headline value",
            r"scoring all $036$ channels or dropping C1--C3 leaves the concordance close to its headline value"),
        "L484 digit_swap 12->21 (of 12 sessions)": (
            r"does not explain the difference ($11$ of $12$ LibriBrain sessions reverse on their own, Appendix~\ref{app:depth})",
            r"does not explain the difference ($11$ of $21$ LibriBrain sessions reverse on their own, Appendix~\ref{app:depth})"),
        "L484 last_digit 11->12 (11 of)": (
            r"does not explain the difference ($11$ of $12$ LibriBrain sessions reverse on their own, Appendix~\ref{app:depth})",
            r"does not explain the difference ($12$ of $12$ LibriBrain sessions reverse on their own, Appendix~\ref{app:depth})"),
        "L497 digit_swap 0.5->5.0": (
            r"regularization-sensitive, while $72$B stays below $0.5$B (Appendix Table~\ref{tab:alpha-grid})",
            r"regularization-sensitive, while $72$B stays below $5.0$B (Appendix Table~\ref{tab:alpha-grid})"),
        "L505 structural digit_swap 80->08": (
            r"broadband reaches a higher peak alignment than $30$--$80$~Hz gamma in $206$ of $213$ recordings",
            r"broadband reaches a higher peak alignment than $30$--$08$~Hz gamma in $206$ of $213$ recordings"),
        "L505 last_digit 206->205": (
            r"broadband reaches a higher peak alignment than $30$--$80$~Hz gamma in $206$ of $213$ recordings",
            r"broadband reaches a higher peak alignment than $30$--$80$~Hz gamma in $205$ of $213$ recordings"),
        "L542 last_digit 2->3 (exponent)": (
            r"Raw best-cell alignment is $10^{-3}$--$10^{-2}$ in MEG and $\approx2\times10^{-2}$ in ECoG, and headline values select cells in-sample at a $6$--$85\%$ cost",
            r"Raw best-cell alignment is $10^{-3}$--$10^{-3}$ in MEG and $\approx2\times10^{-2}$ in ECoG, and headline values select cells in-sample at a $6$--$85\%$ cost"),
        "L542 digit_swap 10->01": (
            r"Raw best-cell alignment is $10^{-3}$--$10^{-2}$ in MEG and $\approx2\times10^{-2}$ in ECoG, and headline values select cells in-sample at a $6$--$85\%$ cost",
            r"Raw best-cell alignment is $01^{-3}$--$10^{-2}$ in MEG and $\approx2\times10^{-2}$ in ECoG, and headline values select cells in-sample at a $6$--$85\%$ cost"),
        "L543 last_digit 6->7": (
            r"Raw best-cell alignment is $10^{-3}$--$10^{-2}$ in MEG and $\approx2\times10^{-2}$ in ECoG, and headline values select cells in-sample at a $6$--$85\%$ cost",
            r"Raw best-cell alignment is $10^{-3}$--$10^{-2}$ in MEG and $\approx2\times10^{-2}$ in ECoG, and headline values select cells in-sample at a $7$--$85\%$ cost"),
        "L592 swap_sent 504->9": (
            r"Its validation recomputes all $504$ stored per-(model, session) values behind the LibriBrain cell to within $10^{-9}$ from cached neural dissimilarity",
            r"Its validation recomputes all $9$ stored per-(model, session) values behind the LibriBrain cell to within $10^{-9}$ from cached neural dissimilarity"),
        # ---- third round (seed-1 round-3 valid rerun, post-commit, 2026-10-05): the one residual miss ----
        "L339 structural digit_swap 95->59": (
            r"Readout concordance on & $\rho(\pi^{\mathrm{RSA}},\pi^{\mathrm{enc}})$ & 95\% interval or note",
            r"Readout concordance on & $\rho(\pi^{\mathrm{RSA}},\pi^{\mathrm{enc}})$ & 59\% interval or note"),
    }


def _null_edit() -> tuple[str, str]:
    # A wording change that touches none of the literal anchors above -- must NOT trip any pin.
    # 2026-10-10: re-anchored on the redrawn Figure 1 caption (the schematic caption it used is gone)
    return ("Rows change one argument of Eq.~\\eqref{eq:agree}.",
            "Rows each change one argument of Eq.~\\eqref{eq:agree}.")


def _receipt_mutants(results: Path):
    """(name, rel_path, json_path_tuple, new_value) -- applied to a tempdir copy of results/."""
    return [
        ("concordance_spearman -0.573 -> -0.999 (Group A)",
         "p4_encoding/concordance32_CORRECTED.json", ("concordance_spearman",), -0.999),
        ("spacefix q2.rho -0.59 -> -0.12 (Group B)",
         "p4_encoding/space_fix_20260925/aggregate/adr0006_registered_verdicts_spacefix.json",
         ("q2", "rho"), -0.12),
        ("selection_tax LibriBrain.n_units 36 -> 99 (Group D)",
         "selection_tax_multicohort.json", ("LibriBrain", "n_units"), 99),
        ("brennan K2 b.branch not determined -> survives (Figure 1 caption)",
         "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json", ("b", "branch"), "survives"),
        ("brennan K2 b.rho -0.248 -> -0.284 (Group L)",
         "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json", ("b", "rho"), -0.284),
        ("brennan K0 drift median 0.885 -> 0.5 (Group L)",
         "paperA_rebuttal_20261004/brennan_preproc_summary_20261009.json", ("K0", "a", "spearman_Db_dt_median"), 0.5),
        ("brennan K2 repeat cv seed1 0.054 -> 0.3 (Group L, cv seed mean)",
         "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json",
         ("repeat_k1_b", "brennan_p4_v2_k1_seed1.json", "arms", "cv", "concordance"), 0.3),
        ("brennan K2 segment-4 RSA CI lower -> +0.1 (Group L, counts)",
         "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json",
         ("cross_8seg_b", "brennan_p4_v2_segment-4_full.json", "rsa_rank_reliability_ci", 0), 0.1),
        ("brennan K2 b enc family verdict -> IDENTIFIABLE (Group L)",
         "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json",
         ("b", "rank_identifiability", "concordance32_brennan_k2_b|encoding [family-level]", "verdict"), "IDENTIFIABLE at this resolution."),
    ]


def selftest(tex_path: Path, results: Path) -> int:
    base_text = tex_path.read_text()
    # Pins anchor against whitespace-normalized text (a sentence can wrap physical lines in the
    # raw .tex source, e.g. "the $48$\nbase checkpoints..."), so mutants must be applied to the same
    # normalized form the pins actually see, not to the raw file text. norm() is idempotent on an
    # already-normalized string, so feeding normalized text into run() (which normalizes again
    # internally) is safe and is exactly what every pin above is matching against.
    base_norm = norm(base_text)
    baseline = run(base_norm, results)
    if baseline.fail:
        print(f"\nSELFTEST ABORTED: baseline (unmutated HEAD) already fails {len(baseline.fail)} pin(s): "
              f"{baseline.fail} -- fix the checker or the tex before trusting any mutant result.")
        return 1

    n_caught, n_missed, missed_names = 0, 0, []

    print("\n--- tex-side mutants (missed-mutation reproductions) ---")
    for name, (old, new) in _tex_mutants().items():
        if old not in base_norm:
            print(f"  ERROR   {name:46s} anchor text not found verbatim in normalized tex (drifted?) -- old={old!r}")
            n_missed += 1
            missed_names.append(name)
            continue
        mutated = base_norm.replace(old, new, 1)
        P = run(mutated, results)
        caught = bool(P.fail)
        print(f"  {'caught ' if caught else 'MISSED '} {name:46s}" + (f" -> {P.fail}" if caught else ""))
        if caught:
            n_caught += 1
        else:
            n_missed += 1
            missed_names.append(name)

    print("\n--- null-edit control (must NOT be caught) ---")
    old, new = _null_edit()
    assert old in base_norm, "null-edit anchor drifted"
    null_mutated = base_norm.replace(old, new, 1)
    P_null = run(null_mutated, results)
    null_ok = not P_null.fail
    print(f"  {'ok (no false positive)' if null_ok else 'FALSE POSITIVE: ' + str(P_null.fail)}")

    print("\n--- receipt-side mutants (tempdir copy of results/) ---")
    n_rcaught, n_rmissed = 0, 0
    for name, rel, keypath, newval in _receipt_mutants(results):
        tmpdir = Path(tempfile.mkdtemp(prefix="paperA_selftest_"))
        try:
            shutil.copytree(results, tmpdir / "results")
            target = tmpdir / "results" / rel
            j = json.loads(target.read_text())
            cur = j
            for k in keypath[:-1]:
                cur = cur[k]
            cur[keypath[-1]] = newval
            target.write_text(json.dumps(j))
            P = run(base_text, tmpdir / "results")
            caught = bool(P.fail)
            print(f"  {'caught ' if caught else 'MISSED '} {name:46s}" + (f" -> {P.fail}" if caught else ""))
            if caught:
                n_rcaught += 1
            else:
                n_rmissed += 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # restored (pristine) results/ must still pass -- paranoia check that the copy/mutate helper
    # above did not leave any state behind that could affect the real directory.
    restored = run(base_text, results)
    restored_ok = not restored.fail

    print(f"\ntex-side mutants {len(_tex_mutants())}, caught {n_caught}, missed {n_missed} {missed_names if missed_names else ''}")
    print(f"receipt-side mutants {len(_receipt_mutants(results))}, caught {n_rcaught}, missed {n_rmissed}")
    print(f"null-edit control: {'ok' if null_ok else 'FALSE POSITIVE'}")
    print(f"real results/ unaffected after selftest: {'ok' if restored_ok else 'CORRUPTED'}")

    ok = (n_missed == 0) and (n_rmissed == 0) and null_ok and restored_ok
    print(f"\nSELFTEST={'ok' if ok else 'BLIND'}")
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default=str(DEFAULT_TEX))
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    tex_path, results = Path(a.tex), Path(a.results)

    if a.selftest:
        return selftest(tex_path, results)

    P = run(tex_path.read_text(), results)
    print(f"\n{'FAILED' if P.fail else 'PASS'}: {P.n - len(P.fail)}/{P.n} pins hold")
    if P.fail:
        print("failed pin ids:", P.fail)
    return 1 if P.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
