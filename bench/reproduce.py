"""bench/reproduce.py — regenerate and verify every headline number in BrainLLM-Bench.

The Datasets & Benchmarks contract is that the released artifact reproduces the paper.
This script recomputes each headline claim directly from the released ``results/`` tree
and compares it against the value printed in the paper, emitting a reproduction report.

Usage:
    python -m bench.reproduce                  # verify all claims, print report
    python -m bench.reproduce --emit-tables    # also write regenerated tables to bench/regenerated/
    python -m bench.reproduce --json report.json

Exit status is non-zero if any claim fails to reproduce, so this doubles as a CI check.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
# results/ lives next to the paper; fall back to a repo-root copy
for _cand in (PROJ / "paper4_temporalalign" / "results", PROJ / "results"):
    if _cand.is_dir():
        RESULTS = _cand
        break
else:  # pragma: no cover
    sys.exit("could not locate a results/ tree")

OUT = PROJ / "bench" / "regenerated"


def _json(rel):
    return json.load(open(RESULTS / rel))


def _tsv(rel):
    return pd.read_csv(RESULTS / rel, sep="\t")


# ---------------------------------------------------------------- claim checks
# Each check returns (recomputed_value, paper_value, tolerance, source).
# Paper values are the numbers as printed in paperA_benchmark.tex, except the provenance pins in
# SUPERSEDED (pre-correction files) and the companion boundary-analysis claims (ids "B.*").


def c_concordance_headline():
    """The paper now prints the corrected value; the published artifact is checked separately."""
    d = _json("p4_encoding/concordance32_CORRECTED.json")
    return d["concordance_spearman"], -0.573, 0.005, "p4_encoding/concordance32_CORRECTED.json"


def c_concordance_ci_low():
    d = _json("p4_encoding/concordance32_CORRECTED.json")
    return d["concordance_ci95"][0], -0.77, 0.01, "p4_encoding/concordance32_CORRECTED.json"


def c_concordance_ci_high():
    d = _json("p4_encoding/concordance32_CORRECTED.json")
    return d["concordance_ci95"][1], -0.28, 0.01, "p4_encoding/concordance32_CORRECTED.json"


def c_family_clustered_ci():
    """Headline row of the family-clustered bootstrap: CI [-0.86, -0.36]."""
    txt = (RESULTS / "p4_encoding/family_clustered_ci_CORRECTED.txt").read_text()
    m = re.search(r"headline.*?\[([-+0-9.]+),\s*([-+0-9.]+)\]", txt)
    if not m:
        raise ValueError("could not parse headline family-clustered CI")
    lo = float(m.group(1))
    return lo, -0.79, 0.01, "p4_encoding/family_clustered_ci_CORRECTED.txt"


def c_n_families():
    txt = (RESULTS / "p4_encoding/family_clustered_ci_CORRECTED.txt").read_text()
    m = re.search(r"n_families=(\d+)", txt)
    return int(m.group(1)), 12, 0, "p4_encoding/family_clustered_ci.txt"


def _depth():
    return _json("p4_encoding/concordance_depth_curve.json")


def c_depth_k1():
    """Concordance at single-session depth, under the corrected encoding arm."""
    d = _json("p4_encoding/concordance_depth_curve_CORRECTED.json")
    return d["curve"]["1"]["mean"], -0.393, 0.01, \
        "p4_encoding/concordance_depth_curve_CORRECTED.json"



def c_depth_sessions_negative():
    """Independent per-session replications that reverse, no averaging."""
    return _depth()["independent_per_session"]["n_negative"], 11, 0, \
        "p4_encoding/concordance_depth_curve.json"


def c_depth_session_p():
    return _depth()["independent_per_session"]["wilcoxon_p"] * 1e3, 0.977, 0.02, \
        "p4_encoding/concordance_depth_curve.json"


def c_rawrsa_symmetry():
    """Reversal survives making both readouts un-partialled (rules out asymmetric confound control).

    Repointed to the corrected value: the stored concordance32_RAWRSA.json (-0.676) carries the
    corrupted encoding arm; the corrected pairing lives in correction_2026_08_derived.json.
    """
    d = _json("p4_encoding/correction_2026_08_derived.json")
    return d["variants"]["raw_rsa_no_partialling"]["corrected"], -0.578, 0.001, \
        "p4_encoding/correction_2026_08_derived.json"


def c_invariance_chantime():
    d = _json("p4_encoding/concordance_repr_chantime_CORRECTED.json")
    return d["concordance_spearman"], -0.531, 0.01, "p4_encoding/concordance_repr_chantime_CORRECTED.json"


def c_invariance_timeavg():
    d = _json("p4_encoding/concordance_repr_timeavg_CORRECTED.json")
    return d["concordance_spearman"], -0.593, 0.01, "p4_encoding/concordance_repr_timeavg_CORRECTED.json"


def c_invariance_allch():
    d = _json("p4_encoding/concordance32_lb_allch_CORRECTED.json")
    return d["concordance_spearman"], -0.651, 0.01, "p4_encoding/concordance32_lb_allch_CORRECTED.json"


def c_stdfeat_centered():
    """Corrected pairing: stored per-model RSA under mean-centred features x corrected encoding."""
    d = _json("p4_encoding/correction_2026_08_stdfeat.json")
    return d["variants"]["mean-centred"]["corrected"], -0.479, 0.001, \
        "p4_encoding/correction_2026_08_stdfeat.json"


def c_stdfeat_standardized():
    """Corrected pairing: per-dimension standardized RSA x corrected encoding."""
    d = _json("p4_encoding/correction_2026_08_stdfeat.json")
    return d["variants"]["standardized"]["corrected"], -0.432, 0.001, \
        "p4_encoding/correction_2026_08_stdfeat.json"


def c_brennan_concordance():
    """The tex prints +0.342, which is the 47-subject v2 merge, not the 19-subject v1 (+0.26)."""
    d = _json("p4_encoding/concordance32_brennan_v2.json")
    return d["concordance_spearman"], +0.342, 0.001, "p4_encoding/concordance32_brennan_v2.json"


def c_ceiling_gamma():
    d = _json("p2_perband_ceiling.json")["per_band"]
    return d["gamma"]["ceiling_SB_raw"], -0.002, 0.001, "p2_perband_ceiling.json"


def c_ceiling_broadband():
    d = _json("p2_perband_ceiling.json")["per_band"]
    return d["broadband"]["ceiling_SB_raw"], 0.135, 0.001, "p2_perband_ceiling.json"


def c_ceiling_delta():
    d = _json("p2_perband_ceiling.json")["per_band"]
    return d["delta"]["ceiling_SB_raw"], 0.244, 0.001, "p2_perband_ceiling.json"


def c_broadband_partial():
    d = _json("p2_perband_ceiling.json")["per_band"]
    return d["broadband"]["alignment_partial"], -0.0036, 0.0002, "p2_perband_ceiling.json"


def c_per_unit_bb_gt_gamma():
    """595 of 639 units show broadband > gamma (93.1%)."""
    pu = _tsv("per_unit_bb_vs_gamma.tsv")
    return int((pu["bb_minus_gamma"] > 0).sum()), 595, 0, "per_unit_bb_vs_gamma.tsv"


def c_per_unit_total():
    pu = _tsv("per_unit_bb_vs_gamma.tsv")
    return len(pu), 639, 0, "per_unit_bb_vs_gamma.tsv"


def _bb_by_cohort():
    d = _tsv("per_unit_bb_vs_gamma.tsv")
    return {c: ((g["bb_minus_gamma"] > 0).sum(), len(g)) for c, g in d.groupby("cohort")}


def c_bb_ecog():
    """ECoG is the consequential cell: it is where the field filters to high gamma."""
    k, n = _bb_by_cohort()["ECoG-ext"]
    return int(k), 27, 0, "per_unit_bb_vs_gamma.tsv"


def c_bb_libribrain():
    k, n = _bb_by_cohort()["LibriBrain"]
    return int(k), 108, 0, "per_unit_bb_vs_gamma.tsv"


def c_bb_cohorts_unanimous():
    """Cohorts in which every unit favours broadband."""
    return sum(1 for k, n in _bb_by_cohort().values() if k == n), 2, 0, \
        "per_unit_bb_vs_gamma.tsv"


def c_bb_all_cohorts_majority():
    """All four cohorts must favour broadband in a majority of units."""
    return sum(1 for k, n in _bb_by_cohort().values() if k > n / 2), 4, 0, \
        "per_unit_bb_vs_gamma.tsv"


def c_leaderboard_bert_libribrain():
    cc = _tsv("cross_cohort_summary.tsv")
    row = cc[(cc.cohort == "LibriBrain") & (cc.llm == "BERT-base")]
    return float(row.rho_best.iloc[0]), 0.0131, 0.0001, "cross_cohort_summary.tsv"


def c_leaderboard_gpt2med_libribrain():
    cc = _tsv("cross_cohort_summary.tsv")
    row = cc[(cc.cohort == "LibriBrain") & (cc.llm == "GPT-2-medium")]
    return float(row.rho_best.iloc[0]), 0.0069, 0.0001, "cross_cohort_summary.tsv"


def c_derived_zou_cells():
    """Zou boundary claim: per-cohort cells recomputed from the analysis, not read from a table.

    R1 2026-10-07 (paperA revision A5): re-scoped, not deleted. The Brennan cell reads the 49-listener
    archive (boundary_brennan_persubject_summary.tsv), which can no longer be recomputed; the paper now
    labels it archived (Fig. 9, App. L "Brennan lineage"). The assertion therefore pins (a) that the stored
    Brennan cell is the archive's (n_total 49 in its source), so a silent substrate swap fails, and (b) that
    the same derive_scorecard rule (primary model mean > 0 and Wilcoxon p < 0.05 -> supports) re-applied to
    the recomputable 47-listener rebuild (brennan47_derived_20261003.json, arm BERT/L1) also reads
    "supports" -- the statement the revised caption makes. MEG-MASC / MEG-SCANS cells unchanged.
    """
    d = _json("scorecard/scorecard_derived.json")["claims"]["Zou et al."]["per_cohort"]
    arch = _tsv("exp_boundary_conditional/boundary_brennan_persubject_summary.tsv").set_index("llm")
    reb = _json("exp_boundary_conditional/brennan47_derived_20261003.json")
    b1 = reb["arms"]["BERT/L1"]
    archive_is_source = int(arch.loc["BERT", "n_total"]) == 49 and reb["n_subjects"] == 47
    rebuild_rule = "supports" if (b1["mean"] > 0 and b1["wilcoxon_p"] < 0.05) else "not supports"
    ok = (d.get("Brennan") == "supports" and archive_is_source and rebuild_rule == "supports"
          and d.get("MEG-MASC") == "refines" and d.get("MEG-SCANS") == "contradicts")
    return int(ok), 1, 0, "scorecard/scorecard_derived.json + exp_boundary_conditional/brennan47_derived_20261003.json"


def c_derived_matches_table():
    """Every derivable verdict must equal the hand-written table's verdict."""
    der = _json("scorecard/scorecard_derived.json")["claims"]
    hand = {r["paper"]: r["verdict"] for r in _tsv("scorecard/scorecard.tsv").to_dict("records")}
    bad = [k for k, v in der.items()
           if v["verdict"] != "NOT_DERIVABLE" and hand.get(k) != v["verdict"]]
    return len(bad), 0, 0, "scorecard/scorecard_derived.json"


def c_reepoch_words():
    """Trial selection recovered exactly when re-epoching an archived subject from raw."""
    d = _json("megmasc_meta/reepoch_validation.json")["trial_selection"]
    return d["recovered_words"], d["archive_rows"], 0, "megmasc_meta/reepoch_validation.json"


def c_reepoch_corr():
    d = _json("megmasc_meta/reepoch_validation.json")["per_epoch_correlation"]
    return d["median"], 0.9999, 0.0002, "megmasc_meta/reepoch_validation.json"


def c_reepoch_worst():
    """No epoch falls below r = 0.99, so the match is not carried by an average."""
    d = _json("megmasc_meta/reepoch_validation.json")["per_epoch_correlation"]
    return d["n_below_0_99"], 0, 0, "megmasc_meta/reepoch_validation.json"


def _tax_curve():
    d = _json("selection_tax_multicohort.json")["_unit_count_control"]
    return {c["n_subjects"]: c["tax_pct_mean"] for c in d["curve"]}, d


def c_tax_n23():
    """Selection tax with all 23 MEG-SCANS subjects."""
    c, _ = _tax_curve()
    return c[23], 13.3, 0.2, "selection_tax_multicohort.json"


def c_tax_n9():
    """Same cohort subsampled to nine subjects: unit count alone raises the tax."""
    c, _ = _tax_curve()
    return c[9], 34.4, 1.5, "selection_tax_multicohort.json"


def c_tax_count_rho():
    _, d = _tax_curve()
    return d["spearman_rho"], -0.96, 0.02, "selection_tax_multicohort.json"


def c_protocol_selftest():
    """The protocol's statistic against known answers. Runs anywhere, unlike --validate."""
    from . import protocol
    return int(bool(protocol.selftest())), 1, 0, "bench/protocol.py --selftest"


def c_protocol_validation():
    """The offline check: every stored per-(model, subject) value the port recomputed."""
    d = _json("scorecard/protocol_validation.json")
    return d["values_matched"], d["values_checked"], 0, "scorecard/protocol_validation.json"


def c_protocol_mismatches():
    d = _json("scorecard/protocol_validation.json")
    return d["mismatches"], 0, 0, "scorecard/protocol_validation.json"


def c_masc_subset_age_p():
    """Largest imbalance between the analyzed 10 and the other 17, before correction."""
    d = _json("megmasc_meta/subset_bias.json")
    return d["tests"]["age"]["mannwhitney_p"], 0.042, 0.002, "megmasc_meta/subset_bias.json"


def c_masc_subset_any_sig():
    """After Holm over the three tests, nothing separates the analyzed subset."""
    d = _json("megmasc_meta/subset_bias.json")
    return int(d["any_significant_after_holm"]), 0, 0, "megmasc_meta/subset_bias.json"


def c_masc_recordings():
    d = _json("megmasc_meta/subset_bias.json")
    return d["recordings_analyzed"], 76, 0, "megmasc_meta/subset_bias.json"


def _toneva_cov():
    d = _json("scorecard/scorecard_derived.json")
    return json.loads(d["claims"]["Toneva & Wehbe"]["evidence"])["_coverage"]


def c_toneva_strict():
    """Cells where middle layers strictly beat both shallower and deeper ones."""
    c = _toneva_cov()
    return c["strict_supports"], 2, 0, "scorecard/scorecard_derived.json"


def c_toneva_weak():
    """Cells where the weaker mid-to-shallow-over-deep reading holds."""
    c = _toneva_cov()
    return c["weak_supports"], 6, 0, "scorecard/scorecard_derived.json"


def c_toneva_cells():
    c = _toneva_cov()
    return c["cells"], 8, 0, "scorecard/scorecard_derived.json"


def c_derived_coverage():
    """How many of the ten verdicts are currently computed rather than asserted."""
    d = _json("scorecard/scorecard_derived.json")
    return d["n_derivable"], 9, 0, "scorecard/scorecard_derived.json"


def c_scorecard_replicate():
    sc = _tsv("scorecard/scorecard.tsv")
    return collections.Counter(sc.verdict)["replicate"], 1, 0, "scorecard/scorecard.tsv"


def c_scorecard_refine():
    sc = _tsv("scorecard/scorecard.tsv")
    return collections.Counter(sc.verdict)["refine"], 8, 0, "scorecard/scorecard.tsv"


def c_scorecard_contradict():
    sc = _tsv("scorecard/scorecard.tsv")
    return collections.Counter(sc.verdict)["contradict"], 1, 0, "scorecard/scorecard.tsv"


def c_width_matched_gap():
    """Class gap survives holding feature width fixed (rules out a wide-vs-narrow account)."""
    # Repointed 2026-09-24 from the superseded class_gap_width_control.json (16.3, p=0.0098,
    # corrupted encoding arm) to the corrected receipt the paper prints (+14.1, two-sided p=0.018).
    d = _json("p4_encoding/class_gap_width_control_CORRECTED.json")["pooled_width_matched"]
    return d["gap"], 14.1, 0.05, "p4_encoding/class_gap_width_control_CORRECTED.json"


def c_width_matched_p():
    d = _json("p4_encoding/class_gap_width_control_CORRECTED.json")["pooled_width_matched"]
    return d["p_family_two_sided"], 0.018, 0.0005, \
        "p4_encoding/class_gap_width_control_CORRECTED.json"


def c_decomp_within_class():
    """Repointed to the recomputation from the corrected encoding arm (2026-08 audit)."""
    d = _json("p4_encoding/concordance_decomposition_libribrain_CORRECTED.json")
    return d["within_class_centred_rho"], -0.252, 0.001, \
        "concordance_decomposition_libribrain_CORRECTED.json"


def c_decomp_class_gap_corr():
    """Causal-vs-non-causal ENC-RSA rank gap under the corrected encoding arm."""
    d = _json("p4_encoding/concordance_decomposition_libribrain_CORRECTED.json")
    return d["class_gap_causal_minus_other"], 25.65, 0.01, \
        "concordance_decomposition_libribrain_CORRECTED.json"


def c_decomp_gap_p_family_corr():
    """Family-clustered permutation p for the corrected gap (10000 perms, two-sided)."""
    d = _json("p4_encoding/concordance_decomposition_libribrain_CORRECTED.json")
    return d["gap_p_family_clustered_two_sided"], 0.0051, 0.001, \
        "concordance_decomposition_libribrain_CORRECTED.json"


def c_decomp_class_gap():
    d = _json("p4_encoding/concordance_decomposition_libribrain.json")
    return d["class_gap_causal_minus_other"], 27.9, 0.1, "concordance_decomposition_libribrain.json"


def c_decomp_gap_p_family():
    d = _json("p4_encoding/concordance_decomposition_libribrain.json")
    return d["gap_p_family_clustered"], 0.00085, 0.0002, "concordance_decomposition_libribrain.json"


def c_mech_causal_rho():
    d = _json("p4_encoding/pr_gap_mechanism.json")
    return d["per_class"]["causal"]["rho"], -0.75, 0.01, "p4_encoding/pr_gap_mechanism.json"


def c_mech_causal_ci_excludes_zero():
    """Within causal, the family-clustered CI upper bound must be below zero."""
    d = _json("p4_encoding/pr_gap_mechanism.json")
    return d["per_class"]["causal"]["family_ci95"][1], -0.05, 0.02, "p4_encoding/pr_gap_mechanism.json"


def c_mech_all_ci_crosses_zero():
    """Pooled over all classes, the same CI crosses zero -- the honest limit of the mechanism."""
    d = _json("p4_encoding/pr_gap_mechanism.json")
    return d["all"]["family_ci95"][1], 0.069, 0.01, "p4_encoding/pr_gap_mechanism.json"


# ---- Paper B (constituent boundary) -------------------------------------------------
_BND = "exp_boundary_conditional"


def _bnd_persub(rel, llm, col="delta"):
    d = _tsv(f"{_BND}/{rel}")
    g = d[d["llm"] == llm]
    return g[col].to_numpy()



def c_b_hz_sib_page_p():
    # paper prints the joint bound "<= 2.8e-5" over both configurations; sibling is the max
    r = _json(f"{_BND}/horizon_dose_sibling.json")
    return float(r["page_trend"]["p"]) * 1e5, 2.8, 0.08, f"{_BND}/horizon_dose_sibling.json"


def c_b_hz_sib_rho():
    r = _json(f"{_BND}/horizon_dose_sibling.json")
    return float(r["spearman_k_retention"]["rho"]), 0.66, 0.005, f"{_BND}/horizon_dose_sibling.json"


def c_b_hz_sib_ret_k0():
    r = _json(f"{_BND}/horizon_dose_sibling.json")
    return float(r["retention_by_level"]["k0"]), 0.29, 0.005, f"{_BND}/horizon_dose_sibling.json"


def c_b_hz_sib_ret_k8():
    r = _json(f"{_BND}/horizon_dose_sibling.json")
    return float(r["retention_by_level"]["k8"]), 1.54, 0.005, f"{_BND}/horizon_dose_sibling.json"


def c_b_hz_head_page_p():
    r = _json(f"{_BND}/horizon_dose_headline.json")
    return float(r["page_trend"]["p"]) * 1e6, 9.5, 0.1, f"{_BND}/horizon_dose_headline.json"


def c_b_hz_head_rho():
    r = _json(f"{_BND}/horizon_dose_headline.json")
    return float(r["spearman_k_retention"]["rho"]), 0.60, 0.005, f"{_BND}/horizon_dose_headline.json"


def c_b_hz_head_ret_k0():
    r = _json(f"{_BND}/horizon_dose_headline.json")
    return float(r["retention_by_level"]["k0"]), 0.41, 0.005, f"{_BND}/horizon_dose_headline.json"


def c_b_surp_d():
    r = _json(f"{_BND}/surprisal_boundary_brennan.json")
    return float(r["cohen_d"]), -0.23, 0.005, f"{_BND}/surprisal_boundary_brennan.json"


def c_b_surp_dhi():
    r = _json(f"{_BND}/surprisal_boundary_brennan.json")
    return float(r["d_upper_95_one_sided"]), 0.014, 0.002, f"{_BND}/surprisal_boundary_brennan.json"


def c_b_surp_pos():
    r = _json(f"{_BND}/surprisal_boundary_brennan.json")
    return int(r["n_positive"]), 17, 0, f"{_BND}/surprisal_boundary_brennan.json"


def c_b_surp_p():
    r = _json(f"{_BND}/surprisal_boundary_brennan.json")
    return float(r["wilcoxon_p"]), 0.11, 0.005, f"{_BND}/surprisal_boundary_brennan.json"


def c_b_lm_page_p():
    r = _json(f"{_BND}/horizon_lengthmatch_sibling.json")
    return float(r["page_trend_scrambled"]["p"]), 0.91, 0.005, f"{_BND}/horizon_lengthmatch_sibling.json"


def c_b_lm_k2():
    r = _json(f"{_BND}/horizon_lengthmatch_sibling.json")
    return float(r["paired_true_vs_scrambled"]["k2"]["mean_true_minus_scrambled"]) * 1e3, 84.0, 0.5, \
        f"{_BND}/horizon_lengthmatch_sibling.json"


def c_b_lm_k8():
    r = _json(f"{_BND}/horizon_lengthmatch_sibling.json")
    return float(r["paired_true_vs_scrambled"]["k8"]["mean_true_minus_scrambled"]) * 1e3, 69.0, 0.5, \
        f"{_BND}/horizon_lengthmatch_sibling.json"


_ADR6 = "p4_encoding/adr0006_registered_verdicts.json"


def c_a_q1_25_enc_rho():
    r = _json(_ADR6)
    return float(r["q1_qwen2.5"]["enc"]["rho"]), -0.89, 0.005, _ADR6


def c_a_q1_25_rsa_rho():
    r = _json(_ADR6)
    return float(r["q1_qwen2.5"]["rsa"]["rho"]), 0.61, 0.005, _ADR6


def c_a_q1_25_top_p():
    r = _json(_ADR6)
    return float(r["q1_qwen2.5"]["enc"]["top_minus_bottom_p"]), 0.0005, 0.0001, _ADR6


def c_a_q1_3_enc_rho():
    r = _json(_ADR6)
    return float(r["q1_qwen3"]["enc"]["rho"]), 0.10, 0.005, _ADR6


def c_a_q1_35_rsa_rho():
    r = _json(_ADR6)
    return float(r["q1_qwen3.5"]["rsa"]["rho"]), -1.00, 0.005, _ADR6


def c_a_q1_35_rsa_ci_hi():
    r = _json(_ADR6)
    return float(r["q1_qwen3.5"]["rsa"]["ci95_session_clustered"][1]), -0.8, 0.005, _ADR6


def c_a_q2_rho():
    r = _json(_ADR6)
    return float(r["q2"]["rho"]), -0.58, 0.005, _ADR6


def c_a_q2_ci_hi():
    r = _json(_ADR6)
    return float(r["q2"]["ci95_family_clustered"][1]), -0.14, 0.005, _ADR6


def c_a_q2_nbase():
    r = _json(_ADR6)
    return int(r["q2"]["n_base"]), 48, 0, _ADR6


def c_a_alpha_edge_72b():
    r = _json("p4_encoding/alpha_probe.json")
    return int(r["chosen_alpha_counts"]["Qwen/Qwen2.5-72B"]["1000000.0"]), 11, 0, \
        "p4_encoding/alpha_probe.json"


def c_a_alpha_edge_05b():
    r = _json("p4_encoding/alpha_probe.json")
    return int(r["chosen_alpha_counts"]["Qwen/Qwen2.5-0.5B"]["1000000.0"]), 11, 0, \
        "p4_encoding/alpha_probe.json"


def c_a_q3_rsa_count():
    r = _json("p4_encoding/adr0006_registered_verdicts_v2.json")
    return int(r["q3_direction_count_rsa"].split("/")[0]), 3, 0, \
        "p4_encoding/adr0006_registered_verdicts_v2.json"


def c_a_q3_enc_count():
    r = _json("p4_encoding/adr0006_registered_verdicts_v2.json")
    return int(r["q3_direction_count_enc"].split("/")[0]), 4, 0, \
        "p4_encoding/adr0006_registered_verdicts_v2.json"


_ADR10 = "p4_encoding/adr0010_q1_masc_verdicts.json"


def c_a_q1m_enc_rho():
    r = _json(_ADR10)
    return float(r["q1_masc"]["enc"]["rho"]), 0.29, 0.005, _ADR10


def c_a_q1m_enc_ci_lo():
    r = _json(_ADR10)
    return float(r["q1_masc"]["enc"]["ci95_subject_clustered"][0]), -0.43, 0.005, _ADR10


def c_a_q1m_enc_ci_hi():
    r = _json(_ADR10)
    return float(r["q1_masc"]["enc"]["ci95_subject_clustered"][1]), 0.64, 0.005, _ADR10


def c_a_q1m_enc_pair_p():
    r = _json(_ADR10)
    return float(r["q1_masc"]["enc"]["top_minus_bottom_p"]), 0.64, 0.005, _ADR10


def c_a_q1m_enc_pair_diff():
    r = _json(_ADR10)
    return float(r["q1_masc"]["enc"]["top_minus_bottom_mean"]) * 1e3, -1.1, 0.05, _ADR10


def c_a_q1m_enc_pair_pos():
    r = _json(_ADR10)
    return int(r["q1_masc"]["enc"]["n_positive_top_minus_bottom"]), 12, 0, _ADR10


def c_a_q1m_enc_gate_lo():
    r = _json(_ADR10)
    return float(r["f3_gate"]["enc"]["ci90"][0]), 0.40, 0.005, _ADR10


def c_a_q1m_enc_sb():
    r = _json(_ADR10)
    return float(r["f3_gate"]["enc"]["split_half"]), 0.81, 0.005, _ADR10


def c_a_q1m_rsa_sb():
    r = _json(_ADR10)
    return float(r["f3_gate"]["rsa"]["split_half"]), 0.76, 0.005, _ADR10


def c_a_q1m_rsa_gate_lo():
    r = _json(_ADR10)
    return float(r["f3_gate"]["rsa"]["ci90"][0]), 0.00, 0.005, _ADR10


def c_b_floor_sb():
    r = _json(f"{_BND}/brennan_rdm_floor.json")
    return float(r["sb_median"]), 0.69, 0.005, f"{_BND}/brennan_rdm_floor.json"


def c_b_floor_ci_lo():
    r = _json(f"{_BND}/brennan_rdm_floor.json")
    return float(r["sb_ci90"][0]), 0.48, 0.005, f"{_BND}/brennan_rdm_floor.json"


def c_b_floor_ci_hi():
    r = _json(f"{_BND}/brennan_rdm_floor.json")
    return float(r["sb_ci90"][1]), 0.83, 0.005, f"{_BND}/brennan_rdm_floor.json"


def c_b_surp_masc_d():
    r = _json(f"{_BND}/surprisal_boundary_masc27.json")
    return float(r["cohen_d"]), 0.21, 0.005, f"{_BND}/surprisal_boundary_masc27.json"


def c_b_surp_masc_p():
    r = _json(f"{_BND}/surprisal_boundary_masc27.json")
    return float(r["wilcoxon_p"]), 0.46, 0.01, f"{_BND}/surprisal_boundary_masc27.json"

def _b47_bert_l1():
    t = pd.read_csv(RESULTS / "exp_boundary_conditional/brennan47_offset_reextract.tsv", sep="\t")
    return t[(t["model"] == "BERT") & (t["arm"] == "L1")]["delta"].to_numpy(float)


_B47 = f"{_BND}/brennan47_offset_reextract.tsv"


def c_b_brennan47_pos():
    return int((_b47_bert_l1() > 0).sum()), 36, 0, _B47


def c_b_brennan47_n():
    return len(_b47_bert_l1()), 47, 0, _B47


def c_b_brennan47_mean():
    return float(_b47_bert_l1().mean()) * 1e3, 31.3, 0.05, _B47


def c_b_brennan47_p():
    from scipy.stats import wilcoxon
    return float(wilcoxon(_b47_bert_l1()).pvalue) * 1e5, 1.3, 0.05, _B47


def c_b_brennan47_between_p():
    d = _json(f"{_BND}/brennan47_derived_20261003.json")
    return d["between_cohorts_BERT_L1_vs_masc27"]["floor_passing_24"]["p_two_sided"] * 1e4, 3.8, 0.05, f"{_BND}/brennan47_derived_20261003.json"


def c_b_brennan_pos():
    v = _bnd_persub("boundary_brennan_persubject.tsv", "BERT")
    return int((v > 0).sum()), 38, 0, f"{_BND}/boundary_brennan_persubject.tsv"


def c_b_brennan_n():
    v = _bnd_persub("boundary_brennan_persubject.tsv", "BERT")
    return len(v), 49, 0, f"{_BND}/boundary_brennan_persubject.tsv"


def c_b_brennan_mean():
    v = _bnd_persub("boundary_brennan_persubject.tsv", "BERT")
    return float(v.mean()) * 1e3, 10.6, 0.1, f"{_BND}/boundary_brennan_persubject.tsv"


def c_b_brennan_p():
    d = _json(f"{_BND}/cohort_comparison_brennan_vs_masc.json")
    return d["brennan"]["wilcoxon_p"] * 1e4, 2.4, 0.1, f"{_BND}/cohort_comparison_brennan_vs_masc.json"


def c_b_masc_pos():
    v = _bnd_persub("boundary_megmasc_persubject_matched.tsv", "BERT")
    return int((v > 0).sum()), 3, 0, f"{_BND}/boundary_megmasc_persubject_matched.tsv"


def c_b_between_cohort_p():
    d = _json(f"{_BND}/cohort_comparison_brennan_vs_masc.json")
    return d["between_cohort"]["p_two_sided"], 0.008, 0.001, f"{_BND}/cohort_comparison_brennan_vs_masc.json"


def c_b_masc_power():
    d = _json(f"{_BND}/cohort_comparison_brennan_vs_masc.json")
    return d["power_at_brennan_d"]["n10_alpha0.05"]["power"], 0.33, 0.02, f"{_BND}/cohort_comparison_brennan_vs_masc.json"


_SENS = f"{_BND}/megmasc_sensitivity.json"


def _masc27(arm="primary_floor_passing"):
    return _json("exp_boundary_conditional/boundary_masc27.json")["per_model"]["BERT"][arm]


def c_masc27_n():
    return _masc27()["n"], 24, 0, "exp_boundary_conditional/boundary_masc27.json"


def c_masc27_upper():
    """The pre-registered primary statistic at full cohort size."""
    return _masc27()["PRIMARY_upper_95"] * 1e3, 3.97, 0.05, \
        "exp_boundary_conditional/boundary_masc27.json"


def c_masc27_excl_p():
    """Non-inferiority p, x1e6, floor-passing primary sample."""
    return _masc27()["PRIMARY_noninferiority_p"] * 1e6, 7.27, 0.2, \
        "exp_boundary_conditional/boundary_masc27.json"


def c_masc27_all_excl_p():
    """Paired all-subjects report: the floor must not be load-bearing."""
    return _masc27("all_subjects")["PRIMARY_noninferiority_p"] * 1e6, 2.34, 0.2, \
        "exp_boundary_conditional/boundary_masc27.json"


def c_masc27_both_exclude():
    d = _json("exp_boundary_conditional/boundary_masc27.json")["per_model"]["BERT"]
    n = sum(int(d[a]["PRIMARY_excludes_brennan_effect"])
            for a in ("primary_floor_passing", "all_subjects"))
    return n, 2, 0, "exp_boundary_conditional/boundary_masc27.json"


def c_xcohort27_p():
    """Between-cohort test at full MEG-MASC size, primary (floor-passing) arm."""
    d = _json("exp_boundary_conditional/cohort_comparison_n27.json")
    return d["between_primary"]["p_two_sided"] * 1e3, 2.31, 0.05, \
        "exp_boundary_conditional/cohort_comparison_n27.json"


def c_xcohort27_level_p():
    """Absolute alignment levels must stay indistinguishable: the null is about the contrast."""
    d = _json("exp_boundary_conditional/cohort_comparison_n27.json")
    return d["alignment_level"]["between_p"], 0.447, 0.01, \
        "exp_boundary_conditional/cohort_comparison_n27.json"


def c_b_masc_power_raw():
    """Power against Brennan's RAW effect, the figure the exclusion claim rests on."""
    d = _json(_SENS)["per_model"]["BERT"]
    return d["power_at_brennan_raw_effect"], 0.84, 0.02, _SENS


def c_b_masc_upper_limit():
    d = _json(_SENS)["per_model"]["BERT"]
    return d["upper_95_t"] * 1e3, 3.25, 0.05, _SENS


def c_b_masc_excl_p():
    d = _json(_SENS)["per_model"]["BERT"]
    return d["noninferiority_p"] * 1e3, 1.238, 0.05, _SENS


def c_b_masc_excl_all_models():
    """All three models must exclude a Brennan-sized effect for the paper's wording."""
    pm = _json(_SENS)["per_model"]
    n = sum(1 for m in ["BERT", "GPT-2", "GPT-2-medium"] if pm[m]["excludes_brennan_effect"])
    return n, 3, 0, _SENS


def c_b_masc_mde():
    """The resolution floor the paper concedes."""
    d = _json(_SENS)["per_model"]["BERT"]
    return d["mde_d_80pct"], 1.00, 0.03, _SENS


def c_b_megscans_pos():
    d = _tsv(f"{_BND}/boundary_megscans_persubject_summary.tsv")
    r = d[d["llm"] == "Mult-BERT"].iloc[0]
    return int(r["n_positive"]), 7, 0, f"{_BND}/boundary_megscans_persubject_summary.tsv"


def c_b_megscans_p():
    d = _tsv(f"{_BND}/boundary_megscans_persubject_summary.tsv")
    r = d[d["llm"] == "Mult-BERT"].iloc[0]
    return float(r["wilcoxon_p"]), 0.0054, 0.0002, f"{_BND}/boundary_megscans_persubject_summary.tsv"


def c_b_ctx_matched_frac():
    d = _json(f"{_BND}/context_control_brennan_v2_L1.json")
    return d["matched_fraction_of_full"] * 100, 59.0, 1.0, f"{_BND}/context_control_brennan_v2_L1.json"


def c_b_ctx_matched_p():
    d = _json(f"{_BND}/context_control_brennan_v2_L1.json")
    return d["delta_matched"]["wilcoxon_p"] * 1e4, 4.7, 0.2, f"{_BND}/context_control_brennan_v2_L1.json"


def c_b_ctx_n():
    d = _json(f"{_BND}/context_control_brennan_v2_L1.json")
    return d["n_subjects"], 47, 0, f"{_BND}/context_control_brennan_v2_L1.json"


def c_b_perm_z():
    d = _json(f"{_BND}/perm_null_brennan_v2_L1.json")
    return d["z_vs_null"], 3.32, 0.05, f"{_BND}/perm_null_brennan_v2_L1.json"


def c_b_perm_p():
    d = _json(f"{_BND}/perm_null_brennan_v2_L1.json")
    return d["p_permutation_one_sided"], 0.002, 0.0005, f"{_BND}/perm_null_brennan_v2_L1.json"


def c_brennan_v2_concordance():
    d = _json("p4_encoding/concordance32_brennan_v2.json")
    return d["concordance_spearman"], 0.34, 0.02, "p4_encoding/concordance32_brennan_v2.json"


def c_brennan_v2_class_gap():
    d = _json("p4_encoding/concordance_decomposition_brennan.json")
    return d["class_gap_causal_minus_other"], -0.38, 0.05, "p4_encoding/concordance_decomposition_brennan.json"


def c_brennan_v2_class_gap_p():
    """The class gap does NOT replicate on the multi-subject cohort."""
    d = _json("p4_encoding/concordance_decomposition_brennan.json")
    return d["gap_p_family_clustered"], 0.53, 0.03, "p4_encoding/concordance_decomposition_brennan.json"


def c_b_acoustic_base():
    d = _json(f"{_BND}/acoustic_control_brennan.json")["delta_base"]
    return d["mean"] * 1e3, 47.3, 0.5, f"{_BND}/acoustic_control_brennan.json"


def c_b_acoustic_controlled():
    """Partialling out preceding silence + sound power does not shrink the contrast."""
    d = _json(f"{_BND}/acoustic_control_brennan.json")["delta_acoustic_controlled"]
    return d["mean"] * 1e3, 50.2, 0.5, f"{_BND}/acoustic_control_brennan.json"


def c_b_acoustic_gapmatched():
    """Onsets vs longest-pause medial words: contrast persists and strengthens."""
    d = _json(f"{_BND}/acoustic_control_brennan.json")["delta_gap_matched"]
    return d["mean"] * 1e3, 78.5, 0.5, f"{_BND}/acoustic_control_brennan.json"


def c_scaling_rho():
    d = _json("p4_encoding/scaling_ladder_test.json")
    return d["size_vs_alignment_spearman"], 0.43, 0.01, "p4_encoding/scaling_ladder_test.json"


def c_scaling_p():
    """Size does not order alignment: the ladder correlation is not significant."""
    d = _json("p4_encoding/scaling_ladder_test.json")
    return d["size_vs_alignment_p"], 0.40, 0.01, "p4_encoding/scaling_ladder_test.json"


def c_scaling_largest_vs_smallest_p():
    """2.8B is indistinguishable from 70M despite 40x the parameters."""
    d = _json("p4_encoding/scaling_ladder_test.json")
    return d["largest_vs_smallest"]["wilcoxon_p"], 0.97, 0.02, "p4_encoding/scaling_ladder_test.json"


def c_scaling_rho_corr():
    """Ladder correlation under the corrected encoding arm: still no size ordering."""
    d = _json("p4_encoding/scaling_ladder_test_CORRECTED.json")
    return d["size_vs_alignment_spearman"], -0.143, 0.001, \
        "p4_encoding/scaling_ladder_test_CORRECTED.json"


def c_scaling_diff_sign_corr():
    """The 2.8B-minus-70M paired difference flips sign under the correction (audit prediction)."""
    d = _json("p4_encoding/scaling_ladder_test_CORRECTED.json")
    ok = d["largest_vs_smallest"]["diff"] < 0 and d["old_vs_new"]["sign_flipped"]
    return int(ok), 1, 0, "p4_encoding/scaling_ladder_test_CORRECTED.json"


def c_lb_heldout_in_sample():
    d = _json("p4_encoding/heldout_cell_selection_libribrain.json")["_pooled"]
    return d["in_sample"] * 1e3, 6.4, 0.1, "p4_encoding/heldout_cell_selection_libribrain.json"


def c_lb_heldout_held_out():
    d = _json("p4_encoding/heldout_cell_selection_libribrain.json")["_pooled"]
    return d["held_out"] * 1e3, 6.0, 0.1, "p4_encoding/heldout_cell_selection_libribrain.json"


def c_lb_heldout_bias_pct():
    """Selection costs only ~6% on the within-listener cohort, vs ~85% across ECoG patients."""
    d = _json("p4_encoding/heldout_cell_selection_libribrain.json")["_pooled"]
    return d["bias_as_pct_of_in_sample"], 6.3, 0.5, "p4_encoding/heldout_cell_selection_libribrain.json"


def c_heldout_in_sample():
    d = _json("ecog_stage1_ext/heldout_cell_selection.json")["_pooled"]
    return d["in_sample"] * 1e3, 9.6, 0.1, "ecog_stage1_ext/heldout_cell_selection.json"


def c_heldout_held_out():
    d = _json("ecog_stage1_ext/heldout_cell_selection.json")["_pooled"]
    return d["held_out"] * 1e3, 1.4, 0.1, "ecog_stage1_ext/heldout_cell_selection.json"


def c_heldout_bias_pct():
    d = _json("ecog_stage1_ext/heldout_cell_selection.json")["_pooled"]
    return d["bias_as_pct_of_in_sample"], 85.0, 1.0, "ecog_stage1_ext/heldout_cell_selection.json"



# ------------------------------------------------- the 1e-12 guard correction (2026-08-04)
# The published concordance was produced by code computing a per-column correlation as
#     num / (sqrt(ssY*ssP) + 1e-12)
# LibriBrain's neural data sits at std 3.95e-07 in physical units, so the denominator is ~1.8e-11
# and that guard is not negligible. It distorts rather than shrinks: ssP is the variance of THAT
# model's own predictions, so a stronger-predicting model gets a larger denominator and is shrunk
# less, which is not monotone in the true correlation and reorders checkpoints packed between
# +0.029 and +0.051.
#
# These four claims are the provenance chain a reader needs. The published artifact is kept
# untouched and still checked above; these establish that (a) the original code reproduces the
# published number exactly, (b) the corrected code gives a different one, and (c) changing ONLY
# the epsilon reproduces the correction, which is what identifies the guard as the cause.


def c_eps_original_repro():
    d = _json("p4_encoding/concordance32_ORIGINAL_repro.json")
    return d["concordance_spearman"], -0.6778, 0.0002, \
        "p4_encoding/concordance32_ORIGINAL_repro.json"


def c_eps_corrected():
    d = _json("p4_encoding/concordance32_CORRECTED.json")
    return d["concordance_spearman"], -0.5733, 0.0002, \
        "p4_encoding/concordance32_CORRECTED.json"


def c_eps_isolated():
    """Changing only 1e-12 -> 1e-30 must land on the corrected value, not the published one."""
    d = _json("p4_encoding/concordance32_EPSILON_ISOLATED.json")
    return d["concordance_spearman"], -0.5733, 0.0002, \
        "p4_encoding/concordance32_EPSILON_ISOLATED.json"


def c_eps_isolation_exact():
    """The isolation and the independent Y-normalisation fix must agree to machine precision.

    Agreement at 1e-12 is the evidence that the two interventions are the same operation: both
    lift the denominator above the guard. Approximate agreement would not settle it.
    """
    a = _json("p4_encoding/concordance32_EPSILON_ISOLATED.json")["concordance_spearman"]
    b = _json("p4_encoding/concordance32_CORRECTED.json")["concordance_spearman"]
    return abs(a - b), 0.0, 1e-12, "epsilon-isolated vs corrected"


def c_masc27_concordance():
    d = _json("p4_encoding/concordance32_masc27.json")
    return d["concordance_spearman"], 0.4886, 0.0005, \
        "p4_encoding/concordance32_masc27.json"



def c_identifiability_worst_ratio():
    """No cohort x readout x level combination falls below spread ~2.4x the paired-difference SE."""
    d = _json("p4_encoding/rank_identifiability_CORRECTED.json")
    return min(v["ratio"] for v in d.values()), 2.35, 0.05, \
        "p4_encoding/rank_identifiability_CORRECTED.json"


def c_identifiability_failures():
    d = _json("p4_encoding/rank_identifiability_CORRECTED.json")
    return sum(1 for v in d.values() if "NOT" in v["verdict"]), 0, 0, \
        "p4_encoding/rank_identifiability_CORRECTED.json"


def c_rdm_reliability_libribrain():
    d = _json("p4_encoding/noise_sweep/libribrain_full_dose0_seed0.json")
    return d["neural_rdm_reliability"], 0.774, 0.005, \
        "p4_encoding/noise_sweep/libribrain_full_dose0_seed0.json"


def c_rdm_reliability_masc():
    d = _json("p4_encoding/noise_sweep/meg_masc_merged27_dose0_seed0.json")
    return d["neural_rdm_reliability"], 0.786, 0.005, \
        "p4_encoding/noise_sweep/meg_masc_merged27_dose0_seed0.json"


def c_rdm_reliability_brennan():
    d = _json("p4_encoding/noise_sweep/brennan_p4_v2_dose0_seed0.json")
    return d["neural_rdm_reliability"], 0.858, 0.005, \
        "p4_encoding/noise_sweep/brennan_p4_v2_dose0_seed0.json"


# --- participation-ratio mechanism (Table 7 / Figure C) --------------------------
# These four numbers and the two clustered CIs were the last place the -0.678 -> -0.573
# correction failed to reach: the appendix paragraph, the table column and the figure each
# carried a different definition of "the gap".  Registering them pins the definition --
# PRIMARY readout scores from concordance32_CORRECTED, families from codes/model_families.py.
_PRGAP = {}


def _prgap():
    if _PRGAP:
        return _PRGAP
    import glob
    import numpy as np
    from scipy.stats import spearmanr
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "codes"))
    from model_families import family
    D = [json.load(open(f)) for f in sorted(glob.glob(str(RESULTS / "p4_pc/pctrunc_*.json")))]
    per = {r["model"]: r for r in _json("p4_encoding/concordance32_CORRECTED.json")["per_model"]}
    pr = {m["model"]: m["participation_ratio"] for m in D}
    gap = {m: per[m]["enc_mean"] - per[m]["rsa_mean"] for m in pr}
    kind = {m["model"]: m["kind"] for m in D}

    def rho(ms):
        return float(spearmanr([pr[m] for m in ms], [gap[m] for m in ms])[0])

    def ci(ms, seed=0, B=10000):
        rng = np.random.RandomState(seed)
        fams = sorted({family(m) for m in ms})
        by = {f: [m for m in ms if family(m) == f] for f in fams}
        bs = []
        for _ in range(B):
            s = [m for f in rng.choice(fams, len(fams), replace=True) for m in by[f]]
            r = spearmanr([pr[m] for m in s], [gap[m] for m in s])[0]
            if np.isfinite(r):
                bs.append(r)
        return [float(v) for v in np.percentile(bs, [2.5, 97.5])]

    allm = list(gap)
    cau = [m for m in gap if kind[m] == "causal"]
    _PRGAP.update(pooled=rho(allm), causal=rho(cau),
                  masked=rho([m for m in gap if kind[m] == "masked"]),
                  encdec=rho([m for m in gap if kind[m] == "encdec"]),
                  pooled_ci=ci(allm), causal_ci=ci(cau))
    return _PRGAP


_PRGAP_SRC = "p4_pc/pctrunc_*.json + p4_encoding/concordance32_CORRECTED.json"


def c_prgap_pooled():
    return _prgap()["pooled"], -0.65, 0.005, _PRGAP_SRC


def c_prgap_causal():
    return _prgap()["causal"], -0.76, 0.005, _PRGAP_SRC


def c_prgap_masked():
    return _prgap()["masked"], 0.07, 0.005, _PRGAP_SRC


def c_prgap_encdec():
    return _prgap()["encdec"], 0.10, 0.005, _PRGAP_SRC


def c_prgap_pooled_ci_hi():
    return _prgap()["pooled_ci"][1], -0.02, 0.02, _PRGAP_SRC


def c_prgap_causal_ci_hi():
    return _prgap()["causal_ci"][1], -0.09, 0.02, _PRGAP_SRC



# ---------------------------------------------------------------- space-fix re-extraction (2026-09-25)
# The documented extractor dropped inter-word spaces for Qwen/Llama tokenizers (transformers 4.57.1),
# so 18 LibriBrain extension checkpoints, the MEG-MASC Qwen2.5 ladder and three small checkpoints on
# MEG-MASC/Brennan were scored on unspaced text. They were re-extracted under the headline (ef17743)
# rule (prereg paper4_temporalalign/PREREG_space_fix_reextract_20260925.md); every result that uses
# them now reads the re-extracted receipts below. The pre-fix claims are kept in SUPERSEDED as
# provenance pins (the paper still prints them once, as the unspaced comparison, in the
# feature-extraction provenance appendix). Summary + mechanical outcomes:
# p4_encoding/space_fix_20260925.json; the TeX side is pinned by codes/check_space_fix_20260926.py.
_SF = "p4_encoding/space_fix_20260925"
_SF6 = f"{_SF}/aggregate/adr0006_registered_verdicts_spacefix.json"
_SF10 = f"{_SF}/aggregate/adr0010_q1_masc_verdicts_spacefix.json"
_SFSUM = "p4_encoding/space_fix_20260925.json"


def c_sf_complete():
    """36 re-extractions, none withdrawn (prereg section 7)."""
    s = _json(_SFSUM)
    n = int(s["summary_meta"]["completeness"].split("/")[0])
    return n if not s["libribrain"]["missing_unspaced"] else -1, 36, 0, _SFSUM


def c_sf_q2_rho():
    return float(_json(_SF6)["q2"]["rho"]), -0.59, 0.005, _SF6


def c_sf_q2_ci_lo():
    return float(_json(_SF6)["q2"]["ci95_family_clustered"][0]), -0.78, 0.005, _SF6


def c_sf_q2_ci_hi():
    return float(_json(_SF6)["q2"]["ci95_family_clustered"][1]), -0.12, 0.005, _SF6


def c_sf_q2_nbase():
    return int(_json(_SF6)["q2"]["n_base"]), 48, 0, _SF6


def c_sf_q1_25_enc_rho():
    return float(_json(_SF6)["q1_qwen2.5"]["enc"]["rho"]), -0.96, 0.005, _SF6


def c_sf_q1_25_rsa_rho():
    return float(_json(_SF6)["q1_qwen2.5"]["rsa"]["rho"]), 0.93, 0.005, _SF6


def c_sf_q1_25_top_p():
    return float(_json(_SF6)["q1_qwen2.5"]["enc"]["top_minus_bottom_p"]), 0.0005, 0.0001, _SF6


def c_sf_q1_25_rsa_top_p():
    return float(_json(_SF6)["q1_qwen2.5"]["rsa"]["top_minus_bottom_p"]), 0.001, 0.0005, _SF6


def c_sf_q1_3_enc_rho():
    """Receipt value (not printed; Figure fig:scaling); the printed item is the label change."""
    return float(_json(_SF6)["q1_qwen3"]["enc"]["rho"]), -0.40, 0.005, _SF6


def c_sf_q1_3_not_supported():
    """Qwen3 registered label: MIXED on the unspaced features, NOT SUPPORTED re-extracted (1 = yes)."""
    return int(_json(_SF6)["q1_qwen3_verdict"].startswith("NOT SUPPORTED")), 1, 0, _SF6


def c_sf_q1_35_rsa_rho():
    return float(_json(_SF6)["q1_qwen3.5"]["rsa"]["rho"]), -0.80, 0.005, _SF6


def c_sf_q1_35_rsa_ci_hi():
    return float(_json(_SF6)["q1_qwen3.5"]["rsa"]["ci95_session_clustered"][1]), -0.80, 0.005, _SF6


def c_sf_q3_rsa():
    return int(_json(_SF6)["q3_direction_count_rsa"].split("/")[0]), 2, 0, _SF6


def c_sf_q3_enc():
    return int(_json(_SF6)["q3_direction_count_enc"].split("/")[0]), 4, 0, _SF6


def _sf_alpha(rung):
    return _json(f"{_SF}/score/libribrain__Qwen_Qwen2.5-{rung}__new.json")["alpha_grid"]


def c_sf_alpha_edge_72b():
    """Original-grid ceiling (1e6) chosen in 11/12 folds on the re-extracted 72B, as in alpha_probe."""
    f = _sf_alpha("72B")["folds"]
    return sum(v["original_alpha"] == 1e6 for v in f.values()), 11, 0, \
        f"{_SF}/score/libribrain__Qwen_Qwen2.5-72B__new.json"


def c_sf_alpha_rho_orig():
    return float(_json(_SFSUM)["outcomes"]["6.4_alpha_grid"]["spearman_size_original"]), -0.964, 0.0005, _SFSUM


def c_sf_alpha_rho_exp():
    return float(_json(_SFSUM)["outcomes"]["6.4_alpha_grid"]["spearman_size_expanded"]), -0.464, 0.0005, _SFSUM


def c_sf_alpha_72b_gap_exp():
    return float(_json(_SFSUM)["outcomes"]["6.4_alpha_grid"]["72B_minus_0.5B_expanded"]), -0.0079, 0.00005, _SFSUM


def c_sf_q1m_rsa_rho():
    return float(_json(_SF10)["q1_masc"]["rsa"]["rho"]), 0.29, 0.005, _SF10


def c_sf_q1m_rsa_ci_lo():
    return float(_json(_SF10)["q1_masc"]["rsa"]["ci95_subject_clustered"][0]), -0.32, 0.005, _SF10


def c_sf_q1m_rsa_ci_hi():
    return float(_json(_SF10)["q1_masc"]["rsa"]["ci95_subject_clustered"][1]), 0.75, 0.005, _SF10


def c_sf_q1m_rsa_pair_diff():
    return float(_json(_SF10)["q1_masc"]["rsa"]["top_minus_bottom_mean"]) * 1e4, 0.9, 0.05, _SF10


def c_sf_q1m_rsa_pair_p():
    return float(_json(_SF10)["q1_masc"]["rsa"]["top_minus_bottom_p"]), 0.71, 0.005, _SF10


def c_sf_q1m_rsa_pair_pos():
    return int(_json(_SF10)["q1_masc"]["rsa"]["n_positive_top_minus_bottom"]), 15, 0, _SF10


def c_sf_q1m_rsa_sb():
    return float(_json(_SF10)["f3_gate"]["rsa"]["split_half"]), 0.83, 0.005, _SF10


def c_sf_q1m_rsa_gate_lo():
    return float(_json(_SF10)["f3_gate"]["rsa"]["ci90"][0]), 0.13, 0.005, _SF10


def c_sf_q1m_enc_sb():
    return float(_json(_SF10)["f3_gate"]["enc"]["split_half"]), 0.70, 0.005, _SF10


def c_sf_q1m_enc_gate_lo():
    return float(_json(_SF10)["f3_gate"]["enc"]["ci90"][0]), -0.16, 0.005, _SF10


def c_sf_q1m_enc_gate_hi():
    return float(_json(_SF10)["f3_gate"]["enc"]["ci90"][1]), 0.94, 0.005, _SF10


def c_sf_masc27_concordance():
    d = _json(f"{_SF}/aggregate/concordance_spacefix_masc27_32.json")
    return d["concordance_spearman"], 0.446, 0.0005, f"{_SF}/aggregate/concordance_spacefix_masc27_32.json"


def c_sf_masc27_ci_lo():
    d = _json(f"{_SF}/aggregate/concordance_spacefix_masc27_32.json")
    return d["concordance_ci95"][0], 0.12, 0.005, f"{_SF}/aggregate/concordance_spacefix_masc27_32.json"


def c_sf_brennan_concordance():
    d = _json(f"{_SF}/aggregate/concordance_spacefix_brennan_32.json")
    return d["concordance_spearman"], 0.366, 0.001, f"{_SF}/aggregate/concordance_spacefix_brennan_32.json"


def c_sf_brennan_ci_lo():
    """Printed as +0.02: the interval now excludes zero, narrowly (prereg 6.8 'flips')."""
    d = _json(f"{_SF}/aggregate/concordance_spacefix_brennan_32.json")
    return d["concordance_ci95"][0], 0.02, 0.005, f"{_SF}/aggregate/concordance_spacefix_brennan_32.json"


CLAIMS = [
    ("prgap.pooled", "PR vs primary readout gap, pooled over 32 checkpoints", c_prgap_pooled),
    ("prgap.causal", "PR vs gap inside the 18 autoregressive checkpoints", c_prgap_causal),
    ("prgap.masked", "PR vs gap inside the masked class", c_prgap_masked),
    ("prgap.encdec", "PR vs gap inside the encoder-decoder class", c_prgap_encdec),
    ("prgap.pooled_ci_hi", "Pooled PR-gap family-clustered CI upper bound", c_prgap_pooled_ci_hi),
    ("prgap.causal_ci_hi", "Causal PR-gap family-clustered CI upper bound", c_prgap_causal_ci_hi),
    ("ident.worst_ratio", "Worst identifiability ratio across 8 combos", c_identifiability_worst_ratio),
    ("ident.failures", "Combinations failing identifiability", c_identifiability_failures),
    ("rdmrel.libribrain", "Neural RDM reliability, LibriBrain dose-0", c_rdm_reliability_libribrain),
    ("rdmrel.masc", "Neural RDM reliability, MEG-MASC dose-0", c_rdm_reliability_masc),
    ("rdmrel.brennan", "Neural RDM reliability, Brennan dose-0", c_rdm_reliability_brennan),
    ("eps.original_repro", "Original code reproduces the published -0.678", c_eps_original_repro),
    ("eps.corrected", "Corrected concordance (scale-invariant denominator)", c_eps_corrected),
    ("eps.isolated", "Changing only the 1e-12 guard gives the corrected value", c_eps_isolated),
    ("eps.isolation_exact", "Isolation and Y-normalisation agree to 1e-12", c_eps_isolation_exact),
    ("masc27.concordance", "MEG-MASC n=27 concordance, corrected code", c_masc27_concordance),
    ("concordance.headline", "Similarity vs encoding rank concordance rho", c_concordance_headline),
    ("concordance.ci_low", "Concordance 95% CI lower bound", c_concordance_ci_low),
    ("concordance.ci_high", "Concordance 95% CI upper bound", c_concordance_ci_high),
    ("concordance.family_ci_low", "Family-clustered CI lower bound", c_family_clustered_ci),
    ("concordance.n_families", "Number of model families", c_n_families),
    ("depth.k1", "Concordance at single-session depth", c_depth_k1),
    ("depth.sessions_neg", "Sessions reversing independently", c_depth_sessions_negative),
    ("depth.session_p", "Per-session reversal, Wilcoxon p (e-3)", c_depth_session_p),
    ("invariance.rawrsa", "Reversal with both readouts un-partialled", c_rawrsa_symmetry),
    ("invariance.chantime", "Both readouts on channel x time", c_invariance_chantime),
    ("invariance.timeavg", "Both readouts time-averaged", c_invariance_timeavg),
    ("invariance.allch", "All 306 channels", c_invariance_allch),
    ("stdfeat.centered", "Mean-centred features", c_stdfeat_centered),
    ("stdfeat.standardized", "Per-dimension standardized features", c_stdfeat_standardized),
    ("cohort.brennan", "Brennan EEG concordance (no reversal)", c_brennan_concordance),
    ("ceiling.gamma", "Per-band reliability ceiling, gamma", c_ceiling_gamma),
    ("ceiling.broadband", "Per-band reliability ceiling, broadband", c_ceiling_broadband),
    ("ceiling.delta", "Per-band reliability ceiling, delta", c_ceiling_delta),
    ("broadband.partial", "Confound-partialled broadband alignment", c_broadband_partial),
    ("perunit.bb_gt_gamma", "Units with broadband > gamma", c_per_unit_bb_gt_gamma),
    ("perunit.total", "Total units", c_per_unit_total),
    ("perunit.bb_ecog", "ECoG units favouring broadband", c_bb_ecog),
    ("perunit.bb_libribrain", "LibriBrain units favouring broadband", c_bb_libribrain),
    ("perunit.bb_unanimous", "Cohorts unanimous for broadband", c_bb_cohorts_unanimous),
    ("perunit.bb_majority", "Cohorts favouring broadband", c_bb_all_cohorts_majority),
    ("leaderboard.bert_libribrain", "LibriBrain BERT-base best rho", c_leaderboard_bert_libribrain),
    ("leaderboard.gpt2med_libribrain", "LibriBrain GPT-2-medium best rho", c_leaderboard_gpt2med_libribrain),
    ("scorecard.derived_zou_cells", "Zou cells (Brennan: archive cell + rebuild re-derivation)", c_derived_zou_cells),
    ("scorecard.derived_vs_table", "Derived verdicts disagreeing with table", c_derived_matches_table),
    ("tax.n23", "Selection tax, 23 subjects", c_tax_n23),
    ("tax.n9", "Selection tax, 9 subjects (count effect)", c_tax_n9),
    ("tax.count_rho", "Tax vs unit count, Spearman rho", c_tax_count_rho),
    ("masc.reepoch_words", "Words recovered re-epoching from raw", c_reepoch_words),
    ("masc.reepoch_corr", "Median per-epoch r vs stored archive", c_reepoch_corr),
    ("masc.reepoch_worst", "Epochs below r=0.99 (0 = none)", c_reepoch_worst),
    ("masc.subset_age_p", "MEG-MASC analyzed-vs-excluded age p", c_masc_subset_age_p),
    ("masc.subset_any_sig", "Imbalances surviving Holm (0 = none)", c_masc_subset_any_sig),
    ("masc.recordings", "MEG-MASC recordings analyzed", c_masc_recordings),
    ("protocol.selftest", "Protocol statistic vs known answers", c_protocol_selftest),
    ("protocol.validated_values", "Stored RSA values recomputed by bench/protocol.py",
     c_protocol_validation),
    ("protocol.mismatches", "Mismatches against stored RSA values", c_protocol_mismatches),
    ("scorecard.derived_coverage", "Verdicts computed, not asserted", c_derived_coverage),
    ("scorecard.toneva_cells", "Mid-layer cells recomputable", c_toneva_cells),
    ("scorecard.toneva_strict", "Cells where mid beats shallow and deep", c_toneva_strict),
    ("scorecard.toneva_weak", "Cells where mid-to-shallow beats deep", c_toneva_weak),
    ("scorecard.replicate", "Scorecard 'replicate' verdicts", c_scorecard_replicate),
    ("scorecard.refine", "Scorecard 'refine' verdicts", c_scorecard_refine),
    ("scorecard.contradict", "Scorecard 'contradict' verdicts", c_scorecard_contradict),
    ("decomp.width_matched_gap", "Class gap at matched feature width", c_width_matched_gap),
    ("decomp.width_matched_p", "Width-matched gap, family-clustered p", c_width_matched_p),
    ("decomp.within_class_rho", "Concordance after removing class means", c_decomp_within_class),
    ("decomp.class_gap", "Causal-vs-non-causal ENC-RSA rank gap", c_decomp_class_gap),
    ("decomp.gap_p_family", "Class gap, family-clustered permutation p", c_decomp_gap_p_family),
    ("decomp.class_gap_corr", "Class gap, corrected encoding arm", c_decomp_class_gap_corr),
    ("decomp.gap_p_family_corr", "Corrected gap, family-clustered p (2-sided)",
     c_decomp_gap_p_family_corr),
    ("mech.causal_rho", "PR-vs-gap within causal class", c_mech_causal_rho),
    ("mech.causal_ci_hi", "Within-causal family CI upper (excludes 0)", c_mech_causal_ci_excludes_zero),
    ("mech.all_ci_hi", "All-class family CI upper (crosses 0)", c_mech_all_ci_crosses_zero),
    # --- Paper B ---
    # 2026-10-04 (BootLoops audit L2): since the A' rebase (10-03) paperB's Brennan claims are the 47-listener
    # re-extraction (B.brennan47_*). The 49-listener archive survives only in paperB's Appendix app:archive and as the
    # two registered (E5) margins; its pins are kept as provenance under B.archive_* and are NOT Brennan claims.
    ("B.brennan47_pos", "Brennan (47 rebuild, arm L1) BERT subjects positive", c_b_brennan47_pos),
    ("B.brennan47_n", "Brennan (47 rebuild) subjects", c_b_brennan47_n),
    ("B.brennan47_mean", "Brennan (47 rebuild, L1) BERT mean delta (x1e-3)", c_b_brennan47_mean),
    ("B.brennan47_p", "Brennan (47 rebuild, L1) BERT Wilcoxon p (x1e-5)", c_b_brennan47_p),
    ("B.brennan47_between_p", "Brennan 47 vs MEG-MASC 24, Mann-Whitney p (x1e-4)", c_b_brennan47_between_p),
    ("B.archive_brennan_pos", "ARCHIVE (app:archive, not recomputable): Brennan-49 BERT positive", c_b_brennan_pos),
    ("B.archive_brennan_n", "ARCHIVE (app:archive): Brennan-49 subjects", c_b_brennan_n),
    ("B.archive_brennan_mean", "ARCHIVE (app:archive; = (E5) margin): Brennan-49 BERT mean (x1e-3)", c_b_brennan_mean),
    ("B.archive_brennan_p", "ARCHIVE (app:archive): Brennan-49 BERT Wilcoxon p (x1e-4)", c_b_brennan_p),
    ("B.masc_pos", "MEG-MASC BERT subjects positive", c_b_masc_pos),
    ("B.archive_between_cohort_p", "ARCHIVE: Brennan-49 vs MEG-MASC-10 between-cohort p", c_b_between_cohort_p),
    ("B.archive_masc_power", "ARCHIVE: MEG-MASC power at the Brennan-49 effect size", c_b_masc_power),
    ("B.archive_masc_power_raw", "ARCHIVE: MEG-MASC power at the Brennan-49 RAW effect", c_b_masc_power_raw),
    ("B.masc27_n", "n=27 rebuild, subjects above the artifact floor", c_masc27_n),
    ("B.masc27_upper95", "n=27 one-sided 95% upper limit (e-3)", c_masc27_upper),
    ("B.masc27_excl_p", "n=27 exclusion of Brennan effect, p (e-6)", c_masc27_excl_p),
    ("B.masc27_all_excl_p", "n=27 all-subjects exclusion, p (e-6)", c_masc27_all_excl_p),
    ("B.masc27_both_arms", "Arms excluding a Brennan-sized effect", c_masc27_both_exclude),
    ("B.xcohort27_p", "Brennan vs n=27 MEG-MASC, p (e-3)", c_xcohort27_p),
    ("B.xcohort27_level", "Absolute alignment levels, between-cohort p", c_xcohort27_level_p),
    ("B.masc_upper95", "MEG-MASC one-sided 95% upper limit (e-3)", c_b_masc_upper_limit),
    ("B.masc_excl_p", "MEG-MASC excludes Brennan-sized effect, p (e-3)", c_b_masc_excl_p),
    ("B.masc_excl_n", "models excluding a Brennan-sized effect", c_b_masc_excl_all_models),
    ("B.masc_mde", "MEG-MASC 80%-power resolution floor |d|", c_b_masc_mde),
    ("B.megscans_pos", "MEG-SCANS mult-BERT positive", c_b_megscans_pos),
    ("B.megscans_p", "MEG-SCANS mult-BERT p", c_b_megscans_p),
    ("B.ctx_n", "Context control subjects (rebuilt preprocessing)", c_b_ctx_n),
    ("B.ctx_matched_frac", "Effect retained under context matching (%)", c_b_ctx_matched_frac),
    ("B.ctx_matched_p", "Context-matched Wilcoxon p (x1e-4)", c_b_ctx_matched_p),
    ("B.perm_z", "Permutation null z", c_b_perm_z),
    ("B.perm_p", "Permutation null p", c_b_perm_p),
    # --- Paper A (cont.) ---
    # --- Brennan multi-subject replication of the reversal (it does not) ---
    ("brennan.concordance", "47-subj Brennan concordance (not -0.68)", c_brennan_v2_concordance),
    ("brennan.class_gap", "47-subj Brennan class gap (vs +27.9)", c_brennan_v2_class_gap),
    ("brennan.class_gap_p", "Class gap does not replicate (p)", c_brennan_v2_class_gap_p),
    ("B.acoustic_base", "Brennan delta, baseline (x1e-3)", c_b_acoustic_base),
    ("B.acoustic_ctrl", "Delta with acoustic confounds partialled", c_b_acoustic_controlled),
    ("B.acoustic_gapmatch", "Delta, onsets vs long-pause medial", c_b_acoustic_gapmatched),
    ("B.hz_sib_page_p", "Horizon trend Page p, sibling (x1e5)", c_b_hz_sib_page_p),
    ("B.hz_sib_rho", "Horizon Spearman rho, sibling", c_b_hz_sib_rho),
    ("B.hz_sib_ret_k0", "Horizon retention at k=0, sibling", c_b_hz_sib_ret_k0),
    ("B.hz_sib_ret_k8", "Horizon retention at k=8, sibling", c_b_hz_sib_ret_k8),
    ("B.hz_head_page_p", "Horizon trend Page p, headline (x1e6)", c_b_hz_head_page_p),
    ("B.hz_head_rho", "Horizon Spearman rho, headline", c_b_hz_head_rho),
    ("B.hz_head_ret_k0", "Horizon retention at k=0, headline", c_b_hz_head_ret_k0),
    ("B.surp_d", "Surprisal-readout boundary d (negative)", c_b_surp_d),
    ("B.surp_d_hi", "Surprisal d one-sided 95% upper bound", c_b_surp_dhi),
    ("B.surp_pos", "Surprisal-readout positive subjects", c_b_surp_pos),
    ("B.surp_p", "Surprisal-readout Wilcoxon p", c_b_surp_p),
    ("B.floor_sb", "Brennan neural RDM split-half (Spearman-Brown)", c_b_floor_sb),
    ("B.floor_ci_lo", "Brennan RDM reliability CI90 lower", c_b_floor_ci_lo),
    ("B.floor_ci_hi", "Brennan RDM reliability CI90 upper", c_b_floor_ci_hi),
    ("B.surp_masc_d", "MASC surprisal-readout d (null)", c_b_surp_masc_d),
    ("B.surp_masc_p", "MASC surprisal-readout Wilcoxon p", c_b_surp_masc_p),
    ("B.lm_page_p", "Scrambled-future Page p (flat)", c_b_lm_page_p),
    ("B.lm_k2", "True-minus-scrambled at k=2 (x1e-3)", c_b_lm_k2),
    ("B.lm_k8", "True-minus-scrambled at k=8 (x1e-3)", c_b_lm_k8),
    ("A.q1_25_enc_rho", "Qwen2.5 ladder encoding rho (negative)", c_a_q1_25_enc_rho),
    ("A.q1_25_rsa_rho", "Qwen2.5 ladder RSA rho (positive)", c_a_q1_25_rsa_rho),
    ("A.q1_25_top_p", "72B vs 0.5B encoding paired p", c_a_q1_25_top_p),
    ("A.q1_3_enc_rho", "Qwen3 ladder encoding rho (weakly positive)", c_a_q1_3_enc_rho),
    ("A.q1_35_rsa_rho", "Qwen3.5 ladder RSA rho (falls)", c_a_q1_35_rsa_rho),
    ("A.q1_35_rsa_ci_hi", "Qwen3.5 ladder RSA CI upper", c_a_q1_35_rsa_ci_hi),
    ("A.q2_rho", "48-base-model concordance rho", c_a_q2_rho),
    ("A.q2_ci_hi", "48-model family-clustered CI upper", c_a_q2_ci_hi),
    ("A.q2_nbase", "Base concordance set size", c_a_q2_nbase),
    ("A.alpha_72b", "Grid-top alpha folds, 72B (of 12)", c_a_alpha_edge_72b),
    ("A.alpha_05b", "Grid-top alpha folds, 0.5B (of 12)", c_a_alpha_edge_05b),
    ("A.q3_rsa", "Instruct-twin direction count, RSA", c_a_q3_rsa_count),
    ("A.q3_enc", "Instruct-twin direction count, encoding", c_a_q3_enc_count),
    ("A.q1m_enc_rho", "MASC ladder encoding rho (flat)", c_a_q1m_enc_rho),
    ("A.q1m_enc_ci_lo", "MASC ladder encoding CI lower", c_a_q1m_enc_ci_lo),
    ("A.q1m_enc_ci_hi", "MASC ladder encoding CI upper", c_a_q1m_enc_ci_hi),
    ("A.q1m_enc_pair_p", "MASC 72B vs 0.5B paired p", c_a_q1m_enc_pair_p),
    ("A.q1m_enc_pair_diff", "MASC 72B-0.5B paired diff (x1e-3)", c_a_q1m_enc_pair_diff),
    ("A.q1m_enc_pair_pos", "MASC 72B>0.5B subject count (of 27)", c_a_q1m_enc_pair_pos),
    ("A.q1m_enc_gate_lo", "MASC encoding leaderboard CI90 lower", c_a_q1m_enc_gate_lo),
    ("A.q1m_enc_sb", "MASC encoding leaderboard split-half", c_a_q1m_enc_sb),
    ("A.q1m_rsa_sb", "MASC similarity leaderboard split-half", c_a_q1m_rsa_sb),
    ("A.q1m_rsa_gate_lo", "MASC similarity CI90 lower (gate fail)", c_a_q1m_rsa_gate_lo),
    ("scaling.rho", "Size vs alignment across Pythia ladder", c_scaling_rho),
    ("scaling.p", "Ladder correlation p (n.s.)", c_scaling_p),
    ("scaling.big_vs_small_p", "2.8B vs 70M paired p", c_scaling_largest_vs_smallest_p),
    ("scaling.rho_corr", "Ladder rho, corrected encoding arm", c_scaling_rho_corr),
    ("scaling.diff_sign_corr", "2.8B-70M diff sign flips (corrected)", c_scaling_diff_sign_corr),
    ("heldout.lb_in_sample", "LibriBrain in-sample best-cell (x1e-3)", c_lb_heldout_in_sample),
    ("heldout.lb_held_out", "LibriBrain held-out cell-selection (x1e-3)", c_lb_heldout_held_out),
    ("heldout.lb_bias_pct", "LibriBrain selection cost (%) vs ECoG 85%", c_lb_heldout_bias_pct),
    ("heldout.in_sample", "ECoG in-sample best-cell rho (x1e-3)", c_heldout_in_sample),
    ("heldout.held_out", "ECoG held-out cell-selection rho (x1e-3)", c_heldout_held_out),
    ("heldout.bias_pct", "Share of best-cell value that is selection (%)", c_heldout_bias_pct),
    # --- space-fix re-extraction (2026-09-25): current values; pre-fix siblings are in SUPERSEDED ---
    ("SF.complete", "Space-fix re-extractions completed (none withdrawn)", c_sf_complete),
    ("SF.q2_rho", "48-base concordance rho, re-extracted", c_sf_q2_rho),
    ("SF.q2_ci_lo", "48-base family-clustered CI lower, re-extracted", c_sf_q2_ci_lo),
    ("SF.q2_ci_hi", "48-base family-clustered CI upper, re-extracted", c_sf_q2_ci_hi),
    ("SF.q2_nbase", "Base concordance set size, re-extracted", c_sf_q2_nbase),
    ("SF.q1_25_enc_rho", "Qwen2.5 ladder encoding rho, re-extracted", c_sf_q1_25_enc_rho),
    ("SF.q1_25_rsa_rho", "Qwen2.5 ladder RSA rho, re-extracted", c_sf_q1_25_rsa_rho),
    ("SF.q1_25_top_p", "72B vs 0.5B encoding paired p, re-extracted", c_sf_q1_25_top_p),
    ("SF.q1_25_rsa_top_p", "72B vs 0.5B similarity paired p, re-extracted", c_sf_q1_25_rsa_top_p),
    ("SF.q1_3_enc_rho", "Qwen3 ladder encoding rho, re-extracted", c_sf_q1_3_enc_rho),
    ("SF.q1_3_not_supported", "Qwen3 registered label now NOT SUPPORTED", c_sf_q1_3_not_supported),
    ("SF.q1_35_rsa_rho", "Qwen3.5 ladder RSA rho, re-extracted", c_sf_q1_35_rsa_rho),
    ("SF.q1_35_rsa_ci_hi", "Qwen3.5 ladder RSA CI upper, re-extracted", c_sf_q1_35_rsa_ci_hi),
    ("SF.q3_rsa", "Instruct-twin direction count, RSA, re-extracted", c_sf_q3_rsa),
    ("SF.q3_enc", "Instruct-twin direction count, encoding, re-extracted", c_sf_q3_enc),
    ("SF.alpha_72b", "Grid-top alpha folds, 72B re-extracted (of 12)", c_sf_alpha_edge_72b),
    ("SF.alpha_rho_orig", "Qwen2.5 size rho, original alpha grid", c_sf_alpha_rho_orig),
    ("SF.alpha_rho_exp", "Qwen2.5 size rho, expanded alpha grid", c_sf_alpha_rho_exp),
    ("SF.alpha_72b_gap", "72B minus 0.5B, expanded grid", c_sf_alpha_72b_gap_exp),
    ("SF.q1m_rsa_rho", "MASC ladder similarity rho, re-extracted", c_sf_q1m_rsa_rho),
    ("SF.q1m_rsa_ci_lo", "MASC ladder similarity CI lower", c_sf_q1m_rsa_ci_lo),
    ("SF.q1m_rsa_ci_hi", "MASC ladder similarity CI upper", c_sf_q1m_rsa_ci_hi),
    ("SF.q1m_rsa_pair_diff", "MASC 72B-0.5B similarity diff (x1e-4)", c_sf_q1m_rsa_pair_diff),
    ("SF.q1m_rsa_pair_p", "MASC 72B vs 0.5B similarity p", c_sf_q1m_rsa_pair_p),
    ("SF.q1m_rsa_pair_pos", "MASC 72B>0.5B similarity subjects (of 27)", c_sf_q1m_rsa_pair_pos),
    ("SF.q1m_rsa_sb", "MASC similarity split-half (gate pass)", c_sf_q1m_rsa_sb),
    ("SF.q1m_rsa_gate_lo", "MASC similarity CI90 lower (gate pass)", c_sf_q1m_rsa_gate_lo),
    ("SF.q1m_enc_sb", "MASC encoding split-half (gate fail)", c_sf_q1m_enc_sb),
    ("SF.q1m_enc_gate_lo", "MASC encoding CI90 lower (gate fail)", c_sf_q1m_enc_gate_lo),
    ("SF.q1m_enc_gate_hi", "MASC encoding CI90 upper", c_sf_q1m_enc_gate_hi),
    ("SF.masc27_concordance", "MEG-MASC n=27 concordance, re-extracted", c_sf_masc27_concordance),
    ("SF.masc27_ci_lo", "MEG-MASC n=27 concordance CI lower", c_sf_masc27_ci_lo),
    ("SF.brennan_concordance", "Brennan n=47 concordance, re-extracted", c_sf_brennan_concordance),
    ("SF.brennan_ci_lo", "Brennan n=47 concordance CI lower (excludes 0)", c_sf_brennan_ci_lo),
]


# Claims kept for provenance: they pin values from result files that a later correction
# superseded (the corrected value is pinned by a sibling claim) and are NOT numbers printed in the
# benchmark paper. They still must reproduce from the files shipped beside the corrected ones.
SUPERSEDED = {
    "decomp.class_gap": "superseded by decomp.class_gap_corr (paper: +25.7)",
    "decomp.gap_p_family": "superseded by decomp.gap_p_family_corr (paper: two-sided p=0.005)",
    "mech.causal_rho": "superseded by prgap.causal (paper: -0.76)",
    "mech.causal_ci_hi": "superseded by prgap.causal_ci_hi (paper: -0.09)",
    "mech.all_ci_hi": "superseded by prgap.pooled_ci_hi (paper: -0.02)",
    "scaling.rho": "superseded by scaling.rho_corr (paper: -0.14)",
    "scaling.p": "pre-correction ladder file; paper prints p=0.79 from the corrected file",
    "scaling.big_vs_small_p": "pre-correction ladder file; paper prints p=0.23 from the corrected file",
    # --- unspaced-features values (space-fix, 2026-09-25): current value pinned by the SF.* sibling;
    #     receipt p4_encoding/space_fix_20260925.json; TeX pins codes/check_space_fix_20260926.py ---
    "A.q2_rho": "space-fix superseded by SF.q2_rho (paper: -0.59; -0.58 printed as the unspaced value)",
    "A.q2_ci_hi": "space-fix superseded by SF.q2_ci_hi (paper: -0.12)",
    "A.q2_nbase": "space-fix superseded by SF.q2_nbase (48, same set, re-extracted)",
    "A.q1_25_enc_rho": "space-fix superseded by SF.q1_25_enc_rho (paper: -0.96)",
    "A.q1_25_rsa_rho": "space-fix superseded by SF.q1_25_rsa_rho (paper: +0.93)",
    "A.q1_25_top_p": "space-fix superseded by SF.q1_25_top_p (paper: p=0.0005)",
    "A.q1_3_enc_rho": "space-fix superseded by SF.q1_3_enc_rho (label now NOT SUPPORTED: SF.q1_3_not_supported)",
    "A.q1_35_rsa_rho": "space-fix superseded by SF.q1_35_rsa_rho",
    "A.q1_35_rsa_ci_hi": "space-fix superseded by SF.q1_35_rsa_ci_hi",
    "A.q3_rsa": "space-fix superseded by SF.q3_rsa (paper: 2/6)",
    "A.q3_enc": "space-fix superseded by SF.q3_enc (paper: 4/6)",
    "A.alpha_72b": "space-fix superseded by SF.alpha_72b (11/12 also on the re-extracted 72B)",
    "A.q1m_enc_rho": "space-fix superseded by SF.q1m_enc_gate_lo (encoding arm now fails F3, indeterminate)",
    "A.q1m_enc_ci_lo": "space-fix superseded by SF.q1m_enc_gate_lo (encoding arm indeterminate)",
    "A.q1m_enc_ci_hi": "space-fix superseded by SF.q1m_enc_gate_hi (encoding arm indeterminate)",
    "A.q1m_enc_pair_p": "space-fix superseded by SF.q1m_rsa_pair_p (paper reports the similarity arm)",
    "A.q1m_enc_pair_diff": "space-fix superseded by SF.q1m_rsa_pair_diff",
    "A.q1m_enc_pair_pos": "space-fix superseded by SF.q1m_rsa_pair_pos",
    "A.q1m_enc_gate_lo": "space-fix superseded by SF.q1m_enc_gate_lo (paper: -0.16)",
    "A.q1m_enc_sb": "space-fix superseded by SF.q1m_enc_sb (paper: 0.70)",
    "A.q1m_rsa_sb": "space-fix superseded by SF.q1m_rsa_sb (paper: 0.83)",
    "A.q1m_rsa_gate_lo": "space-fix superseded by SF.q1m_rsa_gate_lo (paper: 0.13)",
    "masc27.concordance": "space-fix superseded by SF.masc27_concordance (paper: +0.446)",
    "cohort.brennan": "space-fix superseded by SF.brennan_concordance (paper: +0.366)",
    "brennan.concordance": "space-fix superseded by SF.brennan_concordance (paper: +0.366)",
}

# Result files written from features that included unspaced (space-dropped) checkpoints. A claim
# reading one of them must be listed in SUPERSEDED, otherwise the run fails: a stale value cannot
# silently pass as a current paper number.
SPACE_FIX_STALE_SOURCES = {
    "p4_encoding/adr0006_registered_verdicts.json",
    "p4_encoding/adr0006_registered_verdicts_v2.json",
    "p4_encoding/adr0010_q1_masc_verdicts.json",
    "p4_encoding/concordance32_masc27.json",
    "p4_encoding/concordance32_brennan_v2.json",
}


def space_fix_guard(claims, sources):
    """Problems (list of str): stale-source claims not flagged SUPERSEDED, and space-fix SUPERSEDED
    entries whose named SF.* sibling is missing from CLAIMS."""
    ids = {c for c, _, _ in claims}
    probs = []
    for cid, src in sources.items():
        if src in SPACE_FIX_STALE_SOURCES and cid not in SUPERSEDED:
            probs.append(f"{cid}: reads pre-space-fix {src} but is not marked SUPERSEDED")
    for cid, why in SUPERSEDED.items():
        if why.startswith("space-fix superseded by "):
            sib = why.split("by ", 1)[1].split()[0]
            if sib not in ids:
                probs.append(f"{cid}: names sibling {sib}, which is not a claim")
            if cid not in ids:
                probs.append(f"SUPERSEDED lists {cid}, which is not a claim")
    return probs


def claim_scope(cid):
    """'paperA' (benchmark paper), 'superseded' (provenance pin) or 'companion' (a separate
    boundary-effect analysis on the same cohorts, not part of the benchmark submission)."""
    if cid in SUPERSEDED:
        return "superseded"
    if cid.startswith("B."):
        return "companion"
    return "paperA"


def emit_tables():
    """Regenerate the paper's data tables from results/ (Tables 2, ceiling, scorecard)."""
    OUT.mkdir(exist_ok=True)
    _tsv("cross_cohort_summary.tsv").to_csv(OUT / "table_percohort_leaderboard.tsv", sep="\t", index=False)
    _tsv("scorecard/scorecard.tsv").to_csv(OUT / "table_scorecard.tsv", sep="\t", index=False)
    pb = _json("p2_perband_ceiling.json")["per_band"]
    pd.DataFrame(
        [
            {
                "band": b,
                "band_hz": v.get("band_hz"),
                "ceiling_SB_raw": v.get("ceiling_SB_raw"),
                "alignment_partial": v.get("alignment_partial"),
            }
            for b, v in pb.items()
        ]
    ).to_csv(OUT / "table_perband_ceiling.tsv", sep="\t", index=False)
    print(f"regenerated tables -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-tables", action="store_true", help="also regenerate the paper's tables")
    ap.add_argument("--json", metavar="PATH", help="write the reproduction report as JSON")
    args = ap.parse_args()

    print(f"results tree: {RESULTS}")
    print(f"{'claim':34s} {'recomputed':>13s} {'paper':>10s}  status")
    print("-" * 76)

    report, n_fail = [], 0
    for cid, desc, fn in CLAIMS:
        try:
            got, want, tol, src = fn()
        except Exception as e:  # missing file or schema drift
            print(f"{cid:34s} {'ERROR':>13s} {'-':>10s}  {type(e).__name__}: {str(e)[:30]}")
            report.append({"claim": cid, "description": desc, "status": "error", "error": str(e)})
            n_fail += 1
            continue
        ok = abs(got - want) <= tol
        n_fail += not ok
        gs = f"{got:.5g}" if isinstance(got, float) else str(got)
        ws = f"{want:.5g}" if isinstance(want, float) else str(want)
        print(f"{cid:34s} {gs:>13s} {ws:>10s}  {'OK' if ok else 'MISMATCH'}")
        report.append(
            {
                "claim": cid,
                "description": desc,
                "recomputed": got,
                "paper": want,
                "tolerance": tol,
                "source": src,
                "scope": claim_scope(cid),
                "status": "ok" if ok else "mismatch",
            }
        )

    print("-" * 76)
    guard = space_fix_guard(CLAIMS, {r["claim"]: r.get("source") for r in report})
    for g in guard:
        print(f"SPACE-FIX GUARD FAIL: {g}")
    n_fail += len(guard)
    scope = collections.Counter(claim_scope(c) for c, _, _ in CLAIMS)
    print(f"scope: {scope['paperA']} benchmark-paper claims, {scope['superseded']} superseded-value "
          f"provenance pins, {scope['companion']} companion boundary-analysis claims "
          f"(see SUPERSEDED / claim_scope in bench/reproduce.py)")
    print(f"{len(CLAIMS) - (n_fail - len(guard))}/{len(CLAIMS)} claims reproduce from {RESULTS.name}/"
          + ("" if not guard else f"; {len(guard)} space-fix guard failure(s)"))

    if args.emit_tables:
        emit_tables()
    if args.json:
        json.dump(report, open(args.json, "w"), indent=2)
        print(f"report -> {args.json}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
