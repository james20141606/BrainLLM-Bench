#!/usr/bin/env python
"""ADR 0010 registered analysis: Q1-MASC (Qwen2.5 ladder on MEG-MASC merged-27).

The statistic is ADR 0006's Q1 verbatim (adr0006_registered_analysis.q1_scaling)
with subject (27) replacing session (12) as the cluster unit; the input JSON's
per_session arms are subject-keyed because the scoring run used --group_col
subject. The F3 rankability gate consumes the scoring run's own split-half
leaderboard reliability (computed over exactly the 7 ladder rungs), 90% CI must
exclude zero per readout, per ADR 0010's gate clause. The LibriBrain verdicts of
ADR 0006 are final and are quoted from the pinned v1 verdicts file, not rerun.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path("<REPO_ROOT>")
LADDER = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
          "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B",
          "Qwen/Qwen2.5-72B"]
N_BOOT = 10000


def subj_matrix(ps, models):
    subj = sorted(set.intersection(*[set(ps[m]) for m in models]))
    return np.array([[ps[m][s] for s in subj] for m in models], float), subj


def q1_masc(rsa_ps, enc_ps, out):
    ladder = [m for m in LADDER if m in rsa_ps and m in enc_ps]
    out["q1_masc_ladder_models"] = ladder
    if len(ladder) < 4:
        out["q1_masc_verdict"] = f"INDETERMINATE: only {len(ladder)} ladder rungs scored"
        return
    order = np.arange(len(ladder))
    res = {}
    for name, ps in (("rsa", rsa_ps), ("enc", enc_ps)):
        M, subj = subj_matrix(ps, ladder)          # models x subjects
        n_s = M.shape[1]
        rho_pt = float(spearmanr(order, M.mean(1))[0])
        rng = np.random.default_rng(6)
        boots = [spearmanr(order, M[:, rng.integers(0, n_s, n_s)].mean(1))[0]
                 for _ in range(N_BOOT)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        top_vs_bottom = M[-1] - M[0]               # per-subject paired diff
        try:
            p_pair = float(wilcoxon(top_vs_bottom).pvalue)
        except ValueError:
            p_pair = float("nan")
        res[name] = {"rho": rho_pt, "ci95_subject_clustered": ci,
                     "top_minus_bottom_mean": float(top_vs_bottom.mean()),
                     "top_minus_bottom_p": p_pair, "n_subjects": n_s,
                     "n_positive_top_minus_bottom": int((top_vs_bottom > 0).sum())}
    out["q1_masc"] = res
    sup = all(r["rho"] > 0 and r["ci95_subject_clustered"][0] > 0 for r in res.values())
    contra = (not any(r["ci95_subject_clustered"][0] > 0 for r in res.values())) or \
             any(r["rho"] < 0 for r in res.values())
    if sup:
        out["q1_masc_verdict"] = ("SUPPORTED over the modern range: scorecard verdict becomes "
                                  "'contradicted below 3B, supported over the modern ladder' "
                                  "(refine)")
    elif contra:
        out["q1_masc_verdict"] = ("NOT SUPPORTED over the modern range: size does not order "
                                  "brain alignment up to the largest rung scored; state the CI")
    else:
        out["q1_masc_verdict"] = "MIXED: one readout orders, one does not; report both CIs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "paper4_temporalalign/results/p4_encoding/"
                                                 "concordance_masc27_ladder.json"))
    ap.add_argument("--libribrain", default=str(ROOT / "paper4_temporalalign/results/"
                                                       "p4_encoding/adr0006_registered_verdicts.json"))
    ap.add_argument("--out", default=str(ROOT / "paper4_temporalalign/results/p4_encoding/"
                                                "adr0010_q1_masc_verdicts.json"))
    a = ap.parse_args()
    r = json.loads(Path(a.json).read_text())
    lb = json.loads(Path(a.libribrain).read_text())

    out = {"source": a.json, "libribrain_source": a.libribrain,
           "cell": r["cell"], "n_models": r["n_models"]}

    # F3 rankability gate, per readout, from the scoring run's own split-half
    # reliability over exactly the 7 ladder rungs (ADR 0010 gate clause).
    gate = {}
    for name, key in (("rsa", "rsa_reliability"), ("enc", "enc_reliability")):
        sb, ci90 = r[f"{key}_sb"], r[f"{key}_ci90"]
        gate[name] = {"split_half": sb, "ci90": ci90, "passes": bool(ci90[0] > 0)}
    out["f3_gate"] = gate

    q1_masc(r["per_session_rsa"], r["per_session_enc"], out)

    # ADR 0010 cross-cohort statement, applied strictly per readout.
    lb_verdict = lb["q1_qwen2.5_verdict"]
    lb_enc_rho = lb["q1_qwen2.5"]["enc"]["rho"]
    out["libribrain_q1_verdict"] = lb_verdict
    masc_enc = out["q1_masc"]["enc"]
    lines = []
    if not gate["rsa"]["passes"]:
        lines.append("RSA arm: INDETERMINATE by the registered F3 gate (MASC ladder "
                     f"similarity leaderboard split-half {gate['rsa']['split_half']:.3f}, "
                     f"CI90 [{gate['rsa']['ci90'][0]:.3f}, {gate['rsa']['ci90'][1]:.3f}] "
                     "does not exclude zero); not folded into either reading.")
    if gate["enc"]["passes"]:
        both_ns = ("NOT SUPPORTED" in lb_verdict) and \
                  (masc_enc["ci95_subject_clustered"][0] <= 0 or masc_enc["rho"] < 0)
        enc_neg_both = lb_enc_rho < 0 and masc_enc["rho"] < 0
        if both_ns and enc_neg_both:
            lines.append("Encoding arm: size fails to order alignment in two cohorts with "
                         "independent listeners -- the scaling verdict generalizes "
                         "(pre-committed reading 1).")
        elif both_ns:
            lines.append("Encoding arm: NOT SUPPORTED in both cohorts (LibriBrain "
                         f"rho={lb_enc_rho:+.2f} declining; MASC rho={masc_enc['rho']:+.2f}, "
                         f"CI [{masc_enc['ci95_subject_clustered'][0]:+.2f}, "
                         f"{masc_enc['ci95_subject_clustered'][1]:+.2f}] crossing zero, flat). "
                         "Pre-committed reading 1's parenthetical (encoding negative in both) "
                         "is not met: size fails to order encoding alignment in either cohort, "
                         "but the failure shapes differ (decline vs. flat); reported as such, "
                         "with neither cohort privileged.")
        else:
            lines.append("Encoding arm: the scaling verdict is itself cohort-dependent "
                         "(pre-committed reading 2).")
    out["cross_cohort_statement"] = lines
    print(json.dumps(out, indent=2))
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
