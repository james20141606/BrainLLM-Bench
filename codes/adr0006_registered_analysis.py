#!/usr/bin/env python
"""ADR 0006 registered analysis: Q1 (scaling), Q2 (reversal), Q3 (instruct twins).

Written and committed BEFORE the final concordance run over the complete modern set
exists; the decision rules below transcribe docs/adr/0006 verbatim. Run on the
FINAL concordance JSON (per_session_rsa / per_session_enc arms) once every ADR
model has features.

Registration note (recorded in the ELN of 2026-08-13): the ADR's "(32 -> 41)" was
an arithmetic slip; its frozen list implies 32 + 7 base additions. The analysis
follows the ADR's DEFINITION (base checkpoints join the concordance set) and this
script reports the count it actually finds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path("<REPO_ROOT>")
sys.path.insert(0, str(ROOT / "codes"))
from recompute_corrected_stats import family_map  # noqa: E402

LADDERS = {  # ADR 0006 + ADR 0008: registered base-only ladders, one Q1 verdict each
    "qwen2.5": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
                "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B",
                "Qwen/Qwen2.5-72B"],
    "qwen3": ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base",
              "Qwen/Qwen3-8B-Base", "Qwen/Qwen3-14B-Base"],
    "qwen3.5": ["Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-4B-Base",
                "Qwen/Qwen3.5-9B-Base"],
}
POSTTRAINED = {"Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B",
               "Qwen/Qwen3.8-27B"}
INSTRUCT_PAIRS = [("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
                  ("Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct"),
                  ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
                  ("meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-70B-Instruct"),
                  ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"),
                  ("Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B")]
INSTRUCT = {b for _, b in INSTRUCT_PAIRS}
N_BOOT = 10000


def sess_matrix(ps, models):
    sess = sorted(set.intersection(*[set(ps[m]) for m in models]))
    return np.array([[ps[m][s] for s in sess] for m in models], float), sess


def q1_scaling(rsa_ps, enc_ps, out, gen="qwen2.5"):
    ladder = [m for m in LADDERS[gen] if m in rsa_ps and m in enc_ps]
    out[f"q1_{gen}_ladder_models"] = ladder
    if len(ladder) < 4:
        out[f"q1_{gen}_verdict"] = f"INDETERMINATE: only {len(ladder)} ladder rungs scored"
        return
    order = np.arange(len(ladder))
    res = {}
    for name, ps in (("rsa", rsa_ps), ("enc", enc_ps)):
        M, sess = sess_matrix(ps, ladder)          # models x sessions
        n_s = M.shape[1]
        rho_pt = float(spearmanr(order, M.mean(1))[0])
        rng = np.random.default_rng(6)
        boots = [spearmanr(order, M[:, rng.integers(0, n_s, n_s)].mean(1))[0]
                 for _ in range(N_BOOT)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        top_vs_bottom = M[-1] - M[0]               # per-session paired diff
        try:
            p_pair = float(wilcoxon(top_vs_bottom).pvalue)
        except ValueError:
            p_pair = float("nan")
        res[name] = {"rho": rho_pt, "ci95_session_clustered": ci,
                     "top_minus_bottom_mean": float(top_vs_bottom.mean()),
                     "top_minus_bottom_p": p_pair, "n_sessions": n_s}
    out[f"q1_{gen}"] = res
    sup = all(r["rho"] > 0 and r["ci95_session_clustered"][0] > 0 for r in res.values())
    contra = (not any(r["ci95_session_clustered"][0] > 0 for r in res.values())) or \
             any(r["rho"] < 0 for r in res.values())
    if sup:
        out[f"q1_{gen}_verdict"] = ("SUPPORTED over the modern range: scorecard verdict becomes "
                                    "'contradicted below 3B, supported over the modern ladder' "
                                    "(refine)")
    elif contra:
        out[f"q1_{gen}_verdict"] = ("NOT SUPPORTED over the modern range: size does not order "
                                    "brain alignment up to the largest rung scored; state the CI")
    else:
        out[f"q1_{gen}_verdict"] = "MIXED: one readout orders, one does not; report both CIs"


def q2_reversal(rsa_ps, enc_ps, out):
    base = sorted((set(rsa_ps) & set(enc_ps)) - INSTRUCT - POSTTRAINED)
    fam = family_map(base)
    M_r, sess = sess_matrix(rsa_ps, base)
    M_e, _ = sess_matrix(enc_ps, base)
    a, b = M_r.mean(1), M_e.mean(1)
    rho_pt = float(spearmanr(a, b)[0])
    fams = sorted(set(fam.values()))
    byfam = {f: [i for i, m in enumerate(base) if fam[m] == f] for f in fams}
    rng = np.random.default_rng(6)
    boots = []
    for _ in range(N_BOOT):
        idx = [i for f in rng.choice(fams, len(fams), replace=True) for i in byfam[f]]
        if len(set(a[idx])) > 2:
            boots.append(spearmanr(a[idx], b[idx])[0])
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    out["q2"] = {"n_base": len(base), "n_families": len(fams), "rho": rho_pt,
                 "ci95_family_clustered": ci}
    if rho_pt > 0 and ci[0] > 0:
        out["q2_verdict"] = ("F1 FIRED: the published reversal is an artifact of the 2019-era "
                             "model set; declare in the abstract as registered")
    elif rho_pt < 0 and ci[1] < 0:
        out["q2_verdict"] = "REVERSAL SURVIVES modern models (negative, family-clustered CI < 0)"
    else:
        out["q2_verdict"] = "REVERSAL ATTENUATES: report the CI; neither registered pole met"


def q3_instruct(rsa_ps, enc_ps, out):
    rows = {}
    for base_m, ins in INSTRUCT_PAIRS:
        if not all(m in rsa_ps and m in enc_ps for m in (base_m, ins)):
            rows[ins] = "not scored"
            continue
        r = {}
        for name, ps in (("rsa", rsa_ps), ("enc", enc_ps)):
            sess = sorted(set(ps[base_m]) & set(ps[ins]))
            d = np.array([ps[ins][s] - ps[base_m][s] for s in sess], float)
            n = len(d)
            rng = np.random.default_rng(6)
            boots = [d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
            r[name] = {"mean_diff": float(d.mean()),
                       "ci95": [float(np.percentile(boots, 2.5)),
                                float(np.percentile(boots, 97.5))]}
        rows[ins] = r
    out["q3_exploratory"] = rows
    for name in ("rsa", "enc"):
        signs = [np.sign(v[name]["mean_diff"]) for v in rows.values() if isinstance(v, dict)]
        out[f"q3_direction_count_{name}"] = f"{int(sum(s > 0 for s in signs))}/{len(signs)} instruct > base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="final concordance JSON with per_session arms")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.loads(Path(a.json).read_text())
    rsa_ps, enc_ps = r["per_session_rsa"], r["per_session_enc"]
    out = {"source": a.json, "n_scored": len(set(rsa_ps) & set(enc_ps))}
    for gen in LADDERS:
        q1_scaling(rsa_ps, enc_ps, out, gen)
    q2_reversal(rsa_ps, enc_ps, out)
    q3_instruct(rsa_ps, enc_ps, out)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("q3_exploratory")},
                     indent=2))
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
