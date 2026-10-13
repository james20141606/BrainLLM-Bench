#!/usr/bin/env python
"""paperA rebuttal: aggregate the Qwen2.5 ladder arm C (cached features; unscaled C0 vs per-session scaled Cz).

Prereg: paper4_temporalalign/notes/PREREG_paperA_qwen_ladder_rescaled_20261005.md (written before code).
T1/T2 exactly as codes/adr0006_registered_analysis.q1_scaling (order = rung index, 10,000 session resamples with
numpy default_rng(6), percentile CI; Wilcoxon on 72B - 0.5B per session). T3: rho(Cz) - rho(C0) on the same
resamples. Branch rule §4 (checked in the order reverses, holds, vanishes, weakens). Output: qwen_ladder_summary_20261005.json.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
P4 = HERE.parents[1]
N_BOOT = 10000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trend(M, idx_list):
    order = np.arange(M.shape[0])
    rho = float(spearmanr(order, M.mean(1))[0])
    boots = [spearmanr(order, M[:, i].mean(1))[0] for i in idx_list]
    d = M[-1] - M[0]
    return {"rho": rho, "ci95_session_clustered": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "top_minus_bottom_mean": float(d.mean()), "top_minus_bottom_wilcoxon_p": float(wilcoxon(d).pvalue),
            "per_rung_mean": M.mean(1).tolist()}, np.array(boots)


def branch(t):
    rho, (L, U), d, p = t["rho"], t["ci95_session_clustered"], t["top_minus_bottom_mean"], t["top_minus_bottom_wilcoxon_p"]
    if (rho > 0 and L > 0) or (d > 0 and p < 0.05):
        return "reverses"
    if rho < 0 and U < 0 and d < 0 and p < 0.05:
        return "holds"
    if rho >= 0 and L <= 0 <= U and p >= 0.05:
        return "vanishes"
    return "weakens"


def main():
    refs = json.loads((HERE / "qwen_ladder_refs.json").read_text())
    ladder = refs["ladder"]
    recs = {}
    for f in sorted(glob.glob(str(HERE / "qwen_ladder_raw" / "*.json"))):
        j = json.loads(Path(f).read_text()); recs[j["model"]] = j
    missing = [m for m in ladder if m not in recs]
    g2fail = [m for m in recs if not recs[m]["G2_pass"]]
    g3fail = [m for m in recs if recs[m]["G3_feature_identity"] is False]
    out = {"prereg": "paper4_temporalalign/notes/PREREG_paperA_qwen_ladder_rescaled_20261005.md", "arm": "C",
           "missing": missing, "G2_fail": g2fail, "G3_fail": g3fail,
           "inputs_sha256": {Path(f).name: sha(f) for f in sorted(glob.glob(str(HERE / "qwen_ladder_raw" / "*.json")))}}
    excl = sorted(set(missing + g2fail + g3fail))
    if len(excl) >= 2:
        out["branch"] = "INCOMPLETE"
    else:
        use = [m for m in ladder if m not in excl]
        sess = sorted(recs[use[0]]["C0_unscaled"], key=int)
        M0 = np.array([[recs[m]["C0_unscaled"][s]["score"] for s in sess] for m in use])
        Mz = np.array([[recs[m]["Cz_scaled"][s]["score"] for s in sess] for m in use])
        rng = np.random.default_rng(6)
        idx = [rng.integers(0, len(sess), len(sess)) for _ in range(N_BOOT)]
        t0, b0 = trend(M0, idx)
        tz, bz = trend(Mz, idx)
        dd = bz - b0
        adr = json.loads((P4 / "results/p4_encoding/space_fix_20260925/aggregate/adr0006_registered_verdicts_spacefix.json").read_text())
        rsa = adr["q1_qwen2.5"]["rsa"]
        sup = rsa["rho"] > 0 and rsa["ci95_session_clustered"][0] > 0 and tz["rho"] > 0 and tz["ci95_session_clustered"][0] > 0
        out.update({"excluded": excl, "rungs": use, "sessions": sess,
                    "C0_unscaled": {**t0, "branch_same_rule": branch(t0)},
                    "Cz_scaled": {**tz, "branch_same_rule": branch(tz)},
                    "T3_delta_rho": tz["rho"] - t0["rho"],
                    "T3_delta_ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                    "published_registered": adr["q1_qwen2.5"]["enc"],
                    "Q1_rule_with_Cz_enc_and_published_rsa": "SUPPORTED" if sup else "NOT SUPPORTED (rule unchanged)",
                    "branch": branch(tz),
                    "folds_at_1e6": {m: {"C0": int(sum(v["alpha"] == 1e6 for v in recs[m]["C0_unscaled"].values())),
                                         "Cz": int(sum(v["alpha"] == 1e6 for v in recs[m]["Cz_scaled"].values()))} for m in use},
                    "G2_max_abs_diff": {m: recs[m]["G2_max_abs_diff"] for m in use},
                    "jids": {m: recs[m]["jid"] for m in use}})
    (HERE / "qwen_ladder_summary_20261005.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out.get(k) for k in ("branch", "excluded", "T3_delta_rho", "T3_delta_ci95", "Q1_rule_with_Cz_enc_and_published_rsa")}, indent=1))
    for k in ("C0_unscaled", "Cz_scaled"):
        if k in out:
            print(k, {kk: out[k][kk] for kk in ("rho", "ci95_session_clustered", "top_minus_bottom_mean", "top_minus_bottom_wilcoxon_p", "branch_same_rule")})


if __name__ == "__main__":
    main()
