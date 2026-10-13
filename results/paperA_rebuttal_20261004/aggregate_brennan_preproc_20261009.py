#!/usr/bin/env python
"""paperA K0/K1 aggregation and registered decision (PREREG_paperA_brennan_preproc_20261009.md §1, §2).

Input: brennan_preproc_raw/k0k1.json (cluster job). Published comparators: results/p4_encoding/concordance32_brennan_v2.json
(per-listener RSA/ENC, cached features) and noise_sweep/brennan_p4_v2_dose0_seed0.json (RDM reliability 0.8582017).
Production estimators: recompute_corrected_stats.fisher_ci / family_clustered_ci; concordance_decomposition.py run
unchanged on per-model files built from RSA_x and ENC_published. Written before the receipt exists.
Output: brennan_preproc_summary_20261009.json.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
MEG = HERE.parents[2]
P4E = MEG / "paper4_temporalalign/results/p4_encoding"
sys.path.insert(0, str(MEG / "codes"))
import recompute_corrected_stats as RC                                   # noqa: E402

TREAT = ("a", "b", "c")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cat(lo, hi):
    return "above" if lo > 0 else ("below" if hi < 0 else "includes0")


def main():
    k = json.loads((HERE / "brennan_preproc_raw" / "k0k1.json").read_text())
    v2 = json.loads((P4E / "concordance32_brennan_v2.json").read_text())
    rel_pub = json.loads((P4E / "noise_sweep/brennan_p4_v2_dose0_seed0.json").read_text())["neural_rdm_reliability"]
    per = k["per_listener"]
    subs = sorted(per)
    models = list(k["model_kind"])
    kind = k["model_kind"]
    out = {"prereg": "paper4_temporalalign/notes/PREREG_paperA_brennan_preproc_20261009.md",
           "inputs_sha256": {"k0k1.json": sha(HERE / "brennan_preproc_raw/k0k1.json"),
                             "concordance32_brennan_v2.json": sha(P4E / "concordance32_brennan_v2.json")},
           "n_listeners": len(subs), "jid": k["jid"]}
    # gates
    out["gate_pattern_a_identical_all"] = all(per[s]["gate_pattern_a_identical"] for s in subs)
    g2 = {m: max(abs(per[s]["rsa"]["a"][m] - v2["per_session_rsa"][m][s]) for s in subs) for m in models}
    out["G2_max_abs_diff"] = max(g2.values())
    out["G2_fail"] = [m for m, v in g2.items() if v > 1e-8]
    # K0
    k0 = {}
    for t in TREAT:
        sh = np.array([per[s]["K0"][t]["var_share_channel_means_mean"] for s in subs])
        sdt = np.array([per[s]["K0"][t]["spearman_Db_dt"] for s in subs])
        ssn = np.array([per[s]["K0"][t]["spearman_Db_same_sentence"] for s in subs])
        rel = [per[s]["K0"][t]["rdm_reliability_group_sb"] for s in subs]
        rel = [v for v in rel if np.isfinite(v)]
        k0[t] = {"var_share_median": float(np.median(sh)), "var_share_range": [float(sh.min()), float(sh.max())],
                 "spearman_Db_dt_median": float(np.median(sdt)), "spearman_Db_dt_n_pos": int((sdt > 0).sum()),
                 "spearman_Db_dt_range": [float(sdt.min()), float(sdt.max())],
                 "spearman_Db_same_sentence_median": float(np.median(ssn)),
                 "rdm_reliability": float(np.mean(rel)),
                 "rdm_reliability_p5_p95": [float(np.percentile(rel, 5)), float(np.percentile(rel, 95))]}
    out["K0"] = k0
    out["K0_gate_rdm_reliability_reproduces_0.8582"] = bool(abs(k0["a"]["rdm_reliability"] - rel_pub) <= 1e-6)
    a = k0["a"]
    drop_ok = all(k0[t]["spearman_Db_dt_median"] <= 0.5 * a["spearman_Db_dt_median"] for t in ("b", "c"))
    if a["var_share_median"] >= 0.5 and a["spearman_Db_dt_median"] >= 0.05 and a["spearman_Db_dt_n_pos"] >= 40 and drop_ok:
        out["K0_branch"] = "drift mechanism supported"
    elif a["spearman_Db_dt_median"] < 0.02 or a["var_share_median"] < 0.2:
        out["K0_branch"] = "not supported"
    else:
        out["K0_branch"] = "partial"
    dm = k["Dm_vs_time_S01"]
    out["Dm_vs_time_S01"] = {"median_spearman_Dm_dt": float(np.median([v["spearman_Dm_dt"] for v in dm.values()])),
                             "range": [float(min(v["spearman_Dm_dt"] for v in dm.values())),
                                       float(max(v["spearman_Dm_dt"] for v in dm.values()))],
                             "by_class_median": {c: float(np.median([dm[m]["spearman_Dm_dt"] for m in models if kind[m] == c]))
                                                 for c in ("causal", "masked", "encdec")},
                             "per_model": dm}
    # K1
    enc_ps = {m: v2["per_session_enc"][m] for m in models}
    enc_mean = np.array([np.mean([enc_ps[m][s] for s in subs]) for m in models])
    rsa_mean, k1 = {}, {}
    for t in TREAT:
        rps = {m: {s: per[s]["rsa"][t][m] for s in subs} for m in models}
        rsa_mean[t] = np.array([np.mean(list(rps[m].values())) for m in models])
        rho = float(spearmanr(rsa_mean[t], enc_mean)[0])
        fis = RC.fisher_ci(rho, len(models))
        r0, lo, hi, nf = RC.family_clustered_ci(rps, enc_ps, seed=0)
        sweep = [RC.family_clustered_ci(rps, enc_ps, seed=sd)[1:3] for sd in range(20)]
        with tempfile.TemporaryDirectory() as td:
            pm = [{"model": m, "kind": kind[m], "rsa_mean": float(rsa_mean[t][i]), "enc_mean": float(enc_mean[i])}
                  for i, m in enumerate(models)]
            inp, outp = Path(td) / "in.json", Path(td) / "out.json"
            inp.write_text(json.dumps({"per_model": pm}))
            subprocess.run([sys.executable, str(MEG / "codes/concordance_decomposition.py"), str(inp),
                            "--label", f"Brennan K1 {t}", "--out", str(outp)], check=True, capture_output=True,
                           cwd=str(MEG / "codes"))
            dec = json.loads(outp.read_text())
        k1[t] = {"kappa": rho, "fisher_ci95": list(fis), "family_ci95_seed0": [lo, hi],
                 "family_seed_sweep": {"lower_range": [min(x[0] for x in sweep), max(x[0] for x in sweep)],
                                       "upper_range": [min(x[1] for x in sweep), max(x[1] for x in sweep)]},
                 "cat_fisher": cat(*fis), "cat_family": cat(lo, hi),
                 "class_mean_rsa": {c: float(np.mean([rsa_mean[t][i] for i, m in enumerate(models) if kind[m] == c]))
                                    for c in ("causal", "masked", "encdec")},
                 "class_gap": dec["class_gap_causal_minus_other"], "class_gap_p_family": dec["gap_p_family_clustered"],
                 "per_model_rsa_mean": dict(zip(models, map(float, rsa_mean[t])))}
    out["K1"] = k1
    out["S_b"] = float(spearmanr(rsa_mean["a"], rsa_mean["b"])[0])
    out["S_c"] = float(spearmanr(rsa_mean["a"], rsa_mean["c"])[0])
    out["S_b_vs_c"] = float(spearmanr(rsa_mean["b"], rsa_mean["c"])[0])
    unaffected = (out["S_b"] >= 0.9 and out["S_c"] >= 0.9 and
                  all(k1[t]["cat_fisher"] == k1["a"]["cat_fisher"] and k1[t]["cat_family"] == k1["a"]["cat_family"]
                      and np.sign(k1[t]["class_gap"]) == np.sign(k1["a"]["class_gap"]) for t in ("b", "c")))
    if len(out["G2_fail"]) >= 3 or not out["gate_pattern_a_identical_all"]:
        out["K1_branch"] = "INCOMPLETE"
    else:
        out["K1_branch"] = "unaffected" if unaffected else "affected"
    (HERE / "brennan_preproc_summary_20261009.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({x: out[x] for x in ("G2_max_abs_diff", "G2_fail", "gate_pattern_a_identical_all",
                                          "K0_gate_rdm_reliability_reproduces_0.8582", "K0_branch", "S_b", "S_c",
                                          "S_b_vs_c", "K1_branch")}, indent=1))
    for t in TREAT:
        print(t, {x: k1[t][x] for x in ("kappa", "fisher_ci95", "family_ci95_seed0", "class_mean_rsa", "class_gap", "class_gap_p_family")})
        print("  K0", k0[t])
    print("Dm", {x: out["Dm_vs_time_S01"][x] for x in ("median_spearman_Dm_dt", "range", "by_class_median")})


if __name__ == "__main__":
    main()
