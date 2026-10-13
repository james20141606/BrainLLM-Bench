#!/usr/bin/env python
"""paperA rebuttal check (i) aggregation: diag branch + §1.6 per-session-scaled re-score of the 32 checkpoints.

Inputs (pulled from <cluster> $HOME/p4_rebuttal_20261005/session9/): diag.json, rescore/<ii>_<tag>.json (32).
Estimators are the headline's (recompute_corrected_stats.family_clustered_ci, seed 0 and seeds 0-19;
paired delta from p4_strengthen_aggregate_20260926.paired_delta). Branch rule: PREREG §1.6.
Output: session9_summary_20261005.json next to this file.
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEG = HERE.parents[2]
sys.path.insert(0, str(MEG / "codes"))
import recompute_corrected_stats as RC                                   # noqa: E402
from p4_strengthen_aggregate_20260926 import concordance, paired_delta    # noqa: E402

HEAD = MEG / "paper4_temporalalign" / "results" / "p4_encoding" / "concordance32_CORRECTED.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    raw = HERE / "session9_raw"
    diag = json.loads((raw / "diag.json").read_text())
    head = json.loads(HEAD.read_text())
    models = head["models"]
    rsa = {m: head["per_session_rsa"][m] for m in models}
    enc0 = {m: head["per_session_enc"][m] for m in models}
    recs = {}
    for f in sorted(glob.glob(str(raw / "rescore" / "*.json"))):
        j = json.loads(Path(f).read_text()); recs[j["model"]] = j
    missing = [m for m in models if m not in recs]
    g2fail = [m for m in recs if not recs[m]["G2_pass"]]
    out = {"prereg": "paper4_temporalalign/notes/PREREG_paperA_rebuttal_checks_20261004.md §1.6",
           "diag_branch": diag["branch"], "diag_named": diag["named_sessions"], "r_k": diag["D1_r_k"],
           "missing": missing, "G2_fail": g2fail,
           "inputs_sha256": {"diag.json": sha(raw / "diag.json"),
                             **{Path(f).name: sha(f) for f in sorted(glob.glob(str(raw / "rescore" / "*.json")))}}}
    excl = missing + g2fail
    if len(excl) >= 3:
        out["branch"] = "INCOMPLETE"
    else:
        use = [m for m in models if m not in excl]
        encz = {m: recs[m]["per_session_scaled"] and {s: v["score"] for s, v in recs[m]["per_session_scaled"].items()} for m in use}
        c0 = concordance(rsa, enc0, use)
        cz = concordance(rsa, encz, use)
        dci, nd = paired_delta(rsa, enc0, rsa, encz, use)
        seeds = []
        for s in range(20):
            _, lo, hi, _ = RC.family_clustered_ci({m: rsa[m] for m in use}, encz, seed=s)
            seeds.append([lo, hi])
        def br(rho, L, U):
            return "A" if U < 0 else ("B" if rho < 0 and L <= 0 <= U else "C")
        b0 = br(cz["rho"], *cz["family_ci95"])
        bs = [br(cz["rho"], lo, hi) for lo, hi in seeds]
        sub = ("undiminished" if dci[0] <= 0 <= dci[1] else "weakened" if dci[0] > 0 else "strengthened")
        out.update({
            "excluded": excl, "comparator": c0, "per_session_scaled": cz,
            "delta_vs_comparator": cz["rho"] - c0["rho"], "delta_paired_family_ci95": dci, "delta_n_boot_valid": nd,
            "seed_sweep_0_19": {"lower_range": [min(x[0] for x in seeds), max(x[0] for x in seeds)],
                                "upper_range": [min(x[1] for x in seeds), max(x[1] for x in seeds)],
                                "branches": {b: bs.count(b) for b in set(bs)}},
            "branch": b0 if len(set(bs)) == 1 else f"seed-dependent ({ {b: bs.count(b) for b in set(bs)} })",
            "sub_label": sub if b0 == "A" else None,
            "folds_at_1e6_scaled": {m: int(sum(v["alpha"] == 1e6 for v in recs[m]["per_session_scaled"].values())) for m in use},
            "folds_at_1e6_unscaled": {m: int(sum(v["alpha"] == 1e6 for v in recs[m]["unscaled"].values())) for m in use},
            "session9_enc_scaled": {m: recs[m]["per_session_scaled"]["9"]["score"] for m in use},
            "session9_enc_unscaled": {m: recs[m]["unscaled"]["9"]["score"] for m in use},
            "enc_session_means_scaled": {s: float(np.mean([encz[m][s] for m in use])) for s in encz[use[0]]},
            "per_model": {m: {"enc_mean_unscaled": recs[m]["enc_mean_unscaled"], "enc_mean_scaled": recs[m]["enc_mean_scaled"],
                              "G2_max_abs_diff": recs[m]["G2_max_abs_diff"], "jid": recs[m]["jid"]} for m in use}})
    (HERE / "session9_summary_20261005.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out.get(k) for k in ("diag_branch", "branch", "sub_label", "excluded", "delta_paired_family_ci95", "seed_sweep_0_19")}, indent=1))
    if "per_session_scaled" in out:
        print("rho_z", out["per_session_scaled"]["rho"], out["per_session_scaled"]["family_ci95"])


if __name__ == "__main__":
    main()
