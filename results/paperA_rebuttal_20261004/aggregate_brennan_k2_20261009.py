#!/usr/bin/env python
"""paperA K2 aggregation (PREREG_paperA_brennan_preproc_20261009.md §3 + 19:15 note). Written before any K2 receipt.

Inputs (pulled from <cluster> ~/p4_rebuttal_20261005/brennan_k2/): conc/brennan_{b,c}_shard{0-3}.json (partial_out format:
rows, rsa_ps, enc_ps), cross/*.json, repeat/*.json. Also brennan_preproc_raw/k0k1.json (gate: K2 RSA == K1 RSA).
Per treatment: rho, Fisher CI, family CI (seed 0, seeds 0-19), split-half reliabilities, decomposition (#7, the
published script), rank_identifiability (#3, the published script) on a concordance-format file written here.
Branch for Table 1's row (b primary): positive (family lower > 0) / not determined / negative (family upper < 0).
Output: brennan_k2_summary_20261009.json (+ concordance32_brennan_k2_{b,c}.json).
"""
from __future__ import annotations

import glob
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
sys.path.insert(0, str(MEG / "codes"))
import recompute_corrected_stats as RC                                   # noqa: E402
from p4_c4_aggregate import splithalf_rank_reliability                  # noqa: E402

RAW = HERE / "brennan_k2_raw"
P4E = MEG / "paper4_temporalalign/results/p4_encoding"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(MEG / "codes"))


def main():
    k1 = json.loads((HERE / "brennan_preproc_raw/k0k1.json").read_text())
    kinds = k1["model_kind"]
    v2 = json.loads((P4E / "concordance32_brennan_v2.json").read_text())
    out = {"prereg": "paper4_temporalalign/notes/PREREG_paperA_brennan_preproc_20261009.md §3",
           "inputs_sha256": {Path(f).name: sha(f) for f in sorted(glob.glob(str(RAW / "**/*.json"), recursive=True))},
           "published_raw": {"rho_v2": v2["concordance_spearman"], "fisher_v2": v2["concordance_ci95"],
                             "table1_spacefix": 0.3658357771260997}}
    for t in ("b", "c"):
        rsa_ps, enc_ps = {}, {}
        for f in sorted(glob.glob(str(RAW / "conc" / f"brennan_{t}_*.json"))):
            j = json.loads(Path(f).read_text())
            rsa_ps.update(j["rsa_ps"]); enc_ps.update(j["enc_ps"])
        models = [m for m in kinds if m in rsa_ps and m in enc_ps]
        res = {"n_models": len(models), "missing": [m for m in kinds if m not in models]}
        if len(models) < 30:
            res["branch"] = "INCOMPLETE"; out[t] = res; continue
        g = max(abs(rsa_ps[m][s] - k1["per_listener"][s]["rsa"][t][m]) for m in models for s in rsa_ps[m])
        res["gate_rsa_equals_K1_max_abs_diff"] = g
        res["gate_pass"] = bool(g <= 1e-8)
        subs = sorted(rsa_ps[models[0]])
        rsa_m = np.array([np.mean([rsa_ps[m][s] for s in subs]) for m in models])
        enc_m = np.array([np.mean([enc_ps[m][s] for s in subs]) for m in models])
        rho = float(spearmanr(rsa_m, enc_m)[0])
        _, lo, hi, nf = RC.family_clustered_ci({m: rsa_ps[m] for m in models}, {m: enc_ps[m] for m in models}, seed=0)
        sweep = [RC.family_clustered_ci({m: rsa_ps[m] for m in models}, {m: enc_ps[m] for m in models}, seed=s)[1:3]
                 for s in range(20)]
        ik = lambda d: {m: {int(k.lstrip("S")): v for k, v in d[m].items()} for m in models}   # 'S01' -> 1 (order kept)
        rx, cx = splithalf_rank_reliability(ik(rsa_ps))
        ry, cy = splithalf_rank_reliability(ik(enc_ps))
        enc_raw = np.array([np.mean([v2["per_session_enc"][m][s] for s in subs]) for m in models])
        res.update({"rho": rho, "fisher_ci95": list(RC.fisher_ci(rho, len(models))), "family_ci95_seed0": [lo, hi],
                    "family_seed_sweep": {"lower": [min(x[0] for x in sweep), max(x[0] for x in sweep)],
                                          "upper": [min(x[1] for x in sweep), max(x[1] for x in sweep)]},
                    "rsa_reliability_sb": rx, "rsa_ci90": cx, "enc_reliability_sb": ry, "enc_ci90": cy,
                    "enc_rank_vs_raw_spearman": float(spearmanr(enc_m, enc_raw)[0]),
                    "enc_mean_raw_vs_treated": [float(enc_raw.mean()), float(enc_m.mean())],
                    "branch": "positive" if lo > 0 else ("negative" if hi < 0 else "not determined")})
        conc = {"models": models, "per_model": [{"model": m, "kind": kinds[m], "rsa_mean": float(rsa_m[i]),
                                                  "enc_mean": float(enc_m[i])} for i, m in enumerate(models)],
                "per_session_rsa": {m: rsa_ps[m] for m in models}, "per_session_enc": {m: enc_ps[m] for m in models},
                "concordance_spearman": rho, "note": f"K2 treatment {t} (PREREG_paperA_brennan_preproc_20261009)"}
        cf = HERE / f"concordance32_brennan_k2_{t}.json"
        cf.write_text(json.dumps(conc, indent=1))
        with tempfile.TemporaryDirectory() as td:
            run([sys.executable, str(MEG / "codes/concordance_decomposition.py"), str(cf), "--label", f"Brennan K2 {t}",
                 "--out", f"{td}/dec.json"])
            res["decomposition"] = json.loads(Path(f"{td}/dec.json").read_text())
            run([sys.executable, str(MEG / "codes/rank_identifiability.py"), "--files", str(cf), "--out", f"{td}/ri.json"])
            res["rank_identifiability"] = json.loads(Path(f"{td}/ri.json").read_text())
        out[t] = res
    # #4 8 segments and #6 repeat removal: carry the receipts' own summary fields
    out["cross_8seg_b"] = {Path(f).name: {k: v for k, v in json.loads(Path(f).read_text()).items()
                                          if not isinstance(v, (dict, list)) or k in ("rsa_rank_reliability_ci", "enc_rank_reliability_ci")}
                           for f in sorted(glob.glob(str(RAW / "cross" / "*.json")))}
    out["repeat_k1_b"] = {Path(f).name: {k: v for k, v in json.loads(Path(f).read_text()).items()
                                         if not isinstance(v, (dict, list)) or k == "arms"}
                          for f in sorted(glob.glob(str(RAW / "repeat" / "*.json")))}
    for r in out["repeat_k1_b"].values():                      # keep the arms' scalar summaries only
        r["arms"] = {a: {k: v for k, v in d.items() if not isinstance(v, (dict, list))} for a, d in r.get("arms", {}).items()}
    out["cross_8seg_published_raw"] = {Path(f).name: {k: json.loads(Path(f).read_text())[k] for k in
                                       ("n_groups", "concordance", "rsa_rank_reliability", "enc_rank_reliability", "rankable")}
                                       for f in sorted(glob.glob(str(P4E / "crossed_narrative" / "brennan_p4_v2_segment-*_full.json")))}
    (HERE / "brennan_k2_summary_20261009.json").write_text(json.dumps(out, indent=1))
    for t in ("b", "c"):
        r = out.get(t, {})
        print(t, {k: r.get(k) for k in ("n_models", "gate_pass", "rho", "fisher_ci95", "family_ci95_seed0",
                                        "rsa_reliability_sb", "enc_reliability_sb", "enc_rank_vs_raw_spearman", "branch")})
        if "decomposition" in r:
            d = r["decomposition"]
            print("   class gap", d["class_gap_causal_minus_other"], "p_fam", d["gap_p_family_clustered"])
            print("   rank_ident", {k: (v["ratio"], v["tau_median"]) for k, v in r["rank_identifiability"].items()})


if __name__ == "__main__":
    main()
