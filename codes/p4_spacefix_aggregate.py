#!/usr/bin/env python
"""Space-fix re-extraction: aggregation and mechanical verdicts (prereg section 6).

Pre-registration: paper4_temporalalign/PREREG_space_fix_reextract_20260925.md (MEG fed178a).

Builds concordance receipts in the published format with the re-extracted checkpoints' per-session
values substituted (all other values are the published ones, byte-for-byte from the published
receipts), runs the registered analysis scripts UNCHANGED on them (adr0006_registered_analysis.py,
adr0010_q1_masc_analysis.py, copies whose SHA-256 is recorded), recomputes the cross-cohort
32-checkpoint concordances and the alpha-grid table, and applies the section-6 rules.

Reliability, Fisher CI and the H1 rule are p4_concordance32's own functions/lines.
A checkpoint in the unspaced (U) set that was not re-extracted is DROPPED (never kept with its
unspaced values); an N' checkpoint not re-extracted keeps its published values and is flagged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import p4_spacefix_lib as L

U_LB = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B",
        "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-14B-Base",
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct", "Qwen/Qwen3-8B", "Qwen/Qwen3-32B"]
NP_LB = ["Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-4B-Base",
         "Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B"]
U_BASE_LB = U_LB[:12]
MASC_LADDER = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
               "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B"]
SMALL3 = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B"]
QWEN25 = MASC_LADDER


def load(p):
    return json.loads(Path(p).read_text())


TEST_WITH_CACHED = False


def new_scores(cohort):
    """Scores of re-extracted features whose extraction receipt passed every gate.
    --test_with_cached substitutes the G2 (cached-feature) scores instead: an aggregation identity
    test whose verdicts must equal the published ones."""
    out = {}
    if TEST_WITH_CACHED:
        for p in sorted((L.OUTD / "score").glob(f"{cohort}__*__cached.json")):
            r = load(p)
            out[r["model"]] = {"score": r, "extract": None, "score_file": p.name}
        return out
    for p in sorted((L.OUTD / "score").glob(f"{cohort}__*__new.json")):
        r = load(p)
        x = load(L.OUTD / "extract" / f"{cohort}__{L.tag(r['model'])}.json")
        if x.get("status") != "OK" or x["feature_sha256"] != r["feature_sha256"]:
            continue
        out[r["model"]] = {"score": r, "extract": x, "score_file": p.name}
    return out


def check_family_map(models):
    """The registered scripts fall back to name heuristics if model_families cannot be imported;
    assert that the paper's own family map is the one in use."""
    from model_families import family
    from recompute_corrected_stats import family_map
    fm = family_map(models)
    if fm != {m: family(m) for m in models}:
        raise SystemExit("family_map is not codes/model_families.py (fallback heuristics in use)")
    return len(set(fm.values()))


def receipt(template, rsa_ps, enc_ps, kinds, note):
    """The aggregation block of p4_concordance32.main, verbatim in substance."""
    import p4_concordance32 as C
    models = list(rsa_ps)
    rows = [{"model": m, "kind": kinds.get(m, "causal"),
             "rsa_mean": float(np.mean(list(rsa_ps[m].values()))),
             "enc_mean": float(np.mean(list(enc_ps[m].values())))} for m in models]
    rsa_v = [r["rsa_mean"] for r in rows]
    enc_v = [r["enc_mean"] for r in rows]
    # p4_concordance32.main computes reliability while the group keys are still numpy integers
    # (JSON turns them into strings afterwards). splithalf_rank_reliability sorts the keys before
    # permuting, so string keys ("1","10","11",...) give a different split sequence. Restore the
    # integer keys so the published reliability is reproduced (aggregation identity test).
    def _ik(ps):
        return {m: {(int(k) if str(k).lstrip("-").isdigit() else k): v for k, v in s.items()}
                for m, s in ps.items()}
    r_xx, ci_xx = C.splithalf_rank_reliability(_ik(rsa_ps))
    r_yy, ci_yy = C.splithalf_rank_reliability(_ik(enc_ps))
    rho, p = spearmanr(rsa_v, enc_v)
    ci = C.fisher_ci(rho, len(models))
    ceiling = float(np.sqrt(max(r_xx, 0) * max(r_yy, 0))) if (r_xx and r_yy) else float("nan")
    if not np.isfinite(ceiling) or np.isnan(ci[1]):
        verdict = "INDETERMINATE (ceiling or CI not estimable)"
    elif ci[1] < ceiling:
        verdict = "H1 SUPPORTED (CI upper bound below ceiling)"
    else:
        verdict = "H1 FALSIFIED (CI includes the ceiling)"
    return {"prereg": template.get("prereg"), "cell": template["cell"],
            "shared_subsample": template["shared_subsample"], "n_models": len(models),
            "models": models, "per_model": rows,
            "rsa_reliability_sb": r_xx, "rsa_reliability_ci90": ci_xx,
            "enc_reliability_sb": r_yy, "enc_reliability_ci90": ci_yy,
            "attenuation_ceiling": ceiling, "concordance_spearman": float(rho),
            "concordance_p": float(p), "concordance_ci95": ci,
            "attenuation_corrected": (float(rho / ceiling) if ceiling and ceiling > 0 else None),
            "H1_verdict": verdict, "per_session_rsa": rsa_ps, "per_session_enc": enc_ps,
            "note": note}


def substitute(pub, new, drop, keep_flag):
    rsa = {m: dict(v) for m, v in pub["per_session_rsa"].items()}
    enc = {m: dict(v) for m, v in pub["per_session_enc"].items()}
    status = {}
    for m in list(rsa):
        if m in new:
            s = new[m]["score"]
            if set(s["per_session_rsa"]) != set(rsa[m]):
                raise SystemExit(f"group keys differ for {m}")
            rsa[m], enc[m] = s["per_session_rsa"], s["per_session_enc"]
            status[m] = "re-extracted"
        elif m in drop:
            del rsa[m], enc[m]
            status[m] = "NOT RE-EXTRACTED (dropped: unspaced)"
        elif m in keep_flag:
            status[m] = "cached N' (not affected; not re-extracted)"
    return rsa, enc, status


def run_script(script, args):
    r = subprocess.run([sys.executable, str(script)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{script.name} failed: {r.stderr[-2000:]}")
    return r.stdout


def outcome_q2(q2):
    rho, (lo, hi) = q2["rho"], q2["ci95_family_clustered"]
    if rho > 0 and lo > 0:
        return "flips (F1 fired)"
    if hi < 0:
        return "holds"
    if rho < 0:
        return "weakens"
    return "flips"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(L.WORK / "aggregate"))
    ap.add_argument("--test_with_cached", action="store_true")
    a = ap.parse_args()
    global TEST_WITH_CACHED
    TEST_WITH_CACHED = a.test_with_cached
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    local = Path(__file__).resolve().parent / "local"
    S = {"prereg": L.PREREG, "code_sha256": L.code_shas(),
         "registered_scripts_sha256": {p.name: L.sha256_file(p) for p in sorted(local.glob("*.py"))}}

    # ------------------------------------------------------------------ LibriBrain
    pub = load(L.PUB / "concordance_modern_final_v2.json")
    kinds = {r["model"]: r["kind"] for r in pub["per_model"]}
    new_lb = new_scores("libribrain")
    rsa, enc, st = substitute(pub, new_lb, drop=set(U_LB), keep_flag=set(NP_LB))
    missing_u = [m for m in U_LB if m not in new_lb]
    S["n_families_base_check"] = check_family_map(
        [m for m in rsa if m not in {"Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct",
                                     "meta-llama/Llama-3.1-8B-Instruct",
                                     "meta-llama/Llama-3.1-70B-Instruct"} and m not in NP_LB[4:]
         and m not in ("Qwen/Qwen3-8B", "Qwen/Qwen3-32B")])
    rec_lb = receipt(pub, rsa, enc, kinds,
                     "space-fix: re-extracted checkpoints substituted into concordance_modern_final_v2")
    p_lb = out_dir / "concordance_spacefix_libribrain.json"
    p_lb.write_text(json.dumps(rec_lb, indent=2))
    p_v = out_dir / "adr0006_registered_verdicts_spacefix.json"
    run_script(local / "adr0006_registered_analysis.py", ["--json", str(p_lb), "--out", str(p_v)])
    v = load(p_v)
    pub_v = load(L.PUB / "adr0006_registered_verdicts_v2.json")
    S["libribrain"] = {"status": st, "missing_unspaced": missing_u, "n_models": rec_lb["n_models"],
                       "published_57_recomputed_same_code": receipt(
                           pub, pub["per_session_rsa"], pub["per_session_enc"], kinds, "")["concordance_spearman"],
                       "published_57_toplevel_field": pub["concordance_spearman"],
                       "exploratory_57_concordance": {k: rec_lb[k] for k in (
                           "concordance_spearman", "concordance_ci95", "concordance_p",
                           "rsa_reliability_sb", "enc_reliability_sb", "attenuation_ceiling",
                           "H1_verdict")},
                       "adr0006": {k: v[k] for k in v if not k.startswith("q3_exploratory")},
                       "adr0006_q3": v.get("q3_exploratory"),
                       "receipt_sha256": L.sha256_file(p_lb), "verdicts_sha256": L.sha256_file(p_v)}

    O = {}
    # 6.1 Q2
    q2_complete = not any(m in missing_u for m in U_BASE_LB) and v["q2"]["n_base"] == 48
    O["6.1_Q2_48base"] = {"published": pub_v["q2"], "recomputed": v["q2"],
                          "registered_verdict": v["q2_verdict"],
                          "outcome": outcome_q2(v["q2"]) if q2_complete else
                          "WITHDRAWN (not all 12 unspaced base checkpoints re-extracted)"}
    # 6.2 Q1 per generation
    labs = {}
    for g in ("qwen2.5", "qwen3", "qwen3.5"):
        labs[g] = {"published": pub_v[f"q1_{g}_verdict"].split(":")[0],
                   "recomputed": v[f"q1_{g}_verdict"].split(":")[0],
                   "rungs": v[f"q1_{g}_ladder_models"], "stats": v.get(f"q1_{g}")}
    anysup = any(x["recomputed"].startswith("SUPPORTED") for x in labs.values())
    changed = any(x["recomputed"] != x["published"] for x in labs.values())
    O["6.2_Q1_generations"] = {"per_generation": labs,
                               "outcome": "flips" if anysup else ("weakens" if changed else "holds")}
    # 6.3 readout-dependent Qwen2.5
    q = v.get("q1_qwen2.5", {})
    full25 = v["q1_qwen2.5_ladder_models"] == QWEN25
    if q and full25:
        e, r = q["enc"], q["rsa"]
        enc_o = ("holds" if (e["rho"] < 0 and e["ci95_session_clustered"][1] < 0 and
                             e["top_minus_bottom_mean"] < 0 and e["top_minus_bottom_p"] < 0.05)
                 else "flips" if (e["rho"] >= 0 and e["top_minus_bottom_mean"] >= 0) else "weakens")
        rsa_o = ("holds" if (r["rho"] > 0 and r["ci95_session_clustered"][0] > 0 and
                             r["top_minus_bottom_mean"] > 0 and r["top_minus_bottom_p"] < 0.05)
                 else "flips" if (r["rho"] <= 0 and r["top_minus_bottom_mean"] <= 0) else "weakens")
        O["6.3_readout_dependent_qwen25"] = {
            "published": pub_v["q1_qwen2.5"], "recomputed": q, "encoding_falls": enc_o,
            "similarity_rises": rsa_o,
            "readout_dependent_sentence": "kept" if enc_o == rsa_o == "holds" else "removed/rewritten",
            "outcome": f"encoding_falls={enc_o}; similarity_rises={rsa_o}"}
    else:
        O["6.3_readout_dependent_qwen25"] = {
            "outcome": "WITHDRAWN for every sentence naming 72B (ladder incomplete)",
            "rungs": v["q1_qwen2.5_ladder_models"], "recomputed": q}
    # 6.4 alpha grid
    alpha_pub = {r["model"]: r for r in load(L.PUB / "alpha_sensitivity_20260922/summary.json")["model_results"]}
    rows = []
    for m in QWEN25:
        if m in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"):
            rows.append({"model": m, "source": "published (N, unchanged)",
                         "original_mean": alpha_pub[m]["original_mean"],
                         "expanded_mean": alpha_pub[m]["expanded_mean"]})
        elif m in new_lb and "alpha_grid" in new_lb[m]["score"]:
            ag = new_lb[m]["score"]["alpha_grid"]
            rows.append({"model": m, "source": "re-extracted", "original_mean": ag["original_mean"],
                         "expanded_mean": ag["expanded_mean"],
                         "expanded_top_grid_count": ag["expanded_top_grid_count"],
                         "replay_vs_enc_max_abs": ag["replay_vs_enc_max_abs"]})
    if len(rows) == 7:
        orig = [x["original_mean"] for x in rows]
        expd = [x["expanded_mean"] for x in rows]
        d_o, d_e = orig[-1] - orig[0], expd[-1] - expd[0]
        o = "holds" if (d_o < 0 and d_e < 0) else "flips" if (d_o > 0 and d_e > 0) else "weakens"
        O["6.4_alpha_grid"] = {"rows": rows,
                               "spearman_size_original": float(spearmanr(range(7), orig)[0]),
                               "spearman_size_expanded": float(spearmanr(range(7), expd)[0]),
                               "72B_minus_0.5B_original": d_o, "72B_minus_0.5B_expanded": d_e,
                               "outcome": o}
    else:
        O["6.4_alpha_grid"] = {"rows": rows, "outcome": "WITHDRAWN (72B or a rung missing)"}
    # 6.6 twins
    c_r, c_e = v["q3_direction_count_rsa"], v["q3_direction_count_enc"]
    def _cons(s):
        k, n = [int(x) for x in s.split()[0].split("/")]
        return k in (0, n) and n > 0
    O["6.6_twins"] = {"published": [pub_v["q3_direction_count_rsa"], pub_v["q3_direction_count_enc"]],
                      "recomputed": [c_r, c_e],
                      "outcome": "flips" if (_cons(c_r) or _cons(c_e)) else "holds"}

    # ------------------------------------------------------------------ MEG-MASC ladder (ADR 0010)
    pubm = load(L.PUB / "concordance_masc27_ladder.json")
    new_m = new_scores("masc27")
    rsa_m, enc_m, st_m = substitute(pubm, new_m, drop=set(MASC_LADDER), keep_flag=set())
    rec_m = receipt(pubm, rsa_m, enc_m, {}, "space-fix: MASC ladder re-extracted rungs")
    p_m = out_dir / "concordance_spacefix_masc27_ladder.json"
    p_m.write_text(json.dumps(rec_m, indent=2))
    p_mv = out_dir / "adr0010_q1_masc_verdicts_spacefix.json"
    run_script(local / "adr0010_q1_masc_analysis.py",
               ["--json", str(p_m), "--libribrain", str(p_v), "--out", str(p_mv)])
    mv = load(p_mv)
    pub_mv = load(L.PUB / "adr0010_q1_masc_verdicts.json")
    S["masc27_ladder"] = {"status": st_m, "adr0010": mv, "receipt_sha256": L.sha256_file(p_m)}
    if len(rsa_m) == 7 and "q1_masc" in mv:
        e = mv["q1_masc"]["enc"]
        lo, hi = e["ci95_subject_clustered"]
        f3r = mv["f3_gate"]["rsa"]["passes"]
        sup = mv["q1_masc_verdict"].startswith("SUPPORTED")
        if sup or lo > 0:
            o = "flips"
        elif lo <= 0 <= hi and not f3r:
            o = "holds"
        else:
            o = "weakens"
        O["6.5_masc_ladder"] = {"published": {"q1_masc": pub_mv["q1_masc"], "f3": pub_mv["f3_gate"],
                                              "verdict": pub_mv["q1_masc_verdict"]},
                                "recomputed": {"q1_masc": mv["q1_masc"], "f3": mv["f3_gate"],
                                               "verdict": mv["q1_masc_verdict"],
                                               "cross_cohort": mv["cross_cohort_statement"]},
                                "outcome": o}
    else:
        O["6.5_masc_ladder"] = {"outcome": "WITHDRAWN for sentences naming missing rungs",
                                "rungs": list(rsa_m)}

    # 6.7 Antonello rows (derived)
    lb_enc = v.get("q1_qwen2.5", {}).get("enc")
    if lb_enc and full25 and "q1_masc" in mv:
        masc_sup = mv["q1_masc_verdict"].startswith("SUPPORTED")
        if lb_enc["rho"] > 0 and lb_enc["ci95_session_clustered"][0] > 0:
            o = "flips"
        elif lb_enc["rho"] <= 0 and not masc_sup:
            o = "holds"
        else:
            o = "weakens"
        O["6.7_antonello"] = {"outcome": o}
    else:
        O["6.7_antonello"] = {"outcome": "WITHDRAWN for clauses naming missing rungs"}

    # ------------------------------------------------------------------ cross-cohort 32 (6.8)
    for coh, fn in (("masc27", "concordance32_masc27.json"), ("brennan", "concordance32_brennan_v2.json")):
        p32 = load(L.PUB / fn)
        kinds32 = {r["model"]: r["kind"] for r in p32["per_model"]}
        n3 = new_scores(coh)
        r32, e32, s32 = substitute(p32, {m: n3[m] for m in SMALL3 if m in n3},
                                   drop=set(SMALL3), keep_flag=set())
        rc = receipt(p32, r32, e32, kinds32, f"space-fix: {coh} 32-checkpoint concordance")
        pp = out_dir / f"concordance_spacefix_{coh}_32.json"
        pp.write_text(json.dumps(rc, indent=2))
        rho0, (lo0, hi0) = p32["concordance_spearman"], p32["concordance_ci95"]
        rho1, (lo1, hi1) = rc["concordance_spearman"], rc["concordance_ci95"]
        excl0, excl1 = (lo0 > 0 or hi0 < 0), (lo1 > 0 or hi1 < 0)
        if rc["n_models"] != 32:
            o = "WITHDRAWN (a small checkpoint not re-extracted)"
        elif np.sign(rho1) != np.sign(rho0) and excl1:
            o = "flips"
        elif coh == "brennan" and excl1 and not excl0:
            o = "flips"
        elif coh == "masc27" and excl0 and not excl1:
            o = "weakens"
        elif coh == "brennan" and np.sign(rho1) == np.sign(rho0) and abs(rho0) - abs(rho1) > 0.1:
            o = "weakens"
        elif np.sign(rho1) != np.sign(rho0):
            o = "flips"
        else:
            o = "holds"
        O[f"6.8_{coh}_32"] = {"published": [rho0, [lo0, hi0]], "recomputed": [rho1, [lo1, hi1]],
                              "status": s32, "reliability": [rc["rsa_reliability_sb"], rc["enc_reliability_sb"]],
                              "receipt_sha256": L.sha256_file(pp), "outcome": o}

    # ------------------------------------------------------------------ descriptives / gates
    gates = {}
    for p in sorted((L.OUTD / "extract").glob("*.json")):
        x = load(p)
        gates[p.stem] = {"status": x.get("status"), "jid": x.get("jid"),
                         "gpu": x.get("gpu"), "versions": x.get("versions"),
                         "tokenizer_json": x.get("tokenizer_files", {}).get("tokenizer.json"),
                         "T1": x.get("gates", {}).get("T", {}).get("T1_pass"),
                         "T2": x.get("gates", {}).get("T", {}).get("T2_pass"),
                         "W": x.get("gates", {}).get("W"), "I": x.get("gates", {}).get("I"),
                         "G5": x.get("gates", {}).get("G5", {}).get("pass"),
                         "D1_median_row_cos": x.get("D1_new_vs_cached", {}).get("median_row_cos")}
    g2 = {}
    for p in sorted((L.OUTD / "score").glob("*__cached.json")):
        x = load(p)
        g2[p.stem] = {"pass": x.get("G2_pass"), "detail": x.get("G2")}
    S["gates"] = {"extract": gates, "G2": g2}
    S["outcomes"] = O
    (out_dir / "space_fix_20260925.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v.get("outcome") if isinstance(v, dict) else v for k, v in O.items()},
                     indent=1))


if __name__ == "__main__":
    main()
