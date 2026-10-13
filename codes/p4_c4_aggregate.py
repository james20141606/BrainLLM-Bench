#!/usr/bin/env python
"""C4 control, step 3 (local): merge the per-model receipts, apply the gates and the decision rule.

Pre-registration: paper4_temporalalign/PREREG_c4_static32_20260924.md (MEG commit 6e81d61).

Inputs (pulled back from <cluster> $HOME/p4_c4_20260924/):
  paper4_temporalalign/results/p4_encoding/c4_static32/prep_receipt.json
  paper4_temporalalign/results/p4_encoding/c4_static32/per_model/<tag>.json   (32)
  paper4_temporalalign/results/p4_encoding/concordance32_CORRECTED.json       (encoding, C1-C3)
Output:
  paper4_temporalalign/results/p4_encoding/concordance32_c4.json

Estimators are the headline's: recompute_corrected_stats.point / fisher_ci / family_clustered_ci
(10,000 family resamples, seed 0) and a verbatim copy of p4_concordance32.splithalf_rank_reliability
(that module imports cluster-only paths). The copy is checked by reproducing the headline's RSA
reliability 0.98228 from its own per-session values.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "codes"))
import recompute_corrected_stats as RC          # noqa: E402
from model_families import family              # noqa: E402

RES = ROOT / "paper4_temporalalign" / "results" / "p4_encoding"
C4D = RES / "c4_static32"
OUT = RES / "concordance32_c4.json"
PREREG = "paper4_temporalalign/PREREG_c4_static32_20260924.md"
PREREG_COMMIT = "6e81d61"
NOTE = " ".join(sys.argv[1:])     # free-text note recorded in an INCOMPLETE receipt


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def splithalf_rank_reliability(per_sess: dict, n_iter=200, seed=0):
    """Verbatim from p4_concordance32.py (<cluster> copy sha 5c6651e7...), except that session keys are
    cast to int first: the headline ran on int session ids, and JSON receipts carry them as strings,
    whose lexicographic sort ("1","10","11","12","2",...) changes the seeded permutations
    (0.98104 instead of the headline's 0.98228 on the headline's own values; caught by the assert
    below on 2026-09-25, before any C4 value was computed)."""
    per_sess = {m: {int(s): v for s, v in d.items()} for m, d in per_sess.items()}
    models = sorted(per_sess)
    sess = sorted(set.intersection(*[set(per_sess[m]) for m in models]))
    if len(sess) < 4 or len(models) < 3:
        return None, None
    rng = np.random.RandomState(seed)
    rs = []
    for _ in range(n_iter):
        perm = rng.permutation(sess)
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        a = [np.mean([per_sess[m][s] for s in h1]) for m in models]
        b = [np.mean([per_sess[m][s] for s in h2]) for m in models]
        r, _ = spearmanr(a, b)
        if np.isfinite(r):
            rs.append(r)
    rs = np.array(rs)
    sb = 2 * rs / (1 + rs)
    return float(np.median(sb)), [float(np.percentile(sb, 5)), float(np.percentile(sb, 95))]


def paired_family_delta(rsa3, rsa4, enc, models, n_boot=10000, seed=0):
    """Same family resample for both concordances; percentile CI of rho_C4 - rho_C3."""
    sess = sorted(set.intersection(*[set(rsa4[m]) for m in models]))
    a3 = {m: np.mean([rsa3[m][s] for s in sess]) for m in models}
    a4 = {m: np.mean([rsa4[m][s] for s in sess]) for m in models}
    b = {m: np.mean([enc[m][s] for s in sess]) for m in models}
    fams = sorted({family(m) for m in models})
    by = {f: [m for m in models if family(m) == f] for f in fams}
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n_boot):
        pick = rng.choice(len(fams), len(fams), replace=True)
        sel = [m for i in pick for m in by[fams[i]]]
        if len(set(sel)) < 4:
            continue
        r3 = spearmanr([a3[m] for m in sel], [b[m] for m in sel])[0]
        r4 = spearmanr([a4[m] for m in sel], [b[m] for m in sel])[0]
        if np.isfinite(r3) and np.isfinite(r4):
            d.append(r4 - r3)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), len(d)


def main():
    head = json.loads((RES / "concordance32_CORRECTED.json").read_text())
    models32 = head["models"]
    enc = head["per_session_enc"]
    prep = json.loads((C4D / "prep_receipt.json").read_text())

    # check the copied reliability function against the headline receipt
    rxx_head, _ = splithalf_rank_reliability(head["per_session_rsa"])
    assert abs(rxx_head - head["rsa_reliability_sb"]) < 1e-12, rxx_head
    ryy_head, _ = splithalf_rank_reliability(head["per_session_enc"])
    assert abs(ryy_head - head["enc_reliability_sb"]) < 1e-12, ryy_head

    recs, gate_fail, receipts = {}, {}, {}
    for m in models32:
        p = C4D / "per_model" / (m.replace("/", "_") + ".json")
        if not p.exists():
            gate_fail[m] = "no receipt"
            continue
        r = json.loads(p.read_text())
        receipts[m] = {"file": str(p.relative_to(ROOT)), "sha256": sha(p), "jid": r.get("jid"),
                       "hf_revisions": r.get("hf_revisions"), "hf_source": r.get("hf_source"),
                       "l0_sha256": r.get("l0_sha256"), "l0_shape": r.get("l0_shape"),
                       "device": r.get("gpu") or "cpu", "host": r.get("host")}   # deviation 4: route
        g = r.get("gates", {})
        why = []
        if r.get("row_sha256") != prep["row_sha256"]:
            why.append("G1")
        if not g.get("G3", {}).get("pass"):
            why.append("G3")
        if not g.get("G4"):
            why.append("G4")
        if not g.get("G5", {}).get("pass"):
            why.append("G5")
        if m == "gpt2" and not g.get("G3_gpt2_v2_layer0", {}).get("pass"):
            why.append("G3_gpt2_v2")
        if "per_session" not in r:
            why.append("no scores")
        if why:
            gate_fail[m] = "+".join(why)
            continue
        recs[m] = r

    # G2 in the scoring run itself: per-model C3 recomputed next to C4 vs the published values
    g2_worst = max((abs(recs[m]["per_session"][s]["c3"] - head["per_session_rsa"][m][s])
                    for m in recs for s in head["per_session_rsa"][m]), default=float("nan"))
    g2_pass = bool(prep["G2_pass"] and g2_worst <= 1e-8)

    n_fail = len(gate_fail)
    status = "complete" if n_fail == 0 else ("subset" if n_fail <= 2 else "incomplete")
    if status == "incomplete":
        # prereg section 5: >= 3 gate failures -> "C4 run incomplete", ELN only. No C4 statistic
        # is computed or written, so nothing can be read off a partial set.
        res = {"prereg": PREREG, "prereg_commit": PREREG_COMMIT, "status": status,
               "outcome": "INCOMPLETE", "outcome_label": f"{n_fail} models failed gates; ELN only",
               "excluded": gate_fail, "n_passing": len(recs),
               "G1_row_sha256": prep["row_sha256"], "G2_prep_worst_abs_diff": prep["G2_worst_abs_diff"],
               "G2_prep_pass": prep["G2_pass"],
               "G3_failures": {m: {k: (r := json.loads((C4D / "per_model" / (m.replace("/", "_") + ".json"))
                                                       .read_text()))["gates"]["G3"].get(k)
                                   for k in ("median_row_cos", "min_row_cos", "frac_rows_cos_ge_0.9999")}
                               for m in gate_fail if "G3" in gate_fail[m]},
               "superseded_G3fail_receipts": {
                   q.name: {"sha256": sha(q), **{k: json.loads(q.read_text())["gates"]["G3"].get(k)
                            for k in ("median_row_cos", "min_row_cos", "frac_rows_cos_ge_0.9999")}}
                   for q in sorted((C4D / "per_model").glob("*_G3fail.json"))},
               "note": NOTE,
               "inputs": {"prep_receipt.json": sha(C4D / "prep_receipt.json"),
                          "per_model_receipts": receipts},
               "code_sha256": {"p4_c4_aggregate.py": sha(__file__)}}
        OUT.write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ("status", "outcome", "excluded", "n_passing")}, indent=1))
        return
    models = [m for m in models32 if m in recs]
    rsa3 = {m: {s: v["c3"] for s, v in recs[m]["per_session"].items()} for m in models}
    rsa4 = {m: {s: v["c4"] for s, v in recs[m]["per_session"].items()} for m in models}
    encm = {m: enc[m] for m in models}

    rho3, _, sess = RC.point(rsa3, encm, models)
    rho4, _, _ = RC.point(rsa4, encm, models)
    p4 = float(spearmanr([np.mean([rsa4[m][s] for s in sess]) for m in models],
                         [np.mean([encm[m][s] for s in sess]) for m in models])[1])
    f3 = RC.family_clustered_ci(rsa3, encm)
    f4 = RC.family_clustered_ci(rsa4, encm)
    fisher4 = RC.fisher_ci(rho4, len(models))
    dlo, dhi, nd = paired_family_delta(rsa3, rsa4, encm, models)
    rxx4, rxx4_ci = splithalf_rank_reliability(rsa4)
    rxx3, _ = splithalf_rank_reliability(rsa3)
    ryy = head["enc_reliability_sb"]
    ceil4 = float(np.sqrt(max(rxx4, 0) * max(ryy, 0)))

    # decision rule, prereg section 6
    L, U = f4[1], f4[2]
    if not g2_pass:
        outcome, label = "G2_FAILED", "C4 not read: C1-C3 recompute did not match the headline"
    elif status == "incomplete":
        outcome, label = "INCOMPLETE", f"{n_fail} models failed gates; ELN only"
    elif rxx4 < 0.70 and not U < 0:
        outcome, label = "D", "indeterminate: C4 leaves too little reliable ranking variance"
    elif U < 0:
        sub = ("undiminished" if dlo <= 0 <= dhi else
               "weakened" if dlo > 0 else "strengthened")
        outcome, label = "A", f"reversal holds under C4 ({sub})"
    elif rho4 < 0:
        outcome, label = "B", "not determined under C4 (family CI includes zero)"
    else:
        outcome, label = "C", ("reversal removed (becomes agreement)" if L > 0
                               else "reversal removed (vanishes)")

    per_model = []
    for m in models:
        r = recs[m]
        per_model.append({"model": m, "kind": r["kind"], "family": family(m),
                          "rsa_c3_mean": float(np.mean([rsa3[m][s] for s in sess])),
                          "rsa_c4_mean": float(np.mean([rsa4[m][s] for s in sess])),
                          "enc_mean": float(np.mean([encm[m][s] for s in sess])),
                          "median_spearman_ctx_vs_static_rdm": float(np.median(
                              [v["spearman_ld_c4"] for v in r["per_session"].values()])),
                          "n_nonfinite_c4_pairs": int(sum(v["n_nonfinite_c4"]
                                                          for v in r["per_session"].values())),
                          "extractor": r.get("extractor", "v2 cache (deviation 1)"),
                          "G3_mode": r["gates"]["G3"].get("mode", "reextract_vs_fd"),
                          "G3_median_row_cos": r["gates"]["G3"].get("median_row_cos"),
                          "G3_bitwise_equal": r["gates"]["G3"].get("bitwise_equal"),
                          "fallback_frac": r["extractor_record"].get("fallback_frac")})
    classes = {}
    for k in ("causal", "masked", "encdec"):
        rows = [x for x in per_model if x["kind"] == k]
        classes[k] = {"n": len(rows),
                      "mean_rsa_c3": float(np.mean([x["rsa_c3_mean"] for x in rows])),
                      "mean_rsa_c4": float(np.mean([x["rsa_c4_mean"] for x in rows])),
                      "mean_enc": float(np.mean([x["enc_mean"] for x in rows]))}

    res = {"prereg": PREREG, "prereg_commit": PREREG_COMMIT,
           "estimand": "Spearman over models of (session-mean partial RSA | C1..C4) vs "
                       "(session-mean ridge encoding, concordance32_CORRECTED)",
           "c4_definition": "cosine RDM of the model's own hidden_states[0] at the target token(s), "
                            "same extractor/target rule as the 0.5-depth features",
           "status": status, "excluded": gate_fail,
           "deviations": sorted({r.get("deviation") for r in recs.values() if r.get("deviation")}),
           "procedural_notes": ["bert-base-uncased receipt is smoke task 18407672_18 (extract_score "
                                "917c371c, main path code-identical to the array's 9c504800; prereg "
                                "Deviations, note 3)"],
           "n_models": len(models),
           "n_sessions": len(sess),
           "G1_row_sha256": prep["row_sha256"], "G1_meta_sha256": prep["meta_sha256"],
           "G2_prep_worst_abs_diff": prep["G2_worst_abs_diff"],
           "G2_scoring_worst_abs_diff": g2_worst, "G2_pass": g2_pass,
           "rho_c3_recomputed": float(rho3), "rho_c3_published": head["concordance_spearman"],
           "rho_c3_family_ci95": [f3[1], f3[2]],
           "rho_c4": float(rho4), "rho_c4_p": p4, "rho_c4_fisher_ci95": list(fisher4),
           "rho_c4_family_ci95": [L, U], "n_families": f4[3],
           "delta_c4_minus_c3": float(rho4 - rho3), "delta_paired_family_ci95": [dlo, dhi],
           "delta_n_boot_valid": nd,
           "rsa_c4_reliability_sb": rxx4, "rsa_c4_reliability_ci90": rxx4_ci,
           "rsa_c3_reliability_sb_recomputed": rxx3, "enc_reliability_sb": ryy,
           "attenuation_ceiling_c4": ceil4,
           "outcome": outcome, "outcome_label": label,
           "class_means": classes, "per_model": per_model,
           "per_session_rsa_c4": rsa4,
           "inputs": {"concordance32_CORRECTED.json": sha(RES / "concordance32_CORRECTED.json"),
                      "prep_receipt.json": sha(C4D / "prep_receipt.json"),
                      "prep_npz_sha256": prep["prep_npz_sha256"],
                      "per_model_receipts": receipts},
           "code_sha256": {"p4_c4_aggregate.py": sha(__file__),
                           "recompute_corrected_stats.py": sha(ROOT / "codes/recompute_corrected_stats.py"),
                           "model_families.py": sha(ROOT / "codes/model_families.py"),
                           "<cluster>": prep["code_sha256"]},
           "seeds": {"subsample": 42, "family_bootstrap": 0, "n_boot": 10000,
                     "splithalf": 0, "splithalf_iter": 200},
           "jids": {"prep": prep.get("jid"),
                    "per_model": sorted({v["jid"] for v in receipts.values()})}}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in (
        "status", "excluded", "G2_pass", "G2_scoring_worst_abs_diff", "rho_c3_recomputed",
        "rho_c4", "rho_c4_fisher_ci95", "rho_c4_family_ci95", "delta_c4_minus_c3",
        "delta_paired_family_ci95", "rsa_c4_reliability_sb", "attenuation_ceiling_c4",
        "outcome", "outcome_label", "class_means")}, indent=1))


if __name__ == "__main__":
    main()
