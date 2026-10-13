#!/usr/bin/env python3
"""Pin the C4 (static layer-0 embedding) control of the fixed-depth RSA route to its receipts and TeX.

Prereg: paper4_temporalalign/PREREG_c4_static32_20260924.md. Receipts:
results/p4_encoding/concordance32_c4.json (aggregate) and results/p4_encoding/c4_static32/ (prep
receipt + one receipt per model). Everything the aggregate reports is recomputed here from the
per-model receipts with an independent implementation (no import of the aggregator), and every C4
number printed in the TeX is matched to it. Also asserts that the superseded "C4 not run" wording is
gone from the paper. No inference, no writes.

    python codes/check_c4_receipt_20260924.py            # repo layout
    python codes/check_c4_receipt_20260924.py --tex X --results Y   # mutation tests
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[1]

FAMILY_RULES = [("pythia", "pythia"), ("gpt-neo", "gpt-neo"), ("gpt2", "gpt2"), ("opt", "opt"),
                ("qwen", "qwen"), ("llama", "llama"), ("roberta", "roberta"),
                ("deberta", "deberta"), ("electra", "electra"), ("t5", "t5"), ("bart", "bart"),
                ("distilbert", "bert"), ("albert", "bert"), ("bert", "bert")]


def fam(m):
    s = m.lower()
    return next(f for k, f in FAMILY_RULES if k in s)


def rho(a, b):
    return float(spearmanr(a, b)[0])


def fam_boot(means_list, enc, models, n_boot=10000, seed=0):
    """Family bootstrap (same draw order as recompute_corrected_stats.family_clustered_ci)."""
    fams = sorted({fam(m) for m in models})
    by = {f: [m for m in models if fam(m) == f] for f in fams}
    rng = np.random.RandomState(seed)
    outs = [[] for _ in means_list]
    for _ in range(n_boot):
        pick = rng.choice(len(fams), len(fams), replace=True)
        sel = [m for i in pick for m in by[fams[i]]]
        if len(set(sel)) < 4:
            continue
        for k, a in enumerate(means_list):
            outs[k].append(rho([a[m] for m in sel], [enc[m] for m in sel]))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(ROOT / "paper4_temporalalign/tex/paperA_benchmark.tex"))
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    res_dir = Path(a.results) if a.results else ROOT / "paper4_temporalalign/results"
    if not res_dir.is_dir():
        res_dir = ROOT / "results"
    P = res_dir / "p4_encoding"
    tex = Path(a.tex).read_text()
    agg = json.loads((P / "concordance32_c4.json").read_text())
    head = json.loads((P / "concordance32_CORRECTED.json").read_text())
    prep = json.loads((P / "c4_static32/prep_receipt.json").read_text())
    models = head["models"]
    fails = []

    def need(cond, msg):
        if not cond:
            fails.append(msg)

    if agg.get("status") == "incomplete":
        # prereg section 5: >= 3 gate failures -> "C4 run incomplete", ELN only; the paper keeps its
        # C1-C3 scope sentences and prints no C4 statistic.
        n_fail = 0
        for m in models:
            q = P / "c4_static32/per_model" / (m.replace("/", "_") + ".json")
            if not q.exists():
                n_fail += 1
                continue
            g = json.loads(q.read_text()).get("gates", {})
            ok = (g.get("G3", {}).get("pass") and g.get("G4") and g.get("G5", {}).get("pass")
                  and (m != "gpt2" or g.get("G3_gpt2_v2_layer0", {}).get("pass")))
            n_fail += 0 if ok else 1
        need(n_fail >= 3 and n_fail == len(agg["excluded"]), f"incomplete count {n_fail}")
        need(agg["outcome"] == "INCOMPLETE" and "rho_c4" not in agg, "INCOMPLETE receipt shape")
        for keep in ["the model-specific static-embedding control (C4) was not run for this route",
                     "not yet run with the static-embedding control C4",
                     "C4 has not been run for it", "has not been recomputed with C4"]:
            need(keep in tex.replace("\n", " "), f"INCOMPLETE: scope wording must stay: {keep!r}")
        need(r"\label{app:c4}" not in tex, "INCOMPLETE: no C4 appendix")
        if fails:
            print("C4 receipt pin: FAIL")
            for f in fails:
                print("  -", f)
            raise SystemExit(1)
        print(f"C4 receipt pin: PASS (INCOMPLETE: {n_fail} of 32 models fail gates; paper keeps "
              f"C1-C3 scope wording, no C4 number printed)")
        return

    # receipts and gates
    need(prep["G2_pass"] and prep["n_models"] == 32 and prep["G2_worst_abs_diff"] <= 1e-8, "G2 prep")
    need(prep["row_sha256"].startswith("bfe5c58de6dc") and prep["meta_sha256"].startswith("b8e7784006b0"),
         "G1 hashes")
    c3, c4 = {}, {}
    for m in models:
        r = json.loads((P / "c4_static32/per_model" / (m.replace("/", "_") + ".json")).read_text())
        g = r["gates"]
        need(g["G3"]["pass"] and g["G4"] and g["G5"]["pass"], f"gates {m}")
        need(r["row_sha256"] == prep["row_sha256"], f"row hash {m}")
        need(r["l0_shape"][0] == 30000, f"L0 rows {m}")
        need(sorted(r["per_session"], key=int) == [str(i) for i in range(1, 13)], f"sessions {m}")
        if m == "gpt2":
            need(g["G3_gpt2_v2_layer0"]["pass"], "gpt2 v2 layer 0")
        for s, v in r["per_session"].items():
            need(abs(v["c3"] - head["per_session_rsa"][m][s]) <= 1e-8, f"G2 scoring {m} {s}")
        c3[m] = float(np.mean([r["per_session"][s]["c3"] for s in sorted(r["per_session"])]))
        c4[m] = float(np.mean([r["per_session"][s]["c4"] for s in sorted(r["per_session"])]))
        need(abs(c4[m] - next(x["rsa_c4_mean"] for x in agg["per_model"] if x["model"] == m)) < 1e-12,
             f"aggregate per-model C4 {m}")
    enc = {m: float(np.mean([head["per_session_enc"][m][s] for s in sorted(head["per_session_enc"][m])]))
           for m in models}
    r3 = rho([c3[m] for m in models], [enc[m] for m in models])
    r4 = rho([c4[m] for m in models], [enc[m] for m in models])
    need(abs(r3 - head["concordance_spearman"]) < 1e-12, "rho_C3 recompute")
    need(abs(r4 - agg["rho_c4"]) < 1e-12, "rho_C4 recompute")
    z, se = np.arctanh(r4), 1 / np.sqrt(29)
    fz = [np.tanh(z - norm.ppf(0.975) * se), np.tanh(z + norm.ppf(0.975) * se)]
    need(np.allclose(fz, agg["rho_c4_fisher_ci95"], atol=1e-12), "Fisher CI")
    b3, b4 = fam_boot([c3, c4], enc, models)
    ci3 = np.percentile(b3, [2.5, 97.5])
    ci4 = np.percentile(b4, [2.5, 97.5])
    dci = np.percentile(np.array(b4) - np.array(b3), [2.5, 97.5])
    need(np.allclose(ci3, [-0.7855269643048901, -0.2641235334657375], atol=1e-12), "family CI C3")
    need(np.allclose(ci4, agg["rho_c4_family_ci95"], atol=1e-12), "family CI C4")
    need(np.allclose(dci, agg["delta_paired_family_ci95"], atol=1e-12), "paired delta CI")

    # split-half ranking reliability of RSA_C4, independently (int session ids, as the headline)
    def sb_rel(ps, n_iter=200, seed=0):
        ps = {m: {int(k): v for k, v in d.items()} for m, d in ps.items()}
        ms = sorted(ps)
        ss = sorted(set.intersection(*[set(ps[m]) for m in ms]))
        rng = np.random.RandomState(seed)
        out = []
        for _ in range(n_iter):
            perm = rng.permutation(ss)
            h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
            r = rho([np.mean([ps[m][x] for x in h1]) for m in ms],
                    [np.mean([ps[m][x] for x in h2]) for m in ms])
            if np.isfinite(r):
                out.append(r)
        out = np.array(out)
        return float(np.median(2 * out / (1 + out)))
    need(abs(sb_rel(head["per_session_rsa"]) - head["rsa_reliability_sb"]) < 1e-12, "r_xx C3 copy")
    per4 = {m: {s: v["c4"] for s, v in json.loads((P / "c4_static32/per_model" /
            (m.replace("/", "_") + ".json")).read_text())["per_session"].items()} for m in models}
    rxx_ind = sb_rel(per4)
    need(abs(rxx_ind - agg["rsa_c4_reliability_sb"]) < 1e-12, "r_xx(C4) recompute")

    # decision rule (prereg section 6), independently
    L, U = ci4
    rxx = rxx_ind
    if rxx < 0.70 and not U < 0:
        out = "D"
    elif U < 0:
        out = "A"
    elif r4 < 0:
        out = "B"
    else:
        out = "C"
    need(out == agg["outcome"], f"outcome {out} vs {agg['outcome']}")
    need(agg["status"] == "complete" and agg["n_models"] == 32 and agg["G2_pass"], "status")

    # TeX: superseded scope wording must be gone
    texn = " ".join(tex.split())          # line breaks must not hide stale wording
    for bad in ["was not run for this route", "was not run for this readout",
                "not yet run with the static-embedding control",
                "C4 has not been run for it", "has not been recomputed with C4",
                "so its scores do not establish robustness to static-embedding control"]:
        need(bad not in texn, f"stale TeX wording: {bad!r}")
    # TeX: the printed numbers
    body = tex[:tex.index(r"\appendix")]
    s_r4 = f"{r4:+.3f}".replace("+", "+") if r4 >= 0 else f"{r4:.3f}"
    need(f"$\\rho={s_r4}$" in body or f"$\\rho = {s_r4}$" in body, f"body rho_C4 {s_r4}")
    s_ci = f"[{ci4[0]:+.2f}, {ci4[1]:+.2f}]".replace("+-", "-")
    need(s_ci.replace("+", "") in tex.replace("+", ""), f"family CI {s_ci}")
    app = tex[tex.index(r"\label{app:c4}"):]
    app = app[:app.index(r"\end{table}")]
    for mname, cls in (("causal", "causal"), ("masked", "masked"), ("encdec", "encoder--decoder")):
        cm = agg["class_means"][mname]
        need(f"{cm['mean_rsa_c3']:.4f}" in app and f"{cm['mean_rsa_c4']:.4f}" in app, f"class row {mname}")
    need(f"{rxx:.2f}" in app, "C4 reliability in appendix")
    need(f"{r4 - r3:+.3f}".replace("+", "$+$") in app or f"{r4 - r3:+.3f}" in app, "delta in appendix")

    # 2026-09-25 pins added with the outcome-B text (the table's remaining cells and the gate prose)
    def iv(a, b):
        return f"$[{a:.2f}, {b:.2f}]$".replace("-0.00", "0.00")
    need(f"& ${r3:.3f}$ & ${r4:.3f}$ &" in app, "table rho row")
    need(iv(*ci3) in app and iv(*ci4) in app and iv(*dci) in app, "table family / paired CIs")
    z3 = np.arctanh(r3)
    f3 = [np.tanh(z3 - norm.ppf(0.975) * se), np.tanh(z3 + norm.ppf(0.975) * se)]
    need(iv(*f3) in app and iv(*fz) in app, "table Fisher CIs")
    need(f"& ${sb_rel(head['per_session_rsa']):.2f}$ & ${rxx:.2f}$ &" in app, "table reliability row")
    for k, cls in (("causal", "Autoregressive"), ("masked", "Masked"), ("encdec", "Encoder--decoder")):
        cm = agg["class_means"][k]
        need(f"{cls} (${cm['n']}$) & ${cm['mean_rsa_c3']:.4f}$ & ${cm['mean_rsa_c4']:.4f}$ & "
             f"${cm['mean_enc']:.4f}$" in app, f"class row cells {k}")
    capp = " ".join(tex[tex.index(r"\label{app:c4}"):tex.index(r"\label{tab:c4}")].split())
    need(f"reaches ${ci4[1]:+.3f}$" in capp, "upper family CI bound in prose")
    need(agg["G2_scoring_worst_abs_diff"] <= 7e-17 and "to $7\\times10^{-17}$" in capp, "G2 worst diff prose")
    n_re = sum(1 for m in models if m != "gpt2")
    need(f"for ${n_re}$ checkpoints" in capp and n_re == 31, "G3 re-extraction count")
    worst_med = min(json.loads((P / "c4_static32/per_model" / (m.replace("/", "_") + ".json")).read_text())
                    ["gates"]["G3"]["median_row_cos"] for m in models if m != "gpt2")
    worst_frac = min(json.loads((P / "c4_static32/per_model" / (m.replace("/", "_") + ".json")).read_text())
                     ["gates"]["G3"]["frac_rows_cos_ge_0.9999"] for m in models if m != "gpt2")
    need(worst_med > 1 - 1e-10 and worst_frac == 1.0, f"G3 prose bound (min median {worst_med})")
    # outcome-specific wording (prereg section 7); only B is implemented because B is the outcome
    body_n = " ".join(body.split())
    if out == "B":
        for must in [f"On the $32$ checkpoints it is not determined when the $14$ masked and encoder--decoder checkpoints are read up to the target word, as the causal ones are",  # R1 2026-10-07: abstract sentence now also carries the reading-rule result
                     f"or once static embeddings are also partialled out ($\\rho = {r4:.3f}$)",  # 09-25 14:15: "it" scoped to the 32 (acceptance)
                     "static layer-0 embedding (C4) also partialled out, the reversal is not determined",
                     "and adding C4 leaves the reversal not determined",
                     "($C_4$) also partialled out, the reversal is not determined",
                     "under which the fixed-depth reversal of Section~\\ref{sec:concordance} is not determined",  # 09-26 rereview: pointer to 5.3
                     "With C4 added, the fixed-depth reversal is not determined"]:
            need(must in body_n, f"B wording missing in body: {must!r}")
        # round-3 P1-1 (09-25): Section 5.3 and the Figure 3 title agree with outcome B
        # 09-26 AI-review (codex P02): reworded so the change is also stated as not determined
        need("neither the reversal nor its change is determined" in texn and "added it is $-0.415$" in texn,
             "B wording in Section 5.3")
        need("its sign is unresolved after full feature standardization or C4" in texn, "B wording in Figure 3 title")  # 09-26 codex rereview opt-1
        for bad in ["Neither representation, cell nor confound control", "survives representation and control changes"]:
            need(bad not in texn, f"B forbids pre-C4 wording {bad!r}")
        need("With C4 added, the $32$-checkpoint fixed-depth reversal is not determined" in texn,
             "B wording missing in tab:protocols caption")
        need("\\emph{not determined} once C4 is partialled out" in capp, "B verdict in appendix")
        for bad in ["leaves the reversal at", "reversal holds under", "robust to static-embedding",
                    "survives static-embedding", "weakened from"]:
            need(bad not in texn, f"B forbids {bad!r}")
        # the paired Delta interval contains zero: the text may not say C4 weakens the reversal
        need((dci[0] <= 0 <= dci[1]) and "the data do not show that C4 weakens the reversal" in capp,
             "Delta statement")
    else:
        need(False, f"outcome {out}: TeX wording pins for this branch not written")

    if fails:
        print("C4 receipt pin: FAIL")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print(f"C4 receipt pin: PASS (32 models x 12 sessions; rho_C3 {r3:.4f}, rho_C4 {r4:.4f}, "
          f"family CI [{ci4[0]:.3f}, {ci4[1]:.3f}], outcome {out})")


if __name__ == "__main__":
    main()
