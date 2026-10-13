#!/usr/bin/env python3
"""Pin every number in PaperA's main-text Table 1 (label tab:main) to the stored receipts.

Table 1 was added on 2026-09-24 and no earlier checker read it. This script recomputes each
cell from the per-block receipts and compares it, cell by cell and row by row, with the TeX:

  top block   -- MEG-MASC crossed_narrative JSONs (listener halves, story pairs at full and
                 equal length, same- vs other-story partitions with the 10,000-draw bootstrap)
                 and the 225 ds003020 UTS*_*.json receipts (listener halves as the median of
                 per-story medians, 300 story pairs, story-resampled 5,000-draw gap interval);
  caption     -- story means, the "no ds003020 pair of 300 is negative" statement, and the
                 Kendall tau / spread ranges from rank_identifiability_CORRECTED.json;
  bottom block-- concordance32_CORRECTED, correction_2026_08_derived (family CI),
                 adr0006_registered_verdicts_v2 (48 base), the LibriBrain class decomposition,
                 concordance32_masc27 and concordance32_brennan_v2 (stored Fisher-z intervals).

The statistics re-implement, line for line, the helpers the paper's analyses use
(codes/fig_stimulus_generalization.py, codes/ds003020_crossnarrative.py,
codes/partition_contrast_aggregate.py) with the same seeds, so the script is self-contained.
Cells are compared as whole cell strings in fixed row/column positions, so a row or column swap
that preserves the value set fails. It also pins the exploratory dagger on the MEG-MASC story
cells. Read-only; exits 1 on any mismatch.

Usage:  python3 codes/check_table1_receipts.py [--tex PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper4_temporalalign/results"
if not RESULTS.is_dir():
    RESULTS = ROOT / "results"          # release layout
P4 = RESULTS / "p4_encoding"
CN = P4 / "crossed_narrative"
DS = P4 / "ds003020"

FAIL: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAIL.append(msg)


def rj(p: Path) -> dict:
    return json.loads(p.read_text())


def f(v: float, nd: int) -> str:
    s = f"{v:+.{nd}f}"
    return "+" + s[1:] if s in (f"-{0:.{nd}f}",) else s


def ci(lo: float, hi: float, nd: int) -> str:
    return f"[{f(lo, nd)},{f(hi, nd)}]"


# ------------------------------------------------------------------ helpers (mirror the paper's)
def leaderboard(j, ro):
    per = j["per_group"][ro]
    ms = sorted(per)
    return ms, np.array([np.mean(list(per[m].values())) for m in ms])


def agree(a, b):
    (m1, v1), (m2, v2) = a, b
    com = sorted(set(m1) & set(m2))
    return float(spearmanr([v1[m1.index(c)] for c in com], [v2[m2.index(c)] for c in com])[0])


def listener_agreement(j, ro, n_iter=200, seed=0):      # fig_stimulus_generalization.py
    per = j["per_group"][ro]
    ms = sorted(per)
    groups = sorted(set.intersection(*[set(per[m]) for m in ms]))
    M = np.array([[per[m][g] for g in groups] for m in ms])
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_iter):
        p = rng.permutation(len(groups))
        r = spearmanr(M[:, p[: len(p) // 2]].mean(1), M[:, p[len(p) // 2:]].mean(1))[0]
        if np.isfinite(r):
            out.append(r)
    return np.array(out)


def story_pairs(blocks, ro):
    return np.array([agree(leaderboard(a, ro), leaderboard(b, ro)) for a, b in combinations(blocks, 2)])


def ds_leaderboard(cells, ro, subjects=None):             # ds003020_crossnarrative.py
    subs = list(cells) if subjects is None else [s for s in subjects if s in cells]
    models = sorted(set.intersection(*[set(cells[s][ro]) for s in subs]))
    return models, np.array([[cells[s][ro][m] for m in models] for s in subs]).mean(axis=0)


def ds_corr(a, b):
    (ma, va), (mb, vb) = a, b
    com = sorted(set(ma) & set(mb))
    if len(com) < 8:
        return np.nan
    return float(spearmanr([va[ma.index(c)] for c in com], [vb[mb.index(c)] for c in com])[0])


def ds_listener_axis(cells, ro, n_iter=100, seed=0):
    subs = sorted(cells)
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_iter):
        p = rng.permutation(subs)
        r = ds_corr(ds_leaderboard(cells, ro, p[: len(p) // 2]), ds_leaderboard(cells, ro, p[len(p) // 2:]))
        if np.isfinite(r):
            out.append(r)
    return out


# ------------------------------------------------------------------ TeX table parsing
def table_block(tex: str) -> tuple[str, str]:
    pos = tex.index(r"\label{tab:main}")
    start = tex.rfind(r"\begin{table}", 0, pos)
    end = tex.index(r"\end{table}", pos)
    block = tex[start:end]
    cap = block[block.index(r"\caption{"): block.index(r"\label{tab:main}")]
    return block, cap


def rows(block: str) -> dict[str, list[str]]:
    out = {}
    for tab in re.findall(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", block, re.S):
        tab = re.sub(r"\\(toprule|midrule|bottomrule)|\\cmidrule(\([^)]*\))?\{[^}]*\}", "", tab)
        for raw in tab.split(r"\\"):
            cells = [c.strip() for c in raw.split("&")]
            # 09-25 de-numbering: interval cells are set {\scriptsize ...}; read them without the wrapper
            cells = [re.sub(r"^\{\\scriptsize\s*(.*)\}$", r"\1", c, flags=re.S) for c in cells]
            if len(cells) < 2:
                continue
            key = re.sub(r"\^\{?(\\d?dagger)?(\\P)?(\\\|)?\}?", "", cells[0].replace("$", "").replace(r"\quad", ""))
            key = re.sub(r"\s+", " ", key).strip()
            out[key] = cells[1:]
    return out


def norm(c: str) -> str:
    return re.sub(r"\s+", "", c.replace("$", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(ROOT / "paper4_temporalalign/tex/paperA_benchmark.tex"))
    tex = Path(ap.parse_args().tex).read_text()
    block, cap = table_block(tex)
    R = rows(block)

    def row(label: str) -> list[str]:
        if label not in R:                      # exact row label, daggers stripped
            check(False, f"row {label!r} not found; rows are {list(R)}")
            return []
        return R[label]

    def expect(label_prefix: str, want: list[str]) -> None:
        got = [norm(c) for c in row(label_prefix)]
        want_n = [norm(w) for w in want]
        check(got == want_n, f"{label_prefix}: TeX {got} vs receipts {want_n}")

    # ---------------- MEG-MASC crossed narratives
    full = [rj(p) for p in sorted(CN.glob("meg_masc_merged27_story-*_full.json"))]
    cap_ = [rj(p) for p in sorted(CN.glob("meg_masc_merged27_story-*_cap*.json")) if "_p" not in p.stem]
    parts = [rj(p) for p in sorted(CN.glob("meg_masc_merged27_story-*_p*of*.json"))]
    check(len(full) == 4 and len(cap_) == 4 and len(parts) == 8,
          f"receipt counts full/cap/partitions = {len(full)}/{len(cap_)}/{len(parts)} (want 4/4/8)")
    n_list = {j["n_groups"] for j in full}
    check(n_list == {27}, f"MEG-MASC listeners per story {n_list} (want 27)")

    mm = {}
    for ro in ("rsa", "enc"):
        mm[ro, "lst"] = float(np.median(np.concatenate([listener_agreement(j, ro) for j in full])))
        sf, sc = story_pairs(full, ro), story_pairs(cap_, ro)
        check(len(sf) == 6, f"{ro}: 6 story pairs")
        mm[ro, "sf"], mm[ro, "sf_mean"] = float(np.median(sf)), float(np.mean(sf))
        mm[ro, "sc"], mm[ro, "sc_mean"] = float(np.median(sc)), float(np.mean(sc))
        w, b = [], []
        for j1, j2 in combinations(parts, 2):
            (w if j1["block"] == j2["block"] else b).append(agree(leaderboard(j1, ro), leaderboard(j2, ro)))
        w, b = np.array(w), np.array(b)
        rng = np.random.RandomState(0)                        # partition_contrast_aggregate.py
        boot = np.array([rng.choice(w, len(w), True).mean() - rng.choice(b, len(b), True).mean()
                         for _ in range(10000)])
        mm[ro, "part"] = float(w.mean() - b.mean())
        mm[ro, "part_ci"] = tuple(np.percentile(boot, [2.5, 97.5]))
    conc = [j["concordance"] for j in full]

    # ---------------- ds003020
    by = defaultdict(dict)
    for p in sorted(DS.glob("UTS*_*.json")):
        j = rj(p)
        by[j["story"]][j["subject"]] = j
    check(len(by) == 25 and {len(c) for c in by.values()} == {9},
          f"ds003020 stories x listeners = {len(by)} x {sorted({len(c) for c in by.values()})} (want 25 x 9)")
    dz = {}
    for ro in ("rsa", "enc"):
        med = {st: float(np.median(ds_listener_axis(c, ro))) for st, c in sorted(by.items())}
        keep = [st for st, v in med.items() if v >= 0.2]
        check(len(keep) == 25, f"ds003020 {ro}: all 25 stories determined (got {len(keep)})")
        pairs = [(s1, s2, ds_corr(ds_leaderboard(by[s1], ro), ds_leaderboard(by[s2], ro)))
                 for s1, s2 in combinations(keep, 2)]
        rs = np.array([r for *_, r in pairs])
        la = np.array([med[s] for s in keep])
        rng = np.random.RandomState(0)
        boot = []
        for _ in range(5000):
            pick = rng.choice(keep, len(keep), replace=True)
            sub = [r for s1, s2, r in pairs if s1 in pick and s2 in pick]
            if sub:
                boot.append(np.median([med[s] for s in pick]) - np.median(sub))
        dz[ro] = dict(lst=float(np.median(la)), story=float(np.median(rs)), n=len(rs),
                      neg=int((rs < 0).sum()), gap=float(np.median(la) - np.median(rs)),
                      ci=tuple(np.percentile(boot, [2.5, 97.5])))

    # ---------------- top block, cell by cell
    D = r"^\dagger"
    expect("Listener halves", [f(mm["rsa", "lst"], 3), f(mm["enc", "lst"], 3),
                               f(dz["rsa"]["lst"], 3), f(dz["enc"]["lst"], 3)])
    expect("Stories, full length", [f(mm["rsa", "sf"], 3) + D, f(mm["enc", "sf"], 3) + D,
                                    f(dz["rsa"]["story"], 3), f(dz["enc"]["story"], 3)])
    expect("Stories, equal length", [f(mm["rsa", "sc"], 3) + D, f(mm["enc", "sc"], 3) + D, "--", "--"])
    expect("Listener - story gap", ["--", "--", f(dz["rsa"]["gap"], 3), f(dz["enc"]["gap"], 3)])
    expect("95\\% interval, stories resampled", ["", "", ci(*dz["rsa"]["ci"], 3), ci(*dz["enc"]["ci"], 3)])
    expect("Same - other story (partitions)", [f(mm["rsa", "part"], 3) + D, f(mm["enc", "part"], 3) + D,
                                               "--", "--"])
    expect("95\\% interval", [ci(*mm["rsa", "part_ci"], 2), ci(*mm["enc", "part_ci"], 2), "", ""])

    # ---------------- caption statements
    capn = norm(cap)
    check(f"MEG-MASCstorymeansare{f(mm['rsa', 'sf_mean'], 3)}(full)and{f(mm['rsa', 'sc_mean'], 3)}(equallength)"
          in capn, f"caption story means {f(mm['rsa', 'sf_mean'], 3)} / {f(mm['rsa', 'sc_mean'], 3)}")
    check(dz["rsa"]["neg"] == 0 and dz["enc"]["neg"] == 0 and dz["rsa"]["n"] == dz["enc"]["n"] == 300
          and "nods003020storypairof300isnegative" in capn,
          f"caption 'no ds003020 story pair of 300 is negative' (neg rsa/enc = "
          f"{dz['rsa']['neg']}/{dz['enc']['neg']} of {dz['rsa']['n']})")
    # R1 2026-10-10 (PREREG_paperA_brennan_preproc_20261009, K2): the stored Brennan epochs carry no baseline,
    # so the caption's tau/spread range is LibriBrain's alone; Brennan's comes from the K2 receipt, baseline (b)
    # primary and detrend (c) beside it. The un-baselined Brennan entries of rank_identifiability_CORRECTED are
    # no longer quoted anywhere in the caption.
    ident_all = rj(P4 / "rank_identifiability_CORRECTED.json")
    ident = {k: v for k, v in ident_all.items() if k.startswith("concordance32_CORRECTED|")}
    check(len(ident) == 4, f"LibriBrain identifiability: 4 readout x aggregation entries (got {len(ident)})")
    taus = [v["tau_median"] for v in ident.values()]
    ratios = [v["ratio"] for v in ident.values()]
    k2 = rj(RESULTS / "paperA_rebuttal_20261004/brennan_k2_summary_20261009.json")
    def _rng(tr):
        ri = k2[tr]["rank_identifiability"]
        t_ = [v["tau_median"] for v in ri.values()]; r_ = [v["ratio"] for v in ri.values()]
        check(len(ri) == 4, f"K2 {tr}: 4 identifiability entries")
        return f"{min(t_):.2f}--{max(t_):.2f}", f"{min(r_):.2f}--{max(r_):.2f}"
    (tb, rb), (tc, rc_) = _rng("b"), _rng("c")
    want_cap = (f"unithalvesonLibriBraingiveKendall$\\tau={min(taus):.2f}$--${max(taus):.2f}$,withbetween-modelspread"
                f"${min(ratios):.1f}$--${max(ratios):.1f}\\times$thepaired-differenceSE;onBrennanwithaper-epochbaseline"
                f"$\\tau={tb.split('--')[0]}$--${tb.split('--')[1]}$andspread${rb.split('--')[0]}$--${rb.split('--')[1]}\\times$"
                f"(detrended${tc.split('--')[0]}$--${tc.split('--')[1]}$,${rc_.split('--')[0]}$--${rc_.split('--')[1]}\\times$)")
    check(want_cap.replace("$", "") in norm(cap), f"caption tau/spread: LibriBrain {min(taus):.2f}-{max(taus):.2f}, {min(ratios):.1f}-{max(ratios):.1f}x; "
          f"Brennan b {tb}, {rb}x; c {tc}, {rc_}x")

    # ---------------- bottom block
    lb = rj(P4 / "concordance32_CORRECTED.json")
    fam = rj(P4 / "correction_2026_08_derived.json")["headline"]["corrected"]
    # 09-25 space-fix (prereg fed178a, R version): the 48-base test and the MEG-MASC / Brennan 32-checkpoint
    # concordances are read from the re-extraction aggregates; the unspaced originals are pinned in app D
    SF = P4 / "space_fix_20260925/aggregate"
    q2 = rj(SF / "adr0006_registered_verdicts_spacefix.json")["q2"]
    dec = rj(P4 / "concordance_decomposition_libribrain_CORRECTED.json")
    ma = rj(SF / "concordance_spacefix_masc27_32.json")
    br = rj(SF / "concordance_spacefix_brennan_32.json")
    sf = rj(P4 / "space_fix_20260925.json")
    check(sf["outcomes"]["6.1_Q2_48base"]["outcome"] == "holds" and sf["outcomes"]["6.8_masc27_32"]["outcome"] == "holds"
          and sf["outcomes"]["6.8_brennan_32"]["outcome"] == "flips", "space-fix outcomes 6.1 holds / 6.8 holds, flips")
    check(abs(sf["outcomes"]["6.1_Q2_48base"]["recomputed"]["rho"] - q2["rho"]) < 1e-12
          and abs(sf["outcomes"]["6.8_brennan_32"]["recomputed"][0] - br["concordance_spearman"]) < 1e-12
          and abs(sf["outcomes"]["6.8_masc27_32"]["recomputed"][0] - ma["concordance_spearman"]) < 1e-12,
          "summary receipt equals the aggregates")
    # R1 2026-10-07: the Fisher-z interval still excludes zero narrowly (App. D keeps that sentence), but the
    # family-clustered interval -- the standard the paper holds claims to -- is now computed for both cohorts
    # with the paper's own function (codes/recompute_corrected_stats.family_clustered_ci, seed 0, 10,000) and
    # Section 5.3 states it; the old "only just excludes zero" body sentence is gone.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recompute_corrected_stats import family_clustered_ci  # noqa: E402
    fam_ma = family_clustered_ci(ma["per_session_rsa"], ma["per_session_enc"], n_boot=10000, seed=0)
    fam_br = family_clustered_ci(br["per_session_rsa"], br["per_session_enc"], n_boot=10000, seed=0)
    check(abs(fam_ma[0] - ma["concordance_spearman"]) < 1e-12 and abs(fam_br[0] - br["concordance_spearman"]) < 1e-12
          and fam_ma[3] == fam_br[3] == 12, "cohort family CIs: point equals receipt, 12 families")
    check(br["concordance_ci95"][0] > 0 and fam_br[1] < 0 < fam_br[2], "Brennan: Fisher excludes zero, family-clustered includes it (receipts)")
    check(fam_ma[1] > 0, "MEG-MASC: family-clustered interval excludes zero (receipts)")
    bodyn = norm(tex[:tex.index(r"\appendix")])
    # R1 2026-10-10: Section 5.3 now rests "one cohort" on MEG-MASC alone; Brennan is the K2 baseline value
    check("\\textbf{Thereversalbelongstoonecohort.}ThisrestsonMEG-MASCalone." in bodyn
          and f"excludeszero,narrowly(" + ci(fam_ma[1], fam_ma[2], 2) + ")" in bodyn,
          f"Section 5.3 states the MEG-MASC family CI {ci(fam_ma[1], fam_ma[2], 2)} and rests the claim on it alone")
    # K2 Brennan: recompute the family CI of both treatments from the per-listener receipts with the paper's own
    # function and require it to equal the summary, the branch, and the Section 5.3 / Table 1 strings.
    k2c = {}
    for tr in ("b", "c"):
        kc = rj(RESULTS / f"paperA_rebuttal_20261004/concordance32_brennan_k2_{tr}.json")
        fk = family_clustered_ci(kc["per_session_rsa"], kc["per_session_enc"], n_boot=10000, seed=0)
        s = k2[tr]
        z_, s_ = np.arctanh(s["rho"]), 1.96 / np.sqrt(32 - 3)
        check(abs(fk[0] - s["rho"]) < 1e-12 and np.allclose(fk[1:3], s["family_ci95_seed0"], atol=1e-12) and fk[3] == 12
              and abs(kc["concordance_spearman"] - s["rho"]) < 1e-12 and len(kc["per_model"]) == 32
              and np.allclose(np.tanh([z_ - s_, z_ + s_]), s["fisher_ci95"], atol=1e-3)
              and s["branch"] == "not determined" and fk[1] < 0 < fk[2] and s["gate_pass"],
              f"Brennan K2 {tr}: rho {s['rho']:+.3f}, family CI recomputed {fk[1]:+.3f},{fk[2]:+.3f} = summary; "
              f"Fisher = n=32 interval; branch 'not determined'; gate pass")
        k2c[tr] = (s["rho"], s["fisher_ci95"], s["family_ci95_seed0"])
    (rb_, fb_, Fb_), (rc2, fc_, Fc_) = k2c["b"], k2c["c"]
    check(f"Withaper-epochbaselinetheBrennanconcordanceis${f(rb_, 3)}$(family-clusteredCI${ci(*Fb_, 2)}$;"
          f"lineardetrend${f(rc2, 3)}$,${ci(*Fc_, 2)}$),notdetermined".replace("$", "") in bodyn,
          f"Section 5.3 Brennan K2: {f(rb_, 3)} {ci(*Fb_, 2)}; detrend {f(rc2, 3)} {ci(*Fc_, 2)}")
    check("(onBrennantheintervalonlyjustexcludeszero)" not in bodyn, "superseded 'only just excludes zero' body sentence is gone")
    check("onlytheFisher-$z$intervalisstored".replace("$", "") not in bodyn, "superseded 'only the Fisher-z interval is stored' is gone")
    check("Brennan'sintervalincludeszero" not in norm(tex[:tex.index(r"\appendix")]) and "EEGintervalincludingzero" not in norm(tex),
          "no body text calls the Brennan interval undetermined")
    for d, n in ((lb, 32), (ma, 32), (br, 32)):
        z, s = np.arctanh(d["concordance_spearman"]), 1.96 / np.sqrt(n - 3)
        check(np.allclose(np.tanh([z - s, z + s]), d["concordance_ci95"], atol=1e-3),
              f"stored interval is the Fisher-z interval (n={n}, rho={d['concordance_spearman']:+.3f})")
    check(abs(fam["rho"] - lb["concordance_spearman"]) < 1e-12 and fam["n_families"] == q2["n_families"] == 12,
          "family-clustered headline rho equals the corrected receipt; 12 families")
    check(q2["n_base"] == 48, f"48 base checkpoints (got {q2['n_base']})")
    check(len(lb["per_model"]) == len(ma["per_model"]) == len(br["per_model"]) == 32, "32 checkpoints per cohort")
    expect("LibriBrain, 32 checkpoints", [f(lb["concordance_spearman"], 3),
                                          f"{ci(*lb['concordance_ci95'], 2)} [ckpt]; "
                                          f"{ci(*fam['family_clustered_ci95'], 2)} [fam]"])
    # 2026-09-25 (round-3 P1-1): the registered C4 test sits in Table 1 beside the tests that held
    c4 = rj(P4 / "concordance32_c4.json")
    check(c4["outcome"] == "B", f"C4 outcome B (got {c4['outcome']})")
    expect("LibriBrain, 32, C4 also partialled out",
           [f(c4["rho_c4"], 3), f"{ci(*c4['rho_c4_fisher_ci95'], 2)} [ckpt]; {ci(*c4['rho_c4_family_ci95'], 2)} [fam]"])
    check("theC4rowandthecommon-reading-rulerow" in norm(cap)
          and "areregisteredteststhatleftthereversalnotdetermined" in norm(cap)
          and "computedbeforethesubmissiondeadline,notinthesubmittedversion" in norm(cap),
          "caption: C4 and common-reading-rule rows = registered tests, reversal not determined; timing stated")
    # R1 2026-10-07: registered reading-rule test (PREREG_readrule_20260926, branch B) and the per-session
    # target-scale re-score (PREREG_paperA_rebuttal_checks_20261004 section 1.6, branch A)
    rr = rj(RESULTS / "readrule_20260926/readrule_summary.json")
    pl = rr["variants"]["prefix_last"]
    check(rr["status"] == "complete" and rr["primary"]["branch"] == "B" and not rr["missing"] and not rr["gate_failures"]
          and rr["devices"] == ["cpu"], "readrule receipt: complete, branch B, no gate failures, one device class")
    rsa_m = {m["model"]: {"0": m["rsa_mean"]} for m in pl["per_model"]}
    enc_m = {m["model"]: {"0": m["enc_mean"]} for m in pl["per_model"]}
    fam_pl = family_clustered_ci(rsa_m, enc_m, n_boot=10000, seed=0)     # means-only input reproduces the receipt CI
    check(abs(fam_pl[0] - pl["rho"]) < 1e-12 and np.allclose(fam_pl[1:3], pl["family_ci95"], atol=1e-12),
          f"readrule family CI recomputed with the paper's function: {fam_pl[1]:+.4f},{fam_pl[2]:+.4f}")
    check(pl["family_ci95"][0] <= 0 <= pl["family_ci95"][1] and pl["rho"] < 0, "readrule: branch B condition (rho<0, CI includes 0)")
    expect("LibriBrain, 32, common reading rule",
           [f(pl["rho"], 3), f"{ci(*pl['fisher_ci95'], 2)} [ckpt]; {ci(*pl['family_ci95'], 2)} [fam]"])
    s9 = rj(RESULTS / "paperA_rebuttal_20261004/session9_summary_20261005.json")
    pz = s9["per_session_scaled"]
    check(s9["branch"] == "A" and s9["seed_sweep_0_19"]["branches"] == {"A": 20} and not s9["G2_fail"] and not s9["missing"],
          "session-scale re-score: branch A in 20/20 seeds, G2 clean")
    expect("LibriBrain, 32, per-session target scale",
           [f(pz["rho"], 3), f"{ci(*pz['fisher_ci95'], 2)} [ckpt]; {ci(*pz['family_ci95'], 2)} [fam]"])
    # round-3 P1-2: ds003020 encoding uses shuffled within-story TR folds; flagged where it is used
    check("&RSA&encoding&RSA&encoding^\\S\\\\" in norm(block) and
          "^\\SShuffledfive-foldTRsplitswithineachstoryandlistener" in norm(cap),
          "ds003020 encoding column flagged in Table 1 (shuffled within-story TR folds)")
    check("encodingonshuffledwithin-storyTRfolds,whichcaninflateagreement" in norm(tex),
          "Section 5.2 flags the ds003020 encoding folds")
    expect("LibriBrain, 48 base checkpoints", [f(q2["rho"], 2), f"{ci(*q2['ci95_family_clustered'], 2)} [fam]"])
    expect("LibriBrain, class means removed",
           [f(dec["within_class_centred_rho"], 2),
            f"p={dec['within_class_centred_p']:.2f}; {round(100 * dec['between_class_share_of_abs_rho'])}\\% "
            f"of |\\rho| is between classes"])
    expect("MEG-MASC, 27 listeners", [f(ma["concordance_spearman"], 3),
                                       f"{ci(*ma['concordance_ci95'], 2)} [ckpt]; {ci(fam_ma[1], fam_ma[2], 2)} [fam]"])
    # R1 2026-10-10: the un-baselined row is replaced by the K2 baseline (b) and detrend (c) rows
    check(not any(r.startswith("Brennan EEG, 47 listeners") for r in R), "un-baselined Brennan row (+0.366) is gone from Table 1")
    expect("Brennan EEG, 47, per-epoch baseline", [f(rb_, 3), f"{ci(*fb_, 2)} [ckpt]; {ci(*Fb_, 2)} [fam]"])
    expect("Brennan EEG, 47, linear detrend", [f(rc2, 3), f"{ci(*fc_, 2)} [ckpt]; {ci(*Fc_, 2)} [fam]"])
    check(br["shared_subsample"]["n_sessions"] == 47 and ma["shared_subsample"]["n_sessions"] == 27,
          "Brennan 47 / MEG-MASC 27 units in the receipts")
    expect("MEG-MASC, each story alone", [f"{f(min(conc), 3)} to {f(max(conc), 3)}", "sign not determined"])
    check(min(conc) < 0 < max(conc), "per-story concordance changes sign")

    # ---------------- main-text and abstract numbers tied to the same receipts (added 2026-09-24,
    # final review: the width-matched p printed a superseded value and no checker read it)
    body = tex[: tex.index(r"\appendix")]
    abstract = tex[tex.index(r"\begin{abstract}"): tex.index(r"\end{abstract}")]
    bn, an = norm(body), norm(abstract)
    wm = rj(P4 / "class_gap_width_control_CORRECTED.json")["pooled_width_matched"]
    want = (f"gapstays{f(wm['gap'], 1)}ranksover{wm['n']}checkpoints(family-clusteredtwo-sided"
            f"p={wm['p_family_two_sided']:.3f})")
    # 09-25 de-numbering: the width-matched gap and the class-gap p moved from Section 5.3 to Appendix
    # app:mechanism; the body keeps the +25.7 point estimate and points there
    mech = tex[tex.index(r"\label{app:mechanism}"):]
    mech = norm(mech[:mech.index(r"\paragraph")])
    check(want in mech, f"width-matched gap/p in app:mechanism: want '{want}' (class_gap_width_control_CORRECTED)")
    check("withwidthmatchedthegapshrinksandstillsurvivesthefamily-clusteredtest(Appendix~\\ref{app:mechanism})" in bn,
          "Section 5.3 points to app:mechanism for the width-matched gap")
    bdec = rj(P4 / "concordance_decomposition_brennan.json")
    db, dc = k2["b"]["decomposition"], k2["c"]["decomposition"]
    want_b = (f"onBrennanEEGtheclassgapis${f(db['class_gap_causal_minus_other'], 1)}$rankswithaper-epochbaseline"
              f"(family-clustered$p={db['gap_p_family_clustered']:.3f}$;detrended${f(dc['class_gap_causal_minus_other'], 1)}$,"
              f"$p={dc['gap_p_family_clustered']:.3f}$),LibriBrain'ssign,against${f(bdec['class_gap_causal_minus_other'], 1)}$"
              f"($p={bdec['gap_p_family_clustered']:.2f}$)onthestoredepochs")
    want_b = want_b.replace("$", "")
    check(want_b in mech, f"Brennan class gap in app:mechanism: {want_b}")
    check(db["class_gap_causal_minus_other"] > 0 and dec["class_gap_causal_minus_other"] > 0, "K2 b Brennan class gap has LibriBrain's sign (receipts)")
    check(f"classgap(${f(db['class_gap_causal_minus_other'], 1)}$ranks,family-clustered$p={db['gap_p_family_clustered']:.3f}$)hasLibriBrain'ssign".replace("$", "") in bn,
          "Section 5.3 Brennan class gap and p from the K2 receipt")
    check("p=0.010" not in bn and "p=0.0098" not in bn, "superseded width-matched p (0.010/0.0098) absent")
    want = (f"agapof{f(dec['class_gap_causal_minus_other'], 1)}ranks")
    check(want in bn, f"class gap in text: {want}")
    check(f"family-clusteredtwo-sidedp={dec['gap_p_family_clustered_two_sided']:.3f})" in mech,
          f"class-gap two-sided p={dec['gap_p_family_clustered_two_sided']:.3f} in app:mechanism")
    # 09-25 de-numbering: Table 1 intervals are one size smaller than the point estimates
    bottom = block[block.rindex(r"\begin{tabular}"):]
    small = "".join(re.findall(r"\{\\scriptsize [^{}]*\}", block))
    check(block.count("$[") == small.count("$[") and bottom.count(r"{\scriptsize $[") >= 4 and small.count("$[") >= 8,
          "every Table 1 interval is set in \\scriptsize")
    want = f"family-clustered95\\%CI{ci(*fam['family_clustered_ci95'], 2)}".replace("+", "")
    check(want.replace("\\", "\\") in an.replace("+", ""),
          f"abstract headline family-clustered CI {ci(*fam['family_clustered_ci95'], 2)}")
    check(f"\\rho={f(lb['concordance_spearman'], 3)}".replace("+", "") in an,
          "abstract headline rho")
    check(f"(\\rho={f(q2['rho'], 2)},CI{ci(*q2['ci95_family_clustered'], 2)})".replace("+", "")
          in an.replace("+", ""), "abstract 48-base rho and family CI")
    check(f"reliability{lb['rsa_reliability_sb']:.2f}" in an and f"({lb['enc_reliability_sb']:.2f})" in an,
          f"abstract reliabilities {lb['rsa_reliability_sb']:.2f}/{lb['enc_reliability_sb']:.2f}")

    print(f"\nTable 1 receipt pin: {'PASS' if not FAIL else 'FAIL'} ({len(FAIL)} failures)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
