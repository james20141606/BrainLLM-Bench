#!/usr/bin/env python3
"""paperA (#60477) rebuttal: receipt-only recomputation, 2026-10-04. CPU, no GPU, no cluster access.

Reads only released receipts already in the repo; writes one JSON receipt next to this script.
Nothing here changes a published number; this file exists so the rebuttal kit cites a receipt,
not a chat transcript.

Parts
  A  readrule (PREREG_readrule_20260926.md) -- INDEPENDENT route: own Spearman (average ranks),
     own family map hand-copied from tex Table 3 (tab:checkpoints), own cluster bootstrap with a
     different RNG (numpy default_rng). Does not import any production module.
     Inputs: per-session values in results/readrule_20260926/per_model/*.json (14 masked/enc-dec)
     and results/p4_encoding/concordance32_CORRECTED.json (18 causal, and the comparator).
  B  alpha32 (PREREG_alpha32_20260926.md) -- same independent route on
     results/alpha32_20260926/per_model/*.json fold scores; grid-top counts per checkpoint.
  C  family-clustered CI for Brennan and MEG-MASC with the PAPER'S OWN function
     (codes/recompute_corrected_stats.family_clustered_ci, seed 0, 10,000 draws), because the
     point is to apply the paper's own yardstick; plus seeds 0-19 and the independent route.
  D  C4 seed sensitivity with the paper's own function (seeds 0-19).
  E  session-9 facts already visible in released receipts (recorded BEFORE the registered
     diagnostic of notes/PREREG_paperA_rebuttal_checks_20261004.md is run).

Run:  ~/.venvs/figtools/bin/python recompute_paperA_rebuttal_20261004.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P4 = HERE.parent.parent                      # paper4_temporalalign
MEG = P4.parent
RES = P4 / "results"
N_BOOT = 10_000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- independent statistics
def ranks(x):
    """Average ranks (ties share the mean of their positions), 1-based."""
    u, inv, cnt = np.unique(np.asarray(x, float), return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv]


def spear(a, b):
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def fisher(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.959964 * se)), float(np.tanh(z + 1.959964 * se))]


# family map typed from tex Table 3 (tab:checkpoints), NOT imported from model_families.py
TABLE3 = {
    "GPT-2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "Pythia": ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
               "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"],
    "OPT": ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"],
    "GPT-Neo": ["EleutherAI/gpt-neo-125m", "EleutherAI/gpt-neo-1.3B"],
    "Llama": ["meta-llama/Llama-3.2-1B"],
    "Qwen": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"],
    "BERT": ["bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased", "albert-base-v2"],
    "RoBERTa": ["roberta-base", "roberta-large"],
    "DeBERTa": ["microsoft/deberta-v3-base", "microsoft/deberta-v3-large"],
    "ELECTRA": ["google/electra-base-discriminator"],
    "T5": ["t5-small", "t5-base", "t5-large"],
    "BART": ["facebook/bart-base", "facebook/bart-large"],
}
FAM = {m: f for f, ms in TABLE3.items() for m in ms}
assert len(FAM) == 32 and len(TABLE3) == 12


def cluster_boot(xs, ys, models, seed, xs2=None, ys2=None, n_boot=N_BOOT):
    """Resample the 12 families with replacement; Spearman on the pooled members.
    If xs2/ys2 given, also returns the paired difference rho(xs,ys) - rho(xs2,ys2) per draw."""
    rng = np.random.default_rng(seed)
    fams = sorted(set(FAM[m] for m in models))
    members = {f: [i for i, m in enumerate(models) if FAM[m] == f] for f in fams}
    xs, ys = np.asarray(xs), np.asarray(ys)
    out, dif = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(fams), len(fams))
        idx = [i for k in pick for i in members[fams[k]]]
        if len(set(idx)) < 4:
            continue
        r = spear(xs[idx], ys[idx])
        if not np.isfinite(r):
            continue
        out.append(r)
        if xs2 is not None:
            dif.append(r - spear(np.asarray(xs2)[idx], np.asarray(ys2)[idx]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    res = {"ci95": [float(lo), float(hi)], "n_valid": len(out), "seed": seed, "rng": "numpy.default_rng"}
    if xs2 is not None:
        dl, dh = np.percentile(dif, [2.5, 97.5])
        res["paired_delta_ci95"] = [float(dl), float(dh)]
    return res


def splithalf_sb(ps_by_model, models, n_split=200, seed=7):
    """Own split-half ranking reliability: random 6/6 session splits, Spearman of half-means,
    Spearman-Brown on the mean r."""
    rng = np.random.default_rng(seed)
    sess = sorted(ps_by_model[models[0]], key=lambda s: int(s))
    M = np.array([[ps_by_model[m][s] for s in sess] for m in models])
    rs = []
    for _ in range(n_split):
        p = rng.permutation(len(sess)); h = len(sess) // 2
        rs.append(spear(M[:, p[:h]].mean(1), M[:, p[h:]].mean(1)))
    r = float(np.mean(rs))
    return 2 * r / (1 + r)


# ---------------------------------------------------------------- Part A: readrule
def part_a():
    head = json.loads((RES / "p4_encoding" / "concordance32_CORRECTED.json").read_text())
    models = head["models"]
    kind = {d["model"]: d["kind"] for d in head["per_model"]}
    rr = {}
    for f in sorted(glob.glob(str(RES / "readrule_20260926" / "per_model" / "*.json"))):
        j = json.loads(Path(f).read_text())
        rr[j["model"]] = j
    assert len(rr) == 14 and all(kind[m] != "causal" for m in rr), "expected the 14 non-causal"
    sess = [str(s) for s in range(1, 13)]
    # sanity: the receipts' 'cached' per-session values equal the headline per-session values
    worst = max(abs(rr[m]["scores"]["cached"][k][s] - head[h][m][s])
                for m in rr for k, h in (("per_session_rsa", "per_session_rsa"), ("per_session_enc", "per_session_enc"))
                for s in sess)

    def table(rule):
        R, E, Rps, Eps = [], [], {}, {}
        for m in models:
            if m in rr:
                rps, eps = rr[m]["scores"][rule]["per_session_rsa"], rr[m]["scores"][rule]["per_session_enc"]
            else:
                rps, eps = head["per_session_rsa"][m], head["per_session_enc"][m]
            Rps[m], Eps[m] = rps, eps
            R.append(np.mean([rps[s] for s in sess])); E.append(np.mean([eps[s] for s in sess]))
        return np.array(R), np.array(E), Rps, Eps

    R0, E0, R0ps, E0ps = table("full_mean")      # == comparator (cached features re-read)
    # comparator straight from the headline receipt
    Rh = np.array([np.mean([head["per_session_rsa"][m][s] for s in sess]) for m in models])
    Eh = np.array([np.mean([head["per_session_enc"][m][s] for s in sess]) for m in models])
    rho_h = spear(Rh, Eh)
    out = {"inputs_sha256": {"concordance32_CORRECTED.json": sha(RES / "p4_encoding" / "concordance32_CORRECTED.json"),
                             **{f"readrule/{Path(f).name}": sha(f) for f in sorted(glob.glob(str(RES / "readrule_20260926" / "per_model" / "*.json")))}},
           "cached_vs_headline_per_session_max_abs_diff": worst,
           "comparator": {"rho": rho_h, "family": cluster_boot(Rh, Eh, models, 20261004), "fisher": fisher(rho_h, 32)},
           "rules": {}}
    for rule in ("prefix_last", "prefix_mean", "full_last", "full_mean"):
        R, E, Rps, Eps = table(rule)
        rho = spear(R, E)
        fb = cluster_boot(R, E, models, 20261004, Rh, Eh)
        seeds = [cluster_boot(R, E, models, s, n_boot=N_BOOT)["ci95"] for s in range(20)] if rule == "prefix_last" else None
        cls = {}
        for c in ("causal", "masked", "encdec"):
            ix = [i for i, m in enumerate(models) if kind[m] == c]
            cls[c] = {"n": len(ix), "mean_rsa": float(R[ix].mean()), "mean_enc": float(E[ix].mean()),
                      "mean_rsa_published": float(Rh[ix].mean()), "mean_enc_published": float(Eh[ix].mean())}
        ent = {"rho": rho, "fisher_ci95": fisher(rho, 32), "family": fb, "delta_vs_comparator": rho - rho_h,
               "class_means": cls,
               "n_noncausal_enc_up": int(sum(E[i] > Eh[i] for i, m in enumerate(models) if kind[m] != "causal")),
               "n_noncausal_rsa_up": int(sum(R[i] > Rh[i] for i, m in enumerate(models) if kind[m] != "causal"))}
        if rule == "prefix_last":
            ent["seed_sweep_ci95"] = {"seeds": "0-19", "lower_range": [min(s[0] for s in seeds), max(s[0] for s in seeds)],
                                      "upper_range": [min(s[1] for s in seeds), max(s[1] for s in seeds)],
                                      "n_seeds_L_le_0_le_U": int(sum(s[0] <= 0 <= s[1] for s in seeds))}
            ent["splithalf_sb_own"] = {"rsa": splithalf_sb(Rps, models), "enc": splithalf_sb(Eps, models)}
            L, U = fb["ci95"]
            sb_min = min(ent["splithalf_sb_own"].values())
            if sb_min < 0.70:
                br = "D"
            elif U < 0:
                br = "A"
            elif rho < 0 and L <= 0 <= U:
                br = "B"
            else:
                br = "C"
            ent["branch_own_route"] = br
        out["rules"][rule] = ent
    stored = json.loads((RES / "readrule_20260926" / "readrule_summary.json").read_text())
    out["stored_summary_for_comparison"] = {r: {k: stored["variants"][r][k] for k in ("rho", "family_ci95", "delta_paired_family_ci95")}
                                            for r in ("prefix_last", "prefix_mean", "full_last", "full_mean")}
    out["stored_summary_for_comparison"]["primary_branch"] = stored["primary"]["branch"]
    out["stored_summary_sha256"] = sha(RES / "readrule_20260926" / "readrule_summary.json")
    return out


# ---------------------------------------------------------------- Part B: alpha32
def part_b():
    head = json.loads((RES / "p4_encoding" / "concordance32_CORRECTED.json").read_text())
    models = head["models"]
    sess = [str(s) for s in range(1, 13)]
    Rh = np.array([np.mean([head["per_session_rsa"][m][s] for s in sess]) for m in models])
    Eh = np.array([np.mean([head["per_session_enc"][m][s] for s in sess]) for m in models])
    rec = {}
    for f in sorted(glob.glob(str(RES / "alpha32_20260926" / "per_model" / "*.json"))):
        j = json.loads(Path(f).read_text()); rec[j["model"]] = j
    assert set(rec) == set(models)
    arms = {}
    for arm in ("orig", "exp", "exp_pt"):
        E = np.array([np.mean([fd[arm]["score"] for fd in rec[m]["folds"]]) for m in models])
        rho = spear(Rh, E)
        arms[arm] = {"rho": rho, "family": cluster_boot(Rh, E, models, 20261004, Rh, Eh),
                     "seed_sweep_upper_max": max(cluster_boot(Rh, E, models, s)["ci95"][1] for s in range(20)) if arm != "orig" else None}
    top, nontop = {}, {}
    for m in models:
        fl = rec[m]["folds"]
        top[m] = sum(abs(fd["orig"]["alpha"] - 1e6) < 1 for fd in fl)
        nontop[m] = [fd["session"] for fd in fl if abs(fd["orig"]["alpha"] - 1e6) >= 1]
    exp_top = {m: sum(fd["exp"]["alpha"] >= 1e14 * 0.999 for fd in rec[m]["folds"]) for m in models}
    return {"arms": arms,
            "orig_folds_at_1e6": int(sum(top.values())), "orig_folds_total": 12 * len(models),
            "per_checkpoint_orig_top_counts": sorted(set(top.values())),
            "nontop_fold_sessions": sorted({s for v in nontop.values() for s in v}),
            "checkpoints_whose_nontop_fold_is_session9": int(sum(v == ["9"] for v in nontop.values())),
            "exp_ge6_of_12_at_1e14": [m for m in models if exp_top[m] >= 6],
            "stored_primary": json.loads((RES / "alpha32_20260926" / "alpha32_summary.json").read_text())["primary"],
            "stored_summary_sha256": sha(RES / "alpha32_20260926" / "alpha32_summary.json")}


# ---------------------------------------------------------------- Part C/D: paper's own function
def part_c():
    sys.path.insert(0, str(MEG / "codes"))
    import recompute_corrected_stats as RC   # the production estimator, unchanged
    agg = RES / "p4_encoding" / "space_fix_20260925" / "aggregate"
    out = {"production_code_sha256": {"recompute_corrected_stats.py": sha(MEG / "codes" / "recompute_corrected_stats.py"),
                                      "model_families.py": sha(MEG / "codes" / "model_families.py")}, "cohorts": {}}
    for tag, fn in (("brennan", "concordance_spacefix_brennan_32.json"), ("masc27", "concordance_spacefix_masc27_32.json")):
        j = json.loads((agg / fn).read_text())
        rps, eps = j["per_session_rsa"], j["per_session_enc"]
        rho, lo, hi, nf = RC.family_clustered_ci(rps, eps, n_boot=N_BOOT, seed=0)
        sweep = [RC.family_clustered_ci(rps, eps, n_boot=N_BOOT, seed=s)[1:3] for s in range(20)]
        ms = j["models"]
        sess = sorted(set.intersection(*[set(rps[m]) for m in ms]))
        a = [np.mean([rps[m][s] for s in sess]) for m in ms]
        b = [np.mean([eps[m][s] for s in sess]) for m in ms]
        own = cluster_boot(a, b, ms, 20261004)
        out["cohorts"][tag] = {"receipt": str((agg / fn).relative_to(P4)), "receipt_sha256": sha(agg / fn),
                               "n_models": len(ms), "n_units": len(sess),
                               "rho_stored": j["concordance_spearman"], "fisher_ci95_stored": j["concordance_ci95"],
                               "rho": rho, "family_ci95_seed0": [lo, hi], "n_families": nf,
                               "seed_sweep_0_19": {"lower_range": [min(s[0] for s in sweep), max(s[0] for s in sweep)],
                                                   "upper_range": [min(s[1] for s in sweep), max(s[1] for s in sweep)],
                                                   "n_seeds_lower_gt_0": int(sum(s[0] > 0 for s in sweep))},
                               "independent_route": {"rho": spear(a, b), **own}}
    return out


def part_d():
    sys.path.insert(0, str(MEG / "codes"))
    import recompute_corrected_stats as RC
    out = {}
    c4 = json.loads((RES / "p4_encoding" / "concordance32_c4.json").read_text())
    head = json.loads((RES / "p4_encoding" / "concordance32_CORRECTED.json").read_text())
    rsa4, enc = c4["per_session_rsa_c4"], head["per_session_enc"]
    ups = []
    for s in range(20):
        r, lo, hi, _ = RC.family_clustered_ci(rsa4, enc, n_boot=N_BOOT, seed=s)
        ups.append(hi)
    out = {"rho_c4": r, "stored_family_ci95": c4["rho_c4_family_ci95"],
                            "upper_seed0": ups[0], "upper_by_seed_0_19": ups,
                            "n_seeds_upper_lt_0": int(sum(u < 0 for u in ups)),
                            "seed0_is_max": bool(ups[0] == max(ups)),
                            "upper_range": [min(ups), max(ups)]}
    return out


# ---------------------------------------------------------------- Part E: session 9, as already visible
def part_e():
    head = json.loads((RES / "p4_encoding" / "concordance32_CORRECTED.json").read_text())
    models = head["models"]
    sess = [str(s) for s in range(1, 13)]
    enc = np.array([[head["per_session_enc"][m][s] for s in sess] for m in models])
    rsa = np.array([[head["per_session_rsa"][m][s] for s in sess] for m in models])
    per_sess_rho = [spear(rsa[:, k], enc[:, k]) for k in range(12)]
    i9 = sess.index("9")
    oth = [k for k in range(12) if k != i9]
    return {"enc_session9_range": [float(enc[:, i9].min()), float(enc[:, i9].max())],
            "enc_other_sessions_model_mean_range": [float(enc[:, oth].mean(0).min()), float(enc[:, oth].mean(0).max())],
            "enc_session_means": dict(zip(sess, map(float, enc.mean(0)))),
            "rsa_session_means": dict(zip(sess, map(float, rsa.mean(0)))),
            "per_session_concordance": dict(zip(sess, per_sess_rho)),
            "n_sessions_negative": int(sum(r < 0 for r in per_sess_rho))}


PARTS = {"A_readrule_independent": part_a, "B_alpha32_independent": part_b,
         "C_cohort_ci_paper_function": part_c, "D_c4_seed_sweep_paper_function": part_d,
         "E_session9_visible": part_e}

if __name__ == "__main__":
    # usage: script.py PART  -> writes _part_<PART>.json ;  script.py merge -> final receipt
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in PARTS:
        (HERE / f"_part_{arg}.json").write_text(json.dumps(PARTS[arg](), indent=1))
        sys.exit(0)
    receipt = {"script": "results/paperA_rebuttal_20261004/recompute_paperA_rebuttal_20261004.py",
               "script_sha256": sha(__file__), "python": sys.version.split()[0], "numpy": np.__version__}
    for k, f in PARTS.items():
        pf = HERE / f"_part_{k}.json"
        receipt[k] = json.loads(pf.read_text()) if (arg == "merge" and pf.exists()) else f()
    (HERE / "paperA_rebuttal_recompute_20261004.json").write_text(json.dumps(receipt, indent=1))
    if arg == "merge":
        for k in PARTS:
            (HERE / f"_part_{k}.json").unlink(missing_ok=True)
    print("wrote paperA_rebuttal_recompute_20261004.json")
