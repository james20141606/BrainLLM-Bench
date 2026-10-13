#!/usr/bin/env python3
"""Post hoc 30k-word rogue-dose RSA/encoding follow-up for BrainLLM-Bench.

Both readouts use the same frozen primary LibriBrain cell, shared subsample,
middle fixed-depth feature rows, and session grouping as p4_concordance32.py.
Unlike invariance_intervention.py this recomputes ridge on each transformed X.
Each Slurm array task owns one model and writes an atomic per-model receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("<PROJECT>")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "paper4_nn_revision"))
import p4_concordance32 as C  # noqa: E402

MODELS = [
    "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-70m", "bert-base-uncased", "bert-large-uncased",
    "gpt2", "gpt2-medium", "roberta-base", "t5-base", "t5-large",
]
OUT = ROOT / "results/p4_encoding/rogue_encoding_20260922"
REFERENCE = ROOT / "results/p4_encoding/concordance32.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def transformed(X: np.ndarray, share: float, seed: int) -> tuple[np.ndarray, dict]:
    """Freeze one sample vector and feature direction per model across doses.

    Uses the old rogue intervention's variance-share scale and global centering.
    The realized share may differ slightly from its nominal value because the
    random component can covary with X; both are recorded.
    """
    if share == 0:
        return X, {"nominal_share": 0.0, "realized_added_variance_share": 0.0,
                   "rogue_seed": seed, "condition": "raw_baseline"}
    if share == -1:
        return X - X.mean(0, keepdims=True), {
            "nominal_share": 0.0, "realized_added_variance_share": 0.0,
            "rogue_seed": seed, "condition": "center_only"}
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(0, keepdims=True)
    v = rng.normal(size=X.shape[1]); v /= np.linalg.norm(v)
    a = rng.normal(size=(X.shape[0], 1)); a -= a.mean()
    base = float((Xc ** 2).sum())
    scale = np.sqrt(share / (1 - share) * base / max(float((a ** 2).sum()), 1e-12))
    add = scale * a * v[None, :]
    Xt = Xc + add
    realized = float((add ** 2).sum() / max(float((Xt ** 2).sum()), 1e-12))
    return Xt, {"nominal_share": share, "realized_added_variance_share": realized,
                "condition": "center_plus_rank1",
                "rogue_seed": seed, "direction_sha256": sha256_array(v),
                "sample_vector_sha256": sha256_array(a), "scale": float(scale)}


def model_seed(model: str) -> int:
    return int.from_bytes(hashlib.sha256(("rogue-20260922:" + model).encode()).digest()[:4], "big")


def prepare() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame, dict, dict]:
    data = C.DATA
    meta = pd.read_csv(data / "trial_metadata.tsv", sep="\t")
    sel = C.subsample(meta, 2500, 42, "session")
    ms = meta.iloc[sel].reset_index(drop=True)
    groups = ms["session"].to_numpy()
    words = ms["word_text"].astype(str).tolist()
    ep = np.load(data / "epochs_data.npy", mmap_mode="r")
    mask = C.get_neuromag_region_masks(C.get_canonical_neuromag306_order())["temporal_L"]
    s0 = int(round((0.300 - C.TMIN) * C.SFREQ))
    s1 = int(round((0.400 - C.TMIN) * C.SFREQ))
    pat = C.neural_patterns(ep, sel, mask, s0, s1, "broadband")
    Y = pat.reshape(len(pat), int(mask.sum()), s1 - s0).mean(axis=2).astype(np.float64)
    conf = {}
    for s in np.unique(groups):
        w = [words[i] for i in np.where(groups == s)[0]]
        fr, src = C.logfreq_rdm(w)
        if "UNAVAILABLE" in src or np.ptp(fr) == 0:
            raise RuntimeError(f"session {s}: frequency confound unavailable ({src})")
        conf[s] = [C.length_rdm(w), fr, C.ortho_rdm(w)]
    info = {"n_metadata": len(meta), "n_words": len(sel),
            "n_sessions": len(np.unique(groups)), "selected_index_sha256": sha256_array(sel),
            "selected_words_sha256": hashlib.sha256("\n".join(words).encode()).hexdigest(),
            "neural_shape": list(pat.shape), "target_shape": list(Y.shape),
            "target_sha256": sha256_array(Y), "window_samples": [s0, s1],
            "source_metadata": str(data / "trial_metadata.tsv"),
            "source_epochs": str(data / "epochs_data.npy")}
    return sel, groups, pat, words, ms, conf, {"Y": Y, "info": info}


def verify_baseline(model: str, rsa: dict, enc: dict, receipt: dict,
                    tolerance: float) -> dict:
    rr = receipt["per_session_rsa"][model]
    ee = receipt["per_session_enc"][model]
    if set(map(str, rsa)) != set(rr) or set(map(str, enc)) != set(ee):
        raise RuntimeError(f"baseline session keys differ for {model}")
    dr = max(abs(float(rsa[s]) - float(rr[str(s)])) for s in rsa)
    de = max(abs(float(enc[s]) - float(ee[str(s)])) for s in enc)
    if dr > tolerance or de > tolerance:
        raise RuntimeError(f"baseline mismatch {model}: RSA {dr:.3g}, encoding {de:.3g}; tolerance {tolerance}")
    return {"rsa_max_abs_session_diff": dr, "encoding_max_abs_session_diff": de,
            "tolerance": tolerance, "passed": True}


BASELINE_CONFIG_KEYS = (
    "selected_index_sha256", "selected_words_sha256", "target_sha256",
    "selected_feature_sha256", "reference_sha256", "generator_sha256",
    "alpha_grid", "feature_shape",
)
STAGE2_DOSES = {0.0, -1.0, 0.3, 0.7}
CONDITION_FOR_DOSE = {0.0: "raw_baseline", -1.0: "center_only",
                      0.3: "center_plus_rank1", 0.7: "center_plus_rank1"}


def normalized_baseline(prior: dict, model: str, baseline_target: Path) -> dict:
    """Accept historical dose-0 schema, but never change its scores or source file."""
    if prior.get("model") != model or prior.get("baseline_check", {}).get("passed") is not True:
        raise RuntimeError(f"unverified baseline receipt: {baseline_target}")
    if set(float(r["dose"]) for r in prior.get("doses", [])) != {0.0}:
        raise RuntimeError(f"baseline file has non-baseline doses: {baseline_target}")
    for key in BASELINE_CONFIG_KEYS:
        if key not in prior.get("config", {}):
            raise RuntimeError(f"baseline config missing {key}: {baseline_target}")
    raw = dict(prior["doses"][0])
    if raw.get("condition", "raw_baseline") != "raw_baseline":
        raise RuntimeError(f"baseline condition is not raw X: {baseline_target}")
    raw["condition"] = "raw_baseline"
    rsa = raw.get("rsa_per_session", {})
    enc = raw.get("encoding_per_session", {})
    if len(rsa) != 12 or len(enc) != 12 or set(rsa) != set(enc):
        raise RuntimeError(f"baseline requires matching 12-session readouts: {baseline_target}")
    if not np.all(np.isfinite([raw["rsa_mean"], raw["encoding_mean"],
                               *rsa.values(), *enc.values()])):
        raise RuntimeError(f"baseline contains nonfinite score: {baseline_target}")
    return raw


def validate_stage2_receipt(out: dict) -> None:
    rows = out.get("doses", [])
    if len(rows) != 4 or set(float(r["dose"]) for r in rows) != STAGE2_DOSES:
        raise RuntimeError("stage2 receipt missing or repeating a required dose")
    if out.get("baseline_check", {}).get("passed") is not True:
        raise RuntimeError("stage2 receipt lacks verified baseline")
    baseline_sessions = set(next(r for r in rows if r["dose"] == 0)["encoding_per_session"])
    if len(baseline_sessions) != 12:
        raise RuntimeError("stage2 baseline lacks 12 sessions")
    for r in rows:
        dose = float(r["dose"])
        if r.get("condition") != CONDITION_FOR_DOSE[dose]:
            raise RuntimeError(f"stage2 dose {dose} condition mismatch")
        rsa, enc = r.get("rsa_per_session", {}), r.get("encoding_per_session", {})
        if set(rsa) != baseline_sessions or set(enc) != baseline_sessions:
            raise RuntimeError(f"stage2 dose {dose} session roster mismatch")
        if not np.all(np.isfinite([r["rsa_mean"], r["encoding_mean"],
                                   *rsa.values(), *enc.values()])):
            raise RuntimeError(f"stage2 dose {dose} has nonfinite score")


def schema_preflight(baseline_dir: Path, output_dir: Path) -> None:
    """No model compute: normalize all historical baselines and assemble stub JSON.

    Writes ONLY to a caller-selected temporary directory. Stub scores copy
    baseline values and are never scientific results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_config = None
    for index, model in enumerate(MODELS):
        source = baseline_dir / f"model_{index:02d}.json"
        prior = json.loads(source.read_text())
        raw = normalized_baseline(prior, model, source)
        current = {k: prior["config"][k] for k in (
            "selected_index_sha256", "selected_words_sha256", "target_sha256",
            "reference_sha256", "generator_sha256", "alpha_grid")}
        if shared_config is None:
            shared_config = current
        elif current != shared_config:
            raise RuntimeError(f"shared primary input/config mismatch: {source}")
        rows = [raw]
        for dose in (-1.0, 0.3, 0.7):
            stub = dict(raw)
            stub["dose"] = dose
            stub["condition"] = CONDITION_FOR_DOSE[dose]
            stub["transform"] = {"schema_preflight_stub": True, "nominal_share": max(dose, 0.0)}
            rows.append(stub)
        assembled = {"schema_preflight_stub": True, "source_baseline": str(source),
                     "model": model, "kind": prior["kind"], "config": prior["config"],
                     "baseline_check": prior["baseline_check"], "doses": rows}
        validate_stage2_receipt(assembled)
        target = output_dir / f"schema_model_{index:02d}.json"
        target.write_text(json.dumps(assembled, indent=2))
        print(f"[SCHEMA PASS] {index} {model} -> {target}", flush=True)
    print(f"[SCHEMA PREFLIGHT PASS] {len(MODELS)} baselines; temporary stub output only", flush=True)


def run_one(index: int, doses: list[float], out_dir: Path, tolerance: float,
            reuse_baseline: bool) -> None:
    model = MODELS[index]
    receipt = json.loads(REFERENCE.read_text())
    if model not in receipt["models"]:
        raise RuntimeError(f"{model} absent from primary receipt")
    if receipt["shared_subsample"] != {"max_per_session": 2500, "seed": 42,
                                        "n_words": 30000, "n_sessions": 12}:
        raise RuntimeError("primary reference does not use 30k shared words")
    cell = receipt["cell"]
    if cell["region"] != "temporal_L" or cell["window_ms"] != [300.0, 400.0] or cell["depth_frac"] != 0.5:
        raise RuntimeError("primary reference cell differs")
    sel, groups, pat, words, ms, conf, bundle = prepare()
    Y, info = bundle["Y"], bundle["info"]
    fp = C.mid_layer_path(model, C.FEAT)
    F = np.load(fp, mmap_mode="r")
    if len(F) < int(sel.max()) + 1:
        raise RuntimeError(f"{model}: fixed-depth cache has only {len(F)} rows for selection max {sel.max()}")
    X = np.asarray(F[sel], dtype=np.float64)
    info["feature_path"] = str(fp)
    info["feature_file_sha256"] = sha256_file(fp)
    info["selected_feature_sha256"] = sha256_array(X)
    info["feature_shape"] = list(X.shape)
    info["generator_sha256"] = sha256_file(Path(C.__file__))
    info["script_sha256"] = sha256_file(Path(__file__))
    info["reference_sha256"] = sha256_file(REFERENCE)
    info["alpha_grid"] = [float(x) for x in C.ALPHAS]
    info["folds"] = "LeaveOneGroupOut(session); X standardized from training fold only; RidgeCV alpha selected on training fold"
    info["analysis_status"] = "post hoc diagnostic, not preregistered"
    info["rogue_scope"] = "additive sample-varying rank-one intervention; cannot exclude all geometry explanations"
    info["versions"] = {"numpy": np.__version__, "pandas": pd.__version__}
    out = {"model": model, "kind": C.load_models()[model], "config": info,
           "baseline_check": None, "doses": []}
    baseline_target = out_dir / f"model_{index:02d}.json"
    target = (out_dir / f"dose_model_{index:02d}.json") if reuse_baseline else baseline_target
    saved_baseline = None
    if reuse_baseline:
        if not baseline_target.exists():
            raise RuntimeError(f"no verified baseline receipt to reuse: {baseline_target}")
        prior = json.loads(baseline_target.read_text())
        saved_baseline = normalized_baseline(prior, model, baseline_target)
        for key in BASELINE_CONFIG_KEYS:
            if prior["config"][key] != info[key]:
                raise RuntimeError(f"baseline input/config drift in {key}: {baseline_target}")
        out["baseline_check"] = prior["baseline_check"]
        print(f"{model}: reusing verified primary-protocol baseline from {baseline_target}", flush=True)
    for dose in doses:
        if dose == 0 and saved_baseline is not None:
            out["doses"].append(saved_baseline)
            continue
        Xt, transform_meta = transformed(X, dose, model_seed(model))
        rsa = C.rsa_per_session(pat, Xt, words, ms, conf, "session")
        enc = C.encoding_per_session(Xt, Y, groups)
        if dose == 0:
            out["baseline_check"] = verify_baseline(model, rsa, enc, receipt, tolerance)
        out["doses"].append({"dose": dose, "condition": transform_meta["condition"],
                             "transform": transform_meta,
                             "rsa_mean": float(np.mean(list(rsa.values()))),
                             "encoding_mean": float(np.mean(list(enc.values()))),
                             "rsa_per_session": {str(k): float(v) for k, v in rsa.items()},
                             "encoding_per_session": {str(k): float(v) for k, v in enc.items()}})
        print(f"{model} dose={dose:.1f} RSA={out['doses'][-1]['rsa_mean']:+.6f} ENC={out['doses'][-1]['encoding_mean']:+.6f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if reuse_baseline:
        validate_stage2_receipt(out)
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(out, indent=2))
    os.replace(temp, target)
    print(f"WROTE {target}", flush=True)


def aggregate(out_dir: Path, doses: list[float]) -> None:
    rows = []
    for i in range(len(MODELS)):
        p = out_dir / f"{'dose_' if len(doses) > 1 else ''}model_{i:02d}.json"
        if not p.exists():
            raise RuntimeError(f"missing model receipt: {p}")
        d = json.loads(p.read_text())
        if d["model"] != MODELS[i]:
            raise RuntimeError(f"model index mismatch: {p}")
        have = {float(r["dose"]): r for r in d["doses"]}
        if set(doses) - set(have):
            raise RuntimeError(f"missing doses {set(doses)-set(have)}: {p}")
        rows.append(d)
    summary = {"scope": "post hoc 10-model primary-cell diagnostic; recomputed RSA and encoding",
               "models": MODELS, "doses": doses, "per_model": [], "rankings": []}
    for d in rows:
        summary["per_model"].append({"model": d["model"], "kind": d["kind"],
                                    "baseline_check": d["baseline_check"],
                                    "scores": [{k: v for k, v in r.items()
                                                if k in ("dose", "condition", "rsa_mean", "encoding_mean")}
                                               for r in d["doses"]]})
    base_e = np.array([next(r for r in d["doses"] if r["dose"] == 0)["encoding_mean"] for d in rows])
    base_r = np.array([next(r for r in d["doses"] if r["dose"] == 0)["rsa_mean"] for d in rows])
    center_e = (np.array([next(r for r in d["doses"] if r["dose"] == -1)["encoding_mean"]
                          for d in rows]) if -1 in doses else None)
    center_r = (np.array([next(r for r in d["doses"] if r["dose"] == -1)["rsa_mean"]
                          for d in rows]) if -1 in doses else None)
    families = [m.split("/")[0] if "/" in m else m.split("-")[0]
                for m in MODELS]
    # Treat checkpoint families as blocks; ten checkpoints are not ten fully
    # independent architecture draws. This is descriptive uncertainty only.
    blocks = {f: np.where(np.asarray(families) == f)[0] for f in sorted(set(families))}
    rng = np.random.default_rng(20260922)

    def family_bootstrap(a: np.ndarray, b: np.ndarray, n_iter: int = 2000) -> list[float] | None:
        vals = []
        names = list(blocks)
        for _ in range(n_iter):
            chosen = rng.choice(names, size=len(names), replace=True)
            ix = np.concatenate([blocks[x] for x in chosen])
            if len(np.unique(a[ix])) < 2 or len(np.unique(b[ix])) < 2:
                continue
            rho = float(spearmanr(a[ix], b[ix]).statistic)
            if np.isfinite(rho):
                vals.append(rho)
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None

    summary["uncertainty"] = {"method": "family-block bootstrap, 2000 draws, seed 20260922",
                              "families": {k: [MODELS[i] for i in v] for k, v in blocks.items()},
                              "scope": "descriptive; few family blocks, no general geometry exclusion"}
    for dose in doses:
        rv = np.array([next(r for r in d["doses"] if r["dose"] == dose)["rsa_mean"] for d in rows])
        ev = np.array([next(r for r in d["doses"] if r["dose"] == dose)["encoding_mean"] for d in rows])
        record = {"dose": dose,
            "condition": "raw_baseline" if dose == 0 else ("center_only" if dose == -1 else "center_plus_rank1"),
            "rsa_model_order_desc": [MODELS[i] for i in np.argsort(-rv)],
            "encoding_model_order_desc": [MODELS[i] for i in np.argsort(-ev)],
            "rsa_rank_corr_vs_baseline": float(spearmanr(rv, base_r).statistic),
            "rsa_rank_corr_family_ci95": family_bootstrap(rv, base_r),
            "encoding_rank_corr_vs_baseline": float(spearmanr(ev, base_e).statistic),
            "encoding_rank_corr_family_ci95": family_bootstrap(ev, base_e),
            "rsa_encoding_concordance": float(spearmanr(rv, ev).statistic),
            "rsa_encoding_concordance_family_ci95": family_bootstrap(rv, ev),
            "encoding_max_abs_score_shift": float(np.max(np.abs(ev - base_e)))}
        if center_e is not None and center_r is not None:
            record["rsa_rank_corr_vs_center_only"] = float(spearmanr(rv, center_r).statistic)
            record["encoding_rank_corr_vs_center_only"] = float(spearmanr(ev, center_e).statistic)
            record["encoding_max_abs_score_shift_vs_center_only"] = float(np.max(np.abs(ev - center_e)))
        summary["rankings"].append(record)
    target = out_dir / "summary.json"
    target.write_text(json.dumps(summary, indent=2))
    print(f"WROTE {target}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-index", type=int)
    ap.add_argument("--doses", default="0,0.3,0.7")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--reuse-baseline", action="store_true",
                    help="reuse a verified dose-0 receipt after checking hashes/config")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print parsed argv and plan hash without reading caches or writing receipts")
    ap.add_argument("--schema-preflight", action="store_true",
                    help="assemble clearly labelled stub receipts from every historical baseline")
    ap.add_argument("--baseline-dir", type=Path, default=OUT)
    a = ap.parse_args()
    # Slurm --export uses commas to separate variables. Accept a colon-delimited
    # value so DOSES=0:-1:0.3:0.7 reaches the task intact.
    doses = [float(x) for x in a.doses.replace(":", ",").split(",")]
    if len(set(doses)) != len(doses) or 0 not in doses or any((x < 0 and x != -1) or x >= 1 for x in doses):
        raise SystemExit("doses must be unique, include 0; -1 denotes center-only, other doses lie in [0,1)")
    if a.reuse_baseline and set(doses) != {0.0, -1.0, 0.3, 0.7}:
        raise SystemExit("stage2 requires exactly doses {0,-1,0.3,0.7}")
    plan = {"doses": doses, "reuse_baseline": a.reuse_baseline,
            "model_index": a.model_index, "max_per_session": 2500,
            "seed": 42, "alpha_grid": [float(x) for x in C.ALPHAS]}
    print(f"[argv] {sys.argv}", flush=True)
    print(f"[plan_sha256] {hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()}", flush=True)
    if a.dry_run:
        print(f"[parsed_doses] {doses}", flush=True)
        return
    if a.schema_preflight:
        if a.out_dir.resolve() == a.baseline_dir.resolve():
            raise SystemExit("schema-preflight output must be a separate temporary directory")
        schema_preflight(a.baseline_dir, a.out_dir)
        return
    if a.aggregate:
        aggregate(a.out_dir, doses)
    elif a.model_index is not None and 0 <= a.model_index < len(MODELS):
        run_one(a.model_index, doses, a.out_dir, a.tolerance, a.reuse_baseline)
    else:
        raise SystemExit("--model-index must be 0..9 unless --aggregate")


if __name__ == "__main__":
    main()
