"""EXP-A: confound-controlled, noise-ceiling-normalized 2x2 RSA.

Extends path1_2x2_rsa.py. For each (model, subject, layer) we compute, on the
upper-triangular RDM entries:

    raw_rho      = Spearman(neural_RDM, LLM_RDM)
    partial_rho  = partial Spearman(neural_RDM, LLM_RDM | C1..C4)

where the confound RDMs (all built over word_list_shared.txt, in RDM row order)
are word-level / lexical, which is the correct confound family for a word-level
RSA (not the time-series word-rate/position confounds of Hadidi 2025):

    C1 word length          |len_i - len_j|
    C2 log word-frequency   |logfreq_i - logfreq_j|   (wordfreq if importable;
                            else a Zipf proxy from rank in word_list; flagged)
    C3 orthographic         Levenshtein(w_i, w_j)
    C4 static embedding     LLM layer-0 (token-embedding) RDM  -- isolates the
                            CONTEXTUAL contribution; deepest "is it just static
                            embeddings?" control. Per-model, so partialling it
                            removes the model's own non-contextual baseline.

Partialling = residualize both neural-rank and LLM-rank vectors on the design
matrix [1, C1_rank, C2_rank, C3_rank, C4_rank], then Pearson on residuals
(= partial Spearman). C4 is per-model (uses that model's layer 0).

Noise-ceiling: read inter-group reliability for the cell from
reliability/<dataset>_region_time.tsv (mean_rho at region,time_ms). Report
rho / ceiling. Re-run the 2x2 AR x objective mixedlm on partial_rho.

Output (out_dir):
    rsa_confound_per_model_subject.tsv   raw_rho, partial_rho, c4_rho, ceiling, norm
    anova_2x2_partial.json               2x2 on partial_rho (+ raw for comparison)
    confound_summary.json                which confounds active, ceiling used

Usage (on <cluster>):
    python codes/paper4_nn_revision/path1_confound_rsa.py \
        --hidden_root results/path1_libribrain \
        --rdm_cache <scratch>/.../cache/rdm_libribrain_temporalL_BB_t350 \
        --reliability results/reliability/libribrain_region_time.tsv \
        --dataset libribrain --region temporal_L --time_ms 350 \
        --out_dir results/path1_confound_libribrain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

# Reuse the 2x2 cell assignments from the sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from path1_2x2_rsa import DEFAULT_MODELS, model_tag  # noqa: E402


# ---------- confound RDM builders (over the shared word list, RDM row order) ----------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def length_rdm(words):
    L = np.array([len(w) for w in words], dtype=float)
    return pdist(L[:, None], metric="cityblock")


def logfreq_rdm(words):
    """Returns (condensed RDM, source_str). Prefers `wordfreq`; falls back to a
    Zipf proxy from list order (flagged as weaker)."""
    try:
        from wordfreq import zipf_frequency
        z = np.array([zipf_frequency(w, "en") for w in words], dtype=float)
        src = "wordfreq.zipf_frequency(en)"
    except Exception:
        # Proxy: assume word_list roughly frequency-ordered is NOT safe; instead
        # use a flat rank so this confound is inert rather than misleading.
        z = np.zeros(len(words), dtype=float)
        src = "UNAVAILABLE(wordfreq) -> C2 inert"
    return pdist(z[:, None], metric="cityblock"), src


def ortho_rdm(words):
    n = len(words)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = levenshtein(words[i], words[j])
            D[i, j] = D[j, i] = d
    return squareform(D, checks=False)


def rdm_from_feat(feat):
    """Condensed correlation-distance RDM from a (n_words, d) feature matrix."""
    d = pdist(feat, metric="correlation")
    d = np.nan_to_num(d, nan=0.0)
    return d


# ---------- partial Spearman ----------

def _resid(y_rank, X):
    """Residualize rank-vector y on design matrix X (with intercept already in X)."""
    beta, *_ = np.linalg.lstsq(X, y_rank, rcond=None)
    return y_rank - X @ beta


def partial_spearman(neural_ranks, llm_d, confound_ds):
    """neural_ranks: pre-ranked condensed neural RDM.
    llm_d, confound_ds: condensed distance vectors (unranked)."""
    m = len(neural_ranks)
    llm_rank = rankdata(llm_d)
    # Design: intercept + ranked confounds.
    cols = [np.ones(m)]
    for cd in confound_ds:
        if cd is None:
            continue
        cr = rankdata(cd)
        if np.ptp(cr) == 0:  # inert confound (constant) -> skip
            continue
        cols.append(cr)
    X = np.vstack(cols).T
    ny = _resid(neural_ranks.astype(float), X)
    lx = _resid(llm_rank.astype(float), X)
    denom = float(np.sqrt((ny @ ny) * (lx @ lx)))
    return float((ny @ lx) / denom) if denom > 0 else 0.0


def plain_spearman(neural_ranks, llm_d):
    lr = rankdata(llm_d)
    a = neural_ranks - neural_ranks.mean()
    b = lr - lr.mean()
    denom = float(np.sqrt((a @ a) * (b @ b)))
    return float((a @ b) / denom) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden_root", required=True)
    ap.add_argument("--rdm_cache", required=True)
    ap.add_argument("--reliability", required=True, help="reliability tsv for noise ceiling")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--time_ms", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    hidden_root = Path(args.hidden_root)
    rdm_cache = Path(args.rdm_cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- noise ceiling for this cell ---
    rel = pd.read_csv(args.reliability, sep="\t")
    cell = rel[(rel["region"] == args.region) & (np.isclose(rel["time_ms"], args.time_ms))]
    ceiling = float(cell["mean_rho"].iloc[0]) if len(cell) else float("nan")
    print(f"[ceiling] {args.dataset} {args.region}@{args.time_ms}ms = {ceiling:.4f}")

    # --- neural RDMs + shared word list (RDM row order) ---
    sub_paths = sorted(rdm_cache.glob("sub-*.npy"))
    if not sub_paths:
        sys.exit(f"No sub-*.npy under {rdm_cache}")
    sub_rdms = {int(p.stem.split("-")[1]): np.load(p) for p in sub_paths}
    n_words = int((1 + np.sqrt(1 + 8 * len(next(iter(sub_rdms.values()))))) / 2)
    wl_path = rdm_cache / "word_list_shared.txt"
    words = wl_path.read_text().split()
    assert len(words) == n_words, f"word_list {len(words)} != n_words {n_words}"
    print(f"[neural] {len(sub_rdms)} groups, n_words={n_words}")

    # --- confound RDMs (word-level; shared across models except C4) ---
    c1 = length_rdm(words)
    c2, c2_src = logfreq_rdm(words)
    c3 = ortho_rdm(words)
    print(f"[confounds] C1 length, C2 {c2_src}, C3 orthographic, C4 per-model layer0")

    rows = []
    for model_dir in sorted(hidden_root.iterdir()):
        if not model_dir.is_dir() or not (model_dir / "hiddens.npy").exists():
            continue
        meta = json.loads((model_dir / "meta.json").read_text())
        model_id = meta["model"]
        info = DEFAULT_MODELS.get(model_id)
        if info is None:
            if all(k in meta for k in ("arch", "obj", "params_M")):
                info = {"arch": meta["arch"], "obj": meta["obj"], "params_M": meta["params_M"]}
            else:
                print(f"[skip] unknown {model_id}")
                continue
        hiddens = np.load(model_dir / "hiddens.npy")  # (L+1, n_words, d)
        Lp1, nw, _ = hiddens.shape
        if nw != n_words:
            print(f"[skip] {model_id}: n_words {nw} != {n_words}")
            continue
        c4 = rdm_from_feat(hiddens[0])  # static token-embedding RDM (per model)

        # per-(subject, layer): raw & partial. Pick brain-best layer by partial_rho.
        for si, subj in enumerate(sorted(sub_rdms.keys())):
            ranks = sub_rdms[subj].astype(float)
            raw = np.empty(Lp1); par = np.empty(Lp1)
            for li in range(Lp1):
                llm_d = rdm_from_feat(hiddens[li])
                raw[li] = plain_spearman(ranks, llm_d)
                # do not partial layer 0 against its own C4 (would be ~0); for li==0 skip C4
                conf = [c1, c2, c3] + ([c4] if li != 0 else [])
                par[li] = partial_spearman(ranks, llm_d, conf)
            bl = int(np.argmax(par))
            rows.append({
                "model": model_id, "tag": model_tag(model_id),
                "arch": info["arch"], "obj": info["obj"], "params_M": info["params_M"],
                "subject": subj, "best_layer": bl,
                "best_layer_frac": bl / max(Lp1 - 1, 1),
                "raw_rho": float(raw[bl]), "partial_rho": float(par[bl]),
                "raw_rho_at_partial_best": float(raw[bl]),
                "ceiling": ceiling,
                "partial_norm": float(par[bl] / ceiling) if ceiling and ceiling == ceiling else float("nan"),
            })
        print(f"[done] {model_id}  mean partial best={par.max():.4f}")

    df = pd.DataFrame(rows)
    tsv = out_dir / "rsa_confound_per_model_subject.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    print(f"[save] {tsv} ({len(df)} rows)")

    # --- 2x2 on partial_rho (and raw for comparison) ---
    summary = {"ceiling": ceiling, "confound_C2_source": c2_src, "n_obs": int(len(df))}
    try:
        import statsmodels.formula.api as smf
        df["log_params"] = np.log10(df["params_M"])
        for target in ("partial_rho", "raw_rho"):
            res = smf.mixedlm(f"{target} ~ C(arch) * C(obj) + log_params",
                              df, groups=df["subject"]).fit(method="lbfgs", maxiter=300, disp=False)
            summary[target] = {
                "params": {k: float(v) for k, v in res.params.items()},
                "pvalues": {k: float(v) for k, v in res.pvalues.items()},
            }
        # causal vs bidirectional cell means on partial
        cm = df.groupby("obj")["partial_rho"].agg(["mean", "std", "count"]).reset_index()
        summary["partial_obj_means"] = cm.to_dict(orient="records")
    except Exception as e:
        summary["error"] = str(e)
        summary["partial_obj_means"] = (
            df.groupby("obj")["partial_rho"].mean().reset_index().to_dict(orient="records"))
    (out_dir / "anova_2x2_partial.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "confound_summary.json").write_text(json.dumps(
        {"ceiling": ceiling, "C2_source": c2_src, "n_words": n_words,
         "region": args.region, "time_ms": args.time_ms, "dataset": args.dataset}, indent=2))
    print("[save] anova_2x2_partial.json")
    print(json.dumps(summary.get("partial_obj_means", {}), indent=2))


if __name__ == "__main__":
    main()
