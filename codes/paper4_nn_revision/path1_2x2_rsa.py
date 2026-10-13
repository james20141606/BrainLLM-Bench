"""Path 1: per-subject RSA + 2×2 ANOVA decomposition (architecture × objective).

Takes per-model hidden states emitted by path1_extract_hiddens.py
plus per-subject MEG-MASC neural RDMs, computes per-(model, subject, layer)
RSA, picks each model's brain-best layer, then runs a 2×2 ANOVA with

    alignment ~ architecture + objective + architecture:objective + (1|subject)

reporting main effects and interaction.

Output:
    results/path1/rsa_per_model_subject.tsv
    results/path1/anova_2x2.json
    paper4_temporalalign/figures_v2/fig60_path1_2x2_anova_nn.pdf/.png

Usage:
    python codes/paper4_nn_revision/path1_2x2_rsa.py \
        --hidden_root results/path1 \
        --rdm_cache <scratch>/.../rdm_temporal_R_BB_t767 \
        --models_config codes/paper4_nn_revision/path1_models.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Default 2x2 cell assignment for each model name we expect.
# arch: encoder / encdec / decoder
# obj:  causal-LM / bidirectional / span / mixed
# In the 2x2 we collapse to {causal, bidirectional} where:
#   causal = next-token (decoder-only-causal)
#   bidirectional = anything that lets the model attend bidirectionally
#                   (encoder-only MLM, encoder-decoder span, etc.)
DEFAULT_MODELS = {
    # Encoder-only x bidirectional
    "bert-base-uncased":      {"arch": "encoder", "obj": "bidirectional", "params_M": 110},
    "bert-large-uncased":     {"arch": "encoder", "obj": "bidirectional", "params_M": 340},
    "roberta-base":           {"arch": "encoder", "obj": "bidirectional", "params_M": 125},
    # Encoder-decoder x span (BERT-style supervision via masked spans)
    "t5-base":                {"arch": "encdec",  "obj": "bidirectional", "params_M": 220},
    "t5-large":               {"arch": "encdec",  "obj": "bidirectional", "params_M": 770},
    "google/flan-t5-base":    {"arch": "encdec",  "obj": "bidirectional", "params_M": 220},
    "google/flan-t5-large":   {"arch": "encdec",  "obj": "bidirectional", "params_M": 770},
    "google/flan-t5-xl":      {"arch": "encdec",  "obj": "bidirectional", "params_M": 3000},
    # Decoder-only x causal
    "gpt2":                   {"arch": "decoder", "obj": "causal", "params_M": 124},
    "gpt2-medium":            {"arch": "decoder", "obj": "causal", "params_M": 355},
    "EleutherAI/pythia-70m":  {"arch": "decoder", "obj": "causal", "params_M": 70},
    "EleutherAI/pythia-160m": {"arch": "decoder", "obj": "causal", "params_M": 160},
    "EleutherAI/pythia-410m": {"arch": "decoder", "obj": "causal", "params_M": 410},
    "EleutherAI/pythia-1.4b": {"arch": "decoder", "obj": "causal", "params_M": 1400},
    "EleutherAI/pythia-2.8b": {"arch": "decoder", "obj": "causal", "params_M": 2800},
    "meta-llama/Llama-3.2-1B":{"arch": "decoder", "obj": "causal", "params_M": 1200},
    "meta-llama/Llama-3.2-3B":{"arch": "decoder", "obj": "causal", "params_M": 3200},
    "meta-llama/Llama-3.1-8B":{"arch": "decoder", "obj": "causal", "params_M": 8000},
    # Decoder-only x bidirectional-flow / span (FIM-trained, mixed-denoiser)
    "codellama/CodeLlama-7b-hf": {"arch": "decoder", "obj": "bidirectional", "params_M": 7000},
    "google/ul2":             {"arch": "encdec",  "obj": "bidirectional", "params_M": 20000},
    "google/flan-ul2":        {"arch": "encdec",  "obj": "bidirectional", "params_M": 20000},
}


def model_tag(model_id: str) -> str:
    return model_id.replace("/", "__")


def rdm_corr(neural_ranks: np.ndarray, llm_feat: np.ndarray) -> float:
    """Spearman = Pearson on rank vectors of upper-tri RDM entries."""
    from scipy.spatial.distance import pdist
    from scipy.stats import rankdata
    if len(llm_feat) <= 2:
        return 0.0
    llm_d = pdist(llm_feat, metric="correlation")
    if np.isnan(llm_d).any() or (llm_d == 0).all():
        return 0.0
    llm_ranks = rankdata(llm_d)
    a = neural_ranks - neural_ranks.mean()
    b = llm_ranks - llm_ranks.mean()
    denom = float(np.sqrt((a @ a) * (b @ b)))
    return float((a @ b) / denom) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden_root", required=True,
                    help="dir containing one subdir per model with hiddens.npy")
    ap.add_argument("--rdm_cache", required=True,
                    help="dir containing per-subject sub-{N}.npy neural RDMs")
    ap.add_argument("--out_dir", default="results/path1")
    args = ap.parse_args()

    hidden_root = Path(args.hidden_root)
    rdm_cache = Path(args.rdm_cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover cached subjects
    sub_paths = sorted(rdm_cache.glob("sub-*.npy"))
    if not sub_paths:
        sys.exit(f"No sub-*.npy under {rdm_cache}")
    sub_rdms = {int(p.stem.split("-")[1]): np.load(p) for p in sub_paths}
    n_words_neural = int((1 + np.sqrt(1 + 8 * len(next(iter(sub_rdms.values()))))) / 2)
    print(f"[neural] {len(sub_rdms)} subjects, n_words={n_words_neural}")

    rows = []
    for model_dir in sorted(hidden_root.iterdir()):
        if not model_dir.is_dir() or not (model_dir / "hiddens.npy").exists():
            continue
        meta = json.loads((model_dir / "meta.json").read_text())
        model_id = meta["model"]
        info = DEFAULT_MODELS.get(model_id)
        if info is None:
            # fall back to arch/obj/params_M carried in the model's own meta.json
            if all(k in meta for k in ("arch", "obj", "params_M")):
                info = {"arch": meta["arch"], "obj": meta["obj"], "params_M": meta["params_M"]}
            else:
                print(f"[skip] unknown model {model_id} (not in DEFAULT_MODELS, no meta arch/obj)")
                continue
        hiddens = np.load(model_dir / "hiddens.npy")  # (L+1, n_words, d)
        L_plus_1, n_words_llm, _ = hiddens.shape
        if n_words_llm != n_words_neural:
            print(f"[skip] {model_id}: LLM n_words={n_words_llm} != neural {n_words_neural}")
            continue

        # Per-(subject, layer) RSA
        per_layer = np.zeros((L_plus_1, len(sub_rdms)))
        for li in range(L_plus_1):
            feat = hiddens[li]  # (n_words, d)
            for si, (subj, ranks) in enumerate(sorted(sub_rdms.items())):
                per_layer[li, si] = rdm_corr(ranks, feat)

        # Per-subject best layer + alignment score
        for si, subj in enumerate(sorted(sub_rdms.keys())):
            best_layer = int(np.argmax(per_layer[:, si]))
            best_rho = float(per_layer[best_layer, si])
            rows.append({
                "model": model_id,
                "tag": model_tag(model_id),
                "arch": info["arch"],
                "obj": info["obj"],
                "params_M": info["params_M"],
                "subject": subj,
                "best_layer": best_layer,
                "best_layer_frac": best_layer / max(L_plus_1 - 1, 1),
                "best_rho": best_rho,
            })
        print(f"[done] {model_id}  mean best_rho={per_layer.max(axis=0).mean():.3f}")

    df = pd.DataFrame(rows)
    tsv = out_dir / "rsa_per_model_subject.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    print(f"[save] {tsv}  ({len(df)} rows)")

    # 2×2 ANOVA (mixed effects approximation: simple two-way ANOVA on per-subject
    # mean-over-models within each cell, with subject as block).
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        # Center params (log scale) so the 2×2 isn't confounded with size.
        df["log_params"] = np.log10(df["params_M"])
        model = smf.mixedlm(
            "best_rho ~ C(arch) * C(obj) + log_params",
            df, groups=df["subject"],
        )
        res = model.fit(method="lbfgs", maxiter=200, disp=False)
        anova_summary = {
            "n_obs": int(len(df)),
            "n_models": int(df["model"].nunique()),
            "n_subjects": int(df["subject"].nunique()),
            "params": {k: float(v) for k, v in res.params.items()},
            "pvalues": {k: float(v) for k, v in res.pvalues.items()},
            "aic": float(res.aic),
            "bic": float(res.bic),
        }
    except Exception as e:
        # Fallback: per-cell means + Welch's t-tests
        cell = df.groupby(["arch", "obj"])["best_rho"].agg(["mean", "std", "count"])
        anova_summary = {
            "fallback": "no statsmodels; reporting cell means + simple t-tests",
            "cell_means": cell.reset_index().to_dict(orient="records"),
            "error": str(e),
        }
    (out_dir / "anova_2x2.json").write_text(json.dumps(anova_summary, indent=2))
    print(f"[save] anova_2x2.json")
    print(json.dumps(anova_summary, indent=2))


if __name__ == "__main__":
    main()
