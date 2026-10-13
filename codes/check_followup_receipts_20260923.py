#!/usr/bin/env python3
"""Pin the two NEW PaperA post hoc follow-ups to per-model receipts and TeX.

This is independent of bench.reproduce's historical stored-result checks.
It performs no model inference or external writes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper4_temporalalign/tex/paperA_benchmark.tex").read_text()
RESULTS = ROOT / "paper4_temporalalign/results"
if not RESULTS.is_dir():
    RESULTS = ROOT / "results"  # release copy's flattened results layout
ALPHA = RESULTS / "p4_encoding/alpha_sensitivity_20260922"
ROGUE = RESULTS / "p4_encoding/rogue_encoding_20260922"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def near(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) <= tol


def table_before_label(label: str) -> str:
    pos = TEX.index(r"\label{" + label + "}")
    start = TEX.rfind(r"\begin{table}", 0, pos)
    end = TEX.index(r"\end{table}", pos) + len(r"\end{table}")
    return TEX[start:end]


def main() -> None:
    a = load(ALPHA / "summary.json")
    assert a["status"] == "validated"
    correction = a["provenance_correction"]
    assert correction["actual_original_fit_source_git_commit"] == "d89dc54"
    assert correction["actual_original_fit_source_sha256_for_14B_and_72B"] == \
        "bcec091eb03b64eb0f5f7c3f9e8e26661230ff34f1122fe02a9b1871fc28b8d7"
    assert correction["saved_end_of_job_on_disk_sha256_for_14B_and_72B"] == \
        "31593a6efe5e564714f00fb59c10ba515ffebce1634ea6558da0180521279dfa"
    assert correction["fit_and_scoring_difference"].startswith("none;")
    assert correction["raw_receipts_preserved"] is True
    old, new = [], []
    alpha_table = table_before_label("tab:alpha-grid")
    for i, row in enumerate(a["model_results"]):
        d = load(ALPHA / f"qwen25_{i}_full.json")
        assert row["model"] == d["model"] and len(d["folds"]) == 12
        assert {f["session"] for f in d["folds"]} == {str(x) for x in range(1, 13)}
        assert max(abs(f["original_match_error"]) for f in d["folds"]) < 1e-6
        assert near(row["original_mean"], d["original_mean"])
        assert near(row["expanded_mean"], d["expanded_mean"])
        assert row["dimension"] == d["protocol"]["x_shape"][1]
        for val in (row["original_mean"], row["expanded_mean"]):
            assert f"{val:.5f}" in alpha_table, (row["model"], val)
        old.append(d["original_mean"])
        new.append(d["expanded_mean"])
    assert near(spearmanr(range(7), old).statistic, -0.8928571428571429)
    assert near(spearmanr(range(7), new).statistic, -0.4642857142857144)
    # 09-25 space-fix: these receipts are the unspaced-feature run; the TeX now prints them as the
    # "unspaced" comparison (table caption + prose) and the re-extracted values are pinned by
    # check_space_fix_20260926.py
    assert r"$-0.893$ on the unspaced features" in " ".join(TEX.split())
    assert "regularization-sensitive" in TEX
    plateau = load(ALPHA / "qwen25_6_plateau_full.json")
    assert len(plateau["folds"]) == 12
    fixed = [x["candidates"][str(1e14)] for x in plateau["folds"]]
    free = float(np.mean([x["scale_free_heldout_pearson"] for x in fixed]))
    guard = float(np.mean([x["original_epsilon_heldout_score"] for x in fixed]))
    assert near(free, a["seventy_two_b_limit_diagnostic"]["fixed_alpha_scale_free_score_mean_at_1e14"])
    assert near(guard, a["seventy_two_b_limit_diagnostic"]["fixed_alpha_original_epsilon_score_mean_at_1e14"])
    assert r"$0.0449093$" in TEX and r"$0.0450517$" in TEX
    assert r"$0.0456666$" in TEX and r"$0.0458094$" in TEX

    r = load(ROGUE / "summary.json")
    assert len(r["models"]) == 10 and len(r["rankings"]) == 4
    ranks = r["rankings"]
    assert [x["dose"] for x in ranks] == [0.0, -1.0, 0.3, 0.7]
    by_model = {x["model"]: x for x in r["per_model"]}
    for i in range(10):
        d = load(ROGUE / f"dose_model_{i:02d}.json")
        assert d["baseline_check"]["passed"] and d["model"] in by_model
        assert [x["dose"] for x in d["doses"]] == [0.0, -1.0, 0.3, 0.7]
        for actual, agg in zip(d["doses"], by_model[d["model"]]["scores"]):
            assert near(actual["rsa_mean"], agg["rsa_mean"])
            assert near(actual["encoding_mean"], agg["encoding_mean"])
            assert len(actual["rsa_per_session"]) == len(actual["encoding_per_session"]) == 12
    assert near(ranks[1]["encoding_rank_corr_vs_baseline"], 1.0)
    assert near(ranks[2]["encoding_rank_corr_vs_baseline"], 0.406060606060606)
    assert near(ranks[3]["encoding_rank_corr_vs_baseline"], 0.4424242424242424)
    assert r"falls" in TEX and r"to $0.406/0.442$" in TEX   # 09-25 de-numbering: moved from Section 5.3 to app:mechanism
    assert "does not identify" in TEX

    names = [
        ("Pythia-70M", "EleutherAI/pythia-70m"),
        ("Pythia-160M", "EleutherAI/pythia-160m"),
        ("Pythia-410M", "EleutherAI/pythia-410m"),
        ("GPT-2", "gpt2"), ("GPT-2-medium", "gpt2-medium"),
        ("BERT-base", "bert-base-uncased"), ("BERT-large", "bert-large-uncased"),
        ("RoBERTa-base", "roberta-base"), ("T5-base", "t5-base"),
        ("T5-large", "t5-large"),
    ]
    block = TEX[TEX.index(r"\label{tab:rogue-ranks}"):]
    block = block[:block.index(r"\end{table}")]
    for label, model in names:
        pairs = [f"{x['rsa_model_order_desc'].index(model)+1}/{x['encoding_model_order_desc'].index(model)+1}"
                 for x in ranks]
        pattern = re.escape(label) + r"\s*&\s*" + r"\s*&\s*".join(map(re.escape, pairs))
        assert re.search(pattern, block), (label, pairs)
    print("NEW follow-up receipt pin PASS: 7x12 alpha, 72B same-alpha limit, 10x4 rogue, TeX tables/scope")


if __name__ == "__main__":
    main()
