# MEG-LLM-Bench Leaderboard

> These are 1,000-trial harness reference values from 2026-05; they are not the paper's per-cohort leaderboard (Appendix Table 4, `results/cross_cohort_summary.tsv`) and predate later corrections.

Last updated: 2026-05-10

Leaderboard tracks reference baselines and submitted methods. Each entry is the per-task score on each cohort. Headline score = `cohort_consistency` = mean across cohorts of per-task score.

## Tasks

### Task 1: best-layer (Spearman ρ vs gold-standard scan)

Gold-standard reference values from `bench/run.py` (n=1000 trials, seed=42):

| Cohort | GPT-2 / German-GPT-2 | BERT / German-BERT | GPT-2-medium / Mult-BERT |
|--------|----|----|----|
| LibriBrain | GPT-2 L10 (z=+8.4×10⁻³) | BERT L0 (z=+10.3×10⁻³) | GPT-2-medium L22 (z=+7.8×10⁻³) |
| MEG-MASC   | GPT-2 L0 (z=+3.3×10⁻³) | BERT L2 (z=+2.8×10⁻³) | GPT-2-medium L0 (z=+2.4×10⁻³) |
| Brennan    | GPT-2 L11 (z=+3.1×10⁻³) | BERT L10 (null, z=−2×10⁻⁴) | GPT-2-medium L23 (z=+2.6×10⁻³) |
| MEG-SCANS  | German-GPT-2 L11 (z=+4.4×10⁻³) | German-BERT L1 (z=+3.2×10⁻³) | Mult-BERT L0 (z=+3.3×10⁻³) |
| ECoG (median over 5 patients) | GPT-2 L11 (z=+12.4×10⁻³) | BERT L10 (z=+25.0×10⁻³) | GPT-2-medium L12 (z=+6.9×10⁻³) |

| Method | LibriBrain | MEG-MASC | Brennan | MEG-SCANS | ECoG | mean |
|--------|-----------|----------|---------|-----------|------|------|
| **streaming-RSA scan (gold)** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| linear-probe (ridge) | tbd | tbd | tbd | tbd | tbd | tbd |
| mid-LLM (n_layers/2) | tbd | tbd | tbd | tbd | tbd | tbd |
| random | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| your method | — | — | — | — | — | — |

### Task 2: boundary-conditional (sign-consistency: medial > onset)

| Method | LibriBrain | MEG-MASC | Brennan | MEG-SCANS | ECoG | mean |
|--------|-----------|----------|---------|-----------|------|------|
| **paper4 G4 (full layer search)** | 3/3 | 0/3 | 2/3 | 2/3 | n/a | 7/12 |
| paper4 G7 (CLIT × boundary) | 3/3 | 2/3 | 3/3 | 0/3 | n/a | 8/12 |
| random-tercile permutation | ≈50% | ≈50% | ≈50% | ≈50% | n/a | ≈50% |

### Task 3: surprisal-tercile (weighted-mean active layer drift)

| Method | LibriBrain | MEG-MASC | Brennan | MEG-SCANS source | mean |
|--------|-----------|----------|---------|------------------|------|
| **paper4 G3** | +5 layer | +1 layer | +2 layer | 0 (flat) | +2 |
| paper4 G-SRM (source) | tbd | tbd | tbd | rescued: +5 z | tbd |
| unstratified | 0 | 0 | 0 | 0 | 0 |

### Task 4: SAE-feature-transfer (top-50 ∩ LB anchor)

| Method | LB | MEG-MASC | Brennan | MEG-SCANS | ECoG |
|--------|----|----|---------|-----------|------|
| **paper4 CITE-SAE** | 50/50 (self) | 2/50 | 11/50 | 0/50 | 10/50 |
| random features | ≈0.2 | ≈0.2 | ≈0.2 | ≈0.2 | ≈0.2 |

### Task 5: layer-saturation (PFLL L_sat depth %)

| Method | LB BERT | MS Mult-BERT | MS German-BERT | mean |
|--------|---------|--------------|---------------|------|
| **paper4 PFLL** | 11% | 33% | 0% | 15% |
| paper4 G-SRM-PFLL (source-aligned) | n/a | 8% | 0% | 4% |
| linear-probe | tbd | tbd | tbd | tbd |

## Submissions

Submit by:
1. Compute your method's per-cell predictions in the format specified by `tasks.yaml`.
2. Save as TSV with required columns.
3. Run `python -m bench.eval.run_eval submission.tsv --task <name>`.
4. Submit results through the benchmark's public issue tracker (URL released with the public repository at camera-ready) with the leaderboard entry filled in.
