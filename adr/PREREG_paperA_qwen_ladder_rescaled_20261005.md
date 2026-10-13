# PREREG — Qwen2.5 scale-ladder encoding arm under per-session target scaling (paperA #60477, rebuttal evidence)

Written 2026-10-05 ~05:15 Beijing, **before any code for it exists and before any rescaled ladder score
exists**. The commit time is the evidence; deviations go under a dated heading at the end. Rebuttal evidence
only: `paperA_benchmark.tex` is not edited (PDF frozen, MEG `6f43d2ac`); paperB files untouched; 0 GPU.

## 0. Why, and what is already known

- `PREREG_paperA_rebuttal_checks_20261004.md` §1 (run 10-05, MEG `e20f1ef0`): the LibriBrain encoding target is
  scaled once across sessions; session 9 is 76.6× and session 3 5.5× the median session scale (branch U\*).
  Re-scored with per-session scaling, the 32-checkpoint concordance holds (−0.543 [−0.71, −0.22], A), but every
  published encoding score roughly halves relative to the rescaled one (32-model mean 0.046 published vs 0.102
  rescaled) and the 1e6 grid ceiling binds in 352/384 folds published vs 0/384 rescaled.
- §5.4 of the paper (tex `:493-497`) and App L/Table 7 rest the claim "encoding falls with size" on the same
  scorer: registered Q1 on the seven Qwen2.5 rungs (ADR 0006, `codes/adr0006_registered_analysis.q1_scaling`;
  receipt `space_fix_20260925/aggregate/adr0006_registered_verdicts_spacefix.json`): encoding ρ(size rank,
  mean score) = −0.964, session-clustered 95% CI [−1.00, −0.82]; 72B − 0.5B = −0.0274, Wilcoxon p = 0.00049;
  similarity ρ = +0.929 [+0.82, +0.96]. Registered Q1 verdict "NOT SUPPORTED" (rule: SUPPORTED iff both readouts
  have ρ > 0 and CI lower bound > 0).
- **Feature availability (checked 05:10, before this file):** the registered outcome uses re-extracted
  (spaced) features. For 0.5B and 1.5B these equal the cached `llm_embeddings_fd` files bit for bit (App D).
  For 3B, 7B, 14B, 32B, 72B the re-extracted features were deleted after scoring (`p4_spacefix_score.py
  --delete_after`; `~/p4_spacefix_20260925/feat/libribrain/` and `<HPC_ARCHIVE>/p4sf_park/` are empty). The
  only CPU-available features for those five rungs are the **unspaced** cached `llm_embeddings_fd` files
  (mtime 08-16), on which the paper itself quotes the ladder for comparison (encoding ρ −0.893; Table 7 caption
  rows "on the unspaced features"). Their published per-session scores are in
  `space_fix_20260925/score/libribrain__Qwen_Qwen2.5-{3B,7B,14B,32B,72B}__cached.json`.

## 1. Arms

- **R (registered features, rescaled) — primary estimand as asked; NOT RUNNABLE at 0 GPU.** Needs GPU
  re-extraction of five rungs (3B–72B) with the space-fix extractor. Registered here so that, if the author
  approves GPU, it runs under exactly the rules below with no new choices. Until then the paper's claim on the
  registered features is "untested under rescaling".
- **C (cached features, CPU, run now) — fixed-feature contrast.** All seven rungs from `llm_embeddings_fd`
  (0.5B/1.5B = registered features; 3B–72B = unspaced). Two scorers on the same features:
  - C0 unscaled: `p4_concordance32.encoding_per_session` preprocessing verbatim (gate G2);
  - Cz per-session scaled: Y_k ← Y_k / s_k per session before the leave-one-session-out loop, exactly the
    §1.6 follow-up of the 10-04 prereg (same code path, `p4a_session9_scale_20261005.enc_folds`).
  Arm C changes only the scorer, so C0 → Cz isolates the effect of the scale defect on the ladder. It does
  **not** certify the registered-feature value.

## 2. Estimands (both arms, both scorers)

Per rung: per-session encoding scores (12 sessions), mean; similarity unchanged (scale-free; registered RSA
per-session values from the adr0006 source receipt / headline receipt).
- T1 (registered trend statistic): ρ(size rank, mean encoding) over 7 rungs, session-clustered 95% CI exactly
  as `q1_scaling` (10,000 session resamples, `default_rng(6)`, percentile).
- T2: 72B − 0.5B per-session paired difference, mean and two-sided Wilcoxon p (as `q1_scaling`).
- T3: paired change ρ(Cz) − ρ(C0) under the same session resample (same rng), percentile 95% CI.
- Registered Q1 rule re-applied with the rescaled encoding arm and the published similarity arm.
- Descriptive: per-rung mean scores; folds at the 1e6 ceiling; session-9 scores.

## 3. Gates

- G1 subsample rows `bfe5c58de6dc…`, prep `ea2dae19…` (as §1 of the 10-04 prereg).
- G2 per rung: C0 per-session scores reproduce the published ones to |Δ| ≤ 1e-6 (0.5B/1.5B: headline receipt
  `concordance32_CORRECTED.json`; 3B–72B: the `__cached` score receipts). A rung failing G2 is excluded and
  named; ≥ 2 exclusions → INCOMPLETE (7 rungs is already few).
- G3 feature identity: the scored file's SHA-256 equals the `feature_sha256` in the `__cached` receipt
  (3B–72B).

## 4. Decision rule (on arm Cz, T1 and T2; arm R identical if ever run)

- **holds**: ρ_z < 0 and CI upper < 0, and 72B − 0.5B < 0 with Wilcoxon p < 0.05.
- **weakens**: ρ_z < 0 but the CI includes 0, or the 72B − 0.5B contrast is not significant (p ≥ 0.05) while
  still negative.
- **vanishes**: ρ_z ≥ 0 with CI including 0, and 72B − 0.5B not significant.
- **reverses**: ρ_z > 0 with CI lower > 0, or 72B − 0.5B > 0 with p < 0.05.
(Checked in this order: reverses, holds, vanishes, weakens.) The same rule is reported for C0 so the reader
sees the change. If the registered Q1 rule turns SUPPORTED under Cz, that is reported as its own line.

## 5. Wording (kit; revision only after arm R or an author decision)

All branches first sentence (the disclosure the coordinator asked for, in the same place): "The encoding
target in the LibriBrain analyses was scaled once across sessions, and two sessions differ in scale (76.6× and
5.5× the median); scaling each session separately roughly doubles every encoding score (32-checkpoint mean
0.046 → 0.102) and removes the ridge-grid ceiling (352/384 → 0/384 folds)."
- **holds**: "On the Qwen2.5 ladder (cached features), encoding still falls with size under per-session
  scaling (ρ = [ ], session-clustered CI [ ]; 72B − 0.5B = [ ], p = [ ]). The re-extracted features of the
  3B–72B rungs were not retained, so the registered-feature value is not recomputed."
- **weakens**: "…the encoding size trend weakens (ρ = [ ], CI [ ]); 'encoding falls with size' is not
  established under per-session scaling." The kit stops defending the encoding slope.
- **vanishes**: "…encoding no longer orders the ladder (ρ = [ ], CI [ ]); the readout-dependent ordering of §5.4
  rests on the scale defect." Proactive-disclosure item; revision rewrites §5.4's second sentence.
- **reverses**: "…encoding rises with size (ρ = [ ], CI [ ])"; proactive disclosure; §5.4 and the Antonello
  row of Table 8 change; if Q1 flips to SUPPORTED, the scorecard verdict changes too (author).

## 6. Compute

<cpu-partition>, one rung per task, 8 CPU, 48 GB (72B: X is 30,000 × 8,192 float64 ≈ 2 GB), ≤ 3 h; receipts to
`$HOME/p4_rebuttal_20261005/qwen_ladder/`, then `results/paperA_rebuttal_20261004/qwen_ladder_raw/`. 0 GPU.
