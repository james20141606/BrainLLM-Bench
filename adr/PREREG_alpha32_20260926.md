# PREREG — ridge-penalty range of the encoding readout, 32-checkpoint LibriBrain comparison

Written 2026-09-26 ~02:59 Beijing (commit b5a08e2f, 03:00:14), **before any expanded-grid or per-sensor-alpha score of the 32
checkpoints exists** (the only expanded-grid values in the repository are the seven Qwen2.5 rungs
of `alpha_sensitivity_20260922`, of which 0.5B and 1.5B are also in the 32; they are known and
are part of this sample). the cluster was unreachable when this was written. The commit time is the
evidence; deviations go under a dated "Deviations" heading and into the receipt.

Brief: `<coordination-notes>/final_review_20260924/STRENGTHEN_BRIEF.md`; reviewer items:
`<coordination-notes>/ai_review_sim/paperA_cc.md` W5 / R5, reviewer B weakness 2 and question 1, reviewer C weaknesses
3 and 9 / question 3 (per-sensor alpha).

## 1. Question

The encoding readout of the 32-checkpoint comparison uses `RidgeCV(alphas=logspace(1,6,8))`, one
alpha shared across sensors, chosen by GCV on the outer training fold. For Qwen2.5-0.5B the grid
top (1e6) is chosen in 11/12 folds (`alpha_probe.json`); the other 31 checkpoints' alphas were
never stored. **Does the RSA-vs-encoding reversal over the 32 checkpoints remain when the ridge
penalty is chosen, on training data only, from a grid whose top does not bind (1e1–1e14), and
when each sensor gets its own penalty?**

## 2. Estimands

Same 30,000 words, 12 leave-one-session-out folds, cell, depth (cached `llm_embeddings_fd` middle
file), feature standardization (train-fold mean/std + 1e-8), target (window-mean temporal_L
sensors, divided by its global std), and per-fold score (mean over sensors of the held-out
Pearson, the pipeline's scorer with its 1e-12 denominator term), exactly as
`p4_concordance32.encoding_per_session`. Per model and fold, three fits:

- `orig`: `RidgeCV(alphas=logspace(1,6,8))` — the published fit (gate G2).
- **`exp` (primary)**: `RidgeCV(alphas=EXPANDED)`, EXPANDED = logspace(1,6,8) ∪ logspace(7,14,8)
  (the grid of `alpha_sensitivity_20260922.py`), one alpha shared across sensors.
- `exp_pt` (secondary): same grid, `alpha_per_target=True` (one GCV alpha per sensor).

RSA is unaffected by alpha; RSA(m) = published `rsa_mean`.

**Primary estimand:** ρ_exp = Spearman(RSA, ENC_exp) over the 32 checkpoints.
**Comparator:** ρ_0 = −0.5733 (recomputed from the published per-session values in the same
aggregation).

**Secondary (reported, not decision-driving):**
- Δ = ρ_exp − ρ_0, paired family-clustered 95% CI.
- ρ_exp_pt with its family CI; Δ for it.
- Scale-free scorer (no 1e-12 term; zero-variance columns count zero) for `exp` and `exp_pt`:
  ρ recomputed; reported side by side (the 72B diagnostic showed the term matters at 1e14).
- Per checkpoint: selected alpha per fold under `orig` and `exp`; number of folds at the original
  top (1e6) under `orig`; number of folds at the expanded top (1e14) under `exp`; for `exp_pt`, the
  fraction of sensor-fold alphas at 1e14.
- Split-half reliability of ENC_exp (as in the headline, int session keys) and attenuation ceiling.
- Between-class share of |ρ_exp|.

## 3. Inference

`recompute_corrected_stats.family_clustered_ci` unchanged (12 families, 10,000 draws, seed 0).
One primary, no multiplicity correction.

## 4. Sample and compute

The 32 checkpoints of the headline receipt, cached features only (no extraction, no GPU). One
CPU array task per checkpoint; receipts (JSON, per-fold values, SHA-256 of the feature file) to
`$HOME/p4_strengthen_20260926/alpha32/`. Nothing written to /scratch.

## 5. Validity gates

- **G1** rows / metadata hashes as in the C4 prereg.
- **G2 replay identity**: every `orig` fold score must equal the published `per_session_enc`
  value to |Δ| ≤ 1e-6 (the 2026-09-22 replay achieved 2.2e-16). A model failing G2 is excluded
  and named; ≥ 3 failures → "INCOMPLETE", no ρ_exp reported.
- **G5**: all 12 folds present, finite scores.

## 6. Decision rule (mechanical, on ρ_exp and its family CI [L, U])

- **A. Reversal does not depend on the penalty range**: U < 0. Sub-label by the paired Δ CI
  (undiminished / weakened / strengthened, as in the C4 prereg).
- **B. Not determined on the expanded grid**: ρ_exp < 0 and L ≤ 0 ≤ U.
- **C. Reversal depends on the penalty range**: ρ_exp ≥ 0.
- **D. Instrument check (overrides B, C)**: split-half reliability of ENC_exp < 0.70 →
  "indeterminate".
- **Boundary flag (reported with any branch, does not change it)**: the names of checkpoints
  whose `exp` alpha is the grid top (1e14) in ≥ 6/12 folds. Their scores are then at the
  large-penalty plateau, not at an interior optimum; the text says so.

If the scorer choice (with vs without the 1e-12 term) gives a different branch, both branches are
reported and the text says the label depends on the scorer.

## 7. What the paper says (English, filled from the receipt)

Appendix L, after the sentence on the original grid (all branches): "On the original grid the top
penalty ($10^6$) is selected in [k] of the $384$ checkpoint-folds of the $32$-checkpoint
comparison ([n] of $32$ checkpoints in $\ge 6/12$ folds)."

- **A** — "Selecting the penalty from $10^1$--$10^{14}$ on training folds only leaves the readout
  concordance at ρ = [ρ_exp] (family-clustered 95% CI [L, U]; [sub-label], paired Δ CI [..]); with
  one penalty per sensor it is [ρ_exp_pt] ([..]). The reversal does not depend on where the ridge
  grid ends." Main text 0 lines (optional pointer in §5.3: "…and under an expanded, per-sensor
  ridge penalty (Appendix L)").
- **B** — "…gives ρ = [ρ_exp], CI [L, U] including zero: on the expanded grid the reversal is not
  determined." Limitations list gains the item.
- **C** — "…gives ρ = [ρ_exp] ([L, U]): the reversal depends on the ridge-penalty range." Main
  text §5.3 must carry this (hand to the writer / framing).
- **D** — "…the expanded-grid encoding ranking is too unreliable (split-half [r]) to be informative."
- **E (INCOMPLETE)** — the disclosure sentence of the original-grid alpha counts only.

The per-sensor arm and the scale-free scorer are reported with their values and CIs in the same
table, labelled secondary.
