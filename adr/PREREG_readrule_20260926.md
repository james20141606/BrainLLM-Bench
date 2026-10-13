# PREREG — common reading rule for the 14 masked / encoder–decoder checkpoints (LibriBrain, 32-checkpoint readout comparison)

Written 2026-09-26 ~02:58 Beijing (commit b5a08e2f, 03:00:14), **before any feature under the new rule has been extracted and
before any score under it exists.** the cluster was unreachable when this was written (VPN down), so
not even a smoke on real data has run. The commit time of this file is the evidence. After results
exist this file is not edited; deviations are appended below a dated "Deviations" heading and
recorded in the receipt.

Brief: `<coordination-notes>/final_review_20260924/STRENGTHEN_BRIEF.md`; reviewer items:
`<coordination-notes>/ai_review_sim/paperA_cc.md` W8 (undisclosed), reviewer C weakness 5 / question 2, reviewer B
weakness 1.

## 1. Question

In the headline comparison (receipt `results/p4_encoding/concordance32_CORRECTED.json`,
ρ = −0.5733, family-clustered 95% CI [−0.786, −0.264]; 56% of |ρ| is a between-class offset,
`concordance_decomposition_libribrain_CORRECTED.json`), objective class also fixes how a
checkpoint is read (`codes/extract_llm_layers_multi.py:53–55`, `_batch_words`, target rule):

- causal (18): the prefix ending in the target word, read at its last token;
- masked (9) and encoder–decoder (5, encoder stack only): the **whole sentence** (words after the
  target included), read as the **mean of the target word's sub-tokens**.

**If the 14 masked and encoder–decoder checkpoints are read under the causal rule — prefix
ending in the target word, last sub-token of the target word — does the RSA-vs-encoding reversal
over the 32 checkpoints remain?**

The causal rule is the only one all three classes can share: a causal model's state at the target
cannot see words after it, so "full sentence" is not available to the 18 causal checkpoints.

What this does **not** change (stated in the paper whatever the outcome): the tokenization rule
still differs by class on LibriBrain (causal: joined-string; masked/enc–dec: word-list; Appendix
D), and the depth rule, cell, words, sessions and scorers are unchanged.

## 2. Estimands

Same 30,000 words (≤ 2,500 per session, `RandomState(42)`, `p4_concordance32.subsample`), same 12
sessions, same cell (temporal_L, broadband, 300–400 ms), same 0.5 relative depth
(`int(0.5·top + 0.5)`, i.e. the cached `llm_embeddings_fd` middle file), same scorers:

- RSA(m) = mean over sessions of the C1–C3 partial Spearman (`p4_c4_lib.score_sessions`, which
  reproduced the headline per-session values to 6.6e-17 in C4 gate G2).
- ENC(m) = mean over the 12 leave-one-session-out folds of the ridge encoding correlation
  (`p4_concordance32.encoding_per_session`, original grid logspace(1,6,8), unchanged).

Reading rules for a masked / enc–dec checkpoint m (word-list tokenization of the documented
extractor, `_load`/`_forward` imported unchanged, batch 32, fp32, max_length 512):

| rule | input | target vector |
|---|---|---|
| `full_mean` (reference = cached) | whole sentence | mean of the target word's sub-tokens |
| **`prefix_last` (primary)** | words up to and including the target | last sub-token of the target word |
| `prefix_mean` (secondary) | words up to and including the target | mean of the target's sub-tokens |
| `full_last` (secondary) | whole sentence | last sub-token of the target word |

Target sub-tokens are located by `word_ids() == word_position` (special tokens such as [SEP] / </s>
have word id None and are never read). For the 18 causal checkpoints nothing is re-extracted:
their RSA and ENC per-session values are taken from `concordance32_CORRECTED.json` unchanged.

**Primary estimand:** ρ_PL = Spearman(RSA, ENC) over the 32 checkpoints, 14 masked/enc–dec under
`prefix_last`, 18 causal as published.

**Comparator:** ρ_0 = −0.5733, recomputed in the same aggregation from the published per-session
values.

**Secondary (reported, not decision-driving):**
- Δ = ρ_PL − ρ_0, paired family-clustered 95% interval (same family resample for both).
- Checkpoint Fisher 95% CI of ρ_PL.
- Split-half ranking reliability (Spearman–Brown, 200 splits, seed 0, int session keys) of RSA and
  ENC under `prefix_last`; attenuation ceiling.
- Between-class share of |ρ_PL| and within-class concordances (`concordance_decomposition.py`
  logic); class means of RSA and ENC before and after.
- ρ under `prefix_mean` and under `full_last` (splits the change into context direction vs token
  pooling), each with a family CI. Descriptive only.
- Per-model changes in RSA and ENC.

## 3. Inference

Family-clustered 95% CI = `recompute_corrected_stats.family_clustered_ci` unchanged (12 families
of `model_families.py`, 10,000 draws, seed 0, percentile). One primary, no multiplicity
correction. The small-cluster caveat of that interval (12 families, 4 masked, 2 enc–dec) applies
as it does to the headline and is restated in the paper text.

## 4. Sample, compute, storage

- 14 checkpoints: bert-base/large-uncased, roberta-base/large, deberta-v3-base/large,
  electra-base-discriminator, albert-base-v2, distilbert-base-uncased, t5-small/base/large,
  bart-base/large. No additions, no substitutions.
- 30,000 rows only; G1 hashes: rows `bfe5c58de6dc…`, metadata `b8e7784006b0…`.
- Weights to the job's `/dev/shm` (or the shared HF cache read offline); features in `/dev/shm`,
  deleted on exit; only JSON receipts to `$HOME/p4_strengthen_20260926/`. Nothing to /scratch.
- CPU partition (as C4 deviation 4 established admissible) with a GPU sbatch as fallback. The
  device is recorded in each receipt; all 14 receipts of the primary must come from one device
  class (no mixing), whichever delivers all 14 first.
- Neural patterns / orthographic RDMs: the C4 prep file `prep_inputs.npz`
  (SHA-256 `ea2dae19…` asserted), the exact arrays of the headline. The encoding target is derived
  from it exactly as `p4_concordance32.main` does (window mean, float64, global std scaling inside
  `encoding_per_session`).

## 5. Validity gates (checked before the primary is read)

- **G1 rows** as above, in every receipt.
- **G2 scorer identity (per model)**: the cached `llm_embeddings_fd` features are scored in the
  same job; every per-session RSA must match `per_session_rsa` of the headline to |Δ| ≤ 1e-8 and
  every per-session ENC must match `per_session_enc` to |Δ| ≤ 1e-6.
- **G3 extraction identity (per model)**: `full_mean` re-extracted rows vs cached rows: median row
  cosine ≥ 0.99999 **and** ≥ 99.9% of rows with cosine ≥ 0.9999 (the C4 G3 thresholds), and the
  kept layer's file name equals the cached middle file's.
- **G4 alignment (per model, per rule)**: target-word fallback ≤ 5%.
- **G5 completeness**: every feature matrix 30,000 × d, finite, no all-zero rows.

If 1–2 models fail, the primary is computed on the passing models plus the 18 causal, ρ_0 is
recomputed on the same subset, and the exclusions are named. If ≥ 3 fail: "INCOMPLETE", no
ρ_PL is reported as a result, and the paper keeps the disclosure-only wording (§7-E).

## 6. Decision rule (mechanical, on ρ_PL, its family CI [L, U], and the reliabilities)

- **A. Reversal holds under the common reading rule**: U < 0. Sub-label by the paired Δ CI:
  "undiminished" (Δ CI includes 0), "weakened" (Δ CI entirely > 0), "strengthened" (entirely < 0).
- **B. Not determined under the common reading rule**: ρ_PL < 0 and L ≤ 0 ≤ U.
- **C. Reversal depends on the reading rule**: ρ_PL ≥ 0 ("vanishes" if L ≤ 0 ≤ U; "becomes
  agreement" if L > 0).
- **D. Instrument check (overrides B and C)**: if the RSA or ENC split-half reliability under
  `prefix_last` is < 0.70, the outcome is "indeterminate: the common rule leaves too little
  reliable ranking variance".
- **E. INCOMPLETE** per §5.

The between-class share under `prefix_last` is reported next to 56% whatever the branch; it does
not change the branch.

## 7. What the paper says in each case (English, filled mechanically from the receipt)

Appendix D, after the provenance paragraph (all branches, disclosure first):
"Objective class also fixes how a checkpoint is read: causal checkpoints read the prefix ending in
the target word at its last token; masked and encoder--decoder checkpoints read the whole sentence,
including the words after the target, and average the target's sub-tokens (encoder stack only for
encoder--decoder models)."

Then one of:

- **A** — "Re-reading the 14 masked and encoder--decoder checkpoints under the causal rule
  (prefix ending in the target word, last sub-token; Table X) gives ρ = [ρ_PL], family-clustered
  95% CI [L, U] ([sub-label] relative to −0.573; paired Δ CI [..]). The reversal therefore does not
  depend on right context or sub-token pooling; the tokenization rule still differs by class."
  Main text (§5.3, 0 net lines): "…and it survives reading every checkpoint under the causal
  rule (ρ = [ρ_PL]; Appendix D)."
- **B** — "…gives ρ = [ρ_PL], family-clustered CI [L, U] including zero, so under a common reading
  rule the reversal is not determined." Main text: "…and under a common reading rule it is not
  determined (Appendix D)." Limitations list gains this item.
- **C** — "…gives ρ = [ρ_PL] (CI [L, U]): the reversal depends on how masked and encoder--decoder
  checkpoints are read, so it is not attributable to the training objective." Main text §5.3 and
  §6: the class account of the reversal is withdrawn in favour of "reading rule"; the abstract's
  reversal sentence needs the qualifier (hand to the framing writer).
- **D** — "…leaves too little reliable ranking variance (split-half [r]) to be informative."
- **E** — disclosure sentence only; "not tested" stays.

The secondary `prefix_mean` / `full_last` values go into the same table with the label
"descriptive"; no sentence attributes the change to context or pooling unless both secondary
family CIs are reported next to it.
