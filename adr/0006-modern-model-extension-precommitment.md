# 6. Pre-register the modern-model extension of the benchmark (Paper A)

Date: 2026-08-12
Status: Accepted

## Context

The strongest ICLR-specific objection to Paper A (raised independently by an internal reviewer
pass and by the external gpt-5.6-sol verdict) is that the model set is 2019-era: GPT-2, BERT,
RoBERTa, T5, Pythia ≤2.8B. Two published claims are exposed to it directly:

1. **The scaling verdict.** The scorecard contradicts "model size orders brain alignment" over
   70M–2.8B. A reviewer can reply that 2.8B is below the range where scaling laws bite.
2. **The readout reversal.** The −0.573 similarity-vs-encoding reversal on LibriBrain is computed
   over 32 checkpoints whose autoregressive side is dominated by small models; modern
   autoregressive models could in principle move it.

This ADR fixes the model set, the analysis, and the decision rules **before any feature of any
modern model has been extracted**. As of this commit, no Qwen or Llama checkpoint has ever been
run through this project's pipeline; the extraction sbatch does not yet exist.

## Model set (frozen)

One family carries the ladder, to avoid the family-vs-size confound the existing scorecard
already documents:

- **Qwen2.5 base ladder (7 points):** 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B — `Qwen/Qwen2.5-{S}` at
  HF revision `main` as of extraction date, recorded in the extraction report.
- **Cross-family anchors:** Llama-3.1-8B, Llama-3.1-70B (base).
- **Objective contrast (new axis, exploratory):** the instruction-tuned twins
  Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct, Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct.

13 checkpoints. **Fallback, fixed now:** if ≥70B extraction does not fit the available A100
allocation after two attempts (bf16, device_map=auto, layer-streamed), the ladder ends at 32B and
the paper states the cap; smaller models are never dropped to compensate.

## Protocol (unchanged, by construction)

Per-word features under sentence context, one layer at **relative depth 0.5** (round down) — the
same pre-registered rule as the existing 32 — extracted by the same `extract_llm_layers_multi.py`
path with its completeness checks (layer count, row count, sampled zero-rows). Cells, confound
battery, reliability ceilings, session-clustered statistics: identical to the published protocol.
No per-model layer search, no new cells, no re-tuning of anything downstream.

## Primary questions and decision rules (fixed now)

**Q1 — scaling, Qwen ladder, LibriBrain headline cell.**
Statistic: Spearman ρ(log parameters, alignment) per readout, session-clustered bootstrap CI;
plus the 72B-vs-0.5B paired contrast (the analogue of the published 2.8B-vs-70M test).
- *Supported over the modern range* iff ρ > 0 with CI excluding 0 in **both** readouts.
- *Contradicted over the modern range* iff CI excludes 0 in neither, or is negative in either.
- Scorecard update rule: supported → the scaling verdict becomes "contradicted below 3B,
  supported over 0.5B–72B" (refine); not supported → "not supported from 70M to 72B", stated with
  the CI. Either outcome is reported; neither is buried.

**Q2 — does the readout reversal survive modern models?**
The 9 base checkpoints join the concordance set (32 → 41). Statistic: similarity-rank vs
encoding-rank Spearman over 41, family-clustered bootstrap as published.
- *Reversal survives*: point estimate negative, family-clustered CI excluding 0.
- **Falsifier F1**: if the 41-model concordance is positive with CI excluding 0, the published
  reversal is declared an artifact of the 2019-era model set, in the abstract, not a footnote.
- Anything else: "the reversal attenuates with modern models", with the CI.

**Q3 — base vs instruction-tuned (exploratory, labelled as such).**
Per-session paired difference in alignment for the four twins, clustered CI. Reported as a
direction count over 4 pairs; no pooled p. This axis has no prior registration and will be
labelled exploratory regardless of outcome.

**Rankability precondition (F3):** any statistic above enters the paper only if the split-half
rank reliability of the relevant leaderboard, recomputed with the new models included, has a 90%
CI excluding zero. An indeterminate leaderboard is reported as indeterminate.

## What cannot rescue a failed primary

All-sensor variants, other cells, other layers, and per-subject re-selection are sensitivity
analyses. If Q1/Q2 fail their criteria at the headline cell, no variant result reverses the
stated verdict.
