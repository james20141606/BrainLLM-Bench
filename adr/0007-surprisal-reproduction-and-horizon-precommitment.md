# 7. Pre-register the surprisal-readout reproduction and the future-context horizon test (Paper B)

Date: 2026-08-12
Status: Accepted

## Context

Paper B's two named ceilings, per its own text and the external ICLR verdict:

1. It reads out the boundary effect as **representational alignment with BERT**, not as the
   original's **autoregressive surprisal coupling**, so it "cannot adjudicate the motivating
   result directly" (codex verdict, 2026-08-12).
2. Its causal-truncation verdict is **split**: decisive under the sibling configuration
   (retention 0.30, paired CI excluding zero), non-decisive under the headline configuration
   (paired CI crosses zero).

Two experiments address these, and both are cheap enough to run before the ICLR deadline. This
ADR fixes their analyses and decision rules **before either has produced a number**. As of this
commit no surprisal value has ever been computed in this project, and no truncation with
k > 0 future words has ever been run.

## Arm 1 — the boundary effect under its own readout

**Operationalization** (as close to the original as its public methods allow, on our cohorts):
per-word surprisal −log P(w_t | w_<t) from GPT-2 (the family the original's English arm uses),
full left context within the sentence — causal by construction, so no truncation arm is needed
or possible here. Coupling: per subject, the correlation between word-locked neural response
pattern dissimilarity and surprisal-difference dissimilarity is replaced by the simpler and
closer-to-original quantity — the per-word regression of response amplitude at the published
cell (Brennan: all sensors, 400±75 ms; MEG-MASC: Brennan-matched cell) on surprisal, fit
separately for sentence-onset and sentence-medial words, equal counts per subject as in the
published pipeline. Contrast: Δ_coup = coupling_medial − coupling_onset per subject.

**Cohorts:** the 47-subject Brennan rebuild and all 27 MEG-MASC subjects. Fixed seeds, same
subsampling rule as the published per-subject pipeline.

**Decision rules:**
- *Replicates under the original readout*: Δ_coup > 0 with two-sided signed-rank p < 0.05
  (Holm over the two cohorts) **and** ≥ 60% of subjects positive, in at least Brennan.
- *Fails to generalize under its own readout*: the Brennan non-inferiority test excludes a
  boundary-contrast effect of the size the alignment readout found (the same bound logic as
  §3.4 of the paper, bound fixed at the alignment-readout Δ), with p < 0.05.
- Otherwise indeterminate, reported as such.

**Interpretation grid, committed now:**
- Replicates → the original effect is real under its own readout in our data; Paper B's leakage
  claim is thereby **scoped to representational-alignment readouts** and the paper must say the
  boundary effect survives a causal readout — in the abstract, not a footnote.
- Fails → the boundary effect does not generalize to these cohorts even under its own causal
  readout; the leakage finding then explains why the *alignment* version looked positive, and
  the non-replication becomes readout-independent.
- Both outcomes strengthen the paper only if reported in full; this grid is the commitment to
  do so.

## Arm 2 — future-context horizon dose–response

**Design:** the truncation test of §3.5, generalized: the model sees the sentence up to the
current word **plus k future words**, k ∈ {0, 1, 2, 4, 8, full}. k=0 is the published prefix
arm; k=full is the published full arm. Run on **both** configurations (sibling and headline),
reported side by side, 47 subjects, word-level alignment with the 0-lost-words check.

**Prediction from the leakage account, fixed now:** per-subject retention rises monotonically
in k.

**Statistic:** mean retention at each k with subject-clustered bootstrap CI; monotonicity by
Page's trend test across the six ordered levels, α = 0.05, plus Spearman(k, retention) with
subject-clustered CI as the effect size.

**Decision rules:**
- *Graded leakage supported*: Page p < 0.05 and Spearman CI > 0 in the sibling configuration.
- **Falsifier F1**: flat or non-monotone profile (Page p ≥ 0.05 and Spearman CI spanning 0) in
  the sibling configuration refutes the graded-lookahead account; the paper must then say the
  removed information is not graded future context, and the leakage claim weakens accordingly.
- A step only at k=full with a flat 0–8 profile is a pre-named alternative reading (sentence-
  final/global-structure information, not graded lookahead) and will be reported as such, not
  reframed as support.
- The headline configuration is reported descriptively beside the sibling one either way; its
  published non-decisive verdict cannot be overturned by this arm alone.

## What cannot rescue a failed primary

No new cells, layers, medial definitions, or item caps. If Arm 1 is indeterminate and Arm 2
fires F1, the paper keeps its current split-verdict framing and adds both null results.
