# 9. Pre-register the length-matched horizon control (Paper B)

Date: 2026-08-17
Status: Accepted

## Context

The horizon dose-response (ADR 0007 Arm 2) increases the number of future words k visible to the
model — but increasing k also lengthens the input sequence. A skeptic can attribute the retention
rise to sequence length itself (position-embedding regime, attention dilution) rather than to
future CONTENT. This control separates the two before any reviewer asks. As of this commit the
control has never been run.

## Design

Identical to exp_horizon_dose.py in every respect except the appended context: at level k the
model sees words[:i+1] plus k words drawn (fixed seed, per target word) from the SAME story at
least two sentences away — matched length, wrong content. Levels k in {1, 2, 4, 8} plus the k=0
anchor recomputed in-run. Both configurations if budget allows; the sibling (decisive)
configuration is primary. Same subjects, trials, seed, readout, caps.

## Registered predictions and rules

- **Leakage (content) account:** retention under scrambled future stays at the k=0 level; the
  true-future curve of ADR 0007 rises. Statistic: at each k, the per-subject paired difference
  (true-future minus scrambled-future) Delta_k; supported iff the subject-clustered 95% CI of the
  mean paired difference excludes zero at k >= 2, and Page's trend over scrambled levels is
  non-significant (p >= 0.05).
- **Falsifier F1 (length account):** if the scrambled-future curve rises with Page p < 0.05 AND
  its k=8 retention reaches >= half the true-future k=8 retention, the dose-response is declared
  substantially a length artifact, the horizon evidence is demoted from Paper B's headline chain,
  and the paper says so in the abstract.
- Intermediate outcomes are reported in full with both curves in one figure.

## What cannot rescue a failed primary

No re-drawing of scramble sources, no per-subject selection, no alternative readouts. One run per
configuration; the seed is fixed in the script before submission.
