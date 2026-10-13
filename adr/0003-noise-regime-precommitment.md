# 3. Pre-register the noise-regime account of readout concordance

Date: 2026-08-03
Status: Accepted

## Context

Every headline this project has produced is negative, withdrawn, or cohort-specific. This ADR
pre-registers the one account that would make a *positive* claim out of the same data, and fixes in
advance what would refute it.

The two facts to be explained:

| cohort | recording | concordance ρ(RSA rank, encoding rank) over 32 models |
|---|---|---|
| LibriBrain | 1 listener, many sessions, Neuromag-306 MEG | **−0.68** (reversal) |
| Brennan | 47 listeners, shallow each, EEG | **+0.34** (agreement) |

The field's default reading of that table is that the reversal failed to replicate. The account
registered here is the opposite: **the two cohorts differ in measurement reliability, not in
biology, and concordance is a function of reliability.** Agreement in the shallow cohort would then
be uninformative agreement — at low SNR both readouts can only see the same coarse, dominant
stimulus-driven structure, so they cannot disagree. Disagreement appears where the measurement is
good enough to resolve the finer structure the two readouts weight differently.

If that is right, it inverts the policy conclusion. A benchmark built on many shallow subjects
cannot detect readout-contingency *and will falsely reassure*, because it sits below the resolution
at which the readouts can be told apart.

This is registered while the LibriBrain noise sweep does not exist in any form, and while the
n = 27 MEG-MASC concordance has not been computed — jobs 15245837 and 15245838 were submitted
today and are both `PENDING` with zero output bytes as of this commit.

## Hypothesis

**H.** Readout concordance `C` is a monotone decreasing function of neural pattern reliability `R`,
and cohort identity (hardware, listener count, stimulus language) adds nothing once `R` is known.

`R` is defined before any sweep: the split-half reliability of the neural RDM within a unit,
Spearman-Brown corrected — the same estimator already used for ranking reliability, applied one
level down.

**Amendment, 2026-08-03, made before the sweep was run and before any dose produced a number.**
The first wording said to split the unit's *trials*. That is not implementable in this pipeline:
the RDM is built over trials, so two disjoint trial halves give RDMs over different pairs, which
cannot be correlated. `R` is therefore measured by splitting the *channels* into random halves,
building the RDM in each half, and correlating them. This measures the same quantity — how
reliably the pattern geometry is estimated — and is degraded by exactly the independent sensor
noise the sweep injects, while shared neural signal leaves it intact. Recorded here rather than
silently substituted, because the choice was made with the outcome still unknown.

## The decisive experiment: a noise dose on a fixed cohort

Cross-cohort comparison confounds reliability with everything else. So the primary test is an
intervention *within* LibriBrain, where nothing but reliability changes.

Add independent Gaussian sensor noise to the LibriBrain epochs at a dose sweep chosen to walk the
measured `R` down from its native value through Brennan's measured `R`, at a minimum of six doses
including zero. At every dose, recompute on the identical word subsample and the identical 32
frozen checkpoints:

- `R`, the neural RDM reliability;
- the RSA ranking and the encoding ranking;
- `C`, their Spearman concordance;
- **the split-half ranking reliability of each readout separately, with its 90% CI.**

The noise is added to the neural side only, before any band-split or averaging, so both readouts
receive exactly the same degraded data. Seeds fixed; the sweep is run at three seeds and the dose
curve is reported with across-seed spread.

## What counts as confirmation

All three, none optional:

1. `C` is monotone in `R` across the dose sweep, with the across-seed band excluding a flat line.
2. `C` **crosses into positive territory** — not merely decays to zero — by the dose at which `R`
   reaches Brennan's measured `R`, and the value there is within the Brennan CI.
3. At every dose used to support 1 and 2, **both** readouts' split-half ranking reliability CIs
   exclude zero.

## Falsifiers, fixed now

- **F1.** `C` is flat in `R`. H is dead; cohort identity, not reliability, is doing the work, and
  the non-replication stands as a non-replication.
- **F2.** `C` rises toward 0 but does not become positive. Then noise decorrelates the two rankings
  without reproducing Brennan's *agreement*, and the interesting version of H is dead. This is
  reported as such. Attenuation toward zero is the trivial prediction of adding noise to anything
  and is not evidence for H.
- **F3.** `C` moves only at doses where at least one readout's ranking reliability CI includes
  zero. Then the movement is the two rankings becoming individually meaningless, not a law relating
  two meaningful rankings. **This is the most likely failure mode and it is fatal, not a caveat.**
  The claim requires the crossover to occur while both rankings remain internally reliable.
- **F4.** The three real cohorts do not fall near the LibriBrain dose curve at their own measured
  `R` (see below).

## The blind cross-cohort prediction

Registered before job 15245838 reports, and this is the whole point of committing now:

> MEG-MASC (n = 27, moderate data per subject) has reliability between LibriBrain's and Brennan's.
> Under H its concordance must therefore lie **strictly between +0.34 and −0.68**, and must be
> **more negative than Brennan's +0.34**.

The sharp version, also registered: rank the three cohorts by measured `R` and by `C`; under H the
two orderings are identical. With three points that is a 1-in-6 result under the null, which is weak
on its own — it is a consistency check, not proof, and is reported as such. The dose sweep is the
evidence; the cohorts are the calibration.

If MEG-MASC comes back *outside* [−0.68, +0.34], or on the wrong side of Brennan, H is refuted
before the sweep is even run, and this ADR records that it was refuted rather than adjusted.

## The constructive arm

H, if true, names a defect: cosine/correlation distance RDMs are not cross-validated, so their
entries are biased by noise in a way that depends on the pattern geometry, while leave-one-group-out
ridge encoding is cross-validated and its score is not. That predicts a *fix*, and the fix is the
paper's positive deliverable rather than a warning.

Run the entire concordance analysis a second time with the RSA arm replaced by a cross-validated
distance estimator (crossnobis / cross-validated Mahalanobis, the standard unbiased RDM estimator),
holding everything else identical.

- If crossnobis-RSA agrees with encoding **at every dose and in every cohort**, the deliverable is
  constructive and specific: the reversal is an artifact of a biased distance estimator, and the
  field should use cross-validated distances. That is a positive, actionable result.
- If crossnobis-RSA still reverses at high `R`, the disagreement is not an estimator artifact and is
  a real property of what the two readouts measure. Also positive, and a stronger scientific claim,
  but with no fix attached.
- Either outcome is publishable *as registered*. Choosing between them after seeing the result is
  the failure this ADR exists to prevent.

**Amendment, 2026-08-03, before the arm was built.** Checking the repeat structure the estimator
needs turned up a limitation that changes what this arm can conclude, so it is recorded here rather
than in the eventual write-up.

Crossnobis needs two independent measurements of the same condition. The condition cannot be the
word *type*: averaging a type's tokens averages their contextual embeddings too, and a contextual
model whose tokens have been averaged within type is a static embedding, which would collapse the
32-model comparison into a comparison of vocabularies. The condition must therefore be the stimulus
*token*, measured independently more than once.

| cohort | stimulus tokens | independent measurements per token |
|---|---|---|
| MEG-MASC n=27 | 8 224 | 27 (one per subject) |
| Brennan | 2 129 | 47 (one per subject) |
| **LibriBrain** | 177 127 | **1** |

LibriBrain is one subject hearing each word once; its metadata carries no key under which a stimulus
token recurs. So the constructive arm can be built on two cohorts and **not on the cohort the
reversal lives in**. Registered consequences:

- The arm is run on MEG-MASC and Brennan only, and cannot by itself explain the LibriBrain result.
- LibriBrain's RDM is a *single-trial* RDM and, unlike the others, cannot be made anything else.
  This project has previously recorded that single-trial RDMs carry close to zero split-half
  reliability. Whether that applies here is measured directly by the dose-0 run (15246181, task 0),
  and that number is now the one this ADR turns on.
- If the cross-validated RDM makes the two readouts agree on MEG-MASC and Brennan, the honest
  reading is that the reversal is an artifact of single-trial distance estimation surviving only in
  the one cohort that cannot support a better estimate. That is the *demise* of the headline, not
  its rescue, and it is registered here as an acceptable and publishable outcome so that it cannot
  later be reframed as anything else.

## No tuning

No best-dose, best-cell, best-layer, or best-region selection. The cell is the LibriBrain-matched
cell already fixed by ADR 0001. Region and window variants are sensitivity analyses reported
alongside, and cannot rescue a primary that fails.

## Consequences

The prior plan's framing — "brain–LLM leaderboards are measurement-contingent, here is a warning" —
is demoted to a corollary. If H survives, the claim becomes a quantitative one with a threshold and
a predicted crossover, and the cohorts that disagree with us become the evidence for it rather than
the objection to it. If H is refuted under F1–F4, the reversal is reported as LibriBrain-specific,
the portfolio's ceiling is a methods/benchmark venue, and this ADR is the record that we looked.
