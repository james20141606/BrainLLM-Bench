# 2. Pre-register the Armeni readout-concordance replication

Date: 2026-08-02
Status: Accepted — **deprioritised 2026-08-03**, see below

> **Status note, 2026-08-03.** This ADR treats Armeni as decisive because it is the only second deep
> MEG listener in existence. That reasoning belonged to the account in which the reversal was a
> property of the readouts. Under [ADR 0004](0004-repeat-removal-precommitment.md) what governs
> readout agreement is the number of independent measurements per stimulus token, and there Armeni
> offers **3** (one per participant) against MEG-MASC's 27 and Brennan's 47. Armeni becomes one
> supplementary point on a curve rather than the experiment that settles anything.
>
> The pre-registration below remains in force and unmodified — it was committed before the data
> existed and is not being rewritten after the fact. What has changed is its priority, not its
> content. The download is also still blocked: the endpoint truncates deterministically at
> ~6.408 GB (two attempts 893 bytes apart), which is 2.8% of 229.9 GiB.
>
> Recorded because it may matter later: Armeni's participants wore MEG head casts, so *within* a
> participant the sensor space is stable across all ten sessions. That is exactly the
> spatial-correspondence condition ADR 0004's precondition P1 says is missing for cross-subject
> averaging. It does not rescue P1 — Armeni's token repeats would still be cross-participant — but
> if P1 fails and a source-space or hyperalignment pre-registration becomes necessary, this is the
> dataset that could support it.

## Context

Paper A's lead result is that two conventionally applied readouts — raw-feature cosine RSA and
feature-standardized ridge encoding — rank 32 language models in opposite orders on LibriBrain
(ρ = −0.68), each ranking internally reliable. Its two standing weaknesses are that LibriBrain is
one listener, and that the reversal does not reproduce on the shallow-per-subject cohorts.

We have shown that recording depth is not the explanation: scored one session at a time, 11 of 12
LibriBrain sessions reverse independently (p = 0.001), and at single-session depth the concordance
is still −0.47 rather than moving toward Brennan's +0.34. So what the result needs is not more
data per listener but a *second deep listener*, and Armeni is the only such MEG cohort that exists.

Two facts constrain what it can settle, and both must be stated before any number is computed.

**It is not a listener-only replication.** Armeni is CTF/VSM 275-channel at the Donders Centre,
against LibriBrain's Neuromag-306; participants are Dutch listening to English audiobooks. Listener,
hardware and listener-native language change together. It is a cross-hardware, cross-nativeness
stress test, and a failure to replicate cannot be attributed to listener identity alone.

**The frozen model panel is mostly English-trained.** That is deliberate — swapping the panel would
break comparability with LibriBrain — but it makes this a stress test rather than a language-matched
replication. A Dutch or multilingual panel may be reported separately and cannot replace the frozen
arm.

This ADR is written and committed while the Armeni download is incomplete and no Armeni epoch,
alignment value or concordance has been computed. It repeats the pattern of ADR 0001, which is the
strongest methodological asset either paper has.

## Decision

### Channel gate — first, and non-negotiable

Select CTF primary MEG sensors only, via `codes/region_atlas_ctf.py`. Exclude reference channels,
EOG, ECG, stimulus, audio, DC and miscellaneous. Before any analysis runs, record channel names,
counts by type, and **variance share by channel type**, and abort if any non-MEG channel enters the
neural patterns. `pick("all")` is forbidden.

This is not boilerplate. Our ECoG analysis used `pick("all")` on an EDF that typed all 124 channels
as `eeg` and marked none bad; 14 of them were EKG and DC analog channels holding **99.97%** of total
variance, and the resulting "finding" was the stimulus audio track. The gate exists so that failure
cannot recur silently.

### Fixed cell and preprocessing

Frozen from the existing benchmark, with no cohort-specific tuning:

- epochs −200 to +800 ms, baseline −200 to 0 ms, resampled to 250 Hz
- broadband 1–80 Hz
- primary cell: CTF temporal-left sensors, 275–425 ms
- primary layer: relative depth 0.5
- at most 2500 words per session, seed 42, identical word set for both readouts

No best-cell, best-layer or per-listener selection. All-sensor and depth 0.25/0.75 variants are
sensitivity analyses and cannot rescue a failed primary.

### Models

The same 32 frozen checkpoints and family labels as LibriBrain, at the same repository revisions.
Models may not be added or replaced after observing any Armeni result. Report tokenization coverage
and unknown-token rates.

### Readouts

1. **Conventional RSA** — raw-feature cosine RDM, confound-controlled partial Spearman.
2. **Conventional encoding** — feature-standardized ridge, leave-one-session-out, held-out Pearson.
3. **Symmetric control (mandatory)** — RSA after per-dimension feature standardization; encoding
   unchanged, as it already standardizes.

Arm 3 is the point of running this cohort. At 32 models the family-clustered interval for the
standardized case spans [−0.72, +0.30] and cannot adjudicate. If the conventional arm replicates
and the symmetric arm collapses, that is not a failure — it is the finding, and it belongs in the
abstract rather than an appendix.

### Rankability precondition

A listener is analysable only if **both** readouts show positive split-half ranking reliability with
a 90% CI excluding zero. Otherwise that listener is *indeterminate*, not a failed replication.

### Primary estimands

Per listener, across the 32 checkpoints:

- Spearman concordance ρ between the two readouts;
- a family-clustered 95% CI, resampling whole model families;
- the architecture gap `G = mean(RSA rank − enc rank | causal) − mean(· | non-causal)`, positive
  meaning causal models rank worse under RSA and better under encoding, with a family-level
  permutation test.

The naive 32-checkpoint interval is descriptive only and cannot adjudicate anything.

### Decision rule, fixed now

- **Replicates** — every analysable listener is rankable, all have ρ < 0, the family-clustered
  upper CI is below zero in at least two, and every architecture gap carries the LibriBrain sign.
  The paper may then say the conventional disagreement held across a second deeply sampled listener
  on different hardware; it may not say the effect is a population law.
- **Geometry-independent** — the same criterion also passes under the symmetric control. Only this
  outcome licenses any claim that the readouts disagree beyond feature-standardization geometry.
- **Convention-dependent** — the conventional arm passes and the symmetric arm does not. Report as
  the headline: accepted pipelines disagree because their preprocessing contracts differ.
- **Generalization killed** — both readouts reliable, but at least two listeners exclude a
  substantively negative concordance, defined now as ρ ≤ −0.30, and no architecture gap replicates.
  The LibriBrain existence proof survives; any cross-listener claim does not.
- **Indeterminate** — unreliable rankings, or mixed signs without an equivalence result. Reported
  as indeterminate, never reframed as either replication or falsification.

### If fewer than three listeners arrive

The download has so far yielded one participant of three. One deep listener is still the control
LibriBrain lacks and the analysis proceeds, reported as n = 1 with the per-listener criteria above.
It cannot support any claim about consistency *across* listeners, and the decision rule's
"at least two" clauses are then unreachable — in that case the only available verdicts are
*indeterminate* or *generalization killed*.

## Consequences

We accept that a cross-hardware, cross-nativeness cohort is a harder test than a listener-only
replication, and that a failure will be ambiguous between those causes. We prefer that to having no
second deep listener at all, and the ambiguity is stated in advance rather than discovered in
review.

We also accept that arm 3 may collapse. That outcome is pre-declared as a headline result rather
than a limitation, which is the whole reason for fixing it here: after seeing it, the temptation
would be to report it as a robustness check in an appendix.
