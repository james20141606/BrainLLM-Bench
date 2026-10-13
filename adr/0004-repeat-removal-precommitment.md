# 4. Pre-register the repeat-removal test: can the reversal be induced?

Date: 2026-08-03
Status: Accepted — supersedes ADR 0003 as the primary experiment

## Context

ADR 0003 registered a noise-injection sweep to test whether readout concordance is governed by
measurement reliability. Before it ran, two things changed the right design.

**First**, the cohorts' repeat structure turned out to differ in kind, not degree:

| cohort | stimulus tokens | independent measurements per token | concordance |
|---|---|---|---|
| MEG-MASC n=27 | 8 224 | 27 | landing today |
| Brennan | 2 129 | 47 | **+0.34** (agreement) |
| LibriBrain | 177 127 | **1** | **−0.68** (reversal) |

LibriBrain is one listener hearing each word once. Its RDM is a single-trial RDM and no amount of
care can make it anything else. The two cohorts where the readouts *agree* are exactly the two where
a repeat-aware, cross-validated RDM is possible.

**Second**, the noise sweep would only ever have supported a post-hoc diagnosis: *the reversal
appears in the dataset that happens to lack repeats*. That is an observation about a correlation
across three points. The claim worth making is causal, and it is testable in the opposite direction:
start where repeats exist and the readouts agree, **remove the repeats**, and see whether the
reversal appears.

If it can be induced, we have identified a failure mode in neural model benchmarking. If it cannot,
the LibriBrain reversal is an unexplained cohort-specific anomaly and we say so.

## Hypothesis

**H.** The RSA/encoding reversal is produced by estimating dissimilarities from single observations.
Given repeats, a cross-validated RDM yields readout agreement; as repeats are removed the RSA arm
acquires a noise-dependent bias, and concordance degrades and then reverses.

## Design

Cohorts: **MEG-MASC (27 subjects) and Brennan (47 subjects)** — the two with repeats. LibriBrain
cannot enter this experiment and is the regime being simulated, not a condition in it.

**Dose** = `k`, the number of independent subject-measurements averaged per stimulus token, swept
over `k ∈ {1, 2, 4, 8, 16, all}`. Subjects are drawn without replacement per token, seeded, three
seeds. `k = 1` is the LibriBrain regime by construction.

**Symmetry requirement, non-negotiable.** Both readouts consume the *same* `k`-averaged token
patterns. The manipulation is expected to hit RSA harder than encoding — that is the hypothesis —
but it must not be applied to only one arm, or the comparison measures the manipulation rather than
the readouts.

At `k ≥ 2` the RDM is additionally computed in a cross-validated form (disjoint subject folds
supplying the two sides of the distance estimate), which is unbiased under noise. At `k = 1` no
cross-validation is possible; that asymmetry is the point, and both the cross-validated and the
plain estimator are reported at every `k ≥ 2` so the estimator's contribution is separable from the
averaging's.

At every `(k, seed, cohort)` record: RDM reliability `R`, the RSA ranking, the encoding ranking,
concordance `C`, and **each readout's split-half ranking reliability with its 90% CI**.

Word tokens are held fixed across `k` — the same 8 224 / 2 129 tokens at every dose — so nothing
changes but how many measurements each one is estimated from.

## Precondition, added 2026-08-03 before any job ran — the design is void without it

Averaging `k` subjects per token is **not** the same as averaging `k` within-subject repeats.
MEG sensor spaces do not correspond across people: head position differs, anatomy differs, and no
inverse solution is applied here. This project's own encoding diagnostic already said so. Subjects
are not interchangeable draws from one measurement process, and averaging across them in sensor
space could destroy signal rather than average noise.

That is checkable rather than arguable, and cheaply, because the sweep already measures RDM
reliability `R` at every `k`:

> **P1.** Measured `R` must increase monotonically with `k` in both cohorts. If it does not,
> cross-subject averaging is not buying independent measurements in this sensor space, every
> downstream number is uninterpretable, and the experiment is reported as void — not adjusted, not
> rescued with a spatial-alignment step chosen after seeing that it failed.

If P1 fails the honest options are source-space projection or hyperalignment before averaging, and
either would be a new pre-registration, not a repair of this one.

## Primary statistic

Fixed here rather than chosen at analysis time:

    Δρ = ρ(cross-validated RSA at full k, encoding) − ρ(RSA at k = 1, encoding)

meta-analysed across MEG-MASC and Brennan, with uncertainty from resampling **both subjects and
tokens**. Co-primary: **independent-half leaderboard reproducibility** — split subjects into halves,
build a leaderboard in each, and correlate. A leaderboard that does not reproduce between halves
cannot be evidence about anything, whatever Δρ does.

## Confirmation

All of:

1. `C` is positive (agreement) at full `k` in both cohorts, with the cross-validated estimator.
2. `C` falls monotonically as `k` decreases, and at `k = 1` is **negative** in at least one cohort.
3. The `k = 1` concordance is within the CI of LibriBrain's −0.68 in at least one cohort.
4. At every `k` used to support 1–3, both readouts' ranking reliability CIs exclude zero.

## Falsifiers, fixed now

- **F1.** `C` at `k = 1` is still positive in both cohorts. **The reversal cannot be induced by
  removing repeats.** H is dead, and LibriBrain's −0.68 is an unexplained cohort-specific anomaly
  that we report as such rather than explain away.
- **F2.** `C` is already negative at full `k`. Then repeats do not buy agreement and the reversal is
  not an estimation artifact — a different and stronger scientific claim, but it refutes H.
- **F3.** Concordance moves only at doses where a ranking's reliability CI includes zero. Fatal, as
  in ADR 0003: that is two rankings becoming meaningless, not a relationship between two meaningful
  ones.
- **F4.** The cross-validated and plain estimators behave identically at all `k ≥ 2`. Then averaging,
  not cross-validation, carries any effect, and the constructive claim about distance estimators is
  not supported even if `C` moves.

## What the positive deliverable is

Not "crossnobis fixes it" — that is promised by nobody and registered as an open outcome. The
deliverable is a **minimum-reliability criterion for brain–LLM rankings**: the measured `R` below
which model rankings are not identifiable, established by the dose at which concordance stops being
interpretable, together with the diagnostic that any benchmark can run on its own data.

That is positive, actionable, and survives either direction of the result — which is why it, and not
the reversal, is what the paper should be built on.

## Relationship to ADR 0003

ADR 0003's noise-injection sweep is demoted to a **secondary robustness arm**, to be run only if
this experiment produces movement, and then only to check that the effect tracks reliability however
reliability is degraded rather than tracking the specific act of dropping subjects. ADR 0003's
falsifiers remain in force for it. Its dose-0 three-cohort reliability measurement (job 15246181)
remains primary and unaffected: it supplies the common reliability scale this experiment needs.

## Prior art this must be positioned against

Found 2026-08-03 and recorded before the result, because it constrains how large the claim may be.

- **Diedrichsen & Kriegeskorte 2017** (PLoS CB) unify encoding, PCM and RSA and show they test the
  same representational model, with differences arising from distance measure, noise model and
  regularization. This makes the *mechanism* here unsurprising. We may not present "the readouts can
  disagree" as a discovery.
- **Mazurchuk et al. 2023** (Lang Cogn Neurosci), "Stimulus repetition and sample size
  considerations in item-level RSA", already measures how RDM reliability depends on repetition and
  sample size. They do **not** compare RSA against encoding and do **not** test whether repetition
  count changes which model ranks best. Our contribution is that downstream consequence and nothing
  more; if reducing `k` merely lowers reliability without inducing the reversal, Mazurchuk owns the
  story and we have an increment, which the paper must say rather than dress up.
- **Ritchie et al. 2021** (NeuroImage) find noise normalization's benefit to RDM reliability is
  variable across regions and datasets and caution against adding preprocessing steps to RSA
  pipelines. That is a caution against exactly the kind of fix this work would propose, and it is
  cited as such rather than omitted.
- **Schrimpf et al. 2021** (PNAS) and **Goldstein et al. 2022** (Nat Neuro) already own the positive
  claim that the next-token-prediction objective predicts brain alignment. That route is closed and
  is not available as this project's positive result.

## Venue

Updated 2026-08-03. This targets **Imaging Neuroscience or NeuroImage** as the natural home; TMLR
becomes plausible only if the induced reversal is clean, monotonic, replicated in both cohorts, and
changes a substantive conclusion. A top conference is not in scope. The portfolio contains a failed
replication, a withdrawn mechanism, a null intervention, a contaminated side finding, and a
confirmed leakage; presenting the reversal as a broad scientific discovery would be evasive.

Recorded plainly, so it cannot be softened later: **before these jobs run, even TMLR is optimistic.**
If the effect is messy, what exists is a useful negative result or a methods note, not a flagship
positive paper, and that is what it will be called.
