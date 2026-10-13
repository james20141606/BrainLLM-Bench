# 5. Pre-register the crossed subject-by-narrative generalizability test

Date: 2026-08-04
Status: Accepted

## Context

The stimulus-split result — LibriBrain's readout concordance ranging over [−0.720, −0.048] across
random halves of the words while its published interval is [−0.83, −0.43] — demonstrates that the
reported confidence interval omits a variance component. It does not *estimate* that component, and
the reason is a real objection rather than a technicality: **words inside one continuous narrative
are not independent draws from a population of stimuli.** They share topic, syntax, speaker,
register and discourse structure. Two random halves of one audiobook are two dependent samples of
the same text, so their spread is a lower bound on the variance across genuinely different text, and
a poor estimator of it.

This is Clark's (1973) language-as-fixed-effect fallacy: analysing linguistic stimuli as fixed
licenses conclusions about the items used, not about the language. Westfall, Nichols & Yarkoni
(2016/17) brought it to task fMRI; Yarkoni (2020) generalised it. What has not been asked is whether
**model-ranking** benchmarks commit it — and ranking, not effect estimation, is what brain–LLM
leaderboards produce.

The metadata supports the test. Checked before writing this:

| cohort | independent blocks | listeners per block | words per block |
|---|---|---|---|
| MEG-MASC | **4 distinct stories** (`The_Black_Willow`, `cable_spool_fort`, `easy_money`, `lw1`) | 27 each | 32k–177k |
| Brennan | 12 segments of one narrative | 47 each | 6k–10k |
| LibriBrain | 12 sessions | **1 each** | 12k–22k |

MEG-MASC gives a genuine crossed subject-by-narrative design over four separate stories. Brennan
gives twelve contiguous segments of a single narrative — weaker, because segments of one story are
more dependent than four different stories, and it is reported as such. LibriBrain has narrative
blocks but only one listener, so it contributes no crossing and is excluded from the primary.

## Hypothesis

**H.** A brain–LLM leaderboard that is stable across listeners is materially less stable across
narratives, so an interval computed over subjects alone understates the uncertainty in any claim
that generalises to new text.

## Design

Primary cohort: **MEG-MASC, four stories, 27 subjects, the same 32 frozen checkpoints and the same
cell as every other run in this project.**

For each story independently: score all 32 checkpoints with both readouts, per subject; form the
leaderboard and the RSA-vs-encoding concordance for that story alone.

Then compute, and pre-declare as the primary contrast:

    V_subject  = variance of the leaderboard (Kendall tau against the pooled ranking) under
                 bootstrap resampling of the 27 subjects, stimulus held fixed
    V_story    = variance of the same statistic across the four stories, subjects held fixed

with the ratio `V_story / V_subject` as the headline. Co-primary: the six pairwise
between-story leaderboard agreements (Spearman over 32 models), against the within-story
between-subject-half agreement already measured.

Concordance is reported per story with its own interval, and the spread of the four story-level
concordances is compared against the subject-bootstrap interval computed on the pooled data.

## Confirmation

All of:

1. `V_story / V_subject > 2` — narrative choice moves the leaderboard at least twice as much as
   listener choice.
2. Median between-story leaderboard agreement is materially below the between-subject-half
   agreement already measured (Kendall τ ≈ 0.94).
3. At least one pair of stories yields concordances whose intervals do not overlap, or a
   between-story leaderboard agreement whose CI includes zero.

## Falsifiers, fixed now

- **F1.** `V_story / V_subject ≤ 1`. Narratives move the leaderboard no more than listeners do. The
  stimulus-as-fixed-effect concern, while formally correct, is **empirically unimportant in this
  data**, and the paper says exactly that rather than retreating to the formal argument.
- **F2.** Between-story leaderboard agreement is as high as between-subject agreement (τ ≥ 0.9 in
  all six pairs). Same conclusion as F1, stated from the ranking side.
- **F3.** Any story fails the rankability precondition — both readouts' split-half ranking
  reliability CIs must exclude zero within that story. A story whose leaderboard is not individually
  determined cannot contribute to a comparison between leaderboards. `lw1` is the risk at 32k words
  across 27 subjects, roughly 1 187 words per listener.
- **F4.** The four stories differ in length by 5.5x, so a between-story difference could be a
  sample-size effect rather than a narrative effect. **Mandatory control:** repeat the primary with
  all stories subsampled to the smallest story's per-subject word count. If the effect survives only
  at unequal lengths, it is a power artifact and is reported as one.

## What this can and cannot establish

It can establish that model rankings computed on one narrative do not transfer to another, which is
the operative question for any benchmark that reports a single ranking from a single stimulus set.

It cannot estimate the population variance over all English narratives from four stories. Four is a
small sample of texts, and the paper must say so; the claim is a lower bound on stimulus dependence
and a demonstration that the omitted component is non-negligible, not a variance estimate.

## Relationship to earlier ADRs

ADR 0003 (concordance is a function of neural reliability) and ADR 0004 (the reversal can be induced
by removing stimulus repeats) were both **refuted as registered** and are recorded as refuted. This
ADR does not rescue either. It tests a different and, unlike them, pre-theoretically motivated
question, and its falsifiers are written so that an empirically unimportant result is reportable as
such rather than convertible into a formal argument.
