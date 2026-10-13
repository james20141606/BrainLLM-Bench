# 1. Pre-commit the n=27 MEG-MASC analysis before the data exists

Date: 2026-07-29
Status: Accepted

## Context

Paper B's central claim is a cross-cohort dissociation: the constituent-boundary effect
replicates per-subject in Brennan EEG and is null in MEG-MASC. The MEG-MASC arm has run at
n=10 (a subset inherited from a lost preprocessing run), where it excludes a Brennan-sized
effect at p=0.001 but has a resolution floor of |d| >= 1.0.

We are rebuilding the cohort at n=27. Power roughly triples. The result could tighten the null
or invert the paper's central dissociation.

Paper B's subject is analytic flexibility: it argues that pipeline and cell choices move
published brain-LLM results. A paper making that argument cannot choose its own interpretation
rule after seeing the number it will interpret. The ELN is append-only and git-committed with
timestamps, so a rule written now is a verifiable pre-commitment at no cost.

## Decision

Four commitments, fixed before the n=27 boundary result exists.

1. **Scope.** Re-run Paper B's boundary arm first. Whether to propagate to Paper A's MEG-MASC
   cells (leaderboard, three scorecard rows, concordance reliability) is decided afterwards,
   and the mixed-N state is disclosed until then.

2. **Primary statistic: the exclusion bound.** The primary test is non-inferiority -- does the
   n=27 interval still exclude Brennan's raw effect? The Wilcoxon significance test is
   secondary. The non-replication thesis is retained only if the exclusion still holds. A
   merely-underpowered null is not evidence of absence; an exclusion bound is.

3. **Primary cell: Brennan-matched.** 400 +/- 75 ms, all sensors, fixed layer, BERT, sentence
   onset versus all later words, equal-count subsampled. MEG-MASC's own best cell is still
   reported. If it turns positive at n=27 we state the effect exists at a non-matched cell and
   weaken the claim to "does not replicate at the matched cell"; it qualifies the thesis but
   cannot by itself overturn it.

4. **Inclusion: an artifact-rate floor, reported both ways.** Exclude subjects more than 3 MAD
   above the cohort median high-amplitude-epoch rate. Primary analysis on survivors; the
   all-subjects result is always reported alongside.

## Consequences

The paper can state that its interpretation rule predates its data, which is the strongest
available answer to a reviewer who suspects the null was chosen rather than found.

We accept a cost on (2): if the effect is real and small, a design powered to exclude only
Brennan-sized effects will report an exclusion that a larger study overturns. That is the
honest reading of what this sample can resolve, and the resolution floor is stated.

On (4), the metric was chosen by elimination, not preference, and the alternatives were
measured rather than assumed:

- *Epoch count* is degenerate. Every subject heard the same four stories, so counts are exactly
  16464 (two sessions) or 8232 (one). A floor here means "drop single-session subjects" and
  says nothing about quality.
- *RDM split-half reliability* is noise. Per-subject values on the existing archive have median
  +0.0009, range -0.0030 to +0.0038, with 0/10 above 0.05 and 4/10 negative -- a spread the
  same size as the values (results/megmasc_meta/subject_reliability.json). Thresholding it
  would exclude subjects at random while appearing principled.
- *Sensor artifact rate* discriminates: 5.2% to 49.7% high-amplitude epochs across the current
  10, a 10x spread. It is computed from raw amplitude alone, independent of the boundary
  contrast, so it cannot be tuned toward a preferred answer.

Paired reporting is what makes (4) defensible. Paper B criticises both MEG-MASC's inherited
10-of-27 discard and Brennan & Hale's 33-of-49 exclusions, so imposing our own exclusion needs
the reader to see exactly what it did. Note the floor is not free: on the current 10 it would
drop sub-02, sub-03 and sub-13, so the n=27 sample is not a clean superset of the n=10 one, and
that must be stated when the two are compared.
