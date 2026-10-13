# 10. Pre-register the second-cohort scaling ladder (Qwen2.5 on MEG-MASC)

Date: 2026-08-20
Status: Accepted

## Context

The external verdict on Paper A names one weakness text cannot remove: the load-bearing scaling
and reversal results ride on one deeply sampled listener (LibriBrain). This ADR registers the one
experiment that addresses it — the same Qwen2.5 ladder scored on MEG-MASC's 27 subjects under the
already-published MASC protocol. The design is antifragile by construction and we commit to both
readings in advance: replication strengthens the scaling result; non-replication is a further
instance of the paper's own cohort-dependence thesis. Both outcomes enter the paper.

As of this commit, no Qwen2.5 checkpoint above 1.5B has ever been extracted on the MASC word list
(0.5B and 1.5B entered with the frozen 32).

## Design (frozen)

- **Extraction:** Qwen2.5-{3B, 7B, 14B, 32B, 72B} on the MEG-MASC merged-27 trial metadata via
  the existing `extract_llm_layers_multi.py` MASC path (`--context_from_columns`), depths
  0.25/0.5/0.75, same completeness checks. No other models.
- **Scoring:** the published MASC protocol exactly as `concordance32_masc27` ran it (merged-27
  data, kit208 atlas, MASC headline cell, subject as the replication unit, fixed seed). One run,
  all available models, output to a NEW file; no published artifact is rewritten.
- **Statistic (Q1-MASC):** the 0006 Q1 statistic verbatim with subject (27) replacing session
  (12) as the cluster unit: subject-clustered bootstrap of Spearman(ladder order, per-subject-mean
  alignment) per readout, plus the 72B-vs-0.5B paired within-subject contrast.

## Decision rules (fixed now; 0006 wording verbatim per readout)

- *Supported / not supported / mixed*: as in ADR 0006 Q1.
- **Cross-cohort statement, pre-committed:**
  - Both cohorts NOT SUPPORTED (and encoding negative in both): "size fails to order alignment in
    two cohorts with independent listeners" — the scaling verdict generalizes.
  - LibriBrain NOT SUPPORTED but MASC supported (or vice versa): "the scaling verdict is itself
    cohort-dependent" — reported as such, as a further instance of the paper's central claim,
    with neither cohort privileged.
  - MASC indeterminate (rankability gate failed): reported as indeterminate, not folded into
    either reading. Gate: split-half rank reliability of the MASC ladder leaderboard with a 90%
    CI excluding zero, per readout (F3 as in 0006).
- The LibriBrain verdicts of ADR 0006 are final and cannot be revised by this ADR.

## What cannot rescue a failed primary

No cell, layer, or subject selection; no alternative medial definitions; the MASC cell is the one
the published `concordance32_masc27` used. If 72B extraction fails twice on available hardware the
ladder ends at 32B and the paper says so.
