# 8. Amendment to ADR 0006: new-generation Qwen ladders (user-directed recency extension)

Date: 2026-08-15
Status: Accepted (amends 0006; does not alter any 0006 rule or verdict)

## Context

After ADR 0006 was registered, the user pointed out that Qwen2.5 is no longer the newest
generation (Qwen3.5 and Qwen3.8 now exist). The recency objection ADR 0006 was built to answer
is therefore only partially answered by its frozen set. This amendment ADDS new-generation
ladders; it does not replace, re-scope, or re-run anything registered in 0006, whose Q1/Q2/Q3
proceed exactly as written on exactly the frozen 0006 set.

Disclosure: an interim, exploratory scoring of the available Qwen2.5 rungs (ELN 2026-08-13) had
been seen before this amendment was written. Its numbers played no role here — the added families
are fixed by what exists on the Hub, the rules are copied verbatim from 0006, and no rule or
threshold differs from 0006 in any direction.

As of this commit, no Qwen3/3.5/3.8 checkpoint has entered the pipeline.

## Added model set (frozen; HF revision `main` at extraction, recorded per model)

- **Qwen3 base ladder (5):** Qwen3-{0.6B, 1.7B, 4B, 8B, 14B}-Base.
- **Qwen3.5 base ladder (4):** Qwen3.5-{0.8B, 2B, 4B, 9B}-Base.
- **Post-trained top rungs (3, no base variant exists):** Qwen3-32B, Qwen3.5-27B, Qwen3.8-27B.
- **Post-training twins (2):** Qwen3-8B, Qwen3.5-9B (vs their -Base twins).

MoE releases (Qwen3.5-122B/397B, Qwen3.8-2.4T) are excluded: they exceed the available
GPU memory envelope and mix expert routing into the representation, which no frozen model shares.

## Registered analyses (rules copied verbatim from 0006)

**Q1-per-generation.** The 0006 Q1 statistic (session-clustered Spearman over the ladder, plus
top-vs-bottom paired contrast) is computed independently on the Qwen3-Base ladder and the
Qwen3.5-Base ladder, with the same supported / not-supported / mixed rules. Base ladders only:
the post-trained top rungs do NOT enter Q1 (objective changes across the rung otherwise).
Cross-generation statement: whichever way each ladder falls is reported per generation; agreement
across 2.5 / 3 / 3.5 is the replication claim, disagreement is reported as a generation effect.

**Q2 stays base-only and gains the new base rungs.** The 0006 Q2 concordance set (frozen 32 +
0006 base additions) gains the 9 new -Base checkpoints. Post-trained models are excluded from
registered Q2 (the frozen set is pretrained-base throughout); an exploratory concordance
including them is reported as exploratory. The 0006 F1 falsifier applies unchanged to the
registered set.

**Q3 gains two twins.** Qwen3-8B vs -Base and Qwen3.5-9B vs -Base join the 0006 instruct pairs
under the same exploratory rules (per-session paired differences, direction counts, no pooled p).

## What cannot rescue a failed primary

Unchanged from 0006. No per-model layer search; relative depth 0.5 rule; F3 rankability gate.
If a ladder is incomplete at analysis time (extraction failure), the affected generation's Q1 is
reported as indeterminate for the missing rungs, never backfilled by substitution.
