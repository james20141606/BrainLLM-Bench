# P2 decisive control — per-band noise ceiling on LibriBrain

**Cell:** LibriBrain, `temporal_L`, 350 ms, GPT-2 layer 5. N = 12 sessions,
462 shared words. Neural RDMs, GPT-2 RDM, and confound RDMs all built over the
**same 462-word list in the same row order** (verified three ways in-run:
sorted word-intersection == `word_list_shared.txt`; GPT-2 `meta.json`
`word_list_path` == that same file, `n_words` = 462; every condensed RDM has
length 106 491). Ceiling = split-half over the 12 sessions (all 462 balanced
6-vs-6 splits, mean Spearman between half-means), Spearman-Brown corrected to
the full 12-session mean. Alignment = Spearman of GPT-2 L5 RDM vs the
**12-session-mean** neural RDM. Confound-controlled variant residualises both
sides on word length, Zipf frequency (`wordfreq`), and orthographic Levenshtein.
Result file: `results/p2_perband_ceiling.json`. Job 12125679, 547 s.

## (A) Per-band table

RAW (no confound control):

| band | Hz | split-half ceiling (SB) | alignment | normalized = align / ceiling |
|---|---|---|---|---|
| **gamma** | 30–80 | **−0.0020** | −0.00028 | 0.141 † |
| **broadband** | 1–80 | **0.135** | **0.00564** | **0.042** |
| delta | 1–4 | 0.244 | 0.00547 | 0.022 |
| theta | 4–8 | 0.107 | 0.00160 | 0.015 |
| alpha | 8–13 | 0.0029 | −0.00339 | −1.16 † |
| beta | 13–30 | 0.00059 | −0.00012 | −0.20 † |

CONFOUND-CONTROLLED (partial, | length + freq + orthography):

| band | ceiling (SB, partial) | alignment (partial) | normalized (partial) |
|---|---|---|---|
| gamma | −0.0023 | −0.00063 | 0.269 † |
| broadband | 0.158 | **−0.00358** | −0.023 |
| delta | 0.278 | −0.00644 | −0.023 |
| theta | 0.121 | −0.00571 | −0.047 |
| alpha | 0.0042 | −0.00212 | −0.51 † |
| beta | −0.00065 | 0.00064 | −0.98 † |

† **Not interpretable.** The band's split-half ceiling is indistinguishable
from zero (≈0 or negative), so "alignment as a fraction of its own ceiling" is a
0/0 quotient, not a meaningful normalized value.

## (B) The bb-vs-gamma normalized verdict

Literal numbers: `bb_norm = 0.042`, `gamma_norm = 0.141`,
**ratio bb_norm / gamma_norm = 0.30** (raw); −0.08 (partial). Taken at face
value this says broadband does **not** exceed gamma after normalization.

**But that face value is an artifact.** Gamma's split-half word-RDM reliability
on LibriBrain is **−0.002 (SB) ≈ 0**: gamma carries no reproducible word-level
RDM across sessions at all. `gamma_norm = 0.141` is a tiny negative alignment
(−0.00028) divided by a tiny negative ceiling (−0.0020) — a 0/0 quotient. Gamma
is **not** "winning"; it is unnormalizable. The per-band-ceiling test therefore
**cannot** be run as a clean `bb_norm` vs `gamma_norm` comparison on LibriBrain,
because the gamma denominator does not exist.

What *is* well-defined: the three bands with a positive, reliably-measured
ceiling — **broadband (0.135), delta (0.244), theta (0.107)** — are exactly the
low-frequency / high-power bands, and they are the only bands with positive raw
alignment (0.0056, 0.0055, 0.0016). Gamma, alpha, beta all have ceiling ≈ 0 and
alignment ≈ 0. Among the reliably-measured bands the normalized ordering is
broadband 0.042 > delta 0.022 > theta 0.015, but broadband and delta are a near
tie in **raw** alignment (0.0056 vs 0.0055), consistent with broadband being a
1/f low-frequency proxy rather than a distinct carrier.

## (C) What it means for P2's headline — REFRAME (frequency-locus claim does NOT survive)

The decisive question was: does broadband > gamma reflect a frequency-specific
neural locus, or an SNR/reliability advantage? The data give the **SNR /
reliability** answer:

1. Alignment appears in exactly the bands that have a reproducible word-level
   RDM (broadband, delta, theta), and is absent in the bands that do not
   (gamma, alpha, beta). Broadband beats gamma **because gamma has no reliable
   word structure to align to**, not because a mid-frequency carrier was
   identified and out-competed by a low-frequency one.
2. Because gamma's ceiling is ≈ 0, its alignment cannot be "attenuation-
   corrected" — there is no signal floor to divide by. The normalization that
   would adjudicate a genuine frequency locus is undefined for the very band the
   claim contrasts against.
3. Confound control removes even the broadband/delta advantage: at the group-mean
   level, partial alignment goes to ≈ 0 / negative for every band (broadband
   −0.0036, delta −0.0064). The residual, length/frequency/orthography-controlled
   contextual alignment of the group-mean RDM is not positive for any band —
   so the raw broadband alignment is largely lexical-confound-driven here.

**Verdict, stated plainly:** broadband > gamma does **not** survive as a
frequency-locus claim on LibriBrain. The per-band ceiling cannot rescue it —
gamma is unnormalizable (zero reliability). The honest characterisation is that
broadband/low-frequency power is the **only** band carrying a reliable,
LLM-alignable word-level RDM, and gamma carries none; this is an SNR/reliability
phenomenon. P2's headline should be reframed from "identifies a broadband/low-
frequency neural carrier that beats gamma" to "low-frequency/broadband power is
the only band with reliable word-level structure and therefore the only one that
aligns with LLMs — an SNR/reliability effect, not a demonstrated frequency-
specific carrier," with the added caveat that this group-mean alignment does not
survive lexical confound control.

## Caveats
- This is a **group-mean-RDM** analysis (as specified), so it is a stricter and
  differently-aggregated test than P2's per-unit Fisher-z average with per-unit
  best-layer argmax; the group-mean partial going to ≈0 should not be read as
  overturning every per-subject number, but it does undercut the group-level
  frequency-locus reading.
- Ceilings are split-half + Spearman-Brown on the 12-session mean; they are
  numerically larger than the single-session pairwise `mean_rho` (0.0159) in
  `reliability/libribrain_region_time.tsv` because averaging 6 sessions per half
  reduces noise (Spearman-Brown-consistent with 0.0159). The cross-band ratio,
  not the absolute normalization base, is the load-bearing quantity.
- Runnable only where raw epochs survive (LibriBrain, MEG-MASC). Gamma's zero
  word-RDM reliability here echoes the MEG-MASC word-level RDM-noise finding.
