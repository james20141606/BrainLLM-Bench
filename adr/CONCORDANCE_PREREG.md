# Pre-registration: do brain-alignment readouts rank language models the same way?

**Written 2026-07-16, BEFORE running the scaled experiment.** The point of writing
it first is that the analysis choices below cannot then be tuned to whatever the
data turns out to say. If the result contradicts the hypothesis, this file is the
evidence that we said so in advance.

Author: the authors. Status: design frozen; not yet run.

---

## 1. Why this experiment

The field ranks language models by "brain alignment" using at least three readouts
— representational similarity (RSA), linear encoding, and decoding — and treats the
resulting leaderboards as if they measured one thing. Our pilot on six model
families says they do not: RSA and encoding rank models at Spearman
rho = 0.60 while each readout's own split-half ranking reliability is 0.94.

But at n = 6 that is worth nothing: the 95% CI is [-0.41, +0.95]. It excludes
neither zero nor perfect agreement. The pilot cannot distinguish "the readouts
disagree" from "we have six points."

This experiment exists to make that estimate decisive or to kill it.

## 2. The claim, stated so it can fail

**Primary hypothesis (H1).** Observed RSA-vs-encoding rank concordance is
*substantially below the ceiling that each readout's own reliability permits*.

The psychometric argument is the whole design. If a readout's ranking has
split-half reliability `r_xx`, then noise alone caps its correlation with any
other measure. For two readouts the attenuation ceiling is

    ceiling = sqrt(r_xx * r_yy)

Anything below that ceiling is not noise — it is the two readouts measuring
different things. Pilot: ceiling = sqrt(0.943 * 0.943) = 0.943, observed 0.60,
so the attenuation-corrected correlation is 0.60/0.943 = 0.64. The gap (0.34) is
the effect of interest.

**H1 is supported iff** the 95% CI on the observed concordance excludes the
measured ceiling (upper bound < ceiling).

**H1 is falsified iff** the CI includes the ceiling, i.e. the readouts agree as
well as their reliability permits. **If that happens, the finding is dead and this
paper does not get written.** We commit to reporting that outcome. It would be a
real result (the field's readouts are interchangeable after all) but it is not a
paper we would claim novelty for, and we will not rescue it by switching to a
different readout pair or a different cohort post hoc.

**Ambiguous zone:** if the CI excludes the ceiling on one cohort and not the other,
we report both and claim nothing general. See section 7.

## 2b. The pilot's rho = 0.60 is not a usable baseline (found 2026-07-17, before running)

Recording this before the experiment runs, because it changes what the result can
be compared against.

Tracing what `p4_concordance.py` actually reads shows the pilot's two readouts
were not computed on comparable footing:

| | RSA | encoding |
|---|---|---|
| source | `path1_confound_libribrain/rsa_confound_per_model_subject.tsv` | `p4_encoding/families_lb.json` |
| layer rule | **per-model best layer** (`best_layer` column) | middle of a 3-6 layer cache (`feat.shape[1]//2`) |
| words | **462** | **177,127** |

So the pilot's rho = 0.60 confounds readout disagreement with three other
differences: a per-model-argmax layer rule against a fixed-middle rule, a
462-word set against a 177k-word set, and two separately built caches. The
`best_layer` rule is the very thing section 5 forbids, and section 6 of this
document already required the fixed-depth rule for exactly this reason.

**Consequence for framing.** This experiment is not replicating or refuting 0.60.
That number is not interpretable, so it is not the anchor and we do not report
"the pilot found 0.60 and we find X". This is the first clean measurement: both
readouts on the same words, the same sessions, and the same fixed relative depth.
The power table below still uses 0.60, because it is the only prior estimate that
exists and some effect size is needed to size the study -- but it sizes the study,
it is not a hypothesis about what we will see.

**Consequence for design.** Both readouts must run on one shared word set. RSA
cannot use all 177,127 words (the dissimilarity matrix is O(N^2)), so we subsample
per session under a fixed seed, exactly as `stage1_streaming_rsa.py` does at
2500/session, and encoding runs on that same subsample rather than on the full
set. Same words, same sessions, same layer -- the only thing that differs between
the two numbers is the readout, which is the whole point.

## 3. Power

Fisher-z CI on Spearman rho, assuming the pilot's rho = 0.60 and ceiling = 0.943
hold:

| n models | 95% CI | excludes 0 | excludes ceiling |
|---|---|---|---|
| 6 (pilot) | [-0.41, +0.95] | no | no |
| 20 | [+0.21, +0.82] | yes | **yes** |
| 30 | [+0.31, +0.79] | yes | yes |
| 32 | [+0.32, +0.78] | yes | yes |

**n = 20 is the decision threshold; we will run 32** so that dropping models for
QA reasons (section 6) cannot silently push us under it. Committing to 32 in
advance also removes the option of adding models until the CI cooperates.

## 4. Model set (n = 32, frozen)

Chosen to span training objective and scale, which are the axes the field's
claims are about. Frozen here so the set cannot be edited after seeing results.

**Causal / autoregressive (18)**
gpt2, gpt2-medium, gpt2-large, gpt2-xl,
pythia-70m, pythia-160m, pythia-410m, pythia-1b, pythia-1.4b, pythia-2.8b,
opt-125m, opt-350m, opt-1.3b,
gpt-neo-125m, gpt-neo-1.3B,
Llama-3.2-1B, Qwen2.5-0.5B, Qwen2.5-1.5B

**Masked / bidirectional (9)**
bert-base-uncased, bert-large-uncased,
roberta-base, roberta-large,
deberta-v3-base, deberta-v3-large,
electra-base-discriminator, albert-base-v2, distilbert-base-uncased

**Encoder-decoder (5)**
t5-small, t5-base, t5-large, bart-base, bart-large

Rationale: 18/9/5 keeps every objective class large enough that the ranking is not
driven by one family, and the Pythia + GPT-2 + OPT ladders let the same data answer
the scaling question honestly (which the current paper does not — see
the project lab notebook 2026-07-15).

## 5. Cohorts and readouts

**Cohorts.** LibriBrain (462 words, 36 sessions) as primary; MEG-MASC (500 words,
76 sessions, 10 subjects) as the independent replication. ECoG (9 patients) is a
stretch goal, not part of H1.

LibriBrain's 36 sessions come from **one listener**. This does not invalidate H1 —
the unit of analysis is the *model* (n = 32), not the brain — but it does mean
LibriBrain alone cannot show the disagreement generalizes across brains. That is
exactly why MEG-MASC (10 subjects) is required rather than optional.

**Readouts.**
- **RSA**: confound-controlled partial Spearman, the protocol already in the paper.
- **Encoding**: ridge, leave-one-session-out, held-out R.
- **Decoding**: contrastive, included **only** to characterize its reliability, not
  in H1. The pilot puts its split-half ranking reliability at 0.54, meaning a
  decoding leaderboard is close to unrankable. If that replicates at n = 32 it is
  a second, independent finding worth reporting on its own terms.

**Fixed cell rule.** Every model is scored at the cohort's headline cell
(LibriBrain: temporal_L, 350 ms, broadband), fixed in advance, not per-model
argmax. Per-model argmax is what produced the current paper's indefensible
"scaling is contradicted" claim; we do not repeat it.

## 6. Procedure and QA, fixed in advance

1. Extract per-word, per-layer features for all 32 models on each cohort's word
   list. Layer selection: fixed relative depth, not per-model best layer.
2. Compute all three readouts per model at the fixed cell.
3. Compute each readout's split-half ranking reliability over sessions
   (`r_xx`, `r_yy`), median over splits, with a 90% interval.
4. Compute RSA-vs-encoding Spearman `r_xy` over the 32 models; Fisher-z 95% CI.
5. Compare `r_xy` against `ceiling = sqrt(r_xx * r_yy)`. Decide H1 by section 2.
6. Repeat 2-5 independently on MEG-MASC.

**Exclusion rules, decided now:** a model is dropped only if feature extraction
fails or its tokenizer cannot align to the word list — never because of its score.
Every drop gets logged with its reason in the project lab notebook. If drops take
n below 20 the experiment is underpowered and we say so rather than report it.

## 7. Known risks (recorded so they cannot be rationalized later)

- **rho may rise with n.** Thirty-two models span more range than six; a wider
  spread can lift the correlation. If rho rises to ~0.9 the gap closes and H1 dies.
  This is the most likely way the experiment kills the finding, and it is a real
  possibility, not a formality.
- **Reliability may fall with n.** More near-duplicate models (bert-base vs
  distilbert) produce near-ties, which can drop ranking reliability below 0.94.
  That *lowers the ceiling* and shrinks the gap from the other side. Both risks
  push the same way: toward no finding.
- **The two cohorts may disagree.** Then we report both and claim nothing general
  (section 2).
- **n = 1 listener on LibriBrain.** Constrains generality, not H1. MEG-MASC is the
  check.

## 8. What this buys if it lands

A direct, quantified methodological warning to an active subfield: brain-alignment
leaderboards are readout-dependent, and the disagreement exceeds what either
readout's noise can explain. That is useful to people building these benchmarks
regardless of which readout they favor, and it is the one claim in this body of
work whose evidence base can be made decisive with compute we actually have.

If it does not land, the fallback is the broadband result (206/213 recordings, four
cohorts, ECoG 9/9 independent patients, p = 0.002, with a reliability mechanism),
written as the physiology paper the original plan specified. That paper does not
need this experiment.

## 9. Feasibility (checked 2026-07-16, not assumed)

- `ssh <cluster>` reachable; <gpu-partitions> partitions have idle nodes.
- `codes/encoding_analysis.py` is already parameterized by `--models` and
  `--llm_embed_dir`.
- Extraction templates exist (`extract_gpt2_layers_libribrain.py`,
  `extract_arch_ablation.py`, `cinebrain_extract_text_llm.py`).
- Scale is small: 462 words. The RSA dissimilarity vector is 106,491 floats
  (0.85 MB). The memory bottleneck that motivated the deleted "streaming engine"
  does not exist, which is very likely why this experiment was never attempted.

## 10. Critical path (traced on the cluster 2026-07-16, not assumed)

The analysis scripts already exist and are small:

- `code/paper4_nn_revision/p4_encoding_families_lb.py:27` holds the model set as a
  literal `FAMILIES` dict (currently 6). It reads pre-extracted per-word,
  per-layer features from `{fam}_contextual.npy`, shape `(N, nLayers, d)`.
- `code/paper4_nn_revision/p4_concordance.py` consumes the per-model readout
  values (RSA TSV + `families_lb.json`) and does the rank correlations.

~~So the work is: extract `{fam}_contextual.npy` for the 26 new models, extend
`FAMILIES`, rerun. Feature volume is trivial — 462 words x 13 layers x 768 dims
is about 18 MB per model, roughly 600 MB for all 32. Extraction is minutes per
model on an H200 even at 2.8B. **Estimated total: 1-2 GPU-hours.**~~

**Struck 2026-07-17 — this estimate was wrong and is superseded.** It used 462
words, which is the *ceiling analysis's* repeated-word cache, not the number of
epochs extraction runs over (177,127). Measured reality: 6.6 GB per model at full
stacks, so the frozen 32 would have been 500 GB - 1 TB. Resolved by writing only
the pre-registered relative depths (section 6), which the fixed-depth rule made
the correct choice anyway: ~1.6 GB per model, ~75 GB total. Actual extraction cost
was about 10 minutes per model, and the existing `FAMILIES` route turned out to be
unusable regardless (its causal path hardcodes GPT2Model), so the work was a new
extractor rather than a registry edit. See the project lab notebook 2026-07-17.

**Bug found while tracing, must be fixed before the run:** `FAMILIES` labels
`t5` as `"masked"`. T5 is an encoder-decoder, not a masked LM, and lines 95-96
average the `causal` and `masked` groups using that label, so the current
objective-class comparison is contaminated. This pre-registration puts T5 and
BART in their own encoder-decoder class (section 4), which avoids the error
rather than inheriting it. The existing 6-family objective comparison should be
re-checked once T5 is relabelled.

## 11. Order of work

1. Fix the T5 label bug; re-check whether the existing objective comparison moves.
2. Extract features for the 26 new models on LibriBrain (462 words). Log any model
   that fails and why (section 6).
3. Run RSA + encoding at the fixed cell for all 32; compute reliabilities.
4. Decide H1 on LibriBrain by the section 2 rule. **Record the verdict before
   looking at MEG-MASC**, so the replication is a replication.
5. Repeat on MEG-MASC.
6. Report decoding's reliability as a separate result whatever H1 does.
