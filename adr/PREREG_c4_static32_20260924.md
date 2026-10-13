# PREREG — C4 static-embedding control for the fixed-depth RSA route, 32 frozen checkpoints

Written 2026-09-24 (Beijing), **before any layer-0 feature of the 30 missing models has been
extracted and before any C1–C4 score has been computed.** The commit time of this file is the
evidence. After results exist this file is not edited; deviations go to a dated "Deviations"
note in the ELN and in the result receipt.

## 1. Question

The paper's headline readout comparison (LibriBrain, 32 checkpoints frozen in
`CONCORDANCE_PREREG.md` §4 / `MODEL_SET` in `extract_llm_layers_multi.py`) scores similarity (RSA)
as a partial Spearman correlation with three lexical confounds C1–C3 (length, log frequency,
orthographic edit distance) and finds the RSA ranking and the ridge-encoding ranking in opposite
orders, ρ = −0.573 (checkpoint Fisher 95% CI [−0.77, −0.28]; family-clustered 95% CI
[−0.79, −0.26], receipt `results/p4_encoding/concordance32_CORRECTED.json`). The tensor route of
the same paper also partials out C4, each model's own static layer-0 embedding RDM; the
fixed-depth route never did, and the paper says so in six places (Fig. 1 caption, Setup, Intro,
§tensor, Limitations, Table `tab:protocols`).

**Does the RSA-vs-encoding ranking reversal survive when each model's own static layer-0
embedding RDM is partialled out of its RSA score as a fourth confound?**

This is a control on the RSA side only. Encoding scores are not recomputed.

## 2. Estimands

For model m (32), session u (the same 12 LibriBrain sessions), on the same fixed 30,000-word
subsample (≤2,500 per session, `RandomState(42)`, `p4_concordance32.subsample`):

- D^b_u: correlation-distance RDM of the neural pattern (temporal_L, broadband, 300–400 ms) —
  unchanged from `p4_concordance32.py`.
- D^(m)_u: cosine-distance RDM of the model's 0.5-relative-depth features from
  `llm_embeddings_fd` — unchanged.
- C4^(m)_u = cosine-distance RDM of the **same model's layer-0 hidden state**
  (`hidden_states[0]`, the embedding-layer output) read with the **same target-token rule** as
  the 0.5-depth features (causal: last token of the prefix ending at the target word; masked /
  encoder–decoder: mean over the target word's tokens, encoder stack). Cosine, because the model
  RDM of this route is cosine (the tensor route used correlation distance for both its model RDM
  and its C4; the rule carried over is "C4 uses the route's model-RDM metric").

  s^{C4}_{m,u} = ρ_S( D^b_u, D^(m)_u | C1, C2, C3, C4^(m)_u )

  computed with the route's own `partial_spearman` (rank-transform, residualize on
  [1, rank C1..C4], Pearson of residuals). Pairs where any of D^b, D^(m), C4 is non-finite are
  dropped (count reported).

- RSA_C4(m) = mean over the 12 sessions of s^{C4}_{m,u}.
- ENC(m) = `enc_mean` of `concordance32_CORRECTED.json` (per-session values identical, max |Δ| = 0,
  to `concordance_modern_final.json` / `_v2.json` for these 32 models; checked 2026-09-24).

**Primary estimand:** ρ_C4 = Spearman( RSA_C4(m), ENC(m) ) over the 32 models.

**Comparator:** ρ_C3 = −0.5733 (the published C1–C3 value), recomputed in the same scoring run
(see gate G2) so both come from one code path.

**Secondary (reported, not decision-driving):**
- Δ = ρ_C4 − ρ_C3 with a paired family-clustered 95% interval (same family resamples for both).
- Checkpoint Fisher 95% CI of ρ_C4 (n = 32).
- Split-half ranking reliability of RSA_C4 (Spearman–Brown, 200 splits, seed 0, as in
  `splithalf_rank_reliability`) and the attenuation ceiling sqrt(r_xx · r_yy) with r_yy = 0.955
  (encoding reliability of the headline receipt).
- Per-model RSA_C3 → RSA_C4 change; class means (causal / masked / encdec).

## 3. Inference

- Family-clustered 95% CI: `recompute_corrected_stats.family_clustered_ci` unchanged —
  resample the 12 families of `model_families.py` with replacement, 10,000 draws, seed 0,
  percentile interval. (On the headline receipt this reproduces [−0.786, −0.264].)
- α = 0.05, two-sided intervals. No multiplicity correction: one primary.

## 4. Sample and exclusions

- Models: the 32 in `MODEL_SET`, no additions, no substitutions.
- Rows: the 30,000 selected trial rows. Hard gate: SHA-256 of the int64 row-index array =
  `bfe5c58de6dc807aae86ad2b4b2a59d8a7c96e57f10490ab000e2c2fcdb95346`; metadata SHA-256 =
  `b8e7784006b049060cb0383b5f4ad4a191674e540ed571e0aa831c1d6e7e843e`.
- Layer-0 features are extracted **only for these 30,000 rows** (not all 177,127): /scratch is at
  99.86% of quota and the scorer reads no other row. Contexts are the same
  `build_sentence_contexts_from_trial_metadata` entries indexed by the same row numbers, pushed
  through the <cluster> copy of `extract_llm_layers_multi.extract_one` unchanged (batch 32, fp32,
  max_fallback_frac 0.05). Layer 0 of every model is independent of batch composition (no
  cross-token mixing before the first block), so the restriction cannot change it.
- Weights: 30 of the 32 checkpoints are no longer in the HF cache (only gpt2 and
  bert-base-uncased remain). They are re-downloaded **into the job's RAM disk (/dev/shm)**, never
  to /scratch, and deleted at task end. All 32 are ≤ 2.8B parameters.

## 5. Validity gates (checked before the primary is read)

- **G1 rows**: row-index and metadata hashes as above, in every extraction and scoring receipt.
- **G2 pipeline identity**: the scorer recomputes s^{C3}_{m,u} (C1–C3) in the same run. Every one
  of the 32 × 12 values must match `per_session_rsa` of `concordance32_CORRECTED.json` to
  |Δ| ≤ 1e-8, and the recomputed ρ_C3 must be −0.5733 ± 1e-4. If G2 fails, C4 is not read; the
  scorer is debugged and the failure logged.
- **G3 weights/rows identity (per model)**: the extraction job also recomputes the 0.5-depth layer
  on the same 30,000 rows and compares it with the cached `llm_embeddings_fd` rows used by the
  headline. Pass: median per-row cosine ≥ 0.99999 **and** ≥ 99.9% of rows with cosine ≥ 0.9999
  (batch composition differs from the original run, so bitwise equality is not expected above
  layer 0). A model failing G3 has weights or tokenization that differ from the headline run; its
  layer 0 is not the headline model's layer 0.
  Additional check for gpt2: the new layer 0 must equal `llm_embeddings_v2/gpt2/layer_0.npy`
  rows (max |Δ| ≤ 1e-5), the one pre-existing, hash-verified layer-0 cache.
- **G4 alignment**: masked/encdec target-word fallback ≤ 5% (the extractor's own rule).
- **G5 completeness**: every layer-0 file 30,000 × d, finite, no all-zero rows.

If 1–2 models fail G3/G4/G5, the primary is computed on the passing models and ρ_C3 is recomputed
on the same subset; both are reported with the exclusions named. If ≥ 3 fail, the result is
reported in the ELN only as "C4 run incomplete" and the paper keeps its current C1–C3 scope
sentences.

## 6. Decision rule (applied mechanically to ρ_C4, its family-clustered CI and r_xx(C4))

Let [L, U] be the family-clustered 95% CI of ρ_C4.

- **A. Reversal holds under C4**: U < 0.
  Sub-label by Δ's paired family CI: "undiminished" if the Δ CI includes 0; "weakened" if the Δ CI
  lies entirely above 0; "strengthened" if entirely below 0.
- **B. Not determined under C4**: L ≤ 0 ≤ U and ρ_C4 < 0.
- **C. Reversal removed**: ρ_C4 ≥ 0 and L ≤ 0 ≤ U ("vanishes"), or L > 0 ("becomes agreement").
- **D. Instrument check (overrides B and C)**: if r_xx(C4) < 0.70, the C4-partialled similarity
  ranking is too unreliable for its concordance to be informative; the outcome is
  "indeterminate: C4 leaves too little reliable ranking variance" regardless of ρ_C4.

## 7. What the paper says in each case

Numbers in brackets are filled mechanically from the receipt. The six scope sentences listed in §1
are replaced by one short result sentence each (net 0 lines in the body); details go to a new
appendix subsection with a table (per-class C3/C4 RSA, ρ_C3 vs ρ_C4, both CIs, Δ, r_xx) and the
paths of this file and the receipts.

- **A (undiminished/weakened/strengthened)** — Fig. 1 caption: "…(C1--C3) partialled out; adding
  each model's static layer-0 embedding (C4) leaves the reversal at ρ = [ρ_C4]." Setup: "Adding
  the model's own layer-0 embedding as a fourth confound C4 leaves the reversal at ρ = [ρ_C4]
  (family-clustered CI [L, U]; Appendix)." Limitations/Intro: C4 no longer listed as a gap; the
  weakened case says "weakened from −0.573 to [ρ_C4]". **Abstract: unchanged** (it states no C4
  scope).
- **B** — same slots: "with each model's static layer-0 embedding also partialled out (C4) the
  concordance is [ρ_C4], family-clustered CI [L, U] including zero, so the reversal is not
  determined under this control." Intro limits list replaces "not yet run with C4" by this
  finding. **Abstract: one clause added** after "feature geometry is not excluded": "and it is not
  determined once each model's static embedding is also partialled out (ρ = [ρ_C4])".
- **C** — "with C4 partialled out the concordance is [ρ_C4] (CI [L, U]): the reversal on this route
  does not survive static-embedding control." **Abstract: the reversal sentence gets the same
  qualifier**; conclusion's reversal sentence likewise.
- **D** — "adding C4 leaves too little reliable similarity-ranking variance (split-half
  reliability [r_xx]) for the concordance to be informative." Abstract unchanged; limitation kept
  as "not informative under C4" instead of "not run".

**Relation to "feature geometry is not excluded".** C4 removes the part of each model's contextual
RDM that is predictable from its own non-contextual token embedding. It does not standardize or
whiten features, so it does not test the anisotropy/geometry account (Prop. 1, mean-centering and
standardization variants). In every outcome the sentence "feature geometry is not excluded" and the
§"We cannot exclude feature geometry" paragraph stay as they are; no outcome licenses the claim
that geometry is excluded, and outcome A is not described as evidence against a geometry account.

## 8. Not claimed / not done

- No C4 for the 57- or 48-checkpoint sets, MEG-MASC or Brennan (minimum viable version only; the
  paper keeps "C4 not run" wording for those if it mentions them).
- No change to encoding scores, layer rule, cell, or confounds C1–C3.
- A smoke (gpt2, bert-base-uncased only) verifies the pipeline; its scores are not read as a
  result and do not rank anything.

## 9. Compute and artifacts

- Extraction: Slurm array, one task per model, <gpu> (`code/scripts/slurm/run_p4_c4_l0_extract.sbatch`),
  HF weights to /dev/shm; output `data/preprocessed/llm_embeddings_l0_c4sel30k/<tag>/layer_0.npy`
  (~4.6 GB total, fp32) + per-model JSON receipt (file SHA-256, row hash, snapshot revision, dims,
  fallback, G3 statistics). After scoring, the layer-0 files are moved to /archive to free /scratch;
  their SHA-256 stay in the receipts.
- Scoring: CPU partition, `--nshards` over models (`code/scripts/slurm/run_p4_c4_score_cpu.sbatch`),
  `codes/p4_concordance32_c4.py`; aggregation `codes/p4_c4_aggregate.py` →
  `results/p4_encoding/concordance32_c4.json` with input SHAs, script SHAs, seeds, JIDs.

---

## Deviations (appended 2026-09-24 18:50 Beijing; sections 1-9 above are unchanged from commit 6e81d61)

Both deviations were decided after the gates were read and **before any C4 value was read**. The
decision rule (section 6), the gate thresholds (section 5) and the "≥ 3 G3 failures → C4 run
incomplete, ELN only" clause are unchanged.

**Deviation 1 (15:44, ELN 19270c5) — gpt2.** The smoke (18407672_0) failed G3 for gpt2; the headline
gpt2 0.5-depth file `llm_embeddings_fd/gpt2/layer_6.npy` is file-identical to
`llm_embeddings_v2/gpt2/layer_6.npy` (the older `extract_llm_layers.py` extraction). C4 for gpt2 is
therefore `llm_embeddings_v2/gpt2/layer_0.npy`, gated by re-checking that file identity and the
layer-0 SHA-256 (JID 18408635, CPU).

**Deviation 2 (18:50) — the other 17 causal checkpoints.** In the 30-model array (18408587) every
masked and encoder–decoder model passed G3 at median row cosine 1.0000000, and every causal model
failed it (median row cosine 0.786–0.99993; 0.2–6.5% of rows at cosine ≥ 0.9999). Cause, from git and
the 2026-07-17 ELN: the headline causal features were written by array 14100743 (Llama rerun
14103660) with the extractor as committed at **ef17743**, which tokenizes the joined string
`" ".join(prefix)` with the default tokenizer; commit ba9ff1f then switched the extractor to word
lists with `is_split_into_words=True, add_prefix_space=True`, and only the 14 masked/encoder–decoder
models were re-extracted with it. The <cluster> copy used by this control is the post-ba9ff1f version,
so its causal layer 0 is not the headline run's layer 0, which is what G3 exists to detect.
Change: the 17 causal models other than gpt2 are re-run with `extract_one` taken **verbatim** from
git ef17743 (`codes/extract_llm_layers_multi_ef17743.py`, SHA-256
c9b736f703d479d861d44d28cb7d0c555928eddc72371918722cf06edb40a234, no edits) via
`--extractor ef17743` (sbatch `run_p4_c4_l0_ef17743.sbatch`), so that C4's layer 0 and the headline
features come from the same extractor version. G3 is applied unchanged (median row cosine ≥ 0.99999
and ≥ 99.9% of rows ≥ 0.9999 against `llm_embeddings_fd`). If ≥ 3 of the 32 still fail G3/G4/G5,
the outcome is INCOMPLETE per section 5 and the paper keeps its C1–C3 scope sentences. The failed
post-ba9ff1f receipts are kept as `per_model/<tag>_wordids_G3fail.json`.

**Procedural note (deviation 3).** bert-base-uncased's receipt is the smoke task 18407672_18
(section 8 said smoke scores are not read as a result). Its extraction/scoring path is
code-identical to the array's: `p4_c4_extract_score.py` 917c371c… (smoke) and 9c504800… (array)
differ only by the added gpt2 `v2_path`; `p4_c4_lib.py` is 72199e41… in both. It is used as the
bert-base-uncased entry of the 32, not as a separate reading, and nothing was ranked from the smoke.

**Deviation 4 (appended 2026-09-25 02:17 Beijing, before any deviation-2 receipt exists and before
any C4 value of the 17 causal models is read) — hardware.** The deviation-2 GPU array 18440459
(<gpu-partition>) is estimated by Slurm to start after the 10:00 cut-off, and no GPU partition starts
earlier. The same 17 extractions are therefore also run on CPU nodes (partition <cpu-partition>, one task
per model, `codes/scripts/slurm/run_p4_c4_l0_ef17743_cpu.sbatch`): identical command
(`p4_c4_extract_score.py --extractor ef17743`, same code SHA-256s, fp32, batch 32), only
`torch.device("cpu")` instead of CUDA, and torch intra-op threads = the task's cores (numpy/OpenBLAS
stays single-threaded for the scoring workers). CPU and GPU fp32 kernels differ in summation order,
so layer 0 and the 0.5-depth rows are not expected to be bitwise equal to a GPU run; G3 is applied
**unchanged** (median row cosine ≥ 0.99999 and ≥ 99.9% of rows ≥ 0.9999 against the cached GPU
features), so a CPU numerics difference large enough to matter fails G3 and counts toward
INCOMPLETE. Whichever route (CPU array or GPU 18440459) first delivers all 17 receipts is used, in
full (no mixing of routes across models); the other is cancelled. The route used is recorded in
every receipt (`gpu` field null on CPU, `host`) and in the aggregate. Decision rule, gate thresholds
and the ≥ 3-failure INCOMPLETE clause are unchanged.
