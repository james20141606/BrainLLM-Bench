# PREREG — paperA (#60477): are the Brennan numbers driven by unfiltered, un-baselined EEG? (K0, K1, K2)

Written 2026-10-09 ~17:40 Beijing, **before any code for it exists and before any baseline-corrected or
detrended Brennan pattern has been computed**. Commit time is the evidence; deviations go under a dated
heading at the end. Experiment side only: `paperA_benchmark.tex` is not edited (R1 writer integrates later);
paperB files are not touched; 0 GPU; no copy of the 7 GB epoch file (in-memory switch only).

Trigger: `<coordination-notes>/next_cycle_20260927/bootloops_audit_20261003/crosscheck_brennan_preproc_paperA.md`
(<coordination-notes> `66c9532`). Every paperA Brennan number from the 47-listener v2 rebuild reads
`data/brennan_p4_v2/epochs_data.npy` (`build_brennan_epochs_v2.py`, JID 14377998): raw FieldTrip slices, no
filter, no baseline, no re-reference; `p4_concordance32.neural_patterns` applies no filter for "broadband".
Correlation distance cancels a per-trial scalar offset, not a per-channel drift vector. LibriBrain and
MEG-MASC epochs are baseline-corrected (cluster `metadata.json`: `baseline_corrected: true`).

Already known (not results of this registration): S01's `.mat` is unfiltered, 83% of its power is below
0.5 Hz (rebuttal-checks prereg, Deviation 3). paperB's pooled BERT L12 contrast went +4.47 → −0.18 with a
baseline (paperB, not paperA). Published Brennan values: Table 1 +0.366, Fisher [+0.02, +0.63] (space-fix
receipt; 29 checkpoints' features identical to `concordance32_brennan_v2.json`, 3 re-extracted); family CI
[−0.20, +0.69]; RDM reliability 0.858 [0.33, 0.997]; class gap −0.381 (family p 0.525).

## 0. Substrate and the three treatments (shared by K0, K1, K2)

Rows: `p4_concordance32.subsample(meta, 2500, 42, "subject")` (all 2,129 words × 47 listeners = 98,497),
features via `emb_row_idx.npy`. Window 300–400 ms (samples 200–250 at tmin −0.1 s, 500 Hz), all 60 channels.
Treatments applied in memory to each listener's epochs `ep[rows]` (float32 → float64 for b/c):
- **a raw**: exactly `neural_patterns` (gate: identical array);
- **b baseline**: per epoch and channel, subtract the mean of −0.1–0 s (samples 0–49);
- **c detrend**: per epoch and channel, remove a least-squares line over the whole epoch (−0.1 to +0.5 s,
  `scipy.signal.detrend(type="linear")`).
Known limits, stated now: b's pre-stimulus window contains the previous word's response (words ~0.3 s apart);
c's line is fit over the response window itself. Neither is clean; that is why both are run and the
decision needs both.

Features: the cached `llm_embeddings_brennan_fd` middle layer for all 32 checkpoints. For 29 these are the
features behind Table 1; for Llama-3.2-1B, Qwen2.5-0.5B and Qwen2.5-1.5B the re-extracted Brennan features
were deleted after scoring (`~/p4_spacefix_20260925/feat/brennan/` empty), so the cached (pre-fix) ones are
used and the raw comparator is `concordance32_brennan_v2.json` (published comparison value +0.342).

## 1. K0 — neural only (no LLM features except the D^m line)

Per listener and treatment:
- K0.1 share of pattern variance carried by the channel window means:
  Σ_c 50·(m_c − x̄)² / Σ_{c,t} (x_ct − x̄)², averaged over trials (x̄ = the trial's grand mean).
- K0.2 Spearman(D^b, |t_i − t_j|), D^b = correlation-distance RDM of the 3,000-d pattern (as the pipeline),
  t = recording time of word onset (`.vmrk` segment marker + CSV onset, the v2 builder's rule); and
  Spearman(D^b, same-sentence indicator) (CSV `Sentence`).
- K0.3 `noise_regime_sweep.rdm_reliability` verbatim (channel halves, 20 iterations, seed 0). Gate: under a
  it reproduces 0.8582017 to 1e-6.
- Once, not per listener: for each of the 32 models, Spearman(D^m, |Δt|) and Spearman(D^m, same-sentence)
  on the 2,129 words (D^m = cosine, as the pipeline; times from S01).

K0 reading (descriptive; does not gate K1/K2):
- **drift mechanism supported**: under a, median K0.1 ≥ 0.5 and median Spearman(D^b, |Δt|) ≥ 0.05 with ≥ 40
  of 47 listeners positive, **and** under both b and c the median Spearman(D^b, |Δt|) falls by ≥ 50%.
- **not supported**: under a, median Spearman(D^b, |Δt|) < 0.02 or median K0.1 < 0.2.
- **partial**: otherwise.
- RDM reliability: the b value (c beside it) is the number the revision uses in place of 0.858; F4's
  conclusion ("does not track reliability monotonically") is re-read with it, and the "range" sentence is
  re-stated.

## 2. K1 — similarity readout only, 32 models × 47 listeners × {a, b, c}, one <cpu-partition> job

`rsa_per_session` logic verbatim (C1–C3 partial Spearman; confounds built exactly as `p4_concordance32.main`;
neural ranks cached per listener and treatment, which changes nothing numerically).
- Gate G1: rows and metadata as above; `emb_row_idx` length = metadata length.
- Gate G2: under a, every per-listener RSA equals `concordance32_brennan_v2.json` to |Δ| ≤ 1e-8 for all 32
  models. ≥ 3 models failing → INCOMPLETE.
Quantities (RSA_x = 47-listener mean per model under treatment x):
- S_b = Spearman(RSA_a, RSA_b), S_c = Spearman(RSA_a, RSA_c) over 32 models;
- class means of RSA (causal / masked / enc–dec) under a, b, c;
- hybrid concordance κ_x = Spearman(RSA_x, ENC_published) (ENC from `concordance32_brennan_v2.json`), with
  the checkpoint Fisher CI and `recompute_corrected_stats.family_clustered_ci` (seed 0; seeds 0–19 reported);
  CI category ∈ {excludes 0 above, includes 0, excludes 0 below} for each interval;
- class gap (ENC-rank − RSA-rank, causal minus other) and its family-permutation p, by
  `concordance_decomposition.py` on a per-model file built from RSA_x and ENC_published.
Encoding is not recomputed in K1 (the target also changes under b/c; that is K2).

**K1 decision rule (fixed now):**
- **unaffected**: S_b ≥ 0.9 **and** S_c ≥ 0.9 **and** for both b and c the Fisher and family CI categories of
  κ_x equal those of κ_a **and** the class-gap sign is unchanged.
- **affected**: anything else. → K2 runs (§3), within this registration.
The decision is taken on the three treatments together; no treatment is dropped after the fact.

## 3. K2 — registered now, run only if K1 = affected

Full concordance (RSA and encoding) under b (primary) and c (secondary), 32 models, Brennan, via
`p4_concordance32.py` with the same in-memory switch, sharded 4-way per treatment on <cpu-partition> (as
`run_brennan_conc_v2_array.sbatch`). The encoding target Y is the window mean of the treated pattern. Then,
receipt-only, on the new per-listener scores: #2 family CI (paper function, seeds 0–19), #3
`rank_identifiability` (unit-half τ, spread), #7 `concordance_decomposition`; and with the switch: #4 the
8-segment `crossed_narrative.py` run, #6 `repeat_removal_sweep.py` k = 1. #5 is K0.3.
K2 reading for Table 1's Brennan row (on b; c beside it): "positive" if the family CI lower bound > 0;
"not determined" if it includes 0; "negative" if the upper bound < 0. Each derived number (#3, #4, #6, #7) is
reported as recomputed, with its old value beside it.

## 4. Wording (R1 writer and rebuttal kit)

Regardless of outcome (description errors, recorded now):
- §5.3 "Under the identical protocol (same band, window, depth and controls; …)" is false for Brennan:
  "Brennan epochs, unlike LibriBrain and MEG-MASC, are neither filtered nor baseline-corrected (raw FieldTrip
  slices)."
- App. Protocol "each epoch is filtered into six bands … broadband 1–80 Hz" does not hold for the
  fixed-depth concordance route: that route applies no filter for broadband on any cohort, and
  `compute_rdms.FREQ_BANDS["broadband"]` is (1, 150). State which route the band table describes.

Per branch:
- **K1 unaffected**: "With a per-epoch baseline or a linear detrend the Brennan similarity ranking of the 32
  checkpoints correlates ρ = [S_b] / [S_c] with the published one, and the concordance's interval category
  is unchanged; the missing baseline does not drive the Brennan row." Kit: answer on request; Brennan stays
  "positive point estimate, not determined under family-clustered inference" (W13).
- **K1 affected → K2**: "Without baseline correction the Brennan similarity ranking is not stable (ρ = [S_b] /
  [S_c] against the published one). With a baseline the Brennan concordance is ρ = [ ] (family CI [ ]; branch
  [positive / not determined / negative]); the derived Brennan numbers change to [list]." Kit: a
  disclosure item next to W13, because the submitted PDF says "on Brennan the interval only just excludes
  zero"; the author chooses first-round vs on-request (as for readrule). The headline (LibriBrain) and
  MEG-MASC are not on this exposure surface; "the reversal belongs to one cohort" then rests on MEG-MASC alone.
- K0 per §1 reading, one sentence each in App. (reliability; mechanism evidence).

## 5. Compute

K0 and K1 in one <cpu-partition> job (32 CPU, 96 GB, ≤ 6 h), receipts to `$HOME/p4_rebuttal_20261005/brennan_preproc/`,
then `results/paperA_rebuttal_20261004/brennan_preproc_raw/`. K2 only on "affected". 0 GPU.

## Deviations

- 2026-10-09 17:20: clerical — the header says "~17:40"; the file was written 17:05–17:14 and committed at 17:14:15
  (`c78bf4d3`). No content change.
- 2026-10-09 17:20 (before any run): `recompute_corrected_stats.py` and `model_families.py` are not on the cluster, so
  the K1 aggregation (κ_x, family CI, class gap, decision) runs locally on the per-listener receipt with the repo's
  production functions; the cluster job only produces per-listener values. No change to any estimand or rule.
- 2026-10-09 19:15 (after K1 = affected, before any K2 run): K2 implementation choices not fixed by §3, fixed now.
  (i) The published scripts run unchanged through `codes/p4a_k2_switch_20261009.py`, which replaces
  `neural_patterns` / `load_patterns_chunked` with the K1 treatment and `load_models` with the 32-model Brennan
  list. (ii) #4 (8 segments) and #6 (repeat removal) run under b only, the primary treatment; c is run for the
  full concordance only. #6 runs k = 1 with draw seeds 0, 1, 2 (the published grid's three seeds); all three are
  reported. (iii) Gate: K2's per-listener RSA under b and c must equal K1's to |Δ| ≤ 1e-8 (two independent code
  paths onto the same treatment). (iv) #3 is `rank_identifiability`'s unit-half statistics recomputed on the new
  per-listener scores.
- 2026-10-10 01:05 (operational, no analysis change): two K2 concordance shards hit the 5:30 <cpu-partition> limit
  (19480792_0 = b shard 0, 19480792_6 = c shard 2; `partial_out` is written only at the end, so their 16 model
  results were lost). They are re-run one model per task with the published script's `--only` flag, same switch
  and arguments (`run_p4a_k2_conc_only.sbatch`); the aggregator's glob is widened to pick up those files. Nothing
  from the completed shards has been read into any statistic yet (8-segment and repeat receipts not opened).
- 2026-10-10 01:10 correction to the note above: one log line of the repeat-removal k = 1 seed 0 run was seen
  on 10-09 while polling (`[cv] k=1 concordance = +0.2053 … RSA ranking split-half = -0.445`). No other K2 value.
