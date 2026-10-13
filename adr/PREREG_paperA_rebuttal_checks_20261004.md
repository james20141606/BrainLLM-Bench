# PREREG — paperA (#60477) rebuttal checks: session-9 target scale, per-listener Brennan alignment, C4 seed disclosure

Written 2026-10-04 03:40–03:55 Beijing, **before either check has been run**. the cluster is down (no ssh), so no
neural array has been opened for this file and none can be until it is back. The commit time of this
file is the evidence. After results exist this file is not edited; deviations go under a dated
"Deviations" heading at the end and into the receipt.

Scope: rebuttal evidence only. The submitted PDF (MEG `6f43d2ac`, sha256 `b6290676b29f…`) is frozen until
reviews; nothing here edits `paperA_benchmark.tex`. paperB files are not touched; where an outcome bears on
paperB it is flagged for the paperB owner (§2.7).

Source of the two questions: BootLoops audit 2026-10-04,
`<coordination-notes>/next_cycle_20260927/bootloops_audit_20261003/paperA.md` findings S5 / M2 (session 9) and S4
(Brennan v2 alignment verified on S01 only); seed sweep M1.

Cost: (i) and (ii) are CPU-minute diagnostics (no model features, no GPU). The (i) follow-up re-score, run
only under the branches named below, is a CPU array of the alpha32 kind (32 tasks, ~5 min each, cached
features). 0 GPU-h throughout.

---

## 1. Session 9: is it a unit-scale defect of the encoding target?

### 1.0 Already known before this registration (from released receipts; read only)

Recorded by `results/paperA_rebuttal_20261004/recompute_paperA_rebuttal_20261004.py`, part E (receipt
`paperA_rebuttal_recompute_20261004.json`), and the alpha32 receipts:

- `concordance32_CORRECTED.json` `per_session_enc`: in session 9 all 32 checkpoints lie in
  [−0.0018, +0.0046]; the 32-model mean of every other session lies in [0.023, 0.061] (session 3 is the
  low one, 0.023). Session-9 mean 0.0019.
- `per_session_rsa` in session 9 is ordinary: 32-model mean 0.0060 against 0.0047–0.0086 elsewhere.
- Per-session concordance: 11/12 sessions negative; session 9 is the only positive one (+0.033).
- alpha32 `orig` arm: 352/384 checkpoint-folds select the grid top 1e6; every checkpoint does so in exactly
  11/12 folds, and the one fold that selects an interior penalty is, for every checkpoint, the fold that
  holds session 9 out (i.e. the only training set without session 9).
- The encoding scorer (`codes/p4_concordance32.py:125-149`, repo copy; the <cluster> copy `5c6651e7` produced
  the receipt) divides Y by **one global std over all 30,000 rows** and then centres per training fold. It
  does not scale per session. RSA uses correlation distance per word pattern and is insensitive to a
  per-session scalar.
- Paper text: `:1605-1607` ("share one cause: code written as though its inputs were on a unit scale"),
  `:1629-1630` ("the harness does not yet include a test that rescales the neural data").

What is **not** known: the per-session scale of Y. Nobody has opened `prep_inputs.npz` for this question.

### 1.1 Question

Is the target Y of session 9 on a different scale from the other eleven sessions, such that one global
scaling leaves session 9 (or the other eleven) far from unit variance and the ridge penalty selected in
the eleven folds that contain session 9 is set by that scale?

### 1.2 Inputs (cluster)

- `<HPC_HOME>/p4_c4_20260924/prep/prep_inputs.npz`, SHA-256 `ea2dae192744f78d99f6a470012acf402f841ee8d77b21e96266d3250ff27f07`
  asserted (the array of the headline; C4/readrule/alpha32 used it).
- Rows and session labels from the shared subsample via `p4_c4_lib` loaders, G1 hashes rows
  `bfe5c58de6dc…`, metadata `b8e7784006b0…` asserted.
- Y = `pat.reshape(n, n_sens, n_t).mean(axis=2)` in float64, exactly `p4_concordance32.main:262`.

### 1.3 Quantities (diagnostic, CPU minutes)

- **D1 (decision-driving)** s_k = std of Y over the rows of session k, all sensors pooled, k = 1…12.
  r_k = s_k / median_{j≠k} s_j. The primary number is r_9; all twelve r_k are reported.
- D2 per-sensor std profile per session; Spearman between session 9's profile and the median profile of
  the other eleven (descriptive; a channel-type or unit change shows as a profile change).
- D3 per-session mean of Y, fraction of rows with any non-finite value, fraction of all-zero rows.
- D4 per-session std of Y after the global scaling the scorer applies (i.e. s_k / std(Y_all)), so the
  reader sees where each session sits relative to the unit variance the penalty grid assumes.

### 1.4 Plant (must pass before D1 on the real array is read)

On an in-memory copy of Y, multiply session 3's rows by 10. The diagnostic code must return r_3 = 10 ± 1%
(to the median-of-others definition), leave r_9 unchanged to 1e-12, and the rule of §1.5 must classify
the copy as branch U with session 3 named. A second copy with all rows multiplied by 1e-7 must return the
same r_k as the unmodified array to 1e-9 (the ratio is scale-free). If either fails, the diagnostic is
not read and the outcome is "INSTRUMENT FAIL".

### 1.5 Decision rule (on r_9 and the other r_k)

- **U — unit-scale defect specific to session 9**: r_9 > 3 or r_9 < 1/3, and every other r_k in [1/2, 2].
- **U\* — scale heterogeneity not specific to session 9**: any r_k (k ≠ 9) outside [1/2, 2]. All such
  sessions are named. (Takes precedence over U.)
- **Bd — borderline**: r_9 in (2, 3] or [1/3, 1/2), others in [1/2, 2].
- **N — not a scale effect**: r_9 in [1/2, 2] and others in [1/2, 2].
- **E — INSTRUMENT FAIL** per §1.4, or the npz hash does not match.

### 1.6 Follow-up (only under U, U\* or Bd; registered now, run after the branch is known)

Re-score the encoding readout of all 32 checkpoints with **per-session target scaling**: before the
leave-one-session-out loop, Y_k ← Y_k / s_k for every session k (each session's own scalar over all its
rows; a held-out session's per-column Pearson is invariant to its own scalar, so only training changes).
Everything else as `encoding_per_session` (grid logspace(1,6,8), train-fold standardisation, 1e-12 term).
Cached `llm_embeddings_fd` middle-layer features, same 30,000 rows.

- Gate G2: in the same job the unscaled arm must replay the published `per_session_enc` to |Δ| ≤ 1e-6 for
  every checkpoint; ≥ 3 failures → INCOMPLETE, no ρ reported.
- Primary: ρ_z = Spearman(published RSA, ENC_z) over 32; family CI by
  `recompute_corrected_stats.family_clustered_ci` (seed 0, 10,000); **plus seeds 0–19 reported** (lesson
  of C4, §3). Paired Δ vs −0.573 with the same family resample.
- Branch: **A** U < 0 (reversal does not depend on the target scaling); **B** ρ_z < 0 and L ≤ 0 ≤ U;
  **C** ρ_z ≥ 0. If the branch differs across seeds 0–19 the label is "seed-dependent", with the count.
- Reported with any branch: folds at 1e6 per checkpoint under ENC_z; session-9 ENC under ENC_z.
- Under N the follow-up is **not** run (no defect to repair); it is not run "to see".

### 1.7 Wording (filled mechanically)

Kit item 8 (now, all branches) states the fact only: "On the original grid the top penalty (10^6) is
selected in 352 of the 384 checkpoint-folds, 11 of 12 for every checkpoint; the exception is always the fold
that holds session 9 out." On request only (before §1 has a branch): "In session 9 every checkpoint's encoding
score is near zero (−0.002 to +0.005) while its similarity scores are ordinary; we have not yet diagnosed
why." Until §1 has a branch the kit makes **no** statement that encoding is regularisation-limited
"only at 0.5B", and does not use Proposition 1(iii) to explain the reversal.

- **N**: "Session 9's target is on the same scale as the others (ratio [r_9]); the near-zero encoding in
  that session is not a units problem, and its cause is not identified."
- **U / U\* / Bd, then A**: "Session [9 / list]'s encoding target differs in scale ([r_k]×) and the target
  was scaled once across sessions, so [its rows set] the penalty chosen in the folds that contain it. With
  each session scaled separately the concordance is ρ = [ρ_z], family-clustered CI [L, U] ([seed count]);
  the reversal does not depend on it." Revision: App L and the §5.5 / App N failure-mode list gain this
  as a further instance of the unit-scale class the paper already names.
- **… then B or C**: same first sentence, then "…gives ρ = [ρ_z] ([L, U]): the headline's encoding arm
  depends on a target-scaling defect in [session list]." This becomes a proactive-disclosure item of the
  same rank as readrule (kit §0), and the revision changes §5.3 and the abstract (hand to the author).
- **E**: "The per-session scale check did not pass its plant; session 9 remains undiagnosed."

---

## 2. Brennan v2 (47 listeners): per-listener alignment of `.mat` signal to `.vmrk` markers

### 2.0 Already known

- Every Brennan number in the paperA main text and concordance appendix uses the 47-listener v2 epochs
  (`codes/build_brennan_epochs_v2.py`, `data/brennan_p4_v2`, JID 14377998, ELN 2026-07-20T03:16) with the
  `llm_embeddings_brennan_fd` features.
- v2 combines the V1 Deep Blue `.mat` (preprocessed signal, no triggers) with the V2 Deep Blue `.vmrk`
  (segment markers of the raw BrainVision recording). That this pairing shares a time zero was verified on
  S01 only: per-segment word-locked ERP ratio 1.9–5.2× random (ELN 2026-07-18T11:40, commit `ceca1ccd`);
  whole-story ratio 1.14–1.22 (segments of opposite polarity cancel).
- 38 listeners have 12 segment markers, 9 have 11 (segment 1 missing: S02 S24 S26 S27 S29 S30 S34 S35 S49);
  v2 maps segments by label. S32, S36 are absent (failed downloads).
- The archive (49 listeners, `codes/preprocess_brennan.py`) reads raw BrainVision and maps segment onsets
  by label (`:126-138`); the v1 builder dropped every listener with < 12 markers (`build_brennan_epochs.py:25`).
  The archive and the rebuild disagree per listener (Spearman ≤ 0.18; paperB prereg 2026-10-04 (a)).
  Neither side's alignment has been checked per listener. This check covers the v2 side only.

### 2.1 Question

For each of the 47 listeners, are the v2 word onsets (segment marker + CSV onset × 500) locked to the
evoked response in the `.mat` signal, to within 100 ms?

### 2.2 Inputs (cluster)

`data/brennan_raw/S??.mat`, `data/brennan_vmrk/S??.vmrk`, `data/brennan_meta/AliceChapterOne-EEG.csv`;
segment onsets from `build_brennan_epochs_v2.seg_onsets` imported unchanged; 60 electrodes as the builder.
No epochs file is rebuilt and nothing is written to /scratch; per-listener results to `$HOME` as JSON.

### 2.3 Statistic (per listener × segment)

For lag τ ∈ {−1.00, −0.99, …, +1.00} s (201 lags): onsets t_w + τ for every word of the segment; epochs
[−0.1, +0.5] s, baseline (−0.1–0 s) subtracted; average over words; GFP(t) = std over the 60 electrodes;
E(τ) = mean GFP over 0.08–0.30 s. Peak lag τ* = argmax E; peak ratio q = E(τ*) / median{E(τ): |τ| ≥ 0.5 s}.

- segment **aligned**: |τ*| ≤ 0.10 s and q ≥ 1.5;
- segment **displaced**: |τ*| > 0.10 s and q ≥ 1.5;
- segment **flat**: q < 1.5.

Listener: **aligned** if ≥ 9 of its segments are aligned and ≤ 1 is displaced; **misaligned** if ≥ 3
segments are displaced or the median τ* over its non-flat segments is outside ±0.10 s; otherwise
**indeterminate**. Also reported per listener (descriptive): median τ* (audio latency), ±1 s shifted E
relative to E(0).

### 2.4 Plants and controls (must pass before the 47 are read)

- **Positive control**: S01 (the verified listener) must come out "aligned". If it does not, the thresholds
  are mis-set for this SNR; no verdict is issued ("INSTRUMENT FAIL").
- **Plant 1**: S01 with segment 5's marker moved by +0.30 s in memory: segment 5's τ* must be −0.30 s ± 0.02 s
  (the response now precedes the shifted onsets) and it must be classified "displaced"; the other segments
  unchanged.
- **Plant 2**: S01 with every marker moved by −0.25 s: τ* = +0.25 s ± 0.02 s in ≥ 9 segments and the listener
  classified "misaligned".
- If a plant is not recovered: "INSTRUMENT FAIL", and the paper's statement stays "alignment verified on S01".

### 2.5 Decision rule (cohort)

- **ALIGNED**: 0 misaligned and ≥ 40 of 47 aligned (rest indeterminate, named).
- **FEW**: 1–4 misaligned, none of them ≥ 3 of the nine 11-segment listeners.
- **MANY**: ≥ 5 misaligned, or ≥ 3 of the nine 11-segment listeners misaligned.
- **WEAK**: 0 misaligned but > 7 indeterminate: the check lacks power at this SNR for those listeners.
- **E**: INSTRUMENT FAIL per §2.4.

### 2.6 Consequence and wording

- **ALIGNED**: kit and revision: "the `.mat`/`.vmrk` alignment of the 47-listener rebuild was checked per
  listener with a lag scan and planted offsets (aligned [n], indeterminate [m], misaligned 0)."
- **FEW**: the Brennan concordance is recomputed receipt-only from
  `space_fix_20260925/aggregate/concordance_spacefix_brennan_32.json` without the misaligned listeners
  (per-subject RSA/ENC are stored; production `family_clustered_ci`, seed 0, plus seeds 0–19); kit gives
  both; revision labels Brennan numbers "[k] listeners excluded for misalignment" and reruns the
  listener-half and 8-segment analyses on the remainder.
- **MANY**: every Brennan number in paperA (Table 1 row +0.366, listener-half τ, RDM reliability 0.858,
  repeat removal +0.140, per-band +0.052, 8-segment reliability, class gap) is labelled "alignment-dependent,
  not certified" in the kit; becomes a proactive-disclosure item; the revision either rebuilds after the
  builder is fixed or removes Brennan from the cross-cohort contrast (author decision).
- **WEAK / E**: "alignment verified on S01 only; a per-listener check had insufficient power / failed its
  plant", no change to the numbers.

### 2.7 Cross-paper, and the overlapping paperB registration

Every outcome is flagged to the paperB owner (its 47-listener rebuild uses the same v2 epochs); this file
does not edit paperB.

Found while writing this file: paperB amended its own registration at 03:17 today
(`PREREG_paperB_audit_followups_20261004.md` §a.6, before any run). Its **D2** registers a rebuild-alignment
check on the same 47 listeners with a different statistic: per listener × segment, the word-locked GFP peak
(50–250 ms) against the same statistic at onsets shifted by a random 0.5–1.5 s (seed 0, 200 shifts); pass if
the ratio is ≥ 1.2 in every cell. The two checks are complementary, not duplicates: D2 has a random-shift null
but no planted offset and no lag localisation; §2 here has plants, a positive control and a lag estimate.
Rule for the overlap, fixed now: both are run on the same data (one job if possible); each paper applies its
own decision rule; any listener that fails **either** rule is not called "aligned" in paperA, and any
disagreement between the two is reported in the paperA kit with both statistics. Neither registration is
edited to match the other.

---

## 3. C4 seed-sensitivity disclosure (registered sentence, no new run)

The C4 verdict ("not determined", Table 1 C4 row / Table 5 / App J) was registered with seed 0 of
`family_clustered_ci`. Re-running that function with seeds 0–19 (receipt
`results/paperA_rebuttal_20261004/paperA_rebuttal_recompute_20261004.json`, part D) moves the upper bound
of the family-clustered interval across zero: seed 0 reproduces the stored +0.00626 and is the largest of
the 20 upper bounds (range −0.0259 to +0.0063); 17 of 20 are below zero. The registered verdict stays; the sentence below is what
the kit says **only if** a reviewer reports a different seed or asks about Monte Carlo error; it is not
used to strengthen the claim:

> The registered seed gives the conservative end of the Monte Carlo spread: over seeds 0–19 of the same family-clustered bootstrap (10,000 draws each) the upper bound of the C4 interval ranges from −0.026 to +0.006 and is below zero for 17 of 20 seeds. We keep the registered verdict ("not determined") and do not claim the reversal under C4.

Future registrations in this repo state "verdict stable across seeds 0–19, else reported as
seed-dependent".

## Deviations

(none yet)

### 2026-10-05 04:25 Beijing — Deviation 1 (before any run of §1 or §2; no neural array opened yet)

Found while writing the §1 code, before it ran: the §1.4 plant specification is internally inconsistent,
and one of its conditions can fail on a correct instrument.

1. "classify the copy as branch U with session 3 named": branch U is defined for session 9 only (§1.5); a
   ×10 plant on session 3 is, by §1.5, branch **U\*** with session 3 named.
2. "leave r_9 unchanged to 1e-12": r_9's denominator is the median of the other eleven s_j, which includes
   session 3; moving s_3 by ×10 can move that median legitimately. The invariant that does hold is s_9
   (and every s_k, k ≠ 3) unchanged.
3. "r_3 = 10 ± 1%": r_3's denominator excludes session 3, so the planted ratio is exactly 10 × r_3 of the
   unmodified array; the condition is read as r_3(planted) / r_3(unmodified) = 10 ± 1%.

Replacement plant battery (all must pass before the real r_k are read; else branch E):
- P1 (real array, session 3 × 10): r_3(planted)/r_3(unmodified) ∈ [9.9, 10.1]; s_k for every k ≠ 3 unchanged
  to relative 1e-12.
- P2 (real array, all rows × 1e-7): every r_k unchanged to relative 1e-9.
- P3 (synthetic, the decision rule itself; 12 sessions × 2,500 rows × 30 columns, i.i.d. normal, seed 0):
  unmodified → N; session 3 × 10 → U\* naming exactly [3]; session 9 × 10 → U; session 9 × 2.5 → Bd.
The script prints no real r_k unless P1–P3 pass. §2's plants are unchanged.

### 2026-10-05 04:45 Beijing — Deviation 2 (§2 statistic; before any recording is read)

A synthetic plant run of the §2 code (`codes/p4a_brennan_align_20261005.py selftest`; noise plus an evoked kernel
at latency 0.10/0.15/0.20 s after every word onset, word spacing U(0.25, 0.45) or U(0.20, 0.60) s; no recording
touched) shows that the registered statistic is biased: with the per-epoch baseline (−0.1–0 s) subtracted, E(τ)
peaks where the previous response falls into the baseline window, so at **true** alignment τ* = +0.14 to +0.25 s and
every synthetic segment is labelled "displaced". The registered positive control (S01 "aligned") would then fail
for a reason that is the instrument's, not the data's. The planted shifts themselves are recovered exactly
(Δτ* = −0.30 in all six cases) under both statistics.

Changes (fixed now, before S01 or any listener is read):
1. Primary statistic: no per-epoch baseline; the word-averaged trace is demeaned per channel over the whole
   u-range [−1.1, +1.5] s, then GFP and E(τ) as registered. Synthetic: τ* at true alignment −0.06 to +0.08 s, all
   six "aligned", q 3.8–11.0.
2. Plants are judged **relative to the same segment's unplanted τ\***: plant 1 Δτ\*(segment 5) = −0.30 ± 0.02 and
   "displaced", other segments unchanged; plant 2 Δτ\* = +0.25 ± 0.02 in ≥ 9 segments and the listener
   "misaligned". (The registered absolute targets presuppose τ\* = 0 at truth, which no GFP-window statistic
   guarantees.)
3. The registered (baseline) statistic is still computed for S01 and every listener and reported beside the
   primary, labelled "registered statistic, biased on synthetic data"; it does not decide the branch.
Thresholds (|τ\*| ≤ 0.10 s, q ≥ 1.5, listener and cohort rules of §2.3/§2.5) are unchanged.

### 2026-10-05 05:05 Beijing — §2 outcome under Deviation 2, and Deviation 3 (stage 2; written after the S01 gate failed, before any of the 47 is read)

**Outcome as registered (with Deviation 2): branch E, INSTRUMENT FAIL.** JID 19173640 (print crash after the
receipt was written) and 19173831 (clean rerun, same result): S01 "indeterminate" (0 aligned, 2 displaced,
10 flat; q 1.16–2.0, τ* mostly at the ±1 s scan edge); plant 1 Δτ\* = −0.14 (not −0.30), segment 5 "flat";
plant 2 recovered in 1 of 12 segments. The registered (baseline) statistic on S01: "misaligned"
(5 displaced, 7 flat). None of the 47 was read. Per §2.6 the paper's statement stays "alignment verified on
S01 only".

Root cause (instrument, checked on S01 and on synthetic data only): the V1 `.mat` signal is unfiltered
(FieldTrip cfg of S01: hpfilter "no", lpfilter "no", demean "no", detrend "no"); 83% of S01's power is below
0.5 Hz. A word-average over ~180 words does not remove that drift, so E(τ) is dominated by the drift and peaks at
the scan edges. A synthetic random-walk drift added to the selftest reproduces the S01 failure exactly (q ≈ 1.3,
τ\* = −1.0, plants not recovered) and a 0.5 Hz high-pass removes it (`selftest_drift`: τ\* −0.11/−0.06/−0.02,
plants Δτ\* = −0.30 in all three).

**Deviation 3 (stage 2), fixed now.** Disclosure: written after S01 was seen under both statistics; the 47 have
not been read under any statistic.
1. Every listener's 60-channel signal is high-passed at 0.5 Hz (4th-order Butterworth, zero-phase `sosfiltfilt`)
   before the Deviation-2 statistic. Nothing else changes: lags, windows, thresholds (|τ\*| ≤ 0.10 s, q ≥ 1.5),
   listener and cohort rules, the S01 gate and the relative plant criteria.
2. The synthetic run shows τ\* can sit at −0.11 s at true alignment (one of three latencies), so the 0.10 s
   segment threshold is tight; it is **not** relaxed. The listener rule (≥ 9 aligned, ≤ 1 displaced) absorbs
   one such segment.
3. Added descriptive metadata per listener (not decision-driving): FieldTrip `cfg.trl`, `sampleinfo`, sample
   count and filter flags; "time zero = recording start" if trl = [1, N, 0] and sampleinfo starts at 1. A
   listener whose `.mat` does not start at the recording's first sample is named in the receipt.
4. **No further variants.** If the stage-2 gate fails, §2 ends as E and the kit says so.
