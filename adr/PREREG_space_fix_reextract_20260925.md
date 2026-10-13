# PREREG — re-extract the space-dropped Qwen/Llama checkpoints under the headline rule and re-decide every affected claim

Written 2026-09-25 (Beijing), **before any feature of the re-extraction exists and before any score
of it has been computed.** The commit time of this file is the evidence. After results exist this
file is not edited; deviations go to a dated "Deviations" section appended below a rule line and to
the ELN, with the reason.

Author of this file: the computation agent of the coordinating session (it writes code, sbatch,
receipts and ELN only; it does not edit the tex). Background: `<coordination-notes>/final_review_20260924/
paperA_extraction_check.md` (read-only probe of 2026-09-25 02:55) and `acceptance/paperA.md`.

## 1. Question

The documented extractor (`extract_llm_layers_multi.py`, ba9ff1f and later, and its cluster copy
of 2026-08-17) tokenizes causal prefixes as word lists (`is_split_into_words=True`,
`add_prefix_space=True`). Under transformers **4.57.1** (env `<env>`) this drops the space
before every non-initial word for the Qwen2.5/Qwen3 and Llama-3.x tokenizers, so those checkpoints
were scored on text such as `Intheyeareighteen…` (probe: 252/252 word boundaries lost).

**If each affected checkpoint is re-extracted with the rule that produced the 18 headline
autoregressive features (git ef17743: the prefix joined into one string with spaces, default
tokenizer), and everything downstream is held fixed, which of the paper's statements that rest on
these checkpoints still hold?**

## 2. The affected set (fixed now)

### 2.1 Correction to the extraction check, found before this file was written

The extraction check classified all 25 extension checkpoints as unspaced. Its probe ran under
transformers 4.57.1. The run logs show that the Qwen3.5 checkpoints and Qwen3.8-27B were extracted
in a different env, `<HPC_SCRATCH>/envs/<env2>` (transformers **5.15.0**, tokenizers 0.22.2):
sbatch `run_p4_extract_gen35_small.sbatch` (JID 15912520, tasks 0–4) and
`run_p4_extract_gen35_mid.sbatch` (JID 15912521, tasks 1–2) call that env's python, and every one of
those logs shows the transformers-5 weight loader ("Loading weights"), while the 4.57.1 runs show
"Loading checkpoint shards". The same tokenizer-only probe (identical code, `probe_v5.py`, no
weights, no scores), run on 2026-09-25 in `<env2>`, gives **L = 0** (0/128 and 0/124 word
boundaries lost; decode equals the text for 32/32 and 8/8 prefixes) for Qwen3.5-0.8B/2B/4B/9B-Base,
Qwen3.5-9B, Qwen3.5-27B and Qwen3.8-27B. Under 5.15.0 the documented rule gives those seven
checkpoints normal spaced text with a leading space on the first word (state **N′**, the same as the
GPT-2-type checkpoints on the other cohorts), not unspaced text (**U**).
Receipt: `~/p4_spacefix_20260925/probe/probe_v5_out.json` on <cluster> (copied into the receipt folder).

### 2.2 Enumerated set

State of the cached features: **U** = unspaced (affected), **N′** = spaced, first word with leading
space (not affected by the defect), **N** = ef17743 rule (spaced, as the text).

LibriBrain (`llm_embeddings_fd`, scored on the 30,000-word shared subsample):

| # | checkpoint | set | cached state | env of cached run |
|---|---|---|---|---|
| 1–5 | Qwen2.5-3B, -7B, -14B, -32B, -72B | 48 base; Qwen2.5 ladder | U | 4.57.1 |
| 6–7 | Llama-3.1-8B, Llama-3.1-70B | 48 base; twins | U | 4.57.1 |
| 8–12 | Qwen3-0.6B/1.7B/4B/8B/14B-Base | 48 base; Qwen3 ladder | U | 4.57.1 |
| 13–16 | Qwen3.5-0.8B/2B/4B/9B-Base | 48 base; Qwen3.5 ladder | **N′** | 5.15.0 |
| 17–18 | Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct | post-trained; twins | U | 4.57.1 |
| 19–20 | Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct | post-trained; twins | U | 4.57.1 |
| 21–22 | Qwen3-8B, Qwen3-32B | post-trained; twin (8B) | U | 4.57.1 |
| 23–25 | Qwen3.5-9B, Qwen3.5-27B, Qwen3.8-27B | post-trained; twin (9B) | **N′** | 5.15.0 |

So on LibriBrain **18 checkpoints are U (12 of the 16 added base, 6 of the 9 post-trained)** and 7
are N′. Qwen2.5-0.5B/1.5B and Llama-3.2-1B on LibriBrain are N (ef17743 cache) and are not
re-extracted there; they are used only for the identity gate (§4, gate I).

All 25 are re-extracted under the one rule (§3). For the 18 U checkpoints this is the correction.
For the 7 N′ checkpoints it only makes every autoregressive checkpoint on LibriBrain single-rule
(N); they are not affected by the defect, and if one of them is not re-extracted by the cut-off its
cached N′ features are used and flagged (they never carried the defect).

MEG-MASC merged-27 (`llm_embeddings_masc27_fd`, 67,500-word subsample, 27 subjects):

| checkpoint | used in | cached state |
|---|---|---|
| Qwen2.5-0.5B, -1.5B (remapped from `llm_embeddings_masc_fd`, 4.57.1) | ADR 0010 ladder; 32-checkpoint MASC concordance | U |
| Qwen2.5-3B, -7B, -14B, -32B, -72B (JIDs 16073629/30, 4.57.1) | ADR 0010 ladder | U |
| Llama-3.2-1B (remapped, 4.57.1) | 32-checkpoint MASC concordance | U |

Brennan EEG (`llm_embeddings_brennan_fd`, 47-subject v2 set via `emb_row_idx.npy`):
Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B — U (JID 14186392, 4.57.1).

ds003020 fMRI (`llm_embeddings_ds003020_fd`): Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B — U
(JID 15564016). **Not re-extracted in this window** (§8).

MEG-MASC 10-subject `llm_embeddings_masc_fd` and the crossed-narrative / per-story analyses built on
MASC features (Table 1 top panel, "each story alone"): same three small checkpoints U. **Not
re-scored in this window** (§8).

## 3. The corrected rule, and why this one

**Rule (verbatim from git ef17743, file `extract_llm_layers_multi_ef17743.py`, SHA-256
c9b736f703d479d861d44d28cb7d0c555928eddc72371918722cf06edb40a234, unedited):**
tokenizer `AutoTokenizer.from_pretrained(name)` with no extra arguments, `pad_token` fallback to
EOS/UNK; each causal item is `" ".join(sentence_words[:word_position+1])`;
`tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)`; read-out = hidden
state at `attention_mask.sum()-1` at layer index `int(0.5*top+0.5)`, `top = len(hidden_states)-1`.
The code path is the ef17743 `extract_one`, called unchanged. The only substitution is its module
function `_load`, for checkpoints ≥ 3B and all extension checkpoints: the tokenizer lines of
ef17743's `_load` are kept verbatim; the model is loaded exactly as the cached extension run loaded it
(cluster extractor `--big` branch: `AutoModel.from_pretrained(name, output_hidden_states=True,
torch_dtype=torch.bfloat16, device_map="auto")`, `.eval()`), because the ef17743 `_load` (fp32 on
one device) cannot hold a 70B model and would change the numeric precision relative to the cached
extension features. Qwen2.5-0.5B/1.5B and Llama-3.2-1B use ef17743 `extract_one` including its
`_load` (fp32, batch 32), exactly as their LibriBrain headline features were made. Each checkpoint
runs in the env its cached features were made in (`<env>` 4.57.1; `<env2>` 5.15.0 for the
Qwen3.5 series and Qwen3.8-27B, which 4.57.1 cannot load). Batch sizes follow the cached runs
(16 for ≤ 14B, 8 for 27–32B, 4 for 70–72B; 32 for the three small checkpoints).

**Why not a patched documented rule** (word lists with the spaces restored, e.g. `add_prefix_space`
applied per word or spaces prepended to non-initial words):
1. The ef17743 rule is the rule behind every autoregressive number of the 32-checkpoint headline
   (−0.573) and behind the Qwen2.5-0.5B/1.5B rungs of the LibriBrain ladder. Using it makes the 48-
   and 57-checkpoint sets and the Qwen2.5 ladder single-rule on LibriBrain. A patched word-list rule
   would still give N′ (a leading space on the first word) for byte-level tokenizers, so the ladder
   would stay mixed (N at 0.5B/1.5B, N′ above).
2. A patched rule is a third tokenization that never produced any published number; it would need
   its own validation. The ef17743 path has one: re-running it reproduced the cached headline causal
   rows at median row cosine > 1 − 1e-10 (C4 control, gate G3, 17/17 models).
3. The C4 control already showed the word-list rule and the joined-string rule are collinear with
   objective class on LibriBrain; adding a third rule adds a third level to that collinearity.

## 4. Gates (checked before any re-extracted score is read)

- **G1 rows.** LibriBrain: SHA-256 of the int64 row-index array of the shared subsample =
  `bfe5c58de6dc807aae86ad2b4b2a59d8a7c96e57f10490ab000e2c2fcdb95346`, metadata SHA-256 =
  `b8e7784006b049060cb0383b5f4ad4a191674e540ed571e0aa831c1d6e7e843e` (same as C4). MASC and Brennan:
  the row hash computed by the same `p4_concordance32.subsample(meta, 2500, 42, group_col)` call is
  recorded in the prep receipt and asserted in every extraction and scoring receipt of that cohort.
  The context list is built with the builder the cached run used (`build_sentence_contexts_from_
  trial_metadata` for LibriBrain; `build_contexts_from_columns` on `meg_masc_merged27` and on
  `brennan_p4` for MASC/Brennan), its length must equal the cached feature file's row count, and the
  extracted rows are `contexts[emb_rows]` — the same row→context map the cached scorer read.
- **T1/T2 tokenization (per checkpoint × cohort, on every extracted row, before extraction).**
  T1: `decode(real ids, skip_special_tokens=True).strip() == " ".join(prefix)` for ≥ 99.9% of rows,
  and **zero** rows whose decode equals the text after deleting all whitespace but not with it
  (space loss). T2: the read-out position is the last non-pad token and is not a special token, 100%
  of rows. Mismatching rows are listed (up to 20) in the receipt. A checkpoint failing T1/T2 is not
  scored.
- **W weights/pipeline identity (extension checkpoints).** In the same job, with the same loaded
  model, the documented rule (cluster `extract_one`, same env, same batch size) is re-run on the
  first 512 extracted rows and compared with the cached rows. Pass: median row cosine ≥ 0.999 and
  ≥ 99% of rows ≥ 0.99 (bf16 across a different batch composition / GPU split is not bitwise).
  A W failure means the loaded weights, revision or layer differ from the cached run; that
  checkpoint's re-extraction is **not used** and the checkpoint counts as not re-extracted (§7).
- **I identity (smoke, before any extension job).** Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B on
  LibriBrain, all 177,127 contexts in the original batch composition (batch 32, fp32, <gpu> as the
  original array 14100743/14103660): the new code path must reproduce the cached `llm_embeddings_fd`
  0.5-depth file. Reported: fraction of rows bitwise equal, max |Δ|, median/min row cosine. Pass
  criterion (the C4 G3 threshold): median row cosine ≥ 0.99999 and ≥ 99.9% rows ≥ 0.9999; bitwise
  equality is reported, not required (library/driver state since July is not pinned). Also the
  layer index written must equal the cached middle layer file. If gate I fails for any of the three,
  nothing is submitted beyond the smoke; the cause is found first.
- **G2 scorer identity.** The new scorer, run on the **cached** features, must reproduce the
  published per-session values of every re-extracted checkpoint: `concordance_modern_final_v2.json`
  (LibriBrain, 25 checkpoints), `concordance_masc27_ladder.json` (7), `concordance32_masc27.json`
  (Llama-3.2-1B), `concordance32_brennan_v2.json` (3): RSA |Δ| ≤ 1e-8 and encoding |Δ| ≤ 1e-6 per
  session; and for Qwen2.5-3B…72B the expanded-grid alpha replay must reproduce
  `alpha_sensitivity_20260922/qwen25_{2..6}_full.json` fold scores to |Δ| ≤ 1e-6. A cohort whose G2
  fails is not scored until the scorer is fixed.
- **G5 completeness.** Every new feature matrix has the expected rows × dim, is finite, and has no
  all-zero row.
- **D1 descriptive (not a pass/fail gate for scoring).** Median row cosine between new (N) and
  cached rows. Expected < 0.99 for U checkpoints (0.786–0.897 for the three small ones), ≥ 0.99 for
  the seven N′ ones (the GPT-2-type N-vs-N′ range was 0.9965–0.9999). If an N′ checkpoint shows
  < 0.99, or a U checkpoint ≥ 0.999, the §2 classification of that checkpoint is reported as wrong.

## 5. Scoring: identical to the published analysis

Nothing downstream changes. Per cohort the scorer imports, from the cluster copy of
`p4_concordance32.py`, `subsample`, `neural_patterns`, `rsa_per_session`, `encoding_per_session`,
and the confound builders from `path1_confound_rsa.py`; cell, window, band, atlas, group column,
alphas (`logspace(1,6,8)`), seed 42, 2,500 words per group, LOGO folds are those of the published run
of that cohort (LibriBrain: temporal_L/neuromag306/session; MASC: temporal_R/kit208/subject;
Brennan: all/eeg60/subject, tmin −0.1, sfreq 500, `emb_row_idx.npy`). The orthographic confound is
computed once per group with the same `ortho_rdm` and stored as uint8 (asserted integer ≤ 255, as in
C4). The alpha-grid sensitivity uses `alpha_sensitivity_20260922.fit_one` with its `ORIGINAL` and
`EXPANDED` grids unchanged. The 32 frozen checkpoints and all non-affected rows keep their published
per-session values; only the re-extracted checkpoints' per-session values are replaced.

Aggregation uses the registered analysis scripts **unchanged** (`codes/adr0006_registered_analysis.py`
SHA-256 recorded; `codes/adr0010_q1_masc_analysis.py`), run with their `--json/--out` arguments on a
concordance JSON in the published format (`per_session_rsa`, `per_session_enc`), and
`p4_concordance32`'s own reliability / Fisher-CI code for the 57-checkpoint exploratory concordance
and the MASC-32 / Brennan-32 concordances. Family map: `codes/model_families.py` (12 families), family
bootstrap seed and draws as in the script (seed 6, 10,000).

## 6. Estimands and decision rules (applied mechanically; primes denote the re-extracted values)

For each claim: the statistic, the rule, and the sentence the paper should carry in each outcome
(the writing agent chooses wording within the sentence; numbers in brackets are filled from the
summary receipt). "Published" values are from the receipts named.

### 6.1 Q2 — the 48-base readout reversal (abstract "survives the 48 base checkpoints…", §1 L86, Table 1 row "48 base", Fig. concordance32 48-point, §5.3 older-model falsifier)
Statistic: ADR 0006 Q2 on the 48 base checkpoints (32 frozen + 16 added base), ρ′₄₈ and its
family-clustered 95% CI [L, U] (published ρ = −0.582, CI [−0.781, −0.139]). F1 = ρ′>0 and L>0.
- **Holds** (U < 0, registered "REVERSAL SURVIVES"): "The reversal survives the 48 base checkpoints of
  the registered test after re-extracting the 12 space-dropped ones with the headline rule
  (ρ = [ρ′₄₈], family-clustered CI [L, U])." §5.3: the older-model falsifier did not fire.
- **Weakens** (ρ′₄₈ < 0, U ≥ 0, registered "ATTENUATES"): "On the 48 base checkpoints, with the
  space-dropped ones re-extracted, the reversal attenuates (ρ = [ρ′₄₈], CI [L, U] includes zero)."
  The abstract/§1 "survives the 48 base checkpoints" clause is removed; §5.3 says the falsifier did
  not fire but the reversal is not determined on the extended set.
- **Flips** (ρ′₄₈ ≥ 0, or F1 fired): "With the space-dropped checkpoints re-extracted, the reversal
  does not extend to the 48 base checkpoints (ρ = [ρ′₄₈], CI [L, U])"; if F1 fired, ADR 0006
  requires the abstract to state that the reversal is an artifact of the 2019-era model set.

### 6.2 Q1 — scaling per generation (abstract "or across three Qwen generations (up to 72B)", §5.4 "size does not positively order alignment under both readouts in any generation")
Statistic: ADR 0006 Q1 verdict per ladder (Qwen2.5 7 rungs, Qwen3 5, Qwen3.5 4) on the
re-extracted rungs. Published: Qwen2.5 NOT SUPPORTED, Qwen3 MIXED, Qwen3.5 NOT SUPPORTED.
- **Holds**: no ladder SUPPORTED and each ladder's verdict label unchanged. Sentence unchanged, with
  "(re-extracted with spaces)" in the appendix.
- **Weakens**: no ladder SUPPORTED but ≥ 1 label changes (e.g. NOT SUPPORTED ↔ MIXED): the main
  sentence stays; the appendix gives the new per-generation labels and CIs.
- **Flips**: any ladder SUPPORTED: "Size orders alignment under both readouts on the [generation]
  ladder once spaces are restored (ρ_RSA = […], ρ_enc = […])"; the abstract clause "or across three
  Qwen generations" is removed.

### 6.3 Readout-dependent ordering on the Qwen2.5 ladder (§5.4 "encoding falls with size (72B vs 0.5B, paired p=0.0005) while similarity rises (p=0.002)"; Antonello rows)
Statistics: Qwen2.5 Q1 per readout: ρ′ with session-clustered CI, 72B − 0.5B paired mean and
Wilcoxon p (published enc ρ −0.893 [−0.964, −0.821], Δ −0.0261, p 0.0005; RSA ρ +0.607
[+0.429, +0.821], Δ +0.0029, p 0.0024). Note 72B and 0.5B are both N after the fix.
- Encoding "falls": **holds** if ρ′_enc < 0 with CI upper < 0 and Δ′ < 0 with p < 0.05; **flips** if
  ρ′_enc ≥ 0 and Δ′ ≥ 0; **weakens** otherwise.
- Similarity "rises": **holds** if ρ′_RSA > 0 with CI lower > 0 and Δ′ > 0 with p < 0.05; **flips** if
  ρ′_RSA ≤ 0 and Δ′ ≤ 0; **weakens** otherwise.
- The sentence "the ordering is itself readout-dependent" is kept only if both hold. Sentences:
  holds → unchanged numbers replaced by [ρ′, p′]; weakens → "encoding [falls/does not rise] (ρ = […],
  CI […]) and similarity [rises/does not fall] (…); the 72B-versus-0.5B contrast is [Δ′, p′]";
  flips → "with spaces restored, encoding [rises] / similarity [falls] with size (…)", and the
  "readout-dependent" sentence is removed.

### 6.4 Alpha-grid sensitivity (post hoc; tab:alpha-grid, "attenuates the encoding size correlation from −0.893 to −0.464… 72B stays below 0.5B")
Statistic: expanded-grid means for the 7 rungs (0.5B/1.5B from the published receipts, unchanged),
Spearman(size order, expanded means), and 72B − 0.5B under the original and the expanded grid
(published: −0.0261 original, 72B 0.04491 vs 0.5B 0.05550 expanded).
- **Holds**: 72B < 0.5B under both grids. Sentence "72B stays below 0.5B" kept with new numbers.
- **Flips**: 72B > 0.5B under both grids.
- **Weakens**: otherwise (one grid each way). The expanded-grid ρ is reported in every case. This
  remains a post hoc sensitivity analysis in every outcome.

### 6.5 MEG-MASC ladder (ADR 0010; §5.4 "leave encoding flat (ρ=+0.29, CI [−0.43,+0.64]) and the similarity arm fails its rankability gate")
Statistic: ADR 0010 Q1-MASC on the 7 re-extracted rungs, F3 gate per readout, cross-cohort statement
(run with the re-derived LibriBrain verdict file).
- **Holds**: encoding CI includes 0 and the RSA arm still fails F3 (and verdict not SUPPORTED).
- **Weakens**: not SUPPORTED, but encoding CI excludes 0 below zero (then "encoding falls" replaces
  "flat"), or the RSA arm now passes F3 (then its Q1 result is reported instead of "fails its gate").
- **Flips**: ADR 0010 verdict SUPPORTED for a readout that passes F3, or encoding CI excludes 0 above
  zero: "on MEG-MASC, size orders [readout] alignment once spaces are restored (…)".

### 6.6 Base–instruct twins (exploratory; "3/6, 4/6")
Statistic: ADR 0006 Q3 direction counts. **Holds**: neither readout 6/6 or 0/6 ("shifts neither
readout consistently"), counts reported as re-computed. **Flips**: a readout at 6/6 or 0/6
("instruction tuning [raises/lowers] [readout] in all six twins"). No "weakens" state exists for a
direction count; a changed count under "holds" is stated with the new numbers.

### 6.7 Antonello rows (table L1436 "not supported here", L1817 "contradict")
- **Holds** ("contradict" stays): 6.3 encoding does not rise on LibriBrain (ρ′_enc ≤ 0) and 6.5 MASC
  encoding not SUPPORTED.
- **Weakens** → "not supported": ρ′_enc > 0 with CI including 0 on LibriBrain, or MASC encoding CI
  excluding 0 below zero while LibriBrain rises non-significantly.
- **Flips** → "consistent on the Qwen2.5 LibriBrain encoding ladder; not on Pythia": LibriBrain
  Qwen2.5 encoding SUPPORTED (ρ′>0, CI lower > 0).
In every outcome the similarity-arm clause is re-filled from 6.3 and the Pythia clause is unchanged
(Pythia is N).

### 6.8 Cross-cohort concordance rows of Table 1 (MEG-MASC 27 listeners +0.489 [+0.17,+0.72]; Brennan 47 +0.342 [−0.01,+0.62])
Statistic: the 32-checkpoint concordance of each cohort with Qwen2.5-0.5B/1.5B and Llama-3.2-1B
replaced by their re-extracted scores, Fisher CI as published. (After the fix these three are N and
the other 15 autoregressive checkpoints on those cohorts stay N′; stated in the receipt.)
- **Holds**: sign unchanged and "CI excludes 0" status unchanged.
- **Weakens**: MASC CI now includes 0 (sign unchanged), or Brennan |ρ| drops by > 0.1 with the same sign.
- **Flips**: sign changes, or Brennan CI now excludes 0 on either side (then the row reports it).

### 6.9 Exploratory 57-checkpoint concordance (−0.655) — reported with the re-extracted 25; no rule.

### 6.10 Descriptive, no rule: new-vs-cached row cosines (D1), per-model ΔRSA and Δenc, class means.

## 7. Cut-off and incomplete checkpoints

- GPU extraction cut-off **2026-09-26 10:00 Beijing**; scoring cut-off 11:30; verdicts by 12:00.
- A U checkpoint not re-extracted and scored by the cut-off (queue, OOM, W/T/G5 failure) is listed
  as **not re-extracted**, and **every conclusion that uses it is withdrawn** — never backfilled with
  its unspaced features, never replaced by another checkpoint:
  - any of the 12 U base checkpoints missing → the 48-base test (6.1) is withdrawn; the statistic on
    the re-extracted subset may be reported in the receipt as exploratory only, not as the test;
  - a missing Qwen2.5 rung → per ADR 0006's fallback clause the ladder ends at the last contiguous
    re-extracted rung and the paper states the cap; every sentence naming 72B (6.3 contrast, 6.4)
    is withdrawn if 72B is missing; likewise on MASC (6.5);
  - a missing Qwen3 rung → that generation's Q1 is INDETERMINATE for the missing rung (ADR 0008);
  - a missing twin member → that pair is "not scored" and 6.6 counts over the scored pairs, stated.
- An N′ checkpoint not re-extracted by the cut-off keeps its cached N′ features (not affected by the
  defect), flagged "cached N′" in the receipt.
- If gate I or any cohort's G2 fails and is not fixed by the cut-off, the whole affected block is
  reported as not re-extracted.

## 8. Tiers, compute, storage

- Smoke: gate I (three small checkpoints, LibriBrain, <gpu>) and G2 on CPU. Nothing else is submitted
  before gate I passes.
- Tier 1 (≤ 14B; one <gpu>/H100/A100 each, several partitions submitted in parallel, the first to
  start is kept and the duplicates cancelled): Qwen2.5-3B/7B/14B/7B-Instruct, Llama-3.1-8B/-Instruct,
  Qwen3 ×5 base, Qwen3-8B, Qwen3.5 ×4 base, Qwen3.5-9B; plus the three small checkpoints on MASC and
  Brennan.
- Tier 2 (27–32B; 2 GPUs): Qwen2.5-32B, Qwen3-32B, Qwen3.5-27B, Qwen3.8-27B.
- Tier 3 (70–72B; 2 × 80 GB or 2 × H200): Qwen2.5-72B, Qwen2.5-72B-Instruct, Llama-3.1-70B,
  Llama-3.1-70B-Instruct. Weights not in the shared HF cache are downloaded to the job's /dev/shm
  (≈ 140 GB) and deleted at exit. Queue first checked with `sbatch --test-only`; if no start before the
  cut-off is possible, the tier is dropped and §7 applies.
- Qwen2.5 3B–72B extract LibriBrain and MASC rows in the same job (one weight load).
- Storage: weights and HF cache only in /dev/shm or read offline from the existing shared cache;
  nothing written to /scratch except logs < 1 MB; features (0.5-depth layer only, selected rows only)
  go to `$HOME/p4_spacefix_20260925/feat/`, are scored by a CPU job, and are deleted after scoring,
  their SHA-256 kept in the receipt. Shared conda envs are only read.
- GPU jobs do only tokenization gates, extraction and the W comparison; all scoring and aggregation
  run on CPU partitions (the registered aggregation scripts need seconds of CPU).
- Receipts: per-model JSON (model, HF revision, tokenizer.json SHA-256, extractor and script
  SHA-256s, transformers/tokenizers/torch versions, JID, host, GPU, gates, feature SHA-256, timings;
  host names are scrubbed by the supplement builder), per-cohort prep receipt, and the summary
  `results/p4_encoding/space_fix_20260925.json` with every recomputed statistic of §6 and its
  mechanical outcome.

## 9. Not done / not claimed

- ds003020 and the MEG-MASC per-story / listener-half / crossed-narrative rows (Table 1 top panel,
  "each story alone") are not re-scored in this window; the three small checkpoints in them stay U.
  The paper should disclose that 3 of 32 checkpoints in those rows were scored on unspaced text.
- The other autoregressive checkpoints on MASC/Brennan/ds003020 (N′) are not re-extracted.
- No change of cell, layer, confounds, alphas, subsample, or family map; no new checkpoint.
- The masked/encoder–decoder tokenizers are not re-probed here.
- Nothing here tests the geometry account or the class/rule collinearity of the 32-checkpoint
  headline, which is unaffected by the space defect (all 32 are N or N′ on LibriBrain).

## 10. Code (committed with or before the first receipt)

`codes/p4_spacefix_lib.py`, `codes/p4_spacefix_extract.py` (GPU), `codes/p4_spacefix_prep.py`,
`codes/p4_spacefix_score.py` (CPU), `codes/p4_spacefix_aggregate.py`; sbatch under
`codes/scripts/slurm/run_p4_spacefix_*.sbatch`. On <cluster> they run from `~/p4_spacefix_20260925/code/`
and import the frozen pipeline from `<PROJECT>/code` read-only.

---

## Deviations (appended 2026-09-25 evening, after results; the text above is unchanged)

Recorded here so that the pre-registration carries them, not only the lab notebook. None changes a
tokenization rule, a read-out position, a scoring step, a decision rule or an outcome; the evidence for
that is named with each item.

1. **Forward call under transformers 5 (7 checkpoints: Qwen3.5 x5, Qwen3.5-27B, Qwen3.8-27B).** §3 says
   the only substitution is the module function `_load`. Under transformers 5, `from_pretrained(...,
   output_hidden_states=True)` no longer makes hidden states a config default for these checkpoints, so
   ef17743's `model(**inputs).hidden_states` was `None`. For these 7 the cluster extractor's `_forward`
   (which passes `output_hidden_states=True` in the call, the forward their cached features were made
   with) replaced ef17743's forward. Tokenization and read-out are ef17743's. Evidence: gates T1/T2 on
   every row; gate W (the documented rule on the same loaded weights reproduces the cached rows, median
   row cos >= 0.99994). Code: extractor version SHA-256 efa9be79..., committed in 44701779.
2. **Subsample without scikit-learn.** The transformers-5 environment has no scikit-learn, so
   `p4_concordance32` (which imports it) cannot be imported there; `p4_spacefix_lib` falls back to a
   byte-for-byte copy of `p4_concordance32.subsample` and of `mid_layer_path` (421c63f3). Evidence: gate
   G1 asserts the published row-index SHA-256 in every extraction receipt.
3. **Feature storage in bf16.** After the $HOME quota incident (16:00), features of bf16-loaded models
   are stored as their upper 16 bits (lossless for bf16 values; the store asserts the low 16 bits are
   zero and a bit-exact round trip) and the scorer re-checks the float32 SHA-256 recorded at extraction
   (b1d1389d). Evidence: SHA-256 match on every scored feature file.
4. **6.5, encoding arm failing F3.** §6.5 did not enumerate the case in which the encoding arm fails its
   rankability gate after re-extraction. ADR 0010's own gate clause was applied: an arm that fails F3 is
   indeterminate and is not folded into either reading; no cross-cohort encoding statement is made
   (receipt `rule_note`). The similarity arm passing F3 is listed in §6.5 as "weakens" and was applied.
5. **Code committed after its receipts (§10).** The extractor version efa9be79 that produced the 7
   receipts of item 1 was committed at 20:25 (44701779), after those receipts existed; §10 required
   "committed with or before the first receipt". The receipts record the SHA-256, and the committed
   file matches it, so the provenance is verifiable, but the ordering rule was not met.
