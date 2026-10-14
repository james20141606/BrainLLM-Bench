# Verification guide

Detailed description of every offline check in this repository, what it pins and what it
does not establish. `bash verify_release.sh` runs all of them.

## Verify the registered stored-result summaries

From the repository root (Python 3 with numpy, scipy, pandas):

    python -m bench.reproduce

Expected final line: `N/N claims reproduce from results/` with N equal to the
registry size (every registered claim passes; N was 198 at build time). Each
claim recomputes a value from the stored result files and compares it with a
pinned value at the stated tolerance; the source file for each claim is listed
in the output. Not every claim is a number printed in the benchmark paper: the
line above the total gives the split (112 benchmark-paper claims, 33 superseded-value provenance pins, 53 companion boundary-analysis claims). Superseded-value pins
check pre-correction files that ship beside their `_CORRECTED` replacements
(each names the claim that pins the corrected value, see `SUPERSEDED` in
`bench/reproduce.py`); companion claims (`B.*`) belong to a separate
boundary-effect analysis on the same cohorts that is not part of this
paper.

Main-text Table 1 and the main-text numbers pinned with it:

    python codes/check_table1_receipts.py

Expected: `Table 1 receipt pin: PASS (0 failures)`.

The claim registry does **not** cover the alpha/rogue follow-up measurements.
Verify those separately, from the same repository root:

    python codes/check_followup_receipts_20260923.py

Expected: `NEW follow-up receipt pin PASS: 7x12 alpha, 72B same-alpha limit,
10x4 rogue, TeX tables/scope`. This rechecks seven models x twelve sessions
for alpha sensitivity and ten models x four doses for rogue encoding against
the stored per-model receipts and the read-only TeX snapshot. These
are post hoc sensitivity analyses, not new pre-registered claims. The TeX
snapshot is present only for numerical/scope audit and need not compile.

## Discussion-period revision (R1)

The revised manuscript adds results that were computed under registrations
committed before each run; two of them (reading rule, ridge range) finished
before the submission deadline but were not in the submitted PDF, and their
result directories were left out of the submitted supplement. Receipts:
`results/readrule_20260926/` (reading rule, branch B: rho = -0.016,
family-clustered 95% CI [-0.40, +0.34]), `results/alpha32_20260926/` (ridge
range, branch A: -0.657 [-0.80, -0.37]) and `results/paperA_rebuttal_20261004/`
(encoding-target scale per session, branch A: -0.543 [-0.71, -0.22]; rescaled
Qwen2.5 ladder on cached features, holds: -0.786; Brennan per-listener
alignment check, instrument failure E; independent re-computations and seed
sweeps). Registrations: `adr/PREREG_readrule_20260926.md`,
`adr/PREREG_alpha32_20260926.md`, `adr/PREREG_paperA_rebuttal_checks_20261004.md`,
`adr/PREREG_paperA_qwen_ladder_rescaled_20261005.md`, and for the Brennan
preprocessing check (per-epoch baseline, Appendix "Brennan preprocessing")
`adr/PREREG_paperA_brennan_preproc_20261009.md` with receipts
`results/paperA_rebuttal_20261004/brennan_{preproc,k2}_summary_20261009.json`
and `brennan_{preproc,k2}_raw/`. Run logs and scheduler listings are not in this
repository. Verify from the repository root:

    python codes/check_paperA_revision_r1_20261007.py

Expected: `PASS: paperA R1 pins N/N hold` (`--selftest` runs the mutation
test). The cohort family-clustered intervals are recomputed there with
`codes/recompute_corrected_stats.py`.

## Measurement scope

The tensor route uses C1-C4 controls; the fixed-depth RSA comparison uses
C1-C3. results/measurement_scope_20260922.json records that state as of
2026-09-22. Afterwards the static-embedding control C4 was added to the
32-checkpoint LibriBrain comparison under a pre-registration committed before
any C4 score existed (adr/PREREG_c4_static32_20260924.md, including its dated
deviations). Outcome under the registered rule: not determined (rho = -0.415,
family-clustered 95% CI [-0.73, 0.01]; C1-C3 value -0.573). Receipts:
results/p4_encoding/concordance32_c4.json and results/p4_encoding/c4_static32/.
Verify from the repository root:

    python codes/check_c4_receipt_20260924.py

Expected: `C4 receipt pin: PASS (... outcome B)`. C4 was not run for the
57-checkpoint set, MEG-MASC or Brennan. Tensor validation and the
stored-summary harness do not establish end-to-end reproduction of the
fixed-depth sweep.

## Feature-extraction provenance (fixed-depth sweep)

The cached depth-0.5 features behind the fixed-depth scores were not all
written by the extractor as it now stands in `codes/extract_llm_layers_multi.py`.
The 17 autoregressive checkpoints other than GPT-2 small were extracted with an
earlier revision of that script, shipped verbatim as
`codes/extract_llm_layers_multi_ef17743.py`; GPT-2 small comes from the older
`codes/extract_llm_layers.py`. Both tokenize the word prefix as one joined
string with the tokenizer's default settings. The current script passes a word
list (`is_split_into_words=True`, `add_prefix_space=True`), a change made to
fix target-word alignment for masked and encoder-decoder models; only those 14
checkpoints were re-extracted with it. Re-running the current script on the
30,000 scored words changes the autoregressive features: per checkpoint 36%
(GPT-Neo-125M) and 93-99.8% (the other 17) of rows fall below cosine 0.9999
with the cached rows; median row cosine is 0.786 (Llama-3.2-1B), 0.855 and
0.897 (Qwen2.5-0.5B, -1.5B) and 0.9965-0.9999 for the other 15, against
1.0000000 for all 14 masked and encoder-decoder checkpoints. The per-model
gate receipts with these statistics are
`results/p4_encoding/c4_static32/per_model/*_G3fail.json`. Reported numbers
of the 32-checkpoint sweep are computed from the cached features and are
unchanged; re-extracting the autoregressive features with the current script
would not reproduce the fixed-depth scores exactly.

## Tokenization defect and re-extraction

Under transformers 4.57.1 the current script's word-list rule drops the space
before every non-initial word for the Qwen and Llama tokenizers. 18 of the 25
extension checkpoints, the MEG-MASC Qwen2.5 ladder and three small checkpoints
on MEG-MASC, Brennan and ds003020 were first scored on such unspaced text. All
25 extension checkpoints, the MEG-MASC ladder and the three small checkpoints on
MEG-MASC and Brennan (36 extractions) were re-extracted with the joined-string
rule under a pre-registration committed before any re-extracted feature existed
(adr/PREREG_space_fix_reextract_20260925.md). Summary with every registered
outcome: results/p4_encoding/space_fix_20260925.json; per-model receipts:
results/p4_encoding/space_fix_20260925/. ds003020 and the MEG-MASC story
analyses were not re-run. Verify from the repository root:

    python codes/check_space_fix_20260926.py

Expected: `Space-fix pin: PASS (...)`. In the claim registry the re-extracted
values are the `SF.*` claims; every claim that reads a result file computed
from unspaced features is kept only as a superseded provenance pin that names
its `SF.*` sibling, and `bench/reproduce.py` fails if a claim reads such a file
without that mark (`SPACE_FIX_STALE_SOURCES`). Mutation test of those entries:

    python codes/check_bench_space_fix_mutations_20260925.py

Expected: `7/7 mutation cases behaved as expected`.

## Follow-up source provenance

The alpha and rogue jobs were executed from several immutable source versions,
not uniformly from the final working-tree files. Exact historical snapshots
recovered from Git objects are in `codes/frozen_followups_20260923/`, with a
receipt-to-raw-SHA map and a short account of the version differences. Alpha
actually executed raw hashes `bd323b71...` for 0.5B/1.5B and `bcec091e...`
for 3B through 72B. The 14B/72B receipts recorded `31593a6e...` at job end
because the shared disk file had gained a plateau-only extension after their
fit processes started; `summary.json` pins this provenance correction and says
the primary fit/scoring branch did not change. The separate 72B plateau run is
the verified execution of `31593a6e...`. Rogue dose receipts use `516a496a...` for model 00 and
`e3195cd5...` for models 01-09. Package redaction changes some source bytes;
`manifest.json` preserves the original SHA beside the redacted packaged SHA.
The top-level current scripts document the final workflow, while the frozen
snapshots establish which code actually produced each stored receipt.

The old invariance_intervention.json is preserved unchanged. Its encoding
ranking was frozen, so it cannot exclude geometry as a mechanism. The corrected
script labels this RSA-only sensitivity; a small synthetic counterexample can
be run without neural data:

    python codes/rogue_counterexample.py --out /tmp/rogue_check.json

The scorecard transport file reports setting changes and missing evidence,
not a literature replication rate. Historical labels remain for provenance.

## Contents

- `bench/` -- the harness. `bench/reproduce.py` is the claim registry; other
  scripts are the leaderboard/evaluation utilities it documents.
- `results/` -- the derived result files (JSON/TSV) cited by the paper, plus
  `exp_boundary_conditional/`, the files the companion boundary-analysis claims
  (`B.*`) of the registry read.
  These are the exact artifacts the harness reads; superseded originals ship
  unchanged beside their corrected versions (files suffixed `_CORRECTED`),
  as described in the benchmark paper's fragility section.
- `adr/` -- eighteen pre-registration documents (the seventeen the paper lists
  plus the Brennan preprocessing registration of its appendix), committed before the
  corresponding data existed; decision rules and falsifiers were fixed there.
  ADR 0001-0010 (0008 is an amendment), plus `CONCORDANCE_PREREG.md`, the
  registration of the 32-checkpoint readout-concordance test (concordance
  against the reliability ceiling; the paper's Table 1 double dagger), the C4
  control, the space-fix re-extraction, and the four behind the revision
  (reading rule, ridge range, post-submission checks, rescaled Qwen2.5 ladder),
  and the Brennan preprocessing check. ADR 0005 registers the crossed-narrative test,
  ADR 0006/0008 the modern-model extension (48 base checkpoints, scaling
  ladders), ADR 0010 the MEG-MASC ladder. Infrastructure names in these files
  are redacted, and pointers to internal coordination files are replaced by
  `<coordination-notes>`; nothing else is changed (see `manifest.json`).
- `codes/` -- the registered-analysis and experiment scripts, included for
  inspection, with the modules the fixed-depth generator `p4_concordance32.py`
  imports (`compute_rdms.py`, the sensor-region atlases `region_atlas*.py`,
  `extract_llm_layers*.py`, `paper4_nn_revision/path1_*.py`; put `codes/` and
  `codes/paper4_nn_revision/` on the module path). The receipt checkers run
  from this repository; experiment scripts need the unbundled neural recordings,
  cached features and models.
- `paper4_temporalalign/tex/paperA_benchmark.tex` -- read-only manuscript
  snapshot of the R1 manuscript source (MEG@9f2a3165, author block and release
  sentences as in the arXiv version) that the checkers read; not a build-ready
  source distribution. The arXiv abstract is a shortened form of the one here.
- `manifest.json` -- SHA-256 for every other packed file, including each
  original-to-packed path and original SHA. Redaction changes some packaged
  bytes; both hashes are retained rather than pretending the copy is byte-
  identical. The manifest excludes itself because a file cannot contain its
  own cryptographic hash.


## Note on `A1_git_timing`

`codes/check_paperA_revision_r1_20261007.py` checks that the reading-rule summary was committed about two hours before the 2026-09-26 19:59 (+08:00) deadline. That needs the development repository's git history. In this public release the file arrives in the root commit, so the check prints `SKIP` (not `PASS`) and the timing is not verified here.
