# BrainLLM-Bench

A standardized multi-cohort benchmark for **brain–LLM representational alignment**: one fixed
protocol applied across 5 cohorts (MEG, scalp EEG, intracranial ECoG; English and German), so that
two models are comparable by construction.

Companion artifact to: *BrainLLM-Bench: Model Rankings Transfer Across Listeners but Not Always
Across Stimuli or Readouts* (arXiv:XXXX.XXXXX).

> **Why this exists.** Published brain–LLM alignment numbers are not comparable across papers:
> each fixes its own cohort, band, region, latency, layer, confound set, and readout. This
> benchmark fixes all of them once. Its headline finding is that the choice of *readout* is not a
> detail — similarity-based and encoding-based readouts rank the same 32 checkpoints in opposite
> orders on the primary cohort (ρ = −0.573 from the corrected pipeline; an earlier analysis in this
> project gave −0.678 because of a 10⁻¹² guard in a correlation denominator).

---

## 1. Status

What is and is not runnable from a fresh clone:

| Component | Status | Needs |
|---|---|---|
| `reproduce.py` — verify the registered stored-result claims | **works** (count and scope printed on each run, §2) | only the bundled `results/` |
| `regenerated/` — paper tables re-emitted from `results/` | **works** | only the bundled `results/` |
| `protocol.py` — the paper's protocol (band-split, regioning, word-averaged RDM, confound partialization, ceiling) | see §2.1 | cached RDMs + LLM features |
| `run.py --task protocol` — the protocol behind one command | runs given data | preprocessed cohort epochs |
| `run.py --task best-layer` — reference baseline, *not* the protocol (no band-split, no regioning, no confound partialization, no ceiling) | runs given data | preprocessed cohort epochs |
| `run_ecog.py` — ECoG baselines | runs given data | ECoG ds005574 epochs |
| `eval/run_eval.py` — score a submission TSV | works | a submission file |

Cohort epochs are **not** redistributed here (see `data_card.md` for per-cohort licences and
access); the derived per-word results the paper cites *are* bundled.

## 2. Reproducing the paper

Registered claims are recomputed from `results/` and checked against a pinned value. It is a
CI-able check (non-zero exit on any mismatch):

```bash
python -m bench.reproduce                  # verify all claims
python -m bench.reproduce --emit-tables    # re-emit the paper's tables to bench/regenerated/
python -m bench.reproduce --json rep.json  # machine-readable report
```

Current status: **193/193 claims reproduce**. Not all 193 are numbers printed in the benchmark
paper: 112 are; 33 are superseded-value provenance pins (listed in `SUPERSEDED` in `reproduce.py`,
each naming the claim that pins the current value); and 48 (`B.*`) belong to a separate
boundary-effect analysis on the same cohorts that is not part of the benchmark submission. The run
prints this split. Of the 33 superseded pins, 8 check pre-correction files that ship beside their
`_CORRECTED` replacements, and 25 check values computed from features that were tokenized without
inter-word spaces (the 48-checkpoint test, the Qwen ladders, the base--instruct twins, the MEG-MASC
ladder and the MEG-MASC/Brennan 32-checkpoint concordances). Those were re-extracted under a
pre-registration (`paper4_temporalalign/PREREG_space_fix_reextract_20260925.md`); the current
values are the `SF.*` claims, read from `results/p4_encoding/space_fix_20260925.json` and
`results/p4_encoding/space_fix_20260925/`, and their TeX side is pinned by
`codes/check_space_fix_20260926.py`. The unspaced values stay in the registry only as provenance.
A guard fails the run if any claim reads one of the pre-space-fix result files
(`SPACE_FIX_STALE_SOURCES`) without being marked superseded, or if a superseded entry names a
sibling claim that does not exist; `codes/check_bench_space_fix_mutations_20260925.py` mutation-tests
both (expected `7/7`). The claims cover the
concordance headline, its CI
and the family-clustered CI, the invariance controls including the un-partialled symmetry check, the
standardization endpoints, the architecture-class decomposition, the Brennan cross-cohort retest,
the per-band ceilings, the broadband tally, the held-out cell-selection results on both cohorts, the
scaling ladder, the scorecard verdict counts, and the boundary paper's per-subject and control
statistics.

### 2.1 Running the protocol

`bench/protocol.py` is the protocol itself, in one module: band-split (Butterworth, zero-phase)
→ region select → time window → **per-word trial averaging** → rank-transformed RDM →
confound partialization → ceiling normalization. The word averaging matters: on these cohorts a
single-trial RDM has essentially no split-half reliability, so a pipeline that skips it
correlates against noise.

The confound family is word-level, which is the right family for a word-level RSA:

| | confound | RDM |
|---|---|---|
| C1 | word length | `abs(len_i - len_j)` |
| C2 | log frequency | `abs(zipf_i - zipf_j)` (via `wordfreq`; **inert and flagged** if unavailable) |
| C3 | orthographic | Levenshtein |
| C4 | static embedding | the model's own layer-0 RDM, isolating the *contextual* contribution |

Partialling ranks both sides and every confound, residualizes on `[1, C1..C4]`, then correlates
the residuals. C4 is skipped when scoring layer 0, where it would partial a vector against
itself.

```bash
# the protocol end to end, one command
python -m bench.run --cohort libribrain --task protocol \
    --region temporal_L --band broadband --time_s 0.350 \
    --reliability results/reliability/libribrain_region_time.tsv

# check the port against the paper's stored per-(model, subject) numbers
python -m bench.protocol --validate \
    --rdm-cache  <cache>/rdm_libribrain_path1 \
    --hidden-root results/path1_libribrain \
    --stored results/path1_confound_libribrain/rsa_confound_per_model_subject.tsv
```

Regions are the anatomical sensor wedges used for every stored result (frontal, temporal,
parieto-occipital × left/right): `codes/region_atlas.py` for Neuromag-306 and
`codes/region_atlas_kit.py` for KIT-208, in each cohort's canonical channel order. Scalp EEG is
scored on all channels. (Before 2026-09-24 this module split the channel index array in half; that
was not the paper's regioning, and `--validate`, which starts from cached neural RDMs, did not
exercise it.)

## 3. Cohorts

| Cohort | Modality | Language | Sessions | Subjects/patients |
|---|---|---|---|---|
| LibriBrain | MEG (Neuromag-306) | English | 36 recordings (12 sessions × 3 story parts) | 1 |
| MEG-MASC | MEG (KIT-208) | English | 76 recordings (19 sessions × 4 stories) | 10 |
| MEG-SCANS | MEG (Neuromag-306) | German | 92 | 23 |
| Brennan | EEG (61-electrode cap; 60 channels in the epochs) | English | n/a | 49 (47 in the rebuilt readout analyses) |
| ECoG ds005574 | ECoG | English | n/a | 9 |

The task is natural language comprehension throughout. The MEG-MASC tensor archive uses 10 of the
27 released subjects (the readout and crossed-narrative analyses use all 27, rebuilt from raw); those 10 do not differ from the other 17 on any recorded participant variable after
Holm correction (largest: age, p = 0.042 raw / 0.125 adjusted), and all are native English
speakers. Brennan contributes 49 subjects to the boundary and scorecard analyses; the
readout-concordance analyses use 47 (two subjects' public files are corrupt).

## 4. Language models

Six models form the per-cohort tensor leaderboard (paper Appendix Table 4): GPT-2, GPT-2-medium and
BERT-base on the English MEG and ECoG cohorts, and German-GPT-2, German-BERT and multilingual-BERT on
MEG-SCANS. LibriBrain scorecard rows add Pythia-160M, RoBERTa and T5 (retrieval row) and the Pythia 70M–2.8B
and Qwen2.5 0.5B–72B ladders (size row; paper Appendix L). The readout-concordance analysis additionally sweeps a larger
**32-checkpoint** set spanning the same objective, scale, and language axes. Features are per-word
and per-layer under sentence context; German cohorts are scored against the German and
multilingual variants.

## 5. Protocol and primary metric

Models are evaluated **zero-shot**: a submission is a frozen pretrained model, so there is no
training split and no way to fit the neural data.

The fixed pipeline is band-splitting → regioning → RSA → confound partialization → per-band
ceiling normalization. The **primary metric** is the confound-controlled partial Spearman
correlation at each cohort's headline cell, reported as a *fraction of the per-band reliability
ceiling* rather than as a raw correlation — a model is credited for how much of the alignable
structure it captures, not for absolute magnitude, which is small and modality-bound.

Because the neural side, the confound set, the ceiling, and the null are all fixed, two
submissions are comparable by construction.

## 6. Submitting a model, a cohort, or a claim

- **A model.** Extract per-word, per-layer features under sentence context on each cohort's word
  list and score them with `python -m bench.run --task protocol` (§2.1), which applies the same
  band, region, window, confound set and ceiling the paper uses. The older `--task best-layer`
  wrapper is a reference baseline, not the protocol, and its numbers are not comparable to the
  paper's cells.
- **A cohort.** Supply word-onset-aligned epochs plus a channel-to-region map; the protocol
  produces a tensor that drops into the leaderboard and scorecard without re-tuning.
- **A claim.** Each scorecard entry is a restated prior claim plus the cells that test it, so a new
  claim is added by naming cells rather than by writing analysis code.

```bash
python -m bench.eval submission.tsv --cohort libribrain --task best-layer
```

## 7. Tasks

See `tasks.yaml`. The benchmark scores: per-cohort **best-cell alignment** (the leaderboard),
**readout concordance** (similarity vs encoding vs decoding rankings), **band attribution**
(broadband vs narrowband gamma), and the **ten-claim scorecard** audit.

## 8. Licence

Code: MIT. Data: per-cohort licence inherited from upstream (see `data_card.md`). Cohort access:
LibriBrain (Hugging Face `pnpl/LibriBrain`, CC BY-NC 4.0, so LibriBrain-derived files are non-commercial), MEG-MASC (OSF `10.17605/OSF.IO/AG3KJ`, CC0), Brennan (UMich Deep Blue), MEG-SCANS (OpenNeuro `ds006468`, CC0),
ECoG ds005574 (OpenNeuro).

## 9. Citation

```bibtex
@article{chen2026brainllmbench,
  title     = {{BrainLLM-Bench}: Model Rankings Transfer Across Listeners but
               Not Always Across Stimuli or Readouts},
  author    = {Chen, Xupeng and Yue, Michael and Li, Runhao and Cheng, Jiayi and Le, Chenqian},
  journal   = {arXiv preprint arXiv:XXXX.XXXXX},
  year      = {2026}
}
```

## 2026-09-22 measurement-scope correction

`bench.run --task protocol` is the C1-C4 tensor/layer-scan route, not an end-to-end
implementation of the C1-C3 fixed-relative-depth concordance sweep. The latter's
generator is `codes/p4_concordance32.py`; its raw-data/cache requirements remain.
`bench.reproduce` verifies registered summaries against stored outputs, including
legacy scorecard labels; it does not establish scientific validity or a literature
replication rate. Use `scorecard_transport.tsv` for current scope and missing data.
The historical rogue intervention held encoding fixed. Its RSA-only results cannot
exclude a geometry mechanism; `codes/rogue_counterexample.py` demonstrates why.
