# MEG-LLM-Bench Data Card

## Overview

Benchmark spans 5 scored cohorts; the per-cohort tensor leaderboard scores six LLMs (GPT-2, GPT-2-medium and BERT-base on the English MEG and ECoG cohorts; German-GPT-2, German-BERT and multilingual BERT on MEG-SCANS), and the fixed-depth readout comparison scores 32 checkpoints (extended to 57), plus Cam-CAN as an anatomical control that is not scored. We do not redistribute upstream data; users obtain it from the original sources.

Derived tensors and tables inherit each source's licence; those computed from LibriBrain are therefore non-commercial (CC BY-NC 4.0).

## Cohort details

`n released` is what the upstream source distributes; `n analyzed` is what this benchmark scores.
Where they differ the reason is given, because a benchmark that quietly drops cohort members is
not comparable to one that does not.

| Cohort | License | Access path | n released | n analyzed | Stimulus | Language | Modality |
|--------|---------|-------------|-----------|------------|----------|----------|----------|
| **LibriBrain** | CC-BY-NC-4.0 (Hugging Face `pnpl/LibriBrain`, checked 2026-09-26) | https://libribrain.com (Sherlock 1) | 1 subject, 12 sessions (36 recordings) | 1 (36 recordings; 12 sessions in the fixed-depth analyses) | audiobook | English | MEG (Neuromag-306) |
| **MEG-MASC** | CC0-1.0 | OSF `10.17605/OSF.IO/AG3KJ` (2 parts) | 27 subjects, 49 sessions | 10 (tensor) / 27 (rebuilt) | short stories | English | MEG (KIT-208) |
| **Brennan 2019** | CC-BY-4.0 (DataCite record of doi:10.7302/Z29C6VNH, checked 2026-09-26) | UMich DeepBlue | 49 | 49 | audiobook (Alice) | English | EEG (61-ch) |
| **MEG-SCANS** | CC0 (OpenNeuro `ds006468`, checked 2026-09-26) | OpenNeuro `ds006468` | 24 | 23 | audiobook (German lit) | German | MEG (Neuromag-306) |
| **ECoG ds005574** | CC0-1.0 | OpenNeuro `ds005574` | 9 | 9 | narrative | English | iEEG |
| **ds003020** (additional crossed cohort; crossed-narrative analyses only) | CC0 (OpenNeuro `ds003020` v3.1.1, checked 2026-09-26) | OpenNeuro `ds003020` | 9 | 9 | 25 narratives (podcast stories) | English | fMRI |
| **Cam-CAN passive** | DAA application | Cambridge MRC-CBU | 638 | 51 | tones | n/a | MEG (Neuromag-306) |

Where analyzed differs from released:

- **MEG-MASC (10 of 27 in the tensor archive).** The tensor build uses a 10-subject preprocessed
  archive inherited from an earlier preprocessing run, not a quality criterion we applied; the
  readout-concordance and crossed-narrative analyses use all 27 subjects, rebuilt uniformly from the
  raw release.
- **Brennan (49 released; 47 in the rebuilt readout analyses).** The tensor and scorecard
  analyses use every released subject; the rebuilt per-word readout analyses use 47, because two
  subjects' public files are corrupt. Note this cuts the other way from MEG-MASC: Brennan & Hale
  (2019) recruited 52 and analysed 33 (three not analysed after experimenter error, ten below the
  comprehension-quiz criterion, six excluded for excessive noise). We do not apply those
  exclusions, so our Brennan sample is *larger* than
  the source paper's and includes participants its authors judged unusable. The exclusion list is
  not redistributed with the data we hold, so a robustness check restricted to the original 33 is
  not currently runnable.
- **MEG-SCANS (23 of 24).** One subject lacks usable word-aligned epochs.
- **Cam-CAN (51 of 638).** Anatomical control only; not part of the five benchmark cohorts.

## Preprocessing

All cohorts converted to a common format: `epochs_data.npy` of shape `(n_trials, n_channels, n_times)`, sampled at 250 Hz, baseline-corrected (-0.2 to 0 s pre-stimulus), z-scored per channel. Paired with `trial_metadata.tsv` containing per-trial word, sentence-position, surprisal columns.

Source-localization (MEG-SCANS only): MNE inverse + HCP-MMP1 atlas → `per_trial_roi.npy` of shape `(n_trials, 362, n_times)` per subject.

LLM activations: per-cohort `data/preprocessed/<cohort>_llm/<model>/layer_<L>.npy` of shape `(n_trials, d_model)`. Sentence context: 32-word running window for autoregressive models, full-sentence mask for BERT.

## Quality checks

- **Trial alignment**: every trial has matched MEG epoch, LLM activation, and metadata row at the same `trial_index`.
- **Surprisal**: contextual GPT-2 / German-GPT-2 surprisal computed with 25-word running context per session.
- **Source-loc QC**: per-subject FreeSurfer reconstruction passed visual inspection; HCP-MMP1 parcellation applied via `mne.morph_labels`.

## Known limitations

- LibriBrain: single subject; 36 recordings from 12 sessions provide within-subject reliability but no between-subject variance estimate.
- MEG-SCANS: source-loc only on 22/24 subjects (2 subjects had FreeSurfer recon failures).
- ECoG: word_text punctuation stripped → cannot derive sentence-onset markers from punctuation alone.
- Cam-CAN passive: tones not words → CITE-SAE / boundary-conditional / surprisal-tercile tasks do not apply.

## Citation

When using MEG-LLM-Bench, cite:
1. The benchmark paper (this work).
2. Each upstream cohort's original paper (Özdogan 2025 LibriBrain; Gwilliams 2023 MEG-MASC; Brennan & Hale 2019; Habersetzer et al. 2025 MEG-SCANS; Zada 2025 ds005574; Shafto 2014 Cam-CAN; LeBel et al. 2023 ds003020); datasets: Brennan 2018, doi:10.7302/Z29C6VNH (CC BY 4.0, attribution required); OpenNeuro ds003020 v3.1.1, doi:10.18112/openneuro.ds003020.v3.1.1.
