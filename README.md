# BrainLLM-Bench: Model Rankings Transfer Across Listeners but Not Always Across Stimuli or Readouts

Xupeng Chen (NYU), Michael Yue (Columbia University), Runhao Li (University of Southern
California), Jiayi Cheng (NYU), Chenqian Le (NYU)

Paper: arXiv:XXXX.XXXXX

## Summary

Reading a brain-likeness leaderboard as a fact about models assumes its ranking survives a change
of readout, listeners or text. On single-listener MEG, 32 checkpoints get a reliable leaderboard
from representational similarity (reliability 0.98) and an equally reliable one from linear
encoding (0.96), in opposite orders (rho = -0.573, family-clustered 95% CI [-0.79, -0.26]).
BrainLLM-Bench changes readout, listeners and stimulus one at a time. Listeners barely move a
leaderboard: disjoint listener halves agree at median Spearman rho +0.764 to +0.961. The reversal
survives the 48 base checkpoints of the registered test but belongs to one cohort: it does not
reproduce on 27-listener MEG, is not determined on 47-subject English EEG once slow drift is
removed, and is not determined when masked models read only up to the target word or once static
embeddings are partialled out. Text depends on the cohort: similarity leaderboards from different
stories heard by the same 27 listeners do not transfer, while a fully crossed fMRI cohort (25
narratives x 9 listeners) transfers well (+0.778). The benchmark is one protocol over five
language cohorts (MEG, EEG, intracranial ECoG; English and German), six leaderboard models and a
readout sweep of 57 checkpoints. The reversal is a disagreement between measurements, not a
refutation of brain-likeness.

## What this repository contains

The subset of the benchmark that the paper says is released now:

- `bench/` -- the harness (band-splitting, regioning, RSA, confound battery, per-band ceilings,
  clustered nulls) and `bench/reproduce.py`, the registry of 198 stored-result claims
  (112 benchmark-paper claims, 33 superseded-value provenance pins, 53 claims of a separate
  boundary-effect analysis on the same cohorts that is not part of this paper).
- `results/` -- derived result files (statistics and per-session / per-listener summaries) and the
  scorecard (`results/scorecard/`).
- `adr/` -- the pre-registration documents: ADR 0001-0010 (0008 is an amendment), the
  readout-concordance test, the C4 control, the re-extraction, the reading rule, the ridge range,
  the post-submission checks, the rescaled Qwen2.5 ladder, and the Brennan preprocessing check.
- `codes/` -- the analysis scripts, the receipt checkers and the sensor-region atlases
  (`region_atlas*.py`).
- `paper4_temporalalign/tex/paperA_benchmark.tex` -- the manuscript source the checkers read.
- `manifest.json` -- SHA-256 of every file, with the source path and original hash in our project
  repository (files whose bytes changed in redaction say so).

Not included, as the paper states: the alignment tensors, word lists, cached model features and
cached neural dissimilarity matrices. Without them the 504-value tensor validation
(`python -m bench.protocol --validate`) and the experiment scripts cannot be rerun; they will be
released later with an archival DOI. Raw recordings are never redistributed. Run logs and
scheduler listings are not included.

## Verify

Python 3.9+ with numpy, scipy and pandas. From the repository root:

    bash verify_release.sh            # PY=/path/to/python to choose the interpreter

It runs the claim registry (`python -m bench.reproduce`, expected `198/198 claims reproduce`)
and seven checkers that pin Table 1, the main-text numbers, the revision results, the
alpha/rogue follow-ups, the C4 control, the space-fix re-extraction and the feature-extraction
provenance against the stored receipts and the manuscript source. `VERIFY.md` describes each
check and what it does not establish: these are stored-result recomputations, not end-to-end
reruns from raw recordings.

## Citation

```bibtex
@article{chen2026brainllmbench,
  title   = {{BrainLLM-Bench}: Model Rankings Transfer Across Listeners but Not Always Across
             Stimuli or Readouts},
  author  = {Chen, Xupeng and Yue, Michael and Li, Runhao and Cheng, Jiayi and Le, Chenqian},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Data sources and licences

Code (`bench/`, `codes/`, scripts): MIT, see `LICENSE`. Derived result files keep the terms of the
recordings they are computed from (see `bench/data_card.md`): files derived from LibriBrain
(CC BY-NC 4.0) are non-commercial; the Brennan EEG cohort (CC BY 4.0, doi:10.7302/Z29C6VNH)
requires attribution; MEG-MASC, MEG-SCANS (ds006468), the Podcast ECoG dataset (ds005574) and
ds003020 are CC0. Obtain the recordings from their maintainers.

## Contact

Xupeng Chen, xc1490@nyu.edu
