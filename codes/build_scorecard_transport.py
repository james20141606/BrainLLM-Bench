#!/usr/bin/env python3
"""Add estimand/fidelity/availability labels; preserve historical verdict files."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'paper4_temporalalign/results/scorecard'
if not RESULTS.exists():
    RESULTS = ROOT / 'results/scorecard'  # release layout
SCOPE = {
 'Tang et al.': ('analogy', 'fMRI generation -> MEG closed-set retrieval'),
 'Toneva & Wehbe': ('conceptual_extension', 'cohort/modality/context and layer-preference criterion'),
 'Schrimpf et al.': ('analogy_unverified', 'LM-performance correlation -> contextual/static maximum comparison; tensors missing'),
 'Caucheteux et al.': ('analogy', 'forecast-augmented fMRI encoding -> word-shift MEG alignment'),
 'Goldstein et al.': ('conceptual_extension', 'ECoG high-gamma causal LMs -> multiple bands/cohorts/architectures'),
 'Zou et al.': ('conceptual_extension', 'language/model/cohort boundary conditions'),
 'Bhaya-Grossman et al.': ('analogy', 'within-STG physiology -> multilingual/native-model layer-depth crossover'),
 'Feghhi/Hadidi et al.': ('conceptual_extension', 'fMRI/ECoG (reading and listening) confound audit -> MEG listening/disaggregation'),
 'Antonello et al.': ('conceptual_extension', 'fMRI OPT/LLaMA encoding -> MEG Qwen/Pythia fixed-depth encoding'),
 'He et al.': ('conceptual_extension', 'one model family -> architecture-dependent boundary contrast'),
}
# Tested-estimand text for rows whose legacy `our_test` wording no longer matches what the paper
# reports (2026-09-24 final review): the lag test shifts words, not milliseconds; the size test is
# the registered Qwen2.5 + corrected Pythia encoding ladders; the He et al. test is the residual
# disentanglement on LibriBrain; the Zou row cited an appendix number that no longer exists.
TESTED_OVERRIDE = {
 'Tang et al.': 'Trial-level contrastive retrieval of language features from MEG; not continuous generation',
 'Caucheteux et al.': ('Word-shift lag analogue on the LibriBrain headline cell: the model word sequence is shifted '
                       'against the brain (in words) and the peak shift located per post-onset window, GPT-2 and BERT '
                       '(exp_caucheteux/time_resolved_kpeak_libribrain.tsv).'),
 'Zou et al.': '5-cohort x 7-LLM x 2-band boundary-conditional RSA; incl. reverse direction with German LLMs.',
 'Antonello et al.': ('Registered Qwen2.5 0.5B-72B encoding ladder plus the six-point Pythia 70M-2.8B ladder scored per '
                      'session at the pre-registered depth (scaling_ladder_test_CORRECTED.json); the BERT/GPT-2 raw-RSA '
                      'comparison is a secondary arm outside their model class.'),
 'He et al.': ('Iterative residual disentanglement (lexicon, syntax, meaning, reasoning) on LibriBrain; the reasoning '
               'residual split medial versus constituent-onset, read out as alignment per model.'),
}
with (RESULTS/'scorecard.tsv').open() as f:
    records = list(csv.DictReader(f, delimiter='\t'))
assert {r['paper'] for r in records} == set(SCOPE), 'Explicit scope classification required for every row'
rows = []
for r in records:
    fidelity, changed = SCOPE[r['paper']]
    verified = r['paper'] != 'Schrimpf et al.'
    outcome = ({'replicate':'analogue_supported', 'refine':'mixed', 'contradict':'not_supported_here'}[r['verdict']]
               if verified else 'unverified_missing_tensors')
    rows.append({'paper':r['paper'], 'source_estimand':r['claim'], 'tested_estimand':TESTED_OVERRIDE.get(r['paper'], r['our_test']),
                 'fidelity':fidelity, 'changed_dimensions':changed,
                 'output_mapping_available':verified, 'current_scoped_outcome':outcome,
                 'legacy_label_for_provenance':r['verdict']})
with (RESULTS/'scorecard_transport.tsv').open('w') as f:
    writer=csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t');writer.writeheader();writer.writerows(rows)
(RESULTS/'scorecard_transport.json').write_text(json.dumps({
    'scope':'Cross-setting tests, not a direct replication campaign or a literature replication rate.',
    'mapping_status':'Author-coded estimand/fidelity classification; outcome labels mapped from legacy artifacts, with missing-data row withheld.',
    'rows':rows},indent=2)+'\n')
print(f'{len(rows)} scoped rows; {sum(r["output_mapping_available"] for r in rows)} output mappings available; no replication-rate inference')
