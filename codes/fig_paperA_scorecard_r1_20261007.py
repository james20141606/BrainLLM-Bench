#!/usr/bin/env python3
"""paperA revision R1 (2026-10-07): Figure 9 (cross-setting scorecard) redrawn with the three Brennan
cells that read the non-recomputable 49-listener archive marked as archived.

Same receipts and layout as the submitted figure (<coordination-notes>/figure_redraw_20260922_2106/
build_parent_figures.py::paperA9, which wrote figures_v2/integrated_20260923/fig09_redrawn.pdf);
the only change is a dark-hatched overlay plus legend key on the archive cells named in ARCHIVED.
Inputs: results/scorecard/scorecard_transport.tsv, results/scorecard/scorecard_derived.json.
Output: figures_v2/revision_r1_20261007/fig09_scorecard_r1.{svg,pdf,png} via figstyle.compose.export
(Chrome PDF, layout audit, Type-3 check) + a provenance JSON with input sha256s.
"""
import csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P = ROOT / 'paper4_temporalalign'
OUT = P / 'figures_v2/revision_r1_20261007'
sys.path.insert(0, str(HERE / 'figstyle'))
import compose  # noqa: E402

INK = '#20333E'; MUTED = '#60727C'; GRID = '#DEE5E7'; TEAL = '#087F80'; CORAL = '#C95745'
BLUE = '#3E60A0'; GOLD = '#B37F2D'; GREY = '#98A5AA'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 8.5, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'text.color': INK, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'legend.frameon': False, 'hatch.linewidth': 0.6})

# (row paper, cohort) pairs whose stored cell reads the 49-listener Brennan archive
# (derive_scorecard.py: Zou via boundary_brennan_persubject_summary.tsv; Feghhi/Hadidi via
# method_nfce_conformal_brennan.py; He via method_hcd6_boundary_crosscohort.py) -- tex App. L "Brennan lineage".
ARCHIVED = {('Zou et al.', 'Brennan'), ('Feghhi/Hadidi et al.', 'Brennan'), ('He et al.', 'Brennan')}
INPUTS = [P / 'results/scorecard/scorecard_transport.tsv', P / 'results/scorecard/scorecard_derived.json']


def co_name(s):
    s = s.lower().split('/')[0].replace('-', '').replace('_', '')
    for pre, name in (('libri', 'LibriBrain'), ('megmasc', 'MEG-MASC'), ('megscans', 'MEG-SCANS'),
                      ('brennan', 'Brennan'), ('ecog', 'ECoG')):
        if s.startswith(pre):
            return name
    return None


def main():
    tab = list(csv.DictReader(INPUTS[0].open(), delimiter='\t'))
    derived = json.loads(INPUTS[1].read_text())['claims']
    tab.sort(key=lambda r: (r['fidelity'], r['paper']))
    fig, a = plt.subplots(figsize=(5.5, 4.05))
    fig.subplots_adjust(left=.01, right=.99, bottom=.19, top=.84)
    a.set(xlim=(0, 1), ylim=(len(tab) - .5, -.5)); a.axis('off')
    a.text(.01, -1.0, 'Source paper', fontsize=8.5, color=MUTED)
    a.text(.99, -1.0, 'Test scope', ha='right', fontsize=8.5, color=MUTED)
    colors = {'analogy': GOLD, 'conceptual_extension': BLUE, 'analogy_unverified': GREY}
    cohorts = ['LibriBrain', 'MEG-MASC', 'MEG-SCANS', 'Brennan', 'ECoG']; cx = [.435, .535, .635, .735, .835]
    for name, x in zip(cohorts, cx):
        a.text(x, -1.0, name.replace('MEG-', ''), rotation=45, ha='left', fontsize=7.5, color=MUTED)
    outcome_colors = {'supports': TEAL, 'refines': GOLD, 'contradicts': CORAL, 'NOT_DERIVABLE': GREY}
    marked = []
    for i, r in enumerate(tab):
        c = colors[r['fidelity']]
        a.hlines(i + .43, 0, 1, color=GRID, lw=.55)
        a.text(.01, i, r['paper'].replace(' et al.', ''), fontsize=8, va='center')
        a.text(.99, i, {'analogy': 'Analogy', 'conceptual_extension': 'Extension',
                        'analogy_unverified': 'Unverified'}[r['fidelity']], va='center', ha='right', fontsize=7.5, color=c)
        d = derived.get(r['paper'], {}); byco = defaultdict(list)
        for key, v in d.get('per_cohort', {}).items():
            if co_name(key):
                byco[co_name(key)].append(v)
        unavailable = d.get('verdict') == 'NOT_DERIVABLE'
        for co, x in zip(cohorts, cx):
            vals = byco[co]
            if unavailable:
                vals = ['NOT_DERIVABLE']
            if not vals:
                a.text(x, i, '—', ha='center', va='center', fontsize=8, color=GREY); continue
            w = .077 / len(vals)
            arch = (r['paper'], co) in ARCHIVED
            if arch:
                marked.append((r['paper'], co, sorted(vals)))
            for j, v in enumerate(sorted(vals)):
                a.add_patch(Rectangle((x - .0385 + j * w, i - .17), w, .27, fc=outcome_colors[v], ec='white', lw=.5,
                                      hatch='///' if v == 'NOT_DERIVABLE' else None))
                if arch:
                    a.add_patch(Rectangle((x - .0385 + j * w, i - .17), w, .27, fc='none', ec=INK, lw=.6, hatch='\\\\\\\\'))
            if len(vals) > 1:
                a.text(x + .042, i - .035, str(len(vals)), ha='left', va='center', fontsize=7.5, color=MUTED)  # right of the bar (audit: no overlap)
    assert sorted((p, c) for p, c, _ in marked) == sorted(ARCHIVED), marked
    keys = [Rectangle((0, 0), 1, 1, fc=outcome_colors[v], hatch='///' if v == 'NOT_DERIVABLE' else None, ec='white')
            for v in outcome_colors]
    keys.append(Rectangle((0, 0), 1, 1, fc='white', ec=INK, lw=.6, hatch='\\\\\\\\'))
    fig.legend(keys, ['Support here', 'Mixed / conditional', 'Opposite here', 'Unavailable', 'Archive, not recomputable'],
               loc='lower center', bbox_to_anchor=(.5, .065), ncol=5, fontsize=7.5, handlelength=.85, columnspacing=.7)
    fig.text(.5, .035, 'Segments: model / condition outcomes. Counts are conditions, not subjects.', ha='center', fontsize=7.5, color=MUTED)
    fig.text(.5, .006, 'Cross-setting extensions and analogies; not a literature replication rate.', ha='center', fontsize=7.5, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / 'fig09_scorecard_r1.svg'
    with plt.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0}):
        fig.savefig(svg, bbox_inches=None)
    plt.close(fig)
    pdf, png = compose.export(svg, OUT / 'fig09_scorecard_r1.pdf', OUT / 'fig09_scorecard_r1.png', width_in=5.5)
    prov = {'script': str(Path(__file__).relative_to(ROOT)),
            'inputs': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in INPUTS],
            'archived_cells': [{'paper': p, 'cohort': c, 'stored_outcomes': v} for p, c, v in marked],
            'outputs': {str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest() for x in (svg, pdf, png)}}
    (OUT / 'fig09_scorecard_r1.provenance.json').write_text(json.dumps(prov, indent=2))
    print('archived cells marked:', marked)
    print(pdf, png)


if __name__ == '__main__':
    main()
