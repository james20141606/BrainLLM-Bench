#!/usr/bin/env python3
"""paperA appendix Figure fig:scaling (three Qwen ladders x two readouts), redrawn from a concordance
receipt given on the command line. Plotting code is copied from
<coordination-notes>/figure_redraw_20260922_2106/build_parent_figures.py::paperA3 (the published figure), with
only the input path and the output path parameterized. Receipts only; no new data.

  ~/.venvs/figtools/bin/python codes/fig_paperA_scaling_spacefix.py <concordance json> <out stem>
  # published input: paper4_temporalalign/results/p4_encoding/concordance_modern_final_v2.json
  # 2026-09-25 space-fix input: .../space_fix_20260925/aggregate/concordance_spacefix_libribrain.json
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK = '#20333E'; MUTED = '#60727C'; GRID = '#DEE5E7'; TEAL = '#087F80'; CORAL = '#C95745'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.linewidth': .65, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'lines.linewidth': 1.35, 'legend.frameon': False})


def clean(ax, axis='y'):
    ax.set_axisbelow(True); ax.grid(axis=axis, color=GRID, lw=.5); ax.tick_params(length=2.5, pad=3)


def main():
    src, stem = Path(sys.argv[1]), Path(sys.argv[2])
    d = json.loads(src.read_text())
    spec = importlib.util.spec_from_file_location("adr", ROOT / "codes/adr0006_registered_analysis.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ladders = m.LADDERS
    fig, axs = plt.subplots(2, 3, figsize=(5.5, 3.1), sharey='row')
    fig.subplots_adjust(left=.11, right=.99, bottom=.2, top=.87, wspace=.15, hspace=.3)
    rng = np.random.default_rng(11)
    for j, (gen, mods) in enumerate(ladders.items()):
        mods = [x for x in mods if x in d['per_session_rsa'] and x in d['per_session_enc']]
        labs = [x.split('-')[-2] if x.endswith('-Base') else x.split('-')[-1] for x in mods]
        for i, (arm, c) in enumerate([('rsa', TEAL), ('enc', CORAL)]):
            ax = axs[i, j]; ps = d['per_session_' + arm]
            ss = sorted(set.intersection(*[set(ps[x]) for x in mods]))
            M = np.array([[ps[x][s] for s in ss] for x in mods]) * 1000
            draws = rng.integers(0, len(ss), (10000, len(ss))); bs = M[:, draws].mean(2)
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=1); x = np.arange(len(mods))
            ax.fill_between(x, lo, hi, color=c, alpha=.14, lw=0); ax.plot(x, M.mean(1), '-o', c=c, ms=3.5)
            ax.set_xticks(x); ax.set_xticklabels(labs if i == 1 else [''] * len(labs), rotation=45, ha='right')
            clean(ax); ax.tick_params(labelsize=7.5)
            if j == 0:
                ax.set_ylabel(('Similarity' if i == 0 else 'Encoding') + '\n' + r'($\times10^{-3}$)')
        axs[0, j].text(.5, 1.12, gen.replace('qwen', 'Qwen'), transform=axs[0, j].transAxes, ha='center',
                       fontsize=9, fontweight='bold')
    fig.text(.55, .02, 'Model size · session-bootstrap 95% intervals (12 sessions)', ha='center', fontsize=8)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0}):
        for ext in ['pdf', 'png']:
            fig.savefig(f"{stem}.{ext}", dpi=240, bbox_inches=None)
    prov = {"script": str(Path(__file__).resolve()), "input": str(src),
            "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "pdf_sha256": hashlib.sha256(Path(f"{stem}.pdf").read_bytes()).hexdigest()}
    Path(f"{stem}.provenance.json").write_text(json.dumps(prov, indent=2))
    print(stem, "done")


if __name__ == "__main__":
    main()
