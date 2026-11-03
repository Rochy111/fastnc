"""docs/benchmarks.json (tools/bench.py's output) as a bar chart of whole-file
reads: each reader per file, labelled in milliseconds. Needs matplotlib, which
fastnc does not.

    python tools/bench_plot.py [docs/benchmarks.json] [docs/benchmarks_raw.png]
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_table import ROOT, ms  # noqa: E402

SERIES = [('netCDF4', 'netCDF4', '#C9CDCF'), ('h5py', 'h5py', '#919596'),
          ('fastnc loop', 'fastnc, variable loop', '#5BA8E0'), ('fastnc', 'fastnc, Dataset.load()', '#1B4F72')]
LABELS = {'MODIS 1 km L1B': 'MODIS\n1 km L1B', 'MODIS 250 m L1B': 'MODIS\n250 m L1B',
          'RCM SAR winds': 'RCM SAR\nwinds', 'GOES-19 ABI RadF C02': 'GOES-19 ABI\nRadF C02',
          'GOES-19 ABI RadF C13': 'GOES-19 ABI\nRadF C13'}
INK, MUTED, GRID = '#1F2328', '#4A5056', '#D5D8DA'


def plot(rows, out, key='all_raw', title='Reading every variable of a file, raw'):
    rows = [r for r in rows if 'error' not in r.get(key, {'error': 1})]
    fig, ax = plt.subplots(figsize=(20, 6.6), dpi=200)
    wide = 0.8 / len(SERIES)
    for k, (name, label, colour) in enumerate(SERIES):
        at = [(i, r[key][name]) for i, r in enumerate(rows) if r[key].get(name) is not None]
        xs = [i + (k - (len(SERIES) - 1) / 2) * wide for i, _ in at]
        ax.bar(xs, [v for _, v in at], wide, color=colour, edgecolor='white', linewidth=1, label=label, zorder=2)
        for x, (_, v) in zip(xs, at):
            ax.annotate(ms(v), (x, v), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                        fontsize=8.5, color=MUTED)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([LABELS.get(r['file'], '\n'.join(r['file'].rsplit(' ', 1))) for r in rows],
                       fontsize=11, color=INK)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)         # (room for the legend over the tallest bar)
    ax.set_ylabel('ms', fontsize=12, color=INK)
    ax.tick_params(axis='y', labelsize=12, colors=INK, length=0)
    ax.tick_params(axis='x', length=4, color=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(MUTED)
    ax.spines['bottom'].set_color(MUTED)
    ax.set_title(title, loc='left', fontsize=14, color=INK, pad=14)
    ax.legend(loc='upper right', ncol=len(SERIES), frameon=False, fontsize=11.5, labelcolor=INK,
              columnspacing=2, handlelength=2.6)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'docs', 'benchmarks.json')
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'docs', 'benchmarks_raw.png')
    with open(path) as f:
        plot(json.load(f)['rows'], out)
    print('wrote', out)


if __name__ == '__main__':
    main()
