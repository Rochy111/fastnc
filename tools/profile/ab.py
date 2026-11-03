"""Two versions of fastnc compared on the real files, processes alternating,
so that a busy moment falls on both. Two runs of tools/bench.py cannot do this:
the same code differs between them by up to 2x, and such runs both hid real
regressions and suggested false ones.

    git archive <commit> fastnc | tar -x -C /some/old/tree
    python tools/profile/ab.py /some/old/tree . [--rounds 4] [--rows wrf,ku]
    python tools/profile/ab.py . .                    # numba against the image

Each process reads each row 7 times and keeps its best; the table has the
median of those over the rounds. The old tree compiles its own kernels the first
time (~40 s for most rows; run it once alone first).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

from common import fast_thread, path

FILES = dict(d02='wrfout_d02_2024-11-06_01_30_00', wrf='wrfout_d01_2025-10-26_06_20_00.nc',
             m250='MOD02QKM.A2025301.1445.007.2025301204859.nc',
             metop='W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20260922172210_G_O_20260922163559_20260922163659_C_N_T__.nc',
             ku='20260904I1_ku_outer_lfm_VV_L2_pass13_combined.nc', avhrr='NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc',
             rcm='rcm2-sclna-owi-cm-20261014t010230-20261014t010349-00003-xxxxx_ll_gd.nc',
             c13='extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc')
M250 = 'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_250_RefSB'
METOP = 'data/measurement_data/vii_10690'


def child(root, want):
    sys.path.insert(0, root)
    fast_thread()
    import fastnc
    fastnc.warm()
    p = {k: path(v) for k, v in FILES.items()}

    def load(f, masked):
        def go():
            with fastnc.Dataset(p[f]) as d:
                d.set_auto_maskandscale(masked)
                d.load(recursive=True)
        return go

    def window(f, v, key):
        def go():
            with fastnc.File(p[f]) as h:
                return h[v][key]
        return go

    def var(f, v, masked):
        def go():
            with fastnc.Dataset(p[f]) as d:
                x = d[v]
                x.set_auto_maskandscale(masked)
                return x[...]
        return go

    rows = [('d02 load raw', load('d02', False)), ('d02 load masked', load('d02', True)),
            ('m250 File window', window('m250', M250, (0, slice(4000, 4256), slice(2000, 2256)))),
            ('m250 load raw', load('m250', False)), ('c13 load raw', load('c13', False)),
            ('metop File window', window('metop', METOP, (slice(100, 356), slice(1000, 1256)))),
            ('wrf load raw', load('wrf', False)), ('wrf load masked', load('wrf', True)),
            ('ku load raw', load('ku', False)), ('avhrr var masked', var('avhrr', 'avhrr_ch4', True)),
            ('wrf var raw', var('wrf', 'T', False)), ('wrf var masked', var('wrf', 'T', True)),
            ('wrf P var raw', var('wrf', 'P', False)), ('rcm var raw', var('rcm', 'wind_speed', False)),
            ('ku var raw', var('ku', 'Ze', False)), ('avhrr var raw', var('avhrr', 'avhrr_ch4', False)),
            ('c13 var raw', var('c13', 'Rad', False)), ('m250 var raw', var('m250', M250, False))]
    out = {}
    for name, fn in rows:
        if want and not any(w in name for w in want.split(',')):
            continue
        fn()
        best = float('inf')
        for _ in range(7):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
        out[name] = best * 1e3
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old')
    ap.add_argument('new', nargs='?')
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--rows', default='')
    ap.add_argument('--child', action='store_true')
    a = ap.parse_args()
    if a.child:
        child(a.old, a.rows)
        return
    old, new = os.path.abspath(a.old), os.path.abspath(a.new)
    env = dict(os.environ)
    got = {'old': [], 'new': []}
    for r in range(a.rounds):
        for tag, root in (('old', old), ('new', new)) if r % 2 == 0 else (('new', new), ('old', old)):
            if old == new:
                env['FASTNC_IMAGE'] = '0' if tag == 'old' else '1'
            o = subprocess.run([sys.executable, os.path.abspath(__file__), root, '--child', '--rows', a.rows],
                               capture_output=True, text=True, env=env, cwd=root)
            if o.returncode:
                sys.exit(o.stderr[-600:])
            got[tag].append(json.loads(o.stdout.strip().splitlines()[-1]))
    print('%-20s %8s %8s %7s' % ('ms', 'old', 'new', ''))
    for k in got['old'][0]:
        x = statistics.median(g[k] for g in got['old'])
        y = statistics.median(g[k] for g in got['new'])
        print('%-20s %8.2f %8.2f %6.2fx' % (k, x, y, y / x))


if __name__ == '__main__':
    main()
