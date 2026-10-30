"""Fresh-process times (import + open + read), the way a script runs once:
one variable and the whole file, masked as netCDF4-python reads by default,
with fastnc's kernels loading as they do by default, never loading, or
loaded first. Each number: best of 3 processes."""
import os, subprocess, sys
from common import path

CODE = r'''
import sys, time
t0 = time.perf_counter()
lib, p, v, what, policy = sys.argv[1:6]
if lib == 'netCDF4':
    import netCDF4 as m
else:
    import fastnc as m
    from fastnc import _chunks
    if policy == 'never':
        _chunks.LOAD_AFTER = float('inf')
    elif policy == 'first':
        m.warm()
d = m.Dataset(p)
if what == 'var':
    x = d[v][...]
else:
    def walk(g):
        for k, x in g.variables.items():
            x[...]
        for s in g.groups.values():
            walk(s)
    walk(d)
print((time.perf_counter() - t0) * 1e3)
'''

FILES = [('wrfout_d01_2025-10-26_06_20_00.nc', 'T'),
         ('wrfout_d02_2024-11-06_01_30_00', 'T'),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I04'),
         ('NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc', 'lat'),
         ('MOD021KM.A2025301.1445.007.2025301204859.nc', None),
         ('W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20260922172210_G_O_20260922163559_20260922163659_C_N_T__.nc', None)]


def run(*args):
    best = None
    for _ in range(3):
        r = subprocess.run([sys.executable, '-c', CODE] + [str(a) for a in args],
                           capture_output=True, text=True, cwd=path('.'))
        try:
            t = float(r.stdout.strip().splitlines()[-1])
        except Exception:
            return 'err: ' + (r.stderr.strip().splitlines() or ['?'])[-1][:60]
        best = t if best is None else min(best, t)
    return '%6.0f' % best


policies = sys.argv[1].split(',') if len(sys.argv) > 1 else ['default', 'never', 'first']
for f, v in FILES:
    for what in (['var'] if v else []) + ['all']:
        row = ['netCDF4 ' + run('netCDF4', path(f), v, what, '-')]
        row += ['%s %s' % (pol, run('fastnc', path(f), v, what, pol)) for pol in policies]
        print('%-34s %-4s ' % (f[:34], what) + '  '.join(row), flush=True)
