"""docs/benchmarks.json (tools/bench.py's output) as the README's markdown
tables: milliseconds, and fastnc's speed-up in brackets.

    python tools/bench_table.py [docs/benchmarks.json]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ms(x):
    if x is None or isinstance(x, str):
        return '-'
    return '%.0f' % x if x >= 100 else ('%.1f' % x if x >= 1 else '%.2f' % x)


def ratio(ref, ours):
    if not isinstance(ref, (int, float)) or not isinstance(ours, (int, float)) or not ours:
        return ''
    return ' (%.1fx)' % (ref / ours)


def table(rows, key, cols, ours, title):
    out = ['**%s**' % title, '', '| file | ' + ' | '.join(cols) + ' |',
           '|---|' + '---|' * len(cols)]
    for r in rows:
        d = r.get(key)
        if not isinstance(d, dict) or 'error' in d:
            continue
        o = d.get(ours)
        cells = []
        for c in cols:
            v = d.get(c)
            # h5py is compared with fastnc.File (same API), the rest with fastnc
            base = d.get('fastnc.File') if c == 'h5py' and 'fastnc.File' in d else o
            plain = c == ours or c == 'fastnc.File'
            cells.append(ms(v) + ('' if plain else ratio(v, base)) if v is not None else '-')
        out.append('| %s | %s |' % (r['file'], ' | '.join(cells)))
    return '\n'.join(out) + '\n'


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'docs', 'benchmarks.json')
    d = json.load(open(path))
    m = d['machine']
    rows = d['rows']
    print('Measured %s: %s, %d logical CPUs, %s; Python %s, numpy %s, numba %s; h5py %s, '
          'netCDF4 %s, xarray %s, h5netcdf %s, cupy %s.\n' % (
              d['when'], m.get('cpu'), m.get('logical_cpus'), m.get('gpu'), m.get('python'),
              m.get('numpy'), m.get('numba'), m.get('h5py'), m.get('netCDF4'), m.get('xarray'),
              m.get('h5netcdf'), m.get('cupy')))
    print(table(rows, 'all_raw', ['fastnc', 'netCDF4', 'h5py'], 'fastnc',
                'Whole file, every variable, raw (ms)'))
    print(table(rows, 'all_masked', ['fastnc', 'netCDF4'], 'fastnc',
                'Whole file, masked and scaled as netCDF4-python does by default (ms)'))
    print(table(rows, 'var_raw', ['fastnc', 'netCDF4', 'fastnc.File', 'h5py'], 'fastnc',
                'Open + one variable, raw (ms; fastnc.Dataset vs netCDF4, fastnc.File vs h5py)'))
    print(table(rows, 'var_masked', ['fastnc', 'netCDF4'], 'fastnc',
                'Open + one variable, masked and scaled (ms)'))
    print(table(rows, 'window_raw', ['fastnc', 'netCDF4', 'fastnc.File', 'h5py'], 'fastnc',
                'Open + a window of it (ms)'))
    xr = [dict(r, xarray_load=r.get('xarray_load')) for r in rows]
    print(table(xr, 'xarray_load', ['fastnc', 'netcdf4', 'h5netcdf', 'scipy'], 'fastnc',
                'xarray.open_dataset(...).load(), by engine (ms)'))
    print(table(rows, 'first_call', ['fastnc', 'netCDF4', 'fastnc.File', 'h5py'], 'fastnc',
                'First read in a fresh process: import + open + one variable (ms)'))
    print(table(rows, 'gpu_masked', ['fastnc gpu', 'fastnc cpu+upload', 'netCDF4+upload'],
                'fastnc gpu', 'One variable, masked and scaled, to a cupy array (ms)'))
    t1 = ['| file | all threads | 1 thread |', '|---|---|---|']
    for r in rows:
        a = r.get('all_raw', {}).get('fastnc')
        b = r.get('all_raw_1thread')
        if a and b:
            t1.append('| %s | %s | %s (%.1fx) |' % (r['file'], ms(a), ms(b), b / a))
    print('**fastnc, whole file raw: every core vs one (ms)**\n\n' + '\n'.join(t1))


if __name__ == '__main__':
    main()
