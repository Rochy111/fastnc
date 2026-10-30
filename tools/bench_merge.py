"""Merge benchmark runs (tools/bench.py --out ...) cell by cell, keeping the
lowest time of each cell -- every reader and fastnc alike. On this machine
whole runs differ by up to 2x in places (memory state, thermals, the
desktop), more than best-of-5 within a run evens out.

    python tools/bench_merge.py run1.json run2.json [--out docs/benchmarks.json]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: merge(a.get(k), b.get(k)) if k in a and k in b else a.get(k, b.get(k))
                for k in list(a) + [k for k in b if k not in a]}
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return min(a, b)
    return a if a is not None else b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'benchmarks.json'))
    a = ap.parse_args()
    runs = [json.load(open(p)) for p in a.runs]
    out = runs[0]
    for r in runs[1:]:
        rows = {x['file']: x for x in r['rows']}
        out['rows'] = [merge(x, rows[x['file']]) if x['file'] in rows else x for x in out['rows']]
    out['when'] = ' and '.join(r['when'] for r in runs)
    out['note'] = 'each cell the lower of %d runs (tools/bench_merge.py)' % len(runs)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
