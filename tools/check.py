"""Did this change break or slow anything? An answer in about a minute.

    python tools/check.py                 # the image, tests, saved answers, speed against HEAD
    python tools/check.py tests           # any of: image tests answers speed
    python tools/check.py --full          # and every kernel mode, one at a time
    python tools/check.py --against REF   # speed against another commit
    python tools/check.py --save          # (once) the reference libraries' answers

What made a check take twenty minutes was the reference libraries, not fastnc:
h5py takes two minutes over a file fastnc reads in 0.2 s, and its answers do not
change. So they are read once, with --save, into checksums beside the files, and
from then on only fastnc runs. The files are a folder of anything (FASTNC_CORPUS,
or --corpus; several, separated by the path separator): the more foreign the
better, as they have found what no benchmark here did.

Speed is the same files read by the commit and the working tree, processes
taking turns, as tools/profile/ab.py does and for the same reason: the same
code differs by up to 2x between two runs, so a time alone says nothing. A row
is reported only if the rounds agree.
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = min(os.cpu_count() or 4, 8)           # (processes that each run on the pool's threads)
WHOLE = 40 << 20                                # bytes of a variable read whole
QUIET = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
SLOWER = 1.08                                   # a change in speed worth reporting, if the rounds agree


# ------------------------------------------------------------ values as checksums

def checksum(v):
    """Of values, whichever library gave them: their order of bytes aside."""
    import numpy as np
    h = hashlib.blake2b(digest_size=8)

    def add(a):
        if isinstance(a, np.ma.MaskedArray):
            m = np.ma.getmaskarray(a)
            h.update(m.tobytes())
            a = np.ma.getdata(a)[~m]
        if isinstance(a, (str, bytes)):
            h.update(a.encode('utf-8', 'surrogatepass') if isinstance(a, str) else a)
            return
        a = np.asarray(a)
        h.update(repr(a.shape).encode())
        if a.dtype.names:
            for n in a.dtype.names:
                add(a[n])
        elif a.dtype.kind == 'O':
            for x in a.ravel().tolist():
                if 'reference' in repr(type(x)).lower():
                    continue                    # (h5py's say nothing that can be compared)
                add(x)
        elif a.dtype.kind in 'fc':
            # (as doubles: for floats of odd widths libraries choose types of their own)
            b = a.astype(np.complex128 if a.dtype.kind == 'c' else np.float64)
            b = np.where(np.isnan(b), np.nan, b)
            h.update(b.tobytes())
        elif a.dtype.kind in 'iub':
            h.update(a.astype(a.dtype.newbyteorder('=')).tobytes())
        else:
            h.update(a.tobytes())
    add(v)
    return h.hexdigest()


def tries(shape, itemsize):
    """(code, index) of the reads each variable is checked by."""
    import math
    size = math.prod(shape) * itemsize
    out = [('all', ())] if size <= WHOLE else []
    if not shape or not all(shape) or math.prod(shape[1:]) * itemsize > WHOLE:
        return out
    n = shape[0]
    out += [('first', 0), ('mid', n // 2), ('last', -1), ('some', slice(0, min(5, n))), ('steps', slice(0, n, max(1, n // 4)))]
    if len(shape) > 1:
        m = shape[1]
        out += [('pixel', (n // 2, m // 2)), ('box', (slice(0, min(3, n)), slice(0, min(3, m))))]
        if n * itemsize * math.prod(shape[2:]) <= WHOLE:
            out += [('column', (slice(None), m // 2))]
    return out


def nc_vars(g, path=''):
    for k, v in g.variables.items():
        yield path + k, v
    for k, s in g.groups.items():
        yield from nc_vars(s, path + k + '/')


def read_all(items, known, out):
    """Each of (name, object)'s reads; `known`: only those a reference made."""
    for name, x in items:
        try:
            shape = tuple(x.shape)
            item = getattr(x.dtype, 'itemsize', 8) or 8
        except Exception as e:
            out[name] = {'error': type(e).__name__}
            continue
        got = out.setdefault(name, {})
        got['shape'] = list(shape)
        for code, key in tries(shape, item):
            if known is not None and code not in known.get(name, {}):
                continue
            for _ in range(1 if known is None else 2):          # (again: through what a read leaves)
                try:
                    a = x[key]
                    if a is None:
                        break
                    c = checksum(a)
                except MemoryError:
                    break
                except Exception as e:
                    if known is None:
                        break                   # (a reference that cannot: not asked of fastnc)
                    c = 'raised ' + type(e).__name__ + ': ' + str(e)[:70]
                if known is not None and got.setdefault(code, c) != c:
                    got[code] = 'second read differs'
                    break
                got[code] = c


def batch(jobs):
    """Many files in one process: most of a process a file was importing. A
    line of JSON each, as it is done: after a crash the rest go one at a time."""
    drew = False
    for path, kind, known in jobs:
        try:
            got = work(kind, path, known)
        except Exception as e:
            got = {'!': '%s: %s' % (type(e).__name__, str(e)[:100])}
        print(json.dumps([path, kind, got]), flush=True)
        if not drew and 'numba' in sys.modules:     # (what the image holds no kernel for)
            drew = True
            print(json.dumps(['', 'numba', '%s %s' % (kind, os.path.basename(path))]), flush=True)


def worker(kind, path, known):
    print(json.dumps(work(kind, path, known)))


def work(kind, path, known):
    import warnings
    warnings.simplefilter('ignore')
    sys.path.insert(0, ROOT)
    out = {}
    if kind.endswith('5'):
        if kind == 'h5':
            import h5py
            f = h5py.File(path, 'r')
            cls = h5py.Dataset
        else:
            import fastnc
            f = fastnc.File(path)
            cls = fastnc._h5py.Dataset
        if known is None:
            names = []
            f.visititems(lambda n, o: names.append(n) if isinstance(o, cls) else None)
        else:
            names = list(known)
        read_all(((n, f[n]) for n in names), known, out)
    else:
        if kind == 'nc':
            import netCDF4
            f = netCDF4.Dataset(path)
        else:
            import fastnc
            f = fastnc.Dataset(path)
        for name, x in nc_vars(f):
            if known is not None and name not in known:
                continue
            read_all([(name, x)], known, out)
            if known is None or 'attrs' in known.get(name, {}):
                try:
                    out[name]['attrs'] = checksum(sorted((k, repr(checksum(x.getncattr(k)))) for k in x.ncattrs()).__repr__())
                except Exception:
                    out[name].pop('attrs', None)
    f.close()
    return out


def corpus_files(a):
    paths = []
    for top in (a.corpus or '').split(os.pathsep):
        for d, _, fs in os.walk(top) if top else ():
            paths += [os.path.join(d, x) for x in fs if x != 'answers.json']
    return sorted(paths)


def answers_path(a):
    return os.path.join(a.corpus.split(os.pathsep)[0], 'answers.json')


def run_worker(kind, path, known=None, timeout=180):
    """(answers, or None and why)."""
    args = [sys.executable, os.path.abspath(__file__), '--worker', kind, path]
    env = dict(os.environ, FASTNC_IMAGE_BUILD='0')
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, creationflags=QUIET,
                           input=json.dumps(known) if known is not None else '')
    except subprocess.TimeoutExpired:
        return None, 'took over %d s' % timeout
    if r.returncode:
        return None, 'exit %s: %s' % (r.returncode, r.stderr.strip()[-110:].replace('\n', ' '))
    return json.loads(r.stdout.strip().splitlines()[-1]), ''


def save(a):
    """The reference libraries' answers. Minutes, once."""
    t = time.time()
    files = corpus_files(a)

    def one(p):
        h, _ = run_worker('h5', p, timeout=600)
        n, _ = run_worker('nc', p, timeout=600)
        return p, {'h5': h, 'nc': n, 'bytes': os.path.getsize(p)}
    with ThreadPoolExecutor(WORKERS) as ex:
        done = dict(ex.map(one, files))
    with open(answers_path(a), 'w', encoding='utf-8') as out:
        json.dump(done, out)
    print('saved the answers of %d files in %.0f s: %s' % (len(done), time.time() - t, answers_path(a)))
    print('now: python tools/check.py answers --accept  (what differs today is what is known)')


def answers(a):
    path = answers_path(a) if a.corpus else ''
    if not os.path.exists(path):
        return True, 'skipped: no answers saved (--corpus, then --save)'
    with open(path, encoding='utf-8') as f:
        saved = json.load(f)
    known_path = os.path.join(os.path.dirname(path), 'known.json')
    known = {}
    if os.path.exists(known_path) and not a.accept:
        with open(known_path, encoding='utf-8') as f:
            known = json.load(f)

    jobs = []
    for p, v in saved.items():
        for kind, ref in (('f5', v['h5']), ('fn', v['nc'])):
            if ref:                             # (else the reference could not, or it crashed)
                jobs.append((p, kind, {k: x for k, x in ref.items() if 'error' not in x}))
    random.Random(0).shuffle(jobs)              # (the big files evenly among the batches)
    drew = []

    def run_batch(part):
        """{(path, kind): got}, what a crash stopped done alone."""
        env = dict(os.environ, FASTNC_IMAGE_BUILD='0')
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--batch'], capture_output=True, text=True, env=env,
                           creationflags=QUIET, input=json.dumps(part))
        done = {}
        for line in r.stdout.splitlines():
            path, kind, got = json.loads(line)
            if kind == 'numba':
                drew.append(got)
            else:
                done[path, kind] = got
        for p, kind, want in part:
            if (p, kind) not in done:
                got, why = run_worker(kind, p, want)
                done[p, kind] = {'!': why} if got is None else got
        return done
    with ThreadPoolExecutor(WORKERS) as ex:
        results = {}
        for d in ex.map(run_batch, [jobs[i::WORKERS] for i in range(WORKERS)]):
            results.update(d)
    new, gone, reads = [], {}, 0
    for p, kind, want in jobs:
        got = results[p, kind]
        found = []
        if '!' in got:
            found.append('%s: %s' % (kind, got['!']))
        else:
            for name, w in want.items():
                g = got.get(name)
                if g is None:
                    found.append('%s %s: missing' % (kind, name))
                    continue
                for code, d in w.items():
                    reads += 1
                    if g.get(code) != d:
                        found.append('%s %s %s: %s' % (kind, name, code, str(g.get(code))[:80]))
        for x in found:
            line = '%s  %s' % (os.path.basename(p), x)
            if line not in known:
                new.append(line)
            gone[line] = 1
    if a.accept:
        with open(known_path, 'w', encoding='utf-8') as f:
            json.dump(gone, f, indent=0)
        return True, '%d reads; %d differences accepted as known' % (reads, len(gone))
    for line in new[:12]:
        print('   ', line[:130])
    if len(new) > 12:
        print('    ... and %d more' % (len(new) - 12))
    fixed = [k for k in known if k not in gone]
    drew = drew if a.imaged else []             # (with no image of these sources, of course it did)
    for x in sorted(drew):
        print('    numba loaded by', x)
    return not new and not drew, '%d reads of %d files; %d new differences, %d known%s%s' % (
        reads, len(saved), len(new), len(known), ', %d of them gone' % len(fixed) if fixed else '',
        ', numba loaded in %d batches' % len(drew) if drew else '')


# ------------------------------------------------------------ tests

def tests(a):
    """In WORKERS processes, dealt out test by test (see conftest), so that no
    one file of thousands of tests keeps them all waiting."""
    env = dict(os.environ, FASTNC_IMAGE_BUILD='0')

    def one(i):
        e = dict(env, FASTNC_SPLIT='%d/%d' % (i, WORKERS))
        r = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '--tb=line', '-p', 'no:cacheprovider'] + (['-x'] if a.first else []),
                           cwd=ROOT, capture_output=True, text=True, env=e, creationflags=QUIET)
        return r.returncode, r.stdout
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(one, range(WORKERS)))
    passed = failed = 0
    ok = True
    for code, out in res:
        tail = out.strip().splitlines()
        last = tail[-1] if tail else ''
        counts = dict((word, int(n)) for n, word in re.findall(r'(\d+) (passed|failed|error)', last))
        passed += counts.get('passed', 0)
        failed += counts.get('failed', 0) + counts.get('error', 0)
        if code:
            ok = False
            for line in tail:
                if line.startswith(('FAILED', 'ERROR')):
                    print('   ', line[:130])
    if not passed:                              # (a check that saw nothing has not passed)
        return False, 'no test reported passing: ' + res[0][1][-160:].replace('\n', ' ')
    return ok, '%d passed, %d failed' % (passed, failed)


def modes(a):
    """Everything without numba's cache of an image, and without kernels at all
    (what a fresh process reads by): one at a time, as they share a build."""
    ok = True
    lines = []
    for name, extra in (('numba', ['--kernels', 'numba']), ('no kernels', ['--kernels', 'off'])):
        t = time.time()
        r = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '--tb=line', '-p', 'no:cacheprovider'] + extra,
                           cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, FASTNC_IMAGE_BUILD='0'), creationflags=QUIET)
        ok = ok and r.returncode == 0
        tail = r.stdout.strip().splitlines()
        lines.append('%s: %s (%.0f s)' % (name, tail[-1] if tail else r.stderr[-90:], time.time() - t))
        if r.returncode:
            for line in tail:
                if line.startswith('FAILED'):
                    print('   ', line[:130])
    return ok, '; '.join(lines)


# ------------------------------------------------------------ speed

def rows(root, files):
    """(name, what to time). Each short: it runs many times."""
    sys.path.insert(0, root)
    sys.path.insert(1, os.path.join(ROOT, 'tools', 'profile'))
    from common import fast_thread
    fast_thread()
    import fastnc
    fastnc.warm()

    def find(part):
        for p in files:
            if part in os.path.basename(p):
                return p
        p = os.path.join(ROOT, part)
        return p if os.path.exists(p) else None

    def row(part, v, fn, n=1, op=fastnc.File):
        p = find(part)
        if p is None:
            return None

        def go():
            with op(p) as f:
                x = f.variables[v] if op is fastnc.Dataset else f[v]
                for i in range(n):
                    fn(x, i)
        return go

    rng = random.Random(7)
    spots = [(rng.randrange(1 << 20), rng.randrange(1 << 20)) for _ in range(30)]
    METOP = 'W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20260922172210_G_O_20260922163559_20260922163659_C_N_T__.nc'
    VIIRS = 'VJ102IMG_NRT.A2026265.2006.021.nc'
    KU = '20260904I1_ku_outer_lfm_VV_L2_pass13_combined.nc'
    WRF = 'wrfout_d01_2025-10-26_06_20_00.nc'
    C13 = 'extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc'
    I05 = 'observation_data/I05'
    MT = 'data/measurement_data/vii_10690'

    def crop(n, h, w):
        return lambda x, i: x[spots[i][0] % h:spots[i][0] % h + n, spots[i][1] % w:spots[i][1] % w + n]

    def every(p):
        if p is None:
            return None

        def go():
            with fastnc.File(p) as f:
                got = []
                f.visititems(lambda n, o: got.append(o) if isinstance(o, fastnc._h5py.Dataset) else None)
                for o in got:
                    o[()]
        return go

    out = [('metop crops', row(METOP, MT, crop(64, 700, 3000), 30)), ('viirs crops', row(VIIRS, I05, crop(256, 6000, 6000), 30)),
           ('ku rows', row(KU, 'Ze', lambda x, i: x[i], 400)), ('metop rows', row(METOP, MT, lambda x, i: x[i], 400)),
           ('viirs rows', row(VIIRS, I05, lambda x, i: x[i], 300)), ('wrf row across', row(WRF, 'XLAT', lambda x, i: x[0, i % 169], 100)),
           ('wrf whole again', row(WRF, 'T2', lambda x, i: x[()], 50)), ('c13 points', row(C13, 'Rad', lambda x, i: x[2000 + i, 2000], 500)),
           ('wrf Dataset rows', row(WRF, 'T2', lambda x, i: x[0, i % 169], 60, fastnc.Dataset)),
           ('ku masked rows', row(KU, 'Ze', lambda x, i: x[i], 80, fastnc.Dataset)),
           ('wrf open', row(WRF, 'T2', lambda x, i: None)), ('wrf one small', row(WRF, 'T2', lambda x, i: x[()])),
           ('cmip series', row('tas_day_ACCESS', 'tas', lambda x, i: x[:, 10 + i, 20 + i], 6)),
           ('cmip time steps', row('tas_day_ACCESS', 'tas', lambda x, i: x[i], 60)),
           ('cmip all', row('tas_day_ACCESS', 'tas', lambda x, i: x[()])),
           ('ncep Dataset step', row('air.sig995', 'air', lambda x, i: x[i], 3, fastnc.Dataset)),
           ('nwm rows', row('channel_rt', 'streamflow', lambda x, i: x[i], 800)),
           ('glm all', every(find('OR_GLM'))),
           ('million first read', row('many_1000_v1', 'x', lambda x, i: x[4000, 4000])),
           ('million pixels', row('many_1000_v1', 'x', lambda x, i: x[(i * 7919) % 8000, (i * 104729) % 8000], 200))]
    return [(k, v) for k, v in out if v is not None]


def timed(root, files):
    out = {}
    only = os.environ.get('FASTNC_CHECK_ROWS')
    for name, fn in rows(root, files):
        if only and name not in only.split('|'):
            continue
        fn()
        best = 1e9
        for r in range(3):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
            if best > 0.1:                          # (long enough once: a third of the time is here)
                break
        out[name] = best * 1e3
    print(json.dumps(out))


def snapshot(ref):
    """The package at a commit, in a folder kept for the next check."""
    sha = subprocess.run(['git', 'rev-parse', '--short', ref], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    d = os.path.join(tempfile.gettempdir(), 'fastnc-check', sha)
    if not os.path.isdir(os.path.join(d, 'fastnc')):
        os.makedirs(d, exist_ok=True)
        tar = subprocess.run(['git', 'archive', sha, 'fastnc'], cwd=ROOT, capture_output=True, check=True).stdout
        import io
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:
            t.extractall(d, filter='data')
    return d, sha


def speed(a):
    old, sha = snapshot(a.against)
    images = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~/.cache')), 'fastnc', 'kernels')
    env = dict(os.environ, FASTNC_IMAGE_BUILD='0')
    if os.path.isdir(images):
        have = [os.path.join(images, x) for x in os.listdir(images) if '.img.' in x]
        if have:
            env['FASTNC_IMAGE_FILE'] = max(have, key=os.path.getmtime)    # (one for both: no compiling)
    got = {'old': [], 'new': []}
    files = corpus_files(a)
    listing = os.path.join(tempfile.gettempdir(), 'fastnc-check', 'files.json')
    os.makedirs(os.path.dirname(listing), exist_ok=True)
    with open(listing, 'w', encoding='utf-8') as f:
        json.dump(files, f)
    def rounds(n, only=''):
        for r in range(n):                          # (each taking its turn first)
            for tag, root in (('old', old), ('new', ROOT)) if r % 2 == 0 else (('new', ROOT), ('old', old)):
                o = subprocess.run([sys.executable, os.path.abspath(__file__), '--time', root, listing], capture_output=True,
                                   text=True, env=dict(env, FASTNC_CHECK_ROWS=only), creationflags=QUIET)
                if o.returncode:
                    return 'a timing run failed: ' + o.stderr[-200:].replace('\n', ' ')
                got[tag].append(json.loads(o.stdout.strip().splitlines()[-1]))

    def compare():
        worse, better = [], []
        for k in got['new'][0]:
            x = [g[k] for g in got['old'] if k in g]
            y = [g[k] for g in got['new'] if k in g]
            mx, my = statistics.median(x), statistics.median(y)
            if my > mx * SLOWER and min(y) > min(x) * 1.03:
                worse.append((k, mx, my))
            elif mx > my * SLOWER and min(x) > min(y) * 1.03:
                better.append((k, mx, my))
        return worse, better

    failed = rounds(a.rounds)
    if failed:
        return False, failed
    worse, better = compare()
    if worse:                                       # (three rounds can be unlucky: ask again, these alone)
        failed = rounds(6, '|'.join(k for k, _, _ in worse))
        if failed:
            return False, failed
        again = {k for k, _, _ in compare()[0]}
        worse = [(k, x, y) for k, x, y in compare()[0] if k in again]
    for k, x, y in worse:
        print('    SLOWER %-20s %9.3f -> %9.3f ms  %.2fx' % (k, x, y, y / x))
    for k, x, y in better:
        print('    faster %-20s %9.3f -> %9.3f ms  %.2fx' % (k, x, y, y / x))
    return not worse, '%d rows against %s: %d slower, %d faster' % (len(got['new'][0]), sha, len(worse), len(better))


def image(a):
    """This machine's image, built from these sources if need be, as the rest
    then run on it. A build that fails leaves fastnc reading right, by numba
    (0.7 s and 100 MB more), and nothing else says so: seven went unnoticed."""
    code = ('from fastnc import _image\n'
            '_image.build_now()\n'
            'p = _image.current()\n'
            'im = p and _image.Image(p)\n'
            'print("none" if _image.path() is None else "stale" if _image._stale() else '
            '"%d %d" % (sum(map(len, im.kernels.values())), im.left))\n')
    env = {k: v for k, v in os.environ.items() if k not in ('FASTNC_IMAGE_BUILD', 'FASTNC_IMAGE_FILE')}
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, env=env, creationflags=QUIET)
    said = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
    if said == 'none':
        return True, 'skipped: this machine can have none'
    if r.returncode or said == 'stale' or not said:
        return False, 'none could be built from these sources %s' % r.stderr.strip()[-110:].replace('\n', ' ')
    units, left = map(int, said.split())
    a.imaged = True
    return not left, '%d units, %d left out' % (units, left)


def main():
    ap = argparse.ArgumentParser(description='Did this change break or slow anything?')
    ap.add_argument('what', nargs='*', help='image, tests, answers, speed (default: all)')
    ap.add_argument('--corpus', default=os.environ.get('FASTNC_CORPUS', ''))
    ap.add_argument('--against', default='HEAD')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--full', action='store_true', help='also numba and no kernels, one at a time')
    ap.add_argument('--first', action='store_true', help='tests: stop at the first failure')
    ap.add_argument('--save', action='store_true', help='the reference libraries\' answers (minutes, once)')
    ap.add_argument('--accept', action='store_true', help='answers: what differs now is known')
    ap.add_argument('--worker', nargs=2, help=argparse.SUPPRESS)
    ap.add_argument('--time', nargs=2, help=argparse.SUPPRESS)
    ap.add_argument('--batch', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.batch:
        batch(json.loads(sys.stdin.read()))
        return
    if a.worker:
        known = sys.stdin.read() if a.worker[0].startswith('f') else ''
        worker(a.worker[0], a.worker[1], json.loads(known) if known else None)
        return
    if a.time:
        with open(a.time[1], encoding='utf-8') as f:
            timed(a.time[0], json.load(f))
        return
    if a.save:
        save(a)
        return
    steps = {'image': image, 'tests': tests, 'answers': answers, 'speed': speed}
    which = a.what or list(steps)
    start = time.time()
    ok = True
    a.imaged = False

    def step(name):
        t = time.time()
        good, say = steps[name](a)
        return name, good, say, time.time() - t
    done = [step('image')] if 'image' in which else []
    # (side by side, but for speed: its times must have the machine to themselves)
    together = [n for n in which if n not in ('image', 'speed')]
    with ThreadPoolExecutor(2) as ex:
        done += ex.map(step, together)
    if 'speed' in which:
        done.append(step('speed'))
    for name, good, say, took in done:
        ok = ok and good
        print('%-8s %-5s %s  (%.0f s)' % (name, 'ok' if good else 'FAIL', say, took), flush=True)
    if a.full:
        t = time.time()
        good, say = modes(a)
        ok = ok and good
        print('%-8s %-5s %s  (%.0f s)' % ('modes', 'ok' if good else 'FAIL', say, time.time() - t), flush=True)
    print('%s in %.0f s' % ('all ok' if ok else 'PROBLEMS', time.time() - start))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
