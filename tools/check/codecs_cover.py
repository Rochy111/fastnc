"""Which lines of the codec kernels the test chunks never run: the kernels
run as plain Python (NUMBA_DISABLE_JIT) under a line tracer, on every
chunk of the corpus up to a size, and the lines never reached are listed.
Slow; for finding the branches the corpus misses.

    python tools/check/codecs_cover.py [max_chunk_bytes] [file.h5 ...]
"""
import os
import sys

os.environ['NUMBA_DISABLE_JIT'] = '1'

import glob
import numpy as np

from codecs_ref import ROOT
import h5py

from fastnc import _jit
import fastnc._inflate, fastnc._lzf, fastnc._lz4, fastnc._zstd, fastnc._bitshuffle, fastnc._blosc, fastnc._blosc2  # noqa

MODS = [sys.modules['fastnc._' + n] for n in ('inflate', 'lz4', 'lzf', 'zstd', 'bitshuffle', 'blosc', 'blosc2')]
IDS = {32000: 'lzf', 32001: 'blosc', 32004: 'lz4', 32008: 'bitshuffle', 32015: 'zstd', 32026: 'blosc2'}


def ld64(a, i):
    return np.uint64(int.from_bytes(a[i:i + 8].tobytes(), 'little'))


def st64(a, i, v):
    a[i:i + 8] = np.frombuffer(int(v).to_bytes(8, 'little'), np.uint8)


def patch():
    _jit.load()
    for m in MODS:
        for k in ('ld64', 'st64'):
            if k in vars(m):
                setattr(m, k, globals()[k])
        for k, v in list(vars(m).items()):
            py = getattr(v, 'py_func', None)
            if py is not None:
                setattr(m, k, py)


hit = set()
files = {m.__file__ for m in MODS}


def tracer(frame, event, arg):
    if frame.f_code.co_filename in files:
        hit.add((frame.f_code.co_filename, frame.f_lineno))
        return tracer
    return None


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    paths = sys.argv[2:] or sorted(glob.glob(os.path.join(ROOT, 'tests', 'data', 'codecs', '*.h5')))
    patch()
    n = 0
    bad = 0
    for path in paths:
        with h5py.File(path, 'r') as f:
            items = []
            f.visititems(lambda nm, o: items.append(o) if isinstance(o, h5py.Dataset) else None)
            for ds in items:
                pl = ds.id.get_create_plist()
                if pl.get_nfilters() != 1:
                    continue
                fid, flags, cd = pl.get_filter(0)[:3]
                if fid not in IDS:
                    continue
                m = sys.modules['fastnc._' + IDS[fid]]
                nb = ds.dtype.itemsize * int(np.prod(ds.chunks))
                if nb > limit:
                    continue
                cda = np.array([ds.dtype.itemsize, nb, len(cd)] + list(cd), np.int64)
                ws = np.zeros(m.workspace_size(nb), np.uint8)
                for i in range(ds.id.get_num_chunks()):
                    off = ds.id.get_chunk_info(i).chunk_offset
                    mask, raw = ds.id.read_direct_chunk(off)
                    if mask & 1:
                        continue
                    src = np.frombuffer(raw, np.uint8)
                    dst = np.zeros(nb, np.uint8)
                    sys.settrace(tracer)
                    try:
                        r = m.decode(src, 0, len(src), dst, 0, nb, cda, ws)
                    finally:
                        sys.settrace(None)
                    n += 1
                    if r != nb:
                        bad += 1
                        print('   %s %s -> %s' % (ds.name, off, r))
    print('%d chunks, %d not decoded to full size' % (n, bad))
    import ast
    for m in MODS:
        fn = m.__file__
        src = open(fn).read().splitlines()
        tree = ast.parse('\n'.join(src))
        lines = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
                lines.add(node.lineno)
        miss = sorted(l for l in lines if (fn, l) not in hit)
        # module-level statements run at import, not traced
        miss = [l for l in miss if src[l - 1].startswith((' ', '\t'))]
        print('\n%s: %d lines never run' % (os.path.basename(fn), len(miss)))
        for l in miss:
            print('  %4d  %s' % (l, src[l - 1].rstrip()))


if __name__ == '__main__':
    main()
