"""fastnc._pinflate against zlib on the big deflate chunks of the real files:
byte for byte, with the default split and with a much finer one (many
more part boundaries to get right), and how often a part's guessed start
was not where the decode before it ended (redone serially).

    python tools/check/pinflate_real.py [min_raw_bytes]   (default 1 MB)

GOES files (extra/) have no chunk over 1 MB: their chunks over 64 kB are
checked with the fine split only."""
import glob
import os
import sys
import zlib

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from fastnc import _pinflate  # noqa: E402

DATA = os.environ.get('FASTNC_DATA', r'C:\Users\Roch\Desktop\Python\fastnc')
FILES = ['MOD021KM.A2025301.1445.007.2025301204859.nc',
         'MOD02QKM.A2025301.1445.007.2025301204859.nc',
         'VJ102IMG_NRT.A2026265.2006.021.nc',
         'wrfout_d01_2025-10-26_06_20_00.nc',
         'NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc']


def deflate_chunks(path, min_raw):
    """(byte offset, stored size, decoded size, name) of every deflated chunk
    whose decoded size is at least min_raw (deflate the last filter applied
    when writing, i.e. the first undone)."""
    out = []
    with h5py.File(path, 'r') as h:
        def visit(n, d):
            if not (isinstance(d, h5py.Dataset) and d.chunks and d.size):
                return
            dc = d.id.get_create_plist()
            fl = [dc.get_filter(i)[0] for i in range(dc.get_nfilters())]
            if not fl or fl[-1] != 1:
                return
            csz = int(np.prod(d.chunks)) * d.dtype.itemsize
            if csz < min_raw:
                return
            L = []
            d.id.chunk_iter(L.append)
            out.extend((c.byte_offset, c.size, csz, n) for c in L if c.filter_mask == 0)
        h.visititems(visit)
    return out


def check(raw, chunks, configs):
    bad = 0
    tot = {k: [0, 0, 0, 0] for k in configs}       # streams, parts used, redos, bytes redone
    for off, sz, csz, name in chunks:
        ref = zlib.decompress(raw[off:off + sz].tobytes())
        for label, kw in configs.items():
            out = np.zeros(len(ref) + 64, np.uint8)
            info = []
            r = _pinflate.inflate_parallel(raw, off, off + sz, out, 32, 32 + len(ref), info=info, **kw)
            SI = info[-1]
            ok = (r == 32 + len(ref) and out[32:32 + len(ref)].tobytes() == ref
                  and not out[:32].any() and not out[32 + len(ref):].any())
            t = tot[label]
            t[0] += 1
            t[1] += int(SI[0, _pinflate.SI_NUSED])
            t[2] += int(SI[0, _pinflate.SI_NREDO])
            t[3] += int(SI[0, _pinflate.SI_REDOB])
            if not ok:
                bad += 1
                print('  MISMATCH', name, off, sz, label, r)
    return bad, tot


def main():
    min_raw = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
    fine = dict(nthreads=8, min_part=1 << 14, parts_per_thread=4)
    bad = 0
    paths = [os.path.join(DATA, f) for f in FILES]
    for p in paths + sorted(glob.glob(os.path.join(DATA, 'extra', '*.nc'))):
        goes = os.path.dirname(p).endswith('extra')
        chunks = deflate_chunks(p, 1 << 16 if goes else min_raw)
        if goes:
            chunks = [c for c in chunks if c[1] >= 1 << 16][:200]
        if not chunks:
            print('%-48s no chunks' % os.path.basename(p)[:48])
            continue
        raw = np.fromfile(p, np.uint8)
        configs = {'fine': fine} if goes else {'default': {}, 'fine': fine}
        b, tot = check(raw, chunks, configs)
        bad += b
        line = '%-48s %4d chunks %7.1f MB  ' % (os.path.basename(p)[:48], len(chunks),
                                                sum(c[2] for c in chunks) / 1e6)
        for k, (n, used, redo, rb) in tot.items():
            line += ' %s: %d parts used, %d redone (%.2f MB)' % (k, used, redo, rb / 1e6)
        print(line, '' if not b else ' %d MISMATCHES' % b)
    print('ALL IDENTICAL' if not bad else '%d MISMATCHES' % bad)
    return bad


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
