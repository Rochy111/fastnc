"""Inflate alone on one thread, the real files' chunks: this checkout's numba
inflate against libdeflate. Run it from two checkouts to compare versions.

    python tools/profile/inflate_ab.py [checkout]
"""
import mmap
import os
import sys
import time

CODE = sys.argv[1] if len(sys.argv) > 1 else None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import fast_thread, path                # noqa: E402

sys.path.insert(0, CODE or path('.'))
import numpy as np                                  # noqa: E402

import fastnc                                       # noqa: E402
from fastnc import _hdf5, _inflate, _nc4, _zlib     # noqa: E402
from fastnc._jit import njit                        # noqa: E402

fastnc.warm()
fast_thread()
FILES = ['wrfout_d01_2025-10-26_06_20_00.nc', 'VJ102IMG_NRT.A2026265.2006.021.nc',
         'MOD021KM.A2025301.1445.007.2025301204859.nc', 'NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc',
         '20260904I1_ku_outer_lfm_VV_L2_pass13_combined.nc',
         'extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc']


@njit(nogil=True)
def run(src, starts, ends, out, caps, tab, iws):
    n = 0
    for i in range(starts.size):
        r = _inflate.zlib_inflate(src, starts[i], ends[i], out, 0, caps[i], caps[i], tab, iws)
        n += r
    return n


def chunks_of(f):
    fh = open(path(f), 'rb')
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    src = np.frombuffer(mm, np.uint8)
    nc = _nc4.NC4File(_hdf5.H5File(mm), src)
    st, en, cp = [], [], []

    def walk(g):
        for v in g.variables.values():
            ds = v.ds
            if ds.type.dtype is None or not ds.shape or ds.layout.cls != 2:
                continue
            fs = [x.id for x in ds.filters]
            if 1 not in fs:
                continue
            idx = v._reader.index
            cb = int(np.prod(idx.chunks)) * ds.type.size + (4 if 3 in fs else 0)
            cut = 4 if fs[-1] == 3 else 0
            for a, s in zip(idx.addr.tolist(), idx.size.tolist()):
                st.append(a)
                en.append(a + s - cut)
                cp.append(cb)
        for s in g.groups.values():
            walk(s)
    walk(nc.root)
    return src, np.array(st), np.array(en), np.array(cp)


def best(fn, k=5):
    r = []
    for _ in range(k):
        t = time.perf_counter()
        fn()
        r.append(time.perf_counter() - t)
    return min(r) * 1e3


def main():
    print('%-24s %8s %9s' % ('file', 'numba ms', 'libdeflate'))
    for f in FILES:
        src, st, en, cp = chunks_of(f)
        out = np.empty(int(cp.max()) + 64, np.uint8)
        tab, iws = _inflate.workspace()
        tn = best(lambda: run(src, st, en, out, cp, tab, iws))
        tl = float('nan')
        fp, dec = _zlib.native()
        if fp:
            iws[_inflate.NATIVE:_inflate.NATIVE + 2] = fp, dec
            tl = best(lambda: run(src, st, en, out, cp, tab, iws))
            iws[_inflate.NATIVE:_inflate.NATIVE + 2] = 0
        print('%-24s %8.1f %9.1f' % (f[-24:], tn, tl), flush=True)


if __name__ == '__main__':
    main()
