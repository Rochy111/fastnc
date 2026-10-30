"""What the deflate streams of the real files are made of: symbol by
symbol, for a sample of each file's chunks -- literals against matches,
match lengths and distances, blocks and their kinds, how long the literal
codes are, how many chunks are byte-for-byte repeats -- and, of the
decoded bytes, what each byte plane of shuffled data looks like.

    python tools/profile/stream_stats.py [file-substring ...]
"""
import hashlib
import mmap
import sys
import zlib

import numpy as np
from common import fast_thread, path, real_files

sys.path.insert(0, path('.'))
import fastnc                                       # noqa: E402
from fastnc import _chunks, _hdf5, _nc4             # noqa: E402
from numba import njit                              # noqa: E402

fastnc.warm()
fast_thread()
ROOTS = 10
N = 26
S_LIT, S_MATCH, S_MBYTES, S_BLOCK0, S_BLOCK1, S_BLOCK2, S_LITBITS, S_MATCHBITS, S_LONG, S_D1, S_DNEAR, \
    S_RUNS, S_RUNLEN, S_TRANS, S_MAXLIT, S_HDRBITS, S_LIT8, S_LIT6 = range(18)

LBASE = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99,
                  115, 131, 163, 195, 227, 258], np.int64)
LEXT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0],
                np.int64)
DBASE = np.array([1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025,
                  1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577], np.int64)
DEXT = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
                 13, 13], np.int64)
ORDER = np.array([16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15], np.int64)


@njit
def build(lens, n, count, first, sym):
    """Canonical code: count per length, symbols in order."""
    count[:] = 0
    for i in range(n):
        count[lens[i]] += 1
    count[0] = 0
    offs = np.zeros(17, np.int64)
    for k in range(1, 16):
        offs[k + 1 if k < 15 else 16] = offs[k] + count[k]
    for i in range(n):
        if lens[i]:
            sym[offs[lens[i]]] = i
            offs[lens[i]] += 1


@njit
def getbits(src, st, n):
    v = 0
    for k in range(n):
        p = st[0] >> 3
        if p >= src.size:
            st[1] = 1
            return 0
        v |= ((int(src[p]) >> (st[0] & 7)) & 1) << k
        st[0] += 1
    return v


@njit
def decode(src, st, count, sym):
    code = 0
    first = 0
    idx = 0
    for k in range(1, 16):
        code |= getbits(src, st, 1)
        c = count[k]
        if code - c < first:
            return sym[idx + (code - first)]
        idx += c
        first += c
        first <<= 1
        code <<= 1
    st[1] = 1
    return -1


@njit
def walk(src, s):
    """Add the statistics of the raw deflate stream src to s."""
    st = np.zeros(2, np.int64)
    lens = np.zeros(320, np.int64)
    lc, ls = np.zeros(16, np.int64), np.zeros(288, np.int64)
    dc, ds = np.zeros(16, np.int64), np.zeros(32, np.int64)
    cc, cs = np.zeros(16, np.int64), np.zeros(19, np.int64)
    dummy = np.zeros(1, np.int64)
    litlen = np.zeros(288, np.int64)
    prevlit = True
    run = 0
    while True:
        last = getbits(src, st, 1)
        kind = getbits(src, st, 2)
        if st[1]:
            return
        s[S_BLOCK0 + kind] += 1
        if kind == 0:
            st[0] = (st[0] + 7) & ~7
            n = getbits(src, st, 16)
            getbits(src, st, 16)
            st[0] += 8 * n
            s[S_LIT] += n
            s[S_LITBITS] += 8 * n
        elif kind == 3:
            return
        else:
            h0 = st[0]
            if kind == 1:
                lens[:144] = 8
                lens[144:256] = 9
                lens[256:280] = 7
                lens[280:288] = 8
                hlit = 288
                build(lens, 288, lc, dummy, ls)
                lens[:30] = 5
                build(lens, 30, dc, dummy, ds)
            else:
                hlit = getbits(src, st, 5) + 257
                hdist = getbits(src, st, 5) + 1
                hclen = getbits(src, st, 4) + 4
                lens[:19] = 0
                for i in range(hclen):
                    lens[ORDER[i]] = getbits(src, st, 3)
                build(lens, 19, cc, dummy, cs)
                i = 0
                while i < hlit + hdist:
                    y = decode(src, st, cc, cs)
                    if st[1]:
                        return
                    if y < 16:
                        lens[i] = y
                        i += 1
                    else:
                        p = lens[i - 1] if y == 16 else 0
                        r = 3 + getbits(src, st, 2) if y == 16 else (
                            3 + getbits(src, st, 3) if y == 17 else 11 + getbits(src, st, 7))
                        for _ in range(r):
                            lens[i] = p
                            i += 1
                litlen[:hlit] = lens[:hlit]
                m = 0
                for i in range(256):
                    m = max(m, lens[i])
                s[S_MAXLIT] = max(s[S_MAXLIT], m)
                build(lens, hlit, lc, dummy, ls)
                dl = lens[hlit:hlit + hdist].copy()
                build(dl, hdist, dc, dummy, ds)
            s[S_HDRBITS] += st[0] - h0
            while True:
                b0 = st[0]
                y = decode(src, st, lc, ls)
                if st[1]:
                    return
                if y < 256:
                    bits = st[0] - b0
                    s[S_LIT] += 1
                    s[S_LITBITS] += bits
                    s[S_LIT8] += bits <= 8
                    s[S_LIT6] += bits <= 6
                    if not prevlit:
                        s[S_TRANS] += 1
                        s[S_RUNS] += 1
                        s[S_RUNLEN] += run
                        run = 0
                    run += 1
                    prevlit = True
                elif y == 256:
                    break
                else:
                    y -= 257
                    length = LBASE[y] + getbits(src, st, LEXT[y])
                    dsym = decode(src, st, dc, ds)
                    if st[1]:
                        return
                    dist = DBASE[dsym] + getbits(src, st, DEXT[dsym])
                    s[S_MATCH] += 1
                    s[S_MBYTES] += length
                    s[S_MATCHBITS] += st[0] - b0
                    s[S_LONG] += length >= 32
                    s[S_D1] += dist == 1
                    s[S_DNEAR] += dist <= 32
                    if prevlit:
                        s[S_TRANS] += 1
                    prevlit = False
        if last:
            return


def chunks_of(ds, r):
    idx = r.index
    z = [f.id for f in ds.filters]
    return idx, z


def main():
    want = sys.argv[1:]
    files = [f for f in real_files() + ['extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc',
                                        'extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc']
             if not want or any(w in f for w in want)]
    print('%-16s %6s %6s | %5s %5s %5s | %4s %5s %5s | %5s %5s %5s %5s | %5s %5s %6s %5s' % (
        'file', 'chunks', 'dupes', 'lit%', 'byts%', 'bits', 'mlen', 'd=1%', 'd<32%', 'trans',
        'bl/MB', 'dyn%', 'hdr%', 'lit8%', 'lit6%', 'maxlit', 'run'))
    for f in files:
        with open(path(f), 'rb') as fh:
            head = fh.read(4)
        if head[:3] == b'CDF':
            continue
        fh = open(path(f), 'rb')
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        src = np.frombuffer(mm, np.uint8)
        nc = _nc4.NC4File(_hdf5.H5File(mm), src)
        s = np.zeros(N, np.int64)
        seen = {}
        dupes = total = 0
        rawbytes = 0
        comp = 0
        rng = np.random.default_rng(0)

        def walkg(g):
            nonlocal dupes, total, rawbytes, comp
            for v in g.variables.values():
                ds = v.ds
                if ds.type.dtype is None or not ds.shape or ds.layout.cls != 2:
                    continue
                fs = [x.id for x in ds.filters]
                if 1 not in fs or 3 in fs and fs[-1] != 3:
                    continue
                idx = v._reader.index
                n = len(idx.addr)
                for i in range(n):
                    a, b = int(idx.addr[i]), int(idx.size[i])
                    z = src[a:a + b].tobytes()
                    h = hashlib.blake2b(z, digest_size=16).digest()
                    total += 1
                    if h in seen:
                        dupes += 1
                    seen[h] = 1
                pick = rng.choice(n, min(n, 300), replace=False) if n else []
                for i in pick:
                    a, b = int(idx.addr[i]), int(idx.size[i])
                    if fs[-1] == 3:
                        b -= 4
                    comp += b
                    walk(src[a + 2:a + b - 4], s)
            for sub in g.groups.values():
                walkg(sub)
        walkg(nc.root)
        syms = s[S_LIT] + s[S_MATCH]
        out = s[S_LIT] + s[S_MBYTES]
        if not syms:
            continue
        blocks = s[S_BLOCK0] + s[S_BLOCK1] + s[S_BLOCK2]
        print('%-16s %6d %5.1f%% | %5.1f %5.1f %5.2f | %4.1f %5.1f %5.1f | %5.2f %5.1f %5.1f %5.1f | %5.1f %5.1f %6d %5.1f' % (
            f.replace('extra/OR_ABI-L1b-Rad', '')[:16], total, 100 * dupes / max(1, total),
            100 * s[S_LIT] / syms, 100 * s[S_LIT] / out, s[S_LITBITS] / max(1, s[S_LIT]),
            s[S_MBYTES] / max(1, s[S_MATCH]), 100 * s[S_D1] / max(1, s[S_MATCH]),
            100 * s[S_DNEAR] / max(1, s[S_MATCH]), s[S_TRANS] / max(1, syms),
            blocks / (out / 1e6), 100 * s[S_BLOCK2] / max(1, blocks),
            100 * s[S_HDRBITS] / max(1, 8 * comp), 100 * s[S_LIT8] / max(1, s[S_LIT]),
            100 * s[S_LIT6] / max(1, s[S_LIT]), s[S_MAXLIT], s[S_RUNLEN] / max(1, s[S_RUNS])), flush=True)
    print("""
lit%: literals among symbols; byts%: output bytes that are literals; bits: bits a literal;
mlen: bytes a match; d=1 / d<32: matches of distance 1 / under 32; trans: literal<->match
switches per symbol (a branch miss each); bl/MB: blocks per MB of output; dyn: dynamic blocks;
hdr: compressed bits in block headers; lit8 / lit6: literals of at most 8 / 6 bits; run:
literals in a row.""")


if __name__ == '__main__':
    main()
