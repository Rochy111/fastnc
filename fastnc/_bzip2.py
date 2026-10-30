"""bzip2 (HDF5 filter 307, hdf5plugin's / PyTables' H5Zbzip2.c): each chunk
one bzip2 stream; cd: [block size 1-9], which decoding does not need.

Two decoders:

- `decode`, numba, the codec contract: the stream header, then per block
  the symbol map, Huffman tables (a 10-bit lookup table each, the
  canonical limit/base walk for longer codes), selectors, MTF and
  RUNA/RUNB runs, the inverse BWT through one array of 32-bit entries
  (byte and link, as libbzip2's fast mode; in the workspace), the initial
  run-length coding undone, and each block's CRC checked (slicing-by-8,
  in a pass of its own) and the stream's combined CRC. Against Python's
  bz2 (libbzip2) on one thread: 1.0-1.4x on typical chunks (the inverse
  BWT's walk through a 3.6 MB array, one cache miss a byte, bounds both),
  3x on long runs (tools/check/filters_hdf_bzip2.py);
- `decode_py`, Python's bz2 module, which releases the GIL while it works:
  the reference, and the way out for the one thing `decode` refuses,
  randomised blocks (bzip2 0.9.0's, never written by hdf5plugin).

Like H5Zbzip2.c, only the first stream counts (bytes after its end are
ignored), and a stream that ends early is an error.
"""
import bz2

import numpy as np

from ._jit import njit
from ._szip import window

ERR_DATA = -1                   # not bzip2, a bad block, a CRC that does not match
ERR_SPACE = -2                  # output would not fit
ERR_INPUT = -3                  # the stream ends early
TBITS = 10                      # Huffman lookup table bits
MAXBLOCK = 900000               # the largest block (level 9)


def _crc_table():
    """CRC-32 with the polynomial 0x04C11DB7, most significant bit first."""
    t = np.zeros(256, np.uint32)
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = ((c << 1) ^ 0x04C11DB7) if c & 0x80000000 else (c << 1)
            c &= 0xFFFFFFFF
        t[i] = c
    return t


def _crc_tables():
    """Slicing-by-8 tables for the MSB-first CRC-32: T[k][i] is byte i's
    CRC followed by k zero bytes."""
    t = np.zeros((8, 256), np.int64)
    t[0] = _crc_table()
    for k in range(1, 8):
        for i in range(256):
            v = int(t[k - 1, i])
            t[k, i] = ((v << 8) & 0xFFFFFFFF) ^ int(t[0, v >> 24])
    return t


_CRC = _crc_tables()


@njit(cache=True, nogil=True)
def crc32(buf, a, b, crc):
    """bzip2's CRC-32 (MSB first) of buf[a:b], continuing from crc (not
    inverted): eight bytes a step through eight tables."""
    T = _CRC
    i = a
    while i + 8 <= b:
        hi = ((np.int64(buf[i]) << 24) | (np.int64(buf[i + 1]) << 16)
              | (np.int64(buf[i + 2]) << 8) | np.int64(buf[i + 3])) ^ crc
        crc = (T[7, hi >> 24] ^ T[6, (hi >> 16) & 255] ^ T[5, (hi >> 8) & 255] ^ T[4, hi & 255]
               ^ T[3, buf[i + 4]] ^ T[2, buf[i + 5]] ^ T[1, buf[i + 6]] ^ T[0, buf[i + 7]])
        i += 8
    while i < b:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ T[0, (crc >> 24) ^ np.int64(buf[i])]
        i += 1
    return crc


@njit(cache=True, nogil=True)
def _bits(src, pos, k, pend):
    """k (0..56) bits at bit pos, most significant first."""
    return np.int64((window(src, pos, pend) >> np.uint64(1)) >> np.uint64(63 - k))


@njit(cache=True, nogil=True)
def _tables(lens, alpha, limit, base, perm, tab, g):
    """bzip2's canonical decode tables (hbCreateDecodeTables) for group g,
    and a TBITS-bit lookup table (sym << 5 | len; 0: longer code). The
    shortest code length."""
    minl = 32
    maxl = 0
    for i in range(alpha):
        L = lens[g, i]
        if L > maxl:
            maxl = L
        if L < minl:
            minl = L
    pp = 0
    for L in range(minl, maxl + 1):
        for j in range(alpha):
            if lens[g, j] == L:
                perm[g, pp] = j
                pp += 1
    for i in range(23):
        base[g, i] = 0
    for i in range(alpha):
        base[g, lens[g, i] + 1] += 1
    for i in range(1, 23):
        base[g, i] += base[g, i - 1]
    for i in range(23):
        limit[g, i] = 0
    vec = 0
    for L in range(minl, maxl + 1):
        vec += base[g, L + 1] - base[g, L]
        limit[g, L] = vec - 1
        vec <<= 1
    for L in range(minl + 1, maxl + 1):
        base[g, L] = ((limit[g, L - 1] + 1) << 1) - base[g, L]
    for v in range(1 << TBITS):
        tab[g, v] = 0
        zn = minl
        while zn <= TBITS:
            zvec = v >> (TBITS - zn)
            if zvec <= limit[g, zn]:
                k = zvec - base[g, zn]
                if 0 <= k < alpha:
                    tab[g, v] = (perm[g, k] << 5) | zn
                break
            zn += 1
    return minl


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """The bzip2 stream at src[p:pend] into dst[o:oend]; ws: workspace_size()
    bytes. The end of the output, or a negative error code."""
    if pend - p < 4 or src[p] != 0x42 or src[p + 1] != 0x5A or src[p + 2] != 0x68:
        return ERR_DATA
    lev = np.int64(src[p + 3]) - 0x30
    if lev < 1 or lev > 9:
        return ERR_DATA
    maxblock = lev * 100000
    if ws.size < 4 * maxblock:
        return ERR_SPACE
    tt = ws[:4 * maxblock].view(np.uint32)
    pos = (p + 4) * 8
    end = pend * 8
    lens = np.zeros((6, 258), np.int64)
    limit = np.zeros((6, 23), np.int64)
    base = np.zeros((6, 23), np.int64)
    perm = np.zeros((6, 258), np.int64)
    tab = np.zeros((6, 1 << TBITS), np.int64)
    minls = np.zeros(6, np.int64)
    seq = np.zeros(256, np.int64)
    yy = np.zeros(256, np.uint8)
    sel = np.zeros(18002, np.uint8)
    mtfs = np.zeros(6, np.uint8)
    unz = np.zeros(256, np.int64)
    cft = np.zeros(257, np.int64)
    combined = np.int64(0)
    while True:
        if pos + 48 + 32 > end:
            return ERR_INPUT
        m1 = _bits(src, pos, 24, pend)
        m2 = _bits(src, pos + 24, 24, pend)
        pos += 48
        crc_stored = _bits(src, pos, 32, pend)
        pos += 32
        if m1 == 0x177245 and m2 == 0x385090:
            if combined != crc_stored:
                return ERR_DATA
            return o
        if m1 != 0x314159 or m2 != 0x265359:
            return ERR_DATA
        if _bits(src, pos, 1, pend):            # randomised (bzip2 < 0.9.5): not here
            return ERR_DATA
        pos += 1
        optr = _bits(src, pos, 24, pend)
        pos += 24
        used16 = _bits(src, pos, 16, pend)
        pos += 16
        nuse = 0
        for i in range(16):
            if used16 & (0x8000 >> i):
                w = _bits(src, pos, 16, pend)
                pos += 16
                for j in range(16):
                    if w & (0x8000 >> j):
                        seq[nuse] = i * 16 + j
                        nuse += 1
        if nuse == 0:
            return ERR_DATA
        alpha = nuse + 2
        ng = _bits(src, pos, 3, pend)
        pos += 3
        nsel = _bits(src, pos, 15, pend)
        pos += 15
        if ng < 2 or ng > 6 or nsel < 1:
            return ERR_DATA
        for i in range(ng):
            mtfs[i] = i
        for i in range(nsel):
            j = 0
            while _bits(src, pos, 1, pend):
                pos += 1
                j += 1
                if j >= ng:
                    return ERR_DATA
            pos += 1
            if pos > end:
                return ERR_INPUT
            v = mtfs[j]
            while j > 0:
                mtfs[j] = mtfs[j - 1]
                j -= 1
            mtfs[0] = v
            if i < 18002:
                sel[i] = v
        if nsel > 18002:
            nsel = 18002
        for g in range(ng):
            cur = _bits(src, pos, 5, pend)
            pos += 5
            for i in range(alpha):
                while True:
                    if cur < 1 or cur > 20:
                        return ERR_DATA
                    b = _bits(src, pos, 2, pend)
                    if b < 2:                   # 0: this symbol's length is cur
                        pos += 1
                        break
                    pos += 2
                    cur += 1 if b == 2 else -1
                lens[g, i] = cur
            if pos > end:
                return ERR_INPUT
            minls[g] = _tables(lens, alpha, limit, base, perm, tab, g)
        # symbols: MTF values with RUNA/RUNB runs of the front symbol
        for i in range(256):
            yy[i] = i
            unz[i] = 0
        eob = nuse + 1
        nb = 0
        gpos = 0
        gno = -1
        g = 0
        es = -1
        N = 1
        while True:
            if gpos == 0:
                gno += 1
                if gno >= nsel:
                    return ERR_DATA
                g = sel[gno]
                gpos = 50
            gpos -= 1
            w = window(src, pos, pend)
            e = tab[g, np.int64(w >> np.uint64(64 - TBITS))]
            if e:
                pos += e & 31
                sym = e >> 5
            else:                               # a code longer than TBITS bits
                zn = minls[g]
                while True:
                    if zn > 20:
                        return ERR_DATA
                    zvec = np.int64(w >> np.uint64(64 - zn))
                    if zvec <= limit[g, zn]:
                        break
                    zn += 1
                k = zvec - base[g, zn]
                if k < 0 or k >= 258:
                    return ERR_DATA
                sym = perm[g, k]
                pos += zn
            if pos > end:
                return ERR_INPUT
            if sym <= 1:                        # RUNA / RUNB
                if es < 0:
                    es = 0
                    N = 1
                es += N << sym
                N <<= 1
                if N >= 2 * 1024 * 1024:
                    return ERR_DATA
                continue
            if es >= 0:                         # a run ends: the front symbol es times
                uc = seq[yy[0]]
                if nb + es > maxblock:
                    return ERR_DATA
                unz[uc] += es
                for k in range(es):
                    tt[nb + k] = uc
                nb += es
                es = -1
            if sym == eob:
                break
            if nb >= maxblock:
                return ERR_DATA
            nn = sym - 1
            v = yy[nn]
            while nn > 0:                       # move to front
                yy[nn] = yy[nn - 1]
                nn -= 1
            yy[0] = v
            uc = seq[v]
            unz[uc] += 1
            tt[nb] = uc
            nb += 1
        if optr >= nb:
            return ERR_DATA
        # inverse BWT
        cft[0] = 0
        for i in range(256):
            cft[i + 1] = cft[i] + unz[i]
        for i in range(nb):
            uc = tt[i] & np.uint32(255)
            j = cft[uc]
            tt[j] |= np.uint32(i << 8)
            cft[uc] = j + 1
        tp = np.int64(tt[optr] >> np.uint32(8))
        # walk and undo the initial run-length coding (4 equal bytes, then a count)
        ob = o
        last = -1
        run = 0
        for i in range(nb):
            x = tt[tp]
            ch = np.int64(x & np.uint32(255))
            tp = np.int64(x >> np.uint32(8))
            if run == 4:
                if o + ch > oend:
                    return ERR_SPACE
                O = np.uint64(o)
                for k in range(np.uint64(ch)):
                    dst[O + k] = last
                o += ch
                run = 0
                last = -1
                continue
            if ch == last:
                run += 1
            else:
                run = 1
                last = ch
            if o >= oend:
                return ERR_SPACE
            dst[o] = ch
            o += 1
        crc = crc32(dst, ob, o, np.int64(0xFFFFFFFF)) ^ 0xFFFFFFFF
        if crc != crc_stored:
            return ERR_DATA
        combined = (((combined << 1) | (combined >> 31)) & 0xFFFFFFFF) ^ crc


def decode_py(data, out_nbytes, cd):
    """The bzip2 chunk `data` (bytes-like) decoded: bytes, at most
    out_nbytes long. ValueError for a damaged stream, a truncated one, or
    one that decodes to more than out_nbytes."""
    d = bz2.BZ2Decompressor()
    try:
        out = d.decompress(data, max_length=out_nbytes + 1)
    except OSError as e:
        raise ValueError('bzip2: bad stream (%s)' % e)
    if len(out) > out_nbytes:
        raise ValueError('bzip2: chunk decodes to more than %d bytes' % out_nbytes)
    if not d.eof:
        raise ValueError('bzip2: stream ends early')
    return out


def decode_into(data, dst, o, oend):
    """decode_py into dst[o:oend]; the end of the output."""
    out = decode_py(data, oend - o, None)
    n = len(out)
    dst[o:o + n] = np.frombuffer(out, np.uint8)
    return o + n


def workspace_size(chunk_nbytes):
    """The inverse BWT's array: 4 bytes for each byte of the largest block."""
    return 4 * MAXBLOCK
