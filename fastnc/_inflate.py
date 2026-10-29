"""Inflate (RFC 1951) and the zlib wrapper (RFC 1950) in numba, writing
straight into the caller's array.

Why our own: HDF5's deflate chunks are many and small (thousands of 3-400
kB streams in real files) or few and huge (MODIS: 33 MB). A compiled loop
over chunks with no Python between them, output written in place, an
early stop once the rows wanted are out, and the shuffle / byte-swap /
scale that follow done while the chunk is still in cache are what make a
read fast -- none of which zlib's API allows from Python.

The decoder follows libdeflate's design:

- a 64-bit bit buffer refilled without branches (`buf |= ld64(src, p) <<
  bc; p += (63 - bc) >> 3; bc |= 56`), which leaves at least 56 bits, enough
  for a whole length/distance pair, after every refill;
- one table lookup per symbol: an 11-bit literal/length table and an 8-bit
  distance table, with subtables for longer codes. An entry holds the code
  length, the kind of symbol, the extra-bit count and the base value, so a
  length or distance is one lookup, one shift and one add;
- matches copied eight bytes at a time when the distance allows.

Errors (a bad stream, output that would not fit) return a negative code;
nothing is ever written outside [out_start, out_end).
"""
import numpy as np

from ._intrin import ld64, st64
from ._jit import njit

LBITS = 11                      # literal/length main table bits
DBITS = 8                       # distance main table bits
CBITS = 7                       # code-length code: every code fits, no subtables
LSIZE = 2560                    # > ENOUGH(286 symbols, root 11, max 15) = 2342
DSIZE = 640                     # > ENOUGH(30 symbols, root 8, max 15) = 402
CSIZE = 128

# entry: bits 0-4 bits to consume, 5-7 kind, 8-12 extra bits, 16-31 value
K_LEN, K_LIT, K_SUB, K_EOB, K_BAD = 0, 1, 2, 3, 4
BAD = K_BAD << 5

ERR_DATA = -1                   # not a valid deflate / zlib stream
ERR_SPACE = -2                  # output would not fit
ERR_INPUT = -3                  # stream runs past the end of its input

_LBASE = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51,
                   59, 67, 83, 99, 115, 131, 163, 195, 227, 258], np.int64)
_LEXT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                  5, 5, 5, 5, 0], np.int64)
_DBASE = np.array([1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385,
                   513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385,
                   24577], np.int64)
_DEXT = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
                  11, 11, 12, 12, 13, 13], np.int64)
_CLORDER = np.array([16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15],
                    np.int64)

TAB_SIZE = LSIZE + DSIZE + CSIZE
IWS_SIZE = 2720                 # rev 0:320, lens 320:640, sub 640:2688, count, next


def workspace():
    """One thread's tables (uint32) and integer scratch (int64)."""
    return np.zeros(TAB_SIZE, np.uint32), np.zeros(IWS_SIZE, np.int64)


@njit(cache=True, nogil=True)
def _entry(which, s):
    """The table entry of symbol s, less its code length: which 0 is the
    literal/length alphabet, 1 distances, 2 code lengths."""
    if which == 0:
        if s < 256:
            return (s << 16) | (K_LIT << 5)
        if s == 256:
            return K_EOB << 5
        if s < 286:
            return (_LBASE[s - 257] << 16) | (_LEXT[s - 257] << 8) | (K_LEN << 5)
        return BAD
    if which == 1:
        if s < 30:
            return (_DBASE[s] << 16) | (_DEXT[s] << 8) | (K_LEN << 5)
        return BAD
    return s << 16


@njit(cache=True, nogil=True)
def _build(lens, l0, n, table, t0, cap, tbits, which, iws):
    """Fill table[t0:t0+cap] for the canonical code of lens[l0:l0+n].
    0, or ERR_DATA for an over-subscribed or (beyond a single code)
    incomplete code, as zlib rejects them."""
    count = iws[2688:2704]
    nxt = iws[2704:2720]
    rev = iws[0:320]
    for L in range(16):
        count[L] = 0
    for s in range(n):
        count[lens[l0 + s]] += 1
    count[0] = 0
    left = 1
    used = 0
    maxlen = 0
    for L in range(1, 16):
        left = (left << 1) - count[L]
        used += count[L]
        if count[L]:
            maxlen = L
        if left < 0:
            return ERR_DATA
    if used and left > 0 and (which == 2 or maxlen != 1):
        return ERR_DATA                         # incomplete (zlib allows one 1-bit code)
    code = 0
    nxt[0] = 0
    for L in range(1, 16):
        code = (code + count[L - 1]) << 1
        nxt[L] = code
    size = 1 << tbits
    sub = iws[640:640 + size]
    for j in range(size):
        table[t0 + j] = BAD
        sub[j] = 0
    for s in range(n):
        L = lens[l0 + s]
        if L == 0:
            continue
        c = nxt[L]
        nxt[L] = c + 1
        r = 0
        for k in range(L):
            r = (r << 1) | ((c >> k) & 1)
        rev[s] = r
        if L > tbits:
            q = r & (size - 1)
            if L - tbits > sub[q]:
                sub[q] = L - tbits
    pos = size
    for q in range(size):
        sb = sub[q]
        if sb > 0:
            if pos + (1 << sb) > cap:
                return ERR_DATA
            table[t0 + q] = (pos << 16) | (sb << 8) | (K_SUB << 5) | tbits
            for j in range(1 << sb):
                table[t0 + pos + j] = BAD
            pos += 1 << sb
    for s in range(n):
        L = lens[l0 + s]
        if L == 0:
            continue
        r = rev[s]
        ent = _entry(which, s)
        if L <= tbits:
            for j in range(r, size, 1 << L):
                table[t0 + j] = ent | L
        else:
            e = table[t0 + (r & (size - 1))]
            start = np.int64(e >> 16)
            sb = np.int64((e >> 8) & 31)
            Lr = L - tbits
            for j in range(r >> tbits, 1 << sb, 1 << Lr):
                table[t0 + start + j] = ent | Lr
    return 0


@njit(cache=True, nogil=True)
def _fixed(tab, iws):
    lens = iws[320:640]
    for s in range(288):
        lens[s] = 8 if s < 144 else (9 if s < 256 else (7 if s < 280 else 8))
    _build(lens, 0, 288, tab, 0, LSIZE, LBITS, 0, iws)
    for s in range(32):
        lens[s] = 5                             # 30 and 31 complete the code, never valid
    _build(lens, 0, 32, tab, LSIZE, DSIZE, DBITS, 1, iws)


@njit(cache=True, nogil=True)
def inflate(src, p, pend, dst, o0, oend, ostop, tab, iws):
    """Inflate the raw deflate stream at src[p:pend] into dst[o0:oend].
    Stops early once ostop bytes are out (ostop <= oend; pass oend for the
    whole stream). Returns the end of the output (o0 + bytes written) or a
    negative error code."""
    u = np.uint64
    buf = u(0)
    bc = 0
    o = o0
    lens = iws[320:640]
    LM = u((1 << LBITS) - 1)
    DM = u((1 << DBITS) - 1)
    CM = u((1 << CBITS) - 1)
    final = 0
    while final == 0:
        if o >= ostop:
            return o
        # -- block header
        while bc <= 56:
            if p < pend:
                buf |= u(src[p]) << u(bc)
            p += 1
            bc += 8
        final = np.int64(buf & u(1))
        btype = np.int64((buf >> u(1)) & u(3))
        buf >>= u(3)
        bc -= 3
        if btype == 0:
            buf >>= u(bc & 7)
            bc -= bc & 7
            p -= bc >> 3
            buf = u(0)
            bc = 0
            if p + 4 > pend:
                return ERR_INPUT
            ln = np.int64(src[p]) | (np.int64(src[p + 1]) << 8)
            nl = np.int64(src[p + 2]) | (np.int64(src[p + 3]) << 8)
            p += 4
            if ln != (~nl & 0xFFFF):
                return ERR_DATA
            if p + ln > pend:
                return ERR_INPUT
            if o + ln > oend:
                return ERR_SPACE
            k = 0
            while k + 8 <= ln:
                st64(dst, o + k, ld64(src, p + k))
                k += 8
            while k < ln:
                dst[o + k] = src[p + k]
                k += 1
            p += ln
            o += ln
            continue
        if btype == 1:
            _fixed(tab, iws)
        elif btype == 2:
            nlit = np.int64(buf & u(31)) + 257
            ndist = np.int64((buf >> u(5)) & u(31)) + 1
            ncl = np.int64((buf >> u(10)) & u(15)) + 4
            buf >>= u(14)
            bc -= 14
            if nlit > 286 or ndist > 30:
                return ERR_DATA
            for i in range(19):
                lens[i] = 0
            for i in range(ncl):
                if bc < 3:
                    while bc <= 56:
                        if p < pend:
                            buf |= u(src[p]) << u(bc)
                        p += 1
                        bc += 8
                lens[_CLORDER[i]] = np.int64(buf & u(7))
                buf >>= u(3)
                bc -= 3
            # code-length code into the tail of the table area
            if _build(lens, 0, 19, tab, LSIZE + DSIZE, CSIZE, CBITS, 2, iws) < 0:
                return ERR_DATA
            n = nlit + ndist
            i = 0
            while i < n:
                if bc < 16:
                    while bc <= 56:
                        if p < pend:
                            buf |= u(src[p]) << u(bc)
                        p += 1
                        bc += 8
                e = np.int64(tab[LSIZE + DSIZE + np.int64(buf & CM)])
                cl = e & 31
                if cl == 0:
                    return ERR_DATA
                buf >>= u(cl)
                bc -= cl
                sym = e >> 16
                if sym < 16:
                    lens[i] = sym
                    i += 1
                    continue
                if sym == 16:
                    if i == 0:
                        return ERR_DATA
                    rep = 3 + np.int64(buf & u(3))
                    buf >>= u(2)
                    bc -= 2
                    v = lens[i - 1]
                elif sym == 17:
                    rep = 3 + np.int64(buf & u(7))
                    buf >>= u(3)
                    bc -= 3
                    v = 0
                else:
                    rep = 11 + np.int64(buf & u(127))
                    buf >>= u(7)
                    bc -= 7
                    v = 0
                if i + rep > n:
                    return ERR_DATA
                for k in range(rep):
                    lens[i + k] = v
                i += rep
            if lens[256] == 0:
                return ERR_DATA
            if _build(lens, 0, nlit, tab, 0, LSIZE, LBITS, 0, iws) < 0:
                return ERR_DATA
            if _build(lens, nlit, ndist, tab, LSIZE, DSIZE, DBITS, 1, iws) < 0:
                return ERR_DATA
        else:
            return ERR_DATA
        # -- the block's symbols
        while True:
            if p + 8 <= pend:
                buf |= ld64(src, p) << u(bc)
                p += (63 - bc) >> 3
                bc |= 56
            else:
                while bc <= 56:
                    if p < pend:
                        buf |= u(src[p]) << u(bc)
                    p += 1
                    bc += 8
            e = np.int64(tab[np.int64(buf & LM)])
            kind = (e >> 5) & 7
            if kind == K_SUB:
                buf >>= u(LBITS)
                bc -= LBITS
                e = np.int64(tab[(e >> 16) + np.int64(buf & u((1 << ((e >> 8) & 31)) - 1))])
                kind = (e >> 5) & 7
            cl = e & 31
            buf >>= u(cl)
            bc -= cl
            if kind == K_LIT:
                if o >= oend:
                    return ERR_SPACE
                dst[o] = e >> 16
                o += 1
                continue
            if kind != K_LEN:
                if kind == K_EOB:
                    break
                return ERR_DATA
            ex = (e >> 8) & 31
            length = (e >> 16) + np.int64(buf & u((1 << ex) - 1))
            buf >>= u(ex)
            bc -= ex
            e = np.int64(tab[LSIZE + np.int64(buf & DM)])
            kind = (e >> 5) & 7
            if kind == K_SUB:
                buf >>= u(DBITS)
                bc -= DBITS
                e = np.int64(tab[LSIZE + (e >> 16)
                                 + np.int64(buf & u((1 << ((e >> 8) & 31)) - 1))])
                kind = (e >> 5) & 7
            if kind != K_LEN:
                return ERR_DATA
            cl = e & 31
            buf >>= u(cl)
            bc -= cl
            ex = (e >> 8) & 31
            dist = (e >> 16) + np.int64(buf & u((1 << ex) - 1))
            buf >>= u(ex)
            bc -= ex
            s = o - dist
            if s < o0:
                return ERR_DATA
            if o + length > oend:
                return ERR_SPACE
            if dist >= 8 and o + length + 8 <= oend:
                k = 0
                while k < length:
                    st64(dst, o + k, ld64(dst, s + k))
                    k += 8
            elif dist == 1:
                v = dst[s]
                for k in range(length):
                    dst[o + k] = v
            else:
                for k in range(length):
                    dst[o + k] = dst[s + k]
            o += length
            if o >= ostop:
                return o
        if p - (bc >> 3) > pend:
            return ERR_INPUT
    if p - (bc >> 3) > pend:
        return ERR_INPUT
    return o


@njit(cache=True, nogil=True)
def zlib_inflate(src, p, pend, dst, o0, oend, ostop, tab, iws):
    """inflate() of a zlib stream (2-byte header, deflate, Adler-32 trailer,
    which is not checked)."""
    if pend - p < 2:
        return ERR_INPUT
    cmf = np.int64(src[p])
    flg = np.int64(src[p + 1])
    if (cmf & 15) != 8 or (cmf >> 4) > 7 or ((cmf << 8) | flg) % 31 != 0 or (flg & 32):
        return ERR_DATA
    return inflate(src, p + 2, pend, dst, o0, oend, ostop, tab, iws)
