"""Inflate of one big zlib stream on several threads, byte-identical to zlib.

Why: MODIS L1B keeps each band in a few chunks of up to 33 MB, each one
deflate stream; one core inflates 0.45 GB/s of those (radiances barely
compress: almost all literals), so a read of such a band is as slow as its
biggest chunk however many cores there are. Deflate
cannot be split where one likes -- blocks are not byte aligned and matches
reach 32 kB back -- but it can be split speculatively, as pugz (Kerbiriou &
Chikhi) and rapidgzip (Knespel & Brunst 2023) do:

- the compressed range is cut into parts. Part 0 starts from the known
  state and inflates normally. Every other part looks forward from its cut
  for a place a block plausibly starts: a dynamic-Huffman header (not the
  last block) whose code-length code, literal/length code and distance code
  are all valid and complete, or a stored block (LEN = ~NLEN, padding zero);
  then it decodes from there not knowing the 32 kB before it. A match
  reaching into that unknown window writes a 16-bit marker (the window
  position) instead of a byte. Once 32 kB of output hold no marker, nothing
  later can produce one, and the part switches to the ordinary byte decoder
  (the fast loop of `_inflate`). A start that fails to decode is dropped and
  the search goes on from the next bit.
- each part decodes until the first such block start ("sync point") at or
  after the next part's cut -- the same predicate the next part searched
  with, so where part k-1 really stopped is exactly where part k should
  have started.
- then, in order, the chain is checked: part k is used only if it started
  at exactly the sync point (kind and bit) where the verified decode before
  it stopped; its output then goes at its true offset, its markers resolved
  against the true 32 kB before it. A part that guessed wrong (a false
  positive, a part with no sync point, one that ran out of scratch) is
  redone serially from the verified point with the true window. Correctness
  never rests on the guesses: a speculative result is used only after it is
  proven to start where the true decode is.

Speculative output goes to slabs of a shared scratch, taken as a part
needs them (its later regions take runs of 2, 4, 8 ... slabs), so a part
of a skewed stream -- a stretch of fill giving 50 times the average --
never has to stop early; a region continuing another starts with a copy of
its last 32 kB. Touched scratch is about the output plus its 16-bit share.

Three steps on the shared pool (`_pool.share`): (A) the parts of every
stream decode as tasks off one counter (then, if given, other work: the
engine's smaller chunks); the thread finishing a stream's last part walks
its chain (B), writing each part's last 32 kB and any serial redo straight
into place; (C) the bulk copies of the parts' bytes and the marker
resolution of their first stretch, as tasks again. A partial read (`ostop`
under half the output) first covers the compressed share expected to give
that much output plus a margin, then the rest if short.

Same errors as `_inflate.zlib_inflate` (-1 bad data, -2 output would not
fit, -3 input ends early); nothing is written outside dst[o0:oend] or the
module's own scratch.
"""
import threading
import time

import numpy as np

from . import _pool
from ._inflate import (CBITS, CSIZE, DBITS, DSIZE, ERR_DATA, ERR_INPUT, ERR_SPACE, K_EOB,
                       K_LEN, K_LIT, K_SUB, LBITS, LSIZE, _CLORDER, _build, _fixed, workspace)
from ._intrin import ld64, st64, take
from ._jit import intrinsic, njit

W = 32768                       # deflate window

# how a decode ended (>= 0) -- errors are < 0
END_BLOCK = 0                   # at a sync point at or after the target (res R_EK, R_ES)
END_STREAM = 1                  # the last block done
END_STOP = 2                    # ostop reached
END_FULL = 3                    # speculative buffer full: usable up to the block start R_T*
SWITCH = 4                      # (internal) no marker in the last 32 kB
ROOM = 5                        # (internal) the next symbol might not fit

# res: one decode's / one part's record
R_EK, R_ES = 0, 1               # sync point reached: kind (0 header bit, 1 stored LEN byte), bit
R_O = 2                         # true decode: output end
R_NSEG = 3                      # speculative: regions written
R_TK, R_TS, R_TSEG, R_TCNT = 4, 5, 6, 7   # its current block's start (kind, bit), region, count
R_EBIT = 8                      # bit after the last block
R_ST, R_GK, R_GS = 9, 10, 11    # part: how it ended (-1: no start found), its start kind, bit
NRES = 12

# a speculative part's output regions (SEG[part, k])
G_OFF, G_TYP, G_LEAD, G_N = range(4)    # byte offset in the scratch, 1 symbols / 2 bytes, lead, count
NG = 4
MAXSEG = 32

# streams (SI) and parts (PT) tables
SI_LO, SI_PEND, SI_O0, SI_OEND, SI_OSTOP, SI_K, SI_S, SI_O, SI_ST, SI_P0, SI_NP = range(11)
SI_NUSED, SI_NREDO, SI_REDOB = 11, 12, 13   # parts used, serial redos, bytes redone
SI_D, SI_RND = 14, 15           # output array (0 / 1), rounds so far
NSI = 16
PT_S, PT_J, PT_A, PT_B = range(4)        # stream, index in it, cut, next cut (bits)
NPT = 4
J_T, J_DST, J_SRC, J_LEN, J_WIN, J_LO, J_STREAM = range(7)
NJ = 7
J_COPY, J_RESOLVE = 1, 2

PIECE = 1 << 20                 # step C task size (bytes / symbols)


def _make_lut():
    """Which 13-bit values can begin a non-last dynamic block: BFINAL 0,
    BTYPE 2, HLIT <= 29, HDIST <= 29 (one bit position in ~9 passes)."""
    t = np.zeros(1 << 13, np.uint8)
    for x in range(1 << 13):
        if (x & 7) == 4 and ((x >> 3) & 31) <= 29 and ((x >> 8) & 31) <= 29:
            t[x] = 1
    return t


_HLUT = _make_lut()


@intrinsic
def _take_n(typingctx, ctr, i, n):
    """ctr[i]'s value, then n more there: an atomic fetch-and-add (as
    `_intrin.take`, by n), so threads taking runs of slabs get disjoint ones."""
    from numba import types
    from numba.core import cgutils

    def codegen(context, builder, sig, args):
        ary = context.make_array(sig.args[0])(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, sig.args[0], ary, [args[1]])
        return builder.atomic_rmw('add', ptr, args[2], 'acq_rel')
    return types.int64(ctr, i, types.int64), codegen


# ---------------------------------------------------------------- sync points

@njit(cache=True, nogil=True)
def _bits(src, pend, b, n):
    """n <= 56 bits of src at bit b (LSB first); bytes at or past pend read as 0."""
    q = b >> 3
    if q + 8 <= pend:
        w = ld64(src, q)
    else:
        w = np.uint64(0)
        for k in range(8):
            if q + k < pend:
                w |= np.uint64(src[q + k]) << np.uint64(8 * k)
    return np.int64((w >> np.uint64(b & 7)) & np.uint64((1 << n) - 1))


@njit(cache=True, nogil=True)
def _dyn_ok(src, pend, b, tab, iws):
    """Whether a non-last dynamic block header at bit b parses into valid and
    complete codes (code-length, literal/length with an end-of-block code,
    distance). A header the decoder accepts and that is complete passes:
    every zlib-written dynamic block. Uses the code-length table area and
    iws as scratch."""
    h = _bits(src, pend, b, 17)
    if (h & 7) != 4:
        return False
    nlit = ((h >> 3) & 31) + 257
    ndist = ((h >> 8) & 31) + 1
    ncl = ((h >> 13) & 15) + 4
    if nlit > 286 or ndist > 30:
        return False
    lens = iws[320:640]
    for i in range(19):
        lens[i] = 0
    b += 17
    kr = 0
    for i in range(ncl):
        L = _bits(src, pend, b, 3)
        lens[_CLORDER[i]] = L
        if L:
            kr += 128 >> L
        b += 3
    if kr != 128:
        return False
    if _build(lens, 0, 19, tab, LSIZE + DSIZE, CSIZE, CBITS, 2, iws) < 0:
        return False
    n = nlit + ndist
    i = 0
    while i < n:
        x = _bits(src, pend, b, 14)
        e = np.int64(tab[LSIZE + DSIZE + (x & 127)])
        cl = e & 31
        if cl == 0:
            return False
        b += cl
        x >>= cl
        sym = e >> 16
        if sym < 16:
            lens[i] = sym
            i += 1
            continue
        if sym == 16:
            if i == 0:
                return False
            rep = 3 + (x & 3)
            b += 2
            v = lens[i - 1]
        elif sym == 17:
            rep = 3 + (x & 7)
            b += 3
            v = 0
        else:
            rep = 11 + (x & 127)
            b += 7
            v = 0
        if i + rep > n:
            return False
        for k in range(rep):
            lens[i + k] = v
        i += rep
    if lens[256] == 0:
        return False
    s = 0
    for i in range(nlit):
        if lens[i]:
            s += 1 << (15 - lens[i])
    if s != 32768:
        return False
    s = 0
    for i in range(nlit, n):
        if lens[i]:
            s += 1 << (15 - lens[i])
    return s == 32768


@njit(cache=True, nogil=True)
def _stored_ok(src, lo, pend, q):
    """Whether a stored block's LEN can be at byte q: LEN = ~NLEN, the block
    inside the input, and the three bits before the byte boundary zero (the
    header or zero padding, as zlib writes)."""
    if q - 1 < lo or q + 4 > pend:
        return False
    if src[q - 1] & 0xE0:
        return False
    ln = np.int64(src[q]) | (np.int64(src[q + 1]) << 8)
    nl = np.int64(src[q + 2]) | (np.int64(src[q + 3]) << 8)
    if ln != (~nl & 0xFFFF):
        return False
    return q + 4 + ln <= pend


@njit(cache=True, nogil=True)
def _find(src, lo, pend, b, lim, tab, iws):
    """The first sync point at a bit in [b, lim): (kind, bit), or (-1, -1).
    Kind 0 is a dynamic header at that bit, kind 1 a stored block whose LEN
    is at byte bit/8. Most bits are dismissed by one table lookup on 13 bits
    and a Kraft sum over the code-length code."""
    u = np.uint64
    while b < lim:
        q = b >> 3
        if q + 24 <= pend:
            w = ld64(src, q) >> u(b & 7)
            if _HLUT[np.int64(w & u(0x1FFF))]:
                ncl = np.int64((w >> u(13)) & u(15)) + 4
                b2 = b + 17
                w2 = ld64(src, b2 >> 3) >> u(b2 & 7)
                kr = 0
                for i in range(ncl):
                    L = np.int64((w2 >> u(3 * i)) & u(7))
                    if L:
                        kr += 128 >> L
                if kr == 128 and _dyn_ok(src, pend, b, tab, iws):
                    return 0, b
        elif _dyn_ok(src, pend, b, tab, iws):
            return 0, b
        if (b & 7) == 0 and _stored_ok(src, lo, pend, q):
            return 1, b
        b += 1
    return -1, -1


# ---------------------------------------------------------------- decoding

@njit(cache=True, nogil=True)
def _tables(src, pend, p, buf, bc, btype, tab, iws):
    """The tables of a fixed (btype 1) or dynamic (2) block whose 3 header
    bits are consumed. Returns (0 or ERR_DATA, p, buf, bc)."""
    u = np.uint64
    if btype == 1:
        _fixed(tab, iws)
        return 0, p, buf, bc
    lens = iws[320:640]
    CM = u((1 << CBITS) - 1)
    while bc <= 56:
        if p < pend:
            buf |= u(src[p]) << u(bc)
        p += 1
        bc += 8
    nlit = np.int64(buf & u(31)) + 257
    ndist = np.int64((buf >> u(5)) & u(31)) + 1
    ncl = np.int64((buf >> u(10)) & u(15)) + 4
    buf >>= u(14)
    bc -= 14
    if nlit > 286 or ndist > 30:
        return ERR_DATA, p, buf, bc
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
    if _build(lens, 0, 19, tab, LSIZE + DSIZE, CSIZE, CBITS, 2, iws) < 0:
        return ERR_DATA, p, buf, bc
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
            return ERR_DATA, p, buf, bc
        buf >>= u(cl)
        bc -= cl
        sym = e >> 16
        if sym < 16:
            lens[i] = sym
            i += 1
            continue
        if sym == 16:
            if i == 0:
                return ERR_DATA, p, buf, bc
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
            return ERR_DATA, p, buf, bc
        for k in range(rep):
            lens[i + k] = v
        i += rep
    if lens[256] == 0:
        return ERR_DATA, p, buf, bc
    if _build(lens, 0, nlit, tab, 0, LSIZE, LBITS, 0, iws) < 0:
        return ERR_DATA, p, buf, bc
    if _build(lens, nlit, ndist, tab, LSIZE, DSIZE, DBITS, 1, iws) < 0:
        return ERR_DATA, p, buf, bc
    return 0, p, buf, bc


@njit(cache=True, nogil=True)
def _sym8(src, pend, p, buf, bc, dst, dlo, o, oend, ostop, tab):
    """The current block's symbols into dst[o:oend] (matches reach back to
    dlo at most): `_inflate.inflate`'s fast loop, then its careful one.
    Returns (0 at the block's end / END_STOP / an error, p, buf, bc, o)."""
    u = np.uint64
    LM = u((1 << LBITS) - 1)
    DM = u((1 << DBITS) - 1)
    while p + 16 <= pend and o + 274 <= oend and o < ostop:
        buf |= ld64(src, p) << u(bc)
        p += (63 - bc) >> 3
        bc |= 56
        e = np.int64(tab[np.int64(buf & LM)])
        if ((e >> 5) & 7) == K_LIT:
            cl = e & 31
            buf >>= u(cl)
            bc -= cl
            dst[o] = e >> 16
            o += 1
            e = np.int64(tab[np.int64(buf & LM)])
            if ((e >> 5) & 7) == K_LIT:
                cl = e & 31
                buf >>= u(cl)
                bc -= cl
                dst[o] = e >> 16
                o += 1
                e = np.int64(tab[np.int64(buf & LM)])
                if ((e >> 5) & 7) == K_LIT:
                    cl = e & 31
                    buf >>= u(cl)
                    bc -= cl
                    dst[o] = e >> 16
                    o += 1
                    continue
            buf |= ld64(src, p) << u(bc)
            p += (63 - bc) >> 3
            bc |= 56
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
            dst[o] = e >> 16
            o += 1
            continue
        if kind != K_LEN:
            if kind == K_EOB:
                return 0, p, buf, bc, o
            return ERR_DATA, p, buf, bc, o
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
            return ERR_DATA, p, buf, bc, o
        cl = e & 31
        buf >>= u(cl)
        bc -= cl
        ex = (e >> 8) & 31
        dist = (e >> 16) + np.int64(buf & u((1 << ex) - 1))
        buf >>= u(ex)
        bc -= ex
        s = o - dist
        if s < dlo:
            return ERR_DATA, p, buf, bc, o
        if dist >= 16:
            st64(dst, o, ld64(dst, s))
            st64(dst, o + 8, ld64(dst, s + 8))
            k = 16
            while k < length:
                st64(dst, o + k, ld64(dst, s + k))
                st64(dst, o + k + 8, ld64(dst, s + k + 8))
                k += 16
        elif dist >= 8:
            k = 0
            while k < length:
                st64(dst, o + k, ld64(dst, s + k))
                k += 8
        elif dist == 1:
            w = u(dst[s]) * u(0x0101010101010101)
            k = 0
            while k < length:
                st64(dst, o + k, w)
                k += 8
        else:
            for k in range(length):
                dst[o + k] = dst[s + k]
        o += length
    if o >= ostop:
        return END_STOP, p, buf, bc, o
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
                return ERR_SPACE, p, buf, bc, o
            dst[o] = e >> 16
            o += 1
            if o >= ostop and ostop < oend:         # (a full decode goes on: more is an error)
                return END_STOP, p, buf, bc, o
            continue
        if kind != K_LEN:
            if kind == K_EOB:
                return 0, p, buf, bc, o
            return ERR_DATA, p, buf, bc, o
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
            return ERR_DATA, p, buf, bc, o
        cl = e & 31
        buf >>= u(cl)
        bc -= cl
        ex = (e >> 8) & 31
        dist = (e >> 16) + np.int64(buf & u((1 << ex) - 1))
        buf >>= u(ex)
        bc -= ex
        s = o - dist
        if s < dlo:
            return ERR_DATA, p, buf, bc, o
        if o + length > oend:
            return ERR_SPACE, p, buf, bc, o
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
            return END_STOP, p, buf, bc, o


@njit(cache=True, nogil=True)
def _match16(w, base, m, s, length, lastm, cap):
    """w[m:m+length] = w[s:s+length], symbol by symbol as deflate means it;
    positions before base are the unknown window, written as markers (256 +
    window position). Four symbols a word when the distance allows and there
    is room for up to 3 symbols written past the match. Returns where the
    last marker is (or a later position: only the switch to bytes waits a
    little longer)."""
    if s >= base and m - s >= 4 and m + length + 4 <= cap:
        acc = np.uint64(0)
        k = 0
        while k < length:
            x = ld64(w, s + k)
            st64(w, m + k, x)
            acc |= x
            k += 4
        if acc & np.uint64(0xFF00FF00FF00FF00):
            lastm = m + length - 1
        return lastm
    for k in range(length):
        i = s + k
        if i < base:
            w[m + k] = i - base + (W + 256)
            lastm = m + k
        else:
            v = w[i]
            w[m + k] = v
            if v >= 256:
                lastm = m + k
    return lastm


@njit(cache=True, nogil=True)
def _sym16(src, pend, p, buf, bc, w, base, m, cap, lastm, tab):
    """The current block's symbols into w[m:cap] (uint16), w[base - W:base]
    being the unknown window (see _match16). lastm: where the last marker
    is. A fast loop like `_sym8`'s, then a careful one. Returns (0 at the
    block's end / SWITCH once the last 32 kB hold no marker / ROOM when the
    next symbol might not fit -- both between symbols, so decoding can go on
    elsewhere / an error, p, buf, bc, m, lastm)."""
    u = np.uint64
    LM = u((1 << LBITS) - 1)
    DM = u((1 << DBITS) - 1)
    while True:
        if m - 1 - lastm >= W:
            return SWITCH, p, buf, bc, m, lastm
        if m + 274 > cap:
            return ROOM, p, buf, bc, m, lastm
        if p + 16 <= pend:
            buf |= ld64(src, p) << u(bc)
            p += (63 - bc) >> 3
            bc |= 56
            e = np.int64(tab[np.int64(buf & LM)])
            if ((e >> 5) & 7) == K_LIT:     # up to three literals per refill
                cl = e & 31
                buf >>= u(cl)
                bc -= cl
                w[m] = e >> 16
                m += 1
                e = np.int64(tab[np.int64(buf & LM)])
                if ((e >> 5) & 7) == K_LIT:
                    cl = e & 31
                    buf >>= u(cl)
                    bc -= cl
                    w[m] = e >> 16
                    m += 1
                    e = np.int64(tab[np.int64(buf & LM)])
                    if ((e >> 5) & 7) == K_LIT:
                        cl = e & 31
                        buf >>= u(cl)
                        bc -= cl
                        w[m] = e >> 16
                        m += 1
                        continue
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
            w[m] = e >> 16
            m += 1
            continue
        if kind != K_LEN:
            if kind == K_EOB:
                return 0, p, buf, bc, m, lastm
            return ERR_DATA, p, buf, bc, m, lastm
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
            return ERR_DATA, p, buf, bc, m, lastm
        cl = e & 31
        buf >>= u(cl)
        bc -= cl
        ex = (e >> 8) & 31
        dist = (e >> 16) + np.int64(buf & u((1 << ex) - 1))
        buf >>= u(ex)
        bc -= ex
        s = m - dist
        if s < base - W:
            return ERR_DATA, p, buf, bc, m, lastm
        lastm = _match16(w, base, m, s, length, lastm, cap)
        m += length


@njit(cache=True, nogil=True)
def _header(src, lo, pend, p, buf, bc, first, T, tab, iws):
    """The next block's header, after a block (or at the start, first=True,
    when no sync point is looked for). Returns (stop, final, btype, hb, p,
    buf, bc): stop 1 when it is a sync point at or after bit T -- btype is
    then its kind (0 dynamic, 1 stored), hb its bit (8 * the LEN byte for a
    stored block). Else hb is the header's bit, and a stored block's LEN is
    at byte p (buf empty)."""
    u = np.uint64
    while bc <= 56:
        if p < pend:
            buf |= u(src[p]) << u(bc)
        p += 1
        bc += 8
    hb = 8 * p - bc
    if not first and hb >= T and (buf & u(7)) == u(4) and _dyn_ok(src, pend, hb, tab, iws):
        return 1, 0, 0, hb, p, buf, bc
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
        if not first and final == 0 and 8 * p >= T and _stored_ok(src, lo, pend, p):
            return 1, 0, 1, 8 * p, p, buf, bc
    return 0, final, btype, hb, p, buf, bc


@njit(cache=True, nogil=True)
def _decode_true(src, lo, pend, kind, sync, T, dst, dlo, o, oend, ostop, tab, iws, res):
    """The true decode into dst[o:oend] (matches reaching back to dlo at
    most), from a block start -- kind 0: a block header at bit `sync`;
    kind 1: a non-last stored block whose LEN is at byte sync/8 -- until the
    first sync point at or after bit T (not counting the start: END_BLOCK,
    res R_EK/R_ES), the stream's end (END_STREAM) or ostop (END_STOP).
    res[R_O]: the output end. Or an error."""
    u = np.uint64
    buf = u(0)
    bc = 0
    p = sync >> 3
    if kind == 0:
        while bc <= 56:
            if p < pend:
                buf |= u(src[p]) << u(bc)
            p += 1
            bc += 8
        buf >>= u(sync & 7)
        bc -= sync & 7
    first = True
    final = 0
    while final == 0:
        res[R_O] = o
        if o >= ostop:
            return END_STOP
        if first and kind == 1:
            btype = 0
        else:
            stop, final, btype, hb, p, buf, bc = _header(src, lo, pend, p, buf, bc, first, T,
                                                         tab, iws)
            if stop:
                res[R_EK] = btype
                res[R_ES] = hb
                return END_BLOCK
        first = False
        if btype == 0:
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
            _copy(dst, o, src, p, ln)
            o += ln
            p += ln
            continue
        if btype == 3:
            return ERR_DATA
        err, p, buf, bc = _tables(src, pend, p, buf, bc, btype, tab, iws)
        if err < 0:
            return ERR_DATA
        st, p, buf, bc, o = _sym8(src, pend, p, buf, bc, dst, dlo, o, oend, ostop, tab)
        res[R_O] = o
        if st == END_STOP:
            return END_STOP
        if st < 0:
            return st
        if p - (bc >> 3) > pend:
            return ERR_INPUT
    if p - (bc >> 3) > pend:
        return ERR_INPUT
    res[R_EBIT] = 8 * p - bc
    res[R_O] = o
    return END_STREAM


@njit(cache=True, nogil=True)
def _count(seg, k, typ, m, o):
    """Close region k: how many symbols / bytes it holds past its lead."""
    if typ == 1:
        seg[k, G_N] = m - (seg[k, G_OFF] >> 1) - seg[k, G_LEAD]
    else:
        seg[k, G_N] = o - seg[k, G_OFF] - seg[k, G_LEAD]


@njit(cache=True, nogil=True)
def _decode_spec(src, lo, pend, kind, sync, T, B8, W16, seg, k0, slab, sbase, ssize, nslab,
                 tab, iws, res):
    """Speculative decode, the 32 kB before unknown, from a block start
    (as `_decode_true`) to the first sync point at or after bit T or the
    stream's end. Output goes to regions of slabs (ssize bytes each at
    B8[sbase + k * ssize], k = k0 first, more taken off slab[0] as needed;
    W16 is the same memory as uint16), listed in seg[:res[R_NSEG]] as
    (offset, 1 = 16-bit symbols / 2 = bytes, lead, count): the output is
    each region's elements [lead:lead + count] in turn. First 16-bit
    symbols, a match into the unknown window writing a marker; from the
    first 32 kB free of markers on, bytes. A region continuing another
    begins with a copy of the latter's last 32 kB (its lead) so matches
    find them. Out of slabs: END_FULL, the output usable up to the start
    of the block it was in (res R_TK, R_TS: that start; R_TSEG, R_TCNT:
    the region and count there)."""
    u = np.uint64
    off = sbase + k0 * ssize
    rend = off + ssize
    seg[0, G_OFF] = off
    seg[0, G_TYP] = 1
    seg[0, G_LEAD] = 0
    seg[0, G_N] = 0
    nseg = 1
    base = off >> 1
    m = base
    cap = rend >> 1
    lastm = base - 1
    mode16 = True
    o = 0
    dlo = 0
    oend = 0
    buf = u(0)
    bc = 0
    p = sync >> 3
    if kind == 0:
        while bc <= 56:
            if p < pend:
                buf |= u(src[p]) << u(bc)
            p += 1
            bc += 8
        buf >>= u(sync & 7)
        bc -= sync & 7
    res[R_TK] = kind
    res[R_TS] = sync
    res[R_TSEG] = 0
    res[R_TCNT] = 0
    first = True
    final = 0
    while final == 0:
        if first and kind == 1:
            btype = 0
            hb = sync
        else:
            stop, final, btype, hb, p, buf, bc = _header(src, lo, pend, p, buf, bc, first, T,
                                                         tab, iws)
            if stop:
                res[R_EK] = btype
                res[R_ES] = hb
                _count(seg, nseg - 1, 1 if mode16 else 2, m, o)
                res[R_NSEG] = nseg
                return END_BLOCK
        # where to roll back to if the slabs run out within this block
        res[R_TK] = 1 if (first and kind == 1) else 0
        res[R_TS] = hb if not (first and kind == 1) else sync
        res[R_TSEG] = nseg - 1
        if mode16:
            res[R_TCNT] = m - (seg[nseg - 1, G_OFF] >> 1) - seg[nseg - 1, G_LEAD]
        else:
            res[R_TCNT] = o - seg[nseg - 1, G_OFF] - seg[nseg - 1, G_LEAD]
        first = False
        if btype == 3:
            return ERR_DATA
        if btype == 0:
            if p + 4 > pend:
                return ERR_INPUT
            ln = np.int64(src[p]) | (np.int64(src[p + 1]) << 8)
            nl = np.int64(src[p + 2]) | (np.int64(src[p + 3]) << 8)
            p += 4
            if ln != (~nl & 0xFFFF):
                return ERR_DATA
            if p + ln > pend:
                return ERR_INPUT
        else:
            err, p, buf, bc = _tables(src, pend, p, buf, bc, btype, tab, iws)
            if err < 0:
                return ERR_DATA
        while True:
            # room for a stored block, or the next symbol: else a new slab
            need = ln + 8 if btype == 0 else 0
            if mode16:
                st = 0
                if btype == 0:
                    if m + need > cap:
                        st = ROOM
                else:
                    st, p, buf, bc, m, lastm = _sym16(src, pend, p, buf, bc, W16, base, m, cap,
                                                      lastm, tab)
                    if st < 0:
                        return st
                if st == ROOM:
                    if nseg == MAXSEG:
                        return END_FULL
                    g = 1 << (nseg // 4)                # later regions: runs of 2, 4, 8 ... slabs
                    k = _take_n(slab, 0, g)
                    if k + g > nslab:
                        return END_FULL
                    _count(seg, nseg - 1, 1, m, o)
                    off = sbase + k * ssize
                    rend = off + g * ssize
                    base = off >> 1
                    for i in range(W):
                        W16[base + i] = W16[m - W + i]
                    if lastm >= m - W:
                        lastm = base + lastm - (m - W)
                    else:
                        lastm = base - 1
                    seg[nseg, G_OFF] = off
                    seg[nseg, G_TYP] = 1
                    seg[nseg, G_LEAD] = W
                    seg[nseg, G_N] = 0
                    nseg += 1
                    m = base + W
                    cap = rend >> 1
                    continue
                if btype == 0:
                    for i in range(ln):
                        W16[m + i] = src[p + i]
                    m += ln
                    p += ln
                    if m - 1 - lastm >= W:
                        st = SWITCH
                    else:
                        break
                if st == 0:
                    break
                # SWITCH: bytes from here, after the symbols if they fit there
                _count(seg, nseg - 1, 1, m, o)
                b0 = (2 * m + 7) & ~7
                if b0 + W + 65536 + 512 > rend:
                    if nseg == MAXSEG:
                        return END_FULL
                    g = 1 << (nseg // 4)
                    k = _take_n(slab, 0, g)
                    if k + g > nslab:
                        return END_FULL
                    b0 = sbase + k * ssize
                    rend = b0 + g * ssize
                elif nseg == MAXSEG:
                    return END_FULL
                for i in range(W):
                    B8[b0 + i] = W16[m - W + i]
                seg[nseg, G_OFF] = b0
                seg[nseg, G_TYP] = 2
                seg[nseg, G_LEAD] = W
                seg[nseg, G_N] = 0
                nseg += 1
                mode16 = False
                dlo = b0
                o = b0 + W
                oend = rend
                if btype == 0:
                    break
                continue
            # bytes
            if btype == 0:
                if o + need <= oend:
                    _copy(B8, o, src, p, ln)
                    o += ln
                    p += ln
                    break
                st = END_STOP
            else:
                st, p, buf, bc, o = _sym8(src, pend, p, buf, bc, B8, dlo, o, oend, oend - 274,
                                          tab)
                if st < 0:
                    return st
                if st == 0:
                    break
            # END_STOP: near the slab's end, between symbols
            if nseg == MAXSEG:
                return END_FULL
            g = 1 << (nseg // 4)
            k = _take_n(slab, 0, g)
            if k + g > nslab:
                return END_FULL
            _count(seg, nseg - 1, 2, m, o)
            off = sbase + k * ssize
            _copy(B8, off, B8, o - W, W)
            seg[nseg, G_OFF] = off
            seg[nseg, G_TYP] = 2
            seg[nseg, G_LEAD] = W
            seg[nseg, G_N] = 0
            nseg += 1
            dlo = off
            o = off + W
            oend = off + g * ssize
        if p - (bc >> 3) > pend:
            return ERR_INPUT
    if p - (bc >> 3) > pend:
        return ERR_INPUT
    res[R_EBIT] = 8 * p - bc
    _count(seg, nseg - 1, 1 if mode16 else 2, m, o)
    res[R_NSEG] = nseg
    return END_STREAM


# ---------------------------------------------------------------- the parallel steps

@njit(cache=True, nogil=True)
def _part(src, lo, pend, a, b, findmax, B8, W16, seg, slab, sbase, ssize, nslab, tab, iws,
          res):
    """One speculative part: the first sync point in [a, b) (searching at
    most findmax bits) from which decoding works, decoded to the first sync
    point at or after b. res[R_ST] -1 if there is none (or no slab)."""
    lim = min(b, a + findmax)
    x = a
    res[R_ST] = -1
    k0 = take(slab, 0)
    if k0 >= nslab:
        return
    while x < lim:
        k, g = _find(src, lo, pend, x, lim, tab, iws)
        if k < 0:
            return
        st = _decode_spec(src, lo, pend, k, g, b, B8, W16, seg, k0, slab, sbase, ssize, nslab,
                          tab, iws, res)
        if st == END_STREAM and ((res[R_EBIT] + 7) >> 3) + 4 + 16 < pend:
            st = -1                             # ended far from the input's end: a false start
        if st == END_BLOCK or st == END_STREAM or st == END_FULL:
            res[R_ST] = st
            res[R_GK] = k
            res[R_GS] = g
            return
        x = g + 1


@njit(cache=True, nogil=True)
def _copy(dst, d, a, s, n):
    """dst[d:d+n] = a[s:s+n] (not overlapping)."""
    k = 0
    while k + 8 <= n:
        st64(dst, d + k, ld64(a, s + k))
        k += 8
    while k < n:
        dst[d + k] = a[s + k]
        k += 1


@njit(cache=True, nogil=True)
def _resolve(dst, w, s, a, b, O, win, o0):
    """dst[O+a:O+b] from 16-bit symbols w[s+a:s+b]: bytes as they are, a
    marker 256 + i as the byte at dst[win + i] (the true window). ERR_DATA
    for a marker before the stream's output start o0 (a match reaching
    before the stream, as the true decode would find). Eight symbols at a
    time while they hold no marker."""
    u = np.uint64
    HI = u(0xFF00FF00FF00FF00)
    i = a
    while i < b:
        if i + 8 <= b:
            x0 = ld64(w, s + i)
            x1 = ld64(w, s + i + 4)
            if ((x0 | x1) & HI) == u(0):
                y0 = ((x0 & u(0xFF)) | ((x0 >> u(8)) & u(0xFF00)) | ((x0 >> u(16)) & u(0xFF0000))
                      | ((x0 >> u(24)) & u(0xFF000000)))
                y1 = ((x1 & u(0xFF)) | ((x1 >> u(8)) & u(0xFF00)) | ((x1 >> u(16)) & u(0xFF0000))
                      | ((x1 >> u(24)) & u(0xFF000000)))
                st64(dst, O + i, y0 | (y1 << u(32)))
                i += 8
                continue
        n = min(8, b - i)
        for k in range(i, i + n):
            v = np.int64(w[s + k])
            if v < 256:
                dst[O + k] = v
            else:
                x = win + v - 256
                if x < o0:
                    return ERR_DATA
                dst[O + k] = dst[x]
        i += n
    return 0


@njit(cache=True, nogil=True)
def _place_part(dst, B8, W16, seg, nreg, lastn, O, L, o0, JOBS, j0, s):
    """Part output (regions seg[:nreg], the last one's count taken as lastn)
    going to dst[O:O+L]: its last 32 kB written now, the rest queued as jobs
    from row j0. Returns 0 or ERR_DATA."""
    x = 0
    for k in range(nreg):
        n = lastn if k == nreg - 1 else seg[k, G_N]
        typ = seg[k, G_TYP]
        e0 = seg[k, G_OFF] + (seg[k, G_LEAD] if typ == 2 else 2 * seg[k, G_LEAD])
        cut = min(x + n, max(x, L - W))         # [x, cut) later, [cut, x + n) now
        if cut < x + n:
            if typ == 2:
                _copy(dst, O + cut, B8, e0 + cut - x, x + n - cut)
            elif _resolve(dst, W16, e0 >> 1, cut - x, n, O + x, O - W, o0) < 0:
                return ERR_DATA
        if cut > x:
            j = j0 + k
            JOBS[j, J_T] = J_COPY if typ == 2 else J_RESOLVE
            JOBS[j, J_DST] = O + x
            JOBS[j, J_SRC] = e0 if typ == 2 else e0 >> 1
            JOBS[j, J_LEN] = cut - x
            JOBS[j, J_WIN] = O - W
            JOBS[j, J_LO] = o0
            JOBS[j, J_STREAM] = s
        x += n
    return 0


@njit(cache=True, nogil=True)
def _chain(src, dsts, SI, PT, RES, SEG, B8, W16, JOBS, s, tab, iws):
    """Stream s's parts in order once all have run: use each part that
    started exactly where the verified decode reached, redo serially from
    there any that did not. Writes the redone stretches and every used
    part's last 32 kB into place (what the next part's markers and the
    redos read); queues the rest of the copying in JOBS."""
    dst = dsts[SI[s, SI_D]]
    lo = SI[s, SI_LO]
    pend = SI[s, SI_PEND]
    o0 = SI[s, SI_O0]
    oend = SI[s, SI_OEND]
    ostop = SI[s, SI_OSTOP]
    t0 = SI[s, SI_P0]
    npt = SI[s, SI_NP]
    E = PT[t0 + npt - 1, PT_B]
    tmp = np.zeros(NRES, np.int64)
    r = RES[t0]
    st = r[R_ST]
    O = r[R_O]
    kind = r[R_EK]
    sync = r[R_ES]
    j = 1
    while True:
        if st < 0:
            SI[s, SI_ST] = st
            return
        if st == END_STREAM or st == END_STOP or O >= ostop:
            SI[s, SI_ST] = 1
            SI[s, SI_O] = O
            return
        if sync >= E:                           # this round's range done
            SI[s, SI_K] = kind
            SI[s, SI_S] = sync
            SI[s, SI_O] = O
            return
        while PT[t0 + j, PT_B] <= sync:
            j += 1
        t = t0 + j
        r = RES[t]
        rs = r[R_ST]
        if ((rs == END_BLOCK or rs == END_STREAM or rs == END_FULL)
                and r[R_GK] == kind and r[R_GS] == sync):
            seg = SEG[t]
            if rs == END_FULL:
                nreg = r[R_TSEG] + 1
                lastn = r[R_TCNT]
            else:
                nreg = r[R_NSEG]
                lastn = seg[nreg - 1, G_N]
            L = lastn
            for k in range(nreg - 1):
                L += seg[k, G_N]
            if O + L <= oend:
                if _place_part(dst, B8, W16, seg, nreg, lastn, O, L, o0, JOBS, t * MAXSEG,
                               s) < 0:
                    SI[s, SI_ST] = ERR_DATA
                    return
                O += L
                SI[s, SI_NUSED] += 1
                if rs == END_BLOCK:
                    kind = r[R_EK]
                    sync = r[R_ES]
                    continue
                if rs == END_STREAM:
                    st = END_STREAM
                    continue
                kind = r[R_TK]                  # out of slabs: on from its last block start
                sync = r[R_TS]
        # the true decode from here to the next sync point after this part's end
        st = _decode_true(src, lo, pend, kind, sync, PT[t, PT_B], dst, o0, O, oend, ostop,
                          tab, iws, tmp)
        SI[s, SI_NREDO] += 1
        if st >= 0:
            SI[s, SI_REDOB] += tmp[R_O] - O
            O = tmp[R_O]
            kind = tmp[R_EK]
            sync = tmp[R_ES]


@njit(cache=True, nogil=True)
def _steps_ab(src, dsts, SI, PT, RES, SEG, DONE, B8, W16, JOBS, ctr, slab, sbase, ssize, nslab,
              findmax, tab, iws):
    """Take parts off ctr[0] and decode them -- part 0 of a stream truly,
    straight into its output; the others speculatively into slabs. The
    thread finishing a stream's last part walks its chain."""
    n = PT.shape[0]
    while True:
        t = take(ctr, 0)
        if t >= n:
            return
        s = PT[t, PT_S]
        res = RES[t]
        lo = SI[s, SI_LO]
        pend = SI[s, SI_PEND]
        if PT[t, PT_J] == 0:
            res[R_ST] = _decode_true(src, lo, pend, SI[s, SI_K], SI[s, SI_S], PT[t, PT_B],
                                     dsts[SI[s, SI_D]], SI[s, SI_O0], SI[s, SI_O], SI[s, SI_OEND],
                                     SI[s, SI_OSTOP], tab, iws, res)
        else:
            _part(src, lo, pend, PT[t, PT_A], PT[t, PT_B], findmax, B8, W16, SEG[t], slab, sbase,
                  ssize, nslab, tab, iws, res)
        if take(DONE, s) == SI[s, SI_NP] - 1:
            _chain(src, dsts, SI, PT, RES, SEG, B8, W16, JOBS, s, tab, iws)


@njit(cache=True, nogil=True)
def _step_c(dsts, B8, W16, JOBS, CUM, SI, ctr):
    """The queued copies and marker resolutions, PIECE at a time off ctr[0]."""
    total = CUM[CUM.shape[0] - 1]
    while True:
        k = take(ctr, 0)
        if k >= total:
            return
        jb = np.searchsorted(CUM, k, side='right') - 1
        a = (k - CUM[jb]) * PIECE
        b = min(a + PIECE, JOBS[jb, J_LEN])
        s = JOBS[jb, J_STREAM]
        dst = dsts[SI[s, SI_D]]
        if JOBS[jb, J_T] == J_COPY:
            _copy(dst, JOBS[jb, J_DST] + a, B8, JOBS[jb, J_SRC] + a, b - a)
        else:
            e = _resolve(dst, W16, JOBS[jb, J_SRC], a, b, JOBS[jb, J_DST], JOBS[jb, J_WIN],
                         JOBS[jb, J_LO])
            if e < 0:
                SI[s, SI_ST] = e


# ---------------------------------------------------------------- driver

MIN_PART = 1 << 18              # compressed bytes a part gets at least
PARTS_PER_THREAD = 2            # parts per thread (load balance across P and E cores)
FINDMAX = 1 << 22               # bits a part searches for its start at most (512 kB)
SLABF = 1.0                     # a slab: bytes for this many times a part's expected output
SLABS = 2.0                     # slabs a round has: bytes for this many times its expected output
BUDGET = 1 << 30                # scratch (address space) a round aims to stay under (one stream at least)
KEEP = 1 << 30                  # scratch kept between calls up to this size (only pages used count)
SLACK = 1.15                    # partial reads: extra share of input a first round covers

_TLS = threading.local()
_ARENA_LOCK = threading.Lock()
_ARENA = [None]                 # scratch kept for the next call


def _ws():
    ws = getattr(_TLS, 'ws', None)
    if ws is None:
        ws = _TLS.ws = workspace()
    return ws


def _arena(n):
    """n bytes of scratch (8-byte aligned): the cached one if big enough
    and free."""
    with _ARENA_LOCK:
        a = _ARENA[0]
        _ARENA[0] = None
    if a is None or a.size < n:
        a = None                                    # (the old one freed first)
        a = np.empty((max(n, 8) + 7) // 8, np.uint64).view(np.uint8)
    return a


def _keep(a):
    if a.size > KEEP:
        return
    with _ARENA_LOCK:
        if _ARENA[0] is None or _ARENA[0].size < a.size:
            _ARENA[0] = a


def release():
    """Drop the scratch kept between calls."""
    with _ARENA_LOCK:
        _ARENA[0] = None


def _plan(SI, act, T, ppt, min_part, budget, total):
    """This round's parts: for the active streams that fit the scratch
    budget (the first always), cuts at equal shares of the compressed
    input, about total / (T * ppt) bytes each (total: the compressed bytes
    of all the work the threads share, at least these streams'). A partial
    read's first round covers the input expected to give the output wanted
    (if that is under half), later ones the rest. Returns (PT, slab size,
    number of slabs); streams left out get SI_NP 0."""
    rows = []
    chosen = []
    need = 0
    for s in act:
        lo, pend, o0, oend, ostop, k, sync, O = (int(x) for x in SI[s, :SI_ST])
        E = 8 * pend
        ratio = (oend - o0) / max(1, pend - lo)            # output bytes per input byte
        if ostop < oend and SI[s, SI_RND] == 0 and 2 * (ostop - o0) < oend - o0:
            bits = 8 * (ostop - O) / max(ratio, 1e-9) * SLACK + 8 * 65536
            E = min(E, sync + int(bits))
        scratch = (SLABS + 1) * ratio * (E - sync) / 8
        if chosen and need + scratch > budget:
            SI[s, SI_NP] = 0
            continue
        chosen.append((s, E, ratio))
        need += scratch
        SI[s, SI_RND] += 1
    mine = sum(E - int(SI[s, SI_S]) for s, E, _ in chosen)
    target = max(8 * min_part, max(mine, 8 * total) / max(1, T * ppt))
    nspec = 0
    big = 0.0
    out = 0.0
    for s, E, ratio in chosen:
        sync = int(SI[s, SI_S])
        n = max(1, min(4 * T, int((E - sync) / target + 0.5)))
        SI[s, SI_P0] = len(rows)
        SI[s, SI_NP] = n
        cuts = [sync + (E - sync) * j // n for j in range(n)] + [E]
        for j in range(n):
            rows.append((s, j, cuts[j], cuts[j + 1]))
        nspec += n - 1
        if n > 1:
            big = max(big, ratio * (cuts[2] - cuts[1]) / 8)
            out += ratio * (E - cuts[1]) / 8
    ssize = (max(int(SLABF * big), 2 * (65536 + 1024)) + 2 * W + 63) & ~63
    nslab = nspec + int(SLABS * out / ssize) + 4 if nspec else 0    # a first slab each, then room
    return np.array(rows, np.int64).reshape(-1, NPT), ssize, nslab


def inflate_many(src, starts, ends, dst, o0s, oends, ostops=None, nthreads=None,
                 dst2=None, which=None, extra=None, total=0,
                 parts_per_thread=PARTS_PER_THREAD, min_part=MIN_PART, findmax=FINDMAX,
                 budget=BUDGET, info=None):
    """Inflate zlib streams src[starts[i]:ends[i]] into dst[o0s[i]:oends[i]]
    (disjoint ranges) on up to nthreads threads of the pool, each stream cut
    into parts that all decode at once. ostops[i] (default oends[i]): the
    output stream i needs; decoding may stop soon after it.

    dst2, which: a second output array, for streams with which[i] = 1.
    extra: a function each thread runs once its parts are done (in the
    first round): other work sharing the pool with the parts, e.g. the
    engine's smaller chunks. total: the compressed bytes of all that work
    together, which sizes the parts (a stream much smaller than its share
    of the threads is not cut).

    Returns an int64 array: each stream's output end, or a negative error
    code as `_inflate.zlib_inflate`'s. info: a list to append each round's
    tables and step times to, then the stream table (diagnostics)."""
    starts = np.asarray(starts, np.int64)
    ns = len(starts)
    SI = np.zeros((ns, NSI), np.int64)
    SI[:, SI_LO] = starts + 2
    SI[:, SI_PEND] = ends
    SI[:, SI_O0] = o0s
    SI[:, SI_OEND] = oends
    SI[:, SI_OSTOP] = oends if ostops is None else np.minimum(ostops, oends)
    SI[:, SI_S] = 8 * (starts + 2)
    SI[:, SI_O] = o0s
    if which is not None:
        SI[:, SI_D] = which
    for s in range(ns):
        p, pend = int(starts[s]), int(SI[s, SI_PEND])
        if pend - p < 2:
            SI[s, SI_ST] = ERR_INPUT
            continue
        cmf, flg = int(src[p]), int(src[p + 1])
        if (cmf & 15) != 8 or (cmf >> 4) > 7 or ((cmf << 8) | flg) % 31 != 0 or (flg & 32):
            SI[s, SI_ST] = ERR_DATA
    dsts = (dst, dst if dst2 is None else dst2)
    T = max(1, min(nthreads or _pool.threads(), _pool.threads()))
    pending = [extra]
    while True:
        act = [s for s in range(ns) if SI[s, SI_ST] == 0]
        if not act:
            break
        PT, ssize, nslab = _plan(SI, act, T, parts_per_thread, min_part, budget, total)
        B8 = _arena(ssize * nslab)
        W16 = B8.view(np.uint16)
        RES = np.zeros((len(PT), NRES), np.int64)
        SEG = np.zeros((len(PT), MAXSEG, NG), np.int64)
        DONE = np.zeros(ns, np.int64)
        JOBS = np.zeros((MAXSEG * len(PT), NJ), np.int64)
        ctr = np.zeros(1, np.int64)
        slab = np.zeros(1, np.int64)
        more = pending.pop() if pending else None

        def ab():
            tab, iws = _ws()
            _steps_ab(src, dsts, SI, PT, RES, SEG, DONE, B8, W16, JOBS, ctr, slab, 0, ssize,
                      nslab, findmax, tab, iws)
            if more is not None:
                more()
        t0 = time.perf_counter()
        _pool.share(ab, T if more is not None else min(T, len(PT)))
        t1 = time.perf_counter()
        pieces = np.where(JOBS[:, J_T] > 0, -(-JOBS[:, J_LEN] // PIECE), 0)
        CUM = np.concatenate([np.zeros(1, np.int64), np.cumsum(pieces)])
        if CUM[-1]:
            ctr[0] = 0
            _pool.share(lambda: _step_c(dsts, B8, W16, JOBS, CUM, SI, ctr), min(T, int(CUM[-1])))
        _keep(B8)
        total = 0                                   # later rounds: these streams alone
        if info is not None:
            info.append((PT, RES, SEG, int(slab[0]), nslab, ssize, t1 - t0,
                         time.perf_counter() - t1))
    if pending and pending[0] is not None:
        _pool.share(pending[0], T)
    if info is not None:
        info.append(SI)
    return np.where(SI[:, SI_ST] < 0, SI[:, SI_ST], SI[:, SI_O])


def warm():
    """Load (from numba's cache, or compile) every kernel with the argument
    types reads use: a small stream cut in two, from a read-only buffer as
    a file mapping is."""
    import zlib
    x = np.arange(1 << 17, dtype=np.uint32) * np.uint32(2654435761) >> np.uint32(24)
    z = zlib.compress(x.astype(np.uint8).tobytes(), 1)
    src = np.frombuffer(z, np.uint8)
    out = np.empty(1 << 17, np.uint8)
    inflate_parallel(src, 0, len(z), out, 0, len(out), nthreads=2, min_part=1 << 12)


def inflate_parallel(src, p, pend, dst, o0, oend, ostop=None, nthreads=None, **kw):
    """`_inflate.zlib_inflate(src, p, pend, dst, o0, oend, ostop, ...)` on
    up to nthreads threads: the output end, or a negative error code."""
    return int(inflate_many(src, [p], [pend], dst, [o0], [oend],
                            None if ostop is None else [ostop], nthreads, **kw)[0])
