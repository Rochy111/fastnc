"""LZ4 blocks, and the HDF5 lz4 filter (32004) that frames them, in numba.

The filter (HDFGroup hdf5_plugins `H5Zlz4.c`, as hdf5plugin ships it)
writes a chunk as

    uint64 BE  decoded size
    uint32 BE  block size (bytes of decoded data per block, the last one shorter)
    per block: uint32 BE compressed size, then that many bytes: an LZ4 block,
               or the block's bytes as they are when the size equals the
               decoded block size (LZ4 could not shrink it).

Blocks are independent (no match reaches into an earlier block). The same
block decoder serves bitshuffle (32008) and blosc's lz4 / lz4hc codecs.

Decoding is the usual sequence loop -- token, literal run, two-byte
offset, match -- with literals and matches copied eight bytes at a time
when there is room to overshoot inside the output range, and the bounds
checked once per sequence so that nothing outside dst[o:oend] is written
and nothing outside src[p:pend] is read.
"""
import numpy as np

from ._inflate import ERR_DATA, ERR_INPUT, ERR_SPACE
from ._intrin import ld64, st64
from ._jit import njit


@njit(cache=True, nogil=True)
def copy_bytes(src, s, dst, d, n):
    """dst[d:d+n] = src[s:s+n] (non-overlapping, or dst before src)."""
    k = 0
    while k + 8 <= n:
        st64(dst, d + k, ld64(src, s + k))
        k += 8
    while k < n:
        dst[d + k] = src[s + k]
        k += 1


@njit(cache=True, nogil=True)
def copy_match(dst, o, dist, n, oend):
    """dst[o:o+n] = the n bytes starting dist back (an LZ77 match, which may
    overlap its own output). Writes nothing at or past oend; the caller has
    checked o + n <= oend and dist <= o - (start of the window)."""
    if o + n + 32 <= oend:
        _copy_fast(dst, o, dist, n)
        return
    m = oend - 32 - o                           # the bulk, with room to overshoot
    if m > 0:
        _copy_fast(dst, o, dist, m)
    else:
        m = 0
    s = o - dist
    for k in range(m, n):
        dst[o + k] = dst[s + k]


@njit(cache=True, nogil=True)
def _copy_fast(dst, o, dist, n):
    """copy_match with at least 32 bytes of room after o + n (it writes up
    to 31 bytes past the match).

    Long matches at short distances (runs of a byte or of a few-byte
    pattern: constant and periodic data) are written from registers
    holding 32 bytes of the pattern, so that no load reads bytes stored a
    moment before (a load straddling two recent stores stalls for a dozen
    cycles: copying such a match a word at a time runs at a third of the
    speed)."""
    s = o - dist
    if dist >= 32:
        k = 0
        while k < n:
            a = ld64(dst, s + k)
            b = ld64(dst, s + k + 8)
            c = ld64(dst, s + k + 16)
            d = ld64(dst, s + k + 24)
            st64(dst, o + k, a)
            st64(dst, o + k + 8, b)
            st64(dst, o + k + 16, c)
            st64(dst, o + k + 24, d)
            k += 32
        return
    if n >= 48 or dist == 1:
        if dist == 1:
            v0 = np.uint64(dst[s]) * np.uint64(0x0101010101010101)
            v1 = v0
            v2 = v0
            v3 = v0
            step = 32
        else:
            # 32 bytes of the pattern, then stores advancing a whole
            # number of periods each time
            v0 = np.uint64(0)
            v1 = np.uint64(0)
            v2 = np.uint64(0)
            v3 = np.uint64(0)
            j = 0
            for i in range(32):
                x = np.uint64(dst[s + j]) << np.uint64(8 * (i & 7))
                if i < 8:
                    v0 |= x
                elif i < 16:
                    v1 |= x
                elif i < 24:
                    v2 |= x
                else:
                    v3 |= x
                j += 1
                if j == dist:
                    j = 0
            step = dist * (32 // dist)
        k = 0
        while k < n:
            st64(dst, o + k, v0)
            st64(dst, o + k + 8, v1)
            st64(dst, o + k + 16, v2)
            st64(dst, o + k + 24, v3)
            k += step
        return
    if dist >= 8:
        k = 0
        while k < n:
            st64(dst, o + k, ld64(dst, s + k))
            k += 8
        return
    # a short period: the first 8 bytes one by one, then words from the
    # smallest multiple of the period that is at least 8 back
    for k in range(8):
        dst[o + k] = dst[s + k]
    step = dist * ((8 + dist - 1) // dist)
    k = 8
    while k < n:
        st64(dst, o + k, ld64(dst, o + k - step))
        k += 8
    return


@njit(cache=True, nogil=True)
def wild_copy(src, s, dst, d, n):
    """dst[d:d+n] = src[s:s+n], 32 bytes at a time: reads and writes up to
    31 bytes past the ends (the caller has the room); non-overlapping, or
    dst at least 32 bytes before or after src."""
    k = 0
    while k < n:
        a = ld64(src, s + k)
        b = ld64(src, s + k + 8)
        c = ld64(src, s + k + 16)
        e = ld64(src, s + k + 24)
        st64(dst, d + k, a)
        st64(dst, d + k + 8, b)
        st64(dst, d + k + 16, c)
        st64(dst, d + k + 24, e)
        k += 32


@njit(cache=True, nogil=True)
def lz4_block(src, p, pend, dst, o0, o, oend):
    """Decode the LZ4 block src[p:pend] into dst from o on, matches reaching
    back no further than o0, writing nothing at or past oend. Returns the
    end of the output, or a negative error code.

    While far from both ends (one check per sequence), a sequence of at
    most 14 literals and a match of at most 18 bytes from 8 or more back --
    most of them -- is copied with five word loads and stores and no
    other branch, as LZ4's own decoder does; the rest of the block goes
    through the careful loop."""
    pf = pend - 32
    of = oend - 64
    while p <= pf and o <= of:
        tok = np.int64(src[p])
        p += 1
        ll = tok >> 4
        if ll != 15:
            st64(dst, o, ld64(src, p))
            st64(dst, o + 8, ld64(src, p + 8))
            p += ll
            o += ll
        else:
            while True:
                if p >= pend:
                    return ERR_INPUT
                b = np.int64(src[p])
                p += 1
                ll += b
                if b != 255:
                    break
            if ll > pend - p:
                return ERR_INPUT
            if ll > oend - o:
                return ERR_SPACE
            if p + ll + 32 <= pend and o + ll + 32 <= oend:
                wild_copy(src, p, dst, o, ll)
            else:
                copy_bytes(src, p, dst, o, ll)
            p += ll
            o += ll
            if p >= pend:
                return o                        # the last sequence: literals only
            if p + 2 > pend:
                return ERR_INPUT
        dist = np.int64(ld64(src, p) & np.uint64(0xFFFF))   # (p + 8 <= pend here)
        p += 2
        ml = tok & 15
        if ml != 15 and dist >= 8 and dist <= o - o0:  # (o + 24 <= oend here)
            s = o - dist
            st64(dst, o, ld64(dst, s))
            st64(dst, o + 8, ld64(dst, s + 8))
            st64(dst, o + 16, ld64(dst, s + 16))
            o += ml + 4
            continue
        if ml == 15:
            while True:
                if p >= pend:
                    return ERR_INPUT
                b = np.int64(src[p])
                p += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        if dist == 0 or dist > o - o0:
            return ERR_DATA
        if ml > oend - o:
            return ERR_SPACE
        copy_match(dst, o, dist, ml, oend)
        o += ml
    # near the ends: every length checked
    while True:
        if p >= pend:
            return ERR_INPUT
        tok = np.int64(src[p])
        p += 1
        ll = tok >> 4
        if ll == 15:
            while True:
                if p >= pend:
                    return ERR_INPUT
                b = np.int64(src[p])
                p += 1
                ll += b
                if b != 255:
                    break
        if ll > pend - p:
            return ERR_INPUT
        if ll > oend - o:
            return ERR_SPACE
        copy_bytes(src, p, dst, o, ll)
        p += ll
        o += ll
        if p >= pend:
            return o                            # the last sequence: literals only
        if p + 2 > pend:
            return ERR_INPUT
        dist = np.int64(src[p]) | (np.int64(src[p + 1]) << 8)
        p += 2
        ml = tok & 15
        if ml == 15:
            while True:
                if p >= pend:
                    return ERR_INPUT
                b = np.int64(src[p])
                p += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        if dist == 0 or dist > o - o0:
            return ERR_DATA
        if ml > oend - o:
            return ERR_SPACE
        copy_match(dst, o, dist, ml, oend)
        o += ml


@njit(cache=True, nogil=True)
def be32(src, p):
    return ((np.int64(src[p]) << 24) | (np.int64(src[p + 1]) << 16)
            | (np.int64(src[p + 2]) << 8) | np.int64(src[p + 3]))


@njit(cache=True, nogil=True)
def be64(src, p):
    """src[p:p+8] big-endian, or -1 if it does not fit an int64."""
    if src[p] >= 128:
        return -1
    return (be32(src, p) << 32) | be32(src, p + 4)


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """The HDF5 lz4 filter: src[p:pend] (header and blocks) into dst[o:oend].
    Returns the end of the output or a negative error code."""
    if pend - p < 12:
        return ERR_INPUT
    total = be64(src, p)
    bs = be32(src, p + 8)
    if total < 0:
        return ERR_DATA
    if total > oend - o:
        return ERR_SPACE
    if bs > total:
        bs = total
    if bs <= 0 and total > 0:
        return ERR_DATA
    q = p + 12
    d = o
    end = o + total
    while d < end:
        if end - d < bs:
            bs = end - d
        if q + 4 > pend:
            return ERR_INPUT
        cb = be32(src, q)
        q += 4
        if cb > pend - q:
            return ERR_INPUT
        if cb == bs:
            copy_bytes(src, q, dst, d, bs)
        else:
            r = lz4_block(src, q, q + cb, dst, d, d, d + bs)
            if r < 0:
                return r
            if r != d + bs:
                return ERR_DATA
        q += cb
        d += bs
    return end


def workspace_size(chunk_nbytes):
    """No scratch: blocks decode straight into the output."""
    return 0
