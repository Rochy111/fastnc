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
    s = o - dist
    if o + n + 8 <= oend:
        if dist >= 8:
            k = 0
            while k < n:
                st64(dst, o + k, ld64(dst, s + k))
                k += 8
            return
        if dist == 1:
            v = ld64(dst, s) & np.uint64(0xFF)
            v *= np.uint64(0x0101010101010101)
            k = 0
            while k < n:
                st64(dst, o + k, v)
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
    for k in range(n):
        dst[o + k] = dst[s + k]


@njit(cache=True, nogil=True)
def lz4_block(src, p, pend, dst, o0, o, oend):
    """Decode the LZ4 block src[p:pend] into dst from o on, matches reaching
    back no further than o0, writing nothing at or past oend. Returns the
    end of the output, or a negative error code."""
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
        if ll <= 16 and p + 16 <= pend and o + 16 <= oend:
            st64(dst, o, ld64(src, p))          # short run: two words, overshooting
            st64(dst, o + 8, ld64(src, p + 8))
        else:
            if p + ll > pend:
                return ERR_INPUT
            if o + ll > oend:
                return ERR_SPACE
            copy_bytes(src, p, dst, o, ll)
        p += ll
        o += ll
        if p >= pend:
            if p > pend:
                return ERR_INPUT
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
        if o + ml > oend:
            return ERR_SPACE
        if dist >= 8 and ml <= 16 and o + 16 <= oend:
            s = o - dist
            st64(dst, o, ld64(dst, s))
            st64(dst, o + 8, ld64(dst, s + 8))
        else:
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
