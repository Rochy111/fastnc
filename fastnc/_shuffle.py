"""HDF5's byte shuffle (filter 2) undone: H5Zshuffle.c's reverse pass.

The filter stores the first byte of every element, then every second
byte, and so on; the bytes left over when the chunk is not a whole number
of elements (`nbytes % size`) follow unchanged. Chunks of one-byte
elements, or of fewer than two elements, are not shuffled at all.

`unshuffle_into` is the kernel, for whoever has the shuffled bytes at hand
(the deflate path fuses it after inflating); `decode` is it behind the
codec contract. Element sizes 2, 4 and 8 have loops of their own, which
gather a word from the byte planes and store it whole -- LLVM turns them
into vector shuffles; other sizes go plane by plane.
"""
import numpy as np

from ._intrin import ld64, st64
from ._jit import njit

ERR_DATA = -1
ERR_SPACE = -2
ERR_INPUT = -3


@njit(cache=True, nogil=True)
def unshuffle_into(src, s0, n, esize, dst, d0):
    """The n shuffled bytes at src[s0:s0+n], elements of esize bytes, into
    dst[d0:d0+n] unshuffled (H5Zshuffle.c's rule for leftover bytes and
    for chunks it leaves alone). src and dst must not overlap."""
    m = n // esize if esize > 0 else 0
    if esize <= 1 or m <= 1:
        k = 0
        while k + 8 <= n:
            st64(dst, d0 + k, ld64(src, s0 + k))
            k += 8
        while k < n:
            dst[d0 + k] = src[s0 + k]
            k += 1
        return
    u = np.uint64
    if esize == 2:
        a = s0
        b = s0 + m
        j = 0
        while j + 4 <= m:                       # four elements, one store
            w = (u(src[a + j]) | (u(src[b + j]) << u(8))
                 | (u(src[a + j + 1]) << u(16)) | (u(src[b + j + 1]) << u(24))
                 | (u(src[a + j + 2]) << u(32)) | (u(src[b + j + 2]) << u(40))
                 | (u(src[a + j + 3]) << u(48)) | (u(src[b + j + 3]) << u(56)))
            st64(dst, d0 + 2 * j, w)
            j += 4
        while j < m:
            dst[d0 + 2 * j] = src[a + j]
            dst[d0 + 2 * j + 1] = src[b + j]
            j += 1
    elif esize == 4:
        a = s0
        b = s0 + m
        c = s0 + 2 * m
        e = s0 + 3 * m
        j = 0
        while j + 2 <= m:                       # two elements, one store
            w = (u(src[a + j]) | (u(src[b + j]) << u(8))
                 | (u(src[c + j]) << u(16)) | (u(src[e + j]) << u(24))
                 | (u(src[a + j + 1]) << u(32)) | (u(src[b + j + 1]) << u(40))
                 | (u(src[c + j + 1]) << u(48)) | (u(src[e + j + 1]) << u(56)))
            st64(dst, d0 + 4 * j, w)
            j += 2
        while j < m:
            dst[d0 + 4 * j] = src[a + j]
            dst[d0 + 4 * j + 1] = src[b + j]
            dst[d0 + 4 * j + 2] = src[c + j]
            dst[d0 + 4 * j + 3] = src[e + j]
            j += 1
    elif esize == 8:
        for j in range(m):
            w = (u(src[s0 + j]) | (u(src[s0 + m + j]) << u(8))
                 | (u(src[s0 + 2 * m + j]) << u(16)) | (u(src[s0 + 3 * m + j]) << u(24))
                 | (u(src[s0 + 4 * m + j]) << u(32)) | (u(src[s0 + 5 * m + j]) << u(40))
                 | (u(src[s0 + 6 * m + j]) << u(48)) | (u(src[s0 + 7 * m + j]) << u(56)))
            st64(dst, d0 + 8 * j, w)
    else:
        for i in range(esize):
            s = s0 + i * m
            d = d0 + i
            for j in range(m):
                dst[d + j * esize] = src[s + j]
    for k in range(m * esize, n):               # leftover bytes, as they are
        dst[d0 + k] = src[s0 + k]


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """Unshuffle src[p:pend] into dst[o:...] (cd: [element size, chunk
    bytes, 1, shuffle's element size]). The end of the output, or
    ERR_SPACE if it does not fit."""
    n = pend - p
    if n < 0:
        return ERR_INPUT
    if n > oend - o:
        return ERR_SPACE
    esize = cd[3] if cd[2] >= 1 else cd[0]
    unshuffle_into(src, p, n, esize, dst, o)
    return o + n


def workspace_size(chunk_nbytes):
    """No scratch needed."""
    return 0
