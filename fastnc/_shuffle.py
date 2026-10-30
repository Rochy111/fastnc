"""HDF5's byte shuffle (filter 2) undone: H5Zshuffle.c's reverse pass.

The filter stores the first byte of every element, then every second
byte, and so on; the bytes left over when the chunk is not a whole number
of elements (`nbytes % size`) follow unchanged. Chunks of one-byte
elements, or of fewer than two elements, are not shuffled at all.

`unshuffle_into` is the kernel, for whoever has the shuffled bytes at hand
(the deflate path fuses it after inflating); `decode` is it behind the
codec contract. Element sizes 2, 4 and 8 have loops of their own, which
LLVM vectorizes (2, 4) or which transpose 8x8 bytes in registers (8).
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
    for chunks it leaves alone). src and dst must not overlap.

    Byte stores at unsigned indices: numba then adds no negative-index
    fix-up, and LLVM vectorizes the 2- and 4-byte loops (22-55 GB/s here,
    against 2.5 for the same loops with signed indices); 8-byte elements
    go eight at a time through a SWAR transpose (27-38 GB/s)."""
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
    M = u(m)
    S = u(s0)
    D = u(d0)
    if esize == 2:
        for j in range(M):
            dst[D + u(2) * j] = src[S + j]
            dst[D + u(2) * j + u(1)] = src[S + M + j]
    elif esize == 4:
        for j in range(M):
            dst[D + u(4) * j] = src[S + j]
            dst[D + u(4) * j + u(1)] = src[S + M + j]
            dst[D + u(4) * j + u(2)] = src[S + u(2) * M + j]
            dst[D + u(4) * j + u(3)] = src[S + u(3) * M + j]
    elif esize == 8:
        j = 0
        while j + 8 <= m:                       # an 8x8 byte transpose in registers (SWAR)
            r0 = ld64(src, s0 + j)
            r1 = ld64(src, s0 + m + j)
            r2 = ld64(src, s0 + 2 * m + j)
            r3 = ld64(src, s0 + 3 * m + j)
            r4 = ld64(src, s0 + 4 * m + j)
            r5 = ld64(src, s0 + 5 * m + j)
            r6 = ld64(src, s0 + 6 * m + j)
            r7 = ld64(src, s0 + 7 * m + j)
            M1 = u(0x00FF00FF00FF00FF)
            t0 = (r0 & M1) | ((r1 & M1) << u(8))        # 16-bit units: columns 0,2,4,6 of rows 0-1
            t1 = ((r0 >> u(8)) & M1) | (r1 & ~M1)       # columns 1,3,5,7 of rows 0-1
            t2 = (r2 & M1) | ((r3 & M1) << u(8))
            t3 = ((r2 >> u(8)) & M1) | (r3 & ~M1)
            t4 = (r4 & M1) | ((r5 & M1) << u(8))
            t5 = ((r4 >> u(8)) & M1) | (r5 & ~M1)
            t6 = (r6 & M1) | ((r7 & M1) << u(8))
            t7 = ((r6 >> u(8)) & M1) | (r7 & ~M1)
            M2 = u(0x0000FFFF0000FFFF)
            q0 = (t0 & M2) | ((t2 & M2) << u(16))       # 32-bit units: columns 0,4 of rows 0-3
            q2 = ((t0 >> u(16)) & M2) | (t2 & ~M2)      # columns 2,6
            q1 = (t1 & M2) | ((t3 & M2) << u(16))       # columns 1,5
            q3 = ((t1 >> u(16)) & M2) | (t3 & ~M2)      # columns 3,7
            q4 = (t4 & M2) | ((t6 & M2) << u(16))       # (rows 4-7)
            q6 = ((t4 >> u(16)) & M2) | (t6 & ~M2)
            q5 = (t5 & M2) | ((t7 & M2) << u(16))
            q7 = ((t5 >> u(16)) & M2) | (t7 & ~M2)
            M4 = u(0x00000000FFFFFFFF)
            o = d0 + 8 * j
            st64(dst, o, (q0 & M4) | (q4 << u(32)))
            st64(dst, o + 8, (q1 & M4) | (q5 << u(32)))
            st64(dst, o + 16, (q2 & M4) | (q6 << u(32)))
            st64(dst, o + 24, (q3 & M4) | (q7 << u(32)))
            st64(dst, o + 32, (q0 >> u(32)) | (q4 & ~M4))
            st64(dst, o + 40, (q1 >> u(32)) | (q5 & ~M4))
            st64(dst, o + 48, (q2 >> u(32)) | (q6 & ~M4))
            st64(dst, o + 56, (q3 >> u(32)) | (q7 & ~M4))
            j += 8
        for jj in range(u(j), M):
            for i in range(u(8)):
                dst[D + u(8) * jj + i] = src[S + i * M + jj]
    else:
        E = u(esize)
        for j in range(M):
            for i in range(E):
                dst[D + E * j + i] = src[S + i * M + j]
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
