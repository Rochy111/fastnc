"""HDF5's scale-offset filter (filter 6) undone, as H5Zscaleoffset.c's
reverse pass does it, integer and floating-point (D-scaling) alike.

A chunk is a 21-byte header -- minbits (4 bytes, little-endian), the size
of minval in bytes (one byte, 8), minval (little-endian), zeros -- then
every element's minbits low bits, most significant bit first, packed with
no padding. cd values (20): scale type (0 float D-scale, 1 E-scale, never
implemented, 2 integer), scale factor (D, a signed 32-bit int; for
integers the minbits asked for), elements in a chunk, class (0 integer,
1 float), size, sign (integers: 1 two's complement), byte order (0 LE,
1 BE), fill value defined, then the fill value's bytes, in the order of
the machine that wrote the file (little-endian), from cd[8] on.

Decoding, element by element:

- minbits equal to the type's bits: the elements stored as they are, in
  little-endian order (HDF5 converts to its native order before encoding);
- minbits 0: every element's bits are 0 (all equal; no data follows);
- integers: x + minval, in the type's width (wrapping); with a fill value
  defined the all-ones code (2^minbits - 1) is the fill value instead;
- floats: as the C expression HDF5 evaluates, `(type)i / pow_fun(10, D) +
  min` with powf for float and pow for double: the integer converted to the
  float type, divided by 10^D and min (minval's low bytes as a float) added,
  all in the float type; with a fill value defined the all-ones code is the
  fill value.

Then the elements are written in the dataset's byte order.
"""
import math

import numpy as np

from ._intrin import ld64, st64
from ._jit import njit
from ._szip import be64, getbits

ERR_DATA = -1
ERR_SPACE = -2
ERR_INPUT = -3

HEADER = 21                     # minbits 4, minval size 1, minval 16 (8 used)
SO_FLOAT_DSCALE, SO_FLOAT_ESCALE, SO_INT = 0, 1, 2


@njit(cache=True, nogil=True)
def _put(dst, at, w, size, be):
    """w's low `size` bytes at dst[at:], little- or big-endian."""
    if be:
        for t in range(size):
            dst[at + t] = (w >> np.uint64(8 * (size - 1 - t))) & np.uint64(255)
    elif size == 8:
        st64(dst, at, w)
    else:
        for t in range(size):
            dst[at + t] = (w >> np.uint64(8 * t)) & np.uint64(255)


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """A scale-offset chunk at src[p:pend] into dst[o:...]; cd: [element
    size, chunk bytes, n, the filter's 20 cd values]. The end of the output
    or a negative error code."""
    u = np.uint64
    if cd[2] < 8:
        return ERR_DATA
    stype = cd[3]
    D = cd[4]
    if D >= 2 ** 31:                            # (a signed int stored unsigned)
        D -= 2 ** 32
    nel = cd[5]
    cls = cd[6]
    size = cd[7]
    be = cd[9] == 1
    filavail = cd[10] == 1
    if cls == 0:
        if stype != SO_INT or size not in (1, 2, 4, 8):
            return ERR_DATA
    elif cls == 1:
        if stype != SO_FLOAT_DSCALE or size not in (4, 8):
            return ERR_DATA
    else:
        return ERR_DATA
    nout = nel * size
    if nout > oend - o:
        return ERR_SPACE
    if pend - p < HEADER:
        return ERR_INPUT
    minbits = (np.int64(src[p]) | (np.int64(src[p + 1]) << 8) | (np.int64(src[p + 2]) << 16)
               | (np.int64(src[p + 3]) << 24))
    if minbits > size * 8:
        return ERR_DATA
    nmin = min(8, np.int64(src[p + 4]))
    minval = u(0)
    for t in range(nmin):
        minval |= u(src[p + 5 + t]) << u(8 * t)
    q0 = p + HEADER
    if minbits == size * 8:                     # full precision: stored as is (little-endian)
        if pend - q0 < nout:
            return ERR_INPUT
        if not be or size == 1:
            k = 0
            while k + 8 <= nout:
                st64(dst, o + k, ld64(src, q0 + k))
                k += 8
            while k < nout:
                dst[o + k] = src[q0 + k]
                k += 1
        else:
            for i in range(nel):
                for t in range(size):
                    dst[o + i * size + t] = src[q0 + i * size + size - 1 - t]
        return o + nout
    if (nel * minbits + 7) // 8 > pend - q0:
        return ERR_INPUT
    nfill = 0
    if filavail:
        nfill = (size + 3) // 4
        if cd[2] < 8 + nfill:
            return ERR_DATA
    filval = u(0)
    for t in range(size * filavail):            # cd[8:]'s bytes, little-endian words
        filval |= u((cd[11 + t // 4] >> (8 * (t % 4))) & 255) << u(8 * t)
    allones = (u(1) << u(minbits)) - u(1)
    wmask = u(0xFFFFFFFFFFFFFFFF) if size == 8 else (u(1) << u(8 * size)) - u(1)
    if cls == 0:
        minv = minval & wmask
        if minbits == 0:
            w = filval if filavail else minv    # (the zeros all match the 0-bit fill code)
            for i in range(nel):
                _put(dst, o + i * size, w, size, be)
            return o + nout
        pos = q0 * 8
        for i in range(nel):
            q = pos >> 3
            if minbits <= 56 and q + 8 <= pend:
                v = (be64(src, q) << u(pos & 7)) >> u(64 - minbits)
            elif minbits <= 56:
                v = u(getbits(src, pos, minbits, pend))
            else:
                v = ((u(getbits(src, pos, minbits - 32, pend)) << u(32))
                     | u(getbits(src, pos + minbits - 32, 32, pend)))
            pos += minbits
            if filavail and v == allones:
                w = filval
            else:
                w = (v + minv) & wmask
            _put(dst, o + i * size, w, size, be)
        return o + nout
    # floats, D-scaling
    if size == 4:
        pw32 = math.pow(np.float32(10.0), np.float32(D))     # powf, as HDF5
        fb = np.zeros(1, np.float32)
        ub = fb.view(np.uint32)
        ub[0] = np.uint32(minval & u(0xFFFFFFFF))
        fmin = fb[0]
        fill32 = np.int64(allones)              # (minbits < 32: the int it is)
        pos = q0 * 8
        for i in range(nel):
            v = np.int64(0)
            if minbits:
                q = pos >> 3
                if q + 8 <= pend:
                    v = np.int64((be64(src, q) << u(pos & 7)) >> u(64 - minbits))
                else:
                    v = getbits(src, pos, minbits, pend)
                pos += minbits
            if filavail and v == fill32:
                w = filval
            else:
                fb[0] = np.float32(v) / pw32 + fmin             # all in float32
                w = u(ub[0])
            _put(dst, o + i * 4, w, 4, be)
        return o + nout
    pw = math.pow(10.0, np.float64(D))
    db = np.zeros(1, np.float64)
    vb = db.view(np.uint64)
    vb[0] = minval
    dmin = db[0]
    pos = q0 * 8
    for i in range(nel):
        v = u(0)
        if minbits:
            q = pos >> 3
            if minbits <= 56 and q + 8 <= pend:
                v = (be64(src, q) << u(pos & 7)) >> u(64 - minbits)
            elif minbits <= 56:
                v = u(getbits(src, pos, minbits, pend))
            else:
                v = ((u(getbits(src, pos, minbits - 32, pend)) << u(32))
                     | u(getbits(src, pos + minbits - 32, 32, pend)))
            pos += minbits
        if filavail and v == allones:
            w = filval
        else:
            db[0] = np.float64(np.int64(v)) / pw + dmin
            w = vb[0]
        _put(dst, o + i * 8, w, 8, be)
    return o + nout


def workspace_size(chunk_nbytes):
    """No scratch needed."""
    return 0
