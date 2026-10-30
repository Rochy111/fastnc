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
  little-endian order (HDF5 converts to its native order before encoding).
  HDF5 1.14 refuses to read such chunks (its check is `minbits >= bits`,
  one off: h5py raises "filter returned failure"); HDF5 2.0 reads them, and
  so does this;
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
from ._jit import intrinsic, njit
from ._szip import getbits, put_word

ERR_DATA = -1
ERR_SPACE = -2
ERR_INPUT = -3

HEADER = 21                     # minbits 4, minval size 1, minval 16 (8 used)
SO_FLOAT_DSCALE, SO_FLOAT_ESCALE, SO_INT = 0, 1, 2


def _bitcast(name, frm, to):
    def cast(typingctx, x):
        from llvmlite import ir
        from numba import types
        width = 32 if '32' in to else 64
        out = getattr(types, to)
        def codegen(context, builder, sig, args):
            v = args[0]
            if frm == 'float':
                return builder.bitcast(v, ir.IntType(width))
            return builder.bitcast(builder.trunc(v, ir.IntType(width)) if width == 32 else v,
                                   ir.FloatType() if width == 32 else ir.DoubleType())
        return out(getattr(types, 'float%d' % width) if frm == 'float' else types.uint64), codegen
    cast.__name__ = name
    return intrinsic(cast)


f32_bits = _bitcast('f32_bits', 'float', 'uint32')      # a float32's bits
f64_bits = _bitcast('f64_bits', 'float', 'uint64')      # a float64's bits
bits_f32 = _bitcast('bits_f32', 'int', 'float32')       # the float32 of a uint64's low 32 bits
bits_f64 = _bitcast('bits_f64', 'int', 'float64')       # the float64 of a uint64's bits


@njit(cache=True, nogil=True)
def _read(src, pos, minbits, pend):
    """minbits (1..63) bits at bit pos, as a uint64."""
    if minbits <= 56:
        return np.uint64(getbits(src, pos, minbits, pend))
    return ((np.uint64(getbits(src, pos, minbits - 32, pend)) << np.uint64(32))
            | np.uint64(getbits(src, pos + minbits - 32, 32, pend)))


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
    if cls == 0:
        minv = minval                           # (added in the type's width: the store truncates)
        if minbits == 0:
            w = filval if filavail else minv    # (the zeros all match the 0-bit fill code)
            for i in range(nel):
                put_word(dst, o + i * size, w, size, be)
            return o + nout
        pos = q0 * 8
        if minbits > 56:                        # (8-byte integers only)
            for i in range(nel):
                v = _read(src, pos + i * minbits, minbits, pend)
                put_word(dst, o + i * size, filval if filavail and v == allones else v + minv,
                         size, be)
        elif filavail:
            for i in range(nel):
                v = np.uint64(getbits(src, pos + i * minbits, minbits, pend))
                put_word(dst, o + i * size, filval if v == allones else v + minv, size, be)
        else:
            for i in range(nel):
                v = np.uint64(getbits(src, pos + i * minbits, minbits, pend))
                put_word(dst, o + i * size, v + minv, size, be)
        return o + nout
    # floats, D-scaling
    pos = q0 * 8
    if size == 4:
        pw32 = math.pow(np.float32(10.0), np.float32(D))     # powf, as HDF5
        fmin = bits_f32(minval)
        for i in range(nel):
            v = np.uint64(getbits(src, pos + i * minbits, minbits, pend))   # (0 bits: 0)
            if filavail and v == allones:
                w = filval
            else:
                w = u(f32_bits(np.float32(np.int64(v)) / pw32 + fmin))     # all in float32
            put_word(dst, o + i * 4, w, 4, be)
        return o + nout
    pw = math.pow(10.0, np.float64(D))
    dmin = bits_f64(minval)
    for i in range(nel):
        if minbits > 56:
            v = _read(src, pos + i * minbits, minbits, pend)
        else:
            v = np.uint64(getbits(src, pos + i * minbits, minbits, pend))
        if filavail and v == allones:
            w = filval
        else:
            w = f64_bits(np.float64(np.int64(v)) / pw + dmin)
        put_word(dst, o + i * 8, w, 8, be)
    return o + nout


def workspace_size(chunk_nbytes):
    """No scratch needed."""
    return 0
