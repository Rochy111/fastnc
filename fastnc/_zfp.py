"""zfp, and the HDF5 zfp filter (32013, LLNL's H5Z-ZFP), in numba.

The filter keeps no header in a chunk: zfp's own (`zfp_write_header`,
ZFP_HEADER_FULL) is in the cd values, after a first value holding the zfp and
filter versions. It was written as a little-endian bit stream, so the values
from the second on, taken in pairs, are 64-bit words read from bit 0 up:

    32 bits   'z', 'f', 'p', the codec version (5, since zfp 0.5.0)
    52 bits   the field: 2 bits type - 1 (int32, int64, float, double), 2 bits
              dimensions - 1, then 48 bits of sizes - 1, x first: the x axis is
              the fastest, so a chunk's last HDF5 axis (axes of size 1 are left
              out by the filter)
    12 bits   the mode, up to 0xFFE:  0-2047 fixed rate, minbits = maxbits = m + 1;
              2048-2175 fixed precision, maxprec = m - 2047; 2176 reversible;
              above that fixed accuracy, minexp = m - 2177 - 1074
    or 64     0xFFF, then minbits - 1, maxbits - 1 (15 bits each), maxprec - 1
              (7), minexp + 16495 (15): any of those in full ("expert" mode)

Short modes need 4 cd values, long ones 6. The chunk is the bare zfp stream,
one block of 4^d values after the other, x fastest, each taking between
minbits and maxbits bits. The last block along an axis keeps only the values
inside the field: all 4^d of it are coded all the same.

A block of floats: a bit saying it is not all zero, the largest exponent of
its values, then integers; of integers, those alone. Integers are coded by
bit plane, most significant first, up to maxprec planes (fewer for floats:
none below minexp) and maxbits bits: in each plane the coefficients found
significant so far take one bit each, and the rest are run-length coded, a 1
for "another one is set", then zeros up to it. The coefficients, in the order
of PERM, are negabinary; the lifting transform below, undone along each axis
from the slowest to x, gives the values, which for floats are then scaled by
the exponent (2^(emax - 30) for floats, 2^(emax - 62) for doubles).

Reversible mode replaces that transform by a Lorenzo one, with no shifts, and
says first how many planes there are. After the bit saying a block of floats
is not all zero comes one more: 0, integers scaled by the exponent as above;
1 where that would not be exact, no exponent, and the values' bits taken as
sign-magnitude integers.

32-bit values are held in the top half of a uint64, so that one code path
serves both widths: the planes, negabinary and the additions then work as they
are, and only the arithmetic shifts need the bits they bring down cut off.
Additions are on unsigned words, which wrap as the C library's do (numba
assumes signed ones do not overflow).

Written from the algorithm of LLNL's zfp (Lindstrom 2014, BSD licensed), and
checked value for value against H5Z-ZFP 1.1.1 with zfp 1.0.1, and a file of
H5Z-ZFP 1.0.1 with zfp 0.5.5. PERM is as that library decodes one coefficient
at a time: by sum of the indices, then of their squares.
"""
import numpy as np

from ._inflate import ERR_DATA, ERR_INPUT, ERR_SPACE
from ._intrin import _ptr, ld64
from ._jit import intrinsic, njit

MAGIC = 0x0570667A                  # 'z', 'f', 'p', codec version 5, as the word reads
MIN_EXP = -1074
MAX_BITS = 16658                    # (no block of a stream reaches it)

# which coefficient comes next, for 1, 2, 3 and 4 dimensions, each after the last
PERM = np.array((
    0, 1, 2, 3,

    0, 1, 4, 5, 2, 8, 6, 9, 3, 12, 10, 7, 13, 11, 14, 15,

    0, 1, 4, 16, 20, 17, 5, 2, 8, 32, 21, 6, 18, 24, 9, 33, 36, 3, 12, 48,
    22, 25, 37, 40, 34, 10, 7, 19, 28, 13, 49, 52, 41, 38, 26, 23, 29, 53, 11, 35,
    44, 14, 50, 56, 42, 27, 39, 45, 30, 54, 57, 60, 51, 15, 43, 46, 58, 61, 55, 31,
    62, 59, 47, 63,

    0, 1, 4, 16, 64, 5, 80, 17, 68, 65, 20, 2, 8, 32, 128, 84, 81, 69, 21, 6,
    18, 66, 24, 72, 9, 96, 33, 36, 129, 132, 144, 3, 12, 48, 192, 85, 82, 70, 22, 73,
    25, 88, 37, 100, 97, 148, 145, 133, 10, 160, 34, 136, 130, 40, 7, 19, 67, 28, 76, 13,
    112, 49, 52, 193, 196, 208, 86, 89, 101, 149, 161, 137, 41, 134, 38, 164, 26, 152, 146, 104,
    98, 74, 83, 71, 23, 77, 29, 92, 53, 116, 113, 212, 209, 197, 11, 35, 131, 44, 140, 14,
    176, 50, 56, 194, 200, 224, 90, 165, 102, 153, 150, 105, 168, 162, 138, 42, 87, 93, 117, 213,
    27, 75, 99, 39, 135, 147, 108, 45, 141, 156, 30, 78, 177, 180, 54, 114, 120, 57, 198, 210,
    216, 201, 225, 228, 15, 240, 51, 204, 195, 60, 169, 166, 154, 106, 91, 103, 151, 109, 157, 94,
    181, 118, 121, 214, 217, 229, 163, 139, 43, 142, 46, 172, 58, 184, 178, 232, 226, 202, 241, 205,
    61, 199, 55, 244, 31, 220, 211, 124, 115, 79, 170, 167, 155, 107, 158, 110, 173, 122, 185, 182,
    233, 230, 218, 95, 245, 119, 221, 215, 125, 242, 206, 62, 203, 59, 248, 47, 236, 227, 188, 179,
    143, 171, 174, 186, 234, 246, 222, 126, 219, 123, 249, 111, 237, 231, 189, 183, 159, 252, 243, 207,
    63, 175, 250, 187, 238, 235, 190, 253, 247, 223, 127, 254, 251, 239, 191, 255,
), np.uint8)


@intrinsic
def store(typingctx, a, i, v):
    """a[i:] = v's bytes, whatever its type (an unaligned store)."""
    from numba import types

    def codegen(context, builder, sig, args):
        from llvmlite import ir
        bits = sig.args[2].bitwidth
        p = _ptr(context, builder, sig.args[0], args[0], args[1], bits)
        builder.store(builder.bitcast(args[2], ir.IntType(bits)), p, align=1)
        return context.get_dummy_value()
    return types.void(a, i, v), codegen


@intrinsic
def as_double(typingctx, u):
    """The float64 with these bits."""
    from numba import types

    def codegen(context, builder, sig, args):
        from llvmlite import ir
        return builder.bitcast(args[0], ir.DoubleType())
    return types.float64(u), codegen


@njit(cache=True, nogil=True)
def _pow2(e):
    """2.0 ** e as ldexp gives it: subnormal, then zero, far enough down."""
    if e >= -1022:
        return as_double(np.uint64(e + 1023) << np.uint64(52))
    if e >= -1074:
        return as_double(np.uint64(1) << np.uint64(e + 1074))
    return 0.0


@njit(cache=True, nogil=True)
def _word(src, q, pend):
    """The eight bytes at q as a little-endian word, zeros past pend."""
    if q + 8 <= pend:
        return ld64(src, q)
    w = np.uint64(0)
    s = np.uint64(0)
    while q < pend:
        w |= np.uint64(src[q]) << s
        s += np.uint64(8)
        q += 1
    return w


@njit(cache=True, nogil=True)
def _take(src, pos, pend, n):
    """n <= 56 bits from bit pos on."""
    return (_word(src, pos >> 3, pend) >> np.uint64(pos & 7)) & ((np.uint64(1) << np.uint64(n)) - np.uint64(1))


@njit(cache=True, nogil=True)
def _ints(src, pos, pend, bits, kmin, u, size):
    """u[:size] from at most `bits` bits at bit pos, planes 63 down to kmin.
    The bit position after them."""
    one = np.uint64(1)
    nil = np.uint64(0)
    for i in range(size):
        u[i] = 0
    k = 63
    n = 0
    while bits > 0 and k >= kmin:
        sh = np.uint64(k)
        m = n if n < bits else bits             # those already significant
        bits -= m
        i = 0
        while i < m:
            w = _word(src, pos >> 3, pend) >> np.uint64(pos & 7)
            c = m - i if m - i < 56 else 56
            for j in range(c):
                u[i + j] |= ((w >> np.uint64(j)) & one) << sh
            i += c
            pos += c
        w = np.uint64(0)
        left = 0
        while n < size and bits > 0:            # the others: another one set?
            if left == 0:
                w = _word(src, pos >> 3, pend) >> np.uint64(pos & 7)
                left = 56
            bits -= 1
            pos += 1
            left -= 1
            b = w & one
            w >>= one
            if b == nil:
                break
            while n < size - 1 and bits > 0:    # zeros up to it
                if left == 0:
                    w = _word(src, pos >> 3, pend) >> np.uint64(pos & 7)
                    left = 56
                bits -= 1
                pos += 1
                left -= 1
                b = w & one
                w >>= one
                if b != nil:
                    break
                n += 1
            u[n] |= one << sh
            n += 1
        k -= 1
    return pos


@njit(cache=True, nogil=True)
def _lift(a, p, s, keep):
    """Undo the decorrelating transform on a[p], a[p + s] ...: the matrix
    (4 6 -4 -1; 4 2 4 5; 4 -2 4 -5; 4 -6 -4 1) / 4, in integers."""
    x = a[p]
    y = a[p + s]
    z = a[p + 2 * s]
    w = a[p + 3 * s]
    y += np.uint64(np.int64(w) >> 1) & keep
    w -= np.uint64(np.int64(y) >> 1) & keep
    y += w
    w += w
    w -= y
    z += x
    x += x
    x -= z
    y += z
    z += z
    z -= y
    w += x
    x += x
    x -= w
    a[p] = x
    a[p + s] = y
    a[p + 2 * s] = z
    a[p + 3 * s] = w


@njit(cache=True, nogil=True)
def _rev_lift(a, p, s):
    """Undo the reversible (Lorenzo) transform on a[p], a[p + s] ..."""
    x = a[p]
    y = a[p + s]
    z = a[p + 2 * s]
    w = a[p + 3 * s]
    w += z
    z += y
    w += z
    y += x
    z += y
    w += z
    a[p + s] = y
    a[p + 2 * s] = z
    a[p + 3 * s] = w


@njit(cache=True, nogil=True)
def _xform(a, size, rev, keep):
    """Undo the block's transform, along each axis from the slowest to x."""
    s = size >> 2
    while s > 0:
        for b in range(0, size, 4 * s):
            for i in range(s):
                if rev:
                    _rev_lift(a, b + i, s)
                else:
                    _lift(a, b + i, s, keep)
        s >>= 2


@njit(cache=True, nogil=True)
def _put(a, dst, o, kind, s4, s8, cx, cy, cz, cw, sy, sz, sw):
    """The part of the block a[] inside the field, at dst[o:]."""
    for m in range(cw):
        for k in range(cz):
            for j in range(cy):
                b = 64 * m + 16 * k + 4 * j
                t = o + m * sw + k * sz + j * sy
                if kind == 2:
                    for i in range(cx):
                        store(dst, t + 4 * i, np.float32(np.int64(a[b + i]) >> 32) * s4)
                elif kind == 3:
                    for i in range(cx):
                        store(dst, t + 8 * i, np.float64(np.int64(a[b + i])) * s8)
                elif kind == 0:
                    for i in range(cx):
                        store(dst, t + 4 * i, np.uint32(a[b + i] >> np.uint64(32)))
                else:
                    for i in range(cx):
                        store(dst, t + 8 * i, a[b + i])


@njit(cache=True, nogil=True)
def _field(src, pos, pend, dst, o, kind, d, nx, ny, nz, nw, minbits, maxbits, maxprec, minexp, u, a):
    """Every block of the field from bit pos on, into dst[o:]. 0, or a negative
    error code."""
    rev = minexp < MIN_EXP
    isfloat = kind >= 2
    wide = (kind & 1) == 1
    esize = 8 if wide else 4
    ebits = 11 if wide else 8
    ebias = 1023 if wide else 127
    pbits = 6 if wide else 5
    keep = np.uint64(0xFFFFFFFFFFFFFFFF) if wide else np.uint64(0xFFFFFFFF00000000)
    tc = np.uint64(0x7FFFFFFFFFFFFFFF) if wide else np.uint64(0x7FFFFFFF00000000)
    nb = np.uint64(0xAAAAAAAAAAAAAAAA)
    nil = np.uint64(0)
    size = 1 << (2 * d)
    at = (size - 4) // 3
    end = pend * 8
    sy = esize * nx
    sz = sy * ny
    sw = sz * nz
    for bw in range(0, nw, 4):
        cw = nw - bw if nw - bw < 4 else 4
        for bz in range(0, nz, 4):
            cz = nz - bz if nz - bz < 4 else 4
            for by in range(0, ny, 4):
                cy = ny - by if ny - by < 4 else 4
                for bx in range(0, nx, 4):
                    cx = nx - bx if nx - bx < 4 else 4
                    pos0 = pos
                    bare = False                # the integers are the values' bits
                    zero = False
                    scaled = False
                    emax = 0
                    prec = maxprec
                    if isfloat:
                        zero = _take(src, pos, pend, 1) == nil
                        pos += 1
                        if not zero and rev:
                            bare = _take(src, pos, pend, 1) != nil
                            pos += 1
                        scaled = not zero and not bare
                        if scaled:
                            emax = np.int64(_take(src, pos, pend, ebits)) - ebias
                            pos += ebits
                            if not rev:
                                top = emax - minexp + 2 * (d + 1)
                                top = top if top > 0 else 0
                                prec = prec if prec < top else top
                    if zero:
                        for i in range(size):
                            a[i] = 0
                    else:
                        if rev:
                            prec = np.int64(_take(src, pos, pend, pbits)) + 1
                            pos += pbits
                        kmin = 64 - prec if prec < 8 * esize else 64 - 8 * esize
                        pos = _ints(src, pos, pend, maxbits - (pos - pos0), kmin, u, size)
                        for i in range(size):
                            a[PERM[at + i]] = (u[i] ^ nb) - nb
                        _xform(a, size, rev, keep)
                        if bare:
                            for i in range(size):
                                if np.int64(a[i]) < 0:
                                    a[i] ^= tc
                    if pos - pos0 < minbits:
                        pos = pos0 + minbits
                    if pos > end:
                        return ERR_INPUT
                    s8 = 1.0
                    if scaled:
                        s8 = _pow2(emax - (62 if wide else 30))
                    kout = kind if scaled or not isfloat else (1 if wide else 0)
                    _put(a, dst, o + bw * sw + bz * sz + by * sy + bx * esize, kout, np.float32(s8), s8,
                         cx, cy, cz, cw, sy, sz, sw)
    return 0


@njit(cache=True, nogil=True)
def _cdword(cd, i):
    """Word i of the header, the cd values after the first; zeros after them."""
    n = cd[2]
    k = 2 * i + 1
    lo = np.uint64(cd[3 + k]) if k < n else np.uint64(0)
    hi = np.uint64(cd[4 + k]) if k + 1 < n else np.uint64(0)
    return lo | (hi << np.uint64(32))


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """The zfp filter: the stream src[p:pend] into dst[o:oend]; ws: workspace_size()
    bytes. The end of the output, or a negative error code."""
    if cd[2] < 4:
        return ERR_DATA
    w0 = _cdword(cd, 0)
    w1 = _cdword(cd, 1)
    if (w0 & np.uint64(0xFFFFFFFF)) != np.uint64(MAGIC):
        return ERR_DATA
    meta = (w0 >> np.uint64(32)) | ((w1 & np.uint64(0xFFFFF)) << np.uint64(32))
    kind = np.int64(meta & np.uint64(3))
    d = np.int64((meta >> np.uint64(2)) & np.uint64(3)) + 1
    wide = (kind & 1) == 1
    esize = 8 if wide else 4
    width = 48 // d
    sizes = meta >> np.uint64(4)
    nmask = (np.uint64(1) << np.uint64(width)) - np.uint64(1)
    nx = np.int64(sizes & nmask) + 1
    ny = np.int64((sizes >> np.uint64(width)) & nmask) + 1 if d > 1 else 1
    nz = np.int64((sizes >> np.uint64(2 * width)) & nmask) + 1 if d > 2 else 1
    nw = np.int64((sizes >> np.uint64(3 * width)) & nmask) + 1 if d > 3 else 1
    m = np.int64((w1 >> np.uint64(20)) & np.uint64(0xFFF))
    minbits = 1
    maxbits = MAX_BITS
    maxprec = 64
    minexp = MIN_EXP
    if m < 2048:
        minbits = maxbits = m + 1
    elif m < 2176:
        maxprec = m - 2047
    elif m == 2176:
        minexp = MIN_EXP - 1
    elif m < 0xFFF:
        minexp = m - 2177 + MIN_EXP
    elif cd[2] < 6:
        return ERR_DATA
    else:
        v = (w1 >> np.uint64(32)) | ((_cdword(cd, 2) & np.uint64(0xFFFFF)) << np.uint64(32))
        minbits = np.int64(v & np.uint64(0x7FFF)) + 1
        maxbits = np.int64((v >> np.uint64(15)) & np.uint64(0x7FFF)) + 1
        maxprec = np.int64((v >> np.uint64(30)) & np.uint64(0x7F)) + 1
        minexp = np.int64((v >> np.uint64(37)) & np.uint64(0x7FFF)) - 16495
    head = 0                                    # what comes before the integers of a block
    if minexp < MIN_EXP:
        head += 7 if wide and kind >= 2 else 6 if wide or kind >= 2 else 5
    if kind >= 2:
        head += 12 if wide else 9
    if minbits > maxbits or maxprec > 64 or maxbits < head:
        return ERR_DATA
    total = nx * ny * nz * nw * esize
    if esize != cd[0] or total != cd[1]:
        return ERR_DATA
    if total > oend - o:
        return ERR_SPACE
    r = _field(src, p * 8, pend, dst, o, kind, d, nx, ny, nz, nw, minbits, maxbits, maxprec, minexp,
               ws[0:2048].view(np.uint64), ws[2048:4096].view(np.uint64))
    return r if r < 0 else o + total


def workspace_size(chunk_nbytes):
    """A block as coded, and as integers: at most 256 words each."""
    return 4096


def describe(cd):
    """What the header in these cd values says: (codec version, type name,
    dimensions), the codec version None if it is no zfp header at all."""
    n = int(cd[2])
    vals = [int(x) for x in cd[3:3 + n]]
    if n < 4:
        return None, None, None
    w0 = vals[1] | (vals[2] << 32)
    if w0 & 0xFFFFFF != MAGIC & 0xFFFFFF:
        return None, None, None
    return (w0 >> 24) & 255, ('int32', 'int64', 'float', 'double')[(w0 >> 32) & 3], ((w0 >> 34) & 3) + 1


def decode_py(data, out_nbytes, cd):
    """Not a decoder: what the kernel turned down, in words. (The staged path
    asks here for a chunk its kernel could not decode.)"""
    codec, kind, d = describe(cd)
    if codec is None:
        raise IOError('zfp filter: no zfp header in the cd values %r' % ([int(x) for x in cd[3:]],))
    if codec != MAGIC >> 24:
        raise NotImplementedError('zfp filter: codec version %d, from a zfp older than 0.5.0 (only %d is '
                                  'decoded)' % (codec, MAGIC >> 24))
    raise IOError('zfp filter: a chunk of %d bytes could not be decoded as %dD %s: %d bytes of stream'
                  % (out_nbytes, d, kind, len(data)))
