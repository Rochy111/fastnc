"""HDF5's n-bit filter (filter 5) undone, as H5Znbit.c's reverse pass.

The filter keeps only the significant bits of each value -- bits
[offset, offset + precision) of an integer or float of reduced precision
-- packed most significant bit first with no padding, element after
element, member after member. Decoding puts them back at their offset,
every other bit zero, in the type's byte order. Bytes of types the filter
cannot reduce (strings, enums, references, opaque ... inside a compound or
array: "no-op" types) are packed whole, 8 bits each.

cd values describe the type as a tree (H5Z__set_parms_*):
  [0] number of cd values, [1] 1 if nothing needs reducing (the chunk is
  stored as is), [2] elements in a chunk, then the type from [3]:
  atomic   1, size, order (0 LE, 1 BE), precision, offset
  array    2, size, base type
  compound 3, size, nmembers, then per member: offset, member type
  no-op    4, size

The tree is flattened once per chunk into a list of operations (an
atomic value or a run of no-op bytes, at a byte offset in the element),
then the elements are decoded through that list. An atomic value of up to
8 bytes and 56 bits of precision is one bit-field read and a store; wider
ones go byte by byte, as H5Znbit.c does.

Note: the result is the stored bytes of the n-bit type, as HDF5's filter
leaves them. What h5py returns has, in addition, gone through HDF5's type
conversion to a native type (shifted down by the offset, sign-extended,
or a non-IEEE float converted); that is the chunk engine's business.
"""
import numpy as np

from ._intrin import ld64, st64
from ._jit import njit
from ._szip import be64, getbits

ERR_DATA = -1
ERR_SPACE = -2
ERR_INPUT = -3

NBIT_ATOMIC, NBIT_ARRAY, NBIT_COMPOUND, NBIT_NOOPTYPE = 1, 2, 3, 4
NOPS = 6                        # an operation: byte offset, kind, size, order, precision, offset


@njit(cache=True, nogil=True)
def _flatten(cd, i, base, ops, nops, depth):
    """The type whose parameters start at cd[i] (its class code), placed at
    byte `base` of the element, appended to ops as operations from row
    nops. (index after its parameters, rows used), or (-1, 0) for bad
    parameters."""
    n = cd.size
    if depth > 64 or i + 1 >= n:
        return -1, 0
    cls = cd[i]
    size = cd[i + 1]
    if cls == NBIT_ATOMIC:
        if i + 4 >= n or nops >= ops.shape[0]:
            return -1, 0
        order, prec, off = cd[i + 2], cd[i + 3], cd[i + 4]
        if prec > size * 8 or prec + off > size * 8 or size <= 0:
            return -1, 0
        ops[nops, 0] = base
        ops[nops, 1] = NBIT_ATOMIC
        ops[nops, 2] = size
        ops[nops, 3] = order
        ops[nops, 4] = prec
        ops[nops, 5] = off
        return i + 5, nops + 1
    if cls == NBIT_NOOPTYPE:
        if nops >= ops.shape[0] or size < 0:
            return -1, 0
        ops[nops, 0] = base
        ops[nops, 1] = NBIT_NOOPTYPE
        ops[nops, 2] = size
        ops[nops, 3] = 0
        ops[nops, 4] = 8 * size
        ops[nops, 5] = 0
        return i + 2, nops + 1
    if cls == NBIT_ARRAY:
        if i + 3 >= n:
            return -1, 0
        bcls = cd[i + 2]
        bsize = cd[i + 3]
        if bcls == NBIT_NOOPTYPE:               # the whole array as one run of bytes
            if nops >= ops.shape[0]:
                return -1, 0
            ops[nops, 0] = base
            ops[nops, 1] = NBIT_NOOPTYPE
            ops[nops, 2] = size
            ops[nops, 3] = 0
            ops[nops, 4] = 8 * size
            ops[nops, 5] = 0
            return i + 4, nops + 1
        if bsize <= 0 or size < bsize:
            return -1, 0
        cnt = size // bsize
        j = i + 2
        for c in range(cnt):
            j, nops = _flatten(cd, i + 2, base + c * bsize, ops, nops, depth + 1)
            if j < 0:
                return -1, 0
        return j, nops
    if cls == NBIT_COMPOUND:
        if i + 2 >= n:
            return -1, 0
        nmem = cd[i + 2]
        j = i + 3
        used = 0
        for m in range(nmem):
            if j + 2 >= n:
                return -1, 0
            moff = cd[j]
            msize = cd[j + 2]
            used += msize
            if used > size or moff + msize > size:
                return -1, 0                    # (H5Znbit.c's overflow checks)
            j, nops = _flatten(cd, j + 1, base + moff, ops, nops, depth + 1)
            if j < 0:
                return -1, 0
        return j, nops
    return -1, 0


@njit(cache=True, nogil=True)
def _atomic_bytes(dst, at, size, be, prec, off, src, pos, pend):
    """An atomic value byte by byte (H5Z__nbit_decompress_one_atomic): bits
    [off, off+prec) from the stream at bit pos into the size-byte value at
    dst[at:] (whose other bits are already zero). The new bit position."""
    dtl = size * 8
    if not be:
        if (prec + off) % 8 != 0:
            bi = (prec + off) // 8
        else:
            bi = (prec + off) // 8 - 1
        ei = off // 8
        k = bi
        while k >= ei:
            if bi == ei:
                dl = prec
                do = off % 8
            elif k == bi:
                dl = 8 - (dtl - prec - off) % 8
                do = 0
            elif k == ei:
                dl = 8 - off % 8
                do = 8 - dl
            else:
                dl = 8
                do = 0
            dst[at + k] = getbits(src, pos, dl, pend) << do
            pos += dl
            k -= 1
    else:
        bi = (dtl - prec - off) // 8
        if off % 8 != 0:
            ei = (dtl - off) // 8
        else:
            ei = (dtl - off) // 8 - 1
        k = bi
        while k <= ei:
            if bi == ei:
                dl = prec
                do = off % 8
            elif k == bi:
                dl = 8 - (dtl - prec - off) % 8
                do = 0
            elif k == ei:
                dl = 8 - off % 8
                do = 8 - dl
            else:
                dl = 8
                do = 0
            dst[at + k] = getbits(src, pos, dl, pend) << do
            pos += dl
            k += 1
    return pos


@njit(cache=True, nogil=True)
def _atomic(dst, at, size, be, prec, off, src, pos, pend):
    """An atomic value of at most 8 bytes and 56 bits of precision: one
    bit-field read, shifted to its offset, stored in its byte order."""
    q = pos >> 3
    if prec == 0:
        v = np.uint64(0)
    elif q + 8 <= pend:
        v = (be64(src, q) << np.uint64(pos & 7)) >> np.uint64(64 - prec)
    else:
        v = np.uint64(getbits(src, pos, prec, pend))
    v <<= np.uint64(off)
    if be:
        for t in range(size):
            dst[at + t] = (v >> np.uint64(8 * (size - 1 - t))) & np.uint64(255)
    else:
        for t in range(size):
            dst[at + t] = (v >> np.uint64(8 * t)) & np.uint64(255)
    return pos + prec


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """An n-bit chunk at src[p:pend] into dst[o:...]; cd: [element size,
    chunk bytes, n, the filter's cd values]. The end of the output, or a
    negative error code."""
    ncd = cd[2]
    if ncd < 3 or ncd + 3 > cd.size or cd[3] != ncd:
        return ERR_DATA
    if cd[4]:                                   # nothing reduced: stored as is
        n = pend - p
        if n > oend - o:
            return ERR_SPACE
        k = 0
        while k + 8 <= n:
            st64(dst, o + k, ld64(src, p + k))
            k += 8
        while k < n:
            dst[o + k] = src[p + k]
            k += 1
        return o + n
    if ncd < 5:
        return ERR_DATA
    tree = cd[6:3 + ncd]                        # the type, from its class code
    nel = cd[5]
    esize = tree[1]
    if tree[0] not in (NBIT_ATOMIC, NBIT_ARRAY, NBIT_COMPOUND) or esize <= 0 or nel < 0:
        return ERR_DATA
    nout = nel * esize
    if nout > oend - o:
        return ERR_SPACE
    ops = np.empty((esize + 1, NOPS), np.int64)
    j, nops = _flatten(tree, 0, 0, ops, 0, 0)
    if j < 0:
        return ERR_DATA
    bits = 0
    for r in range(nops):
        bits += ops[r, 4]
    if (nel * bits + 7) // 8 > pend - p:
        return ERR_INPUT
    for k in range(o, o + nout):                # (bits outside the fields are zero)
        dst[k] = 0
    pos = p * 8
    if nops == 1 and ops[0, 1] == NBIT_ATOMIC and ops[0, 2] <= 8 and ops[0, 4] <= 56:
        size = ops[0, 2]
        be = ops[0, 3] == 1
        prec = ops[0, 4]
        off = ops[0, 5]
        for e in range(nel):
            pos = _atomic(dst, o + e * esize, size, be, prec, off, src, pos, pend)
        return o + nout
    for e in range(nel):
        at = o + e * esize
        for r in range(nops):
            a = at + ops[r, 0]
            size = ops[r, 2]
            if ops[r, 1] == NBIT_NOOPTYPE:
                for t in range(size):
                    dst[a + t] = getbits(src, pos, 8, pend)
                    pos += 8
            elif size <= 8 and ops[r, 4] <= 56:
                pos = _atomic(dst, a, size, ops[r, 3] == 1, ops[r, 4], ops[r, 5], src, pos, pend)
            else:
                pos = _atomic_bytes(dst, a, size, ops[r, 3] == 1, ops[r, 4], ops[r, 5],
                                    src, pos, pend)
    return o + nout


def workspace_size(chunk_nbytes):
    """No scratch needed."""
    return 0
