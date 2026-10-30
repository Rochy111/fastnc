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
from ._szip import getbits, put_word

ERR_DATA = -1
ERR_SPACE = -2
ERR_INPUT = -3

NBIT_ATOMIC, NBIT_ARRAY, NBIT_COMPOUND, NBIT_NOOPTYPE = 1, 2, 3, 4
NOPS = 6                        # an operation: byte offset, kind, size, order, precision, offset


@njit(cache=True, nogil=True)
def _emit(ops, nops, base, kind, size, order, prec, off):
    """One operation in row nops of ops; the rows used after, -1 if full."""
    if nops >= ops.shape[0]:
        return -1
    ops[nops, 0] = base
    ops[nops, 1] = kind
    ops[nops, 2] = size
    ops[nops, 3] = order
    ops[nops, 4] = prec
    ops[nops, 5] = off
    return nops + 1


@njit(cache=True, nogil=True)
def _flatten(cd, ops):
    """The type tree cd[0:] (from its class code) as operations in ops, an
    element's fields in stream order: the rows used, or -1 for bad
    parameters. A walk with a stack of its own (numba's recursion proved
    unreliable): an array frame repeats its base type, a compound frame
    steps through its members; a finished child hands its parameters' end
    to its parent in `ret`."""
    n = cd.size
    D = 64
    fi = np.empty(D, np.int64)                  # the type's parameters
    fb = np.empty(D, np.int64)                  # its byte offset in the element
    fs = np.zeros(D, np.int64)                  # 0 new, 1 iterating
    fc = np.empty(D, np.int64)                  # array: copies done; compound: members done
    fj = np.empty(D, np.int64)                  # compound: the next member's parameters
    fu = np.empty(D, np.int64)                  # compound: the member sizes so far
    top = 0
    fi[0] = 0
    fb[0] = 0
    fs[0] = 0
    nops = 0
    ret = 0
    while top >= 0:
        i = fi[top]
        base = fb[top]
        if i + 1 >= n:
            return -1
        cls = cd[i]
        size = cd[i + 1]
        if fs[top] == 0:
            if cls == NBIT_ATOMIC:
                if i + 4 >= n:
                    return -1
                prec, off = cd[i + 3], cd[i + 4]
                if size <= 0 or prec < 0 or off < 0 or prec > size * 8 or prec + off > size * 8:
                    return -1
                nops = _emit(ops, nops, base, NBIT_ATOMIC, size, cd[i + 2], prec, off)
                ret = i + 5
                top -= 1
            elif cls == NBIT_NOOPTYPE:
                if size < 0:
                    return -1
                nops = _emit(ops, nops, base, NBIT_NOOPTYPE, size, 0, 8 * size, 0)
                ret = i + 2
                top -= 1
            elif cls == NBIT_ARRAY:
                if i + 3 >= n:
                    return -1
                if cd[i + 2] == NBIT_NOOPTYPE:  # the whole array as one run of bytes
                    if size < 0:
                        return -1
                    nops = _emit(ops, nops, base, NBIT_NOOPTYPE, size, 0, 8 * size, 0)
                    ret = i + 4
                    top -= 1
                else:
                    bsize = cd[i + 3]
                    if bsize <= 0 or size < bsize or top + 1 >= D:
                        return -1
                    fs[top] = 1
                    fc[top] = 0
                    top += 1
                    fi[top] = i + 2
                    fb[top] = base
                    fs[top] = 0
            elif cls == NBIT_COMPOUND:
                if i + 2 >= n:
                    return -1
                fs[top] = 1
                fc[top] = 0
                fj[top] = i + 3
                fu[top] = 0
                ret = i + 3
                continue                        # (to the member loop below)
            else:
                return -1
            if nops < 0:
                return -1
            continue
        # a frame whose child (if any) just finished, its parameters ending at ret
        if cls == NBIT_ARRAY:
            fc[top] += 1
            if fc[top] < size // cd[i + 3]:
                top += 1
                fi[top] = i + 2
                fb[top] = base + fc[top - 1] * cd[i + 3]
                fs[top] = 0
            else:
                top -= 1                        # (ret: the end of the base type's parameters)
            continue
        # a compound: the next member, if any
        if fc[top] > 0:
            fj[top] = ret
        if fc[top] >= cd[i + 2]:
            ret = fj[top]
            top -= 1
            continue
        j = fj[top]
        if j + 2 >= n or top + 1 >= D:
            return -1
        moff = cd[j]
        msize = cd[j + 2]
        fu[top] += msize
        if moff < 0 or msize < 0 or fu[top] > size or moff + msize > size:
            return -1                           # (H5Znbit.c's overflow checks)
        fc[top] += 1
        top += 1
        fi[top] = j + 1
        fb[top] = base + moff
        fs[top] = 0
    return nops


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
    v = np.uint64(getbits(src, pos, prec, pend)) << np.uint64(off)
    if size == 1 or size == 2 or size == 4 or size == 8:
        put_word(dst, at, v, size, be)
    elif be:
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
    nops = _flatten(tree, ops)
    if nops < 0:
        return ERR_DATA
    bits = 0
    for r in range(nops):
        bits += ops[r, 4]
    if (nel * bits + 7) // 8 > pend - p:
        return ERR_INPUT
    pos = p * 8
    size = ops[0, 2]
    if (nops == 1 and ops[0, 1] == NBIT_ATOMIC and size == esize and ops[0, 4] <= 56
            and (size == 1 or size == 2 or size == 4 or size == 8)):
        be = ops[0, 3] == 1                     # (the common case: one value, one store each)
        prec = ops[0, 4]
        off = np.uint64(ops[0, 5])
        for e in range(nel):
            v = np.uint64(getbits(src, pos + e * prec, prec, pend)) << off
            put_word(dst, o + e * esize, v, size, be)
        return o + nout
    for k in range(o, o + nout):                # (bits outside the fields are zero)
        dst[k] = 0
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
