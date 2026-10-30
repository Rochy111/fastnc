"""Bitshuffle (HDF5 filter 32008, bshuf_h5filter.c as hdf5plugin ships
it), alone or with its LZ4 / zstd compression, in numba; and the bit
transpose blosc's bit-shuffle filter shares.

cd values: bitshuffle major and minor version, element size, block size
in elements (0: the default, 8192 bytes' worth, a multiple of 8, at least
128), compression (0 none, 2 lz4, 3 zstd), zstd level.

Uncompressed, a chunk is its blocks one after another, each the bit
transpose of `block` elements: bit row 8*j + k holds bit k of byte j of
every element, element i at bit i % 8 of byte i // 8. The last block
holds the elements left over rounded down to a multiple of 8; the last
(size % 8) elements are stored as they are.

Compressed, the chunk starts with the decoded size (uint64 BE) and the
block size in bytes (uint32 BE), and each block is a uint32 BE compressed
size and an LZ4 block or a zstd frame of the transposed block; the
leftover elements follow, as they are.

The transpose is undone 64 elements at a time: eight words from eight bit
rows, an 8x8 byte transpose of them, then an 8x8 bit transpose of each
word gives one byte of 8 elements; for elements of 2 to 8 bytes another
byte transpose puts each element's bytes together, stored a word at a
time.
"""
import numpy as np

from ._inflate import ERR_DATA, ERR_INPUT, ERR_SPACE
from ._intrin import ld64, st64
from ._jit import njit
from ._lz4 import be32, be64, copy_bytes, lz4_block
from ._zstd import WS_SIZE as ZWS, zstd_decompress

W_Y = 0                         # 64 uint64 of transposed words
W_Z = 512                       # zstd's scratch
W_T = W_Z + ZWS                 # one decompressed block


@njit(cache=True, nogil=True)
def _bits8(x):
    """Transpose the 8x8 bit matrix in x (byte k, bit b) -> (byte b, bit k)."""
    t = (x ^ (x >> np.uint64(7))) & np.uint64(0x00AA00AA00AA00AA)
    x = x ^ t ^ (t << np.uint64(7))
    t = (x ^ (x >> np.uint64(14))) & np.uint64(0x0000CCCC0000CCCC)
    x = x ^ t ^ (t << np.uint64(14))
    t = (x ^ (x >> np.uint64(28))) & np.uint64(0x00000000F0F0F0F0)
    x = x ^ t ^ (t << np.uint64(28))
    return x


@njit(cache=True, nogil=True)
def _bytes8(a0, a1, a2, a3, a4, a5, a6, a7):
    """Transpose the 8x8 byte matrix of words a0..a7 (word i, byte j) ->
    (word j, byte i)."""
    m = np.uint64(0x00FF00FF00FF00FF)
    s = np.uint64(8)
    t = ((a0 >> s) ^ a1) & m
    a1 ^= t
    a0 ^= t << s
    t = ((a2 >> s) ^ a3) & m
    a3 ^= t
    a2 ^= t << s
    t = ((a4 >> s) ^ a5) & m
    a5 ^= t
    a4 ^= t << s
    t = ((a6 >> s) ^ a7) & m
    a7 ^= t
    a6 ^= t << s
    m = np.uint64(0x0000FFFF0000FFFF)
    s = np.uint64(16)
    t = ((a0 >> s) ^ a2) & m
    a2 ^= t
    a0 ^= t << s
    t = ((a1 >> s) ^ a3) & m
    a3 ^= t
    a1 ^= t << s
    t = ((a4 >> s) ^ a6) & m
    a6 ^= t
    a4 ^= t << s
    t = ((a5 >> s) ^ a7) & m
    a7 ^= t
    a5 ^= t << s
    m = np.uint64(0x00000000FFFFFFFF)
    s = np.uint64(32)
    t = ((a0 >> s) ^ a4) & m
    a4 ^= t
    a0 ^= t << s
    t = ((a1 >> s) ^ a5) & m
    a5 ^= t
    a1 ^= t << s
    t = ((a2 >> s) ^ a6) & m
    a6 ^= t
    a2 ^= t << s
    t = ((a3 >> s) ^ a7) & m
    a7 ^= t
    a3 ^= t << s
    return a0, a1, a2, a3, a4, a5, a6, a7


@njit(cache=True, nogil=True)
def bitunshuffle(src, s, dst, d, n, es, dend, y):
    """Undo the bit transpose of one block: n elements (a multiple of 8) of
    es bytes, src[s:s+n*es] -> dst[d:d+n*es]; dst is writable up to dend
    (word stores may run past the block's end when there is room). y: 64
    uint64 of scratch."""
    nb = n >> 3                                 # bytes per bit row
    words = es <= 8 or (es & 7) == 0
    g0 = 0
    while words and g0 + 8 <= nb and d + (8 * g0 + 64) * es + 8 <= dend:
        for j0 in range(0, es, 8):
            jn = min(8, es - j0)
            for j in range(jn):
                r = s + (8 * (j0 + j)) * nb + g0
                a0, a1, a2, a3, a4, a5, a6, a7 = _bytes8(
                    ld64(src, r), ld64(src, r + nb), ld64(src, r + 2 * nb),
                    ld64(src, r + 3 * nb), ld64(src, r + 4 * nb), ld64(src, r + 5 * nb),
                    ld64(src, r + 6 * nb), ld64(src, r + 7 * nb))
                y[8 * j] = _bits8(a0)
                y[8 * j + 1] = _bits8(a1)
                y[8 * j + 2] = _bits8(a2)
                y[8 * j + 3] = _bits8(a3)
                y[8 * j + 4] = _bits8(a4)
                y[8 * j + 5] = _bits8(a5)
                y[8 * j + 6] = _bits8(a6)
                y[8 * j + 7] = _bits8(a7)
            if es == 1:
                for g in range(8):
                    st64(dst, d + 8 * (g0 + g), y[g])
                continue
            z = np.uint64(0)
            for g in range(8):
                # word j (byte j0+j of 8 elements) -> word b (element 8(g0+g)+b)
                b0, b1, b2, b3, b4, b5, b6, b7 = _bytes8(
                    y[g], y[8 + g] if jn > 1 else z, y[16 + g] if jn > 2 else z,
                    y[24 + g] if jn > 3 else z, y[32 + g] if jn > 4 else z,
                    y[40 + g] if jn > 5 else z, y[48 + g] if jn > 6 else z,
                    y[56 + g] if jn > 7 else z)
                e = d + 8 * (g0 + g) * es + j0
                st64(dst, e, b0)
                st64(dst, e + es, b1)
                st64(dst, e + 2 * es, b2)
                st64(dst, e + 3 * es, b3)
                st64(dst, e + 4 * es, b4)
                st64(dst, e + 5 * es, b5)
                st64(dst, e + 6 * es, b6)
                st64(dst, e + 7 * es, b7)
        g0 += 8
    # the rest a group of 8 elements and a byte at a time
    for g in range(g0, nb):
        for j in range(es):
            r = s + 8 * j * nb + g
            x = np.uint64(0)
            for k in range(8):
                x |= np.uint64(src[r + k * nb]) << np.uint64(8 * k)
            x = _bits8(x)
            e = d + 8 * g * es + j
            for b in range(8):
                dst[e + b * es] = np.uint8(x & np.uint64(255))
                x >>= np.uint64(8)


def default_block(es):
    """bshuf_default_block_size: elements per block when cd says 0."""
    b = (8192 // es) // 8 * 8
    return max(b, 128)


@njit(cache=True, nogil=True)
def _default_block(es):
    b = (8192 // es) // 8 * 8
    return max(b, 128)


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """The bitshuffle filter: src[p:pend] into dst[o:oend]."""
    ncd = cd[2]
    if ncd < 3:
        return ERR_DATA
    es = cd[5]
    block = cd[6] if ncd > 3 else 0
    comp = cd[7] if ncd > 4 else 0
    if es <= 0:
        return ERR_DATA
    if comp == 2 or comp == 3:
        if pend - p < 12:
            return ERR_INPUT
        total = be64(src, p)
        block = be32(src, p + 8) // es
        q = p + 12
        if total < 0:
            return ERR_DATA
    elif comp == 0:
        total = pend - p
        q = p
    else:
        return ERR_DATA
    if total % es:
        return ERR_DATA
    if total > oend - o:
        return ERR_SPACE
    size = total // es
    if block == 0:
        block = _default_block(es)
    if block % 8:
        return ERR_DATA
    y = ws[W_Y:W_Y + 512].view(np.uint64)
    nfull = size // block
    last = size % block
    last -= last % 8
    d = o
    for i in range(nfull + (1 if last else 0)):
        bn = block if i < nfull else last
        nbytes = bn * es
        if comp:
            if q + 4 > pend:
                return ERR_INPUT
            c = be32(src, q)
            q += 4
            if c > pend - q:
                return ERR_INPUT
            if W_T + nbytes > ws.shape[0]:
                return ERR_SPACE
            if comp == 2:
                r = lz4_block(src, q, q + c, ws, W_T, W_T, W_T + nbytes)
            else:
                r = zstd_decompress(src, q, q + c, ws, W_T, W_T + nbytes, ws[W_Z:W_Z + ZWS])
            if r < 0:
                return r
            if r != W_T + nbytes:
                return ERR_DATA
            bitunshuffle(ws, W_T, dst, d, bn, es, oend, y)
            q += c
        else:
            if nbytes > pend - q:
                return ERR_INPUT
            bitunshuffle(src, q, dst, d, bn, es, oend, y)
            q += nbytes
        d += nbytes
    n = (size % 8) * es
    if n > pend - q:
        return ERR_INPUT
    copy_bytes(src, q, dst, d, n)
    return o + total


def workspace_size(chunk_nbytes):
    """The transpose scratch, zstd's, and one decompressed block (at most
    the chunk)."""
    return W_T + chunk_nbytes + 64
