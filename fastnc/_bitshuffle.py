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


S32, S16, S8, S4, S2, S1 = (np.uint64(32), np.uint64(16), np.uint64(8), np.uint64(4),
                            np.uint64(2), np.uint64(1))
M32 = np.uint64(0x00000000FFFFFFFF)
M16 = np.uint64(0x0000FFFF0000FFFF)
M8 = np.uint64(0x00FF00FF00FF00FF)
M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
M2 = np.uint64(0x3333333333333333)
M1 = np.uint64(0x5555555555555555)


@njit(cache=True, nogil=True)
def _sw(a, c, sh, m):
    """Swap the bits of a at lane columns with bit `sh` set with those of c
    at columns with it clear (one block swap of a bit transpose)."""
    t = ((a >> sh) ^ c) & m
    return a ^ (t << sh), c ^ t


# The bit transposes of 64 elements, one function per element size. A
# block's bit rows are read 8 bytes (64 elements) at a time: R = 8 * es
# words, word r holding bit r of the 64 elements. The R x R bit matrices in
# each R-bit lane of the words are transposed by log2(R) rounds of block
# swaps (_sw), after which word c, lane L holds element L*R + c whole;
# the words are then put in element order. The swaps are written out (as
# a loop over the rounds would generate them) so that the words stay in
# registers: a loop over a scratch array runs at less than half the speed.


@njit(cache=True, nogil=True)
def _tile8(src, r0, nb, dst, e0):
    """64 one-byte elements."""
    x0, x1, x2, x3 = (ld64(src, r0), ld64(src, r0 + 1 * nb), ld64(src, r0 + 2 * nb), ld64(src, r0 + 3 * nb))
    x4, x5, x6, x7 = (ld64(src, r0 + 4 * nb), ld64(src, r0 + 5 * nb), ld64(src, r0 + 6 * nb), ld64(src, r0 + 7 * nb))
    x0, x4 = _sw(x0, x4, S4, M4)
    x1, x5 = _sw(x1, x5, S4, M4)
    x2, x6 = _sw(x2, x6, S4, M4)
    x3, x7 = _sw(x3, x7, S4, M4)
    x0, x2 = _sw(x0, x2, S2, M2)
    x1, x3 = _sw(x1, x3, S2, M2)
    x4, x6 = _sw(x4, x6, S2, M2)
    x5, x7 = _sw(x5, x7, S2, M2)
    x0, x1 = _sw(x0, x1, S1, M1)
    x2, x3 = _sw(x2, x3, S1, M1)
    x4, x5 = _sw(x4, x5, S1, M1)
    x6, x7 = _sw(x6, x7, S1, M1)
    b0, b1, b2, b3, b4, b5, b6, b7 = _bytes8(x0, x1, x2, x3, x4, x5, x6, x7)
    st64(dst, e0, b0)
    st64(dst, e0 + 8, b1)
    st64(dst, e0 + 16, b2)
    st64(dst, e0 + 24, b3)
    st64(dst, e0 + 32, b4)
    st64(dst, e0 + 40, b5)
    st64(dst, e0 + 48, b6)
    st64(dst, e0 + 56, b7)


@njit(cache=True, nogil=True)
def _put16(dst, e, a0, a1, a2, a3):
    """Words a0..a3 hold 16-bit elements e, e+16, e+32, e+48 (+1, +2, +3)
    in their lanes: a 4x4 transpose of the lanes puts 4 consecutive
    elements in each word."""
    a0, a2 = _sw(a0, a2, S32, M32)
    a1, a3 = _sw(a1, a3, S32, M32)
    a0, a1 = _sw(a0, a1, S16, M16)
    a2, a3 = _sw(a2, a3, S16, M16)
    st64(dst, e, a0)
    st64(dst, e + 32, a1)
    st64(dst, e + 64, a2)
    st64(dst, e + 96, a3)


@njit(cache=True, nogil=True)
def _tile16(src, r0, nb, dst, e0):
    """64 two-byte elements."""
    x0, x1, x2, x3 = (ld64(src, r0), ld64(src, r0 + 1 * nb), ld64(src, r0 + 2 * nb), ld64(src, r0 + 3 * nb))
    x4, x5, x6, x7 = (ld64(src, r0 + 4 * nb), ld64(src, r0 + 5 * nb), ld64(src, r0 + 6 * nb), ld64(src, r0 + 7 * nb))
    x8, x9, x10, x11 = (ld64(src, r0 + 8 * nb), ld64(src, r0 + 9 * nb), ld64(src, r0 + 10 * nb), ld64(src, r0 + 11 * nb))
    x12, x13, x14, x15 = (ld64(src, r0 + 12 * nb), ld64(src, r0 + 13 * nb), ld64(src, r0 + 14 * nb), ld64(src, r0 + 15 * nb))
    x0, x8 = _sw(x0, x8, S8, M8)
    x1, x9 = _sw(x1, x9, S8, M8)
    x2, x10 = _sw(x2, x10, S8, M8)
    x3, x11 = _sw(x3, x11, S8, M8)
    x4, x12 = _sw(x4, x12, S8, M8)
    x5, x13 = _sw(x5, x13, S8, M8)
    x6, x14 = _sw(x6, x14, S8, M8)
    x7, x15 = _sw(x7, x15, S8, M8)
    x0, x4 = _sw(x0, x4, S4, M4)
    x1, x5 = _sw(x1, x5, S4, M4)
    x2, x6 = _sw(x2, x6, S4, M4)
    x3, x7 = _sw(x3, x7, S4, M4)
    x8, x12 = _sw(x8, x12, S4, M4)
    x9, x13 = _sw(x9, x13, S4, M4)
    x10, x14 = _sw(x10, x14, S4, M4)
    x11, x15 = _sw(x11, x15, S4, M4)
    x0, x2 = _sw(x0, x2, S2, M2)
    x1, x3 = _sw(x1, x3, S2, M2)
    x4, x6 = _sw(x4, x6, S2, M2)
    x5, x7 = _sw(x5, x7, S2, M2)
    x8, x10 = _sw(x8, x10, S2, M2)
    x9, x11 = _sw(x9, x11, S2, M2)
    x12, x14 = _sw(x12, x14, S2, M2)
    x13, x15 = _sw(x13, x15, S2, M2)
    x0, x1 = _sw(x0, x1, S1, M1)
    x2, x3 = _sw(x2, x3, S1, M1)
    x4, x5 = _sw(x4, x5, S1, M1)
    x6, x7 = _sw(x6, x7, S1, M1)
    x8, x9 = _sw(x8, x9, S1, M1)
    x10, x11 = _sw(x10, x11, S1, M1)
    x12, x13 = _sw(x12, x13, S1, M1)
    x14, x15 = _sw(x14, x15, S1, M1)
    _put16(dst, e0, x0, x1, x2, x3)
    _put16(dst, e0 + 8, x4, x5, x6, x7)
    _put16(dst, e0 + 16, x8, x9, x10, x11)
    _put16(dst, e0 + 24, x12, x13, x14, x15)


@njit(cache=True, nogil=True)
def _put32(dst, e, a, b):
    """Words a, b hold 32-bit elements (i, 32+i), (i+1, 33+i)."""
    st64(dst, e, (a & M32) | (b << S32))
    st64(dst, e + 128, (a >> S32) | (b & ~M32))


@njit(cache=True, nogil=True)
def _tile32(src, r0, nb, dst, e0):
    """64 four-byte elements."""
    x0, x1, x2, x3 = (ld64(src, r0), ld64(src, r0 + 1 * nb), ld64(src, r0 + 2 * nb), ld64(src, r0 + 3 * nb))
    x4, x5, x6, x7 = (ld64(src, r0 + 4 * nb), ld64(src, r0 + 5 * nb), ld64(src, r0 + 6 * nb), ld64(src, r0 + 7 * nb))
    x8, x9, x10, x11 = (ld64(src, r0 + 8 * nb), ld64(src, r0 + 9 * nb), ld64(src, r0 + 10 * nb), ld64(src, r0 + 11 * nb))
    x12, x13, x14, x15 = (ld64(src, r0 + 12 * nb), ld64(src, r0 + 13 * nb), ld64(src, r0 + 14 * nb), ld64(src, r0 + 15 * nb))
    x16, x17, x18, x19 = (ld64(src, r0 + 16 * nb), ld64(src, r0 + 17 * nb), ld64(src, r0 + 18 * nb), ld64(src, r0 + 19 * nb))
    x20, x21, x22, x23 = (ld64(src, r0 + 20 * nb), ld64(src, r0 + 21 * nb), ld64(src, r0 + 22 * nb), ld64(src, r0 + 23 * nb))
    x24, x25, x26, x27 = (ld64(src, r0 + 24 * nb), ld64(src, r0 + 25 * nb), ld64(src, r0 + 26 * nb), ld64(src, r0 + 27 * nb))
    x28, x29, x30, x31 = (ld64(src, r0 + 28 * nb), ld64(src, r0 + 29 * nb), ld64(src, r0 + 30 * nb), ld64(src, r0 + 31 * nb))
    x0, x16 = _sw(x0, x16, S16, M16)
    x1, x17 = _sw(x1, x17, S16, M16)
    x2, x18 = _sw(x2, x18, S16, M16)
    x3, x19 = _sw(x3, x19, S16, M16)
    x4, x20 = _sw(x4, x20, S16, M16)
    x5, x21 = _sw(x5, x21, S16, M16)
    x6, x22 = _sw(x6, x22, S16, M16)
    x7, x23 = _sw(x7, x23, S16, M16)
    x8, x24 = _sw(x8, x24, S16, M16)
    x9, x25 = _sw(x9, x25, S16, M16)
    x10, x26 = _sw(x10, x26, S16, M16)
    x11, x27 = _sw(x11, x27, S16, M16)
    x12, x28 = _sw(x12, x28, S16, M16)
    x13, x29 = _sw(x13, x29, S16, M16)
    x14, x30 = _sw(x14, x30, S16, M16)
    x15, x31 = _sw(x15, x31, S16, M16)
    x0, x8 = _sw(x0, x8, S8, M8)
    x1, x9 = _sw(x1, x9, S8, M8)
    x2, x10 = _sw(x2, x10, S8, M8)
    x3, x11 = _sw(x3, x11, S8, M8)
    x4, x12 = _sw(x4, x12, S8, M8)
    x5, x13 = _sw(x5, x13, S8, M8)
    x6, x14 = _sw(x6, x14, S8, M8)
    x7, x15 = _sw(x7, x15, S8, M8)
    x16, x24 = _sw(x16, x24, S8, M8)
    x17, x25 = _sw(x17, x25, S8, M8)
    x18, x26 = _sw(x18, x26, S8, M8)
    x19, x27 = _sw(x19, x27, S8, M8)
    x20, x28 = _sw(x20, x28, S8, M8)
    x21, x29 = _sw(x21, x29, S8, M8)
    x22, x30 = _sw(x22, x30, S8, M8)
    x23, x31 = _sw(x23, x31, S8, M8)
    x0, x4 = _sw(x0, x4, S4, M4)
    x1, x5 = _sw(x1, x5, S4, M4)
    x2, x6 = _sw(x2, x6, S4, M4)
    x3, x7 = _sw(x3, x7, S4, M4)
    x8, x12 = _sw(x8, x12, S4, M4)
    x9, x13 = _sw(x9, x13, S4, M4)
    x10, x14 = _sw(x10, x14, S4, M4)
    x11, x15 = _sw(x11, x15, S4, M4)
    x16, x20 = _sw(x16, x20, S4, M4)
    x17, x21 = _sw(x17, x21, S4, M4)
    x18, x22 = _sw(x18, x22, S4, M4)
    x19, x23 = _sw(x19, x23, S4, M4)
    x24, x28 = _sw(x24, x28, S4, M4)
    x25, x29 = _sw(x25, x29, S4, M4)
    x26, x30 = _sw(x26, x30, S4, M4)
    x27, x31 = _sw(x27, x31, S4, M4)
    x0, x2 = _sw(x0, x2, S2, M2)
    x1, x3 = _sw(x1, x3, S2, M2)
    x4, x6 = _sw(x4, x6, S2, M2)
    x5, x7 = _sw(x5, x7, S2, M2)
    x8, x10 = _sw(x8, x10, S2, M2)
    x9, x11 = _sw(x9, x11, S2, M2)
    x12, x14 = _sw(x12, x14, S2, M2)
    x13, x15 = _sw(x13, x15, S2, M2)
    x16, x18 = _sw(x16, x18, S2, M2)
    x17, x19 = _sw(x17, x19, S2, M2)
    x20, x22 = _sw(x20, x22, S2, M2)
    x21, x23 = _sw(x21, x23, S2, M2)
    x24, x26 = _sw(x24, x26, S2, M2)
    x25, x27 = _sw(x25, x27, S2, M2)
    x28, x30 = _sw(x28, x30, S2, M2)
    x29, x31 = _sw(x29, x31, S2, M2)
    x0, x1 = _sw(x0, x1, S1, M1)
    x2, x3 = _sw(x2, x3, S1, M1)
    x4, x5 = _sw(x4, x5, S1, M1)
    x6, x7 = _sw(x6, x7, S1, M1)
    x8, x9 = _sw(x8, x9, S1, M1)
    x10, x11 = _sw(x10, x11, S1, M1)
    x12, x13 = _sw(x12, x13, S1, M1)
    x14, x15 = _sw(x14, x15, S1, M1)
    x16, x17 = _sw(x16, x17, S1, M1)
    x18, x19 = _sw(x18, x19, S1, M1)
    x20, x21 = _sw(x20, x21, S1, M1)
    x22, x23 = _sw(x22, x23, S1, M1)
    x24, x25 = _sw(x24, x25, S1, M1)
    x26, x27 = _sw(x26, x27, S1, M1)
    x28, x29 = _sw(x28, x29, S1, M1)
    x30, x31 = _sw(x30, x31, S1, M1)
    _put32(dst, e0 + 0, x0, x1)
    _put32(dst, e0 + 8, x2, x3)
    _put32(dst, e0 + 16, x4, x5)
    _put32(dst, e0 + 24, x6, x7)
    _put32(dst, e0 + 32, x8, x9)
    _put32(dst, e0 + 40, x10, x11)
    _put32(dst, e0 + 48, x12, x13)
    _put32(dst, e0 + 56, x14, x15)
    _put32(dst, e0 + 64, x16, x17)
    _put32(dst, e0 + 72, x18, x19)
    _put32(dst, e0 + 80, x20, x21)
    _put32(dst, e0 + 88, x22, x23)
    _put32(dst, e0 + 96, x24, x25)
    _put32(dst, e0 + 104, x26, x27)
    _put32(dst, e0 + 112, x28, x29)
    _put32(dst, e0 + 120, x30, x31)


@njit(cache=True, nogil=True)
def _tile64(src, r0, nb, dst, e0, es):
    """Eight bytes of each of 64 elements of es bytes (a multiple of 8): bit
    rows r0.., to dst[e0 + c * es : +8] for element c."""
    x0, x1, x2, x3 = (ld64(src, r0), ld64(src, r0 + 1 * nb), ld64(src, r0 + 2 * nb), ld64(src, r0 + 3 * nb))
    x4, x5, x6, x7 = (ld64(src, r0 + 4 * nb), ld64(src, r0 + 5 * nb), ld64(src, r0 + 6 * nb), ld64(src, r0 + 7 * nb))
    x8, x9, x10, x11 = (ld64(src, r0 + 8 * nb), ld64(src, r0 + 9 * nb), ld64(src, r0 + 10 * nb), ld64(src, r0 + 11 * nb))
    x12, x13, x14, x15 = (ld64(src, r0 + 12 * nb), ld64(src, r0 + 13 * nb), ld64(src, r0 + 14 * nb), ld64(src, r0 + 15 * nb))
    x16, x17, x18, x19 = (ld64(src, r0 + 16 * nb), ld64(src, r0 + 17 * nb), ld64(src, r0 + 18 * nb), ld64(src, r0 + 19 * nb))
    x20, x21, x22, x23 = (ld64(src, r0 + 20 * nb), ld64(src, r0 + 21 * nb), ld64(src, r0 + 22 * nb), ld64(src, r0 + 23 * nb))
    x24, x25, x26, x27 = (ld64(src, r0 + 24 * nb), ld64(src, r0 + 25 * nb), ld64(src, r0 + 26 * nb), ld64(src, r0 + 27 * nb))
    x28, x29, x30, x31 = (ld64(src, r0 + 28 * nb), ld64(src, r0 + 29 * nb), ld64(src, r0 + 30 * nb), ld64(src, r0 + 31 * nb))
    x32, x33, x34, x35 = (ld64(src, r0 + 32 * nb), ld64(src, r0 + 33 * nb), ld64(src, r0 + 34 * nb), ld64(src, r0 + 35 * nb))
    x36, x37, x38, x39 = (ld64(src, r0 + 36 * nb), ld64(src, r0 + 37 * nb), ld64(src, r0 + 38 * nb), ld64(src, r0 + 39 * nb))
    x40, x41, x42, x43 = (ld64(src, r0 + 40 * nb), ld64(src, r0 + 41 * nb), ld64(src, r0 + 42 * nb), ld64(src, r0 + 43 * nb))
    x44, x45, x46, x47 = (ld64(src, r0 + 44 * nb), ld64(src, r0 + 45 * nb), ld64(src, r0 + 46 * nb), ld64(src, r0 + 47 * nb))
    x48, x49, x50, x51 = (ld64(src, r0 + 48 * nb), ld64(src, r0 + 49 * nb), ld64(src, r0 + 50 * nb), ld64(src, r0 + 51 * nb))
    x52, x53, x54, x55 = (ld64(src, r0 + 52 * nb), ld64(src, r0 + 53 * nb), ld64(src, r0 + 54 * nb), ld64(src, r0 + 55 * nb))
    x56, x57, x58, x59 = (ld64(src, r0 + 56 * nb), ld64(src, r0 + 57 * nb), ld64(src, r0 + 58 * nb), ld64(src, r0 + 59 * nb))
    x60, x61, x62, x63 = (ld64(src, r0 + 60 * nb), ld64(src, r0 + 61 * nb), ld64(src, r0 + 62 * nb), ld64(src, r0 + 63 * nb))
    x0, x32 = _sw(x0, x32, S32, M32)
    x1, x33 = _sw(x1, x33, S32, M32)
    x2, x34 = _sw(x2, x34, S32, M32)
    x3, x35 = _sw(x3, x35, S32, M32)
    x4, x36 = _sw(x4, x36, S32, M32)
    x5, x37 = _sw(x5, x37, S32, M32)
    x6, x38 = _sw(x6, x38, S32, M32)
    x7, x39 = _sw(x7, x39, S32, M32)
    x8, x40 = _sw(x8, x40, S32, M32)
    x9, x41 = _sw(x9, x41, S32, M32)
    x10, x42 = _sw(x10, x42, S32, M32)
    x11, x43 = _sw(x11, x43, S32, M32)
    x12, x44 = _sw(x12, x44, S32, M32)
    x13, x45 = _sw(x13, x45, S32, M32)
    x14, x46 = _sw(x14, x46, S32, M32)
    x15, x47 = _sw(x15, x47, S32, M32)
    x16, x48 = _sw(x16, x48, S32, M32)
    x17, x49 = _sw(x17, x49, S32, M32)
    x18, x50 = _sw(x18, x50, S32, M32)
    x19, x51 = _sw(x19, x51, S32, M32)
    x20, x52 = _sw(x20, x52, S32, M32)
    x21, x53 = _sw(x21, x53, S32, M32)
    x22, x54 = _sw(x22, x54, S32, M32)
    x23, x55 = _sw(x23, x55, S32, M32)
    x24, x56 = _sw(x24, x56, S32, M32)
    x25, x57 = _sw(x25, x57, S32, M32)
    x26, x58 = _sw(x26, x58, S32, M32)
    x27, x59 = _sw(x27, x59, S32, M32)
    x28, x60 = _sw(x28, x60, S32, M32)
    x29, x61 = _sw(x29, x61, S32, M32)
    x30, x62 = _sw(x30, x62, S32, M32)
    x31, x63 = _sw(x31, x63, S32, M32)
    x0, x16 = _sw(x0, x16, S16, M16)
    x1, x17 = _sw(x1, x17, S16, M16)
    x2, x18 = _sw(x2, x18, S16, M16)
    x3, x19 = _sw(x3, x19, S16, M16)
    x4, x20 = _sw(x4, x20, S16, M16)
    x5, x21 = _sw(x5, x21, S16, M16)
    x6, x22 = _sw(x6, x22, S16, M16)
    x7, x23 = _sw(x7, x23, S16, M16)
    x8, x24 = _sw(x8, x24, S16, M16)
    x9, x25 = _sw(x9, x25, S16, M16)
    x10, x26 = _sw(x10, x26, S16, M16)
    x11, x27 = _sw(x11, x27, S16, M16)
    x12, x28 = _sw(x12, x28, S16, M16)
    x13, x29 = _sw(x13, x29, S16, M16)
    x14, x30 = _sw(x14, x30, S16, M16)
    x15, x31 = _sw(x15, x31, S16, M16)
    x32, x48 = _sw(x32, x48, S16, M16)
    x33, x49 = _sw(x33, x49, S16, M16)
    x34, x50 = _sw(x34, x50, S16, M16)
    x35, x51 = _sw(x35, x51, S16, M16)
    x36, x52 = _sw(x36, x52, S16, M16)
    x37, x53 = _sw(x37, x53, S16, M16)
    x38, x54 = _sw(x38, x54, S16, M16)
    x39, x55 = _sw(x39, x55, S16, M16)
    x40, x56 = _sw(x40, x56, S16, M16)
    x41, x57 = _sw(x41, x57, S16, M16)
    x42, x58 = _sw(x42, x58, S16, M16)
    x43, x59 = _sw(x43, x59, S16, M16)
    x44, x60 = _sw(x44, x60, S16, M16)
    x45, x61 = _sw(x45, x61, S16, M16)
    x46, x62 = _sw(x46, x62, S16, M16)
    x47, x63 = _sw(x47, x63, S16, M16)
    x0, x8 = _sw(x0, x8, S8, M8)
    x1, x9 = _sw(x1, x9, S8, M8)
    x2, x10 = _sw(x2, x10, S8, M8)
    x3, x11 = _sw(x3, x11, S8, M8)
    x4, x12 = _sw(x4, x12, S8, M8)
    x5, x13 = _sw(x5, x13, S8, M8)
    x6, x14 = _sw(x6, x14, S8, M8)
    x7, x15 = _sw(x7, x15, S8, M8)
    x16, x24 = _sw(x16, x24, S8, M8)
    x17, x25 = _sw(x17, x25, S8, M8)
    x18, x26 = _sw(x18, x26, S8, M8)
    x19, x27 = _sw(x19, x27, S8, M8)
    x20, x28 = _sw(x20, x28, S8, M8)
    x21, x29 = _sw(x21, x29, S8, M8)
    x22, x30 = _sw(x22, x30, S8, M8)
    x23, x31 = _sw(x23, x31, S8, M8)
    x32, x40 = _sw(x32, x40, S8, M8)
    x33, x41 = _sw(x33, x41, S8, M8)
    x34, x42 = _sw(x34, x42, S8, M8)
    x35, x43 = _sw(x35, x43, S8, M8)
    x36, x44 = _sw(x36, x44, S8, M8)
    x37, x45 = _sw(x37, x45, S8, M8)
    x38, x46 = _sw(x38, x46, S8, M8)
    x39, x47 = _sw(x39, x47, S8, M8)
    x48, x56 = _sw(x48, x56, S8, M8)
    x49, x57 = _sw(x49, x57, S8, M8)
    x50, x58 = _sw(x50, x58, S8, M8)
    x51, x59 = _sw(x51, x59, S8, M8)
    x52, x60 = _sw(x52, x60, S8, M8)
    x53, x61 = _sw(x53, x61, S8, M8)
    x54, x62 = _sw(x54, x62, S8, M8)
    x55, x63 = _sw(x55, x63, S8, M8)
    x0, x4 = _sw(x0, x4, S4, M4)
    x1, x5 = _sw(x1, x5, S4, M4)
    x2, x6 = _sw(x2, x6, S4, M4)
    x3, x7 = _sw(x3, x7, S4, M4)
    x8, x12 = _sw(x8, x12, S4, M4)
    x9, x13 = _sw(x9, x13, S4, M4)
    x10, x14 = _sw(x10, x14, S4, M4)
    x11, x15 = _sw(x11, x15, S4, M4)
    x16, x20 = _sw(x16, x20, S4, M4)
    x17, x21 = _sw(x17, x21, S4, M4)
    x18, x22 = _sw(x18, x22, S4, M4)
    x19, x23 = _sw(x19, x23, S4, M4)
    x24, x28 = _sw(x24, x28, S4, M4)
    x25, x29 = _sw(x25, x29, S4, M4)
    x26, x30 = _sw(x26, x30, S4, M4)
    x27, x31 = _sw(x27, x31, S4, M4)
    x32, x36 = _sw(x32, x36, S4, M4)
    x33, x37 = _sw(x33, x37, S4, M4)
    x34, x38 = _sw(x34, x38, S4, M4)
    x35, x39 = _sw(x35, x39, S4, M4)
    x40, x44 = _sw(x40, x44, S4, M4)
    x41, x45 = _sw(x41, x45, S4, M4)
    x42, x46 = _sw(x42, x46, S4, M4)
    x43, x47 = _sw(x43, x47, S4, M4)
    x48, x52 = _sw(x48, x52, S4, M4)
    x49, x53 = _sw(x49, x53, S4, M4)
    x50, x54 = _sw(x50, x54, S4, M4)
    x51, x55 = _sw(x51, x55, S4, M4)
    x56, x60 = _sw(x56, x60, S4, M4)
    x57, x61 = _sw(x57, x61, S4, M4)
    x58, x62 = _sw(x58, x62, S4, M4)
    x59, x63 = _sw(x59, x63, S4, M4)
    x0, x2 = _sw(x0, x2, S2, M2)
    x1, x3 = _sw(x1, x3, S2, M2)
    x4, x6 = _sw(x4, x6, S2, M2)
    x5, x7 = _sw(x5, x7, S2, M2)
    x8, x10 = _sw(x8, x10, S2, M2)
    x9, x11 = _sw(x9, x11, S2, M2)
    x12, x14 = _sw(x12, x14, S2, M2)
    x13, x15 = _sw(x13, x15, S2, M2)
    x16, x18 = _sw(x16, x18, S2, M2)
    x17, x19 = _sw(x17, x19, S2, M2)
    x20, x22 = _sw(x20, x22, S2, M2)
    x21, x23 = _sw(x21, x23, S2, M2)
    x24, x26 = _sw(x24, x26, S2, M2)
    x25, x27 = _sw(x25, x27, S2, M2)
    x28, x30 = _sw(x28, x30, S2, M2)
    x29, x31 = _sw(x29, x31, S2, M2)
    x32, x34 = _sw(x32, x34, S2, M2)
    x33, x35 = _sw(x33, x35, S2, M2)
    x36, x38 = _sw(x36, x38, S2, M2)
    x37, x39 = _sw(x37, x39, S2, M2)
    x40, x42 = _sw(x40, x42, S2, M2)
    x41, x43 = _sw(x41, x43, S2, M2)
    x44, x46 = _sw(x44, x46, S2, M2)
    x45, x47 = _sw(x45, x47, S2, M2)
    x48, x50 = _sw(x48, x50, S2, M2)
    x49, x51 = _sw(x49, x51, S2, M2)
    x52, x54 = _sw(x52, x54, S2, M2)
    x53, x55 = _sw(x53, x55, S2, M2)
    x56, x58 = _sw(x56, x58, S2, M2)
    x57, x59 = _sw(x57, x59, S2, M2)
    x60, x62 = _sw(x60, x62, S2, M2)
    x61, x63 = _sw(x61, x63, S2, M2)
    x0, x1 = _sw(x0, x1, S1, M1)
    x2, x3 = _sw(x2, x3, S1, M1)
    x4, x5 = _sw(x4, x5, S1, M1)
    x6, x7 = _sw(x6, x7, S1, M1)
    x8, x9 = _sw(x8, x9, S1, M1)
    x10, x11 = _sw(x10, x11, S1, M1)
    x12, x13 = _sw(x12, x13, S1, M1)
    x14, x15 = _sw(x14, x15, S1, M1)
    x16, x17 = _sw(x16, x17, S1, M1)
    x18, x19 = _sw(x18, x19, S1, M1)
    x20, x21 = _sw(x20, x21, S1, M1)
    x22, x23 = _sw(x22, x23, S1, M1)
    x24, x25 = _sw(x24, x25, S1, M1)
    x26, x27 = _sw(x26, x27, S1, M1)
    x28, x29 = _sw(x28, x29, S1, M1)
    x30, x31 = _sw(x30, x31, S1, M1)
    x32, x33 = _sw(x32, x33, S1, M1)
    x34, x35 = _sw(x34, x35, S1, M1)
    x36, x37 = _sw(x36, x37, S1, M1)
    x38, x39 = _sw(x38, x39, S1, M1)
    x40, x41 = _sw(x40, x41, S1, M1)
    x42, x43 = _sw(x42, x43, S1, M1)
    x44, x45 = _sw(x44, x45, S1, M1)
    x46, x47 = _sw(x46, x47, S1, M1)
    x48, x49 = _sw(x48, x49, S1, M1)
    x50, x51 = _sw(x50, x51, S1, M1)
    x52, x53 = _sw(x52, x53, S1, M1)
    x54, x55 = _sw(x54, x55, S1, M1)
    x56, x57 = _sw(x56, x57, S1, M1)
    x58, x59 = _sw(x58, x59, S1, M1)
    x60, x61 = _sw(x60, x61, S1, M1)
    x62, x63 = _sw(x62, x63, S1, M1)
    st64(dst, e0, x0)
    st64(dst, e0 + 1 * es, x1)
    st64(dst, e0 + 2 * es, x2)
    st64(dst, e0 + 3 * es, x3)
    st64(dst, e0 + 4 * es, x4)
    st64(dst, e0 + 5 * es, x5)
    st64(dst, e0 + 6 * es, x6)
    st64(dst, e0 + 7 * es, x7)
    st64(dst, e0 + 8 * es, x8)
    st64(dst, e0 + 9 * es, x9)
    st64(dst, e0 + 10 * es, x10)
    st64(dst, e0 + 11 * es, x11)
    st64(dst, e0 + 12 * es, x12)
    st64(dst, e0 + 13 * es, x13)
    st64(dst, e0 + 14 * es, x14)
    st64(dst, e0 + 15 * es, x15)
    st64(dst, e0 + 16 * es, x16)
    st64(dst, e0 + 17 * es, x17)
    st64(dst, e0 + 18 * es, x18)
    st64(dst, e0 + 19 * es, x19)
    st64(dst, e0 + 20 * es, x20)
    st64(dst, e0 + 21 * es, x21)
    st64(dst, e0 + 22 * es, x22)
    st64(dst, e0 + 23 * es, x23)
    st64(dst, e0 + 24 * es, x24)
    st64(dst, e0 + 25 * es, x25)
    st64(dst, e0 + 26 * es, x26)
    st64(dst, e0 + 27 * es, x27)
    st64(dst, e0 + 28 * es, x28)
    st64(dst, e0 + 29 * es, x29)
    st64(dst, e0 + 30 * es, x30)
    st64(dst, e0 + 31 * es, x31)
    st64(dst, e0 + 32 * es, x32)
    st64(dst, e0 + 33 * es, x33)
    st64(dst, e0 + 34 * es, x34)
    st64(dst, e0 + 35 * es, x35)
    st64(dst, e0 + 36 * es, x36)
    st64(dst, e0 + 37 * es, x37)
    st64(dst, e0 + 38 * es, x38)
    st64(dst, e0 + 39 * es, x39)
    st64(dst, e0 + 40 * es, x40)
    st64(dst, e0 + 41 * es, x41)
    st64(dst, e0 + 42 * es, x42)
    st64(dst, e0 + 43 * es, x43)
    st64(dst, e0 + 44 * es, x44)
    st64(dst, e0 + 45 * es, x45)
    st64(dst, e0 + 46 * es, x46)
    st64(dst, e0 + 47 * es, x47)
    st64(dst, e0 + 48 * es, x48)
    st64(dst, e0 + 49 * es, x49)
    st64(dst, e0 + 50 * es, x50)
    st64(dst, e0 + 51 * es, x51)
    st64(dst, e0 + 52 * es, x52)
    st64(dst, e0 + 53 * es, x53)
    st64(dst, e0 + 54 * es, x54)
    st64(dst, e0 + 55 * es, x55)
    st64(dst, e0 + 56 * es, x56)
    st64(dst, e0 + 57 * es, x57)
    st64(dst, e0 + 58 * es, x58)
    st64(dst, e0 + 59 * es, x59)
    st64(dst, e0 + 60 * es, x60)
    st64(dst, e0 + 61 * es, x61)
    st64(dst, e0 + 62 * es, x62)
    st64(dst, e0 + 63 * es, x63)


@njit(cache=True, nogil=True)
def _tile(src, r0, nb, dst, e0, es):
    """64 elements of es bytes (1, 2, 4 or a multiple of 8) from their bit
    rows (src[r0 + r*nb : +8] for bit row r) to dst[e0 : e0 + 64*es]."""
    if es == 4:
        _tile32(src, r0, nb, dst, e0)
    elif es == 2:
        _tile16(src, r0, nb, dst, e0)
    elif es == 1:
        _tile8(src, r0, nb, dst, e0)
    else:
        for j0 in range(0, es, 8):
            _tile64(src, r0 + 8 * j0 * nb, nb, dst, e0 + j0, es)


@njit(cache=True, nogil=True)
def bitunshuffle(src, s, dst, d, n, es, dend, y):
    """Undo the bit transpose of one block: n elements (a multiple of 8) of
    es bytes, src[s:s+n*es] -> dst[d:d+n*es]; dst is writable up to dend
    (word stores may run past the block's end when there is room). y: 64
    uint64 of scratch."""
    nb = n >> 3                                 # bytes per bit row
    g0 = 0
    if es == 1 or es == 2 or es == 4 or (es & 7) == 0:
        # 64 elements at a time: bit rows as words, transposed in lanes
        while g0 + 8 <= nb:
            _tile(src, s + g0, nb, dst, d + 8 * g0 * es, es)
            g0 += 8
    words = es <= 8
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
