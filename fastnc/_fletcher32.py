"""HDF5's Fletcher-32 checksum (filter 3): the last 4 bytes of a chunk are
a checksum of the rest; reading checks it and strips it.

The checksum is H5checksum.c's H5_checksum_fletcher32: 16-bit words taken
big-endian, both sums folded to 16 bits (end-around carry) after every 360
words and at the end, a trailing odd byte as the high byte of a last word.
It is stored little-endian. Like HDF5, a checksum is also accepted with
the bytes of each 16-bit half swapped (what HDF5 before 1.6.3 wrote on
little-endian machines).

Within a run of 360 words the sums cannot overflow 32 bits, so a run is
summed with plain 64-bit integers four words at a time (sum2 grows by
4 sum1 + 4a + 3b + 2c + d) -- short dependency chains instead of one long
one -- and folded exactly where HDF5 folds: the result is bit for bit
HDF5's.
"""
import numpy as np

from ._intrin import ld64, st64
from ._jit import njit

ERR_DATA = -1                   # checksum mismatch
ERR_SPACE = -2
ERR_INPUT = -3                  # shorter than the checksum


@njit(cache=True, nogil=True)
def fletcher32(src, p, n):
    """H5_checksum_fletcher32 of src[p:p+n]."""
    s1 = np.int64(0)
    s2 = np.int64(0)
    words = n >> 1
    q = p
    while words > 0:
        t = min(words, 360)
        words -= t
        e = q + 2 * t
        while q + 8 <= e:                       # four big-endian words from one load
            x = ld64(src, q)
            a = np.int64(((x & np.uint64(0xFF)) << np.uint64(8)) | ((x >> np.uint64(8)) & np.uint64(0xFF)))
            b = np.int64(((x >> np.uint64(8)) & np.uint64(0xFF00)) | ((x >> np.uint64(24)) & np.uint64(0xFF)))
            c = np.int64(((x >> np.uint64(24)) & np.uint64(0xFF00)) | ((x >> np.uint64(40)) & np.uint64(0xFF)))
            d = np.int64(((x >> np.uint64(40)) & np.uint64(0xFF00)) | (x >> np.uint64(56)))
            s2 += 4 * s1 + 4 * a + 3 * b + 2 * c + d
            s1 += a + b + c + d
            q += 8
        while q < e:
            s1 += (np.int64(src[q]) << 8) | np.int64(src[q + 1])
            s2 += s1
            q += 2
        s1 = (s1 & 0xFFFF) + (s1 >> 16)
        s2 = (s2 & 0xFFFF) + (s2 >> 16)
    if n & 1:
        s1 += np.int64(src[q]) << 8
        s2 += s1
        s1 = (s1 & 0xFFFF) + (s1 >> 16)
        s2 = (s2 & 0xFFFF) + (s2 >> 16)
    s1 = (s1 & 0xFFFF) + (s1 >> 16)
    s2 = (s2 & 0xFFFF) + (s2 >> 16)
    return ((s2 << 16) | s1) & 0xFFFFFFFF


@njit(cache=True, nogil=True)
def verify(src, p, pend):
    """0 if the chunk src[p:pend] (checksum included) is intact, else
    ERR_DATA (ERR_INPUT if shorter than a checksum). Nothing is copied:
    an engine that can read in place calls this and then uses
    src[p:pend-4]."""
    n = pend - p - 4
    if n < 0:
        return ERR_INPUT
    q = pend - 4
    stored = (np.int64(src[q]) | (np.int64(src[q + 1]) << 8) | (np.int64(src[q + 2]) << 16)
              | (np.int64(src[q + 3]) << 24))
    f = fletcher32(src, p, n)
    swapped = ((f & 0x00FF00FF) << 8) | ((f >> 8) & 0x00FF00FF)
    if stored != f and stored != swapped:
        return ERR_DATA
    return 0


@njit(cache=True, nogil=True)
def _copy(src, p, n, dst, o):
    k = 0
    while k + 8 <= n:
        st64(dst, o + k, ld64(src, p + k))
        k += 8
    while k < n:
        dst[o + k] = src[p + k]
        k += 1


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """Check the chunk src[p:pend]'s checksum and copy the rest to dst[o:].
    The end of the output, or a negative error code."""
    n = pend - p - 4
    if n < 0:
        return ERR_INPUT
    if n > oend - o:
        return ERR_SPACE
    err = verify(src, p, pend)
    if err < 0:
        return err
    _copy(src, p, n, dst, o)
    return o + n


@njit(cache=True, nogil=True)
def strip(src, p, pend, dst, o, oend, cd, ws):
    """decode() without the check (HDF5 with H5Z_FLAG_SKIP_EDC, h5py's
    dataset access with error detection off)."""
    n = pend - p - 4
    if n < 0:
        return ERR_INPUT
    if n > oend - o:
        return ERR_SPACE
    _copy(src, p, n, dst, o)
    return o + n


def workspace_size(chunk_nbytes):
    """No scratch needed."""
    return 0
