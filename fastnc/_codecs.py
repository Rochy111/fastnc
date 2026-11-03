"""Batch drivers for the filters outside the deflate fast path: one small
kernel per codec, each taking chunks off an atomic counter and decoding
src[ioff:ioff+ilen] into dst[ooff:ooff+olen]. One per codec, not one with
every codec inside: numba compiles a kernel only when it is first called
(szip's takes seconds), and caches each on its own -- a kernel taking the
codec as a first-class function argument would be compiled again in every
process (numba cannot cache it).

skip[i] set: the filter was not applied to that chunk (HDF5's filter mask:
an optional filter that could not shrink it), so its bytes are copied.
outlen[i] gets the decoded length, status[i] a negative error code.
"""
from . import (_bitshuffle, _blosc, _blosc2, _bzip2, _inflate, _lz4, _lzf, _nbit,
               _scaleoffset, _shuffle, _szip, _zfp, _zstd)
from ._intrin import ld64, st64, take
from ._jit import njit


@njit(cache=True, nogil=True)
def _copy(src, p, n, dst, o, room):
    if n > room:
        return -2
    k = 0
    while k + 8 <= n:
        st64(dst, o + k, ld64(src, p + k))
        k += 8
    while k < n:
        dst[o + k] = src[p + k]
        k += 1
    return n


@njit(cache=True, nogil=True)
def run_inflate(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen, tab, iws):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _inflate.zlib_inflate(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i],
                                      ooff[i] + olen[i], tab, iws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_szip(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _szip.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_nbit(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _nbit.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_scaleoffset(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _scaleoffset.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_bzip2(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _bzip2.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_shuffle(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _shuffle.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_lzf(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _lzf.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_lz4(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _lz4.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_bitshuffle(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _bitshuffle.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_zstd(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _zstd.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_blosc(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _blosc.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_blosc2(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _blosc2.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


@njit(cache=True, nogil=True)
def run_zfp(src, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen):
    n = ioff.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        if skip[i]:
            r = _copy(src, ioff[i], ilen[i], dst, ooff[i], olen[i])
        else:
            r = _zfp.decode(src, ioff[i], ioff[i] + ilen[i], dst, ooff[i], ooff[i] + olen[i], cd, ws)
            if r >= 0:
                r -= ooff[i]
        if r < 0:
            status[i] = r
        else:
            outlen[i] = r


# filter id -> (driver, module)
DRIVERS = {
    1: (run_inflate, _inflate),
    2: (run_shuffle, _shuffle),
    4: (run_szip, _szip),
    5: (run_nbit, _nbit),
    6: (run_scaleoffset, _scaleoffset),
    307: (run_bzip2, _bzip2),
    32000: (run_lzf, _lzf),
    32001: (run_blosc, _blosc),
    32004: (run_lz4, _lz4),
    32008: (run_bitshuffle, _bitshuffle),
    32013: (run_zfp, _zfp),
    32015: (run_zstd, _zstd),
    32026: (run_blosc2, _blosc2),
}
