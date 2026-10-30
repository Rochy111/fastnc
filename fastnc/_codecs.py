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
from . import _bzip2, _inflate, _nbit, _scaleoffset, _shuffle, _szip
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


# filter id -> (driver, module) ; codecs merged later register here too
DRIVERS = {
    1: (run_inflate, _inflate),
    2: (run_shuffle, _shuffle),
    4: (run_szip, _szip),
    5: (run_nbit, _nbit),
    6: (run_scaleoffset, _scaleoffset),
    307: (run_bzip2, _bzip2),
}
