"""LZF, h5py's lzf filter (32000), in numba.

The filter (h5py `lzf_filter.c`) stores a chunk as one raw LZF stream, no
header (cd values: filter revision, LZF version, chunk bytes). A chunk LZF
could not shrink is stored unfiltered with the filter's mask bit set; the
chunk engine skips the filter for those.

LZF: a control byte below 32 starts a run of ctrl + 1 literal bytes;
otherwise its top three bits are a match length - 2 (7: one more length
byte follows) and its low five bits the high bits of the distance - 1,
whose low byte follows. Decoded as liblzf's `lzf_decompress` with
CHECK_INPUT, copying eight bytes at a time where the output range allows.
"""
import numpy as np

from ._inflate import ERR_DATA, ERR_INPUT, ERR_SPACE
from ._intrin import ld64, st64
from ._jit import njit
from ._lz4 import copy_bytes, copy_match


@njit(cache=True, nogil=True)
def lzf(src, p, pend, dst, o0, oend):
    """Decode the LZF stream src[p:pend] into dst[o0:oend]. Returns the end
    of the output or a negative error code."""
    o = o0
    while p < pend:
        ctrl = np.int64(src[p])
        p += 1
        if ctrl < 32:
            n = ctrl + 1
            if o + n > oend:
                return ERR_SPACE
            if p + n > pend:
                return ERR_INPUT
            if p + 32 <= pend and o + 32 <= oend:
                for k in range(0, 32, 8):        # (n <= 32)
                    st64(dst, o + k, ld64(src, p + k))
            else:
                copy_bytes(src, p, dst, o, n)
            p += n
            o += n
        else:
            n = ctrl >> 5
            if p >= pend:
                return ERR_INPUT
            if n == 7:
                n += np.int64(src[p])
                p += 1
                if p >= pend:
                    return ERR_INPUT
            dist = ((ctrl & 31) << 8) + np.int64(src[p]) + 1
            p += 1
            n += 2
            if o + n > oend:
                return ERR_SPACE
            if dist > o - o0:
                return ERR_DATA
            copy_match(dst, o, dist, n, oend)
            o += n
    return o


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """The lzf filter: src[p:pend] into dst[o:oend]."""
    if pend <= p:
        return ERR_INPUT
    return lzf(src, p, pend, dst, o, oend)


def workspace_size(chunk_nbytes):
    """No scratch."""
    return 0
