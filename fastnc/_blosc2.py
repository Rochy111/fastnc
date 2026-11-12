"""The HDF5 blosc2 filter (32026, PyTables' hdf5-blosc2 `blosc2_filter.c`
as hdf5plugin ships it) in numba.

The filter stores each HDF5 chunk as a Blosc2 contiguous frame (c-blosc2
README_CFRAME_FORMAT): a msgpack header (lengths, sizes, the codec and
filters, metalayers), the blosc2 chunks, an index chunk holding each
chunk's offset (int64; negative: a chunk of zeros, NaNs or uninitialized
values, not stored), and a trailer.

For chunks of 2 to 8 dimensions (cd values 7 on: rank, chunk shape) the
frame is a B2ND array whose shape is the HDF5 chunk's ("b2nd" metalayer:
shape, chunk shape, block shape). Its blosc2 chunks are padded up to a
whole number of blocks in each dimension, and each block holds one tile
of `blockshape` items in C order; the filter returns the array in C order
(b2nd_get_slice_cbuffer). Otherwise (1-d) the frame holds plain chunks and
the filter returns its first chunk.

The chunks themselves are decoded by `_blosc` (blosc2 format: codecs
blosclz, lz4 / lz4hc, zlib, zstd; filters shuffle, bitshuffle, delta,
truncated precision; zero / run / stored streams; special chunks). Each
B2ND block is decoded into scratch and its tile copied into place.
"""
import numpy as np

from . import _blosc
from ._blosc import W_BLK, W_INT, fill_special, header, block
from ._inflate import ERR_DATA, ERR_INPUT, ERR_SPACE
from ._jit import njit
from ._lz4 import copy_bytes

MAXCHUNKS = 1024                        # chunks in one frame we take
W_OFF = W_BLK                           # the decoded chunk offsets
W_BUF = W_OFF + 8 * MAXCHUNKS           # block buffers from here

# int64 scratch (indices, after _blosc's filters)
I_NDIM = 16
I_SHAPE = 17
I_CSHAPE = 25
I_BSHAPE = 33
I_ECS = 41                              # padded chunk shape
I_CIDX = 49                             # chunk coordinates
I_BIDX = 57                             # block coordinates


@njit(cache=True, nogil=True)
def _be(src, p, n):
    v = 0
    for k in range(n):
        v = (v << 8) | np.int64(src[p + k])
    return v


@njit(cache=True, nogil=True)
def _le64(src, p):
    v = 0
    for k in range(8):
        v |= np.int64(src[p + k]) << (8 * k)
    return v


@njit(cache=True, nogil=True)
def _b2nd_meta(src, p, hend, g):
    """Find the "b2nd" (or "caterva") metalayer in the frame header at
    src[p:hend] and read ndim, shape, chunk shape, block shape into g.
    1 if found, 0 if not, or an error."""
    q = p + 87
    if q + 7 > hend or src[q] != 0x93 or src[q + 1] != 0xCD:
        return 0
    q += 4
    t = np.int64(src[q])
    if t == 0xDE:
        n = _be(src, q + 1, 2)
        q += 3
    elif (t & 0xF0) == 0x80:
        n = t & 15
        q += 1
    else:
        return ERR_DATA
    moff = -1
    for i in range(n):
        if q >= hend:
            return ERR_INPUT
        t = np.int64(src[q])
        if (t & 0xE0) != 0xA0:
            return ERR_DATA
        ln = t & 31
        q += 1
        if q + ln + 5 > hend:
            return ERR_INPUT
        isb = (ln == 4 and src[q] == 98 and src[q + 1] == 50 and src[q + 2] == 110
               and src[q + 3] == 100)                                       # b2nd
        isc = (ln == 7 and src[q] == 99 and src[q + 1] == 97 and src[q + 2] == 116
               and src[q + 3] == 101 and src[q + 4] == 114 and src[q + 5] == 118
               and src[q + 6] == 97)                                        # caterva
        q += ln
        if src[q] != 0xD2:
            return ERR_DATA
        off = _be(src, q + 1, 4)
        q += 5
        if (isb or isc) and moff < 0:
            moff = off
    if moff < 0:
        return 0
    q = p + moff
    if q + 5 > hend or src[q] != 0xC6:
        return ERR_DATA
    ln = _be(src, q + 1, 4)
    q += 5
    end = q + ln
    if end > hend or ln < 4:
        return ERR_INPUT
    if (src[q] & 0xF0) != 0x90:
        return ERR_DATA
    nd = np.int64(src[q + 2])
    if nd < 1 or nd > 8:
        return ERR_DATA
    q += 3
    g[I_NDIM] = nd
    for which in range(3):
        if q >= end or np.int64(src[q]) != 0x90 + nd:
            return ERR_DATA
        q += 1
        w = 8 if which == 0 else 4
        for i in range(nd):
            if q + 1 + w > end or np.int64(src[q]) != (0xD3 if which == 0 else 0xD2):
                return ERR_DATA
            v = _be(src, q + 1, w)
            if w == 4 and v >= (1 << 31):
                return ERR_DATA
            if v < 0:
                return ERR_DATA
            g[I_SHAPE + 8 * which + i] = v
            q += 1 + w
    return 1


@njit(cache=True, nogil=True)
def _special_offset(off, ts, fa, fo, n):
    """A chunk the index marks special (negative offset): zeros, NaNs, or
    uninitialized (zeros)."""
    sv = (off >> 56) & 0xFF
    if sv & 1 or sv & 4:
        return fill_special(fa, 0, 1, ts, 0, 0, fa, fo, n)
    if sv & 2:
        return fill_special(fa, 0, 2, ts, 0, 0, fa, fo, n)
    return ERR_DATA


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """The blosc2 filter: the frame src[p:pend] into dst[o:oend]."""
    g = ws[W_INT:W_INT + 2048].view(np.int64)
    if pend - p < 87:
        return ERR_INPUT
    if not (src[p + 2] == 98 and src[p + 3] == 50 and src[p + 4] == 102 and src[p + 5] == 114
            and src[p + 6] == 97 and src[p + 7] == 109 and src[p + 8] == 101):  # b2frame
        return ERR_DATA
    hlen = _be(src, p + 11, 4)
    flen = _be(src, p + 16, 8)
    gflags = np.int64(src[p + 25])
    if (src[p + 26] & 15) != 0 or ((gflags >> 4) & 3) != 1:
        return ERR_DATA                         # not contiguous, offsets not 64-bit
    # (gflags & 0x80: chunks of variable-length blocks; each chunk says so itself)
    nbytes = _be(src, p + 30, 8)
    cbytes = _be(src, p + 39, 8)
    ts = _be(src, p + 48, 4)
    csize = _be(src, p + 58, 4)
    if hlen < 87 or flen < hlen or flen > pend - p or ts <= 0 or nbytes < 0 or cbytes < 0:
        return ERR_DATA
    if cbytes > flen - hlen:
        return ERR_DATA
    fend = p + flen
    # the offsets index
    ip = p + hlen + cbytes
    nchunks = 0
    if nbytes > 0:
        if fend - ip < 32:
            return ERR_INPUT
        v, onb, obs, ocb, ohdr, osp = header(src, ip, fend, ws)
        if v < 0:
            return v
        if onb % 8 or onb > 8 * MAXCHUNKS:
            return ERR_DATA if onb % 8 else ERR_SPACE
        nchunks = onb // 8
        if csize > 0 and nchunks != (nbytes + csize - 1) // csize:
            return ERR_DATA
        r = _blosc.chunk(src, ip, fend, ws, W_OFF, W_OFF + onb, ws, W_BUF)
        if r < 0:
            return r
    nd = _b2nd_meta(src, p, p + hlen, g)
    if nd < 0:
        return nd
    ncd = cd[2]
    if nd == 0:
        # plain frame: the first chunk
        if nchunks == 0:
            return ERR_DATA
        off = _le64(ws, W_OFF)
        if off < 0:
            n = csize if (nchunks > 1 or nbytes % csize == 0) else nbytes % csize
            if n > oend - o:
                return ERR_SPACE
            e = _special_offset(off, ts, dst, o, n)
            return e if e < 0 else o + n
        cp = p + hlen + off
        if off > cbytes - 32:
            return ERR_DATA
        r = _blosc.chunk(src, cp, p + hlen + cbytes, dst, o, oend, ws, W_BUF)
        if r == o:
            return ERR_DATA
        return r
    # B2ND
    nd = g[I_NDIM]
    if ncd >= 8 + 3 and cd[3 + 7] != nd:
        return ERR_DATA
    size = ts
    for i in range(nd):
        if ncd >= 8 + 3 + nd and g[I_SHAPE + i] != cd[3 + 8 + i]:
            return ERR_DATA
        size *= g[I_SHAPE + i]
    if size > oend - o:
        return ERR_SPACE
    if size == 0:
        return o
    bnit = 1
    enit = 1
    ntot = 1
    for i in range(nd):
        c = g[I_CSHAPE + i]
        bl = g[I_BSHAPE + i]
        if c <= 0 or bl <= 0 or bl > c:
            return ERR_DATA
        e = (c + bl - 1) // bl * bl
        g[I_ECS + i] = e
        bnit *= bl
        enit *= e
        ntot *= (g[I_SHAPE + i] + c - 1) // c
    if ntot != nchunks:
        return ERR_DATA
    bsz = bnit * ts
    nblk = enit // bnit
    ca = W_BUF                                  # the block, decoded
    da = ca + bsz + 64                          # the chunk's first block (delta)
    aa = da + bsz + 64                          # the pipeline's two buffers
    bb = aa + bsz + 64
    if bb + bsz > ws.shape[0]:
        return ERR_SPACE
    dend = p + hlen + cbytes
    for c in range(nchunks):
        # chunk coordinates (C order)
        r = c
        for i in range(nd - 1, -1, -1):
            n = (g[I_SHAPE + i] + g[I_CSHAPE + i] - 1) // g[I_CSHAPE + i]
            g[I_CIDX + i] = r % n
            r //= n
        off = _le64(ws, W_OFF + 8 * c)
        if off >= 0:
            cp = p + hlen + off
            if off > cbytes - 16:
                return ERR_DATA
            v, cnb, cbs, ccb, chdr, csp = header(src, cp, dend, ws)
            if v < 0:
                return v
            if v <= 2 or cnb != enit * ts or (cbs != bsz and not csp):
                return ERR_DATA
            cfl = np.int64(src[cp + 2])
            if not csp and not (cfl & 2) and chdr + 4 * nblk > ccb:
                return ERR_DATA
        for j in range(nblk):
            # this block's data into ws[ca:]
            if off < 0:
                e = _special_offset(off, ts, ws, ca, bsz)
            elif csp:
                e = fill_special(src, cp, csp, ts, ccb, chdr, ws, ca, bsz)
            elif cfl & 2:
                if ccb != cnb + chdr:
                    return ERR_DATA
                copy_bytes(src, cp + chdr + j * bsz, ws, ca, bsz)
                e = 0
            else:
                e = block(src, cp, cp + ccb, j, bsz, False, chdr, v, ws, ca, ws, da, ws, aa, bb)
                if e == 0 and j == 0:
                    copy_bytes(ws, ca, ws, da, bsz)
            if e < 0:
                return e
            # block coordinates, then its tile copied row by row
            r = j
            for i in range(nd - 1, -1, -1):
                n = g[I_ECS + i] // g[I_BSHAPE + i]
                g[I_BIDX + i] = r % n
                r //= n
            _tile(ws, ca, dst, o, g, nd, ts)
    return o + size


@njit(cache=True, nogil=True)
def _tile(ws, ca, dst, o, g, nd, ts):
    """Copy the tile in ws[ca:] (block I_BIDX of chunk I_CIDX) into the C
    order array at dst[o:], clipped to the chunk and to the array."""
    # extents of the tile inside the chunk and the array
    nrows = 1
    for i in range(nd):
        bl = g[I_BSHAPE + i]
        lo = g[I_BIDX + i] * bl                         # in the chunk
        n = min(bl, g[I_CSHAPE + i] - lo)
        lo += g[I_CIDX + i] * g[I_CSHAPE + i]           # in the array
        n = min(n, g[I_SHAPE + i] - lo)
        if n <= 0:
            return
        g[I_BIDX + 8 + i] = n                           # (scratch: extent)
        g[I_BIDX + 16 + i] = lo                         # (scratch: origin)
        if i < nd - 1:
            nrows *= n
    run = g[I_BIDX + 8 + nd - 1] * ts
    for rr in range(nrows):
        # row rr of the tile: its coordinates in all but the last dimension
        r = rr
        so = 0                                          # in the tile
        do = 0                                          # in the array
        sstr = g[I_BSHAPE + nd - 1]
        dstr = g[I_SHAPE + nd - 1]
        for i in range(nd - 2, -1, -1):
            n = g[I_BIDX + 8 + i]
            k = r % n
            r //= n
            so += k * sstr
            do += (g[I_BIDX + 16 + i] + k) * dstr
            sstr *= g[I_BSHAPE + i]
            dstr *= g[I_SHAPE + i]
        do += g[I_BIDX + 16 + nd - 1]
        copy_bytes(ws, ca + so * ts, dst, o + do * ts, run)


def workspace_size(chunk_nbytes):
    """Codec scratch, the chunk index, and four block buffers (a block is
    at most the chunk)."""
    return W_BUF + 4 * (chunk_nbytes + 64)
