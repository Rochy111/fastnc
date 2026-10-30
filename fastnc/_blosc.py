"""Blosc chunks in numba: the HDF5 blosc filter (32001, hdf5-blosc on
c-blosc 1.x, as hdf5plugin ships it), and the chunk decoder that blosc2
(32026, `_blosc2.py`) uses for the chunks inside its frames.

A chunk (c-blosc README_CHUNK_FORMAT) is a 16-byte header -- format
version, codec format version, flags, type size, decoded bytes, block
size, chunk bytes, little-endian -- (blosc2: 32 bytes, with a filter
pipeline and special-value flags), then unless the chunk is stored as is
("memcpyed") the start of every block, then the blocks. A block is one
compressed stream, or (blocks "split", the default for some codecs) one
stream per byte of the type; each stream is an int32 size and the codec's
output, or the bytes as they are when the size is the stream's decoded
size (blosc2 also: size 0 for zeros, negative for a run of one byte).
Codecs: blosclz, lz4 / lz4hc, snappy (blosc1), zlib, zstd. Filters undone
after the codec, per block: byte shuffle, bit shuffle, (blosc2) delta and
truncated precision (nothing to undo).

The rules follow the libraries exactly, including their differences: c-blosc
1.x splits a block only when the type has at most 16 bytes and the block
at least 128 elements, and bit-unshuffles a block only when its element
count is a multiple of 8 (else it is stored as is); c-blosc2 splits every
full block unless told not to, and bit-unshuffles the elements up to a
multiple of 8.
"""
import numpy as np

from ._bitshuffle import bitunshuffle, _bytes8
from ._inflate import ERR_DATA, ERR_INPUT, ERR_SPACE, IWS_SIZE, TAB_SIZE, zlib_inflate
from ._intrin import ld64, st64
from ._jit import njit
from ._lz4 import copy_bytes, copy_match, lz4_block
from ._zstd import WS_SIZE as ZWS, zstd_decompress

# scratch layout (bytes); the block buffers follow at W_BLK
W_Y = 0                                 # bit transpose scratch, 64 uint64
W_TAB = 512                             # inflate tables (uint32)
W_IWS = W_TAB + TAB_SIZE * 4            # inflate scratch (int64)
W_INT = W_IWS + IWS_SIZE * 8            # int64 scratch: filters, b2nd geometry (256)
W_Z = W_INT + 256 * 8                   # zstd's scratch
W_BLK = W_Z + ZWS
W_BLK += (-W_BLK) % 64

# int64 scratch (indices)
I_FLT = 0                               # filters (6), then their meta (6)
I_GEO = 16                              # b2nd geometry (see _blosc2)

NAN32 = 0x7FC00000
NAN64 = 0x7FF8000000000000


@njit(cache=True, nogil=True)
def _le32(src, p):
    v = (np.int64(src[p]) | (np.int64(src[p + 1]) << 8) | (np.int64(src[p + 2]) << 16)
         | (np.int64(src[p + 3]) << 24))
    return v - (1 << 32) if v >= (1 << 31) else v      # int32


@njit(cache=True, nogil=True)
def blosclz(src, p, pend, dst, o0, oend):
    """blosclz_decompress of src[p:pend] into dst[o0:oend]: the end of the
    output or an error (the library's rules: a trailing match is dropped)."""
    if pend == p:
        return o0
    ip = p
    op = o0
    ctrl = np.int64(src[ip]) & 31
    ip += 1
    while True:
        if ctrl >= 32:
            ln = (ctrl >> 5) - 1
            ofs = (ctrl & 31) << 8
            if ln == 6:
                while True:
                    if ip + 1 >= pend:
                        return ERR_INPUT
                    code = np.int64(src[ip])
                    ip += 1
                    ln += code
                    if code != 255:
                        break
            elif ip + 1 >= pend:
                return ERR_INPUT
            code = np.int64(src[ip])
            ip += 1
            ln += 3
            dist = ofs + code + 1
            if code == 255 and ofs == (31 << 8):
                if ip + 1 >= pend:
                    return ERR_INPUT
                dist = (np.int64(src[ip]) << 8) + np.int64(src[ip + 1]) + 8192
                ip += 2
            if op + ln > oend:
                return ERR_SPACE
            if dist > op - o0:
                return ERR_DATA
            if ip >= pend:
                break
            ctrl = np.int64(src[ip])
            ip += 1
            if dist >= 8 and ln <= 16 and op + 16 <= oend:
                s = op - dist
                st64(dst, op, ld64(dst, s))
                st64(dst, op + 8, ld64(dst, s + 8))
            else:
                copy_match(dst, op, dist, ln, oend)
            op += ln
        else:
            ctrl += 1
            if op + ctrl > oend:
                return ERR_SPACE
            if ip + ctrl > pend:
                return ERR_INPUT
            if ip + 32 <= pend and op + 32 <= oend:
                for k in range(0, 32, 8):        # (ctrl <= 32)
                    st64(dst, op + k, ld64(src, ip + k))
            else:
                copy_bytes(src, ip, dst, op, ctrl)
            op += ctrl
            ip += ctrl
            if ip >= pend:
                break
            ctrl = np.int64(src[ip])
            ip += 1
    return op


@njit(cache=True, nogil=True)
def snappy(src, p, pend, dst, o0, oend):
    """A raw snappy stream src[p:pend] into dst[o0:oend]: the end of the
    output or an error."""
    n = 0
    sh = 0
    while True:
        if p >= pend or sh > 28:
            return ERR_INPUT if p >= pend else ERR_DATA
        b = np.int64(src[p])
        p += 1
        n |= (b & 127) << sh
        sh += 7
        if b < 128:
            break
    if n > oend - o0:
        return ERR_SPACE
    end = o0 + n
    o = o0
    while p < pend:
        tag = np.int64(src[p])
        p += 1
        kind = tag & 3
        if kind == 0:
            ln = (tag >> 2) + 1
            if ln > 60:
                k = ln - 60
                if p + k > pend:
                    return ERR_INPUT
                ln = 0
                for j in range(k):
                    ln |= np.int64(src[p + j]) << (8 * j)
                ln += 1
                p += k
            if ln > pend - p:
                return ERR_INPUT
            if ln > end - o:
                return ERR_DATA
            copy_bytes(src, p, dst, o, ln)
            p += ln
            o += ln
            continue
        if kind == 1:
            if p >= pend:
                return ERR_INPUT
            ln = 4 + ((tag >> 2) & 7)
            dist = ((tag >> 5) << 8) | np.int64(src[p])
            p += 1
        elif kind == 2:
            if p + 2 > pend:
                return ERR_INPUT
            ln = (tag >> 2) + 1
            dist = np.int64(src[p]) | (np.int64(src[p + 1]) << 8)
            p += 2
        else:
            if p + 4 > pend:
                return ERR_INPUT
            ln = (tag >> 2) + 1
            dist = (np.int64(src[p]) | (np.int64(src[p + 1]) << 8)
                    | (np.int64(src[p + 2]) << 16) | (np.int64(src[p + 3]) << 24))
            p += 4
        if dist == 0 or dist > o - o0:
            return ERR_DATA
        if ln > end - o:
            return ERR_DATA
        copy_match(dst, o, dist, ln, oend)
        o += ln
    if o != end:
        return ERR_DATA
    return o


@njit(cache=True, nogil=True)
def unshuffle(src, s, dst, d, n, ts):
    """Undo blosc's byte shuffle of one block: n bytes of ts-byte elements
    src[s:s+n] -> dst[d:d+n] (the n % ts trailing bytes as they are)."""
    ne = n // ts
    i = 0
    if ts == 8 or (ts > 8 and ts % 8 == 0):
        while i + 8 <= ne:
            for j0 in range(0, ts, 8):
                r = s + j0 * ne + i
                b0, b1, b2, b3, b4, b5, b6, b7 = _bytes8(
                    ld64(src, r), ld64(src, r + ne), ld64(src, r + 2 * ne),
                    ld64(src, r + 3 * ne), ld64(src, r + 4 * ne), ld64(src, r + 5 * ne),
                    ld64(src, r + 6 * ne), ld64(src, r + 7 * ne))
                e = d + i * ts + j0
                st64(dst, e, b0)
                st64(dst, e + ts, b1)
                st64(dst, e + 2 * ts, b2)
                st64(dst, e + 3 * ts, b3)
                st64(dst, e + 4 * ts, b4)
                st64(dst, e + 5 * ts, b5)
                st64(dst, e + 6 * ts, b6)
                st64(dst, e + 7 * ts, b7)
            i += 8
    elif ts == 4:
        z = np.uint64(0)
        h = np.uint64(32)
        while i + 8 <= ne:
            r = s + i
            b0, b1, b2, b3, b4, b5, b6, b7 = _bytes8(
                ld64(src, r), ld64(src, r + ne), ld64(src, r + 2 * ne), ld64(src, r + 3 * ne),
                z, z, z, z)
            e = d + 4 * i
            st64(dst, e, b0 | (b1 << h))
            st64(dst, e + 8, b2 | (b3 << h))
            st64(dst, e + 16, b4 | (b5 << h))
            st64(dst, e + 24, b6 | (b7 << h))
            i += 8
    elif ts == 2:
        z = np.uint64(0)
        h1 = np.uint64(16)
        h2 = np.uint64(32)
        h3 = np.uint64(48)
        while i + 8 <= ne:
            r = s + i
            b0, b1, b2, b3, b4, b5, b6, b7 = _bytes8(
                ld64(src, r), ld64(src, r + ne), z, z, z, z, z, z)
            e = d + 2 * i
            st64(dst, e, b0 | (b1 << h1) | (b2 << h2) | (b3 << h3))
            st64(dst, e + 8, b4 | (b5 << h1) | (b6 << h2) | (b7 << h3))
            i += 8
    elif ts == 1:
        copy_bytes(src, s, dst, d, n)
        return
    for k in range(i, ne):
        for j in range(ts):
            dst[d + k * ts + j] = src[s + j * ne + k]
    r = n - ne * ts
    copy_bytes(src, s + ne * ts, dst, d + ne * ts, r)


@njit(cache=True, nogil=True)
def _codec(fmt, src, p, pend, dst, o, n, ws):
    """Decode one stream with codec `fmt` into exactly n bytes at dst[o:o+n]:
    0 or an error."""
    if fmt == 0:
        r = blosclz(src, p, pend, dst, o, o + n)
    elif fmt == 1:
        r = lz4_block(src, p, pend, dst, o, o, o + n)
    elif fmt == 2:
        r = snappy(src, p, pend, dst, o, o + n)
    elif fmt == 3:
        r = zlib_inflate(src, p, pend, dst, o, o + n, o + n,
                         ws[W_TAB:W_TAB + TAB_SIZE * 4].view(np.uint32),
                         ws[W_IWS:W_IWS + IWS_SIZE * 8].view(np.int64))
    elif fmt == 4:
        r = zstd_decompress(src, p, pend, dst, o, o + n, ws[W_Z:W_Z + ZWS])
    else:
        return ERR_DATA
    if r < 0:
        return r
    if r != o + n:
        return ERR_DATA
    return 0


@njit(cache=True, nogil=True)
def _delta(fa, fo, ra, ro, n, ts, first):
    """blosc2's delta_decoder, in place on fa[fo:fo+n]; ra[ro:] is the
    chunk's first block (already decoded)."""
    u = ts if (ts == 1 or ts == 2 or ts == 4 or ts == 8) else (8 if ts % 8 == 0 else 1)
    m = (n // u) * u
    if first:
        for b in range(u, m):
            fa[fo + b] ^= fa[fo + b - u]
    else:
        for b in range(m):
            fa[fo + b] ^= ra[ro + b]


@njit(cache=True, nogil=True)
def block(src, p, cend, j, bsize, leftover, hdr, v, fa, fo, ra, ro, ws, a, b):
    """Decode block j (bsize bytes) of the chunk at src[p:cend] into
    fa[fo:fo+bsize]. hdr: header bytes (16 / 32); v: format version; ra[ro:]
    the chunk's decoded first block (delta); ws[a:], ws[b:]: two block-sized
    buffers. 0 or an error."""
    flags = np.int64(src[p + 2])
    ts = np.int64(src[p + 3])
    fmt = flags >> 5
    flt = ws[W_INT:W_INT + 2048].view(np.int64)
    bs_at = p + hdr + 4 * j
    if bs_at + 4 > cend:
        return ERR_INPUT
    off = _le32(src, bs_at)
    if v <= 2:
        if off < 0 or off > cend - p - 4:
            return ERR_DATA
        split = ((flags & 0x10) == 0 and ts <= 16 and bsize // ts >= 128 and not leftover)
    else:
        if off <= 0 or off >= cend - p:
            return ERR_DATA
        split = (flags & 0x10) == 0 and not leftover
    ns = ts if split else 1
    neb = bsize // ns
    if neb == 0:
        return ERR_DATA
    # the filters to undo, last applied first
    nact = 0
    only = 0
    for i in range(6):
        f = flt[I_FLT + i]
        if f != 0 and f != 4:                   # (4: truncated precision, nothing to undo)
            nact += 1
            only = f
    direct = nact == 0 or (nact == 1 and only == 3)
    if direct:
        ca = fa
        co = fo
    else:
        ca = ws
        co = a
    q = p + off
    for k in range(ns):
        if q + 4 > cend:
            return ERR_INPUT
        cb = _le32(src, q)
        q += 4
        t = co + k * neb
        if v >= 3 and cb == 0:
            for i in range(neb):
                ca[t + i] = 0
        elif v >= 3 and cb < 0:
            if q >= cend:
                return ERR_INPUT
            tok = src[q]
            q += 1
            if (tok & 1) == 0 or cb < -255:
                return ERR_DATA
            val = np.uint8((-cb) & 255)
            for i in range(neb):
                ca[t + i] = val
        else:
            if cb < 0 or cb > cend - q:
                return ERR_INPUT
            if cb == neb:
                copy_bytes(src, q, ca, t, neb)
            else:
                e = _codec(fmt, src, q, q + cb, ca, t, neb, ws)
                if e < 0:
                    return e
            q += cb
    if direct:
        if nact == 1:
            _delta(fa, fo, ra, ro, bsize, ts, j == 0)
        return 0
    # the pipeline: current data in ws[a:] (0), ws[b:] (1) or the output (2)
    cur = 0
    y = ws[W_Y:W_Y + 512].view(np.uint64)
    done = 0
    for i in range(5, -1, -1):
        f = flt[I_FLT + i]
        if f == 0 or f == 4:
            continue
        done += 1
        if f == 3:
            if cur != 2:
                copy_bytes(ws, a if cur == 0 else b, fa, fo, bsize)
                cur = 2
            _delta(fa, fo, ra, ro, bsize, ts, j == 0)
            continue
        if f != 1 and f != 2:
            return ERR_DATA
        if done == nact:
            oa = fa
            oo = fo
            nxt = 2
        elif cur == 0:
            oa = ws
            oo = b
            nxt = 1
        else:
            oa = ws
            oo = a
            nxt = 0
        sa = fa if cur == 2 else ws
        so = fo if cur == 2 else (a if cur == 0 else b)
        if f == 1:
            m = flt[I_FLT + 6 + i]
            unshuffle(sa, so, oa, oo, bsize, m if m != 0 else ts)
        else:
            ne = bsize // ts
            if v <= 2:
                if ne % 8 == 0:
                    bitunshuffle(sa, so, oa, oo, ne, ts, oo + bsize, y)
                    copy_bytes(sa, so + ne * ts, oa, oo + ne * ts, bsize - ne * ts)
                else:
                    copy_bytes(sa, so, oa, oo, bsize)
            else:
                ne -= ne % 8
                bitunshuffle(sa, so, oa, oo, ne, ts, oo + bsize, y)
                copy_bytes(sa, so + ne * ts, oa, oo + ne * ts, bsize - ne * ts)
        cur = nxt
    if cur != 2:
        copy_bytes(ws, a if cur == 0 else b, fa, fo, bsize)
    return 0


@njit(cache=True, nogil=True)
def header(src, p, pend, ws):
    """Read the chunk header at src[p:pend]: (version, nbytes, blocksize,
    cbytes, header bytes, special) with the filters and their meta in the
    int64 scratch; version < 0 is an error code."""
    flt = ws[W_INT:W_INT + 2048].view(np.int64)
    if pend - p < 16:
        return ERR_INPUT, 0, 0, 0, 0, 0
    v = np.int64(src[p])
    flags = np.int64(src[p + 2])
    ts = np.int64(src[p + 3])
    nbytes = _le32(src, p + 4)
    bsize = _le32(src, p + 8)
    cbytes = _le32(src, p + 12)
    if nbytes < 0 or cbytes < 16 or ts == 0:
        return ERR_DATA, 0, 0, 0, 0, 0
    for i in range(12):
        flt[I_FLT + i] = 0
    special = 0
    hdr = 16
    if v <= 2:
        if v != 2 or (flags & 8):
            return ERR_DATA, 0, 0, 0, 0, 0
        if nbytes > 0 and bsize <= 0:
            return ERR_DATA, 0, 0, 0, 0, 0
        if flags & 1 and ts > 1:
            flt[I_FLT + 5] = 1
        elif flags & 4:
            flt[I_FLT + 5] = 2
    else:
        if bsize <= 0:
            return ERR_DATA, 0, 0, 0, 0, 0
        if (flags & 5) == 5:
            hdr = 32
            if pend - p < 32 or cbytes < 32:
                return ERR_INPUT, 0, 0, 0, 0, 0
            for i in range(6):
                flt[I_FLT + i] = np.int64(src[p + 16 + i])
                flt[I_FLT + 6 + i] = np.int64(src[p + 24 + i])
            f2 = np.int64(src[p + 30])
            bf = np.int64(src[p + 31])
            special = (bf >> 4) & 7
            if f2 & 1 or bf & 1 or bf & 8 or special > 4:
                return ERR_DATA, 0, 0, 0, 0, 0   # VL blocks, dictionaries, lazy chunks
            if fmt_of(flags) == 6 or fmt_of(flags) == 7:
                return ERR_DATA, 0, 0, 0, 0, 0   # user-defined codecs
        else:
            if flags & 1:
                flt[I_FLT + 5] = 1
            if flags & 4:
                flt[I_FLT + 5] = 2
            if flags & 8:
                flt[I_FLT + 4] = 3
        if nbytes > 0 and bsize > nbytes:
            bsize = nbytes
    if cbytes > pend - p:
        return ERR_INPUT, 0, 0, 0, 0, 0
    return v, nbytes, bsize, cbytes, hdr, special


@njit(cache=True, nogil=True)
def fmt_of(flags):
    return flags >> 5


@njit(cache=True, nogil=True)
def fill_special(src, p, special, ts, cbytes, hdr, fa, fo, n):
    """fa[fo:fo+n] = a blosc2 special chunk's values (zeros, NaN, a repeated
    value; not initialized: zeros). 0 or an error."""
    if special == 1 or special == 4:
        for i in range(n):
            fa[fo + i] = 0
        return 0
    if special == 2:
        if n % ts:
            return ERR_DATA
        if ts == 4:
            for i in range(0, n - 3, 4):
                fa[fo + i] = 0
                fa[fo + i + 1] = 0
                fa[fo + i + 2] = 0xC0
                fa[fo + i + 3] = 0x7F
        elif ts == 8:
            for i in range(0, n - 7, 8):
                for k in range(6):
                    fa[fo + i + k] = 0
                fa[fo + i + 6] = 0xF8
                fa[fo + i + 7] = 0x7F
        else:
            return ERR_DATA
        return 0
    if special == 3:
        vs = cbytes - hdr                       # the value's size
        if vs <= 0 or n % vs:
            return ERR_DATA
        for i in range(0, n, vs):
            for k in range(vs):
                fa[fo + i + k] = src[p + hdr + k]
        return 0
    return ERR_DATA


@njit(cache=True, nogil=True)
def chunk(src, p, pend, dst, o, oend, ws):
    """A whole blosc chunk (blosc1 or blosc2 format) src[p:pend] into dst
    from o on, blocks in order. Returns the end of the output or an error."""
    v, nbytes, bsize, cbytes, hdr, special = header(src, p, pend, ws)
    if v < 0:
        return v
    if nbytes > oend - o:
        return ERR_SPACE
    if nbytes == 0:
        return o
    cend = p + cbytes
    ts = np.int64(src[p + 3])
    if special:
        e = fill_special(src, p, special, ts, cbytes, hdr, dst, o, nbytes)
        return e if e < 0 else o + nbytes
    flags = np.int64(src[p + 2])
    if flags & 2:                               # stored as is
        if cbytes != nbytes + hdr:
            return ERR_DATA
        copy_bytes(src, p + hdr, dst, o, nbytes)
        return o + nbytes
    nblocks = (nbytes + bsize - 1) // bsize
    if hdr + 4 * nblocks > cbytes:
        return ERR_DATA
    a = W_BLK
    b = W_BLK + bsize + 64
    if b + bsize > ws.shape[0]:
        return ERR_SPACE
    for j in range(nblocks):
        n = min(bsize, nbytes - j * bsize)
        e = block(src, p, cend, j, n, n < bsize, hdr, v, dst, o + j * bsize, dst, o, ws, a, b)
        if e < 0:
            return e
    return o + nbytes


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """The HDF5 blosc filter: one blosc chunk src[p:pend] into dst[o:oend]."""
    if pend - p >= 16 and src[p] > 2:
        return ERR_DATA                         # blosc1 reads format version 2 only
    r = chunk(src, p, pend, dst, o, oend, ws)
    if r == o:
        return ERR_DATA                         # (the filter fails on an empty chunk)
    return r


def workspace_size(chunk_nbytes):
    """Codec scratch and two block buffers (a block is at most the chunk)."""
    return W_BLK + 2 * (chunk_nbytes + 64)
