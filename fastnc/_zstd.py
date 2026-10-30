"""Zstandard (RFC 8878) frames in numba, and the HDF5 zstd filter (32015).

The filter (HDFGroup hdf5_plugins `H5Zzstd.c`, in hdf5plugin) stores a
chunk as the output of ZSTD_compress -- one zstd frame, no header of its
own (cd value: the compression level). Bitshuffle and blosc use the same
frames for their zstd blocks. ERA5, CMIP and Zarr archives are zstd.

Everything the format allows, except dictionaries: skippable frames,
several frames in a row, raw / RLE / compressed blocks, raw / RLE /
Huffman (one or four streams, tree described or repeated) literals, FSE
sequences in predefined, RLE, described and repeated modes, repeat
offsets, any window size (the whole frame is decoded into the output, so
the window is everything before). The content checksum is skipped.

Design, as zstd's own decoder:

- backward bit streams read through a 64-bit buffer refilled a word at a
  time (`buf` holds `bc` unread bits in its low bits; a refill shifts in
  as many whole bytes as fit below 64, leaving at least 56). Past the
  stream's start zero bits are shifted in and counted (`dfc`), so the
  decoding loops never branch on the end; whether a stream was read
  exactly is checked after;
- Huffman literals: one lookup per symbol in a table indexed by the next
  `tableLog` bits, the four streams decoded in one interleaved loop;
- sequences decoded and executed in the same pass, an FSE state table
  entry (one uint64) giving the base value, extra-bit count, state bits
  and next-state base at once; literals and matches copied by words.

Scratch (`ws`, `WS_SIZE` bytes): the literal buffer (one block's, 128 KiB)
and the tables, which persist from block to block within a frame as the
format's repeat modes need.
"""
import numpy as np

from ._inflate import ERR_DATA, ERR_INPUT, ERR_SPACE
from ._intrin import ld64, st64
from ._jit import njit
from ._lz4 import copy_bytes, copy_match

BLOCK_MAX = 1 << 17

# scratch layout (bytes)
W_LIT = 0                               # literals: one block's, + slack for word copies
LIT_END = BLOCK_MAX + 64
W_HUF = LIT_END                         # Huffman table, 4096 uint16
W_LL = W_HUF + 8192                     # FSE tables, uint64 entries
W_ML = W_LL + 512 * 8
W_OF = W_ML + 512 * 8
W_FW = W_OF + 256 * 8                   # FSE table of the Huffman weights (64)
W_INT = W_FW + 64 * 8                   # int64 scratch, 1024
WS_SIZE = W_INT + 1024 * 8

# int64 scratch layout (indices)
I_NORM = 0                              # normalized counts, 256
I_NEXT = 256                            # next state per symbol, 256
I_W = 512                               # Huffman weights, 256
I_RANK = 768                            # Huffman rank starts, 16
I_REP = 800                             # the three repeat offsets
I_HLOG = 803                            # Huffman tableLog (0: none yet in this frame)
I_LOG = 804                             # LL / OF / ML tableLog (-1: none yet)

_LL_BASE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22,
                     24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                     32768, 65536], np.int64)
_LL_BITS = np.array([0] * 16 + [1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                15, 16], np.int64)
_ML_BASE = np.array(list(range(3, 35)) + [35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131,
                                          259, 515, 1027, 2051, 4099, 8195, 16387, 32771,
                                          65539], np.int64)
_ML_BITS = np.array([0] * 32 + [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13,
                                14, 15, 16], np.int64)
_LL_NORM = np.array([4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1], np.int64)
_ML_NORM = np.array([1, 4, 3, 2, 2, 2, 2, 2, 2] + [1] * 37 + [-1] * 7, np.int64)
_OF_NORM = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2] + [1] * 15 + [-1] * 5, np.int64)
_MAXSYM = np.array([35, 31, 52], np.int64)      # LL, OF, ML
_MAXLOG = np.array([9, 8, 9], np.int64)
_DEFLOG = np.array([6, 5, 6], np.int64)

U8 = np.uint64(8)


@njit(cache=True, nogil=True)
def _highbit(x):
    """Index of the highest set bit of x > 0."""
    n = 0
    while x > 1:
        x >>= 1
        n += 1
    return n


@njit(cache=True, nogil=True)
def _le32(src, p):
    return (np.int64(src[p]) | (np.int64(src[p + 1]) << 8) | (np.int64(src[p + 2]) << 16)
            | (np.int64(src[p + 3]) << 24))


@njit(cache=True, nogil=True)
def _bits32(src, p, pend, pos):
    """32 bits of src[p:pend] from bit pos on (little-endian, LSB first),
    zeros past the end."""
    b = p + (pos >> 3)
    v = 0
    for k in range(5):
        if b + k < pend:
            v |= np.int64(src[b + k]) << (8 * k)
    return (v >> (pos & 7)) & 0xFFFFFFFF


@njit(cache=True, nogil=True)
def _read_ncount(src, p, pend, norm, maxsym, maxlog):
    """An FSE table description (zstd's FSE_readNCount): fills norm[0..]
    and returns (bytes used or error, number of symbols, tableLog)."""
    for s in range(maxsym + 1):
        norm[s] = 0
    if p >= pend:
        return ERR_INPUT, 0, 0
    pos = 0
    log = (_bits32(src, p, pend, 0) & 15) + 5
    pos = 4
    if log > maxlog:
        return ERR_DATA, 0, 0
    remaining = (1 << log) + 1
    threshold = 1 << log
    nbits = log + 1
    s = 0
    prev0 = False
    while True:
        if prev0:
            while True:
                r = _bits32(src, p, pend, pos) & 3
                pos += 2
                s += r
                if r != 3:
                    break
            if s > maxsym:
                break
        bs = _bits32(src, p, pend, pos)
        mx = 2 * threshold - 1 - remaining
        if (bs & (threshold - 1)) < mx:
            count = bs & (threshold - 1)
            pos += nbits - 1
        else:
            count = bs & (2 * threshold - 1)
            if count >= threshold:
                count -= mx
            pos += nbits
        count -= 1
        if count >= 0:
            remaining -= count
        else:
            remaining += count
        norm[s] = count
        s += 1
        prev0 = count == 0
        if remaining < threshold:
            if remaining <= 1:
                break
            nbits = _highbit(remaining) + 1
            threshold = 1 << (nbits - 1)
        if s > maxsym:
            break
    if remaining != 1 or s > maxsym + 1:
        return ERR_DATA, 0, 0
    n = (pos + 7) >> 3
    if n > pend - p:
        return ERR_INPUT, 0, 0
    return n, s, log


@njit(cache=True, nogil=True)
def _build_fse(norm, nsym, log, tab, nxt, kind):
    """The decoding table of an FSE distribution into tab[0:1<<log]. Entry:
    base value (bits 0-31), extra bits (32-39), state bits (40-47), next
    state base (48-63); kind 0 literal lengths, 1 offsets, 2 match lengths,
    3 plain symbols (base value = symbol). 0 or ERR_DATA."""
    size = 1 << log
    high = size - 1
    for s in range(nsym):
        c = norm[s]
        if c == -1:
            tab[high] = s
            high -= 1
            nxt[s] = 1
        else:
            nxt[s] = c
    pos = 0
    step = (size >> 1) + (size >> 3) + 3
    mask = size - 1
    for s in range(nsym):
        for i in range(norm[s]):
            tab[pos] = s
            pos = (pos + step) & mask
            while pos > high:
                pos = (pos + step) & mask
    if pos != 0:
        return ERR_DATA
    for u in range(size):
        s = np.int64(tab[u])
        ns = nxt[s]
        nxt[s] = ns + 1
        nb = log - _highbit(ns)
        st = (ns << nb) - size
        if kind == 0:
            base = _LL_BASE[s]
            add = _LL_BITS[s]
        elif kind == 2:
            base = _ML_BASE[s]
            add = _ML_BITS[s]
        elif kind == 1:
            base = 1 << s
            add = s
        else:
            base = s
            add = 0
        tab[u] = np.uint64(base | (add << 32) | (nb << 40) | (st << 48))
    return 0


@njit(cache=True, nogil=True)
def _rinit(src, start, end):
    """A backward bit stream src[start:end]: (q, buf, bc), the marker bit
    dropped; bc < 0 for a stream that has none (last byte zero)."""
    last = np.int64(src[end - 1])
    if last == 0:
        return end, np.uint64(0), -1
    h = _highbit(last)
    return end - 1, np.uint64(last & ((1 << h) - 1)), h


@njit(cache=True, nogil=True)
def _refill(src, start, q, buf, bc, dfc):
    """At least 56 unread bits in buf (zeros past the start, counted in dfc)."""
    if bc >= 56:
        return q, buf, bc, dfc
    if q - 8 >= start:
        k = (63 - bc) >> 3
        buf = (buf << np.uint64(k << 3)) | (ld64(src, q - 8) >> np.uint64(64 - (k << 3)))
        return q - k, buf, bc + (k << 3), dfc
    while bc <= 55:
        buf <<= U8
        if q > start:
            q -= 1
            buf |= np.uint64(src[q])
        else:
            dfc += 8
        bc += 8
    return q, buf, bc, dfc


@njit(cache=True, nogil=True)
def _read_huf(src, p, pend, ws, iws):
    """A Huffman tree description at src[p:pend] into the literal table.
    Returns the bytes it took, or an error."""
    w = iws[I_W:I_W + 256]
    norm = iws[I_NORM:I_NORM + 256]
    hb = np.int64(src[p])
    if hb < 128:
        # weights FSE-compressed (tableLog <= 6), two interleaved states
        if hb == 0:
            return ERR_DATA
        end = p + 1 + hb
        if end > pend:
            return ERR_INPUT
        n, nsym, log = _read_ncount(src, p + 1, end, norm, 255, 6)
        if n < 0:
            return n
        fw = ws[W_FW:W_FW + 512].view(np.uint64)
        if _build_fse(norm, nsym, log, fw, iws[I_NEXT:I_NEXT + 256], 3) < 0:
            return ERR_DATA
        start = p + 1 + n
        if start >= end:
            return ERR_INPUT
        q, buf, bc = _rinit(src, start, end)
        if bc < 0:
            return ERR_DATA
        dfc = 0
        q, buf, bc, dfc = _refill(src, start, q, buf, bc, dfc)
        s1 = np.int64((buf >> np.uint64(bc - log)) & np.uint64((1 << log) - 1))
        bc -= log
        s2 = np.int64((buf >> np.uint64(bc - log)) & np.uint64((1 << log) - 1))
        bc -= log
        nw = 0
        while True:
            if nw > 253:
                return ERR_DATA
            q, buf, bc, dfc = _refill(src, start, q, buf, bc, dfc)
            e = fw[s1]
            w[nw] = np.int64(e & np.uint64(0xFFFFFFFF))
            nw += 1
            nb = np.int64((e >> np.uint64(40)) & np.uint64(255))
            s1 = np.int64(e >> np.uint64(48)) + np.int64((buf >> np.uint64(bc - nb))
                                                         & np.uint64((1 << nb) - 1))
            bc -= nb
            if bc < dfc:
                w[nw] = np.int64(fw[s2] & np.uint64(0xFFFFFFFF))
                nw += 1
                break
            if nw > 253:
                return ERR_DATA
            e = fw[s2]
            w[nw] = np.int64(e & np.uint64(0xFFFFFFFF))
            nw += 1
            nb = np.int64((e >> np.uint64(40)) & np.uint64(255))
            s2 = np.int64(e >> np.uint64(48)) + np.int64((buf >> np.uint64(bc - nb))
                                                         & np.uint64((1 << nb) - 1))
            bc -= nb
            if bc < dfc:
                w[nw] = np.int64(fw[s1] & np.uint64(0xFFFFFFFF))
                nw += 1
                break
        used = 1 + hb
    else:
        nw = hb - 127
        nb = (nw + 1) >> 1
        if p + 1 + nb > pend:
            return ERR_INPUT
        for i in range(nw):
            b = np.int64(src[p + 1 + (i >> 1)])
            w[i] = (b >> 4) if (i & 1) == 0 else (b & 15)
        used = 1 + nb
    total = 0
    for i in range(nw):
        if w[i] > 12:
            return ERR_DATA
        if w[i]:
            total += 1 << (w[i] - 1)
    if total == 0:
        return ERR_DATA
    log = _highbit(total) + 1
    if log > 12:
        return ERR_DATA
    rest = (1 << log) - total
    hr = _highbit(rest)
    if (1 << hr) != rest:
        return ERR_DATA
    w[nw] = hr + 1
    nsym = nw + 1
    rank = iws[I_RANK:I_RANK + 16]
    for k in range(16):
        rank[k] = 0
    for i in range(nsym):
        rank[w[i]] += 1
    start = 0
    for k in range(1, log + 1):
        c = rank[k]
        rank[k] = start
        start += c << (k - 1)
    huf = ws[W_HUF:W_HUF + 8192].view(np.uint16)
    for i in range(nsym):
        k = w[i]
        if k:
            n = 1 << (k - 1)
            e = i | ((log + 1 - k) << 8)
            r = rank[k]
            for j in range(n):
                huf[r + j] = e
            rank[k] = r + n
    iws[I_HLOG] = log
    return used


@njit(cache=True, nogil=True)
def _huf1(src, start, end, ws, o, n, log):
    """One Huffman stream src[start:end] -> n literals at ws[o:]. 0 or error."""
    huf = ws[W_HUF:W_HUF + 8192].view(np.uint16)
    if end <= start:
        return ERR_INPUT
    q, buf, bc = _rinit(src, start, end)
    if bc < 0:
        return ERR_DATA
    dfc = 0
    M = np.uint64((1 << log) - 1)
    k = 0
    while k + 4 <= n:
        q, buf, bc, dfc = _refill(src, start, q, buf, bc, dfc)
        for j in range(4):
            e = np.int64(huf[np.int64((buf >> np.uint64(bc - log)) & M)])
            bc -= e >> 8
            ws[o + k + j] = e & 255
        k += 4
    q, buf, bc, dfc = _refill(src, start, q, buf, bc, dfc)
    while k < n:
        e = np.int64(huf[np.int64((buf >> np.uint64(bc - log)) & M)])
        bc -= e >> 8
        ws[o + k] = e & 255
        k += 1
    if bc != dfc or q != start:
        return ERR_DATA
    return 0


@njit(cache=True, nogil=True)
def _huf4(src, p, end, ws, o, n, log):
    """Four Huffman streams (6-byte jump table first) -> n literals at ws[o:]."""
    if end - p < 10:
        return ERR_DATA
    l0 = np.int64(src[p]) | (np.int64(src[p + 1]) << 8)
    l1 = np.int64(src[p + 2]) | (np.int64(src[p + 3]) << 8)
    l2 = np.int64(src[p + 4]) | (np.int64(src[p + 5]) << 8)
    a0 = p + 6
    a1 = a0 + l0
    a2 = a1 + l1
    a3 = a2 + l2
    if l0 == 0 or l1 == 0 or l2 == 0 or a3 >= end:
        return ERR_DATA
    seg = (n + 3) >> 2
    n3 = n - 3 * seg
    if n3 < 0:
        return ERR_DATA
    huf = ws[W_HUF:W_HUF + 8192].view(np.uint16)
    M = np.uint64((1 << log) - 1)
    q0, b0, c0 = _rinit(src, a0, a1)
    q1, b1, c1 = _rinit(src, a1, a2)
    q2, b2, c2 = _rinit(src, a2, a3)
    q3, b3, c3 = _rinit(src, a3, end)
    if c0 < 0 or c1 < 0 or c2 < 0 or c3 < 0:
        return ERR_DATA
    d0 = d1 = d2 = d3 = 0
    o0 = o
    o1 = o + seg
    o2 = o + 2 * seg
    o3 = o + 3 * seg
    k = 0
    while k + 4 <= n3:
        q0, b0, c0, d0 = _refill(src, a0, q0, b0, c0, d0)
        q1, b1, c1, d1 = _refill(src, a1, q1, b1, c1, d1)
        q2, b2, c2, d2 = _refill(src, a2, q2, b2, c2, d2)
        q3, b3, c3, d3 = _refill(src, a3, q3, b3, c3, d3)
        for j in range(4):
            e0 = np.int64(huf[np.int64((b0 >> np.uint64(c0 - log)) & M)])
            e1 = np.int64(huf[np.int64((b1 >> np.uint64(c1 - log)) & M)])
            e2 = np.int64(huf[np.int64((b2 >> np.uint64(c2 - log)) & M)])
            e3 = np.int64(huf[np.int64((b3 >> np.uint64(c3 - log)) & M)])
            c0 -= e0 >> 8
            c1 -= e1 >> 8
            c2 -= e2 >> 8
            c3 -= e3 >> 8
            ws[o0 + k + j] = e0 & 255
            ws[o1 + k + j] = e1 & 255
            ws[o2 + k + j] = e2 & 255
            ws[o3 + k + j] = e3 & 255
        k += 4
    # the tails, one stream at a time
    for i in range(4):
        if i == 0:
            q, buf, bc, dfc, st, oo, nn = q0, b0, c0, d0, a0, o0, seg
        elif i == 1:
            q, buf, bc, dfc, st, oo, nn = q1, b1, c1, d1, a1, o1, seg
        elif i == 2:
            q, buf, bc, dfc, st, oo, nn = q2, b2, c2, d2, a2, o2, seg
        else:
            q, buf, bc, dfc, st, oo, nn = q3, b3, c3, d3, a3, o3, n3
        kk = k
        while kk < nn:
            q, buf, bc, dfc = _refill(src, st, q, buf, bc, dfc)
            m = min(nn - kk, 4)
            for j in range(m):
                e = np.int64(huf[np.int64((buf >> np.uint64(bc - log)) & M)])
                bc -= e >> 8
                ws[oo + kk + j] = e & 255
            kk += m
        if bc != dfc or q != st:
            return ERR_DATA
    return 0


@njit(cache=True, nogil=True)
def _seq_table(src, q, send, mode, which, ws, iws):
    """The table for one sequence symbol type (which: 0 LL, 1 OF, 2 ML) in
    `mode`; returns the new read position or an error."""
    if which == 0:
        tab = ws[W_LL:W_LL + 4096].view(np.uint64)
    elif which == 1:
        tab = ws[W_OF:W_OF + 2048].view(np.uint64)
    else:
        tab = ws[W_ML:W_ML + 4096].view(np.uint64)
    kind = 0 if which == 0 else (1 if which == 1 else 2)
    norm = iws[I_NORM:I_NORM + 256]
    nxt = iws[I_NEXT:I_NEXT + 256]
    if mode == 0:
        if which == 0:
            n = 36
            for s in range(n):
                norm[s] = _LL_NORM[s]
        elif which == 1:
            n = 29
            for s in range(n):
                norm[s] = _OF_NORM[s]
        else:
            n = 53
            for s in range(n):
                norm[s] = _ML_NORM[s]
        _build_fse(norm, n, _DEFLOG[which], tab, nxt, kind)
        iws[I_LOG + which] = _DEFLOG[which]
        return q
    if mode == 1:
        if q >= send:
            return ERR_INPUT
        s = np.int64(src[q])
        if s > _MAXSYM[which]:
            return ERR_DATA
        norm[s] = 1
        for k in range(s):
            norm[k] = 0
        _build_fse(norm, s + 1, 0, tab, nxt, kind)
        iws[I_LOG + which] = 0
        return q + 1
    if mode == 2:
        n, nsym, log = _read_ncount(src, q, send, norm, _MAXSYM[which], _MAXLOG[which])
        if n < 0:
            return n
        if _build_fse(norm, nsym, log, tab, nxt, kind) < 0:
            return ERR_DATA
        iws[I_LOG + which] = log
        return q + n
    if iws[I_LOG + which] < 0:
        return ERR_DATA                         # repeat with no table before
    return q


@njit(cache=True, nogil=True)
def _block(src, p, bend, dst, fstart, o, oend, ws, iws, bsmax):
    """One compressed block src[p:bend] -> dst from o on. Returns the end of
    the output or an error."""
    u = np.uint64
    if bend - p < 2:
        return ERR_INPUT
    # -- literals section
    b0 = np.int64(src[p])
    ltype = b0 & 3
    sf = (b0 >> 2) & 3
    if ltype <= 1:
        if sf == 0 or sf == 2:
            lsize = b0 >> 3
            lh = 1
        elif sf == 1:
            lsize = (b0 >> 4) | (np.int64(src[p + 1]) << 4)
            lh = 2
        else:
            if bend - p < 3:
                return ERR_INPUT
            lsize = (b0 >> 4) | (np.int64(src[p + 1]) << 4) | (np.int64(src[p + 2]) << 12)
            lh = 3
        if lsize > bsmax:
            return ERR_DATA
        if ltype == 0:
            if p + lh + lsize > bend:
                return ERR_INPUT
            copy_bytes(src, p + lh, ws, W_LIT, lsize)    # (one type for both sources)
            q = p + lh + lsize
        else:
            if p + lh >= bend:
                return ERR_INPUT
            v = src[p + lh]
            for k in range(lsize):
                ws[W_LIT + k] = v
            q = p + lh + 1
    else:
        if bend - p < 5:
            return ERR_INPUT
        h = np.int64(src[p]) | (np.int64(src[p + 1]) << 8) | (np.int64(src[p + 2]) << 16)
        four = sf != 0
        if sf <= 1:
            lh = 3
            lsize = (h >> 4) & 0x3FF
            csize = (h >> 14) & 0x3FF
        elif sf == 2:
            lh = 4
            h |= np.int64(src[p + 3]) << 24
            lsize = (h >> 4) & 0x3FFF
            csize = h >> 18
        else:
            lh = 5
            h |= np.int64(src[p + 3]) << 24
            lsize = (h >> 4) & 0x3FFFF
            csize = (h >> 22) | (np.int64(src[p + 4]) << 10)
        if lsize > bsmax:
            return ERR_DATA
        if four and lsize < 6:
            return ERR_DATA
        cend = p + lh + csize
        if cend > bend:
            return ERR_INPUT
        r = p + lh
        if ltype == 2:
            n = _read_huf(src, r, cend, ws, iws)
            if n < 0:
                return n
            r += n
        elif iws[I_HLOG] == 0:
            return ERR_DATA                     # treeless with no tree before
        log = iws[I_HLOG]
        if four:
            e = _huf4(src, r, cend, ws, W_LIT, lsize, log)
        else:
            e = _huf1(src, r, cend, ws, W_LIT, lsize, log)
        if e < 0:
            return e
        q = cend
    lit = ws
    lp = W_LIT
    lend = W_LIT + lsize
    llim = LIT_END
    # -- sequences section
    if q >= bend:
        return ERR_INPUT
    nb = np.int64(src[q])
    q += 1
    o0 = o
    if nb == 0:
        if q != bend:
            return ERR_DATA
        nseq = 0
    else:
        if nb < 128:
            nseq = nb
        elif nb < 255:
            if q >= bend:
                return ERR_INPUT
            nseq = ((nb - 128) << 8) + np.int64(src[q])
            q += 1
        else:
            if q + 2 > bend:
                return ERR_INPUT
            nseq = np.int64(src[q]) + (np.int64(src[q + 1]) << 8) + 0x7F00
            q += 2
        if q >= bend:
            return ERR_INPUT
        modes = np.int64(src[q])
        q += 1
        if modes & 3:
            return ERR_DATA
        q = _seq_table(src, q, bend, modes >> 6, 0, ws, iws)
        if q < 0:
            return q
        q = _seq_table(src, q, bend, (modes >> 4) & 3, 1, ws, iws)
        if q < 0:
            return q
        q = _seq_table(src, q, bend, (modes >> 2) & 3, 2, ws, iws)
        if q < 0:
            return q
        if q >= bend:
            return ERR_INPUT
        llt = ws[W_LL:W_LL + 4096].view(np.uint64)
        oft = ws[W_OF:W_OF + 2048].view(np.uint64)
        mlt = ws[W_ML:W_ML + 4096].view(np.uint64)
        lll = iws[I_LOG]
        ofl = iws[I_LOG + 1]
        mll = iws[I_LOG + 2]
        start = q
        q, buf, bc = _rinit(src, start, bend)
        if bc < 0:
            return ERR_DATA
        dfc = 0
        q, buf, bc, dfc = _refill(src, start, q, buf, bc, dfc)
        sl = np.int64((buf >> u(bc - lll)) & u((1 << lll) - 1))
        bc -= lll
        so = np.int64((buf >> u(bc - ofl)) & u((1 << ofl) - 1))
        bc -= ofl
        sm = np.int64((buf >> u(bc - mll)) & u((1 << mll) - 1))
        bc -= mll
        r0 = iws[I_REP]
        r1 = iws[I_REP + 1]
        r2 = iws[I_REP + 2]
        M32 = u(0xFFFFFFFF)
        M8 = u(255)
        for i in range(nseq):
            el = llt[sl]
            eo = oft[so]
            em = mlt[sm]
            q, buf, bc, dfc = _refill(src, start, q, buf, bc, dfc)
            nb = np.int64((eo >> u(32)) & M8)
            ofv = np.int64(eo & M32) + np.int64((buf >> u(bc - nb)) & ((u(1) << u(nb)) - u(1)))
            bc -= nb
            q, buf, bc, dfc = _refill(src, start, q, buf, bc, dfc)
            nb = np.int64((em >> u(32)) & M8)
            ml = np.int64(em & M32) + np.int64((buf >> u(bc - nb)) & ((u(1) << u(nb)) - u(1)))
            bc -= nb
            nb = np.int64((el >> u(32)) & M8)
            ll = np.int64(el & M32) + np.int64((buf >> u(bc - nb)) & ((u(1) << u(nb)) - u(1)))
            bc -= nb
            if i + 1 < nseq:
                q, buf, bc, dfc = _refill(src, start, q, buf, bc, dfc)
                nb = np.int64((el >> u(40)) & M8)
                sl = np.int64(el >> u(48)) + np.int64((buf >> u(bc - nb)) & ((u(1) << u(nb)) - u(1)))
                bc -= nb
                nb = np.int64((em >> u(40)) & M8)
                sm = np.int64(em >> u(48)) + np.int64((buf >> u(bc - nb)) & ((u(1) << u(nb)) - u(1)))
                bc -= nb
                nb = np.int64((eo >> u(40)) & M8)
                so = np.int64(eo >> u(48)) + np.int64((buf >> u(bc - nb)) & ((u(1) << u(nb)) - u(1)))
                bc -= nb
            # repeat offsets
            if ofv > 3:
                off = ofv - 3
                r2 = r1
                r1 = r0
                r0 = off
            else:
                idx = ofv - 1 + (1 if ll == 0 else 0)
                if idx == 0:
                    off = r0
                else:
                    if idx == 3:
                        off = r0 - 1
                    elif idx == 1:
                        off = r1
                    else:
                        off = r2
                    if idx > 1:
                        r2 = r1
                    r1 = r0
                    r0 = off
            # literals
            if ll > lend - lp:
                return ERR_DATA
            if ll <= 16 and lp + 16 <= llim and o + 16 <= oend:
                st64(dst, o, ld64(lit, lp))
                st64(dst, o + 8, ld64(lit, lp + 8))
            else:
                if o + ll > oend:
                    return ERR_SPACE
                copy_bytes(lit, lp, dst, o, ll)
            o += ll
            lp += ll
            # match
            if off > o - fstart or off <= 0:
                return ERR_DATA
            if o + ml > oend:
                return ERR_SPACE
            if off >= 16 and ml <= 32 and o + 32 <= oend:
                s = o - off
                st64(dst, o, ld64(dst, s))
                st64(dst, o + 8, ld64(dst, s + 8))
                st64(dst, o + 16, ld64(dst, s + 16))
                st64(dst, o + 24, ld64(dst, s + 24))
            else:
                copy_match(dst, o, off, ml, oend)
            o += ml
        if bc != dfc or q != start:
            return ERR_DATA
        iws[I_REP] = r0
        iws[I_REP + 1] = r1
        iws[I_REP + 2] = r2
    # last literals
    n = lend - lp
    if o + n > oend:
        return ERR_SPACE
    copy_bytes(lit, lp, dst, o, n)
    o += n
    if o - o0 > bsmax:
        return ERR_DATA
    return o


@njit(cache=True, nogil=True)
def zstd_decompress(src, p, pend, dst, o, oend, ws):
    """Every zstd frame in src[p:pend] (skippable frames skipped) decoded into
    dst[o:oend], as ZSTD_decompress. Returns the end of the output or a
    negative error code (ERR_SPACE too if ws is smaller than WS_SIZE)."""
    if ws.shape[0] < WS_SIZE:
        return ERR_SPACE
    iws = ws[W_INT:W_INT + 8192].view(np.int64)
    while p < pend:
        if pend - p < 5:
            return ERR_INPUT
        magic = _le32(src, p)
        if (magic & 0xFFFFFFF0) == 0x184D2A50:
            if pend - p < 8:
                return ERR_INPUT
            n = _le32(src, p + 4)
            if n > pend - p - 8:
                return ERR_INPUT
            p += 8 + n
            continue
        if magic != 0xFD2FB528:
            return ERR_DATA
        p += 4
        fhd = np.int64(src[p])
        p += 1
        fcsf = fhd >> 6
        single = (fhd >> 5) & 1
        if fhd & 8:
            return ERR_DATA
        chk = (fhd >> 2) & 1
        didf = fhd & 3
        nd = 0 if didf == 0 else (1 if didf == 1 else (2 if didf == 2 else 4))
        nf = (1 if single else 0) if fcsf == 0 else (2 if fcsf == 1 else (4 if fcsf == 2 else 8))
        if pend - p < (0 if single else 1) + nd + nf:
            return ERR_INPUT
        wsize = 0
        if not single:
            wd = np.int64(src[p])
            p += 1
            wlog = 10 + (wd >> 3)
            if wlog > 31:
                return ERR_DATA
            wsize = (1 << wlog) + ((1 << wlog) >> 3) * (wd & 7)
        did = 0
        for k in range(nd):
            did |= np.int64(src[p + k]) << (8 * k)
        p += nd
        if did != 0:
            return ERR_DATA                     # dictionaries: not supported
        fcs = -1
        if nf == 1:
            fcs = np.int64(src[p])
        elif nf == 2:
            fcs = (np.int64(src[p]) | (np.int64(src[p + 1]) << 8)) + 256
        elif nf == 4:
            fcs = _le32(src, p)
        elif nf == 8:
            if src[p + 7] >= 128:
                return ERR_SPACE
            fcs = _le32(src, p) | (_le32(src, p + 4) << 32)
        p += nf
        if single:
            wsize = fcs
        if fcs > oend - o:
            return ERR_SPACE
        bsmax = min(wsize, BLOCK_MAX)
        fstart = o
        iws[I_REP] = 1
        iws[I_REP + 1] = 4
        iws[I_REP + 2] = 8
        iws[I_HLOG] = 0
        iws[I_LOG] = -1
        iws[I_LOG + 1] = -1
        iws[I_LOG + 2] = -1
        while True:
            if pend - p < 3:
                return ERR_INPUT
            bh = np.int64(src[p]) | (np.int64(src[p + 1]) << 8) | (np.int64(src[p + 2]) << 16)
            p += 3
            last = bh & 1
            bt = (bh >> 1) & 3
            bs = bh >> 3
            if bt == 0:
                if bs > pend - p:
                    return ERR_INPUT
                if bs > oend - o:
                    return ERR_SPACE
                if bs > bsmax:
                    return ERR_DATA
                copy_bytes(src, p, dst, o, bs)
                p += bs
                o += bs
            elif bt == 1:
                if p >= pend:
                    return ERR_INPUT
                if bs > oend - o:
                    return ERR_SPACE
                if bs > bsmax:
                    return ERR_DATA
                v = src[p]
                for k in range(bs):
                    dst[o + k] = v
                p += 1
                o += bs
            elif bt == 2:
                if bs > pend - p:
                    return ERR_INPUT
                if bs > bsmax:
                    return ERR_DATA
                r = _block(src, p, p + bs, dst, fstart, o, oend, ws, iws, bsmax)
                if r < 0:
                    return r
                o = r
                p += bs
            else:
                return ERR_DATA
            if last:
                break
        if fcs >= 0 and o - fstart != fcs:
            return ERR_DATA
        if chk:
            if pend - p < 4:
                return ERR_INPUT
            p += 4
    return o


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """The HDF5 zstd filter: the frame(s) src[p:pend] into dst[o:oend]."""
    return zstd_decompress(src, p, pend, dst, o, oend, ws)


def workspace_size(chunk_nbytes):
    """The literal buffer and tables: WS_SIZE whatever the chunk."""
    return WS_SIZE
