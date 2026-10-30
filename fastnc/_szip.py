"""SZIP (HDF5 filter 4) as HDF5 writes it: through libaec's SZ
compatibility layer, a CCSDS 121.0 adaptive entropy coding stream.

What H5Zszip.c and libaec's sz_compat.c make of a chunk, and what is
undone here:

- H5Zszip.c puts the chunk's size in bytes in front, 4 bytes little-endian,
  then the SZ stream;
- cd values: options mask, pixels per block (J), bits per pixel, pixels per
  scanline. Of the mask only NN (32: the unit-delay predictor, so each
  block holds mapped residuals) and MSB (16: samples of more than 8 bits
  are big-endian, else little) mean anything to the decoder; EC (4), RAW
  (128: no szip header, always set by HDF5), ALLOW_K13 (1) and CHIP (2)
  change nothing in the stream libaec writes;
- 32- and 64-bit pixels are coded as bytes: the SZ layer interleaves the
  chunk first (every element's first byte, then every second byte ... --
  HDF5's shuffle), and codes 8-bit samples; the decoder undoes the
  shuffle after;
- a scanline is one reference sample interval of ceil(pps / J) blocks; when
  pps is not a multiple of J each scanline is padded to whole blocks, and
  the padding is dropped here as it is decoded;
- samples are unsigned (the SZ layer never sets AEC_DATA_SIGNED), of 1, 2 or
  4 bytes (9-16 bits: 2, 17-32: 4; AEC_DATA_3BYTE is not set).

The CCSDS decoder is fastgrib's (fastgrib/_images.py, `_aec`, same author),
taken from GRIB floats to raw samples: each block's coded values into a
block-sized buffer, the inverse preprocessing in place while they are in
L1, then a store loop for the sample width and byte order; a run of zero
blocks is one value repeated. Here it also reads several second-extension
codes from one load, and takes the inverse preprocessing as a select for
noisy blocks. One thread: 1.2x libaec on 8-bit samples, 1.8-2.3x on 16-bit
and on the bytes of 32-bit floats (tools/check/filters_hdf_bench.py).

The bit-level helpers here (bswap64, clz64, be64, window, getbits, st16,
st32, put_word) serve _scaleoffset, _nbit and _bzip2 too.
"""
import numpy as np

from ._intrin import ld64, st64
from ._jit import intrinsic, njit
from ._shuffle import unshuffle_into

ERR_DATA = -1                   # not a valid stream / parameters
ERR_SPACE = -2                  # output would not fit
ERR_INPUT = -3                  # stream ends early

SZ_ALLOW_K13, SZ_CHIP, SZ_EC, SZ_LSB, SZ_MSB, SZ_NN, SZ_RAW = 1, 2, 4, 8, 16, 32, 128
SE_TABLE_SIZE = 90              # libaec: a second-extension code above this is an error


def _se_table():
    """Second extension: code m is the pair (s - d1, d1) with s(s+1)/2 + d1 = m,
    0 <= d1 <= s. (s, d1) for each m up to SE_TABLE_SIZE."""
    t = np.zeros((SE_TABLE_SIZE + 1, 2), np.int64)
    m = 0
    s = 0
    while m <= SE_TABLE_SIZE:
        for d1 in range(s + 1):
            if m <= SE_TABLE_SIZE:
                t[m] = (s, d1)
            m += 1
        s += 1
    return t


_SE = _se_table()


@intrinsic
def bswap64(typingctx, x):
    """A uint64's bytes reversed: LLVM's bswap, one instruction."""
    from llvmlite import ir
    from numba import types
    from numba.core import cgutils
    def codegen(context, builder, sig, args):
        i64 = ir.IntType(64)
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(i64, [i64]),
                                            'llvm.bswap.i64')
        return builder.call(fn, [args[0]])
    return types.uint64(types.uint64), codegen


@intrinsic
def clz64(typingctx, x):
    """The leading zero bits of a uint64 (64 for 0): LLVM's ctlz (lzcnt)."""
    from llvmlite import ir
    from numba import types
    from numba.core import cgutils
    def codegen(context, builder, sig, args):
        i64 = ir.IntType(64)
        fn = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(i64, [i64, ir.IntType(1)]), 'llvm.ctlz.i64')
        return builder.call(fn, [args[0], ir.Constant(ir.IntType(1), 0)])
    return types.int64(types.uint64), codegen


@njit(cache=True, nogil=True)
def be64(src, q):
    """src[q:q+8] as a big-endian uint64 (one load and a bswap; unchecked)."""
    return bswap64(ld64(src, q))


def _st(bits):
    def st(typingctx, a, i, v):
        """a[i:i+n] = v's low n bytes, little-endian (an unaligned store)."""
        from llvmlite import ir
        from numba import types
        from numba.core import cgutils
        def codegen(context, builder, sig, args):
            ary = context.make_array(sig.args[0])(context, builder, args[0])
            p = cgutils.get_item_pointer(context, builder, sig.args[0], ary, [args[1]])
            p = builder.bitcast(p, ir.IntType(bits).as_pointer())
            builder.store(builder.trunc(args[2], ir.IntType(bits)), p, align=1)
            return context.get_dummy_value()
        return types.void(a, i, types.uint64), codegen
    st.__name__ = 'st%d' % bits
    return intrinsic(st)


st32 = _st(32)
st16 = _st(16)


@njit(cache=True, nogil=True)
def put_word(dst, at, w, size, be):
    """w's low `size` (1, 2, 4 or 8) bytes at dst[at:], little-endian, or
    big-endian if be: one store (unchecked)."""
    if be:
        w = bswap64(w) >> np.uint64(64 - 8 * size)
    if size == 8:
        st64(dst, at, w)
    elif size == 4:
        st32(dst, at, w)
    elif size == 2:
        st16(dst, at, w)
    else:
        dst[at] = w & np.uint64(255)


@njit(cache=True, nogil=True)
def _tail64(src, q, pend):
    """src[q:q+8] big-endian, zeros from byte pend on (the stream's end)."""
    x = np.uint64(0)
    for j in range(8):
        x <<= np.uint64(8)
        if q + j < pend:
            x |= np.uint64(src[q + j])
    return x


@njit(cache=True, nogil=True)
def window(src, pos, pend):
    """The stream from bit pos on, most significant bit first, in a uint64
    (57 bits or more of it; zeros past byte pend). One load and a bswap but
    near the end: small enough for LLVM to inline -- a bit read that
    was not inlined made split blocks 4x slower."""
    q = pos >> 3
    if q + 8 <= pend:
        x = be64(src, q)
    else:
        x = _tail64(src, q, pend)
    return x << np.uint64(pos & 7)


@njit(cache=True, nogil=True)
def getbits(src, pos, k, pend):
    """k (0..56) bits, big-endian, from absolute bit position pos of src,
    reading nothing at or past byte pend (zeros there). (The shift by one
    first makes k = 0 give 0 without a branch.)"""
    return np.int64((window(src, pos, pend) >> np.uint64(1)) >> np.uint64(63 - k))


@njit(cache=True, nogil=True)
def _fs_long(src, pos, pend):
    """_fs for a code of 57 zeros or more, or at the stream's end."""
    z = 0
    end = pend * 8
    while pos + z < end:
        w = window(src, pos + z, pend) >> np.uint64(8)
        if w != 0:
            lz = clz64(w) - 8                   # (w: 56 bits)
            return z + lz, z + lz + 1
        z += 56
    return -1, 0


@njit(cache=True, nogil=True)
def _fs(src, pos, pend):
    """A fundamental-sequence codeword at bit pos: (its value, bits used),
    or (-1, 0) if the stream ends first."""
    lz = clz64(window(src, pos, pend))
    if lz <= 56:
        return lz, lz + 1
    return _fs_long(src, pos, pend)


@njit(cache=True, nogil=True)
def _split_block(src, pos, pend, tmp, t, J):
    """The FS codes of a split-sample block's samples t..J-1 into tmp; the
    bit position after them, -1 if the stream ends first. Codes come from
    one 8-byte load while they fit in it, a leading-zero count each."""
    while t < J:
        q = pos >> 3
        if q + 8 > pend:                        # the stream's last bytes: one code at a time
            v, used = _fs(src, pos, pend)
            if v < 0:
                return -1
            pos += used
            tmp[t] = v
            t += 1
            continue
        w = window(src, pos, pend)              # 57 bits or more of the stream
        used = 0
        while t < J:
            lz = clz64(w)
            if lz + 1 + used > 56:              # not wholly within the bits loaded
                break
            tmp[t] = lz
            t += 1
            used += lz + 1
            w <<= np.uint64(lz + 1)
        if used == 0:                           # a code of 56 zeros or more
            v, used = _fs(src, pos, pend)
            if v < 0:
                return -1
            tmp[t] = v
            t += 1
        pos += used
    return pos


@njit(cache=True, nogil=True)
def _se_block(src, pos, pend, tmp, t, J):
    """A second-extension block's samples t..J-1 into tmp (FS codes of
    pairs; after a reference sample the first pair's first value is left
    out). The bit position after, -1 if the stream ends, -2 for a code
    libaec rejects."""
    if t:                                       # after a reference sample: half a pair
        mm, used = _fs(src, pos, pend)
        if mm < 0:
            return -1
        if mm > SE_TABLE_SIZE:
            return -2
        pos += used
        tmp[t] = _SE[mm, 1]
        t += 1
    while t < J:                                # whole pairs, codes from one load while they fit
        w = window(src, pos, pend)
        used = 0
        while t < J:
            lz = clz64(w)
            if lz + 1 + used > 57 or lz > SE_TABLE_SIZE:
                break
            tmp[t] = _SE[lz, 0] - _SE[lz, 1]
            tmp[t + 1] = _SE[lz, 1]
            t += 2
            used += lz + 1
            w <<= np.uint64(lz + 1)
        if used == 0:                           # a code of 57 zeros or more
            mm, used = _fs(src, pos, pend)
            if mm < 0:
                return -1
            if mm > SE_TABLE_SIZE:
                return -2
            tmp[t] = _SE[mm, 0] - _SE[mm, 1]
            tmp[t + 1] = _SE[mm, 1]
            t += 2
        pos += used
    return pos


@njit(cache=True, nogil=True)
def _store(out, at, tmp, m, pixel, msb):
    """tmp[0:m] as samples of `pixel` bytes at out[at:] (big-endian if msb).
    (Unsigned indices: no negative-index fix-ups, loops LLVM can vectorize;
    the common 1- and 2-byte little-endian cases are inlined by the caller.)"""
    u = np.uint64
    A = u(at)
    if pixel == 1:
        for t in range(u(m)):
            out[A + t] = tmp[t]
    elif pixel == 2:
        h = u(0) if msb else u(1)
        for t in range(u(m)):
            v = tmp[t]
            out[A + u(2) * t + h] = v >> 8
            out[A + u(2) * t + (u(1) - h)] = v
    else:
        for t in range(u(m)):
            v = tmp[t]
            if msb:
                out[A + u(4) * t] = v >> 24
                out[A + u(4) * t + u(1)] = v >> 16
                out[A + u(4) * t + u(2)] = v >> 8
                out[A + u(4) * t + u(3)] = v
            else:
                out[A + u(4) * t] = v
                out[A + u(4) * t + u(1)] = v >> 8
                out[A + u(4) * t + u(2)] = v >> 16
                out[A + u(4) * t + u(3)] = v >> 24


@njit(cache=True, nogil=True)
def _fill(out, at, v, m, pixel, msb):
    """m samples of value v at out[at:]."""
    if pixel == 1:
        b = np.uint8(v & 255)
        for t in range(m):
            out[at + t] = b
        return
    for t in range(pixel):
        sh = 8 * (pixel - 1 - t) if msb else 8 * t
        b = np.uint8((v >> sh) & 255)
        for i in range(m):
            out[at + i * pixel + t] = b


@njit(cache=True, nogil=True)
def aec_decode(src, p, pend, out, o, nout, bps, pixel, msb, pp, J, rsi, pps, tmp):
    """The CCSDS 121.0 stream at src[p:pend] into nout samples of `pixel`
    bytes at out[o:]: unsigned samples of bps bits, blocks of J samples,
    reference sample intervals (scanlines) of rsi blocks of which the first
    pps samples are kept, unit-delay preprocessing if pp. 0, or a negative
    error code."""
    if bps > 16:
        idlen = 5
    elif bps > 8:
        idlen = 4
    else:
        idlen = 3
    uncomp = (1 << idlen) - 1
    xmax = (np.int64(1) << bps) - 1
    pos = p * 8
    end = pend * 8
    k = 0                                       # samples out
    while k < nout:
        keep = min(pps, nout - k)               # samples of this scanline kept
        last = k + keep >= nout
        r = 0                                   # samples of the scanline decoded
        b = 0                                   # its blocks decoded
        xh = np.int64(0)                        # the sample before (the prediction)
        at = o + k * pixel
        while b < rsi:
            if last and r >= keep:
                break
            ref = 1 if pp and b == 0 else 0
            opt = getbits(src, pos, idlen, pend)
            pos += idlen
            sel = 1
            if opt == 0:                        # low entropy: zero blocks or second extension
                sel = getbits(src, pos, 1, pend)
                pos += 1
            if ref:                             # a reference sample, as it is
                xh = getbits(src, pos, bps, pend)
                pos += bps
                tmp[0] = xh
            if sel == 0:                        # zero blocks: the prediction (or 0) throughout
                z, used = _fs(src, pos, pend)
                if z < 0:
                    return ERR_INPUT
                pos += used
                z += 1
                if z == 5:                      # the rest of the segment
                    z = min(rsi - b, 64 - b % 64)
                elif z > 5:
                    z -= 1
                n = z * J
                if r < keep:
                    _fill(out, at + r * pixel, xh if pp else 0, min(n, keep - r), pixel, msb)
                r += n
                b += z
                continue
            kk = 0
            if opt == 0:
                pos = _se_block(src, pos, pend, tmp, ref, J)
                if pos == -2:
                    return ERR_DATA
            elif opt == uncomp:
                sh = np.uint64(64 - bps)
                for t in range(ref, J):
                    tmp[t] = np.int64(window(src, pos, pend) >> sh)
                    pos += bps
            else:
                kk = opt - 1
                pos = _split_block(src, pos, pend, tmp, ref, J)
                if kk and pos >= 0:             # the low bits, after all the FS codes
                    sh = np.uint64(64 - kk)
                    if ((pos + J * kk) >> 3) + 8 <= pend:
                        for t in range(ref, J):
                            x = be64(src, pos >> 3) << np.uint64(pos & 7)
                            tmp[t] = (tmp[t] << kk) | np.int64(x >> sh)
                            pos += kk
                    else:
                        for t in range(ref, J):
                            tmp[t] = (tmp[t] << kk) | np.int64(window(src, pos, pend) >> sh)
                            pos += kk
            if pos < 0:
                return ERR_INPUT
            if pp and (opt == uncomp or kk >= bps - 3):
                # noisy samples (raw, or nearly all their bits split off): the
                # residual is often beyond the prediction's reach, so both
                # cases are computed and one taken -- a branch would be
                # mispredicted half the time (f4 bytes 220 -> 350 MB/s)
                for t in range(ref, J):
                    d = tmp[t]
                    th = xmax - xh
                    lo = xh <= th
                    near = xh + ((d >> 1) ^ -(d & 1))
                    far = d if lo else th - d + xh
                    xh = near if d <= 2 * (xh if lo else th) else far
                    tmp[t] = xh
            elif pp:                            # mapped residuals from the sample before
                for t in range(ref, J):
                    d = tmp[t]
                    # within reach of the prediction (nearly always): +d/2 for
                    # even d, -(d + 1)/2 for odd, the parity without a branch
                    if d <= 2 * min(xh, xmax - xh):
                        xh = xh + ((d >> 1) ^ -(d & 1))
                    else:
                        xh = d if xh <= xmax - xh else xmax - d
                    tmp[t] = xh
            if r < keep:
                mm = min(J, keep - r)
                if pixel == 1:                  # (inline: a call each block costs 15%)
                    A = np.uint64(at + r)
                    for t in range(np.uint64(mm)):
                        out[A + t] = tmp[t]
                elif pixel == 2 and not msb:
                    A = np.uint64(at + 2 * r)
                    for t in range(np.uint64(mm)):
                        v = tmp[t]
                        out[A + np.uint64(2) * t] = v
                        out[A + np.uint64(2) * t + np.uint64(1)] = v >> 8
                else:
                    _store(out, at + r * pixel, tmp, mm, pixel, msb)
            r += J
            b += 1
        k += keep
        if pos > end:
            return ERR_INPUT
    return 0


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """An HDF5 szip chunk at src[p:pend] into dst[o:...]; cd: [element size,
    chunk bytes, 4, options mask, pixels per block, bits per pixel, pixels
    per scanline]. ws holds the interleaved bytes of 32/64-bit pixels
    (workspace_size). The end of the output, or a negative error code."""
    if cd[2] < 4:
        return ERR_DATA
    mask = cd[3]
    J = cd[4]
    bpp = cd[5]
    pps = cd[6]
    if pend - p < 4:
        return ERR_INPUT
    nalloc = (np.int64(src[p]) | (np.int64(src[p + 1]) << 8) | (np.int64(src[p + 2]) << 16)
              | (np.int64(src[p + 3]) << 24))
    if nalloc > oend - o:
        return ERR_SPACE
    if J < 2 or J > 64 or J % 2 or pps < 1 or bpp < 1 or (bpp > 32 and bpp != 64):
        return ERR_DATA
    rsi = (pps + J - 1) // J
    if rsi > 4096:
        return ERR_DATA
    pp = (mask & SZ_NN) != 0
    msb = (mask & SZ_MSB) != 0
    tmp = np.empty(64, np.int64)
    if bpp == 32 or bpp == 64:                  # bytes, interleaved
        if ws.size < nalloc:
            return ERR_SPACE
        err = aec_decode(src, p + 4, pend, ws, 0, nalloc, 8, 1, msb, pp, J, rsi, pps, tmp)
        if err < 0:
            return err
        unshuffle_into(ws, 0, nalloc, bpp // 8, dst, o)
        return o + nalloc
    pixel = 1 if bpp <= 8 else (2 if bpp <= 16 else 4)
    nout = nalloc // pixel
    err = aec_decode(src, p + 4, pend, dst, o, nout, bpp, pixel, msb, pp, J, rsi, pps, tmp)
    if err < 0:
        return err
    for i in range(o + nout * pixel, o + nalloc):   # (never, for HDF5's chunks)
        dst[i] = 0
    return o + nalloc


def workspace_size(chunk_nbytes):
    """The interleaved bytes of a chunk of 32- or 64-bit pixels."""
    return int(chunk_nbytes)
