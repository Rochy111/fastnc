"""HDF4's own compression coders, decoded in numba: run-length (RLE),
N-bit and skipping Huffman, as the HDF4 library writes them (crle.c,
cnbit.c, cskphuff.c of HDF 4.2). Deflate is zlib's format and goes
through `_inflate.zlib_inflate`; szip goes through `_szip` when fastnc
has one.

Each decoder follows the codec contract (docs/contracts.md, section 1):
`decode(src, p, pend, dst, o, oend, cd, ws)` decodes src[p:pend] into
dst[o:oend] and returns the end of what it wrote, or -1 (bad data), -3
(input ends early). It stops once dst[o:oend] is full, so a caller can
ask for only the head of a stream; it never writes outside dst[o:oend]
nor reads outside src[p:pend].

`cd` (int64) is [element size, decoded bytes, n, parameters...]:
RLE none; N-bit [nt_size, start_bit, bit_len, sign_ext, fill_one];
skipping Huffman [skip_size].
"""
import numpy as np

from ._jit import njit

ERR_DATA = -1
ERR_INPUT = -3


@njit(cache=True, nogil=True)
def rle_decode(src, p, pend, dst, o, oend, cd, ws):
    """HDF4 RLE: a control byte c; c & 0x80: the next byte repeated
    (c & 0x7f) + 3 times, else the next (c & 0x7f) + 1 bytes as they are."""
    while o < oend:
        if p >= pend:
            return ERR_INPUT
        c = np.int64(src[p])
        p += 1
        if c & 0x80:
            n = (c & 0x7f) + 3
            if p >= pend:
                return ERR_INPUT
            v = src[p]
            p += 1
            if n > oend - o:
                n = oend - o
            for k in range(n):
                dst[o + k] = v
            o += n
        else:
            n = (c & 0x7f) + 1
            if n > oend - o:
                n = oend - o
            if p + n > pend:
                return ERR_INPUT
            for k in range(n):
                dst[o + k] = src[p + k]
            p += n
            o += n
    return o


@njit(cache=True, nogil=True)
def _bits(src, pos, pend, n):
    """n bits (1..32) from bit position pos of src[..pend], most significant
    bit first, zeros past the end (as HDF4's Hbitread at end of file)."""
    v = np.int64(0)
    for k in range(n):
        byte = (pos + k) >> 3
        b = 0
        if byte < pend:
            b = (np.int64(src[byte]) >> (7 - ((pos + k) & 7))) & 1
        v = (v << 1) | b
    return v


@njit(cache=True, nogil=True)
def nbit_decode(src, p, pend, dst, o, oend, cd, ws):
    """HDF4 N-bit: bit_len bits per element, from bit start_bit down, packed
    most significant first; the other bits filled with ones or zeros, and
    sign-extended from the top bit kept if asked. Elements come out in the
    file's (big-endian) byte order."""
    nt = cd[3]
    mask_off = cd[4]
    mask_len = cd[5]
    sign_ext = cd[6] != 0
    fill_one = cd[7] != 0
    if nt < 1 or nt > 16 or mask_len < 1 or mask_off >= nt * 8 or mask_off - mask_len + 1 < 0:
        return ERR_DATA
    # per byte of an element: the bit offset of its top kept bit, how many
    # bits it keeps, and their mask (HCIcnbit_init)
    m_off = np.zeros(16, np.int64)
    m_len = np.zeros(16, np.int64)
    m_mask = np.zeros(16, np.int64)
    mask_top = mask_off
    mask_bot = mask_off - (mask_len - 1)
    top_bit = nt * 8 - 1
    bot_bit = nt * 8 - 8
    for i in range(nt):
        if mask_top >= top_bit:
            if mask_bot <= bot_bit:
                m_off[i] = 7
                m_len[i] = 8
                m_mask[i] = 0xff
            else:
                m_off[i] = 7
                m_len[i] = top_bit - mask_bot + 1
                m_mask[i] = (((1 << m_len[i]) - 1) << (8 - m_len[i])) & 0xff
                break
        elif mask_top >= bot_bit:
            if mask_bot < bot_bit:
                m_off[i] = mask_top - bot_bit
                m_len[i] = mask_top - bot_bit + 1
                m_mask[i] = (1 << m_len[i]) - 1
            else:
                m_off[i] = mask_top - bot_bit
                m_len[i] = mask_top - mask_bot + 1
                m_mask[i] = (((1 << m_len[i]) - 1) << (mask_bot - bot_bit)) & 0xff
                break
        top_bit -= 8
        bot_bit -= 8
    fill = np.zeros(16, np.int64)
    for i in range(nt):
        fill[i] = (0xff & ~m_mask[i]) if fill_one else 0
    sbit = mask_off % 8
    sign_ext_mask = (~((1 << sbit) - 1)) & 0xff
    sign_byte = nt - (mask_off // 8 + 1)
    sign_mask = 1 << sbit
    pos = p * 8
    pend_bits = pend
    sign_bit = 0
    while o + nt <= oend:
        for j in range(nt):
            v = fill[j]
            if m_len[j] > 0:
                bits = _bits(src, pos, pend_bits, m_len[j])
                pos += m_len[j]
                bits = bits << (m_off[j] - m_len[j] + 1)
                v |= m_mask[j] & bits & 0xff
                if sign_ext and j == sign_byte:
                    sign_bit = 1 if (sign_mask & bits) else 0
            dst[o + j] = v
        if sign_ext and sign_bit != (1 if fill_one else 0):
            if sign_bit == 1:
                for j in range(sign_byte):
                    dst[o + j] = 0xff
                dst[o + sign_byte] = dst[o + sign_byte] | sign_ext_mask
            else:
                for j in range(sign_byte):
                    dst[o + j] = 0
                dst[o + sign_byte] = dst[o + sign_byte] & (0xff & ~sign_ext_mask)
        o += nt
    if o < oend:
        return ERR_DATA                             # not a whole number of elements
    return o


@njit(cache=True, nogil=True)
def skphuff_decode(src, p, pend, dst, o, oend, cd, ws):
    """HDF4 skipping Huffman: an adaptive (splay-tree) Huffman code per byte
    position of a skip_size-byte element, bits most significant first."""
    skip = cd[3]
    if skip < 1 or skip > 64:
        return ERR_DATA
    left = np.empty((skip, 256), np.int64)
    right = np.empty((skip, 256), np.int64)
    up = np.empty((skip, 513), np.int64)
    for k in range(skip):
        for i in range(513):
            up[k, i] = (i >> 1) & 0xff                # (uint8 in HDF4)
        for j in range(256):
            left[k, j] = j << 1
            right[k, j] = (j << 1) + 1
    pos = p * 8
    nbits = (pend - p) * 8 + 64 * 8                 # zeros past the end, but not forever
    end_bits = pos + nbits
    sp = 0
    while o < oend:
        a = np.int64(0)
        while True:
            if pos >= end_bits:
                return ERR_INPUT
            byte = pos >> 3
            bit = 0
            if byte < pend:
                bit = (np.int64(src[byte]) >> (7 - (pos & 7))) & 1
            pos += 1
            a = left[sp, a] if bit == 0 else right[sp, a]
            if a > 255:
                break
        plain = a - 256
        # splay (HCIcskphuff_splay)
        a = plain + 256
        while True:
            c = up[sp, a]
            if c != 0:
                d = up[sp, c]
                b = left[sp, d]
                if c == b:
                    b = right[sp, d]
                    right[sp, d] = a
                else:
                    left[sp, d] = a
                if a == left[sp, c]:
                    left[sp, c] = b
                else:
                    right[sp, c] = b
                up[sp, a] = d
                up[sp, b] = c
                a = d
            else:
                a = c
            if a == 0:
                break
        dst[o] = plain
        o += 1
        sp += 1
        if sp >= skip:
            sp = 0
    return o


def workspace_size(chunk_nbytes):
    return 0
