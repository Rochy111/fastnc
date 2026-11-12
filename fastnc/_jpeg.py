"""The HDF5 JPEG filter (32019, HDFGroup hdf5_plugins `H5Zjpeg.c`) in numba.

The filter stores each chunk as one JPEG image (cd values: quality,
columns, rows, 0 mono / 1 RGB; only writing reads them) and decodes it with
libjpeg's defaults, writing the rows one after another, components
interleaved. This decodes the same way, so the bytes are libjpeg's:

- baseline and extended sequential Huffman JPEG, 8-bit samples, any number
  of scans (interleaved or one component each), restart intervals;
- dequantization and the inverse DCT as `jpeg_idct_islow` (jidctint.c,
  libjpeg's default method: 13-bit constants, 2 extra bits after the first
  pass), its results through libjpeg's range-limit table (values wrap and
  clamp as there);
- chroma upsampling as libjpeg-turbo (and IJG's 6b) do by default: the
  "fancy" triangle filters for 2:1 horizontally (h2v1) and 2:1 both ways
  (h2v2) when the component is wider than 2 samples, edge rows and columns
  repeated; other integer ratios by repeating samples. IJG's libjpeg 7 to 9
  upsample by scaled DCTs instead and give slightly different colours;
- YCbCr to RGB with jdcolor.c's tables (16-bit fixed point); the colour
  space as libjpeg decides it (JFIF: YCbCr; an Adobe marker's transform;
  else component IDs 'R', 'G', 'B' mean RGB).

Progressive, lossless, arithmetic-coded, 12-bit and CMYK images are refused
(the filter writes none of them).
"""
import numpy as np

from ._inflate import ERR_DATA, ERR_INPUT, ERR_SPACE
from ._jit import njit

NATURAL = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
                    41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15,
                    23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63],
                   np.int64)


def _idct_limit():
    """libjpeg's post-IDCT range limit (jdmaster.c prepare_range_limit_table),
    indexed by the descaled value & 1023: x + 128 clamped to 0..255 for x in
    -512..511, wrapping beyond."""
    t = np.zeros(1024, np.int64)
    for i in range(1024):
        x = i if i < 512 else i - 1024
        t[i] = min(255, max(0, x + 128))
    return t


IDCT_LIMIT = _idct_limit()


def _ycc_tables():
    """jdcolor.c build_ycc_rgb_table: Cr->R, Cb->B, Cr->G, Cb->G (+1/2)."""
    def fix(x):
        return int(x * (1 << 16) + 0.5)
    t = np.zeros((4, 256), np.int64)
    for i in range(256):
        x = i - 128
        t[0, i] = (fix(1.40200) * x + (1 << 15)) >> 16
        t[1, i] = (fix(1.77200) * x + (1 << 15)) >> 16
        t[2, i] = -fix(0.71414) * x
        t[3, i] = -fix(0.34414) * x + (1 << 15)
    return t


YCC = _ycc_tables()

# bit reader state (an int64 array: numba functions cannot share scalars)
S_POS, S_BUF, S_CNT, S_MARK, S_END = 0, 1, 2, 3, 4


@njit(cache=True, nogil=True)
def _fill(src, st):
    """Bytes into the bit buffer up to more than 48 bits: stuffed 0xFF 0x00
    as 0xFF; at a marker (or the end) zeros, as libjpeg inserts them."""
    pos = st[S_POS]
    buf = st[S_BUF]
    cnt = st[S_CNT]
    end = st[S_END]
    while cnt <= 48:
        b = 0
        if st[S_MARK] == 0 and pos < end:
            b = np.int64(src[pos])
            if b == 0xFF:
                nb = np.int64(src[pos + 1]) if pos + 1 < end else 0xD9
                if nb == 0:
                    pos += 2
                else:
                    st[S_MARK] = nb                     # (left in place, at its 0xFF)
                    b = 0
            else:
                pos += 1
        elif st[S_MARK] == 0:
            st[S_MARK] = 0x100                          # (out of data)
        buf = (buf << 8) | b
        cnt += 8
    st[S_POS] = pos
    st[S_BUF] = buf
    st[S_CNT] = cnt


@njit(cache=True, nogil=True)
def _bits(src, st, n):
    if n == 0:
        return 0
    if st[S_CNT] < n:
        _fill(src, st)
    cnt = st[S_CNT] - n
    v = (st[S_BUF] >> cnt) & ((1 << n) - 1)
    st[S_CNT] = cnt
    st[S_BUF] &= (1 << cnt) - 1
    return v


@njit(cache=True, nogil=True)
def _huff(src, st, t, hmax, hoff, hval, hlook):
    """One Huffman symbol of table t (jdhuff.c's lookahead, then bit by bit);
    -1 for a code no table has."""
    if st[S_CNT] < 16:
        _fill(src, st)
    cnt = st[S_CNT]
    buf = st[S_BUF]
    look = hlook[t, (buf >> (cnt - 9)) & 511]
    nb = look >> 8
    if nb:
        cnt -= nb
        st[S_CNT] = cnt
        st[S_BUF] = buf & ((1 << cnt) - 1)
        return look & 255
    code = (buf >> (cnt - 9)) & 511
    ln = 9
    while ln < 16:
        ln += 1
        code = (code << 1) | ((buf >> (cnt - ln)) & 1)
        if code <= hmax[t, ln]:
            cnt -= ln
            st[S_CNT] = cnt
            st[S_BUF] = buf & ((1 << cnt) - 1)
            return hval[t, hoff[t, ln] + code]
    return -1


@njit(cache=True, nogil=True)
def _extend(v, s):
    return v - (1 << s) + 1 if v < (1 << (s - 1)) else v


@njit(cache=True, nogil=True)
def _make_table(counts, vals, t, hmax, hoff, hval, hlook):
    """jpeg_make_d_derived_tbl: max code and value offset per length, and the
    9-bit lookahead table. 0, or an error for a table no code fits."""
    size = np.zeros(257, np.int64)
    code = np.zeros(257, np.int64)
    p = 0
    for ln in range(1, 17):
        for i in range(counts[ln]):
            if p >= 256:
                return ERR_DATA
            size[p] = ln
            p += 1
    size[p] = 0
    total = p
    c = 0
    si = size[0]
    p = 0
    while size[p]:
        while size[p] == si:
            code[p] = c
            p += 1
            c += 1
        if c >= (1 << si):
            return ERR_DATA
        c <<= 1
        si += 1
    p = 0
    for ln in range(1, 17):
        if counts[ln]:
            hoff[t, ln] = p - code[p]
            p += counts[ln]
            hmax[t, ln] = code[p - 1]
        else:
            hmax[t, ln] = -1
    hmax[t, 17] = 1 << 40
    for i in range(512):
        hlook[t, i] = 0
    p = 0
    for ln in range(1, 10):
        for i in range(counts[ln]):
            lb = code[p] << (9 - ln)
            for k in range(1 << (9 - ln)):
                hlook[t, lb + k] = (ln << 8) | vals[p]
            p += 1
    for i in range(total):
        hval[t, i] = vals[i]
    return 0


@njit(cache=True, nogil=True)
def _idct(coef, q, plane, at, stride, ws):
    """jpeg_idct_islow: coef (natural order) times the quantization table q,
    into plane[at:] (8 rows of `stride`)."""
    for c in range(8):
        z2 = coef[16 + c] * q[16 + c]
        z3 = coef[48 + c] * q[48 + c]
        z1 = (z2 + z3) * 4433
        tmp2 = z1 + z3 * -15137
        tmp3 = z1 + z2 * 6270
        z2 = coef[c] * q[c]
        z3 = coef[32 + c] * q[32 + c]
        tmp0 = (z2 + z3) << 13
        tmp1 = (z2 - z3) << 13
        t10 = tmp0 + tmp3
        t13 = tmp0 - tmp3
        t11 = tmp1 + tmp2
        t12 = tmp1 - tmp2
        o0 = coef[56 + c] * q[56 + c]
        o1 = coef[40 + c] * q[40 + c]
        o2 = coef[24 + c] * q[24 + c]
        o3 = coef[8 + c] * q[8 + c]
        z1 = o0 + o3
        z2 = o1 + o2
        z3 = o0 + o2
        z4 = o1 + o3
        z5 = (z3 + z4) * 9633
        o0 = o0 * 2446
        o1 = o1 * 16819
        o2 = o2 * 25172
        o3 = o3 * 12299
        z1 = z1 * -7373
        z2 = z2 * -20995
        z3 = z3 * -16069 + z5
        z4 = z4 * -3196 + z5
        o0 += z1 + z3
        o1 += z2 + z4
        o2 += z2 + z3
        o3 += z1 + z4
        r = 1 << 10
        ws[c] = (t10 + o3 + r) >> 11
        ws[56 + c] = (t10 - o3 + r) >> 11
        ws[8 + c] = (t11 + o2 + r) >> 11
        ws[48 + c] = (t11 - o2 + r) >> 11
        ws[16 + c] = (t12 + o1 + r) >> 11
        ws[40 + c] = (t12 - o1 + r) >> 11
        ws[24 + c] = (t13 + o0 + r) >> 11
        ws[32 + c] = (t13 - o0 + r) >> 11
    for rw in range(8):
        w = rw * 8
        z2 = ws[w + 2]
        z3 = ws[w + 6]
        z1 = (z2 + z3) * 4433
        tmp2 = z1 + z3 * -15137
        tmp3 = z1 + z2 * 6270
        tmp0 = (ws[w] + ws[w + 4]) << 13
        tmp1 = (ws[w] - ws[w + 4]) << 13
        t10 = tmp0 + tmp3
        t13 = tmp0 - tmp3
        t11 = tmp1 + tmp2
        t12 = tmp1 - tmp2
        o0 = ws[w + 7]
        o1 = ws[w + 5]
        o2 = ws[w + 3]
        o3 = ws[w + 1]
        z1 = o0 + o3
        z2 = o1 + o2
        z3 = o0 + o2
        z4 = o1 + o3
        z5 = (z3 + z4) * 9633
        o0 = o0 * 2446
        o1 = o1 * 16819
        o2 = o2 * 25172
        o3 = o3 * 12299
        z1 = z1 * -7373
        z2 = z2 * -20995
        z3 = z3 * -16069 + z5
        z4 = z4 * -3196 + z5
        o0 += z1 + z3
        o1 += z2 + z4
        o2 += z2 + z3
        o3 += z1 + z4
        r = 1 << 17
        d = at + rw * stride
        plane[d] = IDCT_LIMIT[((t10 + o3 + r) >> 18) & 1023]
        plane[d + 7] = IDCT_LIMIT[((t10 - o3 + r) >> 18) & 1023]
        plane[d + 1] = IDCT_LIMIT[((t11 + o2 + r) >> 18) & 1023]
        plane[d + 6] = IDCT_LIMIT[((t11 - o2 + r) >> 18) & 1023]
        plane[d + 2] = IDCT_LIMIT[((t12 + o1 + r) >> 18) & 1023]
        plane[d + 5] = IDCT_LIMIT[((t12 - o1 + r) >> 18) & 1023]
        plane[d + 3] = IDCT_LIMIT[((t13 + o0 + r) >> 18) & 1023]
        plane[d + 4] = IDCT_LIMIT[((t13 - o0 + r) >> 18) & 1023]


@njit(cache=True, nogil=True)
def _seg_len(src, p, pend):
    if p + 4 > pend:
        return -1
    return (np.int64(src[p + 2]) << 8) | np.int64(src[p + 3])


@njit(cache=True, nogil=True)
def _upsample(plane, poff, pw, dw, dh, hr, vr, W, H, full):
    """One component, dw x dh samples in plane[poff:] (row stride pw), to the
    image's W x H in `full`, as libjpeg-turbo's jdsample.c with fancy
    upsampling on."""
    if hr == 1 and vr == 1:
        for y in range(H):
            for x in range(W):
                full[y * W + x] = plane[poff + y * pw + x]
        return
    if hr == 2 and vr == 1 and dw > 2:
        row = np.empty(2 * dw, np.int64)
        for y in range(H):
            a = poff + y * pw
            v = np.int64(plane[a])
            row[0] = v
            row[1] = (v * 3 + plane[a + 1] + 2) >> 2
            for c in range(1, dw - 1):
                v = np.int64(plane[a + c]) * 3
                row[2 * c] = (v + plane[a + c - 1] + 1) >> 2
                row[2 * c + 1] = (v + plane[a + c + 1] + 2) >> 2
            v = np.int64(plane[a + dw - 1])
            row[2 * dw - 2] = (v * 3 + plane[a + dw - 2] + 1) >> 2
            row[2 * dw - 1] = v
            for x in range(W):
                full[y * W + x] = row[x]
        return
    if hr == 2 and vr == 2 and dw > 2:
        row = np.empty(2 * dw, np.int64)
        for y in range(H):
            i0 = y >> 1
            i1 = i0 - 1 if (y & 1) == 0 else i0 + 1
            i0 = min(i0, dh - 1)
            i1 = min(max(i1, 0), dh - 1)
            a = poff + i0 * pw
            b = poff + i1 * pw
            this = np.int64(plane[a]) * 3 + plane[b]
            nxt = np.int64(plane[a + 1]) * 3 + plane[b + 1]
            row[0] = (this * 4 + 8) >> 4
            row[1] = (this * 3 + nxt + 7) >> 4
            last = this
            this = nxt
            for c in range(1, dw - 1):
                nxt = np.int64(plane[a + c + 1]) * 3 + plane[b + c + 1]
                row[2 * c] = (this * 3 + last + 8) >> 4
                row[2 * c + 1] = (this * 3 + nxt + 7) >> 4
                last = this
                this = nxt
            row[2 * dw - 2] = (this * 3 + last + 8) >> 4
            row[2 * dw - 1] = (this * 4 + 7) >> 4
            for x in range(W):
                full[y * W + x] = row[x]
        return
    for y in range(H):
        a = poff + min(y // vr, dh - 1) * pw
        for x in range(W):
            full[y * W + x] = plane[a + min(x // hr, dw - 1)]


@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """One JPEG image src[p:pend] into dst[o:], rows of interleaved samples
    as the HDF5 filter writes them. The end of the output or an error."""
    if pend - p < 4 or src[p] != 0xFF or src[p + 1] != 0xD8:
        return ERR_DATA
    qt = np.zeros((4, 64), np.int64)
    qset = np.zeros(4, np.int64)
    hmax = np.zeros((8, 18), np.int64)
    hoff = np.zeros((8, 18), np.int64)
    hval = np.zeros((8, 256), np.int64)
    hlook = np.zeros((8, 512), np.int64)
    hset = np.zeros(8, np.int64)
    counts = np.zeros(17, np.int64)
    vals = np.zeros(256, np.int64)
    cid = np.zeros(4, np.int64)
    ch = np.zeros(4, np.int64)
    cv = np.zeros(4, np.int64)
    cq = np.zeros(4, np.int64)
    poff = np.zeros(4, np.int64)
    pwid = np.zeros(4, np.int64)
    st = np.zeros(5, np.int64)
    coef = np.zeros(64, np.int64)
    iws = np.zeros(64, np.int64)
    pred = np.zeros(4, np.int64)
    sc = np.zeros(4, np.int64)
    std = np.zeros(4, np.int64)
    sta = np.zeros(4, np.int64)
    nf = 0
    W = 0
    H = 0
    hmx = 1
    vmx = 1
    mcux = 0
    mcuy = 0
    plane = np.zeros(0, np.uint8)
    interval = 0
    jfif = False
    adobe = -1
    scans = 0
    q = p + 2
    while True:
        # the next marker (fill bytes 0xFF skipped)
        while q < pend and src[q] != 0xFF:
            q += 1                                      # (bytes between segments: skipped, as libjpeg warns)
        while q + 1 < pend and src[q + 1] == 0xFF:
            q += 1
        if q + 1 >= pend:
            return ERR_INPUT
        m = np.int64(src[q + 1])
        if m == 0xD9:
            break
        if m == 0xD8 or (0xD0 <= m <= 0xD7) or m == 0x01:
            q += 2
            continue
        ln = _seg_len(src, q, pend)
        if ln < 2 or q + 2 + ln > pend:
            return ERR_INPUT
        a = q + 4
        e = q + 2 + ln
        if m == 0xC0 or m == 0xC1:
            if nf or ln < 8 or src[a] != 8:
                return ERR_DATA                         # (a second frame, or not 8-bit)
            H = (np.int64(src[a + 1]) << 8) | src[a + 2]
            W = (np.int64(src[a + 3]) << 8) | src[a + 4]
            nf = np.int64(src[a + 5])
            if H <= 0 or W <= 0 or (nf != 1 and nf != 3) or ln < 8 + 3 * nf:
                return ERR_DATA
            for i in range(nf):
                cid[i] = src[a + 6 + 3 * i]
                ch[i] = src[a + 7 + 3 * i] >> 4
                cv[i] = src[a + 7 + 3 * i] & 15
                cq[i] = src[a + 8 + 3 * i]
                if ch[i] < 1 or ch[i] > 4 or cv[i] < 1 or cv[i] > 4 or cq[i] > 3:
                    return ERR_DATA
                hmx = max(hmx, ch[i])
                vmx = max(vmx, cv[i])
            for i in range(nf):
                if hmx % ch[i] or vmx % cv[i]:
                    return ERR_DATA                     # (ratios that are not whole)
            mcux = (W + 8 * hmx - 1) // (8 * hmx)
            mcuy = (H + 8 * vmx - 1) // (8 * vmx)
            tot = 0
            for i in range(nf):
                poff[i] = tot
                pwid[i] = mcux * ch[i] * 8
                tot += pwid[i] * mcuy * cv[i] * 8
            plane = np.zeros(tot, np.uint8)
        elif m == 0xC4:
            while a < e:
                tc = np.int64(src[a]) >> 4
                th = np.int64(src[a]) & 15
                if tc > 1 or th > 3 or a + 17 > e:
                    return ERR_DATA
                n = 0
                for k in range(1, 17):
                    counts[k] = src[a + k]
                    n += counts[k]
                if n > 256 or a + 17 + n > e:
                    return ERR_DATA
                for k in range(n):
                    vals[k] = src[a + 17 + k]
                    if tc == 0 and vals[k] > 15:
                        return ERR_DATA
                t = tc * 4 + th
                if _make_table(counts, vals, t, hmax, hoff, hval, hlook) < 0:
                    return ERR_DATA
                hset[t] = 1
                a += 17 + n
        elif m == 0xDB:
            while a < e:
                pq = np.int64(src[a]) >> 4
                tq = np.int64(src[a]) & 15
                if pq > 1 or tq > 3 or a + 1 + 64 * (pq + 1) > e:
                    return ERR_DATA
                for k in range(64):
                    if pq:
                        v = (np.int64(src[a + 1 + 2 * k]) << 8) | src[a + 2 + 2 * k]
                    else:
                        v = np.int64(src[a + 1 + k])
                    qt[tq, NATURAL[k]] = v
                qset[tq] = 1
                a += 1 + 64 * (pq + 1)
        elif m == 0xDD:
            if ln < 4:
                return ERR_DATA
            interval = (np.int64(src[a]) << 8) | src[a + 1]
        elif m == 0xE0:
            if ln >= 7 and src[a] == 74 and src[a + 1] == 70 and src[a + 2] == 73 and src[a + 3] == 70 \
                    and src[a + 4] == 0:
                jfif = True
        elif m == 0xEE:
            if ln >= 14 and src[a] == 65 and src[a + 1] == 100 and src[a + 2] == 111 and src[a + 3] == 98 \
                    and src[a + 4] == 101:
                adobe = np.int64(src[a + 11])
        elif m == 0xDA:
            if nf == 0:
                return ERR_DATA
            ns = np.int64(src[a])
            if ns < 1 or ns > nf or ln < 6 + 2 * ns:
                return ERR_DATA
            for i in range(ns):
                c = -1
                for k in range(nf):
                    if cid[k] == src[a + 1 + 2 * i]:
                        c = k
                if c < 0:
                    return ERR_DATA
                sc[i] = c
                std[i] = np.int64(src[a + 2 + 2 * i]) >> 4
                sta[i] = (np.int64(src[a + 2 + 2 * i]) & 15) + 4
                if std[i] > 3 or sta[i] > 7 or not hset[std[i]] or not hset[sta[i]] or not qset[cq[c]]:
                    return ERR_DATA
            ss = src[a + 1 + 2 * ns]
            se = src[a + 2 + 2 * ns]
            if ss != 0 or se != 63:
                return ERR_DATA                         # (progressive)
            # the entropy-coded data
            st[S_POS] = e
            st[S_BUF] = 0
            st[S_CNT] = 0
            st[S_MARK] = 0
            st[S_END] = pend
            for i in range(4):
                pred[i] = 0
            if ns == 1:
                c0 = sc[0]
                bw = ((W * ch[c0] + hmx - 1) // hmx + 7) // 8
                bh = ((H * cv[c0] + vmx - 1) // vmx + 7) // 8
                nmcu = bw * bh
            else:
                bw = mcux
                bh = mcuy
                nmcu = mcux * mcuy
            left = interval
            rst = 0
            for mi in range(nmcu):
                if interval:
                    if left == 0:
                        # restart: to the byte, then the RSTn marker
                        st[S_CNT] = 0
                        st[S_BUF] = 0
                        if st[S_MARK] == 0:
                            k = st[S_POS]
                            while k + 1 < pend and not (src[k] == 0xFF and 0xD0 <= src[k + 1] <= 0xD7):
                                k += 1
                            st[S_POS] = k
                        elif st[S_MARK] == 0x100:
                            return ERR_INPUT
                        k = st[S_POS]
                        if k + 1 < pend and src[k] == 0xFF and src[k + 1] == 0xD0 + rst:
                            st[S_POS] = k + 2
                        st[S_MARK] = 0
                        rst = (rst + 1) & 7
                        for i in range(4):
                            pred[i] = 0
                        left = interval
                    left -= 1
                my = mi // bw
                mx = mi - my * bw
                for i in range(ns):
                    c = sc[i]
                    nh = 1 if ns == 1 else ch[c]
                    nv = 1 if ns == 1 else cv[c]
                    for by in range(nv):
                        for bx in range(nh):
                            for k in range(64):
                                coef[k] = 0
                            s = _huff(src, st, std[i], hmax, hoff, hval, hlook)
                            if s < 0:
                                return ERR_DATA
                            d = _extend(_bits(src, st, s), s) if s else 0
                            v = pred[c] + d
                            pred[c] = v
                            coef[0] = ((v + 32768) & 0xFFFF) - 32768     # (JCOEF: 16 bits)
                            k = 1
                            while k < 64:
                                rs = _huff(src, st, sta[i], hmax, hoff, hval, hlook)
                                if rs < 0:
                                    return ERR_DATA
                                r = rs >> 4
                                s = rs & 15
                                if s:
                                    k += r
                                    if k > 63:
                                        return ERR_DATA
                                    coef[NATURAL[k]] = _extend(_bits(src, st, s), s)
                                    k += 1
                                else:
                                    if r != 15:
                                        break
                                    k += 16
                            row = (my * nv + by) * 8
                            col = (mx * nh + bx) * 8
                            if row + 8 > mcuy * cv[c] * 8 or col + 8 > pwid[c]:
                                return ERR_DATA
                            _idct(coef, qt[cq[c]], plane, poff[c] + row * pwid[c] + col, pwid[c], iws)
            scans += 1
            # past the scan's data: to the next marker
            k = st[S_POS]
            while k + 1 < pend:
                if src[k] == 0xFF and src[k + 1] != 0 and not (0xD0 <= src[k + 1] <= 0xD7):
                    break
                k += 1
            q = k
            continue
        elif (0xC2 <= m <= 0xCF) and m != 0xC4 and m != 0xC8 and m != 0xCC:
            return ERR_DATA                             # (progressive, lossless, arithmetic)
        q = e
    if nf == 0 or scans == 0:
        return ERR_DATA
    n = W * H * nf
    if n > oend - o:
        return ERR_SPACE
    full = np.empty(W * H * nf, np.int64)
    for i in range(nf):
        dw = (W * ch[i] + hmx - 1) // hmx
        dh = (H * cv[i] + vmx - 1) // vmx
        _upsample(plane, poff[i], pwid[i], dw, dh, hmx // ch[i], vmx // cv[i], W, H,
                  full[i * W * H:(i + 1) * W * H])
    if nf == 1:
        for k in range(n):
            dst[o + k] = full[k]
        return o + n
    rgb = cid[0] == 82 and cid[1] == 71 and cid[2] == 66
    if jfif:
        rgb = False
    elif adobe >= 0:
        rgb = adobe == 0
    WH = W * H
    if rgb:
        for k in range(WH):
            dst[o + 3 * k] = full[k]
            dst[o + 3 * k + 1] = full[WH + k]
            dst[o + 3 * k + 2] = full[2 * WH + k]
        return o + n
    for k in range(WH):
        yy = full[k]
        cb = full[WH + k]
        cr = full[2 * WH + k]
        dst[o + 3 * k] = min(255, max(0, yy + YCC[0, cr]))
        dst[o + 3 * k + 1] = min(255, max(0, yy + ((YCC[3, cb] + YCC[2, cr]) >> 16)))
        dst[o + 3 * k + 2] = min(255, max(0, yy + YCC[1, cb]))
    return o + n


def workspace_size(chunk_nbytes):
    """Nothing: the decoder makes its tables and planes for each image."""
    return 64
