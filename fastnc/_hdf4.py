"""HDF4, read from the file's bytes, and its SD (scientific data) model
presented as netCDF-c's HDF4 dispatch presents it to netCDF4-python.

The file: a magic number, then linked blocks of data descriptors (DDs:
tag, reference number, offset, length; big-endian), each naming one data
element. Special elements (tag | 0x4000) start with a header saying how
their bytes are really kept: in linked blocks, in an external file,
compressed (deflate, RLE, N-bit, skipping Huffman, szip) or chunked (a
Vdata chunk table, each chunk an element of its own, compressed or not).
Vgroups (tag 1965) list tags and refs of other elements; Vdatas (header
1962, records 1963) are little tables.

The SD interface (mfhdf) keeps, in a Vgroup of class "CDF0.0": one Vgroup
per dimension ("Dim0.0" / "UDim0.0", with a "DimVal0.0/0.1" Vdata giving
its size), one per variable ("Var0.0": its dimensions' Vgroups, its
number type, its data element (tag 702), its attributes) and the file's
attributes ("Attr0.0" Vdatas, one value field each). Coordinate
variables (dimension scales) are variables of their own.

netCDF-c (libhdf4/hdf4file.c) shows every SD variable, coordinate
variables included, in the file's order, in the root group; dimensions
by name, in the order variables first use them, none unlimited (an
unlimited one is as long as the first variable using it); attributes as
stored; types big-endian unless stored little-endian (DFNT_L*); data
model 'NETCDF4'. This module gives the same, reading the bytes itself.

Raw values come back native-endian (contract, docs/contracts.md section
2). Chunked, compressed and contiguous data are decoded through the
chunk engine (`_chunks`) on fastnc's pool: deflated chunks inflate in
numba in parallel, straight into the output, big-endian swapped as
placed; a contiguous array is cut into pseudo-chunks so that the copy and
byte swap run on every core too. RLE, N-bit and skipping Huffman decode
in numba (`_hdf4_codecs`), chunk by chunk on the pool.
"""
import os
import struct

import numpy as np

from . import _chunks, _pool, _select

MAGIC = b'\x0e\x03\x13\x01'

# tags (htags.h)
T_NULL, T_LINKED, T_COMPRESSED, T_CHUNK = 1, 20, 40, 61
T_NT, T_SDD, T_SD, T_NDG = 106, 701, 702, 720
T_VH, T_VS, T_VG = 1962, 1963, 1965
SPECIAL = 0x4000
S_LINKED, S_EXT, S_COMP, S_VLINKED, S_CHUNKED = 1, 2, 3, 4, 5

# coders (hcomp.h)
C_NONE, C_RLE, C_NBIT, C_SKPHUFF, C_DEFLATE, C_SZIP = 0, 1, 2, 3, 4, 5
CODER_NAMES = {C_NONE: 'none', C_RLE: 'rle', C_NBIT: 'nbit', C_SKPHUFF: 'skphuff',
               C_DEFLATE: 'deflate', C_SZIP: 'szip'}

# number types (hntdefs.h)
DFNT_LITEND, DFNT_NATIVE = 0x4000, 0x1000
_NT = {3: 'u1', 4: 'S1', 5: 'f4', 6: 'f8', 20: 'i1', 21: 'u1', 22: 'i2', 23: 'u2',
       24: 'i4', 25: 'u4', 26: 'i8', 27: 'u8'}
_OLD_NT = {1: 4, 2: 22, 3: 5, 4: 24, 5: 20, 6: 22, 7: 6}    # Vdata versions <= 2
# mfhdf's default fill (NC_arrayfill) by the netCDF-2 type each HDF type maps to
_FILL = {3: b'\0', 4: b'\0', 20: b'\x81', 21: b'\x81', 22: b'\x80\x01', 23: b'\x80\x01',
         24: b'\x80\x00\x00\x01', 25: b'\x80\x00\x00\x01',
         5: struct.pack('>f', 9.9692099683868690e+36),
         6: struct.pack('>d', 9.9692099683868690e+36)}

PSEUDO_CHUNK = 1 << 17          # contiguous data is read in pieces this big, on the pool
SMALL_READ = 1 << 16            # reads smaller than this skip the pool (numpy, or one thread)


class FormatError(ValueError):
    """The bytes are not HDF4 as this reader understands it."""


_u16 = struct.Struct('>H').unpack_from
_i16 = struct.Struct('>h').unpack_from
_i32 = struct.Struct('>i').unpack_from
_u32 = struct.Struct('>I').unpack_from


def is_hdf4(head):
    return bytes(head[:4]) == MAGIC


def nt_dtype(nt):
    """The numpy dtype of an HDF4 number type as stored (byte order included)."""
    base = nt & 0xff
    code = _NT.get(base)
    if code is None:
        raise FormatError('number type %d is not supported' % nt)
    if code[0] == 'S' or code[1] == '1':
        return np.dtype(code)
    order = '<' if nt & (DFNT_LITEND | DFNT_NATIVE) else '>'
    return np.dtype(order + code)


def _name(raw):
    return raw.split(b'\0', 1)[0].decode('utf-8', 'replace')


class Elem:
    """How a data element's bytes are kept. kind: 'plain' / 'linked' (pieces:
    [(file offset, length)] making up the bytes in order, length: the
    element's logical length), 'ext' (path, offset), 'comp' (coder, params,
    data: the Elem holding the compressed bytes), 'chunked' (see Chunked),
    'none' (no such element)."""
    __slots__ = ('kind', 'length', 'pieces', 'path', 'offset', 'coder', 'params', 'data', 'chunked')

    def __init__(self, kind, length=0):
        self.kind, self.length = kind, length
        self.pieces = self.path = self.offset = self.coder = self.params = self.data = None
        self.chunked = None


class Chunked:
    """A chunked element's header and chunk table."""
    __slots__ = ('dims', 'chunks', 'nt_size', 'fill', 'coder', 'params', 'table', 'length')


class VGroup:
    __slots__ = ('ref', 'name', 'cls', 'tags', 'refs', 'version')


class VData:
    __slots__ = ('ref', 'name', 'cls', 'interlace', 'nrec', 'ivsize', 'fields', 'version')


class H4File:
    """The HDF4 structure of a file mapped (or held) in `buf`."""

    def __init__(self, buf, path=None):
        self.b = buf
        self.n = len(buf)
        self.path = path
        if self.n < 4 or bytes(buf[:4]) != MAGIC:
            raise FormatError('not an HDF4 file (no magic number)')
        self.dd = {}
        self._vg = {}
        self._vh = {}
        self._read_dds()

    # -- data descriptors
    def _read_dds(self):
        b, n = self.b, self.n
        dt = np.dtype([('tag', '>u2'), ('ref', '>u2'), ('off', '>i4'), ('len', '>i4')])
        pos = 4
        seen = set()
        dd = self.dd
        while pos:
            if pos in seen or pos < 4 or pos + 6 > n:
                if pos in seen:
                    break
                raise FormatError('DD block at %d is outside the file' % pos)
            seen.add(pos)
            ndds = _u16(b, pos)[0]
            nxt = _i32(b, pos + 2)[0]
            if pos + 6 + 12 * ndds > n:
                raise FormatError('DD block at %d runs past the end of the file' % pos)
            a = np.frombuffer(b, dt, ndds, pos + 6)
            tags = a['tag'].tolist()
            refs = a['ref'].tolist()
            offs = a['off'].tolist()
            lens = a['len'].tolist()
            for t, r, o, ln in zip(tags, refs, offs, lens):
                if t == T_NULL or t == 0:
                    continue
                dd[(t, r)] = (o, ln)
            pos = nxt
            if pos < 0:
                break

    def find(self, tag, ref):
        """(special, offset, length) of element (tag, ref), or None."""
        e = self.dd.get((tag, ref))
        if e is not None:
            return False, e[0], e[1]
        if not tag & 0x8000:
            e = self.dd.get((tag | SPECIAL, ref))
            if e is not None:
                return True, e[0], e[1]
        return None

    def _span(self, off, ln, what='element'):
        if off < 0 or ln < 0 or off + ln > self.n:
            raise FormatError('%s at %d (+%d) is outside the file' % (what, off, ln))
        return off, ln

    def element(self, tag, ref):
        """The Elem describing data element (tag, ref)."""
        f = self.find(tag, ref)
        if f is None:
            return Elem('none', -1)
        special, off, ln = f
        if not special:
            if ln == 0 or off == -1:
                e = Elem('plain', 0)
                e.pieces = []
                return e
            self._span(off, ln)
            e = Elem('plain', ln)
            e.pieces = [(off, ln)]
            return e
        self._span(off, ln, 'special header')
        if ln < 2:
            raise FormatError('special element (%d, %d) has no header' % (tag, ref))
        b = self.b
        code = _u16(b, off)[0]
        if code == S_LINKED:
            return self._linked(off, ln)
        if code == S_EXT:
            if ln < 14:
                raise FormatError('external element header too short')
            length, eoff, nlen = _i32(b, off + 2)[0], _i32(b, off + 6)[0], _i32(b, off + 10)[0]
            if nlen < 0 or 14 + nlen > ln:
                raise FormatError('external element header too short')
            e = Elem('ext', length)
            e.path = bytes(b[off + 14:off + 14 + nlen]).split(b'\0', 1)[0].decode('utf-8', 'replace')
            e.offset = eoff
            return e
        if code == S_COMP:
            if ln < 14:
                raise FormatError('compressed element header too short')
            length = _i32(b, off + 4)[0]
            cref = _u16(b, off + 8)[0]
            coder, params = self._coder(off + 10, off + ln)
            e = Elem('comp', length)
            e.coder, e.params = coder, params
            e.data = self.element(T_COMPRESSED, cref)
            if e.data.kind == 'comp' or e.data.kind == 'chunked':
                raise FormatError('compressed data element is itself compressed')
            return e
        if code == S_CHUNKED:
            return self._chunked(off, ln)
        raise FormatError('special element code %d (tag %d ref %d) is not supported' % (code, tag, ref))

    def _coder(self, p, end):
        """(coder, params) of a compression header at p (model, coder, info)."""
        b = self.b
        if p + 4 > end:
            raise FormatError('compression header too short')
        coder = _u16(b, p + 2)[0]
        p += 4
        params = ()
        if coder == C_NBIT:
            if p + 16 > end:
                raise FormatError('N-bit header too short')
            nt = _i32(b, p)[0]
            params = (nt, _u16(b, p + 4)[0], _u16(b, p + 6)[0], _i32(b, p + 8)[0], _i32(b, p + 12)[0])
            # (nt, sign_ext, fill_one, start_bit, bit_len)
        elif coder == C_SKPHUFF:
            if p + 8 > end:
                raise FormatError('skipping Huffman header too short')
            params = (_u32(b, p)[0],)
        elif coder == C_DEFLATE:
            params = (_u16(b, p)[0],) if p + 2 <= end else (0,)
        elif coder == C_SZIP:
            if p + 14 > end:
                raise FormatError('szip header too short')
            params = (_u32(b, p)[0], _u32(b, p + 4)[0], _u32(b, p + 8)[0], b[p + 12], b[p + 13])
            # (pixels, pixels_per_scanline, options_mask, bits_per_pixel, pixels_per_block)
        elif coder not in (C_NONE, C_RLE):
            raise FormatError('compression coder %d is not supported' % coder)
        return coder, params

    def _linked(self, off, ln):
        b = self.b
        if ln < 16:
            raise FormatError('linked-block header too short')
        length, blen, nblocks, lref = (_i32(b, off + 2)[0], _i32(b, off + 6)[0],
                                       _i32(b, off + 10)[0], _u16(b, off + 14)[0])
        if length < 0 or blen <= 0 or nblocks <= 0:
            raise FormatError('bad linked-block header')
        e = Elem('linked', length)
        pieces = []
        got = 0
        first = True
        seen = set()
        while lref and got < length:
            if lref in seen:
                raise FormatError('linked-block table loops')
            seen.add(lref)
            t = self.find(T_LINKED, lref)
            if t is None or t[0]:
                raise FormatError('linked-block table %d missing' % lref)
            toff, tln = self._span(t[1], t[2], 'linked-block table')
            if tln < 2 + 2 * nblocks:
                raise FormatError('linked-block table too short')
            nxt = _u16(b, toff)[0]
            brefs = np.frombuffer(b, '>u2', nblocks, toff + 2).tolist()
            for br in brefs:
                if got >= length:
                    break
                if first:
                    want = blen
                    if br:
                        blk = self.find(T_LINKED, br)
                        if blk is not None:
                            want = blk[2]
                    first = False
                else:
                    want = blen
                want = min(want, length - got)
                if br:
                    blk = self.find(T_LINKED, br)
                    if blk is None or blk[0]:
                        raise FormatError('linked block %d missing' % br)
                    boff, bln = self._span(blk[1], blk[2], 'linked block')
                    if bln < want:
                        raise FormatError('linked block %d too short' % br)
                    pieces.append((boff, want))
                else:
                    pieces.append((-1, want))           # never written: zeros
                got += want
            lref = nxt
        if got < length:
            pieces.append((-1, length - got))
        e.pieces = pieces
        return e

    def _chunked(self, off, ln):
        b = self.b
        end = off + ln
        if ln < 6:
            raise FormatError('chunked header too short')
        hlen = _i32(b, off + 2)[0]
        p = off + 6
        if hlen < 29 or p + hlen > end:
            raise FormatError('bad chunked header length')
        c = Chunked()
        flag = _i32(b, p + 1)[0]
        c.length = _i32(b, p + 5)[0]
        c.nt_size = _i32(b, p + 13)[0]
        tbl_ref = _u16(b, p + 19)[0]
        ndims = _i32(b, p + 25)[0]
        q = p + 29
        if ndims < 1 or ndims > 32 or q + 12 * ndims + 4 > p + hlen:
            raise FormatError('bad chunked header dimensions')
        dims, chunks = [], []
        for j in range(ndims):
            dims.append(_i32(b, q + 4)[0])
            chunks.append(_i32(b, q + 8)[0])
            q += 12
        if any(cl <= 0 for cl in chunks) or any(d < 0 for d in dims) or c.nt_size <= 0:
            raise FormatError('bad chunk dimensions')
        flen = _i32(b, q)[0]
        q += 4
        if flen < 0 or q + flen > p + hlen:
            raise FormatError('bad chunk fill value')
        c.fill = bytes(b[q:q + flen])
        c.dims, c.chunks = tuple(dims), tuple(chunks)
        c.coder, c.params = C_NONE, ()
        q = p + hlen
        if flag & 0xff == S_COMP:
            if q + 6 > end:
                raise FormatError('chunked compression header missing')
            clen = _i32(b, q + 2)[0]
            if _u16(b, q)[0] != S_COMP or clen < 4 or q + 6 + clen > end:
                raise FormatError('bad chunked compression header')
            c.coder, c.params = self._coder(q + 6, q + 6 + clen)
        # the chunk table: a Vdata of (origin[ndims] int32, tag uint16, ref uint16)
        vd = self.vdata(tbl_ref)
        table = []
        if vd.nrec > 0:
            raw = self.read(T_VS, tbl_ref)
            fields = {f[0]: f for f in vd.fields}
            need = ('origin', 'chk_tag', 'chk_ref')
            if any(k not in fields for k in need) or fields['origin'][4] != ndims:
                raise FormatError('bad chunk table')
            fo, ft, fr = (fields[k][3] for k in need)
            rs = vd.ivsize
            if vd.interlace != 0 or rs * vd.nrec > len(raw):
                raise FormatError('chunk table too short')
            for i in range(vd.nrec):
                r0 = i * rs
                origin = struct.unpack_from('>%di' % ndims, raw, r0 + fo)
                table.append((origin, _u16(raw, r0 + ft)[0], _u16(raw, r0 + fr)[0]))
        c.table = table
        e = Elem('chunked', c.length)
        e.chunked = c
        return e

    def length(self, tag, ref):
        """The element's logical length (Hlength), -1 if absent."""
        f = self.find(tag, ref)
        if f is None:
            return -1
        if not f[0]:
            return f[2]
        e = self.element(tag, ref)
        if e.kind == 'chunked':
            c = e.chunked
            return int(np.prod(c.dims)) * c.nt_size
        return e.length

    def read(self, tag, ref, upto=None):
        """The element's bytes (any kind but chunked), up to `upto` of them."""
        return self.read_elem(self.element(tag, ref), upto)

    def read_elem(self, e, upto=None):
        n = e.length if upto is None else min(upto, e.length)
        if e.kind == 'none':
            raise FormatError('element missing')
        if n <= 0:
            return b''
        if e.kind in ('plain', 'linked'):
            return self._gather(e.pieces, n)
        if e.kind == 'ext':
            return read_external(self.path, e.path, e.offset, n)
        if e.kind == 'comp':
            out = np.zeros(n, np.uint8)
            decode_into(self, e, out)
            return out.tobytes()
        raise FormatError('cannot read a chunked element as bytes')

    def _gather(self, pieces, n):
        if len(pieces) == 1 and pieces[0][0] >= 0:
            off = pieces[0][0]
            return bytes(self.b[off:off + n])
        out = bytearray()
        for off, ln in pieces:
            if len(out) >= n:
                break
            out += bytes(ln) if off < 0 else bytes(self.b[off:off + ln])
        return bytes(out[:n])

    # -- Vgroups and Vdatas
    def vgroup(self, ref):
        v = self._vg.get(ref)
        if v is not None:
            return v
        raw = self.read(T_VG, ref)
        n = len(raw)
        if n < 9:
            raise FormatError('Vgroup %d too short' % ref)
        v = VGroup()
        v.ref = ref
        v.version = _u16(raw, n - 5)[0]
        if v.version > 4:
            raise FormatError('Vgroup %d: version %d' % (ref, v.version))
        nv = _u16(raw, 0)[0]
        p = 2
        if p + 4 * nv + 4 > n:
            raise FormatError('Vgroup %d too short' % ref)
        v.tags = list(struct.unpack_from('>%dH' % nv, raw, p))
        p += 2 * nv
        v.refs = list(struct.unpack_from('>%dH' % nv, raw, p))
        p += 2 * nv
        ln = _u16(raw, p)[0]
        v.name = _name(raw[p + 2:p + 2 + ln])
        p += 2 + ln
        if p + 2 > n:
            raise FormatError('Vgroup %d too short' % ref)
        ln = _u16(raw, p)[0]
        v.cls = _name(raw[p + 2:p + 2 + ln])
        self._vg[ref] = v
        return v

    def vdata(self, ref):
        v = self._vh.get(ref)
        if v is not None:
            return v
        raw = self.read(T_VH, ref)
        n = len(raw)
        if n < 15:
            raise FormatError('Vdata %d header too short' % ref)
        v = VData()
        v.ref = ref
        v.version = _u16(raw, n - 5)[0]
        if v.version > 4:
            raise FormatError('Vdata %d: version %d' % (ref, v.version))
        v.interlace = _i16(raw, 0)[0]
        v.nrec = _i32(raw, 2)[0]
        v.ivsize = _u16(raw, 6)[0]
        nf = _i16(raw, 8)[0]
        p = 10
        if nf < 0 or p + 8 * nf > n:
            raise FormatError('Vdata %d header too short' % ref)
        types = struct.unpack_from('>%dh' % nf, raw, p)
        isize = struct.unpack_from('>%dH' % nf, raw, p + 2 * nf)
        offs = struct.unpack_from('>%dH' % nf, raw, p + 4 * nf)
        order = struct.unpack_from('>%dH' % nf, raw, p + 6 * nf)
        p += 8 * nf
        names = []
        for _ in range(nf):
            ln = _i16(raw, p)[0]
            names.append(_name(raw[p + 2:p + 2 + ln]))
            p += 2 + max(ln, 0)
        if p + 4 > n:
            raise FormatError('Vdata %d header too short' % ref)
        ln = _i16(raw, p)[0]
        v.name = _name(raw[p + 2:p + 2 + ln])
        p += 2 + max(ln, 0)
        ln = _i16(raw, p)[0]
        v.cls = _name(raw[p + 2:p + 2 + ln])
        if v.version <= 2:
            types = [_OLD_NT.get(t, t) for t in types]
        v.fields = [(names[i], types[i], isize[i], offs[i], order[i]) for i in range(nf)]
        self._vh[ref] = v
        return v

    def vdata_field(self, v, k=0, nrec=None):
        """Field k of Vdata v: (number type, order, the field's stored bytes
        in its first nrec records, one record after the other)."""
        name, t, isize, off, order = v.fields[k]
        nrec = v.nrec if nrec is None else min(nrec, v.nrec)
        if nrec <= 0 or isize == 0:
            return t, order, b''
        raw = self.read(T_VS, v.ref, upto=v.ivsize * v.nrec)
        if v.interlace == 0:                        # full interlace: records of every field
            rs = v.ivsize
            if len(raw) < rs * nrec or off + isize > rs:
                raise FormatError('Vdata %d records too short' % v.ref)
            if len(v.fields) == 1 and off == 0 and isize == rs:
                return t, order, raw[:rs * nrec]
            return t, order, b''.join(raw[i * rs + off:i * rs + off + isize] for i in range(nrec))
        start = sum(f[2] for f in v.fields[:k]) * v.nrec    # no interlace: field after field
        if len(raw) < start + isize * nrec:
            raise FormatError('Vdata %d records too short' % v.ref)
        return t, order, raw[start:start + isize * nrec]

    def vgroups(self):
        """Every Vgroup ref, ascending (as Vgetid walks them)."""
        return sorted(r for (t, r) in self.dd if t == T_VG)


def read_external(base, name, offset, n):
    """n bytes at offset of an external file, found as HDF4 would: the name
    as given, else next to the HDF file."""
    cands = [name]
    if base and not os.path.isabs(name):
        cands.insert(0, os.path.join(os.path.dirname(os.path.abspath(base)), name))
    for c in cands:
        if os.path.exists(c):
            with open(c, 'rb') as fh:
                fh.seek(offset)
                data = fh.read(n)
            if len(data) < n:
                raise FormatError('external file %s too short' % c)
            return data
    raise FileNotFoundError('external data file %r not found' % name)


# ---------------------------------------------------------------- decoding

def _codec_cd(coder, params, esize, nbytes):
    if coder == C_NBIT:
        nt, sign_ext, fill_one, start_bit, bit_len = params
        size = nt_dtype(nt).itemsize
        return np.array([esize, nbytes, 5, size, start_bit, bit_len, sign_ext, fill_one], np.int64)
    if coder == C_SKPHUFF:
        return np.array([esize, nbytes, 1, params[0]], np.int64)
    return np.array([esize, nbytes, 0], np.int64)


def _stream(f, data_elem):
    """(array, start, end) holding a compressed element's bytes: the mapped
    file itself when they are in one piece, else a gathered copy."""
    e = data_elem
    if e.kind == 'plain' and e.pieces:
        off, ln = e.pieces[0]
        return f.src, off, off + ln
    if e.kind == 'plain':
        return np.zeros(0, np.uint8), 0, 0
    if e.kind == 'linked':
        raw = np.frombuffer(f._gather(e.pieces, e.length), np.uint8)
        return raw, 0, raw.size
    if e.kind == 'ext':
        raw = np.frombuffer(read_external(f.path, e.path, e.offset, e.length), np.uint8)
        return raw, 0, raw.size
    raise FormatError('compressed data stored as %s' % e.kind)


def decode_stream(coder, params, src, p, pend, out, esize=1):
    """Decode a compressed stream src[p:pend] into all of out (uint8);
    returns the bytes written (< 0: error)."""
    n = out.size
    if n == 0:
        return 0
    if coder == C_NONE:
        k = min(n, pend - p)
        out[:k] = src[p:p + k]
        return k
    if coder == C_DEFLATE:
        from . import _inflate
        tab, iws = _inflate.workspace()
        return _inflate.zlib_inflate(src, p, pend, out, 0, n, n, tab, iws)
    from . import _hdf4_codecs as hc
    cd = _codec_cd(coder, params, esize, n)
    ws = np.zeros(1, np.uint8)
    if coder == C_RLE:
        return hc.rle_decode(src, p, pend, out, 0, n, cd, ws)
    if coder == C_NBIT:
        return hc.nbit_decode(src, p, pend, out, 0, n, cd, ws)
    if coder == C_SKPHUFF:
        return hc.skphuff_decode(src, p, pend, out, 0, n, cd, ws)
    if coder == C_SZIP:
        return _szip_decode(params, src, p, pend, out)
    raise NotImplementedError('HDF4 compression %s is not supported' % CODER_NAMES.get(coder, coder))


def _szip_decode(params, src, p, pend, out):
    """szip as HDF4 keeps it: a flag byte (1: stored as is), the number of
    good bytes (int32), then the szip stream."""
    if pend - p < 5:
        return -3
    good = int(_i32(bytes(src[p + 1:p + 5]), 0)[0])
    if src[p] == 1:
        k = min(out.size, good, pend - p - 5)
        out[:k] = src[p + 5:p + 5 + k]
        return k
    try:
        from . import _szip
    except ImportError:
        raise NotImplementedError('szip-compressed HDF4 data needs an szip decoder, '
                                  'which fastnc does not have yet') from None
    pixels, ppsl, mask, bpp, ppb = params
    cd = np.array([0, out.size, 4, mask & ~(1 << 16), ppb, ppsl, bpp], np.int64)
    ws = np.zeros(max(1, _szip.workspace_size(out.size)), np.uint8)
    return _szip.decode(src, p + 5, p + 5 + min(good, pend - p - 5), out, 0, out.size, cd, ws)


def decode_into(f, e, out, esize=1):
    """A compressed element's decoded bytes into out (uint8, its first bytes)."""
    src, p, pend = _stream(f, e.data)
    r = decode_stream(e.coder, e.params, src, p, pend, out, esize)
    if r < 0:
        raise IOError('HDF4 %s data could not be decoded (error %d)'
                      % (CODER_NAMES.get(e.coder, e.coder), r))
    return r


# ---------------------------------------------------------------- SD model

class Dim:
    __slots__ = ('name', 'size')

    def __init__(self, name, size):
        self.name, self.size = name, size


class _Filter:
    __slots__ = ('id', 'cd')

    def __init__(self, fid):
        self.id, self.cd = fid, ()


class _Type:
    __slots__ = ('size',)

    def __init__(self, size):
        self.size = size


class _DS:
    """What `_chunks` needs of a dataset (the fill value asked of the layout
    only when needed: it can mean reading the attributes)."""
    __slots__ = ('name', 'shape', 'filters', 'lay', 'type')

    def __init__(self, name, shape, filters, lay, esize):
        self.name, self.shape, self.filters, self.lay = name, tuple(shape), filters, lay
        self.type = _Type(esize)

    @property
    def fill(self):
        return self.lay.fill


def _attr_value(f, vd, name):
    """An Attr0.0 Vdata's value as netCDF-c gives it and netCDF4-python
    returns it."""
    if not vd.fields:
        return None
    t, order, raw = f.vdata_field(vd, 0)
    base = t & 0xff
    if base in (3, 4):                              # char / uchar: `order` bytes
        data = raw[:order]
        if base == 3:
            a = np.frombuffer(data, np.uint8).copy()
            return a[0] if a.size == 1 else a
        if name == '_FillValue':
            return bytes(data)
        return data.decode('utf-8', 'replace').replace('\x00', '')
    dt = nt_dtype(t)
    count = vd.nrec                                 # one value a record (order 1)
    a = np.frombuffer(raw, dt, min(count, len(raw) // dt.itemsize)).astype(dt.newbyteorder('='))
    if a.size == 1:
        return a[0]
    return a


def _attrs(f, vg):
    """The Attr0.0 Vdatas of a Vgroup, in order: name -> value."""
    out = {}
    for t, r in zip(vg.tags, vg.refs):
        if t != T_VH:
            continue
        vd = f.vdata(r)
        if vd.cls != 'Attr0.0':
            continue
        v = _attr_value(f, vd, vd.name)
        if v is not None:
            out[vd.name] = v
    return out


def _guard(fn):
    """fn, with a truncated or garbled structure (struct / index errors deep in
    the parsing) reported as FormatError."""
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (struct.error, IndexError, OverflowError) as ex:
            raise FormatError('corrupt HDF4 file: %s' % ex) from None
    return wrapped


class Variable:
    """An SD variable (backend contract, docs/contracts.md section 2)."""

    def __init__(self, nc, name, hdftype, dimnames, sdshape, data_ref, attrs, data_tag=T_SD, data_offset=0):
        self.nc = nc
        self.name = name
        self.hdftype = hdftype
        self.stored = nt_dtype(hdftype)
        self.dtype = self.stored.newbyteorder('=') if self.stored.byteorder == '>' else self.stored
        self.dimensions = tuple(dimnames)
        self.sdshape = tuple(sdshape)               # the SD interface's shape (its own records)
        self.shape = self.sdshape
        self.data_ref = data_ref
        self.data_tag = data_tag                    # (old SDG files: scales live in tag 703)
        self.data_offset = data_offset              # bytes into that element
        self._attrs = attrs                         # dict, or the Vgroup to read them from
        self._elem = None
        self._layout = None
        self.filters = {'zlib': False, 'szip': False, 'zstd': False, 'bzip2': False,
                        'blosc': False, 'shuffle': False, 'complevel': 0, 'fletcher32': False}

    @property
    @_guard
    def attrs(self):
        a = self._attrs
        if not isinstance(a, dict):
            a = self._attrs = _attrs(self.nc.f, a)
        return a

    @property
    def endian(self):
        return 'little' if self.hdftype & DFNT_LITEND else 'big'

    @property
    @_guard
    def elem(self):
        if self._elem is None:
            self._elem = (self.nc.f.element(self.data_tag, self.data_ref) if self.data_ref
                          else Elem('none', -1))
        return self._elem

    @property
    def chunks(self):
        e = self.elem
        if e.kind == 'chunked':
            return tuple(e.chunked.chunks)
        return None

    @property
    def compression(self):
        """The HDF4 coder of the data ('deflate', 'rle', ...) or None."""
        e = self.elem
        coder = e.coder if e.kind == 'comp' else e.chunked.coder if e.kind == 'chunked' else C_NONE
        return None if coder == C_NONE else CODER_NAMES.get(coder, str(coder))

    # -- fill values
    def fill_bytes(self):
        """The value SDreaddata gives where no data was written: the
        _FillValue attribute, else mfhdf's default for the type (native order)."""
        esize = self.dtype.itemsize
        fv = self.attrs.get('_FillValue')
        if fv is not None:
            if isinstance(fv, bytes):
                raw = fv
            else:
                raw = np.asarray(fv).tobytes()
            if len(raw) >= esize:
                return raw[:esize]
        d = _FILL.get(self.hdftype & 0xff)
        if d is None:
            return bytes(esize)
        return np.frombuffer(d, self.stored.newbyteorder('>') if self.stored.byteorder != '|'
                             else self.stored).astype(self.dtype).tobytes()

    def _chunk_fill(self, c):
        esize = self.dtype.itemsize
        if len(c.fill) >= esize:
            return np.frombuffer(c.fill[:esize], self.stored).astype(self.dtype).tobytes()
        return self.fill_bytes()

    # -- reading
    @_guard
    def read(self, key=(), out=None, threads=None):
        box, post, drop = _select.normalize(key, self.shape)
        shape = _select.box_shape(box)
        nat = self.dtype
        direct_out = out is not None and post is None and not drop and out.shape == shape \
            and out.dtype == nat and out.flags.c_contiguous
        res = out if direct_out else np.empty(shape, nat)
        if res.size:
            self._read_into(box, res, threads)
        res = _select.finish(res, post, drop)
        if out is not None and not direct_out:
            out[...] = res
            return out
        return res

    def _read_into(self, box, res, threads=None):
        lay = self.layout()
        if lay.kind == 'fill':
            _fill_with(res, lay.fill)
            return
        # the part of the box inside the stored extent along the first dimension
        avail = lay.ds.shape[0] if lay.ds.shape else 0
        s, e, k = box[0]
        m = 0 if e <= s or s >= avail else (min(e, avail) - s + k - 1) // k
        if m < res.shape[0]:
            _fill_with(res[m:], lay.fill)
        if m == 0:
            return
        sub = res[:m] if m < res.shape[0] else res
        sbox = [(s, min(e, s + (m - 1) * k + 1), k)] + list(box[1:])
        if lay.kind == 'decode':
            src, index = self._predecode(lay, sbox, threads)
        else:
            src, index = lay.src, lay.index
        if src is None:
            _fill_with(sub, lay.fill)
            return
        if lay.base is not None and (sub.nbytes < SMALL_READ or not _chunks.ready()):
            # contiguous, kernels not loaded yet: numpy
            whole = np.ndarray(lay.ds.shape, self.stored, buffer=src, offset=lay.base)
            sub[...] = whole[tuple(slice(a, b, c) for a, b, c in sbox)]
            return
        _chunks.read(lay.ds, index, src, sbox, sub, swap=lay.swap, threads=threads)

    def layout(self):
        if self._layout is None:
            self._layout = _layout(self)
        return self._layout

    def _predecode(self, lay, box, threads):
        """Decode the chunks the box touches (coders the chunk engine does not
        run, or bytes not in one piece) into one buffer, on the pool; the
        index of that buffer's chunks (no filters left)."""
        idx = lay.index
        p = _chunks.plan(idx, lay.ds.shape, box)
        rows = p.addr                               # (index rows: addr holds the row number)
        if len(rows) == 0:
            return None, None
        cbytes = int(np.prod(idx.chunks)) * lay.esize
        buf = np.empty(len(rows) * cbytes, np.uint8)
        f = self.nc.f
        jobs = list(enumerate(rows.tolist()))
        errs = []

        def one(job):
            i, row = job
            el = lay.elems[row]
            o = buf[i * cbytes:(i + 1) * cbytes]
            try:
                if el.kind == 'comp':
                    n = decode_into(f, el, o, lay.esize)
                else:
                    raw = np.frombuffer(f.read_elem(el, cbytes), np.uint8)
                    n = raw.size
                    o[:n] = raw
                if n < cbytes:
                    o[n:] = np.frombuffer(lay.raw_fill * (cbytes - n), np.uint8)[:cbytes - n] \
                        if lay.raw_fill else 0
            except Exception as ex:                 # (raised on the caller below)
                errs.append(ex)
        if len(jobs) == 1 or (threads or _pool.threads()) == 1:
            for j in jobs:
                one(j)
        else:
            _pool.map_items(one, jobs)
        if errs:
            raise errs[0]
        table = [(tuple(idx.offsets[row]), i * cbytes, cbytes, 0) for i, row in enumerate(rows.tolist())]
        index = _chunks.ChunkIndex(table, tuple(int(c) for c in idx.chunks), lay.ds.shape)
        index.complete = idx.complete
        return buf, index


def _fill_with(a, fill):
    if a.size == 0:
        return
    esize = a.dtype.itemsize
    if len(fill) == esize:
        a.reshape(-1).view(np.uint8).reshape(-1, esize)[:] = np.frombuffer(fill, np.uint8)
    else:
        a.reshape(-1).view(np.uint8)[:] = 0


class Layout:
    """How to read a variable: kind 'fill' (no data), 'file' (the chunk
    engine over `src` with `index`) or 'decode' (chunks decoded first)."""
    __slots__ = ('kind', 'ds', 'index', 'src', 'swap', '_fill', 'var', 'esize', 'elems', 'raw_fill', 'base')

    @property
    def fill(self):
        if self._fill is None:
            self._fill = self.var.fill_bytes()
        return self._fill


def _layout(v):
    f = v.nc.f
    e = v.elem
    lay = Layout()
    lay.esize = esize = v.dtype.itemsize
    lay.swap = v.stored.byteorder == '>' and esize > 1
    lay.var = v
    lay._fill = None
    lay.src = f.src
    lay.elems = None
    lay.raw_fill = None
    lay.base = None                                 # contiguous: where the array starts in src
    shape = v.sdshape
    rank = len(shape)
    nel = int(np.prod(shape)) if rank else 1
    rowel = int(np.prod(shape[1:])) if rank > 1 else 1
    if e.kind == 'none' or e.length == 0 or nel == 0 or rank == 0:
        lay.kind = 'fill'
        return lay
    if e.kind == 'chunked':
        c = e.chunked
        if len(c.chunks) != rank or c.nt_size != esize:
            raise FormatError('%s: chunk header does not match the variable' % v.name)
        lay._fill = v._chunk_fill(c)
        lay.raw_fill = c.fill[:esize] if len(c.fill) >= esize else None
        cbytes = int(np.prod(c.chunks)) * esize
        grid = [-(-s // cl) for s, cl in zip(shape, c.chunks)]
        rows = []
        elems = []
        direct = c.coder in (C_NONE, C_DEFLATE)
        for origin, tag, ref in c.table:
            if len(origin) != rank or any(o < 0 or o >= g for o, g in zip(origin, grid)):
                continue                            # outside the array: never read
            ce = f.element(tag, ref)
            if ce.kind == 'none':
                continue
            offs = tuple(o * cl for o, cl in zip(origin, c.chunks))
            if c.coder == C_NONE and ce.kind == 'comp':
                direct = False
            if ce.kind == 'comp' and ce.coder != c.coder:
                direct = False
            if ce.kind == 'comp':
                dk = ce.data
            else:
                dk = ce
            if dk.kind != 'plain' or not dk.pieces:
                direct = False
            if c.coder == C_NONE and ce.kind == 'plain' and ce.length < cbytes:
                direct = False                      # a short chunk: copy it out first
            elems.append(ce)
            rows.append(offs)
        n = len(rows)
        if direct:
            table = []
            for offs, ce in zip(rows, elems):
                dk = ce.data if ce.kind == 'comp' else ce
                off, ln = dk.pieces[0]
                table.append((offs, off, ln, 0))
            filters = [_Filter(_chunks.F_DEFLATE)] if c.coder == C_DEFLATE else []
            lay.kind = 'file'
        else:
            table = [(offs, i, 0, 0) for i, offs in enumerate(rows)]
            filters = []
            lay.kind = 'decode'
            lay.elems = elems
        lay.ds = _DS(v.name, shape, filters, lay, esize)
        lay.index = _chunks.ChunkIndex(table, c.chunks, shape) if n else _empty_index(c.chunks, shape)
        if n == 0:
            lay.kind = 'fill'
        return lay
    # one stream for the whole array
    rowbytes = rowel * esize
    if e.kind == 'comp':
        length = min(e.length, nel * esize)
        rows_avail = min(length // rowbytes, shape[0])
        if rows_avail <= 0:
            lay.kind = 'fill'
            return lay
        cshape = (rows_avail,) + tuple(shape[1:])
        dk = e.data
        if e.coder == C_DEFLATE and dk.kind == 'plain' and dk.pieces:
            off, ln = dk.pieces[0]
            lay.kind = 'file'
            lay.ds = _DS(v.name, cshape, [_Filter(_chunks.F_DEFLATE)], lay, esize)
            lay.index = _chunks.ChunkIndex([((0,) * rank, off, ln, 0)], cshape, cshape)
            return lay
        lay.kind = 'decode'
        lay.elems = [e]
        lay.ds = _DS(v.name, cshape, [], lay, esize)
        lay.index = _chunks.ChunkIndex([((0,) * rank, 0, 0, 0)], cshape, cshape)
        return lay
    # contiguous: plain (one piece), linked blocks or external
    skip = v.data_offset
    length = min(e.length - skip, nel * esize)
    rows_avail = min(length // rowbytes, shape[0])
    if rows_avail <= 0:
        lay.kind = 'fill'
        return lay
    if e.kind == 'plain':
        base = e.pieces[0][0] + skip
        src = f.src
    else:
        raw = f.read_elem(e, skip + rows_avail * rowbytes)
        src = np.frombuffer(raw, np.uint8)
        base = skip
    cshape = (rows_avail,) + tuple(shape[1:])
    k = max(1, PSEUDO_CHUNK // max(rowbytes, 1))
    k = min(k, rows_avail)
    table = [((r,) + (0,) * (rank - 1), base + r * rowbytes, min(k, rows_avail - r) * rowbytes, 0)
             for r in range(0, rows_avail, k)]
    lay.kind = 'file'
    lay.src = src
    lay.base = base
    lay.ds = _DS(v.name, cshape, [], lay, esize)
    lay.index = _chunks.ChunkIndex(table, (k,) + tuple(shape[1:]), cshape)
    return lay


def _empty_index(chunks, shape):
    return _chunks.ChunkIndex([], chunks, shape)


class Group:
    """The root group (HDF4 has no netCDF groups)."""

    def __init__(self):
        self.name = '/'
        self.dimensions = {}
        self.variables = {}
        self.groups = {}
        self.attrs = {}


class HDF4File:
    """An HDF4 file read through netCDF-c's HDF4 model (backend contract)."""

    data_model = 'NETCDF4'

    def __init__(self, buf, src=None, path=None):
        try:
            self.f = H4File(buf, path)
        except (struct.error, IndexError) as ex:
            raise FormatError('corrupt HDF4 file: %s' % ex) from None
        self.f.src = src if src is not None else np.frombuffer(buf, np.uint8)
        self.root = Group()
        try:
            self._read_sd()
        except (struct.error, IndexError) as ex:
            raise FormatError('corrupt HDF4 file: %s' % ex) from None

    def close(self):
        pass

    def _read_sd(self):
        f = self.f
        cdf = None
        for r in f.vgroups():
            vg = f.vgroup(r)
            if vg.cls == 'CDF0.0':
                cdf = vg
                break
        sdvars = None
        if cdf is not None:
            try:
                sddims = self._dims(cdf)
                sdvars = self._vars(cdf, sddims)
                self.root.attrs = _attrs(f, cdf)
            except _NoDim:
                sdvars = None                       # (mfhdf then reads the file as below)
        if sdvars is None:
            # no SD Vgroup: the data sets of the older DFSD interface (NDGs, SDGs)
            sddims, sdvars = _read_ndgs(self)
        # netCDF-c: dimensions by name, as variables first use them
        dims = self.root.dimensions
        sizes = {}
        for v, dimidx in sdvars:
            rec_len = v.sdshape[0] if v.sdshape else 0
            for di in dimidx:
                d = sddims[di]
                if d.name not in sizes:
                    sizes[d.name] = d.size if d.size else rec_len
                    dims[d.name] = (sizes[d.name], False)
            v.shape = tuple(sizes[sddims[di].name] for di in dimidx)
            self.root.variables[v.name] = v

    def _dims(self, cdf):
        """hdf_read_dims: the file's SD dimensions (size 0: unlimited)."""
        f = self.f
        dims = []
        have = set()
        for t, r in zip(cdf.tags, cdf.refs):
            if t != T_VG:
                continue
            vg = f.vgroup(r)
            if vg.cls not in ('Dim0.0', 'UDim0.0'):
                continue
            unlim = vg.cls == 'UDim0.0'
            for st, sr in zip(vg.tags, vg.refs):
                if st != T_VH:
                    continue
                vd = f.vdata(sr)
                if vd.cls not in ('DimVal0.0', 'DimVal0.1'):
                    continue
                size = None
                if vd.cls == 'DimVal0.0' and not unlim:
                    size = vd.nrec
                if vd.cls == 'DimVal0.1' or unlim:
                    size = 0 if unlim else self._first_int(vd)
                if (vg.name, size) in have:
                    continue
                have.add((vg.name, size))
                dims.append(Dim(vg.name, size))
        return dims

    def _first_int(self, vd):
        t, order, raw = self.f.vdata_field(vd, 0, 1)
        dt = nt_dtype(t)
        if len(raw) < dt.itemsize:
            raise FormatError('dimension %s has no size' % vd.name)
        return int(np.frombuffer(raw, dt, 1)[0])

    def _vars(self, cdf, dims):
        """hdf_read_vars: the SD variables, in order, with their dimension indexes."""
        f = self.f
        out = []
        byname = {}
        for i, d in enumerate(dims):
            byname.setdefault(d.name, i)
        for t, r in zip(cdf.tags, cdf.refs):
            if t != T_VG:
                continue
            vg = f.vgroup(r)
            if vg.cls != 'Var0.0':
                continue
            dimidx = []
            is_rec = False
            hdftype = None
            data_ref = 0
            bad = False
            for st, sr in zip(vg.tags, vg.refs):
                if st == T_VG:
                    sub = f.vgroup(sr)
                    if sub.cls in ('Dim0.0', 'UDim0.0'):
                        if sub.cls == 'UDim0.0':
                            is_rec = True
                        if sub.name not in byname:
                            raise _NoDim(sub.name)
                        dimidx.append(byname[sub.name])
                elif st == T_SD:
                    data_ref = sr
                elif st == T_NT:
                    nt = f.read(T_NT, sr)
                    if len(nt) < 4:
                        raise FormatError('variable %s: bad number type' % vg.name)
                    hdftype = nt[1]
                    if nt[0] != 1 or nt[3] not in (0, 1):
                        if nt[3] == 4:              # DFNTF_PC: little-endian
                            hdftype |= DFNT_LITEND
                        else:
                            bad = True              # native data of another machine: skipped
            if bad:
                continue
            if hdftype is None or (hdftype & 0xff) not in _NT:
                raise FormatError('variable %s: number type %s not supported' % (vg.name, hdftype))
            esize = nt_dtype(hdftype).itemsize
            shape = []
            for k, di in enumerate(dimidx):
                shape.append(dims[di].size)
            numrecs = 0
            if data_ref:
                numrecs = 1
                if is_rec:
                    rec = int(np.prod([dims[di].size for di in dimidx[1:]])) * esize
                    ln = f.length(T_SD, data_ref)
                    numrecs = ln // rec if ln > 0 and rec > 0 else 0
            if shape and shape[0] == 0:
                shape[0] = numrecs
            v = Variable(self, vg.name, hdftype, [dims[di].name for di in dimidx], shape, data_ref, vg)
            out.append((v, dimidx))
        return out


class _NoDim(Exception):
    pass


def _cstr(b, k):
    """The k-th NUL-terminated string of b ('' past the end)."""
    parts = b.split(b'\0')
    return parts[k].decode('utf-8', 'replace') if k < len(parts) else ''


def _num_attr(raw, nt, n=1):
    dt = nt_dtype(nt)
    a = np.frombuffer(raw, dt, n).astype(dt.newbyteorder('='))
    return a[0] if n == 1 else a


def _annotations(f, tag, ref, atag):
    """Texts of the annotations (DIL 104 / DIA 105) of object (tag, ref), in
    annotation ref order."""
    out = []
    for (t, r) in sorted(k for k in f.dd if k[0] == atag):
        raw = f.read(t, r)
        if len(raw) >= 4 and _u16(raw, 0)[0] == tag and _u16(raw, 2)[0] == ref:
            out.append(raw[4:].split(b'\0', 1)[0].decode('utf-8', 'replace'))
    return out


def _nt_flags(nt):
    """hdf_check_nt: the number type (with DFNT_LITEND) of an NT record, or None."""
    if len(nt) < 4:
        return None
    t = nt[1]
    if nt[0] != 1 or nt[3] not in (0, 1):
        if nt[3] == 4:
            return t | DFNT_LITEND
        return None
    return t


def _read_ndgs(nc):
    """hdf_read_ndgs (mfhdf hdfsds.c): every NDG, then every SDG not linked
    from an NDG, as a variable "Data-Set-<ref>" on new dimensions
    fakeDim<n>, each preceded by a coordinate variable "fakeDim<n>" (its
    scale values, if any)."""
    f = nc.f
    dims = []
    out = []
    seen_sdg = set()
    dimcount = 0
    for gtag in (T_NDG, 700):
        for (t, ndg_ref) in list(f.dd):
            if t != gtag or (gtag == 700 and ndg_ref in seen_sdg):
                continue
            raw = f.read(t, ndg_ref)
            members = [struct.unpack_from('>HH', raw, 4 * i) for i in range(len(raw) // 4)]
            hdftype = 5                             # float32 until an SDD says otherwise
            rank = 0
            dimsizes, scaletypes = [], []
            lref = uref = fref = sref = sdref = 0
            attrs = {}
            for mt, mr in members:
                if mt == T_SDD:
                    sdd = f.read(mt, mr)
                    rank = _i16(sdd, 0)[0]
                    if rank <= 0 or len(sdd) < 2 + 4 * rank + 4 * (rank + 1):
                        raise FormatError('Data-Set-%d: bad dimension record' % ndg_ref)
                    dimsizes = list(struct.unpack_from('>%di' % rank, sdd, 2))
                    q = 2 + 4 * rank
                    nts = []
                    for i in range(rank + 1):
                        nt_tag, nt_ref = struct.unpack_from('>HH', sdd, q + 4 * i)
                        nts.append(_nt_flags(f.read(nt_tag, nt_ref)))
                    if any(x is None for x in nts):
                        raise FormatError('Data-Set-%d: number type of another machine' % ndg_ref)
                    hdftype = nts[0]
                    scaletypes = nts[1:]
                elif mt == 704:
                    lref = mr
                elif mt == 705:
                    uref = mr
                elif mt == 706:
                    fref = mr
                elif mt == 708:                     # coordsys
                    c = f.read(mt, mr).split(b'\0', 1)[0]
                    if c:
                        attrs['coordsys'] = c.decode('utf-8', 'replace')
                elif mt == 703:
                    sref = mr
                elif mt == T_SD:
                    sdref = mr
                elif mt == 731:                     # calibration
                    cal = f.read(mt, mr)
                    ct, it, w = (6, 24, 8) if len(cal) == 36 else (5, 22, 4)
                    for k, name in enumerate(('scale_factor', 'scale_factor_err', 'add_offset',
                                              'add_offset_err')):
                        attrs[name] = _num_attr(cal[k * w:(k + 1) * w], ct)
                    attrs['calibrated_nt'] = _num_attr(cal[4 * w:4 * w + (4 if it == 24 else 2)], it)
                elif mt == 707:                     # valid range: max, min
                    rng = f.read(mt, mr)
                    w = nt_dtype(hdftype).itemsize
                    attrs['valid_max'] = _num_attr(rng[:w], hdftype)
                    attrs['valid_min'] = _num_attr(rng[w:2 * w], hdftype)
                elif mt == 710 and gtag == T_NDG:   # link to the SDG of this NDG
                    lnk = f.read(mt, mr)
                    if len(lnk) >= 8:
                        seen_sdg.add(_u16(lnk, 6)[0])
            labels = f.read(704, lref) + b'\0\0\0' if lref else b''
            units = f.read(705, uref) + b'\0\0\0' if uref else b''
            formats = f.read(706, fref) + b'\0\0\0' if fref else b''
            scales = f.read(703, sref) if sref else b''
            scale_offset = rank
            dimidx = []
            for d in range(rank):
                lv, uv, fv = _cstr(labels, d + 1), _cstr(units, d + 1), _cstr(formats, d + 1)
                name = 'fakeDim%d' % dimcount
                dimcount += 1
                dims.append(Dim(name, dimsizes[d]))
                dimidx.append(len(dims) - 1)
                has_scale = len(scales) > d and scales[d] != 0
                if True:                            # (mfhdf tests `new_dim`, which is FAIL, -1:
                                                    # every dimension gets its variable)
                    st = scaletypes[d]
                    if (st & 0xff) not in _NT:
                        raise FormatError('Data-Set-%d: scale type %d' % (ndg_ref, st))
                    dattrs = {}
                    for k, val in (('long_name', lv), ('units', uv), ('format', fv)):
                        if val:
                            dattrs[k] = val
                    off = 0                         # no scale: read from the element's start, as mfhdf
                    if has_scale:
                        off = scale_offset
                        scale_offset += dimsizes[d] * nt_dtype(st).itemsize
                    v = Variable(nc, name, st, [name], [dimsizes[d]], sref, dattrs, 703, off)
                    out.append((v, [len(dims) - 1]))
            for k, txt in enumerate(_annotations(f, gtag, ndg_ref, 105)):
                attrs['remarks-%d' % (k + 1)] = txt
            for k, txt in enumerate(_annotations(f, gtag, ndg_ref, 104)):
                attrs['anno_label-%d' % (k + 1)] = txt
            for k, val in (('long_name', _cstr(labels, 0)), ('units', _cstr(units, 0)),
                           ('format', _cstr(formats, 0))):
                if val:
                    attrs[k] = val
            if (hdftype & 0xff) not in _NT:
                raise FormatError('Data-Set-%d: number type %d' % (ndg_ref, hdftype))
            v = Variable(nc, 'Data-Set-%d' % ndg_ref, hdftype, [dims[i].name for i in dimidx],
                         dimsizes, sdref, attrs)
            out.append((v, dimidx))
    return dims, out


def read_many(variables, threads=None):
    """Every variable of `variables` (backend Variables of one file) whole,
    raw: those the chunk engine reads straight from the file decoded
    together in one pass of the pool (one shared buffer, biggest chunks
    first), so that files of many deflated arrays -- one stream each, which
    no reader can split -- inflate on every core at once."""
    out = [None] * len(variables)
    if variables and _chunks._READY.is_set():
        batch = []
        total = 0
        for i, v in enumerate(variables):
            lay = v.layout()
            if lay.kind != 'file' or lay.src is not v.nc.f.src or tuple(lay.ds.shape) != tuple(v.shape) \
                    or not v.shape:
                continue
            n = int(np.prod(v.shape)) * v.dtype.itemsize
            if n < SMALL_READ:
                continue                            # (quicker on its own than planned in a batch)
            batch.append((i, v, total, n))
            total += (n + 63) // 64 * 64
        if len(batch) > 1:
            buf = np.empty(total, np.uint8)
            items = []
            for i, v, base, n in batch:
                lay = v.layout()
                a = buf[base:base + n].view(v.dtype).reshape(v.shape)
                out[i] = a
                items.append(_chunks.Item(lay.ds, lay.index, [(0, s, 1) for s in v.shape], a,
                                          lay.swap, base))
            _chunks.read_batch(items, variables[0].nc.f.src, buf, threads)
    for i, v in enumerate(variables):
        if out[i] is None:
            out[i] = v.read((), threads=threads)
    return out
