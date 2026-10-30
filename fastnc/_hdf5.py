"""HDF5, read from the file's bytes: the superblock, object headers, groups
(symbol tables, compact links, dense links in a fractal heap), attributes,
datatypes, dataspaces, fill values, filter pipelines and every chunk index
(v1 B-tree, single chunk, implicit, fixed array, extensible array, v2
B-tree). Pure Python over a memoryview of the mapped file: metadata is
small, and parsing it here instead of through libhdf5 makes opening a
file, and listing a dataset's chunks, a matter of microseconds a chunk.

Everything follows the HDF5 File Format Specification version 3.0; the
section names in comments are its. Addresses in the file are relative to
the superblock's base address; `H5File.addr()` makes them absolute.
"""
import struct

import numpy as np

UNDEF = (1 << 64) - 1           # undefined address (all bits set) at 8 bytes


class FormatError(ValueError):
    """The bytes are not HDF5 as this reader understands it."""


_u8 = struct.Struct('<B').unpack_from
_u16 = struct.Struct('<H').unpack_from
_u32 = struct.Struct('<I').unpack_from
_u64 = struct.Struct('<Q').unpack_from


def _uint(b, at, n):
    """An n-byte little-endian unsigned integer (n = 1..8)."""
    if n == 8:
        return _u64(b, at)[0]
    if n == 4:
        return _u32(b, at)[0]
    if n == 2:
        return _u16(b, at)[0]
    if n == 1:
        return b[at]
    return int.from_bytes(b[at:at + n], 'little')


def _enc_size(x):
    """Bytes needed to encode x (H5VM_limit_enc_size)."""
    return (max(int(x), 1).bit_length() - 1) // 8 + 1


def _log2(x):
    return int(x).bit_length() - 1


# ---------------------------------------------------------------- datatypes

class Datatype:
    """A parsed datatype message (IV.A.2.d). `dtype` is the numpy dtype of
    the stored bytes (byte order included), or None for a type numpy cannot
    hold directly (vlen); `cls` is the HDF5 class number."""
    __slots__ = ('cls', 'size', 'dtype', 'order', 'signed', 'base', 'members',
                 'enum', 'vlen_string', 'charset', 'strpad', 'dims', 'ref_kind',
                 'precision', 'offset', 'tag', 'committed', 'fields')

    @property
    def h5py_dtype(self):
        """The dtype h5py gives this type: an enum of FALSE/TRUE over int8 is
        bool, a compound of two equal floats named r and i is complex."""
        if self.cls == self.ENUM and self.size == 1 and self.enum == {'FALSE': 0, 'TRUE': 1}:
            return np.dtype('?')
        if (self.cls == self.COMPOUND and len(self.members) == 2
                and [m[0] for m in self.members] == ['r', 'i']
                and all(m[2].cls == self.FLOAT for m in self.members)
                and self.members[0][2].size == self.members[1][2].size
                and self.members[1][1] == self.members[0][2].size):
            return np.dtype('%sc%d' % (self.members[0][2].order, self.size))
        return self.dtype

    FIXED, FLOAT, TIME, STRING, BITFIELD, OPAQUE, COMPOUND, REFERENCE, ENUM, VLEN, ARRAY = range(11)

    def __init__(self):
        self.base = self.members = self.enum = self.dims = None
        self.dtype = None
        self.vlen_string = False
        self.charset = self.strpad = 0
        self.order = '<'
        self.signed = False
        self.ref_kind = None
        self.precision = self.offset = 0
        self.fields = None
        self.tag = b''
        self.committed = None

    def __repr__(self):
        return '<Datatype class %d size %d %s>' % (self.cls, self.size, self.dtype)


def parse_datatype(b, at):
    """(Datatype, bytes used) at b[at:]."""
    t = Datatype()
    cv = b[at]
    t.cls = cls = cv & 15
    version = cv >> 4
    f0, f1, f2 = b[at + 1], b[at + 2], b[at + 3]
    t.size = size = _u32(b, at + 4)[0]
    p = at + 8
    if cls in (0, 4):                                   # fixed-point, bitfield
        t.order = '>' if f0 & 1 else '<'
        t.signed = bool(f0 & 8)
        t.offset, t.precision = _u16(b, p)[0], _u16(b, p + 2)[0]
        p += 4
        if size in (1, 2, 4, 8):
            t.dtype = np.dtype('%s%s%d' % (t.order, 'i' if t.signed else 'u', size))
        if cls == 4 and size in (1, 2, 4, 8):
            t.dtype = np.dtype('%su%d' % (t.order, size))
    elif cls == 1:                                      # floating point
        if f0 & 64:
            raise FormatError('VAX-order floating point is not supported')
        t.order = '>' if f0 & 1 else '<'
        t.offset, t.precision = _u16(b, p)[0], _u16(b, p + 2)[0]
        # sign position, exponent position/size, mantissa position/size,
        # exponent bias, mantissa normalization (0 none, 1 msb set, 2 implied)
        t.fields = (f1, b[p + 4], b[p + 5], b[p + 6], b[p + 7], _u32(b, p + 8)[0], (f0 >> 4) & 3)
        p += 12
        if size in (2, 4, 8):
            t.dtype = np.dtype('%sf%d' % (t.order, size))
        elif size == 16:
            t.dtype = np.dtype('%sf16' % t.order) if hasattr(np, 'float128') else None
    elif cls == 2:                                      # time
        t.order = '>' if f0 & 1 else '<'
        p += 2
        t.dtype = np.dtype('%si%d' % (t.order, size))
    elif cls == 3:                                      # string
        t.strpad = f0 & 15
        t.charset = f0 >> 4
        t.dtype = np.dtype('S%d' % size)
    elif cls == 5:                                      # opaque
        n = f0
        t.tag = bytes(b[p:p + n]).rstrip(b'\0')
        p += n
        t.dtype = np.dtype('V%d' % size)
    elif cls == 6:                                      # compound
        nm = f0 | (f1 << 8)
        names, formats, offsets = [], [], []
        members = []
        for _ in range(nm):
            e = p
            while b[e]:
                e += 1
            name = bytes(b[p:e]).decode('utf-8', 'replace')
            if version < 3:
                p += ((e - p) // 8 + 1) * 8
            else:
                p = e + 1
            if version == 1:
                off = _u32(b, p)[0]
                ndim = b[p + 4]
                dims = [_u32(b, p + 16 + 4 * i)[0] for i in range(ndim)]
                p += 32
            elif version == 2:
                off = _u32(b, p)[0]
                dims = []
                p += 4
            else:
                n = _enc_size(size) if size else 1
                off = _uint(b, p, n)
                dims = []
                p += n
            mt, used = parse_datatype(b, p)
            p += used
            members.append((name, off, mt))
            md = mt.dtype if mt.dtype is not None else np.dtype('V%d' % mt.size)
            if dims:
                md = np.dtype((md, tuple(dims)))
            names.append(name)
            formats.append(md)
            offsets.append(off)
        t.members = members
        t.dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets,
                            'itemsize': size})
    elif cls == 7:                                      # reference
        t.ref_kind = f0 & 15
        t.dtype = np.dtype('V%d' % size)
    elif cls == 8:                                      # enum
        nm = f0 | (f1 << 8)
        base, used = parse_datatype(b, p)
        p += used
        names = []
        for _ in range(nm):
            e = p
            while b[e]:
                e += 1
            names.append(bytes(b[p:e]).decode('utf-8', 'replace'))
            p = e + 1 if version >= 3 else p + ((e - p) // 8 + 1) * 8
        vals = np.frombuffer(bytes(b[p:p + nm * base.size]), base.dtype).tolist()
        p += nm * base.size
        t.base = base
        t.enum = dict(zip(names, vals))
        t.dtype = base.dtype
        t.order = base.order
    elif cls == 9:                                      # variable length
        t.vlen_string = (f0 & 15) == 1
        t.strpad = f0 >> 4
        t.charset = f1 & 15
        base, used = parse_datatype(b, p)
        p += used
        t.base = base
    elif cls == 10:                                     # array
        ndim = b[p]
        p += 1 if version >= 3 else 4
        dims = [_u32(b, p + 4 * i)[0] for i in range(ndim)]
        p += 4 * ndim
        if version < 3:
            p += 4 * ndim                               # permutation
        base, used = parse_datatype(b, p)
        p += used
        t.base = base
        t.dims = tuple(dims)
        if base.dtype is not None:
            t.dtype = np.dtype((base.dtype, t.dims))
    else:
        raise FormatError('unknown datatype class %d' % cls)
    return t, p - at


# ---------------------------------------------------------------- messages

class Layout:
    """Data layout (IV.A.2.i): `cls` 0 compact, 1 contiguous, 2 chunked."""
    __slots__ = ('cls', 'address', 'size', 'data', 'chunks', 'index', 'index_info',
                 'single_size', 'single_mask', 'flags', 'version')

    def __init__(self):
        self.address = UNDEF
        self.size = 0
        self.data = None
        self.chunks = None
        self.index = None
        self.index_info = ()
        self.single_size = None
        self.single_mask = 0
        self.flags = 0


class Filter:
    __slots__ = ('id', 'name', 'flags', 'cd')

    def __init__(self, fid, name, flags, cd):
        self.id, self.name, self.flags, self.cd = fid, name, flags, tuple(cd)

    @property
    def optional(self):
        return bool(self.flags & 1)

    def __repr__(self):
        return 'Filter(%d, %r, %r)' % (self.id, self.name, self.cd)


def _parse_filters(b, p):
    version = b[p]
    n = b[p + 1]
    p += 8 if version == 1 else 2
    out = []
    for _ in range(n):
        fid = _u16(b, p)[0]
        p += 2
        nlen = 0
        if version == 1 or fid >= 256:
            nlen = _u16(b, p)[0]
            p += 2
        flags, ncd = _u16(b, p)[0], _u16(b, p + 2)[0]
        p += 4
        name = ''
        if nlen:
            name = bytes(b[p:p + nlen]).split(b'\0')[0].decode('ascii', 'replace')
            p += ((nlen + 7) // 8) * 8 if version == 1 else nlen
        cd = [_u32(b, p + 4 * i)[0] for i in range(ncd)]
        p += 4 * ncd
        if version == 1 and ncd % 2:
            p += 4
        out.append(Filter(fid, name, flags, cd))
    return out


def _parse_dataspace(b, p, L):
    """(shape, maxshape) or (None, None) for a null dataspace."""
    version = b[p]
    rank = b[p + 1]
    flags = b[p + 2]
    if version == 1:
        p += 8
        kind = 1 if rank else 0
    else:
        kind = b[p + 3]
        p += 4
    if kind == 2:
        return None, None
    shape = tuple(_uint(b, p + L * i, L) for i in range(rank))
    p += L * rank
    maxshape = shape
    if flags & 1:
        maxshape = tuple(None if _uint(b, p + L * i, L) == (1 << (8 * L)) - 1
                         else _uint(b, p + L * i, L) for i in range(rank))
    return shape, maxshape


class Attribute:
    __slots__ = ('name', 'dtype', 'shape', 'raw', 'corder', 'file')

    def __init__(self, name, dtype, shape, raw, corder, file):
        self.name, self.dtype, self.shape, self.raw = name, dtype, shape, raw
        self.corder, self.file = corder, file

    def value(self):
        """The attribute's value as numpy (strings as bytes / str objects,
        vlen sequences as object arrays), as h5py would give it."""
        return self.file.decode_values(self.dtype, self.shape, self.raw)


# ---------------------------------------------------------------- objects

class ObjectHeader:
    """One object's messages (type, flags, bytes, creation order), its
    continuation blocks followed."""
    __slots__ = ('address', 'messages', 'flags', 'version')

    def __init__(self, address, messages, flags, version):
        self.address, self.messages, self.flags, self.version = address, messages, flags, version

    def first(self, mtype):
        for m in self.messages:
            if m[0] == mtype:
                return m
        return None

    def all(self, mtype):
        return [m for m in self.messages if m[0] == mtype]


class Group:
    """A group: its links (name -> object header address) in the order HDF5
    iterates them for netCDF (creation order when tracked, else name)."""

    def __init__(self, file, address, name, header=None):
        self.file, self.address, self.name = file, address, name
        self._header = header
        self._links = None

    @property
    def header(self):
        if self._header is None:
            self._header = self.file.object_header(self.address)
        return self._header

    @property
    def links(self):
        if self._links is None:
            self._links = self.file.group_links(self)
        return self._links

    def attributes(self):
        return self.file.attributes(self.header)


class Dataset:
    """A dataset's metadata, parsed from its object header."""

    def __init__(self, file, address, name, header=None):
        self.file, self.address, self.name = file, address, name
        h = header if header is not None else file.object_header(address)
        self.header = h
        f = file
        b = f.b
        m = h.first(1)
        self.shape, self.maxshape = _parse_dataspace(b, m[2], f.L) if m else ((), ())
        m = h.first(3)
        if m is None:
            raise FormatError('dataset without a datatype')
        self.type = f.message_datatype(m)
        self._layout = self._filters = self._fill = None
        self.null = self.shape is None                  # a null dataspace: no values at all
        if self.null:
            self.shape = self.maxshape = ()

    # layout, filters and fill value are parsed when first wanted: opening a
    # file of hundreds of variables needs only their shapes and types
    @property
    def layout(self):
        if self._layout is None:
            self._layout = self.file.parse_layout(self.header.first(8))
        return self._layout

    @property
    def filters(self):
        if self._filters is None:
            m = self.header.first(11)
            self._filters = _parse_filters(self.file.b, m[2]) if m else []
        return self._filters

    @property
    def fill(self):
        if self._fill is None:
            self._fill = (self.file.parse_fill(self.header, self.type),)
        return self._fill[0]

    @property
    def dtype(self):
        return self.type.dtype

    @property
    def chunks(self):
        lay = self.layout
        return tuple(lay.chunks) if lay.cls == 2 else None

    def attributes(self):
        return self.file.attributes(self.header)

    def chunk_table(self):
        """Every stored chunk as (offset tuple in elements, address, stored
        size, filter mask), in file index order."""
        return self.file.chunk_table(self)


# ---------------------------------------------------------------- the file

SIGNATURE = b'\x89HDF\r\n\x1a\n'


def find_superblock(b):
    """The superblock's offset (0, 512, 1024, 2048, ...) or None."""
    n = len(b)
    at = 0
    while at + 8 <= n:
        if bytes(b[at:at + 8]) == SIGNATURE:
            return at
        at = 512 if at == 0 else at * 2
    return None


class H5File:
    """The HDF5 structure of a file mapped (or held) in `buf`."""

    def __init__(self, buf):
        self.buf = buf
        self.b = b = memoryview(buf).cast('B') if not isinstance(buf, memoryview) else buf
        sb = find_superblock(b)
        if sb is None:
            raise FormatError('not an HDF5 file')
        v = b[sb + 8]
        self.sb_version = v
        self._headers = {}
        self._attr_refs = {}
        self._fheaps = {}
        self._gheaps = {}
        if v in (0, 1):
            self.O, self.L = b[sb + 13], b[sb + 14]
            p = sb + 24 + (4 if v == 1 else 0)
            O = self.O
            self.base = _uint(b, p, O)
            p += 4 * O                                  # base, free space, EOF, driver
            # root group symbol table entry
            root = _uint(b, p + O, O)
            cache = _u32(b, p + 2 * O)[0]
            self.root_cache = None
            if cache == 1:
                self.root_cache = (_uint(b, p + 2 * O + 8, O), _uint(b, p + 3 * O + 8, O))
        elif v in (2, 3):
            self.O, self.L = b[sb + 9], b[sb + 10]
            O = self.O
            p = sb + 12
            self.base = _uint(b, p, O)
            root = _uint(b, p + 3 * O, O)
            self.root_cache = None
        else:
            raise FormatError('superblock version %d is not supported' % v)
        self.undef = (1 << (8 * self.O)) - 1
        self.root_address = root
        self.root = Group(self, root, '/')

    # -- addresses and integers
    def addr(self, a):
        return self.base + a

    def u(self, at, n):
        return _uint(self.b, at, n)

    def offset(self, at):
        return _uint(self.b, at, self.O)

    def length(self, at):
        return _uint(self.b, at, self.L)

    # -- object headers
    def object_header(self, address):
        h = self._headers.get(address)
        if h is not None:
            return h
        b = self.b
        at = self.addr(address)
        msgs = []
        if bytes(b[at:at + 4]) == b'OHDR':
            version = b[at + 4]
            flags = b[at + 5]
            p = at + 6
            if flags & 32:
                p += 16
            if flags & 16:
                p += 4
            n = 1 << (flags & 3)
            size = _uint(b, p, n)
            p += n
            self._v2_messages(p, p + size, flags, msgs)
        elif b[at] == 1:
            version = 1
            flags = 0
            nmsg = _u16(b, at + 2)[0]
            size = _u32(b, at + 8)[0]
            self._v1_messages(at + 16, at + 16 + size, msgs, nmsg)
        else:
            raise FormatError('no object header at %d' % at)
        h = ObjectHeader(address, msgs, flags, version)
        self._headers[address] = h
        return h

    def _v1_messages(self, p, end, msgs, nmsg):
        b = self.b
        todo = [(p, end)]
        while todo:
            p, end = todo.pop(0)
            while p + 8 <= end:
                mtype, size, mflags = _u16(b, p)[0], _u16(b, p + 2)[0], b[p + 4]
                data = p + 8
                if mtype == 0x10:
                    todo.append((self.addr(self.offset(data)),
                                 self.addr(self.offset(data)) + self.length(data + self.O)))
                elif mtype:
                    msgs.append((mtype, mflags, data, size, None))
                p = data + size

    def _v2_messages(self, p, end, hflags, msgs):
        b = self.b
        corder = bool(hflags & 4)
        hdr = 6 if corder else 4
        todo = [(p, end)]
        while todo:
            p, end = todo.pop(0)
            while p + hdr <= end:
                mtype, size, mflags = b[p], _u16(b, p + 1)[0], b[p + 3]
                co = _u16(b, p + 4)[0] if corder else None
                data = p + hdr
                if mtype == 0x10:
                    a = self.addr(self.offset(data))
                    n = self.length(data + self.O)
                    if bytes(b[a:a + 4]) != b'OCHK':
                        raise FormatError('bad continuation block at %d' % a)
                    todo.append((a + 4, a + n - 4))      # less signature and checksum
                elif mtype:
                    msgs.append((mtype, mflags, data, size, co))
                p = data + size

    # -- shared / committed messages
    def message_datatype(self, m):
        """The datatype a message holds, following a shared (committed) one."""
        return self.datatype_at(self.b, m[2], bool(m[1] & 2))

    def datatype_at(self, b, p, shared):
        """The datatype at b[p:], or the committed one a shared message there
        points to."""
        if shared:
            version = b[p]
            if version == 1:
                a = _uint(b, p + 8, self.O)
            else:
                kind = b[p + 1]
                if version == 3 and kind == 1:
                    raise FormatError('shared object header messages (SOHM) are not supported')
                a = _uint(b, p + 2, self.O)
            h = self.object_header(a)
            t, _ = parse_datatype(self.b, h.first(3)[2])
            t.committed = a
            return t
        t, _ = parse_datatype(b, p)
        return t

    # -- layout and fill
    def parse_layout(self, m):
        lay = Layout()
        if m is None:
            raise FormatError('dataset without a layout')
        b = self.b
        p = m[2]
        version = b[p]
        lay.version = version
        O, L = self.O, self.L
        if version in (1, 2):
            ndim = b[p + 1]
            lay.cls = b[p + 2]
            p += 8
            if lay.cls != 0:
                lay.address = self.offset(p)
                p += O
            dims = [_u32(b, p + 4 * i)[0] for i in range(ndim)]
            p += 4 * ndim
            if lay.cls == 2:
                lay.chunks = dims[:-1]
                lay.index = 'btree1'
                p += 4
            elif lay.cls == 1:
                lay.size = None                         # computed from the dataspace
            else:
                n = _u32(b, p)[0]
                lay.data = (p + 4, n)
            return lay
        lay.cls = cls = b[p + 1]
        p += 2
        if cls == 0:
            n = _u16(b, p)[0]
            lay.data = (p + 2, n)
        elif cls == 1:
            lay.address = self.offset(p)
            lay.size = self.length(p + O)
        elif cls == 2:
            if version == 3:
                ndim = b[p]
                lay.address = self.offset(p + 1)
                p += 1 + O
                lay.chunks = [_u32(b, p + 4 * i)[0] for i in range(ndim)][:-1]
                lay.index = 'btree1'
            else:
                lay.flags = flags = b[p]
                ndim = b[p + 1]
                enc = b[p + 2]
                p += 3
                lay.chunks = [_uint(b, p + enc * i, enc) for i in range(ndim)][:-1]
                p += enc * ndim
                itype = b[p]
                p += 1
                if itype == 1:
                    lay.index = 'single'
                    if flags & 2:
                        lay.single_size = self.length(p)
                        lay.single_mask = _u32(b, p + L)[0]
                        p += L + 4
                elif itype == 2:
                    lay.index = 'implicit'
                elif itype == 3:
                    lay.index = 'farray'
                    lay.index_info = (b[p],)
                    p += 1
                elif itype == 4:
                    lay.index = 'earray'
                    lay.index_info = tuple(b[p:p + 5])
                    p += 5
                elif itype == 5:
                    lay.index = 'btree2'
                    lay.index_info = (_u32(b, p)[0], b[p + 4], b[p + 5])
                    p += 6
                else:
                    raise FormatError('unknown chunk index type %d' % itype)
                lay.address = self.offset(p)
        elif cls == 3:
            pass                                        # virtual: refused when read
        else:
            raise FormatError('unknown layout class %d' % cls)
        return lay

    def parse_fill(self, h, dtype):
        """The fill value's bytes, or None when undefined (HDF5's default:
        zeros)."""
        b = self.b
        m = h.first(5)
        if m is not None:
            p = m[2]
            version = b[p]
            if version in (1, 2):
                defined = b[p + 3]
                if version == 1 or defined:
                    n = _u32(b, p + 4)[0]
                    return bytes(b[p + 8:p + 8 + n]) if n else None
                return None
            flags = b[p + 1]
            if flags & 32:
                n = _u32(b, p + 2)[0]
                return bytes(b[p + 6:p + 6 + n]) if n else None
            return None
        m = h.first(4)
        if m is not None:
            n = _u32(b, m[2])[0]
            return bytes(b[m[2] + 4:m[2] + 4 + n]) if n else None
        return None

    # -- groups
    def group_links(self, g):
        """{name: (kind, target)} in netCDF's iteration order. kind 'hard'
        (target: object header address) or 'soft' / 'external' (target:
        the path)."""
        h = g.header
        b = self.b
        st = h.first(0x11)
        if st is not None or (g.address == self.root_address and self.root_cache
                              and h.first(0x02) is None and not h.all(0x06)):
            if st is not None:
                btree, heap = self.offset(st[2]), self.offset(st[2] + self.O)
            else:
                btree, heap = self.root_cache
            out = self._symbol_table(btree, heap)
            return dict(sorted(out.items()))
        links = []
        for m in h.all(0x06):
            links.append(self._link(m[2]))
        li = h.first(0x02)
        if li is not None:
            p = li[2]
            flags = b[p + 1]
            p += 2 + (8 if flags & 1 else 0)
            fheap = self.offset(p)
            name_bt = self.offset(p + self.O)
            if fheap != self.undef:
                heap = self.fractal_heap(fheap)
                for rec in self.btree2_records(name_bt):
                    hid = rec[4:4 + heap.id_len]
                    links.append(self._link_from(heap.object(hid), 0))
        if any(l[3] is not None for l in links):
            links.sort(key=lambda l: (l[3] if l[3] is not None else 0, l[0]))
        else:
            links.sort(key=lambda l: l[0])
        return {l[0]: (l[1], l[2]) for l in links}

    def _link(self, p):
        return self._link_from(self.b, p)

    def _link_from(self, b, p):
        flags = b[p + 1]
        p += 2
        kind = 0
        corder = None
        if flags & 8:
            kind = b[p]
            p += 1
        if flags & 4:
            corder = _u64(b, p)[0]
            p += 8
        if flags & 16:
            p += 1
        n = 1 << (flags & 3)
        nlen = _uint(b, p, n)
        p += n
        name = bytes(b[p:p + nlen]).decode('utf-8', 'replace')
        p += nlen
        if kind == 0:
            return (name, 'hard', _uint(b, p, self.O), corder)
        if kind == 1:
            n = _u16(b, p)[0]
            return (name, 'soft', bytes(b[p + 2:p + 2 + n]).decode('utf-8', 'replace'), corder)
        return (name, 'external', None, corder)

    def _cstr(self, p):
        b = self.b
        z = p
        while b[z]:
            z += 1
        return bytes(b[p:z]).decode('utf-8', 'replace')

    def _symbol_table(self, btree, heap):
        b = self.b
        ha = self.addr(heap)
        if bytes(b[ha:ha + 4]) != b'HEAP':
            raise FormatError('bad local heap')
        data = self.addr(self.offset(ha + 8 + 2 * self.L))
        out = {}
        O = self.O
        stack = [btree]
        while stack:
            a = self.addr(stack.pop())
            if bytes(b[a:a + 4]) != b'TREE':
                raise FormatError('bad v1 B-tree node')
            level = b[a + 5]
            n = _u16(b, a + 6)[0]
            p = a + 8 + 2 * O + self.L                  # first child after key 0
            for i in range(n):
                child = self.offset(p)
                p += O + self.L
                if level:
                    stack.append(child)
                    continue
                s = self.addr(child)
                if bytes(b[s:s + 4]) != b'SNOD':
                    raise FormatError('bad symbol table node')
                ns = _u16(b, s + 6)[0]
                e = s + 8
                for _ in range(ns):
                    name = self._cstr(data + self.offset(e))
                    if _u32(b, e + 2 * O)[0] == 2:      # cache type 2: a soft link
                        out[name] = ('soft', self._cstr(data + _u32(b, e + 2 * O + 8)[0]))
                    else:
                        out[name] = ('hard', self.offset(e + O))
                    e += 2 * O + 24
        return out

    # -- attributes
    def attribute_refs(self, h):
        """{name: (buffer, position, creation order)} for every attribute,
        values not decoded; in creation order where tracked, else header
        order (cached per header)."""
        refs = self._attr_refs.get(h.address)
        if refs is not None:
            return refs
        out = []
        b = self.b
        for i, m in enumerate(h.all(0x0C)):
            p = m[2]
            out.append((self._attr_name(b, p), b, p, m[4] if m[4] is not None else i))
        ai = h.first(0x15)
        if ai is not None:
            p = ai[2]
            flags = b[p + 1]
            p += 2 + (2 if flags & 1 else 0)
            fheap = self.offset(p)
            name_bt = self.offset(p + self.O)
            if fheap != self.undef:
                heap = self.fractal_heap(fheap)
                base = len(out)
                for k, rec in enumerate(self.btree2_records(name_bt)):
                    obj = heap.object(rec[:8])
                    corder = _u32(rec, 9)[0] if flags & 1 else base + k
                    out.append((self._attr_name(obj, 0), obj, 0, corder))
        out.sort(key=lambda r: r[3])
        refs = {r[0]: r[1:] for r in out}
        self._attr_refs[h.address] = refs
        return refs

    @staticmethod
    def _attr_name(b, p):
        version = b[p]
        nlen = _u16(b, p + 2)[0]
        q = p + 8 + (1 if version == 3 else 0)
        return bytes(b[q:q + nlen]).split(b'\0')[0].decode('utf-8', 'replace')

    def attribute(self, h, name):
        """One attribute (decoded), or None."""
        r = self.attribute_refs(h).get(name)
        return None if r is None else self._attribute(r[0], r[1], r[2])

    def attributes(self, h):
        """{name: Attribute} in creation order where tracked, else header order."""
        return {k: self._attribute(b, p, co) for k, (b, p, co) in self.attribute_refs(h).items()}

    def _attributes_full(self, h):
        out = []
        for i, m in enumerate(h.all(0x0C)):
            a = self._attribute(self.b, m[2], m[4] if m[4] is not None else i)
            out.append(a)
        ai = h.first(0x15)
        if ai is not None:
            b = self.b
            p = ai[2]
            flags = b[p + 1]
            p += 2 + (2 if flags & 1 else 0)
            fheap = self.offset(p)
            name_bt = self.offset(p + self.O)
            if fheap != self.undef:
                heap = self.fractal_heap(fheap)
                base = len(out)
                for k, rec in enumerate(self.btree2_records(name_bt)):
                    hid = rec[:8]
                    corder = _u32(rec, 9)[0] if flags & 1 else base + k
                    out.append(self._attribute(heap.object(hid), 0, corder))
        out.sort(key=lambda a: a.corder)
        return {a.name: a for a in out}

    def _attribute(self, b, p, corder):
        version = b[p]
        flags = b[p + 1]
        nlen, tlen, slen = _u16(b, p + 2)[0], _u16(b, p + 4)[0], _u16(b, p + 6)[0]
        p += 8
        if version == 3:
            p += 1
        pad = (lambda n: (n + 7) // 8 * 8) if version == 1 else (lambda n: n)
        name = bytes(b[p:p + nlen]).split(b'\0')[0].decode('utf-8', 'replace')
        p += pad(nlen)
        t = self.datatype_at(b, p, bool(flags & 1))
        p += pad(tlen)
        shape, _ = _parse_dataspace(b, p, self.L)
        p += pad(slen)
        if shape is None:
            n = 0
        else:
            n = 1
            for s in shape:
                n *= s
        raw = bytes(b[p:p + n * t.size])
        return Attribute(name, t, shape, raw, corder, self)

    # -- values of special types
    def decode_values(self, t, shape, raw):
        """numpy values from raw bytes of datatype t (vlen through the global heap)."""
        if shape is None:
            return None
        n = 1
        for s in shape:
            n *= s
        if t.cls == Datatype.VLEN:
            out = np.empty(n, object)
            O = self.O
            for i in range(n):
                q = i * t.size
                ln = _u32(raw, q)[0]
                coll = _uint(raw, q + 4, O)
                idx = _u32(raw, q + 4 + O)[0]
                data = self.global_heap_object(coll, idx) if coll else b''
                if t.vlen_string:
                    s = bytes(data[:ln])
                    out[i] = s.decode('utf-8' if t.charset == 1 else 'latin-1', 'replace') \
                        if True else s
                else:
                    bt = t.base
                    out[i] = (self.decode_values(bt, (ln,), bytes(data[:ln * bt.size]))
                              if bt.dtype is None or bt.cls == Datatype.VLEN
                              else np.frombuffer(bytes(data[:ln * bt.size]), bt.dtype).copy())
            return out.reshape(shape)
        if t.dtype is None:
            raise FormatError('cannot decode datatype class %d' % t.cls)
        a = np.frombuffer(raw, t.dtype, n).reshape(shape)
        return a

    def vlen_refs(self, a):
        """For an attribute of variable-length sequences of object references
        (DIMENSION_LIST): the first referenced address of each element, or
        None for an empty one -- without building numpy arrays."""
        t = a.dtype
        O = self.O
        out = []
        raw = a.raw
        n = len(raw) // t.size if t.size else 0
        for i in range(n):
            q = i * t.size
            ln = _u32(raw, q)[0]
            coll = _uint(raw, q + 4, O)
            if not ln or not coll:
                out.append(None)
                continue
            data = self.global_heap_object(coll, _u32(raw, q + 4 + O)[0])
            out.append(_uint(data, 0, O))
        return out

    def global_heap_object(self, coll, idx):
        objs = self._gheaps.get(coll)
        if objs is None:
            b = self.b
            a = self.addr(coll)
            if bytes(b[a:a + 4]) != b'GCOL':
                raise FormatError('bad global heap collection')
            size = self.length(a + 8)
            p = a + 8 + self.L
            end = a + size
            objs = {}
            while p + 8 + self.L <= end:
                i = _u16(b, p)[0]
                if i == 0:
                    break
                n = self.length(p + 8)
                objs[i] = (p + 8 + self.L, n)
                p += 8 + self.L + (n + 7) // 8 * 8
            self._gheaps[coll] = objs
        p, n = objs[idx]
        return self.b[p:p + n]

    # -- fractal heaps and v2 B-trees
    def fractal_heap(self, address):
        h = self._fheaps.get(address)
        if h is None:
            h = self._fheaps[address] = FractalHeap(self, address)
        return h

    def btree2_records(self, address):
        """Every record of a v2 B-tree (bytes objects), in key order."""
        b = self.b
        a = self.addr(address)
        if bytes(b[a:a + 4]) != b'BTHD':
            raise FormatError('bad v2 B-tree header')
        node_size = _u32(b, a + 6)[0]
        rsize = _u16(b, a + 10)[0]
        depth = _u16(b, a + 12)[0]
        root = self.offset(a + 16)
        nroot = _u16(b, a + 16 + self.O)[0]
        if root == self.undef:
            return []
        # bytes of each depth's "records in child" fields (H5B2__hdr_init)
        max_nrec = [(node_size - 10) // rsize]
        cum = [max_nrec[0]]
        cum_size = [0]
        max_nrec_size = _enc_size(max_nrec[0])
        for d in range(1, depth + 1):
            ptr = self.O + max_nrec_size + (cum_size[d - 1] if d > 1 else 0)
            mn = (node_size - (10 + ptr)) // (rsize + ptr)
            max_nrec.append(mn)
            cum.append((mn + 1) * cum[d - 1] + mn)
            cum_size.append(_enc_size(cum[d]))
        out = []

        def node(addr, nrec, d):
            q = self.addr(addr)
            p = q + 6
            if d == 0:
                for i in range(nrec):
                    out.append(bytes(b[p + i * rsize:p + (i + 1) * rsize]))
                return
            recs = [bytes(b[p + i * rsize:p + (i + 1) * rsize]) for i in range(nrec)]
            p += nrec * rsize
            for i in range(nrec + 1):
                child = self.offset(p)
                p += self.O
                cn = _uint(b, p, max_nrec_size)
                p += max_nrec_size
                if d > 1:
                    p += cum_size[d - 1]
                node(child, cn, d - 1)
                if i < nrec:
                    out.append(recs[i])
        node(root, nroot, depth)
        return out

    # -- chunk indexes
    def chunk_table(self, ds):
        """[(offsets, address, size, mask)] for every allocated chunk."""
        lay = ds.layout
        rank = len(ds.shape)
        chunks = lay.chunks
        esize = ds.type.size
        csize = esize
        for c in chunks:
            csize *= c
        filtered = bool(ds.filters)
        if lay.address == self.undef:
            return []
        if lay.index == 'btree1':
            return self._btree1_chunks(lay.address, rank)
        if lay.index == 'single':
            size = lay.single_size if lay.single_size is not None else csize
            return [((0,) * rank, self.addr(lay.address), size, lay.single_mask)]
        nper = [-(-s // c) if c else 0 for s, c in zip(ds.shape, chunks)]
        # implicit and fixed-array indexes count chunks over the maximum shape
        nmax = [-(-(m if m is not None else s) // c) if c else 0
                for s, m, c in zip(ds.shape, ds.maxshape, chunks)]
        if lay.index == 'implicit':
            out = []
            n = 1
            for x in nmax:
                n *= x
            for i in range(n):
                offs = _unravel(i, nmax, chunks)
                if all(o < s for o, s in zip(offs, ds.shape)):
                    out.append((offs, self.addr(lay.address) + i * csize, csize, 0))
            return out
        if lay.index == 'btree2':
            return self._btree2_chunks(lay.address, rank, chunks, filtered, csize)
        if lay.index == 'farray':
            out = []
            for i, (a, s, m) in enumerate(self._farray(lay.address)):
                offs = _unravel(i, nmax, chunks)
                if a != self.undef and all(o < n for o, n in zip(offs, ds.shape)):
                    out.append((offs, self.addr(a), s if s is not None else csize, m))
            return out
        if lay.index == 'earray':
            return self._earray_chunks(ds, nper, chunks, csize)
        raise FormatError('chunk index %r' % lay.index)

    def _btree1_chunks(self, address, rank):
        b = self.b
        O = self.O
        out = []
        ksize = 8 + 8 * (rank + 1)
        stack = [address]
        while stack:
            a = self.addr(stack.pop())
            if bytes(b[a:a + 4]) != b'TREE':
                raise FormatError('bad chunk B-tree node')
            level = b[a + 5]
            n = _u16(b, a + 6)[0]
            p = a + 8 + 2 * O
            kids = []
            for i in range(n):
                size = _u32(b, p)[0]
                mask = _u32(b, p + 4)[0]
                offs = struct.unpack_from('<%dQ' % rank, b, p + 8)
                child = self.offset(p + ksize)
                if level:
                    kids.append(child)
                else:
                    out.append((offs, self.addr(child), size, mask))
                p += ksize + O
            stack.extend(reversed(kids))
        return out

    def _btree2_chunks(self, address, rank, chunks, filtered, csize):
        b = self.b
        a = self.addr(address)
        rsize = _u16(b, a + 10)[0]
        O = self.O
        out = []
        csz_len = rsize - O - 4 - 8 * rank if filtered else 0
        for rec in self.btree2_records(address):
            addr = _uint(rec, 0, O)
            p = O
            if filtered:
                size = _uint(rec, p, csz_len)
                mask = _u32(rec, p + csz_len)[0]
                p += csz_len + 4
            else:
                size, mask = csize, 0
            scaled = struct.unpack_from('<%dQ' % rank, rec, p)
            out.append((tuple(s * c for s, c in zip(scaled, chunks)), self.addr(addr), size, mask))
        return out

    def _elements(self, b, p, n, esize, filtered):
        O = self.O
        out = []
        for i in range(n):
            q = p + i * esize
            a = _uint(b, q, O)
            if filtered:
                cl = esize - O - 4
                out.append((a, _uint(b, q + O, cl), _u32(b, q + O + cl)[0]))
            else:
                out.append((a, None, 0))
        return out

    def _farray(self, address):
        b = self.b
        a = self.addr(address)
        if bytes(b[a:a + 4]) != b'FAHD':
            raise FormatError('bad fixed array header')
        client = b[a + 5]
        esize = b[a + 6]
        page_bits = b[a + 7]
        n = self.length(a + 8)
        db = self.addr(self.offset(a + 8 + self.L))
        if bytes(b[db:db + 4]) != b'FADB':
            raise FormatError('bad fixed array data block')
        p = db + 6 + self.O
        filtered = client == 1
        page = 1 << page_bits
        if n <= page:
            return self._elements(b, p, n, esize, filtered)
        npages = -(-n // page)
        p += (npages + 7) // 8 + 4                      # page init bitmap, checksum
        out = []
        for k in range(npages):
            m = min(page, n - k * page)
            out.extend(self._elements(b, p, m, esize, filtered))
            p += m * esize + 4
        return out

    def _earray_chunks(self, ds, nper, chunks, csize):
        b = self.b
        lay = ds.layout
        a = self.addr(lay.address)
        if bytes(b[a:a + 4]) != b'EAHD':
            raise FormatError('bad extensible array header')
        client = b[a + 5]
        esize, max_bits, iblk_n, dblk_min, sblk_min_ptrs, page_bits = (b[a + 6], b[a + 7], b[a + 8],
                                                                     b[a + 9], b[a + 10], b[a + 11])
        p = a + 12 + 6 * self.L
        iblk = self.addr(self.offset(p))
        filtered = client == 1
        nelm = self.length(a + 12 + 4 * self.L) + 1    # max index set, plus one
        # super blocks (H5EA__hdr_init)
        nsblks = 1 + max_bits - _log2(dblk_min)
        sb = []
        start_idx = start_dblk = 0
        for s in range(nsblks):
            nd = 1 << (s // 2)
            dn = dblk_min * (1 << ((s + 1) // 2))
            sb.append((nd, dn, start_idx, start_dblk))
            start_idx += nd * dn
            start_dblk += nd
        if bytes(b[iblk:iblk + 4]) != b'EAIB':
            raise FormatError('bad extensible array index block')
        q = iblk + 6 + self.O
        elems = self._elements(b, q, iblk_n, esize, filtered)
        q += iblk_n * esize
        ib_nsblks = 2 * _log2(sblk_min_ptrs)
        nd_addrs = 2 * (sblk_min_ptrs - 1)
        dblk_addrs = [self.offset(q + i * self.O) for i in range(nd_addrs)]
        q += nd_addrs * self.O
        sblk_addrs = [self.offset(q + i * self.O) for i in range(nsblks - ib_nsblks)]
        off_size = (max_bits + 7) // 8
        page = 1 << page_bits

        def dblock(addr, dn):
            d = self.addr(addr)
            if bytes(b[d:d + 4]) != b'EADB':
                raise FormatError('bad extensible array data block')
            r = d + 6 + self.O + off_size
            if dn <= page:
                return self._elements(b, r, dn, esize, filtered)
            out = []
            r += 4                                      # header checksum; then pages
            for k in range(-(-dn // page)):
                m = min(page, dn - k * page)
                out.extend(self._elements(b, r, m, esize, filtered))
                r += m * esize + 4
            return out

        idx = len(elems)
        all_elems = list(elems)
        for s in range(nsblks):
            if len(all_elems) >= nelm:
                break
            nd, dn, sidx, sdblk = sb[s]
            if s < ib_nsblks:
                for k in range(nd):
                    da = dblk_addrs[sdblk + k]
                    all_elems.extend(dblock(da, dn) if da != self.undef else [(self.undef, None, 0)] * dn)
            else:
                sa = sblk_addrs[s - ib_nsblks]
                if sa == self.undef:
                    all_elems.extend([(self.undef, None, 0)] * (nd * dn))
                    continue
                r = self.addr(sa)
                if bytes(b[r:r + 4]) != b'EASB':
                    raise FormatError('bad extensible array super block')
                r += 6 + self.O + off_size
                if dn > page:
                    r += (nd * (-(-dn // page)) + 7) // 8   # page init bitmap
                for k in range(nd):
                    da = self.offset(r + k * self.O)
                    all_elems.extend(dblock(da, dn) if da != self.undef else [(self.undef, None, 0)] * dn)
        del idx
        # element i is the chunk at linear index i with the unlimited dimension slowest
        rank = len(nper)
        unlim = [i for i, m in enumerate(ds.maxshape) if m is None]
        order = list(range(rank))
        if unlim and unlim[0] != 0:
            u = unlim[0]
            order = [u] + [i for i in range(rank) if i != u]
        sw = [nper[i] for i in order]
        # the fixed dimensions' counts come from the maximum shape
        maxper = [(-(-m // c) if m is not None else n) for m, c, n in zip(ds.maxshape, chunks, nper)]
        swmax = [maxper[i] for i in order]
        out = []
        total = _prod(nper)
        for i, (ad, s, m) in enumerate(all_elems):
            if ad == self.undef:
                continue
            sc = _unravel_counts(i, swmax)
            if any(x >= n for x, n in zip(sc, sw)):
                continue
            offs = [0] * rank
            for k, d in enumerate(order):
                offs[d] = sc[k] * chunks[d]
            out.append((tuple(offs), self.addr(ad), s if s is not None else csize, m))
        del total
        return out


def _prod(xs):
    n = 1
    for x in xs:
        n *= x
    return n


def _unravel_counts(i, counts):
    out = [0] * len(counts)
    for k in range(len(counts) - 1, 0, -1):
        i, out[k] = divmod(i, counts[k])
    if counts:
        out[0] = i
    return out


def _unravel(i, nper, chunks):
    sc = _unravel_counts(i, nper)
    return tuple(s * c for s, c in zip(sc, chunks))


class FractalHeap:
    """A fractal heap (III.G): managed objects in a doubling table of direct
    blocks, and tiny objects held in the ID itself."""

    def __init__(self, f, address):
        self.f = f
        b = f.b
        a = f.addr(address)
        if bytes(b[a:a + 4]) != b'FRHP':
            raise FormatError('bad fractal heap header')
        O, L = f.O, f.L
        self.id_len = _u16(b, a + 5)[0]
        self.filter_len = _u16(b, a + 7)[0]
        self.flags = b[a + 9]
        self.max_man = _u32(b, a + 10)[0]
        self.huge_btree = f.offset(a + 14 + L)
        self._huge_map = None
        p = a + 14 + L + O + L + O + 8 * L
        self.width = _u16(b, p)[0]
        self.start = f.length(p + 2)
        self.max_direct = f.length(p + 2 + L)
        self.max_heap_bits = _u16(b, p + 2 + 2 * L)[0]
        self.root_rows0 = _u16(b, p + 4 + 2 * L)[0]
        self.root = f.offset(p + 6 + 2 * L)
        self.root_rows = _u16(b, p + 6 + 2 * L + O)[0]
        self.off_size = (self.max_heap_bits + 7) // 8
        self.len_size = min((_log2(self.max_direct) + 7) // 8, _enc_size(self.max_man))
        self.max_direct_rows = _log2(self.max_direct) - _log2(self.start) + 2
        if self.filter_len:
            raise FormatError('filtered fractal heaps are not supported')

    def _huge(self, hid):
        """A huge object: its address and length in the ID itself, or found
        by ID in the heap's v2 B-tree of huge objects (records: address,
        length, ID)."""
        f = self.f
        O, L = f.O, f.L
        if O + L <= self.id_len - 1:
            a, n = _uint(hid, 1, O), _uint(hid, 1 + O, L)
        else:
            if self._huge_map is None:
                self._huge_map = {}
                for rec in f.btree2_records(self.huge_btree):
                    self._huge_map[_uint(rec, O + L, L)] = (_uint(rec, 0, O), _uint(rec, O, L))
            a, n = self._huge_map[_uint(hid, 1, min(self.id_len - 1, 8))]
        a = f.addr(a)
        return f.b[a:a + n]

    def _row_size(self, r):
        return self.start if r == 0 else self.start << (r - 1)

    def object(self, hid):
        """The bytes of the object with heap ID hid."""
        first = hid[0]
        kind = (first >> 4) & 3
        if kind == 2:                                   # tiny: the bytes are in the ID
            if self.id_len <= 18:
                n = (first & 15) + 1
                return memoryview(bytes(hid[1:1 + n]))
            n = (((first & 15) << 8) | hid[1]) + 1
            return memoryview(bytes(hid[2:2 + n]))
        if kind == 1:
            return self._huge(hid)
        if kind != 0:
            raise FormatError('unknown fractal heap ID type')
        off = _uint(hid, 1, self.off_size)
        ln = _uint(hid, 1 + self.off_size, self.len_size)
        blk, boff = self._direct_block(off)
        return self.f.b[blk + (off - boff):blk + (off - boff) + ln]

    def _direct_block(self, off):
        """(absolute block address, block's heap offset) holding heap offset off."""
        f = self.f
        if self.root_rows == 0:
            return f.addr(self.root), 0
        return self._in_indirect(self.root, self.root_rows, off)

    def _in_indirect(self, address, nrows, off):
        f = self.f
        b = f.b
        a = f.addr(address)
        if bytes(b[a:a + 4]) != b'FHIB':
            raise FormatError('bad fractal heap indirect block')
        boff = _uint(b, a + 5 + f.O, self.off_size)
        p = a + 5 + f.O + self.off_size
        rel = off - boff
        pos = 0
        for r in range(nrows):
            rs = self._row_size(r)
            span = rs * self.width
            if rel < pos + span:
                col = (rel - pos) // rs
                entry = r * self.width + col
                child_off = boff + pos + col * rs
                if r < self.max_direct_rows:
                    ca = f.offset(p + entry * f.O)
                    return f.addr(ca), child_off
                # indirect child: entries after all direct ones
                ndirect = min(nrows, self.max_direct_rows) * self.width
                k = ndirect + (r - self.max_direct_rows) * self.width + col
                ca = f.offset(p + k * f.O)
                child_rows = _log2(rs) - _log2(self.start * self.width) + 1
                return self._in_indirect(ca, child_rows, off)
            pos += span
        raise FormatError('heap offset outside the heap')
