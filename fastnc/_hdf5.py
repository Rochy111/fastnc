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
import math
import struct

import numpy as np

UNDEF = (1 << 64) - 1           # undefined address (all bits set) at 8 bytes


class FormatError(ValueError):
    """The bytes are not HDF5 as this reader understands it."""


_u8 = struct.Struct('<B').unpack_from
_MSG4 = struct.Struct('<BHB').unpack_from           # v2 header message: type, size, flags
_MSG6 = struct.Struct('<BHBH').unpack_from          # ... and creation order
_u16 = struct.Struct('<H').unpack_from
_u32 = struct.Struct('<I').unpack_from
_u64 = struct.Struct('<Q').unpack_from
_RANKS = {}                                         # rank -> unpacker of that many 8-byte sizes
_UNLIMITED = (1 << 64) - 1


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


_M32 = 0xFFFFFFFF


def _rot(x, k):
    return ((x << k) | (x >> (32 - k))) & _M32


def lookup3(data, initval=0):
    """Bob Jenkins' lookup3 hashlittle, as H5_checksum_lookup3: the hash
    HDF5 indexes dense attribute and link names by."""
    n = len(data)
    a = b = c = (0xDEADBEEF + n + initval) & _M32
    i = 0
    while n > 12:
        a = (a + data[i] + (data[i + 1] << 8) + (data[i + 2] << 16) + (data[i + 3] << 24)) & _M32
        b = (b + data[i + 4] + (data[i + 5] << 8) + (data[i + 6] << 16) + (data[i + 7] << 24)) & _M32
        c = (c + data[i + 8] + (data[i + 9] << 8) + (data[i + 10] << 16) + (data[i + 11] << 24)) & _M32
        a = (a - c) & _M32; a ^= _rot(c, 4); c = (c + b) & _M32       # noqa: E702
        b = (b - a) & _M32; b ^= _rot(a, 6); a = (a + c) & _M32       # noqa: E702
        c = (c - b) & _M32; c ^= _rot(b, 8); b = (b + a) & _M32       # noqa: E702
        a = (a - c) & _M32; a ^= _rot(c, 16); c = (c + b) & _M32      # noqa: E702
        b = (b - a) & _M32; b ^= _rot(a, 19); a = (a + c) & _M32      # noqa: E702
        c = (c - b) & _M32; c ^= _rot(b, 4); b = (b + a) & _M32       # noqa: E702
        n -= 12
        i += 12
    if n == 0:
        return c
    t = list(data[i:i + n]) + [0] * (12 - n)
    a = (a + t[0] + (t[1] << 8) + (t[2] << 16) + (t[3] << 24)) & _M32
    b = (b + t[4] + (t[5] << 8) + (t[6] << 16) + (t[7] << 24)) & _M32
    c = (c + t[8] + (t[9] << 8) + (t[10] << 16) + (t[11] << 24)) & _M32
    c ^= b; c = (c - _rot(b, 14)) & _M32                            # noqa: E702
    a ^= c; a = (a - _rot(c, 11)) & _M32                            # noqa: E702
    b ^= a; b = (b - _rot(a, 25)) & _M32                            # noqa: E702
    c ^= b; c = (c - _rot(b, 16)) & _M32                            # noqa: E702
    a ^= c; a = (a - _rot(c, 4)) & _M32                             # noqa: E702
    b ^= a; b = (b - _rot(a, 14)) & _M32                            # noqa: E702
    c ^= b; c = (c - _rot(b, 24)) & _M32                            # noqa: E702
    return c


_HASH_OK = [None]               # lookup3 checked against a file's own name index
_HASHES = {}                    # name -> lookup3 hash
_NAME_BYTES = {}                # name -> its UTF-8 bytes
_ATTR_HEAD = struct.Struct('<BBHHH')    # attribute message: version, flags, name / type / space sizes
_BT1_REC = {}                   # rank -> numpy dtype of a v1 B-tree chunk record


def _enc_size(x):
    """Bytes needed to encode x (H5VM_limit_enc_size)."""
    return (max(int(x), 1).bit_length() - 1) // 8 + 1


def _log2(x):
    return int(x).bit_length() - 1


# ---------------------------------------------------------------- datatypes

# IEEE layouts (sign, exponent position, size, mantissa position, size, bias, implied)
IEEE = {2: (15, 10, 5, 0, 10, 15, 2), 4: (31, 23, 8, 0, 23, 127, 2), 8: (63, 52, 11, 0, 52, 1023, 2)}


class Datatype:
    """A parsed datatype message (IV.A.2.d). `dtype` is the numpy dtype of
    the stored bytes (byte order included), or None for a type numpy cannot
    hold directly (vlen); `cls` is the HDF5 class number."""
    __slots__ = ('cls', 'size', 'dtype', 'order', 'signed', 'base', 'members',
                 'enum', 'vlen_string', 'charset', 'strpad', 'dims', 'ref_kind',
                 'precision', 'offset', 'tag', 'committed', 'fields')

    @property
    def h5py_dtype(self):
        """The dtype h5py gives this type: an enum of FALSE/TRUE over any
        integer is bool, a compound of two equal floats named r and i is complex."""
        if self.cls == self.ENUM and self.enum == {'FALSE': 0, 'TRUE': 1}:
            return np.dtype('?')
        if self.complex():
            return np.dtype('%sc%d' % (self.members[0][2].order, self.size))
        if self.cls == self.COMPOUND and self.retyped():
            return np.dtype({'names': [m[0] for m in self.members],
                             'formats': [m[2].h5py_dtype if m[2].retyped() else self.dtype.fields[m[0]][0]
                                         for m in self.members],
                             'offsets': [m[1] for m in self.members], 'itemsize': self.size})
        if self.odd():
            from ._h5read import float_target, int_target, reduced
            if self.cls == self.FLOAT and (self.dtype is None or reduced(self)):
                return float_target(self)
            if self.dtype is None and self.size < 8:
                return int_target(self)
        return self.dtype

    def complex(self):
        return (self.cls == self.COMPOUND and len(self.members) == 2
                and [m[0] for m in self.members] == ['r', 'i']
                and all(m[2].cls == self.FLOAT for m in self.members)
                and self.members[0][2].size == self.members[1][2].size
                and self.members[1][1] == self.members[0][2].size)

    def retyped(self):
        """Whether h5py gives this type's values another type than the stored
        one, at any depth of a compound (bools, complex numbers)."""
        if self.cls == self.ENUM:
            return self.enum == {'FALSE': 0, 'TRUE': 1}
        if self.cls != self.COMPOUND or self.dtype is None:
            return False
        return self.complex() or any(m[2].retyped() for m in self.members)

    def odd(self):
        """Whether this number is not one numpy holds as it stands (bits fewer
        than its size, off bit 0, in other than IEEE's places, of an odd width)."""
        if self.cls not in (self.FIXED, self.BITFIELD, self.FLOAT):
            return False
        return (self.dtype is None or self.offset != 0 or self.precision != 8 * self.size
                or (self.cls == self.FLOAT and self.fields != IEEE.get(self.size)))

    FIXED, FLOAT, TIME, STRING, BITFIELD, OPAQUE, COMPOUND, REFERENCE, ENUM, VLEN, ARRAY, COMPLEX = range(12)

    def holds_vlen(self):
        """Whether its stored bytes hold variable-length values, at any depth."""
        if self.cls == self.VLEN:
            return True
        if self.cls == self.ARRAY:
            return self.base.holds_vlen()
        return self.cls == self.COMPOUND and any(m[2].holds_vlen() for m in self.members)

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
        t.order = 'V' if f0 & 64 else '>' if f0 & 1 else '<'
        t.offset, t.precision = _u16(b, p)[0], _u16(b, p + 2)[0]
        # sign position, exponent position/size, mantissa position/size,
        # exponent bias, mantissa normalization (0 none, 1 msb set, 2 implied)
        t.fields = (f1, b[p + 4], b[p + 5], b[p + 6], b[p + 7], _u32(b, p + 8)[0], (f0 >> 4) & 3)
        p += 12
        if t.order == 'V':
            pass                                        # (see _h5read.odd_float)
        elif size in (2, 4, 8):
            t.dtype = np.dtype('%sf%d' % (t.order, size))
        elif size == 16 and t.fields[:5] == (79, 64, 15, 0, 64) and np.finfo(np.longdouble).nmant == 63:
            t.dtype = np.dtype('%sf16' % t.order)       # (x87's, on a machine that has them)
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
    elif cls == 11:                                     # complex (HDF5 2.0): a pair of a float
        base, used = parse_datatype(b, p)
        p += used
        t.base = base
        t.order = base.order
        if size in (8, 16) and base.cls == 1 and not f0 & 6:
            t.dtype = np.dtype('%sc%d' % (base.order, size))
    else:
        raise FormatError('unknown datatype class %d' % cls)
    return t, p - at


# ---------------------------------------------------------------- messages

class Layout:
    """Data layout (IV.A.2.i): `cls` 0 compact, 1 contiguous, 2 chunked,
    3 virtual (`data`: the global heap collection and index of its mappings)."""
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
    if L == 8:                                      # (every dataset's at open: no Python per axis)
        read = _RANKS.get(rank)
        if read is None:
            read = _RANKS[rank] = struct.Struct('<%dQ' % rank).unpack_from
        shape = read(b, p)
        p += 8 * rank
        maxshape = shape
        if flags & 1:
            maxshape = tuple([None if v == _UNLIMITED else v for v in read(b, p)])
        return shape, maxshape
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
        self._found = {}

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

    def link(self, name):
        """(kind, target) of one link, or None: looked up, when no one has
        needed the others yet. (Names not ASCII go through the whole list,
        which holds them as they decoded; a remote file reads it in one go.)"""
        if self._links is None and name.isascii() and self.file.source is None:
            r = self._found.get(name, False)
            if r is False:
                r = self._found[name] = self.file.find_link(self, name)
            return r
        return self.links.get(name)

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
        if f._closed:
            raise ValueError('the file of dataset %s is closed' % name)
        m = h.first(1)
        if m and m[1] & 2:                              # (its bytes are a reference, not a shape)
            raise FormatError('shared object header messages (SOHM) are not supported')
        self.shape, self.maxshape = _parse_dataspace(b, m[2], f.L) if m else ((), ())
        m = h.first(3)
        if m is None:
            raise FormatError('dataset without a datatype')
        self.type = f.message_datatype(m)
        self._layout = self._filters = self._fill = None
        self.null = self.shape is None                  # a null dataspace: no values at all
        if self.null:
            self.shape = self.maxshape = ()
        if None in self.maxshape and _is_virtual(b, h.first(8)):
            # an unlimited virtual dataset is as long as its sources are now
            # (HDF5 works its extent out when the dataset is opened)
            from . import _vds
            self.shape = _vds.current_shape(self)

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
            self._filters = self.file.message_filters(m) if m else []
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
    def external(self):
        """[(file name, offset, size)] of the raw data files holding the
        values (External Data Files message), or None."""
        if not hasattr(self, '_external'):
            m = self.header.first(7)
            self._external = self.file.parse_external(m) if m is not None else None
        return self._external

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

    def chunk_arrays(self):
        """The chunk table as arrays where the index allows (else None)."""
        return self.file.chunk_arrays(self)


def _is_virtual(b, m):
    """True for a layout message of class 3 (virtual)."""
    return m is not None and b[m[2]] >= 3 and b[m[2] + 1] == 3


# ---------------------------------------------------------------- the file

SIGNATURE = b'\x89HDF\r\n\x1a\n'


_WIDTHS = {2: 'H', 4: 'I', 8: 'Q'}              # sizes of offsets and lengths, as struct has them
VLEN_MANY = 256                 # values from which a kernel is worth its tables of a collection
VLEN_JUMPS = 4                  # ... unless the next value is in another collection this often


def object_array(items, n):
    """A one-dimensional object array of a list, its items left as they are
    (numpy would make a table of equal arrays), without a Python step for each."""
    try:
        return np.fromiter(items, object, n)
    except (TypeError, ValueError):                     # (numpy before 1.23)
        out = np.empty(n, object)
        for i, x in enumerate(items):
            out[i] = x
        return out


def find_superblock(b, need=None):
    """The superblock's offset (0, 512, 1024, 2048, ...) or None."""
    n = len(b)
    at = 0
    while at + 8 <= n:
        if need is not None:
            need(at, 8)
        if bytes(b[at:at + 8]) == SIGNATURE:
            return at
        at = 512 if at == 0 else at * 2
    return None


class H5File:
    """The HDF5 structure of a file mapped (or held) in `buf`.

    source: a remote file (`_remote.Source`) whose bytes arrive as asked
    for: every structure is asked for (`need`) before it is read."""

    def __init__(self, buf, source=None):
        self.buf = buf
        self.b = b = memoryview(buf).cast('B') if not isinstance(buf, memoryview) else buf
        self.source = source
        if source is not None:
            self.need, self.prefetch = source.need, source.prefetch
        sb = find_superblock(b, self.need if source is not None else None)
        if sb is None:
            raise FormatError('not an HDF5 file')
        self.need(sb, 256)
        v = b[sb + 8]
        self.sb_version = v
        self._closed = False
        self.superblock = sb                # (where it is: past a user block of this size)
        self.readers = []               # of datasets (see close)
        self._headers = {}
        self._attr_refs = {}
        self._attr_types = {}           # (shared, datatype bytes, dataspace bytes) -> parsed
        self._pipelines = {}            # filter message bytes -> [Filter]
        self._bt2_cache = {}
        self._compact = {}
        self._fheaps = {}
        self._gheaps = {}
        if v in (0, 1):
            self.O, self.L = b[sb + 13], b[sb + 14]
            p = sb + 24 + (4 if v == 1 else 0)
            O = self.O
            self.base = _uint(b, p, O)
            p += 4 * O                                  # base, free space, EOF, driver
            # root group symbol table entry (the name's place in lengths' width,
            # as the library writes it: not as the format's document says)
            p += self.L
            root = _uint(b, p, O)
            cache = _u32(b, p + O)[0]
            self.root_cache = None
            if cache == 1:
                self.root_cache = (_uint(b, p + O + 8, O), _uint(b, p + 2 * O + 8, O))
        elif v in (2, 3):
            self.O, self.L = b[sb + 9], b[sb + 10]
            O = self.O
            p = sb + 12
            self.base = _uint(b, p, O)
            self.extension = _uint(b, p + O, O)         # (a header of file-wide messages)
            root = _uint(b, p + 3 * O, O)
            self.root_cache = None
        else:
            raise FormatError('superblock version %d is not supported' % v)
        self.undef = (1 << (8 * self.O)) - 1
        self._sohm = None
        if v < 2:
            self.extension = self.undef
        self.root_address = root
        self.root = Group(self, root, '/')
        self.path = None                            # set by the opener for local files
        self._pio = None

    def close(self):
        """Close the positional reader, if one was made, and the files that
        virtual datasets and external storage read from, and let go of what
        was kept decoded."""
        from . import _hold, _quick
        _hold.forget(self)
        _quick.forget()
        r, self._pio = self._pio, None
        if r is not None:
            r.close()
        # nothing here may keep the file mapped: what still holds a dataset
        # (x = f['a'] outlives a `with`) would hold it for as long, gigabytes in
        # memory, and on Windows a file that cannot be deleted
        for r in self.readers:
            r.src = None
            if r._index is not None and hasattr(r._index, '_find'):
                r._index._find = None
            v = r.ds.__dict__.get('_vds')
            if v is not None:
                v.close()
        linked = self.__dict__.pop('linked', None)
        if linked:
            from . import _vds
            _vds.close_linked(linked)
        self.readers = []
        for cache in (self._attr_refs, self._compact, self._fheaps, self._gheaps, self._bt2_cache):
            cache.clear()
        # (a released view, not None: anything parsed late, from a read already
        # under way on another thread, fails with ValueError as a closed file's should)
        closed = memoryview(b'')
        closed.release()
        self.b, self.buf, self._closed = closed, None, True

    # -- remote files: bytes asked for before they are read (local: no-ops)
    def need(self, at, n):
        """Bytes [at, at + n) are about to be read."""

    def prefetch(self, addrs, n):
        """need(a, n) for each a of addrs, in one go."""

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
        if self._closed:
            raise ValueError('the file is closed')
        at = self.addr(address)
        remote = self.source is not None
        if remote:
            self.need(at, 64)
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
            if remote:
                self.need(p, size + 4)
            self._v2_messages(p, p + size, flags, msgs)
        elif b[at] == 1:
            version = 1
            flags = 0
            nmsg = _u16(b, at + 2)[0]
            size = _u32(b, at + 8)[0]
            if remote:
                self.need(at + 16, size)
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
                    self.need(self.addr(self.offset(data)), self.length(data + self.O))
                    todo.append((self.addr(self.offset(data)),
                                 self.addr(self.offset(data)) + self.length(data + self.O)))
                elif mtype:
                    msgs.append(self._message(mtype, mflags, data, size, None))
                p = data + size

    def _message(self, mtype, mflags, data, size, corder):
        """A message as read from a header. One kept in the file's shared heap
        (SOHM) as if it lay there, so that all that reads one goes on as usual."""
        if mflags & 2 and self.b[data] == 3 and self.b[data + 1] == 1:
            at, n = self._shared_heap(mtype).location(data + 2)
            return mtype, mflags & ~2, at, n, corder
        return mtype, mflags, data, size, corder

    def _shared_bytes(self, b, p, mtype):
        """A message kept in the shared heap, whose reference is at b[p:]
        (inside an attribute, which may itself lie in a heap)."""
        return self._shared_heap(mtype).object(bytes(b[p + 2:p + 10]))

    def _shared_heap(self, mtype):
        """The heap that keeps shared messages of a type."""
        heaps = self._sohm
        if heaps is None:
            heaps = self._sohm = {}                     # (the table's own messages are not shared)
            m = self.object_header(self.extension).first(0x0F) if self.extension != self.undef else None
            if m is not None:
                b, O = self.b, self.O
                q = self.addr(self.offset(m[2] + 1))
                count = b[m[2] + 1 + O]
                if bytes(b[q:q + 4]) != b'SMTB':
                    raise FormatError('bad shared message table')
                q += 4
                for _ in range(count):                  # version, kind, message types, sizes, index, heap
                    heaps[_u16(b, q + 2)[0]] = self.offset(q + 14 + O)
                    q += 14 + 2 * O
        bit = 5 if mtype == 4 else mtype                # (the old fill message goes with the new)
        for flags, heap in heaps.items():
            if flags >> bit & 1:
                return self.fractal_heap(heap)
        raise FormatError('a shared message of type %d with no heap' % mtype)

    def _dense(self, heap, rec):
        """An attribute's bytes from its record in a name index, in the shared
        heap if flags (byte 8) say it is shared."""
        return (self._shared_heap(0x0C) if rec[8] & 2 else heap).object(rec[:8])

    def _v2_messages(self, p, end, hflags, msgs):
        b = self.b
        corder = bool(hflags & 4)
        hdr = 6 if corder else 4
        unpack = _MSG6 if corder else _MSG4             # one call per message header
        todo = [(p, end)]
        while todo:
            p, end = todo.pop(0)
            while p + hdr <= end:
                if corder:
                    mtype, size, mflags, co = unpack(b, p)
                else:
                    mtype, size, mflags = unpack(b, p)
                    co = None
                data = p + hdr
                if mtype == 0x10:
                    a = self.addr(self.offset(data))
                    n = self.length(data + self.O)
                    self.need(a, n)
                    if bytes(b[a:a + 4]) != b'OCHK':
                        raise FormatError('bad continuation block at %d' % a)
                    todo.append((a + 4, a + n - 4))      # less signature and checksum
                elif mtype:
                    msgs.append((mtype, mflags, data, size, co) if not mflags & 2
                                else self._message(mtype, mflags, data, size, co))
                p = data + size

    def message_filters(self, m):
        """The filters a message holds (a file's variables share a few
        pipelines, and parsing one took 16 us for each of WRF's 145)."""
        key = bytes(self.b[m[2]:m[2] + m[3]])
        got = self._pipelines.get(key)
        if got is None:
            got = self._pipelines[key] = _parse_filters(self.b, m[2])
        return got

    # -- shared / committed messages
    def message_datatype(self, m):
        """The datatype a message holds, following a shared (committed) one
        (each distinct encoding parsed once per file, as for attributes)."""
        key = (m[1] & 2, bytes(self.b[m[2]:m[2] + m[3]]))
        t = self._attr_types.get(key)
        if t is None:
            t = self._attr_types[key] = self.datatype_at(self.b, m[2], bool(m[1] & 2))
        return t

    def datatype_at(self, b, p, shared):
        """The datatype at b[p:], or the committed one a shared message there
        points to."""
        if shared:
            version = b[p]
            if version == 1:                            # (a symbol table entry: heap field first)
                a = _uint(b, p + 8 + self.L, self.O)
            else:
                if version == 3 and b[p + 1] == 1:
                    return parse_datatype(self._shared_bytes(b, p, 3), 0)[0]
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
            lay.data = (self.offset(p), _u32(b, p + O)[0])  # the mappings' global heap object
        else:
            raise FormatError('unknown layout class %d' % cls)
        return lay

    def parse_external(self, m):
        """The External Data Files message (IV.A.2.g): [(name, offset in
        that file, size)], names from the message's local heap."""
        b = self.b
        p = m[2]
        O, L = self.O, self.L
        used = _u16(b, p + 6)[0]
        heap = self.addr(self.offset(p + 8))
        self.need(heap, 8 + 2 * L + O)
        if bytes(b[heap:heap + 4]) != b'HEAP':
            raise FormatError('bad local heap of an external file list')
        data = self.addr(self.offset(heap + 8 + 2 * L))
        self.need(data, self.length(heap + 8))
        out = []
        p += 8 + O
        for _ in range(used):
            name = self._cstr(data + self.length(p))
            out.append((name, self.length(p + L), self.length(p + 2 * L)))
            p += 3 * L
        return out

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

    def fill_undefined(self, h):
        """Whether the fill value is undefined, as HDF5 decodes the message,
        rather than the default (zeros): h5py's fillvalue then raises."""
        b = self.b
        m = h.first(5)
        if m is None:
            return False
        p = m[2]
        if b[p] in (1, 2):
            return not b[p + 3]
        return bool(b[p + 1] & 16)

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
            if self.source is not None:            # remote: the next ones in one go
                self.prefetch([self.addr(t) for k, t in out.values() if k == 'hard'], 1024)
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
        if self.source is not None:            # remote: the next ones in one go
            self.prefetch([self.addr(l[2]) for l in links if l[1] == 'hard'], 1024)
        return {l[0]: (l[1], l[2]) for l in links}

    def find_link(self, g, name):
        """(kind, target) of the link called `name` (ASCII), or None -- the
        way the library finds one, without reading its siblings: down the
        symbol table's B-tree by name, or the dense name index by hash."""
        h = g.header
        st = h.first(0x11)
        if st is not None or (g.address == self.root_address and self.root_cache
                              and h.first(0x02) is None and not h.all(0x06)):
            if st is not None:
                btree, heap = self.offset(st[2]), self.offset(st[2] + self.O)
            else:
                btree, heap = self.root_cache
            return self._symbol_link(btree, heap, name)
        for m in h.all(0x06):
            l = self._link(m[2])
            if l[0] == name:
                return l[1], l[2]
        li = h.first(0x02)
        if li is None:
            return None
        p = li[2]
        p += 2 + (8 if self.b[p + 1] & 1 else 0)
        fheap = self.offset(p)
        name_bt = self.offset(p + self.O)
        if fheap == self.undef:
            return None
        heap = self.fractal_heap(fheap)
        if _HASH_OK[0] is None:
            # once per process: lookup3 must give the hash the file stores
            recs = self.btree2_records(name_bt)
            if not recs:
                return None
            nm = self._link_from(heap.object(recs[0][4:4 + heap.id_len]), 0)[0]
            _HASH_OK[0] = lookup3(nm.encode('utf-8')) == _u32(recs[0], 0)[0]
        if not _HASH_OK[0]:
            return g.links.get(name)
        want = _HASHES.get(name)
        if want is None:
            want = _HASHES[name] = lookup3(name.encode('ascii'))
        for rec in self.btree2_hashed(name_bt, want):
            l = self._link_from(heap.object(rec[4:4 + heap.id_len]), 0)
            if l[0] == name:
                return l[1], l[2]
        return None

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
        if kind == 64:                                  # a file and a path in it, each ended by a NUL
            n = _u16(b, p)[0]
            there = bytes(b[p + 3:p + 2 + n]).split(b'\0')
            return (name, 'external', (there[0].decode('utf-8', 'replace'), there[1].decode('utf-8', 'replace')), corder)
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
        self.need(ha, 8 + 2 * self.L + self.O)
        if bytes(b[ha:ha + 4]) != b'HEAP':
            raise FormatError('bad local heap')
        data = self.addr(self.offset(ha + 8 + 2 * self.L))
        self.need(data, self.length(ha + 8))
        names = bytes(b[data:data + self.length(ha + 8)])
        out = {}
        O = self.O
        stack = [btree]
        while stack:
            a = self.addr(stack.pop())
            self.need(a, 8 + 2 * O)
            if bytes(b[a:a + 4]) != b'TREE':
                raise FormatError('bad v1 B-tree node')
            level = b[a + 5]
            n = _u16(b, a + 6)[0]
            p = a + 8 + 2 * O + self.L                  # first child after key 0
            self.need(a, 8 + 2 * O + (n + 1) * (O + self.L))
            if self.source is not None:            # remote: the next ones in one go
                self.prefetch([self.addr(self.offset(p + i * (O + self.L))) for i in range(n)], 1024)
            for i in range(n):
                child = self.offset(p)
                p += O + self.L
                if level:
                    stack.append(child)
                    continue
                s = self.addr(child)
                self.need(s, 8)
                if bytes(b[s:s + 4]) != b'SNOD':
                    raise FormatError('bad symbol table node')
                ns = _u16(b, s + 6)[0]
                L = self.L
                self.need(s, 8 + ns * (L + O + 24))
                entry = struct.Struct('<%s%sII4s12x' % (_WIDTHS[L], _WIDTHS[O]))
                for at, target, cache, _, link in entry.iter_unpack(b[s + 8:s + 8 + ns * entry.size]):
                    name = names[at:names.find(b'\0', at)].decode('utf-8', 'replace')
                    if cache == 2:                      # a soft link
                        out[name] = ('soft', self._cstr(data + _u32(link, 0)[0]))
                    else:
                        out[name] = ('hard', target)
        return out

    def _symbol_link(self, btree, heap, name):
        """One name of a symbol table, by bisection down its B-tree, whose
        keys are names in the local heap: (kind, target), or None."""
        b = self.b
        O, L = self.O, self.L
        ha = self.addr(heap)
        self.need(ha, 8 + 2 * L + O)
        if bytes(b[ha:ha + 4]) != b'HEAP':
            raise FormatError('bad local heap')
        data = self.addr(self.offset(ha + 8 + 2 * L))
        self.need(data, self.length(ha + 8))
        want = name.encode('ascii') + b'\0'
        n = len(want)
        a = self.addr(btree)
        while True:
            self.need(a, 8 + 2 * O)
            if bytes(b[a:a + 4]) != b'TREE':
                raise FormatError('bad v1 B-tree node')
            level = b[a + 5]
            lo, hi = 0, _u16(b, a + 6)[0]
            keys = a + 8 + 2 * O
            self.need(a, 8 + 2 * O + (hi + 1) * (O + L))
            top = hi
            while lo < hi:                              # child i: names up to key i + 1
                mid = (lo + hi) // 2
                # (a name's first n bytes sort as the name does: NUL is the least byte)
                q = data + self.length(keys + (mid + 1) * (O + L))
                if bytes(b[q:q + n]) < want:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == top:
                return None
            child = self.addr(self.offset(keys + lo * (O + L) + L))
            if not level:
                break
            a = child
        self.need(child, 8)
        if bytes(b[child:child + 4]) != b'SNOD':
            raise FormatError('bad symbol table node')
        count = _u16(b, child + 6)[0]
        self.need(child, 8 + count * (L + O + 24))
        e = child + 8
        for _ in range(count):
            q = data + self.length(e)
            if bytes(b[q:q + n]) == want:
                if _u32(b, e + L + O)[0] == 2:          # cache type 2: a soft link
                    return 'soft', self._cstr(data + _u32(b, e + L + O + 8)[0])
                return 'hard', self.offset(e + L)
            e += L + O + 24
        return None

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
                    obj = self._dense(heap, rec)
                    corder = _u32(rec, 9)[0] if flags & 1 else base + k
                    out.append((self._attr_name(obj, 0), obj, 0, corder))
        out.sort(key=lambda r: r[3])
        refs = {r[0]: r[1:] for r in out}
        self._attr_refs[h.address] = refs
        return refs

    @staticmethod
    def _attr_name_bytes(b, p):
        version = b[p]
        nlen = b[p + 2] | (b[p + 3] << 8)
        q = p + 8 + (1 if version == 3 else 0)
        return bytes(b[q:q + nlen]).split(b'\0', 1)[0]

    @staticmethod
    def _attr_name(b, p):
        version = b[p]
        nlen = _u16(b, p + 2)[0]
        q = p + 8 + (1 if version == 3 else 0)
        return bytes(b[q:q + nlen]).split(b'\0')[0].decode('utf-8', 'replace')

    def extent(self, address):
        """(shape, maxshape) of the object at address (as _parse_dataspace),
        or None if it has none (a group, a type): without reading the rest
        of its header where its dataspace comes first, as it usually does."""
        h = self._headers.get(address)
        if h is None and self.source is None:
            b = self.b
            at = self.addr(address)
            if bytes(b[at:at + 4]) == b'OHDR':
                flags = b[at + 5]
                p = at + 6 + (16 if flags & 32 else 0) + (4 if flags & 16 else 0)
                n = 1 << (flags & 3)
                end = p + n + _uint(b, p, n)
                p += n
                hdr, unpack = (6, _MSG6) if flags & 4 else (4, _MSG4)
                while p + hdr <= end:
                    m = unpack(b, p)
                    if m[0] == 1 and not m[2] & 2:  # (shared: not here)
                        return _parse_dataspace(b, p + hdr, self.L)
                    if m[0] in (1, 0x10):           # (the rest is elsewhere)
                        break
                    p += hdr + m[1]
        h = self.object_header(address)
        m = h.first(1)
        return None if m is None else _parse_dataspace(self.b, m[2], self.L)

    def may_have_attribute(self, h, name):
        """False only if header h surely has no attribute called `name`
        (bytes, with the NUL that ends it): the file's own bytes searched,
        attribute by attribute, where a name is always inline. True whenever
        it cannot tell: dense or shared attributes, a buffer without find()."""
        b = self.b
        whole = getattr(b, 'obj', b)                # (a view of the mapping: the mapping)
        find = getattr(whole, 'find', None)
        if find is None or len(whole) != len(b):
            return True
        for m in h.messages:
            t = m[0]
            if t == 0x0C:
                if m[1] & 2 or find(name, m[2], m[2] + m[3]) >= 0:
                    return True
            elif t == 0x15:                         # (always there in netCDF-C's headers)
                p = m[2]
                if self.offset(p + 2 + (2 if b[p + 1] & 1 else 0)) != self.undef:
                    return True                     # attributes in a heap
        return False

    def find_attribute(self, h, name):
        """(buffer, position, creation order) of attribute `name`, or None --
        without reading every attribute's name: compact attributes are
        scanned in the header; dense ones are looked up in the name index
        by their lookup3 hash, decoding only the heap objects whose hash
        matches (the check netCDF does on every dataset at open --
        CLASS, NAME, DIMENSION_LIST ... -- read 770 heap objects of
        Metop-SG's 149 datasets before)."""
        refs = self._attr_refs.get(h.address)
        if refs is not None:
            return refs.get(name)
        b = self.b
        compact = self._compact.get(h.address)
        if compact is None:
            # (names as raw bytes up to their NUL: nothing decoded at open)
            compact = self._compact[h.address] = {
                self._attr_name_bytes(b, m[2]): (b, m[2], m[4] if m[4] is not None else i)
                for i, m in enumerate(h.all(0x0C))}
        key = _NAME_BYTES.get(name)
        if key is None:
            key = _NAME_BYTES[name] = name.encode('utf-8')
        r = compact.get(key)
        if r is not None:
            return r
        ai = h.first(0x15)
        if ai is None:
            return None
        p = ai[2]
        flags = b[p + 1]
        p += 2 + (2 if flags & 1 else 0)
        fheap = self.offset(p)
        name_bt = self.offset(p + self.O)
        if fheap == self.undef:
            return None
        if _HASH_OK[0] is None:
            # once per process: lookup3 must give the hash the file stores
            recs = self.btree2_records(name_bt)
            if not recs:
                return None
            heap = self.fractal_heap(fheap)
            obj = self._dense(heap, recs[0])
            nm = self._attr_name(obj, 0).encode('utf-8')
            _HASH_OK[0] = lookup3(nm) == _u32(recs[0], 13)[0]
        if not _HASH_OK[0]:
            return self.attribute_refs(h).get(name)
        want = _HASHES.get(name)
        if want is None:
            want = _HASHES[name] = lookup3(name.encode('utf-8'))
        # (a read asks for some nine names, most of which are not there: the
        # name index once, as a dict, and each asks that)
        hashes = self._bt2_cache.get(name_bt)
        if hashes is None:
            hashes = {}
            for k, rec in enumerate(self.btree2_records(name_bt)):
                hashes.setdefault(_u32(rec, 13)[0], []).append((k, rec))
            self._bt2_cache[name_bt] = hashes
        heap = None
        base = len(h.all(0x0C))
        for k, rec in hashes.get(want, ()):
            if heap is None:
                heap = self.fractal_heap(fheap)
            obj = self._dense(heap, rec)
            if self._attr_name(obj, 0) == name:
                return (obj, 0, _u32(rec, 9)[0] if flags & 1 else base + k)
        return None

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
                    corder = _u32(rec, 9)[0] if flags & 1 else base + k
                    out.append(self._attribute(self._dense(heap, rec), 0, corder))
        out.sort(key=lambda a: a.corder)
        return {a.name: a for a in out}

    def _attribute(self, b, p, corder):
        version, flags, nlen, tlen, slen = _ATTR_HEAD.unpack_from(b, p)
        p += 9 if version == 3 else 8
        if version == 1:                            # (fields padded to 8 bytes)
            pn, pt, ps = (nlen + 7) & ~7, (tlen + 7) & ~7, (slen + 7) & ~7
        else:
            pn, pt, ps = nlen, tlen, slen
        name = bytes(b[p:p + nlen]).split(b'\0', 1)[0].decode('utf-8', 'replace')
        p += pn
        # (a file's attributes mostly repeat a few datatypes and dataspaces:
        # each distinct encoding is parsed once -- a Datatype is not changed
        # once parsed, so attributes can share one)
        sp = p + pt
        key = (flags & 3, bytes(b[p:sp + slen]), tlen)
        ts = self._attr_types.get(key)
        if ts is None:
            if flags & 2:                               # (a dataspace kept in the shared heap)
                shape = _parse_dataspace(self._shared_bytes(b, sp, 1), 0, self.L)[0]
            else:
                shape = _parse_dataspace(b, sp, self.L)[0]
            n = 0
            if shape is not None:
                n = 1
                for s in shape:
                    n *= s
            ts = self._attr_types[key] = (self.datatype_at(b, p, bool(flags & 1)), shape, n)
        t, shape, n = ts
        p = sp + ps
        raw = bytes(b[p:p + n * t.size])
        return Attribute(name, t, shape, raw, corder, self)

    # -- values of special types
    def _vlen_many(self, t, n, raw, strings):
        """Many variable-length values at once: packed by a kernel, then cut,
        with nothing worked out per value but the cut. An object array, or None
        where this does not do. strings: bytes, if that is what is wanted of them
        (going through str and back was most of it, and not always the same bytes)."""
        from . import _chunks
        bt = t.base
        if (n < VLEN_MANY or t.size != 16 or self.O != 8 or self.L != 8 or self.source is not None
                or (not t.vlen_string and (bt.dtype is None or bt.cls == Datatype.VLEN or bt.holds_vlen()))
                or not _chunks.loaded()):
            return None
        src = np.frombuffer(self.b, np.uint8)
        # (values written together lie together: only a read jumping about is left to go value by value)
        coll = np.ndarray(n, '<u8', raw, 4, (16,))
        if np.count_nonzero(coll[1:] != coll[:-1]) * VLEN_JUMPS > n:
            return None
        width = 1 if t.vlen_string else bt.size
        total = int(np.ndarray(n, '<u4', raw, 0, (16,)).sum(dtype=np.uint64)) * width
        if total > src.size:
            return None
        pack = np.empty(total, np.uint8)
        ends = np.empty(n, np.int64)
        got = _chunks.vlen_pack(src, np.frombuffer(raw, np.uint8), n, 16, self.base, width, pack, ends)
        if got < 0:
            return None
        e = ends.tolist()
        s = [0] + e[:-1]
        if t.vlen_string:
            data = pack[:got].tobytes()
            if strings is bytes:
                items = [data[a:b] for a, b in zip(s, e)]
            elif data.isascii():
                data = data.decode('ascii')
                items = [data[a:b] for a, b in zip(s, e)]
            else:
                codec = 'utf-8' if t.charset == 1 else 'latin-1'
                items = [data[a:b].decode(codec, 'replace') for a, b in zip(s, e)]
        else:
            values = pack[:got].view(bt.dtype)
            items = [values[a // width:b // width] for a, b in zip(s, e)]
        return object_array(items, n)

    def decode_values(self, t, shape, raw, strings=str):
        """numpy values from raw bytes of datatype t (vlen through the global heap)."""
        if shape is None:
            return None
        n = 1
        for s in shape:
            n *= s
        if t.cls == Datatype.VLEN:
            out = self._vlen_many(t, n, raw, strings)
            if out is not None:
                return out.reshape(shape)
            out = np.empty(n, object)
            O = self.O
            if self.source is not None:
                self.prefetch(sorted({self.addr(_uint(raw, i * t.size + 4, O)) for i in range(n)}), 4096)
            for i in range(n):
                q = i * t.size
                ln = _u32(raw, q)[0]
                coll = _uint(raw, q + 4, O)
                idx = _u32(raw, q + 4 + O)[0]
                data = self.global_heap_object(coll, idx) if coll else b''
                if t.vlen_string:
                    s = bytes(data[:ln])
                    out[i] = s if strings is bytes else s.decode('utf-8' if t.charset == 1 else 'latin-1', 'replace')
                else:
                    bt = t.base
                    out[i] = (self.decode_values(bt, (ln,), bytes(data[:ln * bt.size]))
                              if bt.dtype is None or bt.cls == Datatype.VLEN
                              else np.frombuffer(bytes(data[:ln * bt.size]), bt.dtype).copy())
            return out.reshape(shape)
        if t.cls == Datatype.ARRAY and t.base.holds_vlen():
            k = 1
            for s in t.dims:
                k *= s
            return self.decode_values(t.base, (n * k,), raw).reshape(tuple(shape) + t.dims)
        if t.odd():
            from . import _h5read
            if t.dtype is None and t.cls == Datatype.FLOAT:
                return _h5read.odd_float(np.frombuffer(raw, 'V%d' % t.size, n), t).reshape(shape)
            if t.dtype is None and t.size < 8:
                return _h5read.odd_int(np.frombuffer(raw, 'V%d' % t.size, n), t).reshape(shape)
            if t.dtype is not None and _h5read.reduced(t):
                return _h5read.convert_reduced(np.frombuffer(raw, t.dtype, n), t).reshape(shape)
        if t.dtype is None:
            raise FormatError('cannot decode datatype class %d' % t.cls)
        a = np.frombuffer(raw, t.dtype, n).reshape(tuple(shape) + (t.dims if t.cls == Datatype.ARRAY else ()))
        return self.decode_members(t, a)

    def decode_members(self, t, a):
        """Compound values with the variable-length values in them decoded,
        as object fields of a new array; others as they are."""
        if t.cls != Datatype.COMPOUND or not t.holds_vlen():
            return a
        fields = []
        for name, _, mt in t.members:
            v = a[name]
            if mt.cls == Datatype.COMPOUND:
                v = self.decode_members(mt, v)
            elif mt.holds_vlen():
                # (bytes as stored: a read has made the numbers around them native)
                stored = (mt.base if mt.cls == Datatype.ARRAY else mt).dtype
                v = self.decode_values(mt, v.shape[:a.ndim] if mt.cls == Datatype.ARRAY else v.shape,
                                       (v if stored is None else v.astype(stored)).tobytes())
            fields.append((name, v))
        out = np.empty(a.shape, np.dtype([(n, v.dtype, v.shape[a.ndim:]) for n, v in fields]))
        for n, v in fields:
            out[n] = v
        return out

    def vlen_refs(self, a):
        """For an attribute of variable-length sequences of object references
        (DIMENSION_LIST): the first referenced address of each element, or
        None for an empty one -- without building numpy arrays."""
        t = a.dtype
        O = self.O
        out = []
        raw = a.raw
        n = len(raw) // t.size if t.size else 0
        if self.source is not None:
            self.prefetch(sorted({self.addr(_uint(raw, i * t.size + 4, O)) for i in range(n)}), 4096)
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
            self.need(a, 8 + self.L)
            if bytes(b[a:a + 4]) != b'GCOL':
                raise FormatError('bad global heap collection')
            size = self.length(a + 8)
            self.need(a, size)
            head = (8 + self.L + 7) // 8 * 8            # (headers keep to multiples of 8)
            p = a + head
            end = a + size
            objs = {}
            while p + head <= end:
                i = _u16(b, p)[0]
                if i == 0:
                    break
                n = self.length(p + 8)
                objs[i] = (p + head, n)
                p += head + (n + 7) // 8 * 8
            self._gheaps[coll] = objs
        p, n = objs[idx]
        return self.b[p:p + n]

    # -- fractal heaps and v2 B-trees
    def fractal_heap(self, address):
        h = self._fheaps.get(address)
        if h is None:
            h = self._fheaps[address] = FractalHeap(self, address)
        return h

    def _bt2_shape(self, address):
        """What walking a v2 B-tree takes: (node size, record size, depth,
        root, records in root, bytes of a child's record count, bytes of
        each depth's "records in child" fields (H5B2__hdr_init))."""
        got = self._bt2_cache.get(('shape', address))
        if got is None:
            got = self._bt2_cache['shape', address] = self._bt2_read_shape(address)
        return got

    def _bt2_read_shape(self, address):
        b = self.b
        a = self.addr(address)
        self.need(a, 22 + self.O + self.L)
        if bytes(b[a:a + 4]) != b'BTHD':
            raise FormatError('bad v2 B-tree header')
        node_size = _u32(b, a + 6)[0]
        rsize = _u16(b, a + 10)[0]
        depth = _u16(b, a + 12)[0]
        root = self.offset(a + 16)
        nroot = _u16(b, a + 16 + self.O)[0]
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
        return node_size, rsize, depth, root, nroot, max_nrec_size, cum_size

    def btree2_hashed(self, address, want, at=0):
        """The records of a v2 B-tree of names whose hash, a u32 at byte `at`
        of the records they sort by, is `want`: only the way down to them."""
        node_size, rsize, depth, root, nroot, max_nrec_size, cum_size = self._bt2_shape(address)
        out = []
        if root == self.undef:
            return out
        b = self.b

        def node(addr, nrec, d):
            q = self.addr(addr)
            self.need(q, node_size)
            p = q + 6
            lo, hi = 0, nrec
            while lo < hi:
                mid = (lo + hi) // 2
                if _u32(b, p + mid * rsize + at)[0] < want:
                    lo = mid + 1
                else:
                    hi = mid
            end = lo
            while end < nrec and _u32(b, p + end * rsize + at)[0] == want:
                end += 1
            if d == 0:
                out.extend(bytes(b[p + i * rsize:p + (i + 1) * rsize]) for i in range(lo, end))
                return
            kids = p + nrec * rsize
            stride = self.O + max_nrec_size + (cum_size[d - 1] if d > 1 else 0)
            for i in range(lo, end + 1):                # (equal hashes may lie either side)
                k = kids + i * stride
                node(self.offset(k), _uint(b, k + self.O, max_nrec_size), d - 1)
                if i < end:
                    out.append(bytes(b[p + i * rsize:p + (i + 1) * rsize]))

        node(root, nroot, depth)
        node = None                                     # (it holds itself, and the file with it)
        return out

    def btree2_records(self, address):
        """Every record of a v2 B-tree (bytes objects), in key order."""
        node_size, rsize, depth, root, nroot, max_nrec_size, cum_size = self._bt2_shape(address)
        if root == self.undef:
            return []
        b = self.b
        out = []

        def node(addr, nrec, d):
            q = self.addr(addr)
            self.need(q, node_size)
            p = q + 6
            if d == 0:
                for i in range(nrec):
                    out.append(bytes(b[p + i * rsize:p + (i + 1) * rsize]))
                return
            recs = [bytes(b[p + i * rsize:p + (i + 1) * rsize]) for i in range(nrec)]
            p += nrec * rsize
            if self.source is not None:            # remote: the next ones in one go
                stride = self.O + max_nrec_size + (cum_size[d - 1] if d > 1 else 0)
                self.prefetch([self.addr(self.offset(p + i * stride)) for i in range(nrec + 1)], node_size)
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
        node = None                                     # (it holds itself, and the file with it)
        return out

    # -- chunk indexes
    def _no_chunks(self, lay):
        """Whether a dataset's index holds nothing: none was made, or a dataset
        shrunk until no chunk was left keeps its array's header and that alone."""
        if lay.address == self.undef:
            return True
        if lay.index in ('farray', 'earray'):
            at = self.addr(lay.address) + (8 + self.L if lay.index == 'farray' else 12 + 6 * self.L)
            self.need(at, self.O)
            return self.offset(at) == self.undef
        return False

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
        if self._no_chunks(lay):
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

    def chunk_arrays(self, ds):
        """The chunk table as arrays (offsets (n, rank) int64, absolute
        addresses, stored sizes, filter masks), or None where only the
        tuple form exists. v1 B-tree leaves are fixed-size records, read a
        node at a time with numpy: Metop-SG's 39 datasets of 840 row chunks
        took 37 ms to index tuple by tuple on every open."""
        lay = ds.layout
        if self._no_chunks(lay):
            z = np.zeros(0, np.int64)
            return np.zeros((0, len(ds.shape)), np.int64), z, z, z
        if self.O != 8 or self.source is not None and lay.index != 'btree1':
            return None
        if lay.index == 'btree1':
            return self._btree1_arrays(lay.address, len(ds.shape))
        rank = len(ds.shape)
        chunks = tuple(lay.chunks)
        csize = ds.type.size * math.prod(chunks)
        grid = [-(-(m if m is not None else s) // c) for s, m, c in zip(ds.shape, ds.maxshape, chunks)]
        if lay.index == 'btree2':
            return self._btree2_arrays(lay.address, rank, chunks, bool(ds.filters), csize)
        if lay.index == 'farray':
            raw, esize, filtered = self._farray_raw(lay.address)
            return self._grid_arrays(ds, grid, chunks, raw, esize, filtered, csize, list(range(rank)))
        if lay.index == 'earray':
            e = self._earray_info(ds, chunks)
            grid = [e.counts[d] for d in range(rank)]
            return self._grid_arrays(ds, grid, chunks, self._earray_raw(e), e.esize, e.filtered, csize, e.order)
        return None

    def _grid_arrays(self, ds, grid, chunks, raw, esize, filtered, csize, order):
        """Index arrays from a run of elements, one for each place on the grid
        (dimensions in `order`, the first slowest). Numpy over all of them: tuple
        by tuple a million took two seconds, to read what a read then skips."""
        O = self.O
        n = len(raw) // esize
        a = np.frombuffer(raw, np.uint8).reshape(n, esize)
        addr = np.ndarray(n, '<u8', raw, 0, (esize,))
        keep = np.nonzero(addr != self.undef)[0]
        at = np.array(np.unravel_index(keep, [grid[d] for d in order]), np.int64)
        offs = np.empty((len(keep), len(chunks)), np.int64)
        for k, d in enumerate(order):
            offs[:, d] = at[k] * chunks[d]
        inside = np.nonzero((offs < np.array(ds.shape, np.int64)).all(axis=1))[0]
        offs, keep = offs[inside], keep[inside]
        z = addr[keep].astype(np.int64) + self.base
        if not filtered:
            return offs, z, np.full(len(keep), csize, np.int64), np.zeros(len(keep), np.int64)
        width = esize - O - 4
        return offs, z, _uints(a[keep], O, width), _uints(a[keep], O + width, 4)

    def _btree2_arrays(self, address, rank, chunks, filtered, csize):
        node_size, rsize, depth, root, nroot, nsize, cum = self._bt2_shape(address)
        z = np.zeros(0, np.int64)
        if root == self.undef:
            return np.zeros((0, rank), np.int64), z, z, z
        b, O = self.b, self.O
        parts = []
        stack = [(root, nroot, depth)]
        while stack:                                    # (all records, in no order: nothing asks for it)
            node, nrec, d = stack.pop()
            p = self.addr(node) + 6
            parts.append(b[p:p + nrec * rsize])
            if d:
                k = p + nrec * rsize
                stride = O + nsize + (cum[d - 1] if d > 1 else 0)
                for i in range(nrec + 1):
                    stack.append((self.offset(k), _uint(b, k + O, nsize), d - 1))
                    k += stride
        raw = b''.join(parts)
        n = len(raw) // rsize
        a = np.frombuffer(raw, np.uint8).reshape(n, rsize)
        addr = np.ndarray(n, '<u8', raw, 0, (rsize,)).astype(np.int64) + self.base
        width = rsize - O - 4 - 8 * rank if filtered else 0
        at = O + width + 4 if filtered else O
        offs = np.ndarray((n, rank), '<u8', raw, at, (rsize, 8)).astype(np.int64) * np.array(chunks, np.int64)
        if not filtered:
            return offs, addr, np.full(n, csize, np.int64), np.zeros(n, np.int64)
        return offs, addr, _uints(a, O, width), _uints(a, O + width, 4)

    def _farray_raw(self, address):
        """A fixed array's elements as one run of bytes; (run, size of an
        element, whether they hold sizes and masks). Unwritten pages: undefined."""
        b = self.b
        a = self.addr(address)
        if bytes(b[a:a + 4]) != b'FAHD':
            raise FormatError('bad fixed array header')
        esize = b[a + 6]
        page = 1 << b[a + 7]
        n = self.length(a + 8)
        bits = p = self.addr(self.offset(a + 8 + self.L)) + 6 + self.O
        if bytes(b[p - 6 - self.O:p - 6 - self.O + 4]) != b'FADB':
            raise FormatError('bad fixed array data block')
        if n <= page:
            return bytes(b[p:p + n * esize]), esize, b[a + 5] == 1
        pages = -(-n // page)
        p += (pages + 7) // 8 + 4
        parts = []
        for k in range(pages):
            m = min(page, n - k * page) * esize
            parts.append(b[p:p + m] if _page_written(b, bits, k) else b'\xff' * m)
            p += m + 4
        return b''.join(parts), esize, b[a + 5] == 1

    def chunk_finder(self, ds):
        """A function from a chunk's element offsets to its (absolute address,
        stored size, filter mask), or None if it is not stored, going down
        the index as the library does: for a read of a few chunks of millions.
        None where an index is not looked into that way."""
        lay = ds.layout
        if self.source is not None or self.O != 8 or self._no_chunks(lay):
            return None
        rank = len(ds.shape)
        chunks = tuple(lay.chunks)
        if lay.index == 'btree1':
            return self._btree1_finder(lay.address, rank)
        csize = ds.type.size * math.prod(chunks)
        if lay.index == 'btree2':
            return self._btree2_finder(lay.address, rank, chunks, bool(ds.filters), csize)
        if lay.index == 'farray':
            return self._farray_finder(ds, chunks, csize)
        if lay.index == 'earray':
            return self._earray_finder(ds, chunks, csize)
        return None

    def _earray_info(self, ds, chunks):
        """An extensible array's header (for the structure: _earray_chunks)."""
        b, O = self.b, self.O
        a = self.addr(ds.layout.address)
        if bytes(b[a:a + 4]) != b'EAHD':
            raise FormatError('bad extensible array header')
        e = _EArray()
        e.filtered = b[a + 5] == 1
        e.esize, max_bits, e.iblk_n, dblk_min, sblk_min_ptrs, page_bits = (b[a + 6], b[a + 7], b[a + 8],
                                                                         b[a + 9], b[a + 10], b[a + 11])
        iblk = self.addr(self.offset(a + 12 + 6 * self.L))
        e.nelm = self.length(a + 12 + 4 * self.L) + 1
        e.page = 1 << page_bits
        e.off_size = (max_bits + 7) // 8
        nsblks = 1 + max_bits - _log2(dblk_min)
        e.sb = []
        start_idx = start_dblk = 0
        for s in range(nsblks):
            nd = 1 << (s // 2)
            dn = dblk_min * (1 << ((s + 1) // 2))
            e.sb.append((nd, dn, start_idx, start_dblk))
            start_idx += nd * dn
            start_dblk += nd
        if bytes(b[iblk:iblk + 4]) != b'EAIB':
            raise FormatError('bad extensible array index block')
        e.first = iblk + 6 + O
        e.addrs = e.first + e.iblk_n * e.esize
        e.ib_nsblks = 2 * _log2(sblk_min_ptrs)
        e.sblks = e.addrs + 2 * (sblk_min_ptrs - 1) * O
        # the unlimited dimension goes slowest (see _earray_chunks)
        rank = len(chunks)
        unlim = [i for i, m in enumerate(ds.maxshape) if m is None]
        e.order = [unlim[0]] + [i for i in range(rank) if i != unlim[0]] if unlim else list(range(rank))
        e.counts = [-(-m // c) if m is not None else -(-s // c) for s, m, c in zip(ds.shape, ds.maxshape, chunks)]
        return e

    def _earray_raw(self, e):
        """An extensible array's elements as one run of bytes, in order; what
        is not there, undefined."""
        b, O = self.b, self.O
        esize, page = e.esize, e.page
        parts = [b[e.first:e.first + e.iblk_n * esize]]
        got = e.iblk_n

        def block(da, dn, bits, first=0):
            if da == self.undef:
                parts.append(b'\xff' * (dn * esize))
                return
            d = self.addr(da)
            if bytes(b[d:d + 4]) != b'EADB':
                raise FormatError('bad extensible array data block')
            r = d + 6 + O + e.off_size
            if dn <= page:
                parts.append(b[r:r + dn * esize])
                return
            r += 4
            for k in range(-(-dn // page)):
                m = min(page, dn - k * page) * esize
                parts.append(b[r:r + m] if bits is None or _page_written(b, bits, first + k) else b'\xff' * m)
                r += m + 4

        for s, (nd, dn, _, sdblk) in enumerate(e.sb):
            if got >= e.nelm:
                break
            got += nd * dn
            if s < e.ib_nsblks:
                for k in range(nd):
                    block(self.offset(e.addrs + (sdblk + k) * O), dn, None)
                continue
            sa = self.offset(e.sblks + (s - e.ib_nsblks) * O)
            if sa == self.undef:
                parts.append(b'\xff' * (nd * dn * esize))
                continue
            r = self.addr(sa)
            if bytes(b[r:r + 4]) != b'EASB':
                raise FormatError('bad extensible array super block')
            r += 6 + O + e.off_size
            bits = r
            pages = -(-dn // page)
            if dn > page:                               # (see _earray_finder on these bits)
                r += nd * ((pages + 7) // 8)
            for k in range(nd):
                block(self.offset(r + k * O), dn, bits, k * pages)
        return b''.join(parts)

    def _earray_finder(self, ds, chunks, csize):
        """(See _earray_chunks for the structure: this goes to one element.)"""
        b, O = self.b, self.O
        e = self._earray_info(ds, chunks)
        filtered, esize, iblk_n, nelm = e.filtered, e.esize, e.iblk_n, e.nelm
        page, off_size, sb, first, addrs = e.page, e.off_size, e.sb, e.first, e.addrs
        ib_nsblks, sblks, order, counts = e.ib_nsblks, e.sblks, e.order, e.counts

        def element(q):
            at = _uint(b, q, O)
            if at == self.undef:
                return None
            if filtered:
                width = esize - O - 4
                return self.addr(at), _uint(b, q + O, width), _u32(b, q + O + width)[0]
            return self.addr(at), csize, 0

        def find(place):
            i = 0
            for d in order:
                i = i * counts[d] + place[d] // chunks[d]
            if i >= nelm:
                return None
            if i < iblk_n:
                return element(first + i * esize)
            j = i - iblk_n
            for s, (nd, dn, sidx, sdblk) in enumerate(sb):
                if j < sidx + nd * dn:
                    break
            k, e = divmod(j - sidx, dn)
            if s < ib_nsblks:
                da = self.offset(addrs + (sdblk + k) * O)
            else:
                sa = self.offset(sblks + (s - ib_nsblks) * O)
                if sa == self.undef:
                    return None
                r = self.addr(sa)
                if bytes(b[r:r + 4]) != b'EASB':
                    raise FormatError('bad extensible array super block')
                r += 6 + O + off_size
                if dn > page:
                    # (room for a bitmap a data block, but their bits run on from one to the next)
                    pages = -(-dn // page)
                    if not _page_written(b, r, k * pages + e // page):
                        return None
                    r += nd * ((pages + 7) // 8)
                da = self.offset(r + k * O)
            if da == self.undef:
                return None
            d = self.addr(da)
            if bytes(b[d:d + 4]) != b'EADB':
                raise FormatError('bad extensible array data block')
            r = d + 6 + O + off_size
            if dn <= page:
                return element(r + e * esize)
            return element(r + 4 + e // page * (page * esize + 4) + (e % page) * esize)
        return find

    def _btree1_finder(self, root, rank):
        b, base = self.b, self.base
        keys = struct.Struct('<%dQ' % (rank + 1)).unpack_from
        ksize = 8 + 8 * (rank + 1)
        step = ksize + 8

        def find(place):
            want = place + (0,)
            a = self.addr(root)
            while True:
                if bytes(b[a:a + 4]) != b'TREE':
                    raise FormatError('bad chunk B-tree node')
                level = b[a + 5]
                lo, hi = 0, _u16(b, a + 6)[0]
                p = a + 24
                while lo < hi:                          # the last key not past it
                    mid = (lo + hi) // 2
                    if keys(b, p + mid * step + 8) <= want:
                        lo = mid + 1
                    else:
                        hi = mid
                if not lo:
                    return None
                q = p + (lo - 1) * step
                if not level:
                    if keys(b, q + 8) != want:
                        return None
                    return _u64(b, q + ksize)[0] + base, _u32(b, q)[0], _u32(b, q + 4)[0]
                a = self.addr(_u64(b, q + ksize)[0])
        return find

    def _btree2_finder(self, address, rank, chunks, filtered, csize):
        node_size, rsize, depth, root, nroot, nsize, cum = self._bt2_shape(address)
        b, O = self.b, self.O
        width = rsize - O - 4 - 8 * rank if filtered else 0
        at = O + width + 4 if filtered else O
        keys = struct.Struct('<%dQ' % rank).unpack_from

        def record(p):
            if filtered:
                return self.addr(_uint(b, p, O)), _uint(b, p + O, width), _u32(b, p + O + width)[0]
            return self.addr(_uint(b, p, O)), csize, 0

        def find(place):
            if root == self.undef:
                return None
            want = tuple(x // c for x, c in zip(place, chunks))
            node, nrec, d = root, nroot, depth
            while True:
                p = self.addr(node) + 6
                lo, hi = 0, nrec
                while lo < hi:
                    mid = (lo + hi) // 2
                    k = keys(b, p + mid * rsize + at)
                    if k == want:
                        return record(p + mid * rsize)
                    if k < want:
                        lo = mid + 1
                    else:
                        hi = mid
                if not d:
                    return None
                k = p + nrec * rsize + lo * (O + nsize + (cum[d - 1] if d > 1 else 0))
                node, nrec, d = self.offset(k), _uint(b, k + O, nsize), d - 1
        return find

    def _farray_finder(self, ds, chunks, csize):
        b, O = self.b, self.O
        a = self.addr(ds.layout.address)
        if bytes(b[a:a + 4]) != b'FAHD':
            raise FormatError('bad fixed array header')
        filtered = b[a + 5] == 1
        esize = b[a + 6]
        page = 1 << b[a + 7]
        n = self.length(a + 8)
        bits = p = self.addr(self.offset(a + 8 + self.L)) + 6 + O
        paged = n > page
        if paged:
            p += (-(-n // page) + 7) // 8 + 4           # page init bitmap, checksum
        nmax = [-(-(m if m is not None else s) // c) for s, m, c in zip(ds.shape, ds.maxshape, chunks)]

        def find(place):
            i = 0
            for x, c, m in zip(place, chunks, nmax):
                i = i * m + x // c
            if paged and not _page_written(b, bits, i // page):
                return None
            q = p + (i // page * (page * esize + 4) + (i % page) * esize if paged else i * esize)
            at = _uint(b, q, O)
            if at == self.undef:
                return None
            if filtered:
                width = esize - O - 4
                return self.addr(at), _uint(b, q + O, width), _u32(b, q + O + width)[0]
            return self.addr(at), csize, 0
        return find

    def _btree1_arrays(self, address, rank):
        b = self.b
        O = self.O
        ksize = 8 + 8 * (rank + 1)
        rec = _BT1_REC.get(rank)
        if rec is None:
            rec = _BT1_REC[rank] = np.dtype([('size', '<u4'), ('mask', '<u4'),
                                             ('offs', '<u8', (rank + 1,)), ('child', '<u8')])
        step = rec.itemsize
        parts = []
        stack = [address]
        while stack:
            a = self.addr(stack.pop())
            self.need(a, 8 + 2 * O)
            if bytes(b[a:a + 4]) != b'TREE':
                raise FormatError('bad chunk B-tree node')
            level = b[a + 5]
            n = _u16(b, a + 6)[0]
            p = a + 8 + 2 * O
            self.need(a, 8 + 2 * O + n * (ksize + O) + ksize)
            if level:
                kids = np.ndarray(n, '<u8', b, p + ksize, (step,)).tolist()
                if self.source is not None:        # remote: the next ones in one go
                    self.prefetch([self.addr(k) for k in kids],
                                  8 + 2 * O + max(n, 64) * (ksize + O) + ksize)
                stack.extend(reversed(kids))
            elif n:                                     # (none: all chunks shrunk away)
                parts.append(b[p:p + n * step])
        if not parts:
            z = np.zeros(0, np.int64)
            return np.zeros((0, rank), np.int64), z, z, z
        # (bytes joined and views made by hand: numpy's fields by name and
        # concatenating records cost microseconds each, for every dataset)
        r = b''.join(parts) if len(parts) > 1 else parts[0]
        n = len(r) // step
        offs = np.ndarray((n, rank), '<u8', r, 8, (step, 8)).astype(np.int64)
        addr = np.ndarray(n, '<u8', r, ksize, (step,)).astype(np.int64) + self.base
        return (offs, addr, np.ndarray(n, '<u4', r, 0, (step,)).astype(np.int64),
                np.ndarray(n, '<u4', r, 4, (step,)).astype(np.int64))

    def _btree1_chunks(self, address, rank):
        b = self.b
        O = self.O
        out = []
        ksize = 8 + 8 * (rank + 1)
        stack = [address]
        while stack:
            a = self.addr(stack.pop())
            self.need(a, 8 + 2 * O)
            if bytes(b[a:a + 4]) != b'TREE':
                raise FormatError('bad chunk B-tree node')
            level = b[a + 5]
            n = _u16(b, a + 6)[0]
            p = a + 8 + 2 * O
            self.need(a, 8 + 2 * O + n * (ksize + O) + ksize)
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
            if self.source is not None:            # remote: the next ones in one go
                self.prefetch([self.addr(k) for k in kids], 8 + 2 * O + max(n, 64) * (ksize + O) + ksize)
            stack.extend(reversed(kids))
        return out

    def _btree2_chunks(self, address, rank, chunks, filtered, csize):
        b = self.b
        a = self.addr(address)
        self.need(a, 12)
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
        self.need(a, 12 + self.L + self.O)
        if bytes(b[a:a + 4]) != b'FAHD':
            raise FormatError('bad fixed array header')
        client = b[a + 5]
        esize = b[a + 6]
        page_bits = b[a + 7]
        n = self.length(a + 8)
        db = self.addr(self.offset(a + 8 + self.L))
        npg = -(-n // (1 << page_bits))
        self.need(db, 6 + self.O + (npg + 7) // 8 + 4 + n * esize + 4 * npg + 4)
        if bytes(b[db:db + 4]) != b'FADB':
            raise FormatError('bad fixed array data block')
        p = db + 6 + self.O
        filtered = client == 1
        page = 1 << page_bits
        if n <= page:
            return self._elements(b, p, n, esize, filtered)
        npages = -(-n // page)
        bits = p
        p += (npages + 7) // 8 + 4                      # page init bitmap, checksum
        out = []
        for k in range(npages):
            m = min(page, n - k * page)
            # (a page never written is not on disk as elements: only its bit says so)
            out.extend(self._elements(b, p, m, esize, filtered) if _page_written(b, bits, k)
                       else [(self.undef, None, 0)] * m)
            p += m * esize + 4
        return out

    def _earray_chunks(self, ds, nper, chunks, csize):
        b = self.b
        lay = ds.layout
        a = self.addr(lay.address)
        self.need(a, 16 + 6 * self.L + self.O)
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
        self.need(iblk, 6 + self.O + iblk_n * esize + (2 * (sblk_min_ptrs - 1) + nsblks) * self.O + 4)
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

        def dblock_size(dn):
            npg = -(-dn // page) if dn > page else 0
            return 6 + self.O + off_size + 4 + dn * esize + 4 * npg + 4

        def dblock(addr, dn, bits=None, first=0):
            d = self.addr(addr)
            self.need(d, dblock_size(dn))
            if bytes(b[d:d + 4]) != b'EADB':
                raise FormatError('bad extensible array data block')
            r = d + 6 + self.O + off_size
            if dn <= page:
                return self._elements(b, r, dn, esize, filtered)
            out = []
            r += 4                                      # header checksum; then pages
            for k in range(-(-dn // page)):
                m = min(page, dn - k * page)
                out.extend(self._elements(b, r, m, esize, filtered) if bits is None or _page_written(b, bits, first + k)
                           else [(self.undef, None, 0)] * m)
                r += m * esize + 4
            return out

        idx = len(elems)
        all_elems = list(elems)
        for s in range(nsblks):
            if len(all_elems) >= nelm:
                break
            nd, dn, sidx, sdblk = sb[s]
            if s < ib_nsblks:
                if self.source is not None:            # remote: the next ones in one go
                    self.prefetch([self.addr(x) for x in dblk_addrs[sdblk:sdblk + nd] if x != self.undef],
                                  dblock_size(dn))
                for k in range(nd):
                    da = dblk_addrs[sdblk + k]
                    all_elems.extend(dblock(da, dn) if da != self.undef else [(self.undef, None, 0)] * dn)
            else:
                sa = sblk_addrs[s - ib_nsblks]
                if sa == self.undef:
                    all_elems.extend([(self.undef, None, 0)] * (nd * dn))
                    continue
                r = self.addr(sa)
                self.need(r, 6 + self.O + off_size + nd * ((-(-dn // page) + 7) // 8) + nd * self.O + 4)
                if bytes(b[r:r + 4]) != b'EASB':
                    raise FormatError('bad extensible array super block')
                r += 6 + self.O + off_size
                bits = r
                pages = -(-dn // page)
                if dn > page:                           # page init bits (see _earray_finder)
                    r += nd * ((pages + 7) // 8)
                if self.source is not None:            # remote: the next ones in one go
                    self.prefetch([self.addr(self.offset(r + k * self.O)) for k in range(nd)
                                   if self.offset(r + k * self.O) != self.undef], dblock_size(dn))
                for k in range(nd):
                    da = self.offset(r + k * self.O)
                    all_elems.extend(dblock(da, dn, bits, k * pages) if da != self.undef
                                     else [(self.undef, None, 0)] * dn)
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


def region_selection(b, rank):
    """A selection as a region reference stores it: None for nothing, 'all', an
    array of (n, rank) points, or arrays (first, last) of (n, rank) inclusive
    blocks. (Versions by H5S__hyper_serialize and H5S__point_serialize.)"""
    kind, version = _u32(b, 0)[0], _u32(b, 4)[0]
    if kind == 0:
        return None
    if kind == 3:
        return 'all'

    def numbers(p, n, width):
        return np.frombuffer(bytes(b[p:p + n * width]), '<u%d' % width).astype(np.int64)

    if kind == 1:                                   # points
        if version == 1:
            n = _u32(b, 20)[0]
            return numbers(24, n * rank, 4).reshape(n, rank)
        width = b[8]
        n = _uint(b, 13, width)
        return numbers(13 + width, n * rank, width).reshape(n, rank)
    if kind != 2:
        raise FormatError('selection of kind %d' % kind)
    if version == 1:
        n = _u32(b, 20)[0]
        a = numbers(24, 2 * n * rank, 4).reshape(n, 2, rank)
        return a[:, 0], a[:, 1]
    flags = b[8]
    width, p = (8, 17) if version == 2 else (b[9], 14)
    if flags & 1:                                   # regular: start, stride, count, block
        start, stride, count, block = numbers(p, 4 * rank, width).reshape(rank, 4).T
        axes = [np.arange(c)[:, None] * s + np.arange(k)[None, :] + a
                for a, s, c, k in zip(start.tolist(), stride.tolist(), count.tolist(), block.tolist())]
        firsts, lasts = [], []
        grid = np.indices([len(x) for x in axes]).reshape(rank, -1)
        for d, x in enumerate(axes):
            firsts.append(x[grid[d], 0])
            lasts.append(x[grid[d], -1])
        return np.array(firsts).T, np.array(lasts).T
    n = _uint(b, p, width)
    a = numbers(p + width, 2 * n * rank, width).reshape(n, 2, rank)
    return a[:, 0], a[:, 1]


class _EArray:
    """What an extensible array's header says."""


def _page_written(b, bits, k):
    """Bit k of a page init bitmap at byte `bits` (highest bit first)."""
    return b[bits + k // 8] & (0x80 >> (k % 8))


def _uints(a, at, width):
    """Little-endian integers of any width, from rows of bytes at column `at`."""
    out = np.zeros(len(a), np.int64)
    for k in range(width):
        out |= a[:, at + k].astype(np.int64) << (8 * k)
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
        f.need(a, 26 + 12 * f.L + 3 * f.O)
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
        a, n = self._huge_at(hid)
        return self.f.b[a:a + n]

    def _huge_at(self, hid):
        """Where a huge object lies, and its length: in the ID itself, or found
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
        f.need(a, n)
        return a, n

    def _row_size(self, r):
        return self.start if r == 0 else self.start << (r - 1)

    def location(self, p):
        """(position in the file, length) of the object whose ID is at p in
        the file."""
        f = self.f
        hid = bytes(f.b[p:p + 8])
        kind = (hid[0] >> 4) & 3
        if kind == 2:                                   # tiny: it lies in the ID
            return p + 1, (hid[0] & 15) + 1
        if kind == 1:
            return self._huge_at(hid)
        off = _uint(hid, 1, self.off_size)
        ln = _uint(hid, 1 + self.off_size, self.len_size)
        blk, boff = self._direct_block(off)
        return blk + (off - boff), ln

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
        if self.f.source is not None:
            self.f.need(blk + (off - boff), ln)
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
        f.need(a, 5 + f.O + self.off_size + nrows * self.width * f.O)
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
