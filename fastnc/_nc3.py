"""netCDF3, read from the file's bytes: classic (CDF-1), 64-bit offset
(CDF-2) and 64-bit data (CDF-5), as the netCDF File Format Specification
describes them. The header is parsed in pure Python (one read, no numba:
0.5 ms for wrfout d02's 140 variables), attribute values decoded only when
first asked for; values go straight into the output array.

Three ways to read values, chosen per request:

- runs: long contiguous runs of the file (a whole variable, whole levels,
  a record variable's records) are positional reads of 1 MiB pieces,
  straight into the output, on the shared thread pool (`_pool.share`),
  each piece byte-swapped in place by a numba kernel right after it lands,
  while it is in the core's cache. Cached file data reaches a user buffer
  at 6 GB/s a thread; many threads overlap that copy and the page faults on
  the fresh output, which are what limits a whole read (~10 GB/s here).
- gather: short or strided runs (a column, every other point, small
  records) are copied out of a read-only mapping of the file, swapped on
  the way: by numpy's strided copy, or, for big requests, by a numba kernel
  on the pool taking 1 MiB blocks of rows off a counter.
- small: anything up to 64 kB is copied out of the mapping by numpy.

numba is never imported here: the kernels are used once something else
has loaded it (KERNELS = None), numpy does the swapping until then (~30%
slower on whole reads, and no second's wait for numba on a first read).

Positional reads are one overlapped handle shared by all threads on
Windows (a handle opened the usual way serializes its readers), preadv(2)
elsewhere. Bytes past the end of a truncated file read as zeros, as
netCDF-C's posixio means to (its zero-fill stops at its buffer's page:
beyond that, netCDF4-python returns whatever the buffer held before).
"""
import os
import struct
import sys
import threading

import numpy as np

from ._jit import njit
from ._intrin import ld64, st64, take
from ._chunks import holds_value, loaded, value_words
from . import _arena, _pool


class FormatError(ValueError):
    """The bytes are not netCDF3 as this reader understands it."""


NC_DIMENSION, NC_VARIABLE, NC_ATTRIBUTE = 10, 11, 12

# nc_type -> (stored big-endian dtype, native dtype); 7..11 are CDF-5's
_TYPES = {
    1: ('i1', 'i1'), 2: ('S1', 'S1'), 3: ('>i2', '=i2'), 4: ('>i4', '=i4'),
    5: ('>f4', '=f4'), 6: ('>f8', '=f8'), 7: ('u1', 'u1'), 8: ('>u2', '=u2'),
    9: ('>u4', '=u4'), 10: ('>i8', '=i8'), 11: ('>u8', '=u8'),
}
_BIG = {t: np.dtype(a) for t, (a, b) in _TYPES.items()}
_NATIVE = {t: np.dtype(b) for t, (a, b) in _TYPES.items()}
_SIZE = {t: d.itemsize for t, d in _BIG.items()}
_SIZES = (0,) + tuple(_SIZE[t] for t in range(1, 12))   # by nc_type

_DATA_MODEL = {1: 'NETCDF3_CLASSIC', 2: 'NETCDF3_64BIT_OFFSET', 5: 'NETCDF3_64BIT_DATA'}

_HEAD = 1 << 16                 # first read of the header; grown when it is longer

PIECE = 1 << 20                 # bytes per positional read on the pool
RUN_MIN = 1 << 16               # runs at least this long are read, not gathered
PAR_MIN = 1 << 21               # requests this big go on the pool
MAP_MAX = 1 << 16               # requests up to this big are copied out of the mapping
SWAP_NUMPY = 1 << 16            # smaller reads swap with numpy
GATHER_PAR = 1 << 20            # strided requests this big go on the pool (with numba)
GATHER_BLOCK = 1 << 20          # output bytes per block a gathering thread takes (smaller:
#                               page faults on the fresh output contend, 64 kB is 3x slower)
KERNELS = None                  # numba kernels: None = once numba is loaded; True; False

_u32 = struct.Struct('>I').unpack_from
_u64 = struct.Struct('>Q').unpack_from
_u32x2 = struct.Struct('>II').unpack_from
_u32u64 = struct.Struct('>IQ').unpack_from


class _Short(Exception):
    """The header goes on past the bytes read so far."""


def _decode_name(raw):
    raw = raw.split(b'\0', 1)[0]                    # (a name may be counted past its end: netCDF-c's ends here)
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw.decode('utf-8', 'replace')


def _att_list(b, p, ver):
    """(offset of the first attribute, number of attributes, offset past the
    list) for the attribute list whose tag is at p."""
    sizes = _SIZES
    if ver == 5:
        tag, n = _u32u64(b, p)
        p += 12
    else:
        tag, n = _u32x2(b, p)
        p += 8
    if tag != NC_ATTRIBUTE and not (tag == 0 and n == 0):
        raise FormatError('netCDF3: bad attribute list tag %d' % tag)
    first = p
    if ver == 5:
        u64, u32u64 = _u64, _u32u64
        for _ in range(n):
            p += 8 + ((u64(b, p)[0] + 3) & ~3)
            t, m = u32u64(b, p)
            if not 0 < t < 12:
                raise FormatError('netCDF3: bad attribute type %d' % t)
            p += 12 + ((m * sizes[t] + 3) & ~3)
    else:
        u32, u32x2 = _u32, _u32x2
        for _ in range(n):
            p += 4 + ((u32(b, p)[0] + 3) & ~3)
            t, m = u32x2(b, p)
            if not 0 < t < 7:
                raise FormatError('netCDF3: bad attribute type %d' % t)
            p += 8 + ((m * sizes[t] + 3) & ~3)
    if p > len(b):
        raise _Short
    return first, n, p


def _decode_atts(b, p, n, ver):
    """The attributes as netCDF4-python returns them (`_get_att`): text ->
    str (UTF-8, bad bytes replaced, every NUL dropped; but a text _FillValue
    stays bytes); one number -> numpy scalar; several -> 1-d array."""
    out = {}
    w = 8 if ver == 5 else 4
    for _ in range(n):
        ln = _u64(b, p)[0] if w == 8 else _u32(b, p)[0]
        p += w
        name = _decode_name(b[p:p + ln])
        p += (ln + 3) & ~3
        t = _u32(b, p)[0]
        m = _u64(b, p + 4)[0] if w == 8 else _u32(b, p + 4)[0]
        p += 4 + w
        nb = m * _SIZE[t]
        if t == 2:
            raw = b[p:p + nb]
            out[name] = raw if name == '_FillValue' else \
                raw.decode('utf-8', 'replace').replace('\x00', '')
        elif m == 1:
            out[name] = np.frombuffer(b, _BIG[t], 1, p)[0]
        else:
            out[name] = np.frombuffer(b, _BIG[t], m, p).astype(_NATIVE[t])
        p += (nb + 3) & ~3
    return out


class _Last:
    """The variable list last parsed: (version, dimensions, its bytes, where it
    began, its rows). The files of a run differ in their global attributes and
    not here, and walking every attribute of every variable was most of an
    open. Equal bytes say the same, whatever file they came from."""
    list = None


def _parse(b, size):
    """The header in bytes b (the file's first bytes; size the file's size):
    (version, numrecs field, dims [(name, length)], global attribute span,
    vars [(name, dimids, att span, nc_type, begin)])."""
    if len(b) < 8 or b[:3] != b'CDF' or b[3] not in (1, 2, 5):
        raise FormatError('not a netCDF3 file (no CDF\\1, \\2 or \\5 signature)')
    ver = b[3]
    try:
        if ver == 5:
            numrecs = _u64(b, 4)[0]
            p = 12
            tag = _u32(b, p)[0]
            n = _u64(b, p + 4)[0]
            p += 12
        else:
            numrecs = _u32(b, 4)[0]
            p = 8
            tag, n = _u32x2(b, p)
            p += 8
        if tag != NC_DIMENSION and not (tag == 0 and n == 0):
            raise FormatError('netCDF3: bad dimension list tag %d' % tag)
        dims = []
        for _ in range(n):
            if ver == 5:
                ln = _u64(b, p)[0]
                p += 8
                name = _decode_name(b[p:p + ln])
                p += (ln + 3) & ~3
                length = _u64(b, p)[0]
                p += 8
            else:
                ln = _u32(b, p)[0]
                p += 4
                name = _decode_name(b[p:p + ln])
                p += (ln + 3) & ~3
                length = _u32(b, p)[0]
                p += 4
            dims.append((name, length))
        gfirst, gn, p = _att_list(b, p, ver)
        ndim = len(dims)
        last = _Last.list
        if last is not None and last[0] == ver and last[1] == ndim and b.startswith(last[2], p):
            shift = p - last[3]                     # (attribute spans are places in the file)
            rows = last[4] if not shift else [(a, i, f + shift, m, t, g) for a, i, f, m, t, g in last[4]]
            return ver, numrecs, dims, gfirst, gn, rows, p + len(last[2])
        start = p
        if ver == 5:
            tag = _u32(b, p)[0]
            n = _u64(b, p + 4)[0]
            p += 12
        else:
            tag, n = _u32x2(b, p)
            p += 8
        if tag != NC_VARIABLE and not (tag == 0 and n == 0):
            raise FormatError('netCDF3: bad variable list tag %d' % tag)
        vars_ = []
        for _ in range(n):
            if ver == 5:
                ln = _u64(b, p)[0]
                p += 8
                name = _decode_name(b[p:p + ln])
                p += (ln + 3) & ~3
                nd = _u64(b, p)[0]
                p += 8
                ids = struct.unpack_from('>%dQ' % nd, b, p)
                p += 8 * nd
            else:
                ln = _u32(b, p)[0]
                p += 4
                name = _decode_name(b[p:p + ln])
                p += (ln + 3) & ~3
                nd = _u32(b, p)[0]
                p += 4
                ids = struct.unpack_from('>%dI' % nd, b, p)
                p += 4 * nd
            if nd and max(ids) >= ndim:
                raise FormatError('netCDF3: variable %r uses dimension id %d of %d'
                                  % (name, max(ids), ndim))
            afirst, an, p = _att_list(b, p, ver)
            t = _u32(b, p)[0]
            if not 0 < t < (12 if ver == 5 else 7):
                raise FormatError('netCDF3: variable %r has bad type %d' % (name, t))
            if ver == 1:
                begin = _u32(b, p + 8)[0]      # (vsize skipped: computed as netCDF-C does)
                p += 12
            elif ver == 2:
                begin = _u64(b, p + 8)[0]
                p += 16
            else:
                begin = _u64(b, p + 12)[0]
                p += 20
            vars_.append((name, ids, afirst, an, t, begin))
        if p > len(b):
            raise _Short
    except struct.error:
        raise _Short
    if p - start <= 1 << 20:
        _Last.list = ver, ndim, b[start:p], start, vars_
    return ver, numrecs, dims, gfirst, gn, vars_, p


def _read_header(fh, size):
    """The header's bytes and its parse, read in one go when it fits in the
    first 64 KiB (else read again, four times as much each time)."""
    n = min(size, _HEAD)
    while True:
        fh.seek(0)
        b = fh.read(n)
        try:
            return b, _parse(b, size)
        except _Short:
            if n >= size:
                raise FormatError('netCDF3: the header is truncated')
            n = min(size, n * 4)


class NC3File:
    """A netCDF3 file opened for reading (fastnc's BackendFile): the header
    parsed, file handles for the positional reads, a mapping made on first
    need."""

    def __init__(self, path, source=None):
        self.path = os.fspath(path)
        self._remote = source           # a remote file (_remote.Source): read from its buffer
        fh = open(self.path, 'rb', buffering=0) if source is None else source.file()
        try:
            self._size = size = os.fstat(fh.fileno()).st_size if source is None else source.size
            b, (ver, numrecs, dims, gfirst, gn, vars_, _) = _read_header(fh, size)
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        self._header = b
        self.version = ver
        self.data_model = _DATA_MODEL[ver]
        self.file_format = self.data_model
        self.disk_format = 'NETCDF3'
        self._rd = None                 # the positional reader
        self._mm = None
        self._src = None
        self._mlock = threading.Lock()
        self._closed = False
        self._spanned = set() if source is None else None   # see _span

        dsize = [ln for _, ln in dims]
        dnames = [name for name, _ in dims]
        rec_id = dsize.index(0) if 0 in dsize else -1
        # element counts and record size as netCDF-C's NC_computeshapes
        recsize = 0
        nrec = 0
        first_rec = None
        recs = []
        sizes = _SIZES
        for v in vars_:
            ids = v[1]
            if ids and ids[0] == rec_id:
                n = 1
                for i in ids[1:]:
                    n *= dsize[i]
                ln = n * sizes[v[4]]
                pad = ln + (-ln & 3)            # padded to 4 bytes
                recsize += pad
                nrec += 1
                if first_rec is None:
                    first_rec = (ln, pad)
                recs.append(v[5])
        if nrec == 1 and recsize == first_rec[1]:
            recsize = first_rec[0]              # one record variable: not padded
        self._recsize = recsize
        stream = numrecs == ((1 << 64) - 1 if ver == 5 else 0xFFFFFFFF)
        if stream:                              # streaming: as many records as the file holds
            numrecs = max(0, (size - min(recs)) // recsize) if recsize else 0
        self.numrecs = numrecs
        self.streaming = stream
        if rec_id >= 0:
            dsize[rec_id] = numrecs

        root = NC3Group(self, '/', gfirst, gn)
        root.dimensions = {name: (n, i == rec_id) for i, (name, n) in
                           enumerate(zip(dnames, dsize))}
        root._rows = {v[0]: v for v in vars_}
        self._dims = dnames, dsize, rec_id
        self.root = root

    def _variable(self, row):
        """The NC3Variable of a row of the header."""
        name, ids, afirst, an, t, begin = row
        dnames, dsize, rec_id = self._dims
        return NC3Variable(self, name, tuple([dnames[i] for i in ids]), tuple([dsize[i] for i in ids]),
                           t, begin, bool(ids) and ids[0] == rec_id, afirst, an)

    # -- I/O

    def _reader(self):
        """The positional reader, made on first need."""
        r = self._rd
        if r is None:
            with self._mlock:
                if self._closed:                    # (by now its name may be another file's)
                    raise ValueError('the file %s is closed' % self.path)
                if self._rd is None:
                    self._rd = _IO(self)
                r = self._rd
        return r

    def _readinto(self, u8, addr, o, n, off):
        """u8[o:o + n] (at address addr + o) filled from the file at off;
        bytes past the end of the file are zeros."""
        avail = self._size - off
        if avail < n:
            if avail <= 0:
                u8[o:o + n] = 0
                return
            u8[o + avail:o + n] = 0
            n = avail
        if self._remote is not None:            # fetched by _read already: a copy
            self._remote.fetch(np.array([off]), np.array([n]))
            u8[o:o + n] = self._remote.buf[off:off + n]
            return
        got = self._reader().read(u8, addr, o, n, off)
        if got < n:                             # the file shrank since it was opened
            u8[o + got:o + n] = 0

    def _source(self):
        """The whole file as a read-only uint8 array (a mapping, made once)."""
        src = self._src
        if src is None:
            with self._mlock:
                if self._src is None:
                    import mmap
                    if self._remote is not None:
                        from ._remote import readonly
                        self._src = readonly(self._remote)
                    elif self._size:
                        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
                        self._src = np.frombuffer(self._mm, np.uint8)
                    else:
                        self._src = np.zeros(0, np.uint8)
                src = self._src
        return src

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._src = None
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:                 # a view is still out there: left to the GC
                pass
            self._mm = None
        if self._rd is not None:
            self._rd.close()
            self._rd = None
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return '<NC3File %r %s>' % (self.path, self.data_model)


class _Handles:
    """Positional reads anywhere: seek + readinto on a file handle of the
    reading thread's own (several threads on one handle would race)."""

    def __init__(self, f):
        self.path = f.path
        self.free = []                  # handles not in use
        self.all = []
        self.lock = threading.Lock()

    def read(self, u8, addr, o, n, off):
        try:
            h = self.free.pop()
        except IndexError:
            h = open(self.path, 'rb', buffering=0)
            with self.lock:
                self.all.append(h)
        try:
            h.seek(off)
            mv = memoryview(u8)[o:o + n]
            got = 0
            while got < n:
                k = h.readinto(mv[got:])
                if not k:
                    break
                got += k
            return got
        finally:
            self.free.append(h)

    def close(self):
        for h in self.all:
            h.close()
        self.all = self.free = []


class _Preadv:
    """Positional reads with preadv(2) on the file's descriptor (POSIX):
    thread-safe, no file position."""

    def __init__(self, f):
        self.fd = f._fh.fileno()

    def read(self, u8, addr, o, n, off):
        mv = memoryview(u8)[o:o + n]
        got = 0
        while got < n:
            k = os.preadv(self.fd, [mv[got:]], off + got)
            if k <= 0:
                break
            got += k
        return got

    def close(self):
        pass


class _Overlapped:
    """Positional reads on Windows: ReadFile with the offset in an OVERLAPPED,
    on one handle opened for overlapped I/O that any number of threads read
    at once. (A handle opened the usual way serializes its reads; a handle
    per thread costs an open per thread and file, ~10% of a 30 MB read.)
    Each read waits on an event of its own, taken from a free list."""

    def __init__(self, f):
        k, OVL = _win32()
        self.k, self.OVL = k, OVL
        import ctypes
        h = k.CreateFileW(f.path, 0x80000000, 7, None, 3, 0x40000000, None)  # GENERIC_READ,
        # share read/write/delete, OPEN_EXISTING, FILE_FLAG_OVERLAPPED
        if h is None or h == ctypes.c_void_p(-1).value:
            e = ctypes.get_last_error()
            raise OSError(None, ctypes.FormatError(e), f.path, e)
        self.h = h
        self.events = []
        self.all = []
        self.lock = threading.Lock()

    def read(self, u8, addr, o, n, off):
        import ctypes
        k = self.k
        try:
            ev = self.events.pop()
        except IndexError:
            ev = k.CreateEventW(None, True, False, None)
            with self.lock:
                self.all.append(ev)
        try:
            ov = self.OVL()
            got = ctypes.c_uint32()
            done = 0
            while done < n:
                p = off + done
                ov.Offset, ov.OffsetHigh, ov.hEvent = p & 0xFFFFFFFF, p >> 32, ev
                ov.Internal = ov.InternalHigh = 0
                m = min(n - done, 1 << 30)
                if not k.ReadFile(self.h, addr + o + done, m, ctypes.byref(got), ctypes.byref(ov)):
                    e = ctypes.get_last_error()
                    if e == 38:                 # ERROR_HANDLE_EOF
                        break
                    if e != 997:                # ERROR_IO_PENDING
                        raise OSError(None, ctypes.FormatError(e), None, e)
                if not k.GetOverlappedResult(self.h, ctypes.byref(ov), ctypes.byref(got), True):
                    e = ctypes.get_last_error()
                    if e == 38:
                        break
                    raise OSError(None, ctypes.FormatError(e), None, e)
                if got.value == 0:
                    break
                done += got.value
            return done
        finally:
            self.events.append(ev)

    def close(self):
        for ev in self.all:
            self.k.CloseHandle(ev)
        self.all = self.events = []
        if self.h is not None:
            self.k.CloseHandle(self.h)
            self.h = None


_WIN32 = None


def _win32():
    """kernel32's file functions through ctypes (they release the GIL)."""
    global _WIN32
    if _WIN32 is None:
        import ctypes
        from ctypes import wintypes as w

        class OVL(ctypes.Structure):
            _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                        ('Offset', w.DWORD), ('OffsetHigh', w.DWORD), ('hEvent', w.HANDLE)]
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD,
                                  w.DWORD, w.HANDLE]
        k.CreateFileW.restype = w.HANDLE
        k.ReadFile.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD, ctypes.POINTER(ctypes.c_uint32),
                               ctypes.POINTER(OVL)]
        k.ReadFile.restype = w.BOOL
        k.GetOverlappedResult.argtypes = [w.HANDLE, ctypes.POINTER(OVL),
                                          ctypes.POINTER(ctypes.c_uint32), w.BOOL]
        k.GetOverlappedResult.restype = w.BOOL
        k.CreateEventW.argtypes = [ctypes.c_void_p, w.BOOL, w.BOOL, ctypes.c_void_p]
        k.CreateEventW.restype = w.HANDLE
        k.CloseHandle.argtypes = [w.HANDLE]
        k.CloseHandle.restype = w.BOOL
        _WIN32 = k, OVL
    return _WIN32


if sys.platform == 'win32':
    _IO = _Overlapped
elif hasattr(os, 'preadv'):
    _IO = _Preadv
else:
    _IO = _Handles


class Attrs(dict):
    """A variable's attributes, and what masking would otherwise work out on
    every read: those it may cast to the variable's type, and which of the
    few that decide its plan there are (_maskscale asks for both by name)."""
    __slots__ = ('safe', 'found', 'toma')

    def __init__(self, values):
        dict.__init__(self, values)
        self.safe, self.found, self.toma = set(), None, {}

    def among(self, names):
        if self.found is None:
            self.found = [n for n in names if n in self]
        return self.found


class NC3Group:
    """The root group of a netCDF3 file (fastnc's BackendGroup); netCDF3 has
    no other groups."""

    def __init__(self, file, name, afirst, an):
        self.file = file
        self.name = name
        self.groups = {}
        self.dimensions = {}
        self._rows = {}                 # name -> its row of the header
        self._made = {}                 # name -> NC3Variable, those asked for so far
        self._variables = None
        self._att = (afirst, an)
        self._attrs = None

    @property
    def attrs(self):
        a = self._attrs
        if a is None:
            f = self.file
            a = self._attrs = _decode_atts(f._header, self._att[0], self._att[1], f.version)
        return a

    @property
    def variables(self):
        """Every variable, in the file's order."""
        v = self._variables
        if v is None:
            make, made = self.file._variable, self._made
            v = self._variables = {k: made.get(k) or make(row) for k, row in self._rows.items()}
        return v

    def peek(self, name):
        """The variable called `name` alone (KeyError if there is none): a
        read of one of WRF's 140 need not make the others."""
        v = self._made.get(name)
        if v is None:
            v = self._made[name] = self.file._variable(self._rows[name])
        return v

    def holds_no_data(self):
        return not self._rows and not self.dimensions

    def __repr__(self):
        return '<NC3Group %r>' % self.name


class NC3Variable:
    """A netCDF3 variable (fastnc's BackendVariable): where its values are
    and how they are laid out; read() gives raw native-endian values."""

    chunks = None
    filters = None              # netCDF4.Variable.filters() is None for netCDF3
    endian = 'native'           # as netCDF4.Variable.endian() says for netCDF3

    def __init__(self, file, name, dimensions, shape, nc_type, begin, is_rec, afirst, an):
        self.file = file
        self.name = name
        self.dimensions = dimensions
        self.shape = shape
        self.nc_type = nc_type
        self.dtype = _NATIVE[nc_type]
        self._big = _BIG[nc_type]
        self.begin = begin
        self.is_record = is_rec
        self._att = (afirst, an)
        self._attrs = None
        self._bstrides = None
        self._quick = None

    @property
    def attrs(self):
        a = self._attrs
        if a is None:
            f = self.file
            a = self._attrs = Attrs(_decode_atts(f._header, self._att[0], self._att[1], f.version))
        return a

    def _strides(self):
        """Bytes in the file from one index to the next, per dimension."""
        es = self.dtype.itemsize
        shape = self.shape
        strides = []
        s = es
        for n in reversed(shape[1:] if self.is_record else shape):
            strides.append(s)
            s *= n
        if self.is_record:
            strides.append(self.file._recsize)
        st = self._bstrides = tuple(reversed(strides))
        return st

    def __repr__(self):
        return '<NC3Variable %r %s %s>' % (self.name, self.shape, self.dtype)

    probes = True                               # (read takes a Probe)

    def read(self, key=(), out=None, probe=None):
        """Raw values for a basic index `key` (one int or slice per
        dimension), native byte order, shaped as numpy shapes x[key]; written
        into `out` when given. probe: a _maskscale.Probe, tested on the way."""
        if out is None:
            q = self._quick
            if q is None:
                f = self.file
                q = self._quick = f._remote is None and small_reads(
                    f, self.begin, self.shape, self._bstrides or self._strides(), self._big, self.dtype)
            if q:
                res = q(key, probe)
                if res is not None:
                    return res
        shape = self.shape
        if not isinstance(key, tuple):
            key = (key,)
        nd = len(shape)
        if len(key) < nd:
            key = key + (slice(None),) * (nd - len(key))
        elif len(key) > nd:
            raise IndexError('too many indices for a %d-d variable' % nd)
        es = self.dtype.itemsize
        base = self.begin
        oshape = []
        dims = []                               # (count, byte stride) of what is selected
        for k, n, bs in zip(key, shape, self._bstrides or self._strides()):
            if type(k) is slice:
                start, stop, step = k.start, k.stop, k.step
                if start is None or stop is None or step is None or start < 0 or stop > n:
                    start, stop, step = k.indices(n)
                if step <= 0:
                    raise IndexError('fastnc: slices need a step > 0')
                c = (stop - start + step - 1) // step if stop > start else 0
                oshape.append(c)
                if c != 1:
                    dims.append((c, step * bs))
                base += start * bs
            else:
                i = int(k)
                if i < 0:
                    i += n
                if not 0 <= i < n:
                    raise IndexError('index %d is out of bounds for a dimension of size %d'
                                     % (int(k), n))
                base += i * bs
        oshape = tuple(oshape)
        dtype = self.dtype
        if out is not None:
            if (out.shape == oshape and out.dtype == dtype and out.flags.c_contiguous
                    and out.flags.writeable):
                res = out
            else:
                res = _arena.empty(oshape, dtype)
        else:
            res = _arena.empty(oshape, dtype)
        if res.size:
            _read(self.file, self._big, es, base, dims, res, probe)
        if out is not None and res is not out:
            np.copyto(out, res)
            return out
        return res


def small_reads(f, begin, shape, strides, big, native):
    """A function of a key giving the values for it, or None for anything but
    a slice of whole numbers a dimension (an integer, Ellipsis, a backward
    step). A variable's layout does not change, so where its values lie is a
    sum, and all a read going through `read` pays for besides that sum is
    working out, each time, what the last read worked out. A read small enough
    is a copy out of the mapping; any other is `_read`'s, with the sum done
    here (done there as well, it cost a loop over time steps 8%). f: as for
    _read; its mapping in f._src."""
    plan = tuple(zip(shape, strides))
    rank = len(shape)
    es = big.itemsize
    ndarray = np.ndarray
    empty = _arena.empty

    def read(key, probe=None, threads=None):
        if key.__class__ is not tuple or len(key) != rank:
            return None
        at = begin
        count, step = [], []
        size = es
        reach = es
        for k, (n, b) in zip(key, plan):
            if k.__class__ is not slice:
                return None
            s, e, st = k.start, k.stop, k.step
            if s.__class__ is not int or e.__class__ is not int or st.__class__ is not int \
                    or st < 1 or s < 0 or e > n or s >= e:
                return None
            c = (e - s - 1) // st + 1
            at += s * b
            count.append(c)
            step.append(st * b)
            size *= c
            reach += (c - 1) * st * b
        src = f._src
        if src is not None and size <= MAP_MAX and at + reach <= f._size:
            return ndarray(tuple(count), big, src, at, tuple(step)).astype(native)
        res = empty(tuple(count), native)
        _read(f, big, es, at, [d for d in zip(count, step) if d[0] != 1], res, probe, threads)
        return res
    return read


def _read(f, big, es, base, dims, res, probe=None, threads=None):
    """res (C-contiguous, native) filled with the elements at base + sum(i_d *
    stride_d) for i_d < count_d over dims [(count_d, stride_d)] (counts > 1).
    big: the type as stored. f: this file, or anything with its _remote, _size,
    _source and _readinto (HDF4's contiguous variables: the same sums)."""
    if f._remote is not None:                   # a remote file: its runs fetched first
        from ._remote import runs
        f._remote.fetch(*runs(base, dims, es))
    merged = []                                 # dims merged where the file is contiguous
    for c, s in dims:
        if merged and merged[-1][1] == c * s:
            merged[-1] = (merged[-1][0] * c, s)
        else:
            merged.append((c, s))
    if not merged or merged[-1][1] == es:
        run = merged.pop()[0] * es if merged else es
        nruns = 1
        for c, _ in merged:
            nruns *= c
        if (nruns == 1 and run > MAP_MAX) or run >= RUN_MIN:
            _read_runs(f, big, es, base, merged, run, nruns, res, probe, threads)
            return
        merged.append((run // es, es))          # (small: a copy out of the mapping)
    _gather(f, big, es, base, merged, res, threads)


def _read_runs(f, big, es, base, outer, run, nruns, res, probe=None, threads=None):
    """Contiguous runs of `run` bytes, at base + the outer dims' offsets, read
    straight into res one after another, then swapped; probe's values looked
    for piece by piece, just swapped and still in cache."""
    total = run * nruns
    flat = res.reshape(-1)
    u8 = flat.view(np.uint8)
    addr = u8.ctypes.data
    swap = es > 1 and not big.isnative
    kern = _kernels()
    if nruns == 1 and total <= PAR_MIN:
        f._readinto(u8, addr, 0, total, base)
        if swap:
            if kern and total > SWAP_NUMPY:
                _swap(u8, 0, total, es)
            else:
                res.byteswap(inplace=True)
        return
    offs = np.array([base], np.int64)
    for c, s in outer:
        offs = (offs[:, None] + np.arange(c, dtype=np.int64) * s).reshape(-1)
    oo = np.arange(nruns, dtype=np.int64) * run
    if run > PIECE:                             # long runs cut into pieces
        k = -(-run // PIECE)
        psz = -(-run // (k * es)) * es
        js = np.arange(k, dtype=np.int64) * psz
        sizes = np.minimum(psz, run - js)
        offs = (offs[:, None] + js).reshape(-1)
        oo = (oo[:, None] + js).reshape(-1)
        nb = np.tile(sizes, nruns)
    else:
        nb = np.full(nruns, run, np.int64)
    fo, oo, nb = offs.tolist(), oo.tolist(), nb.tolist()
    npieces = len(fo)
    small = total <= SWAP_NUMPY or not swap
    # swapped in place by the kernel; without numba, read into this thread's
    # own (cache-warm) buffer and swapped on the way out by numpy's copy
    mode = 0 if small else (1 if kern else 2)
    ctr = iter(range(npieces))
    lock = threading.Lock()
    test = probe is not None and mode == 1 and probe.values.dtype.itemsize == es
    if test:
        w0, w1, low = value_words(probe.values)
    found = []

    def work():
        if mode == 2:
            buf = np.empty(max(nb) // es, big)
            b8 = buf.view(np.uint8)
            baddr = b8.ctypes.data
        while True:
            with lock:
                i = next(ctr, -1)
            if i < 0:
                return
            o, n = oo[i], nb[i]
            if mode == 2:
                f._readinto(b8, baddr, 0, n, fo[i])
                np.copyto(flat[o // es:(o + n) // es], buf[:n // es])
            else:
                f._readinto(u8, addr, o, n, fo[i])
                if mode == 1:
                    _swap(u8, o, o + n, es)
                    if test and holds_value(u8, o, o + n, w0, w1, low):
                        found.append(i)

    nt = 1 if total <= PAR_MIN else _pool.memory_threads(total, min(npieces, threads or _pool.threads()),
                                                         _pool.PASS_SPEED)
    _pool.share(work, nt)
    if swap and small:
        res.byteswap(inplace=True)
    if test and not found:
        probe.proven = True


def _gather(f, big, es, base, dims, res, threads=None):
    """Strided elements copied (and swapped) out of the file's mapping: by
    numpy's strided copy, or, for big requests once numba is loaded, by a
    kernel on the pool taking blocks of rows off a counter."""
    counts = tuple(c for c, _ in dims)
    strides = tuple(s for _, s in dims)
    last = base + sum((c - 1) * s for c, s in dims) + es
    seen = f._spanned
    if seen is not None and strides[0] <= 1 << 14 and last - base >= RUN_MIN and last <= f._size \
            and (counts, strides) not in seen:
        if len(seen) >= 64:
            seen.clear()
        seen.add((counts, strides))
        _span(f, big, es, base, counts, strides, res, threads)
        return
    src = f._source()
    if last > src.size:                         # past the end of the file: zeros there
        tmp = np.zeros(last - base, np.uint8)
        if base < f._size:
            n = min(last, f._size) - base
            tmp[:n] = src[base:base + n]
        src, base = tmp, 0
    total = res.nbytes
    nrows = total // (counts[-1] * es)
    rpb = max(1, GATHER_BLOCK // (counts[-1] * es))
    nblocks = -(-nrows // rpb)
    most = threads or _pool.threads()
    # (the kernel reverses bytes: stored in native order, numpy's copy)
    if total > GATHER_PAR and nblocks > 1 and most > 1 and (es == 1 or not big.isnative) and _kernels():
        ctr = np.zeros(1, np.int64)             # (on one thread numpy's copy is faster)
        c = np.array(counts, np.int64)
        s = np.array(strides, np.int64)
        dst = res.reshape(-1).view(np.uint8)
        _pool.share(lambda: _gather_rows(src, base, c, s, es, dst, ctr, nrows, rpb), min(nblocks, most))
        return
    view = np.ndarray(counts, big, buffer=src, offset=base, strides=strides)
    np.copyto(res.reshape(counts), view)


def _span(f, big, es, base, counts, strides, res, threads=None):
    """Strided elements a few pages apart or less (the outer stride at most
    four), read the first time a selection of this form comes: the bytes they
    lie among read in pieces, each piece's elements copied out while it is in
    cache. On a fresh mapping every page of them would fault, and be unmapped
    at close: a variable of a file of small records, 8.3 ms that way, 4.3 on
    one thread. The next such selection goes through the mapping (`_gather`):
    what a read has faulted in stays, and many of them, at place after place,
    come to more than the faults once."""
    from ._chunks import _pybuf                 # (here: a line at the top would compile every kernel)
    outer, step = counts[0], strides[0]
    per = max(1, PIECE // step)                 # outer indexes to a read
    one = sum((c - 1) * s for c, s in zip(counts[1:], strides[1:])) + es
    size = (per - 1) * step + one
    out = res.reshape(counts)
    inner = counts[1:]
    left = iter(range(0, outer, per))
    lock = threading.Lock()

    def work():
        buf = _pybuf(size)
        addr = buf.ctypes.data
        while True:
            with lock:
                i = next(left, -1)
            if i < 0:
                return
            m = min(per, outer - i)
            f._readinto(buf, addr, 0, (m - 1) * step + one, base + i * step)
            np.copyto(out[i:i + m], np.ndarray((m,) + inner, big, buf, 0, strides))
    _pool.share(work, _pool.memory_threads((outer - 1) * step + one, threads or _pool.threads()))


def _kernels():
    """Whether to use the kernels: KERNELS, or (None) whether they are there
    for the asking -- reading netCDF3 then never pays numba's import itself
    (most of a second in a fresh process), only gains once something did."""
    k = KERNELS
    return loaded() if k is None else k


# ---------------------------------------------------------------- kernels

_M8 = np.uint64(0x00FF00FF00FF00FF)
_M16 = np.uint64(0x0000FFFF0000FFFF)
_S8 = np.uint64(8)
_S16 = np.uint64(16)
_S32 = np.uint64(32)


@njit(cache=True, nogil=True)
def _swap(a, lo, hi, es):
    """The es-byte values in a[lo:hi] (uint8; hi - lo a multiple of es, lo
    at a value's start) byte-reversed in place, eight bytes at a time
    (unaligned-safe word loads and stores)."""
    n8 = (hi - lo) // 8
    if es == 2:
        for k in range(n8):
            i = lo + k * 8
            x = ld64(a, i)
            st64(a, i, ((x & _M8) << _S8) | ((x >> _S8) & _M8))
    elif es == 4:
        for k in range(n8):
            i = lo + k * 8
            x = ld64(a, i)
            x = ((x & _M8) << _S8) | ((x >> _S8) & _M8)
            st64(a, i, ((x & _M16) << _S16) | ((x >> _S16) & _M16))
    elif es == 8:
        for k in range(n8):
            i = lo + k * 8
            x = ld64(a, i)
            x = ((x & _M8) << _S8) | ((x >> _S8) & _M8)
            x = ((x & _M16) << _S16) | ((x >> _S16) & _M16)
            st64(a, i, (x << _S32) | (x >> _S32))
    else:
        n8 = 0
    i = lo + n8 * 8
    while i < hi:                               # the tail, value by value
        for j in range(es // 2):
            t = a[i + j]
            a[i + j] = a[i + es - 1 - j]
            a[i + es - 1 - j] = t
        i += es
    return 0


@njit(cache=True, nogil=True)
def _gather_rows(src, base, counts, strides, es, dst, ctr, nrows, rpb):
    """dst (uint8, C order) = the es-byte big-endian elements of src at base +
    sum(i_d * strides[d]), i_d < counts[d], byte-reversed. Work is shared out
    in blocks of rpb rows (all dimensions but the last) taken off ctr[0]:
    each thread writes only the rows of the blocks it took."""
    nd = counts.size
    n = counts[nd - 1]
    step = strides[nd - 1]
    rowbytes = n * es
    idx = np.zeros(nd, np.int64)
    while True:
        r0 = take(ctr, 0) * rpb
        if r0 >= nrows:
            return 0
        r1 = min(nrows, r0 + rpb)
        rem = r0
        addr = base
        for d in range(nd - 2, -1, -1):         # the block's first row's position
            idx[d] = rem % counts[d]
            rem //= counts[d]
            addr += idx[d] * strides[d]
        for r in range(r0, r1):
            o = r * rowbytes
            a = addr
            if step == es and es > 1:           # a contiguous row: words, then the tail
                n8 = rowbytes // 8
                for k in range(n8):
                    x = ld64(src, a + k * 8)
                    if es == 2:
                        x = ((x & _M8) << _S8) | ((x >> _S8) & _M8)
                    elif es == 4:
                        x = ((x & _M8) << _S8) | ((x >> _S8) & _M8)
                        x = ((x & _M16) << _S16) | ((x >> _S16) & _M16)
                    else:
                        x = ((x & _M8) << _S8) | ((x >> _S8) & _M8)
                        x = ((x & _M16) << _S16) | ((x >> _S16) & _M16)
                        x = (x << _S32) | (x >> _S32)
                    st64(dst, o + k * 8, x)
                for t in range(n8 * 8, rowbytes, es):
                    for k in range(es):
                        dst[o + t + k] = src[a + t + es - 1 - k]
            elif es == 4:
                for j in range(n):
                    dst[o] = src[a + 3]
                    dst[o + 1] = src[a + 2]
                    dst[o + 2] = src[a + 1]
                    dst[o + 3] = src[a]
                    o += 4
                    a += step
            elif es == 8:
                for j in range(n):
                    for k in range(8):
                        dst[o + k] = src[a + 7 - k]
                    o += 8
                    a += step
            elif es == 2:
                for j in range(n):
                    dst[o] = src[a + 1]
                    dst[o + 1] = src[a]
                    o += 2
                    a += step
            else:
                for j in range(n):
                    dst[o] = src[a]
                    o += 1
                    a += step
            d = nd - 2
            while d >= 0:                       # next row
                idx[d] += 1
                addr += strides[d]
                if idx[d] < counts[d]:
                    break
                addr -= counts[d] * strides[d]
                idx[d] = 0
                d -= 1
    return 0
