"""netCDF3, read from the file's bytes: classic (CDF-1), 64-bit offset
(CDF-2) and 64-bit data (CDF-5), as the netCDF File Format Specification
describes them. Metadata is parsed in pure Python from the header (one
read, no numba), attribute values only when first asked for; values are
read straight into the output array.

Two ways to read values, chosen per request:

- runs: when what is asked is long contiguous runs of the file (a whole
  variable, whole levels, a record variable's records), positional reads
  of 1 MiB pieces straight into the output, on the shared thread pool, each
  piece byte-swapped in place right after it lands, while it is still in
  the core's cache. Cached file data reaches the user buffer at 6 GB/s a
  thread, and many threads overlap the page faults on the fresh output.
- gather: short or strided runs (a column, every other point, many small
  records) are copied out of a read-only mapping of the file by numpy's
  strided copy, which swaps the bytes as it goes.

Bytes past the end of a truncated file read as zeros, as netCDF-C's
posixio does (its short reads are zero-filled).
"""
import os
import struct
import sys
import threading

import numpy as np

from ._jit import njit
from ._intrin import ld64, st64
from . import _pool


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

_DATA_MODEL = {1: 'NETCDF3_CLASSIC', 2: 'NETCDF3_64BIT_OFFSET', 5: 'NETCDF3_64BIT_DATA'}

_HEAD = 1 << 16                 # first read of the header; grown when it is longer

PIECE = 1 << 20                 # bytes per positional read on the pool
RUN_MIN = 1 << 16               # runs at least this long are read, not gathered
PAR_MIN = 1 << 21               # requests this big go on the pool
SWAP_NUMPY = 1 << 16            # smaller reads swap with numpy (no numba import)

_u32 = struct.Struct('>I').unpack_from
_u64 = struct.Struct('>Q').unpack_from
_u32x2 = struct.Struct('>II').unpack_from
_u64x2 = struct.Struct('>QQ').unpack_from


class _Short(Exception):
    """The header goes on past the bytes read so far."""


def _decode_name(raw):
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw.decode('utf-8', 'replace')


def _att_list(b, p, ver):
    """(offset of the first attribute, number of attributes, offset past the
    list) for the attribute list whose tag is at p."""
    if ver == 5:
        tag = _u32(b, p)[0]
        n = _u64(b, p + 4)[0]
        p += 12
    else:
        tag, n = _u32x2(b, p)
        p += 8
    if tag != NC_ATTRIBUTE and not (tag == 0 and n == 0):
        raise FormatError('netCDF3: bad attribute list tag %d' % tag)
    first = p
    if ver == 5:
        for _ in range(n):
            ln = _u64(b, p)[0]
            p += 8 + ((ln + 3) & ~3)
            t = _u32(b, p)[0]
            m = _u64(b, p + 4)[0]
            sz = _SIZE.get(t)
            if sz is None:
                raise FormatError('netCDF3: bad attribute type %d' % t)
            p += 12 + ((m * sz + 3) & ~3)
    else:
        for _ in range(n):
            ln = _u32(b, p)[0]
            p += 4 + ((ln + 3) & ~3)
            t, m = _u32x2(b, p)
            sz = _SIZE.get(t)
            if sz is None or t > 6:
                raise FormatError('netCDF3: bad attribute type %d' % t)
            p += 8 + ((m * sz + 3) & ~3)
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
        if ver == 5:
            tag = _u32(b, p)[0]
            n = _u64(b, p + 4)[0]
            p += 12
        else:
            tag, n = _u32x2(b, p)
            p += 8
        if tag != NC_VARIABLE and not (tag == 0 and n == 0):
            raise FormatError('netCDF3: bad variable list tag %d' % tag)
        ndim = len(dims)
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
            for i in ids:
                if i >= ndim:
                    raise FormatError('netCDF3: variable %r uses dimension id %d of %d'
                                      % (name, i, ndim))
            afirst, an, p = _att_list(b, p, ver)
            t = _u32(b, p)[0]
            if t not in _SIZE or (ver != 5 and t > 6):
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
    return ver, numrecs, dims, gfirst, gn, vars_, p


def _read_header(fh, size):
    """The header's bytes and its parse, read in one go when it fits in the
    first 64 KiB (else read again, twice as much each time)."""
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

    def __init__(self, path):
        self.path = os.fspath(path)
        fh = open(self.path, 'rb', buffering=0)
        try:
            self._size = size = os.fstat(fh.fileno()).st_size
            b, (ver, numrecs, dims, gfirst, gn, vars_, hend) = _read_header(fh, size)
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        self._header = b
        self.version = ver
        self.data_model = _DATA_MODEL[ver]
        self.file_format = self.data_model
        self.disk_format = 'NETCDF3'
        self._handles = []              # spare per-thread handles (no pread)
        self._hlock = threading.Lock()
        self._all_handles = []
        self._mm = None
        self._src = None
        self._mlock = threading.Lock()
        self._closed = False

        unlim = [i for i, (_, ln) in enumerate(dims) if ln == 0]
        rec_id = unlim[0] if unlim else -1
        # sizes, element counts and record size as netCDF-C's NC_computeshapes
        specs = []
        recsize = 0
        nrec_vars = 0
        first_rec = None
        for name, ids, afirst, an, t, begin in vars_:
            es = _SIZE[t]
            is_rec = bool(ids) and ids[0] == rec_id
            n = 1
            for k, i in enumerate(ids):
                if not (k == 0 and is_rec):
                    n *= dims[i][1]
            ln = n * es
            if es < 4 and ln % 4:
                ln += 4 - ln % 4                # padded to 4 bytes
            if is_rec:
                recsize += ln
                nrec_vars += 1
                if first_rec is None:
                    first_rec = (n * es, ln, begin)
            specs.append((is_rec, n, ln))
        if nrec_vars == 1 and recsize == first_rec[1]:
            recsize = first_rec[0]              # one record variable: not padded
        self._recsize = recsize
        stream = numrecs == ((1 << 64) - 1 if ver == 5 else 0xFFFFFFFF)
        if stream:                              # streaming: as many records as the file holds
            begin_rec = min((v[5] for v, s in zip(vars_, specs) if s[0]), default=size)
            numrecs = max(0, (size - begin_rec) // recsize) if recsize else 0
        self.numrecs = numrecs
        self.streaming = stream

        root = NC3Group(self, '/', gfirst, gn)
        root.dimensions = {name: ((numrecs, True) if i == rec_id else (ln, False))
                           for i, (name, ln) in enumerate(dims)}
        dnames = [d[0] for d in dims]
        variables = {}
        for (name, ids, afirst, an, t, begin), (is_rec, n, ln) in zip(vars_, specs):
            variables[name] = NC3Variable(self, name, tuple(dnames[i] for i in ids),
                                          tuple(numrecs if (k == 0 and is_rec) else dims[i][1]
                                                for k, i in enumerate(ids)),
                                          t, begin, is_rec, afirst, an)
        root.variables = variables
        self.root = root

    # -- I/O

    def _handle(self):
        """A file handle of this thread's own for seek + readinto (where there
        is no pread: synchronous handles on Windows serialize their reads)."""
        try:
            return self._handles.pop()
        except IndexError:
            fh = open(self.path, 'rb', buffering=0)
            with self._hlock:
                self._all_handles.append(fh)
            return fh

    def _readinto(self, h, mv, off):
        """mv filled from the file at off; bytes past the end are zeros."""
        n = len(mv)
        avail = self._size - off
        if avail < n:
            if avail <= 0:
                mv[:] = b'\0' * n if n < 4096 else bytes(n)
                return
            mv[avail:] = bytes(n - avail)
            mv = mv[:avail]
            n = avail
        got = 0
        if _PREADV:
            fd = self._fh.fileno()
            while got < n:
                k = os.preadv(fd, [mv[got:]], off + got)
                if k <= 0:
                    break
                got += k
        else:
            h.seek(off)
            while got < n:
                k = h.readinto(mv[got:])
                if not k:
                    break
                got += k
        if got < n:                             # the file shrank since it was opened
            mv[got:] = bytes(n - got)

    def _source(self):
        """The whole file as a read-only uint8 array (a mapping, made once)."""
        src = self._src
        if src is None:
            with self._mlock:
                if self._src is None:
                    import mmap
                    if self._size:
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
        for fh in self._all_handles:
            fh.close()
        self._all_handles = []
        self._handles = []
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return '<NC3File %r %s>' % (self.path, self.data_model)


_PREADV = hasattr(os, 'preadv') and sys.platform != 'win32'


class NC3Group:
    """The root group of a netCDF3 file (fastnc's BackendGroup); netCDF3 has
    no other groups."""

    def __init__(self, file, name, afirst, an):
        self.file = file
        self.name = name
        self.groups = {}
        self.dimensions = {}
        self.variables = {}
        self._att = (afirst, an)
        self._attrs = None

    @property
    def attrs(self):
        a = self._attrs
        if a is None:
            f = self.file
            a = self._attrs = _decode_atts(f._header, self._att[0], self._att[1], f.version)
        return a

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
        es = self.dtype.itemsize
        strides = []
        s = es
        for n in reversed(shape[1:] if is_rec else shape):
            strides.append(s)
            s *= n
        if is_rec:
            strides.append(file._recsize)
        self._bstrides = tuple(reversed(strides))   # bytes in the file per index step

    @property
    def attrs(self):
        a = self._attrs
        if a is None:
            f = self.file
            a = self._attrs = _decode_atts(f._header, self._att[0], self._att[1], f.version)
        return a

    def __repr__(self):
        return '<NC3Variable %r %s %s>' % (self.name, self.shape, self.dtype)

    def read(self, key=(), out=None):
        """Raw values for a basic index `key` (one int or slice per
        dimension), native byte order, shaped as numpy shapes x[key]; written
        into `out` when given."""
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
        for k, n, bs in zip(key, shape, self._bstrides):
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
                res = np.empty(oshape, dtype)
        else:
            res = np.empty(oshape, dtype)
        if res.size:
            _read(self.file, self._big, es, base, dims, res)
        if out is not None and res is not out:
            np.copyto(out, res)
            return out
        return res


def _read(f, big, es, base, dims, res):
    """res (C-contiguous, native) filled with the elements at base + sum(i_d *
    stride_d) for i_d < count_d over dims [(count_d, stride_d)] (counts > 1)."""
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
        if nruns == 1 or run >= RUN_MIN:
            _read_runs(f, big, es, base, merged, run, nruns, res)
            return
        merged.append((run // es, es))
    _gather(f, big, es, base, merged, res)


def _read_runs(f, big, es, base, outer, run, nruns, res):
    """Contiguous runs of `run` bytes, at base + the outer dims' offsets, read
    straight into res one after another, then swapped."""
    total = run * nruns
    u8 = res.reshape(-1).view(np.uint8)
    mv = memoryview(u8)
    swap = es > 1
    if nruns == 1 and total <= PAR_MIN:
        h = None if _PREADV else f._handle()
        try:
            f._readinto(h, mv, base)
        finally:
            if h is not None:
                f._handles.append(h)
        if swap:
            if total <= SWAP_NUMPY:
                res.view(res.dtype.newbyteorder()).byteswap(inplace=True)
            else:
                _swap(u8, 0, total, es)
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
    small = total <= SWAP_NUMPY
    ctr = iter(range(npieces))
    lock = threading.Lock()

    def work():
        h = None if _PREADV else f._handle()
        try:
            while True:
                with lock:
                    i = next(ctr, -1)
                if i < 0:
                    return
                o, n = oo[i], nb[i]
                f._readinto(h, mv[o:o + n], fo[i])
                if swap and not small:
                    _swap(u8, o, o + n, es)
        finally:
            if h is not None:
                f._handles.append(h)

    nt = 1 if total <= PAR_MIN else min(npieces, _pool.threads())
    _pool.share(work, nt)
    if swap and small:
        res.view(res.dtype.newbyteorder()).byteswap(inplace=True)


def _gather(f, big, es, base, dims, res):
    """Strided elements copied (and swapped) out of the file's mapping by
    numpy; on the pool, cut along the first dimension, when there are many."""
    counts = tuple(c for c, _ in dims)
    strides = tuple(s for _, s in dims)
    last = base + sum((c - 1) * s for c, s in dims) + es
    src = f._source()
    if last <= src.size:
        view = np.ndarray(counts, big, buffer=src, offset=base, strides=strides)
    else:                                       # past the end of the file: zeros there
        tmp = np.zeros(last - base, np.uint8)
        if base < f._size:
            n = min(last, f._size) - base
            tmp[:n] = src[base:base + n]
        view = np.ndarray(counts, big, buffer=tmp, offset=0, strides=strides)
    dst = res.reshape(counts)
    total = res.nbytes
    c0 = counts[0]
    if total <= PAR_MIN or c0 == 1 or _pool.threads() == 1:
        np.copyto(dst, view)
        return
    nparts = min(c0, 4 * _pool.threads())
    edges = [c0 * i // nparts for i in range(nparts + 1)]
    _pool.map_items(lambda i: np.copyto(dst[edges[i]:edges[i + 1]], view[edges[i]:edges[i + 1]]),
                    range(nparts))


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
