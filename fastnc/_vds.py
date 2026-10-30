"""HDF5 datasets whose values live elsewhere: virtual datasets and
external storage.

A virtual dataset (layout class 3, File Format Specification IV.A.2.i)
keeps a list of mappings in a global heap object. Each names a source
dataset (a file name, "." for this file, and a dataset path), a selection
of the source's elements and a selection of the virtual dataset's, of the
same size: the k-th element of the one is the k-th of the other, each in
its selection's own order (C order for hyperslabs and "all", the stored
order for points). Elements no mapping covers read as the virtual
dataset's fill value, as do those of a source that cannot be opened
(HDF5 does the same); where mappings overlap the later one wins.

Sources are read with fastnc's own readers -- chunked and compressed ones
decode through the engine on every core, many small ones at once on the
pool -- and their files stay mapped in a small cache on the H5File
(`H5File.linked`), closed with it. Source files are found as HDF5 finds
them: an absolute name as it is, then under each HDF5_VDS_PREFIX entry
(`${ORIGIN}`: the virtual dataset's directory), then the virtual dataset's
directory, then the current directory. printf-style names (`%b`: the block
number of an unlimited virtual selection) are expanded, sources looked
for from block 0 until one is missing, as HDF5 does with its default
printf gap of 0.

An unlimited virtual dataset is as long as its sources make it now (HDF5
works the extent out when the dataset is opened, with the default view,
H5D_VDS_LAST_AVAILABLE); `current_shape` does the same.

External storage (the External Data Files message beside a contiguous
layout) keeps the raw values in other files, a list of (name, offset,
size) segments laid end to end, read here with positional reads; bytes
past the end of a file read as zeros, as in HDF5. Relative names are
looked for as HDF5 does (under HDF5_EXTFILE_PREFIX if set, else from the
current directory) and then, where HDF5 would give up, beside the HDF5
file.
"""
import mmap
import os
import re
import threading
from collections import namedtuple

import numpy as np

from . import _arena, _hdf5, _pool, _select

UNLIMITED = (1 << 64) - 1       # H5S_UNLIMITED
SAME_FILE = 0x04                # heap encoding version 1: the source is in this file (".")
PIECE_BYTES = 4 << 20           # mappings smaller than this are read side by side, one thread each
STEP = 1 << 22                  # elements placed one by one: this many at a time
_INF = 1 << 62

SEL_NONE, SEL_POINTS, SEL_HYPERSLABS, SEL_ALL = 0, 1, 2, 3


# ---------------------------------------------------------------- selections

class Axis:
    """One dimension of a regular selection: the first n of the positions
    start + (i // block) * stride + i % block (i = 0, 1, ...)."""
    __slots__ = ('start', 'stride', 'block', 'n')

    def __init__(self, start, stride, block, n):
        if block >= stride:                     # one run (count 1, or blocks touching)
            stride = block = 1
        self.start, self.stride, self.block, self.n = start, stride, block, n

    @classmethod
    def regular(cls, start, stride, count, block, limit=None):
        """A hyperslab's dimension (count / block None: unlimited), only the
        positions below `limit` if given."""
        if block is None:
            ax = cls(start, 1, 1, _INF)
        else:
            ax = cls(start, max(stride, 1) if count != 1 else block, block,
                     _INF if count is None else count * block)
        if limit is not None:
            ax.n = ax.below(limit)
        return ax

    def below(self, x):
        """How many of the positions are below x."""
        if x <= self.start:
            return 0
        q, r = divmod(x - self.start, self.stride)
        return min(self.n, q * self.block + min(r, self.block))

    def at(self, i):
        i = np.asarray(i, np.int64)
        if self.block == 1:
            return self.start + i * self.stride
        return self.start + (i // self.block) * self.stride + i % self.block


class Selection:
    """A dataspace selection as HDF5 serializes it (H5S_SELECT_SERIALIZE):
    `kind` 'none', 'all', 'points' (`points`: coordinates in their order),
    'regular' (`regular`: (start, stride, count, block) per dimension, count
    or block None when unlimited) or 'blocks' (an irregular hyperslab,
    `blocks`: first and last corner of each box; its elements are their
    union in C order)."""

    def __init__(self, kind, rank=None, points=None, regular=None, blocks=None):
        self.kind, self.rank = kind, rank
        self.points, self.regular, self.blocks = points, regular, blocks
        self._union = None

    def __repr__(self):
        if self.kind == 'regular':
            return 'Selection(regular %r)' % (self.regular,)
        if self.kind == 'points':
            return 'Selection(%d points)' % len(self.points)
        if self.kind == 'blocks':
            return 'Selection(%d blocks)' % len(self.blocks)
        return 'Selection(%s)' % self.kind

    def unlimited(self):
        """The dimension along which the selection is unlimited, or -1."""
        if self.kind == 'regular':
            for d, (s, st, c, b) in enumerate(self.regular):
                if c is None or b is None:
                    return d
        return -1

    def axes(self, extent, limit=None):
        """The selection as one Axis per dimension (its elements: their
        product, in C order), or None for points and irregular hyperslabs.
        limit: {dimension: size} keeps only the positions below size there;
        an unlimited dimension stops at the extent otherwise."""
        limit = limit or {}
        if self.kind == 'all':
            return [Axis(0, 1, 1, n if d not in limit else min(n, limit[d]))
                    for d, n in enumerate(extent)]
        if self.kind == 'regular':
            out = []
            for d, (s, st, c, b) in enumerate(self.regular):
                lim = limit.get(d)
                if lim is None and (c is None or b is None):
                    lim = extent[d] if d < len(extent) else 0
                out.append(Axis.regular(s, st, c, b, lim))
            return out
        if self.kind == 'blocks' and len(self.blocks) == 1:
            lo, hi = self.blocks[0]
            return [Axis(int(a), 1, 1, int(z) - int(a) + 1) for a, z in zip(lo, hi)]
        if self.kind == 'points' and len(self.points) == 1:
            return [Axis(int(a), 1, 1, 1) for a in self.points[0]]
        return None

    def npoints(self, extent, limit=None):
        if self.kind == 'none':
            return 0
        ax = self.axes(extent, limit)
        if ax is not None:
            return int(np.prod([a.n for a in ax], dtype=np.int64)) if ax else 1
        return len(self.points) if self.kind == 'points' else len(self.union())

    def union(self):
        """An irregular hyperslab's elements, C order (each once)."""
        if self._union is None:
            rank = self.blocks.shape[2]
            hi = self.blocks[:, 1, :].max(0) + 1
            lin = []
            for lo, up in self.blocks:
                g = np.indices(up - lo + 1).reshape(rank, -1) + lo[:, None]
                lin.append(np.ravel_multi_index(tuple(g), tuple(hi)))
            u = np.unique(np.concatenate(lin))
            self._union = np.stack(np.unravel_index(u, tuple(hi)), axis=1).astype(np.int64)
        return self._union

    def coords(self, k, extent, limit=None):
        """Coordinates of the selection's elements k (an int64 array), as
        an array of shape (len(k), rank)."""
        if self.kind == 'points':
            return self.points[k]
        if self.kind == 'blocks':
            return self.union()[k]
        ax = self.axes(extent, limit)
        if not ax:
            return np.zeros((len(k), 0), np.int64)
        idx = np.unravel_index(k, tuple(a.n for a in ax))
        return np.stack([a.at(i) for a, i in zip(ax, idx)], axis=1)

    def bounds(self, extent):
        """[(first, last)] per dimension (None along an unlimited one), or
        None for an empty selection."""
        if self.kind == 'none':
            return None
        if self.kind == 'all':
            return [(0, n - 1) for n in extent]
        if self.kind == 'regular':
            return [None if c is None or b is None else (s, s + (c - 1) * st + b - 1)
                    for s, st, c, b in self.regular]
        pts = self.points if self.kind == 'points' else self.blocks.reshape(-1, self.blocks.shape[2])
        if not len(pts):
            return None
        return [(int(a), int(z)) for a, z in zip(pts.min(0), pts.max(0))]


def _uint(b, p, n):
    return int.from_bytes(b[p:p + n], 'little')


def _ints(b, p, n, enc):
    """n unsigned integers of enc bytes at b[p]."""
    return np.frombuffer(b, '<u%d' % enc, n, p).astype(np.int64)


def parse_selection(b, p):
    """(Selection, position after it) of the selection serialized at b[p]."""
    kind, version = _uint(b, p, 4), _uint(b, p + 4, 4)
    p += 8
    if kind in (SEL_NONE, SEL_ALL):
        return Selection('none' if kind == SEL_NONE else 'all'), p + 8
    if kind == SEL_POINTS:
        if version == 1:
            rank, n = _uint(b, p + 8, 4), _uint(b, p + 12, 4)
            p += 16
            enc = 4
        elif version == 2:
            enc = b[p]
            rank = _uint(b, p + 1, 4)
            n = _uint(b, p + 5, enc)
            p += 5 + enc
        else:
            raise _hdf5.FormatError('point selection version %d is not supported' % version)
        pts = _ints(b, p, n * rank, enc).reshape(n, rank)
        return Selection('points', rank, points=pts), p + n * rank * enc
    if kind != SEL_HYPERSLABS:
        raise _hdf5.FormatError('unknown selection type %d' % kind)
    if version == 1:
        flags, enc = 0, 4
        p += 8                                          # reserved, length
    elif version == 2:
        flags, enc = b[p], 8
        p += 5                                          # flags, length
    elif version == 3:
        flags, enc = b[p], b[p + 1]
        p += 2
    else:
        raise _hdf5.FormatError('hyperslab selection version %d is not supported' % version)
    rank = _uint(b, p, 4)
    p += 4
    if flags & 1:                                       # regular: start, stride, count, block
        v = [int(x) for x in np.frombuffer(b, '<u%d' % enc, 4 * rank, p)]
        p += 4 * rank * enc
        top = (1 << (8 * enc)) - 1                      # H5S_UNLIMITED in enc bytes
        reg = [(v[4 * d], v[4 * d + 1], None if v[4 * d + 2] == top else v[4 * d + 2],
                None if v[4 * d + 3] == top else v[4 * d + 3]) for d in range(rank)]
        return Selection('regular', rank, regular=reg), p
    n = _uint(b, p, enc)
    p += enc
    blocks = _ints(b, p, n * 2 * rank, enc).reshape(n, 2, rank)
    return Selection('blocks', rank, blocks=blocks), p + n * 2 * rank * enc


# ---------------------------------------------------------------- mappings

class Mapping:
    """One mapping of a virtual dataset: source file and dataset names,
    source selection, virtual selection."""
    __slots__ = ('file_name', 'dset_name', 'src', 'virt', 'printf')

    def __init__(self, file_name, dset_name, src, virt):
        self.file_name, self.dset_name, self.src, self.virt = file_name, dset_name, src, virt
        self.printf = _has_b(file_name) or _has_b(dset_name)


_FORMAT = re.compile(r'%(.)')


def _has_b(name):
    return any(m.group(1) == 'b' for m in _FORMAT.finditer(name))


def _printf(name, i):
    """A source name with %b replaced by block number i (%% by %)."""
    def one(m):
        c = m.group(1)
        if c == '%':
            return '%'
        if c == 'b':
            return str(i)
        raise _hdf5.FormatError('bad format specifier %%%s in virtual source name %r' % (c, name))
    return _FORMAT.sub(one, name)


def _cstr(b, p):
    z = b.index(b'\0', p)
    return b[p:z].decode('utf-8', 'surrogateescape'), z + 1


def parse_mappings(f, lay):
    """The mappings of a virtual dataset's layout: its global heap object
    (version, number of entries, then per entry -- after a flags byte in
    version 1, which HDF5 2.0 writes when it is shorter -- source file name
    unless the source is in this file, source dataset name, source
    selection, virtual selection), checksum checked."""
    coll, idx = lay.data
    blob = bytes(f.global_heap_object(coll, idx))
    version = blob[0]
    if version > 1:
        raise _hdf5.FormatError('virtual dataset mappings of version %d are not supported' % version)
    n = _uint(blob, 1, f.L)
    p = 1 + f.L
    out = []
    for _ in range(n):
        flags = 0
        if version >= 1:
            flags = blob[p]
            p += 1
            if flags & ~SAME_FILE:
                raise _hdf5.FormatError('virtual dataset mapping flags 0x%x are not understood' % flags)
        if flags & SAME_FILE:
            fname = '.'
        else:
            fname, p = _cstr(blob, p)
        dname, p = _cstr(blob, p)
        src, p = parse_selection(blob, p)
        virt, p = parse_selection(blob, p)
        out.append(Mapping(fname, dname, src, virt))
    if p + 4 <= len(blob) and _hdf5.lookup3(blob[:p]) != _uint(blob, p, 4):
        raise _hdf5.FormatError('virtual dataset mappings: checksum mismatch')
    return out


def _clip_extent(reg, slices):
    """The extent along a virtual selection's unlimited dimension (start,
    stride, count, block) that holds exactly `slices` of its positions,
    ending with the last of them (H5S__hyper_get_clip_extent_real, no
    trailing gap)."""
    start, stride, count, block = reg
    if slices <= 0:
        return 0
    if block is None or block == stride or count == 1:
        return start + slices
    q, r = divmod(slices, block)
    if r:
        return start + q * stride + r
    return start + (q - 1) * stride + block


# ---------------------------------------------------------------- files the dataset reads from

class _Linked:
    """An HDF5 file a virtual dataset reads, mapped while its parent is open."""

    def __init__(self, path):
        self.fh = open(path, 'rb')
        try:
            self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
            self.h5 = _hdf5.H5File(self.mm)
        except Exception:
            self.close()
            raise
        self.h5.path = path                             # positional reads for big batches
        self.src = np.frombuffer(self.mm, np.uint8)

    def close(self):
        h5, self.h5 = getattr(self, 'h5', None), None
        if h5 is not None:
            h5.close()
        self.src = None
        mm, self.mm = getattr(self, 'mm', None), None
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                pass                                    # arrays still view it: closed when they go
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def close_linked(linked):
    """Close what H5File.linked holds (called by H5File.close)."""
    for v in linked.values():
        if v is not None:
            try:
                v.close()
            except Exception:
                pass


def _linked(f):
    d = f.__dict__.get('linked')
    if d is None:
        d = f.linked = {}
    return d


def _origin(f):
    """The directory of f's file, or None (remote, file objects)."""
    p = getattr(f, 'path', None)
    return os.path.dirname(os.path.abspath(p)) if p else None


def _candidates(f, name, env):
    """Where HDF5 looks for a file named `name` (H5F_prefix_open_file): an
    absolute name as it is, then its last component (any name: as it is)
    under each prefix of `env`, beside f's file, from the current directory."""
    out = []
    base = name
    if os.path.isabs(name):
        out.append(name)
        base = re.split(r'[\\/]', name)[-1]
    elif os.name == 'nt' and re.match(r'^[A-Za-z]:', name):
        out.append(name)
        base = name[2:]
    origin = _origin(f)
    prefixes = os.environ.get(env)
    if prefixes:
        for pre in prefixes.split(os.pathsep):
            if not pre:
                continue
            if pre.startswith('${ORIGIN}'):
                if origin is None:
                    continue
                pre = origin + pre[len('${ORIGIN}'):]
            out.append(os.path.join(pre, base))
    if origin is not None:
        out.append(os.path.join(origin, base))
    out.append(base)
    return out


def open_hdf5(f, name):
    """The H5File (and its bytes) of a source file named `name`, found as
    HDF5 finds it, kept open with f; None when there is none."""
    linked = _linked(f)
    for path in _candidates(f, name, 'HDF5_VDS_PREFIX'):
        key = ('h5', os.path.normcase(os.path.abspath(path)))
        if key in linked:
            return linked[key]
        if not os.path.isfile(path):
            continue
        try:
            linked[key] = ent = _Linked(path)
            return ent
        except (OSError, ValueError, _hdf5.FormatError):
            continue                                    # not HDF5: HDF5 tries the next
    return None


def find_dataset(h5, path, depth=0):
    """The dataset at `path` in an H5File (soft links followed), or None."""
    parts = [p for p in path.split('/') if p and p != '.']
    g = h5.root
    for i, part in enumerate(parts):
        link = g.links.get(part)
        if link is None:
            return None
        kind, target = link
        if kind == 'soft':
            if depth > 16:
                return None
            here = '/'.join(parts[:i])
            to = target if target.startswith('/') else here + '/' + target
            return find_dataset(h5, to + '/' + '/'.join(parts[i + 1:]), depth + 1)
        if kind != 'hard':
            return None
        h = h5.object_header(target)
        if i == len(parts) - 1:
            return _hdf5.Dataset(h5, target, part, h) if h.first(8) is not None else None
        if h.first(8) is not None or h.first(3) is not None:
            return None
        g = _hdf5.Group(h5, target, part, h)
    return None


class _Source:
    """A source dataset, opened: its Dataset and a Reader (made on first read)."""
    __slots__ = ('ds', 'h5', 'src', '_reader')

    def __init__(self, ds, h5, src):
        self.ds, self.h5, self.src = ds, h5, src
        self._reader = None

    def reader(self, src=None):
        if self._reader is None:
            from . import _h5read
            self._reader = _h5read.Reader(self.h5, self.ds, self.src if self.src is not None else src)
        return self._reader


# ---------------------------------------------------------------- the virtual dataset

_DEPTH = threading.local()


class Virtual:
    """A virtual dataset's mappings and the sources they read."""

    def __init__(self, ds):
        self.ds = ds
        self.f = ds.file
        self.mappings = parse_mappings(ds.file, ds.layout)
        self.src = None                                 # this file's bytes (from the first reader)
        self._sources = {}
        self._printf = {}
        self._parts = None
        self._value = None

    def source(self, file_name, dset_name):
        """The source dataset (a _Source), or None when it cannot be opened."""
        key = (file_name, dset_name)
        if key not in self._sources:
            s = None
            if file_name == '.':
                ds = find_dataset(self.f, dset_name)
                s = _Source(ds, self.f, None) if ds is not None else None
            else:
                ent = open_hdf5(self.f, file_name)
                if ent is not None:
                    ds = find_dataset(ent.h5, dset_name)
                    s = _Source(ds, ent.h5, ent.src) if ds is not None else None
            self._sources[key] = s
        return self._sources[key]

    def printf_sources(self, i):
        """The sources of printf mapping i: block 0, 1, ... up to the first missing one."""
        out = self._printf.get(i)
        if out is None:
            m = self.mappings[i]
            out = []
            while True:
                s = self.source(_printf(m.file_name, len(out)), _printf(m.dset_name, len(out)))
                if s is None:
                    break
                out.append(s)
            self._printf[i] = out
        return out

    def _clip(self, i, m, uv):
        """How far along its unlimited dimension mapping i reaches now, or
        None (its source is missing)."""
        reg = m.virt.regular[uv]
        if m.printf:
            n = len(self.printf_sources(i))
            return _clip_extent(reg, n * (reg[3] or 0))
        s = self.source(m.file_name, m.dset_name)
        us = m.src.unlimited()
        if s is None or us < 0 or us >= len(s.ds.shape):
            return None
        slices = Axis.regular(*m.src.regular[us], limit=s.ds.shape[us]).n
        return _clip_extent(reg, slices)

    def extent(self, stored):
        """The dataset's shape now (H5D__virtual_set_extent_unlim): along an
        unlimited dimension the furthest any mapping's sources reach, at
        least as far as the fixed mappings."""
        rank = len(stored)
        new = [None] * rank
        mins = [0] * rank
        for i, m in enumerate(self.mappings):
            uv = m.virt.unlimited()
            b = m.virt.bounds(stored) if m.virt.kind != 'all' else None
            if b is not None:
                for d in range(min(rank, len(b))):
                    if d != uv and b[d] is not None:
                        mins[d] = max(mins[d], b[d][1] + 1)
            if uv < 0:
                continue
            c = self._clip(i, m, uv)
            if c is not None:
                new[uv] = c if new[uv] is None else max(new[uv], c)
        return tuple(stored[d] if new[d] is None else max(new[d], mins[d]) for d in range(rank))

    def parts(self):
        """What to read for the current shape: [(virtual selection, its
        limits, source, source selection, its limits)], printf mappings
        expanded block by block, unlimited ones clipped where their sources
        end; mappings whose source is missing left out (fill value)."""
        if self._parts is not None:
            return self._parts
        shape = tuple(self.ds.shape)
        out = []
        for i, m in enumerate(self.mappings):
            if m.virt.kind == 'none' or m.src.kind == 'none':
                continue
            uv = m.virt.unlimited()
            if uv < 0:
                s = self.source(m.file_name, m.dset_name)
                if s is not None:
                    out.append((m.virt, None, s, m.src, None))
                continue
            reg = m.virt.regular[uv]
            if m.printf:
                start, stride, count, block = reg
                for k, s in enumerate(self.printf_sources(i)):
                    r = list(m.virt.regular)
                    r[uv] = (start + k * stride, 1, 1, block)
                    out.append((Selection('regular', m.virt.rank, regular=r), {uv: shape[uv]}, s,
                                m.src, None))
                continue
            s = self.source(m.file_name, m.dset_name)
            us = m.src.unlimited()
            if s is None or us < 0:
                continue
            clip = self._clip(i, m, uv)
            if clip is None:
                continue
            vclip = min(clip, shape[uv])
            if vclip == clip:
                sclip = s.ds.shape[us]
            else:                                       # the dataset is shorter: the source to match
                sclip = _clip_extent(m.src.regular[us], Axis.regular(*reg, limit=vclip).n)
            out.append((m.virt, {uv: vclip}, s, m.src, {us: sclip}))
        self._parts = out
        return out

    def value_type(self):
        """(dtype of the values read, fill value as a 0-d array of it)."""
        if self._value is None:
            from . import _h5read
            t = self.ds.type
            raw = self.ds.fill
            vdt = value_dtype(t)[0]
            if t.cls == _hdf5.Datatype.VLEN:
                one = np.empty(1, object)
                one[0] = '' if t.vlen_string else np.zeros(0, _h5read.native(t.base.dtype)
                                                         if t.base.dtype is not None else np.uint8)
                fill = one.reshape(())
            else:
                odd = t.cls == _hdf5.Datatype.FLOAT and t.dtype is None and t.fields is not None \
                    and t.size <= 8
                if raw is None or len(raw) != t.size:
                    fill = np.zeros((), vdt)
                elif odd:
                    fill = _h5read.odd_float(np.frombuffer(raw, 'V%d' % t.size), t).reshape(())
                elif t.dtype is None or t.cls == _hdf5.Datatype.REFERENCE:
                    fill = np.frombuffer(raw, vdt).reshape(()).copy()
                else:
                    a = np.frombuffer(raw, t.dtype)
                    a = _h5read.convert_reduced(a, t) if _h5read.reduced(t) else a.astype(vdt)
                    fill = a.reshape(()).astype(vdt)
            self._value = (vdt, fill)
        return self._value

    # -- reading
    def read_box(self, box, res, threads=None):
        """The values of a box (start, stop, step per dimension) into res."""
        depth = getattr(_DEPTH, 'n', 0)
        if depth > 32:
            raise _hdf5.FormatError('virtual dataset %s maps itself' % self.ds.name)
        _DEPTH.n = depth + 1
        try:
            self._read_box(box, res, threads)
        finally:
            _DEPTH.n = depth

    def _read_box(self, box, res, threads):
        vdt, fill = self.value_type()
        shape = tuple(self.ds.shape)
        plans = []
        for vsel, vlim, s, ssel, slim in self.parts():
            p = _plan(vsel, vlim, s, ssel, slim, shape, box)
            if p is not None:
                plans.append(p)
        if not _covers(plans, res):
            res[...] = fill
        if not plans:
            return
        remote = self.f.source is not None or any(p.source.h5.source is not None for p in plans)
        small = [p for p in plans if p.nbytes(vdt) < PIECE_BYTES]
        if len(small) > 1 and not remote and _disjoint(plans):
            big = [p for p in plans if p.nbytes(vdt) >= PIECE_BYTES]
            for p in big:
                p.run(self, res, vdt, threads)
            _pool.map_items(lambda p: p.run(self, res, vdt, 1), small)
        else:
            for p in plans:                             # in order: where mappings overlap the later wins
                p.run(self, res, vdt, threads)


def value_dtype(t):
    """(dtype of the values a Reader returns for datatype t, True when they
    are its stored values in native order -- so a read can go straight into
    an array of that dtype)."""
    from . import _h5read
    if t.cls == _hdf5.Datatype.VLEN:
        return np.dtype(object), False
    if t.cls == _hdf5.Datatype.FLOAT and t.dtype is None and t.fields is not None and t.size <= 8:
        return _h5read.float_target(t), False
    if t.dtype is None or t.cls == _hdf5.Datatype.REFERENCE:
        return np.dtype('V%d' % t.size), False
    return _h5read.native(t.dtype), not _h5read.reduced(t)


def virtual(ds):
    """The Virtual of a _hdf5.Dataset with layout class 3 (made once)."""
    v = ds.__dict__.get('_vds')
    if v is None:
        v = ds._vds = Virtual(ds)
    return v


VDSmap = namedtuple('VDSmap', ('vspace', 'file_name', 'dset_name', 'src_space'))


class SelectionSpace:
    """A mapping's selection as h5py's Dataset.virtual_sources() gives it
    (an h5py.h5s.SpaceID there): the dataspace's extent and the selection,
    with SpaceID's methods for reading them. Constants as h5py.h5s's."""

    def __init__(self, sel, shape):
        self.sel = sel
        self.shape = tuple(shape)

    def __repr__(self):
        return '<fastnc SelectionSpace %r of %r>' % (self.sel, self.shape)

    def get_simple_extent_dims(self):
        return self.shape

    def get_simple_extent_ndims(self):
        return len(self.shape)

    def get_select_type(self):
        return {'none': SEL_NONE, 'points': SEL_POINTS, 'all': SEL_ALL}.get(self.sel.kind, SEL_HYPERSLABS)

    def _regular(self):
        """(start, stride, count, block) per dimension when the selection is
        a regular hyperslab (as HDF5 rebuilds one from a list of blocks),
        else None."""
        s = self.sel
        if s.kind == 'regular':
            return [(a, st if c not in (1,) else 1, UNLIMITED if c is None else c,
                     UNLIMITED if b is None else b) for a, st, c, b in s.regular]
        if s.kind != 'blocks':
            return None
        u = s.union()
        dims = [np.unique(u[:, d]) for d in range(u.shape[1])]
        if len(u) != int(np.prod([len(p) for p in dims])):
            return None                                 # not a product of positions
        out = []
        for p in dims:
            brk = np.nonzero(np.diff(p) != 1)[0]
            starts = np.concatenate([[p[0]], p[brk + 1]])
            ends = np.concatenate([p[brk], [p[-1]]])
            lens = ends - starts + 1
            if (lens != lens[0]).any():
                return None
            if len(starts) > 1:
                st = np.diff(starts)
                if (st != st[0]).any():
                    return None
                out.append((int(p[0]), int(st[0]), len(starts), int(lens[0])))
            else:
                out.append((int(p[0]), 1, 1, int(lens[0])))
        return out

    def is_regular_hyperslab(self):
        return self._regular() is not None

    def get_regular_hyperslab(self):
        r = self._regular()
        if r is None:
            raise ValueError('not a regular hyperslab selection')
        return tuple(tuple(x[i] for x in r) for i in range(4))

    def get_select_npoints(self):
        return self.sel.npoints(self.shape)

    def get_select_bounds(self):
        s = self.sel
        if s.kind == 'none':
            return None
        if s.kind == 'regular':
            ax = s.axes(self.shape)
            return (tuple(a.start for a in ax), tuple(int(a.at(a.n - 1)) for a in ax))
        b = s.bounds(self.shape)
        return (tuple(x[0] for x in b), tuple(x[1] for x in b))

    def get_select_elem_npoints(self):
        return len(self.sel.points) if self.sel.kind == 'points' else 0

    def get_select_elem_pointlist(self):
        return self.sel.points.astype(np.uint64).copy() if self.sel.kind == 'points' else None


def virtual_sources(ds):
    """[VDSmap(vspace, file_name, dset_name, src_space)] as h5py gives
    them: the virtual selection over the dataset's extent, the source
    selection over an extent HDF5 has not filled in (rank 0) for "all"."""
    out = []
    for m in virtual(ds).mappings:
        if m.src.kind == 'all' or m.src.kind == 'none':
            sshape = ()
        elif m.src.kind == 'regular':
            sshape = tuple(UNLIMITED if c is None or b is None else s + (c - 1) * st + b
                           for s, st, c, b in m.src.regular)
        else:
            pts = m.src.points if m.src.kind == 'points' else m.src.blocks.reshape(-1, m.src.rank)
            sshape = tuple(int(x) + 1 for x in pts.max(0))
        out.append(VDSmap(SelectionSpace(m.virt, ds.shape), m.file_name, m.dset_name,
                          SelectionSpace(m.src, sshape)))
    return out


def current_shape(ds):
    """The shape of an unlimited virtual dataset now (its stored one when
    no source says otherwise)."""
    return virtual(ds).extent(tuple(ds.shape))


def sources(ds):
    """[(virtual selection, file name, dataset name, source selection)] as stored."""
    return [(m.virt, m.file_name, m.dset_name, m.src) for m in virtual(ds).mappings]


def read(reader, key, out=None, threads=None, scalars=True):
    """Raw values of a virtual dataset for an index (as Reader.read)."""
    ds = reader.ds
    v = virtual(ds)
    if v.src is None:
        v.src = reader.src
    box, post, drop = _select.normalize(key, ds.shape)
    shape = _select.box_shape(box)
    vdt = v.value_type()[0]
    direct = out is not None and post is None and not drop and out.shape == shape \
        and out.dtype == vdt and out.flags.c_contiguous
    if direct:
        res = out
    elif vdt == object:
        res = np.empty(shape, object)
    else:
        res = _arena.empty(shape, vdt)
    if res.size:
        v.read_box(box, res, threads)
    res = _select.finish(res, post, drop, scalars)
    if out is not None and not direct:
        out[...] = res
        return out
    return res


# ---------------------------------------------------------------- placing one mapping

def _as_slice(idx):
    """idx (sorted, distinct) as a slice if it is evenly spaced, else None."""
    if len(idx) == 1:
        return slice(int(idx[0]), int(idx[0]) + 1, 1)
    d = np.diff(idx)
    if d[0] > 0 and (d == d[0]).all():
        return slice(int(idx[0]), int(idx[-1]) + 1, int(d[0]))
    return None


def _convert(vals, vdt):
    """Source values in the virtual dataset's type (HDF5 converts them as
    it reads): floats to integers clipped to the integer's range, NaN as 0."""
    if vals.dtype == vdt:
        return vals
    if vdt.kind in 'iu' and vals.dtype.kind == 'f':
        info = np.iinfo(vdt)
        vals = np.clip(np.nan_to_num(vals, nan=0.0), info.min, info.max)
    return vals.astype(vdt)


class _Box:
    """A mapping's part of a read that is a box on both sides: output
    indexes per virtual dimension, source indexes per source dimension
    (each a slice, or an index array when not evenly spaced)."""
    __slots__ = ('source', 'okey', 'oshape', 'skey', 'stake', 'count', 'olo', 'ohi')

    def nbytes(self, vdt):
        return self.count * vdt.itemsize

    def run(self, v, res, vdt, threads):
        rd = self.source.reader(v.src)
        target = None
        sdt, plain = value_dtype(self.source.ds.type)
        if plain and sdt == vdt and all(isinstance(k, slice) for k in self.okey) \
                and self.stake is None and res.ndim:
            view = res[self.okey]
            if view.flags.c_contiguous and view.dtype == vdt:
                target = view.reshape(rd_shape(self.skey))
        vals = rd.read(self.skey, out=target, threads=threads, scalars=False)
        if target is not None and vals is target:
            return
        if self.stake is not None:
            for d, t in enumerate(self.stake):
                if t is not None:
                    vals = np.take(vals, t, axis=d)
        vals = _convert(np.asarray(vals), vdt).reshape(self.oshape)
        if all(isinstance(k, slice) for k in self.okey):
            res[self.okey] = vals
        else:
            res[np.ix_(*[np.arange(k.start, k.stop, k.step) if isinstance(k, slice) else k
                         for k in self.okey])] = vals


def rd_shape(skey):
    return tuple(len(range(k.start, k.stop, k.step)) for k in skey)


class _Elements:
    """A mapping placed element by element (point selections, irregular
    hyperslabs, selections of different shapes)."""
    __slots__ = ('source', 'vsel', 'vlim', 'ssel', 'slim', 'shape', 'sshape', 'box', 'n')
    count = 0

    def nbytes(self, vdt):
        return _INF

    def run(self, v, res, vdt, threads):
        rd = self.source.reader(v.src)
        r0 = np.array([s for s, e, k in self.box], np.int64)
        r1 = np.array([e for s, e, k in self.box], np.int64)
        rk = np.array([k for s, e, k in self.box], np.int64)
        sext = np.array(self.sshape, np.int64)
        for k0 in range(0, self.n, STEP):
            k = np.arange(k0, min(self.n, k0 + STEP), dtype=np.int64)
            vc = self.vsel.coords(k, self.shape, self.vlim)
            inside = ((vc >= r0) & (vc < r1) & ((vc - r0) % rk == 0)).all(axis=1)
            if not inside.any():
                continue
            vc = vc[inside]
            sc = self.ssel.coords(k[inside], self.sshape, self.slim)
            if len(sext) and (sc >= sext).any():
                raise OSError('virtual dataset %s: source %s is smaller than its selection'
                              % (v.ds.name, self.source.ds.name))
            vals = _convert(np.asarray(_gather(rd, sc, threads)), vdt)
            res[tuple(((vc - r0) // rk).T)] = vals


def _gather(rd, sc, threads):
    """Source values at coordinates sc (n, rank): boxes around them read
    and picked from, a box split while it is much bigger than its points."""
    n = len(sc)
    if sc.shape[1] == 0:
        v = np.asarray(rd.read((), threads=threads, scalars=False))
        return np.repeat(v.reshape(1), n)
    lo = sc.min(0)
    hi = sc.max(0) + 1
    if n <= 1 or int(np.prod(hi - lo)) <= 4 * n + (1 << 20):
        vals = rd.read(tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), threads=threads,
                       scalars=False)
        return vals[tuple((sc - lo).T)]
    d = int(np.argmax(hi - lo))
    m = sc[:, d] < lo[d] + (hi[d] - lo[d]) // 2
    a = _gather(rd, sc[m], threads)
    b = _gather(rd, sc[~m], threads)
    out = np.empty((n,) + a.shape[1:], a.dtype)
    out[m] = a
    out[~m] = b
    return out


def _plan(vsel, vlim, source, ssel, slim, shape, box):
    """How one mapping places its part of the box (a _Box, an _Elements),
    or None when it has nothing there."""
    sshape = tuple(source.ds.shape)
    vax = vsel.axes(shape, vlim)
    sax = ssel.axes(sshape, slim)
    nv = vsel.npoints(shape, vlim)
    if nv == 0:
        return None
    ns = ssel.npoints(sshape, slim)
    if ns != nv:
        raise OSError('virtual dataset: %d elements mapped from %d of source %s'
                      % (nv, ns, source.ds.name))
    if vax is not None and sax is not None and \
            [a.n for a in vax if a.n != 1] == [a.n for a in sax if a.n != 1]:
        return _box_plan(vax, sax, source, sshape, box)
    # does the virtual selection touch the box at all?
    b = vsel.bounds(shape)
    if b is not None:
        for (s, e, k), bd in zip(box, b):
            if bd is not None and (bd[1] < s or bd[0] >= e):
                return None
    p = _Elements()
    p.source, p.vsel, p.vlim, p.ssel, p.slim = source, vsel, vlim, ssel, slim
    p.shape, p.sshape, p.box, p.n = shape, sshape, box, nv
    return p


def _box_plan(vax, sax, source, sshape, box):
    paired = []                                         # per virtual dimension: element indexes, or None
    okey = []
    oshape = []
    olo, ohi = [], []
    for ax, (r0, r1, rk) in zip(vax, box):
        lo, hi = ax.below(r0), ax.below(r1)
        if lo >= hi:
            return None
        i = np.arange(lo, hi, dtype=np.int64)
        pos = ax.at(i)
        if rk > 1:
            keep = (pos - r0) % rk == 0
            i, pos = i[keep], pos[keep]
            if not len(i):
                return None
        o = (pos - r0) // rk
        sl = _as_slice(o)
        okey.append(sl if sl is not None else o)
        oshape.append(len(o))
        olo.append(int(o[0]))
        ohi.append(int(o[-1]))
        if ax.n != 1:
            paired.append(i)
    if len(sax) != len(sshape):
        raise OSError('virtual dataset: source %s has rank %d, its selection %d'
                      % (source.ds.name, len(sshape), len(sax)))
    skey, stake = [], []
    take = False
    it = iter(paired)
    for ax, n in zip(sax, sshape):
        i = next(it) if ax.n != 1 else np.zeros(1, np.int64)
        pos = ax.at(i)                                  # (increasing)
        if pos[-1] >= n:
            raise OSError('virtual dataset: source %s is smaller than its selection' % source.ds.name)
        sl = _as_slice(pos)
        if sl is not None:
            skey.append(sl)
            stake.append(None)
        else:
            skey.append(slice(int(pos[0]), int(pos[-1]) + 1, 1))
            stake.append(pos - pos[0])
            take = True
    p = _Box()
    p.source, p.okey, p.oshape = source, tuple(okey), tuple(oshape)
    p.skey, p.stake = tuple(skey), (stake if take else None)
    p.count = int(np.prod(oshape, dtype=np.int64)) if oshape else 1
    p.olo, p.ohi = olo, ohi
    return p


def _disjoint(plans):
    """True when no two plans' output boxes overlap (and all are boxes)."""
    if any(not isinstance(p, _Box) for p in plans):
        return False
    if len(plans) > 2048:
        return False
    if len(plans) < 2:
        return True
    lo = np.array([p.olo for p in plans], np.int64)
    hi = np.array([p.ohi for p in plans], np.int64)
    if lo.ndim == 1 or lo.shape[1] == 0:
        return False
    for i in range(len(plans) - 1):
        ov = ((lo[i + 1:] <= hi[i]) & (lo[i] <= hi[i + 1:])).all(axis=1)
        if ov.any():
            return False
    return True


def _covers(plans, res):
    """True when the plans write every element of res (so it needs no fill)."""
    if not plans or any(not isinstance(p, _Box) for p in plans):
        return False
    if sum(p.count for p in plans) != res.size:
        return False
    return len(plans) == 1 or _disjoint(plans)


# ---------------------------------------------------------------- external storage

def _raw(f, name):
    """A positional reader (and the file's size) of an external raw data
    file, kept open with f."""
    from . import _pio
    linked = _linked(f)
    cands = []
    if not os.path.isabs(name):
        pre = os.environ.get('HDF5_EXTFILE_PREFIX')
        origin = _origin(f)
        if pre and pre != '.':
            if pre.startswith('${ORIGIN}') and origin is not None:
                pre = origin + pre[len('${ORIGIN}'):]
            cands.append(os.path.join(pre, name))
        cands.append(name)
        if origin is not None:
            cands.append(os.path.join(origin, name))    # (beyond HDF5: beside the HDF5 file)
    else:
        cands.append(name)
    for path in cands:
        key = ('raw', os.path.normcase(os.path.abspath(path)))
        if key in linked:
            return linked[key]
        if os.path.isfile(path):
            r = _pio.Reader(path)
            r.size = os.path.getsize(path)
            linked[key] = r
            return r
    raise FileNotFoundError('external raw data file %r of an HDF5 dataset not found (looked at %s)'
                            % (name, ', '.join(map(repr, cands))))


def external_bytes(f, segments, start, n):
    """Bytes [start, start + n) of a dataset stored in external files
    (segments: (name, offset, size) laid end to end), as a uint8 array;
    bytes past the end of a file are zeros."""
    out = np.zeros(n, np.uint8)
    pieces = []
    pos = 0
    end = start + n
    for name, off, size in segments:
        seg_end = _INF if size == UNLIMITED else pos + size
        lo, hi = max(start, pos), min(end, seg_end)
        if lo < hi:
            r = _raw(f, name)
            fo = off + (lo - pos)
            avail = max(0, min(hi - lo, r.size - fo))
            for k in range(0, avail, PIECE_BYTES):
                pieces.append((r, fo + k, lo - start + k, min(PIECE_BYTES, avail - k)))
        pos = seg_end
        if pos >= end:
            break
    if pos < end:
        raise OSError('HDF5 external storage: the files hold %d bytes of a dataset of %d' % (pos, end))
    base = out.ctypes.data
    short = []

    def one(pc):
        r, fo, o, ln = pc
        if r._io.read(out, base, o, ln, fo) < ln:
            short.append(pc)
    _pool.map_items(one, pieces)
    if short:
        raise OSError('HDF5 external storage: a read came back short')
    return out


def read_external(ds, box, res):
    """The stored bytes of a box of an externally stored dataset into res
    (a C-contiguous array of the box's shape, same item size): the bytes
    from its first to its last element read, the elements picked."""
    esize = ds.type.size
    shape = tuple(ds.shape)
    rank = len(shape)
    stride = [esize] * rank
    for d in range(rank - 2, -1, -1):
        stride[d] = stride[d + 1] * shape[d + 1]
    counts = _select.box_shape(box)
    first = sum(s * st for (s, e, k), st in zip(box, stride))
    last = sum((s + (c - 1) * k) * st for (s, e, k), c, st in zip(box, counts, stride)) + esize
    buf = external_bytes(ds.file, ds.external, first, last - first)
    vdt = np.dtype('V%d' % esize)
    picked = np.ndarray(counts, vdt, buffer=buf, offset=0,
                        strides=tuple(k * st for (s, e, k), st in zip(box, stride)))
    res.reshape(-1).view(vdt)[:] = picked.reshape(-1)
