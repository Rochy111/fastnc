"""Mask and scale: what netCDF4-python's `Variable.__getitem__` does to the
raw values after reading them (`_Unsigned`, `_FillValue` and the default
fill values, `missing_value`, `valid_min` / `valid_max` / `valid_range`,
`scale_factor` / `add_offset`, masked array or plain array, char -> str),
reproduced exactly: same type, dtype, values, mask and fill value as
netCDF4-python 1.7.x under the numpy that is installed.

There are two paths. `_reference` is netCDF4-python's own code (1.7.4's
`__getitem__` tail, `_toma`, `_check_safecast`, `chartostring`) carried
over line for line onto an attribute dict, so numpy's own rules decide
every dtype and value: the promotion of `data * scale_factor` (numpy 1's
value-based casting and numpy 2's NEP 50 differ), the masked arithmetic
that puts the unscaled value back under the mask, the fill value a
masked array carries through arithmetic (numpy 2 converts it to the new
dtype, numpy 1 keeps the old one). It handles everything: strings, enums,
compounds, vlens, scalars, odd attributes.

The fast path covers the common case -- a numeric array, primitive or
enum, with float scaling or none -- in one pass over memory: a numba
kernel run on fastnc's threads takes 64k values at a time and, while
they are in cache, writes the scaled values, flags the masked ones
(missing values, fill value, default fill, valid range, NaN) and, only
in blocks where something is masked, writes the mask and puts the
unscaled value back under it as numpy's masked arithmetic does. A mask
nothing is in is never paged in. What the kernel cannot know cheaply --
the result dtype, the fill value the masked array ends up with, the
masked array's bookkeeping -- comes from running the reference path on a
one-element stand-in with the same attributes, so numpy decides those
exactly as it would for the whole array. The values are numpy's to the
bit: each operation is done in the float type numpy does it in (the
result type of `data * scale_factor`, then that of adding
`add_offset`), with the constants cast to it as numpy casts them, as two
roundings (no fastmath, so no fused multiply-add), and integers go to
float by one direct conversion.
"""
import itertools
import threading
import time
import warnings

import numpy

from . import _arena, _chunks, _hw, _pool

from ._chunks import holds_value, value_words
from ._jit import bind, njit
from ._intrin import ld64, take

ma = numpy.ma

# netCDF4.default_fillvals (netcdf.h's NC_FILL_*)
DEFAULT_FILLVALS = {
    'S1': '\0',
    'i1': -127,
    'u1': 255,
    'i2': -32767,
    'u2': 65535,
    'i4': -2147483647,
    'u4': 4294967295,
    'i8': -9223372036854775806,
    'u8': 18446744073709551614,
    'f4': 9.969209968386869e+36,
    'f8': 9.969209968386869e+36,
    'c8': 9.969209968386869e+36,
    'c16': 9.969209968386869e+36,
}

FAST_MIN_SIZE = 1 << 15         # below this the reference path is faster (the fast one: ~70 us fixed)
BLOCK = _hw.block(1 << 16)      # elements a thread takes at a time (fits this machine's L2)
SOLO_BYTES = _hw.l2_bytes() or 1 << 20      # arrays that fit a core's L2 cache are tested for
                                            # masked values on one thread


# ---------------------------------------------------------------------------
# the interface

def apply(raw, attrs, dtype, mask=True, scale=True, always_mask=True,
          chartostring=True, var_dtype=None, xp=numpy, no_fill=False,
          var_shape=None, key=None, threads=None, clean=False):
    """netCDF4-python's Variable.__getitem__ post-processing of raw values
    `raw` (what read() returned) for a variable with attributes `attrs`
    (as getncattr returns them) and stored dtype `dtype` (numpy dtype, or
    str for NC_STRING): _Unsigned, _FillValue / default fill values,
    missing_value, valid_min / valid_max / valid_range, scale_factor /
    add_offset with their dtype promotion, masked array vs plain array,
    char -> str with _Encoding. Same values, dtypes, masks and fill values
    as netCDF4-python 1.7.x.

    `var_dtype`: the netCDF kind when `dtype` alone does not tell it --
    'enum', 'vlen', 'compound' (or the netCDF4 EnumType / VLType /
    CompoundType object; a CompoundType's dtype_view is then used).
    `no_fill`: what nc_inq_var_fill says (netCDF4: `v.get_fill_value() is
    None`); it decides only whether a byte variable without _FillValue is
    masked at the default fill value (it is when filling is on).
    `var_shape`, `key`: the variable's shape and the normalised key read
    (one int or slice per dimension); they decide only whether a char
    variable's last dimension was read whole, as chartostring requires;
    and a 0-d raw of an array variable (every index an int) becomes
    numpy's scalar first, as netCDF4-python's indexing makes it. Missing,
    they mean the whole variable was read.
    `threads`: at most this many threads (default: fastnc's pool).
    `clean`: the read proved (a Probe) that raw holds none of the values
    this variable's mask-only plan looks for; no pass over raw then.
    `xp=cupy`: raw is a cupy array; see _maskscale_gpu.apply.
    """
    if xp is not numpy:
        from . import _maskscale_gpu
        return _maskscale_gpu.apply(raw, attrs, dtype, mask=mask, scale=scale,
                                    always_mask=always_mask, chartostring=chartostring,
                                    var_dtype=var_dtype, no_fill=no_fill,
                                    var_shape=var_shape, key=key)
    kind = _kind(dtype, var_dtype)
    if isinstance(raw, numpy.ndarray) and raw.ndim == 0 and var_shape is not None \
            and len(var_shape) > 0:
        raw = raw[()]               # one value of an array variable: numpy's scalar
    plan = fast_plan(raw, attrs, dtype, kind, mask, scale, no_fill)
    if plan is not None:
        if clean and plan.few is not None:
            return finish(plan, raw, None, None, (0, 0, 0, 0), always_mask)
        if _kernels_ready():
            return run_plan(plan, raw, always_mask, threads)
        t = time.perf_counter()
        r = run_plan_numpy(plan, raw, always_mask, threads)
        _chunks.slow(time.perf_counter() - t)
        return r
    return _reference(raw, attrs, dtype, kind, mask, scale, always_mask, chartostring,
                      no_fill, var_shape, key, var_dtype)


def fast_plan(raw, attrs, dtype, kind, mask, scale, no_fill):
    """The Plan raw goes through the kernels by, or None for the reference path."""
    if (kind in ('primitive', 'enum') and type(raw) is numpy.ndarray
            and raw.ndim > 0 and raw.size >= FAST_MIN_SIZE and raw.flags.c_contiguous):
        return make_plan(raw.dtype, attrs, dtype, kind, mask, scale, no_fill)
    return None


def _kernels_ready():
    """Whether to use the kernels (until they are loaded -- see
    _chunks.LOAD_AFTER -- run_plan_numpy does the work)."""
    return _chunks.ready()


def _kind(dtype, var_dtype):
    """'primitive', 'enum', 'vlen', 'compound' or 'string'."""
    if var_dtype is not None:
        name = var_dtype if isinstance(var_dtype, str) else type(var_dtype).__name__
        name = name.lower()
        for k in ('enum', 'vlen', 'compound'):
            if k in name or (k == 'vlen' and name == 'vltype'):
                return k
        if name in ('string', 'str'):
            return 'string'
    if dtype is str:
        return 'string'
    if getattr(dtype, 'names', None):
        return 'compound'
    return 'primitive'


# ---------------------------------------------------------------------------
# the reference path: netCDF4-python 1.7.4's code on an attribute dict

class _Var:
    """Enough of a netCDF4.Variable for the code carried over: attributes
    by name, dtype, the type flags."""

    def __init__(self, attrs, dtype, kind, mask, scale, always_mask, chartostring,
                 no_fill, var_dtype):
        self.attrs, self.dtype = attrs, dtype
        self.mask, self.scale, self.always_mask = mask, scale, always_mask
        self.chartostring, self.no_fill = chartostring, no_fill
        self._isprimitive = kind == 'primitive'
        self._isenum = kind == 'enum'
        self._isvlen = kind in ('vlen', 'string')
        self._iscompound = kind == 'compound'
        self._dtype_view = None
        if self._iscompound:
            self._dtype_view = getattr(var_dtype, 'dtype_view', None)
            if self._dtype_view is None:
                self._dtype_view = _set_viewdtype(numpy.dtype(dtype))

    def has(self, name):
        return name in self.attrs

    def get(self, name, default=None):
        return self.attrs.get(name, default)


def _safecast(a, b):
    # netCDF4.utils._safecast
    try:
        is_safe = ((a == b) | (numpy.isnan(a) & numpy.isnan(b))).all()
    except Exception:
        try:
            is_safe = (a == b).all()
        except Exception:
            is_safe = False
    return is_safe


def _check_safecast(self, attname, warn=warnings.warn):
    msg = """WARNING: %s not used since it
cannot be safely cast to variable data type""" % attname
    if self.has(attname):
        safe = getattr(self.attrs, 'safe', ())      # (one variable's: its type does not change)
        if attname in safe:
            return True
        att = numpy.array(self.attrs[attname])
    else:
        return False
    try:
        atta = numpy.array(att, self.dtype)
    except ValueError:
        warn(msg)
        return False
    is_safe = _safecast(att, atta)
    if not is_safe:
        warn(msg)
    elif safe != ():
        safe.add(attname)
    return is_safe


def _unpacked_units(self, attname):
    """Whether the valid range attribute `attname` of a packed variable is
    in unpacked units: the NUG and CF rule (netCDF-Java follows it), by
    type -- a range of the type of `scale_factor` / `add_offset`, not that
    of the stored integers. netCDF4-python tests every range against the
    stored values; for such a file (HRD's winds: `valid_range = -900., 900.`
    knots on int16 hundredths of a knot) that masks nearly every value."""
    if not self.has(attname) or self.dtype.kind not in 'iu':
        return False
    adt = numpy.asarray(self.attrs[attname]).dtype
    if adt.kind != 'f' or adt == self.dtype:
        return False
    if not _valid_scaleoffset(self, lambda msg: None):
        return False
    return any(self.has(n) and numpy.asarray(self.attrs[n]).dtype == adt
               for n in ('scale_factor', 'add_offset'))


def _packed_bounds(self, xdt, lo, hi):
    """The stored values (of type xdt, whole numbers) whose unpacked values
    lie in [lo, hi] (either may be None), as the bounds netCDF4-python tests
    stored values against: (low, high) in xdt, each None when nothing is
    outside on that side. Unpacking is numpy's own (`_scale_ops`, one value
    at a time) and monotonic, so each bound masks a run from one end of the
    type, found by bisection: the mask is exactly where the unpacked value
    is outside the range, in the float type it is read in."""
    info = numpy.iinfo(xdt)
    tmin, tmax = int(info.min), int(info.max)

    def masked(k, b, below):
        u = _scale_ops(self, numpy.array([k], xdt))
        return bool((u < b)[0] if below else (u > b)[0])

    low = high = None
    for b, below in ((lo, True), (hi, False)):
        if b is None:
            continue
        at_min, at_max = masked(tmin, b, below), masked(tmax, b, below)
        if at_min and at_max:
            return numpy.array(tmax, xdt), numpy.array(tmin, xdt)   # (every value masked)
        a, z = tmin, tmax
        if at_min:                              # masked from tmin up to a
            while z - a > 1:
                m = (a + z) // 2
                if masked(m, b, below):
                    a = m
                else:
                    z = m
            low = z if low is None else max(low, z)
        elif at_max:                            # masked from z up to tmax
            while z - a > 1:
                m = (a + z) // 2
                if masked(m, b, below):
                    z = m
                else:
                    a = m
            high = a if high is None else min(high, a)
    if low is not None and high is not None and low > high:
        return numpy.array(tmax, xdt), numpy.array(tmin, xdt)
    return (None if low is None else numpy.array(low, xdt),
            None if high is None else numpy.array(high, xdt))


def _valid_bounds(self, view, warn):
    """`_toma`'s valid range bounds (validmin, validmax, each None when not
    given), in the type values are compared in (the unsigned view, if
    any). Ranges in unpacked units (_unpacked_units) become the stored
    values they allow; any other is netCDF4-python's, in stored units. (An
    _Unsigned variable read unscaled holds signed values, among which the
    ones an unpacked range allows are no run: such a range is not applied.)"""
    attrs, dtype = self.attrs, self.dtype
    unp = {n: _unpacked_units(self, n) for n in ('valid_range', 'valid_min', 'valid_max')}
    signed = (view is None and dtype.kind == 'i'
              and self.get('_Unsigned', False) in ['True', 'true'])
    safe = {n: unp[n] or _check_safecast(self, n, warn)
            for n in ('valid_range', 'valid_min', 'valid_max')}
    if safe['valid_range'] and attrs['valid_range'].size == 2:
        names = ('valid_range', 'valid_range')
        vals = (attrs['valid_range'][0], attrs['valid_range'][1])
    else:
        names = ('valid_min', 'valid_max')
        vals = tuple(attrs[n] if safe[n] else None for n in names)
    stored = [None, None]                       # netCDF4-python's bounds, in stored units
    for i in (0, 1):
        if vals[i] is not None and not unp[names[i]]:
            b = numpy.array(vals[i], dtype)
            stored[i] = b.view(view) if view is not None else b
    lo = vals[0] if vals[0] is not None and unp[names[0]] and not signed else None
    hi = vals[1] if vals[1] is not None and unp[names[1]] and not signed else None
    if lo is None and hi is None:
        return stored[0], stored[1]
    validmin, validmax = _packed_bounds(self, numpy.dtype(view) if view is not None else dtype,
                                        lo, hi)
    if stored[0] is not None:                   # (one bound of each kind: valid_min / _max)
        validmin = stored[0] if validmin is None else max(validmin, stored[0])
    if stored[1] is not None:
        validmax = stored[1] if validmax is None else min(validmax, stored[1])
    return validmin, validmax


def _valid_scaleoffset(self, warn=warnings.warn):
    try:
        if self.has('scale_factor'):
            float(self.attrs['scale_factor'])
        if self.has('add_offset'):
            float(self.attrs['add_offset'])
        return True
    except Exception:
        if self.scale:
            warn('invalid scale_factor or add_offset attribute, no unpacking done...')
        return False


def _scale_ops(self, data):
    """The scale_factor / add_offset step of __getitem__."""
    if self.has('scale_factor') and self.has('add_offset'):
        sf, off = self.attrs['scale_factor'], self.attrs['add_offset']
        if off != 0.0 or sf != 1.0:
            data = data * sf + off
        else:
            data = data.astype(sf.dtype)            # issue 913
    elif self.has('scale_factor') and self.attrs['scale_factor'] != 1.0:
        data = data * self.attrs['scale_factor']
    elif self.has('add_offset') and self.attrs['add_offset'] != 0.0:
        data = data + self.attrs['add_offset']
    return data


def _reference(data, attrs, dtype, kind, mask, scale, always_mask, chartostring,
               no_fill, var_shape, key, var_dtype):
    self = _Var(attrs, dtype, kind, mask, scale, always_mask, chartostring, no_fill,
                var_dtype)
    valid_scaleoffset = _valid_scaleoffset(self)
    if self.mask and (self._isprimitive or self._isenum):
        data = _toma(self, data)
    else:
        if self.scale:
            is_unsigned = self.get('_Unsigned', False) in ["true", "True"]
            if is_unsigned and data.dtype.kind == 'i':
                data = data.view('%su%s' % (data.dtype.byteorder, data.dtype.itemsize))
    if self.scale and (self._isprimitive or (self._isvlen and self.dtype != str)) \
            and valid_scaleoffset:
        data = _scale_ops(self, data)
    if self.chartostring and getattr(self.dtype, 'kind', None) == 'S' and \
            getattr(self.dtype, 'itemsize', None) == 1:
        encoding = self.get('_Encoding')
        if encoding is not None:
            shape = tuple(var_shape) if var_shape is not None else tuple(data.shape)
            if len(data.shape) > 0 and len(shape) > 0 and data.shape[-1] == shape[-1]:
                if _count_last(key, shape) == shape[-1]:
                    data = _chartostring(data, encoding=encoding)
    if self._iscompound and self.dtype != self._dtype_view and self.chartostring:
        data = data.view(self._dtype_view)
    return data


def _count_last(key, shape):
    """The count netCDF4's _StartCountStride gives the last dimension for a
    normalised key (whole dimension when there is no key)."""
    if key is None:
        return shape[-1]
    if not isinstance(key, tuple):
        key = (key,)
    k = key[-1] if len(key) == len(shape) else slice(None)
    if isinstance(k, slice):
        return len(range(*k.indices(shape[-1])))
    return 1


class _Masking:
    """Variable._toma, netCDF4-python 1.7.4, in two: here what it works out from
    the attributes, in `run` what it does with values. A file read does not
    change its attributes, so the first is done once a variable: it was a third
    of a small read. One addition: a type netCDF has no default fill value for
    (HDF5 arrays; netCDF-C will not open a file that has one) has none to be
    masked at, and no KeyError."""
    __slots__ = ('view', 'missing', 'fill', 'fill_nan', 'other', 'low', 'high', 'last', 'byte')

    def __init__(self, var, dt, warn):
        dtype, attrs = var.dtype, var.attrs
        default = DEFAULT_FILLVALS.get(dtype.str[1:])
        is_unsigned = var.get('_Unsigned', False) in ["True", "true"]
        view = None
        if var.scale and is_unsigned and dt.kind == 'i':
            view = '%su%s' % (dt.byteorder, dt.itemsize)
        self.view = view
        self.missing = None
        if _check_safecast(var, 'missing_value', warn):
            mval = numpy.array(attrs['missing_value'], dtype)
            if view:
                mval = mval.view(view)
            if mval.shape == ():
                mval = [mval]
            self.missing = []
            for m in mval:
                try:
                    mvalisnan = numpy.isnan(m)
                except TypeError:
                    mvalisnan = False
                self.missing.append((m, mvalisnan))
        safe_fillval = _check_safecast(var, '_FillValue', warn)
        self.fill = self.other = None
        self.fill_nan = False
        if safe_fillval:
            self.fill = numpy.array(attrs['_FillValue'], dtype)
            if view:
                self.fill = self.fill.view(view)
            try:
                self.fill_nan = numpy.isnan(self.fill)
            except Exception:
                self.fill_nan = False
        else:
            no_fill = 1 if var.no_fill else 0
            if (default is not None and not var._isvlen
                    and (no_fill != 1 or dtype.str[1:] not in ['u1', 'i1'])):
                self.other = numpy.array(default, dtype)
        validmin, validmax = _valid_bounds(var, view, warn)
        self.low, self.high = (validmin, validmax) if dtype.kind != 'S' else (None, None)
        if safe_fillval:
            self.last = numpy.array(attrs['_FillValue'], dtype)
        elif default is None or dtype.str[1:] in ['u1', 'i1']:
            self.last = default
        else:
            self.last = numpy.array(default, dtype)
        self.byte = None                            # a char variable's fill, as a number (see run)
        fill = self.fill if self.fill is not None else self.other
        if dtype == numpy.dtype('S1') and dt == dtype and fill is not None and self.missing is None:
            self.byte = numpy.frombuffer(fill.tobytes(), numpy.uint8)[0]

    def run(self, data, always_mask):
        if self.view:
            data = data.view(self.view)
        total = fill_value = None
        if self.byte is not None and data.__class__ is numpy.ndarray:
            # (a byte each: numpy compares them as strings, 40 times slower)
            fill = self.fill if self.fill is not None else self.other
            mask = data.view(numpy.uint8) == self.byte
            masked_values = bool(mask.any())
            if masked_values:
                data = ma.masked_array(data, mask=mask, fill_value=fill)
            else:
                data = ma.masked_array(data)
            if data.shape == () and data.mask.all():
                data = data[()]
            elif not always_mask and not masked_values:
                data = _plain(data)
            return data
        if self.missing is not None:
            mask = None
            for m, isnan in self.missing:
                hit = numpy.isnan(data) if isnan else data == m
                mask = hit if mask is None else mask | hit
            if mask is not None and mask.any():
                fill_value = self.missing[0][0]
                total = mask
        fval = self.fill
        if fval is not None:
            if self.fill_nan:
                mask = numpy.isnan(data)
            else:
                mask = data == fval
                if not mask.any():
                    mask = None
            if mask is not None:
                if fill_value is None:
                    fill_value = fval
                total = mask if total is None else total | mask
        elif self.other is not None:
            mask = data == self.other
            if type(mask) == bool:
                mask = numpy.asarray(mask)
            if mask.any():
                if fill_value is None:
                    fill_value = self.other
                total = mask if total is None else total | mask
        if self.low is not None:
            mask = data < self.low
            total = mask if total is None else total | mask
        if self.high is not None:
            mask = data > self.high
            total = mask if total is None else total | mask
        if fill_value is None:
            fill_value = self.last
        masked_values = total is not None and bool(total.any())
        if masked_values:
            data = ma.masked_array(data, mask=total, fill_value=fill_value)
        else:
            data = ma.masked_array(data)
        if data.shape == () and data.mask.all():
            data = data[()]
        elif not always_mask and not masked_values:
            data = _plain(data)
        return data


def _toma(self, data):
    kept = getattr(self.attrs, 'toma', None)
    key = data.dtype, self.scale, self.no_fill
    m = None if kept is None else kept.get(key)
    if m is None:
        warned = []                                 # (those netCDF4 gives every read: so here)
        m = _Masking(self, data.dtype, warned.append)
        for w in warned:
            warnings.warn(w)
        if kept is not None and not warned:
            kept[key] = m
    return m.run(data, self.always_mask)


def _plain(data):
    """numpy.array(data, copy=False), as netCDF4-python writes it (numpy 2:
    no copy, the same)."""
    if numpy.lib.NumpyVersion(numpy.__version__) >= '2.0.0':
        return numpy.asarray(data)
    return numpy.array(data, copy=False)


def _chartostring(b, encoding='utf-8'):
    # netCDF4.chartostring
    dtype = b.dtype.kind
    if dtype not in ["S", "U"]:
        raise ValueError("type must be string or unicode ('S' or 'U')")
    bs = b.tobytes()
    slen = int(b.shape[-1])
    if encoding in ['none', 'None', 'bytes']:
        a = numpy.array([bs[n1:n1 + slen] for n1 in range(0, len(bs), slen)],
                        'S' + repr(slen))
    else:
        a = numpy.array([bs[n1:n1 + slen].decode(encoding) for n1 in range(0, len(bs), slen)],
                        'U' + repr(slen))
    return a.reshape(b.shape[:-1])


def _set_viewdtype(dt):
    # netCDF4._set_viewdtype: char arrays in a compound seen as strings
    names = dt.names
    formats = []
    for name in names:
        fmt = dt.fields[name][0]
        if fmt.kind == 'V':
            if fmt.shape == ():
                dtx = _set_viewdtype(dt.fields[name][0])
            else:
                if fmt.subdtype[0].kind == 'V':
                    raise TypeError('nested structured dtype arrays not supported')
                elif fmt.subdtype[0].kind == 'S' and len(dt.fields[name][0].shape) == 1:
                    dtx = numpy.dtype('S%s' % dt.fields[name][0].shape[0])
                else:
                    dtx = dt.fields[name][0]
        else:
            dtx = dt.fields[name][0]
        formats.append(dtx)
    return numpy.dtype({'names': names, 'formats': formats}, align=True)


# ---------------------------------------------------------------------------
# the fast path: a plan made from the attributes, one kernel pass

MODE_NONE, MODE_ASTYPE, MODE_MUL, MODE_ADD, MODE_MULADD = 0, 1, 2, 3, 4
# opts[] slots: mask at all, NaN is a missing value, NaN is the fill value,
# the scaling, its two float types (float32 or not)
O_MASK, O_MVNAN, O_FVNAN, O_MODE, O_L1F4, O_L2F4, O_NOPTS = range(7)
# what an equality test stands for (the kernel's eqcat, its stats columns)
CAT_MISSING, CAT_FILL, CAT_DEFAULT = 1, 2, 4


class Plan:
    """What the kernel needs (thresholds in the storage dtype, flags, the
    scale constants) and what the wrapping needs (how the fill value is
    chosen, the dtypes, the masked array's stand-in). Shared between
    variables and threads: nothing changes one once made."""


_DECIDES_PLAN = ('_Unsigned', 'missing_value', '_FillValue', 'valid_range', 'valid_min',
                 'valid_max', 'scale_factor', 'add_offset')
_PLANS = {}                     # what decides a plan -> (Plan or None, its warnings)
PLANS_KEPT = 1024

def make_plan(raw_dtype, attrs, dtype, kind, mask, scale, no_fill, quiet=False):
    """A Plan for the fast path, or None where the reference path must run
    (non-numeric storage, integer or complex scaling, odd attributes).
    quiet: warnings about odd attributes left to the plan of the read."""
    dtype = numpy.dtype(dtype)
    if (raw_dtype != dtype or dtype.kind not in 'iuf' or not dtype.isnative
            or dtype.str[1:] not in DEFAULT_FILLVALS):  # (float16: no kernels for it either)
        return None
    # A plan follows from these and nothing else, so variables that agree on
    # them share one, across files too (186 of WRF's 200 variables); it holds
    # only these, not `attrs`, which would keep a closed file's mapping alive
    own = {}
    key = [dtype, kind, bool(mask), bool(scale), bool(no_fill)]
    among = getattr(attrs, 'among', None)
    for name in _DECIDES_PLAN if among is None else among(_DECIDES_PLAN):
        v = attrs.get(name)
        if v is None:
            continue
        if isinstance(v, str):
            key.append((name, v))
        elif isinstance(v, (numpy.generic, numpy.ndarray)):
            key.append((name, v.dtype, v.shape, v.tobytes()))
            if v.dtype.hasobject:
                key = None                      # (those bytes are pointers)
                break
        else:
            return None                         # (not as getncattr gives them)
        own[name] = v
    if key is None:
        got = None
    else:
        key = tuple(key)
        got = _PLANS.get(key)
    if got is None:
        said = []
        got = (_plan(_Var(attrs if key is None else own, dtype, kind, mask, scale, True, False,
                          no_fill, None), dtype, said.append), said)
        if got[0] is not None:
            got[0].few = _few_values(got[0])
            got[0].words = None if got[0].few is None else value_words(got[0].few)
        if key is not None:
            if len(_PLANS) >= PLANS_KEPT:
                _PLANS.clear()
            _PLANS[key] = got
    p, said = got
    if p is not None and not quiet:             # (else the reference path says them)
        for msg in said:
            warnings.warn(msg)
    return p


class Probe:
    """A test a read makes while its values are in cache, instead of a pass
    over them afterwards: `values` (at most two, in the stored type) are
    all a mask-only plan looks for. The read sets `proven` once it has
    shown that none of them is among what it read; anything it could not
    test leaves that False, and the usual pass runs."""
    __slots__ = ('values', 'proven')

    def __init__(self, values):
        self.values = values
        self.proven = False


def masked_values(p):
    """The values Plan p masks, when masking exactly those is all it does
    and a value equals one only if their bits do; else None. (Not 0.0:
    -0.0 equals it. Not a range, NaN, or a scaled or unsigned view.)"""
    if not p.mask or p.out_dtype is not None or p.view is not None or p.neq == 0:
        return None
    o = p.opts
    if o[O_MVNAN] or o[O_FVNAN] or p.fval_always:
        return None
    xdt = p.xdt
    lo, hi = p.lohi[0], p.lohi[1]
    if xdt.kind == 'f':
        if lo != -numpy.inf or hi != numpy.inf:
            return None
    else:
        t = numpy.iinfo(xdt)
        if lo != t.min or hi != t.max:
            return None
    eq = p.eq[:p.neq]
    if xdt.kind == 'f' and (eq == 0).any():
        return None
    return eq


def probe(attrs, dtype, mask, scale, no_fill):
    """A Probe for a whole-number or float variable with these attributes,
    or None when no test made while reading can stand in for the pass."""
    dtype = numpy.dtype(dtype)
    if dtype.kind not in 'iuf' or dtype.itemsize not in (2, 4, 8):
        return None                             # (x87 floats: padding among their bytes)
    p = make_plan(dtype, attrs, dtype, 'primitive', mask, scale, no_fill, quiet=True)
    return None if p is None or p.few is None else Probe(p.few)


def _few_values(p):
    """masked_values(p) without repeats if they are at most two, each of 2,
    4 or 8 bytes; else None."""
    eq = masked_values(p)
    if eq is None or eq.dtype.itemsize not in (2, 4, 8):
        return None
    t = eq.tolist()
    u = eq[[i for i, v in enumerate(t) if v not in t[:i]]]
    return u if len(u) <= 2 else None


def _nothing_masked(p, data, threads):
    """Whether data holds none of what Plan p masks, when it can tell: a
    test far lighter than flagging every value (no mask to write, nothing
    counted, stopped by the first hit), which is all most arrays need."""
    if p.few is None:
        return False
    c = _Check(p, data)
    if c.x.size <= SOLO_BYTES:
        c.run()
    else:
        _pool.share(c.run, _pool.memory_threads(c.x.size, min(c.x.size // c.step + 1,
                                                              threads or _pool.threads())))
    return c.hit[0] == 0


def _plan(self, dtype, warn):
    p = Plan()
    p.self = self
    valid_scaleoffset = _valid_scaleoffset(self, warn)
    is_unsigned = self.get('_Unsigned', False) in ["True", "true"]
    is_unsigned_int = is_unsigned and dtype.kind == 'i'
    p.view = None
    if self.scale and is_unsigned_int:
        p.view = numpy.dtype('%su%s' % (dtype.byteorder, dtype.itemsize))
    xdt = p.view or dtype                       # the dtype values are tested in
    opts = numpy.zeros(O_NOPTS, numpy.int64)
    p.mask = bool(self.mask)
    opts[O_MASK] = p.mask
    eqs = []                                    # (value, CAT_*): the equality tests
    p.mval_fill = None                          # fill_value if a missing value is hit
    p.fval_fill = None                          # ... if the _FillValue is hit
    p.fval_always = False                       # (a NaN _FillValue: set even if not hit)
    p.dval_fill = None                          # ... if the default fill value is hit
    p.fval_last = None
    # valid range bounds no value is outside of, unless given
    if xdt.kind == 'f':
        vmin, vmax = -numpy.inf, numpy.inf
    else:
        vmin, vmax = numpy.iinfo(xdt).min, numpy.iinfo(xdt).max
    if p.mask:
        if _check_safecast(self, 'missing_value', warn):
            mval = numpy.array(self.attrs['missing_value'], self.dtype)
            if p.view is not None:
                mval = mval.view(p.view)
            if mval.shape == ():
                mval = [mval]
            for m in mval:
                try:
                    isnan = bool(numpy.isnan(m))
                except TypeError:
                    isnan = False
                if isnan:
                    opts[O_MVNAN] = 1
                else:
                    eqs.append((m, CAT_MISSING))
            p.mval_fill = mval[0]
        safe_fillval = _check_safecast(self, '_FillValue', warn)
        if safe_fillval:
            fval = numpy.array(self.attrs['_FillValue'], self.dtype)
            if p.view is not None:
                fval = fval.view(p.view)
            try:
                fvalisnan = bool(numpy.isnan(fval))
            except Exception:
                fvalisnan = False
            if fvalisnan:
                opts[O_FVNAN] = 1
                p.fval_always = True
            else:
                eqs.append((fval, CAT_FILL))
            p.fval_fill = fval
        else:
            nf = 1 if self.no_fill else 0
            if nf != 1 or self.dtype.str[1:] not in ['u1', 'i1']:
                fillval = numpy.array(DEFAULT_FILLVALS[self.dtype.str[1:]], self.dtype)
                # tested against the values as they are (viewed unsigned or not):
                # numpy compares a signed default with unsigned data in a wider
                # type, where it never matches (the defaults are negative)
                if p.view is None:
                    eqs.append((fillval, CAT_DEFAULT))
                p.dval_fill = fillval
        validmin, validmax = _valid_bounds(self, p.view, warn)
        if validmin is not None:
            vmin = validmin
        if validmax is not None:
            vmax = validmax
        byte_type = self.dtype.str[1:] in ['u1', 'i1']
        if safe_fillval:
            p.fval_last = numpy.array(self.attrs['_FillValue'], self.dtype)
        else:
            p.fval_last = numpy.array(DEFAULT_FILLVALS[self.dtype.str[1:]], self.dtype)
            if byte_type:
                p.fval_last = None
    # the tests, in the dtype the values are compared in (at least four
    # equality slots, the last repeated: the kernel tests four at once)
    p.neq = len(eqs)
    while eqs and len(eqs) < 4:
        eqs.append(eqs[-1])
    p.eq = numpy.array([numpy.asarray(v).astype(xdt) for v, c in eqs] or [0] * 4, xdt)
    p.eqcat = numpy.array([c for v, c in eqs] or [0] * 4, numpy.int64)
    p.lohi = numpy.array([numpy.asarray(vmin).astype(xdt), numpy.asarray(vmax).astype(xdt)], xdt)
    # the scaling
    p.mode = MODE_NONE
    sf = off = None
    if self.scale and self._isprimitive and valid_scaleoffset:
        if self.has('scale_factor') and self.has('add_offset'):
            sf, off = self.attrs['scale_factor'], self.attrs['add_offset']
            p.mode = MODE_MULADD if (off != 0.0 or sf != 1.0) else MODE_ASTYPE
        elif self.has('scale_factor') and self.attrs['scale_factor'] != 1.0:
            sf, p.mode = self.attrs['scale_factor'], MODE_MUL
        elif self.has('add_offset') and self.attrs['add_offset'] != 0.0:
            off, p.mode = self.attrs['add_offset'], MODE_ADD
    opts[O_MODE] = p.mode
    sc = numpy.zeros(3, numpy.float64)
    p.out_dtype = None
    if p.mode != MODE_NONE:
        # the dtypes numpy works in, from the same operations on one value
        t = numpy.zeros(1, xdt)
        if p.mode == MODE_ASTYPE:
            l1 = l2 = numpy.dtype(sf.dtype)
        elif p.mode == MODE_MUL:
            l1 = l2 = (t * sf).dtype
        elif p.mode == MODE_ADD:
            l1 = l2 = (t + off).dtype
        else:
            l1 = (t * sf).dtype
            l2 = ((t * sf) + off).dtype
        if l1.kind != 'f' or l2.kind != 'f' or l1.itemsize not in (4, 8) \
                or l2.itemsize not in (4, 8):
            return None                         # integer / complex scaling
        if sf is not None:
            if not isinstance(sf, numpy.generic) or sf.dtype.kind not in 'iuf':
                return None
            sc[0] = numpy.asarray(sf).astype(l1)
        if off is not None:
            if not isinstance(off, numpy.generic) or off.dtype.kind not in 'iuf':
                return None
            sc[1] = numpy.asarray(off).astype(l1)
            sc[2] = numpy.asarray(off).astype(l2)
        opts[O_L1F4] = l1.itemsize == 4
        opts[O_L2F4] = l2.itemsize == 4
        p.out_dtype = l2
    p.sf, p.off = sf, off
    # a NaN scale_factor / add_offset: what NaN times a NaN gives (whose
    # payload) is up to how the compiler that built numpy ordered the
    # operands (MSVC and gcc builds differ), which no kernel can promise to
    # follow: numpy itself does the work then (run_plan_numpy)
    used = ([sf] if p.mode in (MODE_MUL, MODE_MULADD) else []) +         ([off] if p.mode in (MODE_ADD, MODE_MULADD) else [])
    p.nan_const = any(bool(numpy.isnan(numpy.asarray(c, numpy.float64))) for c in used)
    p.opts, p.sc, p.xdt = opts, sc, xdt
    p.sc32 = sc.astype(numpy.float32)           # (exact where used: float32 values)
    return p


def fill_value_of(p, any_mv, any_fv, any_dv):
    """The fill value _toma gives the masked array, from which tests hit."""
    fill_value = None
    if any_mv:
        fill_value = p.mval_fill
    if (any_fv or p.fval_always) and fill_value is None:
        fill_value = p.fval_fill
    if any_dv and fill_value is None:
        fill_value = p.dval_fill
    if fill_value is None and p.fval_last is not None:
        fill_value = p.fval_last
    if fill_value is None:
        fill_value = DEFAULT_FILLVALS[p.self.dtype.str[1:]]
    return fill_value


def finish(p, data, out, msk, stats, always_mask):
    """The result, as netCDF4-python builds it, from the kernel's output:
    `data` the (viewed) raw values, `out` the scaled ones (or None), `msk`
    the mask (valid only if something is masked), stats the counts of
    [missing, fill, default fill, masked] hits."""
    any_mv, any_fv, any_dv, any_m = (bool(s) for s in stats)
    if not p.mask:
        return out if out is not None else data
    if any_m:
        fill_value = fill_value_of(p, any_mv, any_fv, any_dv)
        if out is None:
            return ma.masked_array(data, mask=msk, fill_value=fill_value)
        t = ma.masked_array(numpy.zeros(1, data.dtype), mask=numpy.ones(1, bool),
                            fill_value=fill_value)
    else:
        if not always_mask:
            return out if out is not None else data
        if out is None:
            return ma.masked_array(data)
        t = ma.masked_array(numpy.zeros(1, data.dtype))
    t = _scale_ops(p.self, t)
    res = out.view(type(t))
    res.__dict__.update(t.__dict__)
    res._mask = msk if any_m else ma.nomask
    return res


def run_plan(p, raw, always_mask, threads=None):
    """The fast path: the kernel over raw (C-contiguous) on the pool."""
    if p.nan_const:
        return run_plan_numpy(p, raw, always_mask, threads)
    if _nothing_masked(p, raw, threads):
        return finish(p, raw, None, None, (0, 0, 0, 0), always_mask)
    s = _Pass(p, raw)
    nthr = _pool.memory_threads(raw.nbytes, min(s.nblk, threads or _pool.threads()), _pool.PASS_SPEED)
    _pool.share(s.run, nthr)
    if s.mixed():
        _pool.share(s.zero, nthr)
    return s.result(always_mask)


class _Pass:
    """One array's masking and scaling, for any number of threads at once:
    they take its blocks off its counter until none is left."""

    def __init__(self, p, raw):
        self.p = p
        self.data = data = raw.view(p.view) if p.view is not None else raw
        self.x = x = data.reshape(-1)
        self.out = _arena.empty(data.shape, p.out_dtype) if p.out_dtype is not None else None
        self.o = self.out.reshape(-1) if self.out is not None else numpy.empty(1, numpy.float64)
        self.msk = _arena.empty(data.shape, numpy.bool_) if p.mask else numpy.empty(1, numpy.bool_)
        self.m = self.msk.reshape(-1)
        self.block = BLOCK
        self.nblk = (x.size + BLOCK - 1) // BLOCK
        self.stats = numpy.zeros((self.nblk, 4), numpy.int64)
        self.ctr = numpy.zeros(2, numpy.int64)
        self.call = None

    def run(self):
        p = self.p
        if p.mask or self.out is not None:
            call = self.call
            if call is None:
                call = self.call = bind(maskscale_blocks, (self.x, self.o, self.m, p.eq, p.eqcat, p.neq,
                                                           p.lohi, p.opts, p.sc, p.sc32, self.stats,
                                                           self.ctr, self.block), ())
            call()

    def mixed(self):
        """Whether some blocks are masked and some are not (run first)."""
        s = self.stats
        return self.p.mask and s[:, 3].any() and (s[:, 3] == 0).any()

    def zero(self):
        zero_clean_blocks(self.m, self.stats, self.ctr, self.block)

    def result(self, always_mask):
        return finish(self.p, self.data, self.out, self.msk, self.stats.sum(axis=0), always_mask)


class _Check:
    """One array tested for what its plan masks (see _nothing_masked), for
    any number of threads at once."""

    def __init__(self, p, data):
        self.x = data.reshape(-1).view(numpy.uint8)
        self.step = BLOCK * data.dtype.itemsize
        self.words = p.words
        self.ctr = numpy.zeros(1, numpy.int64)
        self.hit = numpy.zeros(1, numpy.int64)
        self.call = None

    def run(self):
        call = self.call
        if call is None:
            w0, w1, low = self.words
            call = self.call = bind(_holds_blocks, (self.x, self.step, w0, w1, low, self.ctr, self.hit), ())
        call()


def run_plans(items, threads=None):
    """run_plan for every (plan, raw, always_mask, clean), in one pass of
    the pool: a thread goes on to another array when it finds none of the
    last one's blocks left, so small arrays cost no waking of threads and
    the biggest ends up with all of them, whatever the sizes (_pool.walk).
    The results in the order given."""
    res = [None] * len(items)
    biggest = sorted(range(len(items)), key=lambda i: -items[i][1].nbytes)
    checks = {}
    for i in biggest:
        p, raw, always_mask, clean = items[i]
        if clean and p.few is not None:
            res[i] = finish(p, raw, None, None, (0, 0, 0, 0), always_mask)
        elif p.few is not None:
            checks[i] = _Check(p, raw)
    total = sum(it[1].nbytes for it in items)
    _pool.walk([c.run for c in checks.values()], _pool.memory_threads(total, threads or _pool.threads()))
    nthr = _pool.memory_threads(total, threads or _pool.threads(), _pool.PASS_SPEED)
    passes = {}
    for i in biggest:
        p, raw, always_mask, clean = items[i]
        if res[i] is None:
            if i in checks and not checks[i].hit[0]:
                res[i] = finish(p, raw, None, None, (0, 0, 0, 0), always_mask)
            else:
                passes[i] = _Pass(p, raw)
    _pool.walk([s.run for s in passes.values()], nthr)
    _pool.walk([s.zero for s in passes.values() if s.mixed()], nthr)
    for i, s in passes.items():
        res[i] = s.result(items[i][2])
    return res


def xdt_kind_f(dt):
    return numpy.dtype(dt).kind == 'f'


NP_BLOCK = _hw.block(1 << 18)   # elements per block of the numpy path (a multiple of 64:
                                # numpy's SIMD loops then end where they would over the
                                # whole array -- a NaN times a NaN gives the first or the
                                # second operand's payload depending on which loop)


def run_plan_numpy(p, raw, always_mask, threads=None):
    """run_plan's work in numpy ufuncs, block by block on the pool (numpy
    lets go of the GIL inside them): for a process whose kernels have not
    loaded (numba's import is half a second). Each value goes through the
    same operations in the same float types as in the kernel."""
    data = raw.view(p.view) if p.view is not None else raw
    x = data.reshape(-1)
    n = x.size
    out = _arena.empty(data.shape, p.out_dtype) if p.out_dtype is not None else None
    msk = _arena.empty(data.shape, numpy.bool_) if p.mask else None
    nblk = (n + NP_BLOCK - 1) // NP_BLOCK
    stats = numpy.zeros((nblk, 4), numpy.int64)
    mode = int(p.opts[O_MODE])
    l1f4, l2f4 = bool(p.opts[O_L1F4]), bool(p.opts[O_L2F4])
    nan_mv, nan_fv = bool(p.opts[O_MVNAN]), bool(p.opts[O_FVNAN])
    lo, hi = p.lohi[0], p.lohi[1]
    eqs = [(p.eq[e], int(p.eqcat[e])) for e in range(p.neq)]
    f4, f8 = numpy.float32, numpy.float64
    s32, o32, o232 = p.sc32[0], p.sc32[1], p.sc32[2]
    s64, o64, o264 = p.sc[0], p.sc[1], p.sc[2]
    U = 'unsafe'
    nxt = itertools.count()

    def scale(xb, ob):
        if mode == MODE_ASTYPE:
            numpy.copyto(ob, xb, casting=U)
        elif mode == MODE_MUL:
            numpy.multiply(xb, s32 if l1f4 else s64, out=ob, dtype=f4 if l1f4 else f8,
                           casting=U)
        elif mode == MODE_ADD:
            numpy.add(xb, o32 if l1f4 else o64, out=ob, dtype=f4 if l1f4 else f8, casting=U)
        elif l2f4:
            numpy.multiply(xb, s32, out=ob, dtype=f4, casting=U)
            numpy.add(ob, o232, out=ob)
        elif l1f4:
            t = numpy.multiply(xb, s32, dtype=f4, casting=U)
            numpy.add(t, o264, out=ob, dtype=f8, casting=U)
        else:
            numpy.multiply(xb, s64, out=ob, dtype=f8, casting=U)
            numpy.add(ob, o264, out=ob)

    # (bounds no value can be outside of are not tested)
    if xdt_kind_f(p.xdt):
        test_lo, test_hi = not numpy.isneginf(lo), not numpy.isposinf(hi)
    else:
        ii = numpy.iinfo(p.xdt)
        test_lo, test_hi = lo != ii.min, hi != ii.max
    flat_out = out.reshape(-1) if out is not None else None
    flat_msk = msk.reshape(-1) if msk is not None else None
    tls = threading.local()

    def work():
        if p.mask:
            s = tls.s = numpy.empty(NP_BLOCK, numpy.bool_)
            e = numpy.empty(NP_BLOCK, numpy.bool_)
        while True:
            b = next(nxt)
            if b >= nblk:
                return
            i0 = b * NP_BLOCK
            i1 = min(i0 + NP_BLOCK, n)
            xb = x[i0:i1]
            ob = flat_out[i0:i1] if out is not None else None
            if ob is not None:
                scale(xb, ob)
            if not p.mask:
                continue
            sb = s[:i1 - i0]
            eb = e[:i1 - i0]
            sb[...] = False
            if test_lo:
                numpy.less(xb, lo, out=sb)
            if test_hi:
                numpy.greater(xb, hi, out=eb)
                sb |= eb
            for v, c in eqs:
                numpy.equal(xb, v, out=eb)
                sb |= eb
            if nan_mv or nan_fv:
                numpy.isnan(xb, out=eb)
                sb |= eb
            c = int(numpy.count_nonzero(sb))
            stats[b, 3] = c
            if not c:
                continue
            flat_msk[i0:i1] = sb
            if ob is not None:
                if l1f4 and not l2f4:
                    numpy.copyto(ob, xb.astype(f4), where=sb, casting=U)
                else:
                    numpy.copyto(ob, xb, where=sb, casting=U)
            for k, cat in enumerate((CAT_MISSING, CAT_FILL, CAT_DEFAULT)):
                hit = any(bool(numpy.equal(xb, v, out=eb).any()) for v, cc in eqs if cc == cat)
                if (k == 0 and nan_mv) or (k == 1 and nan_fv):
                    hit = hit or bool(numpy.isnan(xb, out=eb).any())
                stats[b, k] = hit
    _pool.share(work, min(nblk, threads or _pool.threads()))
    tot = stats.sum(axis=0)
    if p.mask and tot[3]:
        m = msk.reshape(-1)
        for b in numpy.flatnonzero(stats[:, 3] == 0):
            m[b * NP_BLOCK:(b + 1) * NP_BLOCK] = False
    return finish(p, data, out, msk, tot, always_mask)


# ---------------------------------------------------------------------------
# kernels

@njit(cache=True, nogil=True)
def _flag_hits(x, eq, neq, lo, hi, nan_on, s):
    """s[k] = 1 where x[k] is masked (equal to one of the neq values
    eq -- four at once when there are at most four, the rest repeats --,
    outside [lo, hi] -- the type's own range when no valid range is given
    --, or NaN when nan_on), else 0; returns how many. Branch-free loops
    over a block in cache, indexed from 0 (no negative index to wrap):
    they vectorize."""
    n = x.size
    if neq > 4:
        for k in range(n):
            v = x[k]
            h = (v < lo) | (v > hi) | (nan_on & (v != v))
            for e in range(neq):
                h |= v == eq[e]
            s[k] = h
    elif neq > 0:
        e0, e1, e2, e3 = eq[0], eq[1], eq[2], eq[3]
        if nan_on:
            for k in range(n):
                v = x[k]
                s[k] = (v < lo) | (v > hi) | (v == e0) | (v == e1) | (v == e2) | (v == e3) | (v != v)
        else:
            for k in range(n):
                v = x[k]
                s[k] = (v < lo) | (v > hi) | (v == e0) | (v == e1) | (v == e2) | (v == e3)
    elif nan_on:
        for k in range(n):
            v = x[k]
            s[k] = (v < lo) | (v > hi) | (v != v)
    else:
        for k in range(n):
            v = x[k]
            s[k] = (v < lo) | (v > hi)
    c = 0
    for k in range(n):
        c += s[k]
    return c


@njit(cache=True, nogil=True)
def _scale_block(x, out, mode, l1f4, l2f4, sc, sc32):
    """out = numpy's data * scale_factor + add_offset (or the part
    of it the attributes ask for), each operation in the float type numpy
    does it in (l1f4: the first in float32, l2f4: the second), so each
    rounds as numpy's does. sc / sc32: scale_factor in the first type,
    add_offset in the first and in the second."""
    if mode == 1:                               # data.astype(scale_factor.dtype)
        if l1f4:
            for i in range(x.size):
                out[i] = numpy.float32(x[i])
        else:
            for i in range(x.size):
                out[i] = numpy.float64(x[i])
    elif mode == 2:                             # data * scale_factor
        if l1f4:
            s32 = sc32[0]
            for i in range(x.size):
                out[i] = numpy.float32(x[i]) * s32
        else:
            s = sc[0]
            for i in range(x.size):
                out[i] = numpy.float64(x[i]) * s
    elif mode == 3:                             # data + add_offset
        if l1f4:
            o32 = sc32[1]
            for i in range(x.size):
                out[i] = numpy.float32(x[i]) + o32
        else:
            o = sc[1]
            for i in range(x.size):
                out[i] = numpy.float64(x[i]) + o
    elif mode == 4:                             # data * scale_factor + add_offset
        if l2f4:
            s32 = sc32[0]
            o32 = sc32[2]
            for i in range(x.size):
                out[i] = numpy.float32(x[i]) * s32 + o32
        elif l1f4:
            s32 = sc32[0]
            o = sc[2]
            for i in range(x.size):
                out[i] = numpy.float64(numpy.float32(x[i]) * s32) + o
        else:
            s = sc[0]
            o = sc[2]
            for i in range(x.size):
                out[i] = numpy.float64(x[i]) * s + o


@njit(cache=True, nogil=True)
def _fix_block(x, out, msk, s, eq, eqcat, neq, nan_mv, nan_fv, doscale, l1f4, l2f4, cnt):
    """For a block with masked values (flagged in s by _flag_hits): the
    mask written, and at each masked value the unscaled value cast to the
    result type (as numpy's masked arithmetic leaves it). cnt[0:3]: values
    seen equal to a missing value, the fill value, the default fill -- only
    whether any is wanted, so each is counted only until one is seen. All
    loops branch-free over the block in cache (masked values can be a
    third of an image)."""
    n = x.size
    for k in range(n):
        msk[k] = s[k] != 0
    if doscale:
        if l2f4:
            for k in range(n):
                out[k] = numpy.float32(x[k]) if s[k] else out[k]
        elif l1f4:
            for k in range(n):
                out[k] = numpy.float64(numpy.float32(x[k])) if s[k] else out[k]
        else:
            for k in range(n):
                out[k] = numpy.float64(x[k]) if s[k] else out[k]
    for c in range(3):
        if cnt[c] != 0:
            continue
        cat = 1 << c
        h = 0
        for e in range(neq):
            if eqcat[e] == cat and (e == 0 or eq[e] != eq[e - 1] or eqcat[e - 1] != cat):
                ev = eq[e]
                for k in range(n):
                    h += x[k] == ev
        if (c == 0 and nan_mv) or (c == 1 and nan_fv):
            for k in range(n):
                h += x[k] != x[k]
        cnt[c] = h


@njit(cache=True, nogil=True)
def maskscale_blocks(x, out, msk, eq, eqcat, neq, lohi, opts, sc, sc32, stats, ctr, block):
    """Mask and scale x into out / msk, `block` values at a time taken off
    ctr[0] (one caller per thread). Per block, still in cache: the scaled
    values, the masked ones flagged (in this thread's scratch) and counted,
    and only if there are some the mask and the values under it.
    stats[b, 3] = values masked in block b; stats[b, 0:3] > 0 if a
    missing_value / _FillValue / default fill was hit (in it or an
    earlier block of the same thread). Blocks with nothing masked leave msk untouched (zero_clean_blocks
    clears them if another block has a mask; else the mask is dropped,
    never paged in)."""
    n = x.size
    nblk = stats.shape[0]
    domask = opts[0] != 0
    nan_mv = opts[1] != 0
    nan_fv = opts[2] != 0
    nan_on = nan_mv or nan_fv
    mode = opts[3]
    l1f4 = opts[4] != 0
    l2f4 = opts[5] != 0
    doscale = mode != 0
    lo = lohi[0]
    hi = lohi[1]
    cnt = numpy.zeros(3, numpy.int64)
    s = numpy.empty(block + 8 if domask else 8, numpy.uint8)
    while True:
        b = take(ctr, 0)
        if b >= nblk:
            break
        i0 = b * block
        i1 = min(i0 + block, n)
        xb = x[i0:i1]
        ob = out[i0:i1] if doscale else out
        if doscale:
            _scale_block(xb, ob, mode, l1f4, l2f4, sc, sc32)
        c = 0
        if domask:
            c = _flag_hits(xb, eq, neq, lo, hi, nan_on, s)
        if c:
            _fix_block(xb, ob, msk[i0:i1], s, eq, eqcat, neq, nan_mv, nan_fv, doscale,
                       l1f4, l2f4, cnt)
            stats[b, 0] = cnt[0]                # (this thread's so far: only > 0 counts)
            stats[b, 1] = cnt[1]
            stats[b, 2] = cnt[2]
        stats[b, 3] = c


@njit(cache=True, nogil=True)
def _holds_blocks(x, step, w0, w1, low, ctr, hit):
    """hit[0] = 1 if a block of step bytes (taken off ctr[0]) holds one of
    the values (see _chunks.holds_value); no more are taken after that."""
    n = x.size
    while hit[0] == 0:
        lo = take(ctr, 0) * step
        if lo >= n:
            return
        if holds_value(x, lo, min(lo + step, n), w0, w1, low):
            hit[0] = 1


@njit(cache=True, nogil=True)
def zero_clean_blocks(msk, stats, ctr, block):
    """msk False over the blocks maskscale_blocks found nothing in (blocks
    taken off ctr[1])."""
    n = msk.size
    nblk = stats.shape[0]
    while True:
        b = take(ctr, 1)
        if b >= nblk:
            break
        if stats[b, 3] == 0:
            for i in range(b * block, min(b * block + block, n)):
                msk[i] = False
