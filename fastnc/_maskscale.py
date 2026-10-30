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

from . import _arena, _hw

from ._jit import njit
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


# ---------------------------------------------------------------------------
# the interface

def apply(raw, attrs, dtype, mask=True, scale=True, always_mask=True,
          chartostring=True, var_dtype=None, xp=numpy, no_fill=False,
          var_shape=None, key=None, threads=None):
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
    if (kind in ('primitive', 'enum') and type(raw) is numpy.ndarray
            and raw.ndim > 0 and raw.size >= FAST_MIN_SIZE and raw.flags.c_contiguous):
        plan = make_plan(raw.dtype, attrs, dtype, kind, mask, scale, no_fill)
        if plan is not None:
            if _kernels_ready():
                return run_plan(plan, raw, always_mask, threads)
            from . import _chunks
            t = time.perf_counter()
            r = run_plan_numpy(plan, raw, always_mask, threads)
            _chunks.slow(time.perf_counter() - t)
            return r
    return _reference(raw, attrs, dtype, kind, mask, scale, always_mask, chartostring,
                      no_fill, var_shape, key, var_dtype)


def _kernels_ready():
    """Whether to use the kernels (until they are loaded -- see
    _chunks.LOAD_AFTER -- run_plan_numpy does the work)."""
    from . import _chunks
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
    return is_safe


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


def _toma(self, data):
    # Variable._toma, netCDF4-python 1.7.4
    is_unsigned = self.get('_Unsigned', False) in ["True", "true"]
    is_unsigned_int = is_unsigned and data.dtype.kind == 'i'
    if self.scale and is_unsigned_int:
        dtype_unsigned_int = '%su%s' % (data.dtype.byteorder, data.dtype.itemsize)
        data = data.view(dtype_unsigned_int)
    totalmask = numpy.zeros(data.shape, numpy.bool_)
    fill_value = None
    safe_missval = _check_safecast(self, 'missing_value')
    if safe_missval:
        mval = numpy.array(self.attrs['missing_value'], self.dtype)
        if self.scale and is_unsigned_int:
            mval = mval.view(dtype_unsigned_int)
        mvalmask = numpy.zeros(data.shape, numpy.bool_)
        if mval.shape == ():
            mval = [mval]
        for m in mval:
            try:
                mvalisnan = numpy.isnan(m)
            except TypeError:
                mvalisnan = False
            if mvalisnan:
                mvalmask += numpy.isnan(data)
            else:
                mvalmask += data == m
        if mvalmask.any():
            fill_value = mval[0]
            totalmask += mvalmask
    safe_fillval = _check_safecast(self, '_FillValue')
    if safe_fillval:
        fval = numpy.array(self.attrs['_FillValue'], self.dtype)
        if self.scale and is_unsigned_int:
            fval = fval.view(dtype_unsigned_int)
        try:
            fvalisnan = numpy.isnan(fval)
        except Exception:
            fvalisnan = False
        if fvalisnan:
            mask = numpy.isnan(data)
        elif (data == fval).any():
            mask = data == fval
        else:
            mask = None
        if mask is not None:
            if fill_value is None:
                fill_value = fval
            totalmask += mask
    else:
        no_fill = 1 if self.no_fill else 0
        if not self._isvlen and (no_fill != 1 or self.dtype.str[1:] not in ['u1', 'i1']):
            fillval = numpy.array(DEFAULT_FILLVALS[self.dtype.str[1:]], self.dtype)
            has_fillval = data == fillval
            if type(has_fillval) == bool:
                has_fillval = numpy.asarray(has_fillval)
            if has_fillval.any():
                if fill_value is None:
                    fill_value = fillval
                mask = data == fillval
                totalmask += mask
    validmin = None
    validmax = None
    safe_validrange = _check_safecast(self, 'valid_range')
    safe_validmin = _check_safecast(self, 'valid_min')
    safe_validmax = _check_safecast(self, 'valid_max')
    if safe_validrange and self.attrs['valid_range'].size == 2:
        validmin = numpy.array(self.attrs['valid_range'][0], self.dtype)
        validmax = numpy.array(self.attrs['valid_range'][1], self.dtype)
    else:
        if safe_validmin:
            validmin = numpy.array(self.attrs['valid_min'], self.dtype)
        if safe_validmax:
            validmax = numpy.array(self.attrs['valid_max'], self.dtype)
    if validmin is not None and self.scale and is_unsigned_int:
        validmin = validmin.view(dtype_unsigned_int)
    if validmax is not None and self.scale and is_unsigned_int:
        validmax = validmax.view(dtype_unsigned_int)
    byte_type = self.dtype.str[1:] in ['u1', 'i1']
    if safe_fillval:
        fval = numpy.array(self.attrs['_FillValue'], self.dtype)
    else:
        fval = numpy.array(DEFAULT_FILLVALS[self.dtype.str[1:]], self.dtype)
        if byte_type:
            fval = None
    if self.dtype.kind != 'S':
        if validmin is not None:
            totalmask += data < validmin
        if validmax is not None:
            totalmask += data > validmax
    if fill_value is None and fval is not None:
        fill_value = fval
    if fill_value is None:
        fill_value = DEFAULT_FILLVALS[self.dtype.str[1:]]
    masked_values = bool(totalmask.any())
    if masked_values:
        data = ma.masked_array(data, mask=totalmask, fill_value=fill_value)
    else:
        data = ma.masked_array(data)
    if data.shape == () and data.mask.all():
        data = data[()]
    elif not self.always_mask and not masked_values:
        data = _plain(data)
    return data


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
    a.shape = b.shape[:-1]
    return a


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
    chosen, the dtypes, the masked array's stand-in)."""


def make_plan(raw_dtype, attrs, dtype, kind, mask, scale, no_fill):
    """A Plan for the fast path, or None where the reference path must run
    (non-numeric storage, integer or complex scaling, odd attributes)."""
    dtype = numpy.dtype(dtype)
    if raw_dtype != dtype or dtype.kind not in 'iuf' or not dtype.isnative:
        return None
    for name in ('_Unsigned', 'missing_value', '_FillValue', 'valid_range', 'valid_min',
                 'valid_max', 'scale_factor', 'add_offset'):
        v = attrs.get(name)
        if v is not None and not isinstance(v, (str, numpy.generic, numpy.ndarray)):
            return None                         # (not as getncattr gives them)
    self = _Var(attrs, dtype, kind, mask, scale, True, False, no_fill, None)
    said = []
    p = _plan(self, dtype, said.append)
    if p is not None:                           # (else the reference path says them)
        for msg in said:
            warnings.warn(msg)
    return p


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
        validmin = validmax = None
        safe_validrange = _check_safecast(self, 'valid_range', warn)
        safe_validmin = _check_safecast(self, 'valid_min', warn)
        safe_validmax = _check_safecast(self, 'valid_max', warn)
        if safe_validrange and self.attrs['valid_range'].size == 2:
            validmin = numpy.array(self.attrs['valid_range'][0], self.dtype)
            validmax = numpy.array(self.attrs['valid_range'][1], self.dtype)
        else:
            if safe_validmin:
                validmin = numpy.array(self.attrs['valid_min'], self.dtype)
            if safe_validmax:
                validmax = numpy.array(self.attrs['valid_max'], self.dtype)
        if validmin is not None and p.view is not None:
            validmin = validmin.view(p.view)
        if validmax is not None and p.view is not None:
            validmax = validmax.view(p.view)
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
    from . import _pool
    if p.nan_const:
        return run_plan_numpy(p, raw, always_mask, threads)
    data = raw.view(p.view) if p.view is not None else raw
    x = data.reshape(-1)
    n = x.size
    out = _arena.empty(data.shape, p.out_dtype) if p.out_dtype is not None else None
    o = out.reshape(-1) if out is not None else numpy.empty(1, numpy.float64)
    msk = _arena.empty(data.shape, numpy.bool_) if p.mask else numpy.empty(1, numpy.bool_)
    m = msk.reshape(-1)
    nblk = (n + BLOCK - 1) // BLOCK
    stats = numpy.zeros((nblk, 4), numpy.int64)
    ctr = numpy.zeros(2, numpy.int64)
    nthr = min(nblk, threads or _pool.threads())
    if p.mask or out is not None:
        _pool.share(lambda: maskscale_blocks(x, o, m, p.eq, p.eqcat, p.neq, p.lohi, p.opts,
                                             p.sc, p.sc32, stats, ctr, BLOCK), nthr)
    tot = stats.sum(axis=0)
    if p.mask and tot[3] and (stats[:, 3] == 0).any():
        _pool.share(lambda: zero_clean_blocks(m, stats, ctr, BLOCK), nthr)
    return finish(p, data, out, msk, tot, always_mask)


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
    from . import _pool
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
