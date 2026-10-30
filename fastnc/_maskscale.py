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
kernel run on fastnc's threads reads each raw value once, tests it
against missing values, fill value and valid range, and writes the
scaled value and (only where something is masked) the mask. What the
kernel cannot know cheaply -- the result dtype, the fill value the masked
array ends up with, the masked array's bookkeeping -- comes from running
the reference path on a one-element stand-in with the same attributes, so
numpy decides those exactly as it would for the whole array. The values
are numpy's to the bit: each float operation is done in float64 and
rounded to float32 where numpy works in float32, which for one multiply
or one add gives the float32 result exactly (53 >= 2 * 24 + 2 bits, so
double rounding is innocuous), and integers go to float32 by one direct
conversion.
"""
import warnings

import numpy

from ._jit import njit
from ._intrin import take

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

FAST_MIN_SIZE = 1 << 14         # below this the reference path costs no more
BLOCK = 1 << 16                 # elements a thread takes at a time


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
    variable's last dimension was read whole, as chartostring requires.
    Missing, they mean the whole variable was read.
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
            return run_plan(plan, raw, always_mask, threads)
    return _reference(raw, attrs, dtype, kind, mask, scale, always_mask, chartostring,
                      no_fill, var_shape, key, var_dtype)


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
# opts[] slots
O_MASK, O_MVNAN, O_FV, O_FVNAN, O_DV, O_VMIN, O_VMAX, O_MODE, O_L1F4, O_L2F4, O_NMV, O_NOPTS = \
    range(12)


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
    mvals = []
    p.mval_fill = None                          # fill_value if a missing value is hit
    p.fval_fill = None                          # ... if the _FillValue is hit
    p.fval_always = False                       # (a NaN _FillValue: set even if not hit)
    p.dval_fill = None                          # ... if the default fill value is hit
    p.fval_last = None
    fv = dv = vmin = vmax = 0
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
                    mvals.append(m)
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
            opts[O_FV] = 1
            if fvalisnan:
                opts[O_FVNAN] = 1
                p.fval_always = True
            fv = fval
            p.fval_fill = fval
        else:
            nf = 1 if self.no_fill else 0
            if nf != 1 or self.dtype.str[1:] not in ['u1', 'i1']:
                fillval = numpy.array(DEFAULT_FILLVALS[self.dtype.str[1:]], self.dtype)
                # tested against the values as they are (viewed unsigned or not):
                # numpy compares a signed default with unsigned data in a wider
                # type, where it never matches (the defaults are negative)
                if p.view is None:
                    opts[O_DV] = 1
                    dv = fillval
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
            opts[O_VMIN] = 1
            vmin = validmin
        if validmax is not None:
            opts[O_VMAX] = 1
            vmax = validmax
        byte_type = self.dtype.str[1:] in ['u1', 'i1']
        if safe_fillval:
            p.fval_last = numpy.array(self.attrs['_FillValue'], self.dtype)
        else:
            p.fval_last = numpy.array(DEFAULT_FILLVALS[self.dtype.str[1:]], self.dtype)
            if byte_type:
                p.fval_last = None
    # the thresholds, in the dtype the values are compared in
    p.mv = numpy.array(mvals, xdt).reshape(-1) if mvals else numpy.zeros(1, xdt)
    opts[O_NMV] = len(mvals)
    p.th = numpy.array([numpy.asarray(t).astype(xdt) for t in (fv, dv, vmin, vmax)], xdt)
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
    p.opts, p.sc, p.xdt = opts, sc, xdt
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
    data = raw.view(p.view) if p.view is not None else raw
    x = data.reshape(-1)
    n = x.size
    out = numpy.empty(data.shape, p.out_dtype) if p.out_dtype is not None else None
    o = out.reshape(-1) if out is not None else numpy.empty(1, numpy.float64)
    msk = numpy.empty(data.shape, numpy.bool_) if p.mask else numpy.empty(1, numpy.bool_)
    m = msk.reshape(-1)
    nblk = (n + BLOCK - 1) // BLOCK
    stats = numpy.zeros((nblk, 4), numpy.int64)
    ctr = numpy.zeros(2, numpy.int64)
    nthr = min(nblk, threads or _pool.threads())
    if p.mask or out is not None:
        _pool.share(lambda: maskscale_blocks(x, o, m, p.mv, p.th, p.opts, p.sc, stats,
                                             ctr, BLOCK), nthr)
    tot = stats.sum(axis=0)
    if p.mask and tot[3] and (stats[:, 3] == 0).any():
        _pool.share(lambda: zero_clean_blocks(m, stats, ctr, BLOCK), nthr)
    return finish(p, data, out, msk, tot, always_mask)


# ---------------------------------------------------------------------------
# kernels

@njit(cache=True, nogil=True)
def _hits(v, mv, th, opts):
    """Which tests value v hits: bit 0 missing_value, 1 _FillValue, 2 the
    default fill value, 3 the valid range."""
    h = 0
    if opts[1] != 0 and v != v:                 # a NaN missing_value
        h = 1
    for k in range(opts[10]):                  # the other missing values
        if v == mv[k]:
            h = 1
    if opts[2] != 0:
        if opts[3] != 0:
            if v != v:
                h |= 2
        elif v == th[0]:
            h |= 2
    if opts[4] != 0 and v == th[1]:
        h |= 4
    if opts[5] != 0 and v < th[2]:
        h |= 8
    if opts[6] != 0 and v > th[3]:
        h |= 8
    return h


@njit(cache=True, nogil=True)
def _scaled(v, sc, mode, l1f4, l2f4):
    """numpy's data * scale_factor + add_offset (or the part of it the
    attributes ask for) for one value, in float64 holding the result of
    numpy's float32 / float64 operations exactly."""
    if l1f4:
        a = numpy.float64(numpy.float32(v))     # one rounding, as numpy's cast
    else:
        a = numpy.float64(v)
    if mode == 2:                               # data * scale_factor
        a = a * sc[0]
        if l1f4:
            a = numpy.float64(numpy.float32(a))
    elif mode == 3:                             # data + add_offset
        a = a + sc[1]
        if l1f4:
            a = numpy.float64(numpy.float32(a))
    elif mode == 4:                             # data * scale_factor + add_offset
        a = a * sc[0]
        if l1f4:
            a = numpy.float64(numpy.float32(a))
        a = a + sc[2]
        if l2f4:
            a = numpy.float64(numpy.float32(a))
    return a


@njit(cache=True, nogil=True)
def maskscale_blocks(x, out, msk, mv, th, opts, sc, stats, ctr, block):
    """Mask and scale x into out / msk, BLOCK values at a time taken off
    ctr[0] (one caller per thread). stats[b] = hits in block b of
    [missing_value, _FillValue, default fill, any]; the mask is written
    only for blocks with something masked (the rest is left for
    zero_clean_blocks, and never touched when nothing is masked)."""
    n = x.size
    nblk = stats.shape[0]
    domask = opts[0] != 0
    mode = opts[7]
    l1f4 = opts[8] != 0
    l2f4 = opts[9] != 0
    doscale = mode != 0
    while True:
        b = take(ctr, 0)
        if b >= nblk:
            break
        i0 = b * block
        i1 = min(i0 + block, n)
        c_mv = 0
        c_fv = 0
        c_dv = 0
        c_any = 0
        if domask and doscale:
            for i in range(i0, i1):
                v = x[i]
                h = _hits(v, mv, th, opts)
                if h != 0:
                    c_mv += h & 1
                    c_fv += (h >> 1) & 1
                    c_dv += (h >> 2) & 1
                    c_any += 1
                    if l2f4:                    # numpy puts the unscaled value back
                        out[i] = numpy.float32(v) if l1f4 else numpy.float64(v)
                    else:
                        out[i] = numpy.float64(numpy.float32(v)) if l1f4 else numpy.float64(v)
                else:
                    out[i] = _scaled(v, sc, mode, l1f4, l2f4)
        elif domask:
            for i in range(i0, i1):
                h = _hits(x[i], mv, th, opts)
                if h != 0:
                    c_mv += h & 1
                    c_fv += (h >> 1) & 1
                    c_dv += (h >> 2) & 1
                    c_any += 1
        else:
            for i in range(i0, i1):
                out[i] = _scaled(x[i], sc, mode, l1f4, l2f4)
        if c_any:
            for i in range(i0, i1):
                msk[i] = _hits(x[i], mv, th, opts) != 0
        stats[b, 0] = c_mv
        stats[b, 1] = c_fv
        stats[b, 2] = c_dv
        stats[b, 3] = c_any


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
