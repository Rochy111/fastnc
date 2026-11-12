"""fastnc._maskscale against netCDF4-python's answers, recorded by
tools/check/maskscale_cases.py from netCDF files it wrote with every
combination of storage type and mask / scale attribute that matters
(tests/data/maskscale: the inputs, and the expected results under numpy
1 and numpy 2, whose promotion rules differ). netCDF4-python is not
needed here. Each case runs on the reference path and on the fast kernel
(forced on, in tiny blocks); big random arrays check the kernel on many
threads against the reference path, and the GPU against the CPU."""
import gzip
import json
import os
import sys
import warnings

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools', 'check'))
import maskscale_codec as codec                          # noqa: E402
from helpers import RECORDED_NUMPY                       # noqa: E402

from fastnc import _maskscale as M                       # noqa: E402

DATA = os.path.join(HERE, 'data', 'maskscale')
NP_MAJOR = numpy.lib.NumpyVersion(numpy.__version__).major


def _load():
    with gzip.open(os.path.join(DATA, 'inputs.json.gz')) as f:
        inputs = json.loads(f.read())['inputs']
    with gzip.open(os.path.join(DATA, 'expected_np%d.json.gz' % min(NP_MAJOR, 2))) as f:
        exp = json.loads(f.read())
    return inputs, exp


INPUTS, EXPECTED = _load()


def _decode(rec):
    dtype = str if rec['dtype'] == 'str' else codec.dtype_dec(rec['dtype'])
    attrs = {k: codec.obj_dec(v) for k, v in rec['attrs'].items()}
    key = tuple(slice(*s) if isinstance(s, list) else s for s in rec['key'])
    return codec.arr_dec(rec['raw']), attrs, dtype, key


def _call(rec, flags, fast, attrs=None):
    raw, own, dtype, key = _decode(rec)
    m, s, am, c = flags
    save = M.FAST_MIN_SIZE, M.BLOCK, M.NP_BLOCK
    if fast:
        M.FAST_MIN_SIZE, M.BLOCK, M.NP_BLOCK = 1, 5, 64
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return M.apply(raw, own if attrs is None else attrs, dtype, mask=bool(m),
                           scale=bool(s), always_mask=bool(am), chartostring=bool(c),
                           var_dtype=rec['kind'], no_fill=rec['no_fill'],
                           var_shape=tuple(rec['shape']), key=key)
    finally:
        M.FAST_MIN_SIZE, M.BLOCK, M.NP_BLOCK = save


def _run(rec, flags, fast):
    return codec.outcome(lambda: _call(rec, flags, fast))


VALID = ('valid_range', 'valid_min', 'valid_max')


def _castable(a, dtype):
    a = numpy.asarray(a)
    try:
        b = numpy.array(a, dtype)
        return bool(((a == b) | (numpy.isnan(a) & numpy.isnan(b))).all())
    except Exception:
        return False


def _unpacked_ranges(attrs, dtype):
    """The valid range attributes in unpacked units, by the NUG / CF rule: of
    a float type, that of scale_factor or add_offset, not the stored integers'."""
    if dtype is str or numpy.dtype(dtype).kind not in 'iu':
        return set()
    packing = {numpy.asarray(attrs[k]).dtype for k in ('scale_factor', 'add_offset') if k in attrs}
    if not all(numpy.asarray(attrs[k]).dtype.kind in 'iuf' and numpy.asarray(attrs[k]).ndim == 0
               for k in ('scale_factor', 'add_offset') if k in attrs):
        return set()
    return {k for k in VALID if k in attrs and numpy.asarray(attrs[k]).dtype.kind == 'f'
            and numpy.asarray(attrs[k]).dtype in packing
            and numpy.asarray(attrs[k]).dtype != numpy.dtype(dtype)}


def _unpacked_diff(rec, flags, fast):
    """For a case with a valid range in unpacked units, which netCDF4-python
    applies to the stored values: fastnc's answer against netCDF4-python's
    with that range taken out (fastnc's, which the other cases hold to
    netCDF4-python's), masked besides where the unpacked value (the same
    case read scaled and unmasked) is outside it. None if the range is not
    in unpacked units."""
    raw, attrs, dtype, key = _decode(rec)
    unp = _unpacked_ranges(attrs, dtype)
    if not unp:
        return None
    usable = {k for k in VALID if k in attrs and (k in unp or _castable(attrs[k], dtype))}
    if 'valid_range' in usable and numpy.asarray(attrs['valid_range']).size == 2:
        lo, hi = attrs['valid_range'][0], attrs['valid_range'][1]
        lo_unp = hi_unp = 'valid_range' in unp
        drop = set(VALID) if lo_unp else set()
    else:
        lo, hi = attrs.get('valid_min'), attrs.get('valid_max')
        lo_unp, hi_unp = 'valid_min' in unp, 'valid_max' in unp
        drop = {k for k in ('valid_min', 'valid_max') if k in unp}
    m, s, am, c = flags
    if not s and attrs.get('_Unsigned') in ('true', 'True') and numpy.dtype(dtype).kind == 'i':
        lo_unp = hi_unp = False                 # (signed values: not applied)
    base = _call(rec, flags, fast, {k: v for k, v in attrs.items() if k not in drop})
    got = _call(rec, flags, fast)
    if not m:
        return codec.diff(codec.outcome(lambda: got), codec.outcome(lambda: base))
    u = numpy.asarray(_call(rec, (0, 1, 1, c), fast))
    want = numpy.ma.getmaskarray(base).copy()
    with numpy.errstate(invalid='ignore'):
        if lo_unp:
            want |= u < lo
        if hi_unp:
            want |= u > hi
    have = numpy.ma.getmaskarray(got)
    if not numpy.array_equal(have, want):
        return 'mask differs at %d places' % int((have != want).sum())
    if got is numpy.ma.masked:                  # (one value, masked: numpy's masked constant)
        return '' if want.all() else 'masked constant'
    gd, bd = numpy.ma.getdata(got), numpy.ma.getdata(base)
    if gd.dtype != bd.dtype:
        return 'dtype %s != %s' % (gd.dtype, bd.dtype)
    keep = ~want
    if gd[keep].tobytes() != bd[keep].tobytes():
        return 'values differ'
    if want.any() != isinstance(got, numpy.ma.MaskedArray) and not am:
        return 'masked array when nothing is masked, or the other way'
    return ''


@pytest.fixture(autouse=True, params=['kernel', 'numpy'])
def engine(request, monkeypatch):
    """Every test twice: with the numba kernels, and with the numpy path a
    process uses until they have loaded."""
    monkeypatch.setattr(M, '_kernels_ready', lambda: request.param == 'kernel')
    if request.param == 'numpy':
        monkeypatch.setattr(M, 'NP_BLOCK', 4096)
    return request.param


def _runner(engine):
    return M.run_plan if engine == 'kernel' else M.run_plan_numpy


def _only_fill(d):
    return all(part.startswith('fill_value:') for part in d.split('; '))


def _check(fast):
    """Every recorded case. Under a numpy the answers were not recorded with
    (helpers.recorded_numpy: only the fill value's type can differ, by
    numpy's masked-array rules), a fill value that differs must be the one
    netCDF4-python's own code (the reference path) gives under this numpy.
    A valid range in unpacked units is where fastnc departs from
    netCDF4-python on purpose (_unpacked_diff)."""
    bad = []
    unpacked = 0
    for i, flags, j in EXPECTED['results']:
        rec = INPUTS[i]
        d = _unpacked_diff(rec, flags, fast)
        if d is not None:
            unpacked += 1
        else:
            got = _run(rec, flags, fast)
            d = codec.diff(got, EXPECTED['table'][j])
            if d and not RECORDED_NUMPY and _only_fill(d):
                d = '' if not fast else codec.diff(got, _run(rec, flags, False))
        if d:
            bad.append('%s %s key=%s flags=%s: %s' % (rec['file'], rec['var'], rec['key'], flags, d))
    assert not bad, '%d mismatches, e.g.\n%s' % (len(bad), '\n'.join(bad[:10]))
    assert unpacked > 1000                      # (the departure is tested)


def test_cases_recorded():
    assert len(EXPECTED['results']) > 30000


def test_reference_path_matches_netcdf4():
    _check(fast=False)


def test_fast_path_matches_netcdf4():
    _check(fast=True)


# ---------------------------------------------------------------------------
# big arrays: the kernel on many threads == the reference path

def _big_values(t, n, rng, specials):
    dt = numpy.dtype(t)
    if dt.kind == 'f':
        x = (rng.standard_normal(n) * 1000).astype(dt)
        x[rng.integers(0, n, n // 50)] = numpy.nan
        x[rng.integers(0, n, n // 500)] = numpy.inf
        x[rng.integers(0, n, n // 500)] = -0.0
        x[rng.integers(0, n, n // 500)] = numpy.finfo(dt).tiny / 3
    else:
        ii = numpy.iinfo(dt)
        x = rng.integers(ii.min, ii.max, n, dtype=dt, endpoint=True)
        if dt.itemsize == 8:                        # near float32 / float64 rounding ties
            k = n // 4
            base = rng.integers(1, 1 << 40, k).astype(dt)
            x[:k] = base << numpy.array(rng.integers(0, 22, k), dt) | numpy.array(
                rng.integers(0, 4, k), dt)
    for s in specials:
        x[rng.integers(0, n, n // 200)] = numpy.array(s).astype(dt)
    return x


BIG = [
    ('u2', {'_FillValue': numpy.uint16(65535), 'valid_min': numpy.uint16(0),
            'valid_max': numpy.uint16(65527), 'scale_factor': numpy.float32(1.9991757653770037e-05),
            'add_offset': numpy.float32(0)}, [65535, 65527, 65528]),
    ('u2', {'_FillValue': numpy.uint16(65535), 'valid_range': numpy.array([0, 32767], 'u2')},
     [65535, 32767, 32768]),
    ('u2', {'scale_factor': numpy.float32(0.09482232481241226),
            'add_offset': numpy.float32(-2.0999999046325684), 'valid_min': numpy.uint16(0),
            'valid_max': numpy.uint16(8189), '_FillValue': numpy.uint16(8191)}, [8191, 8189, 8190]),
    ('i2', {'scale_factor': numpy.float64(0.01), '_FillValue': numpy.int16(-32768)}, [-32768]),
    ('i2', {'_Unsigned': 'true', 'scale_factor': numpy.float32(0.5), 'add_offset': numpy.float64(3),
            '_FillValue': numpy.int16(-1), 'missing_value': numpy.array([7, -2], 'i2')}, [-1, 7, -2]),
    ('i1', {'_Unsigned': 'true', 'valid_max': numpy.int8(100)}, [100, 101, -127]),
    ('u1', {}, [255]),
    ('i4', {'scale_factor': numpy.float32(4.291536606615409e-05), 'add_offset': numpy.float32(0),
            'valid_min': numpy.int32(-2097151), 'valid_max': numpy.int32(2097151),
            '_FillValue': numpy.int32(-2097152)}, [-2097152, 2097151, 2097152]),
    ('u4', {'scale_factor': numpy.float32(0.001), 'add_offset': numpy.float32(7)}, [4294967295]),
    ('i8', {'scale_factor': numpy.float32(1.5)}, [-9223372036854775806]),
    ('i8', {'add_offset': numpy.float64(0.25)}, []),
    ('u8', {'scale_factor': numpy.float64(3.0), 'missing_value': numpy.uint64(5)}, [5]),
    ('u8', {'scale_factor': numpy.float32(1), 'add_offset': numpy.float32(0)}, []),
    ('f4', {}, [9.969209968386869e+36]),
    ('f4', {'_FillValue': numpy.float32(numpy.nan), 'scale_factor': numpy.float32(2)}, []),
    ('f4', {'missing_value': numpy.array([numpy.nan, -999], 'f4'), 'add_offset': numpy.float64(273.15)},
     [-999]),
    ('f4', {'valid_range': numpy.array([-100, 100], 'f4'), 'scale_factor': numpy.float64(0.1),
            'add_offset': numpy.float32(-1)}, [100, -100]),
    ('f8', {'_FillValue': numpy.float64(-999.9), 'valid_min': numpy.float64(0),
            'scale_factor': numpy.float32(0.3)}, [-999.9]),
    ('f8', {'scale_factor': numpy.float32(1), 'add_offset': numpy.float32(0)}, []),
]


@pytest.mark.parametrize('case', range(len(BIG)))
@pytest.mark.parametrize('flags', [(1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)])
def test_big_fast_equals_reference(case, flags, engine):
    t, attrs, specials = BIG[case]
    rng = numpy.random.default_rng(case)
    x = _big_values(t, 300003, rng, specials).reshape(3, -1)
    m, s, am = flags
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = codec.outcome(lambda: M._reference(x.copy(), attrs, x.dtype, 'primitive', bool(m),
                                                 bool(s), bool(am), True, False, None, None, None))
        plan = M.make_plan(x.dtype, attrs, x.dtype, 'primitive', bool(m), bool(s), False)
        assert plan is not None
        got = codec.outcome(lambda: _runner(engine)(plan, x, bool(am), threads=8))
    assert codec.diff(got, ref) == ''


def test_big_clean_and_all_masked():
    """No value masked (the mask is never written) and every value masked."""
    x = numpy.arange(200000, dtype='u2')
    for attrs in [{'scale_factor': numpy.float32(0.5)},
                  {'valid_max': numpy.uint16(0), 'scale_factor': numpy.float32(0.5)}]:
        for am in (True, False):
            ref = M._reference(x.copy(), attrs, x.dtype, 'primitive', True, True, am, True,
                               False, None, None, None)
            got = M.apply(x, attrs, x.dtype, always_mask=am)
            assert codec.diff(codec.result_enc(got), codec.result_enc(ref)) == ''


# attribute types netCDF4 cannot write (a _FillValue of another type, as
# HDF5 files made elsewhere have) or values that cannot be cast: the fast
# path against the reference path
ODD = [
    ('u2', {'_FillValue': numpy.float64(65535.0), 'valid_range': numpy.array([0., 1e5])}),
    ('i2', {'_FillValue': numpy.float32(-1.5), 'missing_value': numpy.int32(70000)}),
    ('u1', {'_FillValue': numpy.int16(-1), 'valid_min': numpy.int8(-5)}),
    ('i1', {'valid_min': numpy.uint16(0), 'valid_max': numpy.uint16(127),
            'scale_factor': numpy.float32(0.006138000171631575), '_FillValue': numpy.int8(-1)}),
    ('f4', {'_FillValue': numpy.float64(1e40), 'missing_value': numpy.float64(0.1)}),
    ('f4', {'missing_value': 'oops', 'scale_factor': numpy.float32(2)}),
    ('i4', {'scale_factor': 'x'}),
    ('i4', {'scale_factor': numpy.array([1., 2.])}),
    ('u4', {'_Unsigned': 'false', 'valid_range': numpy.array([1, 2, 3], 'u4'),
            'valid_min': numpy.uint32(4)}),
]


@pytest.mark.parametrize('case', range(len(ODD)))
def test_odd_attributes_fast_equals_reference(case):
    t, attrs = ODD[case]
    rng = numpy.random.default_rng(100 + case)
    x = _big_values(t, 70000, rng, [0, 1, 5, 127, 255]).reshape(7, -1)
    for flags in [(1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)]:
        m, s, am = flags
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = codec.outcome(lambda: M._reference(x.copy(), attrs, x.dtype, 'primitive', bool(m),
                                                     bool(s), bool(am), True, False, None, None,
                                                     None))
            got = codec.outcome(lambda: M.apply(x, attrs, x.dtype, mask=bool(m), scale=bool(s),
                                                always_mask=bool(am)))
        assert codec.diff(got, ref) == '', (flags, codec.diff(got, ref))


# valid ranges in unpacked units (of the type of scale_factor / add_offset):
# masked where the unpacked value is outside, checked by hand
UNPACKED = [
    # HRD's TDR winds: knots, on int16 hundredths of a knot (every wind valid)
    ('i2', {'valid_range': numpy.array([-900, 900], 'f4'), 'scale_factor': numpy.float32(0.01),
            'add_offset': numpy.float32(0)}),
    # HRD's reflectivity: dBZ, with a _FillValue in counts
    ('i2', {'scale_factor': numpy.float32(0.01), 'add_offset': numpy.float32(0),
            'valid_range': numpy.array([-31.9, 70], 'f4'), '_FillValue': numpy.int16(-3200)}),
    # GRIDSAT-B1 / HURSAT: kelvin
    ('i2', {'scale_factor': numpy.float32(0.01), 'add_offset': numpy.float32(200),
            'valid_range': numpy.array([140, 375], 'f4'), '_FillValue': numpy.int16(-31999)}),
    ('i2', {'scale_factor': numpy.float32(0.01), 'valid_range': numpy.array([0, 128], 'f4')}),
    ('u2', {'scale_factor': numpy.float64(-0.5), 'add_offset': numpy.float64(100),
            'valid_min': numpy.float64(-50.25)}),
    ('i1', {'add_offset': numpy.float32(10), 'valid_max': numpy.float32(57.5),
            'valid_min': numpy.int8(-100)}),
    ('i2', {'_Unsigned': 'true', 'scale_factor': numpy.float32(0.1),
            'valid_range': numpy.array([100, 6000], 'f4'), '_FillValue': numpy.int16(-1)}),
    ('i4', {'scale_factor': numpy.float64(1e-3), 'add_offset': numpy.float64(-5),
            'valid_range': numpy.array([-1e5, 2e6])}),
    ('u4', {'scale_factor': numpy.float32(3.5), 'valid_max': numpy.float32(1e9)}),
    ('i8', {'scale_factor': numpy.float64(2.5e-9), 'valid_range': numpy.array([-1e9, 7.5e9])}),
    ('u8', {'scale_factor': numpy.float32(1), 'add_offset': numpy.float32(0),
            'valid_min': numpy.float32(1e18)}),
    ('i2', {'scale_factor': numpy.float32(0.01), 'valid_range': numpy.array([1e6, 2e6], 'f4')}),
    ('i2', {'scale_factor': numpy.float32(0.01), 'valid_range': numpy.array([-1e6, 1e6], 'f4')}),
]


@pytest.mark.parametrize('case', range(len(UNPACKED)))
@pytest.mark.parametrize('flags', [(1, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_unpacked_valid_range(case, flags, engine):
    t, attrs = UNPACKED[case]
    if sys.platform == 'darwin' and NP_MAJOR < 2 and engine == 'kernel' and t == 'u8' and flags[1]:
        # (numpy 1 on Apple silicon makes 2 of these 90,003 uint64 values near 2^57 one float32
        # ulp from the kernels' correctly rounded ones; not so on Windows or Linux, nor with
        # numpy 2: see Accuracy)
        pytest.skip('numpy 1 on macOS rounds some uint64 -> float32 casts apart')
    rng = numpy.random.default_rng(200 + case)
    x = _big_values(t, 90003, rng, [0, 1, -1, -3190, -3191, 7000, 4000]).reshape(3, -1)
    m, s, am = flags
    unsigned = attrs.get('_Unsigned') == 'true'
    xs = x.view('u%d' % x.itemsize) if unsigned and s else x
    sf, off = attrs.get('scale_factor'), attrs.get('add_offset')
    if sf is not None and off is not None and sf == 1 and off == 0:
        u = xs.astype(sf.dtype)
    else:
        u = xs * sf if sf is not None else xs
        u = u + off if off is not None else u
    want = numpy.zeros(x.shape, bool)
    lo, hi = attrs.get('valid_min'), attrs.get('valid_max')
    if 'valid_range' in attrs:
        lo, hi = attrs['valid_range']
    if not (unsigned and not s):                # (signed values read: the range is not applied)
        if lo is not None:
            want |= (u < lo) if lo.dtype.kind == 'f' else (xs < lo)
        if hi is not None:
            want |= u > hi
    if '_FillValue' in attrs:
        want |= xs == numpy.array(attrs['_FillValue']).astype(x.dtype).view(xs.dtype)
    elif not (unsigned and s):                  # (the default fill, as stored, when not viewed)
        want |= x == M.DEFAULT_FILLVALS[x.dtype.str[1:]]
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # (no warning: the range is used)
        ref = M._reference(x.copy(), attrs, x.dtype, 'primitive', bool(m), bool(s), bool(am),
                           True, False, None, None, None)
        plan = M.make_plan(x.dtype, attrs, x.dtype, 'primitive', bool(m), bool(s), False)
        got = _runner(engine)(plan, x, bool(am), threads=8)
    assert codec.diff(codec.outcome(lambda: got), codec.outcome(lambda: ref)) == ''
    assert numpy.array_equal(numpy.ma.getmaskarray(got), want)
    if s:
        assert numpy.array_equal(numpy.ma.getdata(got)[~want], u[~want])
    if case == 0:
        assert not want.any()                   # (every wind)
    if case == 11:
        assert want.all()


def test_unpacked_valid_range_needs_packing():
    """Without scale_factor / add_offset, or with a range of the stored type,
    the range is netCDF4-python's, in stored units."""
    v = M._Var({'valid_range': numpy.array([-900, 900], 'f4')}, numpy.dtype('i2'), 'primitive',
               True, True, True, True, False, None)
    assert not M._unpacked_units(v, 'valid_range')
    v.attrs = {'valid_range': numpy.array([-900, 900], 'i2'), 'scale_factor': numpy.float32(0.01)}
    assert not M._unpacked_units(v, 'valid_range')
    v.attrs = {'valid_range': numpy.array([-900, 900], 'f8'), 'scale_factor': numpy.float32(0.01)}
    assert not M._unpacked_units(v, 'valid_range')
    v.attrs['scale_factor'] = numpy.float64(0.01)
    assert M._unpacked_units(v, 'valid_range')
    assert M._valid_bounds(v, None, warnings.warn) == (None, None)     # (all of int16 valid)
    v.attrs['valid_range'] = numpy.array([-9, 9.5])
    lo, hi = M._valid_bounds(v, None, warnings.warn)
    assert (int(lo), int(hi), lo.dtype, hi.dtype) == (-900, 950, numpy.dtype('i2'), numpy.dtype('i2'))


def test_same_warnings():
    x = numpy.arange(100000, dtype='i2')
    attrs = {'_FillValue': numpy.float32(-1.5), 'scale_factor': numpy.float32(2)}
    for size in (1, 10 ** 9):
        save = M.FAST_MIN_SIZE
        M.FAST_MIN_SIZE = size
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                M.apply(x, attrs, x.dtype)
        finally:
            M.FAST_MIN_SIZE = save
        assert [str(m.message) for m in w] == [
            'WARNING: _FillValue not used since it\ncannot be safely cast to variable data type']


# ---------------------------------------------------------------------------
# GPU == CPU

def _cupy():
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception:
        return None


@pytest.mark.skipif(_cupy() is None, reason='no cupy / GPU')
@pytest.mark.parametrize('case', range(len(BIG)))
def test_gpu_equals_cpu(case):
    cupy = _cupy()
    t, attrs, specials = BIG[case]
    rng = numpy.random.default_rng(case)
    x = _big_values(t, 300003, rng, specials).reshape(3, -1)
    for flags in [(1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)]:
        m, s, am = flags
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cpu = codec.outcome(lambda: M.apply(x, attrs, x.dtype, mask=bool(m), scale=bool(s),
                                                always_mask=bool(am)))

            def gpu():
                r = M.apply(cupy.asarray(x), attrs, x.dtype, mask=bool(m), scale=bool(s),
                            always_mask=bool(am), xp=cupy)
                return r.get() if hasattr(r, 'get') else r
            back = codec.outcome(gpu)
        assert codec.diff(back, cpu) == '', flags


def _nan_cases():
    """float data with NaN payloads (quiet, signalling, negative),
    subnormals, infinities; attributes that make NaNs (inf * 0, inf - inf)
    or round into the subnormal range."""
    rng = numpy.random.default_rng(7)
    f4 = numpy.concatenate([
        (rng.integers(0x7f800001, 0x7fffffff, 3000, dtype='u4')
         | (rng.integers(0, 2, 3000, dtype='u4') << numpy.uint32(31))).view('f4'),
        (rng.integers(1, 0x007fffff, 3000, dtype='u4')
         | (rng.integers(0, 2, 3000, dtype='u4') << numpy.uint32(31))).view('f4'),
        numpy.array([numpy.inf, -numpy.inf, 0.0, -0.0, 1e-38, 3e38], 'f4'),
        (rng.standard_normal(30000) * 1e-36).astype('f4')])
    f8 = numpy.concatenate([
        (rng.integers(0x7ff0000000000001, 0x7fffffffffffffff, 3000, dtype='u8')
         | (rng.integers(0, 2, 3000, dtype='u8') << numpy.uint64(63))).view('f8'),
        f4.astype('f8'), rng.standard_normal(3000) * 1e-40])
    attrs = [{'scale_factor': numpy.float32(0.5)}, {'scale_factor': numpy.float32(0)},
             {'add_offset': numpy.float32(-numpy.inf)},
             {'scale_factor': numpy.float32(3e-5), 'add_offset': numpy.float32(1e-39)},
             {'scale_factor': numpy.float64(0.1), 'add_offset': numpy.float32(2)},
             {'scale_factor': numpy.float32(1), 'add_offset': numpy.float32(0)},
             {'scale_factor': numpy.float32(numpy.nan)},
             {'_FillValue': numpy.float32(numpy.nan), 'scale_factor': numpy.float32(2)},
             {'missing_value': numpy.float32(1e-40), 'scale_factor': numpy.float64(2)}]
    return [(x, a) for x in (f4, f8) for a in attrs]


NAN_CASES = _nan_cases()


@pytest.mark.parametrize('case', range(len(NAN_CASES)))
def test_nan_payloads_subnormals(case, engine):
    """CPU kernel == numpy (the reference path), and GPU == numpy, on NaN
    payloads, subnormals and invalid operations."""
    x, attrs = NAN_CASES[case]
    cupy = _cupy()
    for flags in [(1, 1, 1), (0, 1, 1)]:
        m, s, am = flags
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = codec.outcome(lambda: M._reference(x.copy(), attrs, x.dtype, 'primitive', bool(m),
                                                     bool(s), bool(am), True, False, None, None,
                                                     None))
            plan = M.make_plan(x.dtype, attrs, x.dtype, 'primitive', bool(m), bool(s), False)
            assert plan is not None
            if plan.nan_const and numpy.lib.NumpyVersion(numpy.__version__) < '1.26.0':
                pytest.skip("a NaN times a NaN: numpy < 1.26's payload follows where its SIMD "
                            "loops split, which numpy in blocks and numpy whole differ on")
            got = codec.outcome(lambda: _runner(engine)(plan, x, bool(am), threads=4))
            assert codec.diff(got, ref) == '', ('cpu', flags, codec.diff(got, ref))
            if cupy is not None:
                def gpu():
                    r = M.apply(cupy.asarray(x), attrs, x.dtype, mask=bool(m), scale=bool(s),
                                always_mask=bool(am), xp=cupy)
                    return r.get() if hasattr(r, 'get') else r
                back = codec.outcome(gpu)
                assert codec.diff(back, ref) == '', ('gpu', flags, codec.diff(back, ref))


@pytest.mark.skipif(_cupy() is None, reason='no cupy / GPU')
def test_gpu_matches_netcdf4():
    """Every recorded numeric case through the GPU path (the kernel, or the
    CPU where the kernel does not go)."""
    cupy = _cupy()
    bad = []
    for i, flags, j in EXPECTED['results']:
        rec = INPUTS[i]
        raw, attrs, dtype, key = _decode(rec)
        if raw.dtype.kind not in 'iuf' or raw.ndim == 0:
            continue
        m, s, am, c = flags

        def gpu():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = M.apply(cupy.asarray(raw), attrs, dtype, mask=bool(m), scale=bool(s),
                            always_mask=bool(am), chartostring=bool(c), var_dtype=rec['kind'],
                            no_fill=rec['no_fill'], var_shape=tuple(rec['shape']), key=key,
                            xp=cupy)
            return r.get() if hasattr(r, 'get') else r
        if _unpacked_ranges(attrs, dtype):      # (fastnc's own answer: the CPU's)
            d = codec.diff(codec.outcome(gpu), _run(rec, flags, False))
        else:
            d = codec.diff(codec.outcome(gpu), EXPECTED['table'][j])
        if d:
            bad.append('%s %s key=%s flags=%s: %s' % (rec['file'], rec['var'], rec['key'], flags, d))
    assert not bad, '%d mismatches, e.g.\n%s' % (len(bad), '\n'.join(bad[:10]))
