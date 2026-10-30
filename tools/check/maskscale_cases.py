"""The mask-and-scale comparison harness: netCDF files with every
combination that matters written by netCDF4-python, read back by
netCDF4-python with auto mask and scale on (and its other switches), and
fastnc._maskscale.apply() checked against each answer -- the reference
path and the fast kernel (forced on, in tiny blocks) alike.

    python tools/check/maskscale_cases.py make DIR      # writes DIR/ms_*.nc
    python tools/check/maskscale_cases.py record DIR [OUT]

`make` runs once (the files are then read under every numpy); `record`
under each environment (numpy 2: fastnc-dev, numpy 1: bench-wrf) checks
fastnc and, given OUT (tests/data/maskscale), writes the inputs (raw
values, attributes as getncattr gives them) and that numpy's expected
results, which tests/test_maskscale.py replays without netCDF4-python.

The combinations: storage i1 u1 i2 u2 i4 u4 i8 u8 f4 f8 (netCDF4 and
CDF5; i1 i2 i4 f4 f8 in classic netCDF3), scale_factor / add_offset as
f4 / f8 / ints, neutral (1 and 0) or not, each alone or both;
_FillValue absent / present / NaN / at a bound, filling off;
missing_value scalar / vector / NaN / other type / unsafe; valid_min /
valid_max / valid_range, alone, together, of other types, unsafe,
wrong-sized; _Unsigned true / True / false / TRUE; values on the fill
values, at and past the valid bounds, NaN, inf, -0, subnormals, type
extremes. Read whole, as strided slices, as a row and as one value; plus
scalar variables, char (with and without _Encoding), strings, enums,
compounds and vlens.
"""
import gzip
import itertools
import json
import os
import sys
import warnings

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import maskscale_codec as codec                          # noqa: E402

SHAPE = (4, 6)
KEYS = [(slice(0, 4, 1), slice(0, 6, 1)), (slice(1, 4, 2), slice(0, 6, 2)),
        (2, slice(0, 6, 1)), (3, 5)]
FLAGS = [(1, 1, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
DEFAULTS = {'i1': -127, 'u1': 255, 'i2': -32767, 'u2': 65535, 'i4': -2147483647,
            'u4': 4294967295, 'i8': -9223372036854775806, 'u8': 18446744073709551614,
            'f4': 9.969209968386869e+36, 'f8': 9.969209968386869e+36}
F4, F8 = numpy.float32, numpy.float64


def wrap(v, t):
    """Python number v as a value of numpy type t (integers wrap)."""
    dt = numpy.dtype(t)
    if dt.kind == 'f':
        return dt.type(v)
    bits = 8 * dt.itemsize
    v = int(v) % (1 << bits)
    if dt.kind == 'i' and v >= 1 << (bits - 1):
        v -= 1 << bits
    return dt.type(v)


def domain(t, rng):
    """Natural fill / valid min / valid max / missing values for type t."""
    dt = numpy.dtype(t)
    if dt.kind == 'f':
        return dict(fill=-999.0, vmin=-50.0, vmax=400.5, miss=-888.0, miss2=1e30)
    lo, hi = numpy.iinfo(dt).min, numpy.iinfo(dt).max
    if dt.itemsize == 1:
        return dict(fill=hi if dt.kind == 'u' else -99, vmin=3 if dt.kind == 'u' else -50,
                    vmax=200 if dt.kind == 'u' else 100, miss=7, miss2=lo + 1)
    return dict(fill=hi if dt.kind == 'u' else lo + 1, vmin=-1000 if dt.kind == 'i' else 10,
                vmax=hi // 2 + 3, miss=12345, miss2=hi - 2)


def scale_options(t):
    o = [{}, {'scale_factor': F4(0.01)}, {'scale_factor': F8(0.01)}, {'scale_factor': F4(1)},
         {'scale_factor': F4(-0.5)}, {'add_offset': F4(-5.5)}, {'add_offset': F8(273.15)},
         {'add_offset': F8(0)},
         {'scale_factor': F4(0.1), 'add_offset': F4(-2.1)},
         {'scale_factor': F8(0.001), 'add_offset': F8(100)},
         {'scale_factor': F4(0.0123), 'add_offset': F8(-7.25)},
         {'scale_factor': F8(0.3), 'add_offset': F4(1.5)},
         {'scale_factor': F4(1), 'add_offset': F4(0)}, {'scale_factor': F8(1), 'add_offset': F8(0)},
         {'scale_factor': numpy.int16(2)}, {'scale_factor': numpy.int32(3), 'add_offset': numpy.int32(10)},
         {'scale_factor': F4(2.5), 'add_offset': F4(0)},
         {'scale_factor': F8(1e-5), 'add_offset': F4(1e30)},
         {'scale_factor': F4(3e37)},
         {'scale_factor': F4(1.9991757653770037e-05), 'add_offset': F4(0)},
         {'scale_factor': F4(0.09482232481241226), 'add_offset': F4(-2.0999999046325684)},
         {'add_offset': numpy.int16(-3)}, {'scale_factor': numpy.uint8(4)}]
    return o


def fill_options(t, d):
    dt = numpy.dtype(t)
    o = [None, d['fill'], 'nofill', DEFAULTS[t], d['vmax']]
    if dt.kind == 'f':
        o += [float('nan'), 0.0]
    elif dt.kind == 'i':
        o += [-1]
    return o


def missing_options(t, d):
    dt = numpy.dtype(t)
    o = [{}, {'missing_value': wrap(d['miss'], t)}, {'missing_value': F8(d['miss'])},
         {'missing_value': numpy.array([wrap(d['miss'], t), wrap(d['miss2'], t)])},
         {'missing_value': wrap(d['fill'], t)}]
    if dt.kind == 'f':
        o += [{'missing_value': numpy.array([numpy.nan], t)[0]},
              {'missing_value': numpy.array([numpy.nan, d['miss']], t)}]
    else:
        o += [{'missing_value': F8(0.5)}, {'missing_value': F8(numpy.nan)}]
    return o


def valid_options(t, d):
    dt = numpy.dtype(t)
    a, b = wrap(d['vmin'], t), wrap(d['vmax'], t)
    o = [{}, {'valid_min': a}, {'valid_max': b}, {'valid_min': a, 'valid_max': b},
         {'valid_range': numpy.array([a, b])},
         {'valid_range': numpy.array([a, b], 'f8'), 'valid_min': wrap(d['vmin'] + 1, t)},
         {'valid_range': numpy.array([a, b, b]), 'valid_max': b},
         {'valid_min': F8(float(a) + 0.5)},
         {'valid_max': numpy.int16(100) if dt.itemsize != 2 else numpy.int32(100)}]
    if dt.kind == 'f':
        o += [{'valid_min': numpy.array([numpy.nan], t)[0]}]
    return o


UNSIGNED = [None, 'true', 'True', 'false', 'TRUE']


def values(t, d, attrs, rng):
    """24 raw values hitting fill, default fill, bounds, missing values,
    type extremes and (floats) NaN, inf, -0, subnormals."""
    dt = numpy.dtype(t)
    c = [d['fill'], DEFAULTS[t], d['vmin'], d['vmax'], d['miss'], d['miss2'], 0, 1]
    if dt.kind == 'f':
        fi = numpy.finfo(dt)
        c += [numpy.nan, numpy.inf, -numpy.inf, -0.0, float(fi.tiny) / 4, float(fi.max),
              float(numpy.nextafter(dt.type(d['vmin']), dt.type(-numpy.inf))),
              float(numpy.nextafter(dt.type(d['vmax']), dt.type(numpy.inf))), 0.1, -1.5]
    else:
        ii = numpy.iinfo(dt)
        c += [d['vmin'] - 1, d['vmax'] + 1, int(ii.min), int(ii.max), -1, 2, 100, -100,
              int(ii.max) - 1, 3]
    for v in attrs.values():
        if isinstance(v, (numpy.generic, numpy.ndarray)) and numpy.asarray(v).dtype.kind in 'iuf':
            for x in numpy.asarray(v).reshape(-1)[:2]:
                if numpy.isfinite(x) and abs(float(x)) < 1e300:
                    c.append(x.item())
    while len(c) < 24:
        c.append(rng.integers(-1000, 1000).item() if dt.kind != 'f' else rng.normal() * 500)
    c = c[:24]
    out = numpy.array([wrap(v, t) if dt.kind != 'f' else v for v in c], dt)
    rng.shuffle(out)
    return out.reshape(SHAPE)


def gen_cases(types, rng, n_per_type):
    """(name, type, fill option, attrs) for random combinations, plus every
    scale option and every valid option at least once per type."""
    cases = []
    for t in types:
        d = domain(t, rng)
        so, fo, mo, vo = scale_options(t), fill_options(t, d), missing_options(t, d), valid_options(t, d)
        combos = set()
        for i in range(max(len(so), len(fo), len(mo), len(vo), len(UNSIGNED))):
            combos.add((i % len(so), i % len(fo), i % len(mo), i % len(vo), i % len(UNSIGNED)))
        while len(combos) < n_per_type:
            combos.add((rng.integers(len(so)), rng.integers(len(fo)), rng.integers(len(mo)),
                        rng.integers(len(vo)), rng.integers(len(UNSIGNED))))
        for k, (a, b, c, e, u) in enumerate(sorted(combos)):
            attrs = {}
            attrs.update(so[a])
            attrs.update(mo[c])
            attrs.update(vo[e])
            if UNSIGNED[u] is not None:
                attrs['_Unsigned'] = UNSIGNED[u]
            cases.append(('v_%s_%03d' % (t, k), t, fo[b], attrs, d))
    return cases


def make(dirname):
    import netCDF4
    os.makedirs(dirname, exist_ok=True)
    rng = numpy.random.default_rng(20261029)
    all_t = ['i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f4', 'f8']
    plan = [('ms_nc4.nc', 'NETCDF4', all_t, 120), ('ms_cdf5.nc', 'NETCDF3_64BIT_DATA', all_t, 25),
            ('ms_nc3.nc', 'NETCDF3_CLASSIC', ['i1', 'i2', 'i4', 'f4', 'f8'], 25)]
    skipped = 0
    for fname, fmt, types, n in plan:
        path = os.path.join(dirname, fname)
        if os.path.exists(path):
            os.remove(path)
        ds = netCDF4.Dataset(path, 'w', format=fmt)
        ds.createDimension('y', SHAPE[0])
        ds.createDimension('x', SHAPE[1])
        for name, t, fill, attrs, d in gen_cases(types, rng, n):
            kw = {}
            if fill == 'nofill':
                kw['fill_value'] = False
            elif fill is not None:
                kw['fill_value'] = wrap(fill, t) if numpy.dtype(t).kind != 'f' else fill
            v = ds.createVariable(name, t, ('y', 'x'), **kw)
            for an, av in attrs.items():
                try:
                    v.setncattr(an, av)
                except Exception:
                    skipped += 1
            v.set_auto_maskandscale(False)
            v[:] = values(t, d, attrs, rng)
            # a scalar twin of some
            if name.endswith(('0', '5')):
                s = ds.createVariable(name + '_s', t, (), **kw)
                for an, av in attrs.items():
                    try:
                        s.setncattr(an, av)
                    except Exception:
                        pass
                s.set_auto_maskandscale(False)
                vals = values(t, d, attrs, rng).reshape(-1)
                s[...] = vals[int(name[-3:]) % vals.size]
        if fmt == 'NETCDF4':
            _special(ds, rng)
        ds.close()
    print('written to', dirname, '(%d attributes the format refused)' % skipped)


def _special(ds, rng):
    """char, string, enum, compound and vlen variables."""
    ds.createDimension('c', 5)
    ds.createDimension('n', 3)
    chars = numpy.array([list(b'ab\0de'), list(b'\0\0\0\0\0'), list(b'xyz\xc3\xa9')], 'u1').view('S1')
    for name, attrs, kw in [('c_plain', {}, {}), ('c_enc', {'_Encoding': 'utf-8'}, {}),
                            ('c_enc_fill', {'_Encoding': 'utf-8'}, {'fill_value': b'x'}),
                            ('c_bytes', {'_Encoding': 'none'}, {}),
                            ('c_enc_nofill', {'_Encoding': 'ascii'}, {'fill_value': False}),
                            ('c_enc_miss', {'_Encoding': 'utf-8', 'missing_value': 'y'}, {})]:
        v = ds.createVariable(name, 'S1', ('n', 'c'), **kw)
        for an, av in attrs.items():
            v.setncattr(an, av)
        v.set_auto_chartostring(False)
        v.set_auto_maskandscale(False)
        cc = chars.copy()
        if name == 'c_enc_miss':
            cc = numpy.array([list(b'ayc\0y')] * 3, 'u1').view('S1')
        if name == 'c_bytes':
            cc = numpy.array([list(b'ab\0de'), list(b'\x01\x02\x03\x04\x05'), list(b'xyzzy')], 'u1').view('S1')
        v[:] = cc
    v = ds.createVariable('c_1d', 'S1', ('c',))
    v.setncattr('_Encoding', 'utf-8')
    v.set_auto_chartostring(False)
    v[:] = numpy.array(list(b'hello'), 'u1').view('S1')
    s = ds.createVariable('s_str', str, ('n',))
    s[0], s[1], s[2] = 'alpha', '', 'gé'
    s2 = ds.createVariable('s_str_attrs', str, ('n',))
    s2.setncattr('scale_factor', F4(2))
    s2.setncattr('missing_value', 'alpha')
    s2[0], s2[1], s2[2] = 'alpha', 'b', 'c'
    et = ds.createEnumType('u1', 'cloud_t', {'clear': 0, 'cloudy': 1, 'missing': 255})
    e = ds.createVariable('e_u1', et, ('n',), fill_value=255)
    e.setncattr('scale_factor', F4(0.5))
    e.setncattr('valid_max', numpy.uint8(1))
    e[:] = numpy.array([0, 1, 255], 'u1')
    et2 = ds.createEnumType('i2', 'cat_t', {'a': -1, 'b': 7, 'c': 300})
    e2 = ds.createVariable('e_i2', et2, ('n',))
    e2.setncattr('_Unsigned', 'true')
    e2.setncattr('missing_value', numpy.int16(7))
    e2[:] = numpy.array([-1, 7, 300], 'i2')
    cdt = numpy.dtype([('a', 'f4'), ('b', 'i2'), ('c', 'S1', (4,))])
    ct = ds.createCompoundType(cdt, 'cmp_t')
    cv = ds.createVariable('k_cmp', ct, ('n',))
    cv.setncattr('scale_factor', F4(2))
    cv.setncattr('_Encoding', 'utf-8')
    cv.set_auto_chartostring(False)
    data = numpy.zeros(3, ct.dtype)
    data['a'] = [1.5, -2, 3e10]
    data['b'] = [1, 2, -3]
    data['c'] = numpy.array([list(b'ab\0d'), list(b'wxyz'), list(b'\0\0\0\0')], 'u1').view('S1')
    cv[:] = data
    vt = ds.createVLType('i2', 'vl_t')
    for name, attrs in [('l_plain', {}), ('l_scaled', {'scale_factor': F4(0.5), 'add_offset': F8(1)}),
                        ('l_masked', {'missing_value': numpy.int16(2), '_Unsigned': 'true'})]:
        lv = ds.createVariable(name, vt, ('n',))
        for an, av in attrs.items():
            lv.setncattr(an, av)
        lv.set_auto_maskandscale(False)
        for i, a in enumerate([[1, 2, 3], [-4], [5, 6]]):
            lv[i] = numpy.array(a, 'i2')


def _walk(ds):
    for v in ds.variables.values():
        yield v


def _kind(v):
    if v._iscompound:
        return 'compound'
    if v._isenum:
        return 'enum'
    if v.dtype is str:
        return 'string'
    if v._isvlen:
        return 'vlen'
    return 'primitive'


def _keys(v):
    if v.ndim == 0:
        return [()]
    if v.shape == SHAPE:
        return KEYS
    n = v.shape[-1]
    if v.ndim == 2:
        return [(slice(0, v.shape[0], 1), slice(0, n, 1)), (slice(0, v.shape[0], 2), slice(0, 3, 1)),
                (1, slice(0, n, 1)), (1, 2)]
    return [(slice(0, n, 1),), (slice(1, n, 2),), (2,)]


def key_enc(k):
    return [[s.start, s.stop, s.step] if isinstance(s, slice) else int(s) for s in k]


def key_dec(k):
    return tuple(slice(*s) if isinstance(s, list) else s for s in k)


def read_nc(v, key, flags):
    m, s, am, c = flags
    v.set_auto_mask(bool(m))
    v.set_auto_scale(bool(s))
    v.set_always_mask(bool(am))
    v.set_auto_chartostring(bool(c))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return v[key]


def run_apply(inp, raw, key, flags, fast=False):
    """fastnc's answer (fast: the kernel forced on, in blocks of 5)."""
    from fastnc import _maskscale as M
    m, s, am, c = flags
    save = M.FAST_MIN_SIZE, M.BLOCK
    if fast:
        M.FAST_MIN_SIZE, M.BLOCK = 1, 5
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return M.apply(raw, inp['attrs'], inp['dtype'], mask=bool(m), scale=bool(s),
                           always_mask=bool(am), chartostring=bool(c), var_dtype=inp['kind'],
                           no_fill=inp['no_fill'], var_shape=inp['shape'], key=key)
    finally:
        M.FAST_MIN_SIZE, M.BLOCK = save


def record(dirname, out=None):
    import netCDF4
    inputs, expected, table = [], [], {}
    nbad = ncheck = 0
    for fname in sorted(os.listdir(dirname)):
        if not fname.startswith('ms_'):
            continue
        ds = netCDF4.Dataset(os.path.join(dirname, fname))
        for v in _walk(ds):
            kind = _kind(v)
            attrs = {a: v.getncattr(a) for a in v.ncattrs()}
            no_fill = (v.get_fill_value() is None) if v._isprimitive else False
            dtype = str if v.dtype is str else v.dtype
            inp = dict(attrs=attrs, dtype=dtype, kind=kind, no_fill=no_fill, shape=tuple(v.shape))
            flags_set = FLAGS + ([(1, 1, 1, 0), (0, 0, 1, 0)] if (dtype is not str and
                                 (dtype.kind == 'S' or dtype.names)) else [])
            for key in _keys(v):
                raw = read_nc(v, key, (0, 0, 1, 0))
                if v._isvlen and all(isinstance(k, int) for k in key) and v.ndim:
                    one = numpy.empty((), object)   # one vlen element, as read() gives it
                    one[()] = raw
                    raw = one
                raw = numpy.array(raw) if not isinstance(raw, numpy.ndarray) else raw
                rec_in = {'file': fname, 'var': v.name, 'kind': kind, 'no_fill': no_fill,
                          'shape': list(v.shape), 'key': key_enc(key),
                          'dtype': 'str' if dtype is str else codec.dtype_enc(dtype),
                          'attrs': {a: codec.obj_enc(x) for a, x in attrs.items()},
                          'raw': codec.arr_enc(raw)}
                inputs.append(rec_in)
                for flags in flags_set:
                    exp = codec.outcome(lambda: read_nc(v, key, flags))
                    k = codec.result_key(exp)
                    idx = table.setdefault(k, len(table))
                    expected.append([len(inputs) - 1, list(flags), idx])
                    for fast in (False, True):
                        ncheck += 1
                        got = codec.outcome(lambda: run_apply(inp, raw.copy(), key, flags, fast))
                        d = codec.diff(got, exp)
                        if d:
                            nbad += 1
                            if nbad <= 40:
                                print('MISMATCH %s %s key=%s flags=%s fast=%s attrs=%s: %s' % (
                                    fname, v.name, key_enc(key), flags, fast,
                                    {a: x for a, x in attrs.items()}, d))
        ds.close()
    print('numpy %s: %d results, %d checks, %d mismatches' % (numpy.__version__, len(expected),
                                                             ncheck, nbad))
    if out:
        os.makedirs(out, exist_ok=True)
        tab = [None] * len(table)
        for k, i in table.items():
            tab[i] = json.loads(k)
        np_major = numpy.lib.NumpyVersion(numpy.__version__).major
        blob = json.dumps({'inputs': inputs}, sort_keys=True).encode()
        path = os.path.join(out, 'inputs.json.gz')
        if os.path.exists(path):
            with gzip.open(path) as f:
                if f.read() != blob:
                    print('WARNING: inputs differ from', path, '(rewritten)')
        with gzip.GzipFile(path, 'wb', mtime=0) as f:
            f.write(blob)
        with gzip.GzipFile(os.path.join(out, 'expected_np%d.json.gz' % np_major), 'wb', mtime=0) as f:
            f.write(json.dumps({'numpy': numpy.__version__, 'netCDF4': netCDF4.__version__,
                                'results': expected, 'table': tab}).encode())
        print('recorded in', out)
    return nbad


if __name__ == '__main__':
    if sys.argv[1] == 'make':
        make(sys.argv[2])
    else:
        sys.exit(1 if record(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None) else 0)
