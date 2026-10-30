"""Bit-exact JSON encoding of mask-and-scale inputs (raw arrays, attribute
values as getncattr returns them) and results (ndarray / MaskedArray /
masked constant / numpy scalar, with dtype, data bytes, mask and the
masked array's own _fill_value), shared by the harness that records
netCDF4-python's answers (tools/check/maskscale_cases.py) and the tests
that check fastnc against them without netCDF4-python
(tests/test_maskscale.py). Only builtins go into the JSON, so files
recorded under numpy 2 load under numpy 1 and back."""
import base64
import json

import numpy

ma = numpy.ma


def _b64(b):
    return base64.b64encode(bytes(b)).decode('ascii')


def _unb64(s):
    return base64.b64decode(s.encode('ascii'))


def dtype_enc(dt):
    dt = numpy.dtype(dt)
    if dt.names:
        return {'aligned': bool(dt.isalignedstruct), 'itemsize': dt.itemsize, 'names': list(dt.names),
                'formats': [dtype_enc(dt.fields[n][0]) for n in dt.names],
                'offsets': [dt.fields[n][1] for n in dt.names]}
    if dt.subdtype is not None:
        return {'sub': dtype_enc(dt.subdtype[0]), 'shape': list(dt.subdtype[1])}
    return dt.str


def dtype_dec(e):
    if isinstance(e, str):
        return numpy.dtype(e)
    if 'sub' in e:
        return numpy.dtype((dtype_dec(e['sub']), tuple(e['shape'])))
    return numpy.dtype({'names': e['names'], 'formats': [dtype_dec(f) for f in e['formats']],
                        'offsets': e['offsets'], 'itemsize': e['itemsize']}, align=e['aligned'])


def _unpadded(a):
    """a structured array with its padding bytes zeroed (they are not values)."""
    if a.dtype.names:
        b = numpy.zeros(a.shape, a.dtype)
        for n in a.dtype.names:
            b[n] = a[n]
        return b
    return a


def arr_enc(a):
    a = numpy.asarray(a)
    if a.dtype.kind == 'O':
        return {'obj': [obj_enc(x) for x in a.reshape(-1)], 'shape': list(a.shape)}
    a = _unpadded(a)
    return {'d': dtype_enc(a.dtype), 'shape': list(a.shape),
            'b': _b64(numpy.ascontiguousarray(a).tobytes())}


def arr_dec(e):
    if 'obj' in e:
        out = numpy.empty(len(e['obj']), object)
        for i, x in enumerate(e['obj']):
            out[i] = obj_dec(x)
        return out.reshape(e['shape'])
    dt = dtype_dec(e['d'])
    return numpy.frombuffer(_unb64(e['b']), dt).reshape(e['shape']).copy()


def obj_enc(x):
    """An attribute value or object array element."""
    if x is None:
        return {'t': 'none'}
    if isinstance(x, numpy.generic):            # (before float: numpy.float64 is one)
        return {'t': 'scalar', 'd': dtype_enc(x.dtype), 'b': _b64(_unpadded(numpy.asarray(x)).tobytes())}
    if isinstance(x, bool):
        return {'t': 'bool', 'v': x}
    if isinstance(x, str):
        return {'t': 'str', 'v': x}
    if isinstance(x, bytes):
        return {'t': 'bytes', 'v': _b64(x)}
    if isinstance(x, int):
        return {'t': 'int', 'v': str(x)}
    if isinstance(x, float):
        return {'t': 'float', 'v': numpy.float64(x).tobytes().hex()}
    if isinstance(x, list):
        return {'t': 'list', 'v': [obj_enc(y) for y in x]}
    if isinstance(x, numpy.ndarray):
        return {'t': 'array', 'a': arr_enc(x)}
    raise TypeError('cannot encode %r' % type(x))


def obj_dec(e):
    t = e['t']
    if t == 'none':
        return None
    if t in ('bool', 'str'):
        return e['v']
    if t == 'bytes':
        return _unb64(e['v'])
    if t == 'int':
        return int(e['v'])
    if t == 'float':
        return float(numpy.frombuffer(bytes.fromhex(e['v']), numpy.float64)[0])
    if t == 'list':
        return [obj_dec(y) for y in e['v']]
    if t == 'scalar':
        return numpy.frombuffer(_unb64(e['b']), dtype_dec(e['d']))[0]
    if t == 'array':
        return arr_dec(e['a'])
    raise TypeError(t)


def result_enc(r):
    """What netCDF4-python's answer is, to the bit: its type, dtype, data,
    mask (None for nomask) and the masked array's _fill_value."""
    if r is ma.masked:
        return {'type': 'masked'}
    if isinstance(r, ma.MaskedArray):
        m = r._mask
        fv = r.__dict__.get('_fill_value')
        return {'type': 'MaskedArray', 'cls': type(r).__name__, 'data': arr_enc(r.data),
                'mask': None if m is ma.nomask else arr_enc(m),
                'fill_value': None if fv is None else obj_enc(fv)}
    if isinstance(r, numpy.ndarray):
        return {'type': 'ndarray', 'data': arr_enc(r)}
    if isinstance(r, numpy.generic):
        return {'type': 'scalar', 'data': obj_enc(r)}
    return {'type': type(r).__name__, 'data': obj_enc(r)}


def outcome(fn):
    """result_enc(fn()), or the exception fn raised (its type: netCDF4-python
    raises in a few corners, and so must fastnc)."""
    try:
        r = fn()
    except Exception as e:                      # noqa
        return {'type': 'error', 'error': type(e).__name__, 'msg': str(e)[:200]}
    return result_enc(r)


def result_key(e):
    return json.dumps(e, sort_keys=True)


def diff(got, exp):
    """'' when the encoded results agree, else a short description."""
    if got == exp:
        return ''
    if exp.get('type') == 'error' and got.get('type') == 'error' and got['error'] == exp['error']:
        return ''
    if got.get('type') != exp.get('type'):
        return 'type %s != %s' % (got.get('type'), exp.get('type'))
    out = []
    for k in exp:
        if got.get(k) != exp.get(k):
            if k in ('data', 'mask') and isinstance(exp[k], dict) and isinstance(got.get(k), dict) \
                    and 'shape' in exp[k] and 'b' in exp[k] and 'shape' in got[k] and 'b' in got[k]:
                if got[k]['d'] != exp[k]['d']:
                    out.append('%s dtype %s != %s' % (k, got[k]['d'], exp[k]['d']))
                    continue
                a, b = arr_dec(got[k]), arr_dec(exp[k])
                if a.shape != b.shape:
                    out.append('%s shape %s != %s' % (k, a.shape, b.shape))
                    continue
                av, bv = a.reshape(-1).view(numpy.uint8), b.reshape(-1).view(numpy.uint8)
                bad = numpy.flatnonzero((av != bv).reshape(a.size, -1).any(axis=1)) \
                    if a.size else []
                out.append('%s differs at %d places, e.g. %s: %r != %r' % (
                    k, len(bad), bad[:3], a.reshape(-1)[bad[:3]], b.reshape(-1)[bad[:3]]))
            else:
                out.append('%s: %s != %s' % (k, str(got.get(k))[:120], str(exp.get(k))[:120]))
    return '; '.join(out)
