"""Make tests/data/nc3_expected.npz: what tests/test_nc3.py checks fastnc's
netCDF3 reader against, taken from netCDF4-python (raw values, auto
mask/scale and chartostring off), so the tests need no netCDF4.

Stored: the edge-case files of nc3_make_edge.py (their bytes: they are
small), and for them and the committed tests/data/nc3_*.nc each file's
metadata (data model, dimensions, variables, attributes with their types)
as JSON and every variable's whole value; for the real and synthetic files
that are not committed (wrfout d02, corpus/synthetic/nc3_*), the metadata
and a SHA-256 of each variable's values, for a test that runs where they
are present.

    python tools/check/nc3_make_expected.py
"""
import glob
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import nc3_make_edge                                            # noqa: E402

DATA = os.path.dirname(os.path.dirname(os.path.dirname(ROOT))) if '.claude' in ROOT else ROOT
DATA = os.environ.get('FASTNC_DATA', DATA)
OUT = os.path.join(ROOT, 'tests', 'data', 'nc3_expected.npz')


def encode(v, arrays):
    """An attribute value as JSON, its type kept; numbers go to `arrays`."""
    if isinstance(v, str):
        return {'t': 'str', 'v': v}
    if isinstance(v, bytes):
        return {'t': 'bytes', 'v': v.hex()}
    if isinstance(v, np.generic):
        return {'t': 'scalar', 'dtype': v.dtype.str, 'hex': v.tobytes().hex()}
    if isinstance(v, np.ndarray):
        k = 'a%d' % len(arrays)
        arrays[k] = v
        return {'t': 'array', 'key': k}
    raise TypeError(type(v))


def describe(ds, arrays, values):
    """ds's metadata; values: 'arrays' (whole values into `arrays`) or
    'digest' (a SHA-256 of their bytes)."""
    meta = {'data_model': ds.data_model,
            'dimensions': [[k, len(d), d.isunlimited()] for k, d in ds.dimensions.items()],
            'attrs': [[k, encode(ds.getncattr(k), arrays)] for k in ds.ncattrs()],
            'variables': []}
    for name, v in ds.variables.items():
        x = np.asarray(v[...])
        m = {'name': name, 'dimensions': list(v.dimensions), 'shape': list(v.shape),
             'dtype': v.dtype.str, 'endian': v.endian(), 'filters': v.filters(),
             'chunking': v.chunking(),
             'attrs': [[k, encode(v.getncattr(k), arrays)] for k in v.ncattrs()],
             'value_dtype': x.dtype.str, 'value_shape': list(x.shape)}
        if values == 'arrays':
            k = 'a%d' % len(arrays)
            arrays[k] = x
            m['value'] = k
        else:
            m['sha256'] = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        meta['variables'].append(m)
    return meta


def main():
    tmp = tempfile.mkdtemp(prefix='fastnc_nc3_expected_')
    edge = nc3_make_edge.make_all(tmp)
    arrays = {}
    index = {'edge': {}, 'committed': {}, 'external': {}}

    def ref(path):
        d = netCDF4.Dataset(nc3_make_edge.reference_for(path))
        d.set_auto_maskandscale(False)
        d.set_auto_chartostring(False)
        return d

    for p in edge:
        name = os.path.basename(p)
        k = 'a%d' % len(arrays)
        arrays[k] = np.frombuffer(open(p, 'rb').read(), np.uint8)
        with ref(p) as d:
            index['edge'][name] = {'file': k, 'meta': describe(d, arrays, 'arrays')}
    for p in sorted(glob.glob(os.path.join(ROOT, 'tests', 'data', 'nc3_*.nc'))):
        with ref(p) as d:
            index['committed'][os.path.basename(p)] = describe(d, arrays, 'arrays')
    ext = sorted(glob.glob(os.path.join(DATA, 'corpus', 'synthetic', 'nc3_*.nc')))
    ext += [os.path.join(DATA, 'wrfout_d02_2024-11-06_01_30_00')]
    for p in ext:
        if os.path.exists(p):
            with ref(p) as d:
                rel = os.path.relpath(p, DATA).replace(os.sep, '/')
                index['external'][rel] = describe(d, arrays, 'digest')
    arrays['index'] = np.frombuffer(json.dumps(index).encode('utf-8'), np.uint8)
    np.savez_compressed(OUT, **arrays)
    print('%s: %d arrays, %.0f kB' % (OUT, len(arrays), os.path.getsize(OUT) / 1e3))


if __name__ == '__main__':
    main()
