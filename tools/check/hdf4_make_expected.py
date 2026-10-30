"""The expected values tests/test_hdf4.py checks fastnc against, from the
references: netCDF4-python (netCDF-c's HDF4 dispatch) for the data model,
dimensions, variables, attributes and values, pyhdf for the values netCDF-c
will not read (a variable's records past its own end, which fastnc fills;
data in an external file, which netCDF-c looks for in the working
directory only). Written to tests/data/hdf4/expected.json.

    python tools/check/hdf4_make_expected.py
"""
import hashlib
import json
import os
import sys

import netCDF4
import numpy as np
from pyhdf.SD import SD

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'tests', 'data', 'hdf4')
FILES = ['types.hdf', 'comp.hdf', 'layout.hdf', 'many.hdf']

# the selections tests/test_hdf4.py reads (as text: they go through eval)
SELECTIONS = ['...', '(slice(1, None, 2),)', '(-1,)', '(slice(None, None, -3),)',
              '(slice(1, -1), slice(None, None, 3))', '(Ellipsis, slice(2, 3))']


def digest(a):
    a = np.asarray(a)
    if a.dtype.kind not in 'SU':
        a = a.astype(a.dtype.newbyteorder('='))
    return '%s %s %s' % (a.dtype.str, list(a.shape), hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:24])


def attr_repr(v):
    if isinstance(v, (str, bytes)):
        return [type(v).__name__, v if isinstance(v, str) else v.hex()]
    a = np.asarray(v)
    return [type(v).__name__, a.dtype.str, list(a.shape), a.tobytes().hex()]


def valid(sel, shape):
    try:
        np.empty(shape, np.uint8)[eval(sel)]
        return True
    except IndexError:
        return False


def main():
    out = {'selections': SELECTIONS, 'files': {}}
    cwd = os.getcwd()
    os.chdir(DATA)                                  # (netCDF-c finds external files from here)
    try:
        for fn in FILES:
            d = netCDF4.Dataset(fn)
            d.set_auto_maskandscale(False)
            h = SD(fn)
            e = {'data_model': d.data_model, 'file_format': d.file_format, 'disk_format': d.disk_format,
                 'dimensions': [[k, len(v), v.isunlimited()] for k, v in d.dimensions.items()],
                 'attrs': {k: attr_repr(d.getncattr(k)) for k in d.ncattrs()}, 'variables': {}}
            for k, v in d.variables.items():
                ve = {'dimensions': list(v.dimensions), 'shape': list(v.shape),
                      'dtype': v.dtype.newbyteorder('=').str, 'chunking': v.chunking(),
                      'endian': v.endian(), 'filters': v.filters(),
                      'attrs': {n: attr_repr(v.getncattr(n)) for n in v.ncattrs()}, 'values': {}}
                for sel in SELECTIONS:
                    if not valid(sel, v.shape):
                        continue
                    try:
                        ve['values'][sel] = digest(v[eval(sel)])
                    except Exception:
                        # netCDF-c will not read past a variable's own records: pyhdf's
                        # records, then the fill value fastnc gives (as SDreaddata's)
                        s = h.select(k)
                        own = np.asarray(s.get())
                        full = np.empty(v.shape, own.dtype)
                        fv = v.getncattr('_FillValue') if '_FillValue' in v.ncattrs() else None
                        if fv is None:
                            fv = {'i4': -2147483647, 'i2': -32767, 'f4': 9.9692099683868690e+36,
                                  'f8': 9.9692099683868690e+36}[own.dtype.str[1:]]
                        full[...] = fv
                        full[tuple(slice(0, n) for n in own.shape)] = own
                        ve['values'][sel] = digest(full[eval(sel)])
                        ve['reference'] = 'pyhdf'
                e['variables'][k] = ve
            d.close()
            h.end()
            out['files'][fn] = e
    finally:
        os.chdir(cwd)
    with open(os.path.join(DATA, 'expected.json'), 'w', newline='\n') as f:
        json.dump(out, f, indent=0, sort_keys=False)
    print('wrote', os.path.join(DATA, 'expected.json'))


if __name__ == '__main__':
    sys.exit(main())
