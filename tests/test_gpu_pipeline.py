"""device='gpu' reads through the pipelined path (_gpu._pipelined: one
decode of the whole selection, each band uploaded once its chunks are in),
with bands made tiny so that every chunked corpus variable is cut into
many: the values on the card equal the CPU's."""
import glob
import os

import numpy as np
import pytest

import fastnc
from fastnc import _gpu


def _cupy():
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception:
        return None


pytestmark = pytest.mark.skipif(_cupy() is None, reason='no cupy / GPU')

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.nc'))
               + glob.glob(os.path.join(DATA, '*.h5')))


def _host(x):
    if hasattr(x, 'get') and not isinstance(x, np.ndarray):
        return x.get()
    return x


def _enc(r):
    if np.ma.isMaskedArray(r):
        m = r._mask
        return ('ma', r.dtype.str, r.shape, np.ma.getdata(r).tobytes(),
                None if m is np.ma.nomask else np.asarray(m).tobytes(), repr(r.fill_value))
    a = np.asarray(r)
    return (a.dtype.str, a.shape, repr(a.tolist()) if a.dtype.hasobject else a.tobytes())


@pytest.mark.parametrize('name', FILES)
def test_pipelined_gpu_equals_cpu(name, monkeypatch):
    fastnc.warm()
    monkeypatch.setattr(_gpu, 'PIPE_BAND_BYTES', 64)
    used = []
    real = _gpu._pipelined
    monkeypatch.setattr(_gpu, '_pipelined', lambda *a: used.append(real(*a)) or used[-1])
    bad = []
    with fastnc.Dataset(os.path.join(DATA, name)) as ds:
        for g in ds._walk():
            for vname, v in g.variables.items():
                if not v.shape:
                    continue
                keys = [Ellipsis] + ([(slice(1, None),)] if v.shape[0] > 2 else [])
                for k in keys:
                    for m in (False, True):
                        v.set_auto_maskandscale(m)
                        try:
                            want = _enc(v.read(k))
                        except Exception as e:
                            want = ('raised', type(e).__name__)
                        try:
                            got = _enc(_host(v.read(k, device='gpu')))
                        except Exception as e:
                            got = ('raised', type(e).__name__)
                        if got != want:
                            bad.append((g.path, vname, k, m))
    assert not bad, bad[:5]
    if name in ('nc4.nc', 'filters.h5', 'idx_latest.h5'):
        assert any(used)
