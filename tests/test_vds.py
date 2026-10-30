"""Virtual datasets and external storage against what h5py read from the
same files (tests/data/vds, written and recorded by tools/check/vds_make.py
in tests/data/vds/expected.json): shapes (unlimited ones as the sources
make them now), dtypes, fill values, virtual_sources(), external, and the
values of every selection. No h5py needed."""
import json
import os

import numpy as np
import pytest

import fastnc
import helpers

VDS = os.path.join(helpers.DATA, 'vds')
with open(os.path.join(VDS, 'expected.json')) as _fh:
    EXPECTED = json.load(_fh)

CASES = [(f, p) for f in sorted(EXPECTED['files']) for p in sorted(EXPECTED['files'][f])]


def space_info(sp):
    """What a selection says of itself (as tools/check/vds_make.py records it)."""
    out = {'type': int(sp.get_select_type()), 'shape': [int(x) for x in sp.shape]}
    for what, fn in (('npoints', lambda: int(sp.get_select_npoints())),
                     ('regular', lambda: sp.is_regular_hyperslab() and
                      [[int(x) for x in t] for t in sp.get_regular_hyperslab()]),
                     ('bounds', lambda: [[int(x) for x in t] for t in sp.get_select_bounds()]),
                     ('points', lambda: [[int(x) for x in p] for p in sp.get_select_elem_pointlist()]
                      if sp.get_select_type() == 1 else None)):
        try:
            out[what] = fn()
        except Exception:
            out[what] = 'error'
    return out


_OPEN = {}


def opened(name):
    if name not in _OPEN:
        _OPEN[name] = fastnc.File(os.path.join(VDS, name))
    return _OPEN[name]


def teardown_module():
    for f in _OPEN.values():
        f.close()
    _OPEN.clear()


@pytest.mark.parametrize('name,path', CASES, ids=['%s:%s' % c for c in CASES])
def test_properties(name, path):
    ref = EXPECTED['files'][name][path]
    ds = opened(name)[path]
    got = {'shape': list(ds.shape), 'maxshape': [None if m is None else int(m) for m in ds.maxshape],
           'dtype': helpers.dtype_str(ds.dtype),
           'fillvalue': helpers.strip_repr(helpers.encode(ds.fillvalue)),
           'is_virtual': ds.is_virtual, 'chunks': ds.chunks,
           'external': [list(x) for x in ds.external] if ds.external else None}
    want = {k: ref[k] for k in got}
    want['fillvalue'] = helpers.strip_repr(want['fillvalue'])
    assert got == want
    if ref['is_virtual']:
        srcs = [[m.file_name, m.dset_name, space_info(m.vspace)['type'], space_info(m.src_space)['type']]
                for m in ds.virtual_sources()]
        assert srcs == [[a, b, c['type'], d['type']] for a, b, c, d in ref['sources']]
        for m, r in zip(ds.virtual_sources(), ref['sources']):
            vi = space_info(m.vspace)
            for k in ('npoints', 'regular', 'bounds'):
                if r[2][k] != 'error':
                    assert vi[k] == r[2][k], (k, vi, r[2])


@pytest.mark.parametrize('name,path', CASES, ids=['%s:%s' % c for c in CASES])
def test_values(name, path):
    ref = EXPECTED['files'][name][path]
    ds = opened(name)[path]
    bad = []
    for sel, rec in ref['reads'].items():
        k = () if rec['key'] is None else helpers.key(rec['key'])
        ok, msg = helpers.same_values(helpers.digest(ds[k]), rec['digest'])
        if not ok:
            bad.append('%s: %s' % (sel, msg))
    assert not bad, '\n'.join(bad)


def test_closing_closes_sources():
    f = fastnc.File(os.path.join(VDS, 'vds_cross.h5'))
    f['stack3'][()]
    linked = f._f.linked
    assert any(v is not None for v in linked.values())
    f.close()
    assert all(v is None or v.h5 is None for v in linked.values() if hasattr(v, 'h5'))


def test_vds_prefix(tmp_path, monkeypatch):
    """A source found through HDF5_VDS_PREFIX (here: copied elsewhere)."""
    import shutil
    shutil.copy(os.path.join(VDS, 'vds_cross.h5'), tmp_path / 'vds_cross.h5')
    monkeypatch.setenv('HDF5_VDS_PREFIX', VDS)
    with fastnc.File(str(tmp_path / 'vds_cross.h5')) as f:
        rec = EXPECTED['files']['vds_cross.h5']['/stack3']['reads']['full']['digest']
        assert helpers.same_values(helpers.digest(f['stack3'][()]), rec)[0]
    monkeypatch.delenv('HDF5_VDS_PREFIX')
    with fastnc.File(str(tmp_path / 'vds_cross.h5')) as f:
        v = f['stack3'][()]
        assert (v == np.float32(-1.5)).all()                     # sources missing: fill value
