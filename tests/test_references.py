"""fastnc against what netCDF4-python and h5py read from the same files,
as recorded in tests/data/references_*.json.gz by tools/make_references.py.

Every netCDF variable is read raw (set_auto_maskandscale(False)) and as
netCDF4-python reads it by default, whole and for the standard
selections; every HDF5 dataset raw through fastnc.File; and every
metadata item (dimensions, attributes with their types, filters,
chunking, groups, h5py dataset properties) is compared. Values must be
bit-identical: the digests are sha256 of the values in native byte order.

The corpus set (tests/data) always runs; the synthetic corpus and the
real files skip when not on this machine. `--gpu` compares the netCDF
reads again with device='gpu'.
"""
import os

import numpy as np
import pytest

import helpers
from helpers import digest, encode, jsonable, key, references, same_values, selections, strip_repr

NUMPY1 = int(np.__version__.split('.')[0]) < 2


def expected(ref):
    """A reference digest for this numpy: netCDF4-python's answers differ
    in places between numpy 1 and 2 (the fill_value of a masked array);
    those fields are stored under 'numpy1'."""
    ref = dict(ref)
    over = ref.pop('numpy1', None)
    if NUMPY1 and over:
        ref.update(over)
    return ref


def to_host(x):
    """A GPU result as numpy: cupy arrays copied back; a masked result
    (data and mask on the card) as a numpy masked array."""
    if type(x).__module__.startswith('cupy'):
        return x.get()
    if hasattr(x, 'mask') and hasattr(x, 'data') and not isinstance(x, np.ndarray):
        data, mask = to_host(x.data), to_host(x.mask)
        return np.ma.MaskedArray(data, mask=mask, fill_value=getattr(x, 'fill_value', None))
    return x


def attrs_of(names, get, path):
    out = []
    for n in names:
        try:
            out.append([n, strip_repr(encode(get(n)))])
        except Exception as e:
            out.append([n, 'fastnc raised %s: %s' % (type(e).__name__, e)])
    return out


def ref_attrs(attrs):
    return [[n, strip_repr(v)] for n, v in attrs]


def compare(fields):
    """Assert that every (what, got, want) agrees; one message for all."""
    bad = ['%s: got %r, want %r' % (what, got, want) for what, got, want in fields if got != want]
    assert not bad, '\n'.join(bad)


def nc_reads(rec, mode):
    """(selection name, numpy key, reference digest) of a netCDF variable."""
    out = [('full', Ellipsis, rec[mode])]
    for s, d in rec['sel'].items():
        out.append((s, key(d['key']), d[mode]))
    return out


def check_reads(reads, read, device='cpu'):
    """Read each selection with `read(key)` and compare its digest to the
    reference. References that are a reader's error are skipped (the
    reference reader could not read it: there is nothing to compare);
    if all are, the test skips."""
    bad, skipped = [], []
    for s, k, ref in reads:
        if 'error' in ref:
            skipped.append('%s: %s' % (s, ref['error']))
            continue
        try:
            got = read(k)
        except Exception as e:
            bad.append('%s %r: fastnc raised %s: %s' % (s, k, type(e).__name__, e))
            continue
        if device == 'gpu':
            got = to_host(got)
        ok, msg = same_values(digest(got), expected(ref))
        if not ok:
            bad.append('%s %r: %s' % (s, k, msg))
    if skipped and len(skipped) == len(reads):
        pytest.skip('the reference reader failed: ' + skipped[0])
    assert not bad, '\n'.join(bad)


def nc_group(ds, gp):
    g = ds
    for part in [p for p in gp.split('/') if p]:
        g = g.groups[part]
    return g


# ------------------------------------------------------------ the references themselves (no fastnc needed)

def test_corpus_references_cover_tests_data():
    """Every file in tests/data has references, made from that very file
    (same size): regenerate them after tools/make_corpus.py."""
    refs = references('corpus')
    assert refs, 'tests/data/references_corpus.json.gz is missing: run tools/make_references.py --set corpus'
    files = sorted(f for f in os.listdir(helpers.DATA)
                   if not f.startswith('references_') and os.path.isfile(os.path.join(helpers.DATA, f)))
    assert sorted(refs['files']) == files
    for f in files:
        assert os.path.getsize(os.path.join(helpers.DATA, f)) == refs['files'][f]['size'], f


def test_digest_rules():
    """The digest compares values: byte order does not count, what lies
    under a mask does not count, compound padding does not count; types
    and dtypes do."""
    a = np.arange(12, dtype='<f4').reshape(3, 4)
    assert same_values(digest(a.astype('>f4')), digest(a))[0]
    assert not same_values(digest(a.astype('f8')), digest(a))[0]
    m1 = np.ma.masked_array(a, mask=a > 5, fill_value=np.float32(-1))
    m2 = np.ma.masked_array(np.where(a > 5, np.float32(99), a), mask=a > 5, fill_value=np.float32(-1))
    assert digest(m1) == digest(m2)
    assert digest(m1) != digest(np.ma.masked_array(a, mask=a > 5, fill_value=np.float32(-2)))
    assert digest(m1) != digest(np.ma.masked_array(a, mask=a > 6, fill_value=np.float32(-1)))
    assert digest(a) != digest(np.ma.masked_array(a))
    padded = np.dtype({'names': ['x', 'y'], 'formats': ['f4', 'i2'], 'offsets': [0, 4], 'itemsize': 8})
    p1, p2 = np.zeros(3, padded), np.ones(3, padded)
    p2['x'], p2['y'] = 0, 0
    p1.view('u1')[6::8] = 7                                  # padding bytes differ
    assert digest(p1) == digest(p2)


def test_encode_keeps_types():
    e = [strip_repr(encode(x)) for x in
         ('a', b'a', np.bytes_(b'a'), np.str_('a'), np.float32(1), np.float64(1), np.int16(1),
          np.array([1], 'f4'), ['a'], 1)]
    assert len({repr(x) for x in e}) == len(e)
    assert strip_repr(encode(np.float32(np.nan))) == strip_repr(encode(np.float32(np.nan)))


def test_selections_shapes():
    s = selections((5, 13, 17))
    assert set(s) == {'strided', 'index0', 'window'}
    x = np.zeros((5, 13, 17))
    assert x[key(s['strided'])].shape == (2, 5, 6)
    assert np.zeros((300, 400))[key(selections((300, 400))['strided'])].shape == (32, 32)
    assert x[key(s['index0'])].shape == (13, 17)
    assert x[key(s['window'])].shape == (1, 6, 8)
    assert selections(()) == {}


# ------------------------------------------------------------ netCDF (fastnc.Dataset vs netCDF4-python)

def test_nc_file(opened, nc_file):
    which, name = nc_file
    ref = references(which)['files'][name]['netcdf4']
    ds = opened.get('nc', which, name)
    fields = [('data_model', ds.data_model, ref['data_model']),
              ('file_format', ds.file_format, ref['file_format'])]
    for gp, g in ref['groups'].items():
        grp = nc_group(ds, gp)
        fields += [
            (gp + ' dimensions', [[n, len(d), d.isunlimited()] for n, d in grp.dimensions.items()],
             g['dimensions']),
            (gp + ' variables', list(grp.variables), g['variables']),
            (gp + ' groups', list(grp.groups), g['groups']),
            (gp + ' attributes', attrs_of(grp.ncattrs(), grp.getncattr, gp), ref_attrs(g['attrs'])),
        ]
    compare(fields)


def test_nc_variable(opened, nc_var):
    which, name, vp = nc_var
    ref = references(which)['files'][name]['netcdf4']['variables'][vp]
    ds = opened.get('nc', which, name)
    v = ds[vp.lstrip('/')]
    fields = [
        ('name', v.name, vp.rsplit('/', 1)[1]),
        ('dimensions', list(v.dimensions), ref['dimensions']),
        ('shape', list(v.shape), ref['shape']),
        ('dtype', helpers.dtype_str(v.dtype), ref['dtype']),
        ('ndim', v.ndim, ref['ndim']),
        ('size', int(v.size), ref['size']),
        ('group', v.group().path, ref['group']),
        ('attributes', attrs_of(v.ncattrs(), v.getncattr, vp), ref_attrs(ref['attrs'])),
    ]
    dt = ref['datatype']
    if isinstance(dt, str):
        fields.append(('datatype', helpers.dtype_str(v.datatype), dt))
    else:                                                    # compound / enum / vlen: name and base type
        fields.append(('datatype.name', getattr(v.datatype, 'name', None), dt['name']))
        if 'dtype' in dt:
            fields.append(('datatype.dtype', helpers.dtype_str(getattr(v.datatype, 'dtype', None)), dt['dtype']))
        if 'enum_dict' in dt:
            fields.append(('datatype.enum_dict', jsonable(getattr(v.datatype, 'enum_dict', None)),
                           dt['enum_dict']))
    for what in ('filters', 'chunking', 'endian'):
        r = ref[what]
        if isinstance(r, dict) and set(r) == {'error'}:
            continue
        try:
            got = jsonable(getattr(v, what)())
        except Exception as e:
            got = 'fastnc raised %s: %s' % (type(e).__name__, e)
        if what == 'filters' and isinstance(r, dict) and isinstance(got, dict):
            r, got = dict(r), dict(got)                      # the references are shared: never edit them
            fields +=plugin_filters(got, r, references(which)['files'][name].get('h5py'), vp)
        fields.append((what + '()', got, r))
    compare(fields)


PLUGIN_FILTERS = {'zstd': 32015, 'bzip2': 307, 'blosc': 32001}


def plugin_filters(got, ref, h5, vp):
    """netCDF4-python's filters() says False for zstd, bzip2 and blosc
    whenever netCDF-C cannot load that plugin, so those three are taken
    out of the comparison (from both dicts, in place) and checked against
    the filter pipeline h5py saw instead, when there is one."""
    fields = []
    pipeline = None
    if h5 and 'objects' in h5:
        o = h5['objects'].get(vp)
        if o and isinstance(o.get('filters'), list):
            pipeline = {f[0] for f in o['filters']}
    for k, fid in PLUGIN_FILTERS.items():
        ref.pop(k, None)
        g = got.pop(k, None)
        if pipeline is not None:
            fields.append(('filters()[%r] (h5py: filter %d in the pipeline)' % (k, fid), bool(g), fid in pipeline))
    return fields


def test_nc_values(opened, nc_var, mode, device):
    which, name, vp = nc_var
    rec = references(which)['files'][name]['netcdf4']['variables'][vp]
    ds = opened.get('nc', which, name, device)
    v = ds[vp.lstrip('/')]
    v.set_auto_maskandscale(mode == 'masked')
    try:
        check_reads(nc_reads(rec, mode), v.__getitem__, device)
    finally:
        v.set_auto_maskandscale(True)                         # the opened file is shared by later tests


# ------------------------------------------------------------ HDF5 (fastnc.File vs h5py)

H5_PROPS = ('chunks', 'compression', 'compression_opts', 'shuffle', 'fletcher32')
H5_OPTIONAL = ('maxshape', 'scaleoffset')                   # beyond the contract: compared when fastnc has them


def test_h5_object(opened, h5_obj):
    which, name, p = h5_obj
    ref = references(which)['files'][name]['h5py']['objects'][p]
    f = opened.get('h5', which, name)
    if ref['kind'] == 'error':
        pytest.skip('h5py could not open it: ' + ref['error'])
    obj = f[p] if p != '/' else f
    fields = []
    if ref.get('attrs') is not None and not isinstance(ref['attrs'], dict):
        fields.append(('attrs', attrs_of(list(obj.attrs.keys()), obj.attrs.__getitem__, p),
                       ref_attrs(ref['attrs'])))
    if ref['kind'] == 'group':
        fields.append(('keys', list(obj.keys()), ref['keys']))
        for link in ref['links']:
            if link[1] in ('soft', 'hard'):                  # a link reaches its target's object
                try:
                    got = getattr(obj[link[0]], 'shape', 'group')
                    tgt = f[link[2]]
                    want = getattr(tgt, 'shape', 'group')
                except Exception as e:
                    got, want = 'fastnc raised %s: %s' % (type(e).__name__, e), 'the target'
                fields.append(('link %s -> %s' % (link[0], link[2]), got, want))
    elif ref['kind'] == 'dataset':
        fields += [
            ('shape', None if obj.shape is None else list(obj.shape), ref['shape']),
            ('dtype', helpers.dtype_str(obj.dtype), ref['dtype']),
            ('dtype metadata', helpers.dtype_meta(obj.dtype), ref.get('dtype_meta')),
            ('fillvalue', strip_repr(encode(obj.fillvalue)), strip_repr(ref['fillvalue'])),
        ]
        fields += [(k, jsonable(getattr(obj, k)), ref[k]) for k in H5_PROPS]
        fields += [(k, jsonable(getattr(obj, k)), ref[k]) for k in H5_OPTIONAL
                   if hasattr(obj, k) and not isinstance(ref.get(k), dict)]
    elif ref['kind'] == 'datatype':
        fields.append(('dtype', helpers.dtype_str(obj.dtype), ref['dtype']))
    compare(fields)


def test_h5_values(opened, h5_dataset):
    which, name, p = h5_dataset
    ref = references(which)['files'][name]['h5py']['objects'][p]
    f = opened.get('h5', which, name)
    ds = f[p]
    reads = [('full', (), ref['raw'])]
    for s, d in ref['sel'].items():
        reads.append((s, Ellipsis if d['key'] == '...' else key(d['key']), d['raw']))
    check_reads(reads, ds.__getitem__)


def test_h5_read_direct(opened, h5_dataset):
    """read_direct(out) fills a preallocated array with the same values."""
    which, name, p = h5_dataset
    ref = references(which)['files'][name]['h5py']['objects'][p]
    raw = ref['raw']
    if 'error' in raw or not ref['shape'] or ref['dtype'] == '|O' or 'shape' not in raw:
        pytest.skip('nothing to read directly')
    f = opened.get('h5', which, name)
    ds = f[p]
    out = np.empty(tuple(ref['shape']), ds.dtype)
    if out.shape != tuple(ref['shape']):
        pytest.skip('an array type: numpy folds it into the shape')
    ds.read_direct(out)
    want = expected(raw)
    got = digest(out)
    ok, msg = same_values({k: got[k] for k in ('shape', 'dtype', 'sha256')},
                          {k: want[k] for k in ('shape', 'dtype', 'sha256')})
    assert ok, msg
