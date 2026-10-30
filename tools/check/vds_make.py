"""Write the virtual dataset and external storage test files into
tests/data/vds/ with h5py, and record what h5py reads from them in
tests/data/vds/expected.json (tests/test_vds.py compares fastnc to it
without h5py).

    python tools/check/vds_make.py            # in the dev env (HDF5 1.14): the main set
    python tools/check/vds_make.py            # in an env with HDF5 2.x: the *_v2 files

HDF5 2.0 writes the mappings' heap in a newer encoding when it is shorter
(a flags byte per mapping, no "." for same-file sources): those files come
from the second run, and HDF5 1.14 cannot open them. Each run replaces the
expectations of the files it writes and keeps the others'.

The files: sources (chunked + compressed, big-endian, growable, a scalar,
a soft link, a subdirectory, printf-numbered blocks); virtual datasets
with same-file and cross-file sources, missing sources, fill values,
stacking, tiling, overlaps, strided and block (count/stride/block)
mappings, a VDS of a VDS, type conversion, selections of different shapes,
irregular selections (HDF5 refuses point selections in mappings), every hyperslab encoding version, values
needing 4-byte encodings, unlimited mappings (sources grown after the VDS
was made, interleaved, unlimited blocks, a partial last block, printf
names); external storage in two files, a short file, an unlimited segment.
"""
import json
import os
import sys

import h5py
import numpy as np
from h5py import h5s

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import helpers                                              # noqa: E402

OUT = os.path.join(ROOT, 'tests', 'data', 'vds')
EXPECTED = os.path.join(OUT, 'expected.json')
V2 = int(h5py.version.hdf5_version.split('.')[0]) >= 2
U = h5s.UNLIMITED


def field(shape, seed, dtype='f4'):
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.standard_normal(int(np.prod(shape))), dtype='f8').reshape(shape)
    return a.astype(dtype)


def space(shape, maxshape=None):
    return h5s.create_simple(tuple(shape), tuple(U if m is None else m for m in maxshape)
                             if maxshape else None)


def hyper(sp, start, count, stride=None, block=None, op=h5s.SELECT_SET):
    sp.select_hyperslab(tuple(start), tuple(count), tuple(stride) if stride else None,
                        tuple(block) if block else None, op=op)
    return sp


def make_virtual(group, name, shape, dtype, maps, maxshape=None, fill=None):
    """A VDS from (vspace, file name, dataset name, source space) mappings."""
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_layout(h5py.h5d.VIRTUAL)
    for vs, fname, dname, ss in maps:
        dcpl.set_virtual(vs, fname.encode(), dname.encode(), ss)
    if fill is not None:
        dcpl.set_fill_value(np.array([fill], dtype))
    tid = h5py.h5t.py_create(np.dtype(dtype), logical=1)
    h5py.h5d.create(group.id, name.encode(), tid, space(shape, maxshape), dcpl=dcpl)


def all_of(shape):
    sp = space(shape)
    sp.select_all()
    return sp


# ------------------------------------------------------------ the files (HDF5 1.x)

def sources():
    with h5py.File('src_a.h5', 'w') as f:
        f.create_dataset('chunked', data=field((24, 30), 1), chunks=(6, 8), compression='gzip',
                         shuffle=True)
        f.create_dataset('big_endian', data=np.arange(60, dtype='>i2') * 37 - 900)
        f.create_dataset('ints', data=np.arange(256, dtype='<i4').reshape(16, 16))
        f.create_dataset('grow', data=np.arange(24, dtype='i4').reshape(4, 6), chunks=(2, 6),
                         maxshape=(None, 6))
        f.create_dataset('grow2', data=np.arange(18, dtype='i4').reshape(3, 6) + 1000, chunks=(2, 6),
                         maxshape=(None, 6))
        f.create_dataset('scalar', data=np.float64(2.718281828))
        f.create_dataset('group/deep', data=np.arange(10, dtype='u1') * 3)
        f['link_to_deep'] = h5py.SoftLink('/group/deep')
    with h5py.File('src_b.h5', 'w', libver='latest') as f:
        f.create_dataset('data', data=field((12, 30), 2), chunks=(4, 10), compression='lzf')
    os.makedirs('sub', exist_ok=True)
    with h5py.File(os.path.join('sub', 'src_c.h5'), 'w') as f:
        f.create_dataset('data', data=field((24, 30), 3), chunks=(24, 30), compression='gzip',
                         fletcher32=True)
    for i in range(3):
        with h5py.File('blk_%d.h5' % i, 'w') as f:
            f.create_dataset('x', data=(np.arange(12, dtype='i2') + 100 * i).reshape(3, 4))


def grow_sources():
    """Sources grown after the unlimited virtual datasets were made."""
    with h5py.File('src_a.h5', 'a') as f:
        g = f['grow']
        g.resize((7, 6))
        g[4:] = np.arange(18, dtype='i4').reshape(3, 6) + 500
        g = f['grow2']
        g.resize((5, 6))
        g[3:] = np.arange(12, dtype='i4').reshape(2, 6) + 2000


def vds_same():
    with h5py.File('vds_same.h5', 'w') as f:
        a = f.create_dataset('a', data=np.arange(120, dtype='i4').reshape(10, 12))
        b = f.create_dataset('b', data=np.arange(120, dtype='i4').reshape(10, 12) * -3, chunks=(5, 4),
                             compression='gzip')
        lay = h5py.VirtualLayout((2, 10, 12), 'i4')
        lay[0] = h5py.VirtualSource(a)
        lay[1] = h5py.VirtualSource(b)
        f.create_virtual_dataset('stack', lay, fillvalue=-7)
        lay = h5py.VirtualLayout((3, 10, 12), 'i4')
        lay[1] = h5py.VirtualSource(a)
        f.create_virtual_dataset('partial', lay, fillvalue=99)
        lay = h5py.VirtualLayout((20, 24), 'i4')
        lay[0:10, 0:12] = h5py.VirtualSource(a)
        lay[0:10, 12:24] = h5py.VirtualSource(b)
        lay[10:20, 12:24] = h5py.VirtualSource(a)
        f.create_virtual_dataset('tiles', lay)                  # one tile unmapped: fill 0
        lay = h5py.VirtualLayout((10, 12), 'i4')
        lay[:, :] = h5py.VirtualSource(a)
        lay[2:5, 3:9] = h5py.VirtualSource(b)[0:3, 0:6]
        f.create_virtual_dataset('overlap', lay)                # the later mapping wins
        lay = h5py.VirtualLayout((5, 6), 'i4')
        lay[:, :] = h5py.VirtualSource(a)[::2, ::2]
        f.create_virtual_dataset('sub', lay)


def vds_cross():
    with h5py.File('vds_cross.h5', 'w') as f:
        lay = h5py.VirtualLayout((3, 24, 30), 'f4')
        lay[0] = h5py.VirtualSource('src_a.h5', 'chunked', shape=(24, 30))
        lay[1] = h5py.VirtualSource('sub/src_c.h5', '/data', shape=(24, 30))
        lay[2] = h5py.VirtualSource('nonexistent.h5', 'data', shape=(24, 30))
        f.create_virtual_dataset('stack3', lay, fillvalue=-1.5)
        lay = h5py.VirtualLayout((24, 30), 'f4')
        lay[::2, :] = h5py.VirtualSource('src_b.h5', 'data', shape=(12, 30))
        f.create_virtual_dataset('strided', lay, fillvalue=np.nan)
        lay = h5py.VirtualLayout((20, 30), 'f4')
        lay[h5py.MultiBlockSlice(start=1, stride=5, count=4, block=3), :] = \
            h5py.VirtualSource('src_b.h5', 'data', shape=(12, 30))
        f.create_virtual_dataset('blocks', lay)
        lay = h5py.VirtualLayout((10, 30), 'f4')
        lay[:, ::3] = h5py.VirtualSource('src_b.h5', 'data', shape=(12, 30))[
            h5py.MultiBlockSlice(start=0, stride=6, count=2, block=5), 0:10]
        f.create_virtual_dataset('blocks_both', lay)
        lay = h5py.VirtualLayout((6, 10), 'i4')
        lay[:, :] = h5py.VirtualSource('src_a.h5', 'big_endian', shape=(60,))
        f.create_virtual_dataset('reshape', lay)                # >i2 -> i4, (60,) -> (6, 10)
        lay = h5py.VirtualLayout((12,), 'u1')
        lay[1:11] = h5py.VirtualSource('src_a.h5', '/link_to_deep', shape=(10,))
        f.create_virtual_dataset('deep', lay, fillvalue=255)
        lay = h5py.VirtualLayout((3,), 'f8')
        lay[1] = h5py.VirtualSource('src_a.h5', 'scalar', shape=())
        f.create_virtual_dataset('scalar_map', lay)
        lay = h5py.VirtualLayout((2, 10, 12), 'i4')
        lay[:, :, :] = h5py.VirtualSource('vds_same.h5', 'stack', shape=(2, 10, 12))
        f.create_virtual_dataset('nested', lay)                 # a VDS of a VDS
        lay = h5py.VirtualLayout((4, 4), 'f4')
        lay[:, :] = h5py.VirtualSource('src_a.h5', 'chunked', shape=(24, 30))[3:11:2, 20:24]
        f.create_virtual_dataset('src_strided', lay)


def vds_lowlevel(libver):
    name = 'vds_lowlevel_%s.h5' % libver
    with h5py.File(name, 'w', libver=libver) as f:
        vs = hyper(space((12, 12)), (0, 0), (1, 1), block=(4, 4))
        hyper(vs, (2, 2), (1, 1), block=(4, 6), op=h5s.SELECT_OR)
        ss = hyper(space((16, 16)), (0, 0), (1, 1), block=(6, 6))
        make_virtual(f, 'irregular', (12, 12), 'i4', [(vs, 'src_a.h5', 'ints', ss)])
        vs = hyper(space((2, 70000)), (1, 69990), (1, 1), block=(1, 10))     # 4-byte encoding
        ss = hyper(space((16, 16)), (5, 3), (1, 1), block=(1, 10))
        vs2 = hyper(space((2, 70000)), (0, 3), (2, 3), stride=(1, 30000), block=(1, 2))
        ss2 = hyper(space((16, 16)), (0, 0), (1, 1), block=(2, 6))
        make_virtual(f, 'wide', (2, 70000), 'i4', [(vs, 'src_a.h5', 'ints', ss),
                                                   (vs2, 'src_a.h5', 'ints', ss2)], fill=-2)


def vds_unlim(libver):
    name = 'vds_unlim_%s.h5' % libver
    with h5py.File(name, 'w', libver=libver) as f:
        g = (4, 6)
        vs = hyper(space((4, 6), (None, 6)), (0, 0), (U, 1), block=(1, 6))
        ss = hyper(space(g, (None, 6)), (0, 0), (U, 1), block=(1, 6))
        make_virtual(f, 'rows', (4, 6), 'i4', [(vs, 'src_a.h5', 'grow', ss)], maxshape=(None, 6))
        vs = hyper(space((7, 6), (None, 6)), (0, 0), (U, 1), stride=(2, 1), block=(1, 6))
        ss = hyper(space(g, (None, 6)), (0, 0), (U, 1), block=(1, 6))
        vs2 = hyper(space((7, 6), (None, 6)), (1, 0), (U, 1), stride=(2, 1), block=(1, 6))
        ss2 = hyper(space((3, 6), (None, 6)), (0, 0), (U, 1), block=(1, 6))
        make_virtual(f, 'interleave', (7, 6), 'i4',
                     [(vs, 'src_a.h5', 'grow', ss), (vs2, 'src_a.h5', 'grow2', ss2)],
                     maxshape=(None, 6), fill=-9)
        vs = hyper(space((6, 6), (None, 6)), (2, 0), (1, 1), block=(U, 6))
        ss = hyper(space(g, (None, 6)), (0, 0), (1, 1), block=(U, 6))
        make_virtual(f, 'block_unlim', (6, 6), 'i4', [(vs, 'src_a.h5', 'grow', ss)], maxshape=(None, 6))
        vs = hyper(space((3, 6), (None, 6)), (0, 0), (U, 1), stride=(4, 1), block=(3, 6))
        ss = hyper(space((3, 6), (None, 6)), (0, 0), (U, 1), block=(1, 6))
        make_virtual(f, 'partial_block', (3, 6), 'i4', [(vs, 'src_a.h5', 'grow2', ss)],
                     maxshape=(None, 6), fill=-5)
        vs = hyper(space((0, 4), (None, 4)), (0, 0), (U, 1), stride=(3, 1), block=(3, 4))
        make_virtual(f, 'printf', (0, 4), 'i2', [(vs, 'blk_%b.h5', 'x', all_of((3, 4)))],
                     maxshape=(None, 4), fill=7)
        vs = hyper(space((11, 6), (None, 6)), (0, 0), (U, 1), block=(1, 6))
        ss = hyper(space(g, (None, 6)), (0, 0), (U, 1), block=(1, 6))
        vfix = hyper(space((11, 6), (None, 6)), (9, 0), (1, 1), block=(2, 6))
        make_virtual(f, 'min_dims', (11, 6), 'i4',
                     [(vs, 'src_a.h5', 'grow', ss), (vfix, 'src_a.h5', 'ints', hyper(
                         space((16, 16)), (0, 0), (1, 1), block=(2, 6)))], maxshape=(None, 6))


def external():
    a = np.arange(256, dtype='u1').tobytes()
    with open('ext_a.bin', 'wb') as fh:
        fh.write(a)
    vals = field((6, 5), 4, '>f4').tobytes()
    with open('ext_a.bin', 'r+b') as fh:
        fh.seek(100)
        fh.write(vals[:48])
    with open('ext_b.bin', 'wb') as fh:
        fh.write(vals[48:] + b'tail')
    with open('ext_c.bin', 'wb') as fh:
        fh.write(np.arange(5, dtype='<i8').tobytes())       # 40 bytes for a dataset of 80
    with h5py.File('ext.h5', 'w') as f:
        f.create_dataset('ext2', shape=(6, 5), dtype='>f4',
                         external=[('ext_a.bin', 100, 48), ('ext_b.bin', 0, 72)])
        f.create_dataset('ext_short', shape=(10,), dtype='<i8', external=[('ext_c.bin', 0, 80)])
        f.create_dataset('ext_unlim', shape=(8,), dtype='<u2', external=[('ext_a.bin', 3, U)])


# ------------------------------------------------------------ HDF5 2.x

def vds_v2():
    with h5py.File('vds_v2.h5', 'w', libver='latest') as f:
        a = f.create_dataset('a', data=np.arange(120, dtype='f8').reshape(10, 12) / 7)
        b = f.create_dataset('b', data=np.arange(120, dtype='f8').reshape(10, 12) * -1, chunks=(5, 6),
                             compression='gzip')
        lay = h5py.VirtualLayout((4, 10, 12), 'f8')
        lay[0] = h5py.VirtualSource(a)
        lay[1] = h5py.VirtualSource(b)
        lay[2] = h5py.VirtualSource(a)
        lay[3] = h5py.VirtualSource('src_a.h5', 'ints', shape=(16, 16))[0:10, 0:12]
        f.create_virtual_dataset('mixed', lay, fillvalue=0.5)
        lay = h5py.VirtualLayout((2, 10, 12), 'f8')
        lay[0] = h5py.VirtualSource(b)
        lay[1, 2:8] = h5py.VirtualSource(a)[1:7]
        f.create_virtual_dataset('same', lay, fillvalue=-0.25)


# ------------------------------------------------------------ expectations

def space_info(sp):
    """What a selection says of itself (h5py SpaceID or fastnc's stand-in)."""
    out = {'type': int(sp.get_select_type()), 'shape': [int(x) for x in sp.shape]}
    for what, fn in (('npoints', lambda: int(sp.get_select_npoints())),
                     ('regular', lambda: sp.is_regular_hyperslab() and
                      [[int(x) for x in t] for t in sp.get_regular_hyperslab()]),
                     ('bounds', lambda: [[int(x) for x in t] for t in sp.get_select_bounds()]),
                     ('points', lambda: [[int(x) for x in p] for p in sp.get_select_elem_pointlist()]
                      if sp.get_select_type() == 1 else None)):
        try:
            out[what] = fn()
        except Exception as e:
            out[what] = 'error'
    return out


def describe(ds):
    rec = {'shape': list(ds.shape), 'maxshape': [None if m is None else int(m) for m in ds.maxshape],
           'dtype': helpers.dtype_str(ds.dtype), 'fillvalue': helpers.encode(ds.fillvalue),
           'is_virtual': bool(ds.is_virtual), 'chunks': ds.chunks,
           'external': [list(x) for x in ds.external] if ds.external else None}
    if ds.is_virtual:
        rec['sources'] = [[m.file_name, m.dset_name, space_info(m.vspace), space_info(m.src_space)]
                          for m in ds.virtual_sources()]
    sels = dict(helpers.selections(ds.shape))
    if ds.shape and all(ds.shape):
        sels['cross'] = [['s', n // 4, n - n // 5, 3] for n in ds.shape]
        sels['last'] = [['i', ds.shape[0] - 1]] + [['s', None, None, 2] for _ in ds.shape[1:]]
    reads = {'full': {'key': None, 'digest': helpers.digest(ds[()])}}
    for k, sel in sels.items():
        reads[k] = {'key': sel, 'digest': helpers.digest(ds[helpers.key(sel)])}
    rec['reads'] = reads
    return rec


def record(files):
    exp = {}
    if os.path.exists(EXPECTED):
        with open(EXPECTED) as fh:
            exp = json.load(fh)
    exp.setdefault('files', {})
    exp.setdefault('hdf5', {})
    for name in files:
        recs = {}
        with h5py.File(name, 'r') as f:
            def visit(path, obj):
                if isinstance(obj, h5py.Dataset) and (obj.is_virtual or obj.external):
                    recs['/' + path] = describe(obj)
            f.visititems(visit)
        exp['files'][name] = recs
        exp['hdf5'][name] = h5py.version.hdf5_version
    with open(EXPECTED, 'w') as fh:
        json.dump(exp, fh, indent=1, sort_keys=True)
        fh.write('\n')


def main():
    os.makedirs(OUT, exist_ok=True)
    os.chdir(OUT)                       # external files and VDS sources resolve from here for h5py
    if V2:
        vds_v2()
        record(['vds_v2.h5'])
        return
    sources()
    vds_same()
    vds_cross()
    for libver in ('earliest', 'latest'):
        vds_lowlevel(libver)
        vds_unlim(libver)
    grow_sources()
    external()
    record(['vds_same.h5', 'vds_cross.h5', 'vds_lowlevel_earliest.h5', 'vds_lowlevel_latest.h5',
            'vds_unlim_earliest.h5', 'vds_unlim_latest.h5', 'ext.h5'])


if __name__ == '__main__':
    main()
