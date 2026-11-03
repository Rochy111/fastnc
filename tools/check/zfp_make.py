"""Write tests/data/zfp: streams of every type, dimension and mode made by the real
H5Z-ZFP (through hdf5plugin), with the digests of what it decodes them to, and made.h5
for reads through fastnc.File, so that tests/test_zfp.py needs neither h5py nor the plugin.

    python tools/check/zfp_make.py

Everything is read back from a file closed and opened again: on the handle that wrote
them, HDF5's chunk cache hands back the values as written and the decoder never runs,
which for a lossy mode is not what a reader gets. Deterministic: fixed seeds.
"""
import hashlib
import json
import os

import h5py
import hdf5plugin
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'data', 'zfp')
MODES = {
    'rate': dict(rate=9.5), 'precision': dict(precision=14), 'accuracy': dict(accuracy=0.05),
    'reversible': dict(reversible=True), 'expert': dict(minbits=12, maxbits=500, maxprec=22, minexp=-9),
}
CHUNKS = {1: (22,), 2: (9, 6), 3: (5, 6, 7), 4: (3, 5, 4, 6)}      # partial blocks on every axis
TYPES = ('f4', 'f8', 'i4', 'i8', 'u4', 'u8')


def sha(a):
    a = np.asarray(a)
    return hashlib.sha256(a.astype(a.dtype.newbyteorder('=')).tobytes()).hexdigest()[:32]


def smooth(shape, rng):
    g = np.meshgrid(*[np.linspace(0, 2, n) for n in shape], indexing='ij')
    return sum(np.sin(x * (i + 1.3)) for i, x in enumerate(g)) * 900 + 2000 + rng.normal(0, 1, shape)


def special(t, shape, rng):
    """What a float can hold that plain data does not."""
    n = int(np.prod(shape))
    a = rng.normal(0, 1, n) * np.finfo(t).tiny * 10.0 ** rng.uniform(-6, 2, n)
    for v, m in ((np.nan, 9), (np.inf, 11), (-np.inf, 13), (-0.0, 7), (np.finfo(t).max, 17)):
        a[rng.integers(0, n, n // m + 1)] = v
    return a.astype(t).reshape(shape)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(32013)
    tmp = os.path.join(OUT, 'streams.tmp.h5')
    names = {}
    with h5py.File(tmp, 'w') as f:
        for d, chunk in CHUNKS.items():
            for t in TYPES:
                for mode, kw in MODES.items():
                    name = '%dd-%s-%s' % (d, t, mode)
                    f.create_dataset(name, data=smooth(chunk, rng).astype(t), chunks=chunk, **hdf5plugin.Zfp(**kw))
                    names[name] = t
        for t in ('f4', 'f8'):
            for mode, kw in MODES.items():
                name = 'special-%s-%s' % (t, mode)
                f.create_dataset(name, data=special(t, (8, 8), rng), chunks=(8, 8), **hdf5plugin.Zfp(**kw))
                names[name] = t
    streams = []
    source = None
    with h5py.File(tmp, 'r') as f:
        for name, t in names.items():
            d = f[name]
            fid, _, cd, source = d.id.get_create_plist().get_filter(0)
            streams.append({'name': name, 'dtype': t, 'chunk': list(d.chunks), 'cd': [int(x) for x in cd],
                            'hex': d.id.read_direct_chunk((0,) * d.ndim)[1].hex(), 'sha': sha(d[()])})
    os.remove(tmp)

    made = os.path.join(OUT, 'made.h5')
    x = smooth((48, 40), rng)
    with h5py.File(made, 'w') as f:
        for t, mode in (('>f4', 'rate'), ('>f8', 'reversible'), ('>i4', 'precision'), ('>i8', 'reversible')):
            f.create_dataset('be-%s-%s' % (t[1:], mode), data=x.astype(t), chunks=(16, 16), **hdf5plugin.Zfp(**MODES[mode]))
        f.create_dataset('many', data=x.astype('f4'), chunks=(7, 9), **hdf5plugin.Zfp(rate=12.0))
        f.create_dataset('fletcher', data=x.astype('f8'), chunks=(16, 16), fletcher32=True, **hdf5plugin.Zfp(rate=20.0))
        f.create_dataset('rank5', data=x[:8, :40].reshape(2, 1, 4, 1, 40).astype('f4'), chunks=(1, 1, 4, 1, 8),
                         **hdf5plugin.Zfp(accuracy=0.5))
        d = f.create_dataset('optional', shape=(32, 16), chunks=(16, 16), dtype='f4', **hdf5plugin.Zfp(rate=16.0))
        d[0:16, :] = x[:16, :16].astype('f4')
        d.id.write_direct_chunk((16, 0), np.arange(256, dtype='f4').tobytes(), filter_mask=1)  # (went without it)
        d = f.create_dataset('holes', shape=(32, 16), chunks=(16, 16), dtype='f8', fillvalue=2.5, **hdf5plugin.Zfp(rate=16.0))
        d[0:16, :] = x[:16, :16]
    files = {}
    with h5py.File(made, 'r') as f:
        for name in f:
            a = f[name][()]
            files[name] = {'shape': list(a.shape), 'dtype': a.dtype.newbyteorder('=').name, 'sha': sha(a)}
    with open(os.path.join(OUT, 'expected.json'), 'w') as fp:
        json.dump({'source': source.decode() if isinstance(source, bytes) else source,
                   'streams': streams, 'files': files}, fp, indent=0)
    size = os.path.getsize(os.path.join(OUT, 'expected.json'))
    print(len(streams), 'streams,', len(files), 'datasets,', size, 'bytes of json,', os.path.getsize(made), 'of made.h5')


if __name__ == '__main__':
    main()
