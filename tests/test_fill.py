"""Fill values written over what a file never stored (HDF5 chunks never
written, HDF4 datasets with no data): every element's bytes exactly, a word
at a time where it can (_chunks.fill_with)."""
import numpy as np
import pytest

import fastnc
from fastnc import _chunks


@pytest.mark.parametrize('size', [1, 2, 3, 4, 8, 16])
def test_fill_with_every_byte(size):
    elem = bytes(range(0xEF, 0xEF + size))      # (every byte differs, none is zero)
    for n in (0, 1, 7, 100003):
        out = np.zeros(n * size, np.uint8)
        typed = out.view('u%d' % size) if size in (1, 2, 4, 8) else out.reshape(n, size)
        _chunks.fill_with(typed, elem)
        assert out.tobytes() == elem * n


def test_fill_with_keeps_nan_payloads():
    for dt, bits in (('f4', 0x7fa00001), ('f8', 0x7ff4000000000001)):
        out = np.empty((3, 5), dt)
        elem = np.array(bits, 'u%d' % out.itemsize).tobytes()
        _chunks.fill_with(out, elem)
        assert out.tobytes() == elem * 15


def test_unwritten_chunks_read_as_h5py_does(tmp_path):
    h5py = pytest.importorskip('h5py')
    p = str(tmp_path / 'unwritten.h5')
    with h5py.File(p, 'w') as f:
        for name, dt, fill in (('i2', 'i2', -999), ('be', '>u2', 7), ('f4', 'f4', np.nan),
                               ('be8', '>f8', -1.5), ('u1', 'u1', 3), ('s3', 'S3', b'abc')):
            f.create_dataset(name, (4, 300, 70), dt, chunks=(1, 128, 32), fillvalue=fill,
                             compression='gzip')
        f['part'] = np.zeros((4, 300, 70), 'i2')
        f.create_dataset('half', (4, 300, 70), '>i4', chunks=(1, 128, 32), fillvalue=-5)
        f['half'][:2] = 11
    with h5py.File(p, 'r') as h, fastnc.File(p) as a:
        for name in h:
            want = h[name][()]
            got = a[name][()]
            assert got.dtype == want.dtype.newbyteorder('=') or got.dtype == want.dtype
            assert got.tobytes() == want.astype(got.dtype).tobytes(), name
            box = (slice(1, 4, 2), slice(5, 290, 7), 3)
            assert a[name][box].tobytes() == h[name][box].astype(got.dtype).tobytes(), name
