"""tests/data/codecs/jpeg.h5 and jpeg_expected.json: datasets compressed with
the HDF5 JPEG filter (32019, hdf5_plugins H5Zjpeg.c), and the SHA-1 of what
libjpeg (Pillow's libjpeg-turbo, with libjpeg's default decompression, as
the filter uses it) decodes each chunk into.

No JPEG plugin for HDF5 is at hand, so the filter is declared (optional, as
HDF5 lets a filter it has not loaded be) and each chunk written as it
would store it: one JPEG image of the chunk, made by Pillow (libjpeg's
jpeg_set_defaults with jpeg_set_quality, as the filter's own encoder), rows
of samples, RGB interleaved.

    python tools/check/jpeg_make.py
"""
import hashlib
import io
import json
import os

import h5py
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'data', 'codecs')
JPEG = 32019


def image(h, w, rgb, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1), (xx + yy) * 7 % 256], -1)
    a = np.clip(a + rng.integers(-25, 26, a.shape), 0, 255).astype(np.uint8)
    return a if rgb else a[..., 0].copy()


def main():
    p = os.path.join(OUT, 'jpeg.h5')
    expected = {}
    cases = [('gray', 48, 64, False, {}, (2, 1)), ('gray_odd', 37, 53, False, {}, (1, 1)),
             ('rgb_420', 48, 64, True, {}, (2, 2)), ('rgb_444', 40, 56, True, {'subsampling': 0}, (1, 2)),
             ('rgb_422', 33, 71, True, {'subsampling': 1}, (1, 1)),
             ('rgb_q10', 64, 64, True, {'quality': 10}, (1, 1)),
             ('rgb_q100', 16, 24, True, {'quality': 100}, (3, 1)),
             ('rgb_restart', 64, 80, True, {'restart_marker_blocks': 2}, (1, 1)),
             ('rgb_optimized', 50, 50, True, {'optimize': True}, (1, 1))]
    with h5py.File(p, 'w') as f:
        for name, h, w, rgb, kw, (nr, nc) in cases:
            shape = (h * nr, w * nc, 3) if rgb else (h * nr, w * nc)
            chunks = (h, w, 3) if rgb else (h, w)
            q = kw.get('quality', 75)
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            dcpl.set_chunk(chunks)
            dcpl.set_filter(JPEG, h5py.h5z.FLAG_OPTIONAL, (q, w, h, 1 if rgb else 0))
            sid = h5py.h5s.create_simple(shape)
            h5py.h5d.create(f.id, name.encode(), h5py.h5t.NATIVE_UINT8, sid, dcpl=dcpl)
            ds = f[name]
            d = {}
            for i in range(nr):
                for j in range(nc):
                    im = image(h, w, rgb, hash((name, i, j)) % 1000)
                    b = io.BytesIO()
                    Image.fromarray(im, 'RGB' if rgb else 'L').save(b, 'JPEG', quality=q,
                                                                   **{k: v for k, v in kw.items() if k != 'quality'})
                    raw = b.getvalue()
                    want = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB' if rgb else 'L')).tobytes()
                    off = (i * h, j * w, 0) if rgb else (i * h, j * w)
                    ds.id.write_direct_chunk(off, raw, 0)
                    d[','.join(map(str, off))] = [len(want), hashlib.sha1(want).hexdigest()]
            expected[name] = d
    with open(os.path.join(OUT, 'jpeg_expected.json'), 'w', newline='\n') as fh:
        json.dump(expected, fh, indent=0, sort_keys=True)
        fh.write('\n')
    print('wrote', p, os.path.getsize(p) // 1000, 'kB,', sum(len(d) for d in expected.values()), 'chunks')


if __name__ == '__main__':
    main()
