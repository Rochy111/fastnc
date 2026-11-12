"""tests/data/codecs/blosc2_dict_vl.npz: blosc2 chunks hdf5plugin's filter
does not write -- chunks compressed with a dictionary (use_dict: zstd, lz4,
lz4hc) and chunks of variable-length blocks (blosc2_vlcompress), with and
without one -- and blosc2 frames holding them, as the HDF5 filter (32026)
stores a chunk; with the size and SHA-1 of what c-blosc2 itself decodes
each into, so tests/test_codecs.py checks fastnc against it without
python-blosc2.

    pip install blosc2     (4.x: c-blosc2 3.x)
    python tools/check/blosc2_dict_make.py
"""
import hashlib
import os

import blosc2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'data', 'codecs', 'blosc2_dict_vl.npz')
WORDS = [b'temperature', b'pressure', b'humidity', b'wind_u', b'wind_v', b'cloud', b'rain', b'snow']


def main():
    rng = np.random.default_rng(7)

    def text(n):
        return b' '.join(WORDS[i] for i in rng.integers(0, len(WORDS), n))

    arrs = {
        'f4': (np.sin(np.arange(8000) / 300.0) * 1000).astype('<f4'),
        'i2': (np.arange(12000) % 1000 - 500).astype('<i2'),
        'u1': np.frombuffer(text(4000)[:20000], np.uint8),
        'f8': np.round(np.cos(np.arange(4000) / 70.0) * 1e4, 1).astype('<f8'),
    }
    out = {}

    def keep(key, c, d):
        out[key + '.c'] = np.frombuffer(c, np.uint8)
        out[key + '.n'] = np.array(len(d))
        out[key + '.sha1'] = np.array(hashlib.sha1(d).hexdigest())

    for codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4, blosc2.Codec.LZ4HC):
        for name, a in arrs.items():
            for flt in (blosc2.Filter.SHUFFLE, blosc2.Filter.BITSHUFFLE, blosc2.Filter.NOFILTER):
                if flt == blosc2.Filter.NOFILTER and name != 'u1':
                    continue                            # (floats unshuffled: big, and nothing new)
                for bs in (0, 8192):
                    cp = blosc2.CParams(codec=codec, clevel=5, use_dict=True, typesize=a.itemsize,
                                        filters=[flt], filters_meta=[0], blocksize=bs)
                    c = blosc2.compress2(a.tobytes(), cparams=cp)
                    d = blosc2.decompress2(c)
                    assert d == a.tobytes()
                    keep('dict_%s_%s_%s_%d' % (codec.name, name, flt.name, bs), c, d)
    for codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4, blosc2.Codec.LZ4HC, blosc2.Codec.BLOSCLZ,
                  blosc2.Codec.ZLIB):
        for use_dict in (False, True):
            if use_dict and codec in (blosc2.Codec.BLOSCLZ, blosc2.Codec.ZLIB):
                continue
            blocks = [text(int(n)) for n in rng.integers(20, 400, 12)] + [rng.bytes(1500), b'x']
            c = blosc2.blosc2_ext.vlcompress(blocks, codec=codec, clevel=5, typesize=1, use_dict=use_dict)
            assert list(blosc2.blosc2_ext.vldecompress(c)) == blocks
            d = blosc2.decompress2(c)
            assert d == b''.join(blocks)
            key = 'vl_%s_dict%d' % (codec.name, use_dict)
            keep(key, c, d)
            # the same chunk in a frame, as the HDF5 filter keeps a 1-d chunk
            s = blosc2.SChunk(chunksize=len(d), cparams=blosc2.CParams(typesize=1))
            s.append_chunk(c)
            keep(key + '_frame', s.to_cframe(), s.decompress_chunk(0))
    for codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4):
        a = arrs['f4']
        cp = blosc2.CParams(codec=codec, clevel=5, use_dict=True, typesize=4)
        s = blosc2.SChunk(chunksize=a.nbytes, data=a.tobytes(), cparams=cp)
        keep('dict_%s_frame' % codec.name, s.to_cframe(), s.decompress_chunk(0))
    np.savez_compressed(OUT, **out)
    n = sum(1 for k in out if k.endswith('.c'))
    print('wrote', OUT, n, 'chunks and frames,', os.path.getsize(OUT) // 1000, 'kB')


if __name__ == '__main__':
    main()
