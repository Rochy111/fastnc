"""Long fuzz of fastnc._pinflate against the serial fastnc._inflate:
corrupted / truncated / spliced zlib streams of every kind, cut into
tiny parts. A difference in output or success is a failure; a different
(negative) error code is counted. Nothing may be written outside the
output range.

    python tools/check/pinflate_fuzz.py [iterations] [seed]
"""
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from fastnc import _inflate, _pinflate  # noqa: E402

GUARD = 64


def data(rng, n):
    k = int(rng.integers(0, 5))
    if k == 0:
        return rng.integers(0, 256, n, dtype=np.uint8).tobytes()
    if k == 1:
        return np.cumsum(rng.integers(-3, 4, n // 2 + 1)).astype('<i2').tobytes()[:n]
    if k == 2:
        b = rng.integers(0, 256, 32768, dtype=np.uint8).tobytes()
        return (b * (n // 32768 + 1))[:n]
    if k == 3:
        return bytes(n)
    w = [b'fill', b'65535', b'\n', b'ab', b'radiance ']
    return b''.join(w[i] for i in rng.integers(0, len(w), n // 3 + 1))[:n]


def corrupt(rng, b):
    b = bytearray(b)
    for _ in range(int(rng.integers(1, 3))):
        how = int(rng.integers(0, 5))
        if not b:
            break
        if how == 0:
            i = int(rng.integers(0, len(b)))
            b[i] ^= 1 << int(rng.integers(0, 8))
        elif how == 1:
            b[int(rng.integers(0, len(b)))] = int(rng.integers(0, 256))
        elif how == 2:
            b = b[:int(rng.integers(0, len(b)))]
        elif how == 3:
            i = int(rng.integers(0, len(b)))
            k = int(rng.integers(1, 300))
            b[i:i + k] = rng.integers(0, 256, k, dtype=np.uint8).tobytes()
        else:
            i, j = sorted(int(x) for x in rng.integers(0, len(b), 2))
            k = int(rng.integers(1, 20000))
            b[i:i + k] = bytes(b[j:j + k])
    return bytes(b)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    tab, iws = _inflate.workspace()
    stats = dict(same_ok=0, same_err=0, code_differs=0, fail=0)
    for it in range(iters):
        raw = data(rng, int(rng.integers(1, 400_000)))
        c = zlib.compressobj(int(rng.integers(0, 10)), zlib.DEFLATED, int(rng.integers(9, 16)),
                             int(rng.integers(1, 10)),
                             [zlib.Z_DEFAULT_STRATEGY, zlib.Z_RLE, zlib.Z_HUFFMAN_ONLY,
                              zlib.Z_FIXED, zlib.Z_FILTERED][int(rng.integers(0, 5))])
        comp = corrupt(rng, c.compress(raw) + c.flush())
        src = np.frombuffer(comp, np.uint8)
        room = len(raw) + int(rng.integers(-1000, 1000)) if rng.random() < 0.5 else 4 * len(raw) + 1000
        room = max(room, 1)
        oend = GUARD + room
        ostop = oend if rng.random() < 0.7 else GUARD + int(rng.integers(0, room + 1))
        a = np.full(oend + GUARD, 0xA5, np.uint8)
        b = np.full(oend + GUARD, 0xA5, np.uint8)
        rs = _inflate.zlib_inflate(src, 0, len(comp), a, GUARD, oend, ostop, tab, iws)
        rp = _pinflate.inflate_parallel(src, 0, len(comp), b, GUARD, oend, ostop, nthreads=8,
                                        min_part=int(rng.choice([1 << 10, 1 << 12, 1 << 15])),
                                        parts_per_thread=4)
        bad = np.any(b[:GUARD] != 0xA5) or np.any(b[oend:] != 0xA5)
        if rs >= 0 and ostop == oend:
            bad |= rp != rs or a[GUARD:rs].tobytes() != b[GUARD:rp].tobytes()
        elif rs >= 0:
            k = min(rs, rp) - GUARD
            bad |= rp < min(ostop, rs) or a[GUARD:GUARD + k].tobytes() != b[GUARD:GUARD + k].tobytes()
        elif ostop == oend:
            bad |= rp >= 0
        if bad:
            stats['fail'] += 1
            print('FAIL it=%d serial %d parallel %d (len %d, ostop %d, oend %d)'
                  % (it, rs, rp, len(comp), ostop, oend))
        elif rs >= 0:
            stats['same_ok'] += 1
        elif rp == rs:
            stats['same_err'] += 1
        elif rp < 0:
            stats['code_differs'] += 1
    print(stats)
    return stats['fail']


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
