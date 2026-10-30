"""Longer damaged-input run than tests/test_codecs.py: every compressed
chunk of the committed corpus, truncated at many points and with random
bytes changed, through the same padded / guarded harness (an error or some
output; never a write outside dst[o:oend] or the workspace, never a read
outside src[p:pend]).

    python tools/check/codecs_fuzz.py [trials per chunk] [seed]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

import test_codecs as T  # noqa: E402


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    t0 = time.time()
    n = 0
    for fname in T.FILES:
        for name, key, mask, buf, p, pend, fid, cd, (nb, sha) in T.chunks_of(fname):
            if mask & 1 or pend - p > 300_000:
                continue
            m = T.module(fid)
            size = pend - p
            for _ in range(trials):
                if rng.random() < 0.3:
                    T._run(m, buf, p, p + int(rng.integers(0, size)), cd, nb, nb, rng)
                else:
                    bad = buf[p:pend].copy()
                    for _ in range(int(rng.integers(1, 6))):
                        i = int(rng.integers(0, size)) if rng.random() < 0.6 else int(rng.integers(0, min(size, 64)))
                        bad[i] = rng.integers(0, 256)
                    room = nb if rng.random() < 0.8 else int(rng.integers(0, nb + 1))
                    T._run(m, bad, 0, size, cd, nb, room, rng)
                n += 1
        print('%-16s done, %d runs, %.0f s' % (fname, n, time.time() - t0), flush=True)
    print('no crash, no write outside, no read outside: %d runs' % n)


if __name__ == '__main__':
    main()
