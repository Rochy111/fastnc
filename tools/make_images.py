"""Build the kernel images fastnc ships, one per x86-64 level, into
fastnc/_kernels (docs/internals.md, "Images shipped with fastnc").

    python tools/make_images.py                       # x86-64-v3 and x86-64
    python tools/make_images.py --levels x86-64-v2
    python tools/make_images.py --jobs 2              # levels side by side

The two levels by default are the ones that earn their bytes
(docs/measurements.md, round 15): x86-64-v3 runs as fast as an image built for
this very CPU; x86-64 runs on any x86-64, 3-18% slower on one thread, the
same on many; x86-64-v2 was 2-4% better than x86-64 only.

Each level compiles every kernel with numba told to compile for that level
and nothing more (NUMBA_CPU_NAME, NUMBA_CPU_FEATURES), in a numba cache of its
own under build/numba-<level>-<python>-<numpy>, so a second run of a level only links (about
20 s; a first one compiles, about 6 minutes). Runs at low priority. A level
this machine runs is tried in a process of its own before it is kept; one it
cannot run (x86-64-v4 without AVX-512) is built untried.

Images are for this system, this numpy's major version and these kernel
sources: the wheel for Windows is built on Windows, Linux's on Linux.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fastnc import _image  # noqa: E402

DEFAULT = ('x86-64-v3', 'x86-64')


def command(level, target=None):
    code = 'from fastnc import _image; _image.build_shipped(%r, %r)' % (level, target)
    # (numba's cache tells Python versions apart, not numpy's: one of its own each)
    cache = 'numba-%s-py%d.%d-np%s' % (level, *sys.version_info[:2], np.__version__.split('.')[0])
    env = dict(os.environ, NUMBA_CPU_NAME=level, NUMBA_CPU_FEATURES=_image.level_features(level),
               NUMBA_CACHE_DIR=os.path.join(ROOT, 'build', cache), FASTNC_IMAGE_BUILD='0')
    kw = {'creationflags': 0x00004000 | 0x08000000} if sys.platform == 'win32' else {}   # below normal, no window
    if sys.platform != 'win32':
        kw['preexec_fn'] = lambda: os.nice(10)
    return [sys.executable, '-c', code], dict(env=env, cwd=ROOT, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', default=','.join(DEFAULT))
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--out', default=None, help='a folder other than fastnc/_kernels')
    a = ap.parse_args()
    todo = a.levels.split(',')
    for level in todo:
        if level not in _image.LEVELS:
            sys.exit('no level %r: %s' % (level, ', '.join(_image.LEVELS)))
    print('this machine runs', ', '.join(_image.levels()))
    running = []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < a.jobs:
            level = todo.pop(0)
            target = a.out and os.path.join(a.out, _image.shipped_name(level))
            args, kw = command(level, target)
            running.append((level, time.time(), subprocess.Popen(args, **kw)))
        for item in running[:]:
            level, t, p = item
            if p.poll() is not None:
                running.remove(item)
                print('%-10s %s in %.0f s' % (level, 'built' if p.returncode == 0 else 'FAILED (%d)' % p.returncode,
                                              time.time() - t))
        time.sleep(0.5)
    print('%.0f s in all' % (time.time() - t0))


if __name__ == '__main__':
    main()
