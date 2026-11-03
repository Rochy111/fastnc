"""The tests' fixtures: files opened once per session (`opened`), the
reference tests parametrized from the reference files (one test id per
file, variable or HDF5 object), and the opt-in `--gpu` option, which runs
the value comparisons again with device='gpu'. `--kernels off` runs
everything without numba's kernels (the paths a fresh process reads by). `--refs corpus` limits the
reference tests to tests/data (as CI runs them)."""
import os

import pytest

from helpers import REFERENCE_SETS, locate, references


def pytest_addoption(parser):
    parser.addoption('--gpu', action='store_true', default=False,
                     help="also compare every read with fastnc.Dataset(path, device='gpu')")
    parser.addoption('--kernels', default='on', choices=('on', 'numba', 'off'),
                     help="on: the kernels loaded first, from this machine's image if it has one "
                          "(default); numba: from numba, the image left alone; off: never loaded "
                          "-- every read as a process with no image does it (libdeflate / zlib, "
                          "numpy)")
    parser.addoption('--refs', default=','.join(REFERENCE_SETS),
                     help='reference sets to test, of %s (default: all whose files are here)'
                          % ','.join(REFERENCE_SETS))


def pytest_configure(config):
    config.addinivalue_line('markers', 'gpu: runs on the GPU (only with --gpu)')
    # (processes the tests start have caches of their own: each would start a
    # build of its own, minutes of compiling that no test wants)
    os.environ['FASTNC_IMAGE_BUILD'] = '0'
    from fastnc import _chunks
    if config.getoption('--kernels') != 'on':
        _chunks.IMAGE = False
    if config.getoption('--kernels') == 'off':
        _chunks.LOAD_AFTER = float('inf')
    else:
        _chunks.warm()
    if os.environ.get('FASTNC_NEP50_WARN'):                  # tools/run_tests_numpy1.py --nep50
        import numpy as np
        if hasattr(np, '_set_promotion_state'):              # numpy 1.24-1.26 only
            np._set_promotion_state('weak_and_warn')


# ------------------------------------------------------------ parametrization

def _ids(which, name, path=None):
    return '%s/%s' % (which, name) + ('' if path is None else ':' + path)


def sets_here():
    """{set: number of its files on this machine}; a set with none (the
    real files and the synthetic corpus, in CI) has no tests at all."""
    out = {}
    for which in REFERENCE_SETS:
        refs = references(which)
        out[which] = sum(locate(which, n) is not None for n in refs['files']) if refs else 0
    return out


def pytest_collection_modifyitems(config, items):
    """FASTNC_SPLIT=i/n: every n-th test from the i-th, for n processes (tools/check.py)."""
    split = os.environ.get('FASTNC_SPLIT')
    if split:
        i, n = (int(x) for x in split.split('/'))
        items[:] = items[i::n]


def pytest_report_header(config):
    n = sets_here()
    return 'fastnc references: ' + ', '.join(
        '%s %d/%d files' % (w, n[w], len(references(w)['files'])) if references(w) else '%s (none made)' % w
        for w in REFERENCE_SETS)


def _cases(wanted):
    """(fixture name -> [(id, value)]) for every reference set present."""
    out = {'nc_file': [], 'nc_var': [], 'h5_file': [], 'h5_obj': [], 'h5_dataset': []}
    here = sets_here()
    for which in REFERENCE_SETS:
        refs = references(which)
        if which not in wanted or not refs or not here[which]:
            continue
        for name, rec in refs['files'].items():
            nc, h5 = rec.get('netcdf4') or {}, rec.get('h5py') or {}
            if 'variables' in nc:
                out['nc_file'].append((_ids(which, name), (which, name)))
                for vp in nc['variables']:
                    out['nc_var'].append((_ids(which, name, vp), (which, name, vp)))
            if 'objects' in h5:
                out['h5_file'].append((_ids(which, name), (which, name)))
                for p, o in h5['objects'].items():
                    out['h5_obj'].append((_ids(which, name, p), (which, name, p)))
                    if o.get('kind') == 'dataset':
                        out['h5_dataset'].append((_ids(which, name, p), (which, name, p)))
    return out


_CASES = None


def pytest_generate_tests(metafunc):
    global _CASES
    for fx in ('nc_file', 'nc_var', 'h5_file', 'h5_obj', 'h5_dataset'):
        if fx in metafunc.fixturenames:
            if _CASES is None:
                _CASES = _cases(metafunc.config.getoption('--refs').split(','))
            cases = _CASES[fx]
            metafunc.parametrize(fx, [c[1] for c in cases], ids=[c[0] for c in cases])
    if 'mode' in metafunc.fixturenames:
        metafunc.parametrize('mode', ['raw', 'masked'])
    if 'device' in metafunc.fixturenames:
        devices = ['cpu'] + (['gpu'] if metafunc.config.getoption('--gpu') else [])
        metafunc.parametrize('device', devices)


# ------------------------------------------------------------ opened files

class Opened:
    """fastnc's files, each opened once per session and API (and device):
    a failure to open is remembered and reported by every test of that
    file, without retrying."""

    def __init__(self):
        self.cache = {}

    def path(self, which, name):
        p = locate(which, name)
        if p is None:
            pytest.skip('%s file %s is not on this machine' % (which, name))
        want = references(which)['files'][name]['size']
        if os.path.getsize(p) != want:
            pytest.skip('%s differs from the file the references were made from (%d bytes, not %d)'
                        % (name, os.path.getsize(p), want))
        return p

    def get(self, api, which, name, device='cpu'):
        k = (api, which, name, device)
        if k not in self.cache:
            path = self.path(which, name)
            if device == 'gpu':
                gpu_or_skip()
            try:
                import fastnc
                if api == 'nc':
                    obj = fastnc.Dataset(path) if device == 'cpu' else fastnc.Dataset(path, device='gpu')
                else:
                    obj = fastnc.File(path)
            except Exception as e:
                obj = e
            self.cache[k] = obj
        obj = self.cache[k]
        if isinstance(obj, Exception):
            pytest.fail('fastnc could not open %s (%s): %s: %s' % (name, api, type(obj).__name__, obj),
                        pytrace=False)
        return obj

    def close(self):
        for obj in self.cache.values():
            try:
                obj.close()
            except Exception:
                pass
        self.cache.clear()


@pytest.fixture(scope='session')
def opened():
    o = Opened()
    yield o
    o.close()


_GPU = None


def gpu_or_skip():
    """Skip unless cupy works and sees a GPU."""
    global _GPU
    if _GPU is None:
        try:
            import cupy
            _GPU = cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            _GPU = False
    if not _GPU:
        pytest.skip('no usable GPU (cupy)')
