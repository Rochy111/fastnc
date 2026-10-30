"""The tests' fixtures: files opened once per session (`opened`), the
reference tests parametrized from the reference files (one test id per
file, variable or HDF5 object), and the opt-in `--gpu` option, which runs
the value comparisons again with device='gpu'."""
import pytest

from helpers import REFERENCE_SETS, locate, references


def pytest_addoption(parser):
    parser.addoption('--gpu', action='store_true', default=False,
                     help="also compare every read with fastnc.Dataset(path, device='gpu')")


def pytest_configure(config):
    config.addinivalue_line('markers', 'gpu: runs on the GPU (only with --gpu)')


# ------------------------------------------------------------ parametrization

def _ids(which, name, path=None):
    return '%s/%s' % (which, name) + ('' if path is None else ':' + path)


def _cases():
    """(fixture name -> [(id, value)]) for every reference set present."""
    out = {'nc_file': [], 'nc_var': [], 'h5_file': [], 'h5_obj': [], 'h5_dataset': []}
    for which in REFERENCE_SETS:
        refs = references(which)
        if not refs:
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
                _CASES = _cases()
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
        import os
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
