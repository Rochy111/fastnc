"""A read that proves as it decodes that no value is masked, and so spares the
pass over them (_maskscale.Probe), gives what netCDF4-python gives, bit for
bit: values that are masked, values that only look it (the top byte of a fill
value, -0.0 against 0.0), and everything the test cannot cover. Each case also
says whether a proof should come out, so none can pass by falling back."""
import glob
import os

import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _dataset, _maskscale, _nc3, _pool, _readahead
from helpers import digest, same_values

netCDF4 = pytest.importorskip('netCDF4')

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SHAPE = (6, 40, 50)
CHUNKS = (2, 20, 25)
LAST = (-1, -1, -1)
F4 = float(netCDF4.default_fillvals['f4'])


def bits(u, dt):
    """The value of type dt with the bits of the integer u."""
    return np.array([u], 'u%d' % np.dtype(dt).itemsize).view(dt)[0]


# name: dtype, proved, then: planted (index, value), create (createVariable
# arguments), attrs, part (only the first records written)
FLOAT_CASES = {
    'clean': dict(dt='f4', proved=True),
    'default_hit': dict(dt='f4', proved=False, planted=(LAST, F4)),
    'default_first': dict(dt='f4', proved=False, planted=((0, 0, 0), F4)),
    'top_byte_only': dict(dt='f4', proved=False, planted=(LAST, bits(0x7C000001, 'f4'))),
    'fill_clean': dict(dt='f4', proved=True, create=dict(fill_value=-999.0)),
    'fill_hit': dict(dt='f4', proved=False, create=dict(fill_value=-999.0), planted=(LAST, -999.0)),
    'two_clean': dict(dt='f4', proved=True, create=dict(fill_value=-999.0),
                      attrs=dict(missing_value=np.float32(-888.0))),
    'two_second_hit': dict(dt='f4', proved=False, create=dict(fill_value=-999.0),
                           attrs=dict(missing_value=np.float32(-888.0)), planted=(LAST, -888.0)),
    'two_tops_second_hit': dict(dt='f4', proved=False, create=dict(fill_value=-999.0),
                                attrs=dict(missing_value=np.float32(777.0)), planted=(LAST, 777.0)),
    'three_values': dict(dt='f4', proved=False, attrs=dict(
        missing_value=np.array([-1.0, -2.0, -3.0], 'f4')), planted=(LAST, -2.0)),
    'three_values_clean': dict(dt='f4', proved=False, attrs=dict(
        missing_value=np.array([-1.0, -2.0, -3.0], 'f4'))),
    'zero_fill': dict(dt='f4', proved=False, create=dict(fill_value=0.0), planted=(LAST, -0.0)),
    'nan_fill': dict(dt='f4', proved=False, create=dict(fill_value=np.nan), planted=(LAST, np.nan)),
    'nan_in_data': dict(dt='f4', proved=True, planted=(LAST, np.nan)),
    'inf_in_data': dict(dt='f4', proved=True, planted=(LAST, np.inf)),
    'valid_min': dict(dt='f4', proved=False, attrs=dict(valid_min=np.float32(0.0)),
                      planted=(LAST, -5.0)),
    'valid_range_clean': dict(dt='f4', proved=False,
                              attrs=dict(valid_range=np.array([0.0, 1000.0], 'f4'))),
    'big_endian_clean': dict(dt='f4', proved=True, create=dict(endian='big')),
    'big_endian_hit': dict(dt='f4', proved=False, create=dict(endian='big'), planted=(LAST, F4)),
    'f8_clean': dict(dt='f8', proved=True),
    'f8_hit': dict(dt='f8', proved=False, planted=(LAST, netCDF4.default_fillvals['f8'])),
    'scaled': dict(dt='i2', proved=False, attrs=dict(scale_factor=np.float32(0.5))),
    'unsigned': dict(dt='i2', proved=False, attrs=dict(_Unsigned='true'),
                     planted=(LAST, -1)),
    # (netCDF-C's checksum lies under its shuffle: a chunk may go without either,
    # so these are read stage by stage, and the pass is not spared)
    'fletcher_clean': dict(dt='f4', proved=False, create=dict(fletcher32=True)),
    'fletcher_hit': dict(dt='f4', proved=False, create=dict(fletcher32=True), planted=(LAST, F4)),
    'unshuffled': dict(dt='f4', proved=False, create=dict(shuffle=False), planted=(LAST, F4)),
    'unshuffled_clean': dict(dt='f4', proved=False, create=dict(shuffle=False)),
    'unwritten': dict(dt='f4', proved=False, part=True),
    'unwritten_fill': dict(dt='i4', proved=False, part=True, create=dict(fill_value=-5)),
    # chunks reaching past the variable's edge: fill values there, and only there
    'past_edge': dict(dt='f4', proved=True, create=dict(chunksizes=(4, 30, 30))),
    'past_edge_hit': dict(dt='f4', proved=False, create=dict(chunksizes=(4, 30, 30)),
                          planted=(LAST, F4)),
    'past_edge_inside': dict(dt='f4', proved=False, create=dict(chunksizes=(4, 30, 30)),
                             planted=((1, 29, 29), F4)),
    'past_edge_top_only': dict(dt='f4', proved=False, create=dict(chunksizes=(4, 30, 30)),
                               planted=(LAST, bits(0x7C000001, 'f4'))),
    'past_last_dim': dict(dt='f4', proved=True, create=dict(chunksizes=(2, 40, 30))),
    'past_last_dim_hit': dict(dt='f4', proved=False, create=dict(chunksizes=(2, 40, 30)),
                              planted=((5, 39, 49), F4)),
    'no_fill': dict(dt='i4', proved=True, create=dict(fill_value=False)),
    'no_fill_hit': dict(dt='i4', proved=False, create=dict(fill_value=False),
                        planted=(LAST, netCDF4.default_fillvals['i4'])),
}
INT_CASES = {}
for _dt in ('i2', 'i4', 'i8', 'u2', 'u4', 'u8'):
    INT_CASES[_dt + '_clean'] = dict(dt=_dt, proved=True)
    INT_CASES[_dt + '_hit'] = dict(dt=_dt, proved=False, planted=(LAST, netCDF4.default_fillvals[_dt]))
INT_CASES['i2_top_byte_only'] = dict(dt='i2', proved=False, planted=(LAST, -32768))
CASES = {**FLOAT_CASES, **INT_CASES}


def values(c):
    rng = np.random.default_rng(7)
    a = rng.uniform(1, 100, SHAPE).astype(c['dt'])
    if 'planted' in c:
        at, x = c['planted']
        a[at] = x
    return a


def write(path, cases, form='NETCDF4', **common):
    with netCDF4.Dataset(path, 'w', format=form) as d:
        d.createDimension('t', None if form != 'NETCDF4' else SHAPE[0])
        d.createDimension('y', SHAPE[1])
        d.createDimension('x', SHAPE[2])
        for name, c in cases.items():
            kw = dict(common)
            kw.update(c.get('create', {}))
            v = d.createVariable(name, c['dt'], ('t', 'y', 'x'), **kw)
            v.set_auto_maskandscale(False)
            for k, x in c.get('attrs', {}).items():
                v.setncattr(k, x)
            a = values(c)
            if c.get('part'):
                v[:2] = a[:2]
            else:
                v[:] = a


@pytest.fixture(scope='module')
def path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp('probe') / 'cases.nc')
    write(p, CASES, zlib=True, shuffle=True, complevel=1, chunksizes=CHUNKS)
    return p


@pytest.fixture
def seen(monkeypatch):
    """The `clean` each mask/scale step is told; every array counts."""
    fastnc.warm()
    monkeypatch.setattr(_maskscale, 'FAST_MIN_SIZE', 1)
    calls = []
    real = _maskscale.apply

    def spy(*a, **k):
        calls.append(bool(k.get('clean')))
        return real(*a, **k)
    monkeypatch.setattr(_maskscale, 'apply', spy)
    return calls


def check(got, want, what):
    ok, msg = same_values(digest(got), digest(want))
    assert ok, (what, msg)


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('name', list(CASES))
def test_load(path, name, threads, seen):
    with netCDF4.Dataset(path) as r:
        want = r[name][...]
    with fastnc.Dataset(path) as d:
        got = d.load([name], threads=threads)[name]
    check(got, want, name)
    assert seen == [CASES[name]['proved']]


@pytest.mark.parametrize('name', list(CASES))
def test_always_mask_off(path, name, seen):
    with netCDF4.Dataset(path) as r:
        r[name].set_always_mask(False)
        want = r[name][...]
    with fastnc.Dataset(path) as d:
        d[name].set_always_mask(False)
        got = d.load([name])[name]
    check(got, want, name)


def test_every_variable_at_once(path, seen, monkeypatch):
    told = []
    real = _maskscale.run_plans

    def spy(items, threads=None):
        told.extend(it[3] for it in items)
        return real(items, threads)
    monkeypatch.setattr(_maskscale, 'run_plans', spy)
    with netCDF4.Dataset(path) as r, fastnc.Dataset(path) as d:
        got = d.load()
        for name in CASES:
            check(got[name], r[name][...], name)
    assert told and sum(told) + sum(seen) == sum(c['proved'] for c in CASES.values())


def test_chunks_of_a_big_split_are_not_counted(path, seen, monkeypatch):
    """Decoded in parts, a chunk is never tested: the pass runs."""
    monkeypatch.setattr(_chunks, 'PAR_MIN', 64)
    monkeypatch.setattr(_chunks, '_big', lambda V, var, size, mask, part: np.ones(len(size), bool))
    with netCDF4.Dataset(path) as r, fastnc.Dataset(path) as d:
        check(d.load(['clean'])['clean'], r['clean'][...], 'clean')
    assert seen == [False]


def test_readahead(path, seen, monkeypatch):
    """Whole variables one after another, proofs held with their window."""
    monkeypatch.setattr(_readahead, 'BIG', 1 << 10)
    with netCDF4.Dataset(path) as r, fastnc.Dataset(path) as d:
        for name in CASES:
            check(d[name][...], r[name][...], name)
    assert any(seen)


@pytest.mark.parametrize('off', ['set_auto_mask', 'set_auto_scale', 'set_auto_maskandscale'])
def test_settings_changed_after_the_proof(path, seen, monkeypatch, off):
    """A window decoded under one setting serves a read under another."""
    monkeypatch.setattr(_readahead, 'BIG', 1 << 10)
    names = list(CASES)
    with netCDF4.Dataset(path) as r, fastnc.Dataset(path) as d:
        d[names[0]][...]
        d[names[1]][...]
        for name in names[2:12]:                    # (in the window by now)
            getattr(d[name], off)(False)
            getattr(r[name], off)(False)
            check(d[name][...], r[name][...], name)


# netCDF-3: no chunks, the values tested as they are swapped

NC3_CASES = {
    'clean': dict(dt='f4', proved=True),
    'first': dict(dt='f4', proved=False, planted=((0, 0, 0), F4)),
    'last': dict(dt='f4', proved=False, planted=(LAST, F4)),
    'fill_clean': dict(dt='f4', proved=True, create=dict(fill_value=-999.0)),
    'fill_hit': dict(dt='f4', proved=False, create=dict(fill_value=-999.0), planted=(LAST, -999.0)),
    'two_second_hit': dict(dt='f4', proved=False, create=dict(fill_value=-999.0),
                           attrs=dict(missing_value=np.float32(777.0)), planted=(LAST, 777.0)),
    'two_clean': dict(dt='f4', proved=True, create=dict(fill_value=-999.0),
                      attrs=dict(missing_value=np.float32(777.0))),
    'zero_fill': dict(dt='f4', proved=False, create=dict(fill_value=0.0), planted=(LAST, -0.0)),
    'nan_fill': dict(dt='f4', proved=False, create=dict(fill_value=np.nan), planted=(LAST, np.nan)),
    'valid': dict(dt='f4', proved=False, attrs=dict(valid_min=np.float32(0.0)), planted=(LAST, -5.0)),
    'near': dict(dt='f4', proved=True, planted=(LAST, bits(0x7CF00001, 'f4'))),
    'f8_clean': dict(dt='f8', proved=True),
    'f8_last': dict(dt='f8', proved=False, planted=(LAST, netCDF4.default_fillvals['f8'])),
    'i2_clean': dict(dt='i2', proved=True),
    'i2_last': dict(dt='i2', proved=False, planted=(LAST, netCDF4.default_fillvals['i2'])),
    'i2_near': dict(dt='i2', proved=True, planted=(LAST, -32768)),
    'i4_clean': dict(dt='i4', proved=True),
    'i4_last': dict(dt='i4', proved=False, planted=(LAST, netCDF4.default_fillvals['i4'])),
}


@pytest.fixture(scope='module')
def nc3_path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp('probe3') / 'cases.nc')
    write(p, NC3_CASES, form='NETCDF3_64BIT_OFFSET')
    return p


@pytest.fixture
def tiny(monkeypatch):
    for k, v in dict(KERNELS=True, PIECE=4096, RUN_MIN=256, PAR_MIN=8192, MAP_MAX=64,
                     SWAP_NUMPY=512).items():
        monkeypatch.setattr(_nc3, k, v)


def test_nc3_cases_go_through_the_runs(nc3_path, tiny, seen):
    """(Or the rest of these prove little.)"""
    with fastnc.Dataset(nc3_path) as d:
        d['clean'][...]
    assert seen == [True]


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('name', list(NC3_CASES))
def test_nc3(nc3_path, name, threads, tiny, seen):
    _pool.set_threads(threads)
    try:
        with netCDF4.Dataset(nc3_path) as r, fastnc.Dataset(nc3_path) as d:
            check(d[name][...], r[name][...], name)
            del seen[:]
            check(d.load([name])[name], r[name][...], name)
            assert seen == [NC3_CASES[name]['proved']]
            del seen[:]
            sel = (slice(1, 4), slice(None), slice(None))
            check(d[name][sel], r[name][sel], name)
    finally:
        _pool.set_threads(None)


# the tests themselves

def test_byte_test_is_exact():
    fastnc.warm()
    for n in (0, 1, 7, 8, 9, 16, 23, 64):
        for lo in (0, 3):
            a = np.full(lo + n + 16, 0x55, np.uint8)
            a[lo:lo + n] = 0x54                     # (outside: what must not count)
            assert not _chunks._holds(a, lo, n, 0x55, 0x55)
            assert not _chunks._holds(a, lo, n, 0xD4, 0x53)     # (one bit off)
            for k in range(n):
                b = a.copy()
                b[lo + k] = 0x7C
                assert _chunks._holds(b, lo, n, 0x7C, 0x7C), (n, lo, k)
                assert _chunks._holds(b, lo, n, 0x11, 0x7C), (n, lo, k)
                assert _chunks._holds(b, lo, n, 0x7C, 0x11), (n, lo, k)
                assert not _chunks._holds(b, lo, n, 0xFC, 0xFC)


@pytest.mark.parametrize('es', [2, 4, 8])
def test_value_test_is_exact(es):
    """Every place of a word, each next to a value that is nearly the one
    looked for: the lanes of the word test must not bleed."""
    fastnc.warm()
    rng = np.random.default_rng(es)
    u = 'u%d' % es
    t = int(rng.integers(1, 1 << 62)) & ((1 << (8 * es)) - 1) | 1 << (8 * es - 1)
    w = np.frombuffer(np.array([t], u).tobytes() * (8 // es), np.uint64)[0]
    other = np.frombuffer(np.array([t ^ 0x5A], u).tobytes() * (8 // es), np.uint64)[0]
    low = np.frombuffer(np.full(8 // es, (1 << (8 * es - 1)) - 1, u).tobytes(), np.uint64)[0]
    near = [t ^ (1 << b) for b in range(8 * es)] + [t + 1 & (1 << 8 * es) - 1, 0, (1 << 8 * es) - 1]
    for n in range(8 // es, 40):
        a = np.array([near[i % len(near)] for i in range(n)], u).view(np.uint8)
        hi = a.size
        assert not _chunks.holds_value(a, 0, hi, w, w, low), n
        for k in range(n):
            b = a.copy().view(u)
            b[k] = t
            b = b.view(np.uint8)
            assert _chunks.holds_value(b, 0, hi, w, w, low), (n, k)
            assert _chunks.holds_value(b, 0, hi, other, w, low), (n, k)
            assert _chunks.holds_value(b, 0, hi, w, other, low), (n, k)


def test_a_part_too_short_to_tell_counts_as_found():
    fastnc.warm()
    w = np.uint64(1)
    assert _chunks.holds_value(np.zeros(4, np.uint8), 0, 4, w, w, np.uint64(0x7FFFFFFFFFFFFFFF))


@pytest.mark.parametrize('solo', [0, 1 << 40])
@pytest.mark.parametrize('dt, x', [('f4', F4), ('f8', netCDF4.default_fillvals['f8']),
                                   ('i2', netCDF4.default_fillvals['i2']),
                                   ('i4', netCDF4.default_fillvals['i4'])])
def test_check_before_flagging(dt, x, solo, monkeypatch):
    """A single masked value anywhere is found by the test run first (on the
    pool or not), and the array comes out as the flagging pass gives it."""
    fastnc.warm()
    monkeypatch.setattr(_maskscale, 'SOLO_BYTES', solo)
    monkeypatch.setattr(_maskscale, 'BLOCK', 1024)
    n = 5000 + 3
    a = np.random.default_rng(1).uniform(1, 100, n).astype(dt)
    p = _maskscale.make_plan(a.dtype, {}, a.dtype, 'primitive', True, True, False)
    assert _maskscale._nothing_masked(p, a, 4)
    for at in (0, 1, 1023, 1024, 1025, 2048, n - 2, n - 1):
        b = a.copy()
        b[at] = x
        assert not _maskscale._nothing_masked(p, b, 4), at
        got = _maskscale.run_plan(p, b, True)
        assert np.ma.getmaskarray(got).sum() == 1 and got.mask[at]
        assert np.ma.getdata(got).tobytes() == b.tobytes()
    got = _maskscale.run_plan(p, a, True)
    assert got.mask is np.ma.nomask or not np.ma.getmaskarray(got).any()


def test_a_masked_value_past_the_last_whole_block(monkeypatch):
    """(A last block too short for a word counts as found: flagged as ever.)"""
    fastnc.warm()
    monkeypatch.setattr(_maskscale, 'SOLO_BYTES', 0)
    monkeypatch.setattr(_maskscale, 'BLOCK', 1024)
    a = np.ones(1024 + 1, 'f4')
    p = _maskscale.make_plan(a.dtype, {}, a.dtype, 'primitive', True, True, False)
    assert not _maskscale._nothing_masked(p, a, 2)
    assert not np.ma.getmaskarray(_maskscale.run_plan(p, a, True)).any()


@pytest.mark.parametrize('attrs, dt, want', [
    ({}, 'f4', [F4]),
    ({'_FillValue': np.float32(-1)}, 'f4', [-1.0]),
    ({'_FillValue': np.float32(-1), 'missing_value': np.float32(-1)}, 'f4', [-1.0]),
    ({'_FillValue': np.float32(-1), 'missing_value': np.float32(-2)}, 'f4', None),
    ({'_FillValue': np.float32(0)}, 'f4', None),
    ({'_FillValue': np.float32(np.nan)}, 'f4', None),
    ({'valid_max': np.float32(9)}, 'f4', None),
    ({'scale_factor': np.float32(2)}, 'f4', None),
    ({}, 'i1', None),
    ({}, 'u1', None),
])
def test_what_can_be_probed(attrs, dt, want):
    p = _maskscale.probe(attrs, np.dtype(dt), True, True, False)
    if want is None and len(attrs) == 2:            # (two values: both kept)
        assert sorted(p.values.tolist()) == [-2.0, -1.0]
    elif want is None:
        assert p is None
    else:
        assert p.values.tolist() == want and not p.proven


def test_no_probe_without_masking():
    assert _maskscale.probe({}, np.dtype('f4'), False, True, False) is None


def _encoded(r):
    if np.ma.isMaskedArray(r):
        m = r._mask
        return ('ma', r.dtype.str, r.shape, np.ma.getdata(r).tobytes(),
                None if m is np.ma.nomask else m.tobytes(), type(r.fill_value).__name__,
                np.asarray(r.fill_value).tobytes())
    a = np.asarray(r)
    return (type(r).__name__, a.dtype.str, a.shape, repr(a.tolist()) if a.dtype.hasobject else a.tobytes())


def _loaded(p):
    """Every variable through load()."""
    with fastnc.Dataset(p) as d:
        return {k: None if v is None else _encoded(v) for k, v in d.load(recursive=True).items()}


@pytest.mark.parametrize('threads', [1, 3])
@pytest.mark.parametrize('name', sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.nc'))
                                        + glob.glob(os.path.join(DATA, '*.h5'))))
def test_arrays_masked_together_equal_masked_alone(name, threads, monkeypatch):
    fastnc.warm()
    monkeypatch.setattr(_maskscale, 'FAST_MIN_SIZE', 1)
    p = os.path.join(DATA, name)
    with fastnc.Dataset(p) as d:
        together = d.load(recursive=True, threads=threads)
        alone = {}
        for g in d._walk():
            for n in g.variables:
                alone[(g.path.rstrip('/') + '/' + n).lstrip('/')] = g.load([n], threads=threads)[n]
        assert set(together) == set(alone)
        for k in alone:
            a, b = together[k], alone[k]
            assert (None if a is None else _encoded(a)) == (None if b is None else _encoded(b)), k


def test_masked_together_with_hits_in_some_blocks(monkeypatch):
    """Blocks with and without masked values, in arrays of several sizes."""
    fastnc.warm()
    monkeypatch.setattr(_maskscale, 'BLOCK', 64)
    plan = _maskscale.make_plan(np.dtype('f4'), {}, np.dtype('f4'), 'primitive', True, True, False)
    scaled = {'_FillValue': np.int16(-999), 'scale_factor': np.float32(0.5)}
    splan = _maskscale.make_plan(np.dtype('i2'), scaled, np.dtype('i2'), 'primitive', True, True, False)
    rng = np.random.default_rng(3)
    items = []
    for n in (70, 640, 6400, 64000, 7, 1):
        a = rng.uniform(1, 90, n).astype('f4')
        items.append((plan, a.copy(), True, False))
        a[n // 2] = F4
        items.append((plan, a, True, False))
        i = rng.integers(1, 90, n).astype('i2')
        items.append((splan, i.copy(), True, False))
        i[-1] = -999
        items.append((splan, i, False, False))
    for threads in (1, 4):
        got = _maskscale.run_plans(items, threads)
        for (p, raw, always, _), r in zip(items, got):
            want = _maskscale.run_plan(p, raw.copy(), always, 1)
            assert type(r) is type(want)
            assert np.ma.getdata(r).tobytes() == np.ma.getdata(want).tobytes()
            assert np.ma.getmaskarray(r).tobytes() == np.ma.getmaskarray(want).tobytes()


def test_walk_runs_everything_once():
    done = []
    lock = __import__('threading').Lock()

    def task(i, parts):
        left = [parts]

        def run():
            while True:
                with lock:
                    if not left[0]:
                        return
                    left[0] -= 1
                    done.append(i)
        return run
    tasks = [task(i, p) for i, p in enumerate((500, 3, 1, 0, 40))]
    _pool.walk(tasks, 4)
    assert sorted(done) == [0] * 500 + [1] * 3 + [2] + [4] * 40
    _pool.walk([], 4)


@pytest.mark.parametrize('name', sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.nc'))
                                        + glob.glob(os.path.join(DATA, '*.h5'))))
def test_corpus_same_with_and_without(name, monkeypatch):
    fastnc.warm()
    monkeypatch.setattr(_maskscale, 'FAST_MIN_SIZE', 1)
    p = os.path.join(DATA, name)
    got = _loaded(p)
    monkeypatch.setattr(_maskscale, 'probe', lambda *a: None)
    assert got == _loaded(p)
