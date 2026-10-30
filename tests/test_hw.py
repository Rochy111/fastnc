"""What fastnc finds about the machine, and that its settings follow:
budgets shrink with available memory, blocks with the cache, threads with
what the process may use; tune()'s numbers are kept per machine."""
import json
import os

import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _hw, _pool


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """Nothing found yet, and a cache directory of this test's own."""
    monkeypatch.setenv('FASTNC_CACHE_DIR', str(tmp_path))
    saved = dict(_hw._found)
    _hw._found.clear()
    yield tmp_path
    _hw._found.clear()
    _hw._found.update(saved)


def test_found_makes_sense():
    assert 1 <= _hw.cpus() <= (os.cpu_count() or 1)
    t = _hw.topology()
    assert 1 <= t['perf_cores'] <= t['cores'] <= t['logical']
    total, avail = _hw.memory()
    assert total is None or 0 < avail <= total
    assert _hw.l2_bytes() >= 1 << 16
    info = fastnc.hardware()
    assert info['threads'] == _pool.threads()
    json.dumps(info)                                # (plain data)


def test_quota_limits_cpus(monkeypatch):
    if not hasattr(os, 'cpu_count'):
        pytest.skip('no cpu_count')
    monkeypatch.setattr(_hw, '_quota', lambda: 1.5)
    monkeypatch.setattr(_hw.sys, 'platform', 'linux')
    saved = dict(_hw._found)
    _hw._found.clear()
    try:
        assert _hw.cpus() <= 2
    finally:
        _hw._found.clear()
        _hw._found.update(saved)


@pytest.mark.parametrize('avail, want', [(64 << 30, 1.0), (16 << 30, 1.0), (4 << 30, 0.25),
                                         (1 << 20, _hw.MIN_RAM_SCALE)])
def test_budgets_scale_with_available_memory(avail, want, fresh, monkeypatch):
    monkeypatch.setattr(_hw, 'memory', lambda: (avail * 2, avail))
    assert _hw.ram_scale() == want
    assert _hw.scaled(256 << 20) == max(1 << 20, int((256 << 20) * want))
    assert _hw.scaled(4096) == 4096                 # (never below what was asked, when small)


def test_unknown_memory_changes_nothing(fresh, monkeypatch):
    monkeypatch.setattr(_hw, 'memory', lambda: (None, None))
    assert _hw.ram_scale() == 1.0


@pytest.mark.parametrize('l2, want', [(2 << 20, 1 << 16), (4 << 20, 1 << 16),
                                      (256 << 10, (1 << 13))])
def test_blocks_follow_the_cache(l2, want, fresh, monkeypatch):
    monkeypatch.setattr(_hw, 'l2_bytes', lambda: l2)
    b = _hw.block(1 << 16)
    assert b == want and b % 64 == 0


def test_measurements_are_kept_per_machine(fresh, monkeypatch):
    assert _hw.tuned() == {}
    assert _hw.save_tuned(cpu={'threads': 3, 'load_s': 0.5}) is not None
    assert _hw.tuned() == {'threads': 3, 'load_s': 0.5}
    _hw.save_tuned(cpu={'load_s': 0.4})             # merged
    assert _hw.tuned() == {'threads': 3, 'load_s': 0.4}
    monkeypatch.setattr(_hw, 'cpu_name', lambda: 'another CPU')
    assert _hw.tuned() == {}                        # another machine reads nothing


def test_another_format_is_ignored(fresh):
    with open(os.path.join(str(fresh), 'tune.json'), 'w') as f:
        json.dump({'format': -1, 'cpu_key': _hw.machine_key(), 'cpu': {'threads': 2}}, f)
    assert _hw.tuned() == {}


def test_load_threshold_is_this_machines(fresh, monkeypatch):
    assert _chunks._load_after() == _chunks.LOAD_GUESS
    _hw.save_tuned(cpu={'load_s': 0.3})
    assert _chunks._load_after() == 0.3
    _hw.save_tuned(cpu={'load_s': 99.0})
    assert _chunks._load_after() == _chunks.LOAD_MOST
    monkeypatch.setattr(_chunks, 'LOAD_AFTER', 7.0)
    assert _chunks._load_after() == 7.0


def test_tuned_threads_are_used_but_never_exceed_the_allowed(fresh, monkeypatch):
    monkeypatch.setattr(_pool, '_THREADS', 0)
    monkeypatch.setattr(_pool, '_AUTO', 0)
    _hw.save_tuned(cpu={'threads': 1})
    assert _pool.threads() == 1
    monkeypatch.setattr(_pool, '_AUTO', 0)
    _hw.save_tuned(cpu={'threads': 100000})
    assert _pool.threads() == _hw.cpus()


def test_work_threads_scale_with_work(monkeypatch):
    monkeypatch.setattr(_pool, '_WORK', 1000)
    assert _pool.work_threads(10, 24) == 1
    assert _pool.work_threads(5000, 24) == 5
    assert _pool.work_threads(10 ** 9, 24) == 24


def test_little_output_many_chunks_uses_threads(monkeypatch):
    """One column of a chunked variable: the work is every chunk's decoding,
    however little comes out of it."""
    seen = []
    real = _pool.share
    monkeypatch.setattr(_pool, 'share', lambda w, n: (seen.append(n), real(w, n))[1])
    monkeypatch.setattr(_pool, '_WORK', 64)
    monkeypatch.setattr(_chunks, 'ZGRID_SOLO', 0)
    fastnc.warm()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'idx_latest.h5')
    with fastnc.File(path) as f:
        d = f['btree2_z']
        want = np.asarray(d[()])[:, 3]
        got = d[:, 3]
    assert got.tobytes() == want.tobytes()
    assert max(seen) > 1


def test_tune_finds_what_it_should(fresh):
    r = fastnc.tune(gpu=False, verbose=False)
    assert 1 <= r['threads'] <= _hw.cpus()
    assert r['thread_bytes'] >= 1 << 14
    assert r['decode_bytes_s'] > 0
    assert _hw.tuned()['threads'] == r['threads']
