"""The output arena never hands out memory an array still uses."""
import gc

import numpy as np
import pytest

from fastnc import _arena


def test_views_keep_their_block():
    a = _arena.empty((1024, 512), np.float32)
    a[...] = 7
    keep = [a[3:9, ::4], a.T, a.reshape(-1)[100:2000], a[5]]
    del a
    gc.collect()
    others = [_arena.empty((1024, 512), np.float32) for _ in range(4)]
    for o in others:
        o[...] = -1
    for k in keep:
        assert np.all(k == 7)
    del keep, others
    gc.collect()


def test_block_returns_after_last_view():
    _arena.release()
    a = _arena.empty((2 << 20,), np.uint8)
    v = a[10:20]
    del a
    gc.collect()
    assert _arena.held() == 0                      # the view still holds it
    del v
    gc.collect()
    assert _arena.held() >= 2 << 20
    b = _arena.empty((2 << 20,), np.uint8)          # reuses it
    assert _arena.held() == 0
    del b


def test_small_arrays_are_plain_numpy():
    a = _arena.empty((10, 10), np.float64)
    assert a.flags.owndata


def test_stress_reuse_with_survivors():
    rng = np.random.default_rng(1)
    alive = []
    for i in range(200):
        n = int(rng.integers(1 << 20, 6 << 20)) // 8
        a = _arena.empty((n,), np.float64)
        a[...] = i
        if rng.random() < 0.3:
            alive.append((i, a[::7]))
        del a
        if len(alive) > 20:
            alive.pop(int(rng.integers(len(alive))))
        gc.collect()
        for j, v in alive:
            assert v[0] == j and v[-1] == j


def test_release_and_budget():
    old = _arena.BUDGET
    try:
        _arena.BUDGET = 4 << 20
        blocks = [_arena.empty((3 << 20,), np.uint8) for _ in range(4)]
        del blocks
        gc.collect()
        assert _arena.held() <= 4 << 20
    finally:
        _arena.BUDGET = old
        _arena.release()
    assert _arena.held() == 0


@pytest.mark.parametrize('dtype', ['f4', 'i2', 'u1', 'f8', '?'])
def test_dtypes_and_shapes(dtype):
    a = _arena.empty((300, 700, 5), dtype)
    assert a.shape == (300, 700, 5) and a.dtype == np.dtype(dtype) and a.flags.c_contiguous
    assert a.flags.writeable
