"""The chunk tables of a batch read by one kernel: the plans the usual path
makes, chunk for chunk, and values the same whichever made them."""
import glob
import os

import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _jit, _tree

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.nc')) + glob.glob(os.path.join(DATA, '*.h5')))


def readers(d):
    out = []
    for g in d._walk():
        for v in g.variables.values():
            r = getattr(v._bv, '_reader', None)
            if r is not None and r.ds.shape and not r.ds.null and r.ds.layout.cls == 2:
                out.append(r)
    return out


def rows(p, rank):
    """A plan's chunks as sorted rows of every number that matters."""
    a = np.column_stack([p.addr, p.size, p.mask, p.needend, p.cstart[:, :rank], p.count[:, :rank], p.ostart[:, :rank]])
    return a[np.lexsort(a.T[::-1])]


@pytest.mark.parametrize('name', FILES)
def test_same_plans_as_the_usual_path(name):
    if not _chunks.loaded():
        pytest.skip('kernels off')
    with fastnc.Dataset(os.path.join(DATA, name)) as d:
        rs = readers(d)
        made = _tree.indexes(rs)
        for r, x in zip(rs, made):
            if x is None:
                continue
            ds = r.ds
            box = [(0, n, 1) for n in ds.shape]
            mine = _chunks.plan(x, ds.shape, box)
            r._index = None
            usual = _chunks.plan(r.index, ds.shape, box)
            assert mine.fill_needed == usual.fill_needed
            assert np.array_equal(rows(mine, len(ds.shape)), rows(usual, len(ds.shape))), ds.name
            assert x.chunks.tolist() == r.index.chunks.tolist()


@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('name', FILES)
def test_load_gives_the_same_values(name, masked, monkeypatch):
    def load():
        with fastnc.Dataset(os.path.join(DATA, name)) as d:
            d.set_auto_maskandscale(masked)
            return d.load(recursive=True)
    try:
        got = load()
    except Exception as e:                          # (as the usual path raises, in places)
        monkeypatch.setattr(_tree, 'indexes', lambda rs: [None] * len(rs))
        with pytest.raises(type(e)):
            load()
        return
    monkeypatch.setattr(_tree, 'indexes', lambda rs: [None] * len(rs))
    want = load()
    assert got.keys() == want.keys()
    for k in got:
        a, b = got[k], want[k]
        if isinstance(a, np.ndarray) and a.dtype.kind not in 'OV':
            assert np.ma.getdata(a).tobytes() == np.ma.getdata(b).tobytes(), k
            assert np.ma.getmaskarray(a).tobytes() == np.ma.getmaskarray(b).tobytes(), k


def walk(src, root=0, shape=(4,)):
    n = 8
    C = np.array([[2]], np.int64)
    S = np.array([shape], np.int64)
    a, s, m, e = (np.empty(n, np.int64) for _ in range(4))
    o, c = np.zeros((n, 1), np.int64), np.zeros((n, 1), np.int64)
    return _tree._walk(src, np.array([root], np.int64), 0, np.ones(1, np.int64), C, S, np.ones((1, 1), np.int64),
                       a, s, m, o, c, e, np.empty(2, np.int64), np.empty(16, np.int64))


def node(chunks, offsets):
    """A level-0 node of a rank-1 B-tree (24-byte header, 32-byte entries)."""
    b = bytearray(b'TREE' + bytes([1, 0]) + chunks.to_bytes(2, 'little') + bytes(16))
    for k, o in enumerate(offsets):
        b += (100).to_bytes(4, 'little') + bytes(4) + int(o).to_bytes(8, 'little') + bytes(8) + (1000 + k).to_bytes(8, 'little')
    return np.frombuffer(bytes(b) + bytes(64), np.uint8)


def test_walk_declines_what_it_should():
    assert walk(node(2, [0, 2])) == 2
    assert walk(node(2, [0, 6])) == -1              # past the variable's edge
    assert walk(np.zeros(100, np.uint8)) == -1      # not a node
    assert walk(node(2, [0, 2]), root=1 << 40) == -1
    assert walk(node(9, list(range(9)), ), shape=(100,)) == -1  # more than there is room for


def test_python_edits_keep_the_kernels():
    """Only kernels and what surrounds them count for the compiled code's
    stamp, and the locator giving numba that stamp installs: without it, a
    line added above a kernel orphans it without a word (it did)."""
    a = _jit.kernel_stamp()
    assert a == _jit.kernel_stamp() and len(a) == 8
    assert _jit._locate_by_kernels()
