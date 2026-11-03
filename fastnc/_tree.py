"""The chunk tables of a whole batch read in one call of a kernel.

A load of a file reads hundreds of variables whole, and netCDF-4 and most HDF5
files keep each one's chunks in a version 1 B-tree. Parsing each tree and
planning each variable took a dozen numpy calls a variable, on one thread:
a quarter of a WRF load and half of a Ku-band one, around decoding that had
become cheap. Nothing there needs Python: the numbers are fixed-width and the
plan of a whole dataset follows from them. So `_walk` goes through every tree
and writes every plan into one set of arrays, and each variable gets views of
them. What it does not expect (a chunk past the variable's edge, a damaged
node, more chunks than fit) it declines, and the variables take the usual path,
which knows what to say about them.
"""
import math

import numpy as np

from . import _chunks
from ._intrin import ld64
from ._jit import njit

MIN_VARIABLES = 2               # one variable: nothing to share


@njit(cache=True, nogil=True)
def _walk(src, roots, base, ranks, C, S, CS, addr, size, mask, offs, count, needend, first, stack):
    """Every chunk of the B-trees at roots[v] into addr .. needend, in tree
    order, variable v's from first[v] (chunks C[v], shape S[v], chunk strides
    CS[v]). The number of chunks, or -1 where the usual path should decide."""
    top = src.shape[0]
    cap = addr.shape[0]
    room = stack.shape[0]
    nvar = roots.shape[0]
    i = 0
    for v in range(nvar):
        first[v] = i
        rank = ranks[v]
        ksize = 8 + 8 * (rank + 1)
        step = ksize + 8
        stack[0] = roots[v]
        sp = 1
        while sp > 0:
            sp -= 1
            a = stack[sp] + base
            if a < 0 or a + 24 > top:
                return -1
            if src[a] != 84 or src[a + 1] != 82 or src[a + 2] != 69 or src[a + 3] != 69:    # TREE
                return -1
            n = np.int64(src[a + 6]) | (np.int64(src[a + 7]) << 8)
            p = a + 24
            if p + n * step + ksize + 8 > top:
                return -1
            if src[a + 5] != 0:                         # keys and child nodes
                if sp + n > room:
                    return -1
                for k in range(n - 1, -1, -1):
                    stack[sp] = np.int64(ld64(src, p + k * step + ksize))
                    sp += 1
                continue
            if i + n > cap:
                return -1
            for k in range(n):
                q = p + k * step
                w = np.int64(ld64(src, q))
                size[i] = w & 0xFFFFFFFF
                mask[i] = (w >> 32) & 0xFFFFFFFF
                ne = 0
                for d in range(rank):
                    o = np.int64(ld64(src, q + 8 + 8 * d))
                    if o < 0 or o >= S[v, d]:
                        return -1
                    c = min(C[v, d], S[v, d] - o)
                    offs[i, d] = o
                    count[i, d] = c
                    ne += (c - 1) * CS[v, d]
                needend[i] = ne + 1
                addr[i] = np.int64(ld64(src, q + ksize)) + base
                i += 1
    first[nvar] = i
    return i


class Index:
    """What read_batch wants of a chunk index when the plan is already made."""
    __slots__ = ('chunks', 'plans')


def _eligible(r, f):
    ds = r.ds
    lay = ds.layout
    return (r._index is None and lay.cls == 2 and lay.index == 'btree1' and lay.address != f.undef
            and _chunks.fusable(ds))


def indexes(readers):
    """For readers each reading a whole dataset: an Index with the plan made,
    or None for those left to the usual path (all, unless there are several
    of the usual kind and nothing unexpected in them)."""
    none = [None] * len(readers)
    if len(readers) < MIN_VARIABLES or not _chunks.loaded():
        return none
    f = readers[0].f
    if f.O != 8 or f.source is not None:
        return none
    which = [k for k, r in enumerate(readers) if _eligible(r, f)]
    if len(which) < MIN_VARIABLES:
        return none
    shapes = [readers[k].ds.shape for k in which]
    chunks = [readers[k].ds.chunks for k in which]
    wide = max(len(s) for s in shapes)
    if not wide or any(len(c) != len(s) or not s for c, s in zip(chunks, shapes)):
        return none
    grids = [math.prod(-(-n // c) for n, c in zip(s, cs)) for s, cs in zip(shapes, chunks)]
    cap = sum(grids)
    nvar = len(which)
    pad = [0] * wide

    def rows(lists):
        return np.array([list(x) + pad[len(x):] for x in lists], np.int64)
    C, S = rows(chunks), rows(shapes)
    CS = rows([_chunks.strides(tuple(c)).tolist() for c in chunks])
    addr, size, mask, needend = (np.empty(cap, np.int64) for _ in range(4))
    offs, count = np.zeros((cap, wide), np.int64), np.zeros((cap, wide), np.int64)
    first = np.empty(nvar + 1, np.int64)
    roots = np.array([readers[k].ds.layout.address for k in which], np.int64)
    ranks = np.array([len(s) for s in shapes], np.int64)
    stack = np.empty(1 << 16, np.int64)
    if _walk(readers[0].src, roots, f.base, ranks, C, S, CS, addr, size, mask, offs, count, needend,
             first, stack) < 0:
        return none
    zero = np.zeros_like(offs)
    at = first.tolist()
    got = list(none)
    for j, k in enumerate(which):
        a, b, rank = at[j], at[j + 1], len(shapes[j])
        p = _chunks.Plan()
        p.addr, p.size, p.mask, p.needend = addr[a:b], size[a:b], mask[a:b], needend[a:b]
        p.cstart, p.count, p.ostart = zero[a:b, :rank], count[a:b, :rank], offs[a:b, :rank]
        p.wide = zero[a:b], count[a:b], offs[a:b]
        p.fill_needed = b - a < grids[j]
        x = Index()
        x.chunks = C[j, :rank]
        x.plans = {tuple((0, n, 1) for n in shapes[j]): p}
        got[k] = x
    return got
