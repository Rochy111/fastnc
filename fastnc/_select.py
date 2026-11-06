"""Turning a numpy-style index into what the readers read.

Readers read boxes: one (start, stop, step) per dimension, step > 0. An
index is reduced to such a box plus what numpy does to the box's values
afterwards: dropping dimensions indexed by an integer, reversing those
given a negative step, and picking the elements of integer or boolean
sequences (netCDF4-python's orthogonal indexing: each sequence indexes its
own dimension independently, unlike numpy's fancy indexing).
"""
import numpy as np


_S = object()                   # tags a slice in a memo key (a user's tuple cannot hold it)
_MEMO = {}                      # (shape, key) -> (box, None, drop): a read normalizes its
                                # index up to three times (Variable, fused path, reader)


def normalize(key, shape):
    """(box, post, drop): box is [(start, stop, step)] with step > 0 (and an
    empty box as start == stop); post a tuple of per-dimension indexes to
    apply to the box's values (None if nothing to do); drop the dimensions
    to remove (integer indexes)."""
    if not isinstance(key, tuple):
        key = (key,)
    try:
        # (with the types: 1.0 equals 1 but is no valid index)
        hk = (tuple(shape), tuple((_S, type(k.start), k.start, type(k.stop), k.stop,
                                   type(k.step), k.step) if type(k) is slice else (type(k), k)
                                  for k in key))
        r = _MEMO.get(hk)
    except TypeError:                               # (arrays, lists: not memoized)
        hk = r = None
    if r is not None:
        return list(r[0]), None, r[2]
    r = _normalize(key, shape)
    if hk is not None and r[1] is None:
        if len(_MEMO) >= 4096:
            _MEMO.clear()
        _MEMO[hk] = (tuple(r[0]), None, r[2])
    return r


def whole_scalar(key):
    """Whether key is (), Ellipsis or (Ellipsis,), the indexes of a scalar's
    whole value in every front end (without comparing arrays)."""
    return key is Ellipsis or (type(key) is tuple and (not key or (len(key) == 1 and key[0] is Ellipsis)))


def scalar_nc(key):
    """netCDF4-python's reading of key on a scalar (0-d) variable: (k, pick),
    k the index to read the scalar with (Ellipsis) and pick whether an
    integer picked the value (a raw read then gives a numpy scalar, not a 0-d
    array). Raises IndexError where the key selects nothing.

    netCDF4-python indexes a scalar as if it had shape (1,): v[:], v[0:1],
    v[[0]], v[()], v[...] all give the value. Its integers are not
    bounds-checked (netCDF-C ignores a scalar's start), so any integer from
    -1 up picks the value; a selection of no element (v[1:], v[[False]]) is
    an IndexError. One difference: netCDF4-python 1.7 fails on v[::2] (an
    IndexError of its own, 'tuple index out of range', from a strided slice
    without a stop); here that is the value, as v[0:1:2] is in both."""
    if whole_scalar(key):
        return Ellipsis, False
    k = key if isinstance(key, tuple) else (key,)
    ints = []
    for e in k:
        if e is None or e is Ellipsis or isinstance(e, slice) or np.ndim(e):
            ints.append(e)
            continue
        try:
            i = int(e)                              # (bools, floats, digits: as netCDF4-python)
        except (TypeError, ValueError):
            raise IndexError('only integers, slices, Ellipsis and 1-d integer or boolean '
                             'sequences are valid indices') from None
        if i < -1:
            raise IndexError('index %d is out of bounds for a scalar variable' % i)
        ints.append(0)
    if sum(e is not Ellipsis for e in k) > 1 and not any(e is None for e in k):
        raise ValueError('slicing expression exceeds the number of dimensions of the variable')
    box, _, drop = _normalize(tuple(ints), (1,))
    if drop:
        return Ellipsis, True
    if box[0][0] == box[0][1]:
        raise IndexError('index selects no element of a scalar variable')
    return Ellipsis, False


def scalar_h5(key):
    """h5py's rule for a scalar dataset: only () and Ellipsis index it."""
    if not whole_scalar(key):
        if any(e is None for e in key):
            raise TypeError('Indexing with None (or np.newaxis) is not supported')
        raise ValueError('Illegal slicing argument for scalar dataspace')


def _normalize(key, shape):
    rank = len(shape)
    # expand Ellipsis
    if any(k is Ellipsis for k in key):
        i = next(j for j, k in enumerate(key) if k is Ellipsis)
        rest = [k for k in key[i + 1:] if k is not Ellipsis]
        n = rank - i - len(rest) - sum(1 for k in key[:i] if k is None) - sum(1 for k in rest if k is None)
        key = key[:i] + (slice(None),) * max(n, 0) + tuple(rest)
    if any(k is None for k in key):
        raise IndexError('np.newaxis is not supported in a read; reshape the result')
    if len(key) > rank:
        raise IndexError('too many indices: %d for %d dimensions' % (len(key), rank))
    key = key + (slice(None),) * (rank - len(key))
    box = []
    post = []
    drop = []
    need_post = False
    for d, (k, n) in enumerate(zip(key, shape)):
        if isinstance(k, (int, np.integer)):
            i = int(k)
            if i < 0:
                i += n
            if not 0 <= i < n:
                raise IndexError('index %d is out of bounds for dimension %d of size %d' % (int(k), d, n))
            box.append((i, i + 1, 1))
            post.append(slice(None))
            drop.append(d)
        elif isinstance(k, slice):
            start, stop, step = k.indices(n)
            if step > 0:
                if stop <= start:
                    box.append((start, start, 1))
                else:
                    box.append((start, stop, step))
                post.append(slice(None))
            else:
                # the same elements read forwards, then reversed
                cnt = len(range(start, stop, step))
                if cnt == 0:
                    box.append((0, 0, 1))
                    post.append(slice(None))
                else:
                    last = start + (cnt - 1) * step
                    box.append((last, start + 1, -step))
                    post.append(slice(None, None, -1))
                    need_post = True
        else:
            a = np.asarray(k)
            if isinstance(k, (list, tuple)) and not len(k):
                a = a.astype(np.int64)                  # (numpy takes it for floats)
            if a.dtype == bool:
                if a.ndim != 1 or a.size != n:
                    raise IndexError('boolean index of length %d for dimension %d of size %d'
                                     % (a.size, d, n))
                a = np.nonzero(a)[0]
            if a.ndim == 0:
                return normalize(tuple(key[:d]) + (int(a),) + tuple(key[d + 1:]), shape)
            if a.ndim != 1 or a.dtype.kind not in 'iu':
                raise IndexError('only integers, slices, Ellipsis and 1-d integer or boolean '
                                 'sequences are valid indices')
            a = np.where(a < 0, a + n, a).astype(np.int64)
            if a.size and (a.min() < 0 or a.max() >= n):
                raise IndexError('index out of bounds for dimension %d of size %d' % (d, n))
            if a.size == 0:
                box.append((0, 0, 1))
                post.append(slice(None))
            else:
                lo, hi = int(a.min()), int(a.max()) + 1
                box.append((lo, hi, 1))
                post.append(a - lo)
                need_post = True
    return box, (tuple(post) if need_post else None), tuple(drop)


def box_shape(box):
    return tuple(0 if e <= s else (e - s + k - 1) // k for s, e, k in box)


def finish(values, post, drop, scalars=True):
    """The box's values after the post-indexing and integer dimensions dropped."""
    if post is not None:
        arrays = [p for p in post if isinstance(p, np.ndarray)]
        if len(arrays) <= 1:
            values = values[post]
        else:
            # orthogonal: each sequence indexes its own dimension
            for d, p in enumerate(post):
                if not (isinstance(p, slice) and p == slice(None)):
                    idx = [slice(None)] * values.ndim
                    idx[d] = p
                    values = values[tuple(idx)]
    if drop:
        values = values.reshape([n for d, n in enumerate(values.shape) if d not in drop])
        if values.ndim == 0 and scalars:
            return values[()]                       # every dimension indexed: a numpy scalar
    return values
