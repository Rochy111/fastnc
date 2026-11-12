"""Complex numbers as netCDF stores them, read as numpy complex arrays:
Dataset(path, auto_complex=True), netCDF4-python 1.7's switch, after the
conventions of nc-complex (which netCDF4-python reads them through):

- a compound of two members of one float type (float or double), the first
  named starting with 'r' and the second with 'i', in any case ({r, i},
  {real, imag}, {Re, Im} ...): complex64 / complex128 of the same shape. As
  netCDF4-python, not masked or scaled (a compound is not).
- a float or double variable whose last dimension has length 2 and is named
  `complex`, `ri` or `_pfnc_complex` or the start of one (`r`, `c`, `comp`
  ...: nc-complex compares the file's name with that many letters of each,
  case counting): that dimension becomes the complex one, so the variable loses
  it (dimensions, shape, chunking), and masking and scaling apply to the
  complex values as netCDF4-python applies them.

nc-complex also takes such a dimension anywhere in the variable, and integer
variables: netCDF4-python then drops the last dimension whatever it is, and
reads wrong values or none. fastnc leaves those as stored.
"""
import numpy as np

DIM_NAMES = ('complex', 'ri', '_pfnc_complex')          # (and every start of them)


def complex_dim(name):
    """Whether nc-complex takes a dimension of this name (and length 2)
    for the real and imaginary parts."""
    return bool(name) and any(k.startswith(name) for k in DIM_NAMES)


def kind(bv):
    """'compound', 'dimension', or None: how backend variable bv holds complex
    numbers, if it does."""
    dt = bv.dtype
    if not isinstance(dt, np.dtype):
        return None
    if dt.names is not None:
        if len(dt.names) != 2:
            return None
        (a, b) = (dt.fields[n][0] for n in dt.names)
        if (a.kind != 'f' or a.itemsize not in (4, 8) or a.newbyteorder('=') != b.newbyteorder('=')
                or a.shape or b.shape):
            return None
        if dt.names[0][:1].lower() != 'r' or dt.names[1][:1].lower() != 'i':
            return None
        return 'compound'
    if dt.kind == 'f' and dt.itemsize in (4, 8) and bv.shape and bv.shape[-1] == 2:
        dims = bv.dimensions
        if dims and complex_dim(dims[-1]):
            return 'dimension'
    return None


def _complex_of(f):
    return np.dtype('c8' if f.itemsize == 4 else 'c16')


class ComplexBackend:
    """A backend variable of complex values over one storing them as a
    compound or a last dimension of two floats (see the module)."""

    probes = False

    def __init__(self, inner, kind_, group):
        from ._dataset import Variable
        self._inner = inner
        self.kind = kind_
        self._raw = Variable(group, inner)              # (reads the stored values, raw)
        self._raw.set_auto_maskandscale(False)
        self._raw.set_auto_chartostring(False)
        self.name = inner.name
        self.attrs = inner.attrs
        dt = inner.dtype
        if kind_ == 'compound':
            self._fields = dt.names
            self._part = dt.fields[dt.names[0]][0]
            self.dimensions = inner.dimensions
        else:
            self._part = dt
            self.dimensions = tuple(inner.dimensions[:-1])
        self.dtype = _complex_of(self._part)

    @property
    def shape(self):
        s = self._raw.shape
        return tuple(s[:-1]) if self.kind == 'dimension' else tuple(s)

    @property
    def chunks(self):
        c = self._inner.chunks
        return None if c is None else (tuple(c[:-1]) if self.kind == 'dimension' else tuple(c))

    def __getattr__(self, name):
        if name in ('filters', 'endian', 'no_fill'):
            return getattr(self._inner, name)
        raise AttributeError(name)

    def user_type(self):
        """The datatype netCDF4-python reports: for a compound, its
        CompoundType 'complex' of two unnamed floats; else the complex dtype."""
        if self.kind != 'compound':
            return None
        from ._dataset import CompoundType
        p = self._part.newbyteorder('=')
        return CompoundType('complex', np.dtype([('f0', p), ('f1', p)], align=True))

    def read(self, sel, out=None):
        """The complex values of a box (a tuple of slices, one per dimension)."""
        sel = tuple(sel)
        if self.kind == 'dimension':
            parts = self._raw.read(sel + (slice(None),))
            parts = np.asarray(parts)
            r = np.empty(parts.shape[:-1], self.dtype)
            r.real = parts[..., 0]
            r.imag = parts[..., 1]
        else:
            s = np.asarray(self._raw.read(sel if sel else Ellipsis))
            (a, at), (b, bt) = (s.dtype.fields[n][:2] for n in self._fields)
            if (at == 0 and bt == a.itemsize == s.dtype.itemsize // 2 and a.isnative and b.isnative
                    and s.flags.c_contiguous):
                r = s.view(self.dtype)              # (packed as numpy's complex: the same bytes)
            else:
                r = np.empty(s.shape, self.dtype)
                r.real = s[self._fields[0]]
                r.imag = s[self._fields[1]]
        if out is not None:
            np.copyto(out, r)
            return out
        return r
