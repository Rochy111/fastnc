# Contracts between fastnc's parts

Written before the parts exist, so that they can be built at the same time
and fit. Change a contract only by agreement (tell the lead), never
silently.

## Ground rules (everyone)

- Values are **bit-identical** to the references: raw reads equal h5py /
  netCDF4-python with auto mask+scale off, byte for byte (fastnc returns
  native byte order; netCDF4-python keeps big-endian netCDF-4 / HDF4
  variables big-endian; compare values, not the byte order flag); masked/scaled reads equal netCDF4-python's in every rule
  and in the same float type (one departure on purpose: a valid range in
  unpacked units, `_maskscale._unpacked_units`); GPU equals CPU.
- Python 3.9+, numpy 1.23+ and 2.x (beware NEP 50: arithmetic on numpy
  uint8/int16 scalars stays in that type and wraps; read header fields as
  Python ints). numba is the only compiled dependency; cupy optional.
- Kernels: `from ._jit import njit` (deferred numba import, see
  `fastnc/_jit.py`), `@njit(cache=True, nogil=True)`, no `parallel=True`
  / `prange` (numba's threads spin; fastnc uses its own pool), no
  `inline='always'`, explicit loops, int64/uint64 indices, never mix
  uint64 and int64 in one expression (numba makes it float64). Unaligned
  word loads / stores and an atomic counter are in `fastnc/_intrin.py`.
- Never write outside the output range you were given: kernels run on
  many threads writing into one shared output array.
- Dev env: `C:\Users\Roch\miniconda3\envs\fastnc-dev\python.exe`
  (numpy 2.5, numba 0.68, h5py, netCDF4, xarray, hdf5plugin, pyhdf, cupy,
  nvCOMP). numpy 1.26 env for compatibility runs:
  `C:\Users\Roch\miniconda3\envs\bench-wrf\python.exe` (numba 0.59).
  Ask the lead before installing anything into either.
- Real data files (read-only, never commit, never modify):
  `C:\Users\Roch\Desktop\Python\fastnc\*.nc` and
  `C:\Users\Roch\Desktop\Python\fastnc\wrfout_d02_2024-11-06_01_30_00`.
  Small committed test files: `tests/data/` (made by
  `tools/make_corpus.py`). Older synthetic corpus (not in git):
  `C:\Users\Roch\Desktop\Python\fastnc\corpus\synthetic\`.
- Timing on this machine (i7-13700K, 8P+8E): call
  `tools/profile/common.fast_thread()` on timing threads; other agents
  may be running, so compare A/B alternately and report ratios with the
  spread; the final benchmark table is made later on a quiet machine.
- Commits: `git -c user.name=Roch -c user.email=113571107+Rochy111@users.noreply.github.com commit`,
  message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Never push. Docstrings say what a thing is and why, as prose; comments
  short, trailing the line they explain (fastgrib's style).

## 1. Filter (codec) kernels

One module per codec, `fastnc/_<codec>.py`, each exposing:

```python
@njit(cache=True, nogil=True)
def decode(src, p, pend, dst, o, oend, cd, ws):
    """Decode src[p:pend] into dst[o:oend]. Returns the end of the output
    written (o + bytes written), or a negative error code."""

def workspace_size(chunk_nbytes):      # plain Python
    """Bytes of per-thread scratch decode() needs for a chunk this big."""
```

- `src`, `dst`, `ws`: C-contiguous `uint8` arrays. `src` is usually the
  whole mapped file. `ws` is this thread's scratch, at least
  `workspace_size(...)` bytes, reused from chunk to chunk.
- `cd`: `int64` array `[element_size, chunk_nbytes, n, cd_0 .. cd_{n-1}]`:
  the dataset's element size in bytes, the decoded chunk's size, then the
  filter's client data values from the filter pipeline message.
- `oend - o` is the room available; for the last filter of a pipeline it
  is exactly the decoded chunk size. Errors: -1 bad data, -2 output would
  not fit, -3 input ends early (as `fastnc/_inflate.py`).
- A codec that cannot sensibly be a numba kernel (or loses to a C
  library) may instead provide `decode_py(data, out_nbytes, cd) -> bytes
  or ndarray` (plain Python, should release the GIL inside); the engine
  runs those on a thread pool.
- Filter ids (HDF5): 1 deflate (`_inflate.zlib_inflate`), 2 shuffle,
  3 fletcher32, 4 szip, 5 n-bit, 6 scale-offset, 307 bzip2, 32000 lzf,
  32001 blosc, 32004 lz4, 32008 bitshuffle, 32013 zfp, 32015 zstd,
  32026 blosc2.
  Parameters as hdf5plugin / HDF5 write them (see their filter sources).
- Each codec is tested byte-for-byte against the reference decoder on the
  real files' chunks (where present), `tests/data/filters.h5` and
  generated edge cases (empty, tiny, incompressible, max sizes).

## 2. Backend variables (netCDF3 and HDF5 alike)

The netCDF/h5py-like front end (`Dataset`, `Variable`, `Group`, `File`)
talks to a format backend through these objects:

```python
class BackendFile:
    root: BackendGroup
    data_model: str          # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT_OFFSET', 'NETCDF3_64BIT_DATA',
                             # 'NETCDF4', 'NETCDF4_CLASSIC' (as netCDF4.Dataset.data_model)
    def close(self): ...

class BackendGroup:
    name: str                # '/' for the root
    dimensions: dict         # name -> (size, is_unlimited), in netCDF's order
    variables: dict          # name -> BackendVariable, in netCDF's (varid) order
    groups: dict             # name -> BackendGroup
    attrs: dict              # name -> value, exactly as netCDF4-python returns it

class BackendVariable:
    name: str
    dimensions: tuple        # dimension names
    shape: tuple
    dtype: numpy.dtype       # native byte order (str for NC_STRING)
    attrs: dict              # name -> value, exactly as netCDF4-python's getncattr returns it
    chunks: tuple or None    # None: contiguous
    filters: dict            # as netCDF4.Variable.filters()
    endian: str              # 'little' / 'big' / 'native' as netCDF4.Variable.endian()
    def read(self, key, out=None):
        """Raw values (no mask, no scale) for a numpy basic index `key`
        (tuple of ints / slices with steps, normalised by the front end:
        one entry per dimension, ints and slices only, slices with
        start/stop/step resolved, step > 0). Returns a native-endian ndarray
        shaped as numpy would for that key; writes into `out` if given."""
```

Attribute values as netCDF4-python returns them: text (NC_CHAR) -> `str`;
a numeric attribute of length one -> numpy scalar of its type; longer ->
1-d numpy array; NC_STRING -> `str` or list of `str`.

## 3. Mask and scale

`fastnc/_maskscale.py`:

```python
def apply(raw, attrs, dtype, mask=True, scale=True, always_mask=True,
          chartostring=True, var_dtype=None, xp=numpy):
    """netCDF4-python's Variable.__getitem__ post-processing of raw values
    `raw` (what read() returned) for a variable with attributes `attrs`
    and stored dtype `dtype`: _Unsigned, _FillValue / default fill values,
    missing_value, valid_min / valid_max / valid_range, scale_factor /
    add_offset with their dtype promotion, masked array vs plain array,
    char -> str with _Encoding. Same values, dtypes, masks and fill values
    as netCDF4-python 1.7.x."""
```

A fused single-pass kernel (numba, threads) for the common cases, and a
cupy version giving identical results on the GPU (`xp=cupy`).

## 4. Public API (what tests are written against)

```python
import fastnc
ds = fastnc.Dataset(path)                 # netCDF4.Dataset, read-only
ds.variables / dimensions / groups / ncattrs() / getncattr(n) / ds.attr
ds.data_model, ds.file_format, ds.path, ds['grp/var'], ds.close(), with-block
v = ds.variables['T']
v[...] / v[0, :, 3:10]                    # mask+scale on by default, as netCDF4
v.set_auto_maskandscale(b), set_auto_mask(b), set_auto_scale(b),
v.set_always_mask(b), set_auto_chartostring(b)
v.dimensions, shape, dtype, datatype, ndim, size, ncattrs(), getncattr(n),
v.filters(), chunking(), endian(), group(), name

f = fastnc.File(path)                     # h5py.File, read-only
f['grp/ds'] -> h5py-like Dataset: shape, dtype, chunks, compression,
               compression_opts, shuffle, fletcher32, fillvalue, attrs,
               ds[()], ds[...], ds[key], read_direct(out, sel)
f.keys() / items() / visit / visititems / attrs, Group objects

xarray.open_dataset(path, engine='fastnc')
GPU: fastnc.Dataset(path, device='gpu') or v.read(key, device='gpu')
     -> cupy arrays equal to the CPU result.
```
