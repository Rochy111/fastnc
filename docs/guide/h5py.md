# h5py's API

`fastnc.File` reads HDF5 files, netCDF-4 files included, through h5py's
`File`, read-only. It returns raw values with h5py's dtypes, and h5py's
link and attribute order.

```python
import fastnc

with fastnc.File('VJ102IMG_NRT.A2026265.2006.021.nc') as f:
    print(list(f), dict(f.attrs))
    d = f['observation_data/I05']
    print(d.shape, d.dtype, d.chunks, d.compression, d.compression_opts, d.fillvalue)
    window = d[3000:3512, 3000:3512]
```

## What is there

- **Files and groups:** `File(path_or_url, mode='r')`, `keys()`, `items()`,
  `get()` with `getclass` and `getlink`, `visit()`, `visititems()`, paths,
  soft, hard and external links, and `attrs`.
- **Datasets:**
  - `shape`, `maxshape`, `dtype`, `ndim`, `size`, `nbytes`
  - storage: `chunks`, `compression`, `compression_opts`, `shuffle`,
    `fletcher32`, `scaleoffset`, `fillvalue`, `external`, `is_virtual`,
    `virtual_sources()`
  - `read_direct()`, `len()`, iteration, `numpy.asarray(d)`
  - indexing as h5py does it, field names of compound types included
  - not yet: `dims` (dimension scales are read for `fastnc.Dataset`)
- **Datatypes:** compound, enum, array, variable-length strings and
  sequences, opaque and references (object and region), with h5py's dtype
  metadata (`h5py.check_dtype` style).
- **Storage layouts:**
  - virtual datasets: same file, other files, unlimited, `%b` names
  - external raw storage
  - every chunk index: B-tree v1 and v2, fixed and extensible arrays,
    a single chunk, implicit

## Differences from h5py

- **Read-only.** Creating groups, datasets or attributes raises an error.
- **No h5py import.** Objects are fastnc's own: `isinstance(x, h5py.Dataset)`
  is false. Code that checks types should duck-type or check
  `fastnc.File`'s classes.
- **Speed.** Reads decompress on every core. h5py's chunk cache settings
  are accepted and ignored: fastnc keeps its own cache of decoded chunks
  for small repeated reads.
