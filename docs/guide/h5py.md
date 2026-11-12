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
  `get()` with `getclass` and `getlink` (`HardLink`, `SoftLink`,
  `ExternalLink` objects), `visit()`, `visititems()`, `visit_links()`,
  `visititems_links()`, `require_group()` of an existing group, paths,
  soft, hard and external links, `attrs` (with `attrs.get_id(name)`: an
  attribute's `dtype`, `shape` and `get_storage_size()` without reading
  it), and the file's `filename`,
  `mode`, `driver`, `userblock_size`, `libver`, `swmr_mode`.
- **Datasets:**
  - `shape`, `maxshape`, `dtype`, `ndim`, `size`, `nbytes`
  - storage: `chunks`, `compression`, `compression_opts`, `shuffle`,
    `fletcher32`, `scaleoffset`, `fillvalue`, `external`, `is_virtual`,
    `virtual_sources()`
  - `read_direct()`, `len()` and `d.len()`, iteration, `numpy.asarray(d)`
  - indexing as h5py does it, field names of compound types included
  - `d.asstr()[...]` (strings decoded), `d.astype(t)[...]` (converted as
    HDF5 converts: to a narrower integer, floats truncated and anything out
    of range saturated; NaN as numpy has it), `d.fields(names)[...]`
  - `d.iter_chunks(sel)`, `d.ref` (also on groups: `f[d.ref]` is `d`),
    `d.is_scale`
  - `d.dims`: each dimension's `label` and attached scales, by position or
    name (`d.dims[0][0]`, `d.dims[0].keys()`, `.values()`, `.items()`)
  - `d.memmap()` (fastnc's own): a contiguous dataset's bytes in place, a
    read-only array over the file's mapping in h5py's dtype, no copy
  - not there: the low-level `d.id` API
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
- **Indexing takes more.** Negative steps, decreasing or repeated lists,
  boolean lists and a list on more than one dimension give what numpy gives
  for them where h5py raises. Every index both accept gives the same values
  (checked over the test files' datasets); a float index raises
  `TypeError` in both.
