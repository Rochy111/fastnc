# How fastnc works

A map of the code for anyone changing it. The README says what fastnc
does and how fast; `docs/measurements.md` says why it is built this way;
`docs/contracts.md` holds the interfaces between the parts.

## Modules

| Module | What it holds |
|---|---|
| `_dataset` | The netCDF4-python front end: `Dataset`, `Group`, `Variable`, `Dimension`; `Group.load()` (many variables in one pass) |
| `_h5py` | The h5py front end: `File`, `Group`, `Dataset`, `AttributeManager`, `Datatype`, `Reference`, `Empty` |
| `_xarray` | `engine='fastnc'`: a data store over `_dataset`, xarray's CF decoding on top; `preload=True` |
| `_nc4` | netCDF-4 as netCDF-c reads it out of HDF5: dimension scales, `_Netcdf4Dimid`, `DIMENSION_LIST`, phony dimensions, hidden attributes, NC_CHAR / NC_STRING, compounds with `align=True` |
| `_nc3` | netCDF-3 classic, 64-bit offset, CDF-5 |
| `_hdf4` | HDF4 SD files (as netCDF-c's HDF4 dispatch presents them) |
| `_hdf5` | HDF5 from its bytes: superblocks, object headers, groups (symbol tables, compact and dense links), attributes, datatypes, fill values, filter pipelines, every chunk index |
| `_h5read` | Raw values of one HDF5 dataset for an index (chunked, contiguous, compact, vlen) and `read_many` |
| `_chunks` | The chunk engine: planning a selection over a chunk table, the batch kernel, the zlib path used until numba is loaded |
| `_select` | numpy / netCDF4 indexes reduced to boxes, then post-indexing |
| `_inflate` | Deflate / zlib in numba |
| `_lzf`, `_lz4`, `_zstd`, `_blosc`, `_blosc2`, `_bitshuffle`, `_szip`, `_scaleoffset`, `_nbit`, `_fletcher32`, `_bzip2`, `_shuffle` | The other HDF5 filters, each `decode(src, p, pend, dst, o, oend, cd, ws)` |
| `_maskscale` | netCDF4-python's masking and scaling, numpy rules and a fused numba kernel; `_maskscale_gpu` the same on the card |
| `_gpu` | Reads straight to cupy arrays |
| `_pool` | The thread pool, `share()` and `set_threads()` |
| `_jit`, `_intrin` | numba imported on first use; unaligned word loads/stores and an atomic counter |

## Opening a file

`Dataset(path)` maps the file read-only and looks at its first bytes:
`CDF\x01/\x02/\x05` is netCDF-3 (`_nc3`), the HDF5 signature at 0, 512,
1024, ... is HDF5 (`_nc4` over `_hdf5`), `\x0e\x03\x13\x01` is HDF4.
Nothing else is read in: values stay in the mapping until asked for.

`_hdf5.H5File` parses metadata in Python from a `memoryview` of the
mapping with `struct.unpack_from`: object headers on demand (cached by
address), a group's links when first listed, attributes indexed by name
first and decoded one at a time, a dataset's layout / filters / fill value
when it is first read. `_nc4.NC4File` then builds netCDF's view in netCDF-c's
order -- that order decides phony dimension names and dimension ids:

1. per group, links in creation order (when tracked) or name order:
   dimension scales become dimensions as met (ids from `_Netcdf4Dimid`,
   else the next free id), other datasets become variables, committed
   datatypes become user types, subgroups are read after the group's links;
2. a second pass over the whole tree, children before parents, gives each
   variable its dimensions: `_Netcdf4Coordinates`, else `DIMENSION_LIST`
   (object references to scales anywhere in the file), else phony
   dimensions (`phony_dim_<id>`) -- an existing dimension of the group with
   the same length and unlimitedness that the variable does not use yet,
   or a new one;
3. an unlimited dimension is as long as the longest variable along it,
   worked out once (h5netcdf recomputes it on every question: xarray took
   10 s to open a WRF file through it).

## Reading a chunked dataset

`Variable[key]` → `_select.normalize` reduces the key to a box (start,
stop, step > 0 per dimension) and what numpy does afterwards (drop integer
dimensions, reverse negative steps, pick integer/boolean sequences).
`_chunks.read_batch` takes one or many (dataset, box, output) items:

1. **Plan** (numpy, no Python per chunk): the dataset's chunk table --
   parsed once into arrays of offsets, addresses, sizes, filter masks --
   is intersected with the box dimension by dimension, giving for each
   chunk its first selected element, counts, place in the output, and the
   last decoded byte the selection needs. Chunks missing from the table
   (never written) read as the fill value: the output is pre-filled only
   when the selection touches one.
2. **Kernel** (`_work`, numba, nogil): the caller and the pool's threads
   each run it, taking chunks off an atomic counter, biggest first. Per
   chunk: the filter pipeline in reverse (fletcher32 strips its checksum,
   deflate inflates into the thread's scratch, stopping at the last byte
   the selection needs unless a shuffle spreads every byte plane over the
   chunk), then `_place` copies the selected elements into the output,
   undoing the byte shuffle and swapping big-endian to native in the same
   pass while the chunk is still in cache. A chunk that needs no
   transform and lands contiguously in the output is inflated straight
   there.
3. **Batches**: `read_many` / `Group.load()` put every variable's chunks
   in one task list over one output buffer, so a file of many small
   variables keeps every core busy (wrfout d01: 200 variables, 4 chunks
   each; per variable the cores waited on each one's slowest chunk).

Small reads use fewer threads (a thread per ~32 kB of output at most).

## numba, and the first read of a process

Importing numba and loading the cached kernels takes ~0.8 s once per
process. `_jit` defers the import until a kernel runs; `_chunks` goes
further: until the kernels are loaded, chunks are decoded with `zlib` on
the pool (zlib releases the GIL; it is the reference decoder, so values
are the same), and the first read bigger than 1 MB starts loading numba
in a background thread. The kernel is warmed with the argument types reads
use -- a read-only array for the mapped file (numba types read-only arrays
apart: warming with a writable one compiled the real call again).
`fastnc.warm()` loads everything up front (worker pools:
`ProcessPoolExecutor(initializer=fastnc.warm)`).

numba's cache can serve a stale kernel after an edit; delete
`fastnc/__pycache__/*.nbi *.nbc` before timing.

## Inflate

`_inflate.inflate` follows libdeflate: a 64-bit bit buffer refilled
without branches (`buf |= ld64(src, p) << bc; p += (63 - bc) >> 3; bc |=
56` leaves at least 56 bits), one table lookup per symbol (11-bit
literal/length and 8-bit distance tables with subtables; an entry packs
the code length, kind, extra-bit count and base), and a fast loop while 16
input and 274 output bytes remain: no bounds checks per symbol, up to
three literals per refill, 16-byte match copies, word fills for distance-1
runs. Byte-identical to zlib on every chunk of the real files; 1.0-1.5
GB/s on one core where `zlib.decompress` does 0.4-0.6. Adler-32 is not
checked.

## The GPU

`device='gpu'` reads return cupy arrays. Decompressing deflate on the
card loses (nvCOMP, measured: `tools/profile/nvcomp_deflate.py`), so the
CPU cores decode into a reused page-locked buffer band by band along the
first dimension, each band uploads asynchronously while the next decodes,
and the byte swap and mask/scale run on the card: PCIe carries the packed
integers, not the floats they become.

## Checks

`tools/check/` compares against the reference libraries directly (they
need h5py, netCDF4, xarray): `hdf5_vs_h5py.py` (parser), `read_vs_h5py.py`
(values), `meta_vs_netcdf4.py` (netCDF metadata and values),
`h5py_api.py`, `xarray_vs_netcdf4.py`. `tests/` runs without them, on
`tests/data` and stored reference digests.

## Things the reference libraries do that fastnc copies on purpose

- netCDF4-python builds compound dtypes with `align=True` from the
  members' netCDF types; a fixed-length string member becomes `S1`, keeping
  its first byte. Its padding bytes are uninitialized: compare compounds
  field by field.
- netCDF-c reads one-byte fixed strings as NC_CHAR (`S1`) and longer
  ones as NC_STRING (`str`).
- h5py gives variable-length strings as `bytes` from datasets and `str`
  from attributes; an enum of FALSE/TRUE over int8 as `bool`; a compound
  of two equal floats named r and i as complex.
- HDF5's own `chunk_iter` misreports the offsets of extensible-array
  chunks when the unlimited dimension is not the first; fastnc's mapping is
  checked against the chunks' contents instead.

Where fastnc differs on purpose: `Variable.filters()` reports zstd /
bzip2 / blosc when the file uses them (netCDF4-python reports False when
its plugin is not loadable); values come back in native byte order (as
netCDF4-python; h5py keeps the stored order -- the values are equal).
