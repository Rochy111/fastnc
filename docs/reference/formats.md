# Supported formats

## Files

| Format | Read through | Notes |
|---|---|---|
| netCDF-3 classic | `Dataset`, xarray | record variables, every type |
| netCDF-3 64-bit offset | `Dataset`, xarray | |
| netCDF-3 64-bit data (CDF-5) | `Dataset`, xarray | the unsigned and 64-bit integer types |
| netCDF-4 and netCDF-4 classic | `Dataset`, `File`, xarray | groups, user types, phony dimensions |
| HDF5 | `Dataset`, `File`, xarray | any HDF5 file, including HDF-EOS5 and files with a user block |
| HDF4 (SD) | `Dataset`, xarray | the scientific data sets, as netCDF-c shows them |

## HDF5 storage

- **Layouts:** compact, contiguous, chunked, virtual (same file, other
  files, unlimited, `%b` names) and external raw storage.
- **Chunk indexes:** B-tree v1 and v2, fixed array, extensible array,
  single chunk and implicit, with unlimited dimensions anywhere.
- **Metadata:** superblocks 0 to 3, old and new style groups (symbol
  tables, link messages, dense storage with fractal heaps), dense attribute
  storage, shared object header messages, soft, hard and external links,
  committed datatypes.
- **Datatypes:** integers and floats of every size and byte order
  (including unusual float formats), strings of fixed and variable length,
  compound, enum, array, variable-length sequences, opaque, and object and
  region references.

## Filters

Every filter decodes to the same bytes as its reference library, and all
of them run in numba on every core:

| Filter | HDF5 id | Notes |
|---|---:|---|
| deflate (gzip) | 1 | fastnc's own inflate, 1.0–1.5 GB/s on one core; one big chunk inflates on many cores |
| shuffle | 2 | undone in the same pass as the copy into place |
| fletcher32 | 3 | checked |
| szip | 4 | |
| n-bit | 5 | |
| scale-offset | 6 | |
| LZF | 32000 | |
| blosc | 32001 | with its internal codecs |
| bzip2 | 307 | |
| LZ4 | 32004 | |
| bitshuffle | 32008 | with LZ4 |
| zfp | 32013 | |
| zstd | 32015 | |
| blosc2 | 32026 | |

## Not supported yet

- the JPEG filter
- zstd dictionaries
- blosc2's variable-length blocks and dictionaries
- szip in HDF4 files
- external storage next to a remote file
- writing any file: fastnc reads only
