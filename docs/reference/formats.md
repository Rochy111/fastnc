# Supported formats

## Files

| Format | Read through | Notes |
|---|---|---|
| netCDF-3 classic | `Dataset`, xarray | record variables, every type |
| netCDF-3 64-bit offset | `Dataset`, xarray | |
| netCDF-3 64-bit data (CDF-5) | `Dataset`, xarray | the unsigned and 64-bit integer types |
| netCDF-4 and netCDF-4 classic | `Dataset`, `File`, xarray | groups, user types, phony dimensions |
| HDF5 | `Dataset`, `File`, xarray | any HDF5 file, including HDF-EOS5 and files with a user block |
| HDF4 (SD) | `Dataset`, xarray | the scientific data sets, as netCDF-c shows them; all of them, names taken twice included, by `hdf4_datasets()`; contiguous, chunked, linked and external storage; deflate, RLE, N-bit, skipping Huffman and szip |

## HDF5 storage

- **Layouts:** compact, contiguous, chunked, virtual (same file, other
  files, unlimited, `%b` names) and external raw storage. A remote file's
  external storage and virtual dataset sources are fetched from beside its
  URL, over the same connection settings.
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
| bitshuffle | 32008 | with LZ4 or zstd |
| zfp | 32013 | |
| zstd | 32015 | |
| JPEG | 32019 | as libjpeg decodes with its defaults (libjpeg-turbo's and IJG 6b's colours; IJG's libjpeg 7–9 upsample chroma a little differently); baseline images, as the filter writes them |
| blosc2 | 32026 | dictionaries (zstd, LZ4) and variable-length blocks included |

## Not supported

- writing any file: fastnc reads only
- progressive, lossless and 12-bit JPEG (the JPEG filter writes none)
