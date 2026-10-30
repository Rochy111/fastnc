# Changelog

All notable changes. Values read by fastnc stay bit-identical to
netCDF4-python's and h5py's in every release; a change that alters them is
a bug.

## Unreleased

### Added
- **Readers.** netCDF-3 (classic, 64-bit offset, CDF-5), netCDF-4 and plain
  HDF5 (every chunk index, superblock and group/attribute storage), HDF4
  SD files; HTTP(S), S3 and file-object sources with parallel byte-range
  requests.
- **Front ends.** `fastnc.Dataset` (netCDF4-python's API, read-only:
  groups, dimensions, variables, attributes, user types, automatic
  masking and scaling), `fastnc.File` (h5py's API, read-only),
  `xarray.open_dataset(..., engine='fastnc')` (with `preload=True`),
  `Group.load()` for many variables in one pass, `device='gpu'` reads
  returning cupy arrays.
- **Filters.** deflate, shuffle, fletcher32, szip, scale-offset, n-bit,
  bzip2, lzf, lz4, bitshuffle, zstd, blosc, blosc2 -- all in numba, each
  byte-identical to the reference decoder.
- **Speed.** A libdeflate-style inflate (1.0-1.5 GB/s on one core),
  chunks decoded on every core straight into place with unshuffle and
  byte swap fused, one big chunk inflated on many threads, contiguous data
  as virtual chunks, early stop on partial reads, output memory reused
  between reads, mask/scale in a fused multithreaded kernel (and on the
  GPU).
- **Tests.** Every variable and HDF5 object of the test corpus, the
  synthetic corpus and the real WRF / satellite files compared with what
  netCDF4-python and h5py read, values and metadata, from digests stored
  in `tests/data/references_*.json.gz`; per-part tests for each codec, the
  netCDF-3 and HDF4 readers, mask/scale, the arena, remote reads.
- Packaging (`pyproject.toml`), MIT license, CI on Linux / Windows / macOS,
  Python 3.9-3.13, numpy 1 and 2.

### Known gaps
- HDF5 virtual datasets and external storage, HDF4 szip, zstd
  dictionaries, blosc2 variable-length blocks; reading only.
