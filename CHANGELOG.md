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
- **Fresh processes** read without numba until reading has taken 1 s
  (libdeflate / zlib through ctypes on every core, numpy for mask/scale
  and for unfiltered chunk grids): the first masked read of a WRF
  variable 715 -> 222 ms, a whole VIIRS file 1176 -> 613 ms (netCDF4-python:
  3306).
- **Readahead across variables** read whole in the file's order (xarray's
  load, loops over `variables`): 1.8-3x.
- **Mask/scale fused with decoding** for big scaled variables; **GPU
  reads pipelined** (one decode, each band uploaded as soon as it is in).
- **xarray decoding by lookup table** for 8 / 16-bit data, fused with
  decoding (GOES-19 C02 `.load()` 810 -> 160 ms).
- Word-at-a-time unshuffle, a decoded-chunk cache for repeated small reads
  of shuffled data, fixed-Huffman tables kept between blocks and
  libdeflate-style table builds, faster whole-dataset plans; without the
  kernels, per-chunk work vectorized and many small deflated chunks
  staged (Ku-band 199 -> 32 ms).
- **Adapts to the machine**: `fastnc.hardware()`, `fastnc.tune()`; threads
  follow the CPUs a process may use (affinity, container quotas), memory
  budgets the RAM available, block sizes the L2 cache, and numba's load
  point the load time measured there.
- Threads are chosen by the decoding to do, not the output: one column of
  a VIIRS variable 60.8 -> 3.7 ms. netCDF-3 runs copied and swapped in one
  pass out of the mapping (WRF d02 56 -> 46 ms, again on an open file 55 ->
  31). libdeflate without its checksum: 4-7%.
- Packaging (`pyproject.toml`), MIT license, a CI workflow for Linux /
  Windows / macOS, Python 3.9-3.13, numpy 1 and 2, with and without the
  kernels. (Its runs so far: Windows green throughout; Linux and macOS
  showed numpy-version differences in masked arrays' fill_value types and
  NaN payloads, fixed since; its latest runs were cut short by the
  account's Actions minutes, so the full suite was run locally on Windows
  under numpy 2.5 and 1.26.)

### Known gaps
- HDF4 szip, zstd dictionaries, blosc2 variable-length blocks; reading
  only. (HDF5 virtual datasets and external storage are read.)
