# Changelog

All notable changes. Values read by fastnc stay bit-identical to
netCDF4-python's and h5py's in every release; a change that alters them is
a bug.

## Unreleased

### Added
- **Documentation site** at https://rochy111.github.io/fastnc/: getting
  started, a guide to each front end, remote files, the GPU and threads,
  supported formats, accuracy, settings, every benchmark and a gallery of
  real files.
- **CI on every push to `dev`**, and releases from tags: a `v*` tag runs
  the tests again on every system and publishes the GitHub release from
  this file.

### Fixed
- **Scalar variables read with `v[:]`.** `fastnc.Dataset` raised
  `IndexError: too many indices` for `v[:]`, `v[0]`, `v[0:1]` or `v[[0]]`
  on a 0-d variable (GOES ABI files read their calibration constants this
  way); it now gives what netCDF4-python gives for each, masked or not,
  and refuses what it refuses. `fastnc.File` refuses every index but `()`
  and `...` on a scalar dataset with h5py's `ValueError`.

## 0.1.0 (2026-11-02)

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
  a VIIRS variable 60.8 -> 3.7 ms. libdeflate without its checksum: 4-7%.
- **Wrong results fixed:** a netCDF-4 variable with fewer records than its
  unlimited dimension read at its stored length (a smaller array, an
  IndexError past it) where netCDF4-python pads with fill values, and on the
  GPU raised or gave uninitialized memory as values. All read paths now match
  netCDF4-python. Records short along two unlimited dimensions are the one
  exception: netCDF-C returns uninitialized memory there; fastnc gives the
  stored values in place and fill values elsewhere.
- Opening is lazy: a variable looked up by name reads only that variable, and
  shapes, dimensions and the other variables follow when asked for. Open +
  one variable of WRF d01 8.7 -> 5.0 ms, a window 7.4 -> 3.8; whole-file reads
  gain nothing from this, the work only moves. Two things change: a corrupt
  variable's header raises when it is first used, not at open, and metadata
  not yet asked for can no longer be read after `close()`.
- `load()` masks and scales all arrays in one pass of the pool, with no size
  limit to tune: threads move on to whichever array still has work. Attributes
  are decoded on demand, and variables with the same masking attributes share
  a plan. Masked WRF d01 44.5 -> 32.7 ms, WRF d02 64.5 -> 52.8, MODIS 1 km
  60.5 -> 50.5. Fusing mask/scale with decoding starts at the CPU's last cache,
  read from the machine (`fastnc.hardware()['last_cache_bytes']`).
- A fresh process starts reading sooner: threads start without waiting for
  each, `json` is no longer imported to read the tuning file, and chunks
  decoded before numba loads cost less Python each (GOES-19 C02 164 -> 148
  ms).
- On Windows with mixed core types, a thread that opens a file or reads now
  declares its work as the user's. A script's main thread was moved to an
  efficiency core ~2 s in, and every read after that took 1.6x as long
  (WRF d01 34 -> 55 ms); the benchmark tables, whose threads were protected,
  never showed it.
- Masked reads prove as they decode that nothing is masked, from the byte
  plane of top values in shuffled chunks (or word by word as netCDF-3 runs
  are swapped), and skip the pass over the values: WRF d01 whole file 52 ->
  44 ms, WRF d02 in a loop 90 -> 74, VIIRS 105 -> 95.
- HDF5 null dataspaces through `fastnc.Dataset`: a masked read gives None as
  a raw one does, where it raised and took `load()` of the whole file with
  it; a GPU read gives None, where it returned uninitialized memory as a
  value. `fastnc.File`'s `__array__` and `read_direct` refuse them as h5py
  does.
- Types netCDF has no default fill value for (float16, HDF5 arrays) read
  masked: nothing is masked by default, and an explicit `_FillValue`, range
  or scale still applies. A KeyError there also ended `load()`.
- A closed file's mapping goes at `close()`: pool threads no longer keep it
  alive (on Windows it could not be deleted until the next read).
- **The kernels without numba.** Their machine code is linked into a file of
  fastnc's own, per machine, and later processes call it with numpy and ctypes:
  no numba, no llvmlite, no compiler, at full speed from the first read. A
  fresh process reading a whole file: WRF d02 1004 -> 248 ms, VIIRS 539 -> 399,
  GOES-19 C02 `Rad` 517 -> 353, MODIS 1 km 338 -> 276. The file is built in the
  background; x86-64 Windows and Linux; numba as before elsewhere, or with
  `FASTNC_IMAGE=0`.
- A kernel call through the image holds the GIL for under 2 us (numba's
  dispatcher: 16): each array goes by its address and the kernel's own entry
  reads and checks it, so a thousand small calls from 24 threads no longer queue.
- A batch's chunk tables are read and planned by one kernel, not a dozen numpy
  calls per variable, and its tables are built in one step. Compared with the
  code before these changes, processes alternating: WRF d01 `load()` 25.7 -> 21.9
  ms, masked 31.4 -> 27.4, Ku-band 7.5 -> 6.4, AVHRR masked variable 2.0 -> 1.8.
- Going through a dataset a piece at a time (`for i in range(n): v[i]`): a read
  in a chunk already decoded needs no plan, planning goes straight to the chunks
  a box touches, and a read moving on to the next chunk has the chunks after it
  decoded ahead, on every core. 1,600 rows of VIIRS 139 -> 46 ms, of GOES-19 C13
  132 -> 45, 885 of the Ku-band file 98 -> 14 (h5py: 41, 35, 13).
- Small reads of uncompressed data, contiguous or chunked, go straight to their
  bytes: rows of Metop-SG 99 -> 6 us each (h5py 4.4), AWIPS 91 -> 6.
- Each dataset now has a function of its own for small reads (`_quick`):
  integers and plain slices become bytes in the mapped file or in a decoded
  chunk, in one chunk or across several. A row or a pixel takes 1-2 us against
  h5py's 2-2.5. `fastnc.Dataset` goes through it too: rows 58 -> 26 us on AWIPS,
  57 -> 19 on Ku-band (netCDF4 105, 144).
- Fixed: chunks decoded ahead were not remembered as the way a read was going,
  so the run started over at the end of each one, decoding a chunk twice and
  never more than one ahead. 1,600 rows of VIIRS 49 -> 14 ms, of GOES-19 C13
  48 -> 9 (h5py 43, 34); 2,000 points of C13 73 -> 14 (h5py 53).
- A link is found by name, as the library finds one, not by reading a group's
  whole list: `f['T2']` on WRF 645 -> 47 us (h5py 67). Open, one small variable
  and close now beat h5py on all the files tried: WRF 399 us against 555.
- Fixed, from reading all 465 of the HDF Group's test files and comparing each
  dataset with h5py: a read of unstored data in a dataset of one chunk gave that
  chunk's bytes, not the fill value; fixed strings kept their padding spaces or
  the bytes after a NUL; variable-length values in compounds and array types
  came back as raw bytes; array types failed when big or nested; committed
  datatypes of the old format and partial edge chunks stored unfiltered could not
  be read; the last virtual chunk of a contiguous dataset could reach past the
  end of the file. HDF5 2.0's complex numbers are read, and a dataspace in shared
  object header messages says so instead of being misread as a shape.
- Time series at many places: when two reads running want a small part of the
  same many chunks, the region they span is decoded once and kept, within what
  the machine's output pool may hold, until the file is closed or
  `fastnc.release()`. 30 series of a year of OISST 2115 -> 195 ms (netCDF4 112 s,
  h5py 120 s), of CMIP6 983 -> 120 (94 s, 77 s), of NCEP 138 -> 13.
- Going through a variable a chunk at a time: chunks decoded ahead are as many
  as this machine's memory allows, not three at a pass on 24 threads, and go to
  the read that wants all of one with no copy. 365 time steps of CMIP6, File,
  527 -> 150 ms; of OISST 915 -> 230. Chunks of a closed file are let go at once.
- A dataset's chunk index is looked into, as the library does, not read whole
  for a read of a few chunks (all four kinds: v1 and v2 B-trees, fixed and
  extensible arrays), so a first read no longer grows with the chunks in the
  file: a million chunks 61 -> 0.065 ms in a v1 B-tree, 1805 -> 0.068 in a v2
  (h5py 0.10). Past a share of the chunks, the table is made after all, and for
  the newer kinds it is made in bulk: a read of all of a million, v2, 1934 -> 190
  ms (h5py 6300).
- Whole variables through xarray: masking floats, the only decoding most need,
  happens in the array read, not in two copies of it. OISST 281 -> 146 ms, CMIP6
  223 -> 105 (netcdf4 engine 4775, 3334).
- Files of many tiny variables (GLM: 48, a few kB each): whole small datasets are
  decoded on the spot, a dataset's function is made at its second read, and a
  name index is asked as a dict. Dataset 10.2 -> 8.3 ms, File 4.5 -> 4.0 (netCDF4
  11.8, h5py 5.1).
- Fixed: a group inside itself no longer stops `Dataset` (netCDF-C overflows its
  stack on such a file), and array members of old-format compounds can be read.
- Region references: `File` gives them as h5py does, `dset[ref]` the values and
  `file[ref]` the dataset, as NOAA's VIIRS and CrIS SDR files use them.
- Page bits of extensible arrays run on from one data block to the next: read as
  a bitmap each, half the chunks of a big array looked unwritten, and came back as
  fill values. Unreleased, and what the first run of `tools/check.py` caught.
- Read now: external links through `File` (followed beside the file, as the
  library does), shared object header messages (SOHM), VAX floats, floats wider
  than a double, integers of 3, 5, 6 and 7 bytes (which h5py cannot read), more
  than 8 filters on a dataset, chunks that went without an optional filter
  fastnc does not have, soft links to soft links through `Dataset`, HDF4 szip
  (swapped parameters, and no length in front of the stream), and the BitGroom and
  Granular BitRound filters, which round what is written and undo nothing:
  checked against the HDF Group's printed output.
- Fixed: big-endian enums through `File` came out with their bytes reversed;
  a shuffle goes by the element size it says, not the type's (PyTables,
  variable-length data); netCDF-3 names end at a NUL inside their count (a
  variable was not found under its name); floats of odd layouts in attributes.
- Memory: `close()` really unmaps the file, so it can be deleted at once, and
  a read after it says the file is closed; the working set after a loop over
  OISST 605 -> 140 MB. Output blocks, kept regions and chunks over budget go after
  5 s unused (they were kept for good, up to 4 GB). Read-ahead goes by cores:
  a loop over time steps peaks at 379 MB, not 649.
- `tools/check.py`: the tests in parallel, saved answers of the reference
  libraries, and timed reads against the last commit, in about a minute.
- Small chunks read here and there are decoded on the spot, and chunks of any
  filter go through the cache, not only shuffled ones: h5py's plain
  `compression='gzip'` does not shuffle. 2,000 random pixels of a million chunks
  130 -> 25 ms (h5py 22).
- Fixed: fixed and extensible arrays of chunks kept in pages read their pages
  never written as chunks at address zero, giving the file's own bytes for what
  are fill values; big extensible arrays could not be read at all. Fill values of
  big-endian datasets came out with their bytes reversed.
- A read of a few chunks is planned in plain integers: 70 us of numpy gone from
  each plan. A small variable read over and over is kept from its second read,
  and a few small chunks from their first: 30 random crops of WRF 2.0 -> 0.95 ms.
- `fastnc.File` loops read ahead as `Dataset` loops do, following the order of
  the loop (`visititems` goes by name, iterating a group by creation): WRF d01
  79 -> 43 ms, RCM 10 -> 4.7, Metop-SG 33 -> 21, AVHRR 11.5 -> 7.6.
- Threads for work at memory speed (masks, copies) follow what it is: fewer for
  small arrays, at most one per fast core. Scanning 1 MB took 148 us on 24
  threads, 33 on one. A Metop-SG window 1.18 -> 0.94 ms (h5py 1.4), AVHRR
  masked variable 1.9 -> 1.7.
- Editing plain Python around the kernels no longer recompiles them: numba's
  cache goes by the kernels and their constants, not the file's time.
- A fresh install no longer stops to compile in the middle of a read (~40 s,
  and more for each new type): that happens in a process of its own, and reads
  carry on without kernels meanwhile.
- A path through groups reads those groups alone, not the whole file: Metop-SG,
  Dataset, a window 4.6 -> 2.7 ms, one variable 5.3 -> 2.9. xarray opening a
  root that holds only groups 4.4 -> 0.7 ms (h5netcdf 2.2). The arguments of a
  kernel call are worked out once, not by each thread; a window through
  `fastnc.File` stays about level with h5py's, a little behind (1.6-1.9 ms
  against 1.4-1.5).
- GPU reads of up to one band start no upload thread: WRF `T` 5.5 -> 5.1 ms,
  AVHRR 2.5 -> 2.1, level with or ahead of reading on the CPU and uploading.
- Compressed chunks on storage that pays per request (a spinning disk, a
  network share) are read in a few big requests, found by timing three pages,
  not faulted in 64 kB at a time: 4,750 requests -> 160 for VIIRS.
- The zfp filter (32013, H5Z-ZFP): floats, doubles, 32 and 64-bit integers, 1 to
  4 dimensions, fixed rate, precision and accuracy, reversible and expert modes,
  in numba and with no plugin, bit for bit what H5Z-ZFP decodes.
- Thirty series at places near each other keep the area around them, not the
  whole variable: CMIP6 1,261 -> 79 MB held, 108-179 -> 70 ms; OISST 1,514 -> 95
  MB, 184 -> 153 ms. Places all over keep everything, as before.
- netCDF-3 and HDF4 reach what HDF5's reads did. Faster than netCDF4-python on all
  355 rows measured (18 lost before, nearly all HDF4): AMSR-E whole file 100.9 ->
  11.3 ms (netCDF4 42.9), a DEM's 30 series 52.8 -> 5.1 (116.3), WRF opened 20
  times 14.0 -> 3.0 (55.7). HDF4's `close()` now unmaps.
- Threads' scratch goes when reads stop, with what else is kept: a process left
  idle after a loop over OISST holds 36 MB (224 before; h5py 44).
- Output blocks are the process's own memory, not a section of the page file:
  first writes to a GB 101 -> 43 ms on 24 threads, 531 -> 169 on one.
- Big-endian values are swapped a word at a time: uncompressed chunks 9.0 -> 1.35
  ms (h5py 3.2). A masked small read 30 -> 22 us (netCDF4 95).
- Variable-length values are packed by a kernel: 200,000 strings 247 -> 25 ms (h5py
  34), numbers 48 -> 4.8 (13). ASCII strings through `File` keep their bytes.
- Records one at a time 79 -> 2.3 us (h5py 36), through `Dataset` 209 -> 17 (netCDF4
  106). Chars masked 12 -> 1.4 ms for 1.2 MB (25).
- Read-ahead follows a stride, not only the next chunk: every 2nd step of CMIP6 370
  -> 109 ms, every 6th 124 -> 46.
- Threads sharing an open file no longer decode the same chunks: 8 threads over 240
  steps of CMIP6 decoded 610 chunks, 346 of them again, and now about 280, almost none
  again. Each thread keeps its own read-ahead, so sharing a file costs nothing against
  a file each.
- A row across chunks works out its pieces once for a loop: `x[i]` across five
  compressed chunks 5.4 -> 3.9 us (h5py 3.7). A kernel call of eight arguments 2.36 ->
  2.17 us.
- The kernel image is safer, from what porting it to fastgrib showed. An array of
  any class (a memmap) goes to a kernel by its address: a one-element one went as
  its value, an address to crash at. Arrays numba would not take (byte-swapped,
  unaligned) are not given to a kernel compiled for others. A failed allocation in
  a kernel raises `MemoryError`, and what a kernel raises is numba's own exception,
  not a `RuntimeError`. Growing buffers are in its runtime. A kernel needing one of
  numba's helpers is left to numba, not linked to address 0. The index has a CRC,
  and a forked process no longer moves the file position its parent reads by.
- `Dataset`, `File` and what is in them pickle, opening the file again where they are
  unpickled, so xarray datasets opened with fastnc work with dask's distributed
  scheduler and multiprocessing. Objects reached by a reference know their name.
- Windows over plain chunks by a kernel with nothing to plan: Metop-SG 256 x 256 417
  -> 120 us (h5py 890). Lists, boolean masks, steps and Ellipsis take the same ways:
  a list of 30 times from contiguous data 5.0 -> 0.02 ms (h5py 0.22).
- `close()` leaves a file free at once on Windows, not at the next collection: 111
  of 200 files could not be moved until then.
- A kernel image that cannot be built no longer goes unnoticed, or is built again
  by every process: one bug in the build had left this machine on numba for hours,
  and every filter's driver is now in a fresh install's image.
- From fuzzing against h5py and netCDF4 (over 12 million reads), wrong values fixed:
  threads looking chunks up could read another chunk's values; big-endian complex
  attributes and fills; FALSE/TRUE enums wider than a byte, also in compounds;
  char variables' `_FillValue`; virtual datasets wrapping narrowing conversions
  where HDF5 clips; big-endian numbers in arrays of compounds with variable-length
  members; filters in other than the usual order; a closed `Dataset` giving what
  it had decoded ahead, or a netCDF-3 one reopening its name; old-format files
  whose offsets and lengths differ in width. Now read: empty list selections,
  datasets shrunk to no chunks, array-typed attributes, filters that grow what
  they code, vlens with 4-byte lengths, every shared message, user blocks of 4 kB
  and more through `Dataset`.
- Packaging (`pyproject.toml`), MIT license, a CI workflow for Linux /
  Windows / macOS, Python 3.9-3.13, numpy 1 and 2, with and without the
  kernels. (Its runs so far: Windows green throughout; Linux and macOS
  showed numpy-version differences in masked arrays' fill_value types and
  NaN payloads, fixed since; its latest runs were cut short by the
  account's Actions minutes, so the full suite was run locally on Windows
  under numpy 2.5 and 1.26.)

### Known gaps
- Filters that need decoders fastnc has not got: JPEG, and blosc2's
  variable-length blocks. A metadata cache image in a file; reading only.
  (HDF5 virtual datasets and external storage are read.)
