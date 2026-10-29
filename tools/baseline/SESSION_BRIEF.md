# Build a fast netCDF / HDF reader -- session brief

Prepared 2026-10-28 by `prepare_session.py` (re-run it any time: `"C:\Users\Roch\miniconda3\envs\bench-wrf\python.exe" prepare_session.py`).
You are starting a new project in `C:\Users\Roch\Desktop\WRF\fastnc`. Read this whole brief first.

## The goal

Build a standalone Python library (working name **fastnc** -- ask the user if
they want another) that reads **netCDF3** (classic, 64-bit offset, CDF-5),
**netCDF4 / netCDF4-classic** and **plain HDF5** -- and, as stretch goals,
**HDF4** (MODIS-era NASA products), HDF-EOS and Zarr/NCZarr -- much faster
than today's readers (netCDF4-python, h5py, xarray), with **genuine** speedups:
better algorithms, less copying, parallel decompression of independent chunks.
Not "more power wins".

This repeats, for netCDF/HDF, what was done for GRIB in
`C:\Users\Roch\Desktop\WRF\fastgrib` (public at github.com/Rochy111/fastgrib):
from 15 ms to 1.25 ms a GFS field on one core, bit-identical to wgrib2. Read
its `README.md`, `CONTRIBUTING.md` and `docs/internals.md` before writing
code: the conventions, the tests-against-stored-reference approach and the
hard-won rules there apply here unchanged.

## The rule that is not negotiable

**Values must be bit-identical to the reference readers.** Raw values equal
h5py / netCDF4-python with auto mask+scale off, byte for byte. Decoded values
(mask and scale on) equal netCDF4-python's: `_FillValue`, `missing_value`,
`valid_min/max/range`, `scale_factor`/`add_offset` with *their* dtype
promotion, `_Unsigned`, the masked-array vs NaN choice -- every rule, in the
same float type. A faster reader that is off by one ulp, or masks one point
differently, is a bug. `baseline.json` holds per-variable value digests from
every reader for every corpus file: they are the first tests.

## What to do, in order (as fastgrib was done)

1. **Measure first.** `BASELINE.md` (below) has today's numbers on this
   machine. Profile where the time goes before changing anything: open
   (metadata, B-tree walks), chunk I/O, decompression, shuffle/unshuffle,
   copies into the output array, mask/scale. Compare readers on the *same*
   file: when this brief was first made, xarray took 10-20x longer than
   netCDF4-python / h5py on a compressed netCDF4 wrfout (h5py 0.48 s,
   xarray netcdf4 5.7 s, xarray h5netcdf 10.7 s), while uncompressed netCDF3
   already read at ~2 GB/s -- find out why before deciding where to aim.
2. **Design the reader** -- the big decisions, each settled by measurement:
   - netCDF3 is uncompressed: map the file and view it (record variables are
     interleaved: one strided copy, byte-swapped big-endian -> native in the
     same pass). Expect near memory-bandwidth speed.
   - netCDF4/HDF5: the speedup is in **decompressing independent chunks in
     parallel** and writing each straight into its place in the output.
     `zlib.decompress` releases the GIL, so a thread pool works; libdeflate /
     isa-l / zlib-ng inflate faster than zlib -- measure. Shuffle, fletcher32,
     scale-offset, n-bit: numba kernels. Getting chunk addresses: h5py's
     `get_chunk_info` / `read_direct_chunk` is the quick start; a pure-Python
     parser of the HDF5 superblock, object headers and chunk indexes (v1/v2
     B-trees, fixed/extensible arrays, single chunk) removes the h5py
     dependency and per-chunk overhead -- decide by measuring.
   - Filters beyond deflate (szip, zstd, blosc, bzip2, lz4, lzf): support what
     the corpus has, fall back to h5py for what you don't, and say so.
3. **Optimize**, with every change checked against the digests.
4. **Tests** against stored reference outputs (digests or small arrays in
   `tests/data`), needing neither h5py nor netCDF4-python in CI where
   possible; synthetic files for every format/filter combination.
5. **numpy 1 and 2**, Python 3.9-3.13, Linux / Windows / macOS (CI matrix as
   fastgrib's `.github/workflows/tests.yml`).
6. **Docs**: README with benchmarks against netCDF4-python / h5py / xarray
   (machine, method, numbers), CONTRIBUTING, CHANGELOG, internals.
7. **An xarray backend** (`engine='fastnc'`) once reading is right: that is
   how most users would meet it.
8. **Git from the start**; a GitHub repo and releases only when the user says
   so. **Never publish to PyPI** unless the user asks.

## Lessons from fastgrib -- read these, they cost days

Measuring:
- **Hybrid CPU on Windows** (this i7-13700K, 8P+8E): a script from a terminal
  is background work and its main thread lands on E-cores; single-core
  timings swing 30-50%. Call a `fast_thread()` (in `prepare_session.py`,
  from fastgrib's `_cpu._fast_thread`) on the timing thread and the worker
  threads.
- `time.process_time()` ticks every 15.6 ms on Windows: CPU time only means
  anything over runs longer than a second.
- The tool shell sets `PYTHONDONTWRITEBYTECODE=1`: every import recompiles;
  `unset` it before timing imports or cold start.
- **Page faults are serialized on Windows** (~0.45 ms per 4 MB of fresh
  memory, on any number of threads): time `read(out=preallocated)` to see the
  decoder alone, and don't allocate large scratch per chunk on every core.
- The OS file cache: first read vs later reads differ wildly; report both
  and say which.
- Alternate A/B/A/B and look at the spread, not one run.

numba:
- `@njit(cache=True, nogil=True)`; **no `inline='always'`** (left refcounting
  in hot loops, 10x slower under numba 0.63); unsigned indices (`np.uint64`)
  in hot loops; explicit loops, not slice assignment; branch-free where data
  is unpredictable.
- **No `parallel=True` / `prange`**: numba's threads spin after each loop
  (70 ms of CPU for a 3 ms read). Use your own pool: the caller and helpers
  take items off a shared atomic counter until none is left.
- numba's cache can serve a stale kernel after an edit: delete
  `__pycache__/*.nbi *.nbc` before timing.
- **Import numba lazily** (fastgrib `_jit.py`): listing a file should not pay
  ~250 ms for numba.
- Each new process's first decode pays ~0.5 s (numba import + loading cached
  kernels): provide `warm()`, and document `ProcessPoolExecutor(initializer=
  fastnc.warm)` for worker pools.

Python and numpy:
- **numpy 2 / NEP 50**: arithmetic on numpy uint8/int16 scalars stays in that
  type and wraps (`b[p] * 100`). Read header fields as Python ints
  (`memoryview(buf).cast('B')`, `struct.unpack_from`). Run the suite on numpy
  1.26 with `np._set_promotion_state('weak_and_warn')` to find every site.
- **Fork safety**: a thread pool created before `fork()` is inherited without
  its threads -- workers hang. `os.register_at_fork(after_in_child=...)`
  resets it (fastgrib test: `test_forked_workers`).
- Python 3.9 has no `int.bit_count()`; macOS has no `/proc/meminfo`.

## Working with the user

- Communicate progress plainly: what you measured, what you changed, the
  before/after numbers, what you could not verify.
- Ask before downloading large files, installing packages into existing
  environments, creating the GitHub repo, pushing, or anything public.
- The user's data folders are read-only for this project: never write into
  them; put generated files under `C:\Users\Roch\Desktop\WRF\fastnc`.

## This machine

```
python         3.11.15
platform       Windows-10-10.0.26200-SP0
cpu            Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
logical_cpus   24
numpy          1.26.4
netCDF4        1.7.4
netcdf_c       4.9.3
hdf5_netcdf    1.14.6
python_exe     C:\Users\Roch\miniconda3\envs\bench-wrf\python.exe
h5py           3.16.0
xarray         2025.6.1
h5netcdf       1.8.1
scipy          1.17.1
numba          0.59.1
pyhdf          None
hdf5plugin     None
zarr           None
numcodecs      None
isal           None
zlib_ng        None
deflate        None
hdf5_h5py      2.0.0
```

Use `C:\Users\Roch\miniconda3\envs\bench-wrf\python.exe` (conda env with netCDF4, h5py, xarray, numba).

## The corpus (`corpus.json`)

Real files sampled from: `E:\GPUWRF\runs`, `E:\WRF\STATIONARY\WRF_OUT\run_2025-10-25_18_00_00`
plus generated files in `corpus/synthetic/` (every layout and filter this
machine's libraries can write).

Files by format:
- HDF5 (netCDF4): 15
- HDF5: 8
- netCDF3 64-bit offset: 3
- netCDF3 CDF-5 (64-bit data): 1
- netCDF3 classic: 1

Variable layouts: record 286, chunked 280, contiguous 40
Filters seen: deflate 192, shuffle 180, fletcher32 6, szip 6, lzf 3, bzip2 3

Not generated (missing libraries or unsupported here):
- nc4_zstd: RuntimeError: NetCDF: Filter error: undefined filter encountered: (variable 'T', group '/')
- nc4_blosc_lz: RuntimeError: NetCDF: Filter error: undefined filter encountered: (variable 'T', group '/')
- nc4_blosc_lz4: RuntimeError: NetCDF: Filter error: undefined filter encountered: (variable 'T', group '/')
- nc4_blosc_lz4hc: RuntimeError: NetCDF: Filter error: undefined filter encountered: (variable 'T', group '/')
- nc4_blosc_zlib: RuntimeError: NetCDF: Filter error: undefined filter encountered: (variable 'T', group '/')
- nc4_blosc_zstd: RuntimeError: NetCDF: Filter error: undefined filter encountered: (variable 'T', group '/')
- hdf4_*: pyhdf not installed (pip/conda install pyhdf to cover HDF4)

zstd and blosc: netCDF4-python ships their plugins (`netCDF4/plugins/`,
which this script puts on `HDF5_PLUGIN_PATH`), but in this env the C library
still reports "undefined filter" -- most likely the pip wheel's plugins and
conda's HDF5 DLLs not matching. A clean env (conda-forge `netcdf4` +
`hdf5plugin`) should cover them, and `pyhdf` adds HDF4: **ask the user before
creating or changing environments.** Both filters appear in real archives
(ERA5 on ARCO, CMIP, Zarr stores), so cover them before calling the reader
done.

## Baseline (`baseline.json`, best of 3 reads after a first)

Time to open the file and read every variable completely.

| file | format | raw MB | reader | first s | best s | MB/s |
|---|---|---|---|---|---|---|
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | netCDF4-python raw | 0.352 | 0.352 | 2276 |
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | netCDF4-python mask+scale | 0.492 | 0.490 | 1637 |
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | scipy.io.netcdf | 0.653 | 0.501 | 1601 |
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | xarray scipy | 1.693 | 0.366 | 2194 |
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | xarray netcdf4 | 0.384 | 0.384 | 2086 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | netCDF4-python raw | 0.155 | 0.155 | 2012 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | netCDF4-python mask+scale | 0.193 | 0.193 | 1622 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | scipy.io.netcdf | 0.234 | 0.213 | 1466 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | xarray scipy | 0.143 | 0.143 | 2182 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | xarray netcdf4 | 0.178 | 0.178 | 1751 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | netCDF4-python raw | 0.557 | 0.549 | 368 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | netCDF4-python mask+scale | 0.603 | 0.603 | 335 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | h5py | 0.499 | 0.499 | 405 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | xarray h5netcdf | 10.521 | 10.521 | 19 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | xarray netcdf4 | 6.135 | 5.979 | 34 |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | netCDF4-python raw | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | h5py | 0.023 | 0.022 | 295 |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.026 | 0.025 | 258 |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | xarray netcdf4 | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | netCDF4-python raw | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | h5py | 0.022 | 0.021 | 302 |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.025 | 0.025 | 255 |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | xarray netcdf4 | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | netCDF4-python raw | 0.024 | 0.023 | 277 |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | 0.025 | 0.025 | 262 |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | h5py | 0.021 | 0.021 | 305 |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.025 | 0.025 | 255 |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | xarray netcdf4 | 0.112 | 0.105 | 62 |
| h5_contiguous.h5 | HDF5 | 6.5 | netCDF4-python raw | 0.004 | 0.004 | 1591 |
| h5_contiguous.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | 0.005 | 0.005 | 1326 |
| h5_contiguous.h5 | HDF5 | 6.5 | h5py | 0.002 | 0.002 | 2893 |
| h5_contiguous.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.005 | 0.005 | 1196 |
| h5_contiguous.h5 | HDF5 | 6.5 | xarray netcdf4 | 0.080 | 0.080 | 81 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | netCDF4-python raw | 0.013 | 0.012 | 549 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | 0.013 | 0.013 | 498 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | h5py | 0.010 | 0.010 | 631 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.015 | 0.015 | 445 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | xarray netcdf4 | 0.093 | 0.090 | 72 |
| h5_lzf.h5 | HDF5 | 6.5 | netCDF4-python raw | error: RuntimeError: NetCDF: Filter error: undefined filter encount | | |
| h5_lzf.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | error: RuntimeError: NetCDF: Filter error: undefined filter encount | | |
| h5_lzf.h5 | HDF5 | 6.5 | h5py | 0.009 | 0.007 | 879 |
| h5_lzf.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.011 | 0.011 | 577 |
| h5_lzf.h5 | HDF5 | 6.5 | xarray netcdf4 | error: RuntimeError: NetCDF: Filter error: undefined filter encount | | |
| h5_single_chunk.h5 | HDF5 | 6.5 | netCDF4-python raw | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_single_chunk.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_single_chunk.h5 | HDF5 | 6.5 | h5py | 0.023 | 0.022 | 287 |
| h5_single_chunk.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.026 | 0.026 | 249 |
| h5_single_chunk.h5 | HDF5 | 6.5 | xarray netcdf4 | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_szip.h5 | HDF5 | 6.5 | netCDF4-python raw | 0.032 | 0.031 | 206 |
| h5_szip.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | 0.033 | 0.032 | 200 |
| h5_szip.h5 | HDF5 | 6.5 | h5py | 0.031 | 0.030 | 218 |
| h5_szip.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.034 | 0.034 | 191 |
| h5_szip.h5 | HDF5 | 6.5 | xarray netcdf4 | 0.114 | 0.112 | 58 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | netCDF4-python raw | 0.007 | 0.007 | 1306 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | netCDF4-python mask+scale | 0.009 | 0.009 | 938 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | scipy.io.netcdf | 0.006 | 0.006 | 1447 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | xarray scipy | 0.011 | 0.007 | 1237 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | xarray netcdf4 | 0.010 | 0.010 | 908 |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | netCDF4-python raw | 0.007 | 0.007 | 1326 |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | netCDF4-python mask+scale | 0.009 | 0.009 | 938 |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | scipy.io.netcdf | error: ValueError: Unexpected header. | | |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | xarray scipy | error: ValueError: Unexpected header. | | |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | xarray netcdf4 | 0.009 | 0.009 | 959 |
| nc3_classic.nc | netCDF3 classic | 8.8 | netCDF4-python raw | 0.007 | 0.006 | 1355 |
| nc3_classic.nc | netCDF3 classic | 8.8 | netCDF4-python mask+scale | 0.010 | 0.010 | 916 |
| nc3_classic.nc | netCDF3 classic | 8.8 | scipy.io.netcdf | 0.006 | 0.006 | 1427 |
| nc3_classic.nc | netCDF3 classic | 8.8 | xarray scipy | 0.011 | 0.007 | 1257 |
| nc3_classic.nc | netCDF3 classic | 8.8 | xarray netcdf4 | 0.009 | 0.009 | 948 |
| nc4_bzip2.nc | HDF5 | 7.4 | netCDF4-python raw | 0.218 | 0.214 | 34 |
| nc4_bzip2.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.217 | 0.217 | 34 |
| nc4_bzip2.nc | HDF5 | 7.4 | h5py | 0.218 | 0.214 | 34 |
| nc4_bzip2.nc | HDF5 | 7.4 | xarray h5netcdf | 0.226 | 0.225 | 33 |
| nc4_bzip2.nc | HDF5 | 7.4 | xarray netcdf4 | 0.300 | 0.288 | 26 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | netCDF4-python raw | 0.007 | 0.005 | 1371 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.006 | 0.006 | 1260 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | h5py | 0.004 | 0.003 | 2309 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | xarray h5netcdf | 0.015 | 0.015 | 504 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | xarray netcdf4 | 0.083 | 0.083 | 89 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | netCDF4-python raw | 0.017 | 0.015 | 599 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | netCDF4-python mask+scale | 0.018 | 0.018 | 482 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | h5py | 0.013 | 0.013 | 658 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | xarray h5netcdf | 0.036 | 0.035 | 249 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | xarray netcdf4 | 0.147 | 0.147 | 60 |
| nc4_contiguous.nc | HDF5 | 7.4 | netCDF4-python raw | 0.006 | 0.006 | 1284 |
| nc4_contiguous.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.006 | 0.006 | 1311 |
| nc4_contiguous.nc | HDF5 | 7.4 | h5py | 0.003 | 0.003 | 2358 |
| nc4_contiguous.nc | HDF5 | 7.4 | xarray h5netcdf | 0.015 | 0.014 | 520 |
| nc4_contiguous.nc | HDF5 | 7.4 | xarray netcdf4 | 0.082 | 0.082 | 90 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | netCDF4-python raw | 0.016 | 0.015 | 506 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.015 | 0.015 | 479 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | h5py | 0.014 | 0.013 | 560 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | xarray h5netcdf | 0.025 | 0.025 | 299 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | xarray netcdf4 | 0.092 | 0.088 | 84 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | netCDF4-python raw | 0.011 | 0.009 | 797 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.011 | 0.010 | 708 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | h5py | 0.010 | 0.008 | 895 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | xarray h5netcdf | 0.024 | 0.021 | 359 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | xarray netcdf4 | 0.086 | 0.085 | 87 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | netCDF4-python raw | 0.014 | 0.014 | 530 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.015 | 0.015 | 494 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | h5py | 0.013 | 0.013 | 583 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | xarray h5netcdf | 0.028 | 0.025 | 292 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | xarray netcdf4 | 0.107 | 0.107 | 69 |
| nc4_szip.nc | HDF5 | 7.4 | netCDF4-python raw | 0.038 | 0.038 | 195 |
| nc4_szip.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.040 | 0.039 | 187 |
| nc4_szip.nc | HDF5 | 7.4 | h5py | 0.037 | 0.036 | 202 |
| nc4_szip.nc | HDF5 | 7.4 | xarray h5netcdf | 0.050 | 0.049 | 149 |
| nc4_szip.nc | HDF5 | 7.4 | xarray netcdf4 | 0.136 | 0.135 | 55 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | netCDF4-python raw | 0.058 | 0.054 | 137 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.056 | 0.054 | 136 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | h5py | 0.049 | 0.047 | 158 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | xarray h5netcdf | 0.067 | 0.063 | 117 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | xarray netcdf4 | 0.139 | 0.133 | 56 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | netCDF4-python raw | 0.014 | 0.012 | 598 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.016 | 0.013 | 570 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | h5py | 0.012 | 0.010 | 751 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | xarray h5netcdf | 0.021 | 0.021 | 356 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | xarray netcdf4 | 0.091 | 0.091 | 81 |
| nc4_zlib1.nc | HDF5 | 7.4 | netCDF4-python raw | 0.013 | 0.012 | 600 |
| nc4_zlib1.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.013 | 0.013 | 576 |
| nc4_zlib1.nc | HDF5 | 7.4 | h5py | 0.011 | 0.011 | 687 |
| nc4_zlib1.nc | HDF5 | 7.4 | xarray h5netcdf | 0.023 | 0.023 | 326 |
| nc4_zlib1.nc | HDF5 | 7.4 | xarray netcdf4 | 0.093 | 0.090 | 82 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | netCDF4-python raw | 0.012 | 0.011 | 652 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.012 | 0.012 | 604 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | h5py | 0.010 | 0.010 | 743 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | xarray h5netcdf | 0.025 | 0.023 | 318 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | xarray netcdf4 | 0.093 | 0.089 | 83 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | netCDF4-python raw | 0.015 | 0.014 | 509 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.016 | 0.016 | 474 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | h5py | 0.011 | 0.010 | 732 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | xarray h5netcdf | 0.024 | 0.024 | 310 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | xarray netcdf4 | 0.099 | 0.099 | 75 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | netCDF4-python raw | 0.013 | 0.013 | 589 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.013 | 0.013 | 548 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | h5py | 0.011 | 0.011 | 677 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | xarray h5netcdf | 0.023 | 0.022 | 329 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | xarray netcdf4 | 0.125 | 0.112 | 66 |
