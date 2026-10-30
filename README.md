# fastnc

A fast reader for netCDF and HDF5 in Python: netCDF-3 (classic, 64-bit
offset, CDF-5), netCDF-4 and plain HDF5 (every chunk index, every common
filter), HDF4, from local files, HTTP(S), S3 or file objects. It reads the
same values as netCDF4-python and h5py, bit for bit, many times faster: on
every core, and optionally straight to the GPU.

```python
import fastnc

ds = fastnc.Dataset('wrfout_d01_2025-10-26_06_20_00.nc')   # netCDF4.Dataset's API
t2 = ds['T2'][0]                                         # masked and scaled as netCDF4-python
everything = ds.load()                                   # every variable, all chunks in one pass

f = fastnc.File('VJ102IMG_NRT.A2026265.2006.021.nc')       # h5py.File's API
i05 = f['observation_data/I05'][3000:3512, 3000:3512]

import xarray as xr
xd = xr.open_dataset('MOD021KM.A2025301.1445.007.2025301204859.nc', engine='fastnc')

rad = fastnc.Dataset('s3://noaa-goes19/ABI-L1b-RadF/2026/302/12/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc')['Rad'][:]
gpu = ds['T'].read(device='gpu')                         # a cupy array, same values
```

Read-only. Switching from netCDF4-python or h5py is an import change.

## Values

Bit-identical to the reference readers, and checked:

- **raw** values equal h5py's and netCDF4-python's with mask and scale off,
  byte for byte (fastnc returns native byte order, as netCDF4-python does);
- **masked / scaled** values follow every rule of netCDF4-python 1.7 --
  `_FillValue` and default fill values, `missing_value`, `valid_min` /
  `valid_max` / `valid_range`, `scale_factor` / `add_offset` with their
  dtype promotion (which differs between numpy 1 and 2), `_Unsigned`, the
  masked array vs plain array choice and its fill value -- in the same
  float type;
- **GPU** reads equal CPU reads;
- **metadata** as each reference gives it: netCDF-c's dimensions (phony
  ones included), variable order, attribute values and types, user types;
  h5py's link and attribute order, dtypes and their metadata.

`tests/test_references.py` compares every variable and HDF5 object of the
test corpus (`tests/data`: every layout, chunk index, group and attribute
storage, datatype and filter), of a synthetic corpus and of real WRF and
satellite files (MODIS, VIIRS, Metop-SG, AVHRR, RCM, a Ku-band
scatterometer) with what netCDF4-python and h5py read, from stored
digests: about 6,900 tests, raw and masked, on numpy 1 and 2 (each test a
variable or object, whole and in three selections). The GOES-19 ABI files
(not in the repository) were checked the same way against both readers
directly, with `tools/check/`.

## Speed

Milliseconds, fastnc's speed-up over each reader in brackets
(`tools/bench.py`, tables by `tools/bench_table.py` from
[docs/benchmarks.json](docs/benchmarks.json)). Method: files in the OS
cache, each read once before timing (flushing the cache needs admin rights
on Windows: cold-cache reads were not measured); best of 5 runs with the
readers alternated run by run; raw reads with mask and scale off; fastnc
on every logical CPU unless a row says 1 thread; "first read" is a fresh
Python process, timed from just before the import, best of three
processes (numba is never loaded there: see below). Every variable of every
file means every variable of every group. The files: the user's own WRF
runs (wrfout d01 netCDF-4 deflate+shuffle; d02 netCDF-3) and satellite
products as distributed (MODIS L1B, VIIRS, Metop-SG VII, AVHRR, RCM, a
Ku-band scatterometer, GOES-19 ABI full disk from NOAA's S3 bucket).

Measured 2026-10-30 04:36 and 2026-10-30 04:44: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 24 logical CPUs, NVIDIA GeForce RTX 5070 Ti; Python 3.12.16, numpy 2.5.4, numba 0.68.0; h5py 3.16.0, netCDF4 1.7.4, xarray 2026.9.0, h5netcdf 1.8.2, cupy 14.2.0.

**Whole file, every variable, raw (ms; fastnc: Dataset.load() and the same loop over variables as netCDF4-python, which its ratio is to)**

| file | fastnc | fastnc loop | netCDF4 | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 37.5 | 52.9 | 488 (9.2x) | 443 (11.8x) |
| WRF d02 netCDF3 | 54.1 | 62.2 | 378 (6.1x) | - |
| MODIS 1 km L1B | 66.3 | 84.2 | 906 (10.8x) | 902 (13.6x) |
| MODIS 250 m L1B | 65.3 | 56.6 | 843 (14.9x) | 835 (12.8x) |
| VIIRS I-band | 58.7 | 74.8 | 1771 (23.7x) | 1613 (27.5x) |
| Metop-SG VII | 23.7 | 32.0 | 105 (3.3x) | 84.8 (3.6x) |
| AVHRR HRPT | 24.3 | 20.4 | 180 (8.9x) | 178 (7.3x) |
| RCM SAR winds | 6.1 | 5.9 | 40.5 (6.8x) | 37.5 (6.2x) |
| Ku scatterometer | 16.1 | 20.1 | 66.5 (3.3x) | 55.6 (3.4x) |
| GOES-19 ABI RadF C02 | 74.1 | 74.1 | 2371 (32.0x) | 2286 (30.8x) |
| GOES-19 ABI RadF C13 | 13.5 | 14.5 | 155 (10.7x) | 145 (10.7x) |

**Whole file, masked and scaled as netCDF4-python does by default (ms)**

| file | fastnc | fastnc loop | netCDF4 |
|---|---|---|---|
| WRF d01 netCDF4 | 91.8 | 76.4 | 535 (7.0x) |
| WRF d02 netCDF3 | 75.0 | 98.0 | 485 (4.9x) |
| MODIS 1 km L1B | 145 | 101 | 1099 (10.9x) |
| MODIS 250 m L1B | 74.3 | 56.3 | 1002 (17.8x) |
| VIIRS I-band | 155 | 109 | 3048 (28.0x) |
| Metop-SG VII | 49.9 | 55.5 | 274 (4.9x) |
| AVHRR HRPT | 32.2 | 20.1 | 210 (10.4x) |
| RCM SAR winds | 11.9 | 11.5 | 45.1 (3.9x) |
| Ku scatterometer | 17.3 | 21.7 | 70.2 (3.2x) |
| GOES-19 ABI RadF C02 | 136 | 120 | 4802 (40.0x) |
| GOES-19 ABI RadF C13 | 36.7 | 20.1 | 303 (15.1x) |

**Open + one variable, raw (ms; fastnc.Dataset vs netCDF4, fastnc.File vs h5py)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 11.1 | 43.0 (3.9x) | 5.3 | 26.2 (4.9x) |
| WRF d02 netCDF3 | 3.2 | 16.9 (5.3x) | - | - |
| MODIS 1 km L1B | 39.6 | 307 (7.7x) | 38.1 | 303 (7.9x) |
| MODIS 250 m L1B | 46.1 | 619 (13.4x) | 47.1 | 615 (13.1x) |
| VIIRS I-band | 13.4 | 170 (12.7x) | 11.9 | 164 (13.8x) |
| Metop-SG VII | 6.3 | 15.3 (2.4x) | 2.3 | 3.7 (1.6x) |
| AVHRR HRPT | 2.3 | 16.6 (7.1x) | 1.7 | 14.4 (8.2x) |
| RCM SAR winds | 1.7 | 5.0 (2.9x) | 0.85 | 2.1 (2.5x) |
| Ku scatterometer | 3.7 | 16.4 (4.4x) | 2.3 | 10.8 (4.7x) |
| GOES-19 ABI RadF C02 | 76.4 | 1989 (26.0x) | 66.8 | 1957 (29.3x) |
| GOES-19 ABI RadF C13 | 12.1 | 125 (10.3x) | 13.9 | 120 (8.6x) |

**Open + one variable, masked and scaled (ms)**

| file | fastnc | netCDF4 |
|---|---|---|
| WRF d01 netCDF4 | 11.6 | 44.4 (3.8x) |
| WRF d02 netCDF3 | 4.0 | 20.5 (5.2x) |
| MODIS 1 km L1B | 43.6 | 346 (7.9x) |
| MODIS 250 m L1B | 54.4 | 694 (12.7x) |
| VIIRS I-band | 15.0 | 321 (21.4x) |
| Metop-SG VII | 6.7 | 21.9 (3.3x) |
| AVHRR HRPT | 4.1 | 21.6 (5.3x) |
| RCM SAR winds | 2.4 | 5.2 (2.2x) |
| Ku scatterometer | 4.5 | 16.6 (3.7x) |
| GOES-19 ABI RadF C02 | 103 | 3987 (38.6x) |
| GOES-19 ABI RadF C13 | 20.3 | 244 (12.0x) |

**Open + a window of it (ms)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 9.4 | 21.6 (2.3x) | 3.9 | 5.8 (1.5x) |
| WRF d02 netCDF3 | 1.7 | 4.1 (2.4x) | - | - |
| MODIS 1 km L1B | 20.0 | 113 (5.6x) | 25.8 | 109 (4.2x) |
| MODIS 250 m L1B | 9.9 | 116 (11.8x) | 11.8 | 114 (9.7x) |
| VIIRS I-band | 3.1 | 10.5 (3.4x) | 1.9 | 7.0 (3.8x) |
| Metop-SG VII | 5.1 | 11.6 (2.3x) | 1.1 | 1.2 (1.0x) |
| AVHRR HRPT | 1.6 | 5.2 (3.1x) | 1.3 | 3.1 (2.5x) |
| RCM SAR winds | 1.6 | 4.8 (2.9x) | 0.83 | 1.9 (2.3x) |
| Ku scatterometer | 3.1 | 8.6 (2.7x) | 1.6 | 3.4 (2.2x) |
| GOES-19 ABI RadF C02 | 7.1 | 54.3 (7.7x) | 4.1 | 49.0 (12.0x) |
| GOES-19 ABI RadF C13 | 3.2 | 11.0 (3.4x) | 1.9 | 6.8 (3.6x) |

**xarray.open_dataset(...).load(), by engine (ms)**

| file | fastnc | netcdf4 | h5netcdf | scipy |
|---|---|---|---|---|
| WRF d01 netCDF4 | 96.5 | 559 (5.8x) | 9665 (100.2x) | - |
| WRF d02 netCDF3 | 92.6 | 406 (4.4x) | - | 370 (4.0x) |
| MODIS 1 km L1B | 4.2 | 7.4 (1.8x) | 20.1 (4.8x) | - |
| MODIS 250 m L1B | 3.3 | 6.1 (1.9x) | 15.8 (4.9x) | - |
| VIIRS I-band | 1.8 | 4.7 (2.6x) | 4.2 (2.3x) | - |
| Metop-SG VII | 4.5 | 11.2 (2.5x) | 1.9 (0.4x) | - |
| AVHRR HRPT | 37.0 | 227 (6.1x) | 242 (6.5x) | - |
| RCM SAR winds | 15.3 | 53.7 (3.5x) | 104 (6.8x) | - |
| Ku scatterometer | 26.0 | 81.3 (3.1x) | 993 (38.2x) | - |
| GOES-19 ABI RadF C02 | 146 | 4703 (32.1x) | 4787 (32.7x) | - |
| GOES-19 ABI RadF C13 | 30.6 | 311 (10.2x) | 356 (11.6x) | - |

**First read in a fresh process: import + open + one variable, raw (ms)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 156 | 216 (1.4x) | 147 | 223 (1.5x) |
| WRF d02 netCDF3 | 146 | 186 (1.3x) | - | - |
| MODIS 1 km L1B | 213 | 476 (2.2x) | 208 | 500 (2.4x) |
| MODIS 250 m L1B | 222 | 794 (3.6x) | 234 | 822 (3.5x) |
| VIIRS I-band | 163 | 346 (2.1x) | 163 | 381 (2.3x) |
| Metop-SG VII | 150 | 192 (1.3x) | 141 | 203 (1.4x) |
| AVHRR HRPT | 144 | 186 (1.3x) | 143 | 210 (1.5x) |
| RCM SAR winds | 139 | 174 (1.3x) | 137 | 194 (1.4x) |
| Ku scatterometer | 147 | 186 (1.3x) | 143 | 206 (1.4x) |
| GOES-19 ABI RadF C02 | 296 | 2150 (7.3x) | 300 | 2141 (7.1x) |
| GOES-19 ABI RadF C13 | 161 | 300 (1.9x) | 163 | 321 (2.0x) |

**First read in a fresh process, masked and scaled (the default) (ms)**

| file | fastnc | netCDF4 |
|---|---|---|
| WRF d01 netCDF4 | 162 | 215 (1.3x) |
| WRF d02 netCDF3 | 156 | 190 (1.2x) |
| MODIS 1 km L1B | 241 | 520 (2.2x) |
| MODIS 250 m L1B | 262 | 872 (3.3x) |
| VIIRS I-band | 198 | 501 (2.5x) |
| Metop-SG VII | 158 | 193 (1.2x) |
| AVHRR HRPT | 156 | 193 (1.2x) |
| RCM SAR winds | 146 | 174 (1.2x) |
| Ku scatterometer | 155 | 191 (1.2x) |
| GOES-19 ABI RadF C02 | 514 | 4076 (7.9x) |
| GOES-19 ABI RadF C13 | 189 | 421 (2.2x) |

**One variable, masked and scaled, to a cupy array (ms)**

| file | fastnc gpu | fastnc cpu+upload | netCDF4+upload |
|---|---|---|---|
| WRF d01 netCDF4 | 12.2 | 13.9 (1.1x) | 45.6 (3.7x) |
| WRF d02 netCDF3 | 8.3 | 9.4 (1.1x) | 26.7 (3.2x) |
| MODIS 1 km L1B | 63.3 | 57.9 (0.9x) | 374 (5.9x) |
| MODIS 250 m L1B | 102 | 98.0 (1.0x) | 729 (7.2x) |
| VIIRS I-band | 30.3 | 44.9 (1.5x) | 354 (11.7x) |
| Metop-SG VII | 8.9 | 10.1 (1.1x) | 25.7 (2.9x) |
| AVHRR HRPT | 5.7 | 9.8 (1.7x) | 27.8 (4.9x) |
| RCM SAR winds | 2.9 | 2.8 (1.0x) | 5.8 (2.0x) |
| Ku scatterometer | 6.4 | 5.8 (0.9x) | 17.5 (2.7x) |
| GOES-19 ABI RadF C02 | 86.0 | 226 (2.6x) | 4128 (48.0x) |
| GOES-19 ABI RadF C13 | 36.1 | 50.5 (1.4x) | 282 (7.8x) |

**fastnc, whole file raw: every core vs one (ms)**

| file | all threads | 1 thread |
|---|---|---|
| WRF d01 netCDF4 | 37.5 | 200 (5.3x) |
| WRF d02 netCDF3 | 54.1 | 142 (2.6x) |
| MODIS 1 km L1B | 66.3 | 485 (7.3x) |
| MODIS 250 m L1B | 65.3 | 470 (7.2x) |
| VIIRS I-band | 58.7 | 710 (12.1x) |
| Metop-SG VII | 23.7 | 40.9 (1.7x) |
| AVHRR HRPT | 24.3 | 81.2 (3.3x) |
| RCM SAR winds | 6.1 | 10.4 (1.7x) |
| Ku scatterometer | 16.1 | 25.8 (1.6x) |
| GOES-19 ABI RadF C02 | 74.1 | 1103 (14.9x) |
| GOES-19 ABI RadF C13 | 13.5 | 72.7 (5.4x) |

"fastnc loop" is fastnc used as a drop-in: the same loop over variables
as netCDF4-python's column (readahead decodes the following variables
together); `Dataset.load()` decodes them all in one pass.

Whole benchmark runs differ by up to 2x in places on this machine, more
than best-of-5 within a run evens out, so each cell is the lower of two
full runs, for every reader alike (`tools/bench_merge.py`). Some rows still
read slow inside the benchmark process: run on their own, MODIS 1 km
open + one variable takes 21 ms and its `load()` 43 ms (the tables: 40 and
66).

The GPU pays where values are many and scaled (VIIRS, MODIS, GOES: the
packed integers cross PCIe while the CPU decodes, float32 is made on the
card): 1.2-2.5x over reading on the CPU and uploading the floats. For
small or unscaled variables it is even, and for the small arrays of the
Ku-band, Metop-SG and RCM files it loses (0.8-0.9x). Decompressing deflate on the card was measured and loses on every
file (nVIDIA's nvCOMP: 142 ms for VIIRS where the CPU takes ~75 ms, 4.3 s
for MODIS where it takes 0.1 s), so the CPU always decompresses. The first
GPU call of a process takes ~9 s (CUDA and cupy starting), not counted
above.


### Why it is fast

- **Decompression on every core, straight into place.** Chunks are taken
  off a shared counter by every core in one numba kernel, inflated into a
  per-thread buffer and copied into the output with HDF5's byte shuffle
  undone and big-endian swapped in the same pass, while the chunk is in
  cache -- no per-chunk Python, no intermediate arrays. h5py and
  netCDF4-python decompress one chunk at a time on one core.
- **Its own inflate**, libdeflate-style (1.0-1.5 GB/s on one core on real
  data, where zlib does 0.4-0.6), and every other HDF5 filter (szip,
  scale-offset, n-bit, bzip2, lzf, lz4, bitshuffle, zstd, blosc, blosc2)
  in numba, at or above the C libraries' speed.
- **One big chunk on many cores.** MODIS keeps a band in 1-3 chunks of 33
  MB: a single deflate stream each. fastnc decodes such a stream on every
  core at once (speculatively from guessed block starts, each part
  verified against where the one before it ended).
- **Partial reads stop early**: a window decompresses only up to its last
  row. Shuffled chunks (no early stop possible) read again and again in
  small parts -- a loop over points or levels -- come from a 32 MB cache
  of decoded chunks, as HDF5's own chunk cache does (WRF `T[0,0]` again:
  3.8 ms -> 92 us).
- **Unshuffle a word at a time**: each 8 bytes of every byte plane
  interleaved with shifts that LLVM vectorizes, 15-25 GB/s a core
  instead of 3.
- **Many variables, one pass.** `Dataset.load()` and xarray's
  `preload=True` decode every variable's chunks together, biggest first:
  a WRF file's 200 variables no longer wait on each other.
- **Readahead across variables.** Code that reads whole variables one
  after another in the file's order -- xarray's `.load()`, a loop over
  `ds.variables` -- gets the next ones decoded together in a window (32
  MB, doubling to 256 MB while the reads keep to the order; dropped when
  they do not): a WRF variable of four chunks alone keeps four cores busy.
  xarray's load of wrfout d01 152 -> 86 ms, of the Ku-band file 80 -> 31.
- **xarray's decoding by table.** For 8- and 16-bit packed data (GOES,
  VIIRS, MODIS radiances) xarray's own mask / scale functions are applied
  once to every possible raw value -- at most 65,536 -- and the decoded
  array is a lookup into that table, done as each chunk is decoded: the
  same functions on the same values, so the same bits, without xarray's
  several passes over the array. xarray `.load()`: GOES-19 C02 810 -> 160
  ms (engine='netcdf4': ~5000), VIIRS I-band 417 -> 110.
- **Mask and scale while decoding.** For big scaled variables each chunk
  becomes float32 and a mask while still in cache, so the raw integers
  never go to memory and back (GOES-19 C02 140 -> ~122 ms).
- **Memory reused.** Windows faults in fresh pages slowly (55 ms per GB);
  big outputs reuse already-mapped memory once earlier arrays are gone.
- **Metadata read lazily** from the file's bytes in Python, and netCDF's
  view built once: xarray's netcdf4 engine spent 7 s opening a WRF file in
  an environment where netCDF4-python's plugins fail to load (a 36-48 ms
  probe per variable), h5netcdf 10 s (its unlimited dimensions are
  recomputed on every question); fastnc 30 ms.
- **The GPU** does what it is good at: decompressing deflate on the card
  loses (nvCOMP measured slower than the CPU on every file here), so the
  CPU cores decode the whole selection in one pass and each 8 MB band
  crosses PCIe the moment its last chunk is placed -- the upload hides
  behind the decoding (GOES-19 C02 to the card: 90 ms raw where the CPU
  read alone takes 82) -- and masking and scaling to float32 run on the
  card (masked: 95 ms, against 120 masked on the CPU).
- **A fresh process never waits for numba.** Importing numba and loading
  the kernels takes 0.7-0.9 s (most of it Windows opening ~300 modules)
  and holds the GIL; loaded in the background it slowed the first read
  threefold. So reads start without it -- chunks inflated by libdeflate or
  zlib through ctypes straight into place on every core, chunk grids of
  uncompressed data gathered by numpy in one go, mask/scale by numpy
  block by block on the pool -- and the kernels load only once reading
  without them has taken a second in all (or on `fastnc.warm()`). A
  script that reads a few variables never pays for them.

Where it is no faster: a file of a few tiny variables (opening costs about
the same as the references'), and a fresh process reading little (both
pay ~150 ms for numpy's import; fastnc's own import is ~40 ms).

## Install

```
pip install -e ".[xarray]"      # add "gpu" for cupy (CUDA 12)
```

Python 3.9+, numpy 1.22+ (1.x and 2.x), numba 0.57+. No other dependency;
h5py, netCDF4-python and hdf5plugin are needed only to regenerate the test
references.

## Use

| | |
|---|---|
| `fastnc.Dataset(path_or_url)` | netCDF4.Dataset: `variables`, `dimensions`, `groups`, attributes, `set_auto_maskandscale` and friends, `filters()`, `chunking()`, `endian()`, user types |
| `fastnc.File(path_or_url)` | h5py.File: groups, datasets, `attrs`, `chunks`, `compression`, `fillvalue`, `read_direct`, `visititems` |
| `Group.load(names=None, recursive=False)` | many variables decoded in one pass |
| `Variable.read(key, device='gpu')` | values as a cupy array (masked results: data and mask on the card, `.get()` for numpy) |
| `xr.open_dataset(p, engine='fastnc', preload=False)` | xarray, same Dataset as `engine='netcdf4'` |
| `fastnc.set_threads(n)`, `FASTNC_THREADS` | how many threads (default: every logical CPU) |
| `fastnc.warm()` | load numba and the kernels now (else they load once reading without them has taken a second); for worker pools: `ProcessPoolExecutor(initializer=fastnc.warm)` |
| `fastnc.release()` | free the memory kept between reads |
| `storage_options={...}` | remote: `region`, `endpoint_url`, `headers`, `timeout`, ... |

HDF5 virtual datasets (same-file, cross-file, unlimited, %b names) and
external storage read too. Not supported yet: szip in HDF4, zstd
dictionaries, blosc2 variable-length blocks; writing.

## More

- [docs/internals.md](docs/internals.md): how it works, module by module
- [docs/measurements.md](docs/measurements.md): what was measured before
  and while designing it, including what did not pay off
- [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md)

MIT license.
