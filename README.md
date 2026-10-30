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

Measured 2026-10-30 02:30: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 24 logical CPUs, NVIDIA GeForce RTX 5070 Ti; Python 3.12.16, numpy 2.5.4, numba 0.68.0; h5py 3.16.0, netCDF4 1.7.4, xarray 2026.9.0, h5netcdf 1.8.2, cupy 14.2.0.

**Whole file, every variable, raw (ms; fastnc: Dataset.load() and the same loop over variables as netCDF4-python, which its ratio is to)**

| file | fastnc | fastnc loop | netCDF4 | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 47.6 | 65.8 | 541 (8.2x) | 501 (10.5x) |
| WRF d02 netCDF3 | 62.3 | 76.2 | 425 (5.6x) | - |
| MODIS 1 km L1B | 54.4 | 92.7 | 955 (10.3x) | 956 (17.6x) |
| MODIS 250 m L1B | 49.7 | 59.4 | 889 (15.0x) | 891 (17.9x) |
| VIIRS I-band | 77.8 | 99.0 | 1948 (19.7x) | 1738 (22.3x) |
| Metop-SG VII | 29.4 | 36.5 | 131 (3.6x) | 102 (3.5x) |
| AVHRR HRPT | 13.1 | 15.8 | 198 (12.6x) | 196 (14.9x) |
| RCM SAR winds | 6.6 | 8.7 | 45.0 (5.2x) | 42.6 (6.4x) |
| Ku scatterometer | 16.2 | 23.7 | 75.0 (3.2x) | 62.3 (3.8x) |
| GOES-19 ABI RadF C02 | 113 | 117 | 2585 (22.0x) | 2476 (21.9x) |
| GOES-19 ABI RadF C13 | 10.8 | 13.4 | 165 (12.3x) | 160 (14.8x) |

**Whole file, masked and scaled as netCDF4-python does by default (ms)**

| file | fastnc | fastnc loop | netCDF4 |
|---|---|---|---|
| WRF d01 netCDF4 | 86.0 | 98.9 | 592 (6.0x) |
| WRF d02 netCDF3 | 93.4 | 125 | 547 (4.4x) |
| MODIS 1 km L1B | 85.1 | 114 | 1178 (10.4x) |
| MODIS 250 m L1B | 85.1 | 69.1 | 1053 (15.2x) |
| VIIRS I-band | 148 | 179 | 3192 (17.8x) |
| Metop-SG VII | 58.7 | 63.2 | 311 (4.9x) |
| AVHRR HRPT | 25.1 | 21.7 | 226 (10.4x) |
| RCM SAR winds | 12.0 | 14.6 | 49.1 (3.4x) |
| Ku scatterometer | 22.3 | 29.2 | 81.3 (2.8x) |
| GOES-19 ABI RadF C02 | 269 | 176 | 5565 (31.7x) |
| GOES-19 ABI RadF C13 | 21.6 | 21.6 | 324 (15.0x) |

**Open + one variable, raw (ms; fastnc.Dataset vs netCDF4, fastnc.File vs h5py)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 12.7 | 47.5 (3.7x) | 6.0 | 27.6 (4.6x) |
| WRF d02 netCDF3 | 3.1 | 18.3 (5.8x) | - | - |
| MODIS 1 km L1B | 26.2 | 324 (12.4x) | 24.0 | 321 (13.4x) |
| MODIS 250 m L1B | 43.8 | 648 (14.8x) | 42.8 | 648 (15.1x) |
| VIIRS I-band | 11.9 | 180 (15.2x) | 10.2 | 172 (16.8x) |
| Metop-SG VII | 7.5 | 18.2 (2.4x) | 2.7 | 4.4 (1.7x) |
| AVHRR HRPT | 2.6 | 18.1 (7.0x) | 2.5 | 15.7 (6.4x) |
| RCM SAR winds | 2.3 | 6.0 (2.6x) | 1.4 | 2.5 (1.8x) |
| Ku scatterometer | 4.3 | 18.4 (4.2x) | 2.5 | 11.9 (4.7x) |
| GOES-19 ABI RadF C02 | 115 | 2169 (18.9x) | 101 | 2137 (21.1x) |
| GOES-19 ABI RadF C13 | 10.6 | 136 (12.8x) | 7.9 | 130 (16.5x) |

**Open + one variable, masked and scaled (ms)**

| file | fastnc | netCDF4 |
|---|---|---|
| WRF d01 netCDF4 | 13.8 | 49.0 (3.5x) |
| WRF d02 netCDF3 | 4.2 | 22.4 (5.4x) |
| MODIS 1 km L1B | 31.6 | 375 (11.9x) |
| MODIS 250 m L1B | 50.7 | 738 (14.5x) |
| VIIRS I-band | 15.6 | 344 (22.0x) |
| Metop-SG VII | 7.2 | 25.9 (3.6x) |
| AVHRR HRPT | 3.9 | 23.4 (6.0x) |
| RCM SAR winds | 2.8 | 6.1 (2.2x) |
| Ku scatterometer | 4.4 | 18.3 (4.2x) |
| GOES-19 ABI RadF C02 | 147 | 4205 (28.5x) |
| GOES-19 ABI RadF C13 | 13.1 | 265 (20.2x) |

**Open + a window of it (ms)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 10.9 | 24.5 (2.3x) | 4.4 | 6.4 (1.4x) |
| WRF d02 netCDF3 | 2.1 | 4.8 (2.3x) | - | - |
| MODIS 1 km L1B | 13.9 | 118 (8.5x) | 11.1 | 115 (10.3x) |
| MODIS 250 m L1B | 7.9 | 123 (15.6x) | 7.0 | 121 (17.3x) |
| VIIRS I-band | 3.8 | 12.1 (3.2x) | 2.2 | 7.8 (3.5x) |
| Metop-SG VII | 5.9 | 14.7 (2.5x) | 1.5 | 1.5 (1.0x) |
| AVHRR HRPT | 2.0 | 5.9 (3.0x) | 1.5 | 3.6 (2.3x) |
| RCM SAR winds | 1.9 | 5.7 (2.9x) | 1.1 | 2.2 (2.0x) |
| Ku scatterometer | 3.5 | 10.2 (2.9x) | 1.6 | 3.9 (2.4x) |
| GOES-19 ABI RadF C02 | 6.8 | 57.6 (8.4x) | 4.6 | 52.4 (11.3x) |
| GOES-19 ABI RadF C13 | 3.7 | 12.3 (3.3x) | 2.1 | 7.3 (3.5x) |

**xarray.open_dataset(...).load(), by engine (ms)**

| file | fastnc | netcdf4 | h5netcdf | scipy |
|---|---|---|---|---|
| WRF d01 netCDF4 | 89.9 | 598 (6.7x) | 11540 (128.4x) | - |
| WRF d02 netCDF3 | 101 | 457 (4.5x) | - | 409 (4.0x) |
| MODIS 1 km L1B | 4.8 | 9.1 (1.9x) | 25.7 (5.4x) | - |
| MODIS 250 m L1B | 4.0 | 7.3 (1.8x) | 18.7 (4.7x) | - |
| VIIRS I-band | 2.2 | 6.3 (2.9x) | 5.5 (2.5x) | - |
| Metop-SG VII | 5.5 | 15.2 (2.7x) | 2.3 (0.4x) | - |
| AVHRR HRPT | 44.0 | 243 (5.5x) | 263 (6.0x) | - |
| RCM SAR winds | 23.4 | 59.4 (2.5x) | 123 (5.3x) | - |
| Ku scatterometer | 35.3 | 92.2 (2.6x) | 1197 (33.9x) | - |
| GOES-19 ABI RadF C02 | 940 | 5036 (5.4x) | 4939 (5.3x) | - |
| GOES-19 ABI RadF C13 | 75.6 | 334 (4.4x) | 390 (5.2x) | - |

**First read in a fresh process: import + open + one variable, raw (ms)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 198 | 264 (1.3x) | 187 | 289 (1.5x) |
| WRF d02 netCDF3 | 185 | 231 (1.2x) | - | - |
| MODIS 1 km L1B | 252 | 545 (2.2x) | 247 | 580 (2.3x) |
| MODIS 250 m L1B | 266 | 874 (3.3x) | 260 | 913 (3.5x) |
| VIIRS I-band | 201 | 393 (2.0x) | 201 | 449 (2.2x) |
| Metop-SG VII | 191 | 243 (1.3x) | 191 | 250 (1.3x) |
| AVHRR HRPT | 176 | 228 (1.3x) | 178 | 266 (1.5x) |
| RCM SAR winds | 185 | 221 (1.2x) | 172 | 251 (1.5x) |
| Ku scatterometer | 219 | 228 (1.0x) | 217 | 255 (1.2x) |
| GOES-19 ABI RadF C02 | 459 | 2349 (5.1x) | 453 | 2370 (5.2x) |
| GOES-19 ABI RadF C13 | 203 | 354 (1.7x) | 200 | 389 (1.9x) |

**First read in a fresh process, masked and scaled (the default) (ms)**

| file | fastnc | netCDF4 |
|---|---|---|
| WRF d01 netCDF4 | 204 | 266 (1.3x) |
| WRF d02 netCDF3 | 187 | 232 (1.2x) |
| MODIS 1 km L1B | 304 | 603 (2.0x) |
| MODIS 250 m L1B | 316 | 960 (3.0x) |
| VIIRS I-band | 239 | 567 (2.4x) |
| Metop-SG VII | 197 | 244 (1.2x) |
| AVHRR HRPT | 194 | 242 (1.2x) |
| RCM SAR winds | 187 | 220 (1.2x) |
| Ku scatterometer | 227 | 242 (1.1x) |
| GOES-19 ABI RadF C02 | 731 | 5029 (6.9x) |
| GOES-19 ABI RadF C13 | 246 | 533 (2.2x) |

**One variable, masked and scaled, to a cupy array (ms)**

| file | fastnc gpu | fastnc cpu+upload | netCDF4+upload |
|---|---|---|---|
| WRF d01 netCDF4 | 14.7 | 14.4 (1.0x) | 50.2 (3.4x) |
| WRF d02 netCDF3 | 5.1 | 6.6 (1.3x) | 25.0 (4.9x) |
| MODIS 1 km L1B | 29.1 | 34.8 (1.2x) | 380 (13.1x) |
| MODIS 250 m L1B | 48.6 | 58.2 (1.2x) | 760 (15.7x) |
| VIIRS I-band | 15.2 | 28.7 (1.9x) | 355 (23.3x) |
| Metop-SG VII | 8.7 | 7.6 (0.9x) | 25.8 (3.0x) |
| AVHRR HRPT | 4.3 | 5.4 (1.3x) | 24.9 (5.8x) |
| RCM SAR winds | 3.6 | 2.9 (0.8x) | 7.3 (2.0x) |
| Ku scatterometer | 5.3 | 4.8 (0.9x) | 19.4 (3.7x) |
| GOES-19 ABI RadF C02 | 118 | 292 (2.5x) | 4378 (37.2x) |
| GOES-19 ABI RadF C13 | 11.9 | 21.8 (1.8x) | 279 (23.4x) |

**fastnc, whole file raw: every core vs one (ms)**

| file | all threads | 1 thread |
|---|---|---|
| WRF d01 netCDF4 | 47.6 | 275 (5.8x) |
| WRF d02 netCDF3 | 62.3 | 179 (2.9x) |
| MODIS 1 km L1B | 54.4 | 554 (10.2x) |
| MODIS 250 m L1B | 49.7 | 530 (10.7x) |
| VIIRS I-band | 77.8 | 1021 (13.1x) |
| Metop-SG VII | 29.4 | 48.8 (1.7x) |
| AVHRR HRPT | 13.1 | 109 (8.3x) |
| RCM SAR winds | 6.6 | 19.4 (2.9x) |
| Ku scatterometer | 16.2 | 81.0 (5.0x) |
| GOES-19 ABI RadF C02 | 113 | 1715 (15.2x) |
| GOES-19 ABI RadF C13 | 10.8 | 99.7 (9.2x) |

"fastnc loop" is fastnc used as a drop-in: the same loop over variables
as netCDF4-python's column (readahead decodes the following variables
together); `Dataset.load()` decodes them all in one pass. (GOES-19 C02
masked `load()` measured 269 ms here; it now takes the fused path for big
scaled variables and runs at the loop's speed, ~160-176 ms.)

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
  row.
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
