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
digests: 7,640 tests, raw and masked, on numpy 1 and 2 (each test a
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
Python process, timed from just before the import. Every variable of every
file means every variable of every group. The files: the user's own WRF
runs (wrfout d01 netCDF-4 deflate+shuffle; d02 netCDF-3) and satellite
products as distributed (MODIS L1B, VIIRS, Metop-SG VII, AVHRR, RCM, a
Ku-band scatterometer, GOES-19 ABI full disk from NOAA's S3 bucket).

Measured 2026-10-29 22:53: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 24 logical CPUs, NVIDIA GeForce RTX 5070 Ti; Python 3.12.16, numpy 2.5.4, numba 0.68.0; h5py 3.16.0, netCDF4 1.7.4, xarray 2026.9.0, h5netcdf 1.8.2, cupy 14.2.0.

**Whole file, every variable, raw (ms)**

| file | fastnc | netCDF4 | h5py |
|---|---|---|---|
| WRF d01 netCDF4 | 48.3 | 546 (11.3x) | 518 (10.7x) |
| WRF d02 netCDF3 | 80.9 | 405 (5.0x) | - |
| MODIS 1 km L1B | 52.9 | 959 (18.1x) | 965 (18.2x) |
| MODIS 250 m L1B | 46.6 | 903 (19.4x) | 900 (19.3x) |
| VIIRS I-band | 73.6 | 1886 (25.6x) | 1703 (23.2x) |
| Metop-SG VII | 27.2 | 135 (5.0x) | 101 (3.7x) |
| AVHRR HRPT | 9.7 | 193 (19.8x) | 191 (19.6x) |
| RCM SAR winds | 6.6 | 43.9 (6.6x) | 39.9 (6.0x) |
| Ku scatterometer | 18.4 | 72.5 (3.9x) | 62.1 (3.4x) |
| GOES-19 ABI RadF C02 | 104 | 2549 (24.5x) | 2448 (23.5x) |
| GOES-19 ABI RadF C13 | 10.9 | 161 (14.8x) | 153 (14.1x) |

**Whole file, masked and scaled as netCDF4-python does by default (ms)**

| file | fastnc | netCDF4 |
|---|---|---|
| WRF d01 netCDF4 | 99.0 | 608 (6.1x) |
| WRF d02 netCDF3 | 128 | 515 (4.0x) |
| MODIS 1 km L1B | 121 | 1185 (9.8x) |
| MODIS 250 m L1B | 73.0 | 1078 (14.8x) |
| VIIRS I-band | 149 | 3160 (21.2x) |
| Metop-SG VII | 58.5 | 307 (5.3x) |
| AVHRR HRPT | 26.6 | 224 (8.4x) |
| RCM SAR winds | 13.4 | 48.1 (3.6x) |
| Ku scatterometer | 21.6 | 78.1 (3.6x) |
| GOES-19 ABI RadF C02 | 178 | 5032 (28.2x) |
| GOES-19 ABI RadF C13 | 23.1 | 324 (14.1x) |

**Open + one variable, raw (ms; fastnc.Dataset vs netCDF4, fastnc.File vs h5py)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 12.7 | 45.7 (3.6x) | 6.0 | 27.0 (4.5x) |
| WRF d02 netCDF3 | 3.3 | 17.8 (5.4x) | - | - |
| MODIS 1 km L1B | 26.6 | 320 (12.0x) | 24.1 | 321 (13.3x) |
| MODIS 250 m L1B | 42.5 | 652 (15.3x) | 38.4 | 652 (17.0x) |
| VIIRS I-band | 12.6 | 181 (14.4x) | 9.5 | 176 (18.4x) |
| Metop-SG VII | 7.4 | 19.5 (2.6x) | 2.5 | 4.4 (1.8x) |
| AVHRR HRPT | 2.4 | 17.6 (7.3x) | 2.1 | 15.2 (7.4x) |
| RCM SAR winds | 2.3 | 6.0 (2.6x) | 1.3 | 2.4 (1.8x) |
| Ku scatterometer | 4.8 | 18.3 (3.8x) | 2.6 | 11.7 (4.6x) |
| GOES-19 ABI RadF C02 | 91.4 | 2131 (23.3x) | 90.7 | 2104 (23.2x) |
| GOES-19 ABI RadF C13 | 9.5 | 132 (13.9x) | 7.9 | 125 (15.8x) |

**Open + one variable, masked and scaled (ms)**

| file | fastnc | netCDF4 |
|---|---|---|
| WRF d01 netCDF4 | 14.2 | 47.4 (3.3x) |
| WRF d02 netCDF3 | 3.9 | 21.6 (5.6x) |
| MODIS 1 km L1B | 31.5 | 367 (11.6x) |
| MODIS 250 m L1B | 50.3 | 739 (14.7x) |
| VIIRS I-band | 18.8 | 347 (18.5x) |
| Metop-SG VII | 8.8 | 26.3 (3.0x) |
| AVHRR HRPT | 3.6 | 22.5 (6.3x) |
| RCM SAR winds | 2.7 | 6.1 (2.3x) |
| Ku scatterometer | 4.9 | 18.7 (3.8x) |
| GOES-19 ABI RadF C02 | 149 | 4101 (27.5x) |
| GOES-19 ABI RadF C13 | 15.6 | 256 (16.3x) |

**Open + a window of it (ms)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 11.5 | 24.2 (2.1x) | 4.3 | 6.1 (1.4x) |
| WRF d02 netCDF3 | 1.8 | 4.5 (2.5x) | - | - |
| MODIS 1 km L1B | 13.6 | 117 (8.6x) | 11.9 | 112 (9.5x) |
| MODIS 250 m L1B | 8.0 | 121 (15.1x) | 7.5 | 120 (15.9x) |
| VIIRS I-band | 4.0 | 11.8 (3.0x) | 2.1 | 7.8 (3.8x) |
| Metop-SG VII | 6.7 | 14.1 (2.1x) | 1.6 | 1.8 (1.1x) |
| AVHRR HRPT | 1.9 | 5.8 (3.0x) | 1.4 | 3.4 (2.4x) |
| RCM SAR winds | 2.0 | 5.6 (2.9x) | 1.0 | 2.2 (2.1x) |
| Ku scatterometer | 3.9 | 10.0 (2.5x) | 1.7 | 3.9 (2.3x) |
| GOES-19 ABI RadF C02 | 7.1 | 58.4 (8.2x) | 4.7 | 51.1 (11.0x) |
| GOES-19 ABI RadF C13 | 3.8 | 13.3 (3.5x) | 2.2 | 7.1 (3.2x) |

**xarray.open_dataset(...).load(), by engine (ms)**

| file | fastnc | netcdf4 | h5netcdf | scipy |
|---|---|---|---|---|
| WRF d01 netCDF4 | 174 | 601 (3.5x) | 11121 (64.0x) | - |
| WRF d02 netCDF3 | 102 | 442 (4.3x) | - | 421 (4.1x) |
| MODIS 1 km L1B | 6.0 | 8.5 (1.4x) | 23.2 (3.9x) | - |
| MODIS 250 m L1B | 4.5 | 7.5 (1.7x) | 18.6 (4.1x) | - |
| VIIRS I-band | 2.4 | 5.4 (2.3x) | 5.3 (2.2x) | - |
| Metop-SG VII | 5.6 | 14.1 (2.5x) | 2.7 (0.5x) | - |
| AVHRR HRPT | 43.3 | 241 (5.6x) | 257 (5.9x) | - |
| RCM SAR winds | 33.1 | 58.5 (1.8x) | 118 (3.6x) | - |
| Ku scatterometer | 76.7 | 96.3 (1.3x) | 1186 (15.5x) | - |
| GOES-19 ABI RadF C02 | 889 | 4993 (5.6x) | 4967 (5.6x) | - |
| GOES-19 ABI RadF C13 | 77.2 | 319 (4.1x) | 374 (4.8x) | - |

**First read in a fresh process: import + open + one variable (ms)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 232 | 261 (1.1x) | 198 | 275 (1.4x) |
| WRF d02 netCDF3 | 199 | 241 (1.2x) | - | - |
| MODIS 1 km L1B | 304 | 535 (1.8x) | 300 | 561 (1.9x) |
| MODIS 250 m L1B | 332 | 903 (2.7x) | 318 | 896 (2.8x) |
| VIIRS I-band | 239 | 407 (1.7x) | 235 | 446 (1.9x) |
| Metop-SG VII | 216 | 240 (1.1x) | 197 | 268 (1.4x) |
| AVHRR HRPT | 185 | 240 (1.3x) | 182 | 260 (1.4x) |
| RCM SAR winds | 173 | 212 (1.2x) | 173 | 235 (1.4x) |
| Ku scatterometer | 206 | 224 (1.1x) | 200 | 247 (1.2x) |
| GOES-19 ABI RadF C02 | 677 | 2347 (3.5x) | 660 | 2334 (3.5x) |
| GOES-19 ABI RadF C13 | 251 | 359 (1.4x) | 221 | 374 (1.7x) |

**One variable, masked and scaled, to a cupy array (ms)**

| file | fastnc gpu | fastnc cpu+upload | netCDF4+upload |
|---|---|---|---|
| WRF d01 netCDF4 | 14.1 | 14.9 (1.1x) | 47.0 (3.3x) |
| WRF d02 netCDF3 | 4.6 | 5.9 (1.3x) | 24.0 (5.2x) |
| MODIS 1 km L1B | 28.4 | 34.0 (1.2x) | 375 (13.2x) |
| MODIS 250 m L1B | 43.5 | 57.0 (1.3x) | 749 (17.2x) |
| VIIRS I-band | 20.8 | 28.7 (1.4x) | 362 (17.4x) |
| Metop-SG VII | 8.3 | 9.3 (1.1x) | 26.1 (3.1x) |
| AVHRR HRPT | 3.0 | 5.0 (1.7x) | 25.0 (8.3x) |
| RCM SAR winds | 3.1 | 3.3 (1.1x) | 6.4 (2.1x) |
| Ku scatterometer | 5.2 | 4.8 (0.9x) | 18.5 (3.5x) |
| GOES-19 ABI RadF C02 | 138 | 277 (2.0x) | 4235 (30.6x) |
| GOES-19 ABI RadF C13 | 12.5 | 22.2 (1.8x) | 266 (21.3x) |

**fastnc, whole file raw: every core vs one (ms)**

| file | all threads | 1 thread |
|---|---|---|
| WRF d01 netCDF4 | 48.3 | 272 (5.6x) |
| WRF d02 netCDF3 | 80.9 | 170 (2.1x) |
| MODIS 1 km L1B | 52.9 | 541 (10.2x) |
| MODIS 250 m L1B | 46.6 | 530 (11.4x) |
| VIIRS I-band | 73.6 | 1025 (13.9x) |
| Metop-SG VII | 27.2 | 45.7 (1.7x) |
| AVHRR HRPT | 9.7 | 110 (11.3x) |
| RCM SAR winds | 6.6 | 17.7 (2.7x) |
| Ku scatterometer | 18.4 | 79.1 (4.3x) |
| GOES-19 ABI RadF C02 | 104 | 1467 (14.1x) |
| GOES-19 ABI RadF C13 | 10.9 | 97.2 (8.9x) |

The GPU pays where values are many and scaled (VIIRS, MODIS, GOES: the
packed integers cross PCIe, float32 is made on the card): 1.2-2x over
reading on the CPU and uploading the floats. For small or unscaled
variables it is even, and for the Ku-band file's small arrays it loses
(0.9x). Decompressing deflate on the card was measured and loses on every
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
- **Memory reused.** Windows faults in fresh pages slowly (55 ms per GB);
  big outputs reuse already-mapped memory once earlier arrays are gone.
- **Metadata read lazily** from the file's bytes in Python, and netCDF's
  view built once: xarray's netcdf4 engine spent 7 s opening a WRF file in
  an environment where netCDF4-python's plugins fail to load (a 36-48 ms
  probe per variable), h5netcdf 10 s (its unlimited dimensions are
  recomputed on every question); fastnc 30 ms.
- **The GPU** does what it is good at: decompressing deflate on the card
  loses (nvCOMP measured slower than the CPU on every file here), so the
  CPU cores decode, the packed integers cross PCIe while the next band
  decodes, and masking and scaling to float32 run on the card.

Where it is no faster: a file of a few tiny variables (opening costs about
the same as the references'), and the first read of a process when numba
is loading (reads use zlib meanwhile; `fastnc.warm()` loads it up front).

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
| `fastnc.warm()` | load numba and the kernels now (for worker pools: `ProcessPoolExecutor(initializer=fastnc.warm)`) |
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
