# fastnc

[![tests](https://github.com/Rochy111/fastnc/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Rochy111/fastnc/actions/workflows/tests.yml)
[![docs](https://github.com/Rochy111/fastnc/actions/workflows/docs.yml/badge.svg?branch=main)](https://rochy111.github.io/fastnc/)

**Documentation: [rochy111.github.io/fastnc](https://rochy111.github.io/fastnc/)**

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

## Install

```
pip install git+https://github.com/Rochy111/fastnc
pip install "fastnc[xarray] @ git+https://github.com/Rochy111/fastnc"   # or [all] for cupy (CUDA 12) too
```

From a clone: `pip install -e ".[xarray]"`.

Python 3.9+, numpy 1.23+ (1.x and 2.x), numba 0.57+. No other dependency;
h5py, netCDF4-python and hdf5plugin are needed only to regenerate the test
references. The first second of reading after an install starts a low-priority
process that compiles the kernels (a few minutes, once per machine and
version; `fastnc.tune()` does it and waits). `FASTNC_IMAGE=0` keeps kernels in
numba alone.

## Values

Bit-identical to the reference readers, and checked:

- **raw** values equal h5py's and netCDF4-python's with mask and scale off,
  byte for byte (fastnc returns native byte order, where netCDF4-python
  keeps big-endian netCDF-4 / HDF4 variables so and h5py keeps the stored
  order: the values are equal, only the byte-order flag differs);
- **masked / scaled** values follow every rule of netCDF4-python 1.7 --
  `_FillValue` and default fill values, `missing_value`, `valid_min` /
  `valid_max` / `valid_range`, `scale_factor` / `add_offset` with their
  dtype promotion (which differs between numpy 1 and 2), `_Unsigned`, the
  masked array vs plain array choice and its fill value -- in the same
  float type; except that a valid range typed like `scale_factor` /
  `add_offset` is applied in unpacked units, as the netCDF User Guide and CF
  say (netCDF4-python applies it to the stored integers and masks nearly
  everything in such files);
- **GPU** reads equal CPU reads;
- **metadata** as each reference gives it: netCDF-c's dimensions (phony
  ones included), variable order, attribute values and types, user types
  (unnamed ones named as netCDF-c names them), and `print()`; h5py's link
  and attribute order, dtypes and their metadata, fill values and reprs.
  Across a corpus of 357 real satellite and model files, the metadata of
  every HDF5 dataset is h5py's (492,000 items), and every variable's is
  netCDF4-python's (342,000) but for the byte-order flag above and
  `quantization()` of HDF4 variables, which netCDF4-python refuses.

`tests/test_references.py` compares every variable and HDF5 object of the
test corpus (`tests/data`: every layout, chunk index, group and attribute
storage, datatype and filter), of a synthetic corpus and of real WRF and
satellite files (MODIS, VIIRS, Metop-SG, AVHRR, RCM, a Ku-band
scatterometer) with what netCDF4-python and h5py read, from stored
digests: about 7,000 tests, raw and masked, on numpy 1 and 2 (each test a
variable or object, whole and in three selections). The GOES-19 ABI files
(not in the repository) were checked the same way against both readers
directly, with `tools/check/`.

## Speed

Milliseconds, fastnc's speed-up over each reader in brackets
(`tools/bench.py`, tables by `tools/bench_table.py` from
[docs/benchmarks.json](docs/benchmarks.json)). Method: files in the OS
cache, each read once before timing (reads of files not in the cache are
in [docs/measurements.md](docs/measurements.md): 10-18x h5py from an NVMe);
best of 5 runs with the readers alternated run by run; raw reads with mask and
scale off; fastnc on every logical CPU unless a row says 1 thread; "first read"
is a fresh Python process, timed from just before the import, best of three
processes, with this machine's kernel image (numba is never imported: see
below). Every variable of every
file means every variable of every group. The files: the user's own WRF
runs (wrfout d01 netCDF-4 deflate+shuffle; d02 netCDF-3) and satellite
products as distributed (MODIS L1B, VIIRS, Metop-SG VII, AVHRR, RCM, a
Ku-band scatterometer, GOES-19 ABI full disk from NOAA's S3 bucket).

Measured 2026-11-02 16:26 and 2026-11-02 16:35: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 24 logical CPUs, NVIDIA GeForce RTX 5070 Ti; Python 3.12.16, numpy 2.5.4, numba 0.68.0; h5py 3.16.0, netCDF4 1.7.4, xarray 2026.9.0, h5netcdf 1.8.2, cupy 14.2.0.

These are the slower of fastnc's two cases. The readers take turns, and netCDF4 and
h5py take seconds, so each fastnc read here comes after what it keeps between
reads has been let go (5 s unused), and it starts on fresh memory. A loop of reads
one after another is faster: GOES C02 whole 61 ms, not 94; VIIRS to the GPU 8.6 ms,
not 52.

**Whole file, every variable, raw (ms; fastnc: Dataset.load() and the same loop over variables as netCDF4-python, which its ratio is to)**

| file | fastnc | fastnc loop | netCDF4 | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 31.9 | 41.2 | 493 (12.0x) | 449 (14.1x) |
| WRF d02 netCDF3 | 47.5 | 58.3 | 371 (6.4x) | - |
| MODIS 1 km L1B | 70.0 | 88.1 | 907 (10.3x) | 909 (13.0x) |
| MODIS 250 m L1B | 66.2 | 63.0 | 848 (13.5x) | 846 (12.8x) |
| VIIRS I-band | 69.2 | 79.7 | 1775 (22.3x) | 1625 (23.5x) |
| Metop-SG VII | 19.9 | 25.2 | 105 (4.2x) | 84.0 (4.2x) |
| AVHRR HRPT | 12.5 | 13.0 | 182 (13.9x) | 180 (14.4x) |
| RCM SAR winds | 3.9 | 5.3 | 41.4 (7.8x) | 38.0 (9.7x) |
| Ku scatterometer | 8.4 | 12.0 | 66.8 (5.6x) | 55.7 (6.7x) |
| GOES-19 ABI RadF C02 | 94.2 | 95.9 | 2383 (24.8x) | 2303 (24.5x) |
| GOES-19 ABI RadF C13 | 16.0 | 13.1 | 154 (11.8x) | 147 (9.2x) |

**Whole file, masked and scaled as netCDF4-python does by default (ms)**

| file | fastnc | fastnc loop | netCDF4 |
|---|---|---|---|
| WRF d01 netCDF4 | 36.7 | 45.4 | 540 (11.9x) |
| WRF d02 netCDF3 | 61.2 | 68.3 | 476 (7.0x) |
| MODIS 1 km L1B | 86.1 | 100 | 1107 (11.0x) |
| MODIS 250 m L1B | 79.1 | 71.4 | 1008 (14.1x) |
| VIIRS I-band | 116 | 117 | 3048 (26.2x) |
| Metop-SG VII | 37.7 | 43.1 | 268 (6.2x) |
| AVHRR HRPT | 14.6 | 16.2 | 211 (13.0x) |
| RCM SAR winds | 7.4 | 8.3 | 45.1 (5.4x) |
| Ku scatterometer | 11.1 | 16.0 | 71.4 (4.5x) |
| GOES-19 ABI RadF C02 | 150 | 142 | 4823 (33.8x) |
| GOES-19 ABI RadF C13 | 32.8 | 19.1 | 309 (16.2x) |

**Open + one variable, raw (ms; fastnc.Dataset vs netCDF4, fastnc.File vs h5py)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 5.5 | 43.6 (7.9x) | 4.3 | 26.7 (6.2x) |
| WRF d02 netCDF3 | 2.5 | 17.2 (6.8x) | - | - |
| MODIS 1 km L1B | 36.5 | 312 (8.5x) | 37.7 | 308 (8.2x) |
| MODIS 250 m L1B | 57.2 | 623 (10.9x) | 64.3 | 622 (9.7x) |
| VIIRS I-band | 14.2 | 173 (12.1x) | 11.6 | 167 (14.4x) |
| Metop-SG VII | 2.8 | 15.5 (5.5x) | 1.9 | 3.7 (1.9x) |
| AVHRR HRPT | 2.3 | 17.2 (7.6x) | 2.2 | 14.7 (6.8x) |
| RCM SAR winds | 0.98 | 5.1 (5.2x) | 0.81 | 2.2 (2.7x) |
| Ku scatterometer | 2.6 | 16.9 (6.6x) | 2.0 | 10.9 (5.5x) |
| GOES-19 ABI RadF C02 | 82.8 | 1999 (24.1x) | 84.6 | 1964 (23.2x) |
| GOES-19 ABI RadF C13 | 11.1 | 127 (11.4x) | 11.3 | 121 (10.7x) |

**Open + one variable, masked and scaled (ms)**

| file | fastnc | netCDF4 |
|---|---|---|
| WRF d01 netCDF4 | 5.7 | 44.8 (7.9x) |
| WRF d02 netCDF3 | 2.5 | 20.7 (8.2x) |
| MODIS 1 km L1B | 40.3 | 352 (8.8x) |
| MODIS 250 m L1B | 64.2 | 703 (11.0x) |
| VIIRS I-band | 16.4 | 328 (20.0x) |
| Metop-SG VII | 3.5 | 22.8 (6.5x) |
| AVHRR HRPT | 2.7 | 22.1 (8.3x) |
| RCM SAR winds | 1.2 | 5.3 (4.4x) |
| Ku scatterometer | 2.5 | 16.9 (6.8x) |
| GOES-19 ABI RadF C02 | 112 | 3956 (35.3x) |
| GOES-19 ABI RadF C13 | 19.3 | 247 (12.8x) |

**Open + a window of it (ms)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 3.9 | 21.5 (5.6x) | 2.7 | 5.8 (2.1x) |
| WRF d02 netCDF3 | 1.2 | 4.0 (3.4x) | - | - |
| MODIS 1 km L1B | 25.8 | 114 (4.4x) | 25.4 | 110 (4.3x) |
| MODIS 250 m L1B | 10.0 | 119 (11.8x) | 16.2 | 116 (7.2x) |
| VIIRS I-band | 2.0 | 11.0 (5.5x) | 1.6 | 7.2 (4.4x) |
| Metop-SG VII | 1.7 | 11.7 (6.7x) | 0.83 | 1.2 (1.5x) |
| AVHRR HRPT | 1.2 | 5.3 (4.3x) | 1.1 | 3.2 (2.9x) |
| RCM SAR winds | 0.88 | 4.8 (5.4x) | 0.74 | 2.0 (2.6x) |
| Ku scatterometer | 1.7 | 8.9 (5.2x) | 1.3 | 3.5 (2.7x) |
| GOES-19 ABI RadF C02 | 5.1 | 54.2 (10.6x) | 5.2 | 49.2 (9.4x) |
| GOES-19 ABI RadF C13 | 1.7 | 11.3 (6.5x) | 1.6 | 6.7 (4.2x) |

**xarray.open_dataset(...).load(), by engine (ms)**

| file | fastnc | netcdf4 | h5netcdf | scipy |
|---|---|---|---|---|
| WRF d01 netCDF4 | 93.4 | 563 (6.0x) | 9659 (103.4x) | - |
| WRF d02 netCDF3 | 80.7 | 394 (4.9x) | - | 348 (4.3x) |
| MODIS 1 km L1B | 3.8 | 7.5 (2.0x) | 19.2 (5.1x) | - |
| MODIS 250 m L1B | 3.0 | 6.5 (2.1x) | 16.4 (5.4x) | - |
| VIIRS I-band | 1.8 | 4.9 (2.7x) | 4.4 (2.5x) | - |
| Metop-SG VII | 0.71 | 11.4 (16.1x) | 2.0 (2.8x) | - |
| AVHRR HRPT | 34.1 | 229 (6.7x) | 245 (7.2x) | - |
| RCM SAR winds | 13.2 | 54.7 (4.1x) | 106 (8.0x) | - |
| Ku scatterometer | 22.3 | 81.0 (3.6x) | 1005 (45.1x) | - |
| GOES-19 ABI RadF C02 | 212 | 4790 (22.6x) | 4834 (22.8x) | - |
| GOES-19 ABI RadF C13 | 27.3 | 309 (11.3x) | 354 (12.9x) | - |

**First read in a fresh process: import + open + one variable, raw (ms)**

| file | fastnc | netCDF4 | fastnc.File | h5py |
|---|---|---|---|---|
| WRF d01 netCDF4 | 144 | 218 (1.5x) | 140 | 219 (1.6x) |
| WRF d02 netCDF3 | 137 | 186 (1.4x) | - | - |
| MODIS 1 km L1B | 201 | 484 (2.4x) | 172 | 507 (3.0x) |
| MODIS 250 m L1B | 208 | 802 (3.9x) | 203 | 826 (4.1x) |
| VIIRS I-band | 166 | 347 (2.1x) | 159 | 391 (2.5x) |
| Metop-SG VII | 143 | 188 (1.3x) | 139 | 203 (1.5x) |
| AVHRR HRPT | 141 | 190 (1.4x) | 137 | 216 (1.6x) |
| RCM SAR winds | 137 | 176 (1.3x) | 133 | 196 (1.5x) |
| Ku scatterometer | 141 | 193 (1.4x) | 143 | 212 (1.5x) |
| GOES-19 ABI RadF C02 | 246 | 2160 (8.8x) | 248 | 2163 (8.7x) |
| GOES-19 ABI RadF C13 | 165 | 308 (1.9x) | 152 | 328 (2.2x) |

**First read in a fresh process, masked and scaled (the default) (ms)**

| file | fastnc | netCDF4 |
|---|---|---|
| WRF d01 netCDF4 | 151 | 217 (1.4x) |
| WRF d02 netCDF3 | 145 | 189 (1.3x) |
| MODIS 1 km L1B | 219 | 521 (2.4x) |
| MODIS 250 m L1B | 237 | 886 (3.7x) |
| VIIRS I-band | 176 | 500 (2.8x) |
| Metop-SG VII | 148 | 196 (1.3x) |
| AVHRR HRPT | 150 | 195 (1.3x) |
| RCM SAR winds | 146 | 174 (1.2x) |
| Ku scatterometer | 148 | 191 (1.3x) |
| GOES-19 ABI RadF C02 | 320 | 4448 (13.9x) |
| GOES-19 ABI RadF C13 | 180 | 426 (2.4x) |

**A fresh process reading the whole file, masked and scaled (ms)**

| file | fastnc | netCDF4 |
|---|---|---|
| WRF d01 netCDF4 | 193 | 704 (3.6x) |
| WRF d02 netCDF3 | 217 | 672 (3.1x) |
| MODIS 1 km L1B | 247 | 1285 (5.2x) |
| MODIS 250 m L1B | 249 | 1174 (4.7x) |
| VIIRS I-band | 293 | 3159 (10.8x) |
| Metop-SG VII | 215 | 470 (2.2x) |
| AVHRR HRPT | 168 | 384 (2.3x) |
| RCM SAR winds | 158 | 214 (1.4x) |
| Ku scatterometer | 161 | 253 (1.6x) |
| GOES-19 ABI RadF C02 | 426 | 6001 (14.1x) |
| GOES-19 ABI RadF C13 | 191 | 488 (2.6x) |

**One variable, masked and scaled, to a cupy array (ms)**

| file | fastnc gpu | fastnc cpu+upload | netCDF4+upload |
|---|---|---|---|
| WRF d01 netCDF4 | 5.9 | 6.5 (1.1x) | 46.4 (7.9x) |
| WRF d02 netCDF3 | 11.8 | 14.7 (1.2x) | 32.8 (2.8x) |
| MODIS 1 km L1B | 71.0 | 75.8 (1.1x) | 390 (5.5x) |
| MODIS 250 m L1B | 124 | 151 (1.2x) | 786 (6.3x) |
| VIIRS I-band | 51.6 | 80.7 (1.6x) | 390 (7.6x) |
| Metop-SG VII | 5.4 | 7.7 (1.4x) | 27.1 (5.0x) |
| AVHRR HRPT | 4.4 | 9.3 (2.1x) | 28.9 (6.6x) |
| RCM SAR winds | 1.7 | 1.8 (1.1x) | 6.0 (3.5x) |
| Ku scatterometer | 3.9 | 3.6 (0.9x) | 18.3 (4.7x) |
| GOES-19 ABI RadF C02 | 110 | 264 (2.4x) | 4357 (39.5x) |
| GOES-19 ABI RadF C13 | 39.8 | 58.8 (1.5x) | 292 (7.3x) |

**fastnc, whole file raw: every core vs one (ms)**

| file | all threads | 1 thread |
|---|---|---|
| WRF d01 netCDF4 | 31.9 | 156 (4.9x) |
| WRF d02 netCDF3 | 47.5 | 130 (2.7x) |
| MODIS 1 km L1B | 70.0 | 403 (5.8x) |
| MODIS 250 m L1B | 66.2 | 394 (6.0x) |
| VIIRS I-band | 69.2 | 617 (8.9x) |
| Metop-SG VII | 19.9 | 36.6 (1.8x) |
| AVHRR HRPT | 12.5 | 67.5 (5.4x) |
| RCM SAR winds | 3.9 | 8.5 (2.2x) |
| Ku scatterometer | 8.4 | 16.8 (2.0x) |
| GOES-19 ABI RadF C02 | 94.2 | 927 (9.8x) |
| GOES-19 ABI RadF C13 | 16.0 | 62.2 (3.9x) |

"fastnc loop" is fastnc used as a drop-in: the same loop over variables
as netCDF4-python's column (readahead decodes the following variables
together); `Dataset.load()` decodes them all in one pass.

Whole benchmark runs differ by up to 2x in places on this machine, more
than best-of-5 within a run evens out, so each cell is the lower of two
full runs, for every reader alike (`tools/bench_merge.py`). Timed reads
follow one another, as in any benchmark loop; a short read after the
machine has been idle for a second takes up to 1-2 ms more (GOES C13
4.0 -> 5.4 ms, `tools/profile/idle_reads.py`: the system waking parked
cores), which matters only for the smallest rows.

The GPU pays where values are many and scaled (VIIRS, MODIS, GOES: the
packed integers cross PCIe while the CPU decodes, float32 is made on the
card): 1.1-2.4x over reading on the CPU and uploading the floats. For
small or unscaled variables it is about even, and for the Ku-band file it
loses (0.9x). Decompressing deflate on the card was measured and loses on every
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
  in numba, at or above the C libraries' speed; zfp too, level with the C
  library on one core, a little ahead or behind by mode.
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
- **Nothing to mask, proved while decoding.** Most variables never hold
  their fill value, yet every value used to be checked afterwards. A
  shuffled chunk comes out of inflate as byte planes, and if the fill
  value's top byte is missing from the plane of top bytes, the value is not
  in the chunk: a quarter of the data, still in cache, and no second pass
  (whole WRF d01 file, masked, 52 -> 44 ms). Anything not proved this way is
  checked as before.
- **Many arrays, one pass, again for masking.** `load()` masks and scales all
  its arrays with the threads moving from one to the next as each runs out
  of blocks, instead of waking them for every array. Small arrays cost no
  hand-off, the biggest ends up with every thread, and no size has to be
  chosen (WRF d01 masked 44.5 -> 32.7 ms).
- **Memory reused.** Windows faults in fresh pages slowly (55 ms per GB);
  big outputs reuse already-mapped memory once earlier arrays are gone.
- **Opening reads what you ask for.** `d['T']` touches `T`'s header and
  nothing else; the other variables, dimensions and groups are read when
  something wants them. Reading one variable from each of many files gains
  the most (WRF d01 8.7 -> 5.0 ms); a read of a whole file gains nothing.
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
- **numba is for compiling, not for running.** Importing numba and loading
  the kernels takes 0.7-0.9 s and holds the GIL, for what is a few MB of
  finished machine code. So that code is linked into a file of fastnc's own
  for this machine (x86-64 Windows and Linux), and a process calls it with
  numpy and ctypes alone, in under a millisecond: every process runs at full
  speed from its first read, and worker pools need no warming. The file is
  built in the background once reading has gone on for a second: nothing
  stops to compile, and until it is there reads go without kernels, chunks
  inflated by libdeflate or zlib through ctypes on every core, mask/scale by
  numpy block by block on the pool. Elsewhere, kernels load from numba once
  reading without them has taken as long as loading them does (or on
  `fastnc.warm()`).
- **It watches the storage.** A big read of compressed chunks first times
  three one-page reads. From flash, or from the OS cache, chunks fault in
  through the mapping on every core; from storage that pays per request (a
  disk head, a network round trip) they come in a few big reads: 160 requests
  for VIIRS, not 4,750.
- **The calling thread stays on a fast core.** On Windows with performance
  and efficiency cores, a script without a focused window counts as
  background work, and ~2 s in its main thread is moved to an efficiency
  core for good: every read then took 1.6x as long (WRF d01 34 -> 55 ms).
  Any thread that opens a file or reads with fastnc now declares its work
  as the user's. That stays set on the thread afterwards, for your own code
  too. Other systems and CPUs with one kind of core: nothing happens.

Where it is no faster: a file of a few tiny variables (opening costs about
the same as the references'), and a fresh process reading little (both
pay ~150 ms for numpy's import; fastnc's own import is ~15 ms, h5py's ~90).

## Use

| | |
|---|---|
| `fastnc.Dataset(path_or_url)` | netCDF4.Dataset: `variables`, `dimensions`, `groups`, attributes, `set_auto_maskandscale` and friends, `filters()`, `chunking()`, `endian()`, user types, `auto_complex=True`, `get_variables_by_attributes()`, `print()` |
| `fastnc.File(path_or_url)` | h5py.File: groups, datasets, `attrs`, `chunks`, `compression`, `fillvalue`, `read_direct`, `visititems`, `asstr()`, `astype()`, `fields()`, `iter_chunks()`, `.dims`, references and links |
| `fastnc.MFDataset(files)` | netCDF4.MFDataset, for any file fastnc opens (netCDF4-python's takes netCDF-3 and classic only) |
| `Group.load(names=None, recursive=False)` | many variables decoded in one pass |
| `Variable.read(key, mask=, scale=, nan=)` | those settings for this read alone, nothing changed for other threads; `out=` decodes into your array |
| `set_auto_nan(True)` | masked floats as NaN in plain arrays, as xarray gives them, filled while decoding |
| `memmap()` | an uncompressed variable as a read-only view of the file, no copy (every netCDF-3 variable, contiguous HDF5 / HDF4 ones) |
| `read_async(key)`, `load_async()` | the read on a worker, a `Future` of its result: the next frame is read while this one is used |
| `Dataset.tocdl()` | the header as `ncdump -h` prints it, byte for byte, without the ncdump program |
| `Dataset.hdf4_datasets()` | every HDF4 data set, those sharing a name included (netCDF-c shows only the last) |
| `Variable.read(key, device='gpu')` | values as a cupy array (masked results: data and mask on the card, `.get()` for numpy) |
| `xr.open_dataset(p, engine='fastnc', preload=False)` | xarray, same Dataset as `engine='netcdf4'` |
| `fastnc.set_threads(n)`, `FASTNC_THREADS` | how many threads (default: every logical CPU) |
| `fastnc.warm()` | have the kernels ready now: from this machine's image if it has one, else from numba, waiting for that |
| `fastnc.hardware()` | what fastnc found here: CPUs allowed, cores, memory, cache, the threads it will use, its kernel image |
| `fastnc.tune()` | time real decoding on this machine, once, keep what suits it, and link the kernel image (optional: everything adapts without it) |
| `fastnc.release()` | free the memory kept between reads |
| `storage_options={...}` | remote: `region`, `endpoint_url`, `headers`, `timeout`, ... |

HDF5 virtual datasets (same-file, cross-file, unlimited, %b names), external
storage and links, region references and shared object header messages read
too, a remote file's external and virtual sources fetched from beside its URL.
Every common filter, the JPEG filter and blosc2's dictionaries and
variable-length blocks included. Not supported: writing.

## More

- [The documentation site](https://rochy111.github.io/fastnc/): guides, reference, performance and a gallery
- [docs/internals.md](docs/internals.md): how it works, module by module
- [docs/measurements.md](docs/measurements.md): what was measured before
  and while designing it, including what did not pay off
- [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md)

MIT license.
