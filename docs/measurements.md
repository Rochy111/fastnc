# Where today's readers spend their time

Measured 2026-10-29, before any fastnc code existed, to decide the design.
Machine: i7-13700K (8P+8E, 24 logical), DDR5, RTX 5070 Ti (PCIe 5 x16),
Windows 11. Python 3.11, numpy 1.26.4, h5py 3.16 (HDF5 2.0.0),
netCDF4-python 1.7.4 (netcdf-c 4.9.3, HDF5 1.14.6), xarray 2025.6.1,
h5netcdf 1.8.1, zlib 1.3.1. Files in the OS cache (warm), best of 3 after a
first read; timing threads on P-cores (`tools/profile/common.fast_thread`).
Scripts: `tools/profile/`.

## The real files

| file | HDF5 superblock | groups | vars | raw MB | stored MB | layout | chunks | typical chunk |
|---|---|---|---|---|---|---|---|---|
| wrfout d01 (netCDF4) | v2 | 0 | 210 | 202 | 74 | deflate+shuffle | 635 | 39 kB (max 2.4 MB) |
| wrfout d02 (netCDF3 64-bit offset) | - | - | 140 | 802 | 802 | 1 record: every variable one contiguous block | - | - |
| MOD021KM | v0 | 8 | 41 | 345 | 170 | deflate | 56 | **1-3 chunks per band variable, up to 33 MB each** |
| MOD02QKM | v0 | 8 | 18 | 288 | 169 | deflate | 41 | **6 chunks of 17 MB for the 250 m bands** |
| VJ102IMG | v0 | 2 | 26 | 1035 | 281 | deflate+shuffle | 4042 | 262 kB (32 rows x 6400) |
| Metop-SG VII | v2 | 10 | 149 | 118 | 118 | uncompressed; 45 chunked | **19,433** | 6.3 kB (one row) |
| Ku scatterometer | v2 | 0 | 63 | 4.7 | 4.3 | deflate+shuffle | **15,133** | 3.5 kB (one row) |
| AVHRR HRPT | v0 | 0 | 10 | 59 | 30 | deflate+shuffle | 84 | 375 kB (512x512) |
| RCM SAR winds | v2 | 0 | 26 | 22 | 1.9 | deflate+shuffle | 22 | 1.1 MB (one per var) |
| AWIPS MaxDB | v2 | 0 | 4 | 0.5 | 0.5 | contiguous | 0 | - |

Three regimes the synthetic corpus does not show: **huge single chunks**
(MODIS: parallelism over chunks gives nothing inside one 33 MB chunk),
**tens of thousands of tiny chunks** (Metop-SG, Ku: per-chunk overhead is
everything), and the plain WRF/VIIRS case of hundreds-to-thousands of
mid-size chunks. Superblock v0 (old-style v1 B-trees) and v2 are both common.

## Whole-file reads (open + every variable)

| file | h5py | netCDF4 raw | netCDF4 mask+scale |
|---|---|---|---|
| wrfout d01 | 0.48 s | 0.53 s | 0.57 s |
| wrfout d02 (nc3) | - | 0.43 s (1.9 GB/s) | 0.49 s |
| MOD021KM | 0.99 s | 0.99 s | 1.20 s |
| MOD02QKM | 0.92 s | 0.91 s | 1.09 s |
| VJ102IMG | 1.72 s | 1.89 s | 3.21 s |
| Metop-SG VII | 0.10 s | 0.11 s | 0.29 s |
| Ku scatterometer | 0.064 s | 0.076 s | 0.081 s |
| AVHRR | 0.19 s | 0.20 s | 0.23 s |
| RCM | 0.041 s | 0.043 s | 0.047 s |

## Where the time goes (chunked variables, single thread)

`tools/profile/breakdown.py`: chunk index, raw I/O, `zlib.decompress`,
numpy unshuffle, measured separately.

| file | h5py total | index walk (`chunk_iter`) | read chunk bytes | inflate (zlib) | unshuffle (numpy) |
|---|---|---|---|---|---|
| wrfout d01 | 475 ms | 0.7 ms | 12 ms | 317 ms | 158 ms |
| VJ102IMG | 1782 ms | 1.5 ms | 67 ms | 1533 ms | 688 ms |
| MOD021KM | 972 ms | 0.1 ms | 23 ms | 859 ms | - |
| AVHRR | 198 ms | 0 | 5 ms | 161 ms | 51 ms |
| Ku | 62 ms | 6 ms | 5 ms | 23 ms | 43 ms |
| Metop-SG | 126 ms | 8 ms | 42 ms | - | - |

Inflate is 70-90% of every compressed read. h5py and netCDF4-python both
inflate one chunk at a time on one core; the readers barely differ.
Unshuffle in numpy is a second full pass that costs half as much again.
`get_chunk_info(i)` per chunk is O(n) each (136 ms for Ku's 15k chunks);
`chunk_iter` is 20x cheaper -- per-chunk overhead matters at those counts.

## Prototype: parallel inflate straight into place

`tools/profile/inflate_into.py`: zlib's `uncompress()` through ctypes (which
releases the GIL), each chunk written straight into a preallocated output
(no per-chunk `bytes`), threads taking chunks biggest-first off a counter.

| file | 1 thread | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| VJ102IMG (1035 MB out) | 1260 ms | 312 | 162 | 98 | **75 ms (13.8 GB/s)** |
| wrfout d01 (203 MB) | 256 ms | 63 | 32 | 20 | **16 ms (12.5 GB/s)** |
| MOD021KM (420 MB) | 729 ms | 206 | 119 | 106 | **103 ms** (one 33 MB chunk is the floor) |
| Ku (4.7 MB, 15k chunks) | **29 ms** | 54 | 55 | 52 | 55 (Python per-chunk cost + GIL) |

Even single-threaded, inflating into place beats `zlib.decompress` (1260 vs
1533 ms on VIIRS): no fresh 262 kB object per chunk. Page faults on fresh
memory are serialized on Windows, which is why `zlib.decompress` on 16
threads only reached 497 ms on VIIRS.

## xarray: why it is 10-20x slower on wrfout d01

Not the reading. Two separate metadata bugs:

- **engine='netcdf4' (6-7 s):** xarray calls `Variable.filters()` for every
  variable at open. In this environment that costs **36-48 ms a call**:
  netCDF4-python ships zstd/blosc HDF5 plugins and points
  `HDF5_PLUGIN_PATH` at them, but they do not load against conda's HDF5, so
  every availability check rescans the directory and tries `LoadLibrary`
  again (`has_zstd_filter()` 20 ms, `has_blosc_filter()` 17 ms). With
  `HDF5_PLUGIN_PATH` pointed at an empty folder, `filters()` takes 0.18 ms.
  200 variables x 34 ms = 6.8 s of the 7.3 s open. Environment-specific:
  a healthy install would not see most of this.
- **engine='h5netcdf' (10.5 s):** h5netcdf works out an unlimited
  dimension's size by scanning every variable that uses it, every time it
  is asked: 3,237 `Dimension.size` calls made 734,681 h5py group lookups
  (O(variables^2)). WRF's `Time` is on every variable. This one is
  structural and affects everyone.

The data read under xarray is 0.7 s (netcdf4) / 3.9 s (h5netcdf, which
also pays the lookups during load). A fastnc backend avoids both by
computing dimension sizes once at open.

## netCDF3 vs memory bandwidth (wrfout d02, 802 MB)

| operation | 1 thread | best of 4-24 threads |
|---|---|---|
| memcpy, warm memory | 21 GB/s | 41 GB/s |
| big-endian -> native copy | 12.6 GB/s | 37 GB/s |
| allocate + first touch (page faults) | 6.1 GB/s | 17.6 GB/s |
| `readinto` from OS cache into a buffer | 6.5 GB/s | 21 GB/s (8 threads) |
| fresh `mmap` + copy | 2.7 GB/s | 7.0 GB/s |
| netCDF4-python, all variables | 1.9 GB/s | - |

The ceiling for "give me new native-endian arrays" is set by page faults on
the fresh output plus the cache-to-user copy, not DRAM: about 10-15 GB/s,
i.e. **~60-80 ms for d02 instead of 430 ms**. A fresh mmap is slower than
`readinto` here (mapping faults), so mmap is for lazy, partial access, not
for whole reads. Byte-swapping can ride along with the copy in L2.

## How people read: one variable, a window

| read | h5py | netCDF4 raw | netCDF4 m+s |
|---|---|---|---|
| d01 `T2` (open + read 156 kB) | 0.9 ms | 20.7 ms | 21.9 ms |
| d01 `T` | 28.6 ms | 46.6 ms | 48.3 ms |
| d01 `T[0, 0]` (one level) | 25.5 ms | 46.2 ms | 45.6 ms |
| VIIRS `I05` (83 MB) | 177 ms | 198 ms | 364 ms |
| VIIRS `I05[3000:3256, 3000:3256]` | 8.7 ms | 13.6 ms | 13.4 ms |
| MOD02QKM `EV_250_RefSB[0]` | 345 ms | 351 ms | 393 ms |
| MOD02QKM 256x256 window | **123 ms** | 126 ms | 125 ms |
| Metop-SG `vii_10690` (5.3 MB, 840 chunks) | 6.2 ms | 20.1 ms | 27.2 ms |
| d02 (nc3) `T` | - | 18.8 ms | 23.2 ms |

- netCDF4-python pays ~20 ms just to open d01 (210 variables' metadata).
  Reading `T2` from every hour of a WRF run is mostly opening files.
- One level of `T` costs as much as all of it: the chunk is (1, 62, 85,
  115), all levels.
- A 256x256 window of a MODIS 250 m band inflates a whole 17 MB chunk.
  Rows past the window need not be inflated at all (stop early), and a
  saved inflate checkpoint (32 kB window + bit position, as zran does)
  would make later windows cheap.
- Mask+scale doubles VIIRS (198 -> 364 ms): several numpy passes over the
  array.
