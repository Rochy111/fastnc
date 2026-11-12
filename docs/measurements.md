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

## Later experiments (after the design)

- **Double-literal inflate tables** (one lookup writing two literals when
  both codes fit in the 11-bit table): byte-identical, but slower on the
  quiet machine -- MODIS 824 -> 768 MB/s, VIIRS 1454 -> 1354, wrfout 1200
  -> 1133. In data that barely compresses, literal codes are 8-10 bits, so
  pairs rarely fit; the pairing pass and the extra branch cost more than
  they save. Not kept.
- **Output arena** (big outputs reuse already-mapped memory): kept.
  Repeated whole-file reads 1.3-2.5x faster (VIIRS masked 348 -> 139 ms).
- **Parallel single-stream inflate** for chunks bigger than a thread's
  share: kept. MODIS 250 m band 109 -> 23 ms.
- **Contiguous data as virtual chunks**: kept. GOES-19 full-disk C02 read
  per dataset 102 ms (h5py 2445).
- **Dense-attribute lookups by name hash** (lookup3, self-tested against the
  file's own index once per process): kept, but `Dataset` open is still
  5-8 ms on files of 150-210 variables (netCDF4-python 11-20 ms): the rest
  is header parsing and the DIMENSION_LIST attributes that do exist.
  `fastnc.File` opens in 0.2 ms (h5py 0.17).
- **Numba first-class function arguments** for per-codec specialisation:
  not cacheable (recompiled in every process); one small driver kernel per
  codec instead.
- **Peak memory** (`tools/profile/peak_memory.py`, fresh process, every
  variable read and kept, masked for fastnc / netCDF4-python, raw for
  h5py): VIIRS 3.3 GB fastnc, 3.3 GB netCDF4-python (results 2.3 GB; h5py's
  raw arrays 1.0 GB, peak 1.1); wrfout d01 0.40 / 0.46 GB; GOES full-disk
  C02 4.7 / 6.2 GB. fastnc's peak counts the mapped file's pages (shared
  with the OS file cache) and the raw arrays a masked read made, kept in
  the arena for the next read (`fastnc.release()` returns them); its
  committed memory stays ~100 MB because both are mappings.
- **Inflate table building** (single thread, every deflate chunk of the
  real files, `tools/profile/inflate_check.py`-style): the fixed code's
  tables were rebuilt for every fixed block -- Ku's 13,279 of 15,133 chunks
  are fixed-code streams of ~25 bytes, so building cost 4 us a chunk
  against well under 1 for decoding. Kept between blocks: Ku 59.6 -> 18.7
  ms. Dynamic tables built as libdeflate builds them (sorted by length,
  table doubled between lengths, no clearing pass, no strided writes per
  slot): header + tables 5.3 -> 3.8 us a block, wrfout 158 -> 151 ms,
  VIIRS 655 -> 629, MODIS 478 -> 461, Ku 18.7 -> 13.9. Literal stores at an
  unsigned index (numba wraps signed ones): another ~2%. Kept.
  conda's libdeflate 1.x through ctypes on the same chunks: wrfout 114 ms,
  VIIRS 545, MODIS 362, AVHRR 59 (fastnc now 148 / 606 / 447 / 71).
- **Not paying off in the inflate loop** (no change beyond noise, so not
  kept): the fast loop as a separate function (its bit buffer stops being
  spilled to the stack, but no faster: 147.7 vs 149.7 ms wrfout); the bound
  checks merged into two precomputed limits; libdeflate's entry layout
  (bits to consume in the low 6 bits, so `buf >>= entry` needs no mask, and
  length/distance code plus extra bits consumed in one shift): 145.2 vs
  145.4 ms wrfout, 607 vs 605 VIIRS. The loop is bound by branch misses on
  literal/match, not by the shift chain.
- **Unshuffle a word at a time** (`_chunks._unshuffle2/4/8`): byte by byte
  the shuffled placement ran at 3.1 GB/s (int16) / 3.6 (float32) on one
  core, a third of a shuffled read's decoding; with shifts and masks over
  a word of each byte plane (LLVM vectorizes it) 25 / 15-21 GB/s. Whole
  file raw, every thread: VIIRS 67 -> 54 ms, wrfout 41 -> 38, AVHRR 8.2 ->
  7.3; one thread VIIRS 938 -> 709.
- **Whole-dataset plans without the per-dimension intersection** (a load):
  wrfout raw load 38 -> 32 ms, Ku 11.7 -> 9.7.
- **Decoded-chunk cache for small reads of shuffled chunks** (kept from a
  chunk's second read on, 32 MB in all): repeated `T[0, 0]` of wrfout 3.8
  ms -> 0.09, `T2[0, 10, 10]` 118 -> 45 us, a 16x16 VIIRS window 0.89 ms ->
  0.074; a fresh open's first read unchanged (admitting on the first read
  cost +22% there: four whole 2.4 MB chunks placed and copied).
- **Concurrent readers**: eight user threads reading one open file (raw,
  masked, windows, load) get the values one thread gets, through the zlib
  path and the kernels (`tests/test_threads.py`).

## Round 2 (2026-10-30)

Same machine. Scripts in `tools/profile/`.

**Fresh processes** (`fresh.py`, best of 3 processes, import + open + read).
Importing numba and loading the cached kernels took 0.7-0.9 s (numba 300
ms, the kernels' cache 350 ms, mostly `open_code` on ~300 modules -- Windows
file opening). Started in the background at the first read (the old
design), it made that read 3x slower: the pool's threads queue for the GIL
after every zlib call while the import holds it (WRF d01 `T`, masked: 249
ms with no background load, 824 with). Loading first (`fastnc.warm()`)
cost ~1.1 s in every case. So reads now never wait for numba until 1 s of
reading without it; with ctypes libdeflate and numpy the paths before
that: WRF d01 masked `T` 715 -> 222 ms (netCDF4-python 238), whole VIIRS
file masked 1176 -> 613 (3306), MODIS 1 km 921 -> 547 (1380), Metop-SG
630 -> 365 (533, after gathering unfiltered chunk grids at once).

**Non-temporal stores** (`ntstore.py`: u2 -> f4 scale over 941 MB, plain
stores vs scaled into an L1 scratch and streamed out with
`!nontemporal` 32-byte stores): 24 threads 41.7 vs 40.8 ms, 8 threads 45.6
vs 48.0 -- no gain (Golden Cove already avoids the read-for-ownership on
full-line writes). Not used.

**One chunk on many threads from smaller sizes** (`par_min.py`, PAR_MIN 4
MB / 1 MB / 512 kB / 256 kB): WRF `T` 4.50 / 4.26 / 4.00 / 3.86 ms, but
whole-file loads 22.9 / 26.8 / 25.7 / 28.6 ms. Kept at 4 MB.

**Mask/scale fused with decoding** (`fused_check.py`): GOES-19 C02 Rad 140
-> 117-127 ms, C13 10.8 -> 8.9, VIIRS I05 13.9 -> 11.6, AVHRR 2.9 -> 2.4;
WRF (four chunks per variable, mask only) slower fused (5.2 -> 6.1): such
reads keep the usual path.

**GPU pipeline** (`gpu_breakdown.py`, GOES-19 C02 Rad, 897 MB): CPU read 82
ms, pinned upload 20.6 ms; band-by-band reads with uploads 111 ms; one
decode with each band uploaded as it completes 90 ms (masked: 128 -> 95,
against 120 on the CPU). Band size (`gpu_bands.py`): 8 MB a little ahead
of 32, 4 and 2 MB.

**Readahead across variables** (`readahead_check.py`, loop over variables
/ xarray load): WRF d01 132 -> 60 / 152 -> 86 ms, Ku-band 71 -> 23 / 80 ->
31, RCM 20 -> 8 / 31 -> 18, AVHRR 20 -> 14 / 38 -> 38.

**CI findings.** numpy's masked arrays carry their `fill_value` type through
arithmetic differently in numpy 2.5 than before, and numpy 1's default int
is 32-bit on Windows only; NaN x NaN keeps the payload of the operand the
compiler that built numpy ordered first (MSVC vs gcc builds differ). See
internals, "Values that depend on the platform's numpy".

## Round 3 (2026-10-30)

**xarray decoding by lookup table** (8 / 16-bit data; xarray `.load()`):
GOES-19 C02 810 -> 191 ms with the table looked up after the read, 160 with
it looked up as each chunk is decoded; C13 63 -> 27 -> 22; VIIRS I-band
group 417 -> 132 -> 110; AVHRR 38 -> 28 -> 27.

**The same idea for netCDF4-python's masked read** (value table + flag
table, numba): GOES-19 C02 masked read 118 ms, a bare value-table lookup
while decoding 103 ms before writing any mask -- at most 5-10%; not done.
**In numpy** (the no-kernel path): `np.take` is ~2.3 ns per value on one
thread (default mode; `mode='clip'` 1.6x faster), against ~0.3 ns per ufunc
step, so VIIRS masked went 232 -> 864 ms: reverted. (The xarray no-kernel
lookup uses `mode='clip'`.)

**No-kernel chunk path.** Per-chunk Python runs under the GIL, so on 24
threads it was the limit: GOES-19 C02 raw 233 ms where libdeflate alone
inflates every chunk in 57. Working out per-chunk numbers for all chunks
at once (`_chunks._Quick`): C02 233 -> 145, VIIRS 169 -> 92, WRF d01 63 ->
47. Many small deflated chunks inflated into one staging buffer and placed
by a few numpy copies (`_zgrid`): Ku-band 199 -> 73 ms on 24 threads; on 1
thread 32 ms (2 threads 37, 4 threads 60: each ctypes call gives up the GIL
and takes it back, so for chunks of a few hundred bytes threads only
queue) -- such chunks go on one thread (netCDF4-python: 75 ms).

## Round 4 (2026-10-30): phase by phase

`tools/profile/phases.py` splits each operation into open, chunk planning,
time inside the pool (with how busy the asked-for threads were), mask /
scale, readahead and the rest. What it showed, and what came of it:

- **Handing work to the pool** cost 4.4 us per helper through
  ThreadPoolExecutor (futures, a queue): ~100 us to wake 23, 160 times in
  a WRF masked loop. fastnc's own workers parked on a lock each: 2.8 us
  for any number. Ku-band one variable 4.8 -> 3.8 ms, load 12.1 -> 10.4.
  Helpers still join 55-300 us late (each needs the GIL to enter the
  kernel), so short jobs stay mostly on the caller: thread efficiency of
  12-40% on the small files is that, not overhead.
- **Opening** is most of a small read (WRF d01: 6.3 of 9-11 ms). The root
  group's 150 attributes were all parsed to look for one; attribute names
  were decoded to look them up; datatypes parsed per dataset; each
  attribute header unpacked field by field: WRF open 6.3 -> 5.2 ms,
  Metop-SG 4.2 -> 3.7, xarray open 21.3 -> 20.4 (half of that is xarray's
  own).
- **Inflate** is the floor of the big reads (GOES-19 C02: 60 of 63 ms in
  the pool at 99% efficiency). libdeflate, where installed, now decodes
  whole chunks from inside the numba kernel -- called through a function
  pointer passed as an integer, so the kernels stay cached -- with this
  inflate taking over whenever it declines: load() VIIRS 47 -> 40 ms,
  GOES-19 C02 65 -> 54, AVHRR 5.4 -> 4.8, Ku 3.4 -> 3.1, WRF on one thread
  174 -> 135 (all threads: 18.9 -> 18.5, it is bound elsewhere); MODIS
  (one 33 MB stream per band, decoded in parallel by _pinflate) unchanged.

## Round 5 (2026-10-30): from what the streams contain

`stream_stats.py` walks the real deflate streams symbol by symbol:

| file | chunks repeated | literals of symbols | bytes from literals | bits a literal | mean match | blocks / MB |
|---|---|---|---|---|---|---|
| WRF d01 | 35% | 88% | 28% | 8.0 | 19 | 22 |
| VIIRS | 51% | 91% | 23% | 7.8 | 33 | 19 |
| MODIS 1 km | 0% | 90% | 34% | 7.5 | 17 | 23 |
| GOES-19 C02 | 1% | 86% | 14% | 8.1 | 35 | 13 |
| GOES-19 C13 | 0% | 92% | 25% | 8.0 | 36 | 19 |
| RCM | 9% | 74% | 5% | 7.9 | 57 | 4 |
| Ku | 0% | 99% | 82% | 8.0 | 24 | 2886 |

`inflate_costs.py` times 2,200 chunks one by one and fits time against
their symbol counts (R2 0.90) -- costs without hardware counters:

| one of each | fastnc inflate | libdeflate | share of the time |
|---|---|---|---|
| literal | 1.12 ns | 0.89 ns | ~39% |
| switch between literals and a match | 10.8 ns | 10.0 ns | 33-39% |
| match of distance 2-32 | ~55 ns | ~30 ns | 10-17% |
| byte a match copies | 0.08 ns | 0.15 ns | 8-20% |
| block | 631 ns | 569 ns | ~2% |

So a match costs about 19 ns in every inflate (100 cycles), and it is
not the copying.

**A wrong result found by asking a speed question:** making GPU reads skip
the dimension scan meant checking what `var.shape` was for, and it turned out
to differ from what was read. A variable with fewer records than its
unlimited dimension read 2 records where netCDF4-python gives 6 with fill;
on the GPU `[1:5]` gave rows of uninitialized memory, unmasked. Every test
passed, because no file of the corpus has such a variable, though it is the
ordinary thing in a record file. `tests/test_short.py` now writes them, in
every type, and compares every index and read path with netCDF4-python.
Whether a variable is short takes the longest variable using its dimension,
so a read would need every variable's metadata again; `_cannot_be_short` settles
it from the dataspaces alone (`H5File.extent`, usually the first message of a
header): no unlimited axis in the file reaches past this variable's unlimited
axes, and its fixed axes have no unlimited scale behind them
(`DIMENSION_LIST`). WRF d01 one variable stays 5.0 ms; the check cannot be
turned off, and anything it cannot settle takes the whole scan.

netCDF-C itself was no reference in two corners. Short along two unlimited
dimensions it returns uninitialized memory, and values of other variables.
And before 4.10 (4.9.3 here) a stepped read past the stored records drops
stored values: `short[1::3]` of `[1, 2, fill ...]` gives `[fill, fill]`, against
its own `short[...]`, while 4.10.1 gives `[2, fill]`, as fastnc does everywhere.

**Worked:**
- Opening reads what is asked for. `Dataset` parsed every variable's header,
  shape, type and dimensions, and worked out unlimited lengths, to read one
  variable: that was the gap to `File`. Now `d['T']` reads `T`'s link and
  header alone, and the rest comes when something wants it, whole, so
  phony dimensions keep the numbers netCDF-C gives them. Open alone goes 4.9
  -> 0.1 ms on WRF d01, but that work is moved, not gone. What counts is open
  + one variable: WRF d01 8.7 -> 5.0 ms (with the check for short variables
  below), a window 7.4 -> 3.8; Ku-band 3.1 -> 1.9, RCM 1.6 -> 1.0, Metop-SG
  5.2 -> 4.7. A file loop reading one variable of each gains the most.
  Whole-file reads are level.
  A variable is told from a dimension scale by searching its attribute
  messages for the name `CLASS`, in C. The name, not the value: a value can
  live in the global heap, a name cannot. Dense or shared attributes, or a
  buffer that cannot be searched, fall back to the whole scan.
- Masking many arrays in one pass, with nothing to choose. `load()` decoded
  all variables together, then masked them one after another, waking every
  thread for each. Giving arrays below some size a thread each helped, but
  the size mattered: 16-64 MB was best here, and 256 MB left MODIS 250 m at
  73 ms (49 at 32 MB). A rule from the work itself (an array alone if it
  exceeds one thread's fair share) lost to the fixed size: equal arrays
  all count as too big. So now no such choice exists: `_pool.walk` starts
  threads spread over the arrays, and each goes on to the next unfinished
  one, taking blocks off its counter. Small arrays are done by whoever
  arrived; a big one collects every thread as the rest end. Against the best
  fixed size: Metop-SG 38.3 / 37.3 ms, RCM 5.9 / 5.9, AVHRR 8.7 / 8.7, MODIS
  1 km 53.9 / 53.6, MODIS 250 m 51.6 / 49.3, and level on 1, 4 and 8 threads.
- Attributes decoded when asked for. A masked read wants eight names; each
  variable's every attribute was decoded for them (WRF d01: 1,770).
- One plan for variables that agree on those eight attributes (186 of WRF's
  200), kept across files. It holds a small dict of its own, not the
  variable's attributes: those would keep a closed file's mapping alive.
- `FUSED_LOAD_MIN`, 32 MB, was this CPU's L3 (30 MB) in disguise: below it,
  the raw values are still in cache when the mask pass reads them, so fusing
  saves nothing and costs a pass of the pool (Metop-SG 35 -> 46 ms at 2 MB;
  GOES-19 C13 22 -> 14 above it). Now it is read from the machine.
- Masked `load()`, last commit -> now, best of 4 alternating rounds: WRF d01
  44.5 -> 32.7 ms, in a loop 53.5 -> 41.6, WRF d02 64.5 -> 52.8, MODIS 1 km
  60.5 -> 50.5, GOES-19 C13 14.7 -> 13.4.
- What a fresh process pays before it reads, beyond numpy (150 ms of it, not
  fastnc's): `threading.Thread.start()` waiting for each thread, 5 ms, gone
  with `_thread.start_new_thread`; importing `json` for a 300-byte file, 5
  ms, gone with `ast.literal_eval`; a global lock taken per chunk to ask
  whether a library had been found. And one of mine, found comparing
  against the pushed code: importing numpy.ma before noticing a read is raw.
- A first read's Python per chunk, without numba: `_zlib.stream_inflater`
  holds addresses instead of asking numpy for them chunk by chunk. GOES-19
  C02 (3,616 chunks) 164 -> 148 ms. Libdeflate alone would take ~55: the rest
  is two strided numpy copies per chunk and the GIL changing hands for each.
- The benchmark's whole-file cells were best of 2, one round of it straight
  after netCDF4, while the README said best of 5: the same cell gave GOES-19
  C13 21.7 ms in one run and 69.4 in the next, and MODIS 1 km 93.6 where
  loops gave 50. With five rounds the two runs agree (MODIS 56.7 and 53.6,
  C13 16.1 and 14.7).
- The thread that calls fastnc, kept off the efficiency cores. A script
  with no focused window is background work to Windows, and about 2 s in
  its busy main thread is moved to an E-core, for good. The pool's threads
  had been protected from this for a long time, and so had every timing
  script (`fast_thread`), so no benchmark ever showed what users got. WRF
  d01 read over and over from an ordinary script:

  | | read | open | plain Python | main thread on |
  |---|---|---|---|---|
  | first 2 s | 34 ms | 5.7 | 9.0 ms | P-cores |
  | after 2 s, before | 55 ms | 10.7 | 14.6 ms | E-cores (16-23) |
  | after 2 s, now | 36 ms | | 9.0 ms | P-cores |

  Masked reads went 50 -> 90 ms in the same step. The step comes with no
  file read differently, no new memory and arithmetic on all threads
  unchanged; garbage collection was suspected and is not it (the step
  comes with the collector off, and not on a forced collection). Now any
  thread that opens a file or reads declares itself once. The tables do
  not change: they were always measured this way. They now hold for
  users too. The tradeoff: that thread stays at this QoS afterwards.
- Proving nothing is masked while decoding, not scanning afterwards. Most
  masked variables hold no fill value at all, yet each was read back whole
  and a mask built. With the shuffle filter a chunk comes out of inflate as
  byte planes, so the plane of top bytes can be tested alone: if the fill
  value's top byte is not there, the value is nowhere in the chunk. That
  is a quarter of a float's bytes, still in the core's cache, at 50 GB/s
  against ~1 GB/s for inflate. A variable counts only if every one of its
  chunks was tested and passed; anything else keeps the pass. netCDF-3 has
  no planes: there values are tested word by word just after the swap.
  Best of 4 rounds in alternating processes, whole file masked, ms:

  | | before | check | + HDF5 proof | + netCDF-3 test |
  |---|---|---|---|---|
  | WRF d01 `load()` | 52.1 | 51.1 | 44.5 | 44.4 |
  | WRF d01 loop | 61.2 | 60.5 | 53.3 | 53.8 |
  | WRF d02 `load()` | 63.4 | 65.0 | 62.9 | 62.9 |
  | WRF d02 loop | 90.2 | 83.2 | 82.5 | 74.2 |
  | VIIRS `load()` | 105.3 | 105.3 | 96.8 | 95.1 |

  The "check" is a lighter pass in `run_plan` for what no proof covers:
  word tests that stop at the first hit, with no mask written. In a run of
  its own, taking it away from the rest cost WRF d02 2 ms in `load()` (64.7
  against 62.6) and 6 in the loop (78.1 against 72.4), nothing elsewhere.
  MODIS and GOES-19 did not change: nothing there qualified (their variables
  are scaled, or have ranges or values the test cannot prove).
- HDF5 stores a whole chunk even past a variable's edge, and fills that
  corner with the fill value. WRF's chunks run past the edge in most
  variables, so a first version proved 10 of 202 MB. Now a hit in the plane
  is looked at again over what the read selected: 202 of 202 MB.
- Threads by the decoding to do, not the output: one column of VIIRS I05
  (13 kB from ~200 chunks, on one thread) 60.8 -> 3.7 ms.
- libdeflate without its Adler-32 check, which fastnc's inflate never did:
  4-7% (VIIRS 539 -> 514 ms, MODIS 383 -> 355, GOES-19 C02 822 -> 800).
- Short periods in fastnc's inflate widened to 8 bytes, not copied byte by
  byte: 1-3% (the fit made this look bigger).

**Did not work:**
- Holding threads to cores for reads after a pause. A read straight after a
  pause is slower than one in a loop (masked, 1 s pause: MODIS 1 km 57 -> 65
  ms, VIIRS 97 -> 102, Metop-SG 39 -> 43). Windows' core parking looked the
  likely cause, and one run with every thread pinned seemed to remove the
  cost entirely. It did not hold up. Holding cores for a job's length, workers
  living on a CPU each, the caller pinned, and both: none helped in
  alternating processes (best to worst within the pause cost itself), and
  bound workers made some reads slower (Metop-SG +15%). The first result was
  variation between runs: the same default setup gave MODIS 67 ms in one run,
  91 in the next. What is left looks like the cost of waking up.
- Lowering `FAST_MIN_SIZE`. A profile put the reference path at 115 us an
  array; by the clock it is 8-25 us, and the two paths are within 1 us below
  32k values (the fast one loses when values are masked). The profiler adds
  more than the work. Timing phases by the clock changed the priorities.
- Fusing small arrays to spare memory traffic (see `FUSED_LOAD_MIN` above).
- Scanning was less of a masked read than it looked. With the scan removed
  outright WRF d01 takes 41 ms, not the raw read's 30: 4.6 ms is parsing
  every attribute of 200 variables to find the few that matter (the rest
  of the gap was not pinned down). WRF d02's `load()` barely moved (63.4 ->
  62.9) although 743 of its 802 MB are proved; its raw read is 48 ms and
  dropping the pass for every array reached 53, so most of what is left
  goes with the ~110 smaller arrays, for a reason not found. One HDF5
  variable read alone gains nothing (WRF d01 `T` 10.8 -> 10.7 ms) and
  takes no probe.
- netCDF-3 runs copied and swapped in one pass out of the mapping, not read
  then swapped. It looked like a win (WRF d02 open + load 56 -> 46 ms) and
  was built, then taken out: unmapping 800 MB of touched pages takes 50-80 ms
  (Windows), and a pool thread keeping the last job alive had moved that out
  of the timing. Whole open + load + close, masked: 90 ms with reads, 118
  with the mapping; on one thread reads win outright. Keeping the mapping
  until later (a background unmap) would leave the file undeletable just
  after `close()` on Windows. Benchmark tables caught this, in the masked
  and 1-thread rows; the fast row alone would not have.
- Copying 64 bytes a match unconditionally, to avoid a branch miss on the
  length: no change at all (1019 -> 1019 ms): the copy is not the cost.
- Decoding several chunks at once in one thread (interleaved streams),
  worked out from the fit and not built: literals, the only part it speeds
  up, are ~39% of the time, and every match stalls all streams; at best
  ~1.4x before its overhead, against libdeflate's lead over any numba code.
- A second hardware thread gives 1.47x on one core (a second core 1.85x), so
  hyperthreading already takes much of what interleaving would.
- Pinning threads: 24 threads on a long decode (3 GB of GOES-19) 25.9 GB/s
  placed by the OS, 26.8 with an ideal-processor hint, 27.5 pinned. Not
  worth what pinning does to other processes. (Short runs misled: 2 ms of
  work measured 34 GB/s on 16 threads.)
- VirtualLock to fault fresh memory in one call: 237 ms per GB, against 91
  touching it on one thread and 19 on 24. Page faults do scale with threads
  (an older note here said not).
- Repeated chunks: decoding only one of each saves 12% of VIIRS's inflate and
  nothing elsewhere: repeats are constants, the cheapest chunks there are.
- The mapping for HDF5's scattered unfiltered chunks: Metop-SG open + load
  22 -> 36 ms (again on the same file 13 -> 8): staging stays.
- Lookup tables for netCDF4-style masked reads: at most 5-10%; in numpy,
  `np.take` is 2.3 ns a value and VIIRS went 232 -> 864 ms.
- numba's shifts: suspected of a guard for counts of 64 or more on the
  bit buffer's path. There is none (`shrx`). The literal path is 13
  instructions and a dependency chain of about 7 cycles; libdeflate's lead
  there is not numba's waste.
- CPU and GPU decompressing together: nvCOMP 7.5-9.7 GB/s on files of many
  chunks (0.66 on WRF), the CPU ~26 GB/s: 29% more on the inflate step, about
  10% of a GPU read, for GPU versions of unshuffle and placement. Not built.

**Where a big read's time goes now.** GOES-19 C02 Rad (897 MB): 48 ms on the
CPU, 53 to the card with the upload hidden; masked 78 and 63. About two
thirds of the CPU read is inflate at 99% thread efficiency, the rest writing
the output at memory bandwidth.

## Round 6 (2026-10-31): a fresh process, cold files, what kernels cost

**Fresh processes run at full speed.** Importing numba was the biggest thing
left in a process's first read. Its code is 13 MB for what reads use (48 MB
with every codec), and all numba and LLVM do is find it and fix its addresses.
`_image` (see internals) links it into a file and calls it with numpy and
ctypes. Parity first, same process, best of 8: GOES-19 C02 masked 82.7 ms from
the image, 85.2 from numba; raw 51.8 / 53.4; VIIRS load 47.1 / 46.7; WRF d01
load 27.1 / 27.8: the same machine code. All 8,555 tests pass through it.

Fresh process, import + open + read, best of 3 (ms; numpy's own import is
~140 of each):

| | netCDF4 | no image | image |
|---|---|---|---|
| WRF d02 all variables | 763 | 1004 (numba loaded) | 248 |
| VIIRS all | 3602 | 539 | 399 |
| MODIS 1 km all | 1431 | 338 | 276 |
| Metop-SG all | 531 | 329 | 266 |
| GOES-19 C02 `Rad` | 4605 | 517 | 353 |
| GOES-19 C02 all | 6650 | 649 | 545 |
| WRF d01 all | 803 | 267 | 231 |
| WRF d01 `T` | 251 | 169 | 172 |
| AVHRR `avhrr_ch4` | 235 | 166 | 165 |

What is left above "warm read + import" is page faults on fresh output: see
below. Opening an image takes 0.4 ms, reading what CPUID says 0.15; a unit
0.6 ms. `hashlib` was 4 ms of this: the source stamp uses `zlib.crc32`.

**numba's cache stamp** covered every source file, so an edit of the front end
recompiled everything; now only files with kernels.

**Nothing stops to compile.** A fresh install compiled for ~40 s in the middle
of a read a second in, and a few seconds more for each new type. Now the build
runs in a process of its own and reads go on without kernels until it is done.
A script reading WRF d01 in a loop from empty caches: 1,042 reads in 240 s, the
slowest 335 ms; ~280 ms each until the image appeared at 202 s, then 34 ms, in
the same process, numba never imported there.

**Metop-SG's File window** was the wrong place to look for a gain: at 1.2 ms in
alternating processes before any of this, it was never at the 1.7 of the tables.
In the benchmark it is 1.87 and 1.60 ms in two runs against h5py's 1.51 and 1.39,
a loop of its own 1.15 against 1.44. That file's window still goes either way.

**A path no longer scans the file.** `d['data/measurement_data/x']` scanned
every group to find the groups in the path, and variables with attributes in
dense storage looked like possible dimension scales. A variable with no
unlimited axis needed a look through every dataset to know it is not short;
what is attached to its own axes says so. Metop-SG, Dataset: window 4.6 -> 2.7
ms, one variable 5.3 -> 2.9. xarray opening a root that holds only groups:
4.4 -> 0.7 (h5netcdf 2.2).

**GPU reads of one band** no longer start an upload thread: WRF `T` 5.5 -> 5.1
ms, AVHRR 2.5 -> 2.1. Most of the "loss" in the tables was the benchmark: its
CPU column uploaded a masked array's data, not its mask.

**Cold files, measured at last.** The benchmark never measured them: flushing
the OS cache takes admin rights on Windows. Two ways that do not:

- Opening a file with `FILE_FLAG_NO_BUFFERING` evicts it only if it has not
  been mapped. A first table made this way looked wonderful (VIIRS cold 45 ms)
  and was wrong: counting bytes from the disk (`IOCTL_DISK_PERFORMANCE`, no
  admin needed) showed 0. Mapped pages survive it.
- A copy written with `FILE_FLAG_NO_BUFFERING` is not in the cache. Rewriting
  it before each run gives a file that is truly cold; the counter confirms it.

Whole file, raw, from an NVMe (WD SN850X), median of 5 (ms; requests are what
reached the disk):

| | warm | cold | requests | cold, staged | requests | h5py cold | requests |
|---|---|---|---|---|---|---|---|
| VIIRS | 45 | 93 | 4,743 | 106 | 159 | 1663 | ~640 |
| MODIS 1 km | 45 | 75 | 2,642 | 125 | 132 | 970 | ~140 |
| WRF d01 | 26 | 52 | 1,220 | 65 | 113 | 505 | ~320 |

So from flash a cold file costs about twice a warm one and stays 10-18x ahead
of h5py, because 24 threads faulting at once are a queue depth of 24. But
4,743 requests of 59 kB is what a disk head or a network pays for, 7x h5py's.
Staging takes 30x fewer and costs 10-60% from flash, so it depends on the
storage: see `_pio.Reader.slow`. No spinning disk or network share was at hand:
the request counts are measured, the gain on such storage is not. WRF d02's
141 ms cold is the NVMe's limit (5.7 GB/s).

**Two benchmark runs do not compare two versions.** The full benchmark's fastnc
rows moved by up to 2x between runs of the same code. Yesterday's commit against
today's, in alternating processes on the same machine, is what settles it, and
it found what the tables could not: image kernels were level with numba's on
big reads, as measured above, but slower on many small calls, MODIS 250 m window
1.47x, WRF d02 masked 1.17x. Same code, image against numba: 1.32x and 1.18x.
Flattening arguments in Python cost 11.6 ms of thread time in a d02 load (1,570
calls), under the GIL, with 24 threads queuing for it. Hence the direct entry:
a call 5.5 -> 1.8 us (numba: 16). And a per-file handle for the storage probe
cost AVHRR 10%: it now touches the mapping instead.

Current code against the code before this round, 6 rounds of alternating
processes, median of the best of 7 reads each (ms):

| | before | now | |
|---|---|---|---|
| WRF d01 `load()` | 25.7 | 21.9 | 0.85x |
| WRF d01 `load()` masked | 31.4 | 27.4 | 0.87x |
| Ku-band `load()` | 7.50 | 6.38 | 0.85x |
| AVHRR variable, masked | 1.97 | 1.82 | 0.92x |
| WRF d02 `load()` | 44.2 | 44.9 | 1.01x |
| WRF d02 `load()` masked | 50.7 | 54.5 | 1.08x |
| MODIS 250 m `load()` | 40.6 | 40.7 | 1.00x |
| MODIS 250 m File window | 6.00 | 6.78 | 1.13x (best of the rounds 5.82, 5.47) |
| GOES-19 C13 `load()` | 6.33 | 6.50 | 1.03x |
| Metop-SG File window | 1.13 | 1.16 | 1.03x |

Where the Python of a load went (`tools/profile/quick.py`, ms): chunk tables and
plans, WRF 4.2 -> 0.1, Ku 2.1 -> 0. What is left is spread thin, ~15 us a variable
in plain Python.

**What the benchmark never asked.** Reading whole files and one window hid the
biggest gaps of all. A survey of everyday patterns against the reference (`v[i]`
in a loop, `visititems`, columns, crops, points) found:

- Row by row, h5py won 3x to 8x: 1,600 rows of VIIRS 139 ms against 42, GOES-19
  C13 132 / 35, 885 of Ku-band 98 / 13. An 80-row sample had shown a tie. Each
  read planned over the whole chunk table (numpy over every chunk, O(chunks) a
  read). `ChunkIndex.near` goes from a box to its chunks through a grid table,
  `_hit` answers a read in a decoded chunk with no plan (16 us), and `_ahead`
  decodes the chunks after one read moved into, as many as have been right so
  far, twice over, in one pass of the kernel. 46, 45, 14 ms.
- Uncompressed data was worse still: rows of Metop-SG's chunks 81 ms against
  h5py's 4.5, of AWIPS's contiguous array 23 against 0.8, for what is a memory
  copy, because each went through a plan and the batch kernel. `_chunks._plain`
  works the place out with integers: 99 -> 6.2 us a row (h5py 4.4), 91 -> 6.3.
- Keeping chunks from a dataset's second small read on fixed rows and cost
  random crops 37 -> 57 ms. What tells a loop from random reads is direction: a
  chunk next to the last one (`_beside`).
- `fastnc.File` had no readahead: a loop took 2-3.5x `load()`. With it, WRF 79
  -> 43 ms. The first try made things worse (80 -> 102): it read in creation
  order and `visititems` goes by name, 395 arrays decoded and 6 used. Now
  `visititems` says so, and a loop's first two reads settle the rest.
- Thread rules came from inflate. Masks and copies go ~64x faster:

  | scan | 1 thread | 8 | 24 |
  |---|---|---|---|
  | 256 kB | 17 us | 80 | 163 |
  | 1 MB | 33 | 43 | 148 |
  | 64 MB | 2895 | 677 | 758 |
  | 256 MB | 12156 | 3196 | 3363 |

  24 threads never win, and below ~2 MB a thread helpers only cost.
  `_pool.memory_threads` caps at the machine's fast cores. A mask / scale pass
  is not that fast (`PASS_SPEED`): treating it as a scan cost AVHRR 39%.

**Did not work:**
- Cutting chunks over 256 kB on many threads when a read has fewer chunks than
  threads, to use idle threads: WRF `T` 4.56 -> 5.91 ms, AVHRR 1.25 -> 2.35. For
  chunks this small, the search for block starts costs more than threads give.
- A native parked pool: a share with nothing to do costs 8-80 us. What looked
  like hand-off was too many threads for small work.
- Trimming a fresh process's first load: beyond page faults and `numpy.ma` (8 ms,
  masked results need it), nothing there is over 3 ms.
- Touching fresh output on the pool before decoding. The first read of a big
  array costs twice the next (GOES-19 C02 masked 240 ms, then 85), and at 19
  ms per GB in parallel touching looked like it would pay. It does not: the
  time spent touching (260 ms for that read) equals what decoding then saves.
  That cost is the faults, not their contention.
- Huge pages for output on Linux (`MADV_HUGEPAGE`, under WSL2): 2 GB of first
  touches 142 ms on 24 threads with them off, 146 with them on. Zeroing the
  memory is the cost, in either size of page.
- `PrefetchVirtualMemory` on the mapped file before decoding: 326 requests,
  but VIIRS 45 -> 56 ms when cached (11 ms for the call) and MODIS cold 75 ->
  119. Staging does the job with no such cost when cached.

**After a pause.** `idle_reads.py`: reads in a tight loop against reads after
0.05-1 s of quiet: VIIRS I05 4.7 -> 6.8 ms, GOES C13 3.7 -> 5.3, MODIS 21.5
-> 26, GOES C02 and WRF unchanged. A millisecond or two for cores to wake and
clocks to rise, which a library cannot avoid; loop benchmarks flatter
short reads by that much.

## Round 7 (2026-11-02): reads nothing ever had to plan

Small reads against h5py and netCDF4-python. One read, repeated (us):

| | row | h5py | pixel | h5py |
|---|---|---|---|---|
| Metop-SG | 1.82 | 2.44 | 1.66 | 2.10 |
| AWIPS | 1.50 | 2.11 | 1.39 | 1.96 |
| VIIRS I05 | 1.35 | 2.50 | 0.98 | 2.54 |
| Ku-band | 1.13 | 2.39 | 1.08 | 2.00 |
| GOES-19 C13 | 1.34 | 2.51 | 0.98 | 1.98 |

Loops from a fresh open, processes alternating with the last commit, best of
three rounds (ms):

| | before | now | reference |
|---|---|---|---|
| VIIRS, 1,600 rows | 49.3 | 13.8 | h5py 42.8 |
| GOES-19 C13, 1,600 rows | 47.9 | 9.1 | h5py 34.1 |
| GOES-19 C13, 2,000 points | 73.3 | 13.7 | h5py 52.9 |
| Ku-band, 885 rows | 13.6 | 8.7 | h5py 13.1 |
| Metop-SG, 840 rows | 6.9 | 2.7 | h5py 4.3 |
| WRF `XLAT`, 100 rows | 10.1 | 1.02 | h5py 0.76 |
| WRF `T2`, whole, 200 times | 33.3 | 2.8 | h5py 2.3 |
| Metop-SG, 30 random crops | 5.4 | 4.4 | h5py 5.8 |
| WRF `T2`, 30 random crops | 2.0 | 0.93 | |

Open, one variable, close (us): WRF `T2` 399 against h5py's 555, RCM 686 / 2095,
Ku-band 1292 / 11954, AVHRR 2948 / 54655, Metop-SG 1625 / 4492. Through
`fastnc.Dataset`, a row (us): Metop-SG 50 / netCDF4 222, AWIPS 26 / 105, VIIRS
50 / 161, Ku-band 19 / 144, C13 75 / 190, WRF 34 / 292.

What is left behind h5py is two small variables of WRF, read many times: a
tenth of a millisecond here or there. `T2` and `XLAT` are four 39 kB chunks.

- **A surveyed case hid the biggest bug.** `qk.py` timed C13's first 1,600 rows,
  mostly fill. Rows through the middle, and points (`mid.py`), no better than
  h5py with every core free: 84 chunks took 112 decoding passes, about one a
  chunk. Chunks decoded ahead were not in `index.beside`, so the first read past
  a box looked like a jump: the run restarted, that chunk was decoded twice,
  and the count of chunks ahead never passed one. Doubling had not worked since
  it was written; the gain measured then was this half-working form. Now 16 passes.
- **`_quick`.** The cost of `x[i]` was never the copy: reducing the index, an
  output, planning, converting, for a read whose place is a sum. A closure made
  once holds the sum's terms. Across chunks it goes piece by piece, which replaced
  90 us of plan for a row over two chunks.
- **A link by name.** `group_links` read all 210 of WRF's links through the
  fractal heap to find one. `btree2_hashed` goes down the name index by
  lookup3 hash, `_symbol_link` bisects the symbol table's B-tree, and
  21,057 names and 380 absent ones gave what the whole list gives, in groups of
  up to 5,000 links of either kind.
- **Numpy's fixed cost.** `_plan` made some forty small arrays: 70 us, three times
  over for a miss of two chunks, more than decoding them. `_few_plan` does those
  sums in integers (391 -> 273 us for such a read). A test compares it with
  `_many_plan` on 500 random boxes.
- **What attributes cost.** A masked read asked ten times for attributes that do
  not change, and cast the same five to the variable's type each time. Remembered
  on `Attrs`, with `_quick`, a row through `Dataset` halved.

**Did not work:**
- Keeping every chunk under 256 kB from its first read: WRF crops 2.05 -> 1.25
  ms, but Ku-band 7 -> 20. Its chunks are 2.4 kB, and what keeping costs is ~50 us
  of Python a chunk, whatever the size, which a crop over 8 of them pays 8 times.
  It is the count that decides, so at most four chunks a read.
- `spread` first built all its pieces and then saw they were too many: Metop-SG
  crops 5.5 -> 6.2 ms. Now it counts first, and takes up to 256 uncompressed
  chunks (a copy each), which is what put those crops ahead of h5py.
- In `qk.py` h5py's `WRF` loop (0.15 ms) has its file already open, chunk cache
  warm. `one.py`, closing between reads, is the number to go by.

## Round 8 (2026-11-02): files that were not made for this benchmark

Every test so far had read our own files. So: 13 real files fetched (NOAA OISST,
NCEP, CPC, ERSST, National Water Model, NEX-GDDP CMIP6, xarray's samples), and all
488 of the HDF Group's and Unidata's own test files, each dataset compared with
h5py whole and by ten kinds of small read, each file in a process of its own.

**Correctness.** 59 values were wrong and 67 reads failed, all in the test files
and none in the real ones; most were one of a few causes. Now 26,693 reads of
3,478 datasets agree, but for one virtual dataset whose source files are not
there. Files of a million chunks and tests of what the corpus suggested then found
more than the corpus had. What was silently wrong:

| | gave |
|---|---|
| a read before the one stored chunk of a dataset | that chunk's bytes, not the fill value |
| pages of a fixed or extensible array never written | the file's own header, as values |
| fill values of big-endian datasets | their bytes reversed |
| fixed strings padded with spaces, or ended by a NUL | the padding, the bytes after it |
| a dataspace in shared object header messages | a wrong shape, and an array too big to make (now a clear error) |

What h5py itself does not survive: `bad_nbit_params.h5` and `bad_fletcher32.h5`
bring the process down with an access violation; fastnc reads the one and raises an
error on the other. Not done: VAX floats, SOHM, region references, and a filter
that is not there, on chunks it was skipped for.

**Real files.** Fresh open, ms, through `Dataset` / `File` (`real.py`):

| | Dataset | File | netCDF4 | h5py |
|---|---|---|---|---|
| OISST, a time series | 74 | 74 | 3387 | 3453 |
| OISST, 30 series | 199 | 177 | 103449 | 103302 |
| OISST, all | 133 | 104 | 4323 | 3662 |
| CMIP6, 30 series | 108 | 105 | 72534 | 71977 |
| CMIP6, 60 time steps | 63 | 45 | 494 | 432 |
| CMIP6, all | 93 | 73 | 3012 | 2608 |
| NCEP, 30 series | 12 | 12 | 3203 | 4822 |
| NCEP, one time step | 0.46 | 0.23 | 0.41 | 0.21 |
| NWM land, one time step | 10 | 5.9 | 280 | 145 |
| NWM channels, 3,000 rows | 111 | 6.4 | 390 | 20856 |

- **`_hold`.** 30 series cost 30 times one: a chunk a time step holds every place,
  and each series decoded them all. Two reads running that want under a quarter
  of the same many chunks now keep that region. OISST 2115 -> 195 ms, CMIP6 983 ->
  120, NCEP 138 -> 13. Paying once for a region costs a read at most what reading
  it takes, and only the second read of a program that makes more.
- **Read-ahead by machine, and handed over.** 365 steps of CMIP6 took 527 ms, a
  whole read 75. A fixed 32 MB allowed three chunks of 3.4 MB a pass on 24 threads:
  512 MB gave 200 ms. What was left was copying: each time step, out of a chunk
  decoded ahead that nothing else would ever read, into a new array, page faults
  included. Now such a chunk leaves the cache and is the result. 150 ms, and
  no change from 32 MB to 512 MB. OISST 915 -> 230.

**Many small chunks.** A million chunks of 8 by 8, ms:

| | first read, was | now | h5py | row | whole | h5py whole |
|---|---|---|---|---|---|---|
| v1 B-tree | 61 | 0.065 | 0.098 | 7.9 | 173 | 4470 |
| v2 B-tree | 1805 | 0.068 | 0.095 | 8.4 | 196 | 6390 |
| fixed array | ~2300 (a row) | 0.057 | 0.096 | 3.5 | 238 | 36797 |
| extensible array | failed | 0.107 | 0.103 | 3.0 | 128 | 43860 |

No file here had shown this: the most chunks of any is some 19,000. A first read made
the whole table, O(chunks) against the library's O(log chunks), 18,000 times
h5py's. `LazyIndex` goes down the tree. A table costs ~60 ns a chunk and a lookup
~6 us, so after 1/64 of the chunks in lookups it has paid for the table and makes
it (a rule that needs no tuning to the file: what it costs is at most twice
the better choice). Chunks that do not need the plan's machinery are decoded by
zlib on the spot: 130 -> 12 us for a chunk a read at random.

**Did not work:**
- That rule first gave the rows one by one even once the table was made: 840
  of them on each open, Metop-SG crops 4.2 -> 5.4 ms. All in one go now.
- Reversing every fill value when a dataset is swapped: HDF4's are native already,
  and its reference digests said so.
- `VIIRS fancy rows` (3.3-4.4 ms, h5py 4.0) shows as a loss in some runs. Commits
  from before and after overlap; it is level.

## Round 9 (2026-11-02): as it is used, and a check that takes a minute

**Through xarray,** against its netcdf4 engine (ms): OISST a series 78 / 3629, 30
series 196 / 126811; CMIP6 30 series 118 / 86733; a time mean 1138 / 6497, of
which numpy's nanmean alone is 854 and no file is touched. A month of daily files
by `open_mfdataset`: 217 / 449, of which 160 is opening them, and of the 3.1 ms an
open takes fastnc's part is 0.7: what is left there is xarray's, in any engine.
dask's threads do not fight the pool: `parallel=True` 217 ms, one thread 272.

**A fresh process,** from nothing to a first value, ms: `Dataset` 218, netCDF4 241;
`File` 193, h5py 265.

**Every file netCDF4 opens,** compared through `Dataset` for the first time: on
the 21 real files, no difference in structure, attributes or values. On the HDF
Group's, most is fastnc showing what netCDF-C hides. netCDF-C overflows its
stack on a group inside itself, and cannot read the attributes of two NOAA SDR
files, which fastnc and h5py read alike.

**`tools/check.py`.** A check took twenty minutes, nearly all of it the reference
libraries and processes run one after another:

| | before | now |
|---|---|---|
| the tests | 2 min | 40 s, in 8 processes |
| 62,282 reads against the references | 3.5 min | 24 s, on saved checksums |
| speed against the last commit | 3 min, by eye | 15 s, 20 rows, 3 rounds |
| all of that | 8 min | 57 s |
| and numba, and no kernels | 4 min | 4 min 20 s |

Saving the answers takes 65 s, once. Its first run found a wrong value in code
of the same day (see below). Tried on two faults put in on purpose: the bytes of
big-endian fill values reversed again failed 30 tests, and 1 us added to each
read showed as 1.13x on one row and 1.09x on another, which a single run, varying
by up to 2x, could never show. `answers` did not see the fill values: no file in
the corpus is sparse and big-endian, so it does not replace the tests.

**Did not work:**
- Page init bits read as a bitmap a data block. The room is reserved so, but the
  bits run on from block to block (`k * pages + page`). A million-chunk array read
  as 467,956 chunks and rows of zeros, and tests of smaller arrays passed. The dump
  gave it away: bytes of only 0 and 255 from blocks of two pages.
- A check that reported `tests ok, 0 passed`: pytest's summary line was gone, with
  two `-q`. No tests seen is now a failure.
- The first run of `check.py` took 139 s: a process for each of 1,094 reads of a
  file, mostly importing. Batches of files in 8 processes, 24 s.
- Reading GLM's 48 variables faster: 4.5 -> 4.0 ms by File, 10.2 -> 8.3 by Dataset,
  and no line left above 3%. It stops there.

## Round 10 (2026-11-02): limits, and what a read costs besides time

**More files.** 197 of other projects' (HDF4, PyTables, netCDF-C, netcdf4-python,
h5py): 744 files, 104,870 reads, 21 s. They found big-endian enums reversed through
`File`, a shuffle by a size not the type's, names ending at a NUL inside their
count, and HDF4 szip, which had never worked: its parameters were swapped, and
HDF5's decoder wants a length in front of the stream. A decode by hand in plain
Python gave all 80 values, which is how the swap was found; the same file's
gzip and skipping-Huffman variables are its reference, as netCDF-C cannot read it.
Of 56 differences left: corrupt files, a VDS without its sources, netCDF-C
quirks (`f03tst_open_mem.nc` gives uninitialized memory, `test.cdf` a Windows error
message as numbers, 2-byte strings in vlens split), variables and attributes
netCDF-C hides, a cache image.

BitGroom and Granular BitRound have nothing to undo: checked against the HDF
Group's printed output, 2,048 values to the digits printed, and "Maximum value
in DS1 is 1.84467e+19", which looks wrong and is what their example writes.

zfp (`_zfp`) was checked against the real decoder, which hdf5plugin's H5Z-ZFP
1.1.1 with zfp 1.0.1 is, once `import hdf5plugin` has loaded it: 14,280 datasets
(int32, int64, unsigned, float, double; ranks 1 to 6; fixed rate, precision,
accuracy, reversible, expert; smooth data, noise, subnormals, NaN, any bit
pattern) gave the same bytes, as did the 2,048 values of the HDF Group's file
(H5Z-ZFP 1.0.1, zfp 0.5.5). The first comparison failed by the thousand, and most
of that was the reference: read back on the handle that wrote it, a dataset comes
from HDF5's chunk cache, the input and not what a reader gets (at 4 bits a value
it was 2.35 away from the decoded data). Read a lossy filter from a reopened file.
MB/s decoded on this machine: a whole read on one thread against the plugin
through h5py, like for like, on all 24, and the kernel alone on the stored chunks
(a repeated read of a small dataset times the chunk cache, not the decoder):

| | h5py, 1 thread | fastnc, 1 | fastnc, 24 | kernel alone, 1 |
|---|---|---|---|---|
| 3D float, rate 8 | 311 | 283 | 2188 | 360 |
| 3D float, accuracy 0.01 | 206 | 186 | 1575 | 231 |
| 3D float, reversible | 159 | 174 | 1244 | 186 |
| 3D double, rate 16 | 392 | 398 | 2523 | 474 |
| 3D double, reversible | 179 | 190 | 1374 | 194 |
| 2D float, rate 8 | 249 | 224 | 1956 | 240 |
| 1D double, rate 16 | 265 | 197 | 1741 | 206 |

**CPU.** Many threads are not more work. Total CPU, ms, against h5py:

| | fastnc | h5py |
|---|---|---|
| CMIP6, a series | 703 | 2453 |
| CMIP6, 30 series | 3062 | 99297 |
| CMIP6, 365 time steps | 2141 | 2734 |
| OISST, all of it | 3219 | 3938 |
| VIIRS, 30 crops | 219 | 250 |

CPU in the second after a read: nothing to speak of. No thread spins.

**Memory.** Where h5py was ahead, by far: 1.7 to 2.2 GB in the working set after
closing, against 43 MB. Three causes, none tracked by anything:

- `close()` did not unmap. Readers, the parser's memoryview, slices of heaps in
  attribute caches and the lookups' closures all held the mapping, so `mm.close()`
  raised a BufferError that was passed over as "arrays still view it". The map
  lasted until collection, and on Windows the file could not be deleted. Now
  all of those let go, a read after close raises ValueError, and the working
  set after a loop over OISST is 140 MB, not 605.
- The output pool kept blocks for ever, up to 4 GB, to save 50 ms a GB. Blocks,
  regions and chunks over budget now go after 5 s unused, on one timer that
  runs only while something is kept.
- Read-ahead allowed an eighth of the pool. By cores instead, peak private MB, ms:

| chunks ahead a thread | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| CMIP6 | 372, 199 | 462, 179 | 622, 253 | 732, 258 |
| OISST | 425, 318 | 532, 288 | 728, 250 | 861, 246 |

  2 is the best of CMIP6 and within 15% of OISST's best: past a point a bigger
  buffer costs its own faults. Peak of a loop 649 -> 379 MB.

`import fastnc` is 2 MB and 12 ms. The 72 MB first blamed on it was `fastnc.warm()`,
which loads every kernel; a program loads those it uses. Per-thread scratch stays
(the largest chunk read, a thread), about 100 MB after a run of big chunks.
(Both wrong: see Round 11. The 72 MB was numba, loaded because no image had
been built for hours, and scratch now goes.)

**Did not work:**
- `from ._h5read import ...` in `h5py_dtype` and `decode_values`: the import
  machinery on every attribute, 10 us on an open, caught by `check.py` at 1.11x.
- Three rounds of timing flagged four rows of noise. Flagged rows now get six more.
- The check then flagged `cmip all` at 1.23x, and it was real and harmless: close
  went from 0.03 to 16 ms, collection from 28.7 to 12.7, the sum from 113.7 to
  112.4. Unmapping 243 MB costs what it costs; it now comes when asked for.

## Round 11 (2026-11-02): what nobody had looked at

**The image.** Every build for hours had failed its self test, silently, and each
process that fell back on numba began another. numba names a library after its
kernel, so a cache of object code by that name held one overload a kernel, and
the others were linked to the wrong code; which came last decided whether a build
worked. The check missed it by reusing the last good image, which is the check's
fault: it now builds first, fails if it cannot, and fails if any read loads numba.
That second test found at once that `run_nbit`, `run_shuffle` and `run_inflate` were
in no fresh install's image, and a race: the first kernels come on all threads, and
those arriving while one opened the image went to numba.

Every memory figure of Round 10 had numba in it. After import, and after a loop
and 5 s idle, working set, MB:

| | fastnc then | fastnc now | h5py |
|---|---|---|---|
| imported | 124 (with `warm()`) | 32 | 42 |
| OISST time steps, then idle | 224 | 36 | 44 |

**What "peak" hid.** `PeakPagefileUsage`, used for peaks here, does not count the
output pool: an anonymous `mmap` on Windows is a section, not private memory. So
30 series were not 116 MB but 1.3 GB, for the 5 s it stays. A region is what
makes them hundreds of times faster, but it need not be the whole variable: around
places near each other, a sixteenth of it, and inflating stops at its last
byte. 30 series, ms and MB held, processes alternating:

| | before | after |
|---|---|---|
| CMIP6, close by | 179, 1261 | 70, 79 |
| CMIP6, in a region | 108, 1261 | 70, 79 |
| CMIP6, all over | 107, 1261 | 106, 1261 |
| OISST, close by | 184, 1514 | 153, 95 |
| OISST, all over | 183, 1514 | 182, 1514 |

The first version guessed "near" too easily and made one set of places in two pay
a second pass, 1.4x. Taking everything past an eighth of the values (`_hold.MOST`), one
set in six does (1.35x), and they pay it once. Rows keep everything: a sweep along an
axis would pass again each time the region grew.

Read-ahead at 0, 1 and 2 chunks a thread: 594, 229, 214 ms for CMIP6's time steps,
1023, 1342, 1366 MB with their outputs kept. It stays.

**Fuzzing.** Files and reads made at random, each against h5py or netCDF4: two
rounds, 12.2 million reads in the second, 3.7 million of them threaded, with coverage
counted (`_hold` 1.0 million copies, chunk hand-off 330,000, `_quick` 4.8 million).
Twelve silent wrong results in all (CHANGELOG), none on those paths. No crash came
from fastnc: 71 process deaths, the ones replayed alone in the reference libraries'
code, and 1.8 million reads with fastnc alone in the process died of nothing. One,
seed 70308, never came back and stays unexplained.

Filters were the worst of it. `fusable` looked at ids and not their order, so
a checksum under a shuffle read as garbage. Every order of up to three of six filters:
1,290 datasets, 17 wrong and 151 refused before, all right now. Other orders go stage
by stage, netCDF-C's checksum under its shuffle among them (a chunk that went without
the deflate has no place in the one pass), so a fletcher32 variable written
by netCDF-C loses the mask pass's proof.

**netCDF-3 and HDF4** went from 18 of 355 kinds of read lost to netCDF4-python to
none (CHANGELOG). A masked small read, with what `_toma` works out kept per
variable: 30 -> 22 us; netCDF4 about 95. Big-endian chunks, swapped a word at a
time, ms against h5py: `f4` whole 9.0 -> 1.35 (3.2), `f8` window 3.3 -> 0.84 (1.4).

**The README's tables got slower, and nothing had.** VIIRS whole 47 -> 76 ms, GOES C02
63 -> 94. A/B over the commits showed no change, and one thread none either. The
benchmark alternates readers, netCDF4 takes seconds, and since Round 10 what fastnc
keeps goes after 5 s; a whole GOES read in a loop is 61 ms, 11 s apart 134. It is what
that trade costs, but why so much? First writes to a GB, ms:

| threads | touched already | numpy, `VirtualAlloc` | `mmap(-1)` |
|---|---|---|---|
| 1 | 48 | 158-169 | 531 |
| 8 | 30 | 36-40 | 95 |
| 24 | 31 | 36-43 | 101 |

A pool made to avoid page faults had been buying the dearest kind: an anonymous
mapping on Windows is a section of the page file, and on Linux shared memory. In the
process's own memory GOES 11 s apart is 95 ms. The rest is not found: VIIRS to the
GPU is 8.6 ms in a loop, 37 after idling, and was 32 before any of this.

**Did not work:**
- A test build with a target of `try.img`: `_publish` tidies by the name's prefix up
  to a dash, an empty one here, and took every file beside it.
- Fewer threads for small windows of plain chunks: time is level from 1 to 24. The
  0.2 ms lost to h5py on a first window is the index.

## Round 12 (2026-11-02): what a benchmark of arrays cannot see

Every table so far is of numbers read by slices. So: everything else a program does
with a file, against h5py and netCDF4, in three surveys. Ms, repeated reads:

| | fastnc before | after | h5py |
|---|---|---|---|
| 200,000 variable-length strings | 247 | 25 | 34 |
| 2,000 of them | 3.3 | 0.50 | 0.59 |
| 20,000 vlens of numbers | 48 | 4.8 | 13 |
| 2,000 records one at a time | 159 | 4.6 | 72 |
| 2,000 references | 1.3 | 0.41 | 0.32 |
| 3,000 names of a group | 4.0 | 2.8 | 2.2 |
| a list of 30 times, contiguous | 5.0 | 0.02 | 0.22 |
| a list of 30 times, plain chunks | 1.7 | 0.03 | 0.26 |
| every third time, plain chunks | 0.79 | 0.01 | 0.34 |
| a 256 x 256 window of 256 plain chunks | 0.42 | 0.12 | 0.89 |

| | fastnc before | after | netCDF4 |
|---|---|---|---|
| 2,000 records one at a time | 418 | 33 | 212 |
| 100,000 rows of chars, masked | 12.3 | 1.4 | 25 |
| 2,000 rows of chars, one at a time | 172 | 28 | 170 |

Each was Python for each value, or a plan for each read. Strings went to `str` for
Dataset and back to `bytes` for File, by way of latin-1 where they said ASCII, which
changed bytes past 127. `vlen_pack` packs what a selection refers to in one pass, a
collection walked once for each run of values in it, and the values are cut from one
buffer. `_gather` copies boxes of plain chunks, steps included, with nothing to plan.
Lists and masks read their box as small reads do.

Chars masked at 8 ns a value because numpy compares `S1` as strings: `uint8` does
not. Records through Dataset were a positional read, seven dtypes made anew and a
copy field by field; `dtype_view`, made a property that morning, had added to it.
A record numpy copies by fields leaves the bytes between them as they were, and the
remote tests noticed: records are copied as bytes.

**After idling.** Keeping memory is no longer what a read after a pause pays for:
GOES whole, 7 s after the last read, is the 61 ms of a loop. VIIRS, one variable, is
5 ms in a loop, 14 a second after, with or without what is kept, and with fewer
threads the gap goes (4 threads: 16 and 17). Parked cores: nothing here to fix.

**xarray.** WRF d01 is 21 ms open, 34 load and 6 close, with `decode_cf` and its
`Variable`s flat across the profile. Nothing stands out to take.

**Did not work:**
- v1 B-trees' inner nodes decoded once, for `bisect`: block reads 0.33 -> 0.30 ms, and
  random pixels on a million chunks 1.85x slower: each new node costs 200 lookups.
  Caught by the check, taken out.
- Sparse lists a place at a time: rows 0.11 -> 3.1 ms, each through the general path.
  It stays for boxes over 16 MB. Reading the box the quick way was the answer.
- Comparing lists with numpy: `x[0, a:b, [..]]` moves an axis there, not in h5py.

Left: `x[3, j]` across five plain chunks, 6.8 us against h5py's 3.9; references and
name lists by a quarter; a whole step of 20 chunks read repeatedly, which netCDF4's
bigger cache holds (0.4 ms against 0.2).

**Programs, not reads.** Read-ahead followed a read only to the chunk next to the last.
`range(0, n, 2)` over a variable with a chunk a step got none, a chunk at a time on
one thread, and that is every 6-hourly or monthly loop. A jump repeated along one
dimension is now followed like a step (`_went`). CMIP6, 360 steps' worth, ms:

| every | before | after | chunks decoded ahead / read |
|---|---|---|---|
| 1 | 175 | 175 | 351 / 360 |
| 2 | 370 | 109 | 172 / 180 |
| 3 | 248 | 82 | 112 / 120 |
| 4 | 183 | 62 | 83 / 90 |
| 6 | 124 | 46 | 53 / 60 |
| 10 | 73 | 35 | 30 / 36 |

CPU in all rises with the wall time's fall, as for every step: 344 -> 641 ms at 2.
Nothing is decoded that is not read but the last few. Every 4th step of a year, ms:
File 65, Dataset 99, xarray 105; netCDF4 1155, xarray on it 1201.

An xarray dataset could not be pickled, so dask's distributed scheduler and
multiprocessing failed on it: the store held a file object. `Dataset`, `File` and what
is in them pickle by what they name and open again where unpickled, which netCDF4's
and h5py's do not. With threads, dask's mean over time: 342 ms, netcdf4 3802, h5netcdf
3956. Threads of a program reading steps by turns are still not worth it: 124 ms on
one, 180 on two, 246 on four sharing a file, and 170, 139 with a file each. Probably
two reads at once decoding the same chunks ahead, as nothing says one is under way;
not looked into. h5py: 1830 to 2630.

Over HTTP at 30 ms a request, CMIP6 of 244 MB: a pixel 9 requests and 2 MB, 20 steps
24 requests, 30 series 37 and the whole file's bytes. Nothing to take.

Masking a CMIP6 step costs 0.3 ms, and looked like 5x numpy's `==`. It is not that
test: this variable is 69% masked, land only, and flagging it takes 376 us on a
thread against numpy's 609. The kernel's test is 101 us for 4 MB.
- `Datatype.h5py_dtype` and `retyped` calling each other: a stack overflow, caught by
  the first compound read.
- Reduced integers and float16 widened in `Variable.read` alone: `_quick` and the
  fused path go straight to the file.

## Round 13 (2026-11-02): threads sharing a file

Round 12 left threads reading steps by turns from one open file slower than one thread,
with a guess as to why. Counted: 240 steps of CMIP6, each a chunk.

| threads | ms before | after | chunks decoded before | after | decoded twice before | after |
|---|---|---|---|---|---|---|
| 1 | 134 | 132 | 262 | 262 | 0 | 0 |
| 2 | 179 | 148 | 360 | 295 | 39 | 17 |
| 4 | 138 | 179 | 353 | 267 | 113 | 1 |
| 8 | 240 | 111 | 610 | 276 | 346 | 0 |

Those times are one pass each, and move by 40 ms from run to run: the counts are the
result, and they move too (276 to 318 decoded at 8 threads, 0 to 6 twice). With each case
alone in a process: shared by 1, 2, 4, 8, 16 threads 120, 127, 107, 121, 128 ms, against
136, 158, 122, 114, 129 with a file each. Sharing no longer costs anything. Nor does it
pay: a read uses every core as it is, so more threads share them. h5py: 1800 to 2630.

Three things had to go together, and each alone looked like the answer:

- **Claims.** A read says which chunks it is decoding (`index.coming`); others wait for
  them and leave them out of what they decode ahead.
- **A way each.** `beside`, `last` and `ahead` were the dataset's, so each thread's read
  looked like a jump from the other's and read-ahead started over. They are per thread
  (`_Way`), and a read may carry on from any thread's.
- **What was read.** Claims made repeats worse at 4 threads, 353 -> 443 decoded, which
  showed where they came from: a chunk handed to a read leaves the cache, and to a
  thread behind the others looks new. `index.gone` remembers the last 16,000.

One thread does none of this, and the check's rows show no cost: Ku rows 0.97x, Metop
crops 0.99x. A claim made before a fork is ignored by the child, whose copy of it
nobody would ever set.

**Rows across chunks.** `spread` worked out the same pieces for every row of a loop.
`pieces` keeps them. `x[i]` across five compressed chunks 5.4 -> 3.9 us, h5py 3.7.
Crops that never repeat go on costing what they did.

**The way into a kernel.** Across five plain chunks the same row took 6.1 us by the image
and 4.4 by numba, whose call is the cheaper by some 1.7 us here: the image's exists to
cost next to nothing, and was 2.36 us for `_gather`'s eight arguments. Of that 1.0 made
the words, testing each argument on each call, 0.44 was the call, 0.18 the types.
Which arguments are arrays is known from the types: 2.17 us, and the row 5.9. Under the
GATHER of 4 nothing is to be had: the loop costs 0.75 us a chunk, and the two meet at
about 5 chunks for rows. Columns are another matter: 5.9 by the loop against 8.6 by the
kernel over 4 chunks, 9.4 and 11.1 over 8.

**Measuring.** The check called Metop crops 1.11x slower now and then, and I took it for
real twice: one comparison of 14 rounds by turns said 1.08x, two of 16 said 0.99x. This
row moves by more than the check's margin. And `FASTNC_IMAGE_FILE` named the small `.img`
file that says which image is current, which is not an image: that process went by numba
without a word, and its 4.3 us looked like an image twice as good as itself. Print what a
process is running on before comparing two of them.

Left: a plain row across five chunks, 5.9 us against h5py's 3.7; columns by the kernel, as
above; references and name lists by a quarter; a step of 20 chunks read repeatedly (0.4 ms
against netCDF4's 0.2).

**The image, ported.** fastgrib took the kernel image (its first decode in a fresh
process 760 ms -> 4 ms), and porting it found what fastnc's had not met yet: a memmap
went to a unit as `int()` of itself (an address, for a one-element one), a byte-swapped
array had a native one's key, a failed allocation was written to, errors were all
`RuntimeError`, growing buffers were missing, every numba helper was linked to address 0,
the index had no CRC, and a forked child moved its parent's file position. fastgrib's
answers are here now. Still 340 units, none left out: no kernel here needed a helper.
Not measured: the check's speed step runs the last commit's code on the newest image,
which a new image format turns away, so that step compared nothing this time.

## Round 14 (2026-11-10): a corpus of satellite and model files

33 kinds of satellite and model file: 683 files, 315 of them netCDF / HDF5 / HDF4,
16 GB. Every variable read whole, masked, warm OS cache, fastnc's second read against
netCDF4-python's first (`fastnc.File` against h5py for plain HDF5):

| | netCDF4 / h5py | fastnc | bytes |
|---|---|---|---|
| `Dataset`, every variable | 42.7 s | 7.1 s | 14.9 GB |
| `File`, every dataset | 1.77 s | 0.18 s | 1.3 GB |

By kind, from 1.3x (GCOM AMSR2's small contiguous datasets) to 33x (GOES full disks);
GCOM's netCDF total came out 0.7x, but timed in a loop each of its datasets is faster
(`Land_Ocean Flag 89`, 3.8 MB: 0.91 ms against 1.59): one pass of a 2 ms read lands on
parked cores. 22 variables were slower, all but two under 4 ms and equal in a loop.

The two that were not: MODIS L1B night granules, whose reflective bands hold no data, so
both readers return the fill value. fastnc wrote it a byte column at a time (`_fill_with`,
`_chunks._fill`): 38 MB of 2-byte fill 48 ms, against 1.4 by numpy's `fill` on a word
view. The same code fills HDF5 chunks never written:

| 40 MB, nothing stored | h5py / netCDF4 | fastnc before | after |
|---|---|---|---|
| HDF5 int16, `File` | 5.9 ms | 49.8 | 1.7 |
| HDF5 float32 NaN fill | 5.9 | 26.1 | 1.7 |
| HDF5 >u2, `Dataset` masked | 6.0 (h5py raw) | 35.9 | 4.9 |
| MODIS `EV_1KM_RefSB` raw | 9.6 | 54 | 1.5 |
| MODIS `EV_1KM_RefSB` masked | 46 | 55 | 6.8 |

Values: `tools/check.py` over the same corpus, netCDF4-python's and h5py's answers saved
once: 226,166 reads, and the only differences are the masked reads of the 589 variables
whose valid range is in unpacked units (GridSat-B1, HURSAT-AVHRR, HRD's TDR winds), on
purpose (`_maskscale._unpacked_units`). None in raw reads, none elsewhere.

## Round 15 (2026-11-10): kernels that come with the install

Until now each machine built its own kernel image (`_image`): a fresh install compiled
every kernel with numba, about 6 minutes in the background, reading without kernels
meanwhile, and so did every numba or numpy upgrade and every edit of a kernel. The
question: can the images come with the wheel instead, fast on any machine, under
~100 MB? Python 3.12, numpy 2.5.4, numba 0.68.0 unless said; WSL Ubuntu 22.04 (Python
3.10, numpy 1.26.4, numba 0.63.1) for Linux.

**What an image holds.** This machine's image of the time, 51.1 MB: 158 kernels, 329
units (one per kernel and argument types), none identical to another, 0.3 MB of it
relocations. A unit carries every kernel it calls, so a codec's driver and each of its
decoders each hold the whole decoder (`_blosc2.decode` 393 kB, `_fused._work_ms` 324 kB
each, eight of them). Most of the bytes are the same code over and over, and pages of
zeros between a unit's code and its data:

| stored as | size | |
|---|---|---|
| raw (format 3) | 51.0 MB | |
| zlib-9, each unit alone | 5.4 MB | 9.4x; all 329 decompressed in 59 ms (863 MB/s) |
| xz-6, each unit alone | 4.3 MB | 11.8x, decompression several times slower |
| xz-9e, all units in one stream | 0.7 MB | 71x: the units are mostly one another |

Units are now zlib-compressed one by one (format 4; format 3 is still read), so they are
still mapped one at a time when first called. Mapping one, warm file: 0.11 ms median raw,
0.22 compressed (all 329: 42 and 90 ms; a read maps a handful). Opening an image: 1.3 ms.

**CPU levels.** `tools/make_images.py` compiles each x86-64 level of the psABI in a
process where numba compiles for it alone (`NUMBA_CPU_NAME=x86-64-v3`,
`NUMBA_CPU_FEATURES` its features, so that numba adds none of the host's). At run time
`levels()` takes the levels from CPUID and XGETBV. Images, all 329 units linked:

| | Windows | Linux (numba 0.63) |
|---|---|---|
| x86-64-v3 (AVX2, FMA, BMI2) | 4.92 MB | 4.70 MB |
| x86-64-v2 (SSE4.2, POPCNT) | 4.34 | 4.13 |
| x86-64 (SSE2) | 4.33 | 4.12 |
| this machine's own (raptorlake) | 5.59 | |

Each level against this machine's own image, `tools/profile/ab.py --images`, processes
by turns, 6 rounds, median of each process's best of 7 (ms; the rows read the real
files whole or a window):

| all threads | host | v3 | v2 | x86-64 |
|---|---|---|---|---|
| d02 load masked | 61.8 / 59.8 / 58.9 | 59.9 (0.97x) | 61.8 (1.03x) | 60.2 (1.02x) |
| wrf load raw | 29.1 / 29.0 / 28.7 | 29.0 (1.00x) | 29.0 (1.00x) | 29.3 (1.02x) |
| wrf load masked | 36.1 / 35.6 / 35.2 | 35.6 (0.99x) | 35.8 (1.00x) | 35.6 (1.01x) |
| m250 var raw | 52.2 / 51.7 / 51.5 | 51.2 (0.98x) | 51.7 (1.00x) | 51.9 (1.01x) |
| c13 var raw | 7.37 / 7.49 / 7.44 | 7.41 (1.01x) | 7.37 (0.98x) | 7.35 (0.99x) |
| metop File window | 0.80 / 0.78 / 0.78 | 0.78 (0.98x) | 0.75 (0.96x) | 0.74 (0.95x) |

(host: its own figure in each of the three comparisons.) The other twelve rows were
within the same 0.95-1.03 either way; AVHRR's 1.6 ms raw read swung 0.83-1.08 in all
three. On every core the level makes no difference: these reads wait on memory and on
Huffman decoding, which has nothing to vectorize. On one thread (`FASTNC_THREADS=1`,
same method), where the kernels themselves count:

| one thread | host | v3 | v2 | x86-64 |
|---|---|---|---|---|
| d02 load masked (netCDF3: mask/scale only) | 172 | 171 (1.00x) | 195 (1.14x) | 202 (1.18x) |
| wrf load raw (inflate) | 163 | 162 (0.99x) | 166 (1.03x) | 168 (1.03x) |
| wrf load masked | 171 | 170 (0.99x) | 176 (1.03x) | 178 (1.04x) |
| avhrr var masked | 8.20 | 8.20 (1.00x) | 8.41 (1.02x) | 8.59 (1.04x) |
| m250 var raw (33 MB chunks, inflate) | 352 | 352 (1.00x) | | 354 (0.99x) |
| c13 var raw | 56.4 | 56.8 (1.01x) | | 57.5 (1.01x) |
| ku load raw (15k tiny chunks) | 18.8 | 18.6 (0.99x) | | 18.7 (1.01x) |

x86-64-v3 is as fast as an image made for this very CPU, so a machine's own image
earns nothing over the shipped one: `tune()` no longer builds one where a shipped image
did the work. v2 buys 2-4% over plain x86-64 (4% where mask/scale vectorizes), not
worth 4 MB per image. v4 (AVX-512) was not measured: this CPU has none; given v3 =
host, little is to be expected, and an image this machine cannot run it cannot try
either. Shipped: **x86-64-v3 and x86-64**.

**Fresh processes**, import + open + read, kinds by turns, median of 5 (ms):

| | shipped v3 | shipped x86-64 | own image | old 51 MB | numba loads | no kernels |
|---|---|---|---|---|---|---|
| GOES-19 C13 `Rad` | 198 | 200 | 198 | 192 | 1046 | 204 |
| WRF d01 all | 233 | 232 | 231 | 245 | 1030 | 274 |
| WRF d01 `T` | 190 | 175 | 191 | 184 | 971 | 171 |
| AVHRR `avhrr_ch4` | 171 | 178 | 185 | 186 | 970 | 170 |

Compression costs nothing to be seen here. What a shipped image changes is the first six
minutes of an install and every rebuild: a session that keeps reading runs at full speed
from its first read, never pays numba's 0.8 s, and never compiles.

**What an image depends on.** Built twice from the same sources, units compared byte
for byte:

| A | B | units alike |
|---|---|---|
| numpy 2.5.4 | numpy 1.26.4 (same Python, numba) | 294 of 325; B lacks 4 |
| Python 3.12, numpy 2.5.4 | Python 3.11, numpy 2.4.6 (same numba) | 170 of 325; B lacks the same 4 |
| Python 3.12, numba 0.68.0 | Python 3.11, numba 0.68.1 | 174 of 329 |

The four are `_xarray._gather` with unsigned index arrays, which numba 0.68 does not
compile under numpy 1.26 or 2.4: under numpy 2.5 they are made, elsewhere such a read
goes to numba. Bytes differ with the Python (its bytecode is what numba compiles), yet
the code does the same: the suite under Python 3.11 and numpy 2.4.6 on the image built
under 3.12 and 2.5.4 passes whole (below). So an image is keyed by system and numpy's
major version only, and one image set serves every Python: one wheel per system.

**The suite on shipped images, numba unable to load** (`pytest --kernels shipped`:
`sys.modules['numba'] = None`, only `fastnc/_kernels`):

| where | image | result |
|---|---|---|
| Windows, Python 3.12, numpy 2.5.4 | v3 np2 | 9,783 passed; 2 failed, both tests of numba itself (they now skip without it) |
| Windows, Python 3.12, numpy 2.5.4 | x86-64 np2 | 9,784 passed |
| Windows, Python 3.11, numpy 2.4.6 | the same v3 np2, built under 3.12 | 9,730 passed |
| Windows, Python 3.12, numpy 1.26.4, no h5py/netCDF4/xarray | v3 np1 | 8,635 passed |
| WSL Linux, Python 3.10, numpy 1.26.4 | v3 np1 linux | 5,513 passed; 1 failed with any kernels (h5py 3.14 raises TypeError where newer h5py raises ValueError: not the image) |
| WSL Linux, the same | x86-64 np1 linux | the same |

With numba there, `--kernels numba`, `off` and the default (which now takes the shipped
image) each passed 9,780-9,786; the one test that failed in all three timed numba's load
in a process that found the shipped image instead, and now turns images off.

And as installed: `pip wheel .` with the images in `fastnc/_kernels`, retagged
`py3-none-win_amd64` (`wheel tags`), into a new venv holding numpy and no numba: the
process takes `site-packages/fastnc/_kernels/kernels-x86-64-v3-win-np2-*.img`, numba
neither importable nor imported, WRF `T` bit-identical to netCDF4-python's, 156 ms
import + open + read (671 ms the first time, Python compiling fastnc's own `.pyc`).

**Size of a wheel**, one per system, every Python: v3 and x86-64 for numpy 1 and 2, four
images, ~18.5 MB installed and downloaded (zlib data does not shrink in the zip). The
wheel made above, three images: 13.4 MB. Adding v2 and v4 would make ~37 MB. Units alike
across numpy's versions stored once would take the pair of v3 images from 9.8 to 6.1 MB
(not done: a file of units shared between images is a format change, for later).

**Decided: numba stays, nothing ships yet.** What shipping buys was narrower than
hoped: nothing once a machine has its image, little for a one-shot script (reading
without kernels is as fast for one variable), and the first ~6 minutes after an install
or upgrade, plus a numba-free install. What it costs is every release: four images on
two systems rebuilt and tested by hand (no CI minutes), and the image becomes the only
engine where numba is absent. So numba stays a requirement and fastnc's wheel stays pure;
each machine builds its image as before. Kept from this round, as they help every
install: units compressed (an image on disk 51 -> 5 MB), reads that keep to numpy and
zlib where numba cannot be imported, and the machinery for shipped images, which does
nothing while `fastnc/_kernels` is empty. When fastnc goes to people for whom a
numba-free, instant install matters, the steps below are what is left.

How it would ship then: platform wheels `py3-none-win_amd64` and
`py3-none-manylinux2014_x86_64` holding x86-64-v3 and x86-64 images for numpy 1 and 2,
built locally (Windows here, Linux in WSL or a manylinux container) with one pinned
numba; a pure `py3-none-any` wheel and the sdist for the rest. A Linux image needs only
`malloc`, `free`, `realloc`, `memset`, `memcpy`, `memmove`, `pow`, `powf`, found by name
at load time, so the manylinux2014 tag (and musllinux) holds. Fetching images on first
use was considered and not taken: it needs the network at first read, a place to trust
downloads from, and saves nothing that matters at 18 MB.

**Open risks, if images ship.**
- A unit missing for some argument types: with numba installed it compiles as before
  (and a machine image is built in the background); without numba the read raises
  ImportError naming the cause. The signatures come from the suite and
  `tools/make_signatures.py`; reads the suite never makes could still meet one.
- numpy 1 images lack four `_xarray._gather` units (above).
- x86-64-v4 not built or measured; an image is never tried on a CPU that cannot run it,
  so a level this machine cannot run would ship untested.
- A Python whose objects are laid out otherwise (free-threaded builds) fails
  `_direct_works`, and calls take the slower ctypes path (still correct).
- Images built with a newer numba than a user's numba matters not (the image does not
  load numba); but every kernel edit needs all images rebuilt before a release (the
  name carries `kernel_stamp`, so stale ones are simply not taken).

**Steps to ship images, when that time comes** (none done; `pip install` gives what it
gave before):
1. numba in the metadata. Required everywhere (simplest, as now; ~45 MB of numba and
   llvmlite downloaded and never imported on Windows / Linux x86-64), or by marker,
   `numba>=0.57; platform_machine not in "x86_64 AMD64" or sys_platform == "darwin"`,
   with an extra `fastnc[numba]` for unusual argument types and local images. With the
   marker an sdist install on Linux x86-64 has neither images nor numba: right values,
   reads without kernels.
2. The wheels: `[tool.setuptools.package-data] fastnc = ["_kernels/*.img"]` in
   pyproject.toml (taken out again: the wheel is pure meanwhile), then
   `pip wheel .` after `tools/make_images.py` in a numpy-2 and a numpy-1
   environment with the same numba, then `python -m wheel tags --python-tag py3
   --abi-tag none --platform-tag win_amd64` (Linux: `manylinux2014_x86_64`, built in WSL
   or a manylinux container; `auditwheel` has nothing to repair, the images link no
   library). Images stay out of git (`.gitignore`), so the release steps in
   CONTRIBUTING.md need them.
3. Before each release, `pytest --kernels shipped` on every image of the wheel
   (`FASTNC_IMAGE_FILE=<image>` picks one), Windows and Linux: CI cannot run here.
4. Whether to store units shared across numpy's versions once (9.8 -> 6.1 MB a pair),
   and whether v2 / v4 ever earn a place (v4 needs an AVX-512 machine to measure).

**macOS and ARM.** Not attempted. macOS x86-64: numba emits Mach-O there, so `_link`
needs a Mach-O reader (its relocations: `X86_64_RELOC_UNSIGNED`, `BRANCH`, `SIGNED`)
or numba told an ELF triple, and `_alloc` the `MAP_JIT` way under a hardened runtime.
ARM64 (Linux aarch64, Apple Silicon): the linker knows only `R_X86_64_64` / `PC64`;
ARM64's large code model brings `R_AARCH64_ABS64` and `MOVW_UABS_G0..G3`, calls
`CALL26` (with veneers past 128 MB), the loader must flush the instruction cache, the
CPU probe is `getauxval(AT_HWCAP)` / `sysctl` instead of CPUID, and the entry function
and the `_CPUID` bytes are x86 code. A few days, and a machine of each kind to test on.
Until then those platforms keep numba's own kernels, as today (they have no image of
any kind).

## Round 16 (2026-11-11): windows, as an imagery app reads

Round 14 timed whole variables. An app showing images mostly reads windows, rows and
subsamples, so the same corpus again, one file per case: 68 files, every variable of two
dimensions or more, a 512 x 512 window in the middle, a row, every fourth point both ways
and the first slice along the first axis; warm cache, best of three after a warm-up read
by each library (so a library that keeps what it decoded shows it).

| read | variables | netCDF4-python | fastnc |
|---|---|---|---|
| window | 1612 | 0.61 s | 0.35 s |
| row | 1612 | 0.29 s | 0.15 s |
| first slice | 1612 | 2.34 s | 0.44 s |
| every fourth point | 1312 | 2777 s | 0.62 s |

(netCDF4-python reads a strided selection a point at a time: a quarter of a 1015 x 1367
WRF field 0.95 s, fastnc 1.5 ms. It was timed once, not three times, past 0.5 s, and not
at all past 500,000 values, which left 300 variables out of that row.)

47 reads were slower in fastnc, from three gaps, all in what is kept between reads:

- **One big chunk** (GridSat: each image one 20-40 MB chunk). The cache of decoded chunks
  takes none past half its 32 MB, and `_hold`, which keeps a region of what is read again,
  left any read of 16 chunks or fewer to that cache. Each window, row or point decoded
  the whole chunk: 36, 12 and 12 ms; netCDF4-python, whose chunk cache holds it, 1.2, 0.2
  and 0.16. `_hold` now takes reads the cache would refuse.
- **A few chunks that fit the cache one by one but not together** (a window across four
  5 MB chunks: RADARSAT-2, Landsat, Sentinel-2): 2-3 ms a read against 0.4-0.6. The same
  test, now as `_cached` makes it: by the chunks together. 0.14-0.23 ms.
- **With the kernels loaded, scaled windows of many small chunks** (GOES ABI's rows of
  12 x 10848): masked and scaled reads go through `_fused`, which decodes as it masks and
  never looked at a kept region: 1.9 ms a read with the kernels, 0.3 without them,
  netCDF4-python 0.8. `_fused` now leaves reads `_hold` keeps regions for (`_hold.holds`):
  0.21-0.25 ms. One-off windows of 1/16 to 1/3 of a full disk, which `_fused` still
  served before, time the same either way (two alternating runs, within their noise).

The third gap first looked like a regression from the second fix: the A/B ran the last
commit's code without kernels (its copied files had a stamp no image was built for) and
the new code on the image. With `FASTNC_IMAGE_FILE` set for both, as round 13 says to,
it was there in both, and the fix for it showed.

Left: TRMM VIRS windows across 17 small chunks covering most of each, 0.6 ms against
netCDF4-python's 0.2 (past `CACHE_CHUNKS`, and too much of their chunks for `_hold`).

Also this round, by kind of read:

| | netCDF4-python | fastnc |
|---|---|---|
| open, every attribute, shape and dtype, close (68 files) | 0.51 s | 0.24 s |
| a pixel at random, GOES full disk (masked, scaled) | 471 us | 294 us |
| a row, WRF T2 | 88 us | 21 us |
| GOES full disk as NaN-filled float32 (`set_auto_nan`) | | 32 ms |
| the same as `np.ma.filled(v[:], np.nan)` | | 53 ms |
| the masked array itself | 157 ms | 25 ms |

