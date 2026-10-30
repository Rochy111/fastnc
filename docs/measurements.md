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

**Worked:**
- Threads by the decoding to do, not the output: one column of VIIRS I05
  (13 kB from ~200 chunks, on one thread) 60.8 -> 3.7 ms.
- netCDF-3 runs copied and swapped in one pass out of the mapping, not read
  then swapped: WRF d02 open + load 56 -> 46 ms, again on the same open file
  55 -> 31 ms. (The file's pages mapped once, the pass alone is 23 ms.)
- libdeflate without its Adler-32 check, which fastnc's inflate never did:
  4-7% (VIIRS 539 -> 514 ms, MODIS 383 -> 355, GOES-19 C02 822 -> 800).
- Short periods in fastnc's inflate widened to 8 bytes, not copied byte by
  byte: 1-3% (the fit made this look bigger).

**Did not work:**
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
