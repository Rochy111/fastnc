# How fastnc works

A map of the code for anyone changing it. The README says what fastnc
does and how fast; `docs/measurements.md` says why it is built this way;
`docs/contracts.md` holds the interfaces between the parts.

## Modules

| Module | What it holds |
|---|---|
| `_dataset` | The netCDF4-python front end: `Dataset`, `Group`, `Variable`, `Dimension`, `CompoundType` / `EnumType` / `VLType`; `Group.load()` (many variables in one pass) |
| `_h5py` | The h5py front end: `File`, `Group`, `Dataset`, `AttributeManager`, `Datatype`, `Reference`, `Empty` |
| `_xarray` | `engine='fastnc'`: a data store over `_dataset`, xarray's CF decoding on top; `preload=True`, URLs, file objects |
| `_nc4` | netCDF-4 as netCDF-c reads it out of HDF5: dimension scales, `_Netcdf4Dimid`, `DIMENSION_LIST`, phony dimensions, hidden attributes, NC_CHAR / NC_STRING, compounds with `align=True`, user types, `no_fill` |
| `_nc3` | netCDF-3 classic, 64-bit offset, CDF-5: header; big runs copied and swapped in one pass out of the mapping once the kernels are loaded, else positional reads straight into the output |
| `_hw`, `_tune` | The machine, looked at once a process: CPUs allowed, cores, memory, cache, and the budgets and block sizes that follow; `fastnc.tune()`'s measurements, kept per machine |
| `_hdf4`, `_hdf4_codecs` | HDF4 SD files as netCDF-c's HDF4 dispatch presents them; RLE, N-bit and skipping-Huffman decoders |
| `_hdf5` | HDF5 from its bytes: superblocks, object headers, groups (symbol tables, compact and dense links), attributes, datatypes, fill values, filter pipelines, every chunk index |
| `_h5read` | Raw values of one HDF5 dataset for an index (chunked, contiguous as virtual chunks, compact, vlen, reduced-precision types) and `read_many` |
| `_vds` | HDF5 virtual datasets (mapping lists v0/v1, every selection type, same-file and cross-file sources, unlimited mappings, printf-style names) and external-file storage |
| `_chunks` | The chunk engine: planning a selection over a chunk table (plans cached per box), the batch kernel (with per-band progress counts for the GPU pipeline), big chunks on many threads, the path used until numba is loaded (`_read_python`, `_gather_grid`) and when to load it (`ready()`, `LOAD_AFTER`) |
| `_zlib` | Inflate without numba: libdeflate or zlib found as a shared library, called through ctypes straight into numpy arrays (else `zlib.decompress`) |
| `_fused` | Masked / scaled reads with mask/scale done per chunk while it is in cache; `read_lut`: raw 8 / 16-bit values looked up in a table as each chunk is decoded (xarray's decoding) |
| `_readahead` | Whole variables read in the file's order decoded several at a time (a bounded window) |
| `_staged`, `_codecs` | Filter pipelines beyond deflate / shuffle / fletcher32: stage by stage over all chunks, one small batch kernel per codec |
| `_select` | numpy / netCDF4 indexes reduced to boxes, then post-indexing |
| `_inflate` | Deflate / zlib in numba; whole streams through libdeflate, where installed, called from the kernel by function pointer (`_intrin.call_inflate`) with this inflate as the fallback |
| `_pinflate` | One deflate stream inflated on many threads (speculative, verified) |
| `_lzf`, `_lz4`, `_zstd`, `_blosc`, `_blosc2`, `_bitshuffle`, `_szip`, `_scaleoffset`, `_nbit`, `_fletcher32`, `_bzip2`, `_shuffle` | The other HDF5 filters, each `decode(src, p, pend, dst, o, oend, cd, ws)` |
| `_maskscale` | netCDF4-python's masking and scaling: its own rules carried over line for line, and a fused numba kernel for the common cases; `_maskscale_gpu` the same on the card |
| `_gpu` | Reads straight to cupy arrays: one decode pass, each band uploaded as its last chunk lands |
| `_arena` | Big output arrays carved from pooled, already-mapped memory |
| `_pio` | Big unfiltered batches read with positional reads instead of through a fresh mapping |
| `_remote` | HTTP(S) / S3 / file-object sources: a sparse buffer filled by parallel byte-range requests |
| `_pool` | fastnc's threads (each parked on a lock of its own: a hand-off costs ~3 us for any number), `share()` and `set_threads()` |
| `_jit`, `_intrin` | numba imported on first use, the stale-cache stamp; unaligned word loads/stores and an atomic counter |

## Opening a file

`Dataset(path)` reads the first bytes: `CDF\x01/\x02/\x05` is netCDF-3
(`_nc3`, which reads with its own positional reads), the HDF5 signature at
0, 512, 1024, ... is HDF5 (`_nc4` over `_hdf5`, on a read-only mapping),
`\x0e\x03\x13\x01` is HDF4. Nothing else is read in. A URL or a file
object becomes a `_remote` source instead of a mapping (below).

`_hdf5.H5File` parses metadata in Python from a `memoryview` with
`struct.unpack_from`: object headers on demand (cached by address), a
group's links when first listed, attributes indexed by name first and
decoded one at a time, a dataset's layout / filters / fill value when it
is first read. `_nc4.NC4File` then builds netCDF's view in netCDF-c's
order -- that order decides phony dimension names and dimension ids:

1. per group, links in creation order (when tracked) or name order:
   dimension scales become dimensions as met (ids from `_Netcdf4Dimid`,
   else the next free id), other datasets become variables, committed
   datatypes become user types, subgroups are read after the group's links;
2. a second pass over the whole tree, children before parents, gives each
   variable its dimensions: `_Netcdf4Coordinates`, else `DIMENSION_LIST`
   (object references to scales anywhere in the file), else phony
   dimensions (`phony_dim_<id>`) -- an existing dimension of the group with
   the same length and unlimitedness that the variable does not use yet,
   or a new one;
3. an unlimited dimension is as long as the longest variable along it,
   worked out once (h5netcdf recomputes it on every question: xarray took
   10 s to open a WRF file through it).

User types are named by matching the variable's datatype encoding with the
group's committed datatypes (netCDF-c compares types; the variables carry
copies, not references). `no_fill` -- what netCDF-c's `nc_inq_var_fill`
says, which decides whether byte variables are masked at their default
fill -- is true unless HDF5 stores a fill value (checked on 727 variables).

## Reading a chunked dataset

`Variable[key]` → `_select.normalize` reduces the key to a box (start,
stop, step > 0 per dimension) and what numpy does afterwards (drop integer
dimensions, reverse negative steps, pick integer/boolean sequences).
`_chunks.read_batch` takes one or many (dataset, box, output) items:

1. **Plan** (numpy, no Python per chunk, cached per box on the chunk
   index): the dataset's chunk table -- parsed once into arrays of
   offsets, addresses, sizes, filter masks -- intersected with the box
   dimension by dimension, giving for each chunk its first selected
   element, counts, place in the output, and the last decoded byte the
   selection needs. Chunks missing from the table (never written) read as
   the fill value: the output is pre-filled only when the selection
   touches one.
2. **Kernel** (`_work`, numba, nogil): the caller and the pool's threads
   each run it, taking chunks off an atomic counter, biggest first. Per
   chunk: the filter pipeline in reverse (fletcher32 strips its checksum
   -- off the stored bytes, or off the decompressed ones when it was
   applied first, in which case a later shuffle's byte planes include it --,
   deflate inflates into the thread's scratch, stopping at the last byte
   the selection needs unless a shuffle spreads every byte plane over the
   chunk), then `_place` copies the selected elements into the output,
   undoing the byte shuffle and swapping big-endian to native in the same
   pass while the chunk is still in cache. A chunk that needs no
   transform and is one run of the output is inflated straight there.
3. **Big chunks**: a chunk of 4 MB+ compressed and more than a thread's
   share of the work (MODIS: 33 MB per chunk) is cut into parts that
   `_pinflate` decodes on every thread at once -- each part after the first
   searches for a plausible block start, decodes with 16-bit markers for
   the unknown window, and is used only if its start is exactly where the
   verified part before it ended (else that stretch is redone serially).
   The other chunks run as each thread's parts are taken.
4. **Batches**: `read_many` / `Group.load()` put every variable's chunks
   in one task list over one output buffer, so a file of many small
   variables keeps every core busy.

Small reads use fewer threads (a thread per ~32 kB of output at most).
A small part of a few shuffled chunks (a point, a level, a window: a
shuffle allows no early stop, so each read decoded whole chunks) goes
through a cache of decoded chunks (`_chunks._cached`): from a chunk's
second read on it is decoded whole, by the same batch kernel, and kept
(least recently used out first, 32 MB for all datasets; `fastnc.release()`
empties it).

**Contiguous datasets** of 256 kB or more are read as *virtual chunks*
(`_h5read._virtual_index`): runs of about 1 MB of the file whose addresses
follow from the shape, so they go through the same kernel -- every core,
byte swap and selection fused, batched with the rest.

**Other filters** (`_staged`): szip, scale-offset, n-bit, bzip2, lzf, lz4,
bitshuffle, zstd, blosc, blosc2 run stage by stage over all of a group's
chunks at once between two staging buffers, each stage one codec's batch
kernel in `_codecs` (one kernel per codec, not one with every codec inside:
numba compiles a kernel when first called, szip's in seconds, and caches
each on its own; a kernel taking the codec as a first-class function
argument is compiled again in every process). A last shuffle is left to
the placement kernel. Filter mask bits are honoured per chunk.

**Reduced-precision types** (n-bit data mostly: an integer narrower than
its size or off bit 0, a float with a custom sign / exponent / mantissa
layout, odd sizes too) are converted to the native type as HDF5 converts
them for h5py and netCDF-c (`_h5read.convert_reduced`).

## Masking and scaling

`_maskscale.apply` is netCDF4-python 1.7's own post-processing carried
over onto the attribute dict, so numpy decides every dtype as it does
there (numpy 1 and 2 differ: float32 data times a float64 scale_factor is
float32 on numpy 1, float64 on 2). For numeric arrays of 32k values or
more the rules are worked out once and a numba kernel applies them block
by block on the pool: scaled values written, masked values flagged in
cache, and only where something is masked the mask written and the
unscaled value put back under it. On the GPU, `_maskscale_gpu` does the
same in CUDA (float32 arithmetic in PTX without flush-to-zero, as numpy
keeps subnormals).

## Reading the file: mapping or positional reads

Metadata and compressed chunks are read through the read-only mapping.
A fresh mapping faults in each page on first touch (22 ms per 113 MB here,
worse when many threads fault small pieces at once); with a decompressor at
work those faults hide behind it. A big *unfiltered* batch (contiguous
data, uncompressed chunks: nothing to hide behind) is instead read with
positional reads (`_pio`, reusing `_nc3`'s overlapped / preadv readers)
into one pooled buffer -- byte ranges merged across gaps under 256 kB --
and the kernel decodes from that buffer as from the file. Metop-SG's
19,432 row chunks: fresh open + read all 57 -> 41 ms. Staging compressed
batches was measured and lost (VIIRS 79 -> 93 ms). v1 B-tree chunk
indexes are read a node at a time with numpy (fixed-size records).

## The xarray backend's decoding

xarray applies its CF mask / scale decoding to each variable in a few
numpy passes on one thread -- 2.5 s of a GOES-19 full-disk band's load.
For numeric, non-time variables with mask/scale attributes the backend
decodes a one-element stand-in with xarray's own `CFMaskCoder` and
`CFScaleOffsetCoder`, takes the elementwise functions they produce, and
applies them to the real values block by block on the pool (attributes
and encoding as the coders leave them). The values are xarray's by
construction, whatever xarray version is installed; tests/test_xarray.py
checks them against xarray's `decode_cf` of the same raw values.

## Memory: the output arena

Windows faults in fresh pages slowly (a 1 GB output: ~55 ms of page
faults). Outputs of 1 MB or more come from `_arena`: an array is a view of
`np.frombuffer(block)` over a pooled anonymous mapping; numpy keeps that
array as the base of every view derived from it (the mapping is not an
ndarray, so its base collapsing stops there), and a finalizer returns the
block to the pool when the last view is gone. Pool capped at a quarter of
RAM (4 GB at most); `fastnc.release()` empties it.

## numba, and the first read of a process

Importing numba and loading the cached kernels takes ~0.8 s once per
process. `_jit` defers the import until a kernel runs; `_chunks` goes
further: until the kernels are loaded, chunks are decoded with `zlib` on
the pool (zlib releases the GIL; it is the reference decoder, so values
are the same), and the first read bigger than 1 MB starts loading numba
in a background thread. The kernel is warmed with the argument types reads
use -- a read-only array for the mapped file (numba types read-only arrays
apart: warming with a writable one compiled the real call again).
`fastnc.warm()` loads everything up front (worker pools:
`ProcessPoolExecutor(initializer=fastnc.warm)`).

numba's cache checks only the source file of a kernel itself, not of the
kernels it calls in other modules: after an edit of `_inflate.py` (or an
upgrade: pip leaves `.nbi`/`.nbc` files behind) the engine kept running
the old inflate. `_jit` keeps a stamp of every fastnc source file (name,
size, modification time) in `__pycache__` and clears the whole cache when
it changes.

## Inflate

`_inflate.inflate` follows libdeflate: a 64-bit bit buffer refilled
without branches (`buf |= ld64(src, p) << bc; p += (63 - bc) >> 3; bc |=
56` leaves at least 56 bits), one table lookup per symbol (11-bit
literal/length and 8-bit distance tables with subtables; an entry packs
the code length, kind, extra-bit count and base), and a fast loop while 16
input and 274 output bytes remain: no bounds checks per symbol, up to
three literals per refill, 16-byte match copies, word fills for distance-1
runs. Tables are built as libdeflate builds them (symbols sorted by
length, each length's entries written once and the table doubled up), and
the fixed code's tables stay in place while blocks keep using them
(`tab[FIXED]`; a workspace that is not zeroed must clear it, as `_blosc`
does). Byte-identical to zlib on every chunk of the real files; 0.9-1.7
GB/s on one core on real data where `zlib.decompress` does 0.4-0.6.
Adler-32 is not checked.

## The GPU

`device='gpu'` reads return cupy arrays. Decompressing deflate on the
card loses (nvCOMP, measured: `tools/profile/nvcomp_deflate.py`), so the
CPU cores decode into a reused page-locked buffer band by band along the
first dimension, each band uploads asynchronously while the next decodes,
and the byte swap and mask/scale run on the card: PCIe carries the packed
integers, not the floats they become. Masked results come back as
`_maskscale_gpu.MaskedArray` (data and mask on the card; `.get()` gives the
exact CPU masked array).

## Remote sources

`_remote.Source` keeps a zeroed buffer the size of the file (pages
committed only when written) and a record of the byte ranges present.
Metadata: the parser calls `need(at, n)` before each structure it reads
(a no-op for local files) and `prefetch(addrs, n)` for groups of them
(a group's members' headers, B-tree children, extensible-array blocks);
the first 1 MB and the last 256 kB come at open. Data: the planned chunks'
byte ranges are fetched together before decoding -- merged across gaps
under 1 MB, cut so every connection shares a big read, 16 keep-alive
connections.

## Checks

`tools/check/` compares against the reference libraries directly (they
need h5py, netCDF4, xarray): `hdf5_vs_h5py.py` (parser), `read_vs_h5py.py`
(values), `meta_vs_netcdf4.py` (netCDF metadata and values),
`h5py_api.py`, `xarray_vs_netcdf4.py`, and one per part (`nc3_*`,
`hdf4_*`, `codecs_*`, `filters_hdf_*`, `maskscale_*`, `pinflate_*`,
`remote_*`). `tests/` runs without them, on `tests/data` and stored
reference digests.

## Things the reference libraries do that fastnc copies on purpose

- netCDF4-python builds compound dtypes with `align=True` from the
  members' netCDF types; a fixed-length string member becomes `S1`, keeping
  its first byte. Its padding bytes are uninitialized: compare compounds
  field by field.
- netCDF-c reads one-byte fixed strings as NC_CHAR (`S1`) and longer
  ones as NC_STRING (`str` values, a `VLType`).
- h5py gives variable-length strings as `bytes` from datasets and `str`
  from attributes, with `dtype.metadata` (`vlen`, `h5py_encoding`, `enum`);
  an enum of FALSE/TRUE over int8 as `bool`; a compound of two equal
  floats named r and i as complex; a 5-byte custom float as float64.
- netCDF-3: `chunking()` is None (netCDF-4 and HDF4 say 'contiguous').
- HDF5's own `chunk_iter` misreports the offsets of extensible-array
  chunks when the unlimited dimension is not the first; fastnc's mapping is
  checked against the chunks' contents instead.

Where fastnc differs on purpose: `Variable.filters()` reports zstd /
bzip2 / blosc when the file uses them (netCDF4-python reports False when
its plugin is not loadable); values come back in native byte order (as
netCDF4-python; h5py keeps the stored order -- the values are equal);
reading past a variable's records in HDF4 gives the fill value where
netCDF-c errors.

## Not supported (yet)

HDF4 szip; zstd dictionaries; blosc2 variable-length blocks and
dictionaries; external storage next to a remote file; writing.

## Reading before numba is loaded

numba's import and the kernels' cache take 0.7-0.9 s of a fresh process and
hold the GIL; started in the background at the first big read, that made
the read itself three times slower (each pool thread queues for the GIL
after every chunk). So nothing starts numba behind the reader's back. Until
the kernels are loaded:

- deflate chunks are inflated by `_zlib` (libdeflate, else zlib, through
  ctypes, which lets go of the GIL) straight into a per-thread buffer, or
  into the output when the chunk is a whole contiguous part of it, then
  unshuffled by one numpy copy per byte plane (`_chunks._quick`);
- unfiltered datasets of many chunks are gathered at once: the chunks'
  bytes concatenated in grid order, the grid's axes interleaved by one
  transposing copy (`_chunks._gather_grid`; Metop-SG's 19,000 one-row
  chunks would otherwise cost 25 us of Python each);
- mask/scale runs `_maskscale.run_plan_numpy`: the kernel's steps as numpy
  ufuncs over blocks of 2^18 values on the pool (a multiple of 64, so
  numpy's SIMD loops split where they would over the whole array: NaN
  payloads come out the same).

The time spent so is summed; once it reaches `_chunks.LOAD_AFTER` (1 s) the
next read loads the kernels first, on its own thread (ski rental: never
more than twice what the better choice would have cost). `fastnc.warm()`
loads them at once. `pytest --kernels off` runs the suite on these paths.

## Values that depend on the platform's numpy

Two things in netCDF4-python's answers come from numpy itself and differ
between numpy builds, so fastnc follows the running numpy rather than a
recording:

- a masked array's `fill_value` type after masked arithmetic (numpy 2.5
  converts it to the result type, earlier versions keep the operand's;
  numpy 1's default integer is 32-bit on Windows only). fastnc builds the
  masked array through numpy.ma with the same inputs as netCDF4-python
  (`_maskscale.finish`), so it gets the running numpy's answer; the tests
  compare the type exactly where the references were recorded (numpy 2.5,
  Windows numpy 1.26) and elsewhere against netCDF4-python's own code run
  under that numpy (`_maskscale._reference`).
- which NaN's payload survives NaN times NaN: the operand order numpy's
  compiled loops use (MSVC and gcc builds differ). This only arises with a
  NaN `scale_factor` / `add_offset`; then numpy does the arithmetic
  (`Plan.nan_const`), on the CPU and for GPU reads alike.

## xarray's decoding by lookup table

The xarray backend leaves CF decoding to xarray's own coders but applies
the functions they make (taken from decoding a one-element stand-in) itself
(`_xarray.DecodedArray`). Those functions are elementwise, so for raw
values of 8 or 16 bits they are applied once to an array of every possible
value (`np.arange(2**bits)` viewed as the raw type): the result is a table
of at most 65,536 decoded values, and decoding an array is `table[raw]`,
copied as bits. Same functions, same inputs, same outputs -- the only
operations whose result could depend on where in an array they run are
NaN-times-NaN payload choices, which need a NaN `scale_factor` /
`add_offset`; those keep the block-by-block path. For a chunked read the
lookup happens as each chunk is decoded (`_fused.read_lut`), so the raw
array never exists. tests/test_xarray.py checks the table path against
`xarray.decode_cf` on every possible value of each 8 / 16-bit type under
several attribute sets, and on every corpus variable.

