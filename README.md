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
scatterometer, GOES-19 ABI) with what netCDF4-python and h5py read, from
stored digests: 7,640 comparisons, raw and masked, on numpy 1 and 2.

## Speed

<!-- TABLES -->

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

Not supported yet: HDF5 virtual datasets and external storage, szip in
HDF4, zstd dictionaries, blosc2 variable-length blocks; writing.

## More

- [docs/internals.md](docs/internals.md): how it works, module by module
- [docs/measurements.md](docs/measurements.md): what was measured before
  and while designing it, including what did not pay off
- [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md)

MIT license.
