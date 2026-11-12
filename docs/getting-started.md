# Getting started

## Installation

fastnc needs Python 3.9 or newer and installs with only numpy (1.23 or
newer, 1.x or 2.x) and numba.

```bash
pip install -U git+https://github.com/Rochy111/fastnc
```

Extras add the xarray backend and GPU reads (cupy, CUDA 12):

```bash
pip install "fastnc[xarray] @ git+https://github.com/Rochy111/fastnc"
pip install "fastnc[all] @ git+https://github.com/Rochy111/fastnc"
```

From a clone, use `pip install -e .` or `pip install -e ".[xarray]"`. It is
tested on Linux, Windows and macOS (Intel and Apple silicon), Python 3.9 to
3.13, with numpy 1 and 2. Without an NVIDIA GPU, everything runs on the CPU.
h5py and netCDF4-python are not needed; they are used only to make the test
references.

## The first read after installing

fastnc's decoders are numba kernels compiled for your CPU. They are compiled
once per machine and version, in a low-priority background process that the
first second of reading starts. It takes a few minutes, and nothing waits for
it: until it is done, files are read with libdeflate or zlib through ctypes on
every core, and masking and scaling are done with numpy. To have it done at
once and wait for it:

```python
import fastnc
fastnc.tune()        # times this machine, keeps what suits it, links the kernels
```

The compiled kernels are linked into one file for your CPU (on x86-64 Windows
and Linux), which every later process runs from with numpy and ctypes alone,
without importing numba. So a script's first read is as fast as its tenth.
See [GPU, threads and hardware](guide/hardware.md) for what fastnc finds on
your machine and how to change it.

## Opening a file

`fastnc.Dataset` has netCDF4-python's API:

```python
import fastnc

ds = fastnc.Dataset('wrfout_d01_2025-10-26_06_20_00.nc')
print(ds.dimensions['west_east'].size, list(ds.variables)[:5])
t2 = ds['T2'][0]                  # the first time step: masked and scaled, as netCDF4-python
print(ds['T2'].units, t2.dtype, t2.shape)
```

`fastnc.File` has h5py's API, for any HDF5 file, netCDF-4 included:

```python
f = fastnc.File('VJ102IMG_NRT.A2026265.2006.021.nc')
d = f['observation_data/I05']
print(d.shape, d.dtype, d.chunks, d.compression)
window = d[3000:3512, 3000:3512]  # raw values, as h5py returns them
```

And xarray uses fastnc as an engine:

```python
import xarray as xr
xd = xr.open_dataset('MOD021KM.A2025301.1445.007.2025301204859.nc', engine='fastnc')
```

Each gives the same values as the reader it stands in for: the same bits,
in the same dtype, masked where that reader masks (with one rule fastnc
follows on purpose, a valid range in unpacked units: see
[Accuracy](reference/accuracy.md)). Switching is an import change, and
nothing else in your code has to change.

## Reading a whole file

Reading every variable is where fastnc gains the most, because it decodes
them together:

```python
everything = ds.load()            # {name: array} for every variable, in one pass
```

A loop over `ds.variables` in the file's order gets most of the same gain,
because fastnc reads ahead. See [Many variables at once](guide/many-variables.md).

## Remote files

Paths can be HTTP(S) or S3 URLs. fastnc fetches only the byte ranges a read
needs, on parallel connections:

```python
url = 's3://noaa-goes19/ABI-L1b-RadF/2026/302/12/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc'
rad = fastnc.Dataset(url)['Rad'][:]
```

See [Remote files](guide/remote.md).

## Next steps

- [netCDF4-python's API](guide/netcdf4.md) and [h5py's API](guide/h5py.md) list what each front end supports, and what fastnc adds: zero-copy `memmap()`, `read_async()`, NaN-filled reads, per-read settings, `tocdl()`.
- [xarray](guide/xarray.md) covers the backend and its `preload` option.
- [GPU, threads and hardware](guide/hardware.md) covers reading to the GPU and choosing threads.
- [Performance](speed.md) has the measurements.
