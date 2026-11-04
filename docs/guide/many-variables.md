# Many variables at once

A WRF output file has around 200 variables, most of them a few chunks each.
Read one at a time, each waits for the last, and a variable of four chunks
keeps only four cores busy. fastnc has two ways round that.

## `load()`

```python
import fastnc

ds = fastnc.Dataset('wrfout_d01_2025-10-26_06_20_00.nc')
everything = ds.load()                       # {name: array} for every variable of the group
some = ds.load(['T', 'QVAPOR', 'U', 'V'])    # just these
deep = ds.load(recursive=True)               # every group, keyed by path ('group/var')
```

Every chunk of every variable named is decoded in one pass of the cores,
biggest first, and then all the arrays are masked and scaled in one more
pass. Each value is what `ds[name][...]` would return. A whole WRF d01 file
takes 32 ms this way, against 493 ms for netCDF4-python's loop over its
variables.

## Reading ahead

Code written for netCDF4-python or h5py usually loops:

```python
for name, var in ds.variables.items():
    values = var[:]
```

fastnc notices whole variables being read in the file's order and decodes
the next ones together, in a window of 32 MB that doubles up to 256 MB while
the reads keep to the order. When they stop following it, it stops reading
ahead. So a loop like this, or xarray's `.load()`, gets most of `load()`'s
speed without changing anything: 41 ms for the WRF file.

## Small reads, again and again

A loop over points or levels reads the same chunks many times. Chunks that
cannot be cut short, because they are shuffled, are kept decoded in a 32 MB
cache, as HDF5's own chunk cache does: reading `T[0, 0]` from a WRF file a
second time takes 92 µs, not 3.8 ms.

`fastnc.release()` frees what fastnc keeps between reads: that cache, the
read-ahead window, reused output memory and the GPU's buffers.
