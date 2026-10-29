# fastnc

A fast netCDF / HDF5 reader for Python (CPU and GPU), aiming to replace
netCDF4-python and h5py for reading with bit-identical values.

Status: measuring. Nothing to install yet. See
[docs/measurements.md](docs/measurements.md) for where today's readers
spend their time on real WRF and satellite files.

- `tools/profile/` -- the scripts behind those measurements
- `tools/baseline/` -- the earlier session brief, baseline timings, value
  digests and the script that made the synthetic corpus (`corpus/`, not in git)
