# Changelog

All notable changes. Values read by fastnc stay bit-identical to
netCDF4-python's and h5py's in every release; a change that alters them is
a bug.

## Unreleased

### Added
- Reference tests: every variable and HDF5 object of the test corpus (and,
  where present, the synthetic corpus and real WRF / satellite files)
  compared with what netCDF4-python and h5py read, values and metadata,
  from digests stored in `tests/data/references_*.json.gz`
  (`tools/make_references.py`).
- Packaging (`pyproject.toml`), MIT license, CI on Linux / Windows / macOS,
  Python 3.9-3.13, numpy 1 and 2.
