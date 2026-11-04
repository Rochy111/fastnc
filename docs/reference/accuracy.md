# Accuracy

fastnc returns the same values as netCDF4-python and h5py, bit for bit. A
release that changes one is a bug.

## What "the same" means

- **Raw values** equal h5py's and netCDF4-python's with masking and scaling
  off, byte for byte. fastnc returns native byte order, as netCDF4-python
  does; h5py keeps the stored order, and the values are equal.
- **Masked and scaled values** follow every rule of netCDF4-python 1.7:
  - `_FillValue` and the default fill values
  - `missing_value`
  - `valid_min`, `valid_max` and `valid_range`
  - `scale_factor` and `add_offset`, with their dtype promotion, which
    differs between numpy 1 and 2
  - `_Unsigned`
  - the choice between a masked and a plain array, and its fill value

  All of these come out in the same float type.
- **GPU reads** equal CPU reads.
- **Metadata** is what each reference gives:
  - netCDF-c's dimensions, phony ones included, and variable order
  - attribute values and types, and user types
  - h5py's link and attribute order, and its dtypes with their metadata

## How it is checked

`tests/test_references.py` compares each of these with what netCDF4-python
and h5py read, from digests stored in the repository:

- every variable and HDF5 object of the test corpus (`tests/data`), which
  covers every layout, chunk index, group and attribute storage, datatype
  and filter
- a synthetic corpus
- real WRF and satellite files: MODIS, VIIRS, Metop-SG, AVHRR, RCM and a
  Ku-band scatterometer

That is about 6,900 tests, raw and masked, on numpy 1 and 2. Each test is a
variable or object read whole and in three selections. GOES-19 ABI files,
which are not in the repository, were checked against both readers directly
with `tools/check/`.

CI runs the suite on Linux, Windows and macOS (Intel and Apple silicon),
Python 3.9 to 3.13 and numpy 1 and 2, once with the compiled kernels and
once without them, as a fresh process reads before its kernels are loaded.

## Where fastnc differs on purpose

- **`Variable.filters()`** reports zstd, bzip2 or blosc when a file uses
  them, where netCDF4-python reports `False` if its plugin cannot load.
- **HDF4 past the last record:** reading a variable past its records gives
  the fill value, where netCDF-c raises an error.

[Internals](../internals.md#things-the-reference-libraries-do-that-fastnc-copies-on-purpose)
lists the reference readers' quirks that fastnc copies on purpose, and
[values that depend on numpy](../internals.md#values-that-depend-on-the-platforms-numpy)
the cases where both readers' answers come from numpy itself.
