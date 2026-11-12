# Accuracy

fastnc returns the same values as netCDF4-python and h5py, bit for bit. A
release that changes one is a bug. The exceptions, all on purpose, are
listed [below](#where-fastnc-differs-on-purpose).

## What "the same" means

- **Raw values** equal h5py's and netCDF4-python's with masking and scaling
  off, byte for byte. fastnc returns native byte order. netCDF4-python
  does too for netCDF-3, but keeps a netCDF-4 / HDF5 / HDF4 variable stored
  big-endian (FY-4 AGRI, MODIS) in that order, `dtype` included (`>u2`
  where fastnc says `uint16`), and h5py keeps the stored order everywhere:
  the values are equal, only the byte order flag differs.
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

That is about 7,000 tests, raw and masked, on numpy 1 and 2. Each test is a
variable or object read whole and in three selections. GOES-19 ABI files,
which are not in the repository, were checked against both readers directly
with `tools/check/`.

CI runs the suite on Linux, Windows and macOS (Intel and Apple silicon),
Python 3.9 to 3.13 and numpy 1 and 2, once with the compiled kernels and
once without them, as a fresh process reads before its kernels are loaded.

## Where fastnc differs on purpose

- **A valid range in unpacked units.** The netCDF User Guide and CF say a
  packed variable's `valid_range`, `valid_min` or `valid_max` is in the
  units of the unpacked data when it has the type of `scale_factor` /
  `add_offset` rather than that of the stored integers; netCDF-Java follows
  that. netCDF4-python tests every range against the stored integers, which
  for such files masks nearly everything: NOAA HRD's TDR winds
  (`valid_range = -900., 900.` knots on int16 hundredths of a knot),
  GridSat-B1 and HURSAT-AVHRR brightness temperatures (`140., 375.` kelvin
  on counts). fastnc masks those values where the unpacked value is outside
  the range, in the float type it is read in, whether scaling is on or off.
  (An `_Unsigned` variable read with scaling off holds signed values, which
  such a range cannot be applied to; it is left out there.) A range of the
  stored type, or of any other, is netCDF4-python's, in stored units.
- **`Variable.filters()`** reports zstd, bzip2 or blosc when a file uses
  them, where netCDF4-python reports `False` if its plugin cannot load.
- **Selections of nothing:** an empty list (`v[[]]`) gives an empty array
  where netCDF4-python raises `IndexError`, and an all-False mask on a
  dimension gives numpy's shape (`(0, 13, 17)`) where netCDF4-python's
  shrinks the other dimensions to 1 (`(0, 1, 1)`). Every other index form
  checked (negative steps, numpy integers, masks, lists, Ellipsis anywhere,
  out of range) answers as netCDF4-python does.
- **HDF4 past the last record:** reading a variable past its records gives
  the fill value, where netCDF-c raises an error.

One difference is known that is not on purpose: on macOS (Apple silicon)
with numpy 1, a `uint64` variable read as float32 through a valid range in
unpacked units (`scale_factor = 1`, `add_offset = 0` of type float32) can
come out one float32 step from numpy's answer for a few values far above
2^53 (2 of 90,000 near 2^57 in the tests): numpy there rounds some of these
casts apart from the correctly rounded value the kernels give. Windows,
Linux, and numpy 2 everywhere agree.

[Internals](../internals.md#things-the-reference-libraries-do-that-fastnc-copies-on-purpose)
lists the reference readers' quirks that fastnc copies on purpose, and
[values that depend on numpy](../internals.md#values-that-depend-on-the-platforms-numpy)
the cases where both readers' answers come from numpy itself.
