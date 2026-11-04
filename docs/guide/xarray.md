# xarray

Installing fastnc registers an xarray backend called `fastnc`:

```python
import xarray as xr

xd = xr.open_dataset('wrfout_d01_2025-10-26_06_20_00.nc', engine='fastnc')
xd = xr.open_dataset('VJ102IMG_NRT.A2026265.2006.021.nc', engine='fastnc',
                     group='observation_data')
```

It returns the same `Dataset` as `engine='netcdf4'`: the same variables,
coordinates, attributes and encodings, and after `.load()` the same values,
because xarray's own decoding functions run on fastnc's raw values.

## Options

`open_dataset` takes xarray's usual arguments (`mask_and_scale`,
`decode_times`, `decode_coords`, `drop_variables`, `concat_characters`,
`use_cftime`, `decode_timedelta`, `group`) and two of fastnc's:

| Argument | What it does |
|---|---|
| `preload=True` | reads every variable at open, in one pass of the decoder, so a later `.load()` costs nothing. Use it when you will read most of the file. |
| `storage_options={...}` | for URLs: see [Remote files](remote.md). |

`open_mfdataset(..., engine='fastnc')` works as with any other backend, and
datasets pickle, so dask's distributed workers can open them.

## Why it is faster

- **Lazy variables read ahead.** xarray's `.load()` reads variables one
  after another in the file's order. fastnc sees that and decodes the next
  ones together on every core.
- **Decoding by table.** For 8- and 16-bit packed data, such as GOES,
  VIIRS and MODIS radiances, xarray's mask and scale functions are applied
  once to each possible raw value, at most 65,536 of them, and the decoded
  array is looked up from that table while each chunk is decoded. It is the
  same functions on the same values, so the same bits, without xarray's
  several passes over the array.
- **Opening is quick.** The netcdf4 engine took 7 seconds to open a WRF file
  where netCDF4-python's plugins failed to load, and h5netcdf 10 seconds.
  fastnc took 30 ms.

The [performance page](../speed.md#xarray) has the times by engine.
