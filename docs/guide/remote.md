# Remote files

Any path can be an `http://`, `https://` or `s3://` URL, or a Python file
object, for all three front ends:

```python
import fastnc

url = ('s3://noaa-goes19/ABI-L1b-RadF/2026/302/12/'
       'OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc')
rad = fastnc.Dataset(url)['Rad'][2000:2256, 2000:2256]

f = fastnc.File('https://example.org/data/file.h5')

import xarray as xr
xd = xr.open_dataset(url, engine='fastnc')
```

## What is fetched

fastnc fetches byte ranges, never the whole file, except for HDF4 files,
whose layout it reads in one go:

- **Metadata** comes in aligned 256 kB blocks with read-ahead, because a
  request costs a round trip (40–60 ms to S3) where 256 kB more costs about
  5 ms. The first request takes the start of the file and its end together.
- **Data** for a read is planned first: every chunk it needs is listed, the
  ranges close together are merged, and the rest are fetched on 16 parallel
  connections that stay open between requests.

So reading a window of a big file fetches little more than the chunks under
the window.

Files a remote file points to are fetched from beside it: the raw files of
HDF5 external storage, the files an HDF5 virtual dataset reads from, and
HDF4 external elements. A name relative to the file is taken against its
URL (an absolute one by its last part), over the same headers, timeout and
retries; a file the server does not have is missing, as it would be
locally.

## Options

S3 access is anonymous, which is how NOAA's and NASA's open-data buckets
work. Pass options as `storage_options`:

```python
ds = fastnc.Dataset(url, storage_options={'region': 'us-east-1'})
ds = fastnc.Dataset('s3://bucket/key.nc', storage_options={'endpoint_url': 'https://minio.example.org'})
ds = fastnc.Dataset('https://example.org/protected.nc', storage_options={'headers': {'Authorization': 'Bearer …'}})
```

| Option | What it does |
|---|---|
| `region` | the S3 region of the bucket |
| `endpoint_url` | an S3-compatible server other than AWS |
| `headers` | HTTP headers sent with every request |
| `timeout` | seconds to wait for a response (default 30) |
| `retries` | attempts after a failure (default 5) |

Server errors (5xx), 429 responses, timeouts and dropped connections are
retried with exponential backoff, and other HTTP errors raise `OSError`.
netCDF-c's `#mode=bytes` suffix on a URL is accepted and ignored.

fastnc uses only the standard library for this. For other stores, open the
file with fsspec and pass the file object.
