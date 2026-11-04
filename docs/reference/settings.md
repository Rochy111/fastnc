# Settings

Everything adapts to the machine without settings. These are for when you
want something else.

## Functions

| Call | What it does |
|---|---|
| `fastnc.set_threads(n)` | the threads reading uses in this process |
| `fastnc.threads()` | how many threads reading will use |
| `fastnc.hardware()` | what fastnc found here: CPUs allowed, cores, memory, caches, threads, its kernel image |
| `fastnc.tune(gpu=True, verbose=True)` | time real decoding on this machine once, keep what suits it, and link the kernel image |
| `fastnc.warm()` | have the kernels ready now, waiting if they have to be loaded |
| `fastnc.release()` | free the memory kept between reads, the GPU's buffers included |

## Per read

| Argument | What it does |
|---|---|
| `var.read(key, out=array)` | read into an existing array |
| `var.read(key, device='gpu')` | return a cupy array |
| `var.read(key, threads=n)` | use n threads for this read |
| `Dataset(path, device='gpu')` | make the GPU every read's default |
| `storage_options={...}` | for URLs: `region`, `endpoint_url`, `headers`, `timeout`, `retries` |
| `xr.open_dataset(..., engine='fastnc', preload=True)` | read every variable at open in one pass |

## Environment variables

| Variable | What it does |
|---|---|
| `FASTNC_THREADS` | threads reading uses, when `set_threads()` hasn't set them |
| `FASTNC_IMAGE=0` | never use the kernel image; kernels come from numba |
| `FASTNC_IMAGE_BUILD=0` | never build a kernel image in the background |
| `FASTNC_IMAGE_FILE` | use this kernel image file |
| `FASTNC_CACHE_DIR` | where the kernel image and the tuning results are kept (default: the system's cache directory) |
| `FASTNC_NO_CTYPES_ZLIB` | don't use libdeflate or zlib through ctypes before the kernels are loaded |
| `FASTNC_NO_NATIVE_INFLATE` | don't let the kernels call libdeflate |
