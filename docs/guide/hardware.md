# GPU, threads and hardware

fastnc adapts to the machine it runs on: how many CPUs the process may use,
its cores and caches, its memory and its storage. Nothing has to be set,
and everything below is optional.

## What fastnc found

```python
import fastnc
fastnc.hardware()
```

```text
{'cpu': '...', 'cpus_allowed': 24, 'cpus_total': 24,
 'topology': {...}, 'memory_total': ..., 'memory_available': ...,
 'l2_bytes': ..., 'last_cache_bytes': ..., 'threads': 24, 'ram_scale': 1.0,
 'tuned': {...}, 'tune_file': '...', 'kernel_image': '...'}
```

`cpus_allowed` counts what the process may really use: its CPU affinity
and, in a container, its CPU quota. Buffer sizes follow the caches, and how
much fastnc keeps between reads follows the memory available.

## Threads

Reading uses one thread per CPU the process may use. To change that:

```python
fastnc.set_threads(8)            # for this process
fastnc.threads()                 # what reading will use
ds['T'].read(threads=1)          # for one read
```

or set `FASTNC_THREADS=8` before Python starts. Small reads don't wake the
threads at all: work is shared out only when there is enough of it.

On Windows with performance and efficiency cores, a script without a window
in front counts as background work, and after about 2 seconds its main
thread is moved to an efficiency core for good, where every read takes 1.6
times as long. Any thread that opens a file or reads with fastnc declares
its work as the user's instead, and the declaration stays on the thread for
your own code too. On other systems, and on CPUs with one kind of core,
nothing changes.

## Tuning

```python
fastnc.tune()
```

times real decoding on this machine, which takes a few seconds, and keeps
what suits it per machine: how many threads, how much work each thread
takes, how long loading numba takes here and the GPU's band size. It also
links the kernel image (below) and waits for it. Without it, fastnc uses
measured defaults, so it is never required.

## The kernel image

fastnc's decoders are numba kernels. Importing numba and loading them takes
0.7 to 0.9 seconds and holds the GIL, for a few MB of finished machine code.
So on x86-64 Windows and Linux, that code is linked into one file of
fastnc's own for this machine, and every process calls it with numpy and
ctypes alone, in under a millisecond: the first read of a script runs at
full speed, and worker pools need no warming.

The image is built in a low-priority background process once reading has
gone on for a second, after an install or an upgrade. Each new image is
tested in a separate process before it is used, so a bad one is never seen.
Until it is there, reads go without kernels: chunks are inflated by
libdeflate or zlib through ctypes on every core, and masking and scaling
are done by numpy. Elsewhere, on macOS and ARM, the kernels load from numba
once reading without them has taken as long as loading them would, or at
`fastnc.warm()`.

| Call or variable | What it does |
|---|---|
| `fastnc.warm()` | have the kernels ready now, from the image if there is one, else from numba, and wait |
| `FASTNC_IMAGE=0` | never use the image: kernels come from numba |
| `FASTNC_IMAGE_BUILD=0` | never build an image in the background |
| `FASTNC_CACHE_DIR` | where the image and the tuning results are kept |

## Reading to the GPU

With cupy installed and an NVIDIA GPU, any read can return a cupy array:

```python
rad = ds['Rad'].read(device='gpu')                  # one read
ds = fastnc.Dataset(path, device='gpu')             # every read of this file
```

The values are identical to a CPU read. Masked results come as data and
mask on the card; `.get()` brings them back as a numpy masked array.

The CPU still decompresses, on every core: decompressing deflate on the
card was measured and lost on every file, with nvCOMP taking 142 ms for a
VIIRS file the CPU decodes in about 75. What the GPU does is everything
after: each 8 MB band crosses PCIe as soon as its last chunk is decoded, so
the upload hides behind the decoding, and masking and scaling to float32
run on the card. That pays most where values are many and scaled (GOES,
VIIRS, MODIS: 1.1 to 2.4 times faster than reading on the CPU and uploading
the floats) and breaks about even for small or unscaled variables.

The first GPU read of a process takes about 9 seconds, for CUDA and cupy to
start.

## Memory

fastnc reuses memory between reads, because Windows maps fresh pages slowly
(about 55 ms per GB), and keeps a cache of decoded chunks for small repeated
reads. What it keeps is let go after 5 seconds unused, and
`fastnc.release()` frees it at once.
