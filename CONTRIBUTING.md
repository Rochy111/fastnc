# Contributing to fastnc

Thanks for helping. fastnc has one non-negotiable rule and a few hard-won
ones; this page lists them, then how to set up, test, measure and submit.

**The rule: values stay bit-identical to the reference readers.** Raw reads
equal h5py's and netCDF4-python's with auto mask and scale off, byte for
byte (fastnc returns native byte order, as netCDF4-python does: values
count, not the byte order flag). Reads with mask and scale on equal
netCDF4-python's in every rule -- `_FillValue` and default fill values,
`missing_value`, `valid_min` / `valid_max` / `valid_range`, `scale_factor` /
`add_offset` with their dtype promotion, `_Unsigned`, the masked array
versus plain array choice, the fill value -- in the same float type. GPU
reads equal CPU reads. A faster reader that is off by one ulp, or masks one
point differently, is a bug. The departures on purpose are listed in
`docs/reference/accuracy.md` (a valid range in unpacked units is one); a new
one needs a reference beyond fastnc (a convention, another reader) and a
test of its own.

## Set up

```
git clone https://github.com/Rochy111/fastnc
cd fastnc
pip install -e ".[xarray,test]"        # add 'gpu' (cupy) if you have an NVIDIA GPU
pytest
```

Python 3.9+, numpy 1.23+ (1.x and 2.x), numba 0.57+. The first `pytest`
run compiles the numba kernels; later runs load them from numba's cache in
`fastnc/__pycache__`, or, once one is built, from this machine's kernel image
(`docs/internals.md`). The tests need neither h5py nor netCDF4-python:
those are only for regenerating the test data (below).

## Tests

`tests/test_references.py` checks fastnc against what netCDF4-python and
h5py read from the same files, stored as digests in
`tests/data/references_*.json.gz`:

- every netCDF variable of every file through `fastnc.Dataset`: raw
  (`set_auto_maskandscale(False)`) and as read by default, whole and for
  three standard selections (a strided slice, an integer index on the first
  dimension, a window of the last two), plus its metadata -- dimensions,
  shape, dtype, attributes with their exact types, `filters()`,
  `chunking()`, `endian()`;
- every file's groups, dimensions (size, unlimited), attributes, variable
  order and data model;
- every HDF5 object through `fastnc.File`: groups and their keys and links,
  attributes, dataset properties (shape, dtype, chunks, compression,
  compression options, shuffle, fletcher32, fill value) and raw values,
  whole, for the same selections and through `read_direct`.

Each variable or object is its own test id, e.g.
`test_nc_values[corpus/nc4.nc:/sub/deep-masked-cpu]`; files are opened
once per session. Three sets of references:

| set | files | in git | runs |
|---|---|---|---|
| `corpus` | `tests/data/*` (small, made by `tools/make_corpus.py`) | yes | always, and in CI |
| `synthetic` | `corpus/synthetic/*` (the older generated corpus) | no | when the files are there |
| `real` | the real WRF / satellite files beside the checkout | no | when the files are there |

The references of all three are committed (they are digests, a few hundred
kB). Tests of a set whose files are absent are not collected; `pytest`'s
header says which sets ran. The files are looked for in
`$FASTNC_REAL_DATA`, the checkout and the folders above it (so a git
worktree under `.claude/worktrees/` finds them too).

```
pytest                                   # every set whose files are here
pytest --refs corpus                     # tests/data only, as CI
pytest -k "nc4.nc and masked"            # one file, one mode
pytest --gpu                             # also compare fastnc.Dataset(path, device='gpu')
pytest --kernels shipped                 # only an image in fastnc/_kernels (tools/make_images.py), numba unable to load
python tools/run_tests_numpy1.py         # the same suite under numpy 1.26 (bench-wrf env)
python tools/run_tests_numpy1.py --nep50 # ... and warn wherever numpy 2 would compute differently
```

After any change, `python tools/check.py` says in about a minute whether it broke
or slowed anything: this machine's kernel image built from these sources, the
tests in parallel, then, if `FASTNC_CORPUS` names folders of
files, every dataset against answers saved from h5py and netCDF4, and 20 kinds of
read against `HEAD`, processes taking turns. The reference libraries run once,
with `--save`, which is why it is quick. Files from elsewhere are worth having
there: they found what no test made here did. `--full` adds the suite under
numba and with no kernels at all, before anything is merged.

CI runs `pytest --refs corpus` on Linux, Windows and macOS (Apple
Silicon), Python 3.9-3.13, numpy 1 and 2, and one job with xarray. No CI
runner has an NVIDIA GPU: run `pytest --gpu` on a machine with one before a
release.

### How a value is compared

`tests/helpers.py` reduces a read to a digest, and the reference generator
uses the same code: the Python type of the result (`ndarray`,
`MaskedArray`, `np.float32`, `str` ...), shape, dtype in native-order form,
and the sha256 of the values in native byte order. A masked array adds the
digest of its mask, the number of masked points and its fill value, and its
data is digested with masked points set to zero (what lies under a mask is
not part of the answer). Compound values are digested field by field, so
padding bytes do not count (netCDF4-python leaves them uninitialised).
Object arrays (vlen strings, vlen sequences) and strings are digested
through a canonical JSON form that keeps `str` and `bytes` apart.
Attribute values are stored exactly: type (`str`, `bytes`, numpy scalar of
a given dtype, array, list), dtype and bytes.

netCDF4-python's answers depend in places on numpy's major version: the
numpy type of a masked array's fill value (an integer array's default
999999 is an int32 under numpy 1, an int64 or uint64 under numpy 2; a
packed int16 variable's float32 result keeps an int16 fill value under
numpy 1). Where they differ, the numpy 1 answer is stored under `numpy1` in
that digest and used when the tests run on numpy 1.

A reference reader's error is recorded instead of a digest (netCDF4-python
cannot open some plain HDF5 files; some filters fail in both readers); a
test whose reference is an error skips. netCDF4-python's `filters()` says
`False` for zstd, bzip2 and blosc whenever netCDF-C cannot load the plugin,
so those three are checked against the filter pipeline h5py saw instead.

### Checking real files after a change

The test files are small. For a change to a decoder, also run the `real`
and `synthetic` sets on a machine that has them (`pytest --refs real,synthetic`):
they cover what the corpus cannot (WRF netCDF3 and netCDF4, MODIS / VIIRS /
EUMETSAT HDF5 with compound types, szip, big chunk tables).

## Regenerating the test data

The generated files are never edited by hand. In the dev environment (h5py,
netCDF4-python, hdf5plugin):

```
python tools/make_corpus.py                       # tests/data/*.h5, *.nc (deterministic: fixed seeds)
python tools/make_references.py                   # all three reference sets
python tools/make_references.py --set corpus      # only tests/data's
python tools/make_references.py --set real --jobs 3
python tools/make_references.py --set corpus --check   # do today's readers still agree? (writes nothing)
```

`make_references.py` points `HDF5_PLUGIN_PATH` at hdf5plugin's plugins
and imports hdf5plugin, so both readers decode zstd, blosc, lz4, bzip2 and
bitshuffle. It runs a second interpreter (`--alt-python`, by default the
numpy 1.26 bench-wrf env) for two things: netCDF4-python's numpy 1 answers,
and h5py on HDF5 2.0 for files the dev env's HDF5 1.14 cannot read (the
synthetic `h5_chunk_index_*`, `h5_single_chunk`, `h5_special`); the
reference file says which files were read there. It also compares the two
readers' raw values variable by variable and lists every disagreement in
the reference file's summary (today: only MODIS's compound `Level 1B Swath
Metadata`, where netCDF-C reads a 4-byte string member as one byte).

After `make_corpus.py`, regenerate the corpus references and commit both;
`test_corpus_references_cover_tests_data` fails until you do.

## Writing kernels

fastnc's decoders are numba `@njit(cache=True, nogil=True)` functions run
on fastnc's own thread pool. What has been measured to matter (see
`docs/contracts.md`):

- **Import `njit` from `fastnc._jit`, not from numba.** It defers
  importing numba until the first kernel runs, so listing a file never
  pays for it, and it is what lets a kernel run from the image, without numba.
- **Timing a change to Python around the kernels takes seconds:** those edits
  compile nothing (`_jit.kernel_stamp`). Set `FASTNC_IMAGE_FILE` to the image and
  run `tools/profile/quick.py`. Compare versions with `tools/profile/ab.py`, which
  alternates processes, not from two benchmark runs: from run to run those
  differ by more than most changes.
- **A new module with kernels goes in `_jit.KERNEL_FILES`** (a test checks), and
  a kernel with new argument types in `fastnc/_signatures.py`: after a full
  test run and `_image.build()`, `python tools/make_signatures.py`.
- **A kernel Python calls takes arrays, integers, bools, None and tuples of
  those.** Floats, or a return of an array, keep it out of the image, and a
  call of it then loads numba.
- **No `parallel=True` / `prange`.** numba's threads spin after each loop.
  Spread work with fastnc's pool (`fastnc/_pool.py`): the caller and pool
  threads take items off a shared atomic counter until none is left.
- **No `inline='always'`.** Plain `@njit` helpers are inlined by LLVM
  anyway; numba's own IR inlining left reference counting in hot loops (ten
  times slower under numba 0.63).
- **Explicit loops, int64 / uint64 indices, and never uint64 with int64 in
  one expression** (numba makes the result float64). Unsigned indices in
  hot loops avoid numba's negative-index fix-up; make constants unsigned
  too. Loops, not slice assignment, to copy between arrays.
- **Never write outside the output range you were given**: kernels run on
  many threads writing into one shared output array. Unaligned word loads
  and stores and an atomic counter are in `fastnc/_intrin.py`.
- **No fresh large arrays per chunk** in code that runs on every core: fresh
  memory costs a page fault per 4 kB (on Windows 91 ms per GB on one thread,
  19 ms on 24). Reuse per-thread scratch (`ws`) or write into the caller's
  array.
- **Nothing tuned to one machine.** Thread counts come from `_pool`, memory
  held between reads scales with `_hw.scaled`, block sizes with `_hw.block`;
  what can only be measured belongs in `fastnc.tune()`.
- Codecs follow the `decode(src, p, pend, dst, o, oend, cd, ws)` contract of
  `docs/contracts.md` section 1, and are tested byte for byte against the
  reference decoder.

## Python code and numpy 2

numpy 2 keeps arithmetic on `numpy.uint8` / `int16` scalars in that type
(NEP 50): `b[p] * 100` wraps round. Python code that reads header fields
from a numpy buffer (the mapped file) must read them as Python ints
(`memoryview(buf).cast('B')`, `struct.unpack_from`). To find every site
that behaves differently under numpy 2, run the suite on numpy 1.26 with
`python tools/run_tests_numpy1.py --nep50` (numpy's `weak_and_warn`
promotion state).

## Measuring performance

`tools/profile/` has the scripts behind `docs/measurements.md`. To compare
two versions, alternate them (A, B, A, B ...) and look at the spread, not
one run. Things that will mislead you:

- **Hybrid CPUs on Windows** (Intel 12th gen and later): a script started
  from a terminal counts as background work and Windows parks its main
  thread on an efficiency core, ~2 s in and for good -- single-core timings
  swing 30-50%. fastnc declares any thread that opens a file or reads, so
  reads are safe, but what runs before them is not, nor are other readers:
  call `tools/profile/common.fast_thread()` on timing threads.
- **CPU time on Windows** (`time.process_time()`) ticks every 15.6 ms.
- **`PYTHONDONTWRITEBYTECODE`** makes every import recompile from source.
- **numba's cache** can keep serving an edited kernel's old code: delete
  `fastnc/__pycache__/*.nbi` and `*.nbc` before timing an edit.
- **The OS file cache**: a file's first read and later reads differ
  wildly; say which you report. A file that was ever mapped stays cached
  through `FILE_FLAG_NO_BUFFERING`: `tools/profile/cold.py` shows how to get a
  truly cold one, and checks by counting what left the disk.
- **Fresh memory**: time `read(..., out=array)` to see the decoder alone.

## Submitting

Open an issue first for anything large. A pull request should say what it
changes and why; for performance work, the before and after numbers (with
the machine) and how you checked the values are unchanged (`pytest`, and
the real-file sets where you have them). Keep docstrings as prose saying
what a thing is and why, and comments short, trailing the line they
explain. Add a line to `CHANGELOG.md`'s Unreleased section.

## Branches and commits

- **Work on a branch per change off `dev`** (`perf/libdeflate-in-kernel`,
  `fix/fill-value-type`, `docs/benchmarks`) and commit as often as you
  like there.
- **Land it on `dev` as one commit**: `git merge --squash` the branch,
  then commit with an imperative subject of at most ~72 characters and a
  short body saying why, with its line under "Unreleased" in
  `CHANGELOG.md`. Measurements and rejected alternatives go in
  `docs/measurements.md`, not in commit messages.
- **Before it lands**: the suite passes under numpy 2 and numpy 1
  (`pytest`, `python tools/run_tests_numpy1.py`), with `--kernels off` and
  `--kernels numba`, and with `--gpu` where the GPU path is touched.
- **CI runs on every push to `dev`**: every Python from 3.9 to 3.13, numpy
  1 and 2, Linux, Windows and macOS, compiling fresh. Code commits never
  carry `[skip ci]`.
- **`main` holds releases only**, one squashed commit each, and is never
  rewritten once pushed.
- **Outside contributions** come as pull requests to `dev`.

## Making a release

Versions follow [semantic versioning](https://semver.org/). While fastnc
is 0.x, the minor version (0.1.0 to 0.2.0) is for new features or changed
behaviour, and the patch version (0.1.0 to 0.1.1) for fixes only. The
version is `__version__` in `fastnc/__init__.py`.

1. On `dev`, set `__version__`, turn "Unreleased" in `CHANGELOG.md` into
   the version's heading with the date, and start a new empty "Unreleased"
   above it. Push `dev` and wait for its tests to pass on every system.
2. Squash `dev` onto `main`, tag it and push both:

    ```bash
    git switch main
    git pull
    git merge --squash dev
    git commit -m "Release 0.2.0: what it brings"
    git tag -a v0.2.0 -m "fastnc 0.2.0"
    git push origin main v0.2.0
    ```

   The tag runs `release.yml`: the tests again, then the GitHub release,
   with that version's changelog section as its notes. It stops if the
   tests fail, the tag isn't on `main` or the changelog has no section for
   it. The documentation site rebuilds from `main`.
3. Record the squash in `dev`'s history, so the next squash brings only
   what is new (`main`'s files are `dev`'s already):

    ```bash
    git switch dev
    git merge -s ours origin/main -m "Record 0.2.0's squash on main"
    git push origin dev
    ```

fastnc is not published to PyPI; it installs from GitHub.
