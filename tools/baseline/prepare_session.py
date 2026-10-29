"""Prepare a standalone session to build a fast netCDF / HDF reader, the way
fastgrib was built: find and classify the files it must read, generate the
format/filter combinations real files don't cover, measure today's readers
on all of them, and write the brief the new session starts from.

    python prepare_session.py                      # the default roots below
    python prepare_session.py D:/data E:/more --per-dir 4 --repeat 5

Writes, beside this script:
    corpus/synthetic/      generated files: every netCDF3 / netCDF4 / HDF5 layout
                           and filter this machine's libraries can write
    corpus.json            every file picked: format, variables, chunks, filters
    baseline.json          per file and reader: open / read-all seconds, MB/s,
                           value digests (do the readers agree?)
    BASELINE.md            the same as tables
    SESSION_BRIEF.md       the task for the new session, with the above folded in

Reads only: nothing in the roots is changed. Folders are sampled (--per-dir
files each: a WRF run's hundreds of wrfouts share one format), never walked
file by file.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import date

import numpy as np


def _plugin_path():
    """netCDF4-python ships HDF5 filter plugins (zstd ...) inside its package
    but the C library only finds them through HDF5_PLUGIN_PATH, read when it
    loads -- so set it before netCDF4 is imported."""
    if os.environ.get('HDF5_PLUGIN_PATH'):
        return
    import importlib.util
    spec = importlib.util.find_spec('netCDF4')
    if spec and spec.submodule_search_locations:
        d = os.path.join(list(spec.submodule_search_locations)[0], 'plugins')
        if os.path.isdir(d):
            os.environ['HDF5_PLUGIN_PATH'] = d


_plugin_path()

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOTS = [r'E:\GPUWRF\runs', r'E:\WRF\STATIONARY\WRF_OUT\run_2025-10-25_18_00_00']

MAGIC = {b'CDF\x01': 'netCDF3 classic', b'CDF\x02': 'netCDF3 64-bit offset',
         b'CDF\x05': 'netCDF3 CDF-5 (64-bit data)', b'\x0e\x03\x13\x01': 'HDF4'}
HDF5_SIG = b'\x89HDF\r\n\x1a\n'


# ---------------------------------------------------------------- timing

def fast_thread():
    """Keep this thread off the efficiency cores on Windows (a script started
    from a terminal counts as background work there, and hybrid-CPU timings
    then swing 30-50%: fastgrib's _cpu._fast_thread)."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG),
                        ('StateMask', wintypes.ULONG)]
        k = ctypes.windll.kernel32
        k.GetCurrentThread.restype = wintypes.HANDLE
        st = State(1, 1, 0)
        k.SetThreadInformation(wintypes.HANDLE(k.GetCurrentThread()), 3,
                               ctypes.byref(st), ctypes.sizeof(st))
    except Exception:
        pass


def timed(fn, repeat):
    """(first, best) seconds of fn() over repeat runs, and its last result."""
    first = best = None
    out = None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        d = time.perf_counter() - t
        first = d if first is None else first
        best = d if best is None else min(best, d)
    return first, best, out


# ---------------------------------------------------------------- formats

def file_format(path):
    """The format by its leading bytes (HDF5 may sit after a user block at
    512, 1024, 2048 ... bytes), or None."""
    try:
        with open(path, 'rb') as f:
            head = f.read(4)
            if head in MAGIC:
                return MAGIC[head]
            off = 0
            size = os.path.getsize(path)
            while off < min(size, 1 << 20):
                f.seek(off)
                if f.read(8) == HDF5_SIG:
                    return 'HDF5' + ('' if off == 0 else ' (user block %d)' % off)
                off = 512 if off == 0 else off * 2
    except OSError:
        pass
    return None


def sample_files(roots, per_dir, max_files):
    """Up to per_dir files of every folder under roots (first, middle, last
    by name, then evenly spread), recognised by their leading bytes."""
    picked = []
    for root in roots:
        if not os.path.exists(root):
            print('  (missing root %s)' % root)
            continue
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if not x.startswith(('.', '$', '__pycache__')))
            files = sorted(f for f in files if not f.endswith(('.py', '.txt', '.log', '.png',
                                                               '.jpg', '.json', '.csv', '.md')))
            if not files:
                continue
            idx = sorted({0, len(files) // 2, len(files) - 1} |
                         set(np.linspace(0, len(files) - 1, per_dir).astype(int).tolist()))
            got = 0
            for i in idx:
                p = os.path.join(d, files[i])
                fmt = file_format(p)
                if fmt:
                    picked.append((p, fmt))
                    got += 1
                if got >= per_dir:
                    break
            if len(picked) >= max_files:
                return picked
    return picked


def describe(path, fmt):
    """Variables, shapes, dtypes, chunking and filters of one file."""
    info = {'path': path, 'format': fmt, 'bytes': os.path.getsize(path), 'variables': []}
    try:
        if fmt.startswith('HDF5'):
            import h5py
            with h5py.File(path, 'r') as f:
                info['netcdf4'] = '_NCProperties' in f.attrs or any(
                    'DIMENSION_LIST' in v.attrs for v in f.values() if isinstance(v, h5py.Dataset))
                if '_NCProperties' in f.attrs:
                    p = f.attrs['_NCProperties']
                    info['nc_properties'] = p.decode() if isinstance(p, bytes) else str(p)

                def visit(name, obj):
                    if not isinstance(obj, h5py.Dataset):
                        return
                    dcpl = obj.id.get_create_plist()
                    layout = {0: 'compact', 1: 'contiguous', 2: 'chunked', 3: 'virtual'}.get(
                        dcpl.get_layout(), str(dcpl.get_layout()))
                    filters = []
                    for i in range(dcpl.get_nfilters()):
                        code, _, vals, fname = dcpl.get_filter(i)
                        filters.append('%s(%d)%s' % (fname.decode(errors='replace') or '?', code,
                                                     list(vals) if vals else ''))
                    try:
                        stored = obj.id.get_storage_size()
                    except Exception:
                        stored = None
                    try:
                        nchunks = obj.id.get_num_chunks() if layout == 'chunked' else None
                    except Exception:
                        nchunks = None
                    info['variables'].append({
                        'name': name, 'shape': list(obj.shape), 'dtype': str(obj.dtype),
                        'layout': layout, 'chunks': list(obj.chunks) if obj.chunks else None,
                        'nchunks': nchunks, 'filters': filters,
                        'raw_bytes': int(obj.size * obj.dtype.itemsize) if obj.dtype.kind != 'O' else None,
                        'stored_bytes': stored})
                f.visititems(visit)
        elif fmt.startswith('netCDF3'):
            import netCDF4
            with netCDF4.Dataset(path) as ds:
                info['unlimited'] = [d for d, dim in ds.dimensions.items() if dim.isunlimited()]
                for name, v in ds.variables.items():
                    info['variables'].append({
                        'name': name, 'shape': list(v.shape), 'dtype': str(v.dtype),
                        'layout': 'record' if v.dimensions and v.dimensions[0] in info['unlimited']
                        else 'fixed',
                        'raw_bytes': int(np.prod(v.shape) * v.dtype.itemsize) if v.dtype != str else None,
                        'attrs': sorted(k for k in v.ncattrs() if k in (
                            'scale_factor', 'add_offset', '_FillValue', 'missing_value',
                            'valid_range', 'valid_min', 'valid_max', '_Unsigned'))})
        elif fmt == 'HDF4':
            try:
                from pyhdf.SD import SD
                sd = SD(path)
                for name, (dims, rank, shape, typ, nattr) in sd.datasets().items():
                    info['variables'].append({'name': name, 'shape': list(np.atleast_1d(shape)),
                                              'hdf4_type': typ})
            except ImportError:
                info['note'] = 'pyhdf not installed: HDF4 not described'
    except Exception as e:
        info['error'] = '%s: %s' % (type(e).__name__, e)
    return info


def signature(info):
    """What makes a file a different test case: format, layouts and filters."""
    kinds = sorted({(v.get('layout'), tuple(f.split('(')[0] for f in v.get('filters', [])), v.get('dtype'))
                    for v in info['variables']})
    return info['format'], bool(info.get('netcdf4')), tuple(kinds)


# ---------------------------------------------------------------- synthetic corpus

def field(shape, seed, dtype='f4'):
    """A smooth field with noise -- compresses like model output, not like zeros."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.linspace(0, 6.3, n) for n in shape], indexing='ij')
    v = sum(np.sin((k + 1) * g) for k, g in enumerate(grids)) * 10 + 280
    v = v + rng.normal(0, 0.5, shape)
    return v.astype(dtype)


def make_synthetic(out):
    """Every netCDF3 / netCDF4 / HDF5 storage and filter combination this
    machine's netCDF4-python and h5py can write. Returns [(path, format)]."""
    import netCDF4
    os.makedirs(out, exist_ok=True)
    made, skipped = [], []
    shape3 = (6, 240, 320)            # time, y, x (~1.8 MB float32)

    def nc(name, fmt, **var_kw):
        p = os.path.join(out, name + '.nc')
        if os.path.exists(p):
            made.append(p)
            return
        try:
            with netCDF4.Dataset(p, 'w', format=fmt) as ds:
                ds.createDimension('time', None)
                ds.createDimension('y', shape3[1])
                ds.createDimension('x', shape3[2])
                ds.title = 'fastnc synthetic: ' + name
                for i, (vname, dt) in enumerate((('T', 'f4'), ('QV', 'f4'), ('P', 'f8'))):
                    v = ds.createVariable(vname, dt, ('time', 'y', 'x'), **var_kw)
                    v.units = 'K'
                    v[:] = field(shape3, i, dt)
                if fmt.startswith('NETCDF3') or fmt == 'NETCDF4_CLASSIC':
                    # packed shorts with scale/offset and a fill value (CF packing)
                    s = ds.createVariable('Tpacked', 'i2', ('time', 'y', 'x'), fill_value=-32768,
                                          **var_kw)
                    s.scale_factor, s.add_offset = np.float32(0.01), np.float32(280)
                    s.set_auto_maskandscale(True)
                    data = field(shape3, 9)
                    data[:, :10, :10] = np.nan
                    s[:] = np.ma.masked_invalid(data)
                    b = ds.createVariable('flags', 'i1', ('time', 'y', 'x'), **var_kw)
                    b._Unsigned = 'true'
                    b[:] = (field(shape3, 7) % 256).astype('u1').view('i1')
            made.append(p)
        except Exception as e:
            skipped.append((name, '%s: %s' % (type(e).__name__, e)))
            if os.path.exists(p):
                os.remove(p)

    nc('nc3_classic', 'NETCDF3_CLASSIC')
    nc('nc3_64bit_offset', 'NETCDF3_64BIT_OFFSET')
    nc('nc3_cdf5', 'NETCDF3_64BIT_DATA')
    nc('nc4_contiguous', 'NETCDF4', contiguous=False)       # (unlimited dim: chunked anyway)
    nc('nc4_classic_zlib', 'NETCDF4_CLASSIC', zlib=True, complevel=4)
    nc('nc4_chunked_raw', 'NETCDF4', chunksizes=(1, 120, 160))
    nc('nc4_zlib1', 'NETCDF4', zlib=True, complevel=1)
    nc('nc4_zlib9_shuffle', 'NETCDF4', zlib=True, complevel=9, shuffle=True)
    nc('nc4_zlib_shuffle_fletcher', 'NETCDF4', zlib=True, shuffle=True, fletcher32=True)
    nc('nc4_zlib_bigendian', 'NETCDF4', zlib=True, endian='big')
    nc('nc4_whole_field_chunks', 'NETCDF4', zlib=True, chunksizes=(1, 240, 320))
    nc('nc4_tiny_chunks', 'NETCDF4', zlib=True, chunksizes=(1, 16, 16))
    for comp in ('szip', 'zstd', 'bzip2', 'blosc_lz', 'blosc_lz4', 'blosc_lz4hc', 'blosc_zlib',
                 'blosc_zstd'):
        nc('nc4_' + comp, 'NETCDF4', compression=comp)
    for mode in ('BitGroom', 'BitRound', 'GranularBitRound'):
        nc('nc4_quantize_' + mode, 'NETCDF4', zlib=True, significant_digits=3, quantize_mode=mode)

    # netCDF4 features beyond the classic model: groups, vlen strings, an
    # unsigned type, a compound type -- in one file
    p = os.path.join(out, 'nc4_groups_types.nc')
    if not os.path.exists(p):
        try:
            with netCDF4.Dataset(p, 'w', format='NETCDF4') as ds:
                ds.createDimension('n', 1000)
                g = ds.createGroup('forecast/surface')
                g.createVariable('u8', 'u8', ('n',), zlib=True)[:] = np.arange(1000, dtype='u8') ** 3
                sv = g.createVariable('names', str, ('n',))
                sv[:] = np.array(['station_%d' % i for i in range(1000)], dtype=object)
                ct = ds.createCompoundType(np.dtype([('lat', 'f4'), ('lon', 'f4'), ('id', 'i4')]), 'obs_t')
                cv = ds.createVariable('obs', ct, ('n',))
                a = np.zeros(1000, dtype=[('lat', 'f4'), ('lon', 'f4'), ('id', 'i4')])
                a['lat'], a['lon'], a['id'] = np.linspace(-90, 90, 1000), np.linspace(0, 360, 1000), np.arange(1000)
                cv[:] = a
            made.append(p)
        except Exception as e:
            skipped.append(('nc4_groups_types', '%s: %s' % (type(e).__name__, e)))

    # plain HDF5 (h5py): layouts, h5py-only filters, chunk index types
    try:
        import h5py
    except ImportError:
        h5py = None
        skipped.append(('hdf5_*', 'h5py not installed'))
    if h5py is not None:
        def h5(name, libver=None, **kw):
            p = os.path.join(out, name + '.h5')
            if os.path.exists(p):
                made.append(p)
                return
            try:
                with h5py.File(p, 'w', libver=libver) as f:
                    for i, dt in enumerate(('f4', 'f8', 'i2')):
                        f.create_dataset('v_%s' % dt, data=field(shape3, i + 20, dt), **kw)
                made.append(p)
            except Exception as e:
                skipped.append((name, '%s: %s' % (type(e).__name__, e)))
                if os.path.exists(p):
                    os.remove(p)

        h5('h5_contiguous')
        h5('h5_chunked_gzip', chunks=(1, 120, 160), compression='gzip', compression_opts=6)
        h5('h5_lzf', chunks=(1, 120, 160), compression='lzf')
        h5('h5_szip', chunks=(1, 120, 160), compression='szip')
        h5('h5_gzip_shuffle_fletcher', chunks=(1, 120, 160), compression='gzip', shuffle=True,
           fletcher32=True)
        h5('h5_chunk_index_latest', libver='latest', chunks=(1, 120, 160), compression='gzip',
           maxshape=(None, 240, 320))                        # extensible-array / v2 B-tree index
        h5('h5_chunk_index_fixed_array', libver='latest', chunks=(1, 120, 160), compression='gzip')
        h5('h5_single_chunk', libver='latest', chunks=shape3, compression='gzip')
        # scale-offset (lossy for floats, lossless for ints), n-bit, compact, virtual
        p = os.path.join(out, 'h5_special.h5')
        if not os.path.exists(p):
            try:
                with h5py.File(p, 'w', libver='latest') as f:
                    f.create_dataset('scaleoffset_f4', data=field(shape3, 30), chunks=(1, 120, 160),
                                     scaleoffset=2)
                    f.create_dataset('scaleoffset_i4', data=(field(shape3, 31) * 100).astype('i4'),
                                     chunks=(1, 120, 160), scaleoffset=0)
                    sp = h5py.h5s.create_simple((100,))
                    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
                    dcpl.set_layout(h5py.h5d.COMPACT)
                    h5py.h5d.create(f.id, b'compact', h5py.h5t.NATIVE_FLOAT, sp, dcpl=dcpl).write(
                        h5py.h5s.ALL, h5py.h5s.ALL, np.arange(100, dtype='f4'))
                    a = f.create_dataset('src_a', data=field((240, 320), 32))
                    b = f.create_dataset('src_b', data=field((240, 320), 33))
                    lay = h5py.VirtualLayout(shape=(2, 240, 320), dtype='f4')
                    lay[0] = h5py.VirtualSource(a)
                    lay[1] = h5py.VirtualSource(b)
                    f.create_virtual_dataset('virtual', lay)
                    f.create_dataset('bigendian', data=field(shape3, 34).astype('>f4'),
                                     chunks=(1, 120, 160), compression='gzip')
                made.append(p)
            except Exception as e:
                skipped.append(('h5_special', '%s: %s' % (type(e).__name__, e)))

    # HDF4 (MODIS, older NASA products): only with pyhdf
    try:
        from pyhdf.SD import SD, SDC
        p = os.path.join(out, 'hdf4_sds_deflate.hdf')
        if not os.path.exists(p):
            sd = SD(p, SDC.WRITE | SDC.CREATE)
            for i, (n, t, dt) in enumerate((('T', SDC.FLOAT32, 'f4'), ('Q', SDC.INT16, 'i2'))):
                s = sd.create(n, t, shape3[1:])
                s.setcompress(SDC.COMP_DEFLATE, value=6)
                s[:] = field(shape3[1:], 40 + i, dt)
                s.endaccess()
            sd.end()
        made.append(p)
    except ImportError:
        skipped.append(('hdf4_*', 'pyhdf not installed (pip/conda install pyhdf to cover HDF4)'))
    except Exception as e:
        skipped.append(('hdf4_sds_deflate', '%s: %s' % (type(e).__name__, e)))

    made = sorted(set(made))
    return [(p, file_format(p)) for p in made], skipped


# ---------------------------------------------------------------- baseline readers

def digest(a):
    a = np.ma.getdata(a) if np.ma.isMaskedArray(a) else np.asarray(a)
    if a.dtype.kind == 'O':
        return hashlib.sha1(repr(a.tolist()).encode()).hexdigest()[:16]
    if a.dtype.byteorder not in ('=', '|') and a.dtype.names is None:
        a = a.astype(a.dtype.newbyteorder('='))   # scipy gives netCDF3's big-endian: same values
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def read_netcdf4(path, maskandscale):
    import netCDF4
    out = {}
    with netCDF4.Dataset(path) as ds:
        def walk(g, prefix):
            for name, v in g.variables.items():
                v.set_auto_maskandscale(maskandscale)
                out[prefix + name] = v[...]
            for gname, sub in g.groups.items():
                walk(sub, prefix + gname + '/')
        walk(ds, '')
    return out


def read_h5py(path):
    import h5py
    out = {}
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        f.visititems(visit)
    return out


def read_scipy(path):
    from scipy.io import netcdf_file
    with netcdf_file(path, 'r', mmap=False, maskandscale=False) as f:
        return {k: np.array(v[:]) for k, v in f.variables.items()}


def read_xarray(path, engine):
    import xarray as xr
    kw = {'phony_dims': 'sort'} if engine == 'h5netcdf' else {}
    with xr.open_dataset(path, engine=engine, decode_times=False, cache=False, **kw) as ds:
        ds = ds.load()
        return {k: v.values for k, v in ds.variables.items()}


def readers_for(fmt):
    r = [('netCDF4-python raw', lambda p: read_netcdf4(p, False)),
         ('netCDF4-python mask+scale', lambda p: read_netcdf4(p, True))]
    if fmt.startswith('HDF5'):
        r.append(('h5py', read_h5py))
        r.append(('xarray h5netcdf', lambda p: read_xarray(p, 'h5netcdf')))
    if fmt.startswith('netCDF3'):
        r.append(('scipy.io.netcdf', read_scipy))
        r.append(('xarray scipy', lambda p: read_xarray(p, 'scipy')))
    r.append(('xarray netcdf4', lambda p: read_xarray(p, 'netcdf4')))
    if fmt == 'HDF4':
        r = []
        try:
            from pyhdf.SD import SD

            def read_pyhdf(p):
                sd = SD(p)
                out = {n: sd.select(n)[:] for n in sd.datasets()}
                sd.end()
                return out
            r.append(('pyhdf', read_pyhdf))
        except ImportError:
            pass
    return r


def bench(path, fmt, info, repeat):
    raw = sum(v.get('raw_bytes') or 0 for v in info['variables']) or os.path.getsize(path)
    res = {'path': path, 'format': fmt, 'raw_MB': raw / 1e6, 'readers': {}}
    for name, fn in readers_for(fmt):
        try:
            first, best, out = timed(lambda: fn(path), repeat)
            res['readers'][name] = {
                'first_s': first, 'best_s': best, 'MB_per_s': raw / 1e6 / best if best else None,
                'n_vars': len(out), 'digests': {k: digest(v) for k, v in out.items()}}
        except Exception as e:
            res['readers'][name] = {'error': '%s: %s' % (type(e).__name__, str(e)[:200])}
    # do the raw readers agree, variable by variable?
    raws = [r['digests'] for n, r in res['readers'].items()
            if 'digests' in r and 'mask' not in n and not n.startswith('xarray')]
    if len(raws) > 1:
        common = set.intersection(*(set(d) for d in raws))
        res['raw_readers_disagree_on'] = sorted(k for k in common if len({d[k] for d in raws}) > 1)
    return res


# ---------------------------------------------------------------- reports

def machine():
    import netCDF4
    m = {'python': platform.python_version(), 'platform': platform.platform(),
         'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
         'numpy': np.__version__, 'netCDF4': netCDF4.__version__,
         'netcdf_c': netCDF4.__netcdf4libversion__, 'hdf5_netcdf': netCDF4.__hdf5libversion__,
         'python_exe': sys.executable}
    for mod in ('h5py', 'xarray', 'h5netcdf', 'scipy', 'numba', 'pyhdf', 'hdf5plugin', 'zarr',
                'numcodecs', 'isal', 'zlib_ng', 'deflate'):
        try:
            m[mod] = __import__(mod).__version__
        except Exception:
            m[mod] = None
    try:
        import h5py
        m['hdf5_h5py'] = h5py.version.hdf5_version
    except Exception:
        pass
    return m


def md_tables(corpus, results):
    lines = ['| file | format | raw MB | reader | first s | best s | MB/s |', '|---|---|---|---|---|---|---|']
    for r in results:
        name = os.path.basename(r['path'])
        if 'synthetic' not in r['path']:
            name = '%s / %s' % (os.path.basename(os.path.dirname(os.path.dirname(r['path'])))
                                or os.path.dirname(r['path']), name)
        for rd, x in r['readers'].items():
            if 'error' in x:
                lines.append('| %s | %s | %.1f | %s | error: %s | | |' % (
                    name, r['format'], r['raw_MB'], rd, x['error'][:60].replace('|', '/')))
            else:
                lines.append('| %s | %s | %.1f | %s | %.3f | %.3f | %.0f |' % (
                    name, r['format'], r['raw_MB'], rd, x['first_s'], x['best_s'], x['MB_per_s'] or 0))
    dis = [(os.path.basename(r['path']), r['raw_readers_disagree_on'])
           for r in results if r.get('raw_readers_disagree_on')]
    if dis:
        lines += ['', 'Raw readers disagree (value digests differ) on:']
        lines += ['- %s: %s' % (f, ', '.join(v[:8])) for f, v in dis]
    return '\n'.join(lines)


def corpus_summary(corpus):
    by = Counter(c['format'] + (' (netCDF4)' if c.get('netcdf4') else '') for c in corpus)
    filt = Counter()
    lay = Counter()
    for c in corpus:
        for v in c['variables']:
            lay[v.get('layout')] += 1
            for f in v.get('filters', []):
                filt[f.split('(')[0]] += 1
    out = ['Files by format:'] + ['- %s: %d' % kv for kv in by.most_common()]
    out += ['', 'Variable layouts: ' + ', '.join('%s %d' % kv for kv in lay.most_common())]
    out += ['Filters seen: ' + (', '.join('%s %d' % kv for kv in filt.most_common()) or 'none')]
    return '\n'.join(out)


BRIEF = r"""# Build a fast netCDF / HDF reader -- session brief

Prepared {today} by `prepare_session.py` (re-run it any time: `"{python}" prepare_session.py`).
You are starting a new project in `{here}`. Read this whole brief first.

## The goal

Build a standalone Python library (working name **fastnc** -- ask the user if
they want another) that reads **netCDF3** (classic, 64-bit offset, CDF-5),
**netCDF4 / netCDF4-classic** and **plain HDF5** -- and, as stretch goals,
**HDF4** (MODIS-era NASA products), HDF-EOS and Zarr/NCZarr -- much faster
than today's readers (netCDF4-python, h5py, xarray), with **genuine** speedups:
better algorithms, less copying, parallel decompression of independent chunks.
Not "more power wins".

This repeats, for netCDF/HDF, what was done for GRIB in
`C:\Users\Roch\Desktop\Python\fastgrib` (public at github.com/Rochy111/fastgrib):
from 15 ms to 1.25 ms a GFS field on one core, bit-identical to wgrib2. Read
its `README.md`, `CONTRIBUTING.md` and `docs/internals.md` before writing
code: the conventions, the tests-against-stored-reference approach and the
hard-won rules there apply here unchanged.

## The rule that is not negotiable

**Values must be bit-identical to the reference readers.** Raw values equal
h5py / netCDF4-python with auto mask+scale off, byte for byte. Decoded values
(mask and scale on) equal netCDF4-python's: `_FillValue`, `missing_value`,
`valid_min/max/range`, `scale_factor`/`add_offset` with *their* dtype
promotion, `_Unsigned`, the masked-array vs NaN choice -- every rule, in the
same float type. A faster reader that is off by one ulp, or masks one point
differently, is a bug. `baseline.json` holds per-variable value digests from
every reader for every corpus file: they are the first tests.

## What to do, in order (as fastgrib was done)

1. **Measure first.** `BASELINE.md` (below) has today's numbers on this
   machine. Profile where the time goes before changing anything: open
   (metadata, B-tree walks), chunk I/O, decompression, shuffle/unshuffle,
   copies into the output array, mask/scale. Compare readers on the *same*
   file: when this brief was first made, xarray took 10-20x longer than
   netCDF4-python / h5py on a compressed netCDF4 wrfout (h5py 0.48 s,
   xarray netcdf4 5.7 s, xarray h5netcdf 10.7 s), while uncompressed netCDF3
   already read at ~2 GB/s -- find out why before deciding where to aim.
2. **Design the reader** -- the big decisions, each settled by measurement:
   - netCDF3 is uncompressed: map the file and view it (record variables are
     interleaved: one strided copy, byte-swapped big-endian -> native in the
     same pass). Expect near memory-bandwidth speed.
   - netCDF4/HDF5: the speedup is in **decompressing independent chunks in
     parallel** and writing each straight into its place in the output.
     `zlib.decompress` releases the GIL, so a thread pool works; libdeflate /
     isa-l / zlib-ng inflate faster than zlib -- measure. Shuffle, fletcher32,
     scale-offset, n-bit: numba kernels. Getting chunk addresses: h5py's
     `get_chunk_info` / `read_direct_chunk` is the quick start; a pure-Python
     parser of the HDF5 superblock, object headers and chunk indexes (v1/v2
     B-trees, fixed/extensible arrays, single chunk) removes the h5py
     dependency and per-chunk overhead -- decide by measuring.
   - Filters beyond deflate (szip, zstd, blosc, bzip2, lz4, lzf): support what
     the corpus has, fall back to h5py for what you don't, and say so.
3. **Optimize**, with every change checked against the digests.
4. **Tests** against stored reference outputs (digests or small arrays in
   `tests/data`), needing neither h5py nor netCDF4-python in CI where
   possible; synthetic files for every format/filter combination.
5. **numpy 1 and 2**, Python 3.9-3.13, Linux / Windows / macOS (CI matrix as
   fastgrib's `.github/workflows/tests.yml`).
6. **Docs**: README with benchmarks against netCDF4-python / h5py / xarray
   (machine, method, numbers), CONTRIBUTING, CHANGELOG, internals.
7. **An xarray backend** (`engine='fastnc'`) once reading is right: that is
   how most users would meet it.
8. **Git from the start**; a GitHub repo and releases only when the user says
   so. **Never publish to PyPI** unless the user asks.

## Lessons from fastgrib -- read these, they cost days

Measuring:
- **Hybrid CPU on Windows** (this i7-13700K, 8P+8E): a script from a terminal
  is background work and its main thread lands on E-cores; single-core
  timings swing 30-50%. Call a `fast_thread()` (in `prepare_session.py`,
  from fastgrib's `_cpu._fast_thread`) on the timing thread and the worker
  threads.
- `time.process_time()` ticks every 15.6 ms on Windows: CPU time only means
  anything over runs longer than a second.
- The tool shell sets `PYTHONDONTWRITEBYTECODE=1`: every import recompiles;
  `unset` it before timing imports or cold start.
- **Page faults are serialized on Windows** (~0.45 ms per 4 MB of fresh
  memory, on any number of threads): time `read(out=preallocated)` to see the
  decoder alone, and don't allocate large scratch per chunk on every core.
- The OS file cache: first read vs later reads differ wildly; report both
  and say which.
- Alternate A/B/A/B and look at the spread, not one run.

numba:
- `@njit(cache=True, nogil=True)`; **no `inline='always'`** (left refcounting
  in hot loops, 10x slower under numba 0.63); unsigned indices (`np.uint64`)
  in hot loops; explicit loops, not slice assignment; branch-free where data
  is unpredictable.
- **No `parallel=True` / `prange`**: numba's threads spin after each loop
  (70 ms of CPU for a 3 ms read). Use your own pool: the caller and helpers
  take items off a shared atomic counter until none is left.
- numba's cache can serve a stale kernel after an edit: delete
  `__pycache__/*.nbi *.nbc` before timing.
- **Import numba lazily** (fastgrib `_jit.py`): listing a file should not pay
  ~250 ms for numba.
- Each new process's first decode pays ~0.5 s (numba import + loading cached
  kernels): provide `warm()`, and document `ProcessPoolExecutor(initializer=
  fastnc.warm)` for worker pools.

Python and numpy:
- **numpy 2 / NEP 50**: arithmetic on numpy uint8/int16 scalars stays in that
  type and wraps (`b[p] * 100`). Read header fields as Python ints
  (`memoryview(buf).cast('B')`, `struct.unpack_from`). Run the suite on numpy
  1.26 with `np._set_promotion_state('weak_and_warn')` to find every site.
- **Fork safety**: a thread pool created before `fork()` is inherited without
  its threads -- workers hang. `os.register_at_fork(after_in_child=...)`
  resets it (fastgrib test: `test_forked_workers`).
- Python 3.9 has no `int.bit_count()`; macOS has no `/proc/meminfo`.

## Working with the user

- Communicate progress plainly: what you measured, what you changed, the
  before/after numbers, what you could not verify.
- Ask before downloading large files, installing packages into existing
  environments, creating the GitHub repo, pushing, or anything public.
- The user's data folders are read-only for this project: never write into
  them; put generated files under `{here}`.

## This machine

```
{machine}
```

Use `{python}` (conda env with netCDF4, h5py, xarray, numba).

## The corpus (`corpus.json`)

Real files sampled from: {roots}
plus generated files in `corpus/synthetic/` (every layout and filter this
machine's libraries can write).

{summary}

Not generated (missing libraries or unsupported here):
{skipped}

zstd and blosc: netCDF4-python ships their plugins (`netCDF4/plugins/`,
which this script puts on `HDF5_PLUGIN_PATH`), but in this env the C library
still reports "undefined filter" -- most likely the pip wheel's plugins and
conda's HDF5 DLLs not matching. A clean env (conda-forge `netcdf4` +
`hdf5plugin`) should cover them, and `pyhdf` adds HDF4: **ask the user before
creating or changing environments.** Both filters appear in real archives
(ERA5 on ARCO, CMIP, Zarr stores), so cover them before calling the reader
done.

## Baseline (`baseline.json`, best of {repeat} reads after a first)

Time to open the file and read every variable completely.

{tables}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='*', default=DEFAULT_ROOTS, help='folders holding real files')
    ap.add_argument('--per-dir', type=int, default=2, help='files sampled per folder (default 2)')
    ap.add_argument('--max-files', type=int, default=60, help='real files sampled at most')
    ap.add_argument('--keep-per-signature', type=int, default=1,
                    help='real files kept per distinct format/filter signature (default 1)')
    ap.add_argument('--max-bench-mb', type=float, default=2000,
                    help='skip benchmarking real files bigger than this many raw MB')
    ap.add_argument('--repeat', type=int, default=3, help='reads per reader per file (default 3)')
    ap.add_argument('--no-synthetic', action='store_true')
    args = ap.parse_args()
    fast_thread()

    print('Sampling real files (%d per folder) ...' % args.per_dir)
    t = time.perf_counter()
    real = sample_files(args.roots, args.per_dir, args.max_files)
    print('  %d files in %.1f s' % (len(real), time.perf_counter() - t))

    print('Describing them ...')
    infos = [describe(p, f) for p, f in real]
    keep, seen = [], Counter()
    for info in sorted(infos, key=lambda i: -i['bytes']):
        sig = signature(info)
        if seen[sig] < args.keep_per_signature:
            seen[sig] += 1
            keep.append(info)
    print('  %d distinct cases kept (of %d)' % (len(keep), len(infos)))

    skipped = []
    if not args.no_synthetic:
        print('Generating synthetic files ...')
        synth, skipped = make_synthetic(os.path.join(HERE, 'corpus', 'synthetic'))
        print('  %d files; %d combinations not available here' % (len(synth), len(skipped)))
        keep += [describe(p, f) for p, f in synth if f]

    corpus = [k for k in keep if 'error' not in k]
    with open(os.path.join(HERE, 'corpus.json'), 'w') as f:
        json.dump({'roots': args.roots, 'files': keep, 'skipped': skipped}, f, indent=1, default=str)

    print('Benchmarking %d files x readers (%d reads each) ...' % (len(corpus), args.repeat))
    results = []
    for i, info in enumerate(corpus):
        raw_mb = sum(v.get('raw_bytes') or 0 for v in info['variables']) / 1e6
        if raw_mb > args.max_bench_mb:
            print('  [%d/%d] skip %s (%.0f MB raw)' % (i + 1, len(corpus), info['path'], raw_mb))
            continue
        t = time.perf_counter()
        try:
            results.append(bench(info['path'], info['format'], info, args.repeat))
        except Exception:
            traceback.print_exc()
        print('  [%d/%d] %-50s %.1f s' % (i + 1, len(corpus), os.path.basename(info['path'])[:50],
                                          time.perf_counter() - t))
    with open(os.path.join(HERE, 'baseline.json'), 'w') as f:
        json.dump({'machine': machine(), 'repeat': args.repeat, 'results': results}, f, indent=1)

    tables = md_tables(corpus, results)
    with open(os.path.join(HERE, 'BASELINE.md'), 'w', encoding='utf-8') as f:
        f.write('# Baseline: today\'s readers on the corpus\n\n%s\n\n%s\n' % (corpus_summary(corpus), tables))

    m = machine()
    brief = BRIEF.format(
        today=date.today().isoformat(), here=HERE, python=sys.executable,
        machine='\n'.join('%-14s %s' % kv for kv in m.items()),
        roots=', '.join('`%s`' % r for r in args.roots), summary=corpus_summary(corpus),
        skipped='\n'.join('- %s: %s' % s for s in skipped) or '- nothing', repeat=args.repeat,
        tables=tables)
    with open(os.path.join(HERE, 'SESSION_BRIEF.md'), 'w', encoding='utf-8') as f:
        f.write(brief)
    print('\nWrote corpus.json, baseline.json, BASELINE.md, SESSION_BRIEF.md in', HERE)


if __name__ == '__main__':
    main()
