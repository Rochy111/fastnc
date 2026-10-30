"""Record what h5py and netCDF4-python read from every test file, as
digests: the references fastnc's tests compare against, so that the tests
need neither library.

    python tools/make_references.py                  # all three sets
    python tools/make_references.py --set corpus     # tests/data only
    python tools/make_references.py --set real --jobs 3
    python tools/make_references.py --only nc4.nc --print

Sets, each written to tests/data/references_<set>.json.gz:
    corpus      tests/data/*            (committed: CI's tests use it)
    synthetic   corpus/synthetic/*      (the older generated corpus, not in git)
    real        the real files beside the checkout (*.nc, wrfout_d02_...)

For every file it records, as netCDF4-python sees it: data model, groups,
dimensions, attributes (names, order, values with their types), variables
(dimensions, shape, dtype, attributes, filters(), chunking(), endian()) and
the values of every variable, raw (set_auto_maskandscale(False)) and as
read by default (mask and scale on), whole and for the standard selections
of tests/helpers.py; and as h5py sees it: every object (groups, datasets,
committed types, links), their attributes, dataset properties (chunks,
compression, fill value ...) and raw values, whole and the same
selections. Values are digests (tests/helpers.py: `digest`), so the files
stay small. A reader that fails on a file or a variable has its error
recorded instead. Raw values read by both readers are compared, and every
variable where they differ is listed under 'disagreements'.

Run it in the dev environment (h5py, netCDF4, hdf5plugin). HDF5 files the
dev environment's HDF5 cannot open (written by HDF5 2.0) are read with
h5py from --alt-python (default: the bench-wrf env, h5py on HDF5 2.0); the
reference file says which files.
"""
import argparse
import datetime
import glob
import gzip
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tests'))


def _plugin_path():
    """Filter plugins (zstd, blosc, lz4, bzip2 ...) for both readers.
    HDF5_PLUGIN_PATH points at hdf5plugin's plugins before either library
    loads; but netCDF-C (4.9+) replaces HDF5's plugin search path with its
    own when it initialises, after which h5py finds none. So hdf5plugin
    is also imported, which registers its filters with the HDF5 library
    directly -- in the conda env one HDF5 DLL serves h5py and netCDF4, so
    both then decode them."""
    spec = importlib.util.find_spec('hdf5plugin')
    if spec and spec.submodule_search_locations:
        d = os.path.join(list(spec.submodule_search_locations)[0], 'plugins')
        if os.path.isdir(d):
            os.environ['HDF5_PLUGIN_PATH'] = d
            import hdf5plugin                                 # noqa: F401  (registers the filters)
            return d
    return os.environ.get('HDF5_PLUGIN_PATH')


PLUGIN_PATH = _plugin_path()

import numpy as np                                            # noqa: E402

import helpers                                                # noqa: E402
from helpers import digest, encode, jsonable, loose_digest, selections, key  # noqa: E402

ALT_PYTHON = r'C:\Users\Roch\miniconda3\envs\bench-wrf\python.exe'
REAL_DIR = os.environ.get('FASTNC_REAL_DATA') or next(
    (d for d in helpers.search_dirs() if glob.glob(os.path.join(d, 'wrfout_d02_*'))), ROOT)
HDF5_SIG = b'\x89HDF\r\n\x1a\n'


def file_sets():
    """name -> path for each set."""
    real = sorted(p for p in glob.glob(os.path.join(REAL_DIR, '*.nc')) if '(1)' not in p)
    real += sorted(glob.glob(os.path.join(REAL_DIR, 'wrfout_d02_2024-11-06_01_30_00')))
    corpus = sorted(p for p in glob.glob(os.path.join(helpers.DATA, '*'))
                    if not os.path.basename(p).startswith('references_') and os.path.isfile(p))
    synth = sorted(glob.glob(os.path.join(REAL_DIR, 'corpus', 'synthetic', '*')))
    return {'corpus': corpus, 'synthetic': synth, 'real': real}


def magic(path):
    with open(path, 'rb') as f:
        head = f.read(8)
    if head == HDF5_SIG:
        return 'HDF5'
    if head[:3] == b'CDF':
        return {1: 'netCDF3 classic', 2: 'netCDF3 64-bit offset', 5: 'netCDF3 64-bit data'}.get(head[3], '?')
    if head[:4] == b'\x0e\x03\x13\x01':
        return 'HDF4'
    return 'unknown'


def err(e, path):
    """An error as text, without the machine's paths."""
    s = '%s: %s' % (type(e).__name__, e)
    for p in (path, path.replace('\\', '\\\\'), os.path.dirname(path)):
        s = s.replace(p, os.path.basename(path) if p != os.path.dirname(path) else '')
    return s[:400]


def guarded(fn, path):
    try:
        return fn()
    except Exception as e:                                    # a reader's failure is a result
        return {'error': err(e, path)}


# ------------------------------------------------------------ netCDF4-python

def nc_datatype(dt):
    if isinstance(dt, np.dtype) or dt is str:
        return helpers.dtype_str(dt)
    out = {'kind': type(dt).__name__, 'name': getattr(dt, 'name', None)}
    if getattr(dt, 'dtype', None) is not None:
        out['dtype'] = helpers.dtype_str(dt.dtype)
    if getattr(dt, 'enum_dict', None) is not None:
        out['enum_dict'] = jsonable(dt.enum_dict)
    return out


def nc_attrs(obj, path):
    out = []
    for n in obj.ncattrs():
        try:
            out.append([n, encode(obj.getncattr(n))])
        except Exception as e:
            out.append([n, {'t': 'error', 'error': err(e, path)}])
    return out


def nc_read(v, k, mask, path):
    """Read v[k] with mask and scale on or off; the digest, and the loose
    digest for comparing with h5py."""
    v.set_auto_maskandscale(mask)
    try:
        x = v[k]
    except Exception as e:
        return {'error': err(e, path)}, None
    return safe_digest(x, not mask)


def safe_digest(x, with_loose):
    """digest(x) (and the loose one), or a 'generator:' error saying the
    digest code could not handle the value -- a bug here, not the reader's."""
    try:
        return digest(x), (loose_digest(x) if with_loose else None)
    except Exception as e:
        print('    digest failed on a %s: %s' % (type(x).__name__, e), flush=True)
        return {'error': 'generator: %s: %s' % (type(e).__name__, e)}, None


def nc_reference(path, loose):
    import netCDF4
    try:
        ds = netCDF4.Dataset(path)
    except Exception as e:
        return {'error': err(e, path)}
    out = {}
    with ds:
        out['data_model'] = ds.data_model
        out['file_format'] = ds.file_format
        out['disk_format'] = guarded(lambda: ds.disk_format, path)
        groups, variables = {}, {}

        def walk(g):
            gp = g.path
            groups[gp] = {
                'dimensions': [[n, len(d), d.isunlimited()] for n, d in g.dimensions.items()],
                'attrs': nc_attrs(g, path),
                'variables': list(g.variables),
                'groups': list(g.groups),
            }
            for name, v in g.variables.items():
                vp = (gp.rstrip('/') + '/' + name)
                rec = {
                    'group': gp,
                    'dimensions': list(v.dimensions),
                    'shape': list(v.shape),
                    'dtype': helpers.dtype_str(v.dtype),
                    'datatype': nc_datatype(v.datatype),
                    'ndim': v.ndim,
                    'size': int(v.size),
                    'attrs': nc_attrs(v, path),
                    'filters': guarded(lambda: jsonable(v.filters()), path),
                    'chunking': guarded(lambda: jsonable(v.chunking()), path),
                    'endian': guarded(lambda: v.endian(), path),
                }
                rec['raw'], loose[(vp, 'full')] = nc_read(v, Ellipsis, False, path)
                rec['masked'], _ = nc_read(v, Ellipsis, True, path)
                sel = {}
                for sname, s in selections(v.shape).items():
                    d = {'key': s}
                    d['raw'], loose[(vp, sname)] = nc_read(v, key(s), False, path)
                    d['masked'], _ = nc_read(v, key(s), True, path)
                    sel[sname] = d
                rec['sel'] = sel
                v.set_auto_maskandscale(True)
                variables[vp] = rec
            for sub in g.groups.values():
                walk(sub)

        walk(ds)
        out['groups'] = groups
        out['variables'] = variables
    return out


# ------------------------------------------------------------ h5py

def h5_attrs(obj, path):
    out = []
    try:
        names = list(obj.attrs.keys())
    except Exception as e:
        return {'error': err(e, path)}
    for n in names:
        try:
            out.append([n, encode(obj.attrs[n])])
        except Exception as e:
            out.append([n, {'t': 'error', 'error': err(e, path)}])
    return out


def h5_read(ds, k, path):
    try:
        x = ds[k]
    except Exception as e:
        return {'error': err(e, path)}, None
    return safe_digest(x, True)


LAYOUTS = {0: 'compact', 1: 'contiguous', 2: 'chunked', 3: 'virtual'}


def h5_dataset(ds, p, path, loose):
    rec = {'kind': 'dataset'}
    rec['shape'] = None if ds.shape is None else list(ds.shape)
    rec['dtype'] = helpers.dtype_str(ds.dtype)
    rec['byteorder'] = helpers.byteorder(ds.dtype)
    md = helpers.dtype_meta(ds.dtype)
    if md:
        rec['dtype_meta'] = md
    rec['maxshape'] = guarded(lambda: jsonable(ds.maxshape), path)
    rec['chunks'] = jsonable(ds.chunks)
    rec['compression'] = ds.compression
    rec['compression_opts'] = jsonable(ds.compression_opts)
    rec['shuffle'] = ds.shuffle
    rec['fletcher32'] = ds.fletcher32
    rec['scaleoffset'] = jsonable(ds.scaleoffset)
    rec['fillvalue'] = guarded(lambda: encode(ds.fillvalue), path)
    rec['layout'] = guarded(lambda: LAYOUTS.get(ds.id.get_create_plist().get_layout(), 'other'), path)
    rec['filters'] = guarded(lambda: _filters(ds), path)
    rec['attrs'] = h5_attrs(ds, path)
    rec['raw'], loose[(p, 'full')] = h5_read(ds, (), path)
    sel = {}
    if ds.shape == ():
        d = {'key': '...'}
        d['raw'], loose[(p, 'ellipsis')] = h5_read(ds, Ellipsis, path)
        sel['ellipsis'] = d
    elif ds.shape is not None:
        for sname, s in selections(ds.shape).items():
            d = {'key': s}
            d['raw'], loose[(p, sname)] = h5_read(ds, key(s), path)
            sel[sname] = d
    rec['sel'] = sel
    return rec


def _filters(ds):
    """The filter pipeline as HDF5 states it: [id, flags, client data]."""
    pl = ds.id.get_create_plist()
    out = []
    for i in range(pl.get_nfilters()):
        fid, flags, cd, name = pl.get_filter(i)
        out.append([int(fid), int(flags), [int(c) for c in cd]])
    return out


def h5_reference(path, loose):
    import h5py
    try:
        f = h5py.File(path, 'r')
    except Exception as e:
        return {'error': err(e, path)}
    objects = {}
    seen = {}

    def walk(g, gp):
        rec = {'kind': 'group', 'attrs': h5_attrs(g, path), 'keys': list(g.keys()), 'links': []}
        objects[gp] = rec
        for name in rec['keys']:
            p = gp.rstrip('/') + '/' + name
            try:
                link = g.get(name, getlink=True)
            except Exception as e:
                rec['links'].append([name, 'error', err(e, path)])
                continue
            if isinstance(link, h5py.SoftLink):
                rec['links'].append([name, 'soft', link.path])
                continue
            if isinstance(link, h5py.ExternalLink):
                rec['links'].append([name, 'external', link.filename, link.path])
                continue
            try:
                obj = g[name]
            except Exception as e:
                objects[p] = {'kind': 'error', 'error': err(e, path)}
                continue
            if obj.id in seen:                               # a second hard link to one object
                rec['links'].append([name, 'hard', seen[obj.id]])
                continue
            seen[obj.id] = p
            if isinstance(obj, h5py.Group):
                walk(obj, p)
            elif isinstance(obj, h5py.Dataset):
                objects[p] = h5_dataset(obj, p, path, loose)
            elif isinstance(obj, h5py.Datatype):
                objects[p] = {'kind': 'datatype', 'dtype': helpers.dtype_str(obj.dtype),
                              'attrs': h5_attrs(obj, path)}
                md = helpers.dtype_meta(obj.dtype)
                if md:
                    objects[p]['dtype_meta'] = md
            else:
                objects[p] = {'kind': type(obj).__name__}

    with f:
        out = {'userblock_size': f.userblock_size, 'libver': guarded(lambda: jsonable(f.libver), path)}
        walk(f, '/')
        out['objects'] = objects
    return out


def h5_errors(h5):
    """What h5py failed on in a file's record: the file, or objects."""
    if 'error' in h5:
        return [h5['error']]
    out = []
    for p, o in h5.get('objects', {}).items():
        if o.get('kind') == 'error':
            out.append('%s: %s' % (p, o['error']))
        elif o.get('kind') == 'dataset' and 'error' in o['raw']:
            out.append('%s: %s' % (p, o['raw']['error']))
    return out


def h5_versions():
    import h5py
    return 'h5py %s, HDF5 %s, numpy %s, Python %s' % (
        h5py.__version__, h5py.version.hdf5_version, np.__version__, sys.version.split()[0])


def nc_worker(out_json, path):
    """--nc-worker: the netCDF4-python half of a file, in this interpreter."""
    rec = nc_reference(path, {})
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'netcdf4': rec}, f)


def run_worker(alt_python, mode, path):
    """Run this script's --h5py-worker / --nc-worker on `path` under
    another interpreter; its JSON, or {'worker_error': ...}."""
    tmp = os.path.join(os.environ.get('TEMP', '/tmp'), 'fastnc_ref_%d_%s.json' % (os.getpid(), mode))
    r = subprocess.run([alt_python, os.path.abspath(__file__), '--%s-worker' % mode, tmp, path],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        return {'worker_error': r.stderr[-400:]}
    with open(tmp, encoding='utf-8') as f:
        got = json.load(f)
    os.remove(tmp)
    return got


def merge_numpy1(main, alt):
    """netCDF4-python's answers can depend on numpy's major version (with
    numpy 1 the masked array of a packed int16 variable keeps an int16
    fill_value; numpy 2 makes it float32). Where the numpy 1 env's digest
    of a read differs, the fields that differ are stored under 'numpy1' in
    that digest; tests on numpy 1 use them. Returns how many differ."""
    n = 0
    if 'variables' not in main or 'variables' not in alt:
        return n
    for vp, v in main['variables'].items():
        a = alt['variables'].get(vp)
        if a is None:
            continue
        pairs = [(v[m], a[m]) for m in ('raw', 'masked')]
        for s, d in v['sel'].items():
            if s in a['sel']:
                pairs += [(d[m], a['sel'][s][m]) for m in ('raw', 'masked')]
        for mine, theirs in pairs:
            if 'error' in mine or 'error' in theirs:
                continue
            diff = {k: theirs.get(k) for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)}
            if diff:
                mine['numpy1'] = diff
                n += 1
    return n


def h5_worker(out_json, path):
    """--h5py-worker: the h5py half of a file, in this interpreter."""
    loose = {}
    rec = h5_reference(path, loose)
    rec['reader'] = h5_versions()
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'h5py': rec, 'loose': [[list(k), v] for k, v in loose.items()]}, f)


# ------------------------------------------------------------ one file, a set

def reference_file(path, alt_python):
    fmt = magic(path)
    rec = {'size': os.path.getsize(path), 'format': fmt}
    nc_loose, h5_loose = {}, {}
    t = time.perf_counter()
    rec['netcdf4'] = nc_reference(path, nc_loose)
    have_alt = alt_python and os.path.exists(alt_python)
    if have_alt and 'variables' in rec['netcdf4']:
        got = run_worker(alt_python, 'nc', path)
        if 'netcdf4' in got:
            rec['netcdf4']['numpy1_differs'] = merge_numpy1(rec['netcdf4'], got['netcdf4'])
        else:
            rec['netcdf4']['numpy1_error'] = got.get('worker_error')
    t_nc = time.perf_counter() - t
    t = time.perf_counter()
    if fmt == 'HDF5':
        h5 = h5_reference(path, h5_loose)
        h5['reader'] = h5_versions()
        bad = h5_errors(h5)
        if bad and have_alt:                                 # e.g. HDF5 2.0 files HDF5 1.14 cannot read
            got = run_worker(alt_python, 'h5py', path)
            if 'h5py' in got and len(h5_errors(got['h5py'])) < len(bad):
                first = bad[0]
                h5 = got['h5py']
                h5['note'] = 'read with the %s env: the dev env\'s h5py (%s) failed on %d objects, first: %s' % (
                    os.path.basename(os.path.dirname(alt_python)), h5_versions(), len(bad), first)
                h5_loose = {tuple(k): v for k, v in got['loose']}
        rec['h5py'] = h5
    t_h5 = time.perf_counter() - t
    rec['agreement'] = agreement(rec, nc_loose, h5_loose)
    rec['seconds'] = {'netcdf4': round(t_nc, 3), 'h5py': round(t_h5, 3)}
    return rec


def agreement(rec, nc_loose, h5_loose):
    """Compare netCDF4-python's raw reads with h5py's, variable by variable
    and selection by selection (bytes and str alike; values, shape and
    native dtype)."""
    nc, h5 = rec.get('netcdf4', {}), rec.get('h5py', {})
    if 'variables' not in nc or 'objects' not in h5:
        return {'compared': 0}
    compared, differ, missing = 0, [], []
    for vp in nc['variables']:
        hp = _resolve(h5['objects'], vp)
        o = h5['objects'].get(hp)
        if not o or o.get('kind') != 'dataset':
            missing.append(vp)
            continue
        for (p, sname), a in nc_loose.items():
            if p != vp:
                continue
            b = h5_loose.get((hp, sname))
            if a is None or b is None:
                continue
            compared += 1
            if _norm(a) != _norm(b):
                differ.append({'variable': vp, 'selection': sname,
                               'netcdf4': nc['variables'][vp]['raw'] if sname == 'full'
                               else nc['variables'][vp]['sel'][sname]['raw'],
                               'h5py': o['raw'] if sname == 'full' else o['sel'][sname]['raw']})
    return {'compared': compared, 'differ': differ, 'no_h5py_dataset': missing}


def _resolve(objects, p):
    """The h5py object path a netCDF variable path names: itself, or the
    target of a soft / second hard link (netCDF4-python follows links)."""
    if p in objects:
        return p
    parent, _, name = p.rpartition('/')
    g = objects.get(parent or '/', {})
    for link in g.get('links', []):
        if link[0] == name and link[1] in ('soft', 'hard'):
            return link[2]
    return p


def _norm(d):
    return json.dumps(d)


def _one(args):
    name, path, alt = args
    t = time.perf_counter()
    try:
        rec = reference_file(path, alt)
    except Exception:
        rec = {'error': traceback.format_exc()[-800:]}
    return name, rec, time.perf_counter() - t


def versions():
    import netCDF4
    out = {'netCDF4': 'netCDF4-python %s, netCDF-C %s, HDF5 %s' % (
        netCDF4.__version__, netCDF4.__netcdf4libversion__, netCDF4.__hdf5libversion__),
        'h5py': h5_versions(), 'HDF5_PLUGIN_PATH': 'hdf5plugin' if PLUGIN_PATH else None}
    try:
        import hdf5plugin
        out['hdf5plugin'] = hdf5plugin.version
    except ImportError:
        pass
    return out


ABOUT = """Reference digests of what h5py and netCDF4-python read; made by
tools/make_references.py, compared by tests/test_references.py. Values:
tests/helpers.py digest() of each read (sha256 of the values in native
byte order; masked arrays: mask digest, data digest with masked points
zeroed, fill value; object arrays: sha256 of a canonical JSON). netCDF4
'raw' = after set_auto_maskandscale(False) (always_mask and chartostring
at their defaults), 'masked' = netCDF4-python's defaults. Selections:
helpers.selections(shape). Attribute values: helpers.encode()."""


def make_set(which, paths, alt, jobs, only):
    items = [(os.path.basename(p), p, alt) for p in paths if not only or os.path.basename(p) in only]
    print('%s: %d files' % (which, len(items)))
    files = {}
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            for name, rec, dt in ex.map(_one, items):
                files[name] = rec
                report(name, rec, dt)
    else:
        for it in items:
            name, rec, dt = _one(it)
            files[name] = rec
            report(name, rec, dt)
    files = {k: files[k] for k in sorted(files)}
    doc = {
        'about': ABOUT,
        'set': which,
        'date': datetime.date.today().isoformat(),
        'readers': versions(),
        'alt_reader': alt_versions(alt) if alt and os.path.exists(alt) else None,
        'summary': summary(files),
        'files': files,
    }
    return doc


def alt_versions(alt):
    """The second env: its h5py reads the HDF5 files the dev env's HDF5
    cannot open (marked 'note' in their h5py record); its netCDF4-python,
    on numpy 1, gives the 'numpy1' overrides."""
    code = ('import sys, numpy, h5py, netCDF4; print("h5py %s, HDF5 %s; netCDF4-python %s, netCDF-C %s, '
            'HDF5 %s; numpy %s, Python %s" % (h5py.__version__, h5py.version.hdf5_version, netCDF4.__version__, '
            'netCDF4.__netcdf4libversion__, netCDF4.__hdf5libversion__, numpy.__version__, '
            'sys.version.split()[0]))')
    r = subprocess.run([alt, '-c', code], capture_output=True, text=True)
    return {'env': os.path.basename(os.path.dirname(alt)), 'versions': r.stdout.strip() or r.stderr[-200:]}


def summary(files):
    s = {'files': len(files), 'nc_variables': 0, 'nc_selections': 0, 'h5_datasets': 0,
         'h5_selections': 0, 'nc_errors': [], 'h5_errors': [], 'disagreements': [], 'h5py_alt_env': [],
         'numpy1_overrides': {}}
    for name, r in files.items():
        nc, h5 = r.get('netcdf4', {}), r.get('h5py', {})
        if nc.get('numpy1_differs'):
            s['numpy1_overrides'][name] = nc['numpy1_differs']
        if 'error' in r:
            s['nc_errors'].append([name, '(generator) ' + r['error'][-200:]])
        if 'error' in nc:
            s['nc_errors'].append([name, nc['error']])
        for vp, v in nc.get('variables', {}).items():
            s['nc_variables'] += 1
            s['nc_selections'] += len(v['sel'])
            for mode in ('raw', 'masked'):
                if 'error' in v[mode]:
                    s['nc_errors'].append([name + ':' + vp + ' ' + mode, v[mode]['error']])
        if 'error' in h5:
            s['h5_errors'].append([name, h5['error']])
        if 'note' in h5:
            s['h5py_alt_env'].append(name)
        for p, o in h5.get('objects', {}).items():
            if o.get('kind') == 'dataset':
                s['h5_datasets'] += 1
                s['h5_selections'] += len(o['sel'])
                if 'error' in o['raw']:
                    s['h5_errors'].append([name + ':' + p, o['raw']['error']])
            elif o.get('kind') == 'error':
                s['h5_errors'].append([name + ':' + p, o['error']])
        for d in r.get('agreement', {}).get('differ', []):
            s['disagreements'].append([name, d['variable'], d['selection']])
    return s


def report(name, rec, dt):
    nc, h5 = rec.get('netcdf4', {}), rec.get('h5py', {})
    bits = ['%-60s %6.1f s' % (name[:60], dt)]
    bits.append('nc: ' + ('ERR ' + nc['error'][:60] if 'error' in nc else '%d vars' % len(nc.get('variables', {}))))
    if h5:
        bits.append('h5: ' + ('ERR ' + h5['error'][:60] if 'error' in h5 else '%d objs' % len(h5.get('objects', {}))))
        if 'note' in h5:
            bits.append('(alt env)')
    if 'seconds' in rec:
        bits.append('(nc %.1f s incl. numpy-1 env, h5py %.1f s)' % (rec['seconds']['netcdf4'], rec['seconds']['h5py']))
    a = rec.get('agreement', {})
    if a.get('differ'):
        bits.append('DISAGREE x%d' % len(a['differ']))
    print('  ' + '  '.join(bits), flush=True)
    if 'error' in rec:
        print('  GENERATOR ERROR\n' + rec['error'], flush=True)


def write(doc, which):
    p = helpers.reference_path(which)
    raw = json.dumps(doc, indent=None, separators=(',', ':'), ensure_ascii=True).encode()
    with open(p, 'wb') as f:                                 # mtime 0: the same content gives the same bytes
        with gzip.GzipFile(filename='', mode='wb', fileobj=f, compresslevel=9, mtime=0) as g:
            g.write(raw)
    print('  -> %s (%.1f kB, %.1f kB of JSON)' % (os.path.relpath(p, ROOT), os.path.getsize(p) / 1e3,
                                                  len(raw) / 1e3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--set', choices=helpers.REFERENCE_SETS, action='append',
                    help='which set(s) to make (default: all)')
    ap.add_argument('--only', action='append', help='only these file names (debugging; not written)')
    ap.add_argument('--print', action='store_true', help='print the records instead of writing')
    ap.add_argument('--jobs', type=int, default=1, help='files read in parallel processes')
    ap.add_argument('--alt-python', default=ALT_PYTHON,
                    help='interpreter whose h5py reads HDF5 files the dev env cannot open')
    ap.add_argument('--h5py-worker', nargs=2, metavar=('OUT_JSON', 'FILE'), help=argparse.SUPPRESS)
    ap.add_argument('--nc-worker', nargs=2, metavar=('OUT_JSON', 'FILE'), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.h5py_worker:
        h5_worker(*a.h5py_worker)
        return
    if a.nc_worker:
        nc_worker(*a.nc_worker)
        return
    sets = file_sets()
    for which in a.set or helpers.REFERENCE_SETS:
        doc = make_set(which, sets[which], a.alt_python, a.jobs, a.only)
        s = doc['summary']
        print('  %d files, %d netCDF variables (%d selections), %d h5py datasets (%d selections)' % (
            s['files'], s['nc_variables'], s['nc_selections'], s['h5_datasets'], s['h5_selections']))
        for e in s['disagreements']:
            print('  DISAGREE', e)
        if a.print:
            print(json.dumps(doc, indent=1)[:200000])
        elif a.only:
            print('  (--only: not written)')
        elif doc['files']:
            write(doc, which)


if __name__ == '__main__':
    main()
