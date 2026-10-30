"""The machine fastnc runs on, looked at once a process and lazily -- and
the numbers that follow from it, in place of numbers tuned to one desktop
(fastplot's design).

    _hw.cpus()         logical CPUs this process may run on
    _hw.topology()     cores: physical, the fastest class's, SMT
    _hw.memory()       (total, available) bytes this process may use
    _hw.l2_bytes()     the second-level cache of the fastest core
    _hw.ram_scale()    a factor (<= 1) for memory held between reads
    _hw.block(n)       n elements a thread takes, sized to the cache
    _hw.tuned()        what fastnc.tune() measured on this machine
    _hw.info()         all of it (fastnc.hardware())

CPUs: what the process may use, not what the box has -- its affinity and,
on Linux, a container's CPU quota. Memory: what is available now, and a
container's limit. Budgets that were sized on a desktop with RAM to spare
(the readahead window, the chunk cache, scratch kept between reads) scale
down with ram_scale() on a smaller machine and never up.

What can only be found by running is measured: the time numba and the
kernels take to load is noted whenever they load (every machine then
decides for itself when loading pays), and fastnc.tune() times the rest,
kept in fastnc's cache directory keyed by the CPU, the CPUs allowed, the
GPU and fastnc's version: another machine (or container size) never reads
another's numbers.
"""
import math
import os
import struct
import sys
import threading

REF_RAM = 16 << 30              # available RAM at which the budgets are as written
MIN_RAM_SCALE = 1 / 32          # and the least they shrink to
REF_L2 = 2 << 20                # the L2 the block sizes were measured with
TUNE_FORMAT = 1                 # tune.json's layout: another is ignored

_lock = threading.RLock()
_found = {}                     # name -> value, each looked at once


def _once(fn):
    def get():
        key = fn.__name__
        if key not in _found:
            with _lock:
                if key not in _found:
                    try:
                        _found[key] = fn()
                    except Exception:                   # unknown, then
                        _found[key] = None
        return _found[key]
    get.__name__, get.__doc__ = fn.__name__, fn.__doc__
    return get


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _size(s):
    """'2048K', '32M' -> bytes."""
    s = s.strip().upper()
    mult = {'K': 1 << 10, 'M': 1 << 20, 'G': 1 << 30}.get(s[-1:])
    return int(s[:-1]) * mult if mult else int(s)


def _cpu_list(s):
    """'0-3,8,10-11' -> {0, 1, 2, 3, 8, 10, 11}."""
    out = set()
    for part in (s or '').split(','):
        part = part.strip()
        if part:
            a, _, b = part.partition('-')
            out.update(range(int(a), int(b or a) + 1))
    return out


# ------------------------------------------------------------------ CPUs

def _quota():
    """A Linux container's CPU quota, in CPUs (None: none set)."""
    v = _read('/sys/fs/cgroup/cpu.max')                 # cgroup v2: "quota period" | "max period"
    if v:
        q, _, p = v.partition(' ')
        return int(q) / int(p) if q != 'max' and p else None
    q = _read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')    # cgroup v1
    p = _read('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if q and p and int(q) > 0 and int(p) > 0:
        return int(q) / int(p)
    return None


@_once
def _allowed():
    """The logical CPUs (their numbers) this process may run on, or None."""
    if hasattr(os, 'sched_getaffinity'):
        return set(os.sched_getaffinity(0))
    if sys.platform == 'win32':
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        proc, sysm = ctypes.c_size_t(), ctypes.c_size_t()
        if k32.GetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()),
                                      ctypes.byref(proc), ctypes.byref(sysm)):
            mask = proc.value
            if mask and (os.cpu_count() or 1) <= 64:    # one processor group: the mask says it all
                return {i for i in range(64) if mask >> i & 1}
    return None


@_once
def _cpus():
    n = None
    if hasattr(os, 'process_cpu_count'):                # Python 3.13+
        n = os.process_cpu_count()
    if n is None:
        a = _allowed()
        n = len(a) if a else os.cpu_count() or 1
    if sys.platform.startswith('linux'):
        q = _quota()
        if q:
            n = min(n, max(1, math.ceil(q)))
    return max(1, int(n))


def cpus():
    """Logical CPUs this process may use: its affinity, a container's quota."""
    return _cpus() or os.cpu_count() or 1


# -------------------------------------------------------------- topology

def _win_relations(rel):
    """The records of GetLogicalProcessorInformationEx for relation rel."""
    import ctypes
    k32 = ctypes.windll.kernel32
    size = ctypes.c_uint32(0)
    k32.GetLogicalProcessorInformationEx(rel, None, ctypes.byref(size))
    if not size.value:
        return []
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(rel, buf, ctypes.byref(size)):
        return []
    raw, off, out = buf.raw[:size.value], 0, []
    while off + 8 <= len(raw):
        r, rsize = struct.unpack_from('<II', raw, off)
        if rsize <= 0:
            break
        if r == rel:
            out.append(raw[off:off + rsize])
        off += rsize
    return out


def _win_topology():
    import ctypes
    word = ctypes.sizeof(ctypes.c_void_p)               # KAFFINITY
    allowed = _allowed()
    cores = []
    for rec in _win_relations(0):                       # RelationProcessorCore
        # PROCESSOR_RELATIONSHIP: Flags, EfficiencyClass, Reserved[20], GroupCount,
        # then GroupCount GROUP_AFFINITY (Mask, Group, Reserved[3])
        eff = rec[9]
        ngroups = struct.unpack_from('<H', rec, 8 + 22)[0]
        first = 8 + 24
        step = word + 8
        ids = []
        for g in range(ngroups):
            mask = int.from_bytes(rec[first + g * step:first + g * step + word], 'little')
            group = struct.unpack_from('<H', rec, first + g * step + word)[0]
            ids += [64 * group + i for i in range(8 * word) if mask >> i & 1]
        if allowed is not None:
            ids = [i for i in ids if i in allowed]
        if ids:
            cores.append((eff, len(ids)))
    if not cores:
        return None
    top = max(e for e, _ in cores)
    perf = [n for e, n in cores if e == top]
    return dict(cores=len(cores), logical=sum(n for _, n in cores), perf_cores=len(perf),
                perf_logical=sum(perf), hybrid=len(perf) != len(cores))


def _linux_topology():
    allowed = _allowed() or set(range(os.cpu_count() or 1))
    base = '/sys/devices/system/cpu/cpu%d/'
    core_of, cap = {}, {}
    for c in sorted(allowed):
        cid = _read(base % c + 'topology/core_id')
        if cid is None:
            continue
        pkg = _read(base % c + 'topology/physical_package_id') or '0'
        core_of[c] = (pkg, cid)
        v = _read(base % c + 'cpu_capacity')            # big.LITTLE (ARM)
        if v:
            cap[c] = int(v)
    if not core_of:
        return None
    perf = set(core_of)
    pc = _read('/sys/devices/cpu_core/cpus')            # Intel hybrid: the performance cores
    if pc and _read('/sys/devices/cpu_atom/cpus'):
        perf &= _cpu_list(pc)
    elif cap and len(set(cap.values())) > 1:
        top = max(cap.values())
        perf = {c for c in perf if cap.get(c) == top}
    if not perf:
        perf = set(core_of)
    cores = set(core_of.values())
    pcores = {core_of[c] for c in perf}
    return dict(cores=len(cores), logical=len(core_of), perf_cores=len(pcores),
                perf_logical=len(perf), hybrid=len(pcores) != len(cores))


def _sysctl_int(name):
    import ctypes
    import ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library('c'))
    v, n = ctypes.c_int64(0), ctypes.c_size_t(8)
    if libc.sysctlbyname(name.encode(), ctypes.byref(v), ctypes.byref(n), None, 0):
        return None
    return v.value


def _mac_topology():
    cores, logical = _sysctl_int('hw.physicalcpu'), _sysctl_int('hw.logicalcpu')
    if not cores:
        return None
    pc = _sysctl_int('hw.perflevel0.physicalcpu') or cores     # Apple silicon: the performance cores
    pl = _sysctl_int('hw.perflevel0.logicalcpu') or logical or cores
    return dict(cores=cores, logical=logical or cores, perf_cores=pc, perf_logical=pl,
                hybrid=pc != cores)


@_once
def topology():
    """{'cores', 'logical', 'perf_cores', 'perf_logical', 'hybrid'} of the
    CPUs this process may use (every CPU a core when it cannot be found)."""
    t = None
    try:
        if sys.platform == 'win32':
            t = _win_topology()
        elif sys.platform.startswith('linux'):
            t = _linux_topology()
        elif sys.platform == 'darwin':
            t = _mac_topology()
    except Exception:
        t = None
    n = cpus()
    if t is None:
        t = dict(cores=n, logical=n, perf_cores=n, perf_logical=n, hybrid=False)
    if t['perf_cores'] > n:                             # a quota below the cores: the quota
        t = dict(t, perf_cores=n, perf_logical=min(t['perf_logical'], n))
    return t


# ---------------------------------------------------------------- memory

def _cgroup_limit():
    """(limit, used) of a Linux container's memory, or None."""
    v = _read('/sys/fs/cgroup/memory.max')              # v2
    used = _read('/sys/fs/cgroup/memory.current')
    if v is None:
        v = _read('/sys/fs/cgroup/memory/memory.limit_in_bytes')    # v1
        used = _read('/sys/fs/cgroup/memory/memory.usage_in_bytes')
    if not v or v == 'max' or int(v) >= 1 << 60:
        return None
    return int(v), int(used or 0)


def memory():
    """(total, available) bytes of RAM for this process, looked at now
    (a container's limit counted); (None, None) when it cannot be found."""
    try:
        if sys.platform == 'win32':
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [('len', ctypes.c_ulong), ('load', ctypes.c_ulong),
                            ('total', ctypes.c_ulonglong), ('avail', ctypes.c_ulonglong),
                            ('tp', ctypes.c_ulonglong), ('ap', ctypes.c_ulonglong),
                            ('tv', ctypes.c_ulonglong), ('av', ctypes.c_ulonglong),
                            ('ext', ctypes.c_ulonglong)]
            m = MS()
            m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.total), int(m.avail)
        total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        avail = None
        if sys.platform.startswith('linux'):
            for line in (_read('/proc/meminfo') or '').splitlines():
                if line.startswith('MemAvailable:'):
                    avail = int(line.split()[1]) << 10
            c = _cgroup_limit()
            if c:
                total = min(total, c[0])
                avail = min(avail if avail is not None else total, max(0, c[0] - c[1]))
        elif sys.platform == 'darwin':
            avail = _sysctl_int('hw.memsize_usable')
        return total, avail if avail is not None else total // 2
    except Exception:
        return None, None


@_once
def ram_scale():
    """A factor for what fastnc keeps between reads: 1 with REF_RAM or more
    available (the numbers as measured), less on a smaller machine, in
    proportion, never below MIN_RAM_SCALE. Looked at once a process."""
    avail = memory()[1]
    if not avail:
        return 1.0
    return max(MIN_RAM_SCALE, min(1.0, avail / REF_RAM))


def scaled(nbytes, least=1 << 20):
    """nbytes of a budget on this machine (at least `least`)."""
    return max(min(least, nbytes), int(nbytes * (ram_scale() or 1.0)))


# ----------------------------------------------------------------- cache

def _win_l2():
    best = 0
    for rec in _win_relations(2):                       # RelationCache
        # CACHE_RELATIONSHIP: Level, Associativity, LineSize, CacheSize, Type
        level = rec[8]
        size = struct.unpack_from('<I', rec, 12)[0]
        if level == 2:
            best = max(best, size)
    return best or None


def _linux_l2():
    base = '/sys/devices/system/cpu/cpu0/cache/'
    for i in range(8):
        if _read(base + 'index%d/level' % i) == '2':
            s = _read(base + 'index%d/size' % i)
            if s:
                return _size(s)
    return None


@_once
def l2_bytes():
    """The L2 cache of the CPU's first (on hybrids, largest) core; REF_L2
    when it cannot be found. A cache shared by a cluster counts in full:
    what matters here is the order of magnitude."""
    n = None
    if sys.platform == 'win32':
        n = _win_l2()
    elif sys.platform.startswith('linux'):
        n = _linux_l2()
    elif sys.platform == 'darwin':
        n = _sysctl_int('hw.perflevel0.l2cachesize') or _sysctl_int('hw.l2cachesize')
    return n or REF_L2


def block(n, least=1 << 10):
    """A block of n elements (measured with a 2 MB L2) for this machine's
    L2: smaller where the cache is, never bigger -- a whole multiple of
    64 and of `least`, so that SIMD loops end where they did."""
    b = max(least, int(n * min(1.0, (l2_bytes() or REF_L2) / REF_L2)))
    return b // 64 * 64


# ------------------------------------------------------------- the tuning

def cpu_name():
    """The CPU's model, as the OS names it ('' when it will not say)."""
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
                return str(winreg.QueryValueEx(k, 'ProcessorNameString')[0]).strip()
        except OSError:
            pass
    elif sys.platform.startswith('linux'):
        for line in (_read('/proc/cpuinfo') or '').splitlines():
            if line.lower().startswith(('model name', 'hardware', 'cpu model')):
                return line.partition(':')[2].strip()
    elif sys.platform == 'darwin':
        import subprocess
        try:
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:
            pass
    import platform
    return platform.processor() or ''


def cache_dir():
    """fastnc's cache directory (FASTNC_CACHE_DIR, else the OS's)."""
    d = os.environ.get('FASTNC_CACHE_DIR')
    if d:
        return d
    if sys.platform == 'win32':
        root = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~\\AppData\\Local')
    elif sys.platform == 'darwin':
        root = os.path.expanduser('~/Library/Caches')
    else:
        root = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(root, 'fastnc')


def _tune_path():
    return os.path.join(cache_dir(), 'tune.json')


def machine_key(gpu_name=''):
    """What a measurement is good for: this CPU, the CPUs allowed, the GPU,
    fastnc's minor version."""
    from . import __version__
    return '|'.join([cpu_name(), str(cpus()), gpu_name or '', '.'.join(__version__.split('.')[:2])])


@_once
def _tune_file():
    import json
    try:
        with open(_tune_path()) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return d if d.get('format') == TUNE_FORMAT else {}


def tuned():
    """What has been measured on this machine ({} when nothing has)."""
    d = _tune_file() or {}
    return d.get('cpu', {}) if d.get('cpu_key') == machine_key() else {}


def tuned_gpu(gpu_name):
    """tune()'s results for this GPU on this machine ({}: none)."""
    g = (_tune_file() or {}).get('gpu') or {}
    return g if g.get('key') == machine_key(gpu_name) else {}


def save_tuned(cpu=None, gpu=None):
    """Keep measurements (merged into what is there; atomically: a reader
    sees the old or the new). The path, or None if it cannot be written."""
    import json
    path = _tune_path()
    d = {'format': TUNE_FORMAT}
    try:
        with open(path) as f:
            old = json.load(f)
        if old.get('format') == TUNE_FORMAT:
            d.update(old)
    except (OSError, ValueError):
        pass
    if cpu is not None:
        had = d.get('cpu', {}) if d.get('cpu_key') == machine_key() else {}
        d['cpu_key'], d['cpu'] = machine_key(), dict(had, **cpu)
    if gpu is not None:
        d['gpu'] = gpu                                  # its own 'key': machine_key(its name)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = '%s.%d.tmp' % (path, os.getpid())
        with open(tmp, 'w') as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)
    except OSError:
        return None
    with _lock:
        _found.pop('_tune_file', None)
    return path


def info():
    """What was found on this machine, and what fastnc does with it."""
    from . import _pool
    total, avail = memory()
    return dict(cpu=cpu_name(), cpus_allowed=cpus(), cpus_total=os.cpu_count(),
                topology=topology(), memory_total=total, memory_available=avail,
                l2_bytes=l2_bytes(), threads=_pool.threads(), ram_scale=ram_scale(),
                tuned=tuned(), tune_file=_tune_path() if _tune_file() else None)
