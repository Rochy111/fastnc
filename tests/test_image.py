"""The kernels without numba: the image's linker on made-up objects, its
choices of unit, and (where this machine can have one) a process that reads
through an image without numba ever being imported."""
import os
import struct
import subprocess
import sys
import time
import zlib

import numpy as np
import pytest

from fastnc import _image, _jit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_kernel_files_are_named():
    """A module with kernels left out of KERNEL_FILES would keep running an
    old image and an old numba cache after an edit."""
    here = os.path.join(ROOT, 'fastnc')
    found = set()
    for n in os.listdir(here):
        if n.endswith('.py') and n not in ('_jit.py', '_image.py'):
            with open(os.path.join(here, n), encoding='utf-8') as f:
                text = f.read()
            if '\n@njit' in text or '\n@intrinsic' in text or '\n    @njit' in text:
                found.add(n[:-3])
    assert found == set(_jit.KERNEL_FILES)


def elf(sections, symbols, relocations):
    """A relocatable ELF64 x86-64 object. sections: (name, type, flags,
    align, bytes); symbols: (name, info, section, value); relocations:
    (section it applies to, offset, type, symbol, addend)."""
    names = b'\0'
    strs = b'\0'
    secs = [(0, 0, 0, 1, b'', 0, 0, 0)]             # (name, type, flags, align, data, link, info, ent)
    index = {}

    def name(s):
        nonlocal names
        at = len(names)
        names += s.encode() + b'\0'
        return at
    for sn, ty, fl, al, data in sections:
        index[sn] = len(secs)
        secs.append((name(sn), ty, fl, al, data, 0, 0, 0))
    syms = bytes(24)
    for sn, info, sec, value in symbols:
        syms += struct.pack('<IBBHQQ', len(strs), info, 0, index.get(sec, 0), value, 0)
        strs += sn.encode() + b'\0'
    strtab = len(secs)
    secs.append((name('.strtab'), 3, 0, 1, strs, 0, 0, 0))
    symtab = len(secs)
    secs.append((name('.symtab'), 2, 0, 8, syms, strtab, 1, 24))
    for target in sorted({r[0] for r in relocations}):
        data = b''.join(struct.pack('<QQq', off, (sym + 1) << 32 | ty, add)
                        for sec, off, ty, sym, add in relocations if sec == target)
        secs.append((name('.rela' + target), 4, 0x40, 8, data, symtab, index[target], 24))
    shstr = len(secs)
    secs.append((name('.shstrtab'), 3, 0, 1, b'', 0, 0, 0))
    secs[shstr] = secs[shstr][:4] + (names,) + secs[shstr][5:]
    pos = 64
    table = b''
    body = b''
    for nm, ty, fl, al, data, link, info, ent in secs:
        table += struct.pack('<IIQQQQIIQQ', nm, ty, fl, 0, pos + len(body), len(data), link, info, al, ent)
        body += data
    head = b'\x7fELF\x02\x01\x01' + bytes(9) + struct.pack('<HHIQQQIHHHHHH', 1, 62, 1, 0, 0, 64 + len(body), 0,
                                                           64, 0, 0, 64, len(secs), shstr)
    return head + body + table


CODE, DATA = 0x6, 0x3                               # flags: alloc + exec, alloc + write


def test_link_places_sections_and_fixes_addresses():
    text = b'\x90' * 32
    rodata = b'R' * 16
    obj = elf([('.ltext', 1, CODE, 16, text), ('.lrodata', 1, 2, 16, rodata), ('.ldata', 1, DATA, 8, b'D' * 8),
               ('.eh_frame', 1, 2, 8, b'x' * 8)],
              [('.ltext', 3, '.ltext', 0), ('fn', 0x12, '.ltext', 8), ('far', 0x10, '', 0)],
              [('.ltext', 8, 1, 2, 4),              # a 64-bit address of far (external) + 4
               ('.ltext', 16, 1, 1, 8),             # of fn + 8
               ('.ltext', 24, 24, 1, 0)])           # pc-relative to fn
    code, rel, ext, rodata_end, defined = _image._link([obj])
    assert code[:8] == text[:8]
    assert ext == [(8, 'far', 4)]
    assert rel == [16]                              # the base goes in here at load
    assert struct.unpack_from('<q', code, 16)[0] == defined['fn'] + 8
    assert struct.unpack_from('<q', code, 24)[0] == defined['fn'] - 24
    assert b'x' * 8 not in code                     # (unwind tables: no use without an unwinder)
    assert code[rodata_end:rodata_end + 8] == b'D' * 8
    assert rodata_end % _image.PAGE == 0 and len(code) % _image.PAGE == 0


def test_link_resolves_between_objects():
    a = elf([('.ltext', 1, CODE, 16, bytes(16))],
            [('.ltext', 3, '.ltext', 0), ('entry', 0x12, '.ltext', 0), ('kernel', 0x10, '', 0)],
            [('.ltext', 0, 1, 2, 0)])
    b = elf([('.ltext', 1, CODE, 16, bytes(16))],
            [('.ltext', 3, '.ltext', 0), ('kernel', 0x12, '.ltext', 4)], [])
    code, rel, ext, _, defined = _image._link([a, b])
    assert not ext and rel == [defined['entry']]
    assert struct.unpack_from('<q', code, defined['entry'])[0] == defined['kernel']


@pytest.mark.parametrize('kind, why', [(0x400, 'thread-local'), (None, 'relocation type')])
def test_link_refuses_what_it_cannot_do(kind, why):
    if kind:
        obj = elf([('.tdata', 1, DATA | kind, 8, bytes(8))], [], [])
    else:
        obj = elf([('.ltext', 1, CODE, 16, bytes(16))], [('.ltext', 3, '.ltext', 0)], [('.ltext', 0, 11, 0, 0)])
    with pytest.raises(_image._Unlinkable, match=why):
        _image._link([obj])
    with pytest.raises(_image._Unlinkable):
        _image._link([b'not an object at all'.ljust(64, b'\0')])


@pytest.mark.parametrize('param, arg, ok', [
    (('u1', 1, 'C', True), ('u1', 1, 'C', False), True),    # a writable array for a read-only parameter
    (('u1', 1, 'C', False), ('u1', 1, 'C', True), False),   # not the reverse
    (('i8', 2, 'A', False), ('i8', 2, 'C', False), True),   # any layout takes a contiguous one
    (('i8', 2, 'C', False), ('i8', 2, 'A', False), False),
    (('i8', 1, 'C', False), ('i8', 1, 'A', False), False),  # a strided one is no contiguous one
    (('i8', 1, 'C', False), ('i8', 1, 'F', False), True),
    (('i8', 2, 'C', False), ('i8', 1, 'C', False), False),
    (('i8', 1, 'C', False), ('f8', 1, 'C', False), False),
    ('i8', 'i4', True), ('i4', 'i8', False), ('i8', 'u4', True), ('u8', 'i8', False), ('i4', 'u4', False),
    (('T', 2, 'u1', 1, 'C', True), ('T', 2, 'u1', 1, 'C', False), True),
    (('T', 3, 'u1', 1, 'C', True), ('T', 2, 'u1', 1, 'C', False), False),
])
def test_unit_for_arguments(param, arg, ok):
    assert _image._takes(param, arg) is ok


def test_flatten_is_numbas_form():
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    slots, key = [], []
    assert _image._flatten((a,), slots, key)
    assert slots == [0, 0, 6, 4, a.ctypes.data, 2, 3, 12, 4]
    assert key == [('i4', 2, 'C', False)]
    slots, key = [], []
    assert _image._flatten((a[:, ::2],), slots, key)
    assert slots[5:] == [2, 2, 12, 8]
    assert key == [('i4', 2, 'A', False)]
    slots, key = [], []
    assert _image._flatten((-1, True, None, np.uint8(7), (1, 2)), slots, key)
    assert slots == [2**64 - 1, 1, 0, 7, 1, 2]
    assert key == ['i8', '?', 'n', 'u1', ('t', 2)]
    assert not _image._flatten((1.5,), [], [])


def test_direct_entry_takes_only_what_it_should():
    """The unit's own code reads each array: whatever is not what it takes is
    turned down and goes the long way, which refuses it."""
    from fastnc import _chunks
    if not _chunks.IMAGE or _image.current() is None:
        pytest.skip('no image here')
    im = _image.open_image()
    k = next(x for x in _jit.every_kernel() if x.__name__ == 'holds_value')
    if im is None or k.real is not None:
        pytest.skip('kernels from numba in this process')
    w, low = np.uint64(0x0101010101010101), np.uint64(0x7F7F7F7F7F7F7F7F)
    no, yes = np.zeros(64, np.uint8), np.ones(64, np.uint8)
    k.fast.clear()
    for _ in range(3):                              # (the first goes the long way)
        assert im.call(k, (no, 0, 64, w, w, low)) is False
        assert im.call(k, (yes, 0, 64, w, w, low)) is True
    assert k.fast and all(u.enter is not None for u, _ in k.fast.values())
    assert im.call(k, (np.zeros(32, np.uint16), 0, 64, w, w, low)) is _image.NO
    assert im.call(k, (np.zeros(32, '>u2'), 0, 64, w, w, low)) is _image.NO
    assert im.call(k, (np.zeros((8, 8), np.uint8), 0, 64, w, w, low)) is _image.NO
    assert im.call(k, (np.zeros(128, np.uint8)[::2], 0, 64, w, w, low)) is _image.NO
    assert im.call(k, (yes, 0, 64, w, w, low)) is True     # (and it still works)
    u8 = np.frombuffer(bytes(64), np.uint8)
    # (read-only: its own unit's answer if one was compiled, else turned down)
    assert im.call(k, (u8, 0, 64, w, w, low)) in (False, _image.NO)


def test_arrays_of_any_class_go_by_address(tmp_path):
    """A memmap went to the unit as int() of its one element: an address."""
    from fastnc import _chunks
    if not _chunks.IMAGE or _image.current() is None:
        pytest.skip('no image here')
    im = _image.open_image()
    k = next(x for x in _jit.every_kernel() if x.__name__ == 'holds_value')
    if im is None or k.real is not None:
        pytest.skip('kernels from numba in this process')
    w, low = np.uint64(0x0101010101010101), np.uint64(0x7F7F7F7F7F7F7F7F)

    class Mine(np.ndarray):
        pass
    for n, fill in ((1, 1), (1, 0), (64, 1), (64, 0)):
        p = tmp_path / ('m%d%d' % (n, fill))
        p.write_bytes(bytes([fill]) * n)
        m = np.memmap(p, np.uint8, 'r+', shape=(n,))
        plain = np.array(m)
        want = im.call(k, (plain, 0, n, w, w, low))
        for _ in range(3):                          # (the first goes the long way)
            assert im.call(k, (m, 0, n, w, w, low)) == want
            assert im.call(k, (plain.view(Mine), 0, n, w, w, low)) == want
        del m
    for kinds, (u, words) in k.fast.items():
        assert all(issubclass(kinds[i], int) or issubclass(kinds[i], np.integer) for i in words)


def test_flatten_refuses_what_numba_would_not_take():
    """'>f4' and '<f4' have one key: a swapped array went as a native one."""
    assert not _image._flatten((np.zeros(4, '>f4'),), [], [])
    assert not _image._flatten((np.zeros(17, np.uint8)[1:].view(np.float32),), [], [])
    assert not _image._flatten(((np.zeros(4), np.zeros(4, '>f8')),), [], [])
    assert _image._flatten((np.zeros(4, '<f4'),), [], [])


def test_numba_s_helpers_are_never_linked_as_nothing():
    """Every one was dropped, its place left 0: a unit that called it crashed."""
    ext = [(0, 'numba_do_raise', 0), (8, 'PyErr_Clear', 0), (16, 'NRT_incref', 0), (24, 'numba_cpow', 0),
           (32, 'memcpy', 0), (40, '_ZN5numba3fooE', 0)]
    kept = [n for _, n, _ in _image._usable(ext)]
    assert kept == ['numba_cpow', 'memcpy', '_ZN5numba3fooE']
    assert _image._found('memcpy')
    assert not _image._found('numba_cpow') and not _image._found('_ZN5numba3fooE')


def test_a_kernel_s_error_is_numba_s():
    """numba leaves a pickled (class, arguments, place) for what a kernel raised."""
    import ctypes
    import pickle
    from array import array
    data = pickle.dumps((MemoryError, ('Allocation failed',), None))
    held = ctypes.create_string_buffer(data, len(data))
    pair = ctypes.create_string_buffer(struct.pack('<QI', ctypes.addressof(held), len(data)), 16)
    buf = array('Q', [0] * (3 + _image.RET_WORDS + _image.EXC_WORDS))
    buf[3 + _image.RET_WORDS] = ctypes.addressof(pair)

    def k():
        pass
    e = _image._raised(k, 1, buf, 3)
    assert type(e) is MemoryError and e.args == ('Allocation failed',)
    buf[3 + _image.RET_WORDS] = 0
    assert type(_image._raised(k, 1, buf, 3)) is RuntimeError


def test_the_runtime_has_what_kernels_may_call():
    """Growing buffers were not there: kernels that need them were not linked."""
    pytest.importorskip('numba')
    from numba.core.registry import cpu_target
    e = _image._Elf(_image._runtime_object(cpu_target.target_context.codegen()))
    names = {n for n, bind, _, shndx, _, _ in e.sym if bind and shndx}
    assert {'NRT_MemInfo_alloc_safe_aligned', 'NRT_MemInfo_new', 'NRT_MemInfo_new_varsize',
            'NRT_MemInfo_varsize_alloc', 'NRT_MemInfo_varsize_realloc', 'NRT_MemInfo_varsize_free',
            'NRT_MemInfo_call_dtor', 'NRT_Allocate', 'NRT_Reallocate', 'NRT_Free'} <= names


@pytest.mark.skipif(not hasattr(os, 'pread'), reason='no fork without pread')
def test_reading_a_unit_moves_no_shared_position(tmp_path):
    """A forked child shares the file's position with its parent."""
    p = tmp_path / 'x'
    p.write_bytes(bytes(range(256)) * 4)

    class Im(_image.Image):
        def __init__(self):
            self.file = open(p, 'rb')
    im = Im()
    im.file.seek(5)
    assert im._read(100, 4) == bytes([100, 101, 102, 103])
    assert im.file.tell() == 5
    im.file.close()


def test_direct_entry_reads_this_numpy():
    assert _image._direct_works()
    assert _image._tuple_items() is not None


def test_cpu_id_is_stable():
    cpu = _image._cpu_id()
    if cpu is None:
        pytest.skip('no image on this machine')
    assert cpu == _image._cpu_id() and len(cpu) == 8


def test_image_refused_when_damaged(tmp_path):
    p = tmp_path / 'x.img'
    p.write_bytes(_image.HEAD.pack(_image.MAGIC, _image.FORMAT, 0, 999, 100, 4))
    with pytest.raises(ValueError):
        _image.Image(str(p))
    p.write_bytes(b'something else entirely, long enough for the header....')
    with pytest.raises((ValueError, struct.error)):
        _image.Image(str(p))
    tail = __import__('marshal').dumps(dict(stamp=0, externals=[], kernels={}))
    at = _image.HEAD.size
    for crc in (zlib.crc32(tail) ^ 1, zlib.crc32(tail)):    # (an index damaged: refused before it is read)
        p.write_bytes(_image.HEAD.pack(_image.MAGIC, _image.FORMAT, crc, at + len(tail), at, len(tail)) + tail)
        if crc == zlib.crc32(tail):
            _image.Image(str(p)).file.close()
        else:
            with pytest.raises(ValueError, match='index'):
                _image.Image(str(p))


def test_publish_while_the_old_image_is_open(tmp_path):
    """Processes reading through an image must not keep the next one out
    (Windows refuses to replace an open file), and go on with theirs."""
    target = str(tmp_path / 'kernels-cpu-py3.12-np2-aaaa.img')
    old, new = target + '.1.1', target + '.2.1'
    for name in (old, new):
        with open(name, 'wb') as f:
            f.write(name.encode())
    other = str(tmp_path / 'kernels-cpu-py3.12-np2-bbbb.img.1.1')       # (other sources: not wanted)
    open(other, 'w').close()
    open(target, 'w').close()
    held = open(old, 'rb')
    try:
        _image._publish(new, target)
        with open(target) as f:
            assert os.path.join(str(tmp_path), f.read()) == new
        held.seek(0)
        assert held.read() == old.encode()
        assert os.path.exists(new) and not os.path.exists(other)
    finally:
        held.close()


def test_publish_elsewhere_leaves_the_folder_alone(tmp_path):
    """Tidying is for the folder of images: an image built to a name of
    one's own once took every file beside it."""
    target = str(tmp_path / 'try.img')
    new = target + '.1.1'
    neighbour = tmp_path / 'notes.txt'
    neighbour.write_text('mine')
    open(new, 'w').close()
    _image._publish(new, target)
    assert neighbour.read_text() == 'mine' and os.path.exists(new)


def test_each_overload_is_linked_from_its_own_object():
    """numba names a library after its kernel, overloads alike; an image built
    from whichever loaded last has calls to nothing. Whole, this is what
    check.py's image step sees."""
    pytest.importorskip('numba')
    if _image.current() is None:
        pytest.skip('no image of these sources here')
    im = _image.Image(_image.current())
    try:
        assert not im.left
        assert not [n for n in im.externals if n.startswith('_ZN')]
    finally:
        im.file.close()


@pytest.fixture
def starts(monkeypatch, tmp_path):
    """What build_later would start, nothing being started."""
    if _image.cpu_id() is None:
        pytest.skip('no image on this machine')
    got = []

    class Popen:
        def __init__(self, cmd, **kw):
            got.append(kw)
    monkeypatch.setattr(subprocess, 'Popen', Popen)
    monkeypatch.setenv('FASTNC_CACHE_DIR', str(tmp_path))
    monkeypatch.delenv('FASTNC_IMAGE_BUILD', raising=False)
    monkeypatch.delenv('FASTNC_IMAGE', raising=False)
    monkeypatch.setattr(_image, '_spawned', [False])
    return got


def test_a_build_shows_no_window(starts):
    """A console program started with no console gets a window of its own: a
    detached build's self-test opened one for every build."""
    _image.build_later()
    assert len(starts) == 1
    if sys.platform == 'win32':
        flags = starts[0]['creationflags']
        assert flags & _image.NO_WINDOW and not flags & 0x8     # (DETACHED_PROCESS)
    assert _image._quiet() == ({'creationflags': _image.NO_WINDOW} if sys.platform == 'win32' else {})


def test_a_build_under_way_is_not_started_again(starts):
    os.makedirs(os.path.dirname(_image.path()))
    open(_image.path() + '.lock', 'w').close()
    _image.build_later()
    assert not starts


@pytest.mark.parametrize('kw', [{'compression': 'lzf'}, {'compression': 'gzip'}, {'scaleoffset': 0}], ids=str)
def test_a_first_read_of_any_filter_needs_no_numba(tmp_path, kw):
    """In a fresh process, by whichever path it comes to a kernel: reads not by
    deflate came to one before anything had looked for the image."""
    if _image.current() is None:
        pytest.skip('no image of these sources here')
    h5py = pytest.importorskip('h5py')
    p = str(tmp_path / 'f.h5')
    with h5py.File(p, 'w') as f:
        f.create_dataset('x', data=np.arange(40000, dtype='i4').reshape(200, 200), chunks=(50, 50), **kw)
    code = ('import sys, fastnc\n'
            'f = fastnc.File(sys.argv[1])\n'
            'assert int(f["x"][()].sum()) == 39999 * 40000 // 2\n'
            'print("numba" in sys.modules)\n')
    r = subprocess.run([sys.executable, '-c', code, p], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, FASTNC_IMAGE_BUILD='0'), **_image._quiet())
    assert r.stdout.strip() == 'False', r.stderr[-400:]


def test_every_filter_is_in_a_fresh_install_s_image():
    """What is not listed is compiled only by a read that needs it, by numba,
    however good the image; and any filter may come first, on the file's
    read-only bytes (a list made from one machine's reads left three out)."""
    from fastnc import _codecs, _signatures
    for driver, _ in _codecs.DRIVERS.values():
        name = 'fastnc._codecs.' + driver.__name__
        assert name in _signatures.SIGNATURES, name
        assert any(_signatures.TYPES[sig[0]][3] for sig in _signatures.SIGNATURES[name]), name


def test_a_failed_build_is_tried_again_only_later(starts):
    """Not by every process that then goes by numba, 20 s of CPU each; but
    a failure need not be for good."""
    os.makedirs(os.path.dirname(_image.path()))
    failed = _image.path() + '.failed'
    open(failed, 'w').close()
    _image.build_later()
    assert not starts and _image.coming() is False
    os.utime(failed, (time.time() - 2 * 86400,) * 2)
    _image._spawned[0] = False                      # (once a process: the next one)
    _image.build_later()
    assert len(starts) == 1


def test_no_build_when_told_not_to(starts, monkeypatch):
    monkeypatch.setenv('FASTNC_IMAGE_BUILD', '0')
    _image.build_later()
    assert not starts
    assert _image.coming() is False                 # (and reads go by numba, as with no image)


def test_damaged_unit_is_never_run(tmp_path):
    """A CRC that does not match leaves the call to numba."""
    p = str(tmp_path / 'x.img')
    code = bytes(1 << 16)
    rec = dict(at=_image.HEAD.size, stored=len(code), size=len(code), nrel=0, crc=zlib.crc32(code) ^ 1,
               rodata=1 << 16, entry=0, nslots=1, ret=None, ext=[])
    tail = __import__('marshal').dumps(dict(stamp=0, externals=[], kernels={'m.k': {('i8',): rec}}))
    with open(p, 'wb') as f:
        f.write(_image.HEAD.pack(_image.MAGIC, _image.FORMAT, zlib.crc32(tail),
                                 _image.HEAD.size + len(code) + len(tail), _image.HEAD.size + len(code), len(tail)))
        f.write(code)
        f.write(tail)

    def k():
        pass
    k.__module__, k.__qualname__ = 'm', 'k'

    class Kernel:
        fn = k
        fast = {}
    im = _image.Image(p)
    assert im.call(Kernel, (1,)) is _image.NO


READ_THROUGH_IMAGE = r'''
import sys, zlib
import numpy as np
import fastnc
p = sys.argv[1]
with fastnc.Dataset(p) as d:
    x = d['values']
    got = x[...]
    window = x[3:40, 5:9]
    d.set_auto_maskandscale(False)
    all_ = d.load()['values']
print('numba' in sys.modules, zlib.crc32(np.ma.getdata(got).tobytes()), zlib.crc32(np.ma.getdata(window).tobytes()),
      zlib.crc32(all_.tobytes()))
'''


def test_read_through_an_image_without_numba(tmp_path):
    """A fresh process with this machine's image, if it has one: the same
    values, and numba never imported."""
    netCDF4 = pytest.importorskip('netCDF4')
    from fastnc import _chunks
    if not _chunks.IMAGE or _image.current() is None:
        pytest.skip('no image here')
    f = str(tmp_path / 'one.nc')
    vals = np.ma.masked_equal(np.arange(100 * 40, dtype=np.int32).reshape(100, 40) % 977, 5)
    with netCDF4.Dataset(f, 'w') as d:
        d.createDimension('a', 100)
        d.createDimension('b', 40)
        v = d.createVariable('values', 'i4', ('a', 'b'), zlib=True, shuffle=True, chunksizes=(10, 8),
                             fill_value=5)
        v[...] = vals
    env = dict(os.environ, FASTNC_IMAGE='1')
    r = subprocess.run([sys.executable, '-c', READ_THROUGH_IMAGE, f], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    numba, whole, window, raw = r.stdout.split()
    assert numba == 'False'
    assert int(whole) == zlib.crc32(np.ma.getdata(vals).tobytes())
    assert int(window) == zlib.crc32(np.ma.getdata(vals[3:40, 5:9]).tobytes())
    assert int(raw) == zlib.crc32(vals.filled(5).tobytes())


def test_no_image_when_turned_off(monkeypatch):
    code = "import fastnc; from fastnc import _chunks; print(_chunks.loaded())"
    env = dict(os.environ, FASTNC_IMAGE='0')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.stdout.strip() == 'False'
