"""Files that are not on a local disk: HTTP(S) and S3 URLs (byte ranges,
anonymous) and Python file-like objects (seek / read, as fsspec gives).

The rest of fastnc reads a file's bytes from one array, `src` (the mapped
file). A remote file gets the same: `Source.buf`, a zeroed array the size
of the file (np.zeros: the operating system gives it pages only as they
are written), filled in as the parts are fetched, with a record of the
byte ranges present. Two calls bring bytes in before they are read:

- `need(at, n)`: the HDF5 parser, before it reads a structure (object
  header, B-tree node, heap, ...). A miss fetches whole aligned blocks
  with read-ahead: metadata clusters, and a request costs a round trip
  (40-60 ms to S3) where 256 kB more costs ~5 ms. Local files: a no-op.
- `fetch(addr, size)`: the chunk engine, with every planned chunk's
  address and stored size before its kernel runs (and the contiguous
  reader with a selection's runs). Ranges already present are dropped,
  the rest sorted and merged across gaps under `gap` bytes (fetching a
  small gap costs less than another request), cut into parts so that the
  connections share a big read, and fetched on `connections` threads, each
  keeping its HTTP connection alive between requests.

`prefetch(addrs, n)` is `need` for many structures at once (the children
of a B-tree node, the object headers of a group's members): one parallel
round instead of a round trip each.

netCDF-3 (`_nc3`) plugs in the same way: its backend gets the Source as
`source=` and calls `source.need(0, n)` while it reads the header (the
first `initial` bytes are already here) and `source.fetch(starts, sizes)`
with a variable's byte ranges (a record variable: one range per record)
before reading them from `src`. A backend that does not take `source`
gets the whole file fetched first.

HTTP errors become OSError; 5xx, 429, timeouts and dropped connections are
retried with exponential backoff. No dependencies beyond the standard
library: http.client, one connection per I/O thread and host.
"""
import bisect
import http.client
import os
import re
import socket
import ssl
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BLOCK = 1 << 18             # metadata fetched in aligned blocks of this size
INITIAL = 1 << 20           # the first request: the superblock and root metadata
TAIL = 1 << 18              # and, alongside it, the file's end (HDF5 writes metadata there at close)
GAP = 1 << 20               # data ranges closer than this are fetched as one
PART = 8 << 20              # biggest single request (bigger ranges are split)
MIN_PART = 1 << 20          # a range is split for parallelism down to this size
CONNECTIONS = 16            # I/O threads (HTTP connections) shared by all sources
RETRIES = 5
TIMEOUT = 30.0

_URL = re.compile(r'^(https?|s3)://', re.I)
_POISON = None              # tests: fill new buffers with this byte instead of zeros


def is_url(name):
    return isinstance(name, str) and _URL.match(name) is not None


def open_source(name, storage_options=None):
    """A Source for a URL or a file-like object; None for a local path."""
    opts = dict(storage_options or {})
    if is_url(name):
        return Source(_http_for(name, opts), name.split('#', 1)[0], **_source_opts(opts))
    if hasattr(name, 'read') and hasattr(name, 'seek'):
        label = getattr(name, 'path', None) or getattr(name, 'name', None) or repr(name)
        return Source(FileObject(name), str(label), **_source_opts(opts))
    return None


def readonly(source):
    """The source's buffer as a read-only array: the kernels then see the
    type they see for a mapped file (numba compiles writable arrays apart)."""
    v = source.buf.view()
    v.flags.writeable = False
    return v


def nc3_open(NC3File, buf, src, source):
    """A netCDF-3 backend over buf / src; for a remote file it is given the
    Source (`source=`) to fetch what it reads, or, if it does not take one,
    the whole file is fetched first."""
    if source is None:
        return NC3File(buf, src)
    import inspect
    try:
        takes = 'source' in inspect.signature(NC3File).parameters
    except (TypeError, ValueError):
        takes = False
    if takes:
        return NC3File(buf, src, source=source)
    source.fetch_all()
    return NC3File(buf, src)


def _source_opts(opts):
    keys = ('block_size', 'initial', 'gap', 'part', 'tail')
    return {k: opts[k] for k in keys if k in opts}


def _http_for(url, opts):
    url = url.split('#', 1)[0]                  # netCDF-c's '#mode=bytes'
    if url[:5].lower() == 's3://':
        bucket, _, key = url[5:].partition('/')
        key = urllib.parse.quote(key, safe='/~')
        endpoint = opts.get('endpoint_url')
        region = opts.get('region') or opts.get('region_name')
        if endpoint:
            url = '%s/%s/%s' % (endpoint.rstrip('/'), bucket, key)
        elif '.' in bucket:                     # dotted buckets: path style (TLS names)
            url = 'https://s3.%samazonaws.com/%s/%s' % (region + '.' if region else '', bucket, key)
        else:
            url = 'https://%s.s3.%samazonaws.com/%s' % (bucket, region + '.' if region else '', key)
    return HTTP(url, headers=opts.get('headers'), timeout=opts.get('timeout', TIMEOUT),
                retries=opts.get('retries', RETRIES))


# ---------------------------------------------------------------- the source

_IO = [None]
_IO_LOCK = threading.Lock()


def _io_pool():
    with _IO_LOCK:
        if _IO[0] is None:
            _IO[0] = ThreadPoolExecutor(CONNECTIONS, thread_name_prefix='fastnc-io')
        return _IO[0]


class Source:
    """A remote file's bytes, fetched as asked for into `buf`.

    `requests` and `fetched` count the requests made and the bytes they
    brought (for measuring)."""

    def __init__(self, transport, name, block_size=BLOCK, initial=INITIAL, gap=GAP, part=PART,
                 tail=TAIL):
        self.t = transport
        self.name = name
        self.block = int(block_size)
        self.gap = int(gap)
        self.part = int(part)
        size, head, t0, tail = transport.open(int(initial), int(tail))
        self.size = size
        self.buf = np.zeros(size, np.uint8)
        if _POISON is not None:
            self.buf[:] = _POISON               # tests: bytes read before fetched show up
        self.buf[:len(head)] = np.frombuffer(head, np.uint8)
        self.buf[t0:t0 + len(tail)] = np.frombuffer(tail, np.uint8)
        self._iv = self._intervals(_merge([r for r in ((0, len(head)), (t0, t0 + len(tail))) if r[1] > r[0]], 0))
        self._lock = threading.Lock()
        self.requests = 1 + bool(tail)
        self.fetched = len(head) + len(tail)
        self.on_fetch = None                    # called with (start, end) as each part lands

    @staticmethod
    def _intervals(pairs):
        """(starts, ends, starts array, ends array) of disjoint sorted ranges:
        replaced whole, never changed in place, so that readers on other
        threads see one consistent set without the lock."""
        s = [a for a, _ in pairs]
        e = [b for _, b in pairs]
        return s, e, np.array(s, np.int64), np.array(e, np.int64)

    def has(self, at, end):
        """True when bytes [at, end) are all here."""
        s, e = self._iv[0], self._iv[1]
        i = bisect.bisect_right(s, at) - 1
        return i >= 0 and e[i] >= end

    def need(self, at, n):
        """Make bytes [at, at + n) present: whole blocks around them, and the
        block after (read-ahead), in one request."""
        end = at + n
        s, e = self._iv[0], self._iv[1]
        i = bisect.bisect_right(s, at) - 1
        if i >= 0 and e[i] >= end:
            return
        size = self.size
        if end > size:
            end = size
        if at >= end:
            return
        B = self.block
        a = at - at % B
        self._get([(a, min(size, -(-end // B) * B + B))])

    def prefetch(self, addrs, n):
        """need(a, n) for every a in addrs, in one parallel round: n bytes
        from each (no whole blocks: the nodes of a chunk B-tree lie apart,
        between chunks), merged across gaps of under a block."""
        size = self.size
        rng = []
        for a in addrs:
            if 0 <= a < size and not self.has(a, min(size, a + n)):
                rng.append((a, min(size, a + n)))
        if rng:
            self._get(_merge(rng, self.block))

    def fetch(self, addr, size, gap=None):
        """Make every range [addr[i], addr[i] + size[i]) present (arrays of
        int64): the missing ones merged across gaps under `gap` bytes and
        fetched in parallel."""
        addr = np.asarray(addr, np.int64).ravel()
        end = addr + np.asarray(size, np.int64).ravel()
        keep = end > addr
        if not keep.all():
            addr, end = addr[keep], end[keep]
        if not len(addr):
            return
        np.minimum(end, self.size, out=end)
        s, e = self._iv[2], self._iv[3]
        if len(s):
            i = np.searchsorted(s, addr, 'right') - 1
            miss = (i < 0) | (e[np.maximum(i, 0)] < end)
            if not miss.any():
                return
            addr, end = addr[miss], end[miss]
        o = np.argsort(addr, kind='stable')
        addr, end = addr[o], np.maximum.accumulate(end[o])
        g = self.gap if gap is None else gap
        first = np.flatnonzero(np.concatenate(([True], addr[1:] > end[:-1] + g)))
        last = np.concatenate((first[1:] - 1, [len(addr) - 1]))
        self._get(list(zip(addr[first].tolist(), end[last].tolist())), g)

    def fetch_all(self):
        self._get([(0, self.size)])

    def _missing(self, a, end):
        s, e = self._iv[0], self._iv[1]
        out = []
        i = bisect.bisect_right(s, a) - 1
        if i >= 0 and e[i] > a:
            a = e[i]
        i += 1
        while a < end:
            if i < len(s) and s[i] < end:
                if s[i] > a:
                    out.append((a, s[i]))
                a = max(a, e[i])
                i += 1
            else:
                out.append((a, end))
                break
        return out

    def _get(self, ranges, gap=0):
        """Fetch what is missing of `ranges` ((start, end) pairs)."""
        with self._lock:
            pieces = _merge([p for a, b in ranges for p in self._missing(a, b)], gap)
            if not pieces:
                return
            total = sum(b - a for a, b in pieces)
            par = self.t.parallel
            # parts: small enough that every connection gets some of a big read
            part = max(MIN_PART, min(self.part, -(-total // CONNECTIONS))) if par > 1 else self.part
            jobs = []
            for a, b in pieces:
                n = -(-(b - a) // part)
                step = -(-(b - a) // n)
                jobs.extend((x, min(b, x + step)) for x in range(a, b, step))
            done = []
            buf = self.buf
            t = self.t
            cb = self.on_fetch

            def one(job):
                a, b = job
                t.read_into(a, memoryview(buf[a:b]))
                if cb is not None:
                    cb(a, b)
                return job
            err = None
            if par > 1 and len(jobs) > 1:
                futs = [_io_pool().submit(one, j) for j in jobs]
                for f in futs:
                    try:
                        done.append(f.result())
                    except BaseException as ex:  # keep the parts that arrived
                        err = err or ex
            else:
                for j in jobs:
                    try:
                        done.append(one(j))
                    except BaseException as ex:
                        err = ex
                        break
            self.requests += len(done)
            self.fetched += sum(b - a for a, b in done)
            if done:
                self._iv = self._intervals(_merge(list(zip(self._iv[0], self._iv[1])) + done, 0))
            if err is not None:
                if isinstance(err, OSError):
                    raise err
                raise OSError('reading %s: %s' % (self.name, err)) from err

    def close(self):
        self.t.close()


def _merge(pairs, gap):
    """Sorted (start, end) pairs, overlapping ones (and ones closer than gap) merged."""
    pairs = sorted(pairs)
    out = []
    for a, b in pairs:
        if out and a <= out[-1][1] + gap:
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


# ---------------------------------------------------------------- transports

class FileObject:
    """A Python file-like object (seek / read or readinto): one request at a time."""
    parallel = 1

    def __init__(self, f):
        self.f = f

    def open(self, initial, tail):
        """(size, first bytes, where the last bytes start, last bytes): no
        tail here -- one request at a time, it would cost a round trip."""
        f = self.f
        size = f.seek(0, 2)
        if size is None:
            size = f.tell()
        f.seek(0)
        return size, self._read(0, min(initial, size)), size, b''

    def _read(self, a, n):
        f = self.f
        f.seek(a)
        parts = []
        while n > 0:
            d = f.read(n)
            if not d:
                raise OSError('file object ended %d bytes early' % n)
            parts.append(d)
            n -= len(d)
        return b''.join(parts)

    def read_into(self, a, out):
        f = self.f
        n = len(out)
        if hasattr(f, 'readinto'):
            f.seek(a)
            k = 0
            while k < n:
                r = f.readinto(out[k:])
                if not r:
                    raise OSError('file object ended %d bytes early' % (n - k))
                k += r
        else:
            out[:] = self._read(a, n)

    def close(self):
        pass


class _Retry(Exception):
    pass


_IDLE = {}                  # (scheme, host, port) -> kept-alive connections not in use
_IDLE_LOCK = threading.Lock()


class HTTP:
    """Byte ranges of one URL over HTTP(S). Connections are kept alive and
    shared by every file on the same host (a TLS handshake costs two or
    three round trips: opening many files of one bucket pays it once)."""
    parallel = CONNECTIONS

    def __init__(self, url, headers=None, timeout=TIMEOUT, retries=RETRIES):
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.retries = retries
        self._set_url(url)

    def _set_url(self, url):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ('http', 'https'):
            raise ValueError('not an http(s) URL: %r' % url)
        self.url = url
        self.scheme = u.scheme
        self.host = u.hostname
        self.port = u.port or (443 if u.scheme == 'https' else 80)
        self.path = (u.path or '/') + ('?' + u.query if u.query else '')
        self.proxy = None
        if not urllib.request.proxy_bypass(self.host):
            p = urllib.request.getproxies().get(self.scheme)
            if p:
                p = urllib.parse.urlsplit(p if '://' in p else 'http://' + p)
                self.proxy = p.hostname, p.port or 8080
        self.key = self.scheme, self.host, self.port, self.proxy

    def _conn(self):
        """(a connection, whether it was kept alive from before)."""
        with _IDLE_LOCK:
            idle = _IDLE.get(self.key)
            if idle:
                return idle.pop(), True
        target = self.proxy or (self.host, self.port)
        if self.scheme == 'https':
            c = http.client.HTTPSConnection(*target, timeout=self.timeout,
                                            context=ssl.create_default_context())
            if self.proxy:
                c.set_tunnel(self.host, self.port)
        else:
            c = http.client.HTTPConnection(*target, timeout=self.timeout)
        return c, False

    def _release(self, c):
        with _IDLE_LOCK:
            idle = _IDLE.setdefault(self.key, [])
            if len(idle) < 2 * CONNECTIONS:
                idle.append(c)
                return
        c.close()

    def _range(self, a, b, handle):
        """GET bytes [a, b) and handle(response); b None: to the end."""
        h = dict(self.headers)
        h['Range'] = 'bytes=%d-%s' % (a, '' if b is None else b - 1) if a >= 0 else 'bytes=%d' % a
        delay = 0.25
        attempt = 0
        redirects = 0
        while True:
            c, reused = self._conn()
            try:
                c.request('GET', self.url if self.proxy and self.scheme == 'http' else self.path,
                          headers=h)
                r = c.getresponse()
                st = r.status
                if st in (301, 302, 303, 307, 308) and redirects < 5:
                    loc = r.getheader('Location')
                    region = r.getheader('x-amz-bucket-region')
                    r.read()
                    self._release(c)
                    redirects += 1
                    if loc:
                        self._set_url(urllib.parse.urljoin(self.url, loc))
                        continue
                    if region and '.s3.' in '.' + self.host:
                        self._set_url(re.sub(r'\.s3(\.[a-z0-9-]+)?\.amazonaws\.com',
                                             '.s3.%s.amazonaws.com' % region, self.url, count=1))
                        continue
                    raise RemoteError('HTTP %d %s for %s (no Location)' % (st, r.reason, self.url))
                if st in (429, 500, 502, 503, 504):
                    raise _Retry('HTTP %d %s' % (st, r.reason))
                if st not in (200, 206):
                    body = r.read(500)
                    c.close()
                    msg = re.sub(rb'\s+', b' ', body).decode('utf-8', 'replace')[:300]
                    raise RemoteError('HTTP %d %s for %s%s' % (st, r.reason, self.url,
                                                                ': ' + msg if msg else ''))
                out = handle(r)
                self._release(c)
                return out
            except RemoteError:
                raise
            except (_Retry, OSError, http.client.HTTPException) as ex:
                c.close()
                if reused and not isinstance(ex, (_Retry, socket.timeout)):
                    continue                    # a kept-alive connection the server had closed
                attempt += 1
                if attempt > self.retries:
                    raise RemoteError('reading %s (bytes %d-%s) failed after %d tries: %s'
                                      % (self.url, a, '' if b is None else b - 1, attempt, ex)) from ex
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

    def open(self, initial, tail):
        """(file size, its first bytes, where the last bytes start, last
        bytes): the first `initial` and last `tail` bytes, fetched at once."""
        def ranged(r):
            if r.status == 206:
                cr = r.getheader('Content-Range') or ''
                m = re.match(r'bytes\s+(\d+)-(\d+)/(\d+|\*)', cr)
                if not m or m.group(3) == '*':
                    raise RemoteError('%s: no file size in Content-Range %r' % (self.url, cr))
                return int(m.group(3)), r.read(), int(m.group(1))
            data = r.read()                     # 200: the server sends the whole file
            return len(data), data, 0
        end = _io_pool().submit(self._range, -tail, None, ranged) if tail else None
        size, head, _ = self._range(0, initial, ranged)
        if end is None:
            return size, head, size, b''
        try:
            _, last, t0 = end.result()
        except OSError:
            return size, head, size, b''       # (an empty file: 416)
        return size, head, t0, last

    def read_into(self, a, out):
        n = len(out)

        def into(r):
            if r.status != 206:
                r.read()
                raise RemoteError('%s: the server ignored a byte range request' % self.url)
            k = 0
            while k < n:
                got = r.readinto(out[k:])
                if not got:
                    raise http.client.IncompleteRead(b'', n - k)
                k += got
            if r.read(1):
                raise RemoteError('%s: more bytes than asked for at %d' % (self.url, a))
        self._range(a, a + n, into)

    def close(self):
        pass


class RemoteError(OSError):
    """An HTTP request that failed (status, or retries exhausted)."""
