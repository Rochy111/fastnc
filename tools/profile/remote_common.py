"""Shared by the remote measurements: a logging HTTP proxy (every client's
requests and bytes counted on their way to the real server, at the real
server's latency), and a minimal Python file-like object over HTTP ranges
-- what h5py sees when handed an fsspec file: one GET per read() without
a cache, or fsspec's default 'bytes' read-ahead cache (5 MB blocks)."""
import http.client
import http.server
import io
import queue
import re
import threading
import urllib.parse


class Proxy:
    """http://127.0.0.1:<port>/<path> forwarded to upstream (an https URL
    prefix); `log` holds (method, range, bytes sent) per request."""

    def __init__(self, upstream):
        u = urllib.parse.urlsplit(upstream)
        self.host, self.prefix = u.hostname, u.path.rstrip('/')
        self.log = []
        proxy = self
        idle = queue.LifoQueue()                # kept-alive upstream connections, shared:
        # a client that opens a connection per request must not pay a TLS
        # handshake upstream for it (the real server keeps one alive anyway)

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def _fwd(self, method):
                hdr = {k: v for k, v in self.headers.items() if k.lower() in ('range',)}
                for attempt in range(3):
                    try:
                        c = idle.get_nowait()
                    except queue.Empty:
                        c = http.client.HTTPSConnection(proxy.host, timeout=60)
                    try:
                        c.request(method, proxy.prefix + self.path, headers=hdr)
                        r = c.getresponse()
                        break
                    except (OSError, http.client.HTTPException):
                        c.close()
                else:
                    raise OSError('upstream failed')
                self.send_response(r.status)
                for k in ('Content-Range', 'Content-Type', 'Accept-Ranges', 'Last-Modified', 'ETag',
                          'Content-Length'):
                    if r.getheader(k):
                        self.send_header(k, r.getheader(k))
                self.end_headers()
                n = 0
                while True:                     # streamed, as the real server sends it
                    d = r.read(1 << 18)
                    if not d:
                        break
                    n += len(d)
                    if method != 'HEAD':
                        self.wfile.write(d)
                idle.put(c)
                proxy.log.append((method, hdr.get('Range') or hdr.get('range'), n))

            def do_GET(self):
                self._fwd('GET')

            def do_HEAD(self):
                self._fwd('HEAD')

            def log_message(self, *a):
                pass
        self.srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = 'http://127.0.0.1:%d' % self.srv.server_address[1]

    def counts(self):
        return len(self.log), sum(n for _, _, n in self.log)

    def close(self):
        self.srv.shutdown()


class HTTPFile(io.RawIOBase):
    """A read-only file over HTTP byte ranges (keep-alive connection).
    cache=0: one GET per read (h5py + a plain fsspec file without cache);
    cache=N: fsspec's 'bytes' cache -- a miss fetches N bytes from the read
    position (or the read's size if larger), one block kept."""

    def __init__(self, url, cache=0):
        u = urllib.parse.urlsplit(url)
        self.conn_args = (u.hostname, u.port)
        self.https = u.scheme == 'https'
        self.path = u.path
        self.pos = 0
        self.cache = cache
        self.block = (0, b'')
        self.requests = 0
        self.fetched = 0
        self.c = None
        self.size = self._get(0, 1, size=True)

    def _conn(self):
        if self.c is None:
            cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
            self.c = cls(*self.conn_args, timeout=60)
        return self.c

    def _get(self, a, b, size=False):
        for attempt in range(3):
            try:
                c = self._conn()
                c.request('GET', self.path, headers={'Range': 'bytes=%d-%d' % (a, b - 1)})
                r = c.getresponse()
                data = r.read()
                break
            except (OSError, http.client.HTTPException):
                if self.c is not None:
                    self.c.close()
                self.c = None
        else:
            raise
        self.requests += 1
        self.fetched += len(data)
        if size:
            return int(re.search(r'/(\d+)', r.getheader('Content-Range')).group(1))
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, out):
        n = min(len(out), self.size - self.pos)
        if n <= 0:
            return 0
        a, b = self.pos, self.pos + n
        if self.cache:
            s, d = self.block
            if not (s <= a and b <= s + len(d)):
                e = min(self.size, max(b, a + self.cache))
                self.block = (a, self._get(a, e))
                s, d = self.block
            out[:n] = d[a - s:b - s]
        else:
            out[:n] = self._get(a, b)
        self.pos = b
        return n

    def close(self):
        if self.c is not None:
            self.c.close()
        super().close()
