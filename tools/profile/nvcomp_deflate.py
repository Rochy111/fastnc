"""nvCOMP's deflate on the real files' chunks: zlib header stripped (raw
deflate), compressed bytes uploaded once, decoded as one batch. Times the
upload, the decode (warm), and checks against zlib."""
import sys, time, zlib
import h5py, numpy as np, cupy as cp
from nvidia import nvcomp
from common import path, real_files

def chunks(p):
    h = h5py.File(p, 'r'); raw = np.fromfile(p, np.uint8); jobs = []
    def visit(n, d):
        if isinstance(d, h5py.Dataset) and d.chunks and d.size:
            dc = d.id.get_create_plist()
            if 1 in [dc.get_filter(i)[0] for i in range(dc.get_nfilters())]:
                L = []; d.id.chunk_iter(L.append)
                csz = int(np.prod(d.chunks)) * d.dtype.itemsize
                jobs.extend((c.byte_offset, c.size, csz) for c in L)
    h.visititems(visit); return raw, jobs

for f in (sys.argv[1:] or real_files()):
    try: raw, jobs = chunks(path(f))
    except OSError: continue
    if not jobs: continue
    # one packed host buffer of raw deflate streams
    comp = [raw[o + 2:o + s] for o, s, _ in jobs]
    sizes = np.array([len(c) for c in comp]); padded = (sizes + 15) // 16 * 16   # nvCOMP wants aligned starts
    starts = np.concatenate([[0], np.cumsum(padded)]); offs = starts
    host = np.zeros(starts[-1], np.uint8)
    for c, st in zip(comp, starts): host[st:st + len(c)] = c
    total = sum(j[2] for j in jobs)
    pinned = cp.cuda.alloc_pinned_memory(host.nbytes)
    hp = np.frombuffer(pinned, np.uint8, host.nbytes); hp[:] = host
    for backend in ('DEFAULT', 'CUDA'):
        try:
            codec = nvcomp.Codec(algorithm='Deflate', bitstream_kind=nvcomp.BitstreamKind.RAW,
                                 execution_policy=nvcomp.ExecutionPolicy.LATENCY if hasattr(nvcomp,'ExecutionPolicy') else None,
                                 decompress_backend=getattr(nvcomp, 'NVCOMP_DECOMPRESS_BACKEND_' + backend))
        except Exception as e:
            print(f, backend, 'codec error', e); continue
        best_up = best_dec = 9e9
        for rep in range(4):
            cp.cuda.Device().synchronize(); t = time.perf_counter()
            dev = cp.empty(host.nbytes, cp.uint8); dev.set(hp)
            cp.cuda.Device().synchronize(); t1 = time.perf_counter()
            arrs = nvcomp.as_arrays([dev[starts[i]:starts[i] + sizes[i]] for i in range(len(jobs))])
            out = codec.decode(arrs)
            cp.cuda.Device().synchronize(); t2 = time.perf_counter()
            if rep == 0: first = t2 - t
            else: best_up = min(best_up, t1 - t); best_dec = min(best_dec, t2 - t1)
        # check a sample
        ok = True
        for i in range(0, len(jobs), max(1, len(jobs) // 20)):
            o, s, csz = jobs[i]
            got = cp.asnumpy(cp.asarray(out[i]).view(cp.uint8)).tobytes()
            if got != zlib.decompress(raw[o:o + s].tobytes()): ok = False
        print(f'{f[:36]:36s} {backend:7s} {len(jobs):6d} chunks {total/1e6:7.1f} MB  first {first*1e3:7.1f} ms  '
              f'upload {best_up*1e3:6.1f} ms ({host.nbytes/best_up/1e9:4.1f} GB/s)  decode {best_dec*1e3:6.1f} ms ({total/best_dec/1e9:5.2f} GB/s out)  ok={ok}')
