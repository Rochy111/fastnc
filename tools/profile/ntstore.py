"""Do non-temporal (streaming) stores pay on this machine? Writing a big
output normally costs a read of each line first (read-for-ownership); a
streaming store skips it. Compared: u2 -> f4 scale straight into the output,
vs scaled into an L1-sized scratch and streamed out; and a plain copy."""
import sys, time
import numpy as np
from numba import njit, types
from numba.core import cgutils
from common import fast_thread, path
sys.path.insert(0, path('.'))
import fastnc
fastnc.warm()
from fastnc import _pool
from fastnc._intrin import take
from numba.extending import intrinsic

fast_thread()


@intrinsic
def stream32(typingctx, dst, d, src, s, n):
    """dst[d:d+n] = src[s:s+n] (bytes): 32-byte non-temporal stores over
    the 32-byte-aligned middle of dst, plain stores at the ends."""
    def codegen(context, builder, sig, args):
        from llvmlite import ir
        i8p = ir.IntType(8).as_pointer()
        dary = context.make_array(sig.args[0])(context, builder, args[0])
        sary = context.make_array(sig.args[2])(context, builder, args[2])
        dp = builder.gep(builder.bitcast(dary.data, i8p), [args[1]])
        sp = builder.gep(builder.bitcast(sary.data, i8p), [args[3]])
        n = args[4]
        i64 = ir.IntType(64)
        vty = ir.VectorType(ir.IntType(64), 4)
        # head: up to the first 32-byte boundary of dst
        addr = builder.ptrtoint(dp, i64)
        head = builder.and_(builder.sub(ir.Constant(i64, 0), addr), ir.Constant(i64, 31))
        head = builder.select(builder.icmp_signed('<', n, head), n, head)
        with cgutils.for_range(builder, head) as loop:
            builder.store(builder.load(builder.gep(sp, [loop.index])), builder.gep(dp, [loop.index]))
        rest = builder.sub(n, head)
        nv = builder.lshr(rest, ir.Constant(i64, 5))
        dv = builder.bitcast(builder.gep(dp, [head]), vty.as_pointer())
        sv = builder.bitcast(builder.gep(sp, [head]), vty.as_pointer())
        md = builder.module.add_metadata([ir.Constant(ir.IntType(32), 1)])
        with cgutils.for_range(builder, nv) as loop:
            v = builder.load(builder.gep(sv, [loop.index]), align=1)
            st = builder.store(v, builder.gep(dv, [loop.index]), align=32)
            st.set_metadata('nontemporal', md)
        t0 = builder.add(head, builder.shl(nv, ir.Constant(i64, 5)))
        tail = builder.sub(n, t0)
        with cgutils.for_range(builder, tail) as loop:
            k = builder.add(t0, loop.index)
            builder.store(builder.load(builder.gep(sp, [k])), builder.gep(dp, [k]))
        builder.fence('seq_cst')
        return context.get_dummy_value()
    return types.void(dst, d, src, s, n), codegen


BLK = 8192


@njit(nogil=True)
def scale_plain(x, out, s, o, ctr):
    n = x.size
    nb = (n + BLK - 1) // BLK
    while True:
        b = take(ctr, 0)
        if b >= nb:
            break
        for i in range(b * BLK, min(n, b * BLK + BLK)):
            out[i] = np.float32(x[i]) * s + o


@njit(nogil=True)
def scale_nt(x, out, s, o, ctr):
    n = x.size
    nb = (n + BLK - 1) // BLK
    scr = np.empty(BLK, np.float32)
    ob = out.view(np.uint8)
    sb = scr.view(np.uint8)
    while True:
        b = take(ctr, 0)
        if b >= nb:
            break
        i0 = b * BLK
        m = min(n, i0 + BLK) - i0
        for i in range(m):
            scr[i] = np.float32(x[i0 + i]) * s + o
        stream32(ob, i0 * 4, sb, 0, m * 4)


@njit(nogil=True)
def copy_plain(x, out, ctr):
    n = x.size
    nb = (n + BLK * 4 - 1) // (BLK * 4)
    while True:
        b = take(ctr, 0)
        if b >= nb:
            break
        out[b * BLK * 4:min(n, b * BLK * 4 + BLK * 4)] = x[b * BLK * 4:min(n, b * BLK * 4 + BLK * 4)]


@njit(nogil=True)
def copy_nt(x, out, ctr):
    n = x.size
    nb = (n + BLK * 4 - 1) // (BLK * 4)
    while True:
        b = take(ctr, 0)
        if b >= nb:
            break
        i0 = b * BLK * 4
        stream32(out, i0, x, i0, min(n, i0 + BLK * 4) - i0)


def best(fn, n=7):
    r = []
    for _ in range(n):
        t = time.perf_counter(); fn(); r.append(time.perf_counter() - t)
    return min(r) * 1e3


N = 21696 * 21696
x = (np.arange(N, dtype=np.uint32) % 4093).astype(np.uint16)
out = np.empty(N, np.float32)
out2 = np.empty(N, np.float32)
T = _pool.threads()
for nthr in (1, 8, T):
    def run(k, *a):
        ctr = np.zeros(2, np.int64)
        _pool.share(lambda: k(*a, ctr), nthr)
    tp = best(lambda: run(scale_plain, x, out, np.float32(0.5), np.float32(-3)))
    tn = best(lambda: run(scale_nt, x, out2, np.float32(0.5), np.float32(-3)))
    assert out.tobytes() == out2.tobytes()
    xb = x.view(np.uint8); cb = np.empty_like(xb); cb2 = np.empty_like(xb)
    cp = best(lambda: run(copy_plain, xb, cb))
    cn = best(lambda: run(copy_nt, xb, cb2))
    assert cb.tobytes() == cb2.tobytes() == xb.tobytes()
    print('%2d threads: scale plain %6.1f ms, streamed %6.1f ms | copy %5.0f MB plain %6.1f ms, streamed %6.1f ms'
          % (nthr, tp, tn, xb.nbytes / 1e6, cp, cn))
