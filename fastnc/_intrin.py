"""LLVM intrinsics the kernels need and numba does not offer: unaligned
loads and stores of a machine word out of a byte array, and an atomic
fetch-and-add for sharing work out among threads.

All byte arrays here are C-contiguous uint8 and the machine is
little-endian (x86-64, arm64): `ld64(a, i)` is bytes a[i:i+8] as a
little-endian uint64, one instruction. Nothing is bounds-checked: callers
guarantee the eight bytes are inside the array.
"""
from ._jit import intrinsic


def _ptr(context, builder, aryty, ary, idx, bits):
    from llvmlite import ir
    from numba.core import cgutils
    a = context.make_array(aryty)(context, builder, ary)
    p = cgutils.get_item_pointer(context, builder, aryty, a, [idx])
    return builder.bitcast(p, ir.IntType(bits).as_pointer())


@intrinsic
def ld64(typingctx, a, i):
    """a[i:i+8] as a little-endian uint64 (an unaligned load)."""
    from numba import types
    def codegen(context, builder, sig, args):
        p = _ptr(context, builder, sig.args[0], args[0], args[1], 64)
        return builder.load(p, align=1)
    return types.uint64(a, i), codegen


@intrinsic
def st64(typingctx, a, i, v):
    """a[i:i+8] = v's eight bytes, little-endian (an unaligned store)."""
    from numba import types
    def codegen(context, builder, sig, args):
        p = _ptr(context, builder, sig.args[0], args[0], args[1], 64)
        builder.store(args[2], p, align=1)
        return context.get_dummy_value()
    return types.void(a, i, v), codegen


@intrinsic
def take(typingctx, ctr, i):
    """ctr[i]'s value, then one more there: an atomic fetch-and-add, so each
    of several threads calling it takes a different value (fastgrib's)."""
    from numba import types
    from numba.core import cgutils
    def codegen(context, builder, sig, args):
        ary = context.make_array(sig.args[0])(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, sig.args[0], ary, [args[1]])
        return builder.atomic_rmw('add', ptr, context.get_constant(types.int64, 1),
                                  'acq_rel')
    return types.int64(ctr, i), codegen


@intrinsic
def call_inflate(typingctx, fp, dec, src, a, n, dst, o, cap):
    """libdeflate_deflate_decompress(dec, src+a, n, dst+o, cap, &got) through
    the function pointer fp (an int: no ctypes object in the kernel, so
    numba still caches it); got, or -1 when it fails (a bad stream, more
    than cap, or less: the caller decides what to do then)."""
    from numba import types
    from numba.core import cgutils
    from llvmlite import ir

    def codegen(context, builder, sig, args):
        fpv, decv, sary, av, nv, dary, ov, capv = args
        i8p = ir.IntType(8).as_pointer()
        i64 = ir.IntType(64)
        i32 = ir.IntType(32)
        fnty = ir.FunctionType(i32, [i8p, i8p, i64, i8p, i64, i64.as_pointer()])
        fn = builder.inttoptr(fpv, fnty.as_pointer())
        s = context.make_array(sig.args[2])(context, builder, sary)
        d = context.make_array(sig.args[5])(context, builder, dary)
        sp = builder.gep(builder.bitcast(s.data, i8p), [av])
        dp = builder.gep(builder.bitcast(d.data, i8p), [ov])
        got = cgutils.alloca_once(builder, i64)
        builder.store(ir.Constant(i64, 0), got)
        r = builder.call(fn, [builder.inttoptr(decv, i8p), sp, nv, dp, capv, got])
        ok = builder.icmp_signed('==', r, ir.Constant(i32, 0))
        return builder.select(ok, builder.load(got), ir.Constant(i64, -1))
    return types.int64(fp, dec, src, a, n, dst, o, cap), codegen
