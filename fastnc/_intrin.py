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
