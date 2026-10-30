"""Mask and scale on the GPU (cupy): the same plan as the CPU fast path
(fastnc._maskscale.make_plan decides every rule on the host), one CUDA
kernel over the raw values already on the card, the same values to the
bit.

Equal to numpy / numba because every float operation is the IEEE one in
the type numpy uses: compiled with --fmad=false (no a * b + c fused into
one rounding), and the float32 operations -- the multiply, the add, the
conversions to and from double -- written as PTX without `.ftz`: cupy
always compiles with -ftz=true, which would flush float32 subnormals
(numpy keeps them). float32 values are compared after an exact
conversion to double, for the same reason; double arithmetic is never
flushed.

`apply` returns a cupy array where netCDF4-python returns an ndarray, and
a `MaskedArray` (data and mask on the card, `.get()` giving the very
numpy.ma.MaskedArray the CPU path gives) where it returns a masked
array. NaNs come out as x86 makes them (payload kept, the negative
default NaN for invalid operations), as numpy does on the CPU next to it.
What the kernel does not cover (strings, compounds, scalars,
integer scaling) runs on the CPU and comes back as cupy arrays where it
can, numpy objects otherwise.
"""
import threading

import numpy

from . import _maskscale

ma = numpy.ma

CUDA = r'''
// NaNs as x86 makes them, which numpy and numba give on this machine: an
// operation on a NaN returns that NaN, quieted (the scale_factor's /
// add_offset's when both are: numpy's loops put it first); an invalid
// operation (inf - inf, 0 * inf) the negative default NaN; a conversion
// keeps sign and payload. The GPU's own instructions return one
// canonical NaN instead.
__device__ __forceinline__ bool ms_isnan(float a) { return (__float_as_uint(a) & 0x7fffffffu) > 0x7f800000u; }
__device__ __forceinline__ bool ms_isnan(double a)
{
    return ((unsigned long long)__double_as_longlong(a) & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
}
__device__ __forceinline__ float ms_quiet(float a) { return __uint_as_float(__float_as_uint(a) | 0x00400000u); }
__device__ __forceinline__ double ms_quiet(double a)
{
    return __longlong_as_double(__double_as_longlong(a) | 0x0008000000000000ll);
}

__device__ __forceinline__ double ms_d(float v)
{
    if (ms_isnan(v)) {
        unsigned long long b = __float_as_uint(v);
        return __longlong_as_double((long long)(((b >> 31) << 63) | 0x7ff8000000000000ull
                                                | ((b & 0x007fffffull) << 29)));
    }
    double r;
    asm("cvt.f64.f32 %0, %1;" : "=d"(r) : "f"(v));      // exact, no flush
    return r;
}
__device__ __forceinline__ double ms_d(double v) { return v; }
template <typename T> __device__ __forceinline__ double ms_d(T v) { return (double)v; }

__device__ __forceinline__ float ms_f(float v) { return v; }
__device__ __forceinline__ float ms_f(double v)
{
    if (ms_isnan(v)) {
        unsigned long long b = (unsigned long long)__double_as_longlong(v);
        return __uint_as_float((unsigned int)(((b >> 63) << 31) | 0x7fc00000ull
                                              | ((b >> 29) & 0x007fffffull)));
    }
    float r;
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(r) : "d"(v));   // one rounding, no flush
    return r;
}
template <typename T> __device__ __forceinline__ float ms_f(T v) { return (float)v; }

__device__ __forceinline__ float ms_mul(float a, float b)
{
    if (ms_isnan(b)) return ms_quiet(b);  // (both NaN: the constant's, as numpy's loops give)
    if (ms_isnan(a)) return ms_quiet(a);
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return ms_isnan(r) ? __uint_as_float(0xffc00000u) : r;
}
__device__ __forceinline__ float ms_add(float a, float b)
{
    if (ms_isnan(b)) return ms_quiet(b);  // (both NaN: the constant's, as numpy's loops give)
    if (ms_isnan(a)) return ms_quiet(a);
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return ms_isnan(r) ? __uint_as_float(0xffc00000u) : r;
}
__device__ __forceinline__ double ms_mul(double a, double b)
{
    if (ms_isnan(b)) return ms_quiet(b);  // (both NaN: the constant's, as numpy's loops give)
    if (ms_isnan(a)) return ms_quiet(a);
    double r = __dmul_rn(a, b);
    return ms_isnan(r) ? __longlong_as_double((long long)0xfff8000000000000ull) : r;
}
__device__ __forceinline__ double ms_add(double a, double b)
{
    if (ms_isnan(b)) return ms_quiet(b);  // (both NaN: the constant's, as numpy's loops give)
    if (ms_isnan(a)) return ms_quiet(a);
    double r = __dadd_rn(a, b);
    return ms_isnan(r) ? __longlong_as_double((long long)0xfff8000000000000ull) : r;
}

// the type values are compared in: their own, float32 widened to double
template <typename T> struct ms_key { typedef T t; };
template <> struct ms_key<float> { typedef double t; };
template <typename T> __device__ __forceinline__ T ms_k(T v) { return v; }
__device__ __forceinline__ double ms_k(float v) { return ms_d(v); }

// numpy's data * scale_factor + add_offset (mode 1 astype, 2 mul, 3 add,
// 4 both), float32 throughout / ending in double
template <typename T>
__device__ __forceinline__ float ms_val_f(T v, int mode, const float* sc32)
{
    float a = ms_f(v);
    if (mode == 2) return ms_mul(a, sc32[0]);
    if (mode == 3) return ms_add(a, sc32[1]);
    if (mode == 4) return ms_add(ms_mul(a, sc32[0]), sc32[2]);
    return a;
}
template <typename T>
__device__ __forceinline__ double ms_val_d(T v, int mode, int l1f4, const double* sc,
                                           const float* sc32)
{
    if (l1f4) {
        float a = ms_f(v);
        if (mode == 2) return ms_d(ms_mul(a, sc32[0]));
        if (mode == 3) return ms_d(ms_add(a, sc32[1]));
        if (mode == 4) return ms_add(ms_d(ms_mul(a, sc32[0])), sc[2]);
        return ms_d(a);
    }
    double a = ms_d(v);
    if (mode == 2) return ms_mul(a, sc[0]);
    if (mode == 3) return ms_add(a, sc[1]);
    if (mode == 4) return ms_add(ms_mul(a, sc[0]), sc[2]);
    return a;
}
// under the mask: the unscaled value cast to the result type
template <typename T>
__device__ __forceinline__ void ms_put(float* o, long long i, T v, bool hit, int mode, int l1f4,
                                       const double* sc, const float* sc32)
{
    o[i] = hit ? ms_f(v) : ms_val_f(v, mode, sc32);
}
template <typename T>
__device__ __forceinline__ void ms_put(double* o, long long i, T v, bool hit, int mode, int l1f4,
                                       const double* sc, const float* sc32)
{
    o[i] = hit ? (l1f4 ? ms_d(ms_f(v)) : ms_d(v)) : ms_val_d(v, mode, l1f4, sc, sc32);
}

template <typename T, typename O>
__global__ void ms_kernel(const T* __restrict__ x, O* __restrict__ out,
                          unsigned char* __restrict__ msk, long long n,
                          const T* __restrict__ eq, const long long* __restrict__ eqcat, int neq,
                          const T* __restrict__ lohi, int domask, int nan_mv, int nan_fv,
                          int mode, int l1f4, const double* __restrict__ sc,
                          const float* __restrict__ sc32, unsigned int* flags)
{
    typedef typename ms_key<T>::t K;
    K lo = ms_k(lohi[0]), hi = ms_k(lohi[1]);
    unsigned int acc = 0;              // CAT_* bits seen, 8: anything masked
    long long step = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) {
        T v = x[i];
        bool hit = false;
        if (domask) {
            K kv = ms_k(v);
            int cat = 0;
            for (int e = 0; e < neq; e++)
                if (kv == ms_k(eq[e])) cat |= (int)eqcat[e];
            if (kv != kv) {
                if (nan_mv) cat |= 1;
                if (nan_fv) cat |= 2;
            }
            hit = cat != 0 || kv < lo || kv > hi;
            msk[i] = hit;
            if (hit) acc |= cat | 8;
        }
        if (mode) ms_put(out, i, v, hit, mode, l1f4, sc, sc32);
    }
    for (int s = 16; s > 0; s >>= 1)
        acc |= __shfl_xor_sync(0xffffffffu, acc, s);
    if ((threadIdx.x & 31) == 0 && acc) atomicOr(flags, acc);
}
'''

_CTYPE = {'i1': 'signed char', 'u1': 'unsigned char', 'i2': 'short', 'u2': 'unsigned short',
          'i4': 'int', 'u4': 'unsigned int', 'i8': 'long long', 'u8': 'unsigned long long',
          'f4': 'float', 'f8': 'double'}
OPTIONS = ('--std=c++14', '--fmad=false')
_MODULE = None
_LOCK = threading.Lock()


def _names():
    return ['ms_kernel<%s, %s>' % (t, o) for t in _CTYPE.values() for o in ('float', 'double')]


def module():
    """The kernels, compiled once per process (cupy caches them on disk)."""
    global _MODULE
    with _LOCK:
        if _MODULE is None:
            import cupy
            _MODULE = cupy.RawModule(code=CUDA, options=OPTIONS, name_expressions=_names())
    return _MODULE


class MaskedArray:
    """netCDF4-python's masked result with its data and mask on the GPU:
    `data` (cupy), `mask` (cupy bool, or None for numpy.ma.nomask),
    `fill_value` (the masked array's), and `.get()` -> the
    numpy.ma.MaskedArray the CPU path returns, bit for bit."""

    def __init__(self, data, mask, proto):
        self.data, self.mask, self._proto = data, mask, proto

    @property
    def fill_value(self):
        return self._proto.fill_value

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    def filled(self, fill_value=None):
        """data with fill_value (the array's own if None) where masked."""
        import cupy
        if self.mask is None:
            return self.data.copy()
        fv = self._proto.fill_value if fill_value is None else fill_value
        return cupy.where(self.mask, self.data.dtype.type(fv), self.data)

    def get(self):
        d = self.data.get()
        r = d.view(type(self._proto))
        r.__dict__.update(self._proto.__dict__)
        r._mask = ma.nomask if self.mask is None else self.mask.get().view(numpy.bool_)
        return r

    def __repr__(self):
        return 'fastnc GPU MaskedArray(shape=%s, dtype=%s, masked=%s)' % (
            self.shape, self.dtype, self.mask is not None)


def apply(raw, attrs, dtype, mask=True, scale=True, always_mask=True, chartostring=True,
          var_dtype=None, no_fill=False, var_shape=None, key=None):
    """_maskscale.apply for a cupy array raw (see there)."""
    import cupy
    kind = _maskscale._kind(dtype, var_dtype)
    if (kind in ('primitive', 'enum') and isinstance(raw, cupy.ndarray) and raw.ndim > 0
            and raw.flags.c_contiguous and raw.size > 0):
        plan = _maskscale.make_plan(raw.dtype, attrs, dtype, kind, mask, scale, no_fill)
        if plan is not None:
            return run_plan(plan, raw, always_mask)
    host = raw.get() if isinstance(raw, cupy.ndarray) else raw
    r = _maskscale.apply(host, attrs, dtype, mask=mask, scale=scale, always_mask=always_mask,
                         chartostring=chartostring, var_dtype=var_dtype, no_fill=no_fill,
                         var_shape=var_shape, key=key)
    return to_gpu(r)


def to_gpu(r):
    """A CPU result moved to the card where cupy can hold it."""
    import cupy
    if isinstance(r, ma.MaskedArray) and r is not ma.masked and r.dtype.kind in 'biuf' \
            and r.ndim > 0:
        proto = ma.masked_array(numpy.zeros(1, r.dtype))
        proto.__dict__.update({k: v for k, v in r.__dict__.items() if k != '_mask'})
        m = None if r._mask is ma.nomask else cupy.asarray(r._mask)
        return MaskedArray(cupy.asarray(r.data), m, proto)
    if type(r) is numpy.ndarray and r.dtype.kind in 'biuf':
        return cupy.asarray(r)
    return r


def run_plan(p, raw, always_mask):
    """The kernel over raw (a C-contiguous cupy array) for plan p."""
    import cupy
    data = raw.view(p.view) if p.view is not None else raw
    n = data.size
    out = cupy.empty(data.shape, p.out_dtype) if p.out_dtype is not None else None
    msk = cupy.empty(data.shape, numpy.uint8) if p.mask else None
    if out is None and msk is None:
        return data
    odt = 'double' if (p.out_dtype is not None and p.out_dtype.itemsize == 8) else 'float'
    fn = module().get_function('ms_kernel<%s, %s>' % (_CTYPE[data.dtype.str[1:]], odt))
    flags = cupy.zeros(1, numpy.uint32)
    o = out if out is not None else cupy.empty(1, numpy.float32)
    m = msk if msk is not None else cupy.empty(1, numpy.uint8)
    xdt = data.dtype
    eq = cupy.asarray(p.eq.astype(xdt))
    eqcat = cupy.asarray(p.eqcat)
    lohi = cupy.asarray(p.lohi.astype(xdt))
    sc = cupy.asarray(p.sc)
    sc32 = cupy.asarray(p.sc32)
    threads = 256
    dev = cupy.cuda.Device()
    sms = dev.attributes['MultiProcessorCount']
    blocks = int(max(1, min((n + threads - 1) // threads, sms * 16)))
    fn((blocks,), (threads,),
       (data, o, m, numpy.int64(n), eq, eqcat, numpy.int32(p.neq), lohi,
        numpy.int32(p.opts[_maskscale.O_MASK]), numpy.int32(p.opts[_maskscale.O_MVNAN]),
        numpy.int32(p.opts[_maskscale.O_FVNAN]), numpy.int32(p.opts[_maskscale.O_MODE]),
        numpy.int32(p.opts[_maskscale.O_L1F4]), sc, sc32, flags))
    bits = int(flags.get()[0])
    any_mv, any_fv, any_dv, any_m = bits & 1, bits & 2, bits & 4, bits & 8
    if not p.mask:
        return out
    if not any_m and not always_mask:
        return out if out is not None else data
    if any_m:
        fill_value = _maskscale.fill_value_of(p, any_mv, any_fv, any_dv)
        proto = ma.masked_array(numpy.zeros(1, xdt), mask=numpy.ones(1, bool),
                                fill_value=fill_value)
    else:
        proto = ma.masked_array(numpy.zeros(1, xdt))
    if out is not None:
        proto = _maskscale._scale_ops(p.self, proto)
    return MaskedArray(out if out is not None else data, msk.view(numpy.bool_) if any_m else None,
                       proto)
