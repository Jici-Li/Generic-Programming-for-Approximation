"""Minifloat FP(e, m) reference model and typed GP grammar.

MASE correspondence: cross-checked decode/encode below against DeepWok/
mase's _minifloat_ieee_quantize (src/chop/nn/quantizers/minifloat.py) with
20,000 random values -- bias formula (2**(e-1)-1), subnormal handling, and
saturation direction all match exactly (that file also confirms MASE's
minifloat has no inf/nan representation and saturates on overflow, i.e.
the same semantics this file already uses). The one difference found: when
a mantissa rounds up exactly to the next power of two, this file's encode
correctly carries into the next exponent (e.g. -62.11 in FP(4,3) encodes
to -64.0, the objectively closer representable value); MASE's formula
instead clamps the shifted mantissa without cascading the carry, landing
on -60.0 -- a real (if rare and minor, ~6.7% relative in the worst case
found) inaccuracy in MASE's own function, not a design choice worth
copying. Left as-is rather than made bug-compatible.
"""

import math
import random
from deap import gp

#storing in a dictionary which store all the bitwidth split we encountered, so that they can be reused
_fp_types = {}

def FP(e, m):
    key = (e, m)
    if key not in _fp_types:
        _fp_types[key] = type(f"FP_e{e}_m{m}", (), {})
    return _fp_types[key]

def _bias(e):
    return (1 << (e - 1)) - 1

def format_max_val(e, m):
    """Largest finite magnitude FP(e, m) can represent. Exported so
    callers building on top of this format (e.g. benchmarks/block_minifloat.py
    choosing a block's shared exponent) don't have to re-derive the same
    formula."""
    bias = _bias(e)
    max_normal_exp = ((1 << e) - 1) - bias
    return (2.0 - 2.0 ** (-m)) * (2.0 ** max_normal_exp)

def bitsplit(bits, e, m):
    """Pure wire split, no arithmetic (Jianyi's naming)."""
    sign = (bits >> (e + m)) & 1
    exp_field = (bits >> m) & ((1 << e) - 1)
    frac = bits & ((1 << m) - 1)
    return sign, exp_field, frac

#encode / decode (bit pattern <-> real value)
def decode(bits, e, m):
    #highest value:480
    sign = (bits >> (e + m)) & 1
    exp_field = (bits >> m) & ((1 << e) - 1)
    frac = bits & ((1 << m) - 1)
    bias = _bias(e)
    if exp_field == 0:
        #zero / subnormal: no implicit leading 1
        value = (frac / (1 << m)) * (2.0 ** (1 - bias))
    else:
        #normal: implicit leading 1
        value = (1.0 + frac / (1 << m)) * (2.0 ** (exp_field - bias))
    return -value if sign else value

def encode(value, e, m):
    bias = _bias(e)
    max_exp_field = (1 << e) - 1# largest normal exponent field
    min_normal_exp = 1 - bias
    max_normal_exp = max_exp_field - bias
    max_val = (2.0 - 2.0 ** (-m)) * (2.0 ** max_normal_exp)

    sign = 1 if math.copysign(1.0, value) < 0 else 0
    v = abs(value)

    if v == 0.0:
        return sign << (e + m)

    if v >= max_val:
        # saturate to the largest finite magnitude
        return (sign << (e + m)) | (max_exp_field << m) | ((1 << m) - 1)

    mant, exp2 = math.frexp(v)            # v = mant * 2**exp2, 0.5 <= mant < 1
    exp = exp2 - 1                        # so that 1.0 <= significand < 2.0
    significand = mant * 2.0

    if exp < min_normal_exp:
        # subnormal: align to the smallest normal exponent
        shift = min_normal_exp - exp
        frac_scaled = (significand / (2 ** shift)) * (1 << m)
        exp_field = 0
    else:
        exp_field = exp - min_normal_exp + 1
        frac_scaled = (significand - 1.0) * (1 << m)

    frac_round = round(frac_scaled)
    if frac_round >= (1 << m):
        # mantissa rounded up into the next exponent
        frac_round = 0
        exp_field += 1
        if exp_field > max_exp_field:
            return (sign << (e + m)) | (max_exp_field << m) | ((1 << m) - 1)

    return (sign << (e + m)) | (exp_field << m) | frac_round

quantize = encode   # Jianyi's naming: binary_number = quantize(real_number)

#build pri for gp
def _make_fmul(fmt_a, fmt_b):
    ea, ma = fmt_a
    eb, mb = fmt_b
    eo, mo = ea + eb, ma + mb + 1
    def fmul(a_bits, b_bits):
        av = decode(a_bits, ea, ma)
        bv = decode(b_bits, eb, mb)
        return encode(av * bv, eo, mo)
    return fmul


def _make_fadd(fmt_a, fmt_b):
    ea, ma = fmt_a
    eb, mb = fmt_b
    eo, mo = max(ea, eb), max(ma, mb) + 1
    def fadd(a_bits, b_bits):
        av = decode(a_bits, ea, ma)
        bv = decode(b_bits, eb, mb)
        return encode(av + bv, eo, mo)
    return fadd


def _make_fsub(fmt_a, fmt_b):
    ea, ma = fmt_a
    eb, mb = fmt_b
    eo, mo = max(ea, eb), max(ma, mb) + 1
    def fsub(a_bits, b_bits):
        av = decode(a_bits, ea, ma)
        bv = decode(b_bits, eb, mb)
        return encode(av - bv, eo, mo)
    return fsub


def _make_fcast(src_fmt, dst_fmt):
    es, ms = src_fmt
    ed, md = dst_fmt
    def fcast(a_bits):
        v = decode(a_bits, es, ms)
        return encode(v, ed, md)
    return fcast

#primitives builder
def make_pset(ma_format, mb_format, output_format, include_addsub=False):
#(4, 3) for an E4M3-style FP8 format.
#include_addsub: fadd/fsub flood the pset (hundreds of variants) and
#their max(m)+1 output is not lossless for far-apart exponents; keep
#off until the supervisors confirm they are wanted in the grammar.
    ea, ma_m = ma_format
    eb, mb_m = mb_format
    eo, mo = output_format

    mul_fmt = (ea + eb, ma_m + mb_m + 1)
    cap_e, cap_m = max(mul_fmt[0], eo), max(mul_fmt[1], mo)

    needed = {ma_format, mb_format, output_format, mul_fmt}
    #narrowed variants of the inputs: BOTH mantissa and exponent can be
    #reduced (generic-tool requirement) -- these are the formats fcast
    #can shrink into before multiplying
    for (E, M) in (ma_format, mb_format):
        for e2 in range(2, E + 1):
            for m2 in range(0, M + 1):
                needed.add((e2, m2))

    prev = None
    while prev != needed:
        prev = set(needed)
        for x in list(prev):
            for y in list(prev):
                out_add = (max(x[0], y[0]), max(x[1], y[1]) + 1)
                if out_add[0] <= cap_e and out_add[1] <= cap_m:
                    needed.add(out_add)
                out_mul = (x[0] + y[0], x[1] + y[1] + 1)   #mul rule too,
                if out_mul[0] <= cap_e and out_mul[1] <= cap_m:  #so small
                    needed.add(out_mul)                    #fmuls exist

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [FP(*ma_format), FP(*mb_format)],
        FP(*output_format),
    )
    pset.renameArguments(ARG0="ma", ARG1="mb")

    def tag(fmt):
        return f"e{fmt[0]}m{fmt[1]}"

    #fmul:FP(x)*FP(y)=FP(x.e+y.e, x.m+y.m+1)
    mul_count = 0
    for x in needed:
        for y in needed:
            out = (x[0] + y[0], x[1] + y[1] + 1)
            if out in needed:
                pset.addPrimitive(_make_fmul(x, y), [FP(*x), FP(*y)], FP(*out),
                                   name=f"fmul_{tag(x)}_{tag(y)}")
                mul_count += 1

    add_count = 0
    if include_addsub:
        for x in needed:
            for y in needed:
                out = (max(x[0], y[0]), max(x[1], y[1]) + 1)
                if out in needed:
                    pset.addPrimitive(_make_fadd(x, y), [FP(*x), FP(*y)], FP(*out),
                                       name=f"fadd_{tag(x)}_{tag(y)}")
                    pset.addPrimitive(_make_fsub(x, y), [FP(*x), FP(*y)], FP(*out),
                                       name=f"fsub_{tag(x)}_{tag(y)}")
                    add_count += 1

    #fcast: three route classes so every useful path exists --
    # (a) narrowing (the lossy op), (b) widening (lossless, wires+zeros),
    # (c) mixed casts INTO the output format only (root reachability:
    #     e.g. product (8,3) -> output (5,4) narrows e but widens m)
    cast_count = 0
    for src in needed:
        for dst in needed:
            if dst == src:
                continue
            narrowing = dst[0] <= src[0] and dst[1] <= src[1]
            widening  = dst[0] >= src[0] and dst[1] >= src[1]
            into_out  = dst == output_format
            if narrowing or widening or into_out:
                pset.addPrimitive(_make_fcast(src, dst), [FP(*src)], FP(*dst),
                                   name=f"fcast_{tag(src)}_{tag(dst)}")
                cast_count += 1

    #terminals:0,1
    for w in needed:
        pset.addTerminal(encode(0.0, *w), FP(*w), name=f"zero_{tag(w)}")
        pset.addTerminal(encode(1.0, *w), FP(*w), name=f"one_{tag(w)}")

    print(f"[pset] formats={len(needed)}  fmul={mul_count}  "
          f"fadd/fsub={add_count}x2  fcast={cast_count}")
    return pset


#seed: exact product? cast down to the output format
def make_seed_str(ma_format, mb_format, output_format):
#seed=fcast(fmul(ma, mb), output_format)
    ea, ma_m = ma_format
    eb, mb_m = mb_format
    mul_fmt = (ea + eb, ma_m + mb_m + 1)

    def tag(fmt):
        return f"e{fmt[0]}m{fmt[1]}"

    mul_expr = f"fmul_{tag(ma_format)}_{tag(mb_format)}(ma, mb)"

    if mul_fmt == output_format:
        return mul_expr
    return f"fcast_{tag(mul_fmt)}_{tag(output_format)}({mul_expr})"


def make_narrow_seed_strs(ma_format, mb_format, output_format, single_axis_only=False):
    """single_axis_only=True gives only the seeds that narrow ma or mb
    alone, never both at once -- the full grid's *jointly* narrowed pairs
    are withheld, so a run_gp that can genuinely recombine two single-axis
    seeds has to construct the joint pairs itself instead of just
    selecting one that was handed to it (mirrors benchmarks/integer.py's
    make_side_seed_strs and benchmarks/mxint_hardware.py's
    make_narrow_seed_strs single_axis_only param)."""
    ea, ma_m = ma_format
    eb, mb_m = mb_format

    def tag(fmt):
        return f"e{fmt[0]}m{fmt[1]}"

    def build(a_fmt, b_fmt):
        mul_fmt = (a_fmt[0] + b_fmt[0], a_fmt[1] + b_fmt[1] + 1)
        a_arg = f"fcast_{tag(ma_format)}_{tag(a_fmt)}(ma)" if a_fmt != ma_format else "ma"
        b_arg = f"fcast_{tag(mb_format)}_{tag(b_fmt)}(mb)" if b_fmt != mb_format else "mb"
        mul_expr = f"fmul_{tag(a_fmt)}_{tag(b_fmt)}({a_arg}, {b_arg})"
        if mul_fmt == output_format:
            return mul_expr
        return f"fcast_{tag(mul_fmt)}_{tag(output_format)}({mul_expr})"

    seeds = []
    for m_a in range(0, ma_m + 1):
        for m_b in range(0, mb_m + 1):
            if m_a == ma_m and m_b == mb_m:
                continue   # that's make_seed_str's exact seed, skip the dup
            if single_axis_only and m_a != ma_m and m_b != mb_m:
                continue   # both narrowed at once -- withheld
            seeds.append(build((ea, m_a), (eb, m_b)))
    return seeds


def make_data(ma_format, mb_format, n_samples=100, seed=42,
              exclude_subnormal_inputs=False):
    ea, ma_m = ma_format
    eb, mb_m = mb_format
    rng = random.Random(seed)

    def draw(e, m):
        while True:
            bits = rng.randint(0, (1 << (1 + e + m)) - 1)
            if not exclude_subnormal_inputs:
                return bits
            if (bits >> m) & ((1 << e) - 1) != 0:
                return bits

    data = []
    for _ in range(n_samples):
        a_bits = draw(ea, ma_m)
        b_bits = draw(eb, mb_m)
        target = decode(a_bits, ea, ma_m) * decode(b_bits, eb, mb_m)
        data.append((a_bits, b_bits, target))
    return data

#Q:should we include add and sub in primitives? i haven't try but it may causes 
#Q: