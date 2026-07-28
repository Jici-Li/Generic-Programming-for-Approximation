"""MXInt block reference model and typed GP grammar.

Stage-1 scope:
  * a block shares one power-of-two scale, ``2 ** exponent``;
  * each block element is a signed integer mantissa;
  * GP searches the signed mantissa multiplier;
  * block quantisation and exponent addition remain fixed reference logic.

The separation is deliberate: it gives GP a stable target and an exact seed
before shared-exponent selection, requantisation, and accumulation are added
to the search space.

MASE correspondence: this block structure (one shared exponent, per-element
signed integer mantissa, no per-element exponent field) is also exactly
DeepWok/mase's block_fp.py (Microsoft Floating Point / MSFP) -- see
benchmarks/block_minifloat.py's module docstring for the fuller comparison. MSFP's
real shared-exponent formula differs from choose_shared_exponent below
(MSFP: exponent = clamp(ceil(log2(block_max)), -bias, bias-1) with
bias=2**(exponent_width-1)-1, no width-dependent shift folded in --
compare to this file's MASE-mxint_hardware.py-matching formula, which
*does* fold exponent_width-dependent bias in and, per the project report,
under-uses the mantissa range as a result). Adding an MSFP variant should
be cheap: same MXIntBlock/pset/search/verify, just a different
choose_shared_exponent implementation selectable alongside this one.
"""

from dataclasses import dataclass
import math
import random

from deap import gp


# ==========================================================
# MXInt reference representation
# ==========================================================
@dataclass(frozen=True)
class MXIntBlock:
    exponent: int
    mantissas: tuple[int, ...]
    element_bits: int

    @property
    def block_size(self):
        return len(self.mantissas)


def signed_range(bits):
    if bits < 2:
        raise ValueError("signed MXInt elements need at least 2 bits")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _clip_signed(value, bits):
    lo, hi = signed_range(bits)
    return max(lo, min(int(value), hi))


def choose_shared_exponent(values, exponent_width):
    """Shared exponent, matching MASE's mxint_quant_block exactly (DeepWok/
    mase, src/chop/nn/quantizers/mxint_hardware.py):

        exponent_bias = 2 ** (exponent_width - 1)
        exponent_max  = 2 ** exponent_width - 1 - exponent_bias
        exponent_min  = -exponent_bias
        exponent = clamp(ceil(log2(max_abs)) - exponent_bias, exponent_min, exponent_max)

    MASE's real formula takes no mantissa-width argument at all -- exponent_
    bias depends only on exponent_width. Unlike a "use the full mantissa
    range" formula (what this function computed before this was made MASE-
    bug-compatible), this one under-uses the mantissa's dynamic range
    whenever exponent_width is small relative to the mantissa width (e.g.
    MASE's own defaults width=12, exponent_width=6 make exponent_bias=32,
    which sends nearly every mantissa straight to the int_min/int_max clamp
    for any reasonable max_abs). That's real, faithfully-reproduced MASE
    behaviour, not a bug introduced here -- see the project report/
    discussion for the numeric trace that surfaced this.
    """
    if not values:
        raise ValueError("an MXInt block cannot be empty")
    exponent_bias = 2 ** (exponent_width - 1)
    exponent_max = 2 ** exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias
    max_abs = max(abs(float(value)) for value in values)
    if max_abs == 0.0:
        return max(exponent_min, min(0, exponent_max))
    exponent = math.ceil(math.log2(max_abs)) - exponent_bias
    return max(exponent_min, min(exponent, exponent_max))


def quantize_block(values, element_bits, exponent_width):
    """Quantise real values to one shared exponent and signed mantissas,
    matching MASE's mxint_quant_block: mantissa = clamp(floor(x / 2**exponent),
    int_min, int_max) -- floor, not round-to-nearest (MASE has no rounding
    bias step at all)."""
    values = tuple(float(value) for value in values)
    exponent = choose_shared_exponent(values, exponent_width)
    scale = 2.0 ** exponent
    mantissas = tuple(
        _clip_signed(math.floor(value / scale), element_bits)
        for value in values
    )
    return MXIntBlock(exponent, mantissas, element_bits)


def dequantize_block(block):
    scale = 2.0 ** block.exponent
    return tuple(mantissa * scale for mantissa in block.mantissas)


def reference_multiply_blocks(a, b):
    """Exact element-wise multiplication of two already-quantised blocks."""
    if a.block_size != b.block_size:
        raise ValueError("MXInt block sizes must match")
    return MXIntBlock(
        exponent=a.exponent + b.exponent,
        mantissas=tuple(
            ma * mb for ma, mb in zip(a.mantissas, b.mantissas)
        ),
        element_bits=a.element_bits + b.element_bits,
    )


def reference_dot_blocks(a, b):
    product = reference_multiply_blocks(a, b)
    return sum(product.mantissas) * (2.0 ** product.exponent)


# ==========================================================
# Typed signed-integer GP grammar
# ==========================================================
_sint_types = {}
_uint_types = {}


class Sign:
    """One-bit sign marker used by typed GP expressions."""


def SInt(bits):
    if bits not in _sint_types:
        _sint_types[bits] = type(f"SInt{bits}", (), {})
    return _sint_types[bits]


def UInt(bits):
    if bits not in _uint_types:
        _uint_types[bits] = type(f"UInt{bits}", (), {})
    return _uint_types[bits]


def _clip_unsigned(value, bits):
    return max(0, min(int(value), (1 << bits) - 1))


def _make_smul(x_bits, y_bits):
    def smul(a, b):
        return int(a) * int(b)
    return smul


def _make_sadd(x_bits, y_bits):
    out_bits = max(x_bits, y_bits) + 1

    def sadd(a, b):
        return _clip_signed(int(a) + int(b), out_bits)
    return sadd


def _make_ssub(x_bits, y_bits):
    out_bits = max(x_bits, y_bits) + 1

    def ssub(a, b):
        return _clip_signed(int(a) - int(b), out_bits)
    return ssub


def _make_scast(src_bits, dst_bits):
    """Widening is a numeric no-op (sign extension in hardware).

    Narrowing used to saturate straight to the destination range, which
    isn't the same operation as reducing precision: e.g. scast_4_2 sent
    6, 5, and 3 all to the same clipped value 1, destroying magnitude
    information for anything outside [-2, 1] instead of approximating it.
    Reducing precision while preserving magnitude is a right-shift (round
    to nearest, not floor -- see benchmarks/integer.py's make_seed_str
    fix for the same floor-vs-round distinction), with saturation kept
    only as a safety net for the rounding step's rare boundary overflow.
    """
    def scast(value):
        value = int(value)
        if dst_bits >= src_bits:
            return value
        shift = src_bits - dst_bits
        rounded = (value + (1 << (shift - 1))) >> shift
        return _clip_signed(rounded, dst_bits)
    return scast


def _make_ashr(bits, amount):
    def ashr(value):
        return int(value) >> amount
    return ashr


def _make_sshl(bits, amount):
    """Signed left shift, widening output by `amount` bits -- a free wire
    shift in hardware (compile-time-fixed amount), used to restore the
    scale a scast narrowing dropped before the narrowed operand feeds a
    multiply. Mirrors benchmarks/integer.py's _make_left_shift, and
    exists because narrowing an operand via scast then multiplying
    without compensating leaves the result systematically too small by
    exactly 2**amount (verified: scast_8_5(64)=8; 8*100=800 vs the true
    64*100=6400 -- 87.5% error; <<3 back onto the product recovers it
    exactly). No masking/clipping needed since the output type is always
    wider than the input, so this can never overflow."""
    def sshl(value):
        return int(value) << amount
    return sshl


def _make_abs_magnitude(bits):
    def abs_magnitude(value):
        return abs(int(value))
    return abs_magnitude


def _make_is_negative(bits):
    def is_negative(value):
        return int(value) < 0
    return is_negative


def _xor_sign(a, b):
    return bool(a) ^ bool(b)


def _make_apply_sign(bits):
    def apply_sign(sign, magnitude):
        value = -int(magnitude) if sign else int(magnitude)
        return _clip_signed(value, bits)
    return apply_sign


def _make_umul(x_bits, y_bits):
    out_bits = x_bits + y_bits

    def umul(a, b):
        return _clip_unsigned(int(a) * int(b), out_bits)
    return umul


def _make_uadd(x_bits, y_bits):
    out_bits = max(x_bits, y_bits) + 1

    def uadd(a, b):
        return _clip_unsigned(int(a) + int(b), out_bits)
    return uadd


def _make_ucast(src_bits, dst_bits):
    def ucast(value):
        return _clip_unsigned(value, dst_bits)
    return ucast


def _make_uhigh(src_bits, dst_bits):
    shift = src_bits - dst_bits

    def uhigh(value):
        return (_clip_unsigned(value, src_bits) >> shift) & (
            (1 << dst_bits) - 1
        )
    return uhigh


def _make_ulow(src_bits, dst_bits):
    def ulow(value):
        return _clip_unsigned(value, src_bits) & ((1 << dst_bits) - 1)
    return ulow


def _make_ulshift(src_bits, amount):
    out_bits = src_bits + amount

    def ulshift(value):
        return _clip_unsigned(int(value) << amount, out_bits)
    return ulshift


def make_pset(element_bits, output_bits=None):
    """Build a typed grammar for one MXInt mantissa multiplication."""
    if output_bits is None:
        output_bits = 2 * element_bits
    if output_bits < 2:
        raise ValueError("output_bits must be at least 2")

    cap = max(output_bits, 2 * element_bits) + 2
    needed = set(range(2, element_bits + 1))
    needed.add(output_bits)

    previous = None
    while previous != needed:
        previous = set(needed)
        for x in previous:
            for y in previous:
                if x + y <= cap:
                    needed.add(x + y)
                add_out = max(x, y) + 1
                if add_out <= cap:
                    needed.add(add_out)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [SInt(element_bits), SInt(element_bits)],
        SInt(output_bits),
    )
    pset.renameArguments(ARG0="ma", ARG1="mb")

    for x in sorted(needed):
        for y in sorted(needed):
            mul_out = x + y
            if mul_out in needed:
                pset.addPrimitive(
                    _make_smul(x, y),
                    [SInt(x), SInt(y)],
                    SInt(mul_out),
                    name=f"smul_{x}_{y}",
                )
            add_out = max(x, y) + 1
            if add_out in needed:
                pset.addPrimitive(
                    _make_sadd(x, y),
                    [SInt(x), SInt(y)],
                    SInt(add_out),
                    name=f"sadd_{x}_{y}",
                )
                pset.addPrimitive(
                    _make_ssub(x, y),
                    [SInt(x), SInt(y)],
                    SInt(add_out),
                    name=f"ssub_{x}_{y}",
                )

    for src in sorted(needed):
        for dst in sorted(needed):
            if src != dst:
                pset.addPrimitive(
                    _make_scast(src, dst),
                    [SInt(src)],
                    SInt(dst),
                    name=f"scast_{src}_{dst}",
                )
        for amount in range(1, min(src, element_bits)):
            pset.addPrimitive(
                _make_ashr(src, amount),
                [SInt(src)],
                SInt(src),
                name=f"ashr_{src}_by{amount}",
            )
        for amount in range(1, cap - src + 1):
            if src + amount in needed:
                pset.addPrimitive(
                    _make_sshl(src, amount),
                    [SInt(src)],
                    SInt(src + amount),
                    name=f"sshl_{src}_by{amount}",
                )

    for bits in sorted(needed):
        pset.addTerminal(0, SInt(bits), name=f"zero_{bits}")
        pset.addTerminal(1, SInt(bits), name=f"one_{bits}")
        pset.addTerminal(-1, SInt(bits), name=f"minus_one_{bits}")

    # Signed inputs are converted to an unsigned magnitude before splitting.
    # This avoids treating a negative two's-complement bit pattern as if it
    # were an ordinary positive high/low decomposition.
    for bits in sorted(needed):
        pset.addPrimitive(
            _make_abs_magnitude(bits),
            [SInt(bits)],
            UInt(bits),
            name=f"abs_s{bits}_u{bits}",
        )
        pset.addPrimitive(
            _make_is_negative(bits),
            [SInt(bits)],
            Sign,
            name=f"is_negative_{bits}",
        )
    pset.addPrimitive(_xor_sign, [Sign, Sign], Sign, name="xor_sign")

    uint_widths = set(range(1, cap + 1))
    for x in sorted(uint_widths):
        for y in sorted(uint_widths):
            mul_out = x + y
            if mul_out in uint_widths:
                pset.addPrimitive(
                    _make_umul(x, y),
                    [UInt(x), UInt(y)],
                    UInt(mul_out),
                    name=f"umul_{x}_{y}",
                )
            add_out = max(x, y) + 1
            if add_out in uint_widths:
                pset.addPrimitive(
                    _make_uadd(x, y),
                    [UInt(x), UInt(y)],
                    UInt(add_out),
                    name=f"uadd_{x}_{y}",
                )

    for src in sorted(uint_widths):
        for dst in sorted(uint_widths):
            if src != dst:
                pset.addPrimitive(
                    _make_ucast(src, dst),
                    [UInt(src)],
                    UInt(dst),
                    name=f"ucast_{src}_{dst}",
                )
        for amount in range(1, cap - src + 1):
            pset.addPrimitive(
                _make_ulshift(src, amount),
                [UInt(src)],
                UInt(src + amount),
                name=f"ulshift_{src}_by{amount}",
            )
        pset.addTerminal(0, UInt(src), name=f"uzero_{src}")
        pset.addTerminal(1, UInt(src), name=f"uone_{src}")

    # The first partial-product decomposition splits each input magnitude in
    # half.  More split points can be registered later without changing the
    # seed interface.
    if element_bits % 2 == 0:
        half = element_bits // 2
        pset.addPrimitive(
            _make_uhigh(element_bits, half),
            [UInt(element_bits)],
            UInt(half),
            name=f"uhigh_{element_bits}_{half}",
        )
        pset.addPrimitive(
            _make_ulow(element_bits, half),
            [UInt(element_bits)],
            UInt(half),
            name=f"ulow_{element_bits}_{half}",
        )

    pset.addPrimitive(
        _make_apply_sign(output_bits),
        [Sign, UInt(output_bits)],
        SInt(output_bits),
        name=f"apply_sign_{output_bits}",
    )
    pset.addTerminal(False, Sign, name="positive_sign")
    pset.addTerminal(True, Sign, name="negative_sign")

    return pset


# ==========================================================
# Exact and search-helping seeds
# ==========================================================
def make_seed_str(element_bits, output_bits=None):
    if output_bits is None:
        output_bits = 2 * element_bits
    product_bits = 2 * element_bits
    product = f"smul_{element_bits}_{element_bits}(ma, mb)"
    if product_bits == output_bits:
        return product
    return f"scast_{product_bits}_{output_bits}({product})"


def make_narrow_seed_strs(element_bits, output_bits=None, single_axis_only=False):
    """Seeds that try smaller signed input multipliers before widening.

    single_axis_only=True gives only the seeds that narrow one side while
    leaving the other at full precision -- e.g. for element_bits=4 that's
    (a_bits, b_bits) in {(2,4),(3,4),(4,2),(4,3)}, never both narrowed at
    once. The full grid also contains every *jointly* narrowed pair
    (like (2,3)) pre-built; single_axis_only intentionally withholds those,
    so a run_gp that can genuinely recombine two single-axis seeds has to
    construct the joint pairs itself instead of just picking one that was
    handed to it -- see search/minifloat_search.py's cx_combine_sides for
    why standard crossover often can't do this recombination on its own.
    """
    if output_bits is None:
        output_bits = 2 * element_bits

    seeds = []
    for a_bits in range(2, element_bits + 1):
        for b_bits in range(2, element_bits + 1):
            if a_bits == element_bits and b_bits == element_bits:
                continue
            if single_axis_only and a_bits != element_bits and b_bits != element_bits:
                continue
            a_arg = (
                "ma" if a_bits == element_bits
                else f"scast_{element_bits}_{a_bits}(ma)"
            )
            b_arg = (
                "mb" if b_bits == element_bits
                else f"scast_{element_bits}_{b_bits}(mb)"
            )
            # narrowing an operand via scast drops (element_bits-a_bits)
            # low bits (round-then-shift, see _make_scast) -- multiplying
            # without compensating leaves the result systematically too
            # small by 2**total_shift (see _make_sshl's docstring for the
            # verified numeric example). sshl restores it, same as
            # integer's blockmul_k's compensating <<(2k).
            total_shift = (element_bits - a_bits) + (element_bits - b_bits)
            product_bits = a_bits + b_bits
            product = f"smul_{a_bits}_{b_bits}({a_arg}, {b_arg})"
            if total_shift > 0:
                product = f"sshl_{product_bits}_by{total_shift}({product})"
                product_bits += total_shift
            if product_bits != output_bits:
                product = f"scast_{product_bits}_{output_bits}({product})"
            seeds.append(product)
    return seeds


def make_partial_seed_strs(element_bits, output_bits=None):
    """Return one exact and several approximate signed partial-product seeds.

    The inputs are converted to unsigned magnitudes, split into equal high and
    low halves, and their product sign is restored at the root.  The exact
    decomposition is useful as a reachability check; the remaining seeds omit
    selected low-value partial products to give GP structured approximation
    starting points.
    """
    if output_bits is None:
        output_bits = 2 * element_bits
    if element_bits % 2 != 0:
        return None, []
    if output_bits != 2 * element_bits:
        return None, []

    n = element_bits
    h = n // 2
    product_bits = 2 * h
    cross_bits = product_bits + h

    a_mag = f"abs_s{n}_u{n}(ma)"
    b_mag = f"abs_s{n}_u{n}(mb)"
    a_high = f"uhigh_{n}_{h}({a_mag})"
    a_low = f"ulow_{n}_{h}({a_mag})"
    b_high = f"uhigh_{n}_{h}({b_mag})"
    b_low = f"ulow_{n}_{h}({b_mag})"
    sign = f"xor_sign(is_negative_{n}(ma), is_negative_{n}(mb))"

    hh = (
        f"ulshift_{product_bits}_by{2*h}("
        f"umul_{h}_{h}({a_high}, {b_high}))"
    )
    hl = (
        f"ulshift_{product_bits}_by{h}("
        f"umul_{h}_{h}({a_high}, {b_low}))"
    )
    lh = (
        f"ulshift_{product_bits}_by{h}("
        f"umul_{h}_{h}({a_low}, {b_high}))"
    )
    ll = f"umul_{h}_{h}({a_low}, {b_low})"

    def widen(expr, src_bits):
        if src_bits == output_bits:
            return expr
        return f"ucast_{src_bits}_{output_bits}({expr})"

    def add_at_output(left, right):
        summed = f"uadd_{output_bits}_{output_bits}({left}, {right})"
        return f"ucast_{output_bits + 1}_{output_bits}({summed})"

    hh_w = widen(hh, output_bits)
    hl_w = widen(hl, cross_bits)
    lh_w = widen(lh, cross_bits)
    ll_w = widen(ll, product_bits)

    hh_hl = add_at_output(hh_w, hl_w)
    drop_ll_magnitude = add_at_output(hh_hl, lh_w)
    exact_magnitude = add_at_output(drop_ll_magnitude, ll_w)

    exact = f"apply_sign_{output_bits}({sign}, {exact_magnitude})"
    approximate = [
        f"apply_sign_{output_bits}({sign}, {drop_ll_magnitude})",
        f"apply_sign_{output_bits}({sign}, {hh_hl})",
        f"apply_sign_{output_bits}({sign}, {add_at_output(hh_w, lh_w)})",
        f"apply_sign_{output_bits}({sign}, {hh_w})",
    ]
    return exact, approximate


# ==========================================================
# Block training data
# ==========================================================
def make_data(element_bits, exponent_width, block_size=4, n_blocks=100, seed=42):
    """Generate real blocks, quantise them, and store exact MXInt targets.

    Each item is ``(a_block, b_block, target_block)``.  The target is exact
    multiplication of the quantised inputs, so the exact GP seed has zero
    arithmetic error; input quantisation error remains a separate metric for
    a later end-to-end experiment.

    exponent_width: shared-exponent field width, passed straight through to
    choose_shared_exponent (see its docstring -- this is MASE's own knob,
    decoupled from element_bits).

    No block-level "make every element land in the upper mantissa range"
    rejection sampling here (there used to be one, gated by a mantissa_
    domain flag): under MASE's real shared-exponent formula, a block's
    exponent tracks only its own max element, so non-max elements
    legitimately end up with small mantissas -- that's an inherent property
    of the format, not something retry-until-lucky sampling can fix (it
    just burns 200 tries and returns a non-conforming block anyway). The
    near-zero-mantissa filtering that used to happen here at block-draw
    time now happens per-element at evaluation time instead -- see
    search/mxint_search.py's _block_error.

    The magnitude draw range is deliberately kept inside choose_shared_
    exponent's *unclamped* zone for this exponent_width (roughly
    [2**0, 2**(2*exponent_bias-1)], exponent_bias=2**(exponent_width-1)):
    magnitudes outside that range hit the exponent clamp and can leave an
    *entire block's* mantissas uniformly tiny (e.g. every element in
    {-1,0,1}), which no amount of within-block filtering can fix, since
    there's no large element left to compare against -- verified: block-
    size=4, exponent_width=3, 100 blocks, drawing magnitude exponents from
    the old fixed [-4,4] put 23% of blocks entirely under max|mantissa|=4;
    narrowing the draw to this format's unclamped zone drops that to 0%.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rng = random.Random(seed)
    data = []
    exponent_bias = 2 ** (exponent_width - 1)
    mag_lo, mag_hi = 0, 2 * exponent_bias - 1

    for _ in range(n_blocks):
        # A random power-of-two magnitude produces blocks across several
        # shared exponents while remaining deterministic and easy to inspect.
        a_magnitude = 2.0 ** rng.randint(mag_lo, mag_hi)
        b_magnitude = 2.0 ** rng.randint(mag_lo, mag_hi)
        a_values = [rng.uniform(-a_magnitude, a_magnitude) for _ in range(block_size)]
        b_values = [rng.uniform(-b_magnitude, b_magnitude) for _ in range(block_size)]
        a = quantize_block(a_values, element_bits, exponent_width)
        b = quantize_block(b_values, element_bits, exponent_width)
        data.append((a, b, reference_multiply_blocks(a, b)))

    return data


def make_exhaustive_mantissa_data(element_bits, mantissa_domain=False):
    """All signed mantissa pairs, wrapped as one-element MXInt blocks.

    mantissa_domain: restrict to |ma|, |mb| in the upper half of the
    representable magnitude range (mirrors benchmarks/integer.py's
    mantissa_domain and benchmarks/minifloat.py's
    exclude_subnormal_inputs). Without it, any mantissa narrowed below
    element_bits rounds near-zero values (e.g. 1, -1, 2) straight to 0,
    which is a genuine, correctly-computed answer -- but relative error
    against a nonzero target is then undefined/100% for every one of
    those pairs, swamping the exhaustive worst-case error with the same
    "narrowing a near-zero value is catastrophic in relative terms"
    effect documented for integer and minifloat, not a real difference
    between candidates.
    """
    lo, hi = signed_range(element_bits)
    threshold = 1 << max(0, element_bits - 2)
    data = []
    for ma in range(lo, hi + 1):
        if mantissa_domain and abs(ma) < threshold:
            continue
        for mb in range(lo, hi + 1):
            if mantissa_domain and abs(mb) < threshold:
                continue
            a = MXIntBlock(0, (ma,), element_bits)
            b = MXIntBlock(0, (mb,), element_bits)
            data.append((a, b, reference_multiply_blocks(a, b)))
    return data


def self_check():
    """Cheap reference checks, runnable without starting evolution."""
    block = quantize_block([3.0, 1.0, 0.25, -2.0], 4, exponent_width=2)
    assert block.exponent == 0
    assert block.mantissas == (3, 1, 0, -2)
    assert dequantize_block(block) == (3.0, 1.0, 0.0, -2.0)

    other = quantize_block([1.0, 2.0, -1.0, 0.5], 4, exponent_width=2)
    product = reference_multiply_blocks(block, other)
    assert product.exponent == block.exponent + other.exponent
    assert dequantize_block(product) == tuple(
        a * b
        for a, b in zip(dequantize_block(block), dequantize_block(other))
    )
