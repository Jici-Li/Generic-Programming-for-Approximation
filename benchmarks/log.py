"""Log-domain (LNS -- Logarithmic Number System) reference model and typed
GP grammar -- DeepWok/mase's log.py (src/chop/nn/quantizers/log.py).

A log-quantized value is ``sign * 2**exponent`` -- no mantissa at all, the
exponent (a signed integer) carries the entire representation. Exact
multiplication is therefore just sign XOR and exponent ADD:
``log(a*b) = log(a) + log(b)``. GP here searches an approximate signed
integer ADDER for the exponent field, not a multiplier -- a genuinely
different circuit family from every other benchmark in this project
(int/minifloat/mxint/block_fp/block_minifloat all center on smul).

Why "narrow an operand, then add" (the pattern used everywhere else in
this project) does not transfer here: dequantizing is ``2**exponent``, so
an *absolute* error of eps in the exponent becomes a *multiplicative*
2**eps error in the dequantized value -- off by exactly 1 is already a
100% relative error, off by 0.5 is 41%. Operand-width narrowing has no
gentle regime here the way mantissa narrowing does elsewhere. The
meaningful approximation axis instead is the ADDER'S OWN CARRY CHAIN:
splitting the add into K-bit segments and dropping the carry that would
normally propagate across each segment boundary (a real, established
approximate-adder technique -- e.g. the "Error Tolerant Adder" family).
Smaller segments = cheaper hardware (fewer/no cross-segment carry links)
but a coarser, segment-local notion of "correct".
"""

from dataclasses import dataclass
import math

from deap import gp

from benchmarks.mxint_hardware import SInt, signed_range, _clip_signed


# ==========================================================
# LogValue reference representation
# ==========================================================
@dataclass(frozen=True)
class LogValue:
    sign: bool     # True = negative
    exponent: int  # signed integer: log2 of the magnitude


def quantize(value, width, exponent_bias=None):
    """Matches MASE's _log_quantize: exponent_bits = width - 1 (1 sign bit
    + (width-1) exponent bits), round-to-nearest on log2(|value|)."""
    exponent_bits = width - 1
    if exponent_bias is None:
        exponent_bias = 2 ** (exponent_bits - 1) - 1
    exponent_max = 2 ** exponent_bits - 1 - exponent_bias
    exponent_min = -exponent_bias
    sign = value < 0
    v = abs(value)
    if v == 0.0:
        exponent = exponent_min
    else:
        exponent = round(math.log2(v))
    exponent = max(exponent_min, min(exponent, exponent_max))
    return LogValue(sign, exponent)


def dequantize(lv):
    value = 2.0 ** lv.exponent
    return -value if lv.sign else value


def reference_multiply(a, b):
    """Exact: log(a*b) = log(a) + log(b), sign = XOR. Unlike every other
    format's reference_multiply, this is lossless given already-quantised
    inputs -- integer addition of the exponents introduces no rounding at
    all (there's no mantissa to round)."""
    return LogValue(a.sign != b.sign, a.exponent + b.exponent)


# ==========================================================
# Typed signed-integer GP grammar: approximate ADDER, not multiplier
# ==========================================================
def _make_sadd_split(bits, split_bits):
    """Single-split, carry-truncated signed add: split the two's-
    complement bit pattern into exactly two chunks -- the low split_bits
    bits and the high (bits-split_bits) bits -- add each chunk pair
    modulo 2**chunk_width (discarding the carry crossing the one split
    point), and concatenate the two chunks back into a `bits`-wide
    result.

    split_bits == bits is the *exact* signed add (one chunk, the carry
    has nowhere to go). split_bits < bits drops the one carry that would
    cross the split, at cost bits-1 regardless of *where* the split sits
    (see search/log_search.py's node_cost) -- but *where* it sits controls
    the error a lot: when that carry is genuinely needed, the result
    undershoots the true exponent by exactly 2**split_bits, i.e. relative
    error 1 - 2**-(2**split_bits) -- small split_bits (truncating only
    the low bits) gives a small, bounded error; split_bits near the top
    approaches (but never reaches) 100%.

    This replaced an earlier "uniform K-bit segments tiling the whole
    width" version: for small K that tiled into 3+ segments, multiple
    simultaneously-droppable carries could compound into UNBOUNDED error
    (verified: real counterexamples with relative error ~10**19) -- a
    materially different, much riskier failure mode than this single-
    split design's bounded-under-100% one. See benchmarks/log.py's
    module docstring and the project report/discussion for the numeric
    trace."""
    def sadd_split(a, b):
        a = int(a) & ((1 << bits) - 1)  # raw two's-complement bit pattern
        b = int(b) & ((1 << bits) - 1)
        low_mask = (1 << split_bits) - 1
        low_sum = ((a & low_mask) + (b & low_mask)) & low_mask
        high_w = bits - split_bits
        high_mask = (1 << high_w) - 1
        high_sum = (((a >> split_bits) & high_mask) + ((b >> split_bits) & high_mask)) & high_mask
        result = (high_sum << split_bits) | low_sum
        return _clip_signed(result if result < (1 << (bits - 1)) else result - (1 << bits), bits)
    return sadd_split


def _make_scast(src_bits, dst_bits):
    """Same round-then-shift-then-clip semantics as benchmarks/mxint_hardware
    .py's _make_scast (widening is a numeric no-op; narrowing rounds)."""
    def scast(value):
        value = int(value)
        if dst_bits >= src_bits:
            return value
        shift = src_bits - dst_bits
        rounded = (value + (1 << (shift - 1))) >> shift
        return _clip_signed(rounded, dst_bits)
    return scast


def _segmentation_max_error(exponent_bits, add_bits, split_bits):
    """Exhaustively compute the true worst-case relative error this
    single-split configuration produces, over every representable
    exponent_bits-wide (ea, eb) pair -- used to empirically confirm which
    split points are safe to offer GP at all (see make_pset).

    A single split (this file's *current* _make_sadd_split) is bounded by
    construction (relative error 1 - 2**-(2**split_bits), always < 100%
    for any split_bits < bits) -- unlike an earlier "uniform K-bit
    segments tiling the whole width" design this replaced, where small K
    created 3+ simultaneous drop points and *unbounded* error (verified:
    real counterexamples ~10**19 relative error). This exhaustive check
    is kept anyway as a direct, assumption-free confirmation rather than
    trusting the hand derivation alone -- that hand derivation was wrong
    twice already for the previous design."""
    lo, hi = signed_range(exponent_bits)
    fn = _make_sadd_split(add_bits, split_bits)
    max_err = 0.0
    for ea in range(lo, hi + 1):
        for eb in range(lo, hi + 1):
            exact = ea + eb
            approx = fn(ea, eb)
            d = approx - exact
            err = (2.0 ** d - 1.0) if d >= 0 else (1.0 - 2.0 ** d)
            if err > max_err:
                max_err = err
    return max_err


def _is_safe_segmentation(exponent_bits, add_bits, split_bits, ceiling=1e6):
    """A split point is offered to GP only if its true worst-case error
    (see _segmentation_max_error) stays under a large sanity ceiling.
    Every single-split point is expected to pass this (see
    _segmentation_max_error's docstring); kept as a defensive check
    rather than assumed."""
    return _segmentation_max_error(exponent_bits, add_bits, split_bits) < ceiling


def make_pset(exponent_bits, output_bits=None):
    """Build a typed grammar for one log-domain exponent addition.

    exponent_bits: width of each input's exponent field.
    output_bits: width the circuit must produce (defaults to
    exponent_bits + 1, room for the one extra bit an *exact* signed add
    can need -- segmented/approximate adds never need it since dropping a
    carry can only shrink the represented range, but the type signature
    stays the same for every seg_bits choice so GP can mix them freely).
    """
    if output_bits is None:
        output_bits = exponent_bits + 1

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [SInt(exponent_bits), SInt(exponent_bits)],
        SInt(output_bits),
    )
    pset.renameArguments(ARG0="ea", ARG1="eb")

    # sadd_split{K}_{bits}: only "safe" split points (see
    # _is_safe_segmentation) are offered -- expected to be every K, kept
    # as a defensive filter rather than assumed. All split_bits values
    # share the same SInt(bits) -> SInt(bits) signature, so mixing them
    # is just picking a different primitive at the same tree position.
    add_bits = output_bits  # the add happens at output width; scast below
                             # narrows an operand up to it beforehand
    for split_bits in range(1, add_bits + 1):
        if not _is_safe_segmentation(exponent_bits, add_bits, split_bits):
            continue
        pset.addPrimitive(
            _make_sadd_split(add_bits, split_bits),
            [SInt(add_bits), SInt(add_bits)],
            SInt(add_bits),
            name=f"sadd_split{split_bits}_{add_bits}",
        )

    if exponent_bits != add_bits:
        pset.addPrimitive(
            _make_scast(exponent_bits, add_bits),
            [SInt(exponent_bits)],
            SInt(add_bits),
            name=f"scast_{exponent_bits}_{add_bits}",
        )
        # SInt(exponent_bits) otherwise has terminals but no primitive
        # that *produces* it (only scast, which *consumes* it) -- DEAP's
        # typed tree generation/mutation can still ask to grow a non-
        # terminal node of that type (e.g. regenerating a fresh subtree
        # at an existing ea/eb leaf during mutation) and crashes with
        # "Cannot choose from an empty sequence" when it does. A trivial
        # negate closes that gap generically without changing what the
        # exact/safe-segment seeds compute (they never use it).
        pset.addPrimitive(
            lambda v: _clip_signed(-int(v), exponent_bits),
            [SInt(exponent_bits)],
            SInt(exponent_bits),
            name=f"neg_{exponent_bits}",
        )

    for bits in (exponent_bits, add_bits):
        pset.addTerminal(0, SInt(bits), name=f"zero_{bits}")
        pset.addTerminal(1, SInt(bits), name=f"one_{bits}")
        pset.addTerminal(-1, SInt(bits), name=f"minus_one_{bits}")

    return pset


def make_seed_str(exponent_bits, output_bits=None):
    """Exact seed: full-width (split_bits == add width, i.e. no split at
    all) signed add."""
    if output_bits is None:
        output_bits = exponent_bits + 1
    add_bits = output_bits
    ea = "ea" if exponent_bits == add_bits else f"scast_{exponent_bits}_{add_bits}(ea)"
    eb = "eb" if exponent_bits == add_bits else f"scast_{exponent_bits}_{add_bits}(eb)"
    return f"sadd_split{add_bits}_{add_bits}({ea}, {eb})"


def make_seg_seed_strs(exponent_bits, output_bits=None):
    """One seed per *safe* split point (see _is_safe_segmentation) below
    add_bits (add_bits itself is the exact seed from make_seed_str) -- the
    approximate-adder analogue of every other benchmarks module's
    make_narrow_seed_strs. There's only one axis here (where to split),
    not two independently-narrowable operands, so there's no single-axis/
    joint-combo split to withhold -- every safe split point is handed
    over directly as a hint."""
    if output_bits is None:
        output_bits = exponent_bits + 1
    add_bits = output_bits
    ea = "ea" if exponent_bits == add_bits else f"scast_{exponent_bits}_{add_bits}(ea)"
    eb = "eb" if exponent_bits == add_bits else f"scast_{exponent_bits}_{add_bits}(eb)"
    return [
        f"sadd_split{split_bits}_{add_bits}({ea}, {eb})"
        for split_bits in range(1, add_bits)
        if _is_safe_segmentation(exponent_bits, add_bits, split_bits)
    ]


# ==========================================================
# Training / exhaustive data
# ==========================================================
def make_exhaustive_exponent_data(exponent_bits):
    """Every signed exponent pair -- small domain (a couple thousand at
    most for realistic exponent_bits), exhaustive is cheap and exact, no
    need for random sampling the way real-valued formats need it."""
    lo, hi = signed_range(exponent_bits)
    data = []
    for ea in range(lo, hi + 1):
        for eb in range(lo, hi + 1):
            a = LogValue(False, ea)
            b = LogValue(False, eb)
            data.append((ea, eb, reference_multiply(a, b).exponent))
    return data


def self_check():
    """Cheap reference checks, runnable without starting evolution."""
    a = quantize(8.0, width=8)     # 2**3
    b = quantize(0.5, width=8)     # 2**-1
    assert a.exponent == 3 and not a.sign
    assert b.exponent == -1 and not b.sign
    product = reference_multiply(a, b)
    assert product.exponent == 2 and not product.sign
    assert dequantize(product) == 4.0

    neg = quantize(-2.0, width=8)
    assert neg.sign and neg.exponent == 1
    signed_product = reference_multiply(a, neg)
    assert signed_product.sign  # positive * negative = negative

    exact_add = _make_sadd_split(6, 6)  # split_bits==bits: must be an exact signed add
    for x in range(-32, 32):
        for y in range(-32, 32):
            expected = _clip_signed(x + y, 6)
            if abs(x + y) > 31:
                continue  # exact add can only be tested where it doesn't overflow 6 bits
            assert exact_add(x, y) == expected, (x, y, exact_add(x, y), expected)

    # a genuine single split: dropping the carry into bit 2 undershoots
    # by exactly 2**2=4 when that carry is needed, e.g. 3+3=6 wants to
    # carry out of the low 2 bits (3+3=6 doesn't fit in 2 bits) into the
    # high part, which split_bits=2 discards.
    split2 = _make_sadd_split(6, 2)
    assert split2(3, 3) == 6 - 4
