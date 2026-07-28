"""Plain signed/unsigned integer multiplier reference model and typed GP
grammar.

MASE correspondence: DeepWok/mase's integer.py (fixed-point Qm.n, scale=
2**frac_width, round-or-floor, clamp) and scale_integer.py (per-tensor
scale=2**(width-1)/x_max, round, clamp -- an arbitrary real scale, not a
power of two) both reduce, at the element-multiplier level, to exactly
this file: multiplying the two integer mantissas is bit-identical
regardless of where the implied binary point sits or whether the outer
scale is a power of two or an arbitrary real number -- either way it's a
constant factor applied outside the integer multiply, and (same reasoning
as benchmarks/mxint_hardware.py's block-exponent argument) it cancels out of
the relative-error ratio. Neither MASE file needed a new benchmark/search/
verify module; this one already covers the multiplier hardware for both.
What neither this file nor those two model yet is the *quantize-a-real-
value-into-fixed-point* step itself (make_data here draws raw integers
directly, not real values rounded/floored into Qm.n) -- that's a separate,
not-yet-built concern from the multiplier search this file does.
"""

import random
from deap import gp


# ==========================================================
# Dynamic bitwidth types
# ==========================================================
_bv_types = {}

def BV(n):
    if n not in _bv_types:
        _bv_types[n] = type(f"BV{n}", (), {})
    return _bv_types[n]


# ==========================================================
# Python-level implementations (for GP evaluation)
# ==========================================================
def _mask(n):
    return (1 << n) - 1


def _make_mul(x_bits, y_bits):
    out_bits = x_bits + y_bits
    def mul(a, b):
        return (a * b) & _mask(out_bits)
    return mul


def _make_add(x_bits, y_bits):
    out_bits = max(x_bits, y_bits) + 1
    def add(a, b):
        return (a + b) & _mask(out_bits)
    return add


def _make_sub(x_bits, y_bits):
    out_bits = max(x_bits, y_bits) + 1
    def sub(a, b):
        return (a - b) & _mask(out_bits)
    return sub


def _make_and(x_bits, y_bits):
    out_bits = max(x_bits, y_bits)
    def and_op(a, b):
        return (a & b) & _mask(out_bits)
    return and_op


def _make_or(x_bits, y_bits):
    out_bits = max(x_bits, y_bits)
    def or_op(a, b):
        return (a | b) & _mask(out_bits)
    return or_op


def _make_left_shift(x_bits, k):
    out_bits = x_bits + k
    def lshift(a):
        return (a << k) & _mask(out_bits)
    return lshift


def _make_right_shift(x_bits, k):
    def rshift(a):
        return (a >> k) & _mask(x_bits)
    return rshift


def _make_block_mul(ma_bits, mb_bits, k):
    """Drop the low k bits of each operand before multiplying, then shift
    the (narrower) product back into position -- the 'high-only' block
    decomposition (benchmarks/integer.py's make_block_seed_strs) as one
    atomic, directly-parameterized primitive instead of a fixed 4-node
    trunc+rshift+mul+lshift recipe GP has to assemble from scratch."""
    out_bits = ma_bits + mb_bits
    def block_mul(a, b):
        aH = a >> k
        bH = b >> k
        return ((aH * bH) << (2 * k)) & _mask(out_bits)
    return block_mul


def _make_cross_mul_hi_lo(ma_bits, mb_bits, k):
    """aH*bL, shifted to line up with block_mul_k's aH*bH term -- one of
    the two 'cross' terms block_mul_k drops. Companion to block_mul_k so
    GP can add cross terms back in (recovering accuracy) without having
    to assemble trunc+rshift+mul+lshift by hand for each one."""
    out_bits = ma_bits + mb_bits
    def cross_mul(a, b):
        aH = a >> k
        bL = b & _mask(k)
        return ((aH * bL) << k) & _mask(out_bits)
    return cross_mul


def _make_cross_mul_lo_hi(ma_bits, mb_bits, k):
    """aL*bH, shifted to line up with block_mul_k's aH*bH term -- the other
    cross term."""
    out_bits = ma_bits + mb_bits
    def cross_mul(a, b):
        aL = a & _mask(k)
        bH = b >> k
        return ((aL * bH) << k) & _mask(out_bits)
    return cross_mul


def _make_ite(val_bits):
    def ite_fn(cond, a, b):
        return a if cond else b
    return ite_fn


def _make_trunc(src_bits, dst_bits):
    def trunc(a):
        return a & _mask(dst_bits)
    return trunc


def _make_rcast(src_bits, dst_bits):
    """Saturating round-to-nearest narrow: unlike trunc (a plain bit-mask
    that *wraps* when the input's high bits are set), this rounds first
    then *clips* to the destination range -- matches MASE's integer.py,
    which supports both integer_quantizer (round) and
    integer_floor_quantizer (floor) variants, and matches the round-then-
    clip pattern benchmarks/mxint_hardware.py's scast and benchmarks/
    minifloat.py's encode already use. trunc alone can't express this:
    a naive "add half-ulp then trunc" wraps on overflow instead of
    clamping (verified: rounding ma=15 by 2 bits wraps to 0, a 100% error,
    worse than plain floor truncation -- see the project report/discussion
    for the numeric trace), because trunc has no saturating clip, only a
    bit-mask."""
    shift = src_bits - dst_bits
    max_val = (1 << dst_bits) - 1

    def rcast(a):
        rounded = (int(a) + (1 << (shift - 1))) >> shift
        return min(rounded, max_val)
    return rcast


# ==========================================================
# Primitive set builder
# ==========================================================
def make_pset(ma_bits, mb_bits, output_bits, max_shift=None):
    """
    Build typed PrimitiveSet for approximate multiplier synthesis.
    ma:  BV(ma_bits)
    mb:  BV(mb_bits)
    out: BV(output_bits)
    """
    if max_shift is None:
        max_shift = max(ma_bits, mb_bits)

    # Bitwidths we need to support
    needed = set()
    needed.add(1)
    needed.add(ma_bits)
    needed.add(mb_bits)
    needed.add(output_bits)
    needed.add(ma_bits + mb_bits)  # mul output
    # every width up to max(ma_bits, mb_bits) -- so a single operand can be
    # narrowed to any intermediate width while the other stays exact (the
    # asymmetric, one-side-only analogue of blockmul_k's symmetric drop);
    # without this, only a handful of widths derived from the shift/add
    # closure below exist and most trunc targets are simply untypeable.
    needed.update(range(1, max(ma_bits, mb_bits) + 1))

    # left_shift outputs
    for k in range(1, max_shift + 1):
        needed.add(ma_bits + k)
        needed.add(mb_bits + k)

    # Iteratively expand for add/sub outputs
    prev = None
    cap = max(output_bits, ma_bits + mb_bits, ma_bits + max_shift) + 2
    while prev != needed:
        prev = set(needed)
        for x in list(prev):
            for y in list(prev):
                out = max(x, y) + 1
                if out <= cap:
                    needed.add(out)
        needed = {b for b in needed if b <= cap}

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [BV(ma_bits), BV(mb_bits)],
        BV(output_bits),
    )
    pset.renameArguments(ARG0="ma", ARG1="mb")

    # ---- mul: BV(x) * BV(y) -> BV(x+y) ----
    for x in needed:
        for y in needed:
            out = x + y
            if out in needed:
                pset.addPrimitive(_make_mul(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"mul_{x}_{y}")

    # ---- block_mul_k: BV(ma_bits) * BV(mb_bits) -> BV(ma_bits+mb_bits),
    # dropping k low bits of each operand first -- same in/out types as the
    # top-level mul_{ma_bits}_{mb_bits}, so GP can freely swap between exact
    # and block-approximate multiply at the position an exact seed would use.
    for k in range(1, min(ma_bits, mb_bits)):
        pset.addPrimitive(_make_block_mul(ma_bits, mb_bits, k),
                          [BV(ma_bits), BV(mb_bits)], BV(ma_bits + mb_bits),
                          name=f"blockmul_{k}_{ma_bits}_{mb_bits}")

    # ---- cross_mul_k: the two terms block_mul_k drops (aH*bL, aL*bH) --
    # same in/out types as block_mul_k, so GP can add them back on top of a
    # block_mul_k call using the ordinary add_{out}_{out} primitive.
    for k in range(1, min(ma_bits, mb_bits)):
        pset.addPrimitive(_make_cross_mul_hi_lo(ma_bits, mb_bits, k),
                          [BV(ma_bits), BV(mb_bits)], BV(ma_bits + mb_bits),
                          name=f"crossmul_hilo_{k}_{ma_bits}_{mb_bits}")
        pset.addPrimitive(_make_cross_mul_lo_hi(ma_bits, mb_bits, k),
                          [BV(ma_bits), BV(mb_bits)], BV(ma_bits + mb_bits),
                          name=f"crossmul_lohi_{k}_{ma_bits}_{mb_bits}")

    # ---- add, sub ----
    for x in needed:
        for y in needed:
            out = max(x, y) + 1
            if out in needed:
                pset.addPrimitive(_make_add(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"add_{x}_{y}")
                pset.addPrimitive(_make_sub(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"sub_{x}_{y}")

    # ---- and, or ----
    for x in needed:
        for y in needed:
            out = max(x, y)
            if out in needed:
                pset.addPrimitive(_make_and(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"and_{x}_{y}")
                pset.addPrimitive(_make_or(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"or_{x}_{y}")

    # ---- left_shift ----
    for x in needed:
        for k in range(1, max_shift + 1):
            out = x + k
            if out in needed:
                pset.addPrimitive(_make_left_shift(x, k), [BV(x)], BV(out),
                                  name=f"lshift_{x}_by{k}")

    # ---- logic_right_shift ----
    for x in needed:
        for k in range(1, max_shift + 1):
            pset.addPrimitive(_make_right_shift(x, k), [BV(x)], BV(x),
                              name=f"rshift_{x}_by{k}")

    # ---- ite ----
    for w in needed:
        pset.addPrimitive(_make_ite(w),
                          [BV(1), BV(w), BV(w)], BV(w),
                          name=f"ite_{w}")

    # ---- trunc (wrap-based, floor) ----
    for src in needed:
        for dst in needed:
            if dst < src and dst >= 1:
                pset.addPrimitive(_make_trunc(src, dst), [BV(src)], BV(dst),
                                  name=f"trunc_{src}_{dst}")

    # ---- rcast (saturating round-to-nearest, see _make_rcast's docstring
    # for why trunc alone can't express this) ----
    for src in needed:
        for dst in needed:
            if dst < src and dst >= 1:
                pset.addPrimitive(_make_rcast(src, dst), [BV(src)], BV(dst),
                                  name=f"rcast_{src}_{dst}")

# ---- Terminals: small constants for every bitwidth ----
    # 0 and 1 keep their historical names; 2..7 use const_{value}_{width}
    # so compensation constants are directly reachable by GP.
    for w in needed:
        pset.addTerminal(0, BV(w), name=f"zero_{w}")
        pset.addTerminal(1, BV(w), name=f"one_{w}")
        for c in range(2, min(8, 1 << w)):
            pset.addTerminal(c, BV(w), name=f"const_{c}_{w}")

    return pset


# ==========================================================
# Seed: simple truncated multiplier
# ==========================================================
def make_seed_str(ma_bits, mb_bits, output_bits):
    """
    Seed = trunc(mul(ma, mb), output_bits).
    Precise starting point; GP explores by reducing precision.
    """
    mul_out = ma_bits + mb_bits
    if mul_out > output_bits:
        shift = mul_out - output_bits
        # trunc keeps the LOW bits (see _make_trunc: a & mask) -- must shift
        # the high, magnitude-carrying bits down into that low window first,
        # or trunc silently discards them instead of the low-order precision.
        return (f"trunc_{mul_out}_{output_bits}("
                f"rshift_{mul_out}_by{shift}(mul_{ma_bits}_{mb_bits}(ma, mb)))")
    elif mul_out == output_bits:
        return f"mul_{ma_bits}_{mb_bits}(ma, mb)"
    else:
        raise ValueError(
            f"output_bits ({output_bits}) larger than mul output "
            f"({mul_out}). Extend seed if this configuration is needed."
        )


# ==========================================================
# Training data
# ==========================================================
def make_data(ma_bits, mb_bits, n_samples=100, seed=42, mantissa_domain=False):
    # target is always the true, unscaled product -- error is measured
    # against reality, not against a same-width "ideal" circuit (see
    # make_evaluate/eval_true, which rescale the circuit's output back up
    # by the same shift make_seed_str uses before comparing to this).
    rng = random.Random(seed)
    lo_a = (1 << (ma_bits - 1)) if mantissa_domain else 0
    lo_b = (1 << (mb_bits - 1)) if mantissa_domain else 0
    data = []
    for _ in range(n_samples):
        ma = rng.randint(lo_a, (1 << ma_bits) - 1)
        mb = rng.randint(lo_b, (1 << mb_bits) - 1)
        data.append((ma, mb, ma * mb))
    return data

def make_block_seed_strs(ma_bits, mb_bits, output_bits):
    """Full k-grid of block-decomposition seeds, one pair per split point --
    mirrors minifloat's make_narrow_seed_strs, which enumerates every
    mantissa-narrowing combination instead of a single fixed split. For each
    k in 1..n-1: a 'high-only' seed (blockmul_k alone, cheapest at that k)
    and a 'drop-LL' seed (blockmul_k plus both cross terms, more accurate at
    the same k) -- so GP has an explicit, verified-correct starting point at
    every area level blockmul_k/crossmul_k make reachable, instead of having
    to discover the other k's unaided from a single k=n//2 example.
    """
    seeds = []
    n = ma_bits
    if not (ma_bits == mb_bits and output_bits == 2 * n):
        return seeds

    for k in range(1, n):
        hi = f"blockmul_{k}_{n}_{n}(ma, mb)"                   # BV(2n)
        seeds.append(hi)

        s = f"add_{2*n}_{2*n}({hi}, crossmul_hilo_{k}_{n}_{n}(ma, mb))"      # BV(2n+1)
        s = f"add_{2*n+1}_{2*n}({s}, crossmul_lohi_{k}_{n}_{n}(ma, mb))"     # BV(2n+2)
        seeds.append(f"trunc_{2*n+2}_{2*n}({s})")

    return seeds


def make_side_seed_strs(ma_bits, mb_bits, output_bits):
    """Single-axis seeds: narrow ma only (keep mb exact) or narrow mb only
    (keep ma exact) -- the integer analogue of minifloat's per-axis
    mantissa seeds. The narrowed operand keeps its genuinely smaller BV(w)
    type all the way into the mul node (mul_w_mb_bits or mul_ma_bits_w, NOT
    mul_ma_bits_mb_bits), so area_model.node_cost sees the smaller multiply
    and actually credits the area saving; an outer lshift restores the
    result to output_bits. The *combined* narrowing (both operands narrowed
    at once) is never produced here, so it can be used to test whether GP's
    crossover can genuinely assemble it from two independent single-axis
    parents (see cx_combine_sides in search/integer_search.py) instead of
    just selecting between given seeds.

    Returns {'a<k>': seed_str, ...} for narrowing ma by k, and
    {'b<k>': seed_str, ...} for narrowing mb by k.
    """
    seeds = {}
    if output_bits != ma_bits + mb_bits:
        return seeds

    for k in range(1, ma_bits):
        w = ma_bits - k
        narrow = f"trunc_{ma_bits}_{w}(rshift_{ma_bits}_by{k}(ma))"      # BV(w)
        mulres = f"mul_{w}_{mb_bits}({narrow}, mb)"                      # BV(w+mb_bits)
        seeds[f"a{k}"] = f"lshift_{w + mb_bits}_by{k}({mulres})"         # BV(output_bits)

    for k in range(1, mb_bits):
        w = mb_bits - k
        narrow = f"trunc_{mb_bits}_{w}(rshift_{mb_bits}_by{k}(mb))"      # BV(w)
        mulres = f"mul_{ma_bits}_{w}(ma, {narrow})"                      # BV(ma_bits+w)
        seeds[f"b{k}"] = f"lshift_{ma_bits + w}_by{k}({mulres})"         # BV(output_bits)

    return seeds


def make_side_seed_strs_round(ma_bits, mb_bits, output_bits):
    """Same shape as make_side_seed_strs, but narrows via rcast (saturating
    round-to-nearest) instead of trunc(rshift(...)) (wrap-based floor).

    Verified empirically these are NOT redundant with the floor seeds:
    at ma_bits=mb_bits=4, round beats floor at several (ka,kb) combinations
    (e.g. single-axis k=2 drops from 27.3% to 20.0% worst-case error at the
    *same* area) and is tied at others (k=1, k=3 in this config happen to
    round to the same values floor produces) -- see the project report/
    discussion for the full comparison. Both families are offered as
    hints side by side so GP (and cx_combine_sides) can freely mix floor-
    narrowed and round-narrowed sides rather than being steered toward
    floor exclusively."""
    seeds = {}
    if output_bits != ma_bits + mb_bits:
        return seeds

    for k in range(1, ma_bits):
        w = ma_bits - k
        narrow = f"rcast_{ma_bits}_{w}(ma)"
        mulres = f"mul_{w}_{mb_bits}({narrow}, mb)"
        seeds[f"ar{k}"] = f"lshift_{w + mb_bits}_by{k}({mulres})"

    for k in range(1, mb_bits):
        w = mb_bits - k
        narrow = f"rcast_{mb_bits}_{w}(mb)"
        mulres = f"mul_{ma_bits}_{w}(ma, {narrow})"
        seeds[f"br{k}"] = f"lshift_{ma_bits + w}_by{k}({mulres})"

    return seeds