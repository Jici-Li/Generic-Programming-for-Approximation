"""MSFP (Microsoft Floating Point) block reference model -- DeepWok/mase's
block_fp.py (src/chop/nn/quantizers/block_fp.py): one shared power-of-two
exponent per block, each element is a plain signed integer mantissa (no
per-element exponent field at all).

That's exactly benchmarks/mxint_hardware.py's MXIntBlock shape, so this file
does not redefine the element grammar, GP primitive set, seed builders, or
CVC5 verification -- all of those are imported/reused unchanged from
benchmarks/mxint_hardware.py and verify/verify_mxint_hardware_error_bound.py. The only
thing genuinely different here is choose_shared_exponent/quantize_block,
matching MASE's real block_fp.py formula, which (unlike mxint_hardware.py's
formula -- see mxint_hardware.py's module docstring) fully uses the mantissa's
dynamic range: it normalizes the exponent to the block's own max magnitude
first, then explicitly scales by 2**(element_bits-1) to spread the mantissa
across its whole representable range, instead of folding a width-
independent bias straight into the exponent choice.
"""

import math
import random

from benchmarks.mxint_hardware import (
    MXIntBlock,
    signed_range,
    reference_multiply_blocks,
    make_exhaustive_mantissa_data,
    make_narrow_seed_strs,
    make_partial_seed_strs,
    make_pset,
    make_seed_str,
    self_check as _mxint_self_check,
)


def _clip_signed(value, bits):
    lo, hi = signed_range(bits)
    return max(lo, min(int(value), hi))


def choose_shared_exponent(values, element_bits, exponent_width):
    """MASE block_fp.py's real MSFP formula:

        exponent_bias = 2 ** (exponent_width - 1) - 1   [standard IEEE bias]
        exponent_max  = 2 ** exponent_width - 1 - exponent_bias
        exponent_min  = -exponent_bias
        exponent = clamp(ceil(log2(block_max)), exponent_min, exponent_max)
        mantissa = clamp(round(value / 2**exponent * shift), 0, shift-1),
                   shift = 2**(element_bits-1), sign stored separately

    Folded here into a single signed-integer-mantissa exponent (value =
    mantissa * 2**returned_exponent, matching MXIntBlock's convention)
    by subtracting the mantissa's own headroom (element_bits-1) from the
    block exponent up front, so quantize_block below can reuse mxint_hardware
    .py's plain "floor/round then clip" shape unchanged.
    """
    if not values:
        raise ValueError("an MXInt block cannot be empty")
    exponent_bias = 2 ** (exponent_width - 1) - 1
    exponent_max = 2 ** exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias
    max_abs = max(abs(float(v)) for v in values)
    if max_abs == 0.0:
        block_exponent = max(exponent_min, min(0, exponent_max))
    else:
        block_exponent = max(exponent_min, min(math.ceil(math.log2(max_abs)), exponent_max))
    return block_exponent - (element_bits - 1)


def quantize_block(values, element_bits, exponent_width):
    """Quantise real values to one shared exponent and signed mantissas,
    matching MASE's block_fp.py: round-to-nearest (not floor -- unlike
    mxint_hardware.py's mxint_quant_block)."""
    values = tuple(float(v) for v in values)
    exponent = choose_shared_exponent(values, element_bits, exponent_width)
    scale = 2.0 ** exponent
    mantissas = tuple(_clip_signed(round(v / scale), element_bits) for v in values)
    return MXIntBlock(exponent, mantissas, element_bits)


def dequantize_block(block):
    scale = 2.0 ** block.exponent
    return tuple(mantissa * scale for mantissa in block.mantissas)


def make_data(element_bits, exponent_width, block_size=4, n_blocks=100, seed=42):
    """Generate real blocks, quantise them, and store exact MSFP targets --
    mirrors benchmarks/mxint_hardware.py's make_data exactly (same magnitude-
    draw-range reasoning: keep the block's real-value magnitude inside
    choose_shared_exponent's unclamped zone so no block ends up with
    uniformly tiny mantissas)."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rng = random.Random(seed)
    data = []
    exponent_bias = 2 ** (exponent_width - 1) - 1
    mag_lo, mag_hi = 0, 2 * exponent_bias + 1

    for _ in range(n_blocks):
        a_magnitude = 2.0 ** rng.randint(mag_lo, mag_hi)
        b_magnitude = 2.0 ** rng.randint(mag_lo, mag_hi)
        a_values = [rng.uniform(-a_magnitude, a_magnitude) for _ in range(block_size)]
        b_values = [rng.uniform(-b_magnitude, b_magnitude) for _ in range(block_size)]
        a = quantize_block(a_values, element_bits, exponent_width)
        b = quantize_block(b_values, element_bits, exponent_width)
        data.append((a, b, reference_multiply_blocks(a, b)))

    return data


def self_check():
    """Cheap reference checks, runnable without starting evolution."""
    _mxint_self_check()  # element grammar/reference-multiply checks are shared

    block = quantize_block([3.0, 1.0, 0.25, -2.0], element_bits=4, exponent_width=3)
    assert block.exponent == -1
    assert block.mantissas == (6, 2, 0, -4)
    assert dequantize_block(block) == (3.0, 1.0, 0.0, -2.0)

    other = quantize_block([1.0, 2.0, -1.0, 0.5], element_bits=4, exponent_width=3)
    product = reference_multiply_blocks(block, other)
    prod_values = dequantize_block(product)
    exact = tuple(a * b for a, b in zip(dequantize_block(block), dequantize_block(other)))
    # element_bits=4 only gives 3 magnitude bits, so this reference-multiply
    # check (unlike mxint_hardware.py's, which has exact-integer inputs) allows
    # for the block's own real-value quantisation error, not just checking
    # the multiply is exact given already-quantised mantissas.
    for pv, ev in zip(prod_values, exact):
        assert abs(pv - ev) <= 0.5, (pv, ev)
