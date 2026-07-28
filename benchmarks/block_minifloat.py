"""Block Minifloat (BM) block reference model, built directly on top of
benchmarks/minifloat.py's element-level FP(e, m) grammar -- DeepWok/
mase's block_minifloat.py (src/chop/nn/quantizers/block_minifloat.py).

Renamed from this project's earlier "mxfp_mult.py": each block element
keeps its own full exponent field *and* mantissa (a real per-element
minifloat value), and only an exponent *bias* is shared across the block.
MASE's own docstring for the format: ``2**-bias_shared x [(-1)^s1 x
2**exponent1 x mantissa1, (-1)^s2 x 2**exponent2 x mantissa2, ...]``.

Stage-1 scope (mirrors benchmarks/mxint_hardware.py exactly, just with FP
elements instead of signed-integer ones):
  * a block shares one exponent *bias*, applied here as an outer power-of-
    two scale ``2 ** exponent`` for the same numeric effect (MASE folds the
    bias directly into each element's own exponent field at decode time;
    this file applies it as a separate multiplicative factor after
    decoding each element with its normal, fixed bias -- both parameterise
    the exact same representable set of values, see choose_shared_exponent
    below for the conversion between the two);
  * GP searches the element-level FP multiplier;
  * block quantisation and shared-exponent selection remain fixed
    reference logic, exactly like benchmarks/mxint_hardware.py's stage-1 split.

The element multiplier grammar (make_pset, make_seed_str,
make_narrow_seed_strs) is *not* redefined here -- it's imported straight
from benchmarks/minifloat.py, because a block-minifloat element
genuinely is a minifloat value. Everything new in this file is the
block-level wrapper around it.

Note: DeepWok/mase's block_fp.py (Microsoft Floating Point / MSFP) is a
*different* format -- one shared exponent per block, elements are plain
signed integer mantissas with no per-element exponent field at all. That
one is structurally benchmarks/mxint_hardware.py's MXIntBlock, built
separately as benchmarks/block_fp.py (reusing mxint's element grammar/
search/verify wholesale, just a different choose_shared_exponent).
"""

from dataclasses import dataclass
import math
import random

from benchmarks.minifloat import (
    decode, encode, format_max_val,
    make_pset, make_seed_str, make_narrow_seed_strs,
)


# ==========================================================
# Block Minifloat reference representation
# ==========================================================
@dataclass(frozen=True)
class BlockMinifloatBlock:
    exponent: int
    elements: tuple[int, ...]        # encoded FP(e, m) bit patterns
    element_format: tuple[int, int]  # (e, m)

    @property
    def block_size(self):
        return len(self.elements)


def choose_shared_exponent(values, element_format, exponent_bias_width):
    """Shared exponent, matching MASE's block_minifloat.py's real formula:

        per_block_bias = clamp(floor(log2(block_max)), 0, 2**exponent_bias_width - 1)
        (each element then minifloat-decodes using per_block_bias instead
        of its format's normal fixed bias (1<<(e-1))-1)

    Folded here into a single outer power-of-two scale (this file's existing
    parameterisation: value = 2**exponent * decode_minifloat(element)) by
    converting MASE's substituted-bias form into an equivalent outer-scale
    form: decoding element bits with the format's own FIXED bias gives
    2**(local_exp_field - fixed_bias) * mantissa; MASE's version gives
    2**(local_exp_field - per_block_bias) * mantissa; these are equal when
    the outer scale equals 2**(fixed_bias - per_block_bias) -- same
    representable values, this file just keeps the block-level shift
    external instead of substituting it into the decode step.

    exponent_bias_width: number of bits used to encode the shared bias
    (MASE's own separate knob from the element format's own exponent
    width e).
    """
    e, m = element_format
    if not values:
        raise ValueError("a Block Minifloat block cannot be empty")
    fixed_bias = (1 << (e - 1)) - 1
    max_abs = max(abs(float(v)) for v in values)
    if max_abs == 0.0:
        per_block_bias = 0
    else:
        per_block_bias = math.floor(math.log2(max_abs))
    per_block_bias = max(0, min(per_block_bias, (1 << exponent_bias_width) - 1))
    return fixed_bias - per_block_bias


def quantize_block(values, element_format, exponent_bias_width):
    """Quantise real values to one shared exponent (bias) and per-element
    FP(e, m) encodings."""
    e, m = element_format
    values = tuple(float(v) for v in values)
    exponent = choose_shared_exponent(values, element_format, exponent_bias_width)
    scale = 2.0 ** exponent
    elements = tuple(encode(v / scale, e, m) for v in values)
    return BlockMinifloatBlock(exponent, elements, element_format)


def dequantize_block(block):
    e, m = block.element_format
    scale = 2.0 ** block.exponent
    return tuple(decode(bits, e, m) * scale for bits in block.elements)


def reference_multiply_blocks(a, b):
    """Exact element-wise multiplication of two already-quantised blocks,
    encoded into the lossless product format (ea+eb, ma+mb+1) -- same
    "exact reference, no rounding" contract as benchmarks/mxint_hardware.py's
    reference_multiply_blocks."""
    if a.block_size != b.block_size:
        raise ValueError("Block Minifloat block sizes must match")
    ea, ma = a.element_format
    eb, mb = b.element_format
    eo, mo = ea + eb, ma + mb + 1
    products = tuple(
        encode(decode(x, ea, ma) * decode(y, eb, mb), eo, mo)
        for x, y in zip(a.elements, b.elements)
    )
    return BlockMinifloatBlock(a.exponent + b.exponent, products, (eo, mo))


def reference_dot_blocks(a, b):
    product = reference_multiply_blocks(a, b)
    return sum(dequantize_block(product))


# ==========================================================
# Training / exhaustive data
# ==========================================================
def make_data(element_format, exponent_bias_width, block_size=4, n_blocks=100, seed=42,
             mantissa_domain=False):
    """Generate real blocks, quantise them, and store exact targets.

    mantissa_domain: draw each element from the upper half of the block's
    magnitude range instead of the full [-magnitude, magnitude] -- same
    reasoning and same rejection-sampling-at-the-quantised-level fix as
    benchmarks/mxint_hardware.py's make_data (a block-shared scale means even
    a real value drawn from the "upper half" can still quantise to a
    small-magnitude element relative to that block's own scale).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    e, m = element_format
    rng = random.Random(seed)
    data = []
    fmax = format_max_val(e, m)
    # "small" relative to the element format: below the format's smallest
    # normal magnitude, same threshold spirit as mxint's 1<<(element_bits-2).
    small_threshold = 2.0 ** (1 - ((1 << (e - 1)) - 1))

    def draw_block(magnitude):
        for _ in range(200):
            values = [rng.uniform(magnitude / 2.0, magnitude) *
                      (1 if rng.random() < 0.5 else -1)
                      for _ in range(block_size)]
            block = quantize_block(values, element_format, exponent_bias_width)
            if all(abs(decode(bits, e, m)) >= small_threshold for bits in block.elements):
                return block
        return block  # give up after 200 tries, return the last attempt

    for _ in range(n_blocks):
        a_magnitude = fmax * (2.0 ** rng.randint(-6, 0))
        b_magnitude = fmax * (2.0 ** rng.randint(-6, 0))
        if mantissa_domain:
            a = draw_block(a_magnitude)
            b = draw_block(b_magnitude)
        else:
            a_values = [rng.uniform(-a_magnitude, a_magnitude) for _ in range(block_size)]
            b_values = [rng.uniform(-b_magnitude, b_magnitude) for _ in range(block_size)]
            a = quantize_block(a_values, element_format, exponent_bias_width)
            b = quantize_block(b_values, element_format, exponent_bias_width)
        data.append((a, b, reference_multiply_blocks(a, b)))

    return data


def make_exhaustive_element_data(element_format):
    """Every finite (non-inf/NaN) element bit-pattern pair, each wrapped as
    a one-element, zero-exponent block -- the block-minifloat analogue of
    benchmarks/mxint_hardware.py's make_exhaustive_mantissa_data. Exponent bias
    doesn't matter here (block of one, always its own max), so this is
    unchanged by the choose_shared_exponent formula update."""
    e, m = element_format
    width = 1 + e + m
    max_exp_field = (1 << e) - 1
    data = []
    for a_bits in range(1 << width):
        if ((a_bits >> m) & max_exp_field) == max_exp_field:
            continue  # inf/NaN bit pattern
        for b_bits in range(1 << width):
            if ((b_bits >> m) & max_exp_field) == max_exp_field:
                continue
            a = BlockMinifloatBlock(0, (a_bits,), element_format)
            b = BlockMinifloatBlock(0, (b_bits,), element_format)
            data.append((a, b, reference_multiply_blocks(a, b)))
    return data


def self_check():
    """Cheap reference checks, runnable without starting evolution."""
    element_format = (4, 3)
    ebw = 3
    block = quantize_block([3.0, 1.0, 0.25, -2.0], element_format, ebw)
    values = dequantize_block(block)
    for v, expected in zip(values, (3.0, 1.0, 0.25, -2.0)):
        assert abs(v - expected) / max(abs(expected), 1e-9) < 0.05, (v, expected)

    other = quantize_block([1.0, 2.0, -1.0, 0.5], element_format, ebw)
    product = reference_multiply_blocks(block, other)
    prod_values = dequantize_block(product)
    exact = tuple(a * b for a, b in zip(values, dequantize_block(other)))
    for pv, ev in zip(prod_values, exact):
        assert abs(pv - ev) / max(abs(ev), 1e-9) < 0.05, (pv, ev)
