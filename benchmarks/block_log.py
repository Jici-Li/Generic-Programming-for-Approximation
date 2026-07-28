"""Block log-domain (LNS) reference model -- DeepWok/mase's block_log.py
(src/chop/nn/quantizers/block_log.py): a block shares one exponent *bias*,
each element still keeps its own log-domain exponent (relative to that
shared bias). MASE's own formula:

    per_block_bias = clamp(2**exponent_bits - 1 - ceil(log2(block_max)),
                            0, 2**exponent_bias_width - 1)

then every element in the block is log-quantized (benchmarks/log.py's
quantize) using that shared bias instead of the format's own default one.

Unlike benchmarks/mxint_hardware.py/block_fp.py/block_minifloat.py, there's no
separate outer block-level multiplicative scale here -- the shared bias is
substituted directly into each element's own quantize step (same
substituted-bias pattern as block_minifloat.py, see that file's module
docstring), so dequantizing an element needs nothing beyond benchmarks/
log.py's own dequantize.

Multiplication is still exactly log(a*b) = log(a) + log(b): the block's
shared bias doesn't change the elementwise operation at all, only how real
values map onto (sign, exponent) pairs in the first place -- so the GP
search target, grammar, and CVC5 verifier are *exactly* benchmarks/
log.py's / verify/verify_log_error_bound.py's, completely unmodified.
This file only adds the block-level quantization wrapper.
"""

from dataclasses import dataclass
import math
import random

from benchmarks.log import LogValue, quantize, dequantize, reference_multiply


@dataclass(frozen=True)
class LogBlock:
    values: tuple  # LogValue entries, already quantised with the shared bias

    @property
    def block_size(self):
        return len(self.values)


def choose_shared_bias(values, width, exponent_bias_width):
    """MASE's real block_log.py formula."""
    exponent_bits = width - 1
    max_abs = max(abs(float(v)) for v in values) if values else 0.0
    per_block_max_exponent = 0 if max_abs == 0.0 else math.ceil(math.log2(max_abs))
    bias = (2 ** exponent_bits - 1) - per_block_max_exponent
    return max(0, min(bias, (1 << exponent_bias_width) - 1))


def quantize_block(values, width, exponent_bias_width):
    bias = choose_shared_bias(values, width, exponent_bias_width)
    return LogBlock(tuple(quantize(v, width, exponent_bias=bias) for v in values))


def dequantize_block(block):
    return tuple(dequantize(lv) for lv in block.values)


def reference_multiply_blocks(a, b):
    """Exact, elementwise -- log(a*b) = log(a)+log(b) doesn't involve the
    shared bias at all once each element is already a LogValue."""
    if a.block_size != b.block_size:
        raise ValueError("block_log block sizes must match")
    return LogBlock(tuple(
        reference_multiply(av, bv) for av, bv in zip(a.values, b.values)
    ))


def make_data(width, exponent_bias_width, block_size=4, n_blocks=100, seed=42):
    """Generate real-valued blocks, quantise them, and store exact
    block_log targets."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rng = random.Random(seed)
    data = []
    for _ in range(n_blocks):
        a_magnitude = 2.0 ** rng.randint(-8, 8)
        b_magnitude = 2.0 ** rng.randint(-8, 8)
        a_values = [rng.uniform(-a_magnitude, a_magnitude) for _ in range(block_size)]
        b_values = [rng.uniform(-b_magnitude, b_magnitude) for _ in range(block_size)]
        a = quantize_block(a_values, width, exponent_bias_width)
        b = quantize_block(b_values, width, exponent_bias_width)
        data.append((a, b, reference_multiply_blocks(a, b)))
    return data


def self_check():
    """Cheap reference checks, runnable without starting evolution."""
    block = quantize_block([8.0, 0.5, -2.0, 4.0], width=8, exponent_bias_width=4)
    values = dequantize_block(block)
    for v, expected in zip(values, (8.0, 0.5, -2.0, 4.0)):
        assert abs(v - expected) / abs(expected) < 1e-9, (v, expected)

    other = quantize_block([2.0, 2.0, 2.0, 2.0], width=8, exponent_bias_width=4)
    product = reference_multiply_blocks(block, other)
    prod_values = dequantize_block(product)
    exact = tuple(a * b for a, b in zip(values, dequantize_block(other)))
    for pv, ev in zip(prod_values, exact):
        assert abs(pv - ev) / abs(ev) < 1e-9, (pv, ev)
