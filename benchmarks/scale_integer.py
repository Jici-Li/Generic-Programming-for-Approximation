"""Thin alias for DeepWok/mase's scale_integer.py (per-tensor absmax-scaled
integer quantizer: scale = 2**(width-1) / x_max, an arbitrary real number,
not a power of two).

At the element-multiplier level this reduces to exactly benchmarks/
integer.py too: the per-tensor scale is a constant applied outside the
integer multiply and cancels out of the relative-error ratio, whether that
constant is a power of two (mxint_hardware.py/block_fp.py) or an arbitrary
real number (scale_integer.py) makes no difference to what the multiplier
itself has to do. This file exists only so the file name matches MASE's
for discoverability; no new logic lives here.
"""

from benchmarks.integer import (  # noqa: F401
    BV,
    make_pset,
    make_seed_str,
    make_side_seed_strs,
    make_side_seed_strs_round,
    make_block_seed_strs,
    make_data,
)
