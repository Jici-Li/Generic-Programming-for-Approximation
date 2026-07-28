"""
Formal verification: does an MXInt element-multiplier GP candidate satisfy
the relative error bound for every (ma, mb) signed mantissa pair?

Mirrors verify/verify_integer_error_bound.py's integer structure closely (same
division-free cross-multiplication for the percentage-error check, same
exception-path solver cleanup to avoid the cross-call segfault documented
there -- confirmed to matter again while validating verify/
mxint_cvc5_translate.py: ~500 back-to-back solver instances without
del+gc.collect() segfaulted at the very end of that run). "exact" is the
plain, untruncated ma*mb (matching benchmarks/mxint_hardware.py's _make_smul,
which has no masking at all) -- the block's shared exponent is a
multiplicative constant applied equally to approx and exact, so it cancels
out of the relative-error ratio exactly, same reasoning search/
block_minifloat_search.py's reuse of verify_minifloat_error_bound_native relies on.
Verifying the element multiplier alone (ignoring block scale) is
therefore sufficient and sound.
"""

import gc

import cvc5
from cvc5 import Kind

from verify.mxint_hardware_cvc5_translate import gp_to_cvc5


TIMEOUT_MS = 30000


def _parse_bv_value(term):
    """Convert a CVC5 (signed) BitVector value term into a Python int."""
    s = str(term)
    if s.startswith('#b'):
        v = int(s[2:], 2)
        width = len(s) - 2
    elif s.startswith('#x'):
        v = int(s[2:], 16)
        width = (len(s) - 2) * 4
    elif s.startswith('(_'):
        parts = s.strip('()').split()
        val_str = parts[1]
        if not val_str.startswith('bv'):
            raise ValueError(f"Cannot parse BV value: {s!r}")
        v = int(val_str[2:])
        width = int(parts[2])
    else:
        raise ValueError(f"Cannot parse BV value: {s!r}")
    if v >= (1 << (width - 1)):
        v -= (1 << width)
    return v


def verify_error_bound(individual, pset, element_bits, output_bits, alpha,
                       timeout_ms=TIMEOUT_MS, mantissa_domain=False):
    """
    Returns (verified: bool, counterexample: dict|None).
    verified = True  -> the property holds for all (ma, mb) (UNSAT)
    verified = False -> a counterexample was found (SAT), or unknown/error

    mantissa_domain: restrict ma, mb to the upper half of the representable
    magnitude range (|value| >= 2**(element_bits-2)), matching benchmarks/
    mxint_hardware.py's make_data mantissa_domain=True -- must match whatever
    training used, same reasoning as integer's mantissa_domain and
    minifloat's exclude_subnormal_inputs.
    """
    try:
        solver = cvc5.Solver()
        solver.setLogic("QF_BVNIRA")
        solver.setOption("produce-models", "true")
        solver.setOption("tlimit-per", str(timeout_ms))

        ma_bv = solver.mkConst(solver.mkBitVectorSort(element_bits), 'ma')
        mb_bv = solver.mkConst(solver.mkBitVectorSort(element_bits), 'mb')

        if mantissa_domain:
            threshold = 1 << max(0, element_bits - 2)
            pos_thresh = solver.mkBitVector(element_bits, threshold)
            neg_thresh = solver.mkBitVector(element_bits, (-threshold) & ((1 << element_bits) - 1))
            for bv in (ma_bv, mb_bv):
                in_domain = solver.mkTerm(Kind.OR,
                    solver.mkTerm(Kind.BITVECTOR_SGE, bv, pos_thresh),
                    solver.mkTerm(Kind.BITVECTOR_SLE, bv, neg_thresh))
                solver.assertFormula(in_domain)

        approx_bv = gp_to_cvc5(individual, ma_bv, mb_bv, solver)

        approx_real = solver.mkTerm(Kind.TO_REAL,
            solver.mkTerm(Kind.BITVECTOR_SBV_TO_INT, approx_bv))
        ma_real = solver.mkTerm(Kind.TO_REAL,
            solver.mkTerm(Kind.BITVECTOR_SBV_TO_INT, ma_bv))
        mb_real = solver.mkTerm(Kind.TO_REAL,
            solver.mkTerm(Kind.BITVECTOR_SBV_TO_INT, mb_bv))
        exact_real = solver.mkTerm(Kind.MULT, ma_real, mb_real)

        zero_real = solver.mkReal(0)
        solver.assertFormula(solver.mkTerm(Kind.DISTINCT, exact_real, zero_real))

        # percentage error, division-free: |approx - exact| > alpha*|exact|
        diff = solver.mkTerm(
            Kind.ITE,
            solver.mkTerm(Kind.GT, approx_real, exact_real),
            solver.mkTerm(Kind.SUB, approx_real, exact_real),
            solver.mkTerm(Kind.SUB, exact_real, approx_real),
        )
        exact_abs = solver.mkTerm(
            Kind.ITE,
            solver.mkTerm(Kind.GEQ, exact_real, zero_real),
            exact_real,
            solver.mkTerm(Kind.NEG, exact_real),
        )
        alpha_int_num = int(round(alpha * 10000))
        alpha_rat = solver.mkReal(alpha_int_num, 10000)
        threshold_term = solver.mkTerm(Kind.MULT, alpha_rat, exact_abs)

        violation = solver.mkTerm(Kind.GT, diff, threshold_term)
        solver.assertFormula(violation)

        result = solver.checkSat()

        if result.isUnsat():
            del solver
            gc.collect()
            return True, None

        if result.isSat():
            ma_val = _parse_bv_value(solver.getValue(ma_bv))
            mb_val = _parse_bv_value(solver.getValue(mb_bv))
            approx_val = _parse_bv_value(solver.getValue(approx_bv))
            exact_val = ma_val * mb_val
            err = abs(approx_val - exact_val) / abs(exact_val) if exact_val else float('inf')
            del solver
            gc.collect()
            return False, {
                'ma': ma_val, 'mb': mb_val,
                'approx': approx_val, 'exact': exact_val,
                'error': err,
            }

        del solver
        gc.collect()
        return False, {'reason': 'timeout_or_unknown'}

    except Exception as e:
        message = str(e)
        if 'solver' in locals():
            del solver
        gc.collect()
        return False, {'reason': 'exception', 'message': message}
