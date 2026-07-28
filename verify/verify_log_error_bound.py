"""Formal verification: does a log-domain (LNS) approximate exponent-adder
GP candidate satisfy the relative error bound for every (ea, eb) signed
exponent pair?

The dequantized value is sign * 2**exponent, so relative error of the
*value* is a function of the *exponent difference* d = approx_exp -
exact_exp alone:
    d >= 0:  relative error = 2**d - 1          (unbounded as d grows)
    d <  0:  relative error = 1 - 2**d           (bounded below 1)
This file precomputes, in plain Python, the integer thresholds d_max/d_min
such that "relative error > alpha" is exactly "d > d_max or d < d_min" --
that keeps the CVC5 query pure integer/BitVector arithmetic (no symbolic
exponentiation needed at all), unlike verify_error_bound.py/
verify_mxint_error_bound.py which have to build a genuine multiplication
or float comparison into the SMT formula.
"""

import gc
import math

import cvc5
from cvc5 import Kind

from verify.log_cvc5_translate import gp_to_cvc5


TIMEOUT_MS = 30000


def _parse_bv_value(term):
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


def _error_thresholds(alpha):
    """Return (d_min, d_max): the exponent-difference d=approx-exact
    violates the alpha bound iff d < d_min or d > d_max."""
    d_max = math.floor(math.log2(1.0 + alpha))
    if alpha >= 1.0:
        d_min = -math.inf
    else:
        d_min = math.ceil(math.log2(1.0 - alpha))
    return d_min, d_max


def verify_error_bound(individual, pset, exponent_bits, output_bits, alpha,
                       timeout_ms=TIMEOUT_MS):
    """
    Returns (verified: bool, counterexample: dict|None).
    verified = True  -> the property holds for all (ea, eb) (UNSAT)
    verified = False -> a counterexample was found (SAT), or unknown/error
    """
    d_min, d_max = _error_thresholds(alpha)

    try:
        solver = cvc5.Solver()
        solver.setLogic("QF_BVNIA")
        solver.setOption("produce-models", "true")
        solver.setOption("tlimit-per", str(timeout_ms))

        ea_bv = solver.mkConst(solver.mkBitVectorSort(exponent_bits), 'ea')
        eb_bv = solver.mkConst(solver.mkBitVectorSort(exponent_bits), 'eb')

        approx_bv = gp_to_cvc5(individual, ea_bv, eb_bv, solver)

        # exact exponent sum, computed at output_bits width (wide enough
        # that two exponent_bits-wide signed values can never overflow it
        # -- output_bits is exponent_bits+1 by construction in
        # benchmarks/log.py's make_pset).
        ea_wide = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, output_bits - exponent_bits), ea_bv)
        eb_wide = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, output_bits - exponent_bits), eb_bv)
        exact_bv = solver.mkTerm(Kind.BITVECTOR_ADD, ea_wide, eb_wide)

        # diff = approx - exact can range across nearly the *entire*
        # output_bits signed span in either direction (verified: found a
        # real counterexample, ea=eb=-32 with exponent_bits=6, where
        # approx=0 and exact=-64 -- diff=+64 doesn't fit in a 7-bit signed
        # value at all, silently wrapping to -64 and making a genuine
        # violation look like a false "VERIFIED"). Same "two N-bit signed
        # values need N+1 bits to add/subtract without overflow" rule
        # used everywhere else in this project for sadd's output width,
        # just previously forgotten here for this specific subtraction.
        approx_diff = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, 1), approx_bv)
        exact_diff = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, 1), exact_bv)
        diff_int = solver.mkTerm(Kind.BITVECTOR_SBV_TO_INT,
            solver.mkTerm(Kind.BITVECTOR_SUB, approx_diff, exact_diff))

        violation_terms = [solver.mkTerm(Kind.GT, diff_int, solver.mkInteger(d_max))]
        if d_min != -math.inf:
            violation_terms.append(solver.mkTerm(Kind.LT, diff_int, solver.mkInteger(int(d_min))))
        violation = violation_terms[0]
        for t in violation_terms[1:]:
            violation = solver.mkTerm(Kind.OR, violation, t)
        solver.assertFormula(violation)

        result = solver.checkSat()

        if result.isUnsat():
            del solver
            gc.collect()
            return True, None

        if result.isSat():
            ea_val = _parse_bv_value(solver.getValue(ea_bv))
            eb_val = _parse_bv_value(solver.getValue(eb_bv))
            approx_val = _parse_bv_value(solver.getValue(approx_bv))
            exact_val = ea_val + eb_val
            d = approx_val - exact_val
            err = (2.0 ** d - 1.0) if d >= 0 else (1.0 - 2.0 ** d)
            del solver
            gc.collect()
            return False, {
                'ea': ea_val, 'eb': eb_val,
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
