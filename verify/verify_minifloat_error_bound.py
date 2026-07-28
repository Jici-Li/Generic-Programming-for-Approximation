"""
Formal verification for minifloat GP candidates using CVC5's native
FloatingPoint theory (see verify/minifloat_cvc5_translate.py for why: the
hand-rolled Real-arithmetic path in verify/minifloat_cvc5_translate.py is
validated correct but can't *prove* universal properties in reasonable
time, even trivial ones -- native FP proves the same kind of property in
~0.01s instead of timing out).

The relative-error threshold check stays entirely in native FloatingPoint
arithmetic too (double precision -- e11m52, natively supported without the
fp-exp flag), instead of bridging to generic Real arithmetic via
FLOATINGPOINT_TO_REAL at the end. That bridge was measured to be the
actual bottleneck, not an inherent solver limit: a candidate that timed
out at 90s with the Real-arithmetic threshold check verified in 0.6s once
the whole query was rewritten to stay in FP theory end to end. Double
precision (52 mantissa bits) is vastly more precision than any value in
this project's tiny formats needs, so this introduces no meaningful
rounding of its own relative to the "exact" Real-arithmetic comparison.

"verified" here means: for every (ma_bits, mb_bits) pair whose raw
exponent field is not the reserved top value, AND for which no
intermediate/output value in the candidate's computation overflows to
infinity, the relative error is within alpha. That's a real, sound
guarantee -- just over a slightly smaller domain than the full bit-pattern
space (see verify/minifloat_cvc5_translate.py's module docstring for exactly
why, and why closing that last gap would reintroduce the scalability
problem this file exists to avoid).
"""

import gc
import struct

import cvc5
from cvc5 import Kind

from verify.minifloat_cvc5_translate import gp_to_cvc5_native, decode_native, widen, UnsupportedFormat, _RM


TIMEOUT_MS = 90000
_DBL_E, _DBL_M = 11, 52


def _parse_bv_value(term):
    s = str(term)
    if s.startswith('#b'):
        return int(s[2:], 2)
    if s.startswith('#x'):
        return int(s[2:], 16)
    if s.startswith('(_'):
        parts = s.strip('()').split()
        val_str = parts[1]
        if val_str.startswith('bv'):
            return int(val_str[2:])
    raise ValueError(f"Cannot parse BV value: {s!r}")


def _fp_dbl_to_float(term):
    """A model value for a double-precision (11, 53) FloatingPoint term ->
    Python float, via its raw 64-bit IEEE representation."""
    _, _, bv_term = term.getFloatingPointValue()
    bits = _parse_bv_value(bv_term)
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def verify_error_bound(individual, pset, ma_format, mb_format, output_format, alpha,
                       timeout_ms=TIMEOUT_MS, exclude_subnormal_inputs=False):
    """
    Returns (verified: bool, counterexample: dict|None).
    verified = True  -> the property holds over the checked domain (UNSAT)
    verified = False -> a counterexample was found (SAT), or unknown/error

    See module docstring for exactly what domain "verified" covers.

    exclude_subnormal_inputs: must match whatever benchmarks.minifloat
    .make_data was called with for this run (same reasoning as integer's
    mantissa_domain param) -- if training samples excluded subnormal-
    magnitude ma/mb but this stays False, CVC5 will happily find real
    counterexamples in the region GP was never scored against, and
    "VERIFIED" would otherwise be a promise over a wider domain than what
    was actually searched.
    """
    ea, ma_m = ma_format
    eb, mb_m = mb_format
    wa, wb = 1 + ea + ma_m, 1 + eb + mb_m

    try:
        solver = cvc5.Solver()
        solver.setLogic("ALL")
        solver.setOption("fp-exp", "true")
        solver.setOption("produce-models", "true")
        solver.setOption("tlimit-per", str(timeout_ms))

        ma_bv = solver.mkConst(solver.mkBitVectorSort(wa), 'ma')
        mb_bv = solver.mkConst(solver.mkBitVectorSort(wb), 'mb')

        if exclude_subnormal_inputs:
            ma_sub_exp = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, wa - 2, ma_m), ma_bv)
            mb_sub_exp = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, wb - 2, mb_m), mb_bv)
            solver.assertFormula(solver.mkTerm(Kind.DISTINCT, ma_sub_exp, solver.mkBitVector(ea, 0)))
            solver.assertFormula(solver.mkTerm(Kind.DISTINCT, mb_sub_exp, solver.mkBitVector(eb, 0)))

        # exclude the one exponent field native FP treats as inf/nan but
        # our format treats as an ordinary (saturating) value -- see
        # verify/minifloat_cvc5_translate.py's module docstring.
        ma_exp = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, wa - 2, ma_m), ma_bv)
        mb_exp = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, wb - 2, mb_m), mb_bv)
        solver.assertFormula(solver.mkTerm(Kind.DISTINCT, ma_exp, solver.mkBitVector(ea, (1 << ea) - 1)))
        solver.assertFormula(solver.mkTerm(Kind.DISTINCT, mb_exp, solver.mkBitVector(eb, (1 << eb) - 1)))

        # exact = decode(ma) * decode(mb), computed in a format wide enough
        # to hold the product exactly (FLOATINGPOINT_MULT needs both
        # operands in the same sort). Widened up front so the inf/nan
        # domain-exclusion checks below and the final double-precision
        # comparison both operate on a single, already-widened term.
        mul_eo, mul_mo = ea + eb, ma_m + mb_m + 1
        ma_wide = widen(solver, decode_native(solver, ma_bv, ea, ma_m), mul_eo, mul_mo)
        mb_wide = widen(solver, decode_native(solver, mb_bv, eb, mb_m), mul_eo, mul_mo)
        exact = solver.mkTerm(Kind.FLOATINGPOINT_MULT,
                              solver.mkRoundingMode(cvc5.RoundingMode.ROUND_NEAREST_TIES_TO_EVEN),
                              ma_wide, mb_wide)

        domain_exclusions = []
        approx = gp_to_cvc5_native(individual, ma_bv, mb_bv, solver, domain_exclusions)

        exact_finite = solver.mkTerm(Kind.NOT, solver.mkTerm(Kind.OR,
            solver.mkTerm(Kind.FLOATINGPOINT_IS_INF, exact),
            solver.mkTerm(Kind.FLOATINGPOINT_IS_NAN, exact)))
        solver.assertFormula(exact_finite)
        if domain_exclusions:
            any_bad = domain_exclusions[0]
            for extra in domain_exclusions[1:]:
                any_bad = solver.mkTerm(Kind.OR, any_bad, extra)
            solver.assertFormula(solver.mkTerm(Kind.NOT, any_bad))

        solver.assertFormula(solver.mkTerm(Kind.NOT, solver.mkTerm(Kind.FLOATINGPOINT_IS_ZERO, exact)))

        # everything below stays in double-precision FP -- no Real theory,
        # no FLOATINGPOINT_TO_REAL bridge (measured to be the actual
        # bottleneck; see module docstring).
        rm = solver.mkRoundingMode(_RM)
        exact_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, _DBL_E, _DBL_M + 1), rm, exact)
        approx_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, _DBL_E, _DBL_M + 1), rm, approx)
        diff = solver.mkTerm(Kind.FLOATINGPOINT_ABS,
                             solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, approx_dbl, exact_dbl))
        exact_abs = solver.mkTerm(Kind.FLOATINGPOINT_ABS, exact_dbl)
        alpha_const = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, _DBL_E, _DBL_M + 1),
                                    rm, solver.mkReal(alpha))
        threshold = solver.mkTerm(Kind.FLOATINGPOINT_MULT, rm, alpha_const, exact_abs)

        violation = solver.mkTerm(Kind.FLOATINGPOINT_GT, diff, threshold)
        solver.assertFormula(violation)

        result = solver.checkSat()

        if result.isUnsat():
            del solver
            gc.collect()
            return True, None

        if result.isSat():
            ma_val = _parse_bv_value(solver.getValue(ma_bv))
            mb_val = _parse_bv_value(solver.getValue(mb_bv))
            exact_val = _fp_dbl_to_float(solver.getValue(exact_dbl))
            approx_val = _fp_dbl_to_float(solver.getValue(approx_dbl))
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

    except UnsupportedFormat as e:
        if 'solver' in locals():
            del solver
        gc.collect()
        return False, {'reason': 'unsupported_format', 'message': str(e)}

    except Exception as e:
        message = str(e)
        if 'solver' in locals():
            del solver
        gc.collect()
        return False, {'reason': 'exception', 'message': message}
