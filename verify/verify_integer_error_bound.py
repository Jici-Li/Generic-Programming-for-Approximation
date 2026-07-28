"""
Formal verification: does the GP candidate satisfy the error bound
for all inputs in the specified range?

Uses:
  - typed BitVector for approx (GP candidate expression)
  - Real for exact (mathematical multiplication)
  - percentage error: |approx / exact - 1| <= alpha
"""

import gc
import os
import cvc5
from cvc5 import Kind

from verify.integer_cvc5_translate import gp_to_cvc5, cast_bv


TIMEOUT_MS = int(os.environ.get('GP_CVC5_TIMEOUT_MS', 30000))


def _parse_bv_value(term):
    """Convert a CVC5 BitVector value term into a Python int."""
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


def verify_error_bound(individual, pset, ma_bits, mb_bits, output_bits, alpha,
                       timeout_ms=TIMEOUT_MS, mantissa_domain=False):
    """
    Returns (verified: bool, counterexample: dict|None).
    verified = True  -> the property holds for all inputs (UNSAT)
    verified = False -> a counterexample was found (SAT), or unknown

    mantissa_domain: must match whatever benchmarks.integer.make_data was
    called with for this run. It only restricts the *training* samples GP
    is scored against; if the real spec allows small-magnitude ma/mb, this
    must stay False, since "VERIFIED" here would otherwise be a promise
    over a narrower domain than the circuit will actually see.
    """
    try:
        solver = cvc5.Solver()
        solver.setLogic("QF_BVNIRA")
        solver.setOption("produce-models", "true")
        solver.setOption("tlimit-per", str(timeout_ms))

        # ---- BitVector inputs (unsigned within full bit-width range) ----
        ma_bv = solver.mkConst(solver.mkBitVectorSort(ma_bits), 'ma')
        mb_bv = solver.mkConst(solver.mkBitVectorSort(mb_bits), 'mb')

        # ---- restrict to the same "top bit set" domain make_data samples
        # from when mantissa_domain=True (benchmarks/integer.py) ----
        if mantissa_domain:
            lo_a = solver.mkBitVector(ma_bits, 1 << (ma_bits - 1))
            lo_b = solver.mkBitVector(mb_bits, 1 << (mb_bits - 1))
            solver.assertFormula(
                solver.mkTerm(Kind.BITVECTOR_UGE, ma_bv, lo_a))
            solver.assertFormula(
                solver.mkTerm(Kind.BITVECTOR_UGE, mb_bv, lo_b))

        # ---- Translate GP tree into a CVC5 BitVector term ----
        approx_bv = gp_to_cvc5(individual, ma_bv, mb_bv, solver)
        approx_width = approx_bv.getSort().getBitVectorSize()

        # ---- Convert approx to Real (unsigned), rescaled back up ----
        # output_bits can be narrower than the true product (ma_bits+mb_bits);
        # the circuit's raw value is then an implicitly-shifted approximation
        # (see benchmarks/integer.py make_seed_str), so it has to be
        # rescaled before comparing against the true, unscaled exact product.
        shift = max(0, (ma_bits + mb_bits) - output_bits)
        approx_real = solver.mkTerm(
            Kind.TO_REAL,
            solver.mkTerm(Kind.BITVECTOR_TO_NAT, approx_bv)
        )
        if shift:
            approx_real = solver.mkTerm(
                Kind.MULT, approx_real, solver.mkReal(1 << shift)
            )

        # ---- exact = ma * mb as Real ----
        ma_real = solver.mkTerm(
            Kind.TO_REAL,
            solver.mkTerm(Kind.BITVECTOR_TO_NAT, ma_bv)
        )
        mb_real = solver.mkTerm(
            Kind.TO_REAL,
            solver.mkTerm(Kind.BITVECTOR_TO_NAT, mb_bv)
        )
        exact_real = solver.mkTerm(Kind.MULT, ma_real, mb_real)

        # ---- Assert exact > 0 to avoid division by zero ----
        zero_real = solver.mkReal(0)
        solver.assertFormula(solver.mkTerm(Kind.GT, exact_real, zero_real))

        # ---- Percentage error, division-free: |approx - exact| > alpha * exact ----
        # equivalent to |approx/exact - 1| > alpha given exact_real > 0
        # (asserted above) -- Kind.DIVISION in this mixed BV/Real theory is
        # expensive for the solver, cross-multiplying avoids it entirely.
        diff = solver.mkTerm(
            Kind.ITE,
            solver.mkTerm(Kind.GT, approx_real, exact_real),
            solver.mkTerm(Kind.SUB, approx_real, exact_real),
            solver.mkTerm(Kind.SUB, exact_real, approx_real),
        )

        # alpha as an exact rational
        alpha_int_num = int(round(alpha * 10000))
        alpha_rat = solver.mkReal(alpha_int_num, 10000)
        threshold = solver.mkTerm(Kind.MULT, alpha_rat, exact_real)

        # violation
        violation = solver.mkTerm(Kind.GT, diff, threshold)
        solver.assertFormula(violation)

        result = solver.checkSat()

        if result.isUnsat():
            del solver
            gc.collect()
            return True, None

        if result.isSat():
            ma_val = _parse_bv_value(solver.getValue(ma_bv))
            mb_val = _parse_bv_value(solver.getValue(mb_bv))
            approx_val = _parse_bv_value(solver.getValue(approx_bv)) << shift
            exact_val = ma_val * mb_val
            err = abs(approx_val - exact_val) / exact_val \
                if exact_val else float('inf')
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
        # same cleanup as the other return paths -- an uncleaned solver left
        # alive by an exception here (e.g. an unhandled terminal in
        # gp_to_cvc5) was observed to crash a *later*, unrelated
        # verify_error_bound() call in the same process (segfault).
        message = str(e)
        if 'solver' in locals():
            del solver
        gc.collect()
        return False, {'reason': 'exception', 'message': message}


# ==========================================================
# Quick self-test
# ==========================================================
if __name__ == "__main__":
    from deap import base, creator, gp
    from benchmarks.integer import make_pset, make_seed_str

    MA_BITS, MB_BITS, OUTPUT_BITS = 4, 4, 8

    print(f"Config: ma={MA_BITS}-bit, mb={MB_BITS}-bit, out={OUTPUT_BITS}-bit")
    pset = make_pset(MA_BITS, MB_BITS, OUTPUT_BITS)

    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    seed_str = make_seed_str(MA_BITS, MB_BITS, OUTPUT_BITS)
    print(f"\nTest 1: seed (exact truncated multiplier), alpha=0.01")
    print(f"  {seed_str}")
    seed_ind = creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset))
    verified, ce = verify_error_bound(seed_ind, pset,
                                      MA_BITS, MB_BITS, OUTPUT_BITS, 0.01)
    print(f"  -> verified={verified}, ce={ce}")

    print(f"\nTest 2: add(ma, mb) as approx, alpha=0.05 (should FAIL)")
    bad_str = f"add_{MA_BITS}_{MB_BITS}(ma, mb)"
    add_out = max(MA_BITS, MB_BITS) + 1
    if add_out < OUTPUT_BITS:
        # need padding, use trunc
        bad_str = bad_str
    bad_ind = creator.Individual(gp.PrimitiveTree.from_string(bad_str, pset))
    verified, ce = verify_error_bound(bad_ind, pset,
                                      MA_BITS, MB_BITS, OUTPUT_BITS, 0.05)
    print(f"  -> verified={verified}, ce={ce}")