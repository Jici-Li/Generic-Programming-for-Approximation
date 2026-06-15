"""
verify/verify_error_bound.py

Uses CVC5 to formally verify that a GP-found expression satisfies
the error constraint on ALL possible inputs (not just training samples).
"""

import cvc5
from cvc5 import Kind
from verify.cvc5_translate import gp_to_cvc5

SCALE      = 1000
TIMEOUT_MS = 10_000


def verify_error_bound(individual, pset, input_bits, alpha,
                       timeout_ms=TIMEOUT_MS):
    try:
        bv_width = input_bits * 2 + 8

        solver = cvc5.Solver()
        solver.setLogic("QF_BV")
        solver.setOption("tlimit", str(timeout_ms))
        solver.setOption("produce-models", "true")

        bv_sort = solver.mkBitVectorSort(bv_width)
        ma_bv   = solver.mkConst(bv_sort, 'ma')
        mb_bv   = solver.mkConst(bv_sort, 'mb')

        def bv(v):
            return solver.mkBitVector(bv_width, int(v) & ((1 << bv_width) - 1))

        lo = bv(2 ** input_bits)
        hi = bv(2 ** (input_bits + 1) - 1)
        solver.assertFormula(solver.mkTerm(Kind.BITVECTOR_UGE, ma_bv, lo))
        solver.assertFormula(solver.mkTerm(Kind.BITVECTOR_ULE, ma_bv, hi))
        solver.assertFormula(solver.mkTerm(Kind.BITVECTOR_UGE, mb_bv, lo))
        solver.assertFormula(solver.mkTerm(Kind.BITVECTOR_ULE, mb_bv, hi))

        exact = solver.mkTerm(Kind.BITVECTOR_MULT, ma_bv, mb_bv)

        print(f"    [CVC5] expression: {str(individual)[:100]}", flush=True)
        print(f"    [CVC5] nodes: {len(list(individual))}", flush=True)
        approx = gp_to_cvc5(individual, ma_bv, mb_bv, solver, bv_width)

        diff = solver.mkTerm(
            Kind.ITE,
            solver.mkTerm(Kind.BITVECTOR_UGT, approx, exact),
            solver.mkTerm(Kind.BITVECTOR_SUB, approx, exact),
            solver.mkTerm(Kind.BITVECTOR_SUB, exact, approx),
        )

        alpha_int    = int(alpha * SCALE)
        diff_scaled  = solver.mkTerm(Kind.BITVECTOR_MULT, diff,  bv(SCALE))
        exact_scaled = solver.mkTerm(Kind.BITVECTOR_MULT, exact, bv(alpha_int))
        violation    = solver.mkTerm(Kind.BITVECTOR_UGT, diff_scaled, exact_scaled)

        print(f"    [CVC5] building formula for alpha={alpha}...", flush=True)
        solver.assertFormula(violation)
        print(f"    [CVC5] calling checkSat...", flush=True)
        result = solver.checkSat()
        print(f"    [CVC5] done: {result}", flush=True)

        if result.isUnsat():
            return True, None

        if result.isSat():
            ma_val     = _bv_to_int(solver.getValue(ma_bv))
            mb_val     = _bv_to_int(solver.getValue(mb_bv))
            exact_val  = ma_val * mb_val
            approx_val = _bv_to_int(solver.getValue(approx))
            err        = abs(approx_val - exact_val) / exact_val \
                         if exact_val else float('inf')
            return False, {
                'ma': ma_val, 'mb': mb_val,
                'approx': approx_val, 'exact': exact_val,
                'error': err,
            }

        print(f"    CVC5 unknown/timeout for alpha={alpha}")
        return False, {'reason': 'timeout_or_unknown'}

    except Exception as e:
        print(f"    CVC5 exception: {e}")
        return False, {'reason': 'exception', 'message': str(e)}


def _bv_to_int(bv_term):
    s = str(bv_term)
    if s.startswith('#b'):
        return int(s[2:], 2)
    if s.startswith('#x'):
        return int(s[2:], 16)
    return int(s)


if __name__ == "__main__":
    import random
    from deap import base, creator, gp
    from benchmarks.int_mult import make_pset, make_seed_str

    INPUT_BITS = 4
    ALPHA      = 0.05
    pset       = make_pset(INPUT_BITS)

    if hasattr(creator, "FitnessMin"):  del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    seed_str = make_seed_str(INPUT_BITS + 1)
    seed_ind = creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset))
    print(f"\nTest 1: seed (exact), alpha={ALPHA}")
    verified, ce = verify_error_bound(seed_ind, pset, INPUT_BITS, ALPHA)
    print("  PASSED" if verified else f"  FAILED ce={ce}")

    bad_ind = creator.Individual(
        gp.PrimitiveTree.from_string("left_shift(ma, 3)", pset)
    )
    print(f"\nTest 2: left_shift(ma,3), alpha={ALPHA}")
    verified, ce = verify_error_bound(bad_ind, pset, INPUT_BITS, ALPHA)
    print(f"  Correctly rejected ce={ce}" if not verified else "  Should have failed")