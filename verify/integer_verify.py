
from cvc5 import Kind

def cast_bv(slv, x, n, signed=True):
    if n <= 0:
        raise ValueError("target width n must be positive")
    src_width = x.getSort().getBitVectorSize()
    if src_width == n:
        return x
    if src_width > n:
        return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, n - 1, 0), x)
    extension = n - src_width
    kind = Kind.BITVECTOR_SIGN_EXTEND if signed else Kind.BITVECTOR_ZERO_EXTEND
    return slv.mkTerm(slv.mkOp(kind, extension), x)

def gp_to_cvc5(individual, ma_bv, mb_bv, solver):
    nodes = list(individual)

    def convert(idx):
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        if not hasattr(node, 'arity') or node.arity == 0:
            if name in ('ma', 'ARG0'):
                return ma_bv, idx + 1
            if name in ('mb', 'ARG1'):
                return mb_bv, idx + 1
            if name.startswith('zero_'):
                w = int(name[len('zero_'):])
                return solver.mkBitVector(w, 0), idx + 1
            if name.startswith('one_'):
                w = int(name[len('one_'):])
                return solver.mkBitVector(w, 1), idx + 1
            if name.startswith('const_'):
                _, c_str, w_str = name.split('_')
                return solver.mkBitVector(int(w_str), int(c_str)), idx + 1
            raise ValueError(f"Unknown terminal: {name!r}")

        next_idx = idx + 1

        if name.startswith('mul_'):
            x, y = map(int, name[len('mul_'):].split('_'))
            out_bits = x + y
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_MULT, l, r), next_idx

        if name.startswith('blockmul_'):
            k_str, ma_str, mb_str = name[len('blockmul_'):].split('_')
            k, ma_bits, mb_bits = int(k_str), int(ma_str), int(mb_str)
            out_bits = ma_bits + mb_bits
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            a_h = solver.mkTerm(Kind.BITVECTOR_LSHR, a, solver.mkBitVector(ma_bits, k))
            b_h = solver.mkTerm(Kind.BITVECTOR_LSHR, b, solver.mkBitVector(mb_bits, k))
            a_h = cast_bv(solver, a_h, out_bits, signed=False)
            b_h = cast_bv(solver, b_h, out_bits, signed=False)
            prod = solver.mkTerm(Kind.BITVECTOR_MULT, a_h, b_h)
            shift_amount = solver.mkBitVector(out_bits, 2 * k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, prod, shift_amount), next_idx

        if name.startswith('crossmul_hilo_'):
            k_str, ma_str, mb_str = name[len('crossmul_hilo_'):].split('_')
            k, ma_bits, mb_bits = int(k_str), int(ma_str), int(mb_str)
            out_bits = ma_bits + mb_bits
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            a_h = solver.mkTerm(Kind.BITVECTOR_LSHR, a, solver.mkBitVector(ma_bits, k))
            mask = solver.mkBitVector(mb_bits, (1 << k) - 1)
            b_l = solver.mkTerm(Kind.BITVECTOR_AND, b, mask)
            a_h = cast_bv(solver, a_h, out_bits, signed=False)
            b_l = cast_bv(solver, b_l, out_bits, signed=False)
            prod = solver.mkTerm(Kind.BITVECTOR_MULT, a_h, b_l)
            shift_amount = solver.mkBitVector(out_bits, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, prod, shift_amount), next_idx

        if name.startswith('crossmul_lohi_'):
            k_str, ma_str, mb_str = name[len('crossmul_lohi_'):].split('_')
            k, ma_bits, mb_bits = int(k_str), int(ma_str), int(mb_str)
            out_bits = ma_bits + mb_bits
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            mask = solver.mkBitVector(ma_bits, (1 << k) - 1)
            a_l = solver.mkTerm(Kind.BITVECTOR_AND, a, mask)
            b_h = solver.mkTerm(Kind.BITVECTOR_LSHR, b, solver.mkBitVector(mb_bits, k))
            a_l = cast_bv(solver, a_l, out_bits, signed=False)
            b_h = cast_bv(solver, b_h, out_bits, signed=False)
            prod = solver.mkTerm(Kind.BITVECTOR_MULT, a_l, b_h)
            shift_amount = solver.mkBitVector(out_bits, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, prod, shift_amount), next_idx

        if name.startswith('add_'):
            x, y = map(int, name[len('add_'):].split('_'))
            out_bits = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_ADD, l, r), next_idx

        if name.startswith('sub_'):
            x, y = map(int, name[len('sub_'):].split('_'))
            out_bits = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_SUB, l, r), next_idx

        if name.startswith('and_'):
            x, y = map(int, name[len('and_'):].split('_'))
            out_bits = max(x, y)
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_AND, l, r), next_idx

        if name.startswith('or_'):
            x, y = map(int, name[len('or_'):].split('_'))
            out_bits = max(x, y)
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_OR, l, r), next_idx

        if name.startswith('lshift_'):
            rest = name[len('lshift_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            out_bits = x + k
            child, next_idx = convert(next_idx)
            child_ext = cast_bv(solver, child, out_bits, signed=False)
            shift_amount = solver.mkBitVector(out_bits, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, child_ext, shift_amount), next_idx

        if name.startswith('rshift_'):
            rest = name[len('rshift_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            child, next_idx = convert(next_idx)
            # k >= x: the shift constant itself no longer fits in an x-bit
            # bitvector (e.g. rshift_1_by3 needs to encode "3" in 1 bit),
            # but this exact combination is a real, well-defined primitive
            # in benchmarks/integer.py (_make_right_shift: (a>>k)&mask(x)
            # is always 0 once k>=x) -- make_pset generates rshift_x_byk
            # for every k up to max_shift regardless of x, so narrow-x/
            # large-k combinations do exist in the pset. Match the Python
            # semantics directly instead of trying to encode an
            # unrepresentable shift-amount constant.
            if k >= x:
                return solver.mkBitVector(x, 0), next_idx
            shift_amount = solver.mkBitVector(x, k)
            return solver.mkTerm(Kind.BITVECTOR_LSHR, child, shift_amount), next_idx

        if name.startswith('ite_'):
            w = int(name[len('ite_'):])
            cond, next_idx = convert(next_idx)
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            zero1 = solver.mkBitVector(1, 0)
            cond_bool = solver.mkTerm(Kind.DISTINCT, cond, zero1)
            a_cast = cast_bv(solver, a, w, signed=False)
            b_cast = cast_bv(solver, b, w, signed=False)
            return solver.mkTerm(Kind.ITE, cond_bool, a_cast, b_cast), next_idx

        if name.startswith('trunc_'):
            src, dst = map(int, name[len('trunc_'):].split('_'))
            child, next_idx = convert(next_idx)
            return cast_bv(solver, child, dst, signed=False), next_idx

        if name.startswith('rcast_'):
            src, dst = map(int, name[len('rcast_'):].split('_'))
            child, next_idx = convert(next_idx)
            shift = src - dst
            bias = 1 << (shift - 1)
            wide = cast_bv(solver, child, src + 1, signed=False)
            biased = solver.mkTerm(Kind.BITVECTOR_ADD, wide,
                                   solver.mkBitVector(src + 1, bias))
            shift_amount = solver.mkBitVector(src + 1, shift)
            rounded = solver.mkTerm(Kind.BITVECTOR_LSHR, biased, shift_amount)

            narrow = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, dst, 0), rounded)
            max_val = solver.mkBitVector(dst + 1, (1 << dst) - 1)
            overflowed = solver.mkTerm(Kind.BITVECTOR_UGT, narrow, max_val)
            clamped = solver.mkTerm(Kind.ITE, overflowed, max_val, narrow)
            return solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, dst - 1, 0), clamped), next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    result, _ = convert(0)
    return result

import gc
import os
import cvc5
from cvc5 import Kind


# overridable via GP_CVC5_TIMEOUT_MS -- wider bit-widths (16x16+) need a
# longer CVC5 budget to actually resolve to VERIFIED/counterexample instead
# of timing out inconclusive.
TIMEOUT_MS = int(os.environ.get('GP_CVC5_TIMEOUT_MS', 30000))

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

def verify_error_bound(individual, pset, ma_bits, mb_bits, output_bits, alpha,
                       timeout_ms=TIMEOUT_MS, mantissa_domain=False):
    # Returns (verified, counterexample) -- SAT here is what actually feeds
    # the CEGIS loop in search/integer_search.py: a returned counterexample
    # gets appended to the training set and GP re-runs, so this function is
    # where "keep testing for edge cases" concretely happens, not just the
    # loop that calls it.
    try:
        solver = cvc5.Solver()
        solver.setLogic("QF_BVNIRA")
        solver.setOption("produce-models", "true")
        solver.setOption("tlimit-per", str(timeout_ms))

        ma_bv = solver.mkConst(solver.mkBitVectorSort(ma_bits), 'ma')
        mb_bv = solver.mkConst(solver.mkBitVectorSort(mb_bits), 'mb')

        if mantissa_domain:
            # must match benchmarks/integer.py's make_data mantissa_domain
            # flag -- this restricts the *verification* domain the same way
            # that flag restricts the *training* domain (near-zero ma/mb
            # excluded, since narrowing them to zero causes unbounded
            # relative error); mismatching the two makes "VERIFIED" a
            # promise over a narrower domain than GP was scored on.
            lo_a = solver.mkBitVector(ma_bits, 1 << (ma_bits - 1))
            lo_b = solver.mkBitVector(mb_bits, 1 << (mb_bits - 1))
            solver.assertFormula(
                solver.mkTerm(Kind.BITVECTOR_UGE, ma_bv, lo_a))
            solver.assertFormula(
                solver.mkTerm(Kind.BITVECTOR_UGE, mb_bv, lo_b))

        approx_bv = gp_to_cvc5(individual, ma_bv, mb_bv, solver)
        approx_width = approx_bv.getSort().getBitVectorSize()

        shift = max(0, (ma_bits + mb_bits) - output_bits)
        approx_real = solver.mkTerm(
            Kind.TO_REAL,
            solver.mkTerm(Kind.BITVECTOR_TO_NAT, approx_bv)
        )
        if shift:
            approx_real = solver.mkTerm(
                Kind.MULT, approx_real, solver.mkReal(1 << shift)
            )

        ma_real = solver.mkTerm(
            Kind.TO_REAL,
            solver.mkTerm(Kind.BITVECTOR_TO_NAT, ma_bv)
        )
        mb_real = solver.mkTerm(
            Kind.TO_REAL,
            solver.mkTerm(Kind.BITVECTOR_TO_NAT, mb_bv)
        )
        exact_real = solver.mkTerm(Kind.MULT, ma_real, mb_real)

        zero_real = solver.mkReal(0)
        solver.assertFormula(solver.mkTerm(Kind.GT, exact_real, zero_real))

        diff = solver.mkTerm(
            Kind.ITE,
            solver.mkTerm(Kind.GT, approx_real, exact_real),
            solver.mkTerm(Kind.SUB, approx_real, exact_real),
            solver.mkTerm(Kind.SUB, exact_real, approx_real),
        )

        alpha_int_num = int(round(alpha * 10000))
        alpha_rat = solver.mkReal(alpha_int_num, 10000)
        threshold = solver.mkTerm(Kind.MULT, alpha_rat, exact_real)

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

        message = str(e)
        if 'solver' in locals():
            del solver
        gc.collect()
        return False, {'reason': 'exception', 'message': message}

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

        bad_str = bad_str
    bad_ind = creator.Individual(gp.PrimitiveTree.from_string(bad_str, pset))
    verified, ce = verify_error_bound(bad_ind, pset,
                                      MA_BITS, MB_BITS, OUTPUT_BITS, 0.05)
    print(f"  -> verified={verified}, ce={ce}")