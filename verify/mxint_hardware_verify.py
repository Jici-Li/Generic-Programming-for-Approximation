from cvc5 import Kind

from verify.integer_verify import cast_bv

def _signed_const(slv, width, value):
    return slv.mkBitVector(width, value & ((1 << width) - 1))

def _clip_signed_bv(slv, wide_val, dst_bits):
    wide_bits = wide_val.getSort().getBitVectorSize()
    min_dst = -(1 << (dst_bits - 1))
    max_dst = (1 << (dst_bits - 1)) - 1
    min_c = _signed_const(slv, wide_bits, min_dst)
    max_c = _signed_const(slv, wide_bits, max_dst)
    below = slv.mkTerm(Kind.BITVECTOR_SLT, wide_val, min_c)
    above = slv.mkTerm(Kind.BITVECTOR_SGT, wide_val, max_c)
    clamped = slv.mkTerm(Kind.ITE, below, min_c,
                         slv.mkTerm(Kind.ITE, above, max_c, wide_val))
    return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, dst_bits - 1, 0), clamped)

def _clip_unsigned_bv(slv, wide_val, dst_bits):
    wide_bits = wide_val.getSort().getBitVectorSize()
    max_dst = (1 << dst_bits) - 1
    max_c = slv.mkBitVector(wide_bits, max_dst)
    above = slv.mkTerm(Kind.BITVECTOR_UGT, wide_val, max_c)
    clamped = slv.mkTerm(Kind.ITE, above, max_c, wide_val)
    return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, dst_bits - 1, 0), clamped)

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
            if name.startswith('minus_one_'):
                w = int(name[len('minus_one_'):])
                return _signed_const(solver, w, -1), idx + 1
            if name.startswith('uzero_'):
                w = int(name[len('uzero_'):])
                return solver.mkBitVector(w, 0), idx + 1
            if name.startswith('uone_'):
                w = int(name[len('uone_'):])
                return solver.mkBitVector(w, 1), idx + 1
            if name.startswith('zero_'):
                w = int(name[len('zero_'):])
                return solver.mkBitVector(w, 0), idx + 1
            if name.startswith('one_'):
                w = int(name[len('one_'):])
                return solver.mkBitVector(w, 1), idx + 1
            if name == 'positive_sign':
                return solver.mkFalse(), idx + 1
            if name == 'negative_sign':
                return solver.mkTrue(), idx + 1
            raise ValueError(f"Unknown terminal: {name!r}")

        next_idx = idx + 1

        if name.startswith('smul_'):
            x, y = map(int, name[len('smul_'):].split('_'))
            out_bits = x + y
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits)
            r = cast_bv(solver, right, out_bits)
            return solver.mkTerm(Kind.BITVECTOR_MULT, l, r), next_idx

        if name.startswith('sadd_') or name.startswith('ssub_'):
            op = name[:4]
            x, y = map(int, name[len(op) + 1:].split('_'))
            out_bits = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits)
            r = cast_bv(solver, right, out_bits)
            kind = Kind.BITVECTOR_ADD if op == 'sadd' else Kind.BITVECTOR_SUB
            return solver.mkTerm(kind, l, r), next_idx

        if name.startswith('scast_'):
            x, y = map(int, name[len('scast_'):].split('_'))
            child, next_idx = convert(next_idx)
            if y >= x:
                return cast_bv(solver, child, y), next_idx
            shift = x - y
            wide_bits = x + 2
            wide = cast_bv(solver, child, wide_bits)
            bias = _signed_const(solver, wide_bits, 1 << (shift - 1))
            biased = solver.mkTerm(Kind.BITVECTOR_ADD, wide, bias)
            shift_amount = solver.mkBitVector(wide_bits, shift)
            rounded = solver.mkTerm(Kind.BITVECTOR_ASHR, biased, shift_amount)
            return _clip_signed_bv(solver, rounded, y), next_idx

        if name.startswith('ashr_'):
            rest = name[len('ashr_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            child, next_idx = convert(next_idx)
            shift_amount = solver.mkBitVector(x, k)
            return solver.mkTerm(Kind.BITVECTOR_ASHR, child, shift_amount), next_idx

        if name.startswith('sshl_'):
            rest = name[len('sshl_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            out_bits = x + k
            child, next_idx = convert(next_idx)
            child_ext = cast_bv(solver, child, out_bits)
            shift_amount = solver.mkBitVector(out_bits, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, child_ext, shift_amount), next_idx

        if name.startswith('umul_'):
            x, y = map(int, name[len('umul_'):].split('_'))
            out_bits = x + y
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_MULT, l, r), next_idx

        if name.startswith('uadd_'):
            x, y = map(int, name[len('uadd_'):].split('_'))
            out_bits = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_ADD, l, r), next_idx

        if name.startswith('ucast_'):
            x, y = map(int, name[len('ucast_'):].split('_'))
            child, next_idx = convert(next_idx)
            if y >= x:
                return cast_bv(solver, child, y, signed=False), next_idx
            return _clip_unsigned_bv(solver, child, y), next_idx

        if name.startswith('uhigh_'):
            src, dst = map(int, name[len('uhigh_'):].split('_'))
            shift = src - dst
            child, next_idx = convert(next_idx)
            shift_amount = solver.mkBitVector(src, shift)
            shifted = solver.mkTerm(Kind.BITVECTOR_LSHR, child, shift_amount)
            return solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, dst - 1, 0), shifted), next_idx

        if name.startswith('ulow_'):
            src, dst = map(int, name[len('ulow_'):].split('_'))
            child, next_idx = convert(next_idx)
            return solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, dst - 1, 0), child), next_idx

        if name.startswith('ulshift_'):
            rest = name[len('ulshift_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            out_bits = x + k
            child, next_idx = convert(next_idx)
            child_ext = cast_bv(solver, child, out_bits, signed=False)
            shift_amount = solver.mkBitVector(out_bits, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, child_ext, shift_amount), next_idx

        if name.startswith('abs_s'):
            # abs_s{n}_u{n} -- the two widths are always equal (see
            # benchmarks/mxint_hardware.py's make_pset)
            n = int(name[len('abs_s'):].split('_u')[0])
            child, next_idx = convert(next_idx)
            zero_c = solver.mkBitVector(n, 0)
            is_neg = solver.mkTerm(Kind.BITVECTOR_SLT, child, zero_c)
            neg = solver.mkTerm(Kind.BITVECTOR_NEG, child)
            return solver.mkTerm(Kind.ITE, is_neg, neg, child), next_idx

        if name.startswith('is_negative_'):
            bits = int(name[len('is_negative_'):])
            child, next_idx = convert(next_idx)
            zero_c = solver.mkBitVector(bits, 0)
            return solver.mkTerm(Kind.BITVECTOR_SLT, child, zero_c), next_idx

        if name == 'xor_sign':
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.XOR, left, right), next_idx

        if name.startswith('apply_sign_'):
            bits = int(name[len('apply_sign_'):])
            sign, next_idx = convert(next_idx)
            magnitude, next_idx = convert(next_idx)
            mag_wide = cast_bv(solver, magnitude, bits + 1, signed=False)
            neg_wide = solver.mkTerm(Kind.BITVECTOR_NEG, mag_wide)
            wide_signed = solver.mkTerm(Kind.ITE, sign, neg_wide, mag_wide)
            return _clip_signed_bv(solver, wide_signed, bits), next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    result, _ = convert(0)
    return result


import gc

import cvc5
from cvc5 import Kind


TIMEOUT_MS = 60000

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

def verify_error_bound(individual, pset, element_bits, output_bits, alpha,
                       timeout_ms=TIMEOUT_MS, mantissa_domain=False):
    try:
        solver = cvc5.Solver()
        solver.setLogic("QF_BVNIRA")
        solver.setOption("produce-models", "true")
        solver.setOption("tlimit-per", str(timeout_ms))

        ma_bv = solver.mkConst(solver.mkBitVectorSort(element_bits), 'ma')
        mb_bv = solver.mkConst(solver.mkBitVectorSort(element_bits), 'mb')

        if mantissa_domain:
            # restrict ma, mb to the upper half of the representable range --
            # must match benchmarks/mxint_hardware.py's make_data
            # mantissa_domain=True, same reasoning as integer's
            # mantissa_domain and minifloat's exclude_subnormal_inputs.
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


def _bv_sext(solver, bv, src, dst):
    if src == dst:
        return bv
    return solver.mkTerm(solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, dst - src), bv)

def _bv_zext(solver, bv, src, dst):
    if src == dst:
        return bv
    return solver.mkTerm(solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, dst - src), bv)

def _add_verify_setup(individual, element_bits, timeout_ms, mantissa_domain,
                      output_bits=None, exclude_overflow=False):
    # shared setup for verify_add_error_bound/verify_mxint_add_ulp_bound:
    # pure QF_BV (never QF_BVNIRA/Real) -- empirically, going through
    # BITVECTOR_SBV_TO_INT into Real/Int arithmetic (as verify_error_bound/
    # verify_mxint_ulp_bound do for Multiply, where the ma*mb ground truth
    # is genuinely nonlinear and needs it) makes CVC5 time out even on
    # this Add pset's exact, zero-error seed (>30s where the equivalent
    # Multiply check is near-instant) -- reformulating ma+mb and the
    # alpha/ULP comparison entirely in bitvector arithmetic (linear, no
    # NIRA needed since Add's ground truth is a sum, not a product) solves
    # in ~0.01s for the same seed.
    solver = cvc5.Solver()
    solver.setLogic("QF_BV")
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
    approx_w = approx_bv.getSort().getBitVectorSize()
    wide = approx_w + element_bits + 4  # generous headroom, no overflow in sub/add below

    ma_w = _bv_sext(solver, ma_bv, element_bits, wide)
    mb_w = _bv_sext(solver, mb_bv, element_bits, wide)
    approx_w_bv = _bv_sext(solver, approx_bv, approx_w, wide)
    exact = solver.mkTerm(Kind.BITVECTOR_ADD, ma_w, mb_w)
    zero_w = solver.mkBitVector(wide, 0)
    solver.assertFormula(solver.mkTerm(Kind.DISTINCT, exact, zero_w))

    if exclude_overflow:
        # A fixed-width accumulator (output_bits < element_bits+1, e.g.
        # DotProduct's Accum stage) MUST saturate on true overflow -- no
        # circuit can avoid that, it's an intrinsic limit of the output
        # register width, not something GP search can improve (same
        # "format-intrinsic floor" phenomenon as E4M3's own quantization
        # floor, just from saturation instead of rounding). Restricting
        # verification to inputs where the exact sum actually fits in
        # output_bits keeps the accuracy claim meaningful (asserts genuine
        # circuit quality) rather than vacuously failing on inputs no
        # accumulator width could ever handle.
        assert output_bits is not None
        out_min = -(1 << (output_bits - 1))
        out_max = (1 << (output_bits - 1)) - 1
        min_c = solver.mkBitVector(wide, out_min & ((1 << wide) - 1))
        max_c = solver.mkBitVector(wide, out_max & ((1 << wide) - 1))
        solver.assertFormula(solver.mkTerm(Kind.BITVECTOR_SGE, exact, min_c))
        solver.assertFormula(solver.mkTerm(Kind.BITVECTOR_SLE, exact, max_c))

    diff = solver.mkTerm(
        Kind.ITE,
        solver.mkTerm(Kind.BITVECTOR_SGT, approx_w_bv, exact),
        solver.mkTerm(Kind.BITVECTOR_SUB, approx_w_bv, exact),
        solver.mkTerm(Kind.BITVECTOR_SUB, exact, approx_w_bv),
    )
    exact_abs = solver.mkTerm(
        Kind.ITE,
        solver.mkTerm(Kind.BITVECTOR_SGE, exact, zero_w),
        exact,
        solver.mkTerm(Kind.BITVECTOR_NEG, exact),
    )
    return solver, ma_bv, mb_bv, approx_bv, wide, diff, exact_abs

def verify_add_error_bound(individual, pset, element_bits, output_bits, alpha,
                           timeout_ms=TIMEOUT_MS, mantissa_domain=False, exclude_overflow=False):
    # Add/Accum analogue of verify_error_bound: ground truth is ma+mb, not
    # ma*mb -- reflects reference_add_blocks' semantics (both operands
    # already share one exponent, so a straight integer sum is exact,
    # unlike Multiply where the ground truth genuinely is the product).
    # violation check: diff/exact_abs > alpha  <=>  diff*10000 >
    # round(alpha*10000)*exact_abs -- integer cross-multiplication instead
    # of a Real division, keeping this in pure QF_BV (see
    # _add_verify_setup).
    try:
        solver, ma_bv, mb_bv, approx_bv, wide, diff, exact_abs = _add_verify_setup(
            individual, element_bits, timeout_ms, mantissa_domain,
            output_bits=output_bits, exclude_overflow=exclude_overflow)

        alpha_num = int(round(alpha * 10000))
        mul_w = wide + 20
        lhs = solver.mkTerm(Kind.BITVECTOR_MULT,
                            _bv_zext(solver, diff, wide, mul_w),
                            solver.mkBitVector(mul_w, 10000))
        rhs = solver.mkTerm(Kind.BITVECTOR_MULT,
                            solver.mkBitVector(mul_w, alpha_num),
                            _bv_zext(solver, exact_abs, wide, mul_w))
        violation = solver.mkTerm(Kind.BITVECTOR_UGT, lhs, rhs)
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
            exact_val = ma_val + mb_val
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


def verify_mxint_add_ulp_bound(individual, pset, element_bits, output_bits, ulp_threshold,
                               timeout_ms=TIMEOUT_MS, mantissa_domain=False, exclude_overflow=False):
    # Add/Accum analogue of verify_mxint_ulp_bound (ma+mb ground truth,
    # same absolute-bound criterion since integer mantissas are evenly
    # spaced). Pure QF_BV, see _add_verify_setup.
    try:
        solver, ma_bv, mb_bv, approx_bv, wide, diff, _exact_abs = _add_verify_setup(
            individual, element_bits, timeout_ms, mantissa_domain,
            output_bits=output_bits, exclude_overflow=exclude_overflow)

        threshold_term = solver.mkBitVector(wide, ulp_threshold)
        violation = solver.mkTerm(Kind.BITVECTOR_UGT, diff, threshold_term)
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
            exact_val = ma_val + mb_val
            del solver
            gc.collect()
            return False, {
                'ma': ma_val, 'mb': mb_val,
                'approx': approx_val, 'exact': exact_val,
                'ulp_err': abs(approx_val - exact_val),
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


def verify_mxint_ulp_bound(individual, pset, element_bits, output_bits, ulp_threshold,
                           timeout_ms=TIMEOUT_MS, mantissa_domain=False):
    # ULP-threshold analogue of verify_error_bound, for consistency with
    # minifloat's verify_mult_ulp_bound/verify_add_ulp_bound (same
    # faithful-rounding-family criterion project-wide, not FP-only).
    # "ULP" for an integer mantissa is just 1 LSB -- integers are evenly
    # spaced, unlike FP where spacing scales with the exponent -- so the
    # criterion here is a plain absolute bound (|approx - exact| <=
    # ulp_threshold), not the floor/ceil-grid construction the FP
    # verifiers need to handle non-uniform spacing. This also drops the
    # alpha*|exact| SCALING term from the violation check (replaced by a
    # constant), which is one fewer layer of nonlinear arithmetic for
    # CVC5 to reason about than verify_error_bound's alpha-relative bound
    # -- the remaining nonlinearity (ma*mb as the ground truth) is
    # inherent to comparing against the true product at all, not
    # something switching from alpha to an absolute bound can remove.
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

        approx_int = solver.mkTerm(Kind.BITVECTOR_SBV_TO_INT, approx_bv)
        ma_int = solver.mkTerm(Kind.BITVECTOR_SBV_TO_INT, ma_bv)
        mb_int = solver.mkTerm(Kind.BITVECTOR_SBV_TO_INT, mb_bv)
        exact_int = solver.mkTerm(Kind.MULT, ma_int, mb_int)

        solver.assertFormula(solver.mkTerm(Kind.DISTINCT, exact_int, solver.mkInteger(0)))

        diff = solver.mkTerm(
            Kind.ITE,
            solver.mkTerm(Kind.GT, approx_int, exact_int),
            solver.mkTerm(Kind.SUB, approx_int, exact_int),
            solver.mkTerm(Kind.SUB, exact_int, approx_int),
        )
        threshold_term = solver.mkInteger(ulp_threshold)

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
            del solver
            gc.collect()
            return False, {
                'ma': ma_val, 'mb': mb_val,
                'approx': approx_val, 'exact': exact_val,
                'ulp_err': abs(approx_val - exact_val),
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
