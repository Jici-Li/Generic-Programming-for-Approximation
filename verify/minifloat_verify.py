import re
from fractions import Fraction

from cvc5 import Kind, RoundingMode

_RM = RoundingMode.ROUND_NEAREST_TIES_TO_EVEN

def _mkReal_exact(slv, value):#transfer  a float to rational number(precious)
    frac = Fraction(value)
    return slv.mkReal(frac.numerator, frac.denominator)

def _assert_input_range(solver, val_dbl, value_range, rm, dbl_e, dbl_m):
    if value_range is None:
        return
    lo, hi = value_range
    mk_dbl = lambda v: solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, dbl_e, dbl_m + 1),
                                     rm, _mkReal_exact(solver, v))
    solver.assertFormula(solver.mkTerm(Kind.FLOATINGPOINT_LEQ, mk_dbl(lo), val_dbl))
    solver.assertFormula(solver.mkTerm(Kind.FLOATINGPOINT_LEQ, val_dbl, mk_dbl(hi)))

class UnsupportedFormat(Exception):
    pass

def decode_native(slv, bits_bv, e, m):
    if m == 0:
        raise UnsupportedFormat(f"m=0 has no native FP representation (e={e})")
    return slv.mkTerm(slv.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_IEEE_BV, e, m + 1), bits_bv)#using the function in cvc5:FLOATINGPOINT_TO_FP_FROM_IEEE to decode the bitvector to floating point number

def decode_native_satmax(slv, bits_bv, e, m):
    if m == 0:
        raise UnsupportedFormat(f"m=0 has no native FP representation (e={e})")
    width = e + m + 1
    sign = slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, width - 1, width - 1), bits_bv)
    expf = slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, width - 2, m), bits_bv)
    is_max_exp = slv.mkTerm(Kind.EQUAL, expf, slv.mkBitVector(e, (1 << e) - 1))
    is_neg = slv.mkTerm(Kind.EQUAL, sign, slv.mkBitVector(1, 1))
    pos_inf = slv.mkFloatingPointPosInf(e, m + 1)
    neg_inf = slv.mkFloatingPointNegInf(e, m + 1)
    inf_signed = slv.mkTerm(Kind.ITE, is_neg, neg_inf, pos_inf)
    native = decode_native(slv, bits_bv, e, m)
    return slv.mkTerm(Kind.ITE, is_max_exp, inf_signed, native)#check if the number is abormal, if not, return native for this number

def widen(slv, fp_val, e, m):
    if m == 0:
        raise UnsupportedFormat(f"m=0 has no native FP representation (e={e})")
    return slv.mkTerm(slv.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, e, m + 1), slv.mkRoundingMode(_RM), fp_val)#transfer the number to a widen format with widen bitwidth to avoid rounding

def _clip_unsigned_bv(slv, wide_val, dst_bits):
    wide_bits = wide_val.getSort().getBitVectorSize()
    max_dst = (1 << dst_bits) - 1
    max_c = slv.mkBitVector(wide_bits, max_dst)
    above = slv.mkTerm(Kind.BITVECTOR_UGT, wide_val, max_c)
    clamped = slv.mkTerm(Kind.ITE, above, max_c, wide_val)
    return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, dst_bits - 1, 0), clamped)#clip the wide value to the dst_bits, if the wide value is larger than the max value of dst_bits, return the max value of dst_bits

def _zext(slv, x, n):
    src = x.getSort().getBitVectorSize()
    if src == n:
        return x
    if src > n:
        return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, n - 1, 0), x)
    return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_ZERO_EXTEND, n - src), x)#zero extend the x to n bits, if the src is larger than n, return the lower n bits of x

def _usub_bv(slv, left, right, out_bits):
    ext = out_bits + 1
    l = _zext(slv, left, ext)
    r = _zext(slv, right, ext)
    diff = slv.mkTerm(Kind.BITVECTOR_SUB, l, r)
    is_neg = slv.mkTerm(Kind.BITVECTOR_SLT, diff, slv.mkBitVector(ext, 0))
    clamped = slv.mkTerm(Kind.ITE, is_neg, slv.mkBitVector(ext, 0), diff)
    return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, out_bits - 1, 0), clamped)#subtract two unsigned bitvectors, if the result is negative, return 0, otherwise return the result

def _ushr_bv(slv, value, amount, x_bits):
    y_bits = amount.getSort().getBitVectorSize()
    w = max(x_bits, y_bits)
    v_w = _zext(slv, value, w)
    a_w = _zext(slv, amount, w)
    shifted = slv.mkTerm(Kind.BITVECTOR_LSHR, v_w, a_w)
    return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, x_bits - 1, 0), shifted)#logical right shift of an unsigned bitvector by a variable amount, return the lower x_bits of the resultg

def _ushl_bv(slv, value, amount, x_bits):
    y_bits = amount.getSort().getBitVectorSize()
    w = x_bits + y_bits
    v_w = _zext(slv, value, w)
    a_w = _zext(slv, amount, w)
    shifted = slv.mkTerm(Kind.BITVECTOR_SHL, v_w, a_w)
    return _clip_unsigned_bv(slv, shifted, x_bits)

def _lzc_bv(slv, value, n_bits, out_bits):
    result = slv.mkBitVector(out_bits, n_bits)
    for pos in range(0, n_bits):
        bit = slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, pos, pos), value)
        is_one = slv.mkTerm(Kind.EQUAL, bit, slv.mkBitVector(1, 1))
        count_here = slv.mkBitVector(out_bits, n_bits - 1 - pos)
        result = slv.mkTerm(Kind.ITE, is_one, count_here, result)
    return result#count the leading zeros of an unsigned bitvector, return the count as a bitvector of out_bits

def _sticky_bv(slv, value, amount, x_bits):
    shr = _ushr_bv(slv, value, amount, x_bits)
    y_bits = amount.getSort().getBitVectorSize()
    w = x_bits + y_bits
    shr_w = _zext(slv, shr, w)
    amount_w = _zext(slv, amount, w)
    shl_back = slv.mkTerm(Kind.BITVECTOR_SHL, shr_w, amount_w)
    shl_back_trunc = slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, x_bits - 1, 0), shl_back)
    differs = slv.mkTerm(Kind.DISTINCT, value, shl_back_trunc)
    return slv.mkTerm(Kind.ITE, differs, slv.mkBitVector(1, 1), slv.mkBitVector(1, 0))

def gp_to_cvc5(individual, ma_bv, mb_bv, solver, src_format=(4, 3)):
    e, m = src_format
    nodes = list(individual)

    def convert(idx):
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        if not hasattr(node, 'arity') or node.arity == 0:
            if name in ('ma', 'ARG0'):
                return ma_bv, idx + 1
            if name in ('mb', 'ARG1'):
                return mb_bv, idx + 1
            if name.startswith('const_'):
                _, c_str, w_str = name.split('_')
                return solver.mkBitVector(int(w_str), int(c_str)), idx + 1
            if name.startswith('uzero_'):
                w = int(name[len('uzero_'):])
                return solver.mkBitVector(w, 0), idx + 1
            if name.startswith('uone_'):
                w = int(name[len('uone_'):])
                return solver.mkBitVector(w, 1), idx + 1
            if name == 'positive_sign':
                return solver.mkBitVector(1, 0), idx + 1
            if name == 'negative_sign':
                return solver.mkBitVector(1, 1), idx + 1
            raise ValueError(f"Unknown terminal: {name!r}")

        next_idx = idx + 1

        if name.startswith('sign_e'):
            child, next_idx = convert(next_idx)
            return solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, e + m, e + m), child), next_idx

        if name.startswith('expf_e'):
            child, next_idx = convert(next_idx)
            return solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, e + m - 1, m), child), next_idx

        if name.startswith('sig_e'):
            child, next_idx = convert(next_idx)
            frac = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m - 1, 0), child)
            hidden = solver.mkBitVector(1, 1)
            return solver.mkTerm(Kind.BITVECTOR_CONCAT, hidden, frac), next_idx

        if name.startswith('encode_e'):
            sign, next_idx = convert(next_idx)
            expf, next_idx = convert(next_idx)
            frac, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_CONCAT, solver.mkTerm(Kind.BITVECTOR_CONCAT, sign, expf), frac), next_idx

        if name == 'xor_sign':
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_XOR, left, right), next_idx

        if name == 'bit_to_u1':
            child, next_idx = convert(next_idx)
            return child, next_idx

        if name.startswith('msb_u'):
            n = int(name[len('msb_u'):])
            child, next_idx = convert(next_idx)
            return solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, n - 1, n - 1), child), next_idx

        if name.startswith('ugt_u'):
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            gt_bool = solver.mkTerm(Kind.BITVECTOR_UGT, left, right)
            return solver.mkTerm(Kind.ITE, gt_bool, solver.mkBitVector(1, 1), solver.mkBitVector(1, 0)), next_idx

        if name.startswith('ite_u'):
            n = int(name[len('ite_u'):])
            cond, next_idx = convert(next_idx)
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            cond_bool = solver.mkTerm(Kind.EQUAL, cond, solver.mkBitVector(1, 1))
            return solver.mkTerm(Kind.ITE, cond_bool, a, b), next_idx

        if name.startswith('umul_'):
            x, y = map(int, name[len('umul_'):].split('_'))
            out = x + y
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = _zext(solver, left, out)
            r = _zext(solver, right, out)
            return solver.mkTerm(Kind.BITVECTOR_MULT, l, r), next_idx

        if name.startswith('uadd_'):
            x, y = map(int, name[len('uadd_'):].split('_'))
            out = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = _zext(solver, left, out)
            r = _zext(solver, right, out)
            return solver.mkTerm(Kind.BITVECTOR_ADD, l, r), next_idx

        if name.startswith('uaddc_'):
            x, y = map(int, name[len('uaddc_'):].split('_'))
            out = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            cin, next_idx = convert(next_idx)  # Sign (1-bit bitvector)
            l = _zext(solver, left, out)
            r = _zext(solver, right, out)
            c = _zext(solver, cin, out)
            sum1 = solver.mkTerm(Kind.BITVECTOR_ADD, l, r)
            return solver.mkTerm(Kind.BITVECTOR_ADD, sum1, c), next_idx

        if name.startswith('ucast_'):
            x, y = map(int, name[len('ucast_'):].split('_'))
            child, next_idx = convert(next_idx)
            if y >= x:
                return _zext(solver, child, y), next_idx
            return _clip_unsigned_bv(solver, child, y), next_idx

        if name.startswith('uhigh_'):
            src, dst = map(int, name[len('uhigh_'):].split('_'))
            child, next_idx = convert(next_idx)
            shift_amount = solver.mkBitVector(src, src - dst)
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
            out = x + k
            child, next_idx = convert(next_idx)
            child_ext = _zext(solver, child, out)
            shift_amount = solver.mkBitVector(out, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, child_ext, shift_amount), next_idx

        if name.startswith('usub_'):
            x, y = map(int, name[len('usub_'):].split('_'))
            out = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return _usub_bv(solver, left, right, out), next_idx

        if name.startswith('ushr_'):
            x, y = map(int, name[len('ushr_'):].split('_'))
            value, next_idx = convert(next_idx)
            amount, next_idx = convert(next_idx)
            return _ushr_bv(solver, value, amount, x), next_idx

        if name.startswith('ushl_'):
            x, y = map(int, name[len('ushl_'):].split('_'))
            value, next_idx = convert(next_idx)
            amount, next_idx = convert(next_idx)
            return _ushl_bv(solver, value, amount, x), next_idx

        if name.startswith('lzc_u'):
            n = int(name[len('lzc_u'):])
            out_bits = int(node.ret.__name__[len('UInt'):])
            child, next_idx = convert(next_idx)
            return _lzc_bv(solver, child, n, out_bits), next_idx

        if name.startswith('sticky_'):
            x, y = map(int, name[len('sticky_'):].split('_'))
            value, next_idx = convert(next_idx)
            amount, next_idx = convert(next_idx)
            return _sticky_bv(solver, value, amount, x), next_idx

        if name.startswith('unot_'):
            x = int(name[len('unot_'):])
            child, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_NOT, child), next_idx

        if name.startswith('orbit_u'):
            x = int(name[len('orbit_u'):])
            value, next_idx = convert(next_idx)
            flag, next_idx = convert(next_idx)
            flag_x = _zext(solver, flag, x)
            return solver.mkTerm(Kind.BITVECTOR_OR, value, flag_x), next_idx

        if name == 'ite_sign':
            cond, next_idx = convert(next_idx)
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            cond_bool = solver.mkTerm(Kind.EQUAL, cond, solver.mkBitVector(1, 1))
            return solver.mkTerm(Kind.ITE, cond_bool, a, b), next_idx

        if name.startswith('uconcat_'):
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_CONCAT, left, right), next_idx

        if name.startswith('sigmul_'):
            x, y = map(int, name[len('sigmul_'):].split('_'))
            out = x + y
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = _zext(solver, left, out)
            r = _zext(solver, right, out)
            return solver.mkTerm(Kind.BITVECTOR_MULT, l, r), next_idx

        if name.startswith('usubwrap_'):
            x, y = map(int, name[len('usubwrap_'):].split('_'))
            out = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = _zext(solver, left, out)
            r = _zext(solver, right, out)
            return solver.mkTerm(Kind.BITVECTOR_SUB, l, r), next_idx

        if name.startswith('roundadd_'):
            x, y = map(int, name[len('roundadd_'):].split('_'))
            out = max(x, y)
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            cin, next_idx = convert(next_idx)
            l = _zext(solver, left, out)
            r = _zext(solver, right, out)
            c = _zext(solver, cin, out)
            sum1 = solver.mkTerm(Kind.BITVECTOR_ADD, l, r)
            return solver.mkTerm(Kind.BITVECTOR_ADD, sum1, c), next_idx

        if name.startswith('ueq_u'):
            m2 = re.match(r'ueq_u(\d+)_is(\d+)', name)
            n, v = int(m2.group(1)), int(m2.group(2))
            child, next_idx = convert(next_idx)
            eq_bool = solver.mkTerm(Kind.EQUAL, child, solver.mkBitVector(n, v))
            return solver.mkTerm(Kind.ITE, eq_bool, solver.mkBitVector(1, 1), solver.mkBitVector(1, 0)), next_idx

        if name == 'exc_lookup':
            child, next_idx = convert(next_idx)
            table = {0: 0, 1: 0, 4: 0, 5: 1, 6: 2, 9: 2, 10: 2}
            result = solver.mkBitVector(2, 3)
            for k, v in table.items():
                is_k = solver.mkTerm(Kind.EQUAL, child, solver.mkBitVector(4, k))
                result = solver.mkTerm(Kind.ITE, is_k, solver.mkBitVector(2, v), result)
            return result, next_idx

        if name == 'excpostnorm_lookup':
            child, next_idx = convert(next_idx)
            table = {0: 1, 1: 2, 2: 0, 3: 0}
            result = solver.mkBitVector(2, 3)
            for k, v in table.items():
                is_k = solver.mkTerm(Kind.EQUAL, child, solver.mkBitVector(2, k))
                result = solver.mkTerm(Kind.ITE, is_k, solver.mkBitVector(2, v), result)
            return result, next_idx

        if name.startswith('classify_e'):
            rest = name[len('classify_e'):]
            e_val_str, m_val_str = rest.split('m')
            e_val, m_val = int(e_val_str), int(m_val_str)
            child, next_idx = convert(next_idx)
            exp_mask_val = (1 << e_val) - 1

            expf = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, e_val + m_val - 1, m_val), child)
            frac = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m_val - 1, 0), child)

            exp_zero = solver.mkTerm(Kind.EQUAL, expf, solver.mkBitVector(e_val, 0))
            exp_infty = solver.mkTerm(Kind.EQUAL, expf, solver.mkBitVector(e_val, exp_mask_val))
            frac_zero = solver.mkTerm(Kind.EQUAL, frac, solver.mkBitVector(m_val, 0))
            repr_subnormal_bit = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m_val - 1, m_val - 1), frac)
            repr_subnormal = solver.mkTerm(Kind.EQUAL, repr_subnormal_bit, solver.mkBitVector(1, 1))

            infinity = solver.mkTerm(Kind.AND, exp_infty, frac_zero)
            not_repr_subnormal = solver.mkTerm(Kind.NOT, repr_subnormal)
            zero_flag = solver.mkTerm(Kind.AND, exp_zero, not_repr_subnormal)
            not_frac_zero = solver.mkTerm(Kind.NOT, frac_zero)
            nan_flag = solver.mkTerm(Kind.AND, exp_infty, not_frac_zero)
            result = solver.mkTerm(Kind.ITE, zero_flag, solver.mkBitVector(2, 0),
                     solver.mkTerm(Kind.ITE, infinity, solver.mkBitVector(2, 2),
                     solver.mkTerm(Kind.ITE, nan_flag, solver.mkBitVector(2, 3),
                     solver.mkBitVector(2, 1))))
            return result, next_idx

        if name.startswith('input_ieee_e'):
            rest = name[len('input_ieee_e'):]
            e_val_str, m_val_str = rest.split('m')
            e_val, m_val = int(e_val_str), int(m_val_str)
            child, next_idx = convert(next_idx)
            exp_mask_val = (1 << e_val) - 1

            sign = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, e_val + m_val, e_val + m_val), child)
            expf = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, e_val + m_val - 1, m_val), child)
            frac = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m_val - 1, 0), child)

            exp_zero = solver.mkTerm(Kind.EQUAL, expf, solver.mkBitVector(e_val, 0))
            exp_infty = solver.mkTerm(Kind.EQUAL, expf, solver.mkBitVector(e_val, exp_mask_val))
            frac_zero = solver.mkTerm(Kind.EQUAL, frac, solver.mkBitVector(m_val, 0))
            repr_subnormal_bit = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m_val - 1, m_val - 1), frac)
            repr_subnormal = solver.mkTerm(Kind.EQUAL, repr_subnormal_bit, solver.mkBitVector(1, 1))

            if m_val > 1:
                frac_low = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m_val - 2, 0), frac)
                sfrac_sub = solver.mkTerm(Kind.BITVECTOR_CONCAT, frac_low, solver.mkBitVector(1, 0))
            else:
                sfrac_sub = solver.mkBitVector(m_val, 0)
            exp_zero_and_subnorm = solver.mkTerm(Kind.AND, exp_zero, repr_subnormal)
            sfrac = solver.mkTerm(Kind.ITE, exp_zero_and_subnorm, sfrac_sub, frac)

            infinity = solver.mkTerm(Kind.AND, exp_infty, frac_zero)
            not_repr_subnormal = solver.mkTerm(Kind.NOT, repr_subnormal)
            zero_flag = solver.mkTerm(Kind.AND, exp_zero, not_repr_subnormal)
            not_frac_zero = solver.mkTerm(Kind.NOT, frac_zero)
            nan_flag = solver.mkTerm(Kind.AND, exp_infty, not_frac_zero)
            exn = solver.mkTerm(Kind.ITE, zero_flag, solver.mkBitVector(2, 0),
                  solver.mkTerm(Kind.ITE, infinity, solver.mkBitVector(2, 2),
                  solver.mkTerm(Kind.ITE, nan_flag, solver.mkBitVector(2, 3),
                  solver.mkBitVector(2, 1))))

            packed = solver.mkTerm(Kind.BITVECTOR_CONCAT, exn,
                     solver.mkTerm(Kind.BITVECTOR_CONCAT, sign,
                     solver.mkTerm(Kind.BITVECTOR_CONCAT, expf, sfrac)))
            return packed, next_idx

        if name.startswith('output_ieee_e'):
            rest = name[len('output_ieee_e'):]
            eo_str, mo_str = rest.split('m')
            eo_val, mo_val = int(eo_str), int(mo_str)
            exn, next_idx = convert(next_idx)
            sign, next_idx = convert(next_idx)
            expf, next_idx = convert(next_idx)
            frac, next_idx = convert(next_idx)
            exp_mask_val = (1 << eo_val) - 1

            exn_is0 = solver.mkTerm(Kind.EQUAL, exn, solver.mkBitVector(2, 0))
            exn_is1 = solver.mkTerm(Kind.EQUAL, exn, solver.mkBitVector(2, 1))
            exn_le2 = solver.mkTerm(Kind.BITVECTOR_ULE, exn, solver.mkBitVector(2, 2))
            exp_zero = solver.mkTerm(Kind.EQUAL, expf, solver.mkBitVector(eo_val, 0))

            out_sign = solver.mkTerm(Kind.ITE, exn_le2, sign, solver.mkBitVector(1, 0))

            one_mo = solver.mkBitVector(mo_val, 1)
            frac_shr1 = solver.mkTerm(Kind.BITVECTOR_LSHR, frac, one_mo)
            hidden_bit_tag = solver.mkBitVector(mo_val, 1 << (mo_val - 1))
            subnorm_frac = solver.mkTerm(Kind.BITVECTOR_OR, hidden_bit_tag, frac_shr1)
            exn_is1_and_expzero = solver.mkTerm(Kind.AND, exn_is1, exp_zero)
            exn_and_1 = _zext(solver, solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, 0, 0), exn), mo_val)
            frac_r = solver.mkTerm(Kind.ITE, exn_is0, solver.mkBitVector(mo_val, 0),
                     solver.mkTerm(Kind.ITE, exn_is1_and_expzero, subnorm_frac,
                     solver.mkTerm(Kind.ITE, exn_is1, frac,
                     exn_and_1)))

            exp_r = solver.mkTerm(Kind.ITE, exn_is0, solver.mkBitVector(eo_val, 0),
                    solver.mkTerm(Kind.ITE, exn_is1, expf,
                    solver.mkBitVector(eo_val, exp_mask_val)))

            packed = solver.mkTerm(Kind.BITVECTOR_CONCAT, out_sign,
                     solver.mkTerm(Kind.BITVECTOR_CONCAT, exp_r, frac_r))
            return packed, next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    result, _ = convert(0)
    return result


import gc
import struct

import cvc5

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
    _, _, bv_term = term.getFloatingPointValue()
    bits = _parse_bv_value(bv_term)
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def verify_add_ulp_bound(individual, src_format, out_format, ulp_threshold,
                         timeout_ms=TIMEOUT_MS, exclude_subnormal_inputs=True,
                         ma_range=None, mb_range=None):
    e, m = src_format
    eo, mo = out_format
    in_width = e + m + 1
    out_sig = mo + 1

    try:
        solver = cvc5.Solver()
        solver.setLogic("ALL")
        solver.setOption("fp-exp", "true")
        solver.setOption("produce-models", "true")
        solver.setOption("tlimit-per", str(timeout_ms))

        ma_bv = solver.mkConst(solver.mkBitVectorSort(in_width), 'ma')
        mb_bv = solver.mkConst(solver.mkBitVectorSort(in_width), 'mb')#Assume there's two unknown number ma&mb

        ma_exp = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, in_width - 2, m), ma_bv)
        mb_exp = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, in_width - 2, m), mb_bv)
        if exclude_subnormal_inputs:
            solver.assertFormula(solver.mkTerm(Kind.DISTINCT, ma_exp, solver.mkBitVector(e, 0)))
            solver.assertFormula(solver.mkTerm(Kind.DISTINCT, mb_exp, solver.mkBitVector(e, 0)))
        solver.assertFormula(solver.mkTerm(Kind.DISTINCT, ma_exp, solver.mkBitVector(e, (1 << e) - 1)))
        solver.assertFormula(solver.mkTerm(Kind.DISTINCT, mb_exp, solver.mkBitVector(e, (1 << e) - 1)))

        rm = solver.mkRoundingMode(_RM)
        ma_dbl = widen(solver, decode_native(solver, ma_bv, e, m), _DBL_E, _DBL_M)
        mb_dbl = widen(solver, decode_native(solver, mb_bv, e, m), _DBL_E, _DBL_M)
        _assert_input_range(solver, ma_dbl, ma_range, rm, _DBL_E, _DBL_M)
        _assert_input_range(solver, mb_dbl, mb_range, rm, _DBL_E, _DBL_M)
        exact_dbl = solver.mkTerm(Kind.FLOATINGPOINT_ADD, rm, ma_dbl, mb_dbl)
        exact_finite = solver.mkTerm(Kind.NOT, solver.mkTerm(Kind.OR,
            solver.mkTerm(Kind.FLOATINGPOINT_IS_INF, exact_dbl),
            solver.mkTerm(Kind.FLOATINGPOINT_IS_NAN, exact_dbl)))
        solver.assertFormula(exact_finite)
        solver.assertFormula(solver.mkTerm(Kind.NOT, solver.mkTerm(Kind.FLOATINGPOINT_IS_ZERO, exact_dbl)))

        from benchmarks.minifloat import _bias
        min_normal_val = 2.0 ** (1 - _bias(eo))
        min_normal_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, _DBL_E, _DBL_M + 1),
                                       rm, _mkReal_exact(solver, min_normal_val))
        exact_abs_dbl = solver.mkTerm(Kind.FLOATINGPOINT_ABS, exact_dbl)
        is_underflow = solver.mkTerm(Kind.FLOATINGPOINT_LT, exact_abs_dbl, min_normal_dbl)
        is_neg = solver.mkTerm(Kind.FLOATINGPOINT_IS_NEG, exact_dbl)

        zero_pos = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, eo, out_sig), rm, solver.mkReal(0))
        zero_neg = solver.mkTerm(Kind.FLOATINGPOINT_NEG, zero_pos)
        min_normal_pos = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, eo, out_sig),
                                       rm, _mkReal_exact(solver, min_normal_val))
        min_normal_neg = solver.mkTerm(Kind.FLOATINGPOINT_NEG, min_normal_pos)
        zero_signed = solver.mkTerm(Kind.ITE, is_neg, zero_neg, zero_pos)
        min_normal_signed = solver.mkTerm(Kind.ITE, is_neg, min_normal_neg, min_normal_pos)
        nudge = solver.mkTerm(Kind.FLOATINGPOINT_MULT, rm, exact_abs_dbl,
                              solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, _DBL_E, _DBL_M + 1),
                                           rm, _mkReal_exact(solver, 2.0 ** -40)))
        exact_for_ceil = solver.mkTerm(Kind.FLOATINGPOINT_ADD, rm, exact_dbl, nudge)
        exact_for_floor = solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, exact_dbl, nudge)

        ceil_native = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, eo, out_sig),
                                    solver.mkRoundingMode(RoundingMode.ROUND_TOWARD_POSITIVE), exact_for_ceil)
        floor_native = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, eo, out_sig),
                                     solver.mkRoundingMode(RoundingMode.ROUND_TOWARD_NEGATIVE), exact_for_floor)

        candidate_lo = solver.mkTerm(Kind.ITE, is_underflow, zero_signed, floor_native)
        candidate_hi = solver.mkTerm(Kind.ITE, is_underflow, min_normal_signed, ceil_native)

        lo_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, _DBL_E, _DBL_M + 1), rm, candidate_lo)
        hi_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, _DBL_E, _DBL_M + 1), rm, candidate_hi)
        dist_lo = solver.mkTerm(Kind.FLOATINGPOINT_ABS, solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, exact_dbl, lo_dbl))
        dist_hi = solver.mkTerm(Kind.FLOATINGPOINT_ABS, solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, exact_dbl, hi_dbl))
        lo_is_nearer = solver.mkTerm(Kind.FLOATINGPOINT_LT, dist_lo, dist_hi)
        round_nearest_dbl = solver.mkTerm(Kind.ITE, lo_is_nearer, lo_dbl, hi_dbl)

        ulp_size_dbl = solver.mkTerm(Kind.FLOATINGPOINT_ABS,
                                     solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, hi_dbl, lo_dbl))

        candidate_bv = gp_to_cvc5(individual, ma_bv, mb_bv, solver, src_format=src_format)
        approx = decode_native_satmax(solver, candidate_bv, eo, mo)
        approx_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, _DBL_E, _DBL_M + 1), rm, approx)

        diff = solver.mkTerm(Kind.FLOATINGPOINT_ABS,
                             solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, approx_dbl, round_nearest_dbl))
        threshold_const = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, _DBL_E, _DBL_M + 1),
                                        rm, _mkReal_exact(solver, ulp_threshold))
        threshold_dbl = solver.mkTerm(Kind.FLOATINGPOINT_MULT, rm, threshold_const, ulp_size_dbl)

        violation = solver.mkTerm(Kind.FLOATINGPOINT_GT, diff, threshold_dbl)
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


def verify_mult_ulp_bound(individual, src_format, out_format, ulp_threshold,
                          timeout_ms=TIMEOUT_MS,
                          ma_range=None, mb_range=None):
    #generate some basic setup:bitwidth, exponent&mantissa value
    e, m = src_format
    eo, mo = out_format
    in_width = e + m + 1
    out_sig = mo + 1

    try:
        solver = cvc5.Solver()
        solver.setLogic("ALL")
        solver.setOption("fp-exp", "true")
        solver.setOption("produce-models", "true")
        solver.setOption("tlimit-per", str(timeout_ms))

        ma_bv = solver.mkConst(solver.mkBitVectorSort(in_width), 'ma')
        mb_bv = solver.mkConst(solver.mkBitVectorSort(in_width), 'mb')#setup ma&mb as two unknown bitvector

        ma_exp = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, in_width - 2, m), ma_bv)#get the exponent and mantissa 
        mb_exp = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, in_width - 2, m), mb_bv)
        ma_frac = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m - 1, 0), ma_bv)
        mb_frac = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m - 1, 0), mb_bv)
        ma_exp_zero = solver.mkTerm(Kind.EQUAL, ma_exp, solver.mkBitVector(e, 0))
        mb_exp_zero = solver.mkTerm(Kind.EQUAL, mb_exp, solver.mkBitVector(e, 0))#decide if the exponent is all 0
        ma_top_frac_bit = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m - 1, m - 1), ma_frac)
        mb_top_frac_bit = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, m - 1, m - 1), mb_frac)#get the firstt bit of mantissa
        ma_top_frac_zero = solver.mkTerm(Kind.EQUAL, ma_top_frac_bit, solver.mkBitVector(1, 0))
        mb_top_frac_zero = solver.mkTerm(Kind.EQUAL, mb_top_frac_bit, solver.mkBitVector(1, 0))#decide if it's 0
        ma_is_flush_zero = solver.mkTerm(Kind.AND, ma_exp_zero, ma_top_frac_zero)
        mb_is_flush_zero = solver.mkTerm(Kind.AND, mb_exp_zero, mb_top_frac_zero)#if the expoent and the first bit of mantissa is all 0 then flush the number to 0
        either_flush_zero = solver.mkTerm(Kind.OR, ma_is_flush_zero, mb_is_flush_zero)#if one is 0, then product is 0
        solver.assertFormula(solver.mkTerm(Kind.DISTINCT, ma_exp, solver.mkBitVector(e, (1 << e) - 1)))
        solver.assertFormula(solver.mkTerm(Kind.DISTINCT, mb_exp, solver.mkBitVector(e, (1 << e) - 1)))#the exponent cant be all 1 either

        rm = solver.mkRoundingMode(_RM)
        ma_dbl = widen(solver, decode_native(solver, ma_bv, e, m), _DBL_E, _DBL_M)
        mb_dbl = widen(solver, decode_native(solver, mb_bv, e, m), _DBL_E, _DBL_M)#transfer the value to a larger floating point format to avoid overflow
        _assert_input_range(solver, ma_dbl, ma_range, rm, _DBL_E, _DBL_M)
        _assert_input_range(solver, mb_dbl, mb_range, rm, _DBL_E, _DBL_M)
        exact_dbl = solver.mkTerm(Kind.FLOATINGPOINT_MULT, rm, ma_dbl, mb_dbl)#get the exact product
        exact_finite = solver.mkTerm(Kind.NOT, solver.mkTerm(Kind.OR,
            solver.mkTerm(Kind.FLOATINGPOINT_IS_INF, exact_dbl),
            solver.mkTerm(Kind.FLOATINGPOINT_IS_NAN, exact_dbl)))
        solver.assertFormula(exact_finite)#if the product is not abnormal, store it

        from benchmarks.minifloat import _bias
        min_normal_val = 2.0 ** (1 - _bias(eo))#the smallest number can be represented
        min_normal_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, _DBL_E, _DBL_M + 1),
                                       rm, _mkReal_exact(solver, min_normal_val))#exact product to floating number
        exact_abs_dbl = solver.mkTerm(Kind.FLOATINGPOINT_ABS, exact_dbl)
        is_underflow = solver.mkTerm(Kind.FLOATINGPOINT_LT, exact_abs_dbl, min_normal_dbl)
        is_neg = solver.mkTerm(Kind.FLOATINGPOINT_IS_NEG, exact_dbl)#decide if there's underflow and if the product is negative

        zero_pos = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, eo, out_sig), rm, solver.mkReal(0))
        zero_neg = solver.mkTerm(Kind.FLOATINGPOINT_NEG, zero_pos)
        min_normal_pos = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, eo, out_sig),
                                       rm, _mkReal_exact(solver, min_normal_val))
        min_normal_neg = solver.mkTerm(Kind.FLOATINGPOINT_NEG, min_normal_pos)
        zero_signed = solver.mkTerm(Kind.ITE, is_neg, zero_neg, zero_pos)
        min_normal_signed = solver.mkTerm(Kind.ITE, is_neg, min_normal_neg, min_normal_pos)#decide if, the leading bit is 1 the negative, otherwise positive

        nudge = solver.mkTerm(Kind.FLOATINGPOINT_MULT, rm, exact_abs_dbl,
                              solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, _DBL_E, _DBL_M + 1),
                                           rm, _mkReal_exact(solver, 2.0 ** -40)))
        exact_for_ceil = solver.mkTerm(Kind.FLOATINGPOINT_ADD, rm, exact_dbl, nudge)
        exact_for_floor = solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, exact_dbl, nudge)#the nudge is create to avoid exact integer like 15, if there's integer , the ceil and floor will be the same, so we need to add a small number to make sure they are different

        ceil_native = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, eo, out_sig),
                                    solver.mkRoundingMode(RoundingMode.ROUND_TOWARD_POSITIVE), exact_for_ceil)
        floor_native = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, eo, out_sig),
                                     solver.mkRoundingMode(RoundingMode.ROUND_TOWARD_NEGATIVE), exact_for_floor)#using the round_towards func to get the ceil and floor value of the product

        candidate_lo = solver.mkTerm(Kind.ITE, either_flush_zero, zero_signed,
                                     solver.mkTerm(Kind.ITE, is_underflow, zero_signed, floor_native))
        candidate_hi = solver.mkTerm(Kind.ITE, either_flush_zero, zero_signed,
                                     solver.mkTerm(Kind.ITE, is_underflow, min_normal_signed, ceil_native))#both candidate_lo and candidate_hi are the two nearest floating point number to the exact product, if the product is underflow, then the lower bound is 0, and the upper bound is the smallest normal number

        lo_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, _DBL_E, _DBL_M + 1), rm, candidate_lo)
        hi_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, _DBL_E, _DBL_M + 1), rm, candidate_hi)
        dist_lo = solver.mkTerm(Kind.FLOATINGPOINT_ABS, solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, exact_dbl, lo_dbl))
        dist_hi = solver.mkTerm(Kind.FLOATINGPOINT_ABS, solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, exact_dbl, hi_dbl))#get the absolue value of the difference between the exact product and the two nearest floating point number
        lo_is_nearer = solver.mkTerm(Kind.FLOATINGPOINT_LT, dist_lo, dist_hi)
        round_nearest_dbl = solver.mkTerm(Kind.ITE, lo_is_nearer, lo_dbl, hi_dbl)#see which line it's nearer to

        ulp_size_dbl = solver.mkTerm(Kind.FLOATINGPOINT_ABS,
                                     solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, hi_dbl, lo_dbl))#get how big 1 ulp is

        candidate_bv = gp_to_cvc5(individual, ma_bv, mb_bv, solver, src_format=src_format)#get the real gp expression output
        approx = decode_native_satmax(solver, candidate_bv, eo, mo)#candidate output to floating point number
        approx_dbl = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, _DBL_E, _DBL_M + 1), rm, approx)#transfer the candidate output to double format

        diff = solver.mkTerm(Kind.FLOATINGPOINT_ABS,
                             solver.mkTerm(Kind.FLOATINGPOINT_SUB, rm, approx_dbl, round_nearest_dbl))
        threshold_const = solver.mkTerm(solver.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, _DBL_E, _DBL_M + 1),
                                        rm, _mkReal_exact(solver, ulp_threshold))
        threshold_dbl = solver.mkTerm(Kind.FLOATINGPOINT_MULT, rm, threshold_const, ulp_size_dbl)#using ulp_number*width to get how big exact;y we can tolerate

        violation = solver.mkTerm(Kind.FLOATINGPOINT_GT, diff, threshold_dbl)
        solver.assertFormula(violation)#give the solver the constraint that the difference between the candidate output and the exact product is greater than the threshold

        result = solver.checkSat()#start check

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
